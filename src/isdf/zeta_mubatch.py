"""μ-batch ζ fit kernels: Z_q(G) by centroid batches; see docs/architecture/zeta_fit_mubatch.md.

Per μ batch ``B`` (a union of whole centroid orbits) every rank builds, on
its own orbit-closed r blocks and for every q of the full zone,

    D^X_{k̄,cd}(μ, r) = Σ_n w^X_n ψ_{n k̄ c}(r_μ) ψ*_{n k̄ d}(r)        (X = L, R; raw parents k̄)
    Z_q(μ, r)         = isdf.core.parent_projector_kconv(D^L, D^R)    (typed unfold + k-convolution)

completes the LR+RL normal equations and keeps the stored q rows.  One
all-to-all per r sub-block moves the rows to their owners over the full
grid (q-owned on the R4 tier, μ-owned otherwise); each owner applies
``e^{-iq·r}`` and a local full-box FFT and keeps the ζ sphere.  ``Z_q(G)``
is written once into :class:`ZStore`; no accumulator and no distributed FFT
exist.  ``C_q⁺`` acts on μ only, so it is applied afterwards on the sphere:
``ζ_q(G) = C_q⁺ Z_q(G)`` (:class:`ZetaG`).

One route serves every deck: a group of one operation is the identity
plan (singleton orbits), whose parents are the full grid.

Layouts (P ranks, flat index ``p = x·P_y + y``):

* ψ(r) block cache ``(n_bc, n_parent, bc_w, ns, P·R)`` at
  ``P(None, None, None, None, ('x','y'))``: rank p holds every band on its
  r blocks.  Built once (one band→r all-to-all per band chunk).
* ψ(G) resident ``(n_bc, n_parent, P·b_p, ns, ngkmax)`` at
  ``P(None, None, ('x','y'), None, None)`` (the plane route): each rank
  evaluates its own bands on every rank's planes of the current sub-block
  through the ψ cylinder, then one plane→owner all-to-all.
"""
from __future__ import annotations

import math
import time
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import io_callback as _io_callback
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.shard_map import shard_map
from common.fft_helpers import local_fftn3, local_ifftn3
from common.wfn_transforms import (
    _norm_split,
    _plane_geometry,
    _tile_plane_slots,
    apply_bloch_phase_at,
    to_rchunk_inner,
)
from runtime.padding import axis_mask, pad_to_axis, padded_axis, strip_axis

_XY = ('x', 'y')
_kernel_cache: dict = {}


def _mesh_size(mesh: Mesh) -> int:
    return int(mesh.shape['x']) * int(mesh.shape['y'])


def _mesh_id(mesh: Mesh) -> tuple:
    return (tuple(mesh.axis_names), tuple(int(v) for v in mesh.devices.shape),
            tuple(int(d.id) for d in mesh.devices.flat))


# ---------------------------------------------------------------------------
# r blocks
# ---------------------------------------------------------------------------

class RSpec(NamedTuple):
    """Static geometry of the rank r blocks (the per-block tables are operands).

    Rank p owns ``n_sub`` sub-blocks of ``r_s`` transport slots; their
    concatenation over ``(p, s)`` is a permutation of the (padded) grid.
    ``route`` names the ψ source: ``'cache'`` (the block cache, any point
    set) or ``'planes'`` (plane regeneration: each sub-block lies on at most
    ``n_pl`` planes of ``plane_axis``).
    """
    route: str
    n_ranks: int
    n_sub: int
    r_s: int
    n_pl: int
    plane_axis: int
    fft_grid: tuple

    @property
    def R(self) -> int:
        return self.n_sub * self.r_s


def r_blocks(k_unfold_plan, fft_grid, n_ranks: int, *, route: str,
             r_s_target: int):
    """The rank r blocks of the fit and their typed-unfold tables.

    Returns ``(rs, points (P, n_sub, r_s), local_perm (P, n_sub, n_rows,
    r_s), wraps (P, n_sub, n_rows, r_s, 3), planes (P, n_sub, n_pl))``.
    A symmetric group takes :func:`gw.centroid_k_unfold.orbit_r_blocks`
    (orbit-closed, owner-contiguous).  A group of one operation (singleton
    orbits, identity tables) on the plane route takes whole planes of the
    largest axis per rank, so no plane is regenerated for half its points;
    ``P`` beyond that axis refuses (pencils are not implemented).
    """
    from gw.centroid_k_unfold import orbit_r_blocks
    plan = k_unfold_plan
    fft_grid = tuple(int(v) for v in fft_grid)
    P_ = int(n_ranks)
    n_rows = int(np.asarray(plan.sym_perm).shape[0])
    if route == 'planes' and int(plan.n_sym_spatial) == 1:
        axis = int(np.argmax(fft_grid))
        n_a, (n_b, n_c), (b_ax, c_ax) = _plane_geometry(fft_grid, axis)
        if P_ > n_a:
            raise ValueError(
                f"GATE zeta-mubatch-pencils: got P={P_} ranks for the "
                f"plane-regenerated ψ(r) route, want P <= {n_a} (planes of the "
                f"largest box axis {axis} of {fft_grid}); why: a rank would own "
                "no plane, and the pencil split is not implemented.  Fix: a P at "
                "which the ψ(r) block cache fits.")
        ps = n_b * n_c
        n_pg = max(1, int(r_s_target) // ps)
        n_pl_rank = -(-n_a // P_)
        n_pg = min(n_pg, n_pl_rank)
        n_sub = -(-n_pl_rank // n_pg)
        plane = (np.arange(P_)[:, None, None] * (n_sub * n_pg)
                 + np.arange(n_sub)[None, :, None] * n_pg
                 + np.arange(n_pg)[None, None, :])            # (P, n_sub, n_pg)
        inp = np.arange(ps)
        coords = [None, None, None]
        coords[axis] = plane[..., None]
        coords[b_ax] = (inp // n_c)[None, None, None, :]
        coords[c_ax] = (inp % n_c)[None, None, None, :]
        nx, ny, nz = fft_grid
        flat = coords[0] * ny * nz + coords[1] * nz + coords[2]
        flat = np.where(plane[..., None] < n_a, flat, -1).reshape(P_, n_sub, n_pg * ps)
        planes = np.where(plane < n_a, plane, -1)
        r_s = n_pg * ps
        perm = np.broadcast_to(np.arange(r_s, dtype=np.int32),
                               (P_, n_sub, n_rows, r_s)).copy()
        wraps = np.zeros((P_, n_sub, n_rows, r_s, 3), dtype=np.int32)
        rs = RSpec('planes', P_, n_sub, r_s, n_pg, axis, fft_grid)
        return rs, flat.astype(np.int32), perm, wraps, planes.astype(np.int32)
    ob = orbit_r_blocks(plan, fft_grid, P_, r_s_target=int(r_s_target), route=route)
    planes = np.asarray(ob.planes, dtype=np.int32)
    rs = RSpec(route, P_, int(ob.n_sub), int(ob.r_s), int(planes.shape[-1]),
               int(ob.plane_axis), fft_grid)
    return (rs, np.asarray(ob.points, np.int32), np.asarray(ob.local_perm, np.int32),
            np.asarray(ob.wraps, np.int32), planes)


def box_from_slots(points: np.ndarray, n_rtot: int) -> np.ndarray:
    """Inverse of a transport table: the (rank, sub-block, slot) index of every grid point."""
    flat = np.asarray(points).reshape(-1)
    inv = np.full((int(n_rtot),), -1, dtype=np.int64)
    live = flat >= 0
    inv[flat[live]] = np.flatnonzero(live)
    if np.any(inv < 0):
        raise ValueError("box_from_slots: the r blocks do not cover the grid")
    return inv.astype(np.int32)


# ---------------------------------------------------------------------------
# Band slots
# ---------------------------------------------------------------------------

def band_slot_tables(psi_G_store, *, band_start: int, nb_face: int,
                     weight_l, weight_r):
    """Per band chunk: global face band of each transport slot and its L/R weight.

    Slot ``p·b_p + j`` of chunk ``bc`` is band ``lo + p·bpd_bc + j`` when
    ``j < bpd_bc`` (the ``PsiGStore`` per-rank layout), else a zero pad.
    Returns host ``(band_rel (n_bc, P·b_p) int32, w_l, w_r (n_bc, P·b_p))``;
    a pad or out-of-face slot has weight zero.
    """
    bcr = tuple((int(lo), int(hi)) for lo, hi in psi_G_store.band_chunk_ranges)
    b_p = int(psi_G_store.local_band_chunk_shape[1])
    P_ = int(psi_G_store.band_chunk_carrier) // max(b_p, 1)
    w_l = np.asarray(weight_l, dtype=np.float64)
    w_r = np.asarray(weight_r, dtype=np.float64)
    n_bc = len(bcr)
    rel = np.full((n_bc, P_ * b_p), -1, dtype=np.int32)
    for bc, (lo, hi) in enumerate(bcr):
        bpd = psi_G_store._bpd_per_bc[bc]
        for p in range(P_):
            for j in range(bpd):
                rel[bc, p * b_p + j] = lo - int(band_start) + p * bpd + j
    ok = (rel >= 0) & (rel < int(nb_face))
    safe = np.clip(rel, 0, int(nb_face) - 1)
    return (np.where(ok, rel, -1).astype(np.int32),
            np.where(ok, w_l[safe], 0.0), np.where(ok, w_r[safe], 0.0))


# ---------------------------------------------------------------------------
# ψ(r) sources
# ---------------------------------------------------------------------------

def build_psi_block_cache(psi_G_store, *, mesh: Mesh, rs: RSpec,
                          points: np.ndarray) -> jax.Array:
    """ψ on every rank's r blocks, all bands: built once for all batches.

    Each rank full-box IFFTs its own band shard (``to_rchunk_inner``, the
    incumbent transform), gathers every rank's block points, and one
    all-to-all per band chunk moves bands → r blocks.  Returns
    ``(n_bc, n_parent, P·b_p, ns, P·R)`` at ``P(None, None, None, None, ('x','y'))``.
    """
    fft_grid = rs.fft_grid
    n_rtot = math.prod(fft_grid)
    nk, b_p, ns, ngkmax = (int(v) for v in psi_G_store.local_band_chunk_shape)
    n_bc = len(psi_G_store.band_chunk_ranges)
    gather = np.clip(np.asarray(points).reshape(-1), 0, n_rtot - 1).astype(np.int32)
    key = ('psi_block_cache', _mesh_id(mesh), id(psi_G_store), rs, nk, b_p, ns,
           ngkmax, n_bc, hash(gather.tobytes()))
    fn = _kernel_cache.get(key)
    if fn is None:
        store = psi_G_store
        sds = jax.ShapeDtypeStruct((nk, b_p, ns, ngkmax), jnp.complex128)

        def _host(x_idx, y_idx, bc_idx):
            return store.read_local_band_chunk(x_idx, y_idx, bc_idx)

        @partial(shard_map, mesh=mesh, in_specs=(P(), P()),
                 out_specs=P(None, None, None, None, _XY), check_vma=False)
        def _local(g_index, kvecs):
            x_idx = jax.lax.axis_index('x')
            y_idx = jax.lax.axis_index('y')

            def body(carry, bc):
                g = _io_callback(_host, sds, x_idx, y_idx, bc, ordered=False)
                r = to_rchunk_inner(g, g_index, fft_grid, jnp.int32(0), n_rtot,
                                    kvecs_frac=kvecs, norm="ortho")
                r = jnp.take(r, jnp.asarray(gather), axis=-1)   # every rank's slots
                r = jax.lax.all_to_all(r, _XY, split_axis=3, concat_axis=1,
                                       tiled=True)
                return carry, r

            _, out = jax.lax.scan(body, jnp.int32(0),
                                  jnp.arange(n_bc, dtype=jnp.int32), unroll=1)
            return out

        fn = jax.jit(_local)
        _kernel_cache[key] = fn
    return fn(psi_G_store.g_index, psi_G_store.kvecs_frac)


def plane_cylinder(psi_G_store, rs: RSpec):
    """The store's ψ cylinder along the plane axis (``psi_cylinder_tables``)."""
    from common.wfn_transforms import psi_cylinder_tables
    return psi_cylinder_tables(
        psi_G_store.g_index, rs.fft_grid, rs.plane_axis,
        ngkmax=int(psi_G_store.local_band_chunk_shape[3]))


# ---------------------------------------------------------------------------
# One μ batch: Z_q(μ_B, G) rows on their owners
# ---------------------------------------------------------------------------

def make_batch_kernel(*, mesh: Mesh, rs: RSpec, plan, kgrid, fft_grid, ns: int,
                      b: int, n_bc: int, bc_w: int, nb_face: int,
                      q_sel, q_axis, q_neg, sphere_idx, qvec_frac, row_chunk: int,
                      source: str, rows: str = 'mu', psi_G_store=None,
                      cylinder=None, vertex=None, stop_at: str | None = None,
                      k_chunk: int | None = None):
    """Compile-once executable for one batch; see the module docstring.

    ``source``: ``'cache'`` (ψ block cache), ``'resident'`` (ψ(G) on
    device, plane route) or ``'host'`` (ψ(G) host store via io_callback,
    plane route).  ``plan`` is the fit's ``CentroidKUnfoldPlan`` (parents,
    actions, spin representation).  Returns ``fn(src, X_B, band_rel, w_l,
    w_r, kvecs, cyl, geo, batch) -> rows`` with ``geo = (points, r_perm,
    r_wrap, planes, box_from_slot, sphere)`` (:func:`r_blocks`; ``sphere`` the
    ``sphere_idx`` table as an operand, not an HLO constant) and ``batch =
    (left_perm, left_L)`` of this batch (``gw.centroid_k_unfold.orbit_mu_batches``).

    ``q_axis`` is the stored-q ``runtime.padding.PaddedAxis`` (logical Q,
    carrier a multiple of P) the store shares; ``sphere_idx`` is already at
    the store's G-tile carrier width (padded once, by the caller).

    ``rows`` names who owns a row after the transpose: ``'q'`` (the R4
    q-local tier: rank p gets whole stored q's, ``Q_pad/P`` of them, for
    every batch centroid; output ``(Q_pad, b, ngk)`` at
    ``P(('x','y'), None, None)``) or ``'mu'`` (rank p gets the batch slots
    ``p·c + [0, c)`` for every q; output ``(Q, b, ngk)`` at
    ``P(None, ('x','y'), None)``).

    ``k_chunk`` (plane route): parents regenerated per step, a divisor of
    ``n_parent`` (the planner's; bounds the band→plane transient, which
    holds P·b_p bands of every regenerated parent on the rank's planes).

    ``stop_at`` (debug split timers only: ``'psi'``, ``'gemm'``, ``'kconv'``,
    ``'transpose'``) builds the same kernel truncated after that stage,
    returning a checksum, so stage walls come from differences.
    """
    from isdf.core import parent_projector_kconv, _conv_kpair_static_gamma
    from ffi.fft import make_fused_conv_kparent
    P_ = rs.n_ranks
    nk_src = int(plan.n_parent)
    nk = int(np.prod(kgrid))
    kc = nk_src if k_chunk is None else int(k_chunk)
    if nk_src % kc:
        raise ValueError(f"make_batch_kernel: k_chunk {kc} must divide n_parent {nk_src}")
    n_kc = nk_src // kc
    if rows == 'mu' and b % P_:
        raise ValueError(f"make_batch_kernel: batch {b} must be a multiple of P={P_}")
    c = b // P_ if rows == 'mu' else 0
    q_sel = np.asarray(q_sel, dtype=np.int32)
    Q = int(q_axis.logical)
    if rows not in ('q', 'mu') or q_sel.shape != (Q,) or q_axis.divisor != P_:
        raise ValueError(f"make_batch_kernel: rows={rows!r}, q_sel {q_sel.shape} "
                         f"vs {q_axis}")
    Q_pad, Qloc = q_axis.carrier, q_axis.carrier // P_
    # Pad q rows gather any stored q and are zeroed by the mask.
    q_take = np.asarray(pad_to_axis(q_sel, q_axis, axis=0, fill=int(q_sel[0])))
    q_live = np.asarray(axis_mask(q_axis, dtype=np.float64))
    q_neg = None if q_neg is None else np.asarray(q_neg, dtype=np.int32)
    sphere = np.asarray(sphere_idx, dtype=np.int32)
    ngkmax = int(sphere.shape[1])
    qv = np.asarray(qvec_frac, dtype=np.float64)
    nx, ny, nz = (int(s) for s in fft_grid)
    if vertex is None:
        vertex = (np.arange(ns), np.ones(ns, dtype=np.complex128))
    n_rows = Qloc * b if rows == 'q' else Q * c
    cs = max(1, min(int(row_chunk), n_rows))
    rows_ax = padded_axis(n_rows, cs, name="μ-batch row-FFT scan rows")
    n_fft = rows_ax.carrier // cs
    key = ('batch', _mesh_id(mesh), rs, id(plan), tuple(kgrid), ns, b, n_bc, bc_w,
           nb_face, hash(q_sel.tobytes()), q_axis,
           None if q_neg is None else hash(q_neg.tobytes()),
           sphere.shape, hash(qv.tobytes()), cs, source, rows,
           None if psi_G_store is None else id(psi_G_store),
           tuple(int(v) for v in np.asarray(vertex[0])),
           tuple(complex(v) for v in np.asarray(vertex[1])), stop_at, kc)
    hit = _kernel_cache.get(key)
    if hit is not None:
        return hit

    if source in ('resident', 'host'):
        if rs.route != 'planes':
            raise ValueError("ψ(G) sources run on the plane route")
        cyl_index, cyl_axis, plane_from_col = cylinder
        n_a, (n_b, n_c), _ = _plane_geometry(fft_grid, rs.plane_axis)
        norm2, a_scale = _norm_split("ortho", n_a, n_b * n_c, inverse=True)
        n_col = int(cyl_index.shape[1])
        ps = n_b * n_c
    if source == 'host':
        store = psi_G_store
        b_p = int(store.local_band_chunk_shape[1])
        ngk_psi = int(store.local_band_chunk_shape[3])
        host_sds = jax.ShapeDtypeStruct((nk_src, b_p, ns, ngk_psi), jnp.complex128)

        def _host(x_idx, y_idx, bc_idx):
            return store.read_local_band_chunk(x_idx, y_idx, bc_idx)

    # The k-convolution arm the r-chunk loop uses: native conv_kparent where
    # its gate resolves it (CUDA auto), else the XLA tail.
    p_l, ph_l = _conv_kpair_static_gamma(None, ns)
    pair_kernel = make_fused_conv_kparent(
        mesh, kgrid, ns, (b, rs.r_s), perm_l=p_l, phase_l=ph_l, perm_r=p_l,
        phase_r=ph_l)
    phx = np.exp(-2j * np.pi * qv[:, 0:1] * (np.arange(nx) / nx)[None, :])
    phy = np.exp(-2j * np.pi * qv[:, 1:2] * (np.arange(ny) / ny)[None, :])
    phz = np.exp(-2j * np.pi * qv[:, 2:3] * (np.arange(nz) / nz)[None, :])

    src_spec = (P(None, None, None, None, _XY) if source == 'cache'
                else P(None, None, _XY, None, None) if source == 'resident'
                else P())
    geo_spec = (P(), P(_XY, None, None, None), P(_XY, None, None, None, None),
                P(), P(), P())

    @partial(shard_map, mesh=mesh,
             in_specs=(src_spec, P(), P(), P(), P(), P(), P(), geo_spec, P()),
             out_specs=(P(_XY, None, None) if rows == 'q'
                        else P(None, _XY, None)), check_vma=False)
    def _local(src, X_B, band_rel, w_l, w_r, kvecs, cyl, geo, batch):
        points, r_perm, r_wrap, planes, box_from_slot, sph = geo
        l_perm, l_wrap = batch
        x_idx = jax.lax.axis_index('x')
        y_idx = jax.lax.axis_index('y')
        p = x_idx * int(mesh.shape['y']) + y_idx

        def g_tile(bc):
            """This rank's ψ(G) band shard of chunk bc, (n_parent, b_p, ns, ngk)."""
            return (src[bc] if source == 'resident' else _io_callback(
                _host, host_sds, x_idx, y_idx, bc, ordered=False))

        def psi_at(g, s, k0, upto=None):
            """ψ(kc, bc_w, ns, r_s) of parents [k0, k0+kc) on sub-block s (plane route).

            ``upto`` (debug split timers): a checksum after the cylinder
            1D DFT ('psi_dft'), the band→plane all-to-all ('psi_a2a') or the
            2D plane IFFTs ('psi_ifft')."""
            g = jax.lax.dynamic_slice_in_dim(g, k0, kc, axis=0)
            ngk = int(g.shape[-1])
            gpad = jnp.concatenate(
                [g, jnp.zeros(g.shape[:3] + (1,), g.dtype)], axis=-1)
            ci, cax, pfc = cyl
            ci = jax.lax.dynamic_slice_in_dim(ci, k0, kc, axis=0)
            idx = jnp.clip(ci, 0, ngk).reshape(kc, 1, 1, -1)
            cyl_v = jnp.take_along_axis(gpad, idx, axis=-1).reshape(
                kc, g.shape[1], ns, n_col, -1)           # (k, b_p, s, col, n_s)
            pl_all = planes[:, s, :].reshape(-1)          # every rank's planes
            ph = jnp.exp((2j * jnp.pi / n_a) * (
                cax.astype(jnp.float64)[:, None]
                * jnp.maximum(pl_all, 0).astype(jnp.float64)[None, :])) * a_scale
            ph = jnp.where((pl_all >= 0)[None, :], ph, 0)
            F = jnp.einsum('kbscj,jp->kbspc', cyl_v, ph)
            if upto == 'psi_dft':
                return jnp.sum(jnp.abs(F))
            F = jax.lax.all_to_all(F, _XY, split_axis=3, concat_axis=1,
                                   tiled=True)             # (k, bc_w, s, n_pl, col)
            if upto == 'psi_a2a':
                return jnp.sum(jnp.abs(F))
            F = jnp.concatenate(
                [F, jnp.zeros(F.shape[:4] + (1,), F.dtype)], axis=-1)
            stack = jnp.take(F, pfc, axis=-1)              # (k, bc_w, s, n_pl, ps)
            r = local_ifftn3(stack.reshape(*stack.shape[:4], n_b, n_c),
                             axes=(-2, -1), norm=norm2)
            if upto == 'psi_ifft':
                return jnp.sum(jnp.abs(r))
            r = r.reshape(kc, bc_w, ns, rs.n_pl * ps)
            pts = points[p, s]
            slot, on = _tile_plane_slots(pts, planes[p, s], fft_grid, rs.plane_axis)
            r = jnp.take(r, jnp.clip(slot, 0, rs.n_pl * ps - 1), axis=-1)
            r = jnp.where(on[None, None, None, :], r, 0)
            return apply_bloch_phase_at(
                r, jax.lax.dynamic_slice_in_dim(kvecs, k0, kc, axis=0),
                fft_grid, pts)

        def sub_body(carry, s):
            buf, chk = carry
            if stop_at in ('psi', 'psi_dft', 'psi_a2a', 'psi_ifft'):
                acc = chk
                if source == 'cache':
                    acc = acc + jnp.sum(jnp.abs(jax.lax.dynamic_slice_in_dim(
                        src, s * rs.r_s, rs.r_s, axis=4)))
                else:
                    for bc in range(n_bc):
                        g = g_tile(bc)
                        for kci in range(n_kc):
                            acc = acc + jnp.sum(jnp.abs(psi_at(
                                g, s, kci * kc, upto=stop_at)))
                return (buf, acc), None
            if source == 'cache':
                # Every band is resident on this r block: one GEMM over all
                # of them (K = n_bc·bc_w), no accumulator round trips.
                psi = jax.lax.dynamic_slice_in_dim(
                    src, s * rs.r_s, rs.r_s, axis=4)   # (c, k, n, b, r)
                rel = jnp.clip(band_rel.reshape(-1), 0, nb_face - 1)
                x = jnp.take(X_B, rel, axis=3).reshape(nk_src, ns, b, n_bc, bc_w)
                # One GEMM for both windows: [x·w_L | x·w_R] along μ.
                xlr = jnp.concatenate([x * w_l[None, None, None],
                                       x * w_r[None, None, None]], axis=2)
                D = jnp.einsum('kamcn,cknbr->kambr', xlr, jnp.conj(psi))
                D_l, D_r = D[:, :, :b], D[:, :, b:]
            else:
                def band_body(D, bc):
                    g = g_tile(bc)
                    rel = band_rel[bc]
                    x = jnp.take(X_B, jnp.clip(rel, 0, nb_face - 1), axis=3)
                    # One GEMM for both windows: [x·w_L | x·w_R] along μ.
                    xlr = jnp.concatenate([x * w_l[bc][None, None, None, :],
                                           x * w_r[bc][None, None, None, :]], axis=2)

                    def k_body(D, kci):
                        k0 = kci * kc
                        sl = lambda a: jax.lax.dynamic_slice_in_dim(a, k0, kc, axis=0)
                        dD = jnp.einsum('kamn,knbr->kambr', sl(xlr),
                                        jnp.conj(psi_at(g, s, k0)))
                        return jax.lax.dynamic_update_slice_in_dim(
                            D, sl(D) + dD, k0, axis=0), None

                    D, _ = jax.lax.scan(
                        k_body, D, jnp.arange(n_kc, dtype=jnp.int32), unroll=1)
                    return D, None

                D, _ = jax.lax.scan(
                    band_body,
                    jnp.zeros((nk_src, ns, 2 * b, ns, rs.r_s), dtype=jnp.complex128),
                    jnp.arange(n_bc, dtype=jnp.int32), unroll=1)
                D_l, D_r = D[:, :, :b], D[:, :, b:]
            if stop_at == 'gemm':
                return (buf, chk + jnp.sum(jnp.abs(D_l)) + jnp.sum(jnp.abs(D_r))), None
            Z = parent_projector_kconv(
                D_l, D_r, plan=plan, left_perm=l_perm, left_L=l_wrap,
                right_perm=r_perm[0, s], right_L=r_wrap[0, s], kgrid=kgrid,
                vertex_l=vertex, vertex_r=vertex,
                pair_kernel=pair_kernel)                # (nk, b, r_s)
            if stop_at == 'kconv':
                return (buf, chk + jnp.sum(jnp.abs(Z))), None
            if q_neg is not None:
                Z = Z + jnp.conj(jnp.take(Z, jnp.asarray(q_neg), axis=0))
            if rows == 'q':
                Z = (jnp.take(Z, jnp.asarray(q_take), axis=0)
                     * jnp.asarray(q_live)[:, None, None])  # (Q_pad, b, r_s)
                Zt = jax.lax.all_to_all(Z, _XY, split_axis=0, concat_axis=2,
                                        tiled=True)         # (Qloc, b, P·r_s)
                Zt = Zt.reshape(Qloc, b, P_, 1, rs.r_s)
            else:
                Z = jnp.take(Z, jnp.asarray(q_sel), axis=0)  # (Q, b, r_s)
                Zt = jax.lax.all_to_all(Z, _XY, split_axis=1, concat_axis=2,
                                        tiled=True)         # (Q, c, P·r_s)
                Zt = Zt.reshape(Q, c, P_, 1, rs.r_s)
            return (jax.lax.dynamic_update_slice_in_dim(buf, Zt, s, axis=3),
                    chk), None

        lead = (Qloc, b) if rows == 'q' else (Q, c)
        buf = jnp.zeros(lead + (P_, rs.n_sub, rs.r_s), dtype=jnp.complex128)
        (buf, chk), _ = jax.lax.scan(
            sub_body, (buf, jnp.float64(0.0)),
            jnp.arange(rs.n_sub, dtype=jnp.int32), unroll=1)
        if stop_at in ('psi', 'psi_dft', 'psi_a2a', 'psi_ifft', 'gemm', 'kconv'):
            return jnp.zeros(lead + (ngkmax,), jnp.complex128) + chk
        if stop_at == 'transpose':
            return jnp.zeros(lead + (ngkmax,), jnp.complex128) + jnp.sum(
                jnp.abs(buf))
        buf = buf.reshape(n_rows, P_ * rs.R)
        if rows == 'q':
            q_row_all = jnp.minimum(
                p * Qloc + jnp.arange(n_rows, dtype=jnp.int32) // b, Q - 1)
        else:
            q_row_all = jnp.arange(n_rows, dtype=jnp.int32) // c
        buf = pad_to_axis(buf, rows_ax, axis=0)
        q_row_all = pad_to_axis(q_row_all, rows_ax, axis=0)
        phx_d, phy_d, phz_d = (jnp.asarray(phx), jnp.asarray(phy),
                               jnp.asarray(phz))

        def fft_body(out, i):
            sub = jax.lax.dynamic_slice_in_dim(buf, i * cs, cs, axis=0)
            qr = jax.lax.dynamic_slice_in_dim(q_row_all, i * cs, cs, axis=0)
            box = jnp.take(sub, box_from_slot, axis=-1).reshape(cs, nx, ny, nz)
            box = (box * phx_d[qr][:, :, None, None]
                   * phy_d[qr][:, None, :, None] * phz_d[qr][:, None, None, :])
            G = local_fftn3(box, axes=(-3, -2, -1), norm='backward').reshape(
                cs, nx * ny * nz)
            val = jnp.take_along_axis(G, sph[qr], axis=-1,
                                      mode='promise_in_bounds')
            return jax.lax.dynamic_update_slice_in_dim(out, val, i * cs,
                                                       axis=0), None

        out, _ = jax.lax.scan(
            fft_body, jnp.zeros((n_fft * cs, ngkmax), dtype=jnp.complex128),
            jnp.arange(n_fft, dtype=jnp.int32), unroll=1)
        return strip_axis(out, rows_ax, axis=0).reshape(*lead, ngkmax)

    fn = jax.jit(_local)
    _kernel_cache[key] = fn
    return fn


def gather_batch_centroids(psi_mun, slots, *, mesh: Mesh) -> jax.Array:
    """``X_B = ψ_{n k̄ s}(r_μ)`` for one batch's packed centroids, replicated.

    ``psi_mun`` is the raw-parent face ``(n_parent, ns, μ_pad, nb)``;
    ``slots`` ``(b,)`` packed indices, ``-1`` for a pad slot (zero column).
    Bytes ``n_parent·ns·b·nb·16`` per rank (VI3, b=80: 0.13 GB).
    """
    key = ('gather_X', _mesh_id(mesh), tuple(int(v) for v in psi_mun.shape),
           int(np.size(slots)))
    fn = _kernel_cache.get(key)
    if fn is None:
        @partial(jax.jit, out_shardings=NamedSharding(mesh, P()))
        def fn(face, idx):
            x = jnp.take(face, jnp.clip(idx, 0, face.shape[2] - 1), axis=2)
            return jnp.where((idx >= 0)[None, None, :, None], x, 0)
        _kernel_cache[key] = fn
    from common.collectives import device_put_process_local
    idx_dev = device_put_process_local(np.asarray(slots, dtype=np.int32),
                                       NamedSharding(mesh, P()))
    return fn(psi_mun, idx_dev)


# ---------------------------------------------------------------------------
# The Z store: write-once batches, streamed G tiles
# ---------------------------------------------------------------------------

def _reorder_rank_major(x, P_: int, n_batch: int, c: int, mu_pad: int, axis: int):
    """μ axis in (rank, batch, j) order → packed (batch, rank, j), cut to ``mu_pad``."""
    sh = x.shape
    x = x.reshape(*sh[:axis], P_, n_batch, c, *sh[axis + 1:])
    x = jnp.swapaxes(x, axis, axis + 1)
    x = x.reshape(*sh[:axis], P_ * n_batch * c, *sh[axis + 1:])
    return jax.lax.slice_in_dim(x, 0, mu_pad, axis=axis)


class ZStore:
    """Write-once ``Z_q(μ, G)``; the GPU holds only the batch being written.

    Tile-major: G is cut into ``n_Gt`` tiles of ``G_tile`` slots, so a batch
    write and a G-tile read are both a few large contiguous blocks (the
    owner's ``(q_tile, μ_batch, G_tile)`` chunking; SlabIO datasets are
    contiguous, so the tiling lives in the dataset shape).  One resource,
    placed by the memory planner:

    * ``'disk'``: a slab_io scratch dataset ``(n_Gt, Q, n_batch·b, G_tile)``
      in packed μ order.  A batch is one hyperslab at μ offset ``β·b``; a
      G tile is one hyperslab.
    * ``'host'``: one numpy tile ``(n_Gt, Q, n_batch·c, G_tile)`` per
      addressable device (the ``PsiGStore`` pattern): each rank keeps the
      rows it computed.

    ponytail: the store never lives on the device (owner simplification,
    2026-09-23).  A small deck whose whole store would fit beside the batch
    (CrI3 8x8 P16: 1.1 GB/rank) pays one host round trip, ~1 s; the
    upgrade path is a device placement here, priced by the planner.

    :meth:`read_tile` returns ``Z[:, :, tile]`` either q-local
    ``(Q_pad, μ_pad, G_tile)`` at ``P(('x','y'), None, None)`` or G-split
    ``(Q, μ_pad, G_tile)`` at ``P(None, None, ('x','y'))``; the host/device
    tiers reach that layout with one all-to-all per tile.
    """

    def __init__(self, *, mesh: Mesh, q_axis, mu_pad: int, g_axis, b: int,
                 placement: str, rows: str = 'mu',
                 scratch_path: str | None = None, packed_from_slot=None,
                 n_batch: int | None = None):
        """``q_axis``: stored q rows (carrier a multiple of P); ``g_axis``: the
        ζ sphere cut into whole G tiles (divisor = ``G_tile``).  Both are
        ``runtime.padding.PaddedAxis`` records the kernel and the finalize share."""
        self.mesh = mesh
        self.P = _mesh_size(mesh)
        self.q_axis, self.g_axis = q_axis, g_axis
        if q_axis.divisor != self.P:
            raise ValueError(f"ZStore: {q_axis} must divide over P={self.P}")
        self.Q, self.mu_pad, self.n_G = q_axis.logical, int(mu_pad), g_axis.carrier
        self.rows = str(rows)
        self.b, self.c = int(b), int(b) // self.P
        if self.rows == 'mu' and self.b % self.P:
            raise ValueError(f"ZStore(rows='mu'): batch {b} must be a multiple "
                             f"of P={self.P} (each rank owns b/P of its rows)")
        self.g_tile = g_axis.divisor
        self.n_Gt = g_axis.carrier // g_axis.divisor
        self.n_batch = (-(-self.mu_pad // self.b) if n_batch is None
                        else int(n_batch))
        self.Q_pad = q_axis.carrier
        self.placement = str(placement)
        # The store's μ axis is batch-slot order (β·b + j); readers gather the
        # packed carrier from it.  Contiguous batches make this a prefix.
        pfs = (np.arange(self.mu_pad, dtype=np.int32) if packed_from_slot is None
               else np.asarray(packed_from_slot, dtype=np.int32))
        if pfs.shape != (self.mu_pad,):
            raise ValueError(f"ZStore: packed_from_slot has shape {pfs.shape}")
        self._pfs = pfs
        self.bytes_written = 0
        self.bytes_read = 0
        self.t_write = 0.0
        self.t_read = 0.0
        if self.rows == 'q':
            self._init_q_owned(scratch_path)
            return
        local_shape = (self.n_Gt, self.Q, self.n_batch * self.c, self.g_tile)
        if self.placement == 'host':
            self._host = {dev.id: np.zeros(local_shape, np.complex128)
                          for dev in mesh.local_devices}
        elif self.placement == 'disk':
            from file_io.slab_io import SlabIO
            if scratch_path is None:
                raise ValueError("ZStore(disk) needs scratch_path")
            self.path = str(scratch_path)
            self._io = SlabIO(self.path, mode='w', mesh=mesh)
            self._io.create_dataset(
                'Z', shape=(self.n_Gt, self.Q, self.n_batch * self.b,
                            self.g_tile), dtype=np.complex128)
        else:
            raise ValueError(f"ZStore: unknown placement {placement!r}")

    # -- q-owned chunks (the R4 tier) --------------------------------------
    # (Q_pad, n_Gt, n_batch·b, G_tile) over q: a batch write is whole
    # (q, G_tile, μ_B) chunks per writer; a G-tile read is one contiguous
    # μ×G_tile block per q, local on host and device.
    def _init_q_owned(self, scratch_path):
        self.Qloc = self.Q_pad // self.P
        shape = (self.Q_pad, self.n_Gt, self.n_batch * self.b, self.g_tile)
        if self.placement == 'host':
            # Tile-major on the host: a G-tile read is one contiguous block.
            self._host = {dev.id: np.zeros((self.n_Gt, self.Qloc) + shape[2:],
                                           np.complex128)
                          for dev in self.mesh.local_devices}
        elif self.placement == 'disk':
            from file_io.slab_io import SlabIO
            if scratch_path is None:
                raise ValueError("ZStore(disk) needs scratch_path")
            self.path = str(scratch_path)
            self._io = SlabIO(self.path, mode='w', mesh=self.mesh)
            self._io.create_dataset('Z', shape=shape, dtype=np.complex128)
        else:
            raise ValueError(f"ZStore: unknown placement {self.placement!r}")

    def _write_q(self, beta, rows):
        chunks = _q_chunks(self.mesh, self.b, self.n_Gt, self.g_tile)(rows)
        if self.placement == 'host':
            lo = int(beta) * self.b
            for shard in chunks.addressable_shards:
                self._host[shard.device.id][:, :, lo:lo + self.b, :] = np.moveaxis(
                    np.asarray(shard.data), 1, 0)
        else:
            self._io.write_slab('Z', chunks, offset=(0, 0, int(beta) * self.b, 0))
        return chunks

    def _read_q(self, t):
        n_slot = self.n_batch * self.b
        if self.placement == 'host':
            return _host_tile_to_device(
                self.mesh, P(_XY, None, None), (self.Q_pad, n_slot, self.g_tile),
                {dev: self._host[dev.id][t] for dev in self.mesh.local_devices})
        self._io.sync_writes()
        raw = self._io.read_slab(
            'Z', shape=(self.Q_pad, 1, n_slot, self.g_tile),
            offset=(0, int(t), 0, 0), mesh=self.mesh,
            partition_spec=P(_XY, None, None, None))
        return _q_disk_tile(self.mesh, n_slot)(raw)

    # -- write ------------------------------------------------------------
    def write_batch(self, beta: int, rows: jax.Array) -> None:
        """``rows (Q, b, n_Gt·G_tile)`` at ``P(None, ('x','y'), None)`` for batch ``beta``."""
        t0 = time.perf_counter()
        if self.rows == 'q':
            tiled = self._write_q(beta, rows)
            jax.block_until_ready(tiled)
            self.bytes_written += self.Q * self.b * self.n_Gt * self.g_tile * 16
            self.t_write += time.perf_counter() - t0
            return
        tiled = _tile_rows(self.mesh, self.Q, self.b, self.n_Gt, self.g_tile)(rows)
        if self.placement == 'host':
            lo = int(beta) * self.c
            for shard in tiled.addressable_shards:
                self._host[shard.device.id][:, :, lo:lo + self.c, :] = np.asarray(
                    shard.data)
        else:
            self._io.write_slab('Z', tiled, offset=(0, 0, int(beta) * self.b, 0))
        jax.block_until_ready(tiled)
        self.bytes_written += self.Q * self.b * self.n_Gt * self.g_tile * 16
        self.t_write += time.perf_counter() - t0

    # -- read -------------------------------------------------------------
    def read_tile(self, t: int, *, layout: str) -> jax.Array:
        """G tile ``t``: ``layout='q'`` q-local, ``'g'`` G-split (see class doc)."""
        if layout not in ('q', 'g'):
            raise ValueError(f"ZStore.read_tile: layout {layout!r}")
        t0 = time.perf_counter()
        t = int(t)
        if self.rows == 'q':
            if layout != 'q':
                raise ValueError("ZStore(rows='q') serves q-local tiles only")
            out = self._read_q(t)
        elif self.placement == 'disk':
            self._io.sync_writes()
            if layout == 'q':
                raw = self._io.read_slab(
                    'Z', shape=(1, self.Q_pad, self.n_batch * self.b, self.g_tile),
                    offset=(t, 0, 0, 0), mesh=self.mesh,
                    partition_spec=P(None, _XY, None, None))
            else:
                raw = self._io.read_slab(
                    'Z', shape=(1, self.Q, self.n_batch * self.b, self.g_tile),
                    offset=(t, 0, 0, 0), mesh=self.mesh,
                    partition_spec=P(None, None, None, _XY))
            out = _disk_tile(self.mesh, layout, self.n_batch * self.b)(raw)
        else:
            local = _host_tile_to_device(
                self.mesh, P(None, _XY, None),
                (self.Q, self.P * self.n_batch * self.c, self.g_tile),
                {dev: self._host[dev.id][t] for dev in self.mesh.local_devices})
            out = _rows_to_layout(self.mesh, layout, self.P, self.n_batch,
                                  self.c, self.n_batch * self.b, self.q_axis)(local)
        # Store slot order → packed centroid order (a prefix for contiguous
        # batches, a gather for orbit batches).
        out = _slots_to_packed(self.mesh, layout)(out, jnp.asarray(self._pfs))
        # Not blocked: the caller prefetches tile t+1 behind tile t's work.
        self.bytes_read += self.Q * self.mu_pad * self.g_tile * 16
        self.t_read += time.perf_counter() - t0
        return out

    def close(self) -> None:
        if self.placement == 'host':
            self._host.clear()
        else:
            self._io.close()
            if jax.process_index() == 0:
                import os
                try:
                    os.unlink(self.path)
                except OSError:
                    pass

    def receipt(self) -> str:
        return (f"Z store: placement={self.placement}, rows={self.rows}, "
                f"(Q={self.Q}, μ={self.mu_pad}, "
                f"N_G={self.n_G}) as {self.n_Gt} tiles x {self.g_tile}, "
                f"{self.n_batch} batches of {self.b}; written "
                f"{self.bytes_written / 1e9:.2f} GB in {self.t_write:.2f} s, "
                f"read {self.bytes_read / 1e9:.2f} GB in {self.t_read:.2f} s "
                f"(global volumes)")


def _tile_rows(mesh, Q, b, n_Gt, g_tile):
    """``(Q, b, n_Gt·G_tile)`` μ-sharded → ``(n_Gt, Q, b, G_tile)`` μ-sharded (local)."""
    key = ('tile_rows', _mesh_id(mesh), Q, b, n_Gt, g_tile)
    fn = _kernel_cache.get(key)
    if fn is None:
        @partial(shard_map, mesh=mesh, in_specs=(P(None, _XY, None),),
                 out_specs=P(None, None, _XY, None), check_vma=False)
        def _f(r):
            q, c, _ = r.shape
            return jnp.transpose(r.reshape(q, c, n_Gt, g_tile), (2, 0, 1, 3))
        fn = jax.jit(_f)
        _kernel_cache[key] = fn
    return fn


def _rows_to_layout(mesh, layout, P_, n_batch, c, mu_pad, q_axis):
    """Rank-local rows ``(Q, P·n_batch·c, g)`` (μ in rank-major order) → q-local or G-split."""
    key = ('rows_to_layout', _mesh_id(mesh), layout, P_, n_batch, c, mu_pad, q_axis)
    fn = _kernel_cache.get(key)
    if fn is None:
        out_spec = P(_XY, None, None) if layout == 'q' else P(None, None, _XY)

        @partial(shard_map, mesh=mesh, in_specs=(P(None, _XY, None),),
                 out_specs=out_spec, check_vma=False)
        def _f(x):                                        # (Q, n_batch·c, g)
            if layout == 'q':
                x = pad_to_axis(x, q_axis, axis=0)
                x = jax.lax.all_to_all(x, _XY, split_axis=0, concat_axis=1,
                                       tiled=True)        # (Q_pad/P, P·n_b·c, g)
            else:
                x = jax.lax.all_to_all(x, _XY, split_axis=2, concat_axis=1,
                                       tiled=True)        # (Q, P·n_b·c, g/P)
            return _reorder_rank_major(x, P_, n_batch, c, mu_pad, axis=1)
        fn = jax.jit(_f)
        _kernel_cache[key] = fn
    return fn


def _disk_tile(mesh, layout, mu_pad):
    key = ('disk_tile', _mesh_id(mesh), layout, mu_pad)
    fn = _kernel_cache.get(key)
    if fn is None:
        in_spec = (P(None, _XY, None, None) if layout == 'q'
                   else P(None, None, None, _XY))
        out_spec = P(_XY, None, None) if layout == 'q' else P(None, None, _XY)

        @partial(shard_map, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec,
                 check_vma=False)
        def _f(raw):
            return jax.lax.slice_in_dim(raw[0], 0, mu_pad, axis=1)
        fn = jax.jit(_f)
        _kernel_cache[key] = fn
    return fn


# ---------------------------------------------------------------------------
# ζ held as (Z store, C⁺): formed tile by tile, never materialized whole
# ---------------------------------------------------------------------------


def zeta_shell_slots(gvec_components, ngk_per_q, q_frac, bvec, q_full_frac,
                     fft_grid):
    """Per stored q, the sphere slots with |q+G| ≤ max_q' |q'| (the G≈0 shell).

    Every parent G a full-zone literal G=0 unfolds from, and every Coulomb
    head slot (argmin |q+G|), lies in it: |q_p + G_p| = |q_full| under an
    orthogonal operation.  Slot 0 (G = 0) comes first.  Returns host
    ``(slots (Q, n_shell) int32, gvec (Q, 3, n_shell) int32)``; pad slots
    repeat slot 0 in ``slots`` and carry the FFT-box pad sentinel (never a
    sphere G, so no exact-G search can match it) in ``gvec``.
    """
    from common.gvec_fft_box import fft_box_pad_sentinel
    sentinel = np.asarray(fft_box_pad_sentinel(tuple(fft_grid))[0], np.int32)
    gv = np.asarray(gvec_components, dtype=np.int64)       # (Q, 3, ngkmax)
    B = np.asarray(bvec, dtype=np.float64)
    qf = np.asarray(q_frac, dtype=np.float64)
    k = np.einsum('qin,ij->qnj', gv + qf[:, :, None], B)   # Cartesian q+G
    k2 = np.sum(k * k, axis=-1)
    qmax2 = float(np.max(np.sum((np.asarray(q_full_frac) @ B) ** 2, axis=-1)))
    ngk = np.asarray(ngk_per_q, dtype=np.int64)
    lists = []
    for q in range(gv.shape[0]):
        live = np.flatnonzero(k2[q, :ngk[q]] <= qmax2 * (1 + 1e-9) + 1e-12)
        live = np.r_[0, live[live != 0]]
        lists.append(live)
    n = max(len(l) for l in lists)
    slots = np.zeros((gv.shape[0], n), dtype=np.int32)
    g_out = np.zeros((gv.shape[0], 3, n), dtype=np.int32)
    for q, l in enumerate(lists):
        slots[q, :len(l)] = l
        g_out[q, :, :len(l)] = gv[q][:, l]
        if len(l) < n:
            g_out[q, :, len(l):] = sentinel[:, None]
    return slots, g_out


class ZetaG:
    """ζ_q(G) = C_q⁺ Z_q(G) held as its :class:`ZStore` and the C⁺ factor.

    Presents the metadata a G-flat ζ reader presents (``n_rmu``,
    ``zeta_layout``, ``gvec_components``) so the V_q and head-channel
    consumers take it in place of a ``ZetaLoader``; their data comes through
    :meth:`contract_v`, which streams G tiles once: C⁺ applied on each tile,
    ``V_q += conj(ζ) diag(v_q) ζᵀ`` accumulated, the G≈0 shell kept, and each
    ζ tile written only when a file is wanted.  On the ``local`` tier every
    rank owns whole q's (the factor's R4 batch layout) and the pass moves no
    data but the store read; the only resident accumulator is V (Q·μ²/P).
    """

    zeta_layout = 'G_flat'

    def __init__(self, store, *, mesh, L_q, lu_piv, q_chunk_size, solver_kind,
                 zeta_gather, batched_route, n_rmu_solve, n_rmu, mu_basis,
                 ngk_per_q, gvec_components, shell_slots, shell_gvec, path):
        self.store = store
        self.mesh = mesh
        self.L_q, self.lu_piv = L_q, lu_piv
        self.q_chunk_size = int(q_chunk_size)
        self.solver_kind = str(solver_kind)
        self.zeta_gather = str(zeta_gather)
        self.batched_route = str(batched_route)
        self.n_rmu_solve = int(n_rmu_solve)
        self.n_rmu = int(n_rmu)
        self.n_rmu_disk = int(n_rmu)
        self.mu_basis = mu_basis
        self.ngk_per_q = np.asarray(ngk_per_q, dtype=np.int32)
        self.gvec_components = np.asarray(gvec_components, dtype=np.int32)
        self.ngkmax = int(self.gvec_components.shape[-1])
        self.shell_slots = np.asarray(shell_slots, dtype=np.int32)
        self.shell_gvec = np.asarray(shell_gvec, dtype=np.int32)
        self.path = str(path)
        self.shell = None
        self.receipt = ""

    @property
    def q_local(self) -> bool:
        return self.store.rows == 'q'


    # -- the one pass ---------------------------------------------------
    def contract_v(self, v_table, *, zeta_io=None, print_fn=print):
        """Stream every G tile once; return V (Q, μ_pad, μ_pad) at ``P(None,'x','y')``.

        ``v_table`` is ``(Q, ngkmax)`` v(q+G) on the stored sphere.  V and
        the shell come back in the canonical (file) centroid order.  With
        ``zeta_io`` the masked ζ tiles are also written to ``zeta_q_G``.
        """
        t0 = time.perf_counter()
        st = self.store
        # Pad q rows and G-tile slots carry v = 0, ngk = 0: inert in V.
        qa, ga = st.q_axis, st.g_axis
        v = np.asarray(pad_to_axis(pad_to_axis(
            jnp.asarray(v_table, dtype=jnp.complex128), qa, axis=0), ga, axis=1))
        ngk = np.asarray(pad_to_axis(jnp.asarray(self.ngk_per_q, jnp.int32), qa, axis=0))
        sl = np.asarray(pad_to_axis(jnp.asarray(self.shell_slots, jnp.int32), qa, axis=0))
        from common.collectives import device_put_process_local
        if self.q_local:
            layout = 'q'
            v_dev = device_put_process_local(
                v, NamedSharding(self.mesh, P(_XY, None)))
            ngk_dev = device_put_process_local(
                ngk, NamedSharding(self.mesh, P(_XY)))
            sl_dev = device_put_process_local(
                sl, NamedSharding(self.mesh, P(_XY, None)))
        else:
            layout = 'g'
            rep = NamedSharding(self.mesh, P())
            v_dev = device_put_process_local(np.asarray(strip_axis(v, qa, axis=0)), rep)
            ngk_dev = device_put_process_local(np.asarray(strip_axis(ngk, qa, axis=0)), rep)
            sl_dev = device_put_process_local(np.asarray(strip_axis(sl, qa, axis=0)), rep)
        dbg = _debug_enabled()
        step = _v_tile_kernel(self.mesh, layout, self.solver_kind,
                              self.n_rmu_solve, st.g_tile, debug_m=dbg)
        mu = int(st.mu_pad)
        V, M, shell = _zero_accumulators(self.mesh, layout, st.Q_pad, st.Q, mu,
                                         int(sl.shape[1]), debug_m=dbg)
        L_arg = self.L_q
        if layout == 'g':
            L_arg = jax.lax.with_sharding_constraint(
                self.L_q, NamedSharding(self.mesh, P()))
        nxt = st.read_tile(0, layout=layout)
        for t in range(st.n_Gt):
            # Tile t+1 is read while tile t is contracted.
            Zt = nxt
            nxt = st.read_tile(t + 1, layout=layout) if t + 1 < st.n_Gt else None
            V, M, shell, zt = step(L_arg, Zt, v_dev, ngk_dev, sl_dev,
                                   jnp.int32(t), V, M, shell)
            if zeta_io is not None:
                self._write_tile(zeta_io, zt, t * st.g_tile)
            del Zt, zt
        if not dbg:
            M = None
        V = _finish_v(self.mesh, layout, st.Q)(V)
        shell = _finish_shell(self.mesh, layout, st.Q)(shell)
        if self.mu_basis is not None:
            V = self.mu_basis.unpack_operator(V)
            shell = self.mu_basis.unpack_axis(shell, 1)
        self.shell = shell
        if M is not None:
            V_cmc = _v_from_m(self.mesh, layout, self.solver_kind,
                              self.n_rmu_solve)(L_arg, M)
            V_cmc = _finish_v(self.mesh, layout, st.Q)(V_cmc)
            if self.mu_basis is not None:
                V_cmc = self.mu_basis.unpack_operator(V_cmc)
            d = float(jnp.linalg.norm(V_cmc - V) / jnp.linalg.norm(V))
            if jax.process_index() == 0:
                print_fn(f"  μ-batch V check: conj(C+) M conj(C+) vs zeta-first "
                         f"rel {d:.3e} (production keeps zeta-first)")
        self.receipt = (f"  μ-batch V_q: {st.n_Gt} G tiles, {layout}-layout, "
                        f"{time.perf_counter() - t0:.2f}s (store read "
                        f"{st.t_read:.2f}s); zeta file "
                        f"{'written' if zeta_io is not None else 'not written'}")
        if jax.process_index() == 0:
            print_fn(self.receipt)
        return V

    def _write_tile(self, zeta_io, zt, g0):
        """One masked ζ tile into ``zeta_q_G`` (canonical μ order, clipped)."""
        if zt.sharding.spec[0] is not None:     # q-local → the writer's layout
            zt = jax.lax.with_sharding_constraint(
                zt, NamedSharding(self.mesh, P(None, _XY, None)))
        zt = zt[:self.store.Q]
        if self.mu_basis is not None:
            zt = self.mu_basis.unpack_axis(zt, 1)
        zeta_io.write_slab('zeta_q_G', zt, offset=(0, 0, int(g0)))

    def head_columns(self, sel):
        """ζ at the stored-sphere slots ``sel (Q, j)`` from the kept shell."""
        pos = np.zeros_like(np.asarray(sel, dtype=np.int32))
        for q in range(pos.shape[0]):
            for j in range(pos.shape[1]):
                hit = np.flatnonzero(self.shell_slots[q] == int(sel[q, j]))
                if hit.size == 0:
                    raise ValueError(
                        "GATE zeta-mubatch-shell: got head slot "
                        f"{int(sel[q, j])} at stored q {q}, want a slot inside "
                        "the kept G≈0 shell; why: the head channel would read a "
                        "column the μ-batch fit did not keep.")
                pos[q, j] = hit[0]
        return pos

    def close(self):
        self.store.close()
        self.L_q = self.lu_piv = self.shell = None


def check_mubatch_solve(rows: str, zeta_gather: str, solver_kind: str) -> None:
    """The planned Z-row ownership must match the resolved solve tier.

    The factor is always the rank-truncation ``B`` of :mod:`isdf.cplus`;
    the tier says who holds it: q-local (q-owned rows) or replicated.
    """
    want = 'local' if rows == 'q' else 'replicated'
    if str(zeta_gather) != want or str(solver_kind) != 'replicated_rank_truncate':
        raise ValueError(
            f"GATE zeta-mubatch-solve-tier: got tier {zeta_gather!r} / kind "
            f"{solver_kind!r} for {rows}-owned Z rows, want tier {want!r} with "
            "the whole-tile rank-truncation factor; why: the streamed finalize "
            "applies C⁺ (isdf.cplus) itself, q-local or replicated.  The "
            "distributed 2D factor application (regime A) is not implemented.")


def _debug_enabled() -> bool:
    from runtime import debug_print_enabled
    return bool(debug_print_enabled())


def _logical_solve(solver_kind: str, n_log: int):
    """ζ = C⁺Z at the logical μ extent through the conditioning seam."""
    from isdf import cplus
    from runtime.padding import solve_at_logical
    return lambda B, Z: solve_at_logical(cplus.apply, int(n_log), (B,), Z)


def _acc_specs(layout):
    """Accumulator layout: q-local blocks, or per-rank partial sums over local G."""
    return P(_XY, None, None) if layout == 'q' else P(_XY, None, None, None)


def _v_tile_kernel(mesh, layout, solver_kind, n_log, g_tile, *, debug_m):
    """One G tile: ζ = C⁺Z, V += conj(ζ) v ζᵀ, shell gather (and M in debug).

    ``layout='q'``: F (Q_pad, μ, μ) and Z (Q_pad, μ, g) q-local; V, shell and
    M accumulate on the q owner.  ``layout='g'``: F replicated, Z G-split;
    each rank accumulates its partial sums over its G columns into a leading
    rank axis, reduced once by :func:`_finish_v`.
    """
    key = ('v_tile', _mesh_id(mesh), layout, solver_kind, int(n_log),
           int(g_tile), bool(debug_m))
    fn = _kernel_cache.get(key)
    if fn is not None:
        return fn
    one = _logical_solve(solver_kind, n_log)
    acc = _acc_specs(layout)
    if layout == 'q':
        in_specs = (P(_XY, None, None), P(_XY, None, None), P(_XY, None),
                    P(_XY), P(_XY, None), P(), acc, acc, acc)
        z_spec = P(_XY, None, None)
    else:
        in_specs = (P(), P(None, None, _XY), P(), P(), P(), P(), acc, acc, acc)
        z_spec = P(None, None, _XY)

    @partial(shard_map, mesh=mesh, in_specs=in_specs,
             out_specs=(acc, acc, acc, z_spec), check_vma=False)
    def k(F, Z, v, ngk, sl, t, V, M, S):
        n_g = Z.shape[-1]
        g_idx = t * g_tile + jnp.arange(n_g, dtype=jnp.int32)
        if layout == 'g':
            g_idx = g_idx + jax.lax.axis_index(_XY) * n_g
        mask = g_idx[None, :] < ngk[:, None]                    # (q, g)
        zeta = jnp.where(mask[:, None, :], jax.vmap(one)(F, Z), 0)
        vt = jnp.where(mask, jnp.take(v, jnp.clip(g_idx, 0, v.shape[-1] - 1),
                                      axis=1), 0)
        dV = jnp.einsum('qmg,qg,qng->qmn', jnp.conj(zeta), vt, zeta)
        hit = (sl[:, None, :] == g_idx[None, :, None]).astype(zeta.dtype)
        dS = jnp.einsum('qmg,qgs->qms', zeta, hit)
        if layout == 'g':
            dV, dS = dV[None], dS[None]
        V = V + dV
        S = S + dS
        if debug_m:
            dM = jnp.einsum('qmg,qg,qng->qmn', jnp.conj(Z), vt, Z)
            M = M + (dM[None] if layout == 'g' else dM)
        return V, M, S, zeta

    fn = jax.jit(k, donate_argnums=(6, 7, 8))
    _kernel_cache[key] = fn
    return fn


def _zero_accumulators(mesh, layout, Q_pad, Q, mu, n_sh, *, debug_m):
    P_ = _mesh_size(mesh)
    sh = NamedSharding(mesh, _acc_specs(layout))
    if layout == 'q':
        vs, ss = (Q_pad, mu, mu), (Q_pad, mu, n_sh)
    else:
        vs, ss = (P_, Q, mu, mu), (P_, Q, mu, n_sh)
    z = lambda shape: jax.jit(lambda: jnp.zeros(shape, jnp.complex128),
                              out_shardings=sh)()
    return z(vs), (z(vs) if debug_m else z(vs[:1] + (1,) * (len(vs) - 1))), z(ss)


def _finish_v(mesh, layout, Q):
    """Accumulator → ``(Q, μ, μ)`` at ``P(None, 'x', 'y')``."""
    key = ('finish_v', _mesh_id(mesh), layout, int(Q))
    fn = _kernel_cache.get(key)
    if fn is None:
        out = NamedSharding(mesh, P(None, 'x', 'y'))

        @partial(jax.jit, out_shardings=out)
        def fn(V):
            return (V[:Q] if layout == 'q' else jnp.sum(V, axis=0))
        _kernel_cache[key] = fn
    return fn


def _finish_shell(mesh, layout, Q):
    """Shell accumulator → ``(Q, μ, n_shell)`` at ``P(None, ('x','y'), None)``."""
    key = ('finish_shell', _mesh_id(mesh), layout, int(Q))
    fn = _kernel_cache.get(key)
    if fn is None:
        out = NamedSharding(mesh, P(None, _XY, None))

        @partial(jax.jit, out_shardings=out)
        def fn(S):
            return (S[:Q] if layout == 'q' else jnp.sum(S, axis=0))
        _kernel_cache[key] = fn
    return fn


def _v_from_m(mesh, layout, solver_kind, n_log):
    """Debug check: V = conj(C⁺) M conj(C⁺) by the same per-q solver (two applications)."""
    key = ('v_from_m', _mesh_id(mesh), layout, solver_kind, int(n_log))
    fn = _kernel_cache.get(key)
    if fn is None:
        one = _logical_solve(solver_kind, n_log)
        acc = _acc_specs(layout)
        f_spec = P(_XY, None, None) if layout == 'q' else P()

        @partial(shard_map, mesh=mesh, in_specs=(f_spec, acc), out_specs=acc,
                 check_vma=False)
        def k(F, M):
            if layout == 'g':
                M = jax.lax.psum(M[0], _XY)
            X = jax.vmap(one)(F, jnp.conj(M))                 # C⁻¹ M*
            Vc = jnp.conj(jax.vmap(one)(F, jnp.conj(jnp.swapaxes(X, -1, -2))))
            if layout == 'g':
                Vc = jnp.where(jax.lax.axis_index(_XY) == 0, Vc, 0)[None]
            return Vc
        fn = jax.jit(k)
        _kernel_cache[key] = fn
    return fn


def _q_chunks(mesh, b, n_Gt, g_tile):
    """q-owned rows ``(Q_pad, b, n_Gt·G_tile)`` → chunks ``(Q_pad, n_Gt, b, G_tile)`` (local)."""
    key = ('q_chunks', _mesh_id(mesh), b, n_Gt, g_tile)
    fn = _kernel_cache.get(key)
    if fn is None:
        @partial(shard_map, mesh=mesh, in_specs=(P(_XY, None, None),),
                 out_specs=P(_XY, None, None, None), check_vma=False)
        def _f(r):
            q = r.shape[0]
            return jnp.transpose(r.reshape(q, b, n_Gt, g_tile), (0, 2, 1, 3))
        fn = jax.jit(_f)
        _kernel_cache[key] = fn
    return fn


def _q_disk_tile(mesh, mu_pad):
    key = ('q_disk_tile', _mesh_id(mesh), int(mu_pad))
    fn = _kernel_cache.get(key)
    if fn is None:
        @partial(shard_map, mesh=mesh, in_specs=(P(_XY, None, None, None),),
                 out_specs=P(_XY, None, None), check_vma=False)
        def _f(raw):
            return jax.lax.slice_in_dim(raw[:, 0], 0, mu_pad, axis=1)
        fn = jax.jit(_f)
        _kernel_cache[key] = fn
    return fn


def _host_tile_to_device(mesh, spec, shape, local_tiles):
    """This process's host tiles → one sharded array, through the collectives service.

    ``common.collectives.restore_from_host`` is the one place this process
    places host shards on its own devices (no cross-process traffic).
    """
    from common.collectives import HostSpill, restore_from_host
    return restore_from_host(HostSpill(
        shape=tuple(int(v) for v in shape),
        sharding=NamedSharding(mesh, spec),
        shards=[(dev, np.ascontiguousarray(a)) for dev, a in local_tiles.items()]))


def _slots_to_packed(mesh, layout):
    """Store slot order → packed centroid order on the (unsharded) μ axis."""
    key = ('slots_to_packed', _mesh_id(mesh), layout)
    fn = _kernel_cache.get(key)
    if fn is None:
        spec = P(_XY, None, None) if layout == 'q' else P(None, None, _XY)

        @partial(shard_map, mesh=mesh, in_specs=(spec, P()), out_specs=spec,
                 check_vma=False)
        def _g(x, pfs):
            # -1: a layout pad centroid, in no batch; its Z row is exactly zero.
            y = jnp.take(x, jnp.clip(pfs, 0, None), axis=1)
            live = (pfs >= 0)[None, :, None]
            return jnp.where(live, y, 0)
        fn = jax.jit(_g)
        _kernel_cache[key] = fn
    return fn
