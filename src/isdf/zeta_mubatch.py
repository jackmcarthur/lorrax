"""μ-batch ζ fit kernels: Z_q(G) by centroid batches; see docs/architecture/zeta_fit_mubatch.md.

Per μ batch ``B`` every rank builds, on its own block ``R_p`` of the real
grid and for every q of the full zone,

    D^X_{k,ab}(μ, r) = Σ_n w^X_n ψ_{nka}(r_μ) ψ*_{nkb}(r)         (X = L, R)
    Z_q(μ, r)        = Σ_k Σ_ab D^L_{k,ab}(μ, r) conj(D^R_{k+q,ab}(μ, r))

(the pair GEMM and the k-convolution of ``isdf.core._tile_tail``, on the
unfolded full BZ, so no orbit-closed tile exists), completes the LR+RL
normal equations and keeps the stored q rows.  One all-to-all per r
sub-block moves the rows ``(q, μ_B)`` to their owners over the full grid;
each owner applies ``e^{-iq·r}`` and a local full-box FFT and keeps the ζ
sphere.  ``Z_q(G)`` is written once into :class:`ZStore`; no accumulator
and no distributed FFT exist.  ``C_q⁺`` acts on μ only, so the caller
applies it afterwards on the sphere: ``ζ_q(G) = C_q⁺ Z_q(G)``.

Layouts (P ranks, flat index ``p = x·P_y + y``):

* ψ(r) block cache ``(n_bc, nk, bc_w, ns, P·R)`` at
  ``P(None, None, None, None, ('x','y'))``: rank p holds every band on its
  r block.  Built once (one band→r all-to-all per band chunk).
* ψ(G) resident ``(n_bc, nk, P·b_p, ns, ngkmax)`` at
  ``P(None, None, ('x','y'), None, None)`` (the plane route): each rank
  evaluates its own bands on every rank's current planes through the ψ
  cylinder, then one plane→owner all-to-all.
* Z store ``(Q, μ_pad, N_G)`` at ``P(None, ('x','y'), None)``: batch β
  holds ``c = b/P`` consecutive μ of every rank's μ block, so each rank
  writes only its own rows.
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
    apply_bloch_phase_at,
    to_rchunk_inner,
)

_XY = ('x', 'y')
_kernel_cache: dict = {}


def _mesh_size(mesh: Mesh) -> int:
    return int(mesh.shape['x']) * int(mesh.shape['y'])


def _mesh_id(mesh: Mesh) -> tuple:
    return (tuple(mesh.axis_names), tuple(int(v) for v in mesh.devices.shape),
            tuple(int(d.id) for d in mesh.devices.flat))


# ---------------------------------------------------------------------------
# r-block geometry
# ---------------------------------------------------------------------------

class RBlocks(NamedTuple):
    """The r partition of one fit: rank p owns ``R`` transport slots.

    ``route='cache'``: flat grid blocks ``[p·R, (p+1)·R)`` (no box axis).
    ``route='planes'``: whole planes ``[p·n_pl, (p+1)·n_pl)`` of ``axis``
    (the largest box axis), ``R = n_pl·ps``.  Either way the transport
    order of rank p's slots is ``R = n_sub·r_s`` sub-blocks, and the
    concatenation over p of all slots is a permutation of the padded grid
    that :func:`transport_to_box` inverts.
    """
    route: str
    n_ranks: int
    R: int
    r_s: int
    n_sub: int
    axis: int
    n_pl: int
    n_pg: int
    ps: int
    fft_grid: tuple


def make_r_blocks(fft_grid, n_ranks: int, *, route: str, r_sub: int) -> RBlocks:
    """Partition the grid for ``n_ranks``; ``r_sub`` is the sub-block width
    (points for ``cache``, planes for ``planes``)."""
    fft_grid = tuple(int(s) for s in fft_grid)
    n_rtot = math.prod(fft_grid)
    P_ = int(n_ranks)
    if route == 'cache':
        R0 = -(-n_rtot // P_)
        r_s = max(1, min(int(r_sub), R0))
        n_sub = -(-R0 // r_s)
        return RBlocks('cache', P_, n_sub * r_s, r_s, n_sub, -1, 0, 0, 0,
                       fft_grid)
    if route == 'planes':
        axis = int(np.argmax(fft_grid))
        n_a, (n_b, n_c), _ = _plane_geometry(fft_grid, axis)
        if P_ > n_a:
            raise ValueError(
                "GATE zeta-mubatch-pencils: got P="
                f"{P_} ranks for the plane-regenerated psi(r) route, want P <= "
                f"{n_a} (planes along the largest box axis {axis} of "
                f"{fft_grid}); why: a rank would own no plane, and the pencil "
                "split over two axes is not implemented.  Fix: a P at which "
                "the psi(r) cache fits (the planner then takes the flat-block "
                "route), or fewer ranks.")
        n_pl0 = -(-n_a // P_)
        n_pg = max(1, min(int(r_sub), n_pl0))
        n_sub = -(-n_pl0 // n_pg)
        n_pl = n_sub * n_pg
        ps = n_b * n_c
        return RBlocks('planes', P_, n_pl * ps, n_pg * ps, n_sub, axis, n_pl,
                       n_pg, ps, fft_grid)
    raise ValueError(f"make_r_blocks: unknown route {route!r}")


def block_points(rb: RBlocks) -> np.ndarray:
    """``(P, n_sub, r_s)`` flat grid index of every transport slot (-1 = pad)."""
    n_rtot = math.prod(rb.fft_grid)
    if rb.route == 'cache':
        pts = np.arange(rb.n_ranks * rb.R, dtype=np.int64)
        pts = np.where(pts < n_rtot, pts, -1)
        return pts.reshape(rb.n_ranks, rb.n_sub, rb.r_s).astype(np.int32)
    n_a, (n_b, n_c), (b_ax, c_ax) = _plane_geometry(rb.fft_grid, rb.axis)
    plane = np.arange(rb.n_ranks * rb.n_pl).reshape(rb.n_ranks, rb.n_pl)
    inp = np.arange(rb.ps)
    coords = [None, None, None]
    coords[rb.axis] = plane[:, :, None]
    coords[b_ax] = (inp // n_c)[None, None, :]
    coords[c_ax] = (inp % n_c)[None, None, :]
    nx, ny, nz = rb.fft_grid
    flat = (coords[0] * ny * nz + coords[1] * nz + coords[2])
    flat = np.where(plane[:, :, None] < n_a, flat, -1)
    return flat.reshape(rb.n_ranks, rb.n_sub, rb.r_s).astype(np.int32)


def transport_to_box(rows, rb: RBlocks):
    """``(..., P·R)`` transport order → ``(..., nx, ny, nz)`` box order."""
    lead = rows.shape[:-1]
    if rb.route == 'cache':
        n_rtot = math.prod(rb.fft_grid)
        return rows[..., :n_rtot].reshape(*lead, *rb.fft_grid)
    n_a, (n_b, n_c), (b_ax, c_ax) = _plane_geometry(rb.fft_grid, rb.axis)
    x = rows.reshape(*lead, rb.n_ranks * rb.n_pl, n_b, n_c)[..., :n_a, :, :]
    nl = len(lead)
    # (a, b, c) → grid axis order.
    order = [None, None, None]
    order[rb.axis], order[b_ax], order[c_ax] = nl, nl + 1, nl + 2
    return jnp.transpose(x, tuple(range(nl)) + tuple(order))


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

def build_psi_block_cache(psi_G_store, *, mesh: Mesh, rb: RBlocks) -> jax.Array:
    """ψ on every rank's r block, all bands: built once for all batches.

    Each rank full-box IFFTs its own band shard (``to_rchunk_inner``, the
    same transform as the incumbent cache), then one all-to-all per band
    chunk moves bands → r blocks.  Returns
    ``(n_bc, nk, P·b_p, ns, P·R)`` at ``P(None, None, None, None, ('x','y'))``.
    """
    if rb.route != 'cache':
        raise ValueError("build_psi_block_cache: needs the flat-block route")
    fft_grid = rb.fft_grid
    n_rtot = math.prod(fft_grid)
    nk, b_p, ns, ngkmax = (int(v) for v in psi_G_store.local_band_chunk_shape)
    n_bc = len(psi_G_store.band_chunk_ranges)
    P_ = rb.n_ranks
    pad = P_ * rb.R - n_rtot
    key = ('psi_block_cache', _mesh_id(mesh), id(psi_G_store), rb, nk, b_p, ns,
           ngkmax, n_bc)
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
                if pad:
                    r = jnp.pad(r, ((0, 0), (0, 0), (0, 0), (0, pad)))
                r = jax.lax.all_to_all(r, _XY, split_axis=3, concat_axis=1,
                                       tiled=True)
                return carry, r

            _, out = jax.lax.scan(body, jnp.int32(0),
                                  jnp.arange(n_bc, dtype=jnp.int32), unroll=1)
            return out

        fn = jax.jit(_local)
        _kernel_cache[key] = fn
    return fn(psi_G_store.g_index, psi_G_store.kvecs_frac)


def plane_cylinder(psi_G_store, rb: RBlocks):
    """The store's ψ cylinder along the plane axis (``psi_cylinder_tables``)."""
    from common.wfn_transforms import psi_cylinder_tables
    return psi_cylinder_tables(
        psi_G_store.g_index, rb.fft_grid, rb.axis,
        ngkmax=int(psi_G_store.local_band_chunk_shape[3]))


# ---------------------------------------------------------------------------
# One μ batch: Z_q(μ_B, G) rows on their owners
# ---------------------------------------------------------------------------

def _kconv_tail(D_l, D_r, kgrid, vertex):
    """Z_q(μ, r) = FFT_k Σ_ab conj(IFFT_k conj D^L_ab) · φ_ab IFFT_k conj D^R_{π(ab)}.

    The identity-plan arm of ``isdf.core._tile_tail`` (both transforms
    ``norm='forward'``).  ``D`` is ``(ns, ns, nk, b, r)``; ``vertex`` the
    static current-vertex output permutation/phase (identity for charge).
    """
    ns = int(D_l.shape[0])
    nkx, nky, nkz = kgrid
    b, r = int(D_l.shape[3]), int(D_l.shape[4])
    sh = (nkx, nky, nkz, b, r)
    perm, phase = vertex
    pairs = np.arange(ns * ns, dtype=np.int32)
    rpairs = (np.asarray(perm)[:, None] * ns
              + np.asarray(perm)[None, :]).reshape(-1).astype(np.int32)
    phases = (np.asarray(phase)[:, None]
              * np.asarray(phase)[None, :]).reshape(-1).astype(np.complex128)
    Dl = D_l.reshape(ns * ns, *D_l.shape[2:])
    Dr = D_r.reshape(ns * ns, *D_r.shape[2:])

    def spin(acc, args):
        i, j, ph = args
        Pl = local_ifftn3(jnp.conj(Dl[i]).reshape(sh), axes=(0, 1, 2),
                          norm='forward')
        Pr = local_ifftn3(jnp.conj(Dr[j]).reshape(sh), axes=(0, 1, 2),
                          norm='forward')
        return acc + jnp.conj(Pl) * (ph * Pr), None

    acc, _ = jax.lax.scan(
        spin, jnp.zeros(sh, dtype=jnp.complex128),
        (jnp.asarray(pairs), jnp.asarray(rpairs), jnp.asarray(phases)),
        unroll=1)
    return local_fftn3(acc, axes=(0, 1, 2), norm='forward').reshape(
        nkx * nky * nkz, b, r)


def make_batch_kernel(*, mesh: Mesh, rb: RBlocks, kgrid, fft_grid, nk: int,
                      ns: int, b: int, n_bc: int, bc_w: int, nb_face: int,
                      q_sel, q_neg, sphere_idx, qvec_frac, row_chunk: int,
                      source: str, psi_G_store=None, cylinder=None,
                      vertex=None):
    """Compile-once executable for one batch; see the module docstring.

    ``source``: ``'cache'`` (ψ block cache), ``'resident'`` (ψ(G) on
    device, plane route) or ``'host'`` (ψ(G) host store via io_callback,
    plane route).  Returns ``fn(src, X_B, band_rel, w_l, w_r, pts, kvecs)
    -> rows (Q, P·c, ngkmax)`` at ``P(None, ('x','y'), None)`` where rank p's
    ``c = b/P`` rows are its batch centroids.
    """
    P_ = rb.n_ranks
    if b % P_:
        raise ValueError(f"make_batch_kernel: batch {b} must be a multiple of P={P_}")
    c = b // P_
    q_sel = (np.arange(nk, dtype=np.int32) if q_sel is None
             else np.asarray(q_sel, dtype=np.int32))
    Q = int(q_sel.size)
    q_neg = None if q_neg is None else np.asarray(q_neg, dtype=np.int32)
    sphere = np.asarray(sphere_idx, dtype=np.int32)
    ngkmax = int(sphere.shape[1])
    qv = np.asarray(qvec_frac, dtype=np.float64)
    nx, ny, nz = (int(s) for s in fft_grid)
    if vertex is None:
        vertex = (np.arange(ns), np.ones(ns, dtype=np.complex128))
    n_rows = Q * c
    cs = max(1, min(int(row_chunk), n_rows))
    n_fft = -(-n_rows // cs)
    key = ('batch', _mesh_id(mesh), rb, tuple(kgrid), nk, ns, b, n_bc, bc_w,
           nb_face, hash(q_sel.tobytes()),
           None if q_neg is None else hash(q_neg.tobytes()),
           hash(sphere.tobytes()), hash(qv.tobytes()), cs, source,
           None if psi_G_store is None else id(psi_G_store),
           tuple(int(v) for v in np.asarray(vertex[0])),
           tuple(complex(v) for v in np.asarray(vertex[1])))
    hit = _kernel_cache.get(key)
    if hit is not None:
        return hit

    if source in ('resident', 'host'):
        if rb.route != 'planes':
            raise ValueError("ψ(G) sources run on the plane route")
        cyl_index, cyl_axis, plane_from_col = cylinder
        n_a, (n_b, n_c), _ = _plane_geometry(fft_grid, rb.axis)
        norm2, a_scale = _norm_split("ortho", n_a, rb.ps, inverse=True)
        cyl_axis_np = np.asarray(jax.device_get(cyl_axis), dtype=np.float64)
        # Global plane coordinates of sub-block s on every rank: (n_sub, P·n_pg).
        planes = (np.arange(P_)[None, :, None] * rb.n_pl
                  + np.arange(rb.n_sub)[:, None, None] * rb.n_pg
                  + np.arange(rb.n_pg)[None, None, :]).reshape(rb.n_sub, -1)
        ph_np = (np.exp(2j * np.pi / n_a * cyl_axis_np[None, :, None]
                        * planes[:, None, :].astype(np.float64)) * a_scale)
        n_col = int(cyl_index.shape[1])
    if source == 'host':
        store = psi_G_store
        b_p = int(store.local_band_chunk_shape[1])
        ngk_psi = int(store.local_band_chunk_shape[3])
        host_sds = jax.ShapeDtypeStruct((nk, b_p, ns, ngk_psi), jnp.complex128)

        def _host(x_idx, y_idx, bc_idx):
            return store.read_local_band_chunk(x_idx, y_idx, bc_idx)

    phx = np.exp(-2j * np.pi * qv[:, 0:1] * (np.arange(nx) / nx)[None, :])
    phy = np.exp(-2j * np.pi * qv[:, 1:2] * (np.arange(ny) / ny)[None, :])
    phz = np.exp(-2j * np.pi * qv[:, 2:3] * (np.arange(nz) / nz)[None, :])

    src_spec = (P(None, None, None, None, _XY) if source == 'cache'
                else P(None, None, _XY, None, None) if source == 'resident'
                else P())

    @partial(shard_map, mesh=mesh,
             in_specs=(src_spec, P(), P(), P(), P(), P(), P(), P()),
             out_specs=P(None, _XY, None), check_vma=False)
    def _local(src, X_B, band_rel, w_l, w_r, pts, kvecs, cyl):
        x_idx = jax.lax.axis_index('x')
        y_idx = jax.lax.axis_index('y')
        p = x_idx * int(mesh.shape['y']) + y_idx
        pts_p = pts[p]                                    # (n_sub, r_s)

        def psi_at(bc, s):
            """ψ(nk, bc_w, ns, r_s) of band chunk bc on sub-block s."""
            if source == 'cache':
                return jax.lax.dynamic_slice_in_dim(
                    src[bc], s * rb.r_s, rb.r_s, axis=3)
            g = (src[bc] if source == 'resident' else _io_callback(
                _host, host_sds, x_idx, y_idx, bc, ordered=False))
            ngk = int(g.shape[-1])
            gpad = jnp.concatenate(
                [g, jnp.zeros(g.shape[:3] + (1,), g.dtype)], axis=-1)
            ci, _, pfc = cyl
            idx = jnp.clip(ci, 0, ngk).reshape(nk, 1, 1, -1)
            cyl_v = jnp.take_along_axis(gpad, idx, axis=-1).reshape(
                nk, g.shape[1], ns, n_col, -1)             # (k, b_p, s, col, n_s)
            ph = jnp.asarray(ph_np)[s]                     # (n_s, P·n_pg)
            F = jnp.einsum('kbscj,jp->kbspc', cyl_v, ph)    # all ranks' planes
            F = jax.lax.all_to_all(F, _XY, split_axis=3, concat_axis=1,
                                   tiled=True)             # (k, bc_w, s, n_pg, col)
            F = jnp.concatenate(
                [F, jnp.zeros(F.shape[:4] + (1,), F.dtype)], axis=-1)
            stack = jnp.take(F, pfc, axis=-1)              # (k, bc_w, s, n_pg, ps)
            r = local_ifftn3(stack.reshape(*stack.shape[:4], n_b, n_c),
                             axes=(-2, -1), norm=norm2)
            r = r.reshape(nk, bc_w, ns, rb.r_s)
            return apply_bloch_phase_at(r, kvecs, fft_grid, pts_p[s])

        def sub_body(rows, s):
            def band_body(carry, bc):
                D_l, D_r = carry
                psi = psi_at(bc, s)
                rel = band_rel[bc]
                x = jnp.take(X_B, jnp.clip(rel, 0, nb_face - 1), axis=3)
                yc = jnp.conj(psi)
                D_l = D_l + jnp.einsum(
                    'kamn,knbr->abkmr', x * w_l[bc][None, None, None, :], yc)
                D_r = D_r + jnp.einsum(
                    'kamn,knbr->abkmr', x * w_r[bc][None, None, None, :], yc)
                return (D_l, D_r), None

            z0 = jnp.zeros((ns, ns, nk, b, rb.r_s), dtype=jnp.complex128)
            (D_l, D_r), _ = jax.lax.scan(
                band_body, (z0, z0), jnp.arange(n_bc, dtype=jnp.int32),
                unroll=1)
            Z = _kconv_tail(D_l, D_r, kgrid, vertex)        # (nk, b, r_s)
            if q_neg is not None:
                Z = Z + jnp.conj(jnp.take(Z, jnp.asarray(q_neg), axis=0))
            Z = jnp.take(Z, jnp.asarray(q_sel), axis=0)     # (Q, b, r_s)
            Zt = jax.lax.all_to_all(Z, _XY, split_axis=1, concat_axis=2,
                                    tiled=True)             # (Q, c, P·r_s)
            Zt = Zt.reshape(Q, c, P_, 1, rb.r_s)
            return jax.lax.dynamic_update_slice_in_dim(rows, Zt, s, axis=3), None

        rows = jnp.zeros((Q, c, P_, rb.n_sub, rb.r_s), dtype=jnp.complex128)
        rows, _ = jax.lax.scan(sub_body, rows,
                               jnp.arange(rb.n_sub, dtype=jnp.int32), unroll=1)
        rows = rows.reshape(Q * c, P_ * rb.R)
        q_row_all = jnp.arange(Q * c, dtype=jnp.int32) // c
        pad_rows = n_fft * cs - n_rows
        if pad_rows:
            rows = jnp.pad(rows, ((0, pad_rows), (0, 0)))
            q_row_all = jnp.pad(q_row_all, (0, pad_rows))
        phx_d, phy_d, phz_d = (jnp.asarray(phx), jnp.asarray(phy),
                               jnp.asarray(phz))
        sph = jnp.asarray(sphere)

        def fft_body(out, i):
            sub = jax.lax.dynamic_slice_in_dim(rows, i * cs, cs, axis=0)
            qr = jax.lax.dynamic_slice_in_dim(q_row_all, i * cs, cs, axis=0)
            box = transport_to_box(sub, rb)                 # (cs, nx, ny, nz)
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
        return out[:n_rows].reshape(Q, c, ngkmax)

    fn = jax.jit(_local)
    _kernel_cache[key] = fn
    return fn




def batch_slots(mu_pad: int, b: int, beta: int) -> np.ndarray:
    """Packed centroid of each slot of batch ``beta`` (``β·b + slot``; -1 past μ_pad).

    Batches are contiguous in the packed order; slot ``p·c + j`` is owned by
    rank p after the transpose, so rank p's rows are ``β·b + p·c + [0, c)``.
    """
    s = int(beta) * int(b) + np.arange(int(b))
    return np.where(s < int(mu_pad), s, -1).astype(np.int32)


def gather_batch_centroids(psi_mun_full, slots, *, mesh: Mesh) -> jax.Array:
    """``X_B = ψ_{nks}(r_μ)`` for the batch's packed centroids, replicated.

    ``psi_mun_full`` is the full-BZ face ``(nk, ns, μ_pad, nb)``; ``slots``
    ``(b,)`` packed indices, ``-1`` for a pad slot (zero column).  Bytes
    ``nk·ns·b·nb·16`` per rank (VI3, b=100: 0.17 GB).
    """
    key = ('gather_X', _mesh_id(mesh), tuple(int(v) for v in psi_mun_full.shape),
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
    return fn(psi_mun_full, idx_dev)


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
    * ``'device'``: the same rank-local tile on the GPU (the planner's
      shortcut when it fits beside the batch working set at no cost in
      batch size).

    :meth:`read_tile` returns ``Z[:, :, tile]`` either q-local
    ``(Q_pad, μ_pad, G_tile)`` at ``P(('x','y'), None, None)`` or G-split
    ``(Q, μ_pad, G_tile)`` at ``P(None, None, ('x','y'))``; the host/device
    tiers reach that layout with one all-to-all per tile.
    """

    def __init__(self, *, mesh: Mesh, Q: int, mu_pad: int, n_G: int, b: int,
                 g_tile: int, placement: str, scratch_path: str | None = None):
        self.mesh = mesh
        self.P = _mesh_size(mesh)
        self.Q, self.mu_pad, self.n_G = int(Q), int(mu_pad), int(n_G)
        self.b, self.c = int(b), int(b) // self.P
        if self.b % self.P:
            raise ValueError(f"ZStore: batch {b} must be a multiple of P={self.P}")
        self.g_tile = int(g_tile)
        self.n_Gt = -(-self.n_G // self.g_tile)
        self.n_batch = -(-self.mu_pad // self.b)
        self.Q_pad = -(-self.Q // self.P) * self.P
        self.placement = str(placement)
        self.bytes_written = 0
        self.bytes_read = 0
        self.t_write = 0.0
        self.t_read = 0.0
        local_shape = (self.n_Gt, self.Q, self.n_batch * self.c, self.g_tile)
        if self.placement == 'device':
            self._dev = jax.jit(
                lambda: jnp.zeros((self.n_Gt, self.Q, self.P * self.n_batch
                                   * self.c, self.g_tile), jnp.complex128),
                out_shardings=NamedSharding(mesh, P(None, None, _XY, None)))()
        elif self.placement == 'host':
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

    @property
    def device_bytes_per_rank(self) -> int:
        """Persistent device bytes this store holds (0 off device)."""
        if self.placement != 'device':
            return 0
        return self.n_Gt * self.Q * self.n_batch * self.c * self.g_tile * 16

    # -- write ------------------------------------------------------------
    def write_batch(self, beta: int, rows: jax.Array) -> None:
        """``rows (Q, b, n_Gt·G_tile)`` at ``P(None, ('x','y'), None)`` for batch ``beta``."""
        t0 = time.perf_counter()
        tiled = _tile_rows(self.mesh, self.Q, self.b, self.n_Gt, self.g_tile)(rows)
        if self.placement == 'device':
            self._dev = _device_batch_update(self.mesh, self.c)(
                self._dev, tiled, jnp.int32(beta))
        elif self.placement == 'host':
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
        if self.placement == 'disk':
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
            out = _disk_tile(self.mesh, layout, self.mu_pad)(raw)
        else:
            if self.placement == 'device':
                local = _device_take_tile(self.mesh)(self._dev, jnp.int32(t))
            else:
                arrays = [jax.device_put(self._host[dev.id][t], dev)
                          for dev in self.mesh.local_devices]
                local = jax.make_array_from_single_device_arrays(
                    (self.Q, self.P * self.n_batch * self.c, self.g_tile),
                    NamedSharding(self.mesh, P(None, _XY, None)), arrays)
            out = _rows_to_layout(self.mesh, layout, self.P, self.n_batch,
                                  self.c, self.mu_pad, self.Q_pad)(local)
        jax.block_until_ready(out)
        self.bytes_read += self.Q * self.mu_pad * self.g_tile * 16
        self.t_read += time.perf_counter() - t0
        return out

    def close(self) -> None:
        if self.placement == 'device':
            self._dev = None
        elif self.placement == 'host':
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
        return (f"Z store: placement={self.placement}, (Q={self.Q}, μ={self.mu_pad}, "
                f"N_G={self.n_G}) as {self.n_Gt} tiles x {self.g_tile}, "
                f"{self.n_batch} batches of {self.b}; written "
                f"{self.bytes_written / 1e9:.2f} GB in {self.t_write:.2f} s, "
                f"read {self.bytes_read / 1e9:.2f} GB in {self.t_read:.2f} s "
                f"(global volumes; device resident "
                f"{self.device_bytes_per_rank / 1e9:.2f} GB/rank)")


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


def _device_batch_update(mesh, c):
    key = ('batch_update', _mesh_id(mesh), int(c))
    fn = _kernel_cache.get(key)
    if fn is None:
        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, None, _XY, None), P(None, None, _XY, None), P()),
                 out_specs=P(None, None, _XY, None), check_vma=False)
        def _upd(store, tiled, beta):
            return jax.lax.dynamic_update_slice_in_dim(store, tiled, beta * c,
                                                       axis=2)
        fn = jax.jit(_upd, donate_argnums=(0,))
        _kernel_cache[key] = fn
    return fn


def _device_take_tile(mesh):
    key = ('take_tile', _mesh_id(mesh))
    fn = _kernel_cache.get(key)
    if fn is None:
        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, None, _XY, None), P()),
                 out_specs=P(None, _XY, None), check_vma=False)
        def _t(store, t):
            return jax.lax.dynamic_index_in_dim(store, t, axis=0, keepdims=False)
        fn = jax.jit(_t)
        _kernel_cache[key] = fn
    return fn


def _rows_to_layout(mesh, layout, P_, n_batch, c, mu_pad, Q_pad):
    """Rank-local rows ``(Q, P·n_batch·c, g)`` (μ in rank-major order) → q-local or G-split."""
    key = ('rows_to_layout', _mesh_id(mesh), layout, P_, n_batch, c, mu_pad, Q_pad)
    fn = _kernel_cache.get(key)
    if fn is None:
        out_spec = P(_XY, None, None) if layout == 'q' else P(None, None, _XY)

        @partial(shard_map, mesh=mesh, in_specs=(P(None, _XY, None),),
                 out_specs=out_spec, check_vma=False)
        def _f(x):                                        # (Q, n_batch·c, g)
            if layout == 'q':
                Q = x.shape[0]
                if Q_pad > Q:
                    x = jnp.pad(x, ((0, Q_pad - Q), (0, 0), (0, 0)))
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

_LOCAL_KINDS = ('replicated_rank_truncate', 'replicated_cholesky',
                'sharded_cholesky')


def zeta_shell_slots(gvec_components, ngk_per_q, q_frac, bvec, q_full_frac):
    """Per stored q, the sphere slots with |q+G| ≤ max_q' |q'| (the G≈0 shell).

    Every parent G a full-zone literal G=0 unfolds from, and every Coulomb
    head slot (argmin |q+G|), lies in it: |q_p + G_p| = |q_full| under an
    orthogonal operation.  Slot 0 (G = 0) comes first.  Returns host
    ``(slots (Q, n_shell) int32, gvec (Q, 3, n_shell) int32)``; pad slots
    repeat slot 0 in ``slots`` and carry the FFT-box sentinel of the first
    pad column (or G = 0 when the sphere has none) in ``gvec``.
    """
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
            pad_col = gv[q][:, ngk[q]] if ngk[q] < gv.shape[2] else gv[q][:, 0]
            g_out[q, :, len(l):] = pad_col[:, None]
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
        return (self.zeta_gather == 'local' and self.solver_kind in _LOCAL_KINDS
                and not hasattr(self.L_q, 'nbatch'))

    # -- the one pass ---------------------------------------------------
    def contract_v(self, v_table, *, zeta_io=None, print_fn=print):
        """Stream every G tile once; return V (Q, μ_pad, μ_pad) at ``P(None,'x','y')``.

        ``v_table`` is ``(Q, ngkmax)`` v(q+G) on the stored sphere.  V and
        the shell come back in the canonical (file) centroid order.  With
        ``zeta_io`` the masked ζ tiles are also written to ``zeta_q_G``.
        """
        t0 = time.perf_counter()
        st = self.store
        v = np.zeros((st.Q_pad, st.n_Gt * st.g_tile), dtype=np.complex128)
        v[:st.Q, :self.ngkmax] = np.asarray(v_table, dtype=np.complex128)
        ngk = np.zeros((st.Q_pad,), np.int32)
        ngk[:st.Q] = self.ngk_per_q
        sl = np.zeros((st.Q_pad, self.shell_slots.shape[1]), np.int32)
        sl[:st.Q] = self.shell_slots
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
            v_dev = device_put_process_local(v[:st.Q], rep)
            ngk_dev = device_put_process_local(ngk[:st.Q], rep)
            sl_dev = device_put_process_local(sl[:st.Q], rep)
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
        for t in range(st.n_Gt):
            Zt = st.read_tile(t, layout=layout)
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


def _debug_enabled() -> bool:
    from runtime import debug_print_enabled
    return bool(debug_print_enabled())


def _logical_solve(solver_kind: str, n_log: int):
    from isdf.core import _zeta_logical_solvers
    (_ridge, _lu, tri, pinv, _pinvT) = _zeta_logical_solvers(int(n_log))
    if solver_kind == 'replicated_rank_truncate':
        return pinv
    if solver_kind in ('replicated_cholesky', 'sharded_cholesky'):
        return tri
    raise ValueError(
        f"GATE zeta-mubatch-v-route: got solver kind {solver_kind!r}, want one "
        f"of {_LOCAL_KINDS}; why: the streamed V applies the per-q whole-tile "
        "factor itself.  Fix: the replicated/local charge tiers (the default).")


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
