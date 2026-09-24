"""μ-batch ζ fit, route G: Z_q(G) by centroid batches; see docs/architecture/zeta_fit_mubatch.md.

Per μ batch ``B``, with conj ψ(G) of the full zone sharded over G slots:

    X_B = ψ_{nks}(r_μ)                        one psum of the ranks' partial DFTs
    D̃^X_k(a, μ, b, G) = Σ_n w^X_n X_{nka}(r_μ) conj c_{nkb}(G)   (G-space GEMM, local)
    one all-to-all: G split → μ owner (rank p owns batch slots p·c + [0, c))
    on the owner, per plane group: cylinder → planes → D(k, μ, r_plane);
        Z_q(μ, r) = isdf.core.parent_projector_kconv(D^L, D^R) (identity plan);
        LR+RL completion; e^{-iq·r}, 2D FFT, ζ-sphere columns, axis phase
        accumulate Z_q(μ, G)

``Z_q(G)`` is written once into :class:`ZStore` (pinned host tiles or a
slab_io file); no accumulator over r and no distributed FFT exist.  ``C_q⁺``
acts on μ only, so it is applied afterwards on the sphere, G tile by G tile:
``ζ_q(G) = C_q⁺ Z_q(G)`` (:class:`ZetaG`, through :mod:`isdf.cplus`).
"""
from __future__ import annotations

import math
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.shard_map import shard_map
from common.fft_helpers import local_fftn3
from common.wfn_transforms import _plane_geometry
from runtime.padding import axis_mask, pad_to_axis, padded_axis, strip_axis

_XY = ('x', 'y')
_kernel_cache: dict = {}


def _mesh_size(mesh: Mesh) -> int:
    return int(mesh.shape['x']) * int(mesh.shape['y'])


def _mesh_id(mesh: Mesh) -> tuple:
    return (tuple(mesh.axis_names), tuple(int(v) for v in mesh.devices.shape),
            tuple(int(d.id) for d in mesh.devices.flat))


# ---------------------------------------------------------------------------
# Route G: the pair GEMM in G space, one all-to-all to the μ owners, planes
# on the owner (docs/architecture/zeta_fit_mubatch.md, "Route G")
# ---------------------------------------------------------------------------

def identity_kplan(k_full_frac, centroid_fft_idx, fft_grid, mesh, ns):
    """The one-operation ``CentroidKUnfoldPlan`` on the full zone: every k is
    its own parent, so :func:`isdf.core.parent_projector_kconv` is the plain
    k-convolution (its identity-plan arm, native or XLA)."""
    from types import SimpleNamespace
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from symmetry_maps import spinor_rotation_for_sym_row
    kf = np.asarray(k_full_frac, dtype=np.float64)
    nk = int(kf.shape[0])
    ops = np.eye(3, dtype=np.int64)[None]
    U = np.eye(2, dtype=np.complex128)[None]
    sym = SimpleNamespace(
        sym_matrices=ops, translations=np.zeros((1, 3)),
        irr_idx_k=np.arange(nk, dtype=np.int32), sym_idx_k=np.zeros(nk, np.int32),
        unfolded_kpts=kf, kirr_fullids=np.arange(nk),
        spinor_action=lambda rows, *, nspinor: spinor_rotation_for_sym_row(
            U, np.asarray(rows), 2, nspinor=nspinor, R_cart=ops))
    return build_centroid_k_unfold_plan(sym, np.asarray(centroid_fft_idx), fft_grid,
                                        mesh, nspinor=int(ns), parent_k_frac=kf)


def zeta_plane_tables(gvec_components, ngk_per_q, fft_grid, axis, g_axis):
    """ζ-sphere slot → (in-plane column, axis Miller index) per stored q, at
    the store's G-tile carrier width (pad slots: column 0, index 0, masked
    downstream by ``ngk``)."""
    n_a, (n_b, n_c), (b_ax, c_ax) = _plane_geometry(fft_grid, axis)
    gv = np.asarray(gvec_components, dtype=np.int64)          # (Q, 3, ngk)
    col = (gv[:, b_ax] % n_b) * n_c + gv[:, c_ax] % n_c
    ga = gv[:, axis]
    live = np.arange(gv.shape[-1])[None, :] < np.asarray(ngk_per_q)[:, None]
    col, ga = np.where(live, col, 0), np.where(live, ga, 0)
    return (np.asarray(pad_to_axis(col.astype(np.int32), g_axis, axis=1)),
            np.asarray(pad_to_axis(ga.astype(np.int32), g_axis, axis=1)))


def make_route_g_kernel(*, mesh: Mesh, plan_id, kgrid, fft_grid, ns: int, b: int,
                        q_sel, q_axis, q_neg, qvec_frac, n_col: int, n_s: int,
                        n_pg: int, axis: int):
    """Compile-once executable for one μ batch on route G.

    Returns ``fn(psi_bar, w_l, w_r, kvecs, g3, xmu, live, cyl, zt) -> rows``
    ``(Q, b, N_G)`` at ``P(None, ('x','y'), None)`` (μ-owned; rank p owns the
    batch slots ``p·c + [0, c)``, ``c = b/P``):

    1. ``X_B = ψ(r_μ)`` by a direct DFT of each rank's ψ G slice, one psum;
    2. the pair GEMM in G space on the rank's slice
       (:func:`isdf.pair_kernels.pair_projectors_lr`, stored ``conj c``);
    3. ONE all-to-all, G split → μ owner (``[L | R]`` rows owner-major);
    4. on the owner, per group of ``n_pg`` planes normal to ``axis``: the
       sphere → cylinder gather, the axis DFT onto the planes, the 2D FFT
       and Bloch phase → ``D(k, μ, r_plane)``; the k-convolution
       (``parent_projector_kconv`` on ``plan_id``, the identity plan);
       LR+RL completion; ``e^{-iq·r}``, forward 2D FFT, the ζ-sphere
       columns and the axis phase accumulate ``Z_q(μ, G)``.

    Operands: ``psi_bar (nk, nb, ns, ngk_pad)`` = conj ψ(G) on the full
    zone, G sharded ``P(None, None, None, ('x','y'))``; ``g3 (nk, ngk_pad, 3)``
    Miller index of each slot, same G sharding; ``xmu (b, 3)`` fractional
    coordinates of the batch slots and ``live (b,)`` their mask; ``cyl`` =
    :func:`common.wfn_transforms.psi_cylinder_tables` of the ψ sphere;
    ``zt`` = :func:`zeta_plane_tables`.
    """
    from isdf.core import parent_projector_kconv, _conv_kpair_static_gamma
    from isdf.pair_kernels import pair_projectors_lr
    from ffi.fft import make_fused_conv_kparent
    P_ = _mesh_size(mesh)
    if b % P_:
        raise ValueError(f"make_route_g_kernel: batch {b} must be a multiple of P={P_}")
    c = b // P_
    nk = int(np.prod(kgrid))
    N = int(np.prod(fft_grid))
    n_a, (n_b, n_c), (b_ax, c_ax) = _plane_geometry(fft_grid, axis)
    ps = n_b * n_c
    pl_ax = padded_axis(n_a, int(n_pg), name="route-G plane groups")
    n_grp = pl_ax.carrier // int(n_pg)
    q_sel = np.asarray(q_sel, dtype=np.int32)
    Q = int(q_axis.logical)
    q_neg = np.asarray(q_neg, dtype=np.int32)
    qv = np.asarray(qvec_frac, dtype=np.float64)
    r_pl = int(n_pg) * ps
    p_l, ph_l = _conv_kpair_static_gamma(None, ns)
    pair_kernel = make_fused_conv_kparent(mesh, kgrid, ns, (c, r_pl), perm_l=p_l,
                                          phase_l=ph_l, perm_r=p_l, phase_r=ph_l)
    n_rows = int(np.asarray(plan_id.sym_perm).shape[0])
    l_perm = np.broadcast_to(np.arange(c, dtype=np.int32), (n_rows, c)).copy()
    l_wrap = np.zeros((n_rows, c, 3), np.int32)
    r_perm = np.broadcast_to(np.arange(r_pl, dtype=np.int32), (n_rows, r_pl)).copy()
    r_wrap = np.zeros((n_rows, r_pl, 3), np.int32)
    ib = (np.arange(ps) // n_c).astype(np.float64)
    ic = (np.arange(ps) % n_c).astype(np.float64)
    key = ('route_g', _mesh_id(mesh), id(plan_id), tuple(kgrid), tuple(fft_grid), ns, b,
           hash(q_sel.tobytes()), q_axis, hash(q_neg.tobytes()), hash(qv.tobytes()),
           int(n_col), int(n_s), int(n_pg), int(axis))
    hit = _kernel_cache.get(key)
    if hit is not None:
        return hit

    G_ = P(None, None, None, _XY)

    @partial(shard_map, mesh=mesh,
             in_specs=(G_, P(), P(), P(), P(None, _XY, None), P(), P(), P(), P()),
             out_specs=P(None, _XY, None), check_vma=False)
    def _local(psi_bar, w_l, w_r, kvecs, g3, xmu, live, cyl, zt):
        ci, cax, pfc = cyl
        zcol, zga = zt
        # 1. X_B = ψ_{nks}(r_μ) = Σ_G c e^{2πi(k+G)·x_μ}/√N, one psum
        kg = kvecs[:, None, :] + g3.astype(jnp.float64)            # (k, Gp, 3)
        ph = jnp.exp(2j * jnp.pi * jnp.einsum('kgd,md->kgm', kg, xmu))
        X = jnp.einsum('knsg,kgm->knsm', jnp.conj(psi_bar), ph) / np.sqrt(N)
        X = jax.lax.psum(X, _XY) * live[None, None, None, :]
        # 2. pair GEMM in G space on this rank's slice (one band chunk)
        D_l, D_r = pair_projectors_lr(X[None], lambda bc: psi_bar,
                                      w_l[None], w_r[None])     # (k, s, b, s, Gp)
        # 3. one all-to-all: G split -> μ owners, [L | R] owner-major
        D = jnp.stack([D_l, D_r], axis=2).reshape(nk, ns, 2, P_, c, ns, -1)
        D = jnp.moveaxis(D, 3, 2).reshape(nk, ns, P_ * 2 * c, ns, -1)
        D = jax.lax.all_to_all(D, _XY, split_axis=2, concat_axis=4, tiled=True)
        D = jnp.concatenate([D, jnp.zeros(D.shape[:4] + (1,), D.dtype)], axis=-1)
        ngk1 = int(D.shape[-1])

        def group(acc, gi):
            a0 = gi * n_pg + jnp.arange(n_pg)
            on = (a0 < n_a).astype(jnp.float64)
            pa = jnp.exp(-2j * jnp.pi * cax.astype(jnp.float64)[:, None]
                         * a0[None, :] / n_a) * on[None, :]         # (n_s, n_pg)

            def one_k(Dk, k):
                cy = jnp.take(D[k], jnp.clip(ci[k], 0, ngk1 - 1).reshape(-1), axis=-1)
                cy = cy.reshape(ns, 2 * c, ns, n_col, n_s)
                F = jnp.einsum('asbcj,jp->asbpc', cy, pa)            # (…, n_pg, n_col)
                F = jnp.concatenate([F, jnp.zeros(F.shape[:4] + (1,), F.dtype)], -1)
                st = jnp.take(F, pfc, axis=-1).reshape(ns, 2 * c, ns, n_pg, n_b, n_c)
                d = local_fftn3(st, axes=(-2, -1), norm='backward')  # Σ e^{-iG·r}
                bl = jnp.exp(-2j * jnp.pi * (
                    kvecs[k, axis] * a0[:, None] / n_a
                    + kvecs[k, b_ax] * ib[None, :] / n_b
                    + kvecs[k, c_ax] * ic[None, :] / n_c)) / np.sqrt(N)
                d = d.reshape(ns, 2 * c, ns, n_pg, ps) * bl
                return jax.lax.dynamic_update_index_in_dim(
                    Dk, d.reshape(ns, 2 * c, ns, r_pl), k, axis=0), None

            Dk, _ = jax.lax.scan(
                one_k, jnp.zeros((nk, ns, 2 * c, ns, r_pl), jnp.complex128),
                jnp.arange(nk, dtype=jnp.int32), unroll=1)
            Z = parent_projector_kconv(
                Dk[:, :, :c], Dk[:, :, c:], plan=plan_id, left_perm=l_perm,
                left_L=l_wrap, right_perm=r_perm, right_L=r_wrap, kgrid=kgrid,
                pair_kernel=pair_kernel)                           # (nk, c, r_pl)
            Z = Z + jnp.conj(jnp.take(Z, jnp.asarray(q_neg), axis=0))
            Z = jnp.take(Z, jnp.asarray(q_sel), axis=0).reshape(Q, c, n_pg, ps)
            qin = jnp.exp(-2j * jnp.pi * (jnp.asarray(qv[:, b_ax])[:, None] * ib[None, :] / n_b
                                          + jnp.asarray(qv[:, c_ax])[:, None] * ic[None, :] / n_c))
            Z = (Z * qin[:, None, None, :]).reshape(Q, c, n_pg, n_b, n_c)
            Fz = local_fftn3(Z, axes=(-2, -1), norm='backward').reshape(Q, c, n_pg, ps)
            for j in range(int(n_pg)):
                val = jnp.take_along_axis(Fz[:, :, j], zcol[:, None, :], axis=-1)
                za = jnp.exp(-2j * jnp.pi * (jnp.asarray(qv[:, axis])[:, None]
                                            + zga.astype(jnp.float64)) * a0[j] / n_a) * on[j]
                acc = acc + val * za[:, None, :]
            return acc, None

        acc, _ = jax.lax.scan(
            group, jnp.zeros((Q, c, int(zcol.shape[-1])), jnp.complex128),
            jnp.arange(n_grp, dtype=jnp.int32), unroll=1)
        return acc

    fn = jax.jit(_local)
    _kernel_cache[key] = fn
    return fn


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
    * ``'host'``: pinned tiles ``(Q, b, G_tile)`` per (G tile, batch) in a
      ``file_io.HostTileStore``, sharded like the rows: each rank keeps the
      rows it computed.

    Rows are μ-owned (rank p owns batch slots ``p·c + [0, c)``, ``c = b/P``),
    as route G produces them.

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
                 placement: str, scratch_path: str | None = None,
                 packed_from_slot=None, n_batch: int | None = None):
        """``q_axis``: stored q rows (carrier a multiple of P); ``g_axis``: the
        ζ sphere cut into whole G tiles (divisor = ``G_tile``).  Both are
        ``runtime.padding.PaddedAxis`` records the kernel and the finalize share."""
        self.mesh = mesh
        self.P = _mesh_size(mesh)
        self.q_axis, self.g_axis = q_axis, g_axis
        if q_axis.divisor != self.P:
            raise ValueError(f"ZStore: {q_axis} must divide over P={self.P}")
        self.Q, self.mu_pad, self.n_G = q_axis.logical, int(mu_pad), g_axis.carrier
        self.b, self.c = int(b), int(b) // self.P
        if self.b % self.P:
            raise ValueError(f"ZStore: batch {b} must be a multiple of P={self.P} "
                             "(each rank owns b/P of its rows)")
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
        if self.placement == 'host':
            # Pinned, tile-major host tiles (file_io.HostTileStore): one
            # (Q, b, G_tile) tile per (G tile, batch), sharded like the rows.
            from file_io.host_tile_store import HostTileStore
            self._hts = HostTileStore(
                mesh=mesh, grid=(self.n_Gt, self.n_batch),
                tile_shape=(self.Q, self.b, self.g_tile), spec=P(None, _XY, None),
                dtype=np.complex128)
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

    # -- write ------------------------------------------------------------
    def write_batch(self, beta: int, rows: jax.Array) -> None:
        """``rows (Q, b, n_Gt·G_tile)`` at ``P(None, ('x','y'), None)`` for batch ``beta``."""
        t0 = time.perf_counter()
        if self.placement == 'host':
            # Asynchronous D2H; the next batch computes meanwhile.
            for t, tile in enumerate(_split_g_tiles(self.mesh, self.n_Gt,
                                                    self.g_tile)(rows)):
                self._hts.put_tile_async((t, int(beta)), tile)
        else:
            tiled = _tile_rows(self.mesh, self.Q, self.b, self.n_Gt, self.g_tile)(rows)
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
            out = _disk_tile(self.mesh, layout, self.n_batch * self.b)(raw)
        else:
            if t == 0:
                self._hts.wait()
            local = _concat_batches(self.mesh, self.n_batch)(
                tuple(self._hts.get_tile_async((t, beta))
                      for beta in range(self.n_batch)))
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
            self._hts.close()
        else:
            self._io.close()
            if jax.process_index() == 0:
                import os
                try:
                    os.unlink(self.path)
                except OSError:
                    pass

    def receipt(self) -> str:
        return (f"Z store: placement={self.placement}, μ-owned rows, "
                f"(Q={self.Q}, μ={self.mu_pad}, "
                f"N_G={self.n_G}) as {self.n_Gt} tiles x {self.g_tile}, "
                f"{self.n_batch} batches of {self.b}; written "
                f"{self.bytes_written / 1e9:.2f} GB in {self.t_write:.2f} s, "
                f"read {self.bytes_read / 1e9:.2f} GB in {self.t_read:.2f} s "
                f"(global volumes)")


def _split_g_tiles(mesh, n_Gt, g_tile):
    """``(Q, b, n_Gt·G_tile)`` at ``P(None, XY, None)`` → the ``n_Gt`` G tiles, same sharding."""
    key = ('split_g_tiles', _mesh_id(mesh), int(n_Gt), int(g_tile))
    fn = _kernel_cache.get(key)
    if fn is None:
        sh = NamedSharding(mesh, P(None, _XY, None))
        fn = jax.jit(lambda r: tuple(
            jax.lax.slice_in_dim(r, t * g_tile, (t + 1) * g_tile, axis=2)
            for t in range(n_Gt)), out_shardings=tuple(sh for _ in range(n_Gt)))
        _kernel_cache[key] = fn
    return fn


def _concat_batches(mesh, n_batch):
    """One G tile's per-batch ``(Q, b, G_tile)`` tiles → rank-local rows in
    (batch, slot) order, ``(Q, P·n_batch·c, G_tile)`` at ``P(None, XY, None)``."""
    key = ('concat_batches', _mesh_id(mesh), int(n_batch))
    fn = _kernel_cache.get(key)
    if fn is None:
        spec = P(None, _XY, None)
        fn = jax.jit(shard_map(lambda ts: jnp.concatenate(ts, axis=1), mesh=mesh,
                               in_specs=(tuple(spec for _ in range(n_batch)),),
                               out_specs=spec, check_vma=False))
        _kernel_cache[key] = fn
    return fn


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
        """The solve tier decides the finalize layout: q-local reads each G
        tile onto q owners (one all-to-all per tile from μ-owned rows)."""
        return self.zeta_gather == 'local'


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
