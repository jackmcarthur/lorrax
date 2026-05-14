"""V_q orchestrator for the G-flat ζ on-disk format.

This is the post-G-flat rewrite of the V_q hot loop.  The legacy
:func:`gw.v_q_tile.compute_V_q_tile` is replaced wholesale when the
on-disk ζ is in WFN.h5-style per-q sphere layout — most of its
complexity (Case A/B chooser, ``μ × ν`` tiling, in-kernel FFT, shared
sphere conversion) goes away because:

* ``ζ̃`` already lives on the per-q sphere on disk (no FFT here).
* The contract chunks over **G** (a fixed-cost reduction axis), not μ
  / ν — one G-chunk is a small GEMM, and the V[μ,ν] output is the
  whole problem at once.
* One q at a time; q-batching can come back as an outer vmap if a
  future profile shows the per-q launch latency dominating.

Async I/O — kept from the legacy driver — is the only orchestration
trick we keep: a worker thread reads ζ̃_{q+1} while the compute thread
contracts ζ̃_q.  At per-q read size ``n_rmu × ngkmax × 16 B`` (typical
MoS2 3×3 ~50 MB) the overlap matters more than the chooser/tiling
machinery.

Math:

    V_q[μ, ν] = Σ_G  conj(ζ̃_{q,μ}(G)) · v(q+G) · ζ̃_{q,ν}(G)
    g0_μ(q)   = ζ̃_{q,μ}(G=0)               # = ζ̃[μ, 0] by sphere convention

IBZ unfold runs post-loop via the existing helpers in ``v_q_tile``
(``_unfold_v_q_ibz_to_full``, ``_unfold_g0_ibz_to_full``); the V_q
output sharding ``P(None, 'x', 'y')`` matches.
"""
from __future__ import annotations

import queue
import threading
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

if TYPE_CHECKING:
    from file_io.zeta_loader import ZetaLoader
    from file_io.zeta_reader import ZetaReader


# ---------------------------------------------------------------------------
# Inner kernel: ζ_q (μ, G_padded) + v_q (G_padded,) → V_q (μ, μ) at P('x','y')
# ---------------------------------------------------------------------------

_PER_Q_KERNEL_CACHE: dict = {}


def _make_per_q_kernel(mesh_xy: Mesh, n_rmu_L: int, n_rmu_R: int,
                       ngkmax: int, g_chunk: int,
                       *, write_g0: bool, same_zeta: bool):
    """Compile-once kernel for the per-q contract + dynamic_update_slice
    into the (V_acc, g0_acc) buffers.

    Single signature handles both:
      * Charge / diagonal bispinor tiles: ``same_zeta=True``; caller
        passes ``zeta_R_q is zeta_L_q`` and the kernel re-shards one
        buffer for the two operands of the einsum.
      * Bispinor off-diagonal tiles: ``same_zeta=False``; caller passes
        two separate buffers (potentially different ``n_rmu_*``).

    Returns ``fn(V_acc, g0_acc, zeta_L_q, zeta_R_q, v_q, q_idx)
              -> (V_new, g0_new)``.
    Donates the two accumulators so the per-q update is in-place.
    """
    key = (id(mesh_xy), int(n_rmu_L), int(n_rmu_R), int(ngkmax),
           int(g_chunk), bool(write_g0), bool(same_zeta))
    hit = _PER_Q_KERNEL_CACHE.get(key)
    if hit is not None:
        return hit

    blk_xy_sh = NamedSharding(mesh_xy, P(('x', 'y'), None))
    blk_x_sh = NamedSharding(mesh_xy, P('x', None))
    blk_y_sh = NamedSharding(mesh_xy, P('y', None))
    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    V_block_sh = NamedSharding(mesh_xy, P('x', 'y'))
    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))
    g0_block_sh = NamedSharding(mesh_xy, P('x'))
    v_sh = NamedSharding(mesh_xy, P(None))

    n_chunks = ngkmax // g_chunk

    @partial(jax.jit, donate_argnums=(0, 1))
    def fn(V_acc, g0_acc, zeta_L_q, zeta_R_q, v_q, q_idx):
        # zeta_*_q come in as (1, n_rmu_*, ngkmax) from per-q reads;
        # drop the q axis.  In the same_zeta case ``zeta_R_q is
        # zeta_L_q`` (caller-aliased); we still take separate views
        # so the two reshardings below are explicit.
        zeta_L_3d = jax.lax.with_sharding_constraint(zeta_L_q, blk_xy_sh)
        if same_zeta:
            zeta_R_3d = zeta_L_3d
        else:
            zeta_R_3d = jax.lax.with_sharding_constraint(zeta_R_q, blk_xy_sh)
        zeta_L = jax.lax.with_sharding_constraint(zeta_L_3d[0], blk_x_sh)
        zeta_R = jax.lax.with_sharding_constraint(zeta_R_3d[0], blk_y_sh)
        v_q = jax.lax.with_sharding_constraint(v_q, v_sh)

        V_q = jnp.zeros((n_rmu_L, n_rmu_R), dtype=zeta_L.dtype)
        V_q = jax.lax.with_sharding_constraint(V_q, V_block_sh)

        # G-chunked accumulation via ``lax.scan`` — compiles once and
        # executes n_chunks times.  Replaces the historical static
        # Python loop, which unrolled the HLO ``n_chunks ×`` and grew
        # compile time linearly with the system size (CrI3 6×6 80 Ry
        # has n_chunks ~ 14; MoS2 3×3 has 1).  See module docstring
        # for the math: ``V[μ,ν] += conj(L_chunk) · v · R_chunkᵀ``.
        def _g_chunk_body(V_carry, i):
            start = i * g_chunk
            L_chunk = jax.lax.dynamic_slice_in_dim(
                zeta_L, start, g_chunk, axis=-1)        # (n_rmu_L/p_x, g_chunk)
            R_chunk = jax.lax.dynamic_slice_in_dim(
                zeta_R, start, g_chunk, axis=-1)        # (n_rmu_R/p_y, g_chunk)
            v_chunk = jax.lax.dynamic_slice_in_dim(
                v_q, start, g_chunk, axis=0)            # (g_chunk,)
            L_w = jnp.conj(L_chunk) * v_chunk[None, :]
            return V_carry + L_w @ R_chunk.T, None
        V_q, _ = jax.lax.scan(
            _g_chunk_body, V_q, jnp.arange(n_chunks, dtype=jnp.int32))
        V_q = jax.lax.with_sharding_constraint(V_q, V_block_sh)

        q_idx_32 = q_idx.astype(jnp.int32)
        zero32 = jnp.int32(0)
        V_new = jax.lax.dynamic_update_slice(
            V_acc, V_q[None, :, :], (q_idx_32, zero32, zero32))
        V_new = jax.lax.with_sharding_constraint(V_new, V_sh)

        if write_g0:
            g0_q = zeta_L[:, 0]                          # (n_rmu_L/p_x,)
            g0_q = jax.lax.with_sharding_constraint(g0_q, g0_block_sh)
            g0_new = jax.lax.dynamic_update_slice(
                g0_acc, g0_q[None, :], (q_idx_32, zero32))
            g0_new = jax.lax.with_sharding_constraint(g0_new, g0_sh)
        else:
            g0_new = g0_acc

        return V_new, g0_new

    _PER_Q_KERNEL_CACHE[key] = fn
    return fn


# ---------------------------------------------------------------------------
# Small shared helpers (used by both the charge wrapper and the bispinor
# tile loop in gw.v_q_bispinor)
# ---------------------------------------------------------------------------

def _resolve_ibz_q_list(*, sym, centroid_indices, kgrid, fft_grid, verbose):
    """Pick IBZ q's via centroid orbit closure, fall back to full BZ.

    Returns ``(q_irr_kgrid_int, q_irr_frac, q_full_to_irr_idx,
    q_full_to_irr_sym, sym_perm, use_ibz)``.  When ``use_ibz`` is
    False the *_idx / *_sym / sym_perm fields are None; caller skips
    the post-loop unfold.
    """
    nkx, nky, nkz = kgrid
    use_ibz = False
    q_irr_kgrid_int = None
    q_full_to_irr_idx = None
    q_full_to_irr_sym = None
    sym_perm = None
    if sym is not None and centroid_indices is not None:
        from centroid.orbit_syms import compute_centroid_sym_perm
        n_tran = int(np.asarray(sym.sym_matrices).shape[0])
        cent_idx = np.asarray(centroid_indices, dtype=np.int32)
        try:
            # ``extend_trs=True`` so ``sym_perm`` has shape
            # ``(2·n_tran, n_rmu)`` — the second half duplicates the
            # spatial rows and is indexed by the TRS-augmented sym
            # values returned by ``sym.irr_idx_q/sym_idx_q``.  See
            # ``compute_centroid_sym_perm`` docstring and the audit
            # report (``reports/trs_sym_audit_2026-05-14``).
            sym_perm = compute_centroid_sym_perm(
                cent_idx,
                sym_matrices=np.asarray(sym.sym_matrices[:n_tran]),
                translations=np.asarray(sym.translations[:n_tran]),
                fft_grid=np.asarray(fft_grid, dtype=np.int32),
                extend_trs=True,
            )
        except RuntimeError as exc:
            if verbose and jax.process_index() == 0:
                print(f"  V_q g-flat: centroid orbit closure failed — "
                      f"full-BZ iteration.  Reason: "
                      f"{exc.args[0].splitlines()[0] if exc.args else exc}")
            sym_perm = None
        if sym_perm is not None:
            q_irr_kgrid_int = sym.q_irr_kgrid_int
            q_full_to_irr_idx = sym.irr_idx_q
            q_full_to_irr_sym = sym.sym_idx_q
            use_ibz = True

    if not use_ibz:
        q_irr_kgrid_int = np.array(
            [(qx, qy, qz) for qx in range(nkx)
             for qy in range(nky) for qz in range(nkz)],
            dtype=np.int32)

    # BGW wrap: q > kg/2 → q - kg.  Same convention the writer used
    # when building the per-q gvec_components on disk.
    kg_arr = np.asarray(kgrid, dtype=np.float64)
    q_irr_wrapped = np.where(
        q_irr_kgrid_int > kg_arr / 2,
        q_irr_kgrid_int - kg_arr,
        q_irr_kgrid_int).astype(np.float64)
    q_irr_frac = q_irr_wrapped / kg_arr
    return (q_irr_kgrid_int, q_irr_frac,
            q_full_to_irr_idx, q_full_to_irr_sym, sym_perm, use_ibz)


def _pick_g_chunk(ngkmax: int, target: int = 4096) -> int:
    """Largest divisor of ``ngkmax`` that is ≤ ``target``."""
    for c in range(min(target, int(ngkmax)), 0, -1):
        if ngkmax % c == 0:
            return int(c)
    return int(ngkmax)


def _make_read_q(zeta_loader, n_rmu_padded: int, mesh_xy: Mesh):
    """Return a ``read_q(q_idx) -> (1, n_rmu_padded, ngkmax)`` callable.

    Both :class:`ZetaLoader` (test bench) and :class:`ZetaReader`
    (production) can serve a G-flat slab; their call signatures
    differ.  Detect which one we have once, then dispatch.

    Also exposes ``read_all_ibz(n_q_ibz) -> (n_q_ibz, n_rmu_padded, ngkmax)``
    as a single-call batched variant.  Used by the V_q tile orchestrator
    to read the whole IBZ slab once at the start, avoiding the
    ``n_q_ibz`` separate ``read_slab`` calls (each of which produces a
    distinct ``_FfiBackend.read_slab.<locals>._per_rank`` closure id and
    triggers a JAX trace cache miss).
    """
    has_load = callable(getattr(zeta_loader, 'load', None))
    has_read_slab = callable(getattr(zeta_loader, 'read_zeta_G_slab', None))
    if not (has_load or has_read_slab):
        raise TypeError(
            "v_q_g_flat: zeta_loader must expose .load() (ZetaLoader) "
            "or .read_zeta_G_slab() (ZetaReader); got "
            f"{type(zeta_loader).__name__} with neither.")
    n_rmu_logical = int(zeta_loader.n_rmu)
    spec = P(None, ('x', 'y'), None)

    def read_q(q_idx: int) -> jax.Array:
        if has_load:
            return zeta_loader.load(
                q=[int(q_idx)], layout='G_flat', sharding=spec)
        # ZetaReader path: per-q phase is already baked in on disk;
        # pass a dummy qvec for the unused ``qvec_batch_frac`` arg.
        return zeta_loader.read_zeta_G_slab(
            q_offset=int(q_idx), q_count=1,
            mu_offset=0, mu_count=int(n_rmu_padded),
            qvec_batch_frac=jnp.zeros((1, 3), dtype=jnp.float64),
            sphere_idx=None,
            mesh=mesh_xy, valid_mu=n_rmu_logical,
        )

    def read_all_ibz(n_q_ibz: int) -> jax.Array:
        """Single batched read of all IBZ q's → ``(n_q_ibz, n_rmu_padded, ngkmax)``."""
        if has_load:
            return zeta_loader.load(
                q=list(range(int(n_q_ibz))), layout='G_flat', sharding=spec)
        return zeta_loader.read_zeta_G_slab(
            q_offset=0, q_count=int(n_q_ibz),
            mu_offset=0, mu_count=int(n_rmu_padded),
            qvec_batch_frac=jnp.zeros((int(n_q_ibz), 3), dtype=jnp.float64),
            sphere_idx=None,
            mesh=mesh_xy, valid_mu=n_rmu_logical,
        )

    # Attach the batched variant onto ``read_q`` so callers can pick.
    read_q.read_all_ibz = read_all_ibz
    return read_q


# ---------------------------------------------------------------------------
# Per-tile core (one (μ_L, ν_L) tile)
# ---------------------------------------------------------------------------

def _compute_V_q_g_flat_one_tile(
    zeta_L_loader,
    zeta_R_loader,                     # None ⇒ same_zeta=True
    *,
    v_per_G_builder,                   # callable(q_irr_frac, gvec_components) -> (n_q, ngkmax) c128
    kgrid, fft_grid, bvec, mesh_xy,
    n_rmu_L: int,
    n_rmu_R: int,
    g_chunk: int | None,
    sym, centroid_indices,             # IBZ closure check is on the L centroids
    write_g0: bool,
    timing_label: str,
    verbose: bool,
) -> tuple[jax.Array, jax.Array | None]:
    """Compute one bispinor / scalar (μ_L, ν_L) tile end-to-end.

    Loops q-by-q over the IBZ (or full BZ), reads ζ_L (and ζ_R if
    distinct) from G-flat disk, contracts via the per-q + G-chunked
    kernel into a single ``(n_q_ibz, n_rmu_L_padded, n_rmu_R_padded)``
    buffer, and post-loop unfolds IBZ → full-BZ via the existing
    centroid double-permute (V_q is bilinear in ζ; no τ phase).

    Returns ``(V_qmunu_full_BZ, g0_or_None)`` at the production
    shardings ``P(None, 'x', 'y')`` and ``P(None, 'x')``.
    """
    same_zeta = (zeta_R_loader is None) or (zeta_R_loader is zeta_L_loader)
    if str(getattr(zeta_L_loader, 'zeta_layout', '')) != 'G_flat':
        raise ValueError(
            f"_compute_V_q_g_flat_one_tile[{timing_label}]: zeta_L "
            f"layout must be 'G_flat'; got "
            f"{getattr(zeta_L_loader, 'zeta_layout', None)!r}")
    if (not same_zeta and
            str(getattr(zeta_R_loader, 'zeta_layout', '')) != 'G_flat'):
        raise ValueError(
            f"_compute_V_q_g_flat_one_tile[{timing_label}]: zeta_R "
            "layout must be 'G_flat'.")

    from .v_q_tile import (_unfold_v_q_ibz_to_full,
                            _unfold_g0_ibz_to_full)

    # ---- IBZ list + per-tile v(q+G) -----------------------------------
    (_q_int, q_irr_frac,
     full_to_irr_idx, full_to_irr_sym,
     sym_perm, use_ibz) = _resolve_ibz_q_list(
        sym=sym, centroid_indices=centroid_indices,
        kgrid=kgrid, fft_grid=fft_grid, verbose=verbose)
    n_q_ibz = int(q_irr_frac.shape[0])

    gvec_components = np.asarray(
        zeta_L_loader.gvec_components, dtype=np.int32)
    if gvec_components.shape[0] != n_q_ibz:
        raise ValueError(
            f"_compute_V_q_g_flat_one_tile[{timing_label}]: ζ_L on "
            f"disk has {gvec_components.shape[0]} q's; resolved IBZ "
            f"has {n_q_ibz}.  Mismatch — was the file written with the "
            f"same write_ibz_only setting?")
    if not same_zeta:
        gvec_R = np.asarray(zeta_R_loader.gvec_components, dtype=np.int32)
        if gvec_R.shape != gvec_components.shape:
            raise ValueError(
                f"_compute_V_q_g_flat_one_tile[{timing_label}]: ζ_L vs "
                f"ζ_R gvec_components shape mismatch "
                f"({gvec_components.shape} vs {gvec_R.shape}).  Both "
                f"files must be written with matching zeta_cutoff_ry "
                f"and q-layout.")
    ngkmax = int(gvec_components.shape[-1])

    v_q_table = np.asarray(
        v_per_G_builder(q_irr_frac, gvec_components),
        dtype=np.complex128)                                # (n_q_ibz, ngkmax)
    if v_q_table.shape != (n_q_ibz, ngkmax):
        raise ValueError(
            f"_compute_V_q_g_flat_one_tile[{timing_label}]: "
            f"v_per_G_builder returned shape {v_q_table.shape}; "
            f"expected ({n_q_ibz}, {ngkmax}).")

    g_chunk = int(g_chunk) if g_chunk else _pick_g_chunk(ngkmax)
    if ngkmax % g_chunk != 0:
        raise ValueError(
            f"_compute_V_q_g_flat_one_tile[{timing_label}]: g_chunk="
            f"{g_chunk} does not divide ngkmax={ngkmax}.")
    n_chunks = ngkmax // g_chunk

    # ---- μ padding to mesh-product per side ---------------------------
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])
    _proc = p_x * p_y
    def _pad(n: int) -> int:
        return n + (_proc - n % _proc) % _proc
    n_rmu_L_padded = _pad(int(n_rmu_L))
    n_rmu_R_padded = _pad(int(n_rmu_R))
    if verbose and jax.process_index() == 0:
        print(f"  V_q g-flat [{timing_label}]: n_q_ibz={n_q_ibz}, "
              f"ngkmax={ngkmax}, g_chunk={g_chunk} ({n_chunks}/q), "
              f"n_rmu_L={n_rmu_L}→{n_rmu_L_padded}, "
              f"n_rmu_R={n_rmu_R}→{n_rmu_R_padded}, "
              f"unfold={'IBZ→full' if use_ibz else 'full-BZ'}",
              flush=True)

    # ---- Accumulators + v_q on device --------------------------------
    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    V_acc = jax.jit(lambda: jnp.zeros(
        (n_q_ibz, n_rmu_L_padded, n_rmu_R_padded), dtype=jnp.complex128),
        out_shardings=V_sh)()
    if write_g0:
        g0_sh = NamedSharding(mesh_xy, P(None, 'x'))
        g0_acc = jax.jit(lambda: jnp.zeros(
            (n_q_ibz, n_rmu_L_padded), dtype=jnp.complex128),
            out_shardings=g0_sh)()
    else:
        # The jit kernel still needs a buffer to donate; allocate a
        # tiny placeholder we won't read.
        g0_sh = NamedSharding(mesh_xy, P(None, 'x'))
        g0_acc = jax.jit(lambda: jnp.zeros(
            (n_q_ibz, n_rmu_L_padded), dtype=jnp.complex128),
            out_shardings=g0_sh)()
    v_q_dev = jax.device_put(
        v_q_table, NamedSharding(mesh_xy, P(None, None)))

    kernel = _make_per_q_kernel(
        mesh_xy, n_rmu_L_padded, n_rmu_R_padded, ngkmax, g_chunk,
        write_g0=write_g0, same_zeta=same_zeta)

    read_L = _make_read_q(zeta_L_loader, n_rmu_L_padded, mesh_xy)
    read_R = (read_L if same_zeta
              else _make_read_q(zeta_R_loader, n_rmu_R_padded, mesh_xy))

    # ---- Pre-read all IBZ ζ̃ slabs in ONE batched call ---------------
    # The historical per-q PHDF5 read inside the kernel loop interleaved
    # with NCCL collectives and was the root cause of the async-prefetch
    # deadlock.  At MoS2 3×3 the full ζ̃_L is ~50 MB / rank; at CrI3 6×6
    # 80 Ry it's ~0.8 GB / rank — both comfortable.
    #
    # 2026-05-12: switched from ``concatenate([read_L(q) for q in ...])``
    # (n_q_ibz separate ``read_slab`` calls; each one created a fresh
    # ``_per_rank`` closure and triggered a JAX trace-cache miss in the
    # FFI shard_map dispatch) to ONE batched ``read_all_ibz`` call.
    # Net effect on MoS2 3×3 bispinor: 63 read_slab calls (9 q × 7 tiles)
    # → 7 calls; the corresponding ``jit__per_rank`` retraces drop from
    # ~63 to 7.
    import time as _t
    _read_t0 = _t.perf_counter()
    zeta_L_all = read_L.read_all_ibz(n_q_ibz)               # (n_q_ibz, n_rmu_L_padded, ngkmax)
    if same_zeta:
        zeta_R_all = zeta_L_all
    else:
        zeta_R_all = read_R.read_all_ibz(n_q_ibz)
    jax.block_until_ready(zeta_L_all)
    if not same_zeta:
        jax.block_until_ready(zeta_R_all)
    _read_total = _t.perf_counter() - _read_t0
    if verbose and jax.process_index() == 0:
        print(f"    [{timing_label}] pre-read all {n_q_ibz} IBZ ζ̃ slabs "
              f"(1 batched call): {_read_total:.2f}s", flush=True)

    # ---- Sync per-q kernel loop on device-resident ζ̃ ---------------
    for q in range(n_q_ibz):
        _t1 = _t.perf_counter()
        zeta_L_q = jax.lax.dynamic_slice_in_dim(
            zeta_L_all, q, 1, axis=0)                       # (1, n_rmu_L_padded, ngkmax)
        zeta_R_q = (zeta_L_q if same_zeta
                    else jax.lax.dynamic_slice_in_dim(
                        zeta_R_all, q, 1, axis=0))
        V_acc, g0_acc = kernel(
            V_acc, g0_acc, zeta_L_q, zeta_R_q, v_q_dev[q], jnp.int32(q))
        jax.block_until_ready(V_acc)
        _t_k = _t.perf_counter() - _t1
        if verbose and jax.process_index() == 0:
            print(f"    [{timing_label}] q={q}/{n_q_ibz}: "
                  f"kernel={_t_k:.2f}s", flush=True)
    del zeta_L_all
    if not same_zeta:
        del zeta_R_all

    # ---- IBZ → full-BZ unfold (centroid double-permute) -------------
    if use_ibz:
        # ``sym_perm`` came from ``compute_centroid_sym_perm(..., extend_trs=True)``
        # so its shape[0] is ``2·ntran``; the second half encodes the
        # TRS-augmented rows (centroid permutation unchanged under TRS,
        # but the unfold helper conjugates V_q at TRS-tagged q's).
        n_sym_spatial = int(np.asarray(sym_perm).shape[0]) // 2
        V_acc = _unfold_v_q_ibz_to_full(
            V_acc, full_to_irr_idx=full_to_irr_idx,
            full_to_irr_sym=full_to_irr_sym,
            sym_perm=sym_perm, mesh_xy=mesh_xy,
            n_sym_spatial=n_sym_spatial)
        if write_g0:
            g0_acc = _unfold_g0_ibz_to_full(
                g0_acc, full_to_irr_idx=full_to_irr_idx,
                full_to_irr_sym=full_to_irr_sym,
                sym_perm=sym_perm, mesh_xy=mesh_xy,
                n_sym_spatial=n_sym_spatial)

    V_qmunu = jax.lax.with_sharding_constraint(V_acc, V_sh)
    if write_g0:
        return V_qmunu, jax.lax.with_sharding_constraint(g0_acc, g0_sh)
    return V_qmunu, None


# ---------------------------------------------------------------------------
# Public charge entry point (CC tile only)
# ---------------------------------------------------------------------------

def compute_all_V_q_g_flat(
    zeta_loader,                       # ZetaLoader | ZetaReader (G-flat)
    *,
    kgrid: tuple[int, int, int],
    fft_grid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    mesh_xy: Mesh,
    n_rmu: int,
    sys_dim: int,
    bdot: np.ndarray | None = None,
    bare_coulomb_cutoff_ry: float | None = None,
    bgw_v_grid_fn=None,
    g_chunk: int | None = None,
    verbose: bool = True,
    sym=None,
    centroid_indices: np.ndarray | None = None,
    async_prefetch: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """V_q^{0,0} (charge-channel CC tile) on a G-flat-on-disk ζ file.

    Thin wrapper that builds the bare-Coulomb ``v(q+G)`` per-q-sphere
    builder and dispatches to :func:`_compute_V_q_g_flat_one_tile`
    with ``zeta_R=None`` (same_zeta) and ``write_g0=True``.  The
    ``async_prefetch`` argument is accepted for compatibility with
    the legacy dispatcher's env-var knob but currently has no effect
    — the sync per-q loop is already ~6× faster than the legacy
    μ × ν tile driver on MoS2 3×3.

    See :func:`_compute_V_q_g_flat_one_tile` for the math + I/O flow.
    """
    if sys_dim not in (2, 3):
        raise NotImplementedError(
            f"compute_all_V_q_g_flat: sys_dim must be 2 or 3 "
            f"(0-D box per-q v(G) not wired); got {sys_dim}.")
    from .compute_vcoul import compute_v_q_per_G

    def _bare_v_per_G(q_irr_frac, gvec_components):
        v = compute_v_q_per_G(
            q_irr_frac, gvec_components,
            bvec=bvec, cell_volume=cell_volume,
            sys_dim=sys_dim, vcoul_cutoff_ry=bare_coulomb_cutoff_ry,
            bdot=bdot,
        )                                                   # (n_q_ibz, ngkmax) f64
        # Optional BGW vcoul overlay — host-side scatter from BGW's
        # full-FFT-grid v into the per-q WFN.h5 sphere positions.
        if bgw_v_grid_fn is not None:
            nx, ny, nz = (int(s) for s in fft_grid)
            for qi in range(q_irr_frac.shape[0]):
                v_full = np.asarray(
                    bgw_v_grid_fn(tuple(q_irr_frac[qi]))).reshape(-1)
                miller = gvec_components[qi]                # (3, ngkmax)
                ix = miller[0] % nx
                iy = miller[1] % ny
                iz = miller[2] % nz
                v_at_sphere = v_full[ix * ny * nz + iy * nz + iz]
                v[qi] = np.where(v_at_sphere != 0.0, v_at_sphere, v[qi])
        return v.astype(np.complex128)

    return _compute_V_q_g_flat_one_tile(
        zeta_loader, None,
        v_per_G_builder=_bare_v_per_G,
        kgrid=kgrid, fft_grid=fft_grid, bvec=bvec,
        mesh_xy=mesh_xy,
        n_rmu_L=int(n_rmu), n_rmu_R=int(n_rmu),
        g_chunk=g_chunk,
        sym=sym, centroid_indices=centroid_indices,
        write_g0=True,
        timing_label='CC',
        verbose=verbose,
    )


__all__ = ["compute_all_V_q_g_flat", "_compute_V_q_g_flat_one_tile",
            "_resolve_ibz_q_list", "_pick_g_chunk", "_make_read_q"]
