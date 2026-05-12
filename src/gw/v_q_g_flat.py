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


def _make_per_q_kernel(mesh_xy: Mesh, n_rmu: int, ngkmax: int,
                       g_chunk: int, *, write_g0: bool):
    """Compile-once kernel for the per-q contract + dynamic_update_slice
    into the (V_acc, g0_acc) buffers.

    Returns ``fn(V_acc, g0_acc, zeta_q, v_q, q_idx) -> (V_new, g0_new)``.
    Donates the two accumulators so the per-q update is in-place.
    """
    key = (id(mesh_xy), int(n_rmu), int(ngkmax), int(g_chunk), bool(write_g0))
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
    def fn(V_acc, g0_acc, zeta_q, v_q, q_idx):
        # zeta_q comes in as (1, n_rmu, ngkmax) from a per-q read; drop
        # the q axis so the kernel works on a single tile.
        zeta = zeta_q[0]                                 # (n_rmu, ngkmax)
        zeta = jax.lax.with_sharding_constraint(zeta, blk_xy_sh)

        # Two views for the GEMM-shape einsum:
        #   conj(L)·v  on x-sharded μ_L
        #   R         on y-sharded μ_R
        # Output (μ_L_x, μ_R_y) is the V_q block at P('x','y').
        zeta_L = jax.lax.with_sharding_constraint(zeta, blk_x_sh)
        zeta_R = jax.lax.with_sharding_constraint(zeta, blk_y_sh)
        v_q = jax.lax.with_sharding_constraint(v_q, v_sh)

        V_q = jnp.zeros((n_rmu, n_rmu), dtype=zeta.dtype)
        V_q = jax.lax.with_sharding_constraint(V_q, V_block_sh)

        # G-chunked accumulation.  Static loop over n_chunks (small —
        # ngkmax / g_chunk is O(few) to O(10) — keeps the jit fast).
        # TODO(g-prefetch): a future opt-in could fuse this with an
        # in-jit double-buffer of two G-chunks worth of ζ slices to
        # hide the dynamic_slice latency; current shape is one chunk
        # per kernel iter.
        for i in range(n_chunks):
            start = i * g_chunk
            L_chunk = jax.lax.dynamic_slice_in_dim(
                zeta_L, start, g_chunk, axis=-1)        # (n_rmu_L/p_x, g_chunk)
            R_chunk = jax.lax.dynamic_slice_in_dim(
                zeta_R, start, g_chunk, axis=-1)        # (n_rmu_R/p_y, g_chunk)
            v_chunk = jax.lax.dynamic_slice_in_dim(
                v_q, start, g_chunk, axis=0)            # (g_chunk,)
            L_w = jnp.conj(L_chunk) * v_chunk[None, :]
            V_q = V_q + L_w @ R_chunk.T
        V_q = jax.lax.with_sharding_constraint(V_q, V_block_sh)

        # dynamic_update_slice wants all start_indices in the same int
        # dtype; the spatial axes default to int32 in numpy-like land
        # while the python literal ``0`` is int64.  Force-cast.
        q_idx_32 = q_idx.astype(jnp.int32)
        zero32 = jnp.int32(0)
        V_new = jax.lax.dynamic_update_slice(
            V_acc, V_q[None, :, :], (q_idx_32, zero32, zero32))
        V_new = jax.lax.with_sharding_constraint(V_new, V_sh)

        if write_g0:
            # G=0 lives at sphere position 0 by writer construction.
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
# Public entry point
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
    async_prefetch: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """V_q orchestrator for the G-flat on-disk layout.

    Iterates q one at a time over the writer's IBZ (or full BZ if the
    file was written without IBZ reduction), reads the per-q ζ̃ slab,
    builds ``v(q+G)`` from the on-disk per-q components, contracts via
    :func:`gw.compute_vcoul._v_q_per_q_g_chunked_jit`, and writes the
    result into a single ``(n_q_ibz, n_rmu, n_rmu)`` output buffer.
    Post-loop, applies the IBZ → full-BZ centroid double-permute via
    the existing :func:`gw.v_q_tile._unfold_v_q_ibz_to_full` helper
    (V_q is bilinear in ζ — no τ phase, same math as before).

    Parameters mirror the legacy :func:`gw.compute_vcoul.compute_all_V_q`
    with two differences:
    * ``zeta_loader`` is a :class:`file_io.zeta_loader.ZetaLoader` or
      :class:`file_io.zeta_reader.ZetaReader` whose ``zeta_layout`` is
      ``'G_flat'``.  Caller is expected to verify; this function
      raises if it isn't.
    * ``g_chunk`` is the G-axis chunk size.  ``None`` (default) picks
      the largest divisor of ``ngkmax`` ≤ 4096.

    Returns ``(V_qmunu, g0_mu_all)`` at the legacy shardings
    ``P(None, 'x', 'y')`` and ``P(None, 'x')``.
    """
    if str(getattr(zeta_loader, 'zeta_layout', '')) != 'G_flat':
        raise ValueError(
            "compute_all_V_q_g_flat: zeta_loader.zeta_layout must be "
            f"'G_flat'; got {getattr(zeta_loader, 'zeta_layout', None)!r}.")
    if sys_dim not in (2, 3):
        raise NotImplementedError(
            f"compute_all_V_q_g_flat: sys_dim must be 2 or 3 "
            f"(0-D box truncation per-q v(G) not wired yet); got {sys_dim}.")

    from .compute_vcoul import compute_v_q_per_G
    from .v_q_tile import (_unfold_v_q_ibz_to_full,
                            _unfold_g0_ibz_to_full)

    # ---- IBZ resolution -------------------------------------------------
    nkx, nky, nkz = kgrid
    nq_full = nkx * nky * nkz
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
            sym_perm = compute_centroid_sym_perm(
                cent_idx,
                sym_matrices=np.asarray(sym.sym_matrices[:n_tran]),
                translations=np.asarray(sym.translations[:n_tran]),
                fft_grid=np.asarray(fft_grid, dtype=np.int32),
            )
        except RuntimeError as exc:
            if verbose and jax.process_index() == 0:
                print(f"  V_q g-flat: centroid orbit closure failed — "
                      f"full-BZ iteration.  Reason: "
                      f"{exc.args[0].splitlines()[0] if exc.args else exc}")
            sym_perm = None
        if sym_perm is not None:
            (q_irr_kgrid_int, q_full_to_irr_idx,
             q_full_to_irr_sym, _) = sym.find_irreducible_qpoints()
            use_ibz = True

    if not use_ibz:
        q_irr_kgrid_int = np.array(
            [(qx, qy, qz) for qx in range(nkx)
             for qy in range(nky) for qz in range(nkz)],
            dtype=np.int32)
    n_q_ibz = int(q_irr_kgrid_int.shape[0])
    # BGW wrap: q > kg/2 → q - kg.  Same convention the writer used
    # when building the per-q gvec_components on disk.
    kg_arr = np.asarray(kgrid, dtype=np.float64)
    q_irr_wrapped = np.where(
        q_irr_kgrid_int > kg_arr / 2,
        q_irr_kgrid_int - kg_arr,
        q_irr_kgrid_int).astype(np.float64)
    q_irr_frac = q_irr_wrapped / kg_arr

    # ---- Per-q v(q+G) on the disk components ----------------------------
    gvec_components = np.asarray(
        zeta_loader.gvec_components, dtype=np.int32)        # (n_q_ibz, 3, ngkmax)
    if gvec_components is None or int(gvec_components.shape[0]) != n_q_ibz:
        raise ValueError(
            "compute_all_V_q_g_flat: zeta_loader.gvec_components is "
            f"missing or shape-mismatched (got {None if gvec_components is None else gvec_components.shape}, "
            f"expected leading axis {n_q_ibz}).  Was the file written "
            f"with the G-flat writer?")
    ngkmax = int(gvec_components.shape[-1])

    v_q_table = compute_v_q_per_G(
        q_irr_frac, gvec_components,
        bvec=bvec, cell_volume=cell_volume,
        sys_dim=sys_dim, vcoul_cutoff_ry=bare_coulomb_cutoff_ry,
        bdot=bdot,
    )                                                       # (n_q_ibz, ngkmax)

    # BGW vcoul overlay (host-side, before transfer to device).
    if bgw_v_grid_fn is not None:
        for qi in range(n_q_ibz):
            v_scaled_bgw = np.asarray(
                bgw_v_grid_fn(tuple(q_irr_frac[qi]))).reshape(-1)
            # The BGW grid lives on the full FFT box; gather to per-q sphere.
            sphere_idx = zeta_loader.gvec_components[qi]      # (3, ngkmax)
            # flat-FFT index for each per-q G: ix*ny*nz + iy*nz + iz
            # with Miller indices already in fftfreq-compatible form.
            nx, ny, nz = fft_grid
            ix = sphere_idx[0] % nx
            iy = sphere_idx[1] % ny
            iz = sphere_idx[2] % nz
            flat = ix * ny * nz + iy * nz + iz
            v_bgw_at_sphere = v_scaled_bgw[flat]
            v_q_table[qi] = np.where(
                v_bgw_at_sphere != 0.0, v_bgw_at_sphere, v_q_table[qi])

    # ---- g_chunk pick ---------------------------------------------------
    if g_chunk is None:
        # Largest divisor of ngkmax ≤ 4096.
        target = 4096
        g_chunk = ngkmax
        for c in range(min(target, ngkmax), 0, -1):
            if ngkmax % c == 0:
                g_chunk = c
                break
    g_chunk = int(g_chunk)
    if ngkmax % g_chunk != 0:
        raise ValueError(
            f"compute_all_V_q_g_flat: g_chunk={g_chunk} does not divide "
            f"ngkmax={ngkmax}.  Pass an explicit divisor of ngkmax, "
            f"or accept the auto pick.")
    n_chunks = ngkmax // g_chunk
    if verbose and jax.process_index() == 0:
        print(f"  V_q g-flat: n_q_ibz={n_q_ibz}, ngkmax={ngkmax}, "
              f"g_chunk={g_chunk} ({n_chunks} chunks/q), "
              f"unfold={'IBZ→full' if use_ibz else 'full-BZ'}")

    # ---- Pad n_rmu to mesh-product so the V output shards cleanly ------
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])
    _proc = p_x * p_y
    n_rmu_logical = int(n_rmu)
    n_rmu_padded = (n_rmu_logical + (_proc - n_rmu_logical % _proc) % _proc)
    if n_rmu_padded != n_rmu_logical and verbose and jax.process_index() == 0:
        print(f"  V_q g-flat: μ pad n_rmu={n_rmu_logical}→{n_rmu_padded} "
              f"(mesh={p_x}×{p_y}={_proc}-divisible)")

    # ---- Allocate accumulators -----------------------------------------
    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))
    V_acc = jax.jit(lambda: jnp.zeros(
        (n_q_ibz, n_rmu_padded, n_rmu_padded), dtype=jnp.complex128),
        out_shardings=V_sh)()
    g0_acc = jax.jit(lambda: jnp.zeros(
        (n_q_ibz, n_rmu_padded), dtype=jnp.complex128),
        out_shardings=g0_sh)()

    # ---- v_q to device, replicated -------------------------------------
    v_sh = NamedSharding(mesh_xy, P(None))
    v_q_dev = jax.device_put(
        jnp.asarray(v_q_table.astype(np.complex128)),
        NamedSharding(mesh_xy, P(None, None)))

    # ---- Kernel ----------------------------------------------------------
    kernel = _make_per_q_kernel(
        mesh_xy, n_rmu_padded, ngkmax, g_chunk, write_g0=True)

    # ---- Per-q loop with optional async prefetch -----------------------
    # The async pattern (borrowed from the legacy v_q_tile driver):
    # rank-0 background thread issues the next collective read while
    # rank-0 main thread waits on the current compute.  All ranks must
    # call read_slab in lock-step (it's a collective), so the
    # "prefetch" thread is really running the collective from rank-0
    # while every rank waits — which works because the SlabIO read
    # under PHDF5 is synchronous from each rank's POV.  For h5py-
    # allgather, the read is rank-0-then-broadcast; the prefetch
    # thread still serialises correctly.
    #
    # TODO(q-batch): a future opt-in could batch K q's into a single
    # read + a vmap'd kernel, amortising both the read latency and
    # the per-q kernel launch.  Marked here as the natural seam.

    zeta_disk_sh = NamedSharding(mesh_xy, P(None, ('x', 'y'), None))

    # Both ZetaReader and ZetaLoader can serve a G-flat slab, but their
    # call signatures differ.  Detect which one we have once, then
    # use the matching call inside ``read_q``.
    has_load = hasattr(zeta_loader, 'load') and callable(
        getattr(zeta_loader, 'load', None))
    has_read_slab = hasattr(zeta_loader, 'read_zeta_G_slab') and callable(
        getattr(zeta_loader, 'read_zeta_G_slab', None))
    if not (has_load or has_read_slab):
        raise TypeError(
            "compute_all_V_q_g_flat: zeta_loader must expose .load() "
            "(ZetaLoader) or .read_zeta_G_slab() (ZetaReader); "
            f"got {type(zeta_loader).__name__} with neither.")
    # Logical μ extent on disk; pad rows above this are zero by writer
    # construction and stay zero on read (SlabIO valid_shape clip).
    _n_rmu_logical = int(zeta_loader.n_rmu)

    def read_q(q_idx: int) -> jax.Array:
        """Per-q slab read sharded P(None, ('x','y'), None) → (1, μ, ngkmax)."""
        if has_load:
            return zeta_loader.load(
                q=[int(q_idx)], layout='G_flat',
                sharding=P(None, ('x', 'y'), None))
        # ZetaReader path: read_zeta_G_slab returns (Q, μ, ngkmax)
        # already in the same sharding.  ``qvec_batch_frac`` is ignored
        # on the G-flat-disk branch (the per-q phase is baked into
        # the on-disk tensor by the writer).  Pass a tiny dummy.
        return zeta_loader.read_zeta_G_slab(
            q_offset=int(q_idx), q_count=1,
            mu_offset=0, mu_count=int(n_rmu_padded),
            qvec_batch_frac=jnp.zeros((1, 3), dtype=jnp.float64),
            sphere_idx=None,
            mesh=mesh_xy, valid_mu=_n_rmu_logical,
        )

    if not async_prefetch or n_q_ibz <= 1:
        # Synchronous straight-line loop.  Light per-q progress logging
        # so a stuck kernel is visible immediately.
        import time as _t
        for q in range(n_q_ibz):
            _t0 = _t.perf_counter()
            zeta_q = read_q(q)
            _t_read = _t.perf_counter() - _t0
            _t1 = _t.perf_counter()
            V_acc, g0_acc = kernel(
                V_acc, g0_acc, zeta_q, v_q_dev[q],
                jnp.int32(q))
            jax.block_until_ready(V_acc)
            _t_k = _t.perf_counter() - _t1
            if verbose and jax.process_index() == 0:
                print(f"  V_q g-flat q={q}/{n_q_ibz}: "
                      f"read={_t_read:.2f}s, kernel={_t_k:.2f}s",
                      flush=True)
    else:
        # Single-step prefetch.  Holds at most TWO ζ_q slabs in flight.
        # The thread queue carries the *future* read; the main thread
        # awaits it before calling the kernel, then immediately issues
        # the next prefetch.
        prefetch_q: "queue.Queue[jax.Array | None]" = queue.Queue(maxsize=1)
        stop_flag = threading.Event()

        def prefetcher():
            try:
                for q in range(1, n_q_ibz):
                    if stop_flag.is_set():
                        break
                    prefetch_q.put(read_q(q))
                prefetch_q.put(None)            # sentinel
            except Exception as e:                # noqa: BLE001 — surface to main
                prefetch_q.put(e)

        zeta_curr = read_q(0)
        worker = threading.Thread(target=prefetcher, daemon=True,
                                    name="v_q_g_flat_prefetch")
        worker.start()
        try:
            for q in range(n_q_ibz):
                V_acc, g0_acc = kernel(
                    V_acc, g0_acc, zeta_curr, v_q_dev[q],
                    jnp.int32(q))
                if q + 1 < n_q_ibz:
                    next_item = prefetch_q.get()
                    if isinstance(next_item, Exception):
                        raise next_item
                    if next_item is None:
                        raise RuntimeError(
                            "compute_all_V_q_g_flat: prefetcher returned "
                            "sentinel before loop end")
                    zeta_curr = next_item
        finally:
            stop_flag.set()
            # Drain any pending sentinel so the worker exits cleanly.
            try:
                while not prefetch_q.empty():
                    prefetch_q.get_nowait()
            except queue.Empty:
                pass
            worker.join(timeout=5.0)

    # Force last kernel iteration to complete before the unfold reads it.
    V_acc.block_until_ready()

    # ---- IBZ → full-BZ unfold (centroid double-permute) ---------------
    if use_ibz:
        V_acc = _unfold_v_q_ibz_to_full(
            V_acc,
            full_to_irr_idx=q_full_to_irr_idx,
            full_to_irr_sym=q_full_to_irr_sym,
            sym_perm=sym_perm,
            mesh_xy=mesh_xy,
        )
        g0_acc = _unfold_g0_ibz_to_full(
            g0_acc,
            full_to_irr_idx=q_full_to_irr_idx,
            full_to_irr_sym=q_full_to_irr_sym,
            sym_perm=sym_perm,
            mesh_xy=mesh_xy,
        )

    V_qmunu = jax.lax.with_sharding_constraint(V_acc, V_sh)
    g0_mu_all = jax.lax.with_sharding_constraint(g0_acc, g0_sh)
    return V_qmunu, g0_mu_all


__all__ = ["compute_all_V_q_g_flat"]
