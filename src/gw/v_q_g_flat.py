"""V_q orchestrator for the G-flat ζ on-disk format.

This is the post-G-flat rewrite of the V_q hot loop.  It replaced the
old r-space tile driver (``gw/v_q_tile.py``, deleted 2026-07-02) when
the on-disk ζ is in WFN.h5-style per-q sphere layout — most of the old
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

IBZ unfold runs post-loop via ``common.symmetry_maps.unfold_v_q`` (V_q)
and the local :func:`_unfold_g0_ibz_to_full` (g0); the V_q output
sharding ``P(None, 'x', 'y')`` matches.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .gw_config import env_bool

if TYPE_CHECKING:
    from file_io.zeta_loader import ZetaLoader


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
        # Drop the size-1 q axis FIRST, then reshard the real (μ, G) tensor to
        # μ-on-x (L) / μ-on-y (R).  The previous code staged through
        # P(('x','y'), None) — sharding the size-1 q axis over all 4 devices —
        # which XLA cannot reshard to the μ-sharded layout, so it fell back to a
        # full replicate-then-repartition ("[SPMD] Involuntary full
        # rematerialization" on the V_q g-flat tensor).  Indexing [0] first
        # keeps the reshard on the μ axis (a clean all-to-all / all-gather).
        zeta_L = jax.lax.with_sharding_constraint(zeta_L_q[0], blk_x_sh)
        zeta_R_src = zeta_L_q if same_zeta else zeta_R_q
        zeta_R = jax.lax.with_sharding_constraint(zeta_R_src[0], blk_y_sh)
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
            _g_chunk_body, V_q, jnp.arange(n_chunks, dtype=jnp.int32), unroll=1)
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
    q_full_to_irr_sym, sym_perm, L_table, use_ibz)``.  When
    ``use_ibz`` is False the *_idx / *_sym / sym_perm / L_table fields
    are None; caller skips the post-loop unfold.

    ``L_table`` is the per-(sym, μ) integer real-space lattice wrap
    captured by ``compute_centroid_sym_perm``; ``unfold_v_q`` uses it
    to build the umklapp phase ``exp(2π i q · (L_μ − L_ν))``.
    """
    nkx, nky, nkz = kgrid
    use_ibz = False
    q_irr_kgrid_int = None
    q_full_to_irr_idx = None
    q_full_to_irr_sym = None
    sym_perm = None
    L_table = None
    # DEBUG: set LORRAX_FORCE_FULL_BZ=1 to bypass the IBZ cascade and
    # compute V_q at all full-BZ q's directly.  Useful for isolating
    # whether residuals come from unfold_v_q vs the rest of the pipeline.
    #
    # ONE grammar for this knob, shared with the four other sites that read
    # it (``gw/gw_init.py`` x3, ``gw/screening.py``).  The old
    # ``bool(int(os.environ.get(...)))`` took decimal digits only, so
    # ``=true``/``=on``/``=yes`` raised a bare ``invalid literal for int()``
    # from deep inside the V_q cascade rather than doing what they say.
    _force_full_bz = env_bool('LORRAX_FORCE_FULL_BZ', False)
    if sym is not None and centroid_indices is not None and not _force_full_bz:
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
            sym_perm, L_table = compute_centroid_sym_perm(
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
            L_table = None
        if sym_perm is not None:
            # Bake the μ pad into the tables ONCE at construction:
            # identity tail on the permutation (pad centroids map to
            # themselves), zero tail on the umklapp wrap (pad centroids
            # never wrap).  Consumers (``unfold_v_q``,
            # ``_unfold_g0_ibz_to_full``, gw_jax's W unfold) then
            # REQUIRE an exact extent match instead of each re-padding
            # per site — the too-small/too-large guards there replace a
            # silent ``promise_in_bounds`` OOB gather (the TRS-bug
            # failure shape) with a loud error.
            from runtime.padding import padded_mu_extent
            n_rmu_log = int(sym_perm.shape[-1])
            n_rmu_pad = padded_mu_extent(n_rmu_log, int(jax.device_count()))
            if n_rmu_pad > n_rmu_log:
                tail = np.broadcast_to(
                    np.arange(n_rmu_log, n_rmu_pad, dtype=sym_perm.dtype),
                    (sym_perm.shape[0], n_rmu_pad - n_rmu_log))
                sym_perm = np.concatenate([sym_perm, tail], axis=-1)
                L_table = np.concatenate(
                    [L_table,
                     np.zeros((L_table.shape[0], n_rmu_pad - n_rmu_log, 3),
                              dtype=L_table.dtype)], axis=1)
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
            q_full_to_irr_idx, q_full_to_irr_sym, sym_perm, L_table, use_ibz)


def _pick_g_chunk(ngkmax: int, target: int = 4096) -> int:
    """Largest divisor of ``ngkmax`` that is ≤ ``target``."""
    for c in range(min(target, int(ngkmax)), 0, -1):
        if ngkmax % c == 0:
            return int(c)
    return int(ngkmax)


def _make_read_all_ibz(zeta_loader, n_rmu_padded: int, mesh_xy: Mesh):
    """Return ``read_all_ibz(n_q_ibz) -> (n_q_ibz, n_rmu_padded, ngkmax)``.

    One read shape: ``ZetaLoader.read_zeta_G_slab`` at ``n_rmu_padded``
    rows.  SlabIO zero-fills past the dataset's own μ extent
    (decisions.md 2026-08-04), so the caller states the extent it wants
    to consume and nothing else.
    The batched single-call form avoids the ``n_q_ibz`` separate
    ``read_slab`` closures (each one a distinct
    ``_FfiBackend.read_slab.<locals>._per_rank`` closure id) that would
    each cost a JAX trace cache miss.
    """
    def read_all_ibz(n_q_ibz: int) -> jax.Array:
        return zeta_loader.read_zeta_G_slab(
            q_offset=0, q_count=int(n_q_ibz),
            mu_offset=0, mu_count=int(n_rmu_padded),
            qvec_batch_frac=jnp.zeros((int(n_q_ibz), 3), dtype=jnp.float64),
            sphere_idx=None,
            mesh=mesh_xy,
        )

    return read_all_ibz


# ---------------------------------------------------------------------------
# Per-tile core (one (μ_L, ν_L) tile)
# ---------------------------------------------------------------------------

def _compute_V_q_g_flat_one_tile(
    zeta_L_loader,
    zeta_R_loader,                     # None ⇒ same_zeta=True
    *,
    v_per_G_builder,                   # callable(q_irr_frac, gvec_components) -> (n_q, ngkmax) c128
    kgrid, fft_grid, mesh_xy,
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
    # ``n_rmu_*`` is the logical centroid count for each side — read off the
    # loader so callers don't repeat themselves.
    n_rmu_L = int(zeta_L_loader.n_rmu)
    n_rmu_R = n_rmu_L if same_zeta else int(zeta_R_loader.n_rmu)
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

    from common.symmetry_maps import unfold_v_q

    # ---- IBZ list + per-tile v(q+G) -----------------------------------
    (_q_int, q_irr_frac,
     full_to_irr_idx, full_to_irr_sym,
     sym_perm, L_table, use_ibz) = _resolve_ibz_q_list(
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
    # ``padded_mu_extent`` = the same round-up (+ test-only
    # LORRAX_EXTRA_MU_PAD rows) as ``Meta.n_rmu_padded`` — the V tiles
    # built here must match the ψ-side μ extent exactly.
    from runtime.padding import padded_mu_extent
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])
    _proc = p_x * p_y
    def _pad(n: int) -> int:
        return padded_mu_extent(int(n), _proc)
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
    # ``g0_acc`` is also the donate-target when ``write_g0=False`` — the
    # per-q kernel still needs the buffer; the contents are simply unread.
    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))
    g0_acc = jax.jit(lambda: jnp.zeros(
        (n_q_ibz, n_rmu_L_padded), dtype=jnp.complex128),
        out_shardings=g0_sh)()
    # Process-local placement, NOT plain ``jax.device_put``: the latter
    # fires JAX's hidden ``assert_equal`` all-gather on a multi-process
    # mesh — P × nq × ngkmax × 8 B of pure assertion traffic (scorecard
    # AA.1).  ``v_q_table`` is a pure function of the q-grid + cutoff,
    # identical on every rank; ``LORRAX_CHECK_REPLICA=1`` re-arms the check.
    from common.collectives import device_put_process_local
    v_q_dev = device_put_process_local(
        v_q_table, NamedSharding(mesh_xy, P(None, None)))

    kernel = _make_per_q_kernel(
        mesh_xy, n_rmu_L_padded, n_rmu_R_padded, ngkmax, g_chunk,
        write_g0=write_g0, same_zeta=same_zeta)

    read_L = _make_read_all_ibz(zeta_L_loader, n_rmu_L_padded, mesh_xy)
    read_R = (read_L if same_zeta
              else _make_read_all_ibz(zeta_R_loader, n_rmu_R_padded, mesh_xy))

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
    zeta_L_all = read_L(n_q_ibz)                            # (n_q_ibz, n_rmu_L_padded, ngkmax)
    if same_zeta:
        zeta_R_all = zeta_L_all
    else:
        zeta_R_all = read_R(n_q_ibz)
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
        # ``L_table`` is the per-(sym, μ) integer lattice wrap; the
        # umklapp phase ``exp(2π i q_irr · (L_μ − L_ν))`` is essential
        # for non-cubic / non-symmorphic systems.
        n_sym_spatial = int(np.asarray(sym_perm).shape[0]) // 2
        V_acc = unfold_v_q(
            V_acc, irr_idx=full_to_irr_idx, sym_idx=full_to_irr_sym,
            sym_perm=sym_perm, L_table=L_table, q_irr_frac=q_irr_frac,
            mesh_xy=mesh_xy, n_sym_spatial=n_sym_spatial)
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
    zeta_loader,                       # ZetaLoader (G-flat)
    *,
    kgrid: tuple[int, int, int],
    fft_grid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    mesh_xy: Mesh,
    sys_dim: int,
    bdot: np.ndarray | None = None,
    bare_coulomb_cutoff_ry: float | None = None,
    bgw_v_grid_fn=None,
    mc_average_vcoul_body: bool = True,
    g_chunk: int | None = None,
    verbose: bool = True,
    sym=None,
    centroid_indices: np.ndarray | None = None,
) -> tuple[jax.Array, jax.Array]:
    """V_q^{0,0} (charge-channel CC tile) on a G-flat-on-disk ζ file.

    Thin wrapper that builds the bare-Coulomb ``v(q+G)`` per-q-sphere
    builder and dispatches to :func:`_compute_V_q_g_flat_one_tile`
    with ``zeta_R=None`` (same_zeta) and ``write_g0=True``.  The sync
    per-q loop is already ~6× faster than the legacy μ × ν tile driver
    on MoS2 3×3.

    See :func:`_compute_V_q_g_flat_one_tile` for the math + I/O flow.
    """
    if sys_dim not in (2, 3):
        raise NotImplementedError(
            f"compute_all_V_q_g_flat: sys_dim must be 2 or 3 "
            f"(0-D box per-q v(G) not wired); got {sys_dim}.")
    # compute_v_q_per_G is gw's wfn-facing translation over the vcoul door
    # (old bvec/cell_volume/sys_dim signature) and correctly stays a gw
    # import; build_v_head_miniBZ_avg_3d is a pure service symbol, so its
    # true dependency is the door (replumbed 2026-08-07).
    from vcoul import build_v_head_miniBZ_avg_3d
    from .compute_vcoul import compute_v_q_per_G

    # 3D bulk: precompute the mini-BZ-averaged v(q, G=0) table once.
    # The IBZ → full-BZ V_q unfold is bilinear in ζ and inherits this
    # head value through ``unfold_v_q``'s centroid-permute + L-phase, so
    # injecting at every IBZ q is sufficient — no separate full-BZ pass.
    # 2D ``f2d → 0`` regularizes v at G=0 already; the MC flag is a 3D-
    # only refinement and is silently no-op'd for sys_dim=2.
    _v_head_table = None
    if mc_average_vcoul_body and sys_dim == 3:
        _v_head_table = build_v_head_miniBZ_avg_3d(
            kgrid, bvec, cell_volume)

    def _bare_v_per_G(q_irr_frac, gvec_components):
        v = compute_v_q_per_G(
            q_irr_frac, gvec_components,
            bvec=bvec, cell_volume=cell_volume,
            sys_dim=sys_dim, vcoul_cutoff_ry=bare_coulomb_cutoff_ry,
            bdot=bdot,
            v_head_miniBZ=_v_head_table,
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
        kgrid=kgrid, fft_grid=fft_grid,
        mesh_xy=mesh_xy,
        g_chunk=g_chunk,
        sym=sym, centroid_indices=centroid_indices,
        write_g0=True,
        timing_label='CC',
        verbose=verbose,
    )


__all__ = ["compute_all_V_q_g_flat", "_compute_V_q_g_flat_one_tile",
            "_resolve_ibz_q_list", "_pick_g_chunk", "_make_read_all_ibz"]


# Relocated from the deleted gw/v_q_tile.py (2026-07-02) — the only
# survivor of the r-space tile subsystem; sole consumer is compute_all_V_q_g_flat.
def _unfold_g0_ibz_to_full(
    g0_ibz: jax.Array,
    *,
    full_to_irr_idx: np.ndarray,
    full_to_irr_sym: np.ndarray,
    sym_perm: np.ndarray,
    mesh_xy: Mesh,
    n_sym_spatial: int | None = None,
) -> jax.Array:
    """Expand ``g0_ibz (n_qpt_irr, μ)`` → ``(n_q_full, μ)``.

    g0 transforms like a single ζ leg (not bilinear), so under {S | τ}:

        g0_full[q, π_s(μ)] = e^{-i (Sq + 0)·τ_s} · g0_ibz[i(q), μ]

    However, the **only** downstream use of g0 is the Γ slot (q=0),
    where S = identity ⇒ no phase, no permutation.  For q ≠ Γ the
    head-correction code in ``head_correction.py`` does not consume
    ``g0_acc[q]``.  So we apply the centroid permutation (correct
    structure) but omit the τ-phase (its effect is unobservable; the
    Γ phase is exp(0) = 1 anyway).  If a future consumer needs the
    correct phase, set ``include_tau_phase=True``.

    TRS-augmented rows: g0 is a single ζ leg, so the TRS rule is a
    plain conjugation: ``g0_{full}^{TRS-q, π_s(μ)} = conj(g0_{ibz}^{i(q), μ})``.
    Pass ``n_sym_spatial=ntran`` along with a
    ``compute_centroid_sym_perm(..., extend_trs=True)`` table to enable
    the conj branch.  Without ``n_sym_spatial``, no TRS conj is
    applied; any TRS-augmented sym index in ``full_to_irr_sym`` then
    hits the hard-fail guard below.
    """
    # Trivial-IBZ short-circuit (ntran=1, no centroid permutation, no
    # μ-pad).  See ``unfold_v_q`` for the rationale; we
    # keep the same guard here so the no-op pass-through doesn't even
    # dispatch a jit.
    idx_np = np.asarray(full_to_irr_idx)
    sym_np = np.asarray(full_to_irr_sym)
    if (idx_np.shape[0] == int(g0_ibz.shape[0])
            and np.array_equal(idx_np, np.arange(idx_np.shape[0]))
            and np.all(sym_np == 0)
            and int(g0_ibz.shape[-1]) == int(np.asarray(sym_perm).shape[-1])):
        return g0_ibz

    # TRS-aware bounds check, mirroring ``unfold_v_q``.
    n_sym_perm = int(np.asarray(sym_perm).shape[0])
    max_sym = int(sym_np.max()) if sym_np.size else -1
    if max_sym >= n_sym_perm:
        raise ValueError(
            f"_unfold_g0_ibz_to_full: full_to_irr_sym contains value "
            f"{max_sym} (TRS-augmented row index) but sym_perm has only "
            f"{n_sym_perm} rows.  Build sym_perm via "
            f"``compute_centroid_sym_perm(..., extend_trs=True)`` to "
            f"cover the TRS-augmented half of ``sym_mats_k``, and pass "
            f"``n_sym_spatial=ntran`` here.")
    if n_sym_spatial is not None and int(n_sym_spatial) * 2 != n_sym_perm:
        raise ValueError(
            f"_unfold_g0_ibz_to_full: n_sym_spatial={n_sym_spatial} is "
            f"inconsistent with sym_perm.shape[0]={n_sym_perm}.  "
            f"sym_perm must have shape (2·n_sym_spatial, n_rmu).")

    # The μ pad is baked into ``sym_perm`` at construction
    # (``_resolve_ibz_q_list``); its identity tail argsorts to itself,
    # so ``inv_perm`` needs no per-site pad-extension.  REQUIRE an
    # exact extent match (both directions) — same rationale as
    # ``unfold_v_q``: a mismatch would feed the ``promise_in_bounds``
    # gather below with out-of-range indices and clip silently.
    inv_perm = np.argsort(sym_perm, axis=-1).astype(np.int32)
    if int(g0_ibz.shape[-1]) != int(inv_perm.shape[-1]):
        raise ValueError(
            f"_unfold_g0_ibz_to_full: g0 μ-extent {int(g0_ibz.shape[-1])}"
            f" != sym_perm extent {int(inv_perm.shape[-1])}.  Tables "
            f"carry the μ pad from construction (_resolve_ibz_q_list); "
            f"pass g0 at the same (padded) extent.")
    full_to_irr_idx = np.asarray(full_to_irr_idx, dtype=np.int32)
    full_to_irr_sym = np.asarray(full_to_irr_sym, dtype=np.int32)
    inv_perm_j = jnp.asarray(inv_perm)
    idx_j = jnp.asarray(full_to_irr_idx)
    sym_j = jnp.asarray(full_to_irr_sym)

    # TRS mask: same convention as ``unfold_v_q``.
    if n_sym_spatial is not None:
        trs_mask_np = (sym_np >= int(n_sym_spatial))
    else:
        trs_mask_np = np.zeros(sym_np.shape, dtype=bool)
    trs_mask_j = jnp.asarray(trs_mask_np)

    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))

    @partial(jax.jit, out_shardings=g0_sh)
    def _do_unfold(g0):
        perm_q = inv_perm_j[sym_j]                                  # (n_q_full, n_rmu_padded)
        g0_at_irr = g0[idx_j]                                       # (n_q_full, μ)
        # ``mode='promise_in_bounds'`` to skip the take_along_axis
        # OOB-fill helper that breaks under shard_map+x64; same fix
        # as ``unfold_v_q``.
        g0_full = jnp.take_along_axis(
            g0_at_irr, perm_q, axis=1, mode='promise_in_bounds')
        # TRS rows: ζ-leg conjugation (g0 is a single ζ leg, not
        # bilinear, so the TRS conj does NOT cancel — apply it).
        g0_full = jnp.where(trs_mask_j[:, None], jnp.conj(g0_full), g0_full)
        return g0_full

    return _do_unfold(g0_ibz)
