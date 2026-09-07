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

The tile builder retains q parents; the scalar consumer unfolds ``V_q``.
The one-leg literal-``G=0`` coefficient unfolds after the tile loop.  The latter must inspect
the parent G table: a star operation can map a nonzero parent G onto the
full-zone literal G=0.  The V_q output sharding ``P(None, 'x', 'y')`` matches.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


if TYPE_CHECKING:                       # pragma: no cover — typing only
    # The DOOR, top-level name only.  No ``ensure_on_path()`` bootstrap is
    # needed here and none is added: this import never executes (the
    # module has ``from __future__ import annotations``, so the annotation
    # it feeds is a string), and a runtime path edit smuggled into a
    # typing-only block is how a "type-checking import" stops being one.
    from zeta_loader import ZetaLoader


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
        passes ``zeta_R_all is zeta_L_all`` and the kernel re-shards one
        buffer for the two operands of the einsum.
      * Bispinor off-diagonal tiles: ``same_zeta=False``; caller passes
        two separate slabs (potentially different ``n_rmu_*``).

    Returns ``fn(V_acc, g0_acc, zeta_L_all, zeta_R_all, v_q_all, q_idx)
              -> (V_new, g0_new)``.
    Donates the two accumulators so the per-q update is in-place.

    THE ζ SLICES ARE TAKEN INSIDE THIS JIT, off the already-traced
    ``q_idx``, and the whole ``(n_q, μ, G)`` slabs come in whole.  The
    caller used to slice them eagerly, which emitted one or two
    ``dynamic_slice_in_dim`` executables plus a gather for ``v_q`` per q
    outside any jit — O(n_q) dispatches and host round trips, and one
    live ``(1, n_rmu_padded, ngkmax)`` device temporary per iteration
    that only the caller's ``block_until_ready`` kept from queueing up
    (3.28 GB global for a second copy of ζ_L at CrI3 6×6 80 Ry).  Moving
    the slices in here is what makes gating that sync safe: the
    temporaries now live and die inside one executable, and the donated
    accumulator chain serialises the executions anyway.  Same shape as
    the sibling per-q tier in ``isdf/core.py`` (``_solve_one_q_and_update``),
    for the same reason: one compiled shape for the whole loop.

    Only the three slice operands are new arguments; positions 0 and 1
    are still the two donated accumulators, so ``donate_argnums`` is
    unchanged — and the ζ slabs, which every iteration reuses, are
    deliberately NOT donated.
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
    # The two slabs arrive exactly as ``ZetaLoader.read_zeta_G_slab``
    # returns them — q replicated, μ over ('x','y') — and the v(q+G) table
    # replicated by ``device_put_process_local``.  Now that the per-q slice
    # is taken in here, the SLAB is what crosses the jit boundary, so the
    # entry sharding is worth stating rather than inheriting: it is what
    # keeps the reshard below on the one (μ, G) face instead of on the whole
    # (n_q, μ, ngkmax) tensor.  Measured a no-op at the production layout on
    # a 2×2 CPU mesh (identical HLO with and without) — it is a contract
    # against a caller that hands the kernel a differently-sharded slab, not
    # a fix for something GSPMD is doing today.
    zeta_all_sh = NamedSharding(mesh_xy, P(None, ('x', 'y'), None))
    v_all_sh = NamedSharding(mesh_xy, P(None, None))

    n_chunks = ngkmax // g_chunk

    @partial(jax.jit, donate_argnums=(0, 1))
    def fn(V_acc, g0_acc, zeta_L_all, zeta_R_all, v_q_all, q_idx):
        q_idx_32 = q_idx.astype(jnp.int32)
        zero32 = jnp.int32(0)

        zeta_L_all = jax.lax.with_sharding_constraint(
            zeta_L_all, zeta_all_sh)
        zeta_R_src_all = zeta_L_all if same_zeta else (
            jax.lax.with_sharding_constraint(zeta_R_all, zeta_all_sh))
        v_q_all = jax.lax.with_sharding_constraint(v_q_all, v_all_sh)

        # Per-q slices off the TRACED q.  ``zeta_*_q`` are (1, n_rmu_*,
        # ngkmax); the size-1 q axis is dropped immediately below.
        zeta_L_q = jax.lax.dynamic_slice_in_dim(
            zeta_L_all, q_idx_32, 1, axis=0)
        # Drop the size-1 q axis FIRST, then reshard the real (μ, G) tensor to
        # μ-on-x (L) / μ-on-y (R).  The previous code staged through
        # P(('x','y'), None) — sharding the size-1 q axis over all 4 devices —
        # which XLA cannot reshard to the μ-sharded layout, so it fell back to a
        # full replicate-then-repartition ("[SPMD] Involuntary full
        # rematerialization" on the V_q g-flat tensor).  Indexing [0] first
        # keeps the reshard on the μ axis (a clean all-to-all / all-gather).
        # THIS ORDER IS THE POINT — do not fold the [0] into the slice's
        # sharding constraint.
        #
        # ONE ``[0]`` PER SLAB, BOUND TO A NAME, and in the same_zeta case
        # the two operands share it; then an ``optimization_barrier`` on the
        # (μ, ngkmax) face.  Both are there for the same measured reason and
        # neither is cosmetic.  The (1, μ, G) slice has a degenerate leading
        # axis, so XLA is free to prefer the ``{2,0,1}`` minor-to-major for
        # it — which is a genuinely different physical layout for the
        # (n_q, μ, G) slab it is sliced from — and it answers by copying the
        # WHOLE parameter into that layout, once per call.  Measured in the
        # optimized HLO on a 2×2 CPU mesh: two ``c128[n_q,μ/p,ngkmax] copy``
        # ops on the distinct-ζ tile, one on the shared-ζ tile.  Binding the
        # squeeze once removes the shared-ζ copy; the barrier removes both,
        # and costs nothing at run time (it is a scheduling fence, not an op).
        # The collectives are unchanged either way — 2 all-gather + 1
        # collective-permute, identical to the eager-slice kernel this
        # replaced.
        zeta_L_face = jax.lax.optimization_barrier(zeta_L_q[0])
        zeta_R_face = (zeta_L_face if same_zeta
                       else jax.lax.optimization_barrier(
                           jax.lax.dynamic_slice_in_dim(
                               zeta_R_src_all, q_idx_32, 1, axis=0)[0]))
        zeta_L = jax.lax.with_sharding_constraint(zeta_L_face, blk_x_sh)
        zeta_R = jax.lax.with_sharding_constraint(zeta_R_face, blk_y_sh)
        v_q = jax.lax.with_sharding_constraint(
            jax.lax.dynamic_slice_in_dim(v_q_all, q_idx_32, 1, axis=0)[0],
            v_sh)

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

def _resolve_ibz_q_list(*, sym, centroid_indices, kgrid, fft_grid,
                        context="V_q / W q-grid reduction",
                        return_resolution=False, mu_basis=None):
    """Pick IBZ q's via centroid orbit closure, fall back to full BZ.

    Returns ``(q_irr_kgrid_int, q_irr_frac, q_full_to_irr_idx,
    q_full_to_irr_sym, sym_perm, L_table, use_ibz)``.  With
    ``return_resolution=True``, the resolution carrying the closure verdict
    is appended.  When
    ``use_ibz`` is False the *_idx / *_sym / sym_perm / L_table fields
    are None; caller skips the post-loop unfold.

    ``L_table`` is the per-(sym, μ) integer real-space lattice wrap
    captured by ``centroid_source_map_and_wrap``; ``unfold_isdf_operator`` uses
    it
    to build the umklapp phase ``exp(2π i q · (L_μ − L_ν))``.

    THE CLOSURE DECISION IS NOT TAKEN HERE.  It is taken once, in
    ``symmetry_maps.resolve_qgrid_symmetry``, and announced once by
    ``gw.qgrid_symmetry.resolve_qgrid_symmetry_tables``; this function
    consumes the resolution and shapes it into the seven-tuple its three
    callers already read.

    THE ``verbose`` ARGUMENT IS GONE.  It gated exactly one thing — the
    line that said the fallback had happened — and ``gw/screening.py``
    passed ``verbose=False``, which is why the W Dyson solve could drop
    from ``n_q_ibz`` blocks to ``n_q_full`` without a word.  A knob whose
    only effect is to hide a degradation is not a verbosity knob.
    """
    nkx, nky, nkz = kgrid
    use_ibz = False
    q_irr_kgrid_int = None
    q_full_to_irr_idx = None
    q_full_to_irr_sym = None
    sym_perm = None
    L_table = None
    res = None
    if sym is not None and centroid_indices is not None:
        from .qgrid_symmetry import resolve_qgrid_symmetry_tables
        res = resolve_qgrid_symmetry_tables(
            sym=sym, centroid_indices=centroid_indices, fft_grid=fft_grid,
            context=context)
        if res.use_ibz:
            sym_perm, L_table = res.tables()
        if sym_perm is not None and mu_basis is not None:
            # In-memory consumers (W, χ) hold their operators in the run's
            # packed centroid order: conjugate the canonical tables into it.
            # The pad bake below is then a no-op (the packed extent is
            # already the complete-mesh carrier).  V is built from the
            # canonical ζ file and passes no basis.
            sym_perm, L_table = mu_basis.pack_tables(sym_perm, L_table)
        if sym_perm is not None:
            # Bake the μ pad into the tables ONCE at construction:
            # identity tail on the permutation (pad centroids map to
            # themselves), zero tail on the umklapp wrap (pad centroids
            # never wrap).  Consumers (``unfold_isdf_operator``,
            # ``unfold_isdf_one_leg``, gw_jax's W unfold) then
            # REQUIRE an exact extent match instead of each re-padding
            # per site — the too-small/too-large guards there replace a
            # silent ``promise_in_bounds`` OOB gather (the TRS-bug
            # failure shape) with a loud error.
            from runtime.padding import padded_mu_extent
            n_rmu_log = int(sym_perm.shape[-1])
            n_rmu_pad = (mu_basis.n_packed if mu_basis is not None else
                         padded_mu_extent(n_rmu_log, int(jax.device_count())))
            if n_rmu_pad > n_rmu_log:
                tail = np.broadcast_to(
                    np.arange(n_rmu_log, n_rmu_pad, dtype=sym_perm.dtype),
                    (sym_perm.shape[0], n_rmu_pad - n_rmu_log))
                tail = np.where(np.all(sym_perm == -1, axis=1)[:, None], -1, tail)
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

    # The symmetry service owns BGW's strict half-grid tie convention.
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import bgw_integer_q_to_fractional
    q_irr_frac = bgw_integer_q_to_fractional(q_irr_kgrid_int, kgrid)
    result = (q_irr_kgrid_int, q_irr_frac,
              q_full_to_irr_idx, q_full_to_irr_sym,
              sym_perm, L_table, use_ibz)
    return (*result, res) if return_resolution else result


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
    is_charge_cc: bool,
    write_g0: bool,
    one_leg_action: str,
    qgrid_policy=None,
    source_component: int | None = None,
    timing_label: str,
    verbose: bool,
) -> tuple[jax.Array, jax.Array | None]:
    """Contract one q-parent V tile and its separately transported full-q G=0 leg."""
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

    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import unfold_isdf_one_leg

    # ---- IBZ list + per-tile v(q+G) -----------------------------------
    (_q_int, q_irr_frac,
     full_to_irr_idx, full_to_irr_sym,
     sym_perm, L_table, use_ibz) = _resolve_ibz_q_list(
        sym=sym, centroid_indices=centroid_indices,
        kgrid=kgrid, fft_grid=fft_grid,
        context=f"V_q g-flat tile [{timing_label}]")
    n_q_ibz = int(q_irr_frac.shape[0])

    policy = qgrid_policy
    unfold_sym = full_to_irr_sym
    if use_ibz:
        n_sym_spatial = int(np.asarray(sym_perm).shape[0]) // 2
        if policy is None:
            from .qgrid_symmetry import qgrid_trs_policy_for
            policy = qgrid_trs_policy_for(
                sym=sym, irr_idx_q=full_to_irr_idx,
                sym_idx_q=full_to_irr_sym, kgrid=tuple(kgrid),
                n_sym_spatial=n_sym_spatial,
                context=f"V_q / one-leg [{timing_label}]")
        unfold_sym = np.asarray(policy.unfold_sym_idx, dtype=np.int32)
        if unfold_sym.shape != np.asarray(full_to_irr_sym).shape:
            raise ValueError(
                f"_compute_V_q_g_flat_one_tile[{timing_label}]: shared "
                "QgridTrsPolicy has the wrong q extent.")

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
    def _pad(n: int) -> int:
        return padded_mu_extent(int(n), mesh_xy)
    n_rmu_L_padded = _pad(int(n_rmu_L))
    n_rmu_R_padded = _pad(int(n_rmu_R))
    if verbose and jax.process_index() == 0:
        print(f"  V_q g-flat [{timing_label}]: n_q_ibz={n_q_ibz}, "
              f"ngkmax={ngkmax}, g_chunk={g_chunk} ({n_chunks}/q), "
              f"n_rmu_L={n_rmu_L}→{n_rmu_L_padded}, "
              f"n_rmu_R={n_rmu_R}→{n_rmu_R_padded}, "
              f"storage={'q-IBZ' if use_ibz else 'full-BZ'}",
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
        # On an IBZ the literal full-zone G=0 may be a nonzero parent G;
        # selecting disk slot zero inside this kernel is therefore unsafe.
        # The service action below reads the exact parent slot while the
        # same zeta slab is still resident.
        write_g0=bool(write_g0 and not use_ibz), same_zeta=same_zeta)

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

    # ---- Per-q kernel loop on device-resident ζ̃ ---------------------
    # The loop hands the kernel the WHOLE slabs and a traced q; the three
    # per-q slices (ζ_L, ζ_R, v_q) happen inside its jit.  What used to be
    # here was the eager form of exactly those slices — one or two
    # ``dynamic_slice_in_dim`` executables and a ``v_q_dev[q]`` gather per
    # iteration, outside any jit, plus a host→device transfer for the loop
    # counter.
    #
    # THE SYNC AND THE SLICES MOVED TOGETHER, and the order matters.  The
    # ``block_until_ready`` below is vestigial as a correctness device: it
    # landed in ac735cca8 when the loop body still did a collective PHDF5
    # read per q, and ordered the kernel before the next read; 0880066a1
    # hoisted that read into ``read_all_ibz`` above and left the sync
    # behind.  But while the slices were still eager it was also the only
    # backpressure on them — drop it alone and up to n_q live
    # ``(1, n_rmu_L_padded, ngkmax)`` temporaries queue, a second copy of
    # ζ_L (3.28 GB global at CrI3 6×6 80 Ry).  With the slices inside the
    # executable there is nothing left to queue: the donated (V_acc, g0_acc)
    # chain makes each call depend on the previous one's output, so the
    # executions serialise on their own and the transients live and die
    # inside one program.  What is left for the sync to do is make the
    # per-q number below a KERNEL time rather than a dispatch time — so it
    # is gated on the same condition as the print it feeds.
    #
    # Multiplier for both: ``v_q_bispinor`` calls this function once per
    # UNIQUE_TILE, so the counts here are per tile (7 tiles × 9 q = 63
    # syncs on the MoS2 3×3 bispinor deck).
    _time_each_q = bool(verbose) and jax.process_index() == 0
    # Hoisted: ``jnp.int32(q)`` inside the loop was a host→device transfer
    # per iteration.
    q_idx_dev = [jnp.int32(q) for q in range(n_q_ibz)]
    for q in range(n_q_ibz):
        _t1 = _t.perf_counter()
        V_acc, g0_acc = kernel(
            V_acc, g0_acc, zeta_L_all, zeta_R_all, v_q_dev, q_idx_dev[q])
        # The wait consumes a sharded accumulator, so it cannot sit under
        # the rank-0 timing gate.  All processes rendezvous; only rank 0
        # formats the diagnostic (INVARIANTS row 21).
        jax.block_until_ready(V_acc)
        if _time_each_q:
            print(f"    [{timing_label}] q={q}/{n_q_ibz}: "
                  f"kernel={_t.perf_counter() - _t1:.2f}s", flush=True)

    if write_g0 and use_ibz:
        g0_acc = unfold_isdf_one_leg(
            zeta_L_all,
            gvec_components=gvec_components,
            sym=sym,
            sym_idx=unfold_sym,
            sym_perm=sym_perm,
            L_table=L_table,
            q_irr_frac=q_irr_frac,
            kgrid=kgrid,
            mesh_xy=mesh_xy,
            component_action=one_leg_action,
            source_component=source_component,
        )
    del zeta_L_all
    if not same_zeta:
        del zeta_R_all

    # ---- IBZ → full-BZ unfold (centroid double-permute) -------------
    if use_ibz:
        # ``sym_perm`` came from ``centroid_source_map_and_wrap(...,
        # extend_trs=True)`` so its shape[0] is ``2·ntran``; the second half
        # encodes the TRS-augmented rows (centroid permutation unchanged under
        # TRS, but the unfold helper conjugates V_q at TRS-tagged q's).
        # ``L_table`` is the per-(sym, μ) integer lattice wrap; the
        # umklapp phase ``exp(2π i q_irr · (L_μ − L_ν))`` is essential
        # for non-cubic / non-symmorphic systems.
        n_sym_spatial = int(np.asarray(sym_perm).shape[0]) // 2
        if is_charge_cc:
            # TIME REVERSAL IS EXPLICIT, NEVER ASSUMED HERE. This block used to
            # compose q with −q through Θ and project the self-negative
            # rows unconditionally; on a ferromagnet that fabricates a
            # symmetry the reference verdict says is absent. The policy reads
            # ``sym.trs_allowed`` and
            # this site no longer contains a TRS branch of its own.
            # The point-group covariance the unfold below ASSUMES of the
            # finite ζ basis, measured on the stored parents while they are
            # still the pre-unfold wedge.  The q↔−q gate downstream is
            # structurally blind to it at a self-negative q (there it
            # degenerates to "V_q is real"); this is not.
            cov = policy.measure_covariance(
                V_acc, q_irr_frac=q_irr_frac,
                q_irr_full_idx=sym.q_irr_full_idx,
                sym_mats_k=sym.sym_mats_k, sym_perm=sym_perm,
                L_table=L_table)
            V_acc, removed = policy.project_fixed_q(
                V_acc, sym.q_irr_full_idx)
            if jax.process_index() == 0:
                from common import sanity
                sanity.report_parent_covariance(
                    "V_q[CC] IBZ parents", cov, removed=removed)
        # THE PRE-UNFOLD BLOCK, OFFERED TO WHOEVER IS WRITING THE RESTART.
        # This is the array the q_irr format persists — the design's
        # load-bearing decision, because ``unfold(stored)`` is then the
        # SAME CALL on the SAME ARGUMENTS the line below makes, an identity
        # rather than a property that depends on the op-selection policy.
        # It exists for exactly one statement, which is why the offer is
        # here and not at the writer.
        #
        # A NO-OP unless a driver has opened a capture scope, so the
        # compute path takes no restart decision and this line costs a list
        # check on every other run.  Only the CC tile is offered: the
        # bispinor CT/TT tiles are not restart tensors.
        if is_charge_cc:
            from .restart_q_storage import deposit_pre_unfold
            deposit_pre_unfold(
                "V_qmunu", V_acc,
                n_rmu_logical=int(zeta_L_loader.n_rmu),
                q_irr_frac=q_irr_frac, irr_idx_q=full_to_irr_idx,
                sym_idx_q=unfold_sym, sym_perm=sym_perm,
                L_table=L_table, n_sym_spatial=n_sym_spatial)

    V_qmunu = jax.lax.with_sharding_constraint(V_acc, V_sh)
    if write_g0:
        g0_spec = (P(None, 'x') if int(g0_acc.ndim) == 2
                   else P(None, None, 'x'))
        return V_qmunu, jax.lax.with_sharding_constraint(
            g0_acc, NamedSharding(mesh_xy, g0_spec))
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
    # import; build_v_head_miniBZ_fn_3d is a pure service symbol, so its
    # true dependency is the door (replumbed 2026-08-07).  ORDER IS
    # LOAD-BEARING: .compute_vcoul runs the service path bootstrap at its
    # module scope, so it must be imported BEFORE the door — the blind
    # audit arm measured the swapped order dying with ModuleNotFoundError
    # in a stripped process where nothing else had bootstrapped yet.
    from .compute_vcoul import compute_v_q_per_G
    from vcoul import build_v_head_miniBZ_fn_3d

    # 3D bulk: build the mini-BZ-averaged head ⟨v(K+δq)⟩ ONCE, as a
    # function of the Cartesian K = q+G.  ``v_qG_table`` evaluates it at
    # every slot attaining argmin |q+G| — see its HEAD SLOT note; the old
    # per-q table keyed on the Miller-(0,0,0) label was not equivariant
    # under q → −q and cost V_q 6.0e−3 of reciprocity against a 1.16e−7
    # floor.
    # The IBZ → full-BZ V_q unfold is bilinear in ζ and inherits this
    # head value through ``unfold_isdf_operator``'s centroid-permute + L-phase,
    # so injecting at every IBZ q is sufficient — no separate full-BZ pass.
    # 2D ``f2d → 0`` regularizes v at G=0 already; the MC flag is a 3D-
    # only refinement and is silently no-op'd for sys_dim=2.
    _v_head_fn = None
    if mc_average_vcoul_body and sys_dim == 3:
        _v_head_fn = build_v_head_miniBZ_fn_3d(
            kgrid, bvec, cell_volume)

    def _bare_v_per_G(q_irr_frac, gvec_components):
        v = compute_v_q_per_G(
            q_irr_frac, gvec_components,
            bvec=bvec, cell_volume=cell_volume,
            sys_dim=sys_dim, vcoul_cutoff_ry=bare_coulomb_cutoff_ry,
            bdot=bdot,
            v_head_fn=_v_head_fn,
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

    V_q, g0 = _compute_V_q_g_flat_one_tile(
        zeta_loader, None,
        v_per_G_builder=_bare_v_per_G,
        kgrid=kgrid, fft_grid=fft_grid,
        mesh_xy=mesh_xy,
        g_chunk=g_chunk,
        sym=sym, centroid_indices=centroid_indices,
        is_charge_cc=True,
        write_g0=True,
        one_leg_action="scalar",
        timing_label='CC',
        verbose=verbose,
    )

    from symmetry_maps import unfold_isdf_operator
    from .qgrid_symmetry import qgrid_trs_policy_for
    _, q_frac, irr, rows, perm, wraps, reduced = _resolve_ibz_q_list(
        sym=sym, centroid_indices=centroid_indices,
        kgrid=kgrid, fft_grid=fft_grid)
    if reduced:
        policy = qgrid_trs_policy_for(
            sym=sym, irr_idx_q=irr, sym_idx_q=rows, kgrid=kgrid,
            n_sym_spatial=len(perm) // 2, context="scalar V consumer")
        V_q = unfold_isdf_operator(
            V_q, irr_idx=irr, sym_idx=policy.unfold_sym_idx,
            sym_perm=perm, L_table=wraps, q_irr_frac=q_frac,
            mesh_xy=mesh_xy, n_sym_spatial=policy.n_sym_spatial)
    return V_q, g0


def compute_head_channel_zeta(
    zeta_loader,
    *,
    kgrid: tuple[int, int, int],
    fft_grid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    mesh_xy: Mesh,
    sys_dim: int,
    bdot: np.ndarray | None = None,
    bare_coulomb_cutoff_ry: float | None = None,
    mc_average_vcoul_body: bool = True,
    sym=None,
    centroid_indices: np.ndarray | None = None,
    verbose: bool = True,
):
    """The q != 0 Coulomb head channel, in the centroid basis, on the full BZ.

    Returns ``(g_head, table)`` where ``g_head`` is
    ``(n_q_full, k, n_rmu_padded)`` complex128 at ``P(None, None, 'x')`` and
    ``table`` is the :class:`vcoul.HeadSlotTable` on the IBZ q-list.
    ``g_head[q, j, :]`` is ``zeta(q, mu, G_j)`` for the j-th slot attaining
    ``argmin |q+G|``, already multiplied by the tie mask — so padding
    columns, Γ, and any q whose head slot the bare-Coulomb cutoff zeroes are
    EXACT zeros and the projector

        P_q = sum_j conj(g_head[q, j]) (x) g_head[q, j]

    is the head-slot part of ``V_q`` divided by the value ``v_qG_table`` put
    there.  Consumed by ``gw.head_channel``; built only when a deck turns
    ``mc_average_placement`` on, which is why this is a second short read of
    the same ζ slabs rather than a third output of the V_q hot loop — the
    default V_q path stays byte-for-byte and compile-for-compile unchanged.

    Each selected source column is unfolded by
    ``symmetry_maps.unfold_isdf_one_leg`` with its actual parent Miller
    vector, including the centroid source gather, L phase, nonsymmorphic tau
    phase and measured antiunitary convention.  A tied set maps onto a tied
    set under the little group (``|q+G|`` is invariant), so the sum over
    columns is independent of their image ordering.
    """
    from .compute_vcoul import compute_v_q_per_G  # bootstrap, see the CC path
    from vcoul import (CoulombGeometry, build_v_head_miniBZ_fn_3d, get_kernel,
                       head_slot_table)

    del compute_v_q_per_G  # imported for the service-path bootstrap only

    if str(getattr(zeta_loader, 'zeta_layout', '')) != 'G_flat':
        raise ValueError(
            "compute_head_channel_zeta: zeta layout must be 'G_flat'; got "
            f"{getattr(zeta_loader, 'zeta_layout', None)!r}")

    (_q_int, q_irr_frac,
     full_to_irr_idx, full_to_irr_sym,
     sym_perm, L_table, use_ibz) = _resolve_ibz_q_list(
        sym=sym, centroid_indices=centroid_indices,
        kgrid=kgrid, fft_grid=fft_grid,
        context="head-channel zeta")
    n_q_ibz = int(q_irr_frac.shape[0])
    gvec_components = np.asarray(zeta_loader.gvec_components, dtype=np.int32)

    # THE SAME estimator object the V_q path builds — same seed, same draw
    # count, same centrosymmetrisation.  Built here rather than shared
    # because the two calls are in different stages and a deterministic pure
    # function is cheaper to rebuild than to thread.
    #
    # NOT GATED ON ``mc_average_vcoul_body``.  That flag decides whether the
    # average is substituted into V — i.e. what Sigma_X receives — and the
    # placement mode decides where the average lands in W.  Gating the head
    # function on the flag made ``mc_average_placement = bgw`` with the flag
    # off a silent no-op (<v> == v_c => r == 1), which is precisely the
    # "knob that quietly does nothing" failure this feature is supposed to
    # be immune to.  The flag is honoured where it belongs: in ``v_in_V``,
    # the value the production tile actually carries.
    v_head_fn = None
    if sys_dim == 3:
        v_head_fn = build_v_head_miniBZ_fn_3d(kgrid, bvec, cell_volume)
    del mc_average_vcoul_body

    table = head_slot_table(
        get_kernel(sys_dim), q_irr_frac, gvec_components,
        geometry=CoulombGeometry(bvec=bvec, cell_volume=cell_volume,
                                 bdot=bdot, fft_grid=fft_grid),
        vcoul_cutoff_ry=bare_coulomb_cutoff_ry,
        v_head_fn=v_head_fn,
    )

    from runtime.padding import padded_mu_extent
    n_rmu_padded = padded_mu_extent(
        int(zeta_loader.n_rmu),
        int(mesh_xy.shape['x']) * int(mesh_xy.shape['y']))

    read_all = _make_read_all_ibz(zeta_loader, n_rmu_padded, mesh_xy)
    zeta_all = read_all(n_q_ibz)              # (n_q_ibz, mu_pad, ngkmax)

    policy = None
    if use_ibz:
        from .qgrid_symmetry import qgrid_trs_policy_for
        policy = qgrid_trs_policy_for(
            sym=sym, irr_idx_q=full_to_irr_idx,
            sym_idx_q=full_to_irr_sym, kgrid=tuple(kgrid),
            n_sym_spatial=int(np.asarray(sym_perm).shape[0]) // 2,
            context="head-channel one-leg")
        from symmetry_maps import unfold_isdf_one_leg

    sel_dev = jnp.asarray(np.asarray(table.sel, dtype=np.int32))
    mask_dev = jnp.asarray(np.asarray(table.mask, dtype=np.float64),
                           dtype=jnp.complex128)
    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))

    # (n_q, mu, k) gather on the UNSHARDED G axis, then transpose.  Done
    # per j so each intermediate is (n_q, mu) — the same class as g0_acc —
    # and so the sharding constraint lands on the shape the unfold wants.
    cols = []
    for j in range(int(table.sel.shape[1])):
        col = jnp.take_along_axis(
            zeta_all, sel_dev[:, None, j:j + 1], axis=2)[:, :, 0]
        col = col * mask_dev[:, j:j + 1]
        col = jax.lax.with_sharding_constraint(col, g0_sh)
        if use_ibz:
            source_slot = np.broadcast_to(
                np.asarray(table.sel[:, j], dtype=np.int32)[:, None, None],
                (n_q_ibz, 3, 1))
            source_g = np.take_along_axis(
                gvec_components, source_slot, axis=2)[:, :, 0]
            col = unfold_isdf_one_leg(
                col,
                source_gvec_components=source_g,
                sym=sym,
                sym_idx=policy.unfold_sym_idx,
                sym_perm=sym_perm,
                L_table=L_table,
                q_irr_frac=q_irr_frac,
                kgrid=kgrid,
                mesh_xy=mesh_xy,
                component_action="scalar",
            )
        cols.append(jax.lax.with_sharding_constraint(col, g0_sh))
    del zeta_all

    g_head = jnp.stack(cols, axis=1)          # (n_q_full, k, mu)
    g_head = jax.lax.with_sharding_constraint(
        g_head, NamedSharding(mesh_xy, P(None, None, 'x')))
    if verbose and jax.process_index() == 0:
        live = np.asarray(table.v_bare) > 0.0
        eta = np.zeros_like(np.asarray(table.v_bare))
        eta[live] = (np.asarray(table.v_avg)[live]
                     / np.asarray(table.v_bare)[live] - 1.0)
        print(f"  head channel: {int((table.mult > 0).sum())}/{n_q_ibz} IBZ q "
              f"carry a head slot, k={int(table.sel.shape[1])}, "
              f"tie histogram="
              f"{dict(zip(*[c.tolist() for c in np.unique(table.mult, return_counts=True)]))}",
              flush=True)
        # Per-shell eta, printed as data rather than summarised: it is the
        # ONE input the whole rescale is a function of, it is directly
        # comparable to BerkeleyGW's own vcoul dumps, and a run whose log
        # carries it can be audited without re-running anything.
        for qi in np.nonzero(live)[0]:
            print(f"    head slot q[{qi}] |q+G|^2={float(table.len2[qi]):.6f} "
                  f"mult={int(table.mult[qi])} v_c={float(table.v_bare[qi]):.6f} "
                  f"<v>={float(table.v_avg[qi]):.6f} eta={eta[qi]:.6f}",
                  flush=True)
    return g_head, table, full_to_irr_idx


__all__ = ["compute_all_V_q_g_flat", "_compute_V_q_g_flat_one_tile",
            "_resolve_ibz_q_list", "_pick_g_chunk", "_make_read_all_ibz",
            "compute_head_channel_zeta"]
