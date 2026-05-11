"""
Unified V_q tile kernel — single source of truth for V_q(μ, ν) construction.

Replaces the old ``_make_V_q_caseA_kernel`` / ``_make_V_q_caseB_kernel`` pair
plus the two outer drivers ``_compute_all_V_q_sharded`` /
``_compute_all_V_q_replicated`` with one inner kernel and one outer driver.

The unified function handles every V_q tile shape that occurs in production:

    * ``same_zeta=True``  — V^{0,0} self-contraction (the scalar charge tile and
      the bispinor diagonal tiles).  One read, one FFT, two one-axis gathers.
    * ``same_zeta=False`` — bispinor V^{μ_L, ν_L} (L ≠ R) tiles.  Two reads,
      two FFTs, no g0 (head term is only meaningful for V^{0,0}).
    * Case A (μ fits in budget)  — full μ × full μ in one shot per q-batch.
    * Case B (μ-tiled)            — single q, μ × ν loop.

Coulomb dimensionality is provided externally via ``v_per_G_fn`` and
``phase_fn`` callables.  The inner kernel applies ``v_per_G`` ONCE on the L
side of the contraction:

    V_block(μ, ν) = Σ_G  conj(ζ_L(G)) · v(K) · ζ_R(G)

This works for any real-valued (signed) weight ``v(K)``; in particular the
signed transverse-projector weight ``v(K) · (-K̂_i K̂_j)`` used by the
bispinor V^{μ_L, ν_L} (L ≠ R) tiles is handled with no special casing.

I/O backends
------------
The driver reads ζ via ``SlabIO.read_slab(..., partition_spec=P(None, None,
('x','y')))``.  Both the PHDF5/FFI and h5py-allgather backends honour this
partition_spec — PHDF5 lands ζ directly in μ-on-XY layout, allgather rank-0
reads then ``device_put``s with the same sharding.  The unified kernel
sees the same input layout either way.

Synchronisation
---------------
The Python loop does NOT call ``.block_until_ready()`` on intermediate
arrays.  Each kernel call is async-dispatched and returns lazily; the next
iteration's ``read_slab`` runs on the host thread while the GPU finishes
the previous iteration's compute, so PHDF5 collective reads naturally
overlap with V_q compute.
"""

import os
from functools import partial
from math import gcd

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.fft_helpers import make_sharded_fftn_3d


# ----------------------------------------------------------------------------
# Memory-model chooser
# ----------------------------------------------------------------------------
#
# Chooses (q_chunk, μ_chunk) given a per-rank budget.  See the big comment in
# ``compute_vcoul.py`` (kept there for now since other modules also reference
# it) for the full derivation.  The chooser models two stages with different
# working-axis shardings:
#
#   (a) Gathered/post-sphere stage on (Q, μ, n_G)  sharded P_min     (one axis)
#   (b) FFT stage on (Q, μ, n_rtot)                sharded P_prod    (both axes)
#
# Pre-2026-05-06 only stage (a) was modelled; CrI3 4×4×6 with μ=1500 hit OOM
# at the predicted 41 GB because stage (b) actually controls when n_rtot
# >> n_G.  Take ``max(a, b)`` to size q_chunk.
#
# The FFT stage's overlap coefficient is env-tunable for A/B testing and
# for distinct-ζ tile callers.  The default 5.0 reflects the cuFFT
# workspace measured-but-not-cheaply-known footprint: input ζ slab + the
# post-(reshape × phase) intermediate alive alongside the FFT scratch
# (cuFFT plans for mixed-radix 75×75×200 typically allocate 2-3× the
# input as scratch on top of the input + output buffers).
# ``query_fft_peak_bytes`` in fft_helpers gives the AOT-exact value per
# shape; we'd need to call it at chooser time, paying a ~1 s compile per
# unique shape.  For now an over-conservative flat 5× covers same-ζ; the
# ``_choose_v_q_chunks`` distinct-ζ path doubles this further (10×) since
# both ζ_L_disk and ζ_R_disk are alive simultaneously at the kernel input
# boundary.  Override with ``LORRAX_V_Q_FFT_COEF=<float>`` for A/B tests.
_Q_COMPUTE_COEF_FFT = float(os.environ.get('LORRAX_V_Q_FFT_COEF', '5.0'))


def _gather_stage_per_q_bytes(*, n_rmu: int, n_G: int,
                              p_x: int, p_y: int) -> float:
    """Per-rank, per-Q peak memory of the gather + contract stage.

    Three sharded slabs are alive simultaneously at the contract: the
    post-sphere ζ_G ``(Q, μ/p_prod, n_G)``, plus the two one-axis-gathered
    slabs ζ_μ_X ``(Q, μ/p_x, n_G)`` and ζ_ν_Y ``(Q, μ/p_y, n_G)``.  Sum
    of per-rank live bytes per Q is

        N_zeta · μ · n_G · (1/p_x + 1/p_y + 1/(p_x · p_y))

    No magic constant: the formula is just "three slabs alive at
    contract time".  Replaces the legacy empirical coefficient
    ``_Q_COMPUTE_COEF_GATHER = 4.4`` (calibrated on MoS2 / Si 10³ on a
    different mesh and over-predicting on CrI3 by ~2× — it never bit
    perf because the FFT stage dominates ``max(gather, fft)``, but it
    was the last system-specific magic in the chooser).
    """
    return 16.0 * float(n_rmu) * float(n_G) * (
        1.0 / float(p_x) + 1.0 / float(p_y) +
        1.0 / (float(p_x) * float(p_y))
    )


_v_q_aot_cache: dict = {}
_v_q_full_kernel_aot_cache: dict = {}


def _aot_full_kernel_peak(
    *,
    q_chunk: int,
    mu_size: int,
    nu_size: int,
    n_rmu_L: int,
    n_rmu_R: int,
    nq_total: int,
    n_rtot: int,
    n_G_sph: int,
    fft_grid: tuple[int, int, int],
    sphere_idx,
    mesh_xy: Mesh,
    same_zeta: bool,
    write_g0: bool,
) -> int | None:
    """AOT-compile the *full* production V_q tile kernel and return
    per-rank peak bytes from XLA ``compiled.memory_analysis()``.

    This captures everything inside the kernel boundary at once: the
    ζ_disk inputs, the FFT input + output + cuFFT plan workspace, the
    sphere-pick output, the two one-axis gathered ζ_X/ζ_Y slabs, the
    contract V_block, the DUS update on V_acc/g0_acc, and any aliasing
    XLA negotiates between them.  Replaces the standalone-FFT
    ``_aot_fft_model`` measurement (which captured only the FFT
    primitive's input + output + workspace and missed the production
    kernel's surrounding live buffers + aliasing).

    Cost: one ``jit.lower(*specs).compile()`` per unique shape (~1-2 s
    on the V_q kernel's HLO size).  Cached by shape + same_zeta +
    write_g0 in ``_v_q_full_kernel_aot_cache``.

    The chooser uses this for shrink-retry sizing: if peak > budget,
    drop the unused cache entries at the end (the production run only
    keeps the final selected shape's compiled object).

    Returns ``None`` on AOT-compile failure — caller falls back to the
    slope+intercept FFT model.
    """
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])
    cache_key = (
        int(q_chunk), int(mu_size), int(nu_size),
        int(n_rmu_L), int(n_rmu_R), int(nq_total),
        int(n_rtot), int(n_G_sph), tuple(int(s) for s in fft_grid),
        p_x, p_y, bool(same_zeta), bool(write_g0),
    )
    hit = _v_q_full_kernel_aot_cache.get(cache_key)
    if hit is not None:
        return hit

    try:
        kernel = _make_V_q_tile_kernel(
            sphere_idx=sphere_idx, n_G_sph=n_G_sph,
            fft_shape=tuple(int(s) for s in fft_grid),
            mesh_xy=mesh_xy,
            q_chunk=int(q_chunk),
            mu_size=int(mu_size), nu_size=int(nu_size),
            n_rmu_L=int(n_rmu_L), n_rmu_R=int(n_rmu_R),
            same_zeta=bool(same_zeta), write_g0=bool(write_g0))

        # Specs must mirror the kernel's in_shardings exactly.
        V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
        g0_sh = NamedSharding(mesh_xy, P(None, 'x'))
        zeta_disk_sh = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
        v_per_G_sh = NamedSharding(mesh_xy, P(None, None))
        phase_sh = NamedSharding(mesh_xy, P(None, None, None, None, None))
        rep = NamedSharding(mesh_xy, P())
        nx, ny, nz = (int(s) for s in fft_grid)

        # All shapes match what the kernel's signature expects.
        V_acc_spec = jax.ShapeDtypeStruct(
            (int(nq_total), int(n_rmu_L), int(n_rmu_R)),
            jnp.complex128, sharding=V_sh)
        g0_acc_spec = jax.ShapeDtypeStruct(
            (int(nq_total), int(n_rmu_L)),
            jnp.complex128, sharding=g0_sh)
        zeta_L_spec = jax.ShapeDtypeStruct(
            (int(q_chunk), int(n_rtot), int(mu_size)),
            jnp.complex128, sharding=zeta_disk_sh)
        v_per_G_spec = jax.ShapeDtypeStruct(
            (int(q_chunk), int(n_G_sph)),
            jnp.complex128, sharding=v_per_G_sh)
        phase_spec = jax.ShapeDtypeStruct(
            (int(q_chunk), 1, nx, ny, nz),
            jnp.complex128, sharding=phase_sh)
        int_spec = jax.ShapeDtypeStruct((), jnp.int32, sharding=rep)

        if same_zeta:
            specs = (V_acc_spec, g0_acc_spec,
                     zeta_L_spec, v_per_G_spec, phase_spec,
                     int_spec, int_spec, int_spec)
        else:
            zeta_R_spec = jax.ShapeDtypeStruct(
                (int(q_chunk), int(n_rtot), int(nu_size)),
                jnp.complex128, sharding=zeta_disk_sh)
            specs = (V_acc_spec, g0_acc_spec,
                     zeta_L_spec, zeta_R_spec, v_per_G_spec, phase_spec,
                     int_spec, int_spec, int_spec)

        compiled = kernel.lower(*specs).compile(
            compiler_options={"xla_gpu_memory_limit_slop_factor": 10000})
        # AOT GPU peak = XLA-visible buffers + cuFFT plan scratch.  The
        # cuFFT scratch is queried via cufftGetSize on jaxlib's loaded
        # libcufft.so (see runtime.aot_memory) — no slope+intercept guess.
        from runtime.aot_memory import aot_kernel_peak_bytes
        breakdown = aot_kernel_peak_bytes(compiled)
        if (bool(int(os.environ.get('LORRAX_V_Q_AOT_VERBOSE', '0') or 0))
                and jax.process_index() == 0):
            print(f"  V_q full-kernel AOT breakdown q_chunk={q_chunk}: "
                  f"compiled={breakdown.compiled_peak/1e9:.2f} GB + "
                  f"cuFFT scratch={breakdown.cufft_scratch/1e9:.2f} GB "
                  f"= {breakdown.total/1e9:.2f} GB "
                  f"({len(breakdown.fft_specs)} fft op(s))")
        peak = breakdown.total
        _v_q_full_kernel_aot_cache[cache_key] = peak
        return peak
    except Exception:
        return None


def _drop_unused_v_q_kernel_cache_entries(
    keep_cache_keys: set[tuple],
) -> int:
    """Drop entries from ``_v_q_tile_kernel_cache`` whose cache_key is
    NOT in ``keep_cache_keys``.

    Called by the chooser after shrink-retry settles on a final
    ``q_chunk`` / ``mu_chunk``: the AOT compiles for unused candidate
    shapes leak into the cache via ``_make_V_q_tile_kernel``'s
    populate-on-build, so we scrub them.  Returns the count dropped.

    Note: JAX's internal pjit/lower compile cache also holds these
    artifacts; we don't reach inside that.  The Python-level
    LORRAX kernel cache that we own is the one we clean.
    """
    stale = [k for k in _v_q_tile_kernel_cache
             if k not in keep_cache_keys]
    for k in stale:
        _v_q_tile_kernel_cache.pop(k, None)
    return len(stale)


def _aot_fft_model(
    *,
    n_rmu: int,
    fft_grid: tuple[int, int, int],
    mesh_xy: Mesh,
    same_zeta: bool,
) -> tuple[int, int] | None:
    """AOT-measure the V_q production-kernel FFT stage memory and split
    into a Q-linear cost + Q-fixed workspace overhead.

    Compiles a tiny ``shard_map`` jit using ``make_sharded_fftn_3d`` —
    the SAME primitive the production ``_make_V_q_tile_kernel`` uses
    (NOT ``make_jittable_local_fftn_3d``, which compiles a different
    HLO with 3 sequential 1D-FFT plans and a different memory profile).
    Reads ``compiled.memory_analysis()`` and returns
    (slope, intercept) such that

        peak_fft_per_rank(Q) = slope · Q + intercept

    Method: measure at Q=1 and Q=2; take slope = peak(2) − peak(1) and
    intercept = peak(1) − slope.  Caches by (n_rmu, fft_grid,
    p_x, p_y, same_zeta).

    Distinct-ζ tiles double the slope (two ζ_disk inputs alive at the
    kernel boundary).  The cuFFT plan workspace stays the same.

    Returns ``None`` on AOT-compile failure — caller falls back to the
    flat heuristic.
    """
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])
    cache_key = (int(n_rmu), tuple(int(s) for s in fft_grid),
                 p_x, p_y, bool(same_zeta))
    hit = _v_q_aot_cache.get(cache_key)
    if hit is not None:
        return hit

    try:
        spec = P(None, ('x', 'y'), None, None, None)
        sharding = NamedSharding(mesh_xy, spec)
        # Production primitive — single shard_map'd 3-D cuFFT plan.
        local_fftn = make_sharded_fftn_3d(mesh_xy, spec, spec)
        jit_fft = jax.jit(local_fftn, out_shardings=sharding)

        def _peak_at_Q(Q: int) -> int:
            in_spec = jax.ShapeDtypeStruct(
                (Q, n_rmu, *fft_grid), jnp.complex128, sharding=sharding)
            compiled = jit_fft.lower(in_spec).compile(
                compiler_options={"xla_gpu_memory_limit_slop_factor": 10000})
            m = compiled.memory_analysis()
            return (int(m.temp_size_in_bytes)
                    + int(m.argument_size_in_bytes)
                    + int(m.output_size_in_bytes)
                    - int(m.alias_size_in_bytes))

        peak_q1 = _peak_at_Q(1)
        peak_q2 = _peak_at_Q(2)
        slope = max(0, peak_q2 - peak_q1)            # Q-linear cost
        intercept = max(0, peak_q1 - slope)          # Q-fixed workspace
        if not same_zeta:
            slope *= 2
        result = (slope, intercept)
        _v_q_aot_cache[cache_key] = result
        return result
    except Exception:
        return None


def _choose_v_q_chunks(
    *,
    n_rmu: int,
    n_G: int,
    n_q_total: int,
    budget_bytes: float,
    p_x: int,
    p_y: int,
    n_rtot: int | None = None,
    same_zeta: bool = True,
    mesh_xy: Mesh | None = None,
    fft_grid: tuple[int, int, int] | None = None,
    sphere_idx=None,
    write_g0: bool = True,
) -> dict:
    """Pick (q_chunk, μ_chunk) per the memory model documented above.

    Returns a dict with keys:
        q_chunk       : int — q-points per kernel call
        mu_chunk      : int — μ block size (== n_rmu in Case A)
        n_mu_blocks   : int — number of μ blocks in the tile loop (1 in A)
        tiled         : bool — True ⟺ Case B (μ × ν tiling)
        per_rank_peak : float — predicted per-rank peak bytes
        ref_bytes     : float — bytes reserved for V_ref + g0_ref per rank
        aligned       : bool — True ⟺ μ_chunk = N_μ/(P_x·k); ref slice-set
                              lands fully inside one rank's V_ref shard
                              (no collective on update)
    """
    N_zeta = 16.0  # c128
    # p_min was the conservative gather divisor used by the legacy
    # _Q_COMPUTE_COEF_GATHER heuristic; gone now that the gather stage
    # is modelled analytically as 1/p_x + 1/p_y + 1/(p_x·p_y).
    p_prod = float(int(p_x) * int(p_y))

    # Accumulator reservation (V_ref + g0_ref, per-rank sharded bytes).
    v_ref_bytes  = N_zeta * n_q_total * n_rmu * n_rmu / p_prod
    g0_ref_bytes = N_zeta * n_q_total * n_rmu / max(1, int(p_x))
    ref_bytes = v_ref_bytes + g0_ref_bytes
    B_compute = float(budget_bytes) - ref_bytes

    v_per_q_bytes = N_zeta * n_G  # sqrt_v(q+G) replicated per rank

    # Distinct ζ_L vs ζ_R: both disk slabs alive simultaneously at the
    # kernel input boundary, so the FFT-stage footprint doubles.  XLA can
    # free one buffer as the FFT consumes it, but the peak before either
    # FFT starts already has both buffers alive.
    fft_coef = _Q_COMPUTE_COEF_FFT * (1.0 if same_zeta else 2.0)

    one_q_gather = _gather_stage_per_q_bytes(
        n_rmu=int(n_rmu), n_G=int(n_G),
        p_x=int(p_x), p_y=int(p_y))

    # Prefer the AOT (slope, intercept) model from XLA memory_analysis
    # over the flat ``_Q_COMPUTE_COEF_FFT`` heuristic.  Models the FFT
    # stage as ``Q · per_q_input + fixed_workspace`` per rank.
    aot_slope = aot_intercept = None
    if (mesh_xy is not None and fft_grid is not None and
            n_rtot is not None and n_rtot > 0):
        aot = _aot_fft_model(
            n_rmu=int(n_rmu), fft_grid=tuple(int(s) for s in fft_grid),
            mesh_xy=mesh_xy, same_zeta=bool(same_zeta))
        if aot is not None:
            aot_slope, aot_intercept = aot
    if aot_slope is not None:
        # Linear-in-Q variable footprint per rank (ζ_disk slab).  The
        # fixed workspace is added once below in the q_chunk equation.
        one_q_fft = float(aot_slope)
        if bool(int(os.environ.get('LORRAX_V_Q_AOT_VERBOSE', '0') or 0)) and jax.process_index() == 0:
            print(f"  V_q AOT FFT model: slope={aot_slope/1e9:.3f} GB/Q, "
                  f"intercept={aot_intercept/1e9:.3f} GB (fixed workspace)")
    elif n_rtot is not None and n_rtot > 0:
        one_q_fft = fft_coef * N_zeta * n_rmu * float(n_rtot) / p_prod
    else:
        one_q_fft = 0.0
    one_q_bytes = max(one_q_gather, one_q_fft)
    # Q-fixed FFT workspace (cuFFT plan scratch) — only present when we
    # ran the AOT measurement; the heuristic path doesn't separate it.
    fft_workspace_fixed = float(aot_intercept) if aot_intercept is not None else 0.0

    # Case A check: can we hold *one* q's worth (whichever stage dominates)?
    # When the AOT model provides a fixed workspace, subtract it from the
    # compute budget once — it doesn't scale with q_chunk.
    B_compute_after_fixed = B_compute - fft_workspace_fixed
    slack_after_one_q = B_compute_after_fixed - (one_q_bytes + v_per_q_bytes)
    if slack_after_one_q < 0:
        # Case B — single q, tile μ × ν.  Gather-stage budget on μ_chunk:
        # peak_gather(μ_chunk) = (1/p_x + 1/p_y + 1/(p_x·p_y)) · 16 · μ_chunk · n_G
        gather_per_mu_row = _gather_stage_per_q_bytes(
            n_rmu=1, n_G=int(n_G),
            p_x=int(p_x), p_y=int(p_y))
        mu_chunk_max_gather = int(
            (B_compute_after_fixed - v_per_q_bytes) /
            max(1.0, gather_per_mu_row))
        if aot_slope is not None:
            # AOT-exact FFT slope scales linearly with μ (batch dim of
            # the FFT) at fixed Q=1; divide by n_rmu to get the per-μ-row
            # variable cost, then size mu_chunk to the remaining slack.
            mu_chunk_max_fft = int(
                (B_compute_after_fixed - v_per_q_bytes) /
                (aot_slope / max(1, int(n_rmu))))
        elif n_rtot is not None and n_rtot > 0:
            # Use the same same/distinct-aware coefficient for the tiled
            # μ × ν Case B FFT-stage cap.
            mu_chunk_max_fft = int(
                (B_compute - v_per_q_bytes) * p_prod /
                (fft_coef * N_zeta * float(n_rtot)))
        else:
            mu_chunk_max_fft = mu_chunk_max_gather
        mu_chunk_max = max(1, min(mu_chunk_max_gather, mu_chunk_max_fft))
        mu_chunk_max = min(int(n_rmu), int(mu_chunk_max))

        # CONSERVATIVE CHOICE of μ_chunk — must simultaneously satisfy:
        #   (i)  μ_chunk divides N_μ/P_x  AND  μ_chunk divides N_μ/P_y
        #   (ii) μ_chunk is divisible by P_x·P_y
        # Both hold simultaneously when μ_chunk divides gcd(N_μ/P_x,
        # N_μ/P_y) AND is a multiple of P_x·P_y.
        mu_per_x = int(n_rmu) // int(p_x) if int(n_rmu) % int(p_x) == 0 else 0
        mu_per_y = int(n_rmu) // int(p_y) if int(n_rmu) % int(p_y) == 0 else 0
        aligned_parent = gcd(mu_per_x, mu_per_y) if (mu_per_x and mu_per_y) else 0
        snap = int(p_x) * int(p_y) or 1

        aligned = False
        mu_chunk = None
        if aligned_parent and aligned_parent % snap == 0:
            candidates = sorted(
                (d for d in range(snap, aligned_parent + 1, snap)
                 if aligned_parent % d == 0 and d <= mu_chunk_max),
                reverse=True)
            if candidates:
                mu_chunk = candidates[0]
                aligned = True
        if mu_chunk is None:
            mu_chunk = max(snap, mu_chunk_max - (mu_chunk_max % snap))

        n_mu_blocks = (int(n_rmu) + mu_chunk - 1) // mu_chunk
        peak_gather = _gather_stage_per_q_bytes(
            n_rmu=int(mu_chunk), n_G=int(n_G),
            p_x=int(p_x), p_y=int(p_y))
        if aot_slope is not None:
            peak_fft = (aot_slope / max(1, int(n_rmu))) * mu_chunk + fft_workspace_fixed
        elif n_rtot is not None and n_rtot > 0:
            peak_fft = fft_coef * N_zeta * mu_chunk * float(n_rtot) / p_prod
        else:
            peak_fft = 0.0
        peak = max(peak_gather, peak_fft) + v_per_q_bytes + ref_bytes
        return dict(
            q_chunk=1, mu_chunk=int(mu_chunk), n_mu_blocks=int(n_mu_blocks),
            tiled=True, aligned=aligned,
            per_rank_peak=peak, ref_bytes=ref_bytes,
        )

    # Case A — fit at least one q with full μ.  Maximise q_chunk in
    # the post-fixed-workspace budget: q_chunk · (per_q_var + v_per_q)
    # ≤ B_compute_after_fixed.
    q_max = int(B_compute_after_fixed /
                (one_q_bytes + v_per_q_bytes)) if one_q_bytes > 0 else n_q_total
    q_chunk = max(1, min(n_q_total, q_max))
    peak = (q_chunk * one_q_bytes + q_chunk * v_per_q_bytes
            + fft_workspace_fixed + ref_bytes)

    # Optional: AOT-compile the *full* production kernel at the chosen
    # q_chunk and read its memory_analysis() — this captures the actual
    # peak including everything XLA tracks (cuFFT scratch, gather temps,
    # contract intermediates, DUS overlap, alias choices XLA makes
    # inside the production kernel that the standalone-FFT measurement
    # misses).  Shrink-retry if over budget.  Default-on; opt out with
    # ``LORRAX_V_Q_AOT_FULL_KERNEL=0`` (e.g. to debug the AOT path
    # itself, or in environments where the AOT compile fails).
    _aot_full = bool(int(os.environ.get('LORRAX_V_Q_AOT_FULL_KERNEL', '1') or 0))
    if (_aot_full and mesh_xy is not None and fft_grid is not None
            and n_rtot is not None and n_rtot > 0):
        cache_keys_visited: set[tuple] = set()
        _aot_verbose = bool(int(os.environ.get(
            'LORRAX_V_Q_AOT_VERBOSE', '0') or 0)) and jax.process_index() == 0
        attempt_q = int(q_chunk)
        verified_q = None
        verified_peak = None
        # Up to 4 shrink attempts before falling back to the cheap
        # analytical pick.
        for _ in range(4):
            if attempt_q < 1:
                break
            # The cache key in _v_q_tile_kernel_cache uses (n_rmu, n_rmu)
            # for ν_size=μ_size=N_μ in Case A.
            cache_keys_visited.add((
                'unified', id(mesh_xy), int(attempt_q),
                int(n_rmu), int(n_rmu),
                int(n_rmu), int(n_rmu), int(n_G),
                tuple(int(s) for s in fft_grid),
                id(sphere_idx), bool(same_zeta), bool(write_g0),
            ))
            full_peak = _aot_full_kernel_peak(
                q_chunk=attempt_q,
                mu_size=int(n_rmu), nu_size=int(n_rmu),
                n_rmu_L=int(n_rmu), n_rmu_R=int(n_rmu),
                nq_total=int(n_q_total),
                n_rtot=int(n_rtot), n_G_sph=int(n_G),
                fft_grid=tuple(int(s) for s in fft_grid),
                sphere_idx=sphere_idx,
                mesh_xy=mesh_xy,
                same_zeta=bool(same_zeta), write_g0=bool(write_g0))
            if full_peak is None:
                break  # AOT failed — fall back to analytical pick
            if _aot_verbose:
                print(f"  V_q full-kernel AOT: q_chunk={attempt_q} "
                      f"predicted peak/rank={full_peak/1e9:.2f} GB "
                      f"(budget={budget_bytes/1e9:.2f} GB)")
            if float(full_peak) <= float(budget_bytes):
                verified_q = attempt_q
                verified_peak = full_peak
                break
            # Shrink; aim for ~0.7× of current attempt rounded down.
            new_attempt = max(1, int(attempt_q * 0.7))
            if new_attempt >= attempt_q:
                new_attempt = attempt_q - 1
            attempt_q = new_attempt
        if verified_q is not None:
            # Drop kernel-cache entries we built but won't use.
            keep = {(
                'unified', id(mesh_xy), int(verified_q),
                int(n_rmu), int(n_rmu),
                int(n_rmu), int(n_rmu), int(n_G),
                tuple(int(s) for s in fft_grid),
                id(sphere_idx), bool(same_zeta), bool(write_g0),
            )}
            _drop_unused_v_q_kernel_cache_entries(keep)
            q_chunk = verified_q
            peak = float(verified_peak) + ref_bytes
        # else: fall through with the analytically-picked q_chunk + peak.
    return dict(
        q_chunk=q_chunk, mu_chunk=int(n_rmu), n_mu_blocks=1,
        tiled=False, aligned=True,
        per_rank_peak=peak, ref_bytes=ref_bytes,
    )


# ----------------------------------------------------------------------------
# Unified inner V_q tile kernel
# ----------------------------------------------------------------------------

_v_q_tile_kernel_cache: dict = {}


def _make_V_q_tile_kernel(
    *,
    sphere_idx,
    n_G_sph: int,
    fft_shape: tuple[int, int, int],
    mesh_xy: Mesh,
    q_chunk: int,
    mu_size: int,
    nu_size: int,
    n_rmu_L: int,
    n_rmu_R: int,
    same_zeta: bool,
    write_g0: bool,
):
    """Build the unified inner V_q kernel.

    Parameters
    ----------
    sphere_idx : jax.Array | None
        G-sphere indices into the flat FFT box (or None when no cutoff).
    n_G_sph : int
        Post-cutoff G-vector count (== prod(fft_shape) when sphere_idx is None).
    fft_shape : (nx, ny, nz)
        FFT-box dims.
    mesh_xy : Mesh
        2-D device mesh with axes ('x', 'y').
    q_chunk : int
        q-points per kernel call (Case A: q-batch size; Case B: 1).
    mu_size, nu_size : int
        Per-call block sizes (cache key).  In Case A both equal n_rmu_L /
        n_rmu_R; in Case B they're the chunked sizes (with min(mu_chunk,
        n_rmu - offset) for the last partial block).
    n_rmu_L, n_rmu_R : int
        Total μ counts on the left / right ζ — set the V_acc accumulator
        extents.  Equal for V^{0,0}; may differ for bispinor.
    same_zeta : bool (static)
        True → one ζ input, one FFT, two gathers.  False → two ζ inputs,
        two FFTs, two gathers.
    write_g0 : bool (static)
        Only valid when same_zeta and (μ_lo == ν_lo).  Diagonal blocks
        write ζ_μ(G=0) into the g0 accumulator.

    Returns
    -------
    A jit-compiled kernel with signature::

        kernel(V_acc, g0_acc,
               zeta_mu_disk, [zeta_nu_disk,]   # second arg only when not same_zeta
               v_per_G_batch, phase_batch,
               q_lo_dyn, mu_lo_dyn, nu_lo_dyn) -> (V_acc', g0_acc')

    plus the convenience attributes ``zeta_disk_sh``, ``V_sh``, ``g0_sh``.
    """
    cache_key = (
        'unified', id(mesh_xy), q_chunk, mu_size, nu_size,
        n_rmu_L, n_rmu_R, n_G_sph, tuple(fft_shape),
        id(sphere_idx), bool(same_zeta), bool(write_g0),
    )
    hit = _v_q_tile_kernel_cache.get(cache_key)
    if hit is not None:
        return hit

    nx, ny, nz = fft_shape
    n_rtot = nx * ny * nz

    # Sharding specs.  These match the chi0 / CCT / ZCT / sigma chain
    # downstream: V_block lands natively in P(None, 'x', 'y').
    blk_xy_sh = NamedSharding(mesh_xy, P(None, ('x', 'y'), None))
    blk_xy_5d_spec = P(None, ('x', 'y'), None, None, None)
    blk_xy_5d_sh = NamedSharding(mesh_xy, blk_xy_5d_spec)
    blk_x_sh = NamedSharding(mesh_xy, P(None, 'x', None))
    blk_y_sh = NamedSharding(mesh_xy, P(None, 'y', None))
    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))
    phase_sh = NamedSharding(mesh_xy, P(None, None, None, None, None))  # (Q,1,nx,ny,nz)
    v_per_G_sh = NamedSharding(mesh_xy, P(None, None))                    # (Q, n_G_sph)
    zeta_disk_sh = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
    rep = NamedSharding(mesh_xy, P())

    _local_fftn_3d = make_sharded_fftn_3d(
        mesh_xy, blk_xy_5d_spec, blk_xy_5d_spec)

    # ------------------------------------------------------------------
    # Top-level read+FFT+sphere helper.
    #
    # Takes a ζ slab in the disk layout ``P(None, None, ('x','y'))``
    # (μ flat-sharded across the (x,y) mesh axes), reshards μ → leading
    # second axis with the same flat ('x','y') sharding ``P(None,
    # ('x','y'), None)``, runs the 5-D μ-XY-sharded FFT (custom-
    # partitioned cuFFT plan), and gathers onto the flat-index G sphere.
    # Returns ``(zeta_G, g0_blk)``:
    #   zeta_G : (Q, μ_per_rank, n_G_sph)  P(None, ('x','y'), None)
    #   g0_blk : (Q, μ_per_rank)            P(None, ('x','y'))
    #
    # No v(K) multiply here — that is applied once on the L side post-
    # gather (see the kernel body below).
    # ------------------------------------------------------------------
    def _zeta_disk_to_G(zeta_disk, phase_batch):
        zeta_mu_r = jax.lax.with_sharding_constraint(
            jnp.transpose(zeta_disk, (0, 2, 1)), blk_xy_sh)
        Q, mu_per_rank, _ = zeta_mu_r.shape
        zeta_5d = jax.lax.with_sharding_constraint(
            zeta_mu_r.reshape(Q, mu_per_rank, nx, ny, nz) * phase_batch,
            blk_xy_5d_sh)
        zeta_box = _local_fftn_3d(zeta_5d)
        zeta_box = jax.lax.with_sharding_constraint(
            zeta_box.reshape(Q, mu_per_rank, n_rtot), blk_xy_sh)
        g0_blk = zeta_box[:, :, 0]
        if sphere_idx is not None:
            zeta_G = jnp.take(zeta_box, sphere_idx, axis=-1)
        else:
            zeta_G = zeta_box
        zeta_G = jax.lax.with_sharding_constraint(zeta_G, blk_xy_sh)
        return zeta_G, g0_blk

    # ------------------------------------------------------------------
    # Shared kernel body.  ``same_zeta`` is a Python-static bool — when
    # True, the second ζ_R G-space form is just an alias of ζ_L_G
    # (single read + single FFT path).  When False, ζ_R is read+FFT'd
    # independently from a distinct file.  Everything after the read
    # branch is identical (two one-axis gathers, one-sided v multiply,
    # einsum, DUS write).
    # ------------------------------------------------------------------
    def _kernel_body(V_acc, g0_acc,
                     zeta_L_disk, zeta_R_disk_or_None,
                     v_per_G_batch, phase_batch,
                     q_lo_dyn, mu_lo_dyn, nu_lo_dyn):
        zeta_L_G, g0_mu = _zeta_disk_to_G(zeta_L_disk, phase_batch)
        if same_zeta:
            zeta_R_G = zeta_L_G                       # static reuse
        else:
            zeta_R_G, _ = _zeta_disk_to_G(
                zeta_R_disk_or_None, phase_batch)     # second read+FFT

        zeta_mu_X = jax.lax.with_sharding_constraint(zeta_L_G, blk_x_sh)
        zeta_nu_Y = jax.lax.with_sharding_constraint(zeta_R_G, blk_y_sh)

        # One-sided v(K) multiply on the L (μ) side — handles signed
        # transverse-projector weights for bispinor off-diagonal tiles;
        # bit-identical to symmetric-√v for V^{0,0}.
        zeta_mu_X = jax.lax.with_sharding_constraint(
            zeta_mu_X * v_per_G_batch[:, None, :], blk_x_sh)

        V_block = jnp.einsum('qmG,qnG->qmn',
                             jnp.conj(zeta_mu_X), zeta_nu_Y,
                             optimize=True)
        V_block = jax.lax.with_sharding_constraint(V_block, V_sh)

        V_new = jax.lax.dynamic_update_slice(
            V_acc, V_block, (q_lo_dyn, mu_lo_dyn, nu_lo_dyn))
        V_new = jax.lax.with_sharding_constraint(V_new, V_sh)

        if write_g0:
            g0_mu = jax.lax.with_sharding_constraint(g0_mu, g0_sh)
            g0_new = jax.lax.dynamic_update_slice(
                g0_acc, g0_mu, (q_lo_dyn, mu_lo_dyn))
            g0_new = jax.lax.with_sharding_constraint(g0_new, g0_sh)
        else:
            g0_new = g0_acc
        return V_new, g0_new

    # Two jit signatures — same/distinct ζ differ only in argument count.
    # The bodies share ``_kernel_body`` via the Python-static ``same_zeta``
    # flag; XLA gets the right HLO either way.
    if same_zeta:
        @partial(jax.jit,
                 in_shardings=(V_sh, g0_sh, zeta_disk_sh,
                               v_per_G_sh, phase_sh, rep, rep, rep),
                 out_shardings=(V_sh, g0_sh),
                 donate_argnums=(0, 1))
        def _kernel(V_acc, g0_acc,
                    zeta_L_disk,
                    v_per_G_batch, phase_batch,
                    q_lo_dyn, mu_lo_dyn, nu_lo_dyn):
            return _kernel_body(
                V_acc, g0_acc, zeta_L_disk, None,
                v_per_G_batch, phase_batch,
                q_lo_dyn, mu_lo_dyn, nu_lo_dyn)
    else:
        assert not write_g0, (
            "write_g0=True is only valid when same_zeta=True (V^{0,0} "
            "self-contraction); the bispinor V^{μ_L, ν_L} (L ≠ R) head "
            "term is not defined.")

        @partial(jax.jit,
                 in_shardings=(V_sh, g0_sh, zeta_disk_sh, zeta_disk_sh,
                               v_per_G_sh, phase_sh, rep, rep, rep),
                 out_shardings=(V_sh, g0_sh),
                 donate_argnums=(0, 1))
        def _kernel(V_acc, g0_acc,
                    zeta_L_disk, zeta_R_disk,
                    v_per_G_batch, phase_batch,
                    q_lo_dyn, mu_lo_dyn, nu_lo_dyn):
            return _kernel_body(
                V_acc, g0_acc, zeta_L_disk, zeta_R_disk,
                v_per_G_batch, phase_batch,
                q_lo_dyn, mu_lo_dyn, nu_lo_dyn)

    _kernel.zeta_disk_sh = zeta_disk_sh
    _kernel.V_sh = V_sh
    _kernel.g0_sh = g0_sh
    _v_q_tile_kernel_cache[cache_key] = _kernel
    return _kernel


_v_q_tile_gflat_kernel_cache: dict = {}


def _make_V_q_tile_kernel_Gflat(
    *,
    n_G_sph: int,
    g0_sphere_idx: int,
    mesh_xy: Mesh,
    q_chunk: int,
    mu_size: int,
    nu_size: int,
    n_rmu_L: int,
    n_rmu_R: int,
    same_zeta: bool,
    write_g0: bool,
):
    """V_q tile kernel — G-flat ζ input variant.

    Identical mathematics to :func:`_make_V_q_tile_kernel` except the ζ
    inputs are already in G-flat layout (the FFT + per-q phase + sphere
    gather happen upstream in :class:`file_io.zeta_reader.ZetaReader`
    rather than inside this kernel).

    Per-kernel work shrinks to:
      * an optional shard cast to ``P(None, 'x', None)`` / ``P(None, 'y', None)``
        for the μ / ν halves of the contraction,
      * a one-sided ``v(K)`` multiply on the L side,
      * the ``einsum('qmG,qnG->qmn')`` contraction with one-axis gathers,
      * (optional) g0 gather at ``G_idx = g0_sphere_idx``.

    Parameters
    ----------
    n_G_sph : int
        Post-cutoff G-vector count (the trailing axis of the input ζ_G).
    g0_sphere_idx : int
        Index in ``[0, n_G_sph)`` of the G = (0, 0, 0) entry.  The
        reader builds ζ_G on a sphere ``sphere_idx`` whose 0th entry
        is conventionally Γ; if a different convention is used,
        pass that index here.  Only consulted when ``write_g0`` is set.
    mesh_xy, q_chunk, mu_size, nu_size, n_rmu_L, n_rmu_R,
    same_zeta, write_g0
        Mirror the r-space kernel's parameters.

    Returns
    -------
    A jit-compiled kernel with signature::

        same_zeta=True:
          kernel(V_acc, g0_acc, zeta_L_G, v_per_G_batch,
                 q_lo_dyn, mu_lo_dyn, nu_lo_dyn) -> (V_acc', g0_acc')

        same_zeta=False:
          kernel(V_acc, g0_acc, zeta_L_G, zeta_R_G, v_per_G_batch,
                 q_lo_dyn, mu_lo_dyn, nu_lo_dyn) -> (V_acc', g0_acc')

    plus the convenience attributes ``zeta_G_sh``, ``V_sh``, ``g0_sh``.
    """
    cache_key = (
        'gflat', id(mesh_xy), q_chunk, mu_size, nu_size,
        n_rmu_L, n_rmu_R, n_G_sph,
        bool(same_zeta), bool(write_g0),
        int(g0_sphere_idx) if write_g0 else -1,
    )
    hit = _v_q_tile_gflat_kernel_cache.get(cache_key)
    if hit is not None:
        return hit

    blk_xy_sh = NamedSharding(mesh_xy, P(None, ('x', 'y'), None))
    blk_x_sh = NamedSharding(mesh_xy, P(None, 'x', None))
    blk_y_sh = NamedSharding(mesh_xy, P(None, 'y', None))
    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))
    v_per_G_sh = NamedSharding(mesh_xy, P(None, None))
    rep = NamedSharding(mesh_xy, P())

    def _kernel_body(V_acc, g0_acc,
                     zeta_L_G, zeta_R_G_or_None,
                     v_per_G_batch,
                     q_lo_dyn, mu_lo_dyn, nu_lo_dyn):
        # Already G-flat with μ on the ('x','y') product axis.
        zeta_L_G = jax.lax.with_sharding_constraint(zeta_L_G, blk_xy_sh)
        if same_zeta:
            zeta_R_G = zeta_L_G
        else:
            zeta_R_G = jax.lax.with_sharding_constraint(
                zeta_R_G_or_None, blk_xy_sh)

        # Re-shard for the einsum's two-axis split.
        zeta_mu_X = jax.lax.with_sharding_constraint(zeta_L_G, blk_x_sh)
        zeta_nu_Y = jax.lax.with_sharding_constraint(zeta_R_G, blk_y_sh)

        # One-sided v(K) multiply.
        zeta_mu_X = jax.lax.with_sharding_constraint(
            zeta_mu_X * v_per_G_batch[:, None, :], blk_x_sh)

        V_block = jnp.einsum('qmG,qnG->qmn',
                              jnp.conj(zeta_mu_X), zeta_nu_Y,
                              optimize=True)
        V_block = jax.lax.with_sharding_constraint(V_block, V_sh)
        V_new = jax.lax.dynamic_update_slice(
            V_acc, V_block, (q_lo_dyn, mu_lo_dyn, nu_lo_dyn))
        V_new = jax.lax.with_sharding_constraint(V_new, V_sh)

        if write_g0:
            # ζ_L(G=0) = entry at sphere index ``g0_sphere_idx``.
            g0_mu = zeta_L_G[:, :, int(g0_sphere_idx)]
            g0_mu = jax.lax.with_sharding_constraint(g0_mu, g0_sh)
            g0_new = jax.lax.dynamic_update_slice(
                g0_acc, g0_mu, (q_lo_dyn, mu_lo_dyn))
            g0_new = jax.lax.with_sharding_constraint(g0_new, g0_sh)
        else:
            g0_new = g0_acc

        return V_new, g0_new

    if same_zeta:
        @partial(jax.jit,
                 in_shardings=(V_sh, g0_sh, blk_xy_sh, v_per_G_sh,
                                rep, rep, rep),
                 out_shardings=(V_sh, g0_sh),
                 donate_argnums=(0, 1))
        def _kernel(V_acc, g0_acc, zeta_L_G,
                    v_per_G_batch,
                    q_lo_dyn, mu_lo_dyn, nu_lo_dyn):
            return _kernel_body(
                V_acc, g0_acc, zeta_L_G, None,
                v_per_G_batch,
                q_lo_dyn, mu_lo_dyn, nu_lo_dyn)
    else:
        assert not write_g0, (
            "write_g0=True is only valid when same_zeta=True (V^{0,0} "
            "self-contraction); bispinor V^{μ_L,ν_L} L≠R head is "
            "not defined."
        )

        @partial(jax.jit,
                 in_shardings=(V_sh, g0_sh, blk_xy_sh, blk_xy_sh,
                                v_per_G_sh, rep, rep, rep),
                 out_shardings=(V_sh, g0_sh),
                 donate_argnums=(0, 1))
        def _kernel(V_acc, g0_acc, zeta_L_G, zeta_R_G,
                    v_per_G_batch,
                    q_lo_dyn, mu_lo_dyn, nu_lo_dyn):
            return _kernel_body(
                V_acc, g0_acc, zeta_L_G, zeta_R_G,
                v_per_G_batch,
                q_lo_dyn, mu_lo_dyn, nu_lo_dyn)

    _kernel.zeta_G_sh = blk_xy_sh
    _kernel.V_sh = V_sh
    _kernel.g0_sh = g0_sh
    _v_q_tile_gflat_kernel_cache[cache_key] = _kernel
    return _kernel


def _make_g0_dummy(mesh_xy, nq_total, n_rmu_L):
    """Sharded dummy g0 accumulator for tiles without a head.

    The unified kernel always takes a g0_acc argument (it's in
    ``donate_argnums``); when the caller doesn't want the head, we carry
    one dummy zero slab through the loop so donation ownership stays
    explicit without per-block allocation.
    """
    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))

    @partial(jax.jit, out_shardings=g0_sh)
    def _zeros():
        return jnp.zeros((nq_total, n_rmu_L), dtype=jnp.complex128)

    return _zeros()


# ----------------------------------------------------------------------------
# Outer driver — single function for every V_q tile shape and both backends
# ----------------------------------------------------------------------------


def compute_V_q_tile(
    *,
    zeta_L_io,
    zeta_R_io=None,
    v_per_G_fn,
    phase_fn,
    sphere_idx,
    fft_grid: tuple[int, int, int],
    mesh_xy: Mesh,
    kgrid: tuple[int, int, int],
    n_rmu_L: int,
    n_rmu_R: int | None = None,
    V_acc=None,
    g0_acc=None,
    chooser_choice: dict | None = None,
    budget_bytes: float | None = None,
    bgw_v_grid_overlay_fn=None,
    verbose: bool = True,
    timing_label: str = "compute_V_q_tile",
    q_list_kgrid_int: np.ndarray | None = None,
    use_g_flat_zeta: bool = False,
):
    """Outer driver for the unified V_q tile.

    Single function; covers:
      * V^{0,0} (same_zeta=True) and bispinor V^{μ_L, ν_L} (same_zeta=False)
      * Case A (q-batched) and Case B (μ × ν tiled)
      * PHDF5/FFI and h5py-allgather backends (dispatched inside SlabIO)

    Parameters
    ----------
    zeta_L_io, zeta_R_io : SlabIO
        Open SlabIO handles in mode='r'.  When ``zeta_R_io is None`` (or
        ``zeta_R_io is zeta_L_io``) the kernel runs in single-ζ mode (one
        FFT, two gathers) — the V^{0,0} self-contraction.
    v_per_G_fn : callable((Q,3) qvec_np) -> jnp.ndarray (Q, n_G_sph) c128
        Returns the per-G weight applied on the L side of the contraction.
        Any real-valued (signed) weight allowed; for V^{0,0} this is
        ``v(q+G)`` (real, ≥ 0) and the math is bit-identical to the old
        symmetric-√v form.  For bispinor V^{μ_L, ν_L} it is
        ``v(K) · t_{μ_L, ν_L}(K)`` with signed transverse-projector ``t``.
    phase_fn : callable((Q,3) qvec_np) -> jnp.ndarray (Q, 1, nx, ny, nz) c128
        Per-q FFT-box phase factor ``exp(-2πi q·r)``.  Independent of the
        v side (factored out so any Coulomb dimensionality plugs in).
    sphere_idx : jax.Array | None
        G-sphere indices into the flat FFT box, or None when no cutoff.
    fft_grid : (nx, ny, nz)
    mesh_xy : Mesh
        2-D device mesh with axes ('x', 'y').
    kgrid : (nkx, nky, nkz)
    n_rmu_L, n_rmu_R : int
        Total μ counts.  ``n_rmu_R`` defaults to ``n_rmu_L`` (V^{0,0}).
    V_acc : jnp.ndarray (n_q_total, n_rmu_L, n_rmu_R), P(None,'x','y')
        Donated mutable accumulator.  Caller pre-allocates with
        ``jnp.zeros`` of the right layout, OR pass None to have this
        function allocate it.
    g0_acc : jnp.ndarray | None (n_q_total, n_rmu_L), P(None, 'x')
        Donated g0 accumulator — only used when ``same_zeta`` and the
        caller wants the head term.  Pass None for bispinor off-diagonal
        tiles or when the caller doesn't need g0.
    chooser_choice : dict | None
        Output of :func:`_choose_v_q_chunks`.  If None, the chooser is
        run with ``budget_bytes`` (mandatory in that case).
    budget_bytes : float | None
        Memory budget per device, bytes.  Required when chooser_choice is None.
    bgw_v_grid_overlay_fn : callable | None
        Optional host-side overlay applied to ``v_per_G`` before passing
        to the kernel.  Signature::
            f(qvec_np_batch, v_per_G_native_np) -> v_per_G_overlaid_np
        Used for byte-reproducible BGW comparisons (``use_bgw_vcoul=true``).
    verbose : bool
    timing_label : str
    q_list_kgrid_int : np.ndarray | None
        Optional (n_q, 3) int array of q-points (kgrid coords) to
        iterate.  When ``None`` (default), the full BZ kgrid is built
        in the canonical flat-q order.  When given, ``nq_total`` is
        replaced by ``len(q_list_kgrid_int)`` for V_acc allocation
        AND ``zeta_q`` read offsets — the caller is responsible for
        ensuring ζ on disk is indexed in the same order (typically
        an IBZ subset via ``SymMaps.find_irreducible_qpoints()``).
    use_g_flat_zeta : bool
        Opt-in (default False) to consume ζ in **G-flat** layout via
        the new :class:`file_io.zeta_reader.ZetaReader` read path —
        the FFT + per-q phase + sphere gather happen in the reader
        rather than inside the V_q kernel.  When True,
        ``zeta_L_io`` / ``zeta_R_io`` must be :class:`ZetaReader`
        instances (the legacy r-space path expects raw
        :class:`SlabIO` handles).  The dispatch is internal — the
        kernel call signature differs between the two paths but
        callers don't see it.

    Returns
    -------
    (V_acc, g0_acc_or_None)
        Both buffers in their final filled state.  When ``g0_acc`` was None
        on entry, returns ``(V_acc, None)``.

    Notes
    -----
    * The chooser's ``aligned`` knob and the (mu_size, nu_size) cache key
      give the standard 4-compile-shape envelope (or 8 with write_g0).  In
      the chooser-aligned case all interior iterations share one shape.
    * The ``LORRAX_V_Q_MU_CHUNK`` env override is honoured by the dispatcher
      ``compute_all_V_q`` before this function — this driver does not
      consult env vars.
    """
    from common import timing
    from common.progress import LoopProgress

    nkx, nky, nkz = kgrid
    # ``nq_total`` is the size of the q-axis the orchestrator iterates
    # over AND the leading axis of V_acc / g0_acc / on-disk zeta_q.
    # Default = full BZ; IBZ caller passes ``q_list_kgrid_int`` to
    # iterate only the IBZ subset (the V_q centroid-double-permute
    # unfold to full BZ is the caller's responsibility).
    if q_list_kgrid_int is None:
        q_list_np = np.array(
            [(qx, qy, qz) for qx in range(nkx)
             for qy in range(nky) for qz in range(nkz)],
            dtype=np.int32)
    else:
        q_list_np = np.asarray(q_list_kgrid_int, dtype=np.int32)
        if q_list_np.ndim != 2 or q_list_np.shape[1] != 3:
            raise ValueError(
                f"q_list_kgrid_int must be (n_q, 3) int; "
                f"got shape {q_list_np.shape}")
    nq_total = int(q_list_np.shape[0])
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])

    n_rtot = int(np.prod(fft_grid))
    if n_rmu_R is None:
        n_rmu_R = n_rmu_L

    same_zeta = (zeta_R_io is None) or (zeta_R_io is zeta_L_io)

    if same_zeta and n_rmu_L != n_rmu_R:
        raise ValueError(
            f"compute_V_q_tile: same_zeta=True requires n_rmu_L == n_rmu_R, "
            f"got {n_rmu_L} vs {n_rmu_R}.")

    # n_rmu_L / n_rmu_R as passed by callers are LOGICAL centroid counts
    # (the on-disk extent of zeta_q).  We pad each to the mesh device-
    # product (``p_x · p_y``) so every interior sharding — single-axis
    # P(None, 'x', None), product P(None, ('x','y'), None), the V output
    # P(None, 'x', 'y'), the disk read P(None, None, ('x','y')) — divides
    # cleanly without any in-jit uneven-WSC fallbacks.  Trailing pad
    # rows/cols are zero-filled (SlabIO reads ``valid_shape=`` zero-fill;
    # internal allocs start at jnp.zeros).  Output V_acc is returned at
    # PADDED extent; callers carry the logical sizes alongside and pass
    # ``valid_shape=(..., n_rmu_L_logical, n_rmu_R_logical)`` to SlabIO
    # writes so on-disk extent stays logical and round-trips across
    # process counts.  Mirrors the band-axis pattern (Meta.b_id_4_user
    # vs Meta.b_id_4) at common/meta.py:99-100 + load_wfns.py:952-959.
    _proc_devices = p_x * p_y
    n_rmu_L_logical = int(n_rmu_L)
    n_rmu_R_logical = int(n_rmu_R)
    def _round_up_to_mesh(n: int) -> int:
        rem = n % _proc_devices
        return n + ((_proc_devices - rem) if rem else 0)
    n_rmu_L = _round_up_to_mesh(n_rmu_L_logical)
    n_rmu_R = _round_up_to_mesh(n_rmu_R_logical)
    if (n_rmu_L != n_rmu_L_logical or n_rmu_R != n_rmu_R_logical) \
            and verbose and jax.process_index() == 0:
        print(f"  V_q tile: padding μ for sharding — "
              f"n_rmu_L={n_rmu_L_logical}→{n_rmu_L}, "
              f"n_rmu_R={n_rmu_R_logical}→{n_rmu_R} "
              f"(mesh={p_x}×{p_y}={_proc_devices}-divisible)")

    if sphere_idx is not None:
        n_G_sph = int(np.asarray(sphere_idx).shape[0])
    else:
        n_G_sph = fft_grid[0] * fft_grid[1] * fft_grid[2]

    # ``wants_g0`` is needed by the chooser's full-kernel AOT path
    # (kernel cache key includes ``write_g0``), so we resolve it here
    # before calling the chooser; the same value is reused below to
    # decide whether to allocate g0_acc and to drive the per-iter
    # ``do_write_g0`` static flag.
    wants_g0 = (g0_acc is not None) or same_zeta

    if chooser_choice is None:
        if budget_bytes is None:
            raise ValueError(
                "compute_V_q_tile: must pass either chooser_choice or "
                "budget_bytes.")
        chooser_choice = _choose_v_q_chunks(
            n_rmu=max(n_rmu_L, n_rmu_R), n_G=n_G_sph, n_q_total=nq_total,
            budget_bytes=budget_bytes, p_x=p_x, p_y=p_y,
            n_rtot=n_rtot, same_zeta=same_zeta,
            mesh_xy=mesh_xy, fft_grid=fft_grid,  # enable AOT FFT peak
            sphere_idx=sphere_idx,
            write_g0=(same_zeta and wants_g0),
        )
    # Diagnostic env override: force a specific q_chunk in Case A,
    # bypassing the chooser's memory model.  Useful for stress-testing
    # whether the chooser leaves headroom on the table.
    _force_q = int(os.environ.get('LORRAX_V_Q_Q_CHUNK', '0') or 0)
    if _force_q > 0 and _force_q <= nq_total and not chooser_choice.get('tiled'):
        _orig_q_chunk = int(chooser_choice.get('q_chunk', 0))
        chooser_choice = dict(chooser_choice)
        chooser_choice['q_chunk'] = _force_q
        if jax.process_index() == 0:
            print(f"  [LORRAX_V_Q_Q_CHUNK] forcing q_chunk={_force_q} "
                  f"(chooser had {_orig_q_chunk})")

    q_chunk = int(chooser_choice['q_chunk'])
    mu_chunk = int(chooser_choice['mu_chunk'])
    n_mu_blocks_L = int(chooser_choice.get('n_mu_blocks_L',
                                            chooser_choice['n_mu_blocks']))
    n_mu_blocks_R = int(chooser_choice.get('n_mu_blocks_R',
                                            chooser_choice['n_mu_blocks']))

    # Allocate accumulators if caller didn't.
    V_sh_full = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    g0_sh_full = NamedSharding(mesh_xy, P(None, 'x'))
    if V_acc is None:
        @partial(jax.jit, out_shardings=V_sh_full)
        def _init_V():
            return jnp.zeros((nq_total, n_rmu_L, n_rmu_R), dtype=jnp.complex128)
        V_acc = _init_V()

    wants_g0 = g0_acc is not None
    if g0_acc is None and same_zeta:
        # Caller didn't pre-allocate but the head term defaults to "wanted"
        # for V^{0,0}; allocate it.  Distinct-ζ off-diagonal callers pass
        # ``g0_acc=None`` explicitly to opt out — they get a cheap dummy
        # below (one alloc; reused across the whole tile loop).
        @partial(jax.jit, out_shardings=g0_sh_full)
        def _init_g0():
            return jnp.zeros((nq_total, n_rmu_L), dtype=jnp.complex128)
        g0_acc = _init_g0()
        wants_g0 = True
    # Distinct-ζ off-diagonal tiles still need *some* g0 buffer flowing
    # through the kernel signature (donate_argnums tracks position 1), so
    # we allocate a single small dummy zero slab and reuse it across all
    # iterations — XLA donates and returns the same buffer in-place.
    g0_work = g0_acc if wants_g0 else _make_g0_dummy(
        mesh_xy, nq_total, n_rmu_L)

    kgrid_arr = np.array([nkx, nky, nkz], dtype=np.float64)
    _read_spec = P(None, None, ('x', 'y'))

    def _qvec_wrap(qx, qy, qz):
        qvec = np.array([qx, qy, qz], dtype=np.float64)
        return np.where(qvec > kgrid_arr / 2, qvec - kgrid_arr, qvec)

    def _v_phase_batch(qvec_list, pad_to: int):
        """Build (v_per_G, phase) for a batch of q's, with optional pad
        and optional BGW vcoul overlay.  Both outputs are jax arrays.
        """
        n_actual = len(qvec_list)
        qvec_np = np.stack([np.asarray(qw, dtype=np.float64)
                            for qw in qvec_list], axis=0)
        if n_actual < pad_to:
            pad = np.tile(qvec_np[0:1], (pad_to - n_actual, 1))
            qvec_np = np.concatenate([qvec_np, pad], axis=0)
        v_per_G = v_per_G_fn(qvec_np)        # (Q, n_G_sph) c128
        phase = phase_fn(qvec_np)            # (Q, 1, nx, ny, nz) c128
        if bgw_v_grid_overlay_fn is not None:
            v_per_G = bgw_v_grid_overlay_fn(qvec_np, v_per_G)
        return v_per_G, phase

    vq_progress = LoopProgress(
        nq_total, print, title="V_q tile",
        item_name="q-point", max_updates=min(nq_total, 20))

    # ζ read uses ``zeta_io.read_slab(..., partition_spec=_read_spec)``.
    # PHDF5 backend lands the slab directly in μ-on-XY layout; allgather
    # backend rank-0 reads then ``device_put``s with the same sharding.
    # Either way the kernel input is properly sharded.
    #
    # === Single nested loop covering both Case A (q-batched, μ_chunk=N_μ,
    # n_mu_blocks=1) and Case B (q_chunk=1, μ-tiled, n_mu_blocks≥1).  The
    # chooser sets ``q_chunk`` and ``mu_chunk`` such that exactly one of
    # ``q_chunk > 1`` or ``n_mu_blocks > 1`` is true, never both.  The
    # ``on_diag`` static check picks the single-FFT (single-read) path
    # only when ``same_zeta`` AND the μ/ν blocks coincide; off-diagonal
    # iterations always read+FFT a second ζ.
    # Iterate the q-list passed by the caller (full BZ or IBZ subset).
    # The on-disk ``zeta_q`` leading axis must be indexed in the same
    # order as ``q_list_np`` — under IBZ the writer slices to
    # ``q_irr_full_idx`` order; here ``q_flat0`` = position in the
    # iteration list (= IBZ index on disk).
    q_coords = [tuple(int(v) for v in row) for row in q_list_np]
    q_batches = [q_coords[i:i + q_chunk]
                 for i in range(0, nq_total, q_chunk)]
    mu_blocks = [(i * mu_chunk, min(mu_chunk, n_rmu_L - i * mu_chunk))
                 for i in range(n_mu_blocks_L)]
    nu_blocks = [(j * mu_chunk, min(mu_chunk, n_rmu_R - j * mu_chunk))
                 for j in range(n_mu_blocks_R)]

    # Per-stage timing breakdown (env-gated).  Set
    # ``LORRAX_V_Q_TIME_STAGES=1`` to get a per-batch breakdown of read
    # vs kernel time.  Each iter calls ``block_until_ready`` only once at
    # the end of the kernel block to keep the read-then-async-dispatch
    # pattern intact while still attributing wall to the right stage.
    _time_stages = bool(int(os.environ.get('LORRAX_V_Q_TIME_STAGES', '0') or 0))
    import time as _time
    t_read = 0.0
    t_kernel = 0.0
    t_vphase = 0.0

    with timing.section(timing_label):
        q_cursor = 0
        for q_batch in q_batches:
            actual = len(q_batch)
            qvecs = [_qvec_wrap(*c) for c in q_batch]
            if _time_stages:
                _t0 = _time.perf_counter()
            v_per_G_b, phase_b = _v_phase_batch(qvecs, pad_to=actual)
            if _time_stages:
                jax.block_until_ready(v_per_G_b)
                t_vphase += _time.perf_counter() - _t0
            # ``q_flat0`` = leading-q offset into the on-disk ``zeta_q``.
            # Under IBZ writes the disk axis follows iteration order
            # (= IBZ index); under the default full-BZ path iteration
            # order matches the flat-q convention exactly.  Either way
            # ``q_cursor`` (the running iteration index) is the correct
            # offset.
            q_flat0 = q_cursor

            for mu_i, (mu_lo, mu_size) in enumerate(mu_blocks):
                if _time_stages:
                    _t0 = _time.perf_counter()
                # ``valid_shape`` clips to the LOGICAL on-disk extent.
                # For interior μ-blocks fully within the logical range
                # this is mu_size (no clip).  For the last block when
                # padding ran past the logical extent, the residual is
                # max(0, n_rmu_L_logical - mu_lo); SlabIO reads the
                # logical prefix and zero-fills the padded tail.
                mu_valid = max(0, min(int(mu_size),
                                      n_rmu_L_logical - int(mu_lo)))
                if use_g_flat_zeta:
                    # ζ_L delivered G-flat by the reader (FFT + phase +
                    # sphere happen inside :meth:`ZetaReader.read_zeta_G_slab`).
                    zeta_mu = zeta_L_io.read_zeta_G_slab(
                        q_offset=q_flat0, q_count=actual,
                        mu_offset=mu_lo, mu_count=mu_size,
                        phase_batch=phase_b,
                        sphere_idx=sphere_idx,
                        mesh=mesh_xy, valid_mu=mu_valid)
                else:
                    zeta_mu = zeta_L_io.read_slab(
                        'zeta_q',
                        shape=(actual, n_rtot, mu_size),
                        valid_shape=(actual, n_rtot, mu_valid),
                        dtype=np.complex128,
                        offset=(q_flat0, 0, mu_lo),
                        mesh=mesh_xy, partition_spec=_read_spec)
                if _time_stages:
                    jax.block_until_ready(zeta_mu)
                    t_read += _time.perf_counter() - _t0
                for nu_j, (nu_lo, nu_size) in enumerate(nu_blocks):
                    on_diag = (
                        same_zeta and mu_i == nu_j and
                        mu_lo == nu_lo and mu_size == nu_size
                    )
                    do_write_g0 = same_zeta and wants_g0 and on_diag
                    if on_diag:
                        # Single-read path — ζ_μ aliases ζ_ν.
                        if use_g_flat_zeta:
                            kernel = _make_V_q_tile_kernel_Gflat(
                                n_G_sph=n_G_sph,
                                g0_sphere_idx=0,
                                mesh_xy=mesh_xy,
                                q_chunk=actual, mu_size=mu_size,
                                nu_size=nu_size,
                                n_rmu_L=n_rmu_L, n_rmu_R=n_rmu_R,
                                same_zeta=True, write_g0=do_write_g0)
                            if _time_stages:
                                _t0 = _time.perf_counter()
                            V_acc, g0_work = kernel(
                                V_acc, g0_work,
                                zeta_mu, v_per_G_b,
                                jnp.int32(q_cursor),
                                jnp.int32(mu_lo), jnp.int32(nu_lo))
                        else:
                            kernel = _make_V_q_tile_kernel(
                                sphere_idx=sphere_idx, n_G_sph=n_G_sph,
                                fft_shape=fft_grid, mesh_xy=mesh_xy,
                                q_chunk=actual, mu_size=mu_size,
                                nu_size=nu_size,
                                n_rmu_L=n_rmu_L, n_rmu_R=n_rmu_R,
                                same_zeta=True, write_g0=do_write_g0)
                            if _time_stages:
                                _t0 = _time.perf_counter()
                            V_acc, g0_work = kernel(
                                V_acc, g0_work,
                                zeta_mu, v_per_G_b, phase_b,
                                jnp.int32(q_cursor),
                                jnp.int32(mu_lo), jnp.int32(nu_lo))
                        if _time_stages:
                            jax.block_until_ready(V_acc)
                            t_kernel += _time.perf_counter() - _t0
                    else:
                        # Distinct ζ_R — second read.
                        if _time_stages:
                            _t0 = _time.perf_counter()
                        nu_valid = max(0, min(int(nu_size),
                                              n_rmu_R_logical - int(nu_lo)))
                        if use_g_flat_zeta:
                            zeta_nu = zeta_R_io.read_zeta_G_slab(
                                q_offset=q_flat0, q_count=actual,
                                mu_offset=nu_lo, mu_count=nu_size,
                                phase_batch=phase_b,
                                sphere_idx=sphere_idx,
                                mesh=mesh_xy, valid_mu=nu_valid)
                        else:
                            zeta_nu = zeta_R_io.read_slab(
                                'zeta_q',
                                shape=(actual, n_rtot, nu_size),
                                valid_shape=(actual, n_rtot, nu_valid),
                                dtype=np.complex128,
                                offset=(q_flat0, 0, nu_lo),
                                mesh=mesh_xy, partition_spec=_read_spec)
                        if _time_stages:
                            jax.block_until_ready(zeta_nu)
                            t_read += _time.perf_counter() - _t0
                        if use_g_flat_zeta:
                            kernel = _make_V_q_tile_kernel_Gflat(
                                n_G_sph=n_G_sph,
                                g0_sphere_idx=0,
                                mesh_xy=mesh_xy,
                                q_chunk=actual, mu_size=mu_size,
                                nu_size=nu_size,
                                n_rmu_L=n_rmu_L, n_rmu_R=n_rmu_R,
                                same_zeta=False, write_g0=False)
                            if _time_stages:
                                _t0 = _time.perf_counter()
                            V_acc, g0_work = kernel(
                                V_acc, g0_work,
                                zeta_mu, zeta_nu, v_per_G_b,
                                jnp.int32(q_cursor),
                                jnp.int32(mu_lo), jnp.int32(nu_lo))
                        else:
                            kernel = _make_V_q_tile_kernel(
                                sphere_idx=sphere_idx, n_G_sph=n_G_sph,
                                fft_shape=fft_grid, mesh_xy=mesh_xy,
                                q_chunk=actual, mu_size=mu_size,
                                nu_size=nu_size,
                                n_rmu_L=n_rmu_L, n_rmu_R=n_rmu_R,
                                same_zeta=False, write_g0=False)
                            if _time_stages:
                                _t0 = _time.perf_counter()
                            V_acc, g0_work = kernel(
                                V_acc, g0_work,
                                zeta_mu, zeta_nu, v_per_G_b, phase_b,
                                jnp.int32(q_cursor),
                                jnp.int32(mu_lo), jnp.int32(nu_lo))
                        if _time_stages:
                            jax.block_until_ready(V_acc)
                            t_kernel += _time.perf_counter() - _t0
                        del zeta_nu
                del zeta_mu
            q_cursor += actual
            for _ in range(actual):
                vq_progress.step()
        vq_progress.finish()

    if _time_stages and verbose and jax.process_index() == 0:
        total = t_read + t_kernel + t_vphase
        print(f"  V_q tile stage breakdown (host-blocked, "
              f"under-counts overlap):  read={t_read:.1f}s  "
              f"kernel={t_kernel:.1f}s  v+phase={t_vphase:.1f}s  "
              f"total={total:.1f}s")

    return V_acc, (g0_work if wants_g0 else None)


# ----------------------------------------------------------------------------
# IBZ → full-BZ unfold (post-loop)
# ----------------------------------------------------------------------------
#
# V_q is a pure centroid-axis double-permute under {S | τ} — the τ-phases
# in the ζ transformation rule cancel because V is bilinear in ζ:
#
#     V_{Sq, π_s(μ), π_s(ν)} = V_{q, μ, ν}                         (eq. 3)
#
# So given V_q_ibz on the IBZ wedge and the full-to-IBZ tables produced by
# ``SymMaps.find_irreducible_qpoints`` + ``orbit_syms.compute_centroid_sym_perm``,
# we can recover V_q on the full BZ as a pure index gather — no FFT, no
# phase multiply.  Cost: one fancy index, scales as n_q_full · μ² ·
# 16 / mesh_size on the (μ, ν) sharding.

def _unfold_v_q_ibz_to_full(
    V_q_ibz: jax.Array,
    *,
    full_to_irr_idx: np.ndarray,
    full_to_irr_sym: np.ndarray,
    sym_perm: np.ndarray,
    mesh_xy: Mesh,
) -> jax.Array:
    """Expand ``V_q_ibz (n_qpt_irr, μ, μ)`` to ``(n_q_full, μ, μ)``.

    The mapping is eq. 3 of ``reports/zeta_ibz_2026-05-11/report.md``:

        V_full[q, μ', ν'] = V_ibz[i(q), π_{s(q)}^{-1}(μ'), π_{s(q)}^{-1}(ν')]

    where ``i(q) = full_to_irr_idx[q]`` and ``s(q) = full_to_irr_sym[q]``.
    No τ-phase — bilinearity in ζ cancels them.

    Parameters
    ----------
    V_q_ibz
        (n_qpt_irr, n_rmu, n_rmu) c128, sharded ``P(None, 'x', 'y')``.
    full_to_irr_idx
        (n_q_full,) int.
    full_to_irr_sym
        (n_q_full,) int.
    sym_perm
        (n_sym, n_rmu) int — the forward permutation
        ``r_{π_s(μ)} ≡ S_s r_μ + τ_s``.
    mesh_xy
        Device mesh; the output is constrained to ``P(None, 'x', 'y')``.

    Returns
    -------
    V_q_full
        (n_q_full, n_rmu, n_rmu) c128, sharded ``P(None, 'x', 'y')``.
    """
    # Inverse permutation: ``inv_perm[s, π_s(μ)] = μ`` → argsort along μ.
    inv_perm = np.argsort(sym_perm, axis=-1).astype(np.int32)
    inv_perm_j = jnp.asarray(inv_perm)                              # (n_sym, n_rmu)
    idx_j = jnp.asarray(np.asarray(full_to_irr_idx, dtype=np.int32))
    sym_j = jnp.asarray(np.asarray(full_to_irr_sym, dtype=np.int32))

    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))

    @partial(jax.jit, out_shardings=V_sh)
    def _do_unfold(V_ibz):
        # Per-q μ-axis mapping: shape (n_q_full, n_rmu).
        perm_q = inv_perm_j[sym_j]                                  # (n_q_full, n_rmu)
        V_at_irr = V_ibz[idx_j]                                     # (n_q_full, μ, μ)
        # Gather along μ (axis 1):
        V_perm_mu = jnp.take_along_axis(
            V_at_irr, perm_q[:, :, None], axis=1)
        # Gather along ν (axis 2):
        V_full = jnp.take_along_axis(
            V_perm_mu, perm_q[:, None, :], axis=2)
        return V_full

    return _do_unfold(V_q_ibz)


def _unfold_g0_ibz_to_full(
    g0_ibz: jax.Array,
    *,
    full_to_irr_idx: np.ndarray,
    full_to_irr_sym: np.ndarray,
    sym_perm: np.ndarray,
    mesh_xy: Mesh,
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
    """
    inv_perm = np.argsort(sym_perm, axis=-1).astype(np.int32)
    inv_perm_j = jnp.asarray(inv_perm)
    idx_j = jnp.asarray(np.asarray(full_to_irr_idx, dtype=np.int32))
    sym_j = jnp.asarray(np.asarray(full_to_irr_sym, dtype=np.int32))

    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))

    @partial(jax.jit, out_shardings=g0_sh)
    def _do_unfold(g0):
        perm_q = inv_perm_j[sym_j]                                  # (n_q_full, n_rmu)
        g0_at_irr = g0[idx_j]                                       # (n_q_full, μ)
        g0_full = jnp.take_along_axis(g0_at_irr, perm_q, axis=1)
        return g0_full

    return _do_unfold(g0_ibz)
