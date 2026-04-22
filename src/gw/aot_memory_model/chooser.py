"""Joint (chunk_r, band_chunk) chooser for the ISDF fit.

Gives ``gw_init.compute_optimal_chunks`` an AOT-based alternative to
its hand-derived per-stage byte formulas.  Minimises total FLOPs over
the full r-chunk loop subject to a per-device memory budget:

    total_flops(chunk_r, bc) =
        ceil(n_rtot / chunk_r) · predict_flops_per_call(chunk_r, bc)
    peak_bytes(chunk_r, bc)  =
        predict_peak(chunk_r, bc)

The memory fit sets which (chunk_r, bc) values are feasible; the cost
fit breaks ties by picking the cheapest among those.  At the MoS2 3×3
scale the chooser prefers chunk_r = n_rtot (one big r-chunk) because
the per-call ZCT/solve overhead reruns per r-chunk — same insight the
user called out in pre-compaction messages.

The (chunk_r, bc, k_chunk) space is enumerated on a coarse grid and
evaluated by closed-form prediction — no new AOT compilation at
runtime, no jax imports needed.  k_chunk is held at ``n_k`` (full k
per call) until ``band_chunk == 1``, matching the user's stated
priority ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .core import (
    SysDims, Knobs, MeshSpec,
    get_kernel, load_fit, predict_peak,
)
from .cost import load_cost_fit, predict_flops_per_call


@dataclass
class ChunkChoice:
    """Best feasible ``(chunk_r, band_chunk)`` under the budget."""
    chunk_r: int
    band_chunk: int
    k_chunk: int
    num_r_chunks: int
    num_bc_chunks: int
    peak_bytes: float
    per_call_flops: float
    total_flops: float
    budget_bytes: float
    # Non-fatal diagnostics
    note: str = ""


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _p_divisible_candidates(max_val: int, p: int, min_val: int = 1) -> list[int]:
    """Return a coarse grid of p-divisible candidates in ``[min_val, max_val]``.

    We keep it sparse — powers of 2 up to max plus the endpoint —
    because the fit is so linear that dense sampling adds no value.
    """
    pts = set()
    v = max(p, min_val)
    while v <= max_val:
        # Round DOWN to nearest p-multiple (except 0).
        v_p = (v // p) * p
        if v_p >= p:
            pts.add(v_p)
        v *= 2
    pts.add(max_val - (max_val % p) if max_val >= p else max_val)
    return sorted(x for x in pts if x >= min_val)


def _enumerate_candidates(
    sys: SysDims, mesh: MeshSpec,
    *,
    chunk_r_max: int | None = None,
    chunk_r_min: int = 1024,
    band_chunk_values: Sequence[int] = (8, 16, 32, 64, 128),
) -> Iterable[tuple[int, int]]:
    """Yield ``(chunk_r, band_chunk)`` candidates on a coarse grid.

    ``chunk_r`` candidates are p-divisible (p = p_x · p_y) since the
    inner shard_map requires it.  The band_chunk grid is an explicit
    small list because the round-up-to-pad behaviour ties it to a few
    discrete sweet spots; the chooser picks from the subset that
    divides n_b (to avoid one oversized remainder chunk).
    """
    p = mesh.p_x * mesh.p_y
    n_rtot = sys.n_r or (
        sys.fft_shape[0] * sys.fft_shape[1] * sys.fft_shape[2])
    chunk_r_max = chunk_r_max if chunk_r_max is not None else n_rtot

    # Always include the full band count so a feasible 1-bc-chunk fit
    # (current heuristic's preferred mode) is on the candidate list.
    bc_grid = sorted(set(list(band_chunk_values) + [sys.n_b]))

    for cr in _p_divisible_candidates(chunk_r_max, p, chunk_r_min):
        for bc in bc_grid:
            if bc > sys.n_b:
                continue
            if bc % p != 0 and bc < p:
                continue  # band-sharding requires bc ≥ p in most kernels
            yield (cr, bc)


def choose_chunks_aot(
    sys: SysDims, mesh: MeshSpec,
    *,
    budget_bytes: float,
    kernel_name: str = "fit_one_rchunk",
    tag: str = "current",
    chunk_r_max: int | None = None,
    band_chunk_values: Sequence[int] = (8, 16, 32, 64, 128),
) -> ChunkChoice:
    """Pick ``(chunk_r, band_chunk)`` minimizing total FLOPs subject to
    ``predict_peak(...) ≤ budget_bytes``.

    Raises ``ValueError`` if no feasible combination exists within the
    candidate grid; the caller should react by lowering ``chunk_r_min``
    or enlarging the mesh.
    """
    kernel = get_kernel(kernel_name)
    mem_fit = load_fit(kernel_name, tag=tag)
    cost_fit = load_cost_fit(kernel_name, tag=tag)

    n_rtot = sys.n_r or (
        sys.fft_shape[0] * sys.fft_shape[1] * sys.fft_shape[2])

    best: ChunkChoice | None = None
    for (cr, bc) in _enumerate_candidates(
            sys, mesh, chunk_r_max=chunk_r_max,
            band_chunk_values=band_chunk_values):
        knobs = Knobs.of(chunk_r=cr, band_chunk=bc)
        peak = predict_peak(mem_fit, kernel, sys, knobs, mesh)
        if peak > budget_bytes:
            continue

        per_call = predict_flops_per_call(cost_fit, kernel, sys, knobs, mesh)
        num_r = _ceil_div(n_rtot, cr)
        num_bc = _ceil_div(sys.n_b, bc)
        total = num_r * per_call

        cand = ChunkChoice(
            chunk_r=cr, band_chunk=bc,
            k_chunk=sys.n_k,  # always full-k until bc hits 1
            num_r_chunks=num_r, num_bc_chunks=num_bc,
            peak_bytes=peak, per_call_flops=per_call,
            total_flops=total, budget_bytes=budget_bytes,
        )
        # Tiebreak (total_flops equal): prefer bigger (chunk_r, band_chunk)
        # — fewer driver-level Python iterations on the r-axis, fewer
        # pair-density einsums on the bc-axis (small matmuls underuse
        # the GPU).  Ties happen when the fit's bc-scaling primitives
        # land on zero β, which is common in lean DoE sweeps.
        if best is None:
            best = cand
            continue
        _cand_key = (-cand.total_flops, cand.chunk_r, cand.band_chunk)
        _best_key = (-best.total_flops, best.chunk_r, best.band_chunk)
        if _cand_key > _best_key:
            best = cand

    if best is None:
        raise ValueError(
            f"No feasible (chunk_r, band_chunk) under budget "
            f"{budget_bytes/1e9:.2f} GB — try a bigger mesh, lower "
            "chunk_r_min, or off-device gspace (use_phdf5_gspace=True).")
    return best


# ===========================================================================
# Analytic chooser — closed-form inversion of the bilinear peak bound
# ===========================================================================
#
# For fit_one_rchunk the peak-bytes model, after grouping primitives by
# their scaling class (PRIMITIVE_CLASSES on the kernel), collapses to
#
#     peak(chunk_r, bc) = α₀ + α_cr · chunk_r + α_bc · bc
#                              + α_crbc · (chunk_r · bc)
#
# At a fixed bc this is LINEAR in chunk_r — so
#
#     chunk_r_max(bc) = (M − α₀ − α_bc · bc) / (α_cr + α_crbc · bc)
#
# …solvable in closed form.  The chooser then 1-D searches over a
# handful of discrete bc candidates (mesh-divisible, bounded by n_b),
# computing total cost at chunk_r_max(bc).  No 2-D grid enumeration.
#
# Total FLOPs:
#     cost(chunk_r, bc) =
#        ceil(n_rtot / chunk_r) · [F_fixed + F_wfnFFT(bc)]
#
# F_wfnFFT(bc) = n_b · C_per_band + (n_b / bc) · C_launch.  The
# C_launch term is a calibration hook: default 0 (pure FLOPs, faithful
# to cost_analysis), can be set post-hoc from runtime measurements to
# capture the "small-chunk performance hit" in kernel dispatch.


@dataclass
class AlphaFit:
    """Regrouped memory-fit coefficients for analytic inversion."""
    alpha0: float
    alpha_cr: float
    alpha_bc: float
    alpha_crbc: float


def _group_alpha(
    kernel, mem_fit, sys: SysDims, mesh: MeshSpec,
) -> AlphaFit:
    """Regroup the 7 primitive β·T contributions into the 4 scaling
    classes.  Each primitive T_i is evaluated at chunk_r=1, bc=1 so
    the returned coefficient already carries (sys, mesh) scaling.

    Requires the kernel class to declare ``PRIMITIVE_CLASSES``.
    """
    classes = getattr(kernel, "PRIMITIVE_CLASSES", None)
    if not classes:
        raise ValueError(
            f"Kernel {kernel.name!r} has no PRIMITIVE_CLASSES — "
            "analytic chooser can't regroup its fit.")
    unit_knobs = Knobs.of(chunk_r=1, band_chunk=1)
    alpha = {"const": float(mem_fit.intercept), "cr": 0.0, "bc": 0.0, "crbc": 0.0}
    for name, coef in zip(mem_fit.feature_names, mem_fit.coefs):
        cls = classes.get(name)
        if cls is None:
            raise ValueError(
                f"PRIMITIVE_CLASSES missing entry for {name!r}")
        T_unit = kernel.PRIMITIVES[name](sys, unit_knobs, mesh)
        alpha[cls] += coef * T_unit
    # γ correction — memory_analysis is an upper bound; XLA schedules
    # tighter at runtime.  γ captures runtime_peak / aot_predicted from
    # calibration.  Applied uniformly to α₀ / α_cr / α_bc / α_crbc so
    # feasibility bounds match real GPU usage, not the worst case.
    gamma = float(getattr(mem_fit, "gamma", 1.0)) or 1.0
    return AlphaFit(
        alpha0=alpha["const"] * gamma,
        alpha_cr=alpha["cr"] * gamma,
        alpha_bc=alpha["bc"] * gamma,
        alpha_crbc=alpha["crbc"] * gamma,
    )


def _p_divisible_floor(x: float, p: int) -> int:
    """Round DOWN to the nearest non-zero multiple of p."""
    v = int(x) // p * p
    return max(0, v)


def _largest_even_divisor_leq(n: int, cap: int, p: int) -> int:
    """Largest divisor of ``n`` that is ≤ cap AND a multiple of ``p``.

    Used to pick chunk sizes that:
      * make every chunk SAME size (no remainder → one compile shape)
      * are mesh-divisible (required for shard_map)

    If no divisor of n fits, falls back to the largest p-multiple ≤ cap
    (which means the last chunk is smaller — 2 compile shapes).  The
    caller sees this via ``chunk % n != 0``.
    """
    cap = max(p, int(cap))
    # Enumerate divisors of n in descending order; pick first ≤ cap
    # that's also p-divisible.  n typically ≤ few×10⁴ so this is cheap.
    best_div = 0
    d = 1
    while d * d <= n:
        if n % d == 0:
            for cand in (d, n // d):
                if cand <= cap and cand % p == 0 and cand > best_div:
                    best_div = cand
        d += 1
    if best_div > 0:
        return best_div
    # No clean divisor — fall back to plain p-multiple (2 compile shapes)
    return (cap // p) * p


def choose_chunks_analytic(
    sys: SysDims, mesh: MeshSpec,
    *,
    budget_bytes: float,
    kernel_name: str = "fit_one_rchunk",
    tag: str = "current",
    band_chunk_values: Sequence[int] | None = None,
    fft_launch_overhead_flops: float = 0.0,
    fft_per_band_flops: float | None = None,
    q_gather_min: int = 1,
) -> ChunkChoice:
    """Closed-form chunk chooser.

    For every ``bc`` candidate we evaluate
      1. ``chunk_r_max(bc) = (M − α₀ − α_bc·bc) / (α_cr + α_crbc·bc)``
      2. round down to p-divisible and clamp to ``n_rtot``
      3. compute ``total_cost`` at that ``chunk_r_max`` and keep the
         candidate with minimal cost.

    Returns the same :class:`ChunkChoice` dataclass as the grid-search
    chooser so callers can swap without changing downstream code.

    Parameters
    ----------
    fft_launch_overhead_flops
        Per-call overhead for the wavefunction FFT, captured as
        additional FLOPs per-call.  When > 0 the cost term becomes
        ``total_launch = (n_b / bc) · C_launch`` per r-chunk — smaller
        ``bc`` → more launches → more overhead.  Default 0 (FLOPs
        model is pure compute, honest to ``cost_analysis``).  Populate
        from runtime measurements via ``profile_chunks_vs_model``.
    fft_per_band_flops
        Per-band wavefunction-FFT FLOPs.  When ``None``, inferred from
        the cost fit via the ``bc_fft`` primitive.  Override only for
        calibration experiments.
    q_gather_min
        Minimum q-chunk gather size the driver can support.  The
        allgather in ``fit_zeta_chunked_to_h5`` replicates
        ``q_gather × n_rmu × chunk_r × 16`` bytes per device after
        each r-chunk — this bounds chunk_r from above.  Default 1
        means "driver can always pick q_gather=1 if it has to" — if
        the caller knows a higher floor (e.g. 2 for performance),
        passing it here gives a tighter cr bound.
    """
    kernel = get_kernel(kernel_name)
    mem_fit = load_fit(kernel_name, tag=tag)
    cost_fit = load_cost_fit(kernel_name, tag=tag)

    alpha = _group_alpha(kernel, mem_fit, sys, mesh)
    n_rtot = sys.n_r or (
        sys.fft_shape[0] * sys.fft_shape[1] * sys.fft_shape[2])
    p = mesh.p_x * mesh.p_y

    # bc candidates: p-divisible in [p, n_b], with n_b itself included.
    if band_chunk_values is None:
        pow2 = []
        v = max(p, 8)
        while v <= sys.n_b:
            if v % p == 0 and v <= sys.n_b:
                pow2.append(v)
            v *= 2
        pow2.append(sys.n_b)
        band_chunk_values = sorted(set(pow2))

    best: ChunkChoice | None = None
    # Post-jit allgather bound: fit_zeta_chunked_to_h5 replicates
    # ``q_gather × n_rmu × chunk_r × 16`` bytes across all devices
    # after each r-chunk.  The driver itself chooses q_gather
    # dynamically to fit, but chunk_r still bounds it: for the
    # minimum-feasible q_gather_min, the replicated slab must fit.
    _allgather_per_cr = 16.0 * q_gather_min * sys.n_rmu  # bytes per cr-unit
    # Leave a 20% slack to account for concurrent live arrays during
    # the gather (zeta input + NCCL temp + output both live).
    _allgather_cr_cap = int(0.8 * budget_bytes / max(1.0, _allgather_per_cr))

    for bc in band_chunk_values:
        if bc < p or bc > sys.n_b:
            continue

        # Closed-form: peak(cr, bc) = α₀ + α_cr·cr + α_bc·bc + α_crbc·cr·bc ≤ M
        # → cr ≤ (M − α₀ − α_bc·bc) / (α_cr + α_crbc·bc)
        denom = alpha.alpha_cr + alpha.alpha_crbc * bc
        numer = budget_bytes - alpha.alpha0 - alpha.alpha_bc * bc
        if denom <= 0 or numer <= 0:
            continue
        cr_budget = numer / denom
        cr = min(int(cr_budget), n_rtot, _allgather_cr_cap)
        cr = _p_divisible_floor(cr, p)
        if cr <= 0:
            continue

        num_r = _ceil_div(n_rtot, cr)
        per_call = predict_flops_per_call(cost_fit, kernel, sys,
                                          Knobs.of(chunk_r=cr, band_chunk=bc),
                                          mesh)
        # Optional: small-bc launch penalty.  Added only to TOTAL cost,
        # not per-call FLOPs, so the AOT fit stays faithful to
        # cost_analysis.
        launch_penalty = (sys.n_b / max(1, bc)) * fft_launch_overhead_flops
        total = num_r * (per_call + launch_penalty)

        # Peak at the chosen (cr, bc) — for validation/logging we
        # re-predict exactly rather than trusting the inversion
        # (handles p-divisible rounding).
        peak = (alpha.alpha0 + alpha.alpha_cr * cr
                + alpha.alpha_bc * bc + alpha.alpha_crbc * cr * bc)

        cand = ChunkChoice(
            chunk_r=cr, band_chunk=bc, k_chunk=sys.n_k,
            num_r_chunks=num_r, num_bc_chunks=_ceil_div(sys.n_b, bc),
            peak_bytes=peak, per_call_flops=per_call,
            total_flops=total, budget_bytes=budget_bytes,
            note=f"analytic (alpha0={alpha.alpha0/1e9:.2f}GB, "
                 f"cr={alpha.alpha_cr/1e6:.1f}MB/cr, "
                 f"bc={alpha.alpha_bc/1e6:.1f}MB/bc, "
                 f"crbc={alpha.alpha_crbc/1e3:.2f}kB/(cr·bc))",
        )
        if best is None:
            best = cand
            continue
        # Tiebreak identically to the grid chooser.
        _cand_key = (-cand.total_flops, cand.chunk_r, cand.band_chunk)
        _best_key = (-best.total_flops, best.chunk_r, best.band_chunk)
        if _cand_key > _best_key:
            best = cand

    if best is None:
        raise ValueError(
            f"No feasible (chunk_r, band_chunk) under budget "
            f"{budget_bytes/1e9:.2f} GB for system {sys}; "
            f"α₀={alpha.alpha0/1e9:.2f} GB (already above budget?).")
    return best


def describe_chunks(choice: ChunkChoice) -> str:
    """Format a one-line summary, for the existing fit_zeta log line."""
    util = 100.0 * choice.peak_bytes / max(1.0, choice.budget_bytes)
    return (f"AOT chooser: chunk_r={choice.chunk_r} "
            f"band_chunk={choice.band_chunk} "
            f"({choice.num_r_chunks}×{choice.num_bc_chunks} jits, "
            f"peak={choice.peak_bytes/1e9:.2f} GB / "
            f"{choice.budget_bytes/1e9:.2f} GB = {util:.0f}%, "
            f"total={choice.total_flops/1e9:.1f} GF)")


# ===========================================================================
# 20 / 80 heuristic chooser — simpler and DoE-independent
# ===========================================================================
#
# Motivation: the analytic chooser's cost model nominally wants to
# minimise FLOPs, but total FLOPs for the ISDF fit is essentially
# chunk-invariant (same work, just partitioned differently) — per-call
# overhead dominates.  So "optimise FLOPs" degenerates to "make bc=1
# with the biggest possible chunk_r", which is bad for GPU
# occupancy and creates too-small pair-density einsums.
#
# The practical rule — reserve a fraction of the budget for the
# wavefunction workspace and let the rest cover the rchunk work:
#
#     wfn_budget  = frac · budget          (default frac = 0.2)
#     rchunk_budget = (1 - frac) · budget
#
# Within wfn_budget, size (band_chunk × k_chunk) so the FFT workspace
# fits.  Production's load_psi_rchunk_fft β[psi_G]=3 tells us each
# wavefunction needs ~3× its own size in transient workspace (input +
# output + cuFFT scratch).  So:
#
#     per_wfn_bytes = 3 · 16 · n_s · n_r / (p_x · p_y)
#     n_wfn_max    = wfn_budget / per_wfn_bytes
#
# Split n_wfn_max into (k_chunk × band_chunk) the way the user asked:
# if n_wfn_max >= n_k, use all k per call (k_chunk = n_k) and set
# band_chunk = n_wfn_max / n_k.  Otherwise, band_chunk = 1 and
# k_chunk = n_wfn_max.
#
# Within rchunk_budget, the dominant cost is 4 pair-sized temps from
# the ZCT stage (β[pair]=4 from zct_lr, confirmed in fit_one_rchunk).
# Solve:
#
#     4 · 16 · n_k · n_rmu · chunk_r / (p_x · p_y)  +  base_persistent
#                                                ≤ rchunk_budget
#     → chunk_r ≤ (rchunk_budget − base) / (64 · n_k · n_rmu / P)
#
# where base_persistent is the always-alive centroid + L_q bytes
# (tiny at MoS2/Si scales; we compute it exactly).


def choose_chunks_heuristic(
    sys: SysDims,
    mesh: MeshSpec,
    *,
    budget_bytes: float,
    wfn_workspace_frac: float = 0.20,
    pair_temp_count: int = 4,
    fft_workspace_multiplier: int = 3,
    q_gather_min: int = 1,
) -> ChunkChoice:
    """Budget-split chunk chooser.

    Reserves ``wfn_workspace_frac`` of the budget for the wavefunction
    FFT workspace (driving band_chunk × k_chunk), and the rest for the
    r-chunk's pair-density temps and intermediates (driving chunk_r).

    No DoE, no AOT fit lookups — just closed-form algebra on
    physically-motivated coefficients from the small-kernel fits:

        β[psi_G]=3      (load_psi_rchunk_fft — per-wfn FFT workspace)
        β[PrBr]=4       (zct_lr — concurrent pair-sized ZCT temps)

    Much more robust than the regressed chooser across untested
    (system, mesh) combinations — has one free parameter
    (``wfn_workspace_frac``) instead of a 7-primitive fit with known
    DoE degeneracies.
    """
    p_x, p_y = int(mesh.p_x), int(mesh.p_y)
    P = p_x * p_y
    n_k = int(sys.n_k)
    mu = int(sys.n_rmu)
    n_s = int(sys.n_s)
    n_b = int(sys.n_b)
    fft_shape = sys.fft_shape
    n_r = sys.n_r or (fft_shape[0] * fft_shape[1] * fft_shape[2])

    # --- 1. Split budget ---
    wfn_budget = wfn_workspace_frac * budget_bytes
    rchunk_budget = budget_bytes - wfn_budget

    # --- 2. Pick (band_chunk, k_chunk) for the wfn workspace ---
    # Bytes for one (k, band) wavefunction shard + its FFT workspace.
    per_wfn_bytes = fft_workspace_multiplier * 16.0 * n_s * n_r / P
    n_wfn_max = int(wfn_budget / max(1.0, per_wfn_bytes))

    if n_wfn_max >= n_k:
        # Fits all k at once — keep k_chunk = n_k and size band_chunk
        # within the remaining budget.  Prefer band_chunk that
        # DIVIDES n_b so every band chunk has the same size (one
        # compile shape inside the r-chunk loop).
        k_chunk = n_k
        bc_cap = max(1, n_wfn_max // n_k)
        band_chunk = _largest_even_divisor_leq(n_b, bc_cap, P)
        if band_chunk <= 0:
            # No divisor of n_b fits; fall back to a p-multiple
            # (accepts 2 compile shapes for the remainder bc-chunk).
            band_chunk = max(P, (bc_cap // P) * P)
    else:
        # Even one band per call doesn't fit all k; shrink k_chunk.
        band_chunk = 1
        k_chunk = max(1, n_wfn_max)

    # --- 3. Pick chunk_r from the rchunk_budget ---
    # Persistent (always alive): 2 centroids + psi_G cache + L_q.
    nb_sum = sys.nb_sum
    base_centroid = 16.0 * n_k * mu * nb_sum * n_s / p_x
    base_psiG     = 16.0 * n_k * n_b * n_s * n_r / P
    base_Lq       = 16.0 * n_k * mu * mu / P      # sharded L_q
    base_persistent = base_centroid + base_psiG + base_Lq

    # Dominant chunk_r-linear term: pair_temp_count × 16·n_k·μ·cr/P
    cr_per_byte = pair_temp_count * 16.0 * n_k * mu / P

    headroom = rchunk_budget - base_persistent
    if headroom <= 0:
        # Persistent alone exceeds 80% budget — fallback: use whole
        # budget and accept a tight fit.
        headroom = budget_bytes - base_persistent
        if headroom <= 0:
            raise ValueError(
                f"Persistent state ({base_persistent/1e9:.2f} GB) "
                f"exceeds full budget ({budget_bytes/1e9:.2f} GB)")

    # Post-jit allgather bound (from Phase C earlier work):
    allgather_cap = int(0.8 * budget_bytes / max(1.0, 16.0 * q_gather_min * mu))

    n_rtot = sys.n_r or (fft_shape[0] * fft_shape[1] * fft_shape[2])
    cr_cap = int(min(n_rtot, headroom / cr_per_byte, allgather_cap))
    # Prefer chunk_r that DIVIDES n_rtot evenly — one compile shape
    # for every r-chunk (no remainder recompile).  Falls back to a
    # p-multiple if no divisor fits (accepts 2 compile shapes).
    chunk_r = _largest_even_divisor_leq(n_rtot, cr_cap, P)
    if chunk_r <= 0:
        raise ValueError(
            f"No feasible chunk_r under budget {budget_bytes/1e9:.2f} GB; "
            f"persistent {base_persistent/1e9:.2f} GB, "
            f"allgather cap {allgather_cap}")

    # --- 4. Compute predicted peak for logging ---
    predicted_peak = base_persistent + cr_per_byte * chunk_r
    # Plus wfn workspace contribution (transient, per call):
    wfn_contribution = band_chunk * k_chunk * per_wfn_bytes
    predicted_peak_total = predicted_peak + wfn_contribution

    def _ceil_div_local(a, b):
        return (a + b - 1) // b

    num_r = _ceil_div_local(n_rtot, chunk_r)
    num_bc = _ceil_div_local(n_b, band_chunk) * _ceil_div_local(n_k, k_chunk)

    return ChunkChoice(
        chunk_r=chunk_r,
        band_chunk=band_chunk,
        k_chunk=k_chunk,
        num_r_chunks=num_r,
        num_bc_chunks=num_bc,
        peak_bytes=predicted_peak_total,
        per_call_flops=0.0,  # N/A — heuristic doesn't use FLOPs
        total_flops=0.0,
        budget_bytes=budget_bytes,
        note=(f"heuristic {int(100*wfn_workspace_frac)}/{int(100*(1-wfn_workspace_frac))}: "
              f"wfn={wfn_contribution/1e9:.2f}GB (cap {wfn_budget/1e9:.2f}GB), "
              f"rchunk={cr_per_byte*chunk_r/1e9:.2f}GB, "
              f"persistent={base_persistent/1e9:.2f}GB"),
    )
