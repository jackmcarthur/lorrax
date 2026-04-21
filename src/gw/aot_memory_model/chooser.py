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


def describe_chunks(choice: ChunkChoice) -> str:
    """Format a one-line summary, for the existing fit_zeta log line."""
    util = 100.0 * choice.peak_bytes / max(1.0, choice.budget_bytes)
    return (f"AOT chooser: chunk_r={choice.chunk_r} "
            f"band_chunk={choice.band_chunk} "
            f"({choice.num_r_chunks}×{choice.num_bc_chunks} jits, "
            f"peak={choice.peak_bytes/1e9:.2f} GB / "
            f"{choice.budget_bytes/1e9:.2f} GB = {util:.0f}%, "
            f"total={choice.total_flops/1e9:.1f} GF)")
