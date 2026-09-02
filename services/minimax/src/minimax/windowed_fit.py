"""Phase-bounded reciprocal fitting on an explicit candidate-time family.

This is an offline scalar fitter for crossing windows.  Candidate times are
provided by the caller (for example, a union of a shipped pane family and the
generic measured-fit dictionary).  A pivoted basis chooses a nested candidate
order and the incumbent nonnegative Lawson solve fits one fixed weight phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr

from minimax.frequency_fit import _nonnegative_lawson
from minimax.reciprocal_fit import (
    ReciprocalMeasureProblem,
    delivered_error,
    evaluate_rule,
    rule_amplification,
)


Array = np.ndarray


@dataclass(frozen=True)
class PhaseBoundedReciprocalFit:
    """Fixed-phase, nonnegative-coefficient reciprocal rule."""

    time_nodes: Array
    weights: Array
    coefficients: Array
    sampled_relative_residual: float
    delivered_error: float
    amplification_p99: float
    amplification_max: float
    candidate_indices: Array
    target_met: bool

    @property
    def node_count(self) -> int:
        return int(self.time_nodes.size)


def _deduplicate_candidates(candidates: Array) -> Array:
    order = np.argsort(np.asarray(candidates).real, kind="stable")
    values = np.asarray(candidates, dtype=np.complex128)[order]
    kept = []
    for value in values:
        if not kept or all(abs(value - previous)
                           > 1.0e-10 * max(abs(value), abs(previous), 1.0)
                           for previous in kept):
            kept.append(value)
    return np.asarray(kept, dtype=np.complex128)


def _training_denominators(problem: ReciprocalMeasureProblem,
                           frequency_count: int,
                           cell_count: int) -> Array:
    frequency_index = np.unique(np.linspace(
        0, problem.frequencies.size - 1,
        min(int(frequency_count), problem.frequencies.size)).astype(int))
    cumulative = np.cumsum(problem.cell_masses)
    mass_targets = np.linspace(0.0, cumulative[-1],
                               min(int(cell_count), cumulative.size))
    # Stable mass-quantile samples retain both support edges and do not re-bin
    # or alter the caller's measure.
    cell_index = np.unique(np.searchsorted(cumulative, mass_targets,
                                           side="left"))
    return problem.denominators[np.ix_(frequency_index, cell_index)].ravel()


def fit_phase_bounded_candidates(
    problem: ReciprocalMeasureProblem,
    candidate_times: Array,
    *,
    target_error: float,
    phase: complex = -1.0j,
    max_rank: int = 128,
    training_frequency_count: int = 16,
    training_cell_count: int = 320,
    lawson_iterations: int = 6,
) -> PhaseBoundedReciprocalFit:
    r"""Fit ``1/d`` with ``weights = phase * nonnegative_coefficients``.

    Candidate selection is nested and deterministic.  Pivoted QR is performed
    on the relative-residual atoms

    ``d * phase * exp(1j * candidate_time * d)``

    over a bounded training subset.  Each trial rank is refitted by the same
    nonnegative Lawson routine as :func:`fit_damped_reciprocal` and judged on
    every denominator in ``problem``.  The returned residual and amplification
    are actual measurements even when ``target_met`` is false.
    """
    target = float(target_error)
    if not 0.0 < target < 1.0:
        raise ValueError("target_error must lie in (0,1)")
    unit_phase = complex(phase)
    if not np.isfinite(unit_phase) or not abs(unit_phase) > 0.0:
        raise ValueError("phase must be finite and nonzero")
    unit_phase /= abs(unit_phase)
    candidates = np.asarray(candidate_times, dtype=np.complex128)
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("candidate_times must be a nonempty 1d array")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("candidate_times must be finite")
    candidates = _deduplicate_candidates(candidates)
    rank_limit = min(int(max_rank), candidates.size)
    if rank_limit < 2:
        raise ValueError("at least two candidate times are required")

    train_d = _training_denominators(
        problem, int(training_frequency_count), int(training_cell_count))
    train_atoms = (train_d[:, None] * unit_phase
                   * np.exp(1.0j * train_d[:, None] * candidates[None, :]))
    pivot = qr(train_atoms, mode="economic", pivoting=True)[2]
    full_d = problem.denominators

    coarse = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, rank_limit]
    trial_ranks = sorted(set(rank for rank in coarse if rank <= rank_limit))
    best = None
    first_met_position = None
    for position, rank in enumerate(trial_ranks):
        indices = np.sort(pivot[:rank])
        coefficient = _nonnegative_lawson(
            train_atoms[:, indices], int(lawson_iterations))
        live = coefficient > 0.0
        indices, coefficient = indices[live], coefficient[live]
        times = candidates[indices]
        weights = unit_phase * coefficient
        residual = np.abs(1.0 - full_d * evaluate_rule(times, weights, full_d))
        error = float(np.max(residual))
        row = (error, indices, coefficient, times, weights)
        if best is None or error < best[0]:
            best = row
        if error <= target:
            best = row
            first_met_position = position
            break

    if first_met_position is not None:
        upper = trial_ranks[first_met_position]
        lower = (1 if first_met_position == 0
                 else trial_ranks[first_met_position - 1])
        for rank in range(lower + 1, upper):
            indices = np.sort(pivot[:rank])
            coefficient = _nonnegative_lawson(
                train_atoms[:, indices], int(lawson_iterations))
            live = coefficient > 0.0
            indices, coefficient = indices[live], coefficient[live]
            times = candidates[indices]
            weights = unit_phase * coefficient
            residual = np.abs(
                1.0 - full_d * evaluate_rule(times, weights, full_d))
            error = float(np.max(residual))
            if error <= target:
                best = (error, indices, coefficient, times, weights)
                break

    error, indices, coefficient, times, weights = best
    delivered, _ = delivered_error(problem, times, weights)
    p99, peak = rule_amplification(times, weights, problem)
    return PhaseBoundedReciprocalFit(
        time_nodes=np.asarray(times, dtype=np.complex128),
        weights=np.asarray(weights, dtype=np.complex128),
        coefficients=np.asarray(coefficient, dtype=np.float64),
        sampled_relative_residual=float(error),
        delivered_error=float(np.max(delivered)),
        amplification_p99=float(p99),
        amplification_max=float(peak),
        candidate_indices=np.asarray(indices, dtype=np.int64),
        target_met=bool(error <= target * (1.0 + 1.0e-12)),
    )


__all__ = ["PhaseBoundedReciprocalFit", "fit_phase_bounded_candidates"]
