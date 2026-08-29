"""Measure-driven support windows and true-error budget apportionment.

This module is physics-free.  A caller supplies complex support points,
nonnegative delivered masses, and a real frequency segment.  The support is
partitioned along a logarithmic distance-to-segment coordinate, with a
crossing region isolated from the two sign-definite sides.  Error budgets are
then apportioned by ``delivered_mass * measured_difficulty``.

The returned scale spans are measurements, not inferred certificates.  A
caller that requires a maximum span must check them before fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class MeasureWindow:
    """One nonempty, disjoint portion of a measured complex support."""

    name: str
    kind: str
    member_indices: Array
    delivered_mass: float
    mass_fraction: float
    scale_min: float
    scale_max: float
    scale_span: float


@dataclass(frozen=True)
class WindowErrorBudget:
    """One share of an absolute true-error budget."""

    name: str
    delivered_mass: float
    measured_difficulty: float
    apportionment_weight: float
    absolute_error_budget: float


def _mass_quantile_groups(indices: Array, scale: Array, masses: Array,
                          count: int) -> list[Array]:
    """Split ``indices`` into nonempty, stable equal-mass quantiles."""
    if indices.size == 0:
        return []
    order = indices[np.argsort(np.log(scale[indices]), kind="stable")]
    ordered_mass = masses[order]
    cumulative = np.cumsum(ordered_mass)
    total = float(cumulative[-1])
    if not total > 0.0:
        return []
    midpoint = cumulative - 0.5 * ordered_mass
    labels = np.minimum((midpoint * int(count) / total).astype(int),
                        int(count) - 1)
    return [order[labels == label] for label in range(int(count))
            if np.any(labels == label)]


def _allocate_side_counts(side_indices: list[Array], masses: Array,
                          available: int) -> list[int]:
    """Give each nonempty sign side one bin, then split the heaviest share."""
    counts = [1 for _ in side_indices]
    while sum(counts) < int(available):
        scores = [float(np.sum(masses[index])) / counts[position]
                  for position, index in enumerate(side_indices)]
        counts[int(np.argmax(scores))] += 1
    return counts


def partition_measure_windows(
    internal_sums: Array,
    masses: Array,
    frequencies: Array,
    *,
    window_count: int = 5,
) -> tuple[MeasureWindow, ...]:
    """Partition support into 2--5 measure-quantile windows.

    A point crosses the requested segment when its real part lies inside the
    closed frequency interval.  All crossing points share one dedicated
    window.  Points below and above the segment are never mixed; the remaining
    window count is divided between those nonempty sides in proportion to
    delivered mass, and each side is split into stable equal-mass quantiles of

    ``scale = hypot(real distance to the segment, abs(imag(internal_sum)))``.

    Empty quantiles are dropped.  Every input point belongs to exactly one
    returned window and delivered mass is conserved to roundoff.
    """
    sums = np.asarray(internal_sums, dtype=np.complex128)
    weight = np.asarray(masses, dtype=np.float64)
    omega = np.asarray(frequencies, dtype=np.float64)
    if sums.ndim != 1 or sums.size == 0 or sums.shape != weight.shape:
        raise ValueError("internal_sums and masses must be matching nonempty 1d arrays")
    if omega.ndim != 1 or omega.size == 0:
        raise ValueError("frequencies must be a nonempty 1d array")
    if not (np.all(np.isfinite(sums)) and np.all(np.isfinite(weight))
            and np.all(np.isfinite(omega))):
        raise ValueError("support, masses, and frequencies must be finite")
    if np.any(weight < 0.0) or not np.any(weight > 0.0):
        raise ValueError("masses must be nonnegative with positive total")
    if not 2 <= int(window_count) <= 5:
        raise ValueError("window_count must lie in [2,5]")

    omega_lo = float(np.min(omega))
    omega_hi = float(np.max(omega))
    below = np.nonzero(sums.real < omega_lo)[0]
    crossing = np.nonzero((sums.real >= omega_lo)
                          & (sums.real <= omega_hi))[0]
    above = np.nonzero(sums.real > omega_hi)[0]
    sides = [("below", below), ("above", above)]
    sides = [(name, indices) for name, indices in sides if indices.size]
    region_count = len(sides) + int(crossing.size > 0)
    if region_count == 0:
        raise ValueError("support has no points")
    if int(window_count) < region_count:
        raise ValueError(
            f"window_count={window_count} cannot isolate {region_count} nonempty regions")

    distance = np.where(sums.real < omega_lo, omega_lo - sums.real,
                        np.where(sums.real > omega_hi,
                                 sums.real - omega_hi, 0.0))
    scale = np.hypot(distance, np.abs(sums.imag))
    if np.any(scale <= 0.0):
        raise ValueError("zero scale in support; broaden or exclude singular points")

    available_for_sides = int(window_count) - int(crossing.size > 0)
    if sides:
        side_counts = _allocate_side_counts(
            [indices for _, indices in sides], weight, available_for_sides)
    else:
        side_counts = []

    groups: list[tuple[str, Array]] = []
    for (side_name, indices), count in zip(sides, side_counts):
        quantiles = _mass_quantile_groups(indices, scale, weight, count)
        groups.extend((f"{side_name}_{position}", group)
                      for position, group in enumerate(quantiles))
    if crossing.size:
        groups.append(("crossing", crossing))

    # A support wholly inside the segment has no sign side to subdivide.
    # Split the crossing mass only to satisfy the public 2-window minimum;
    # both windows retain the explicit crossing kind.
    if len(groups) == 1 and int(window_count) >= 2 and sums.size >= 2:
        only_name, only_indices = groups[0]
        split = _mass_quantile_groups(only_indices, scale, weight, 2)
        groups = [(f"{only_name}_{position}", group)
                  for position, group in enumerate(split)]

    total_mass = float(np.sum(weight))
    windows = []
    for name, indices in groups:
        if indices.size == 0:
            continue
        member_scale = scale[indices]
        member_mass = float(np.sum(weight[indices]))
        kind = ("crossing" if name.startswith("crossing")
                else name.split("_", 1)[0])
        windows.append(MeasureWindow(
            name=name,
            kind=kind,
            member_indices=np.asarray(indices, dtype=np.int64),
            delivered_mass=member_mass,
            mass_fraction=member_mass / total_mass,
            scale_min=float(np.min(member_scale)),
            scale_max=float(np.max(member_scale)),
            scale_span=float(np.max(member_scale) / np.min(member_scale)),
        ))

    membership = np.concatenate([window.member_indices for window in windows])
    if (membership.size != sums.size
            or not np.array_equal(np.sort(membership), np.arange(sums.size))):
        raise RuntimeError("window membership is not a disjoint support partition")
    conserved = sum(window.delivered_mass for window in windows)
    if abs(conserved - total_mass) > 1.0e-12 * max(total_mass, 1.0):
        raise RuntimeError("window partition did not conserve delivered mass")
    if not 2 <= len(windows) <= 5:
        raise RuntimeError(f"partition returned {len(windows)} windows, expected 2--5")
    return tuple(windows)


def apportion_true_error(
    windows: tuple[MeasureWindow, ...],
    measured_difficulties: Array,
    total_absolute_error: float,
) -> tuple[WindowErrorBudget, ...]:
    """Distribute a true absolute-error budget by mass times difficulty."""
    difficulty = np.asarray(measured_difficulties, dtype=np.float64)
    if difficulty.shape != (len(windows),):
        raise ValueError("measured_difficulties must have one entry per window")
    if not np.all(np.isfinite(difficulty)) or np.any(difficulty <= 0.0):
        raise ValueError("measured_difficulties must be finite and positive")
    if not np.isfinite(total_absolute_error) or not total_absolute_error > 0.0:
        raise ValueError("total_absolute_error must be finite and positive")
    score = np.asarray([window.delivered_mass for window in windows]) * difficulty
    fractions = score / np.sum(score)
    return tuple(WindowErrorBudget(
        name=window.name,
        delivered_mass=window.delivered_mass,
        measured_difficulty=float(difficulty[index]),
        apportionment_weight=float(fractions[index]),
        absolute_error_budget=float(total_absolute_error * fractions[index]),
    ) for index, window in enumerate(windows))


__all__ = [
    "MeasureWindow",
    "WindowErrorBudget",
    "partition_measure_windows",
    "apportion_true_error",
]
