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


def _tail_refined_quantiles(bins_per_axis: int) -> Array:
    """Return the delivered-planner quantile edges for one support axis."""
    bins = int(bins_per_axis)
    if bins < 4:
        raise ValueError("bins_per_axis must be at least 4")
    interior = np.linspace(0.04, 0.96, max(2, bins - 3))
    return np.concatenate(([0.0, 0.01], interior, [0.99, 1.0]))


def _lattice_axis(values: Array, quantiles: Array) -> Array:
    """Quantile lattice nodes, retaining a singleton degenerate axis."""
    return np.unique(np.quantile(np.asarray(values, dtype=np.float64), quantiles))


def _base_and_refined_axes(values: Array, bins_per_axis: int):
    """Build both quantile axes with one sort of the same support values."""
    base_q = _tail_refined_quantiles(bins_per_axis)
    refined_q = _tail_refined_quantiles(2 * int(bins_per_axis))
    all_q = np.unique(np.concatenate((base_q, refined_q)))
    sampled = np.quantile(np.asarray(values, dtype=np.float64), all_q)
    base = np.unique(sampled[np.searchsorted(all_q, base_q)])
    refined = np.unique(sampled[np.searchsorted(all_q, refined_q)])
    return base, refined


def _axis_cloud_weights(values: Array, nodes: Array):
    """Return cloud-in-cell node indices and weights on one axis."""
    if nodes.size == 1:
        zero = np.zeros(values.size, dtype=np.int64)
        return ((zero, np.ones(values.size, dtype=np.float64)),)
    lower = np.clip(np.searchsorted(nodes, values, side="right") - 1,
                    0, nodes.size - 2)
    width = nodes[lower + 1] - nodes[lower]
    fraction = np.where(
        width > 0.0,
        (values - nodes[lower]) / np.where(width > 0.0, width, 1.0),
        0.0,
    )
    return ((lower, 1.0 - fraction), (lower + 1, fraction))


def _build_tail_refined_lattice(
    internal_sums: Array,
    masses: Array,
    real_nodes: Array,
    imag_nodes: Array,
) -> tuple[Array, Array]:
    real_cloud = _axis_cloud_weights(internal_sums.real, real_nodes)
    imag_cloud = _axis_cloud_weights(internal_sums.imag, imag_nodes)
    grid = np.zeros(real_nodes.size * imag_nodes.size, dtype=np.float64)
    for real_index, real_weight in real_cloud:
        for imag_index, imag_weight in imag_cloud:
            flat_index = real_index * imag_nodes.size + imag_index
            grid += np.bincount(
                flat_index,
                weights=masses * real_weight * imag_weight,
                minlength=grid.size)
    nodes = (real_nodes[:, None]
             + 1.0j * imag_nodes[None, :]).reshape(-1)
    live = grid > 0.0
    return nodes[live], grid[live]


def tail_refined_lattice_measure(
    internal_sums: Array,
    masses: Array,
    *,
    bins_per_axis: int = 25,
) -> tuple[Array, Array, Array, Array]:
    """Compress a delivered measure onto a tail-refined complex lattice.

    This is the production form of the DEV-80 ``lattice_measure`` prototype.
    Each real and imaginary axis uses count-quantile edges at 0%, 1%, then a
    regular 4%--96% interior, 99%, and 100%.  Every support point shares its
    delivered mass bilinearly among the four surrounding nodes, so the
    representation is a piecewise-linear density rather than a centroid
    point mass.  A second lattice at twice the requested resolution is
    returned for caller-side refinement checks.  Degenerate one-value axes
    are represented by one node and conserve mass exactly.

    Parameters
    ----------
    internal_sums : ndarray, shape (n_support,)
        Complex support values in caller units.
    masses : ndarray, shape (n_support,)
        Nonnegative delivered mass for each support value.
    bins_per_axis : int, optional
        Nominal count-quantile resolution per complex axis.

    Returns
    -------
    cells, cell_masses, refined_cells, refined_cell_masses : ndarray
        Base and doubled-resolution complex lattices.  Each mass array sums
        to the input delivered mass to roundoff.
    """
    sums = np.asarray(internal_sums, dtype=np.complex128).reshape(-1)
    weight = np.asarray(masses, dtype=np.float64).reshape(-1)
    if sums.size == 0 or sums.shape != weight.shape:
        raise ValueError("internal_sums and masses must be matching nonempty arrays")
    if not (np.all(np.isfinite(sums)) and np.all(np.isfinite(weight))):
        raise ValueError("support and masses must be finite")
    if np.any(weight < 0.0) or not np.any(weight > 0.0):
        raise ValueError("masses must be nonnegative with positive total")
    live = weight > 0.0
    sums, weight = sums[live], weight[live]
    total = float(np.sum(weight))
    real_base, real_refined = _base_and_refined_axes(
        sums.real, int(bins_per_axis))
    imag_base, imag_refined = _base_and_refined_axes(
        sums.imag, int(bins_per_axis))
    cells, cell_mass = _build_tail_refined_lattice(
        sums, weight, real_base, imag_base)
    refined_cells, refined_mass = _build_tail_refined_lattice(
        sums, weight, real_refined, imag_refined)
    for label, got in (("base", cell_mass), ("refined", refined_mass)):
        error = abs(float(np.sum(got)) - total)
        if error > 1.0e-12 * max(total, 1.0):
            raise RuntimeError(
                f"{label} tail-refined lattice did not conserve mass: "
                f"{error:.3e}")
    return cells, cell_mass, refined_cells, refined_mass
