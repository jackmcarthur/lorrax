"""Reusable positive minimax fitting for complex-frequency resolvents.

This module contains only scalar offline fitting.  It knows nothing about
wavefunctions, MPA pole files, JAX, sharding, or a particular material.
Callers provide the physical sample domain and candidate time nodes, then
serialize the returned arrays through their existing plan interface.
Dense/exchange sampling remains numerical evidence; it is not a
continuum certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr
from scipy.optimize import nnls

from minimax.sector import reciprocal_sector_rule


Array = np.ndarray


@dataclass(frozen=True)
class DampedReciprocalFit:
    """Positive rotated-Laplace rule fitted to complex rectangles."""

    nodes: Array
    weights: Array
    sampled_max_error: float
    contour_angle: float
    amplification: float


def _rectangle_samples(rectangles: Array, count: int, *, midpoint: bool) -> Array:
    points = []
    axis = ((np.arange(count) + 0.5) / count if midpoint
            else np.linspace(0.0, 1.0, count))
    for x_lo, x_hi, gamma_lo, gamma_hi in rectangles:
        x = np.exp(np.log(x_lo) + axis * np.log(x_hi / x_lo))
        gamma = gamma_lo + axis * (gamma_hi - gamma_lo)
        points.extend((x[:, None] - 1.0j * gamma[None, :]).ravel())
    return np.asarray(points, dtype=np.complex128)


def _nonnegative_lawson(matrix: Array, iterations: int) -> Array:
    row_weight = np.ones(matrix.shape[0], dtype=np.float64)
    coefficient = np.zeros(matrix.shape[1], dtype=np.float64)
    for _ in range(int(iterations)):
        root = np.sqrt(row_weight)
        system = np.concatenate(
            (matrix.real * root[:, None], matrix.imag * root[:, None]))
        coefficient = nnls(
            system, np.r_[root, np.zeros_like(root)],
            maxiter=max(20 * matrix.shape[1], 1))[0]
        residual = np.abs(1.0 - matrix @ coefficient)
        floor = max(float(np.max(residual)) * 1.0e-3, 1.0e-30)
        row_weight *= np.maximum(residual, floor)
        row_weight /= np.mean(row_weight)
        row_weight = np.minimum(row_weight, 1.0e12)
    return coefficient


def fit_damped_reciprocal(
    rectangles: Array,
    *,
    target_error: float,
    max_rank: int = 96,
    training_points: int = 12,
    validation_points: int = 48,
    contour_count: int = 5,
    lawson_iterations: int = 4,
) -> DampedReciprocalFit:
    r"""Fit one positive rule to ``1/(x-i gamma)`` over rectangles.

    Each input row is ``(x_min, x_max, gamma_min, gamma_max)`` with
    ``x_min > 0`` and ``gamma_min >= 0``.  The returned arrays satisfy

    ``1/d ~= sum(weights * exp(-d * nodes))``.

    The reported error is measured on an independent tensor midpoint grid;
    it is numerical evidence, not a continuum certificate.
    """
    cells = np.asarray(rectangles, dtype=np.float64)
    if cells.ndim != 2 or cells.shape[1] != 4 or cells.shape[0] == 0:
        raise ValueError("rectangles must have shape (n,4)")
    if (not np.all(np.isfinite(cells)) or np.any(cells[:, 0] <= 0.0)
            or np.any(cells[:, 1] < cells[:, 0])
            or np.any(cells[:, 2] < 0.0)
            or np.any(cells[:, 3] < cells[:, 2])):
        raise ValueError("invalid damped-reciprocal rectangle")
    if not 0.0 < float(target_error) < 1.0:
        raise ValueError("target_error must lie in (0,1)")

    scale = float(np.min(np.hypot(cells[:, 0], cells[:, 2])))
    radial_max = float(np.max(np.hypot(cells[:, 1], cells[:, 3]))) / scale
    seed = reciprocal_sector_rule(
        1.0, radial_max,
        relative_error=max(float(target_error) * 1.0e-5, 1.0e-14))
    scalar_nodes = np.asarray(seed.times, dtype=np.float64)
    train = _rectangle_samples(cells / np.array([scale, scale, scale, scale]),
                               int(training_points), midpoint=False)
    heldout = _rectangle_samples(cells / np.array([scale, scale, scale, scale]),
                                 int(validation_points), midpoint=True)

    phi_lo = float(np.min(np.arctan2(cells[:, 2], cells[:, 1])))
    phi_hi = float(np.max(np.arctan2(cells[:, 3], cells[:, 0])))
    middle = 0.5 * (phi_lo + phi_hi)
    radius = min(0.25, 0.5 * (np.pi / 2.0 - (phi_hi - phi_lo)),
                 middle, np.pi / 2.0 - middle)
    angles = (np.asarray([middle]) if int(contour_count) == 1 or radius <= 0.0
              else np.linspace(middle - radius, middle + radius,
                               int(contour_count)))
    problems = []
    for angle in angles:
        contour = np.exp(1.0j * angle)
        basis = (train[:, None] * contour
                 * np.exp(-train[:, None] * contour * scalar_nodes[None, :]))
        pivots = qr(basis, mode="economic", pivoting=True)[2]
        problems.append((float(angle), contour, basis, pivots))

    best = None
    for trial_rank in range(2, int(max_rank) + 1):
        candidates = []
        for angle, contour, basis, pivots in problems:
            indices = np.sort(pivots[:trial_rank])
            coefficient = _nonnegative_lawson(
                basis[:, indices], int(lawson_iterations))
            live = coefficient > 0.0
            indices, coefficient = indices[live], coefficient[live]
            residual = 1.0 - (
                heldout[:, None] * contour
                * np.exp(-heldout[:, None] * contour
                         * scalar_nodes[indices][None, :])) @ coefficient
            candidates.append((float(np.max(np.abs(residual))), angle,
                               contour, indices, coefficient))
        best = min(candidates, key=lambda row: (row[0], row[1]))
        if best[0] <= float(target_error):
            error, angle, contour, indices, coefficient = best
            nodes = contour * scalar_nodes[indices] / scale
            weights = contour * coefficient / scale
            amplification = float(np.sum(np.abs(coefficient)))
            return DampedReciprocalFit(
                np.asarray(nodes, dtype=np.complex128),
                np.asarray(weights, dtype=np.complex128), error, angle,
                amplification)
    raise RuntimeError(
        f"no rule reached {target_error:g} through rank {max_rank}; "
        f"best sampled residual was {best[0]:.6g}")
