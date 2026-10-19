"""Analytic sinc quadrature for ``1/z`` in a complex sector.

For ``z`` in the fourth quadrant, rotate the Laplace contour by ``pi/4``
and apply the trapezoidal rule after ``s = exp(y)``.  The node count grows
logarithmically with the radial dynamic range.  This routine is also a
well-resolved candidate dictionary for the fitted rules in
``minimax.frequency_fit``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class SectorRule(NamedTuple):
    times: np.ndarray
    weights: np.ndarray
    contour: complex
    error_bound: float
    amplification: float


def reciprocal_sector_rule(
    radial_min: float,
    radial_max: float,
    *,
    relative_error: float = 1.0e-8,
    max_nodes: int = 4096,
) -> SectorRule:
    r"""Return ``1/z ~= sum(w * exp(-z*c*s))`` on a fourth-quadrant sector.

    The domain is ``radial_min <= abs(z) <= radial_max`` and
    ``-pi/2 <= arg(z) <= 0``.  ``times`` are the positive ``s`` values;
    ``weights`` include the contour factor ``c = exp(i*pi/4)``.
    """
    r_min, r_max, tol = map(float, (radial_min, radial_max, relative_error))
    if not (0.0 < r_min <= r_max < np.inf):
        raise ValueError("radial bounds must satisfy 0 < min <= max < inf")
    if not 0.0 < tol < 1.0 or int(max_nodes) < 1:
        raise ValueError("relative_error must be in (0,1) and max_nodes positive")

    theta = np.pi / 4.0
    contour = np.exp(1.0j * theta)
    ratio = r_max / r_min
    decay = np.cos(theta)
    best = None
    for margin in np.linspace(0.02, 0.45, 173):
        strip = theta - margin
        constant = 2.0 / np.sin(margin)
        step = 2.0 * np.pi * strip / np.log1p(3.0 * constant / tol)
        left = tol * (1.0 - np.exp(-step)) / (3.0 * step * ratio)
        k_min = int(np.floor(1.0 + np.log(left) / step))
        tail = np.log(3.0 * ratio / (tol * decay)) / decay
        k_max = int(np.ceil(np.log(tail) / step))
        candidate = (k_max - k_min + 1, -step, k_min, k_max, margin, constant)
        if best is None or candidate < best:
            best = candidate

    n_nodes, neg_step, k_min, k_max, margin, constant = best
    if n_nodes > int(max_nodes):
        raise ValueError(
            f"sector rule needs {n_nodes} nodes, above max_nodes={max_nodes}")
    step = -neg_step
    k = np.arange(k_min, k_max + 1, dtype=np.float64)
    scaled = np.exp(step * k)
    times = scaled / r_min
    weights = contour * step * scaled / r_min
    strip = theta - margin
    bound = (
        constant / np.expm1(2.0 * np.pi * strip / step)
        + step * np.exp(step * (k_min - 1.0))
        / (1.0 - np.exp(-step)) * ratio
        + ratio / decay * np.exp(-decay * np.exp(step * k_max))
    )
    if bound > tol * (1.0 + 1.0e-12):
        raise ArithmeticError("constructed sector bound exceeds requested error")
    amplification = np.sum(
        np.abs(weights) * np.exp(-r_min * decay * times)) * r_min
    return SectorRule(times, weights, complex(contour), float(bound),
                      float(amplification))


__all__ = ["SectorRule", "reciprocal_sector_rule"]
