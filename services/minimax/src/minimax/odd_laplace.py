"""Shared positive-time nodes for the even and odd parts of a resolvent.

The caller supplies an existing approximation to ``x/(x²+omega²)``. This
routine keeps those time nodes and adds only the nodes needed to represent
``omega/(x²+omega²)`` to the requested absolute accuracy. It knows no GW
bands, occupations, units, or device layout.
"""
from __future__ import annotations

import numpy as np


def _lawson_weights(basis, target, *, iterations=60):
    """Weights-only sup-norm fit with column scaling and deterministic order."""
    scale = np.linalg.norm(basis, axis=0)
    scale[scale == 0.0] = 1.0
    scaled = basis / scale
    weights = np.ones(target.size)
    best_coefficients, best_error = None, np.inf
    for _ in range(iterations):
        root = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(
            scaled * root[:, None], target * root, rcond=None)
        residual = scaled @ coefficients - target
        error = float(np.max(np.abs(residual)))
        if error < best_error:
            best_coefficients, best_error = coefficients / scale, error
        weights *= np.abs(residual) + 1.0e-30
        weights /= weights.sum()
    return np.asarray(best_coefficients, dtype=np.float64), best_error


# Measured 2026-09-01 on MoS2-, CrI3- and Si-like windows: 5, 1 and 5
# extra nodes reached 1e-6. Sixteen is a refusal ceiling, not a target count.
def augment_odd_laplace(times, x_min, x_max, omega, *, tolerance,
                        max_extra=16, grid_size=4096, candidates=48):
    """Return ``(times, odd_weights, added_count, sampled_max_error)``.

    The original nodes remain in their original order. The gate is on the
    absolute error of ``omega/(x²+omega²)`` over ``[x_min,x_max]``. A missed
    gate refuses; it never returns an under-resolved odd channel.
    """
    times = np.asarray(times, dtype=np.float64)
    x_min, x_max, omega, tolerance = map(float, (x_min, x_max, omega, tolerance))
    if (times.ndim != 1 or not times.size or np.any(times <= 0)
            or not np.all(np.isfinite(times)) or not np.isfinite(x_min)
            or not np.isfinite(x_max) or not 0 < x_min <= x_max
            or not np.isfinite(omega) or omega <= 0 or not np.isfinite(tolerance)
            or tolerance <= 0 or max_extra < 0 or grid_size < 2 or candidates < 1):
        raise ValueError("Invalid odd Laplace interval, times, or tolerance")
    x = np.geomspace(x_min, x_max, int(grid_size))
    target = omega / (x * x + omega * omega)
    proposal = np.geomspace(float(times.min()) / 8.0,
                            float(times.max()) * 8.0, int(candidates))
    # Build every candidate column once. The old loop rebuilt both existing
    # columns and the proposed column for each Lawson fit.
    base = np.exp(-x[:, None] * times)
    pool = np.exp(-x[:, None] * proposal)
    selected = []
    current = times.copy()
    odd_weights, error = _lawson_weights(base, target)
    while error > tolerance and len(selected) < int(max_extra):
        best = None
        for j, value in enumerate(proposal):
            if np.any(np.abs(np.log(value / current)) < 1.0e-9):
                continue
            trial = np.column_stack((base, pool[:, j]))
            weights, trial_error = _lawson_weights(trial, target)
            if best is None or trial_error < best[0]:
                best = (trial_error, j, weights)
        if best is None:
            break
        error, j, odd_weights = best
        selected.append(j)
        current = np.append(current, proposal[j])
        base = np.column_stack((base, pool[:, j]))
    if error > tolerance:
        raise RuntimeError(
            "GATE odd_kernel_representation: odd reciprocal kernel missed "
            f"accuracy: error={error:.3e}, tolerance={tolerance:.3e}, "
            f"extra_nodes={len(selected)}, max_extra={int(max_extra)}, "
            f"interval=[{x_min:.6g},{x_max:.6g}], omega={omega:.6g}")
    return current, odd_weights, len(selected), float(error)
