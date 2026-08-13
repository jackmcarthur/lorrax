"""Reusable positive minimax fitting for complex-frequency resolvents.

This module contains only scalar offline fitting.  It knows nothing about
wavefunctions, MPA pole files, JAX, sharding, or a particular material.
Callers provide the physical sample domain and candidate time nodes, then
serialize the returned arrays through their existing plan interface.

For fixed nodes the complex Chebyshev problem is represented by a regular
polygon in the residual plane and solved as a linear program.  If ``rho`` is
the polygon optimum for ``n_directions`` directions, then
``rho / cos(pi / n_directions)`` bounds the circular residual on the fitted
sample set.  Dense/exchange sampling remains numerical evidence; it is not a
continuum certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import linprog


Array = np.ndarray


@dataclass(frozen=True)
class PositiveChebyshevFit:
    """Fixed-node positive complex Chebyshev result."""

    weights: Array
    polygon_radius: float
    circular_sample_bound: float
    iterations: int


@dataclass(frozen=True)
class SharedSineFit:
    """One positive sine dictionary shared by several complex frequencies."""

    times: Array
    weights: Array
    sampled_max_error: float
    sampled_error_by_frequency: Array
    worst_delta_by_frequency: Array
    history: tuple[dict[str, float | int], ...]

    @property
    def sine_rank(self) -> int:
        return int(self.times.size)

    @property
    def executed_contour_nodes(self) -> int:
        """Each sine atom is the exact union of its ``+i`` and ``-i`` arms."""
        return 2 * self.sine_rank


def positive_complex_chebyshev(
    basis: Array,
    target: Array,
    *,
    n_directions: int = 16,
    time_limit: float = 120.0,
) -> PositiveChebyshevFit:
    """Minimize a polygonal complex infinity norm with nonnegative weights.

    Parameters
    ----------
    basis
        Complex matrix ``(n_sample, n_node)``.
    target
        Complex target vector ``(n_sample,)``.
    n_directions
        Number of equally spaced supporting directions for the residual
        disk.  Must be at least four.

    Returns
    -------
    PositiveChebyshevFit
        Nonnegative physical weights and both the polygon optimum and its
        circular-norm upper bound on the supplied samples.
    """
    basis = np.asarray(basis, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)
    if (basis.ndim != 2 or target.shape != (basis.shape[0],)
            or basis.shape[0] == 0 or basis.shape[1] == 0):
        raise ValueError("basis must be (n_sample,n_node) and target (n_sample,)")
    if not (np.all(np.isfinite(basis)) and np.all(np.isfinite(target))):
        raise ValueError("basis and target must be finite")
    if int(n_directions) < 4:
        raise ValueError("n_directions must be at least four")

    scale = np.max(np.abs(basis), axis=0)
    if np.any(scale <= np.finfo(np.float64).tiny):
        raise ValueError("basis contains a numerically zero column")
    normalized = basis / scale[None, :]
    blocks = []
    bounds = []
    for theta in (2.0 * np.pi * np.arange(int(n_directions))
                  / int(n_directions)):
        phase = np.exp(-1.0j * theta)
        blocks.append(np.column_stack((
            np.real(phase * normalized), -np.ones(target.size))))
        bounds.append(np.real(phase * target))

    result = linprog(
        c=np.r_[np.zeros(basis.shape[1]), 1.0],
        A_ub=np.vstack(blocks),
        b_ub=np.concatenate(bounds),
        bounds=[(0.0, None)] * (basis.shape[1] + 1),
        method="highs",
        options={"time_limit": float(time_limit)},
    )
    if not result.success:
        result = linprog(
            c=np.r_[np.zeros(basis.shape[1]), 1.0],
            A_ub=np.vstack(blocks),
            b_ub=np.concatenate(bounds),
            bounds=[(0.0, None)] * (basis.shape[1] + 1),
            method="highs-ipm",
            options={"time_limit": float(time_limit)},
        )
    if not result.success:
        raise RuntimeError(f"complex Chebyshev solve failed: {result.message}")

    rho = float(result.x[-1])
    return PositiveChebyshevFit(
        weights=np.asarray(result.x[:-1], dtype=np.float64) / scale,
        polygon_radius=rho,
        circular_sample_bound=(
            rho / math.cos(math.pi / int(n_directions))),
        iterations=int(result.nit),
    )


def shared_sine_system(
    delta: Array,
    frequencies: Array,
    times: Array,
    *,
    scale: Array | None = None,
) -> tuple[Array, Array]:
    r"""Return the shared finite-``z`` screening system.

    It fits

    .. math::

       K_z(\Delta) = -\frac{2\Delta}{\Delta^2-z^2}
       = -2\int_0^\infty e^{izt}\sin(\Delta t)\,dt.

    ``scale`` is a nonnegative per-frequency row scale.  Choosing
    ``scale=Im(z)`` gives the dimensionless damped-line relative metric used
    by the near-real screening campaign without changing sample locations or
    performing analytic continuation.
    """
    delta = np.asarray(delta, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.complex128)
    times = np.asarray(times, dtype=np.float64)
    if (delta.ndim != 1 or frequencies.ndim != 1 or times.ndim != 1
            or min(delta.size, frequencies.size, times.size) == 0):
        raise ValueError("delta, frequencies, and times must be nonempty vectors")
    if np.any(times <= 0.0):
        raise ValueError("time nodes must be positive")
    if scale is None:
        row_scale = np.ones(frequencies.size, dtype=np.float64)
    else:
        row_scale = np.asarray(scale, dtype=np.float64)
        if row_scale.shape != frequencies.shape or np.any(row_scale < 0.0):
            raise ValueError("scale must be a nonnegative per-frequency vector")

    sine = np.sin(delta[:, None] * times[None, :])
    basis = (-2.0
             * np.exp(1.0j * frequencies[:, None, None]
                      * times[None, None, :])
             * sine[None, :, :]
             * row_scale[:, None, None])
    exact = (-2.0 * delta[None, :]
             / (delta[None, :] ** 2 - frequencies[:, None] ** 2)
             * row_scale[:, None])
    return basis.reshape(-1, times.size), exact.reshape(-1)


def resolvent_system(
    denominators: Array,
    contour: complex,
    times: Array,
) -> tuple[Array, Array]:
    r"""Return a relative-residual system for a rotated Laplace rule.

    The returned system fits

    .. math::

       1 \approx d\,c\sum_j h_j e^{-c d t_j},\qquad h_j\ge0,

    directly on the caller's actual complex denominator samples.
    """
    denominators = np.asarray(denominators, dtype=np.complex128).reshape(-1)
    times = np.asarray(times, dtype=np.float64)
    contour = complex(contour)
    if denominators.size == 0 or times.ndim != 1 or times.size == 0:
        raise ValueError("denominators and times must be nonempty")
    if np.any(times <= 0.0) or not np.isclose(abs(contour), 1.0):
        raise ValueError("times must be positive and contour must have unit modulus")
    decay = np.real(contour * denominators)
    if np.any(decay <= 0.0):
        raise ValueError("contour is inadmissible for at least one denominator")
    basis = (denominators[:, None] * contour
             * np.exp(-contour * denominators[:, None] * times[None, :]))
    return basis, np.ones(denominators.size, dtype=np.complex128)


def _score_shared_sine(
    delta: Array,
    frequencies: Array,
    times: Array,
    weights: Array,
    scale: Array,
    *,
    chunk_size: int,
) -> tuple[Array, Array]:
    per_frequency = np.zeros(frequencies.size, dtype=np.float64)
    locations = np.zeros(frequencies.size, dtype=np.float64)
    for left in range(0, delta.size, int(chunk_size)):
        values = delta[left:left + int(chunk_size)]
        basis, exact = shared_sine_system(
            values, frequencies, times, scale=scale)
        error = np.abs(basis @ weights - exact).reshape(
            frequencies.size, values.size)
        indices = np.argmax(error, axis=1)
        maxima = error[np.arange(frequencies.size), indices]
        improve = maxima > per_frequency
        per_frequency[improve] = maxima[improve]
        locations[improve] = values[indices[improve]]
    return per_frequency, locations


def fit_shared_sine_rule(
    frequencies: Array,
    candidate_times: Array,
    delta_interval: tuple[float, float],
    *,
    row_scale: Array | None = None,
    solve_points: int = 257,
    validation_points: int = 16_384,
    exchange_rounds: int = 2,
    maxima_per_frequency: int = 4,
    n_directions: int = 16,
    relative_support_cut: float = 1.0e-12,
) -> SharedSineFit:
    """Fit one positive sine dictionary to a finite complex-frequency set."""
    frequencies = np.asarray(frequencies, dtype=np.complex128)
    times = np.asarray(candidate_times, dtype=np.float64)
    lo, hi = map(float, delta_interval)
    if not (0.0 < lo < hi):
        raise ValueError("delta_interval must satisfy 0 < lower < upper")
    if row_scale is None:
        scale = np.ones(frequencies.size, dtype=np.float64)
    else:
        scale = np.asarray(row_scale, dtype=np.float64)
    active = np.linspace(lo, hi, int(solve_points))
    dense = np.unique(np.r_[
        lo, hi,
        lo + (np.arange(int(validation_points)) + 0.5)
        * (hi - lo) / int(validation_points),
    ])
    history: list[dict[str, float | int]] = []

    for iteration in range(int(exchange_rounds) + 1):
        basis, exact = shared_sine_system(
            active, frequencies, times, scale=scale)
        fit = positive_complex_chebyshev(
            basis, exact, n_directions=int(n_directions))
        weights = fit.weights

        damping = np.maximum(frequencies.imag, 0.0)
        effective = weights[None, :] * np.exp(
            -damping[:, None] * times[None, :])
        live = np.max(effective, axis=0) > (
            float(np.max(effective)) * float(relative_support_cut))
        if not np.all(live):
            times = times[live]
            basis, exact = shared_sine_system(
                active, frequencies, times, scale=scale)
            fit = positive_complex_chebyshev(
                basis, exact, n_directions=int(n_directions))
            weights = fit.weights

        per_frequency, locations = _score_shared_sine(
            dense, frequencies, times, weights, scale, chunk_size=2048)
        history.append({
            "iteration": iteration,
            "active_points": int(active.size),
            "rank": int(times.size),
            "sampled_max_error": float(np.max(per_frequency)),
            "polygon_radius": fit.polygon_radius,
            "circular_active_bound": fit.circular_sample_bound,
        })
        if iteration == int(exchange_rounds):
            break

        additions = []
        for index in np.argsort(per_frequency)[-int(maxima_per_frequency):]:
            additions.append(locations[index])
        enlarged = np.unique(np.r_[active, additions])
        if enlarged.size == active.size:
            break
        active = enlarged

    return SharedSineFit(
        times=np.asarray(times, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        sampled_max_error=float(np.max(per_frequency)),
        sampled_error_by_frequency=per_frequency,
        worst_delta_by_frequency=locations,
        history=tuple(history),
    )


def select_smallest_rank(
    candidates: Iterable[tuple[Array, Array]],
    system: Callable[[Array], tuple[Array, Array]],
    target_error: float,
    *,
    n_directions: int = 16,
) -> tuple[Array, PositiveChebyshevFit]:
    """Return the first increasing-rank positive rule meeting a sample gate.

    ``candidates`` yields ``(nodes, validation_basis)`` pairs.  ``system``
    receives the nodes and returns the solve basis/target.  The validation
    basis must have the same columns and is evaluated against the same target;
    callers with a different validation target should perform their own gate.
    """
    for nodes, validation_basis in candidates:
        nodes = np.asarray(nodes)
        basis, target = system(nodes)
        fit = positive_complex_chebyshev(
            basis, target, n_directions=n_directions)
        validation_basis = np.asarray(validation_basis, dtype=np.complex128)
        if validation_basis.shape[1] != fit.weights.size:
            raise ValueError("validation basis has the wrong node dimension")
        if validation_basis.shape[0] != target.size:
            raise ValueError("validation basis and solve target differ in length")
        error = float(np.max(np.abs(validation_basis @ fit.weights - target)))
        if error <= float(target_error):
            return nodes, fit
    raise RuntimeError("no candidate rank met the requested sampled error")
