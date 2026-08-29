r"""Delivered-error fitting of ``1/d`` on a weighted complex support.

This module contains only scalar offline fitting.  It knows nothing about
wavefunctions, MPA pole files, JAX, sharding, or a particular material.
The caller supplies real external frequencies, complex internal sums, and
nonnegative cell masses; denominators are ``d = frequency - internal_sum``.
The fitted rule is ``Q(d) = sum(weights * exp(1j * time_nodes * d))`` with
unrestricted complex weights.

The objective is the delivered branch error of the owner's guide, not a
uniform sup norm: at each requested frequency the kernel error
``|Q(d) - 1/d|`` is integrated against the cell masses and normalized by
the delivered mass ``sum(mass / |d|)``, and the fit minimizes the maximum
of that ratio over the requested frequencies.  Accuracy therefore relaxes
continuously where little spectral weight exists, and no part of a
bounding rectangle is fitted merely for convenience.

Every error this module reports is measured on the supplied cells; it is
numerical evidence, not a continuum certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack

Array = np.ndarray


@dataclass(frozen=True)
class ReciprocalMeasureProblem:
    """Weighted complex support for one branch fit.

    ``frequencies`` are the real requested external values; the fit
    protects every entry.  ``internal_sums`` are the complex signed sums
    the branch subtracts, and ``cell_masses`` their nonnegative spectral
    masses (occupation x k-weight x |B| in the caller's units; the scale
    cancels in the delivered error).  Cells with ``|d| < excluded_radius``
    are dropped from both the error and its normalization, and the dropped
    mass is reported, never silently discarded.  ``zero_weight_sum``
    imposes ``Q(0) = 0`` (``sum(weights) = 0``), the linear form of the
    odd principal-value constraint for deliberately regularized real
    poles.  The module is dimensionless: the rescale stays with the
    caller.
    """

    frequencies: Array
    internal_sums: Array
    cell_masses: Array
    excluded_radius: float = 0.0
    normalization_floor: float = 0.0
    zero_weight_sum: bool = False

    def __post_init__(self) -> None:
        frequency = np.asarray(self.frequencies, dtype=np.float64)
        sums = np.asarray(self.internal_sums, dtype=np.complex128)
        mass = np.asarray(self.cell_masses, dtype=np.float64)
        if frequency.ndim != 1 or frequency.size == 0:
            raise ValueError("frequencies must be a nonempty 1d real array")
        if sums.ndim != 1 or sums.size == 0 or sums.shape != mass.shape:
            raise ValueError(
                "internal_sums and cell_masses must be matching nonempty 1d arrays")
        if not (np.all(np.isfinite(frequency)) and np.all(np.isfinite(sums))
                and np.all(np.isfinite(mass))):
            raise ValueError("problem arrays must be finite")
        if np.any(mass < 0.0) or not np.any(mass > 0.0):
            raise ValueError("cell_masses must be nonnegative with positive total")
        if not 0.0 <= float(self.excluded_radius) < np.inf:
            raise ValueError("excluded_radius must be finite and nonnegative")
        if not 0.0 <= float(self.normalization_floor) < np.inf:
            raise ValueError("normalization_floor must be finite and nonnegative")
        object.__setattr__(self, "frequencies", frequency)
        object.__setattr__(self, "internal_sums", sums)
        object.__setattr__(self, "cell_masses", mass)

    @property
    def denominators(self) -> Array:
        """All ``d[i, j] = frequencies[i] - internal_sums[j]``."""
        return self.frequencies[:, None] - self.internal_sums[None, :]

    def retained(self) -> tuple[Array, Array, Array]:
        """Per-frequency kept mask, delivered mass, and excluded mass.

        A cell is kept at a frequency when ``|d| >= excluded_radius``; a
        genuinely zero denominator is singular and always excluded.  The
        delivered mass is ``sum(mass / |d|)`` over kept cells plus the
        normalization floor.
        """
        magnitude = np.abs(self.denominators)
        floor = max(float(self.excluded_radius), 0.0)
        kept = magnitude > max(floor, 0.0) if floor > 0.0 else magnitude > 0.0
        with np.errstate(divide="ignore"):
            inverse = np.where(kept, 1.0 / np.where(kept, magnitude, 1.0), 0.0)
        delivered = inverse @ self.cell_masses + float(self.normalization_floor)
        excluded = (~kept) @ self.cell_masses
        return kept, delivered, excluded


@dataclass(frozen=True)
class ComplexTimeRule:
    """Fitted complex-exponential rule and its measured evidence.

    ``1/d ~= sum(weights * exp(1j * time_nodes * d))`` on the problem's
    support.  ``node_families`` records which candidate family produced
    each retained node, in the same order.  ``delivered_error_by_frequency``
    is the fitting-measure evidence; ``sampled_max_error`` is its maximum.
    ``amplification`` is the mass-weighted p99 of
    ``sum(|w * exp(1j t d)|) / |Q(d)|`` and ``amplification_max`` its
    maximum, both measured on the fitting cells.  ``method`` is
    ``"sector"`` when the certified rotated sinc rule answered (its
    ``sampled_max_error`` is then still measured on the cells; the
    analytic bound travels in ``error_bound``) and ``"measured_crossing"``
    for the fitted rule, where ``error_bound`` is None.
    """

    time_nodes: Array
    weights: Array
    node_families: tuple[str, ...]
    delivered_error_by_frequency: Array
    sampled_max_error: float
    excluded_mass_fraction: float
    amplification: float
    amplification_max: float
    method: str
    error_bound: float | None = None
    # One row per accepted growth step, in acceptance order:
    # (time_re, time_im, family, delivered_error_after) — enough to rebuild
    # any prefix rule for convergence-versus-node-count studies.
    growth_history: tuple[tuple[float, float, str, float], ...] = ()

    @property
    def node_count(self) -> int:
        return int(self.time_nodes.size)

    def one_line(self) -> str:
        bound = ("" if self.error_bound is None
                 else f" analytic bound {self.error_bound:.2e}")
        return (f"{self.method}: {self.node_count} nodes, delivered error "
                f"{self.sampled_max_error:.2e},{bound} amplification p99 "
                f"{self.amplification:.3g} max {self.amplification_max:.3g}")


def evaluate_rule(times: Array, weights: Array, denominators: Array) -> Array:
    """``Q(d) = sum(weights * exp(1j * times * d))`` at each denominator."""
    d = np.asarray(denominators, dtype=np.complex128)
    phases = np.exp(1.0j * d[..., None] * np.asarray(times)[None, :])
    return phases @ np.asarray(weights, dtype=np.complex128)


def delivered_error(problem: ReciprocalMeasureProblem,
                    times: Array, weights: Array) -> tuple[Array, Array]:
    """Delivered branch error and excluded mass fraction per frequency.

    Returns ``(error_by_frequency, excluded_fraction_by_frequency)`` where
    the error at one frequency is
    ``sum(mass * |Q(d) - 1/d|) / (sum(mass / |d|) + floor)`` over kept
    cells.  This is the exact reporting metric; the LP inside the solver
    sees only its polygonal relaxation.
    """
    kept, delivered, excluded = problem.retained()
    d = problem.denominators
    with np.errstate(divide="ignore", invalid="ignore"):
        truth = np.where(kept, 1.0 / np.where(kept, d, 1.0), 0.0)
    residual = np.abs(evaluate_rule(times, weights, d) - truth)
    numerator = np.where(kept, residual, 0.0) @ problem.cell_masses
    total_mass = float(np.sum(problem.cell_masses))
    return numerator / delivered, excluded / total_mass


def rule_amplification(times: Array, weights: Array,
                       problem: ReciprocalMeasureProblem) -> tuple[float, float]:
    """Mass-weighted p99 and maximum of the term-cancellation ratio.

    ``kappa(d) = sum(|w * exp(1j t d)|) / max(|Q(d)|, tiny)`` measured on
    the problem's kept cells at every requested frequency.
    """
    kept, _, _ = problem.retained()
    d = problem.denominators
    magnitude = np.abs(np.exp(1.0j * d[..., None] * np.asarray(times)[None, :])
                       * np.asarray(weights)[None, None, :]).sum(axis=-1)
    value = np.abs(evaluate_rule(times, weights, d))
    kappa = (magnitude / np.maximum(value, 1.0e-300))[kept]
    mass = np.broadcast_to(problem.cell_masses[None, :], d.shape)[kept]
    order = np.argsort(kappa, kind="stable")
    cumulative = np.cumsum(mass[order])
    p99 = kappa[order][np.searchsorted(cumulative, 0.99 * cumulative[-1])]
    return float(p99), float(np.max(kappa))


def _polygon_directions(count: int) -> Array:
    return np.exp(-1.0j * 2.0 * np.pi * np.arange(int(count)) / int(count))


def solve_fixed_time_weights(
    problem: ReciprocalMeasureProblem,
    times: Array,
    *,
    polygon_directions: int = 16,
    conditioning_slack: float = 1.0e-3,
    conditioning_pass: bool = True,
    objective_scale: float = 1.0,
) -> tuple[Array, float]:
    r"""Best complex weights for fixed time nodes, by linear program.

    Minimizes ``T = max_i sum_j(mass_j u_ij) / delivered_i`` with
    ``u_ij >= |Q(d_ij) - 1/d_ij|`` approximated by a regular polygon of
    ``polygon_directions`` half-planes; the polygon under-reads ``|r|`` by
    at most ``1/cos(pi/M)`` and the caller re-measures with the exact
    metric.  ``objective_scale`` multiplies the residual system so the
    optimum sits at order one — the natural optimum is the relative
    delivered error, which at tight targets falls below the solver's
    absolute feasibility tolerances and reports as zero; pass roughly
    ``1/target_error``.  A second solve then minimizes ``sum|weights|``
    inside ``(1 + conditioning_slack)`` of the first optimum, spending a
    bounded slice of accuracy on cancellation control.  Returns
    ``(weights, polygon_objective)`` with the objective unscaled.
    """
    nodes = np.asarray(times, dtype=np.complex128)
    if nodes.ndim != 1 or nodes.size == 0:
        raise ValueError("times must be a nonempty 1d array")
    if int(polygon_directions) < 8:
        raise ValueError("polygon_directions must be at least 8")
    if not 0.0 < float(objective_scale) < np.inf:
        raise ValueError("objective_scale must be positive and finite")

    kept, delivered, _ = problem.retained()
    d = problem.denominators
    n_frequency, _ = d.shape
    keep_i, keep_j = np.nonzero(kept & (problem.cell_masses[None, :] > 0.0))
    n_kept = keep_i.size
    if n_kept == 0:
        raise ValueError("no cell survives the excluded region")

    d_flat = d[keep_i, keep_j]
    # Node gauge: reference each atom at its own best-decaying support point
    # so separate exponentials stay moderate even when their product is; the
    # gauge is folded back into the returned weights exactly.
    exponent = 1.0j * d_flat[:, None] * nodes[None, :]
    log_peak = np.max(exponent.real, axis=0)
    atoms = np.exp(exponent - log_peak[None, :])
    gauge = np.exp(-log_peak)
    truth = 1.0 / d_flat
    row_scale = (problem.cell_masses[keep_j] / delivered[keep_i]
                 * float(objective_scale))

    directions = _polygon_directions(polygon_directions)
    n_nodes = nodes.size
    n_var = 2 * n_nodes + n_kept + 1  # Re(a), Im(a), scaled u, T

    rows, cols, values, rhs = [], [], [], []
    row = 0
    for direction in directions:
        turned = direction * atoms * row_scale[:, None]
        sample = np.arange(n_kept)
        rows.append(np.repeat(row + sample, n_nodes))
        cols.append(np.tile(np.arange(n_nodes), n_kept))
        values.append(turned.real.ravel())
        rows.append(np.repeat(row + sample, n_nodes))
        cols.append(np.tile(n_nodes + np.arange(n_nodes), n_kept))
        values.append(-turned.imag.ravel())
        rows.append(row + sample)
        cols.append(2 * n_nodes + sample)
        values.append(np.full(n_kept, -1.0))
        rhs.append((direction * truth * row_scale).real)
        row += n_kept
    per_frequency = coo_matrix(
        (np.ones(n_kept), (keep_i, np.arange(n_kept))),
        shape=(n_frequency, n_kept))
    rows.append(per_frequency.row + row)
    cols.append(2 * n_nodes + per_frequency.col)
    values.append(per_frequency.data)
    rows.append(row + np.arange(n_frequency))
    cols.append(np.full(n_frequency, n_var - 1))
    values.append(np.full(n_frequency, -1.0))
    rhs.append(np.zeros(n_frequency))
    row += n_frequency

    inequality = coo_matrix(
        (np.concatenate(values),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(row, n_var)).tocsc()
    upper = np.concatenate(rhs)

    equality = None
    equality_rhs = None
    if problem.zero_weight_sum:
        real_row = np.concatenate([gauge.real, -gauge.imag, np.zeros(n_kept + 1)])
        imag_row = np.concatenate([gauge.imag, gauge.real, np.zeros(n_kept + 1)])
        equality = np.vstack([real_row, imag_row])
        equality_rhs = np.zeros(2)

    cost = np.zeros(n_var)
    cost[-1] = 1.0
    bounds = ([(None, None)] * (2 * n_nodes)
              + [(0.0, None)] * n_kept + [(0.0, None)])
    first = linprog(cost, A_ub=inequality, b_ub=upper, A_eq=equality,
                    b_eq=equality_rhs, bounds=bounds, method="highs")
    if not first.success:
        raise RuntimeError(
            f"weight solve failed for {n_nodes} nodes over {n_kept} cells: "
            f"{first.message}")
    objective = float(first.x[-1])
    solution = first.x

    if conditioning_pass and n_nodes > 1:
        # Spend the declared slack on cancellation control: same residual
        # polygon, T pinned, minimize sum |a| through per-node envelope
        # variables c with (sign_r * Re a + sign_i * Im a) / sqrt(2) <= c,
        # the four-direction outer polygon of |a|.
        pinned = float(objective) * (1.0 + float(conditioning_slack)) + 1.0e-30
        n_var2 = n_var + n_nodes
        sample = np.arange(n_nodes)
        envelope_blocks = []
        for sign_r, sign_i in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            block = coo_matrix(
                (np.concatenate([np.full(n_nodes, sign_r / np.sqrt(2.0)),
                                 np.full(n_nodes, sign_i / np.sqrt(2.0)),
                                 np.full(n_nodes, -1.0)]),
                 (np.tile(sample, 3),
                  np.concatenate([sample, n_nodes + sample, n_var + sample]))),
                shape=(n_nodes, n_var2))
            envelope_blocks.append(block)
        stacked = vstack(
            [hstack([inequality, coo_matrix((row, n_nodes))])]
            + envelope_blocks).tocsc()
        stacked_rhs = np.concatenate([upper, np.zeros(4 * n_nodes)])
        cost2 = np.zeros(n_var2)
        cost2[n_var:] = np.abs(gauge)
        bounds2 = bounds[:-1] + [(0.0, pinned)] + [(0.0, None)] * n_nodes
        equality2 = None if equality is None else np.hstack(
            [equality, np.zeros((2, n_nodes))])
        second = linprog(cost2, A_ub=stacked, b_ub=stacked_rhs,
                         A_eq=equality2, b_eq=equality_rhs,
                         bounds=bounds2, method="highs")
        if second.success:
            solution = second.x[:n_var]
            objective = float(second.x[n_var - 1])

    weights = (solution[:n_nodes] + 1.0j * solution[n_nodes:2 * n_nodes]) * gauge
    return (np.asarray(weights, dtype=np.complex128),
            objective / float(objective_scale))


__all__ = [
    "ReciprocalMeasureProblem",
    "ComplexTimeRule",
    "evaluate_rule",
    "delivered_error",
    "rule_amplification",
    "solve_fixed_time_weights",
]
