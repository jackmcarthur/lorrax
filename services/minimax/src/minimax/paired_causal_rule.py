"""Paired causal real-time minimax rules for a broadened reciprocal.

This is the eta-preserving counterpart of an odd-only sine fit.  A rule has
real nonnegative times and real (not necessarily positive) coefficients::

    Q(x + i beta) = -1j * sum(h * exp(1j * t * (x + 1j * beta))).

The real and imaginary parts are therefore a sine/cosine Hilbert pair.  The
service fits the complete reciprocal, never an independently regularized odd
part.  It returns *undamped* weights ``-1j*h``: a GW adapter can apply its
external eta once while W(t) supplies any additional pole damping.

Only scalar offline fitting lives here.  There are no wavefunctions, spectral
weights, JAX arrays, or material-dependent objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.linalg import lstsq, qr
from scipy.special import roots_legendre

from minimax.uniform_rule import box_samples


Array = np.ndarray


@dataclass(frozen=True)
class PairedCausalRule:
    """One fixed-phase causal rule and its independent box diagnostics."""

    times: Array
    weights: Array
    coefficients: Array
    box: tuple[float, float, float, float]
    requested_rank: int
    numerical_rank: int
    candidate_count: int
    horizon_beta: float
    sup_error: float
    kappa_max: float
    scaled_condition: float
    seconds: float

    @property
    def node_count(self) -> int:
        return int(self.times.size)

    def one_line(self) -> str:
        return (
            f"paired causal rule: {self.node_count} nodes, "
            f"T*beta_min={self.horizon_beta:g}, sup {self.sup_error:.2e}, "
            f"kappa {self.kappa_max:.3g}, cond {self.scaled_condition:.3g}"
        )


def evaluate_paired_rule(times: Array, coefficients: Array,
                         denominators: Array) -> Array:
    """Evaluate ``-i sum h_j exp(i t_j d)`` without eta duplication."""

    d = np.asarray(denominators, dtype=np.complex128)
    t = np.asarray(times, dtype=np.float64)
    h = np.asarray(coefficients, dtype=np.float64)
    return -1.0j * np.exp(1.0j * d[..., None] * t) @ h


def _heldout_box_samples(box: tuple[float, float, float, float]) -> Array:
    """A dense midpoint cloud disjoint from the public fit cloud.

    Midpoints in one coordinate paired with edge/midpoint values of the other
    retain all four boundary directions without repeating any fit point.
    """

    re_lo, re_hi, im_lo, im_hi = box
    fit_like = box_samples(
        re_lo, re_hi, im_lo, im_hi, per_unit=8.0, n_im=12)
    levels = np.unique(fit_like.imag)
    if levels.size == 1:
        beta_mid = levels
    else:
        beta_mid = np.sqrt(levels[:-1] * levels[1:])
    rows = []
    for beta in np.r_[levels[[0, -1]], beta_mid]:
        near = levels[np.argmin(np.abs(levels - beta))]
        line = np.sort(np.unique(fit_like[fit_like.imag == near].real))
        mids = 0.5 * (line[:-1] + line[1:])
        if mids.size:
            rows.append(mids + 1.0j * beta)
    # Sample the real-support edges at imaginary midpoints.  Those points are
    # disjoint from the fit cloud because its imaginary levels are endpoints
    # or powers on the unshifted logarithmic grid.
    if beta_mid.size:
        rows.append(re_lo + 1.0j * beta_mid)
        rows.append(re_hi + 1.0j * beta_mid)
    out = np.unique(np.concatenate(rows))
    return np.asarray(out, dtype=np.complex128)


def _currency(denominators: Array, relative: bool, beta_min: float) -> Array:
    return (np.abs(denominators) if relative
            else np.full(np.asarray(denominators).shape, float(beta_min)))


def _stack(matrix: Array, target: Array) -> tuple[Array, Array]:
    return (np.concatenate((matrix.real, matrix.imag), axis=0),
            np.concatenate((target.real, target.imag), axis=0))


def _lawson_real(matrix: Array, target: Array, iterations: int,
                 rcond: float) -> tuple[Array, int, float]:
    """Complex minimax-oriented refit with real coefficients and no AtA."""

    row_weight = np.ones(target.size, dtype=np.float64)
    coefficient = np.zeros(matrix.shape[1], dtype=np.float64)
    numerical_rank = 0
    scaled_condition = np.inf
    for _ in range(max(1, int(iterations))):
        root = np.sqrt(row_weight)
        weighted = matrix * root[:, None]
        rhs = target * root
        stacked, stacked_rhs = _stack(weighted, rhs)
        column_norm = np.linalg.norm(stacked, axis=0)
        live = column_norm > np.finfo(float).tiny
        if not np.any(live):
            break
        normalized = stacked[:, live] / column_norm[live]
        solution, _residues, numerical_rank, singular = lstsq(
            normalized, stacked_rhs, cond=float(rcond), lapack_driver="gelsd")
        coefficient.fill(0.0)
        coefficient[live] = solution / column_norm[live]
        if singular.size and numerical_rank:
            scaled_condition = float(
                singular[0] / max(singular[numerical_rank - 1],
                                  np.finfo(float).tiny))
        residual = np.abs(target - matrix @ coefficient)
        floor = max(float(np.max(residual)) * 1.0e-3, 1.0e-30)
        row_weight *= np.maximum(residual, floor)
        row_weight /= max(float(np.mean(row_weight)), np.finfo(float).tiny)
        row_weight = np.minimum(row_weight, 1.0e12)
    return coefficient, int(numerical_rank), scaled_condition


def build_paired_causal_rule_ladder(
    box,
    ranks,
    *,
    horizon_beta: float,
    candidate_factor: int = 3,
    lawson_iterations: int = 8,
    rcond: float = 1.0e-13,
) -> tuple[PairedCausalRule, ...]:
    """Fit a nested rank ladder for ``1/d`` on an upper-half-plane box.

    The box and error currencies match :func:`build_uniform_rule`.  A single
    pivoted QR determines a deterministic nested time order; every requested
    rank then receives an independent column-scaled SVD Lawson refit and is
    judged on a denser midpoint cloud that was not used by the fit.
    """

    started = time.perf_counter()
    values = tuple(map(float, box))
    re_lo, re_hi, im_lo, im_hi = values
    if not (np.isfinite(values).all() and re_lo <= re_hi
            and 0.0 < im_lo <= im_hi):
        raise ValueError(f"invalid support box {box!r}")
    requested = tuple(sorted(set(int(rank) for rank in ranks)))
    if not requested or requested[0] < 2:
        raise ValueError("ranks must contain integers >=2")
    hb = float(horizon_beta)
    if not np.isfinite(hb) or hb <= 0.0:
        raise ValueError("horizon_beta must be finite and positive")
    factor = int(candidate_factor)
    if factor < 2:
        raise ValueError("candidate_factor must be at least two")

    horizon = hb / im_lo
    candidate_count = max(96, factor * requested[-1])
    xg, _wg = roots_legendre(candidate_count)
    candidates = 0.5 * horizon * (xg + 1.0)
    train = box_samples(re_lo, re_hi, im_lo, im_hi)
    relative = re_lo > 0.0 or re_hi < 0.0
    rho = _currency(train, relative, im_lo)
    atoms = -1.0j * np.exp(1.0j * train[:, None] * candidates[None, :])
    target = 1.0 / train
    scaled_atoms = rho[:, None] * atoms
    scaled_target = rho * target
    stacked_atoms, _ = _stack(scaled_atoms, scaled_target)
    column_norm = np.linalg.norm(stacked_atoms, axis=0)
    normalized = stacked_atoms / np.maximum(
        column_norm, np.finfo(float).tiny)[None, :]
    pivot = qr(normalized, mode="economic", pivoting=True)[2]

    check = _heldout_box_samples(values)
    rho_check = _currency(check, relative, im_lo)
    rules = []
    for rank in requested:
        indices = np.sort(pivot[:rank])
        times = candidates[indices]
        matrix = scaled_atoms[:, indices]
        coefficient, numerical_rank, condition = _lawson_real(
            matrix, scaled_target, int(lawson_iterations), float(rcond))
        model = evaluate_paired_rule(times, coefficient, check)
        error = np.abs(rho_check * (model - 1.0 / check))
        term_mass = rho_check * (
            np.abs(np.exp(1.0j * check[:, None] * times[None, :]))
            @ np.abs(coefficient))
        rules.append(PairedCausalRule(
            times=np.asarray(times, dtype=np.float64),
            weights=np.asarray(-1.0j * coefficient, dtype=np.complex128),
            coefficients=np.asarray(coefficient, dtype=np.float64),
            box=values,
            requested_rank=rank,
            numerical_rank=numerical_rank,
            candidate_count=candidate_count,
            horizon_beta=hb,
            sup_error=float(np.max(error)),
            kappa_max=float(np.max(term_mass)),
            scaled_condition=float(condition),
            seconds=time.perf_counter() - started,
        ))
    return tuple(rules)


def build_paired_causal_rule(box, rank: int, *, horizon_beta: float,
                             **kwargs) -> PairedCausalRule:
    """Single-rank convenience wrapper for the nested builder."""

    return build_paired_causal_rule_ladder(
        box, (rank,), horizon_beta=horizon_beta, **kwargs)[0]


__all__ = [
    "PairedCausalRule", "build_paired_causal_rule",
    "build_paired_causal_rule_ladder", "evaluate_paired_rule",
]
