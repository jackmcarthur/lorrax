"""Measure-weighted reduced-order quadrature (ROQ) node discovery.

A deterministic alternative to the candidate-tolerance ladder: node
discovery by linear algebra on the actual delivered measure.

For one causal support group with half-plane sign ``sigma`` and contour
``c = exp(i*angle)``, the exact representation

    1/d = -i*sigma*c * integral_0^inf exp(i*sigma*c*u*d) du

is overresolved with Gauss-Legendre nodes ``u_l`` on ``[0, horizon]``.
The snapshot family ``exp(i*sigma*c*u_l*d)`` on the group's fit cells,
weighted by delivered mass and per-frequency normalization, is reduced by
SVD; QDEIM (pivoted QR on the right singular basis) selects physical time
columns; universal complex weights come from the production IRLS solver;
scoring uses only the production delivered metrics.  Unlike a uniform
Hankel lift, the measure's mass anisotropy is inside the inner product,
so node selection sees it (measured on Na: branch error <= 1e-5 with 69
nodes for cond 39+18 / val 12 at angles 0/-65/-58 deg, vs 137 production
window-tau pairs; provenance: sandbox claim 512 lineage, ROQ study).

The snapshot reduction accumulates the candidate Gram per frequency; its
squared conditioning only blurs the candidate subspace, never the final
answer — the refit and all reported errors are computed downstream on the
exact cells.  Everything here is deterministic: rerunning a fit
reproduces times and weights bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
from scipy.special import roots_legendre

from minimax.reciprocal_fit import (ReciprocalMeasureProblem, delivered_error,
                                    rule_amplification,
                                    solve_fixed_time_weights_fast)

__all__ = [
    "RoqGroup", "RoqRule", "fit_roq_group", "fit_roq_branch",
    "roq_select_times", "branch_delivered_error", "branch_noise_gate",
]


@dataclass(frozen=True)
class RoqGroup:
    """One causal support group (e.g. cond resonant, cond tails, val all)."""

    name: str
    fit: ReciprocalMeasureProblem
    validation: ReciprocalMeasureProblem
    sigma: int
    angle_deg: float = 0.0
    horizon: float = 0.0  # contour length in inverse caller-energy units

    def contour(self) -> complex:
        return complex(np.exp(1j * np.deg2rad(self.angle_deg)))


@dataclass(frozen=True)
class RoqRule:
    """A fitted group rule with its validation metrics."""

    group: str
    times: np.ndarray
    weights: np.ndarray
    rank: int
    max_error: float
    kappa_p99: float
    kappa_max: float
    singular_ratio: float


def _candidates(group: RoqGroup, base_nodes: int):
    if not group.horizon > 0.0:
        raise ValueError(f"group {group.name!r} needs a positive horizon")
    x, q = roots_legendre(int(base_nodes))
    u = 0.5 * group.horizon * (x + 1.0)
    return group.sigma * group.contour() * u, 0.5 * group.horizon * q


def _weighted_subspace(group: RoqGroup, times: np.ndarray, gl: np.ndarray,
                       max_rank: int):
    """Right singular basis of the mass/frequency-weighted snapshot family."""
    problem = group.fit
    d = problem.denominators
    mass = problem.cell_masses
    growth = float(np.max(-np.imag(group.sigma * group.contour() * d)))
    if growth * group.horizon > 60.0:
        raise ValueError(
            f"group {group.name!r}: atoms grow along the contour (peak "
            f"exponent {growth * group.horizon:.1f}); the half-plane sign or "
            f"contour angle does not decay on this support")
    delivered = (1.0 / np.abs(d)) @ mass
    row_weight = (mass[None, :] / delivered[:, None]) / d.shape[0]
    n = times.size
    gram = np.zeros((n, n), dtype=np.complex128)
    root_gl = np.sqrt(gl)
    for k in range(d.shape[0]):
        snap = np.exp(1j * d[k, :, None] * times[None, :])
        snap *= np.sqrt(row_weight[k])[:, None]
        snap *= root_gl[None, :]
        gram += snap.conj().T @ snap
    lo = max(0, n - int(max_rank))
    values, vectors = la.eigh(gram, subset_by_index=[lo, n - 1])
    order = np.argsort(values)[::-1]
    singular = np.sqrt(np.maximum(values[order], 0.0))
    return singular, vectors[:, order]


def roq_select_times(group: RoqGroup, rank: int, *, base_nodes: int = 0,
                     max_rank: int = 0):
    """QDEIM-selected causal times and the subspace singular ratio."""
    rank = int(rank)
    max_rank = int(max_rank) or max(2 * rank, rank + 8)
    base_nodes = int(base_nodes) or max(4 * max_rank, 96)
    times, gl = _candidates(group, base_nodes)
    singular, basis = _weighted_subspace(group, times, gl, max_rank)
    if rank > basis.shape[1]:
        raise ValueError(f"rank {rank} exceeds computed subspace "
                         f"{basis.shape[1]} for group {group.name!r}")
    pivots = la.qr(basis[:, :rank].T, mode="economic", pivoting=True)[2]
    selected = np.sort(pivots[:rank])
    ratio = float(singular[rank - 1] / singular[0]) if singular[0] > 0 else 0.0
    return times[selected], ratio


def _score(group: RoqGroup, times: np.ndarray, weights: np.ndarray, rank: int,
           ratio: float) -> RoqRule:
    error, _ = delivered_error(group.validation, times, weights)
    p99, peak = rule_amplification(times, weights, group.validation)
    return RoqRule(group.name, times, weights, rank, float(np.max(error)),
                   float(p99), float(peak), ratio)


def fit_roq_group(group: RoqGroup, target: float, *, ranks=None,
                  base_nodes: int = 0) -> RoqRule:
    """First rank whose IRLS-refit rule meets ``target`` on validation.

    Returns the best rule found even on a miss — check ``max_error``.
    """
    ranks = list(ranks) if ranks is not None else list(range(6, 65, 2))
    best = None
    for rank in ranks:
        times, ratio = roq_select_times(group, rank, base_nodes=base_nodes,
                                        max_rank=max(ranks))
        weights, _ = solve_fixed_time_weights_fast(
            group.fit, times, iterations=55, stall_iterations=6)
        rule = _score(group, times, weights, rank, ratio)
        if best is None or rule.max_error < best.max_error:
            best = rule
        if rule.max_error <= float(target):
            return rule
    return best


def branch_delivered_error(groups, rules, *, which: str = "validation"):
    """Combined branch delivered error by frequency over several groups."""
    numerator = denominator = 0.0
    for group, rule in zip(groups, rules):
        problem = getattr(group, which)
        d = problem.denominators
        q = np.exp(1j * d[..., None] * rule.times[None, None, :]) @ rule.weights
        numerator = numerator + np.abs(q - 1.0 / d) @ problem.cell_masses
        denominator = denominator + (1.0 / np.abs(d)) @ problem.cell_masses
    return numerator / denominator


def branch_noise_gate(groups, rules, target: float, *,
                      noise: float = 6.0e-8, safety: float = 0.05):
    """Mass-share-aggregated amplification test at the branch level."""
    shares = np.array([(1.0 / np.abs(g.fit.denominators))
                       @ g.fit.cell_masses for g in groups]).sum(axis=1)
    shares = shares / shares.sum()
    effective = float(sum(s * r.kappa_p99 for s, r in zip(shares, rules)))
    return effective * noise <= safety * float(target), effective


def fit_roq_branch(groups, start_rules, *, iterations: int = 60,
                   stall_iterations: int = 6):
    """Jointly refit group weights against the combined branch error.

    Node families stay per group (take ``start_rules`` from
    ``fit_roq_group``).  One Lawson multiplier per frequency couples the
    groups; each iteration's weighted least squares then decouples per
    group because the groups partition the branch cells.  Returns
    ``(rules, branch_error_by_frequency)`` scored on validation.
    """
    n_freq = groups[0].fit.denominators.shape[0]
    branch_mass = np.zeros(n_freq)
    packs = []
    for group, rule in zip(groups, start_rules):
        d2 = group.fit.denominators
        mass = group.fit.cell_masses
        branch_mass += (1.0 / np.abs(d2)) @ mass
        d = d2.reshape(-1)
        freq = np.repeat(np.arange(n_freq), d2.shape[1])
        cell_mass = np.tile(mass, n_freq)
        exponent = 1j * d[:, None] * rule.times[None, :]
        peak = np.max(exponent.real, axis=0)
        packs.append((d, freq, cell_mass, np.exp(exponent - peak[None, :]),
                      np.exp(-peak)))
    coeff = [rule.weights / pack[4]
             for rule, pack in zip(start_rules, packs)]
    residual = [pack[3] @ y - 1.0 / pack[0]
                for pack, y in zip(packs, coeff)]

    def branch_by_frequency(residuals):
        total = np.zeros(n_freq)
        for (d, freq, cell_mass, _, _), r in zip(packs, residuals):
            total += np.bincount(
                freq, weights=cell_mass / branch_mass[freq] * np.abs(r),
                minlength=n_freq)
        return total

    lawson = np.full(n_freq, 1.0 / n_freq)
    best_error, best_coeff, stall = np.inf, coeff, 0
    for _ in range(int(iterations)):
        by = branch_by_frequency(residual)
        lawson *= by / by.max() + 1.0e-12
        lawson /= lawson.sum()
        new_coeff, new_residual = [], []
        for (d, freq, cell_mass, atoms, _), res in zip(packs, residual):
            floor = (1.0e-12 * np.max(np.abs(1.0 / d))
                     + 1.0e-6 * np.mean(np.abs(res)))
            row = lawson[freq] * (cell_mass / branch_mass[freq])
            row /= np.maximum(np.abs(res), floor)
            root = np.sqrt(row)
            y = np.linalg.lstsq(root[:, None] * atoms, root * (1.0 / d),
                                rcond=None)[0]
            new_coeff.append(y)
            new_residual.append(atoms @ y - 1.0 / d)
        coeff, residual = new_coeff, new_residual
        error = float(branch_by_frequency(residual).max())
        if error < best_error * (1.0 - 1.0e-3):
            stall = 0
        else:
            stall += 1
        if error < best_error:
            best_error, best_coeff = error, [y.copy() for y in coeff]
        if stall >= int(stall_iterations):
            break
    rules = [_score(group, rule.times, y * pack[4], rule.rank,
                    rule.singular_ratio)
             for group, rule, pack, y in zip(groups, start_rules, packs,
                                             best_coeff)]
    return rules, branch_delivered_error(groups, rules)
