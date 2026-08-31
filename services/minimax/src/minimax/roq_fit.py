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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import time

import numpy as np
import scipy.linalg as la
from scipy.special import roots_legendre

from minimax.reciprocal_fit import (ReciprocalMeasureProblem, delivered_error,
                                    rule_amplification,
                                    single_core_blas,
                                    solve_fixed_time_weights_fast)

__all__ = [
    "RoqWindow", "RoqGroup", "RoqRule", "RoqBranchEvidence", "RoqPlan",
    "fit_roq_group", "fit_roq_branch", "roq_select_times",
    "branch_delivered_error", "branch_noise_gate",
    "plan_measure_adapted_roq",
]


# These are production policy, not user dials.  The fixed scan contains the
# measured Na optimum (-58 degrees) and its -55-degree near miss.  Wider
# rotations stop at -75 degrees because the contour then grows on common
# valence supports.  Claim 522 is the provenance for this grid.
# Measured on the frozen Na valence support (2026-08-31): node count is FLAT
# from -70 to -90 deg while kappa falls to exactly 1.00 at -85, so the useful
# region is a narrow band near the imaginary axis, not a sweep from 0.  The
# sign-definite limit is 90 deg (a Laplace rule), which is what the certified
# noncrossing tables already are; crossing support wants a few degrees off it.
# The -58 deg the earlier study chose costs one extra node and 18% more
# cancellation than -85 on the same support.  Real-time (0 deg) is retained
# ONLY as the last entry: it is the operating point the shipped fitter used
# and it misses this support by four orders of magnitude, so it must never be
# tried first.
_ANGLE_SCAN_DEG = (-85.0, -80.0, -70.0, -58.0, 0.0)
_ANGLE_PROBE_RANK = 12
_ANGLE_BASE_NODES = 64
_MIN_RANK = 6
_MAX_RANK = 64
_FINAL_BASE_NODES = 192

# The controlled crossing scaling study (claim 528, 2026-08-31) measured
# worst-seed ranks N = 1.464*A - 0.534 at 1e-3 and
# N = 2.024*A - 0.058 at 1e-4, with R^2 = 0.9963/0.9983.  They replace the
# old fixed rank-64 ceiling once a crossing support grows beyond its DFT
# size.  The caller supplies a measured energy-drift margin; it is geometry,
# not a user tolerance or a new planner dial.
_LOOSE_RANK_SLOPE = 1.464
_LOOSE_RANK_INTERCEPT = -0.534
_TIGHT_RANK_SLOPE = 2.024
_TIGHT_RANK_INTERCEPT = -0.058
_RANK_LAW_TARGET_BREAK = 1.0e-3

# The ROQ study used horizons 260/85/27 Ry^-1 for the Na resonant/tail/valence
# groups.  Five inverse low-percentile energy scales reproduces 249/83/31 on
# the frozen measures.  The 0.01% quantile ignores only negligible delivered
# mass; eta remains the hard lower energy scale.
_HORIZON_DECAY_LENGTHS = 5.0
_HORIZON_MASS_QUANTILE = 1.0e-4

_NOISE_FLOOR = 6.0e-8
_NOISE_SHARE = 0.05
_MAX_WORKERS = max(1, min(4, os.cpu_count() or 1))


@dataclass(frozen=True)
class RoqWindow:
    """One product window supplied to the production ROQ planner.

    ``fit`` selects nodes and weights.  ``validation`` is the refined
    lattice used for every acceptance decision.  ``target`` is the window's
    apportioned relative delivered-error budget.  ``branch`` and ``sigma``
    group windows that may share one causal rule.
    """

    name: str
    fit: ReciprocalMeasureProblem
    validation: ReciprocalMeasureProblem
    target: float
    branch: str
    sigma: int


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
    angle_deg: float = 0.0
    horizon: float = 0.0
    windows: tuple[str, ...] = ()
    target_met: bool = True
    noise_passed: bool = True
    search_evaluations: int = 0


@dataclass(frozen=True)
class RoqBranchEvidence:
    """Validation evidence for the rules selected for one causal branch."""

    branch: str
    strategy: str
    target: float
    max_error: float
    kappa_p99: float
    noise_passed: bool
    node_count: int
    window_errors: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class RoqPlan:
    """Selected rules and refined-lattice evidence from one planner call."""

    rules: tuple[RoqRule, ...]
    branches: tuple[RoqBranchEvidence, ...]
    planning_seconds: float
    planning_breakdown: tuple[tuple[str, float], ...] = ()


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


def _prepare_subspace(group: RoqGroup, base_nodes: int, max_rank: int):
    """Build the rank-independent candidate basis once for a rank search."""
    times, gl = _candidates(group, base_nodes)
    singular, basis = _weighted_subspace(group, times, gl, max_rank)
    return times, singular, basis


def _select_prepared(times: np.ndarray, singular: np.ndarray,
                     basis: np.ndarray, rank: int, name: str):
    """Select one QDEIM rank from a prepared candidate basis."""
    rank = int(rank)
    if rank > basis.shape[1]:
        raise ValueError(f"rank {rank} exceeds computed subspace "
                         f"{basis.shape[1]} for group {name!r}")
    pivots = la.qr(basis[:, :rank].T, mode="economic", pivoting=True)[2]
    selected = np.sort(pivots[:rank])
    ratio = float(singular[rank - 1] / singular[0]) if singular[0] > 0 else 0.0
    return times[selected], ratio


def roq_select_times(group: RoqGroup, rank: int, *, base_nodes: int = 0,
                     max_rank: int = 0):
    """QDEIM-selected causal times and the subspace singular ratio."""
    rank = int(rank)
    max_rank = int(max_rank) or max(2 * rank, rank + 8)
    base_nodes = int(base_nodes) or max(4 * max_rank, 96)
    prepared = _prepare_subspace(group, base_nodes, max_rank)
    return _select_prepared(*prepared, rank, group.name)


def _score(group: RoqGroup, times: np.ndarray, weights: np.ndarray, rank: int,
           ratio: float, *, windows: tuple[str, ...] = (),
           target: float = np.inf, evaluations: int = 0) -> RoqRule:
    error, _ = delivered_error(group.validation, times, weights)
    p99, peak = rule_amplification(times, weights, group.validation)
    passed = p99 * _NOISE_FLOOR <= _NOISE_SHARE * float(target)
    return RoqRule(group.name, times, weights, rank, float(np.max(error)),
                   float(p99), float(peak), ratio, group.angle_deg,
                   group.horizon, windows, float(np.max(error)) <= target,
                   passed, evaluations)


def _fit_prepared(group: RoqGroup, prepared, rank: int, *, target: float,
                  windows: tuple[str, ...] = (), quick: bool = False,
                  evaluations: int = 0) -> RoqRule:
    """Fit and validate one rank from an already built snapshot basis."""
    times, ratio = _select_prepared(*prepared, rank, group.name)
    weights, _ = solve_fixed_time_weights_fast(
        group.fit, times,
        iterations=16 if quick else 45,
        stall_iterations=3 if quick else 5,
        conditioning_pass=not quick)
    return _score(group, times, weights, rank, ratio, windows=windows,
                  target=target, evaluations=evaluations)


def fit_roq_group(group: RoqGroup, target: float, *, ranks=None,
                  base_nodes: int = 0,
                  windows: tuple[str, ...] = ()) -> RoqRule:
    """Find the smallest passing rank by bisection on one snapshot basis.

    Search fits use a short deterministic IRLS solve.  The chosen rank is
    refit with the full production settings and accepted only from
    ``group.validation``.  If the final solve misses, larger ranks are tried;
    a complete miss returns the best measured rule with ``target_met=False``.
    """
    ranks = sorted(set(int(rank) for rank in (
        ranks if ranks is not None else range(_MIN_RANK, _MAX_RANK + 1))))
    if not ranks or ranks[0] < 1:
        raise ValueError("ranks must contain positive integers")
    base_nodes = int(base_nodes) or max(4 * max(ranks), 96)
    prepared = _prepare_subspace(group, base_nodes, max(ranks))
    cache = {}

    def probe(index: int) -> RoqRule:
        rank = ranks[index]
        if rank not in cache:
            cache[rank] = _fit_prepared(
                group, prepared, rank, target=target, windows=windows,
                quick=True, evaluations=len(cache) + 1)
        return cache[rank]

    # The error curve is generally decreasing.  Bisection cuts the old
    # 363-prefix scan to O(log N); checking the two lower neighbours catches
    # the small QDEIM pivot non-monotonicity seen in the study.
    lo, hi = -1, len(ranks) - 1
    if not probe(hi).target_met:
        final = _fit_prepared(
            group, prepared, ranks[hi], target=target, windows=windows,
            evaluations=len(cache))
        return final
    while hi - lo > 1:
        mid = (lo + hi) // 2
        rule = probe(mid)
        if rule.target_met:
            hi = mid
        else:
            lo = mid
    first = max(0, hi - 2)
    passing = [i for i in range(first, hi + 1) if probe(i).target_met]
    chosen = min(passing) if passing else hi

    best = None
    for index in range(chosen, len(ranks)):
        final = _fit_prepared(
            group, prepared, ranks[index], target=target, windows=windows,
            evaluations=len(cache))
        if best is None or final.max_error < best.max_error:
            best = final
        if final.target_met and final.noise_passed:
            return final
        # A quick probe that was optimistic normally needs one extra rank;
        # keep the refusal bounded if the full fit reveals a broader miss.
        if index >= chosen + 3:
            break
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


def _merge_problems(windows, which: str) -> ReciprocalMeasureProblem:
    """Concatenate product-window cells without changing their masses."""
    problems = [getattr(window, which) for window in windows]
    first = problems[0]
    for problem in problems[1:]:
        if not np.array_equal(problem.frequencies, first.frequencies):
            raise ValueError("consolidated ROQ windows need identical frequencies")
        if (problem.excluded_radius != first.excluded_radius
                or problem.normalization_floor != first.normalization_floor
                or problem.zero_weight_sum != first.zero_weight_sum):
            raise ValueError(
                "consolidated ROQ windows need identical measure options")
    return ReciprocalMeasureProblem(
        first.frequencies,
        np.concatenate([problem.internal_sums for problem in problems]),
        np.concatenate([problem.cell_masses for problem in problems]),
        excluded_radius=first.excluded_radius,
        normalization_floor=first.normalization_floor,
        zero_weight_sum=first.zero_weight_sum)


def _merged_group(windows, name: str, angle: float, horizon: float) -> RoqGroup:
    sigmas = {window.sigma for window in windows}
    if sigmas - {-1, 1} or len(sigmas) != 1:
        raise ValueError("one ROQ group needs one half-plane sign, +1 or -1")
    return RoqGroup(name, _merge_problems(windows, "fit"),
                    _merge_problems(windows, "validation"), sigmas.pop(),
                    angle, horizon)


def _combined_target(windows, which: str = "validation") -> float:
    """Branch budget implied by the windows' apportioned targets."""
    delivered = []
    for window in windows:
        problem = getattr(window, which)
        _, mass, _ = problem.retained()
        delivered.append(mass)
    delivered = np.asarray(delivered)
    target = np.asarray([window.target for window in windows])[:, None]
    return float(np.max(np.sum(target * delivered, axis=0)
                        / np.sum(delivered, axis=0)))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray,
                       quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, float(quantile) * cumulative[-1])
    return float(values[order[min(index, order.size - 1)]])


def _derive_horizon(problem: ReciprocalMeasureProblem, eta: float) -> float:
    d = problem.denominators
    magnitude = np.abs(d).ravel()
    mass = np.broadcast_to(problem.cell_masses, d.shape).ravel() / magnitude
    scale = _weighted_quantile(magnitude, mass, _HORIZON_MASS_QUANTILE)
    return _HORIZON_DECAY_LENGTHS / max(float(eta), scale)


def _angle_decays(group: RoqGroup, angle: float) -> bool:
    contour = np.exp(1j * np.deg2rad(float(angle)))
    rate = np.imag(group.sigma * contour * group.fit.denominators)
    return bool(np.min(rate) > 0.0)


def _probe_angle(group: RoqGroup, angle: float, target: float,
                 windows: tuple[str, ...]) -> tuple[RoqGroup, RoqRule] | None:
    candidate = RoqGroup(group.name, group.fit, group.validation, group.sigma,
                         angle, group.horizon)
    if not _angle_decays(candidate, angle):
        return None
    prepared = _prepare_subspace(candidate, _ANGLE_BASE_NODES,
                                 _ANGLE_PROBE_RANK)
    times, ratio = _select_prepared(*prepared, _ANGLE_PROBE_RANK,
                                    candidate.name)
    weights, _ = solve_fixed_time_weights_fast(
        candidate.fit, times, iterations=10, stall_iterations=2,
        conditioning_pass=False)
    rule = _score(candidate, times, weights, _ANGLE_PROBE_RANK, ratio,
                  windows=windows, target=target)
    return candidate, rule


def _product_group_seed(windows, eta: float):
    """Build one angle-free group and its derived target and horizon."""
    names = tuple(window.name for window in windows)
    label = "+".join(names)
    target = _combined_target(windows)
    seed = _merged_group(windows, label, 0.0, 1.0)
    horizon = _derive_horizon(seed.fit, eta)
    seed = RoqGroup(seed.name, seed.fit, seed.validation, seed.sigma,
                    0.0, horizon)
    return seed, target, names


def _fit_production_rank(group: RoqGroup, target: float,
                         windows: tuple[str, ...],
                         angle_probe: RoqRule,
                         rank_ceiling: int) -> RoqRule:
    """Bracket from the angle probe, test one midpoint, and fully refit."""
    rank_ceiling = max(_MIN_RANK, int(rank_ceiling))
    base_nodes = max(_FINAL_BASE_NODES, 3 * rank_ceiling)
    prepared = _prepare_subspace(group, base_nodes, rank_ceiling)
    cache = {}

    def quick(rank: int) -> RoqRule:
        rank = int(rank)
        if rank not in cache:
            cache[rank] = _fit_prepared(
                group, prepared, rank, target=target, windows=windows,
                quick=True, evaluations=len(cache) + 1)
        return cache[rank]

    lower = _MIN_RANK
    upper = _ANGLE_PROBE_RANK
    if not angle_probe.target_met or not quick(upper).target_met:
        lower = upper
        for candidate in dict.fromkeys((18, 27, 40, 60, rank_ceiling)):
            if candidate > rank_ceiling:
                continue
            upper = candidate
            if quick(upper).target_met:
                break
            lower = upper
        else:
            return _fit_prepared(
                group, prepared, rank_ceiling, target=target, windows=windows,
                evaluations=len(cache))
    midpoint = (lower + upper) // 2
    if lower < midpoint < upper and quick(midpoint).target_met:
        upper = midpoint

    best = None
    for rank in range(upper, min(rank_ceiling, upper + 3) + 1):
        final = _fit_prepared(
            group, prepared, rank, target=target, windows=windows,
            evaluations=len(cache))
        if best is None or final.max_error < best.max_error:
            best = final
        if final.target_met and final.noise_passed:
            return final
    return best


def _fit_product_groups(subsets, eta: float, rank_ceiling: int):
    """Fit independent product groups through one bounded CPU work pool."""
    records = []
    for key, windows in subsets.items():
        seed, target, names = _product_group_seed(windows, eta)
        records.append((key, seed, target, names))

    jobs = [(key, seed, target, names, angle)
            for key, seed, target, names in records
            for angle in _ANGLE_SCAN_DEG]
    with single_core_blas(), ThreadPoolExecutor(
            max_workers=_MAX_WORKERS) as executor:
        results = list(executor.map(
            lambda job: (job[0], _probe_angle(
                job[1], job[4], job[2], job[3])), jobs))
        probes_by_key = {key: [] for key in subsets}
        for key, probe in results:
            if probe is not None:
                probes_by_key[key].append(probe)

        selected = {}
        for key, probes in probes_by_key.items():
            if not probes:
                label = "+".join(window.name for window in subsets[key])
                raise ValueError(
                    f"group {label!r}: atoms grow at every production "
                    "contour angle")
            selected[key] = min(probes, key=lambda item: (
                item[1].max_error, item[1].kappa_p99,
                _ANGLE_SCAN_DEG.index(item[0].angle_deg)))

        fitted = list(executor.map(
            lambda record: (
                record[0],
                selected[record[0]][0],
                _fit_production_rank(
                    selected[record[0]][0], record[2], record[3],
                    selected[record[0]][1], rank_ceiling)),
            records))
    return {key: (group, rule) for key, group, rule in fitted}


def _fit_product_group(windows, eta: float,
                       rank_ceiling: int = _MAX_RANK
                       ) -> tuple[RoqGroup, RoqRule]:
    """Derive contour and rank for one product group."""
    return _fit_product_groups(
        {(0,): tuple(windows)}, eta, rank_ceiling)[(0,)]


def _try_whole_below(windows, eta: float, node_cap: int, rank_ceiling: int):
    """Test a whole-branch rule only where it can beat the fallback."""
    seed, target, names = _product_group_seed(windows, eta)
    angles = tuple(angle for angle in _ANGLE_SCAN_DEG if angle < 0.0
                   and _angle_decays(seed, angle))
    if not angles:
        return None
    with single_core_blas(), ThreadPoolExecutor(
            max_workers=_MAX_WORKERS) as executor:
        probes = list(executor.map(
            lambda angle: _probe_angle(seed, angle, target, names),
            angles))
    probes = [probe for probe in probes if probe is not None]
    if not probes:
        return None
    group, _ = min(probes, key=lambda item: (
        item[1].max_error, item[1].kappa_p99,
        _ANGLE_SCAN_DEG.index(item[0].angle_deg)))
    cap = max(_MIN_RANK, min(int(node_cap), int(rank_ceiling)))
    base_nodes = max(_ANGLE_BASE_NODES, 3 * cap)
    prepared = _prepare_subspace(group, base_nodes, cap)
    cap_rule = _fit_prepared(group, prepared, cap, target=target,
                             windows=names)
    if not (cap_rule.target_met and cap_rule.noise_passed):
        return group, cap_rule
    rule = fit_roq_group(group, target, ranks=range(_MIN_RANK, cap + 1),
                         base_nodes=base_nodes, windows=names)
    return group, rule


def _decay_partition(windows) -> tuple[tuple[int, ...], ...]:
    """Group windows that can share at least one rotated decaying contour."""
    signatures = {True: [], False: []}
    for index, window in enumerate(windows):
        group = _merged_group([window], window.name, 0.0, 1.0)
        valid = tuple(angle for angle in _ANGLE_SCAN_DEG
                      if _angle_decays(group, angle))
        signatures[any(angle < 0.0 for angle in valid)].append(index)
    return tuple(tuple(indices) for indices in signatures.values() if indices)


def _branch_evidence(branch: str, strategy: str, windows, groups,
                     rules) -> RoqBranchEvidence | None:
    target = _combined_target(windows)
    error = branch_delivered_error(groups, rules)
    passed, kappa = branch_noise_gate(groups, rules, target)
    window_errors = []
    for window in windows:
        rule = next(rule for rule in rules if window.name in rule.windows)
        value, _ = delivered_error(window.validation, rule.times, rule.weights)
        window_errors.append((window.name, float(np.max(value))))
    evidence = RoqBranchEvidence(
        branch, strategy, target, float(np.max(error)), kappa,
        bool(passed and all(rule.noise_passed for rule in rules)),
        int(sum(rule.rank for rule in rules)), tuple(window_errors))
    if evidence.max_error > target or not evidence.noise_passed:
        return None
    return evidence


def _planned_rank_ceiling(windows, support_margin: float,
                          max_nodes: int) -> int:
    """Size the ROQ basis from measured support plus measured drift.

    ``support_margin`` has the same energy unit as each denominator.  The
    dimensionless planning radius is ``(max|Re d| + margin) / min|Im d|``.
    The production scaling laws above turn that radius into a rank ceiling;
    ``max_nodes`` remains only the caller's resource certificate.
    """
    estimates = []
    for window in windows:
        denominator = np.asarray(window.validation.denominators)
        gamma_min = float(np.min(np.abs(denominator.imag)))
        if not gamma_min > 0.0:
            raise ValueError(
                f"window {window.name!r} has no damped support")
        A_dim = ((float(np.max(np.abs(denominator.real)))
                  + float(support_margin)) / gamma_min)
        if float(window.target) <= _RANK_LAW_TARGET_BREAK:
            slope, intercept = _TIGHT_RANK_SLOPE, _TIGHT_RANK_INTERCEPT
        else:
            slope, intercept = _LOOSE_RANK_SLOPE, _LOOSE_RANK_INTERCEPT
        estimates.append(int(np.ceil(slope * A_dim + intercept)))
    return max(_MIN_RANK, min(int(max_nodes), max(_MAX_RANK, *estimates)))


def plan_measure_adapted_roq(windows, eta: float, *,
                             support_margin: float = 0.0,
                             max_nodes: int = _MAX_RANK) -> RoqPlan:
    """Build the smallest validated product-window ROQ plan.

    Parameters
    ----------
    windows
        Per-window fit and refined-validation measures, apportioned targets,
        branch labels, and half-plane signs as :class:`RoqWindow` objects.
    eta
        Physical broadening in the same energy unit as the measures.  It is
        the only contour-scale input; angle, horizon, and rank are derived.
    support_margin
        Energy displacement measured by the self-consistent caller after its
        first map.  It sizes the basis for support drift but never changes the
        fit or validation measure.  Zero preserves one-shot planning.
    max_nodes
        Resource ceiling supplied by the caller, not an accuracy dial.

    Returns
    -------
    RoqPlan
        Rules and achieved refined-lattice evidence.  Whole-branch rules are
        tried first.  A decay-compatible product-window partition and then
        individual product windows are fallbacks; no explicit ``(n,p)`` pair
        evaluator exists.  If none passes, this function refuses.
    """
    started = time.perf_counter()
    windows = tuple(windows)
    if not windows:
        raise ValueError("ROQ planning needs at least one product window")
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be positive and finite")
    if not np.isfinite(support_margin) or support_margin < 0.0:
        raise ValueError("support_margin must be finite and nonnegative")
    if int(max_nodes) < 1:
        raise ValueError("max_nodes must be positive")
    for window in windows:
        if not np.isfinite(window.target) or window.target <= 0.0:
            raise ValueError(f"window {window.name!r} needs a positive target")

    rank_ceiling = _planned_rank_ceiling(
        windows, float(support_margin), int(max_nodes))
    by_branch = {}
    for window in windows:
        by_branch.setdefault(window.branch, []).append(window)

    partition_started = time.perf_counter()
    branch_data = {}
    initial_subsets = {}
    for branch, branch_windows_list in by_branch.items():
        branch_windows = tuple(branch_windows_list)
        all_indices = tuple(range(len(branch_windows)))
        partition = _decay_partition(branch_windows)
        strategy = ("whole_branch" if partition == (all_indices,)
                    else "decay_compatible")
        for indices in partition:
            key = (branch, indices)
            initial_subsets[key] = tuple(
                branch_windows[i] for i in indices)
        branch_data[branch] = (branch_windows, all_indices, strategy,
                               partition)

    partition_seconds = time.perf_counter() - partition_started
    fit_started = time.perf_counter()
    group_cache = _fit_product_groups(initial_subsets, eta, rank_ceiling)
    fit_seconds = time.perf_counter() - fit_started
    selected_rules = []
    evidence_rows = []
    consolidation_seconds = 0.0
    fallback_seconds = 0.0
    for branch, data in branch_data.items():
        branch_windows, all_indices, strategy, partition = data
        candidates = []
        groups = [group_cache[(branch, indices)][0]
                  for indices in partition]
        rules = [group_cache[(branch, indices)][1]
                 for indices in partition]
        row = _branch_evidence(branch, strategy, branch_windows,
                               groups, rules)
        if row is not None:
            candidates.append((row, tuple(rules)))

        if partition != (all_indices,):
            cap = (row.node_count - 1) if row is not None else rank_ceiling
            consolidation_started = time.perf_counter()
            whole = _try_whole_below(
                branch_windows, eta, cap, rank_ceiling)
            consolidation_seconds += (
                time.perf_counter() - consolidation_started)
            if whole is not None:
                whole_group, whole_rule = whole
                whole_row = _branch_evidence(
                    branch, "whole_branch", branch_windows,
                    [whole_group], [whole_rule])
                if whole_row is not None:
                    candidates.append((whole_row, (whole_rule,)))

        # Individual product windows are a refusal fallback, not prepaid
        # alternatives.  Consolidated winners are already no less accurate;
        # fitting every legacy window made the first Na prototype 61 s.
        if not candidates:
            partition = tuple((i,) for i in range(len(branch_windows)))
            missing = {
                (branch, indices): tuple(branch_windows[i] for i in indices)
                for indices in partition
                if (branch, indices) not in group_cache}
            if missing:
                fallback_started = time.perf_counter()
                group_cache.update(_fit_product_groups(
                    missing, eta, rank_ceiling))
                fallback_seconds += time.perf_counter() - fallback_started
            groups = [group_cache[(branch, indices)][0]
                      for indices in partition]
            rules = [group_cache[(branch, indices)][1]
                     for indices in partition]
            row = _branch_evidence(branch, "per_window", branch_windows,
                                   groups, rules)
            if row is not None:
                candidates.append((row, tuple(rules)))
        if not candidates:
            attempts = "; ".join(
                f"{rule.group}: rank={rule.rank}/{rank_ceiling}, "
                f"residual={rule.max_error:.6g}, "
                f"amplification_p99={rule.kappa_p99:.6g}"
                for rule in rules)
            raise RuntimeError(
                f"branch {branch!r}: no product-window ROQ plan meets the "
                "refined delivered-error and noise gates; " + attempts)
        row, rules = min(candidates, key=lambda item: (
            item[0].node_count, item[0].max_error, item[0].strategy))
        selected_rules.extend(rules)
        evidence_rows.append(row)

    total = time.perf_counter() - started
    measured = (partition_seconds + fit_seconds + consolidation_seconds
                + fallback_seconds)
    breakdown = (
        ("partition", partition_seconds),
        ("angle_and_rank_fits", fit_seconds),
        ("whole_branch_challenge", consolidation_seconds),
        ("per_window_fallback", fallback_seconds),
        ("selection_and_scoring", max(0.0, total - measured)),
    )
    return RoqPlan(tuple(selected_rules), tuple(evidence_rows), total,
                   breakdown)
