r"""Support-adaptive search for complex time nodes fitting ``1/d``.

This module grows, exchanges, and prunes the time nodes of a
``reciprocal_fit`` rule.  It knows only complex points, masses, candidate
nodes, and errors; branches, occupations, and pole models stay with the
caller.

The search is deterministic by construction: the candidate dictionary is
generated on fixed ladders and sorted, every tie carries a declared
tiebreak (score, then family label, then node magnitude — the second and
third keys exist only so equal scores resolve identically on every
machine), and there is no random state anywhere.

A support whose kept mass lies inside one quarter-plane sector is
answered by the certified rotated sinc rule of ``minimax.sector`` — an
analytic bound, no optimizer.  Everything else is fitted and validated on
the supplied cells; those numbers are numerical evidence, not a continuum
certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minimax.reciprocal_fit import (
    ComplexTimeRule,
    ReciprocalMeasureProblem,
    delivered_error,
    evaluate_rule,
    rule_amplification,
    solve_fixed_time_weights,
)
from minimax.sector import reciprocal_sector_rule

Array = np.ndarray


@dataclass(frozen=True)
class ComplexTimeSearchOptions:
    """Search request for ``fit_reciprocal_measure``.

    ``target_error`` is the delivered branch error to reach at every
    requested frequency; ``max_nodes`` is a refusal ceiling, not a goal.
    ``growth_cap`` bounds the log magnitude any candidate atom may reach
    on the kept support, keeping crossing-capable nodes representable in
    the runtime dtype.  ``fit_frequency_count`` bounds how many
    frequencies enter one linear program during the search; the exact
    metric is always re-measured on all of them and the worst offenders
    are exchanged in.
    """

    target_error: float
    max_nodes: int = 512
    polygon_directions: int = 16
    candidate_times_per_octave: int = 8
    candidate_angle_count: int = 7
    candidates_per_round: int = 1
    exchange_rounds: int = 8
    conditioning_slack: float = 1.0e-3
    growth_cap: float = 30.0
    fit_frequency_count: int = 8
    prune_trials_per_round: int = 6
    # The certified sector rule is simple and carries an analytic bound,
    # but it pays for the whole radial annulus whether or not the measure
    # occupies it; disable the shortcut to let the measured fit compete on
    # sign-definite supports too.
    sector_shortcut: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < float(self.target_error) < 1.0:
            raise ValueError("target_error must lie in (0,1)")
        if int(self.max_nodes) < 1:
            raise ValueError("max_nodes must be positive")
        if int(self.candidate_angle_count) < 1 or int(self.candidates_per_round) < 1:
            raise ValueError("candidate counts must be positive")


def support_arc(problem: ReciprocalMeasureProblem) -> tuple[float, float, float, float]:
    """Radial and angular extent of the kept denominator support.

    Returns ``(radial_min, radial_max, arc_center, arc_width)`` over every
    kept cell at every requested frequency, using unfloored magnitudes —
    eligibility for a sign-definite family is decided on the analytic
    support, never manufactured by a numerical floor.  The arc is the
    smallest angle interval covering all kept denominators.
    """
    kept, _, _ = problem.retained()
    live = kept & (problem.cell_masses[None, :] > 0.0)
    d = problem.denominators[live]
    if d.size == 0:
        raise ValueError("no cell survives the excluded region")
    magnitude = np.abs(d)
    angles = np.sort(np.angle(d))
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2.0 * np.pi]))
    widest = int(np.argmax(gaps))
    width = 2.0 * np.pi - float(gaps[widest])
    start = angles[(widest + 1) % angles.size]
    center = float(np.angle(np.exp(1.0j * (start + 0.5 * width))))
    return float(np.min(magnitude)), float(np.max(magnitude)), center, width


def _sector_rule_rotated(problem: ReciprocalMeasureProblem,
                         options: ComplexTimeSearchOptions) -> ComplexTimeRule | None:
    """Certified sector rule when the whole kept support allows one.

    The shipped rule covers ``-pi/2 <= arg(z) <= 0``; a support arc no
    wider than ``pi/2`` is rotated onto that sector exactly:
    ``1/d = exp(1j*alpha) * (1/z)`` at ``z = exp(1j*alpha) * d``, so the
    returned nodes are ``t = 1j * c * exp(1j*alpha) * s`` and the rotation
    folds into the weights.  Its pointwise relative bound dominates the
    delivered error, so the analytic bound transfers.
    """
    radial_min, radial_max, center, width = support_arc(problem)
    if width > 0.5 * np.pi or problem.zero_weight_sum:
        return None
    alpha = -0.25 * np.pi - center  # rotate the arc center onto -pi/4
    try:
        rule = reciprocal_sector_rule(
            radial_min, radial_max,
            relative_error=float(options.target_error),
            max_nodes=int(options.max_nodes))
    except ValueError:
        # The certified family refuses its node ceiling here; the measured
        # fit may still answer within it, so the caller falls through.
        return None
    turn = np.exp(1.0j * alpha)
    times = 1.0j * rule.contour * turn * rule.times
    weights = turn * rule.weights
    error, excluded = delivered_error(problem, times, weights)
    if float(np.max(error)) > float(options.target_error) * (1.0 + 1.0e-9):
        raise ArithmeticError(
            "constructed sector rule exceeds its analytic bound on the support")
    p99, peak = rule_amplification(times, weights, problem)
    return ComplexTimeRule(
        time_nodes=np.asarray(times, dtype=np.complex128),
        weights=np.asarray(weights, dtype=np.complex128),
        node_families=("sector",) * int(times.size),
        delivered_error_by_frequency=error,
        sampled_max_error=float(np.max(error)),
        excluded_mass_fraction=float(np.max(excluded)),
        amplification=p99,
        amplification_max=peak,
        method="sector",
        error_bound=float(rule.error_bound),
    )


def candidate_time_dictionary(
        problem: ReciprocalMeasureProblem,
        options: ComplexTimeSearchOptions) -> tuple[Array, tuple[str, ...]]:
    """Deterministic candidate complex times for the measured fit.

    Three families, all filtered by the growth cap on the kept support:

    - ``ray``: log-spaced magnitudes along angles spanning the decaying
      directions of the support (the rotated-Laplace continuum, including
      every intermediate contour angle);
    - ``oscillatory``: real times of both signs, resolving real-axis
      structure down to the smallest kept denominator (the crossing
      family);
    - ``short_time``: a short ladder near ``1/radial_max`` for isolated
      high-energy weight.

    Magnitude ladders carry ``candidate_times_per_octave`` points per
    octave.  The dictionary is deduplicated and sorted by
    ``(family, magnitude, angle)`` so downstream ties are reproducible.
    """
    kept, _, _ = problem.retained()
    live = kept & (problem.cell_masses[None, :] > 0.0)
    d = problem.denominators[live]
    radial_min, radial_max, center, width = support_arc(problem)

    per_octave = int(options.candidate_times_per_octave)
    span = 4.0 * radial_max / radial_min
    count = max(2, int(np.ceil(np.log2(span) * per_octave)))
    magnitudes = np.geomspace(0.25 / radial_max, 4.0 / radial_min, count)

    # Decaying directions: an atom exp(1j*t*d) with t = m*exp(1j*chi)
    # decays on d = r*exp(1j*theta) when sin(theta + chi) > 0, so the
    # angle window shared by the whole arc is (-theta_lo, pi - theta_hi).
    theta_lo = center - 0.5 * width
    theta_hi = center + 0.5 * width
    chi_lo, chi_hi = -theta_lo, np.pi - theta_hi
    angle_count = int(options.candidate_angle_count)
    if chi_hi > chi_lo:
        margin = 0.02 * (chi_hi - chi_lo)
        ray_angles = np.linspace(chi_lo + margin, chi_hi - margin, angle_count)
        short_angle = 0.5 * (chi_lo + chi_hi)
    else:
        # Crossing support with no universally decaying direction: search
        # around the two real-time directions; the growth cap filters what
        # the support cannot represent.
        ray_angles = np.linspace(-0.45 * np.pi, 0.45 * np.pi, angle_count)
        short_angle = 0.0

    times, families = [], []
    for angle in np.atleast_1d(ray_angles):
        times.append(magnitudes * np.exp(1.0j * float(angle)))
        families.extend(["ray"] * magnitudes.size)
    for sign in (1.0, -1.0):
        times.append(sign * magnitudes)
        families.extend(["oscillatory"] * magnitudes.size)
    short = np.geomspace(0.05 / radial_max, 1.0 / radial_max,
                         max(2, per_octave))
    times.append(short * np.exp(1.0j * short_angle))
    families.extend(["short_time"] * short.size)

    flat = np.concatenate(times)
    labels = np.asarray(families)

    growth = np.max(-(flat[None, :] * d[:, None]).imag, axis=0)
    representable = growth <= float(options.growth_cap)
    flat, labels = flat[representable], labels[representable]
    if flat.size == 0:
        raise RuntimeError(
            "every candidate time violates the growth cap "
            f"{options.growth_cap:g} on this support; widen growth_cap or "
            "shrink the excluded region")

    order = np.lexsort((np.angle(flat), np.abs(flat), labels))
    flat, labels = flat[order], labels[order]
    keep = np.ones(flat.size, dtype=bool)
    seen: list[complex] = []
    for index, value in enumerate(flat):
        if any(abs(value - other) <= 1.0e-9 * abs(value) for other in seen):
            keep[index] = False
        else:
            seen.append(complex(value))
    return flat[keep], tuple(labels[keep])


def _fit_subproblem(problem: ReciprocalMeasureProblem,
                    frequency_indices: Array) -> ReciprocalMeasureProblem:
    return ReciprocalMeasureProblem(
        frequencies=problem.frequencies[frequency_indices],
        internal_sums=problem.internal_sums,
        cell_masses=problem.cell_masses,
        excluded_radius=problem.excluded_radius,
        normalization_floor=problem.normalization_floor,
        zero_weight_sum=problem.zero_weight_sum)


def _residual_scores(problem: ReciprocalMeasureProblem, times: Array,
                     weights: Array, candidates: Array) -> Array:
    """Mass-weighted matched-filter score of each candidate atom."""
    kept, delivered, _ = problem.retained()
    live = kept & (problem.cell_masses[None, :] > 0.0)
    d = problem.denominators[live]
    scale = (np.broadcast_to(problem.cell_masses[None, :], kept.shape)[live]
             / np.broadcast_to(delivered[:, None], kept.shape)[live])
    truth = 1.0 / d
    residual = (evaluate_rule(times, weights, d) - truth) if times.size else -truth
    exponent = 1.0j * d[:, None] * candidates[None, :]
    atoms = np.exp(exponent - np.max(exponent.real, axis=0)[None, :])
    correlation = np.abs((scale * residual.conj()) @ atoms)
    energy = np.sqrt(np.clip((scale @ (np.abs(atoms) ** 2)), 1.0e-300, None))
    return correlation / energy


def fit_reciprocal_measure(problem: ReciprocalMeasureProblem,
                           options: ComplexTimeSearchOptions) -> ComplexTimeRule:
    """Fit ``1/d`` on a weighted complex support to the delivered target.

    A sign-definite support inside one quarter-plane sector returns the
    certified rotated sinc rule.  Otherwise the rule is grown greedily
    from the candidate dictionary: each round scores candidates against
    the current weighted residual, re-solves the weight program for the
    few best, keeps the best exact-metric improvement, and exchanges the
    worst uncovered frequencies into the fitting set.  Once the target
    holds on every requested frequency the rule is pruned one node at a
    time with refitting.  Raises when the target is not reachable within
    ``max_nodes``, naming both levers.
    """
    if options.sector_shortcut:
        sector = _sector_rule_rotated(problem, options)
        if sector is not None and sector.node_count <= int(options.max_nodes):
            return sector

    candidates, candidate_families = candidate_time_dictionary(problem, options)
    n_frequency = problem.frequencies.size
    fit_count = min(int(options.fit_frequency_count), n_frequency)
    stride = max(1, n_frequency // fit_count)
    fit_set = list(dict.fromkeys(
        list(range(0, n_frequency, stride)) + [n_frequency - 1]))

    active = np.zeros(candidates.size, dtype=bool)
    growth_history: list[tuple[float, float, str, float]] = []
    # Seeding: a short sector-like ladder along the central decaying ray
    # when one exists (the guide's "sector nodes plus a small crossing
    # seed"), else the shortest crossing-capable atoms.  Ties resolve by
    # magnitude order, deterministically.
    families_array = np.asarray(candidate_families)
    ray_index = np.nonzero(families_array == "ray")[0]
    if ray_index.size:
        ray_angle = np.angle(candidates[ray_index])
        distinct_angles = np.unique(np.round(ray_angle, 12))
        central = distinct_angles[
            int(np.argmin(np.abs(distinct_angles - np.median(ray_angle))))]
        ladder = ray_index[np.abs(np.round(ray_angle, 12) - central) == 0.0]
        ladder = ladder[np.argsort(np.abs(candidates[ladder]))]
        seed_count = max(1, min(6, ladder.size, int(options.max_nodes) // 4))
        picks = ladder[np.linspace(0, ladder.size - 1, seed_count).astype(int)]
    else:
        picks = np.argsort(np.abs(candidates))[:4]
    for index in np.unique(picks):
        active[int(index)] = True

    target = float(options.target_error)

    def solve_on(indices: list[int],
                 mask: Array) -> tuple[Array, Array, float, Array]:
        sub = _fit_subproblem(problem, np.asarray(sorted(indices)))
        weights, _ = solve_fixed_time_weights(
            sub, candidates[mask],
            polygon_directions=int(options.polygon_directions),
            conditioning_slack=float(options.conditioning_slack),
            conditioning_pass=False,
            objective_scale=1.0 / target)
        error, _ = delivered_error(problem, candidates[mask], weights)
        return candidates[mask], weights, float(np.max(error)), error

    times, weights, best_error, error_by_frequency = solve_on(fit_set, active)
    for index in np.nonzero(active)[0]:
        growth_history.append((float(candidates[index].real),
                               float(candidates[index].imag),
                               candidate_families[index], best_error))

    exchanges_since_growth = 0
    while best_error > target:
        # Exchange before growth: cover the worst uncovered frequency with
        # the nodes already retained before spending a new node on it.
        worst = int(np.argmax(error_by_frequency))
        if (worst not in fit_set
                and exchanges_since_growth < int(options.exchange_rounds)):
            fit_set.append(worst)
            exchanges_since_growth += 1
            times, weights, best_error, error_by_frequency = solve_on(
                fit_set, active)
            continue
        if int(np.count_nonzero(active)) >= int(options.max_nodes):
            raise RuntimeError(
                f"no rule reached {target:g} within max_nodes="
                f"{int(options.max_nodes)}; best delivered error was "
                f"{best_error:.6g} — raise max_nodes or relax target_error")
        scores = _residual_scores(problem, times, weights, candidates)
        scores[active] = -np.inf
        # Tiebreak: score, then family label, then magnitude — declared so
        # equal scores resolve identically everywhere.
        order = np.lexsort((np.abs(candidates),
                            np.asarray(candidate_families),
                            -scores))
        trial_rows = []
        for index in order[: int(options.candidates_per_round)]:
            trial = active.copy()
            trial[int(index)] = True
            solved = solve_on(fit_set, trial)
            trial_rows.append((solved[2], int(index), solved))
        _, chosen, solved = min(trial_rows, key=lambda row: (row[0], row[1]))
        active[chosen] = True
        times, weights, best_error, error_by_frequency = solved
        growth_history.append((float(candidates[chosen].real),
                               float(candidates[chosen].imag),
                               candidate_families[chosen], best_error))
        exchanges_since_growth = 0

    # Prune with refitting, weakest node first; a removal survives only if
    # the exact metric still meets the target on every frequency.
    while int(np.count_nonzero(active)) > 1:
        live_index = np.nonzero(active)[0]
        strength = np.abs(weights)
        removal_order = live_index[np.lexsort((live_index, strength))]
        removal_order = removal_order[: int(options.prune_trials_per_round)]
        removed = False
        for index in removal_order:
            trial = active.copy()
            trial[int(index)] = False
            try:
                candidate_solution = solve_on(fit_set, trial)
            except RuntimeError:
                continue
            if candidate_solution[2] <= target:
                active = trial
                times, weights, best_error, error_by_frequency = (
                    candidate_solution)
                removed = True
                break
        if not removed:
            break

    sub = _fit_subproblem(problem, np.asarray(sorted(fit_set)))
    weights, _ = solve_fixed_time_weights(
        sub, candidates[active],
        polygon_directions=int(options.polygon_directions),
        conditioning_slack=float(options.conditioning_slack),
        conditioning_pass=True,
        objective_scale=1.0 / target)
    times = candidates[active]
    error, excluded = delivered_error(problem, times, weights)
    if float(np.max(error)) > target:
        # The conditioning pass holds its slack on the fitting set; if the
        # full-frequency metric slipped past the target, keep the stricter
        # pre-conditioning weights instead of returning a miss.
        weights, _ = solve_fixed_time_weights(
            sub, times,
            polygon_directions=int(options.polygon_directions),
            conditioning_pass=False,
            objective_scale=1.0 / target)
        error, excluded = delivered_error(problem, times, weights)
    p99, peak = rule_amplification(times, weights, problem)
    families = tuple(np.asarray(candidate_families)[active])
    return ComplexTimeRule(
        time_nodes=times,
        weights=weights,
        node_families=families,
        delivered_error_by_frequency=error,
        sampled_max_error=float(np.max(error)),
        excluded_mass_fraction=float(np.max(excluded)),
        amplification=p99,
        amplification_max=peak,
        method="measured_crossing",
        growth_history=tuple(growth_history),
    )


__all__ = [
    "ComplexTimeSearchOptions",
    "support_arc",
    "candidate_time_dictionary",
    "fit_reciprocal_measure",
]
