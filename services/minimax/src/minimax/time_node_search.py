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

Search schedule.  The growth loop's cost is (number of weight solves and
scoring passes) x (cost of each); ``ComplexTimeSearchOptions`` carries
independently disableable schedule toggles that attack both factors: a
coarse-to-fine ladder of equal-mass re-binned measures walked before the
full-resolution stage (with a fidelity guard that ends a rung the moment
the full metric stops following it), batched candidate acceptance while
the error is far from target, a per-stage precomputed candidate atom
table reused by every scoring pass, a bounded weakest-first prune, a
bounded exchange-in/exchange-out fitting frequency set, a reduced solver
depth far from target, and a per-problem candidate-dictionary cache for
tolerance ladders.  Every toggle keeps the declared tie-breaks and the
exact acceptance metric: a rule is accepted only when the measured
delivered error on the full problem meets the target.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from minimax.reciprocal_fit import (
    ComplexTimeRule,
    ReciprocalMeasureProblem,
    single_core_blas,
    delivered_error,
    evaluate_rule,
    rule_amplification,
    solve_fixed_time_weights,
    solve_fixed_time_weights_fast,
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

    Schedule toggles (each independently disableable; all preserve the
    declared tie-breaks and the exact full-problem acceptance metric):

    - ``coarse_to_fine`` grows the rule on an equal-mass re-binning of
      the measure to at most ``coarse_cell_cap`` supercells (total mass
      conserved) with ``coarse_fit_frequency_count`` fitting
      frequencies, down to ``coarse_stop_factor`` times the target only
      — the coarse metric's fidelity floor sits above the target itself,
      so the final stretch always runs on the full metric, which also
      grows further whenever refinement misses.  Growth-history rows
      record the delivered error of the stage that accepted them.
    - ``batched_growth`` accepts up to ``batch_size`` best-scored
      candidates in one weight solve while the error is more than
      ``batch_error_factor`` times the target, and reverts to
      single-candidate acceptance close to it.  Batch members must be
      pairwise separated by ``batch_diversity`` in relative time-plane
      distance — the top scores otherwise land on near-copies of one
      atom, which spend nodes without spanning new directions.
    - ``precompute_atoms`` builds the candidate atom matrix
      ``exp(1j*d*t)`` once per stage and reuses it for every scoring
      pass instead of re-exponentiating the whole dictionary each round.
    - ``cheap_prune`` tries only the ``cheap_prune_trials`` weakest
      nodes and stops after ``cheap_prune_failure_stop`` consecutive
      failed removal attempts.
    - ``dictionary_cache`` memoizes ``candidate_time_dictionary`` per
      (support, dictionary options), so a tolerance ladder on one
      problem builds its dictionary once.
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
    # Warm start: exact times to activate before growth (typically the
    # previous tolerance rung's rule when walking a ladder).  They join
    # the dictionary as family "seed" and remain prunable.
    seed_times: tuple = ()
    # "irls" solves each weight subproblem by reweighted least squares in
    # milliseconds and judges by the exact measured metric; "lp" restores
    # the certified polygonal program (tens of seconds per solve at
    # production sizes — measured 12-33 s — so search cost is O(nodes)
    # solves either way, just three orders apart).
    weight_solver: str = "irls"
    # Return the best rule found (target_met=False) instead of raising when
    # the node ceiling is reached; actual errors are reported either way.
    return_best_on_miss: bool = False
    # Search-schedule toggles; see the class docstring.
    coarse_to_fine: bool = True
    coarse_cell_cap: int = 60
    coarse_fit_frequency_count: int = 6
    coarse_polygon_directions: int = 8
    coarse_stop_factor: float = 3.0
    # Refinement ladder: (cell_cap, stop_factor) rungs walked in order
    # before the full-resolution stage; empty means the single rung
    # (coarse_cell_cap, coarse_stop_factor).  Each rung grows on the
    # re-binned measure until stop_factor x target, then sheds overshoot
    # by pruning at that rung's cost.  Each rung's stop factor is its
    # measured fidelity floor: refined past it, a rule starts fitting the
    # supercell centroids instead of the measure and the next stage pays
    # to undo that.
    coarse_rungs: tuple = ((60, 3.0), (150, 1.2))
    batched_growth: bool = True
    batch_size: int = 3
    batch_error_factor: float = 10.0
    batch_diversity: float = 0.2
    precompute_atoms: bool = True
    cheap_prune: bool = True
    cheap_prune_trials: int = 3
    cheap_prune_failure_stop: int = 2
    dictionary_cache: bool = True
    # Warm-start each IRLS weight solve from the previous accepted state
    # (new nodes enter at zero weight).  MEASURED NEGATIVE on the toy
    # ensemble and therefore off by default: the biased start stalls the
    # Lawson equilibration in the previous basin, weight quality drops,
    # and the greedy spends more nodes and more solves (P4 cond/primary
    # 1e-4: 38 nodes / 63 s cold against 50 nodes / 109 s warm).  Kept as
    # a toggle for supports where the trade may differ.
    warm_start: bool = False
    # Carry the fitting frequency set across stages: the frequencies a
    # rung exchanged in are the binding ones, and re-discovering them at
    # the next stage costs one full re-solve each.
    carry_fit_set: bool = True
    # While the error is more than batch_error_factor x target the weight
    # subproblem runs at a reduced IRLS iteration ceiling: far from the
    # target the greedy only needs a rough ranking, and the exact sweep
    # depth matters only where single-candidate acceptance takes over.
    light_when_far: bool = True
    light_iterations: int = 16
    # Bound the fitting set: when an exchange pushes it past
    # fit_set_cap_factor x fit_frequency_count members, the member whose
    # frequency currently has the smallest delivered error leaves.  Solve
    # rows scale with the fitting set, and an unbounded set accumulates
    # every frequency ever exchanged in whether or not it still binds.
    cap_fit_set: bool = True
    fit_set_cap_factor: float = 3.0
    # Rung fidelity guard (always on with coarse_to_fine): once a rung's
    # own error is within 10x of its stop target, every acceptance also
    # measures the FULL metric, and the rung ends as soon as its metric
    # keeps improving (>=20% per acceptance) while the full metric stops
    # following (<5%) — refining a supercell measure past the point where
    # the true measure follows is fiction, and on a wide-grid crossing
    # support an unguarded rung burned the entire node budget on it.

    def __post_init__(self) -> None:
        if not 0.0 < float(self.target_error) < 1.0:
            raise ValueError("target_error must lie in (0,1)")
        if int(self.max_nodes) < 1:
            raise ValueError("max_nodes must be positive")
        if str(self.weight_solver) not in ("irls", "lp"):
            raise ValueError("weight_solver must be exactly 'irls' or 'lp'")
        seeds = np.asarray(self.seed_times, dtype=np.complex128).reshape(-1)
        if seeds.size > int(self.max_nodes):
            raise ValueError(
                f"seed_times has {seeds.size} nodes but max_nodes="
                f"{int(self.max_nodes)}")
        if not np.all(np.isfinite(seeds)):
            raise ValueError("seed_times must contain only finite values")
        for index, value in enumerate(seeds):
            if index and np.any(
                    np.abs(value - seeds[:index])
                    <= 1.0e-9 * max(abs(value), 1.0)):
                raise ValueError("seed_times must not contain duplicates")
        if int(self.candidate_angle_count) < 1 or int(self.candidates_per_round) < 1:
            raise ValueError("candidate counts must be positive")
        if int(self.coarse_cell_cap) < 1 or int(self.coarse_fit_frequency_count) < 1:
            raise ValueError("coarse stage sizes must be positive")
        if float(self.coarse_stop_factor) < 1.0:
            raise ValueError("coarse_stop_factor must be at least 1")
        if int(self.batch_size) < 1 or float(self.batch_error_factor) < 1.0:
            raise ValueError(
                "batch_size must be positive and batch_error_factor at least 1")
        if not 0.0 <= float(self.batch_diversity) < 1.0:
            raise ValueError("batch_diversity must lie in [0,1)")
        if int(self.cheap_prune_trials) < 1 or int(self.cheap_prune_failure_stop) < 1:
            raise ValueError("cheap prune bounds must be positive")
        if int(self.light_iterations) < 1:
            raise ValueError("light_iterations must be positive")
        if float(self.fit_set_cap_factor) < 1.0:
            raise ValueError("fit_set_cap_factor must be at least 1")
        for rung in self.coarse_rungs:
            cap, stop = rung
            if int(cap) < 1 or float(stop) < 1.0:
                raise ValueError(
                    "each coarse rung needs a positive cell cap and a stop "
                    "factor of at least 1")


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
    # Sequential first-wins dedup, identical comparisons to the historical
    # Python loop (each value against every previously KEPT one), with the
    # inner scan vectorized: the O(n^2) pure-Python pass measured seconds
    # on production dictionaries.
    keep = np.ones(flat.size, dtype=bool)
    seen = np.empty(flat.size, dtype=np.complex128)
    n_seen = 0
    for index, value in enumerate(flat):
        if n_seen and bool(np.any(np.abs(value - seen[:n_seen])
                                  <= 1.0e-9 * abs(value))):
            keep[index] = False
        else:
            seen[n_seen] = value
            n_seen += 1
    return flat[keep], tuple(labels[keep])


#: Per-problem dictionary memo for tolerance ladders (toggle:
#: ``dictionary_cache``).  Keyed by a digest of the support arrays and the
#: dictionary-shaping options; bounded FIFO so a long campaign cannot grow
#: it without limit.  Entries are returned by reference and are treated as
#: immutable by every consumer in this module.
_DICTIONARY_CACHE: dict[bytes, tuple[Array, tuple[str, ...]]] = {}
_DICTIONARY_CACHE_LIMIT = 8


def _cached_candidate_dictionary(
        problem: ReciprocalMeasureProblem,
        options: ComplexTimeSearchOptions) -> tuple[Array, tuple[str, ...]]:
    if not options.dictionary_cache:
        return candidate_time_dictionary(problem, options)
    digest = hashlib.sha256()
    for payload in (problem.frequencies, problem.internal_sums,
                    problem.cell_masses):
        digest.update(np.ascontiguousarray(payload).tobytes())
    digest.update(np.float64(problem.excluded_radius).tobytes())
    digest.update(f"{int(options.candidate_times_per_octave)}:"
                  f"{int(options.candidate_angle_count)}:"
                  f"{float(options.growth_cap)!r}".encode())
    key = digest.digest()
    hit = _DICTIONARY_CACHE.get(key)
    if hit is None:
        hit = candidate_time_dictionary(problem, options)
        while len(_DICTIONARY_CACHE) >= _DICTIONARY_CACHE_LIMIT:
            _DICTIONARY_CACHE.pop(next(iter(_DICTIONARY_CACHE)))
        _DICTIONARY_CACHE[key] = hit
    return hit


def _coarsened_measure(problem: ReciprocalMeasureProblem,
                       cap: int) -> ReciprocalMeasureProblem:
    """Equal-mass re-binning of the cell measure to at most ``cap`` cells.

    Mass-quantile columns along ``Re s`` subdivided by mass-quantile rows
    along ``Im s`` (the same construction as the adaptive histogram that
    produces production measures); each supercell carries the summed mass
    at the mass-weighted centroid, so total mass is conserved.  Two
    rejected designs, both measured on the toy ensemble: a mass-topped
    cell SELECTION keeps only 29% of an equal-mass histogram's mass and
    missed the full metric by 6x at handoff; sensitivity-weighted cuts
    (``mass / distance-to-grid``) starved the far tail of resolution and
    doubled the node count on the near-crossing branch.  How faithful a
    rung can stay is instead policed at run time by the fidelity guard in
    the growth loop.
    """
    live = problem.cell_masses > 0.0
    sums = problem.internal_sums[live]
    masses = problem.cell_masses[live]
    if sums.size <= int(cap):
        return problem
    columns = max(1, int(round(np.sqrt(4.0 * float(cap)))))
    rows = max(1, int(cap) // columns)

    def quantile_groups(values: Array, weights: Array, count: int) -> Array:
        """Group ids splitting sorted ``values`` into ~equal-weight runs."""
        order = np.argsort(values, kind="stable")
        cumulative = np.cumsum(weights[order])
        midpoint = cumulative - 0.5 * weights[order]
        edges = np.linspace(0.0, cumulative[-1], count + 1)[1:-1]
        group = np.empty(values.size, dtype=np.int64)
        group[order] = np.searchsorted(edges, midpoint, side="right")
        return group

    column_id = quantile_groups(sums.real, masses, columns)
    cell_id = np.full(sums.size, -1, dtype=np.int64)
    for column in range(columns):
        inside = column_id == column
        if not np.any(inside):
            continue
        row_id = quantile_groups(sums.imag[inside], masses[inside], rows)
        cell_id[inside] = column * rows + row_id
    labels, compact = np.unique(cell_id, return_inverse=True)
    mass = np.bincount(compact, weights=masses, minlength=labels.size)
    centroid = (
        np.bincount(compact, weights=masses * sums.real, minlength=labels.size)
        + 1.0j * np.bincount(compact, weights=masses * sums.imag,
                             minlength=labels.size)) / mass
    return ReciprocalMeasureProblem(
        frequencies=problem.frequencies,
        internal_sums=centroid,
        cell_masses=mass,
        excluded_radius=problem.excluded_radius,
        normalization_floor=problem.normalization_floor,
        zero_weight_sum=problem.zero_weight_sum)


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


class _CandidateAtomTable:
    """Candidate atoms ``exp(1j*d*t)``, gauged per column, built once per
    (problem, dictionary) and reused by every scoring pass (toggle:
    ``precompute_atoms``).

    The scores are the same mass-weighted matched filter as
    ``_residual_scores``; the current rule's residual is evaluated through
    the gauged columns (``exp(x - m) * exp(m)`` instead of a fresh
    ``exp(x)``), so scores agree with the recomputing path to
    floating-point roundoff.  Scores only rank candidates — every accepted
    node is still judged by the exact delivered metric.
    """

    def __init__(self, problem: ReciprocalMeasureProblem,
                 candidates: Array) -> None:
        kept, delivered, _ = problem.retained()
        live = kept & (problem.cell_masses[None, :] > 0.0)
        keep_i, keep_j = np.nonzero(live)
        d = problem.denominators[live]
        self.frequency_index = keep_i
        self.n_frequency = problem.frequencies.size
        self.scale = problem.cell_masses[keep_j] / delivered[keep_i]
        self.truth = 1.0 / d
        exponent = 1.0j * d[:, None] * candidates[None, :]
        self.log_peak = np.max(exponent.real, axis=0)
        exponent -= self.log_peak[None, :]
        self.atoms = np.exp(exponent, out=exponent)
        self.energy = np.sqrt(np.clip(
            self.scale @ (np.abs(self.atoms) ** 2), 1.0e-300, None))

    def _residual(self, active: Array, weights: Array) -> Array:
        if np.any(active):
            ungauged = np.asarray(weights) * np.exp(self.log_peak[active])
            return self.atoms[:, active] @ ungauged - self.truth
        return -self.truth

    def scores(self, active: Array, weights: Array) -> Array:
        residual = self._residual(active, weights)
        correlation = np.abs((self.scale * residual.conj()) @ self.atoms)
        return correlation / self.energy

    def delivered_max_error(self, active: Array, weights: Array) -> float:
        """Max-over-frequency delivered error of the active rule, evaluated
        through the gauged columns (agrees with ``delivered_error`` to
        floating-point roundoff)."""
        residual = self._residual(active, weights)
        by_frequency = np.bincount(self.frequency_index,
                                   weights=self.scale * np.abs(residual),
                                   minlength=self.n_frequency)
        return float(np.max(by_frequency))


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

    The schedule toggles on ``options`` (coarse-to-fine staging, batched
    acceptance, precomputed scoring atoms, bounded pruning, dictionary
    caching) change how many solves the search spends, never what it
    accepts: acceptance always requires the measured delivered error on
    the full problem to meet the target.  BLAS is pinned to one thread
    for the whole fit — see ``reciprocal_fit.single_core_blas``.
    """
    with single_core_blas():
        return _fit_reciprocal_measure_pinned(problem, options)


def _validated_seed_times(
        problem: ReciprocalMeasureProblem,
        options: ComplexTimeSearchOptions) -> Array:
    """Validate warm nodes on the same retained support as the dictionary."""
    warm = np.asarray(options.seed_times, dtype=np.complex128).reshape(-1)
    if not warm.size:
        return warm
    kept, _, _ = problem.retained()
    live = kept & (problem.cell_masses[None, :] > 0.0)
    denominator = problem.denominators[live]
    growth = np.max(
        -(warm[None, :] * denominator[:, None]).imag, axis=0)
    bad = np.nonzero(growth > float(options.growth_cap))[0]
    if bad.size:
        index = int(bad[0])
        raise ValueError(
            f"seed_times[{index}] violates growth_cap="
            f"{float(options.growth_cap):g} on the retained support: "
            f"log magnitude {float(growth[index]):.6g}")
    return warm


def _fit_reciprocal_measure_pinned(
        problem: ReciprocalMeasureProblem,
        options: ComplexTimeSearchOptions) -> ComplexTimeRule:
    warm = _validated_seed_times(problem, options)
    if options.sector_shortcut:
        sector = _sector_rule_rotated(problem, options)
        if sector is not None and sector.node_count <= int(options.max_nodes):
            return sector

    candidates, candidate_families = _cached_candidate_dictionary(
        problem, options)
    if warm.size:
        labels = list(candidate_families)
        append = []
        for value in warm:
            close = np.nonzero(
                np.abs(value - candidates)
                <= 1.0e-9 * max(abs(value), 1.0))[0]
            if close.size:
                labels[int(close[0])] = "seed"
            else:
                append.append(value)
        if append:
            candidates = np.concatenate(
                [candidates, np.asarray(append, np.complex128)])
            labels.extend(["seed"] * len(append))
        candidate_families = tuple(labels)

    active = np.zeros(candidates.size, dtype=bool)
    growth_history: list[tuple[float, float, str, float]] = []
    # Seeding: a short sector-like ladder along the central decaying ray
    # when one exists (the guide's "sector nodes plus a small crossing
    # seed"), else the shortest crossing-capable atoms.  Ties resolve by
    # magnitude order, deterministically.
    families_array = np.asarray(candidate_families)
    seed_index = np.nonzero(families_array == "seed")[0]
    ray_index = np.nonzero(families_array == "ray")[0]
    if seed_index.size:
        picks = seed_index
    elif ray_index.size:
        ray_angle = np.angle(candidates[ray_index])
        distinct_angles = np.unique(np.round(ray_angle, 12))
        central = distinct_angles[
            int(np.argmin(np.abs(distinct_angles - np.median(ray_angle))))]
        ladder = ray_index[np.abs(np.round(ray_angle, 12) - central) == 0.0]
        ladder = ladder[np.argsort(np.abs(candidates[ladder]))]
        seed_count = max(1, min(6, ladder.size, int(options.max_nodes) // 4))
        picks = ladder[np.linspace(0, ladder.size - 1, seed_count).astype(int)]
    else:
        picks = np.argsort(np.abs(candidates))[
            :min(4, int(options.max_nodes))]
    for index in np.unique(picks):
        active[int(index)] = True

    target = float(options.target_error)
    atom_tables: dict[
        int, tuple[ReciprocalMeasureProblem, _CandidateAtomTable]] = {}
    small_full_problem = problem.denominators.size <= 1024
    tight_target = target <= 1.0e-6

    def weight_solve(sub: ReciprocalMeasureProblem, nodes: Array,
                     conditioning: bool, polygon: int,
                     start_weights: Array | None = None,
                     light: bool = False) -> Array:
        if options.weight_solver == "lp":
            weights, _ = solve_fixed_time_weights(
                sub, nodes,
                polygon_directions=int(polygon),
                conditioning_slack=float(options.conditioning_slack),
                conditioning_pass=conditioning,
                objective_scale=1.0 / target)
        else:
            if tight_target:
                extra = {"iterations": 48, "stall_iterations": 12}
            elif small_full_problem:
                extra = {"iterations": 64, "stall_iterations": 16}
            else:
                extra = ({"iterations": int(options.light_iterations)}
                         if light else {})
            weights, _ = solve_fixed_time_weights_fast(
                sub, nodes,
                conditioning_slack=float(options.conditioning_slack),
                conditioning_pass=conditioning,
                start_weights=start_weights, **extra)
        return weights

    def solve_on(stage: ReciprocalMeasureProblem, indices: list[int],
                 mask: Array, polygon: int,
                 warm: tuple[Array, Array] | None = None,
                 light: bool = False) -> tuple[Array, Array, float, Array]:
        sub = _fit_subproblem(stage, np.asarray(sorted(indices)))
        start_weights = None
        if warm is not None and options.warm_start:
            previous_mask, previous_weights = warm
            start_weights = np.zeros(int(np.count_nonzero(mask)),
                                     dtype=np.complex128)
            shared = previous_mask & mask
            if np.any(shared):
                inside_mask = np.cumsum(mask) - 1
                inside_previous = np.cumsum(previous_mask) - 1
                shared_index = np.nonzero(shared)[0]
                start_weights[inside_mask[shared_index]] = (
                    previous_weights[inside_previous[shared_index]])
        weights = weight_solve(sub, candidates[mask], False, polygon,
                               start_weights, light)
        error, _ = delivered_error(stage, candidates[mask], weights)
        return candidates[mask], weights, float(np.max(error)), error

    def atom_table(stage: ReciprocalMeasureProblem) -> _CandidateAtomTable:
        entry = atom_tables.get(id(stage))
        if entry is None or entry[0] is not stage:
            table = _CandidateAtomTable(stage, candidates)
            atom_tables[id(stage)] = (stage, table)
            return table
        return entry[1]

    def residual_scores(stage: ReciprocalMeasureProblem, mask: Array,
                        times: Array, weights: Array) -> Array:
        if options.precompute_atoms:
            return atom_table(stage).scores(mask, weights)
        return _residual_scores(stage, times, weights, candidates)

    def full_metric_error(mask: Array, times: Array, weights: Array) -> float:
        """Max delivered error on the FULL problem (rung fidelity guard)."""
        if options.precompute_atoms:
            return atom_table(problem).delivered_max_error(mask, weights)
        error, _ = delivered_error(problem, times, weights)
        return float(np.max(error))

    def batch_pick(order: Array, scores: Array, batch: int) -> list[int]:
        """Top-scored batch with pairwise time-plane diversity.

        Walks the declared order and keeps a candidate only when it sits
        at relative distance above ``batch_diversity`` from every member
        already kept this round; the scan is bounded so a dictionary of
        near-duplicates cannot turn one pick into a full sweep.
        """
        separation = float(options.batch_diversity)
        chosen: list[int] = []
        for index in order[: max(64, 8 * batch)]:
            index = int(index)
            if not np.isfinite(scores[index]):
                break
            value = candidates[index]
            close = any(abs(value - candidates[other])
                        <= separation * max(abs(value),
                                            abs(candidates[other]))
                        for other in chosen)
            if not close:
                chosen.append(index)
            if len(chosen) >= batch:
                break
        return chosen

    def grow(stage: ReciprocalMeasureProblem, fit_frequency_count: int,
             polygon: int, record_initial: bool, raise_on_ceiling: bool,
             stage_target: float, carried_fit_set: list[int] | None = None,
             carried_warm: tuple[Array, Array] | None = None,
             guard_fidelity: bool = False,
             ) -> tuple[list[int], Array, Array, float, Array]:
        """Grow ``active`` (in place) until the stage metric meets
        ``stage_target``, or — with ``guard_fidelity`` — until the full
        metric stops following the stage metric."""
        n_frequency = stage.frequencies.size
        fit_count = min(int(fit_frequency_count), n_frequency)
        stride = max(1, n_frequency // fit_count)
        fit_set = list(dict.fromkeys(
            list(range(0, n_frequency, stride)) + [n_frequency - 1]
            + (carried_fit_set or [])))
        guard_last: tuple[float, float] | None = None
        times, weights, best_error, error_by_frequency = solve_on(
            stage, fit_set, active, polygon, carried_warm)
        best_seen = best_error
        if record_initial:
            for index in np.nonzero(active)[0]:
                growth_history.append((float(candidates[index].real),
                                       float(candidates[index].imag),
                                       candidate_families[index], best_error))

        exchanges_since_growth = 0
        while best_error > stage_target:
            # Exchange before growth: cover the worst uncovered frequency
            # with the nodes already retained before spending a new node
            # on it.
            far = best_error > float(options.batch_error_factor) * target
            light = far and options.light_when_far
            worst = int(np.argmax(error_by_frequency))
            if (worst not in fit_set
                    and exchanges_since_growth < int(options.exchange_rounds)):
                fit_set.append(worst)
                if options.cap_fit_set:
                    cap = max(2, int(np.ceil(
                        float(options.fit_set_cap_factor) * fit_count)))
                    if len(fit_set) > cap:
                        slack = min(
                            (member for member in fit_set if member != worst),
                            key=lambda member: (error_by_frequency[member],
                                                member))
                        fit_set.remove(slack)
                exchanges_since_growth += 1
                times, weights, best_error, error_by_frequency = solve_on(
                    stage, fit_set, active, polygon, (active, weights),
                    light=light)
                best_seen = min(best_seen, best_error)
                continue
            budget = int(options.max_nodes) - int(np.count_nonzero(active))
            if budget <= 0:
                if raise_on_ceiling:
                    raise RuntimeError(
                        f"no rule reached {target:g} within max_nodes="
                        f"{int(options.max_nodes)}; best delivered error was "
                        f"{best_seen:.6g} — raise max_nodes or relax "
                        "target_error")
                break
            scores = residual_scores(stage, active, times, weights)
            scores[active] = -np.inf
            # Tiebreak: score, then family label, then magnitude — declared
            # so equal scores resolve identically everywhere.
            order = np.lexsort((np.abs(candidates), families_array, -scores))
            batch = 1
            if options.batched_growth and far:
                batch = min(int(options.batch_size), budget)
            chosen_batch = batch_pick(order, scores, batch) if batch > 1 else []
            if batch > 1 and len(chosen_batch) > 1:
                warm = (active.copy(), weights)
                for index in chosen_batch:
                    active[index] = True
                times, weights, best_error, error_by_frequency = solve_on(
                    stage, fit_set, active, polygon, warm, light=light)
                for index in chosen_batch:
                    growth_history.append((float(candidates[index].real),
                                           float(candidates[index].imag),
                                           candidate_families[index],
                                           best_error))
            else:
                trial_rows = []
                for index in order[: int(options.candidates_per_round)]:
                    trial = active.copy()
                    trial[int(index)] = True
                    solved = solve_on(stage, fit_set, trial, polygon,
                                      (active, weights), light=light)
                    trial_rows.append((solved[2], int(index), solved))
                _, chosen, solved = min(trial_rows,
                                        key=lambda row: (row[0], row[1]))
                active[chosen] = True
                times, weights, best_error, error_by_frequency = solved
                growth_history.append((float(candidates[chosen].real),
                                       float(candidates[chosen].imag),
                                       candidate_families[chosen], best_error))
            best_seen = min(best_seen, best_error)
            exchanges_since_growth = 0
            if guard_fidelity and best_error <= 10.0 * stage_target:
                full_error = full_metric_error(active, times, weights)
                if guard_last is not None:
                    last_rung, last_full = guard_last
                    # The rung keeps improving but the full metric no
                    # longer follows: everything below is supercell
                    # fiction, so hand over to the next stage now.
                    if (best_error <= 0.8 * last_rung
                            and full_error >= 0.95 * last_full):
                        guard_last = (best_error, full_error)
                        break
                guard_last = (best_error, full_error)
        return fit_set, times, weights, best_error, error_by_frequency

    def prune_rule(stage: ReciprocalMeasureProblem, fit_set: list[int],
                   polygon: int, prune_target: float, cheap: bool,
                   start_weights: Array,
                   ) -> tuple[Array, Array, float, Array] | None:
        """Prune ``active`` (in place) with refitting, weakest node first.

        A removal survives only if the stage metric still meets
        ``prune_target`` on every frequency.  ``cheap`` bounds the trials
        per round and stops after consecutive failures; the full policy
        retries until a whole round removes nothing.
        """
        nonlocal active
        state = None
        weights = start_weights
        consecutive_failures = 0
        while int(np.count_nonzero(active)) > 1:
            live_index = np.nonzero(active)[0]
            strength = np.abs(weights)
            removal_order = live_index[np.lexsort((live_index, strength))]
            trials = (int(options.cheap_prune_trials) if cheap
                      else int(options.prune_trials_per_round))
            removal_order = removal_order[:trials]
            removed = False
            stopped = False
            for index in removal_order:
                trial = active.copy()
                trial[int(index)] = False
                try:
                    candidate_solution = solve_on(stage, fit_set, trial,
                                                  polygon, (active, weights))
                except RuntimeError:
                    candidate_solution = None
                if (candidate_solution is not None
                        and candidate_solution[2] <= prune_target):
                    active = trial
                    state = candidate_solution
                    weights = candidate_solution[1]
                    removed = True
                    consecutive_failures = 0
                    break
                consecutive_failures += 1
                if (cheap and consecutive_failures
                        >= int(options.cheap_prune_failure_stop)):
                    stopped = True
                    break
            if not removed or stopped:
                break
        return state

    # Coarse-to-fine: grow on the highest-mass sub-measure first, then
    # refine on the full problem — the full stage below grows further only
    # if the exact full-problem metric still misses the target.
    rungs = tuple(options.coarse_rungs) or (
        (int(options.coarse_cell_cap), float(options.coarse_stop_factor)),)
    run_coarse = False
    carried_fit_set: list[int] | None = None
    carried_warm: tuple[Array, Array] | None = None
    if options.coarse_to_fine:
        recorded_initial = False
        for cell_cap, stop_factor in rungs:
            if problem.cell_masses.size <= int(cell_cap):
                continue
            coarse_problem = _coarsened_measure(problem, int(cell_cap))
            coarse_target = target * float(stop_factor)
            coarse_fit_set, _, coarse_weights, coarse_error, _ = grow(
                coarse_problem, int(options.coarse_fit_frequency_count),
                int(options.coarse_polygon_directions),
                record_initial=not recorded_initial, raise_on_ceiling=False,
                stage_target=coarse_target,
                carried_fit_set=carried_fit_set, carried_warm=carried_warm,
                guard_fidelity=True)
            run_coarse = True
            recorded_initial = True
            if coarse_error <= coarse_target:
                # Shed batch and exchange overshoot at this rung's cost
                # before the next rung pays more per solve: the full prune
                # policy runs here because each trial is cheap.
                pruned_rung = prune_rule(
                    coarse_problem, coarse_fit_set,
                    int(options.coarse_polygon_directions),
                    coarse_target, cheap=False, start_weights=coarse_weights)
                if pruned_rung is not None:
                    coarse_weights = pruned_rung[1]
            if options.carry_fit_set:
                carried_fit_set = list(coarse_fit_set)
            carried_warm = (active.copy(), coarse_weights)

    fit_set, times, weights, best_error, error_by_frequency = grow(
        problem, int(options.fit_frequency_count),
        int(options.polygon_directions),
        record_initial=not run_coarse,
        raise_on_ceiling=not bool(options.return_best_on_miss),
        stage_target=target,
        carried_fit_set=carried_fit_set, carried_warm=carried_warm)

    # Prune with refitting, weakest node first; a removal survives only if
    # the exact metric still meets the target on every frequency.
    pruned = prune_rule(problem, fit_set, int(options.polygon_directions),
                        target, cheap=bool(options.cheap_prune),
                        start_weights=weights)
    if pruned is not None:
        times, weights, best_error, error_by_frequency = pruned

    sub = _fit_subproblem(problem, np.asarray(sorted(fit_set)))
    loop_weights = weights
    weights = weight_solve(sub, candidates[active], True,
                           int(options.polygon_directions))
    times = candidates[active]
    error, excluded = delivered_error(problem, times, weights)
    if float(np.max(error)) > target:
        # The conditioning pass holds its slack on the fitting set; if the
        # full-frequency metric slipped past the target, keep the stricter
        # pre-conditioning weights instead of returning a miss.
        weights = weight_solve(sub, times, False,
                               int(options.polygon_directions))
        error, excluded = delivered_error(problem, times, weights)
    if float(np.max(error)) > target:
        # The growth loop already accepted these same nodes and weights on
        # the exact full metric.  A cold final polish is allowed to improve
        # that state, never to replace it with a miss.
        weights = loop_weights
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
        target_met=bool(float(np.max(error))
                        <= float(options.target_error) * (1.0 + 1.0e-12)),
        growth_history=tuple(growth_history),
    )


__all__ = [
    "ComplexTimeSearchOptions",
    "support_arc",
    "candidate_time_dictionary",
    "fit_reciprocal_measure",
]
