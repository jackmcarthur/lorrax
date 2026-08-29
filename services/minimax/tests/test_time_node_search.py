"""``minimax.time_node_search`` through the door: paths, targets, twins.

Every support here is an explicit deterministic grid — no random state —
and every fit is small (at most 12 frequencies, at most 12 cells) with a
loose target, so the whole file runs on a laptop in well under a minute.

The cells follow the owner's Phase-1 program: the sector path answers a
sign-definite support with an analytic bound; a crossing support is fitted
and measured within the node budget; the delivered measure relaxes where
mass vanishes (the wrong-sign-tail continuity cell) and where cells are
excluded by radius; and the anticausal red twin asserts the discriminating
growth-cap signature rather than "a number changed".
"""

import numpy as np
import pytest

from minimax import (
    ComplexTimeSearchOptions,
    ReciprocalMeasureProblem,
    candidate_time_dictionary,
    fit_reciprocal_measure,
    support_arc,
)

FAMILY_LABELS = {"sector", "ray", "oscillatory", "short_time"}


def _sector_problem() -> ReciprocalMeasureProblem:
    """d = omega + level - 0.4j: one quarter-plane, no crossing."""
    omega = np.linspace(0.0, 5.0, 5)
    levels = np.geomspace(3.0, 70.0, 7)
    return ReciprocalMeasureProblem(
        frequencies=omega,
        internal_sums=-levels + 1.0j * 0.4,
        cell_masses=np.ones(levels.size))


def _crossing_problem(gamma_sign: float) -> ReciprocalMeasureProblem:
    """d = omega - pole + gamma_sign * 0.5j with Re(d) crossing zero."""
    omega = np.linspace(-4.0, 4.0, 9)
    poles = np.linspace(-3.0, 3.0, 7)
    return ReciprocalMeasureProblem(
        frequencies=omega,
        internal_sums=poles - gamma_sign * 1.0j * 0.5,
        cell_masses=np.ones(poles.size))


# ---------------------------------------------------------------------------
#  the two answer paths
# ---------------------------------------------------------------------------

def test_a_sign_definite_support_is_answered_by_the_certified_sector_rule():
    rule = fit_reciprocal_measure(
        _sector_problem(),
        ComplexTimeSearchOptions(target_error=1.0e-3, max_nodes=48))
    assert rule.method == "sector"
    assert rule.error_bound is not None
    assert 0.0 < rule.error_bound <= 1.0e-3
    assert rule.sampled_max_error <= 1.0e-3
    assert rule.node_count <= 48
    assert set(rule.node_families) == {"sector"}
    assert np.all(np.isfinite(rule.time_nodes))
    assert np.all(np.isfinite(rule.weights))


def test_a_support_crossing_the_real_axis_is_fitted_within_target_and_node_budget():
    problem = _crossing_problem(gamma_sign=1.0)
    _, _, _, width = support_arc(problem)
    assert width > 0.5 * np.pi  # the premise: no sector can serve this
    rule = fit_reciprocal_measure(
        problem,
        ComplexTimeSearchOptions(target_error=1.0e-3, max_nodes=48,
                                 fit_frequency_count=6))
    assert rule.method == "measured_crossing"
    assert rule.error_bound is None
    assert rule.sampled_max_error <= 1.0e-3
    assert rule.node_count <= 48
    assert set(rule.node_families) <= FAMILY_LABELS
    assert np.all(np.isfinite(rule.time_nodes))
    assert np.all(np.isfinite(rule.weights))


def test_an_exact_complex_resolvent_with_denominators_of_both_real_signs_is_fitted():
    u = np.concatenate([-np.geomspace(0.5, 8.0, 6)[::-1],
                        np.geomspace(0.5, 8.0, 6)])
    problem = ReciprocalMeasureProblem(
        frequencies=np.array([0.0]),
        internal_sums=-u - 1.0j * 0.5,   # d = u + 0.5j, both real signs
        cell_masses=np.ones(u.size))
    rule = fit_reciprocal_measure(
        problem, ComplexTimeSearchOptions(target_error=1.0e-3, max_nodes=48))
    assert rule.method == "measured_crossing"
    assert rule.sampled_max_error <= 1.0e-3
    assert rule.node_count <= 48
    assert set(rule.node_families) <= FAMILY_LABELS


# ---------------------------------------------------------------------------
#  scale coverage of the retained nodes
# ---------------------------------------------------------------------------

def test_two_spectral_bumps_at_separated_scales_retain_both_short_and_long_time_nodes():
    near = np.array([-2.5, -2.0, 2.0, 2.5])
    far = np.linspace(70.0, 130.0, 7)
    u = np.concatenate([near, far])
    problem = ReciprocalMeasureProblem(
        frequencies=np.array([0.0]),
        internal_sums=-u - 1.0j * 0.3,   # bumps near |d|~2 and |d|~100
        cell_masses=np.concatenate([np.ones(near.size),
                                    10.0 * np.ones(far.size)]))
    rule = fit_reciprocal_measure(
        problem, ComplexTimeSearchOptions(target_error=1.0e-3, max_nodes=48))
    assert rule.sampled_max_error <= 1.0e-3
    magnitudes = np.abs(rule.time_nodes)
    assert float(np.min(magnitudes)) <= 0.05   # the |d|~100 bump's node
    assert float(np.max(magnitudes)) >= 0.25   # the |d|~2 bump's node
    assert set(rule.node_families) <= FAMILY_LABELS


# ---------------------------------------------------------------------------
#  the delivered measure relaxes where mass vanishes or is excluded
# ---------------------------------------------------------------------------

def test_a_vanishing_mass_wrong_sign_tail_costs_at_most_a_few_nodes_over_the_sector_answer():
    """Continuity in the vanishing-mass limit.

    One cell just across the crossing line (Re(d) < 0) carrying 1e-12 of
    the total mass flips the path from sector to measured, and the
    delivered measure must all but ignore it: the fitted rule stays within
    a few nodes of the pure sector answer.
    """
    pure = fit_reciprocal_measure(
        _sector_problem(),
        ComplexTimeSearchOptions(target_error=1.0e-3, max_nodes=48))
    assert pure.method == "sector"

    base = _sector_problem()
    tainted = ReciprocalMeasureProblem(
        frequencies=base.frequencies,
        internal_sums=np.concatenate(
            [base.internal_sums, [0.5 + 0.4j]]),  # d = omega - 0.5 - 0.4j
        cell_masses=np.concatenate(
            [base.cell_masses, [1.0e-12 * float(np.sum(base.cell_masses))]]))
    rule = fit_reciprocal_measure(
        tainted,
        ComplexTimeSearchOptions(target_error=1.0e-3, max_nodes=48,
                                 fit_frequency_count=5))
    assert rule.method == "measured_crossing"
    assert rule.sampled_max_error <= 1.0e-3
    assert rule.node_count <= pure.node_count + 4


def test_cells_inside_the_excluded_radius_are_reported_and_do_not_constrain_the_fit():
    """A huge-mass cell at |d| ~ 1e-3 sits under excluded_radius = 0.5.

    Constrained, that cell would demand a rule accurate at three decades
    below the rest of the support, unreachable at this target and budget;
    excluded, the fit succeeds and the dropped mass is reported exactly.
    """
    levels = np.geomspace(3.0, 70.0, 7)
    sums = np.concatenate([-levels + 1.0j * 0.4, [-1.0e-3 + 0.0j]])
    masses = np.concatenate([np.ones(levels.size), [1.0e6]])
    problem = ReciprocalMeasureProblem(
        frequencies=np.array([0.0]),
        internal_sums=sums,
        cell_masses=masses,
        excluded_radius=0.5)
    rule = fit_reciprocal_measure(
        problem, ComplexTimeSearchOptions(target_error=1.0e-3, max_nodes=48))
    assert rule.sampled_max_error <= 1.0e-3
    assert rule.node_count <= 48
    expected = 1.0e6 / float(np.sum(masses))
    assert abs(rule.excluded_mass_fraction - expected) <= 1.0e-12 * expected


# ---------------------------------------------------------------------------
#  support_arc
# ---------------------------------------------------------------------------

def test_support_arc_reports_the_hand_built_radial_and_angular_extent():
    dens = np.array([2.0 * np.exp(-0.3j),
                     9.0 * np.exp(0.1j),
                     5.0 * np.exp(-0.6j)])
    problem = ReciprocalMeasureProblem(
        frequencies=np.array([0.0]),
        internal_sums=-dens,             # d = dens exactly
        cell_masses=np.ones(3))
    radial_min, radial_max, center, width = support_arc(problem)
    assert abs(radial_min - 2.0) <= 1.0e-9
    assert abs(radial_max - 9.0) <= 1.0e-9
    assert abs(width - 0.7) <= 1.0e-9    # angles span [-0.6, 0.1]
    assert abs(center - (-0.25)) <= 1.0e-9


def test_support_arc_refuses_an_emptied_support_by_name_and_serves_the_shrunk_radius():
    dens = np.array([2.0 * np.exp(-0.3j),
                     9.0 * np.exp(0.1j),
                     5.0 * np.exp(-0.6j)])
    emptied = ReciprocalMeasureProblem(
        frequencies=np.array([0.0]),
        internal_sums=-dens,
        cell_masses=np.ones(3),
        excluded_radius=20.0)
    with pytest.raises(ValueError) as excinfo:
        support_arc(emptied)
    assert "excluded region" in str(excinfo.value)
    sibling = ReciprocalMeasureProblem(
        frequencies=np.array([0.0]),
        internal_sums=-dens,
        cell_masses=np.ones(3),
        excluded_radius=3.0)             # drops only the |d| = 2 cell
    radial_min, _, _, _ = support_arc(sibling)
    assert abs(radial_min - 5.0) <= 1.0e-9


# ---------------------------------------------------------------------------
#  the candidate dictionary
# ---------------------------------------------------------------------------

def test_the_candidate_dictionary_is_deterministic_label_sorted_and_growth_capped():
    u = np.concatenate([-np.geomspace(0.5, 8.0, 6)[::-1],
                        np.geomspace(0.5, 8.0, 6)])
    problem = ReciprocalMeasureProblem(
        frequencies=np.array([0.0]),
        internal_sums=-u - 1.0j * 0.5,
        cell_masses=np.ones(u.size))
    options = ComplexTimeSearchOptions(target_error=1.0e-3, growth_cap=30.0)
    times, families = candidate_time_dictionary(problem, options)
    again_times, again_families = candidate_time_dictionary(problem, options)
    assert np.array_equal(times, again_times)
    assert families == again_families
    assert set(families) <= {"ray", "oscillatory", "short_time"}
    assert list(families) == sorted(families)
    labels = np.asarray(families)
    for label in set(families):
        group = np.abs(times[labels == label])
        assert np.all(np.diff(group) >= -1.0e-12)
    kept, _, _ = problem.retained()
    d = problem.denominators[kept & (problem.cell_masses[None, :] > 0.0)]
    growth = np.max(-(times[None, :] * d[:, None]).imag, axis=0)
    assert np.all(growth <= 30.0 + 1.0e-9)


# ---------------------------------------------------------------------------
#  the red twin
# ---------------------------------------------------------------------------

def test_an_anticausal_twin_of_the_crossing_support_is_refused_on_growth_or_fitted_with_bounded_atoms():
    """gamma -> -gamma of the crossing case: growth in the decay direction.

    The contract gives one of two discriminating signatures, and this cell
    asserts whichever arrives: either the candidate growth cap refuses by
    name, or the fit still meets the target with every retained atom's
    magnitude bounded by e^30 on the kept support — never a silent third
    thing.
    """
    problem = _crossing_problem(gamma_sign=-1.0)
    options = ComplexTimeSearchOptions(target_error=1.0e-3, max_nodes=48,
                                       fit_frequency_count=6,
                                       growth_cap=30.0)
    try:
        rule = fit_reciprocal_measure(problem, options)
    except RuntimeError as refusal:
        assert "growth_cap" in str(refusal) or "growth cap" in str(refusal)
        return
    assert rule.sampled_max_error <= 1.0e-3
    assert set(rule.node_families) <= FAMILY_LABELS
    kept, _, _ = problem.retained()
    d = problem.denominators[kept & (problem.cell_masses[None, :] > 0.0)]
    for node in rule.time_nodes:
        log_magnitude = float(np.max(-(node * d).imag))
        assert log_magnitude <= 30.0 + 1.0e-9
