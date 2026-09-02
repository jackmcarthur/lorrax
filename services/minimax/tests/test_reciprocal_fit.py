"""``minimax.reciprocal_fit`` through the door: refusals, metric, LP contract.

Everything here is deterministic — explicit grids, no random state — and
laptop-fast.  The refusal cells follow the service rule that every check
ships with the case where it returns FALSE: each ``pytest.raises`` asserts
the message names the offending quantity and sits beside a sibling that
differs only in that quantity and is accepted.

The delivered-error cell recomputes the documented metric by hand, in
plain ``cmath`` arithmetic, and holds the module to it at 1e-12 relative:
``error_i = sum_j(mass_j |Q(d_ij) - 1/d_ij|) / sum_j(mass_j / |d_ij|)``
over kept cells, with ``Q(d) = sum(weights * exp(1j * time_nodes * d))``.
"""

import cmath

import numpy as np
import pytest

from minimax import (
    ReciprocalMeasureProblem,
    delivered_error,
    evaluate_rule,
    rule_amplification,
    solve_fixed_time_weights,
)

_FREQ = np.array([1.0, 2.5])
_SUMS = np.array([0.5 - 0.4j, -1.0 + 0.3j, 3.0 - 1.0j])
_MASS = np.array([0.7, 1.3, 0.4])


# ---------------------------------------------------------------------------
#  ReciprocalMeasureProblem refusals, each with its passing sibling
# ---------------------------------------------------------------------------

def test_empty_frequencies_are_refused_by_name_and_a_single_frequency_is_accepted():
    with pytest.raises(ValueError) as excinfo:
        ReciprocalMeasureProblem(np.array([]), _SUMS, _MASS)
    assert "frequencies" in str(excinfo.value)
    sibling = ReciprocalMeasureProblem(np.array([1.0]), _SUMS, _MASS)
    assert sibling.denominators.shape == (1, 3)


def test_an_empty_support_is_refused_by_name_and_a_single_cell_support_is_accepted():
    with pytest.raises(ValueError) as excinfo:
        ReciprocalMeasureProblem(_FREQ, np.array([], dtype=np.complex128),
                                 np.array([]))
    assert "internal_sums" in str(excinfo.value)
    sibling = ReciprocalMeasureProblem(_FREQ, np.array([0.5 - 0.4j]),
                                       np.array([1.0]))
    assert sibling.denominators.shape == (2, 1)


def test_a_sums_masses_shape_mismatch_is_refused_by_name_and_matching_shapes_are_accepted():
    with pytest.raises(ValueError) as excinfo:
        ReciprocalMeasureProblem(_FREQ, _SUMS, _MASS[:2])
    text = str(excinfo.value)
    assert "internal_sums" in text and "cell_masses" in text
    sibling = ReciprocalMeasureProblem(_FREQ, _SUMS, _MASS)
    assert sibling.cell_masses.shape == sibling.internal_sums.shape


def test_a_negative_cell_mass_is_refused_by_name_and_a_zero_mass_beside_a_positive_one_is_accepted():
    with pytest.raises(ValueError) as excinfo:
        ReciprocalMeasureProblem(_FREQ, _SUMS, np.array([0.7, -1.3, 0.4]))
    assert "cell_masses" in str(excinfo.value)
    sibling = ReciprocalMeasureProblem(_FREQ, _SUMS, np.array([0.7, 0.0, 0.4]))
    assert float(np.sum(sibling.cell_masses)) > 0.0


def test_all_zero_cell_masses_are_refused_by_name_and_a_positive_total_is_accepted():
    with pytest.raises(ValueError) as excinfo:
        ReciprocalMeasureProblem(_FREQ, _SUMS, np.zeros(3))
    assert "cell_masses" in str(excinfo.value)
    sibling = ReciprocalMeasureProblem(_FREQ, _SUMS, np.array([0.0, 0.0, 1.0]))
    assert float(np.sum(sibling.cell_masses)) > 0.0


def test_nonfinite_entries_in_any_problem_array_are_refused_and_the_finite_arrays_are_accepted():
    with pytest.raises(ValueError) as excinfo:
        ReciprocalMeasureProblem(np.array([1.0, np.nan]), _SUMS, _MASS)
    assert "finite" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        ReciprocalMeasureProblem(_FREQ, np.array([0.5, np.inf + 0.0j, 3.0]),
                                 _MASS)
    assert "finite" in str(excinfo.value)
    sibling = ReciprocalMeasureProblem(_FREQ, _SUMS, _MASS)
    assert np.all(np.isfinite(sibling.denominators))


def test_a_negative_or_infinite_excluded_radius_is_refused_by_name_and_a_finite_one_is_accepted():
    with pytest.raises(ValueError) as excinfo:
        ReciprocalMeasureProblem(_FREQ, _SUMS, _MASS, excluded_radius=-0.1)
    assert "excluded_radius" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        ReciprocalMeasureProblem(_FREQ, _SUMS, _MASS, excluded_radius=np.inf)
    assert "excluded_radius" in str(excinfo.value)
    sibling = ReciprocalMeasureProblem(_FREQ, _SUMS, _MASS,
                                       excluded_radius=0.25)
    assert sibling.excluded_radius == 0.25


# ---------------------------------------------------------------------------
#  evaluate_rule and delivered_error against hand arithmetic
# ---------------------------------------------------------------------------

def test_evaluate_rule_matches_the_hand_computed_exponential_sum_at_explicit_points():
    times = np.array([0.4 + 0.0j, -1.1 + 0.2j])
    weights = np.array([0.3 - 0.2j, 1.1 + 0.7j])
    dens = np.array([1.5 - 0.4j, -2.0 + 0.9j, 0.3 + 3.0j])
    values = evaluate_rule(times, weights, dens)
    for k, d in enumerate(dens):
        hand = sum(w * cmath.exp(1j * t * complex(d))
                   for t, w in zip(times, weights))
        assert abs(values[k] - hand) <= 1.0e-12 * abs(hand)


def test_delivered_error_reproduces_the_hand_computed_metric_for_a_one_node_rule():
    problem = ReciprocalMeasureProblem(_FREQ, _SUMS, _MASS)
    t0 = 0.35 - 0.15j
    w0 = 1.2 + 0.5j
    error, excluded = delivered_error(
        problem, np.array([t0]), np.array([w0]))
    assert error.shape == _FREQ.shape
    for i, frequency in enumerate(_FREQ):
        numerator = 0.0
        delivered_mass = 0.0
        for s, m in zip(_SUMS, _MASS):
            d = complex(frequency) - complex(s)
            numerator += m * abs(w0 * cmath.exp(1j * t0 * d) - 1.0 / d)
            delivered_mass += m / abs(d)
        hand = numerator / delivered_mass
        assert abs(error[i] - hand) <= 1.0e-12 * hand
        assert excluded[i] == 0.0


# ---------------------------------------------------------------------------
#  solve_fixed_time_weights: the zero-weight-sum equality
# ---------------------------------------------------------------------------

def _resolvent_problem(zero_weight_sum: bool) -> ReciprocalMeasureProblem:
    """Cells ``1/(u + 0.5j)`` with ``u`` of both signs, one frequency."""
    u = np.concatenate([-np.geomspace(0.5, 8.0, 6)[::-1],
                        np.geomspace(0.5, 8.0, 6)])
    return ReciprocalMeasureProblem(
        frequencies=np.array([0.0]),
        internal_sums=-u - 1.0j * 0.5,
        cell_masses=np.ones(u.size),
        zero_weight_sum=zero_weight_sum)


_ZERO_SUM_GRID = np.concatenate(
    [np.geomspace(0.05, 4.0, 10),
     -np.geomspace(0.05, 4.0, 10)]).astype(np.complex128)


def test_zero_weight_sum_forces_the_returned_weights_to_cancel_at_the_origin():
    weights, _ = solve_fixed_time_weights(
        _resolvent_problem(zero_weight_sum=True), _ZERO_SUM_GRID,
        objective_scale=1.0e3)
    assert np.all(np.isfinite(weights))
    assert abs(np.sum(weights)) <= 1.0e-8 * float(np.sum(np.abs(weights)))


def test_without_zero_weight_sum_the_same_solve_does_not_cancel_at_the_origin():
    """The discriminating sibling: the equality above is not vacuous."""
    weights, _ = solve_fixed_time_weights(
        _resolvent_problem(zero_weight_sum=False), _ZERO_SUM_GRID,
        objective_scale=1.0e3)
    assert np.all(np.isfinite(weights))
    assert abs(np.sum(weights)) > 1.0e-6 * float(np.sum(np.abs(weights)))


# ---------------------------------------------------------------------------
#  rule_amplification structure
# ---------------------------------------------------------------------------

def test_rule_amplification_is_finite_with_p99_at_most_the_maximum_and_both_at_least_one():
    problem = ReciprocalMeasureProblem(
        frequencies=np.array([0.5, 1.5]),
        internal_sums=np.array([-2.0 + 0.3j, 1.0 + 0.3j, 3.0 - 0.5j]),
        cell_masses=np.array([1.0, 2.0, 0.5]))
    times = np.array([0.2, 0.9])
    weights = np.array([1.0 + 0.0j, -0.8 + 0.1j])
    p99, peak = rule_amplification(times, weights, problem)
    assert np.isfinite(p99) and np.isfinite(peak)
    assert 1.0 <= p99 <= peak
