"""Measure-weighted ROQ pathway: determinism, targets, branch fit, gate."""
import numpy as np
import pytest

from minimax import (ReciprocalMeasureProblem, RoqGroup, branch_noise_gate,
                     delivered_error, fit_roq_branch, fit_roq_group)


def _measure(rng, n, lo, hi, width_floor):
    real = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    imag = -(width_floor + rng.gamma(2.0, 0.05, n))
    return real + 1j * imag, rng.random(n) * np.exp(-rng.uniform(0.0, 8.0, n))


@pytest.fixture(scope="module")
def groups():
    rng = np.random.default_rng(3)
    omega = np.linspace(0.0, 0.4, 9)
    def build(name, n_fit, n_val, lo, hi, angle, horizon):
        fit = ReciprocalMeasureProblem(omega, *_measure(rng, n_fit, lo, hi, 0.02))
        val = ReciprocalMeasureProblem(omega, *_measure(rng, n_val, lo, hi, 0.02))
        return RoqGroup(name, fit, val, sigma=1, angle_deg=angle,
                        horizon=horizon)
    return (build("tails", 400, 900, 1.0, 8.0, -55.0, 60.0),
            build("resonant", 300, 700, 0.05, 0.8, 0.0, 300.0))


def test_group_fit_meets_target_and_is_deterministic(groups):
    tails, _ = groups
    first = fit_roq_group(tails, 1.0e-4, ranks=range(6, 41, 2))
    second = fit_roq_group(tails, 1.0e-4, ranks=range(6, 41, 2))
    assert first.max_error <= 1.0e-4
    assert first.kappa_p99 < 10.0
    np.testing.assert_array_equal(first.times, second.times)
    np.testing.assert_array_equal(first.weights, second.weights)


def test_rule_scores_with_production_metric(groups):
    tails, _ = groups
    rule = fit_roq_group(tails, 1.0e-4, ranks=range(6, 41, 2))
    error, _ = delivered_error(tails.validation, rule.times, rule.weights)
    assert float(np.max(error)) == rule.max_error


def test_branch_joint_fit_and_noise_gate(groups):
    tails, resonant = groups
    seed_tails = fit_roq_group(tails, 1.0e-4, ranks=range(6, 41, 2))
    seed_res = fit_roq_group(resonant, 5.0e-4, ranks=range(8, 49, 2))
    rules, branch = fit_roq_branch([tails, resonant], [seed_tails, seed_res])
    assert float(np.max(branch)) <= 5.0e-4
    passed, effective = branch_noise_gate([tails, resonant], rules, 1.0e-3)
    assert passed and effective < 100.0


def test_wrong_half_plane_is_loud(groups):
    tails, _ = groups
    wrong = RoqGroup("wrong", tails.fit, tails.validation, sigma=-1,
                     angle_deg=-55.0, horizon=60.0)
    with pytest.raises(ValueError, match="grow"):
        fit_roq_group(wrong, 1.0e-4, ranks=[10, 14])
