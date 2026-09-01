"""Measure-weighted ROQ pathway: determinism, targets, branch fit, gate."""
import json
from pathlib import Path

import numpy as np
import pytest

from minimax import (ReciprocalMeasureProblem, RoqGroup, RoqWindow,
                     branch_noise_gate, delivered_error, fit_roq_branch,
                     fit_roq_group, plan_measure_adapted_roq)


_FROZEN_NA = Path(
    "/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/"
    "runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/"
    "evidence/causal_hankel/na_reconstructed_problems_v1.npz")
_FROZEN_WINDOWS = Path(
    "/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/"
    "reports/tr_broken_gnppm_and_new_minimax_2026-08-28/"
    "hankel_agent_export/na_measures/windows.json")


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


def test_production_rank_interpolates_and_accepts_exact_search_score(
        groups, monkeypatch):
    from minimax import roq_fit

    target = 1.0e-4
    calls = []

    def fake_fit(group, prepared, rank, **kwargs):
        quick = bool(kwargs.get("quick"))
        calls.append((rank, quick))
        error = target * np.exp(0.2 * (40 - rank))
        return roq_fit.RoqRule(
            group.name, np.arange(rank), np.ones(rank), rank, error,
            2.0, 2.0, 1.0, target_met=error <= target,
            noise_passed=True)

    monkeypatch.setattr(roq_fit, "_rank_ceiling", lambda group: 100)
    monkeypatch.setattr(roq_fit, "_prepare_subspace",
                        lambda group, base_nodes, max_rank: None)
    monkeypatch.setattr(roq_fit, "_fit_prepared", fake_fit)
    angle_probe = fake_fit(groups[0], None, 12, quick=True)
    angle_probe = roq_fit.RoqRule(
        **{**angle_probe.__dict__, "target_met": False})

    rule = roq_fit._fit_production_rank(
        groups[0], target, ("window",), angle_probe)

    assert rule.rank == 40
    assert rule.max_error == target
    assert all(quick for _, quick in calls[1:])


def test_quick_rank_fit_defers_amplification(groups, monkeypatch):
    from minimax import roq_fit

    group = groups[0]
    prepared = roq_fit._prepare_subspace(group, 32, 8)
    calls = []
    monkeypatch.setattr(
        roq_fit, "rule_amplification",
        lambda *args: calls.append(args) or (2.0, 3.0))

    searched = roq_fit._fit_prepared(
        group, prepared, 8, target=1.0e-3, quick=True)
    assert calls == []
    assert np.isnan(searched.kappa_p99)
    assert not searched.noise_passed

    scored = roq_fit._score(
        group, searched.times, searched.weights, searched.rank,
        searched.singular_ratio, target=1.0e-3)
    assert len(calls) == 1
    assert scored.kappa_p99 == 2.0
    assert scored.kappa_max == 3.0


def test_production_planner_refuses_a_growing_product_window(groups):
    tails, _ = groups
    window = RoqWindow("wrong", tails.fit, tails.validation, 1.0e-4,
                       "branch", sigma=-1)
    with pytest.raises(ValueError, match="grow"):
        plan_measure_adapted_roq([window], eta=0.02)


def _load_frozen_na():
    if not (_FROZEN_NA.exists() and _FROZEN_WINDOWS.exists()):
        pytest.skip("frozen Na measure export is not mounted")
    registry = json.loads(_FROZEN_WINDOWS.read_text())
    windows = []
    with np.load(_FROZEN_NA, allow_pickle=False) as data:
        for index, row in enumerate(registry["windows"]):
            frequency = data[f"p{index}_frequencies"]
            fit = ReciprocalMeasureProblem(
                frequency, data[f"p{index}_internal"],
                data[f"p{index}_mass"])
            validation = ReciprocalMeasureProblem(
                frequency, data[f"p{index}_validation_internal"],
                data[f"p{index}_validation_mass"])
            windows.append(RoqWindow(
                row["name"], fit, validation,
                row["relative_residual_target"], row["branch"],
                row["sigma_half_plane_sign"]))
    return tuple(windows), float(registry["eta_ry"])


@pytest.fixture(scope="module")
def frozen_na_plans():
    windows, eta = _load_frozen_na()
    return (plan_measure_adapted_roq(windows, eta),
            plan_measure_adapted_roq(windows, eta))


def test_frozen_na_angle_selection_and_branch_consolidation(frozen_na_plans):
    plan, _ = frozen_na_plans
    valence = next(row for row in plan.branches if row.branch == "val")
    conduction = next(row for row in plan.branches if row.branch == "cond")
    assert valence.strategy == "whole_branch"
    assert valence.node_count == 12
    assert conduction.strategy == "decay_compatible"
    # Angles come from the scanned grid, which is anchored near the imaginary
    # axis: node count is flat from -70 to -90 deg on this support while kappa
    # falls to 1.00 at -85, so pinning one exact triple would freeze a choice
    # the measurement says is a plateau.  Assert membership and the property
    # that matters instead.
    from minimax.roq_fit import _ANGLE_SCAN_DEG
    assert all(rule.angle_deg in _ANGLE_SCAN_DEG for rule in plan.rules)
    valence_rule = next(rule for rule in plan.rules if len(rule.windows) == 3)
    assert valence_rule.angle_deg < 0.0


def test_frozen_na_plan_is_bit_deterministic(frozen_na_plans):
    first, second = frozen_na_plans
    assert len(first.rules) == len(second.rules)
    for left, right in zip(first.rules, second.rules):
        assert (left.group, left.rank, left.angle_deg, left.horizon,
                left.max_error, left.kappa_p99, left.windows) == (
                    right.group, right.rank, right.angle_deg, right.horizon,
                    right.max_error, right.kappa_p99, right.windows)
        np.testing.assert_array_equal(left.times, right.times)
        np.testing.assert_array_equal(left.weights, right.weights)


def test_frozen_na_node_accuracy_and_noise_acceptance(frozen_na_plans):
    plan, _ = frozen_na_plans
    assert sum(rule.rank for rule in plan.rules) == 54
    assert sum(rule.rank for rule in plan.rules) <= 69
    achieved = {row.branch: row.max_error for row in plan.branches}
    # Actual aggregate errors of the accepted 137-node production plan.
    assert achieved["cond"] <= 1.35633e-4
    assert achieved["val"] <= 2.43017e-5
    assert all(row.noise_passed for row in plan.branches)
    assert all(rule.noise_passed for rule in plan.rules)
    assert all(rule.kappa_p99 * 6.0e-8
               <= 0.05 * max(rule.max_error, 1.0e-12)
               for rule in plan.rules)
