import numpy as np
import pytest

from minimax.measure_windows import (
    apportion_true_error,
    partition_measure_windows,
)


def test_partition_isolates_crossing_and_conserves_membership_and_mass():
    support = np.array([
        -8.0 + 0.4j, -3.0 + 0.5j,
        1.0 - 0.3j, 4.0 - 0.7j,
        7.0 - 0.4j, 12.0 - 0.8j, 30.0 - 1.2j,
    ])
    masses = np.array([1.0, 2.0, 3.0, 1.0, 5.0, 2.0, 4.0])
    windows = partition_measure_windows(
        support, masses, np.linspace(0.0, 5.0, 11), window_count=5)

    assert 2 <= len(windows) <= 5
    crossing = [window for window in windows if window.kind == "crossing"]
    assert len(crossing) == 1
    assert np.array_equal(np.sort(crossing[0].member_indices), [2, 3])
    members = np.concatenate([window.member_indices for window in windows])
    assert np.array_equal(np.sort(members), np.arange(support.size))
    assert sum(window.delivered_mass for window in windows) == pytest.approx(
        masses.sum())
    assert all(np.isfinite(window.scale_span) and window.scale_span >= 1.0
               for window in windows)


def test_sign_definite_support_uses_nonempty_mass_quantiles():
    support = -np.geomspace(1.0, 100.0, 12) + 0.5j
    masses = np.arange(1.0, 13.0)
    windows = partition_measure_windows(
        support, masses, np.linspace(0.0, 5.0, 11), window_count=3)

    assert len(windows) == 3
    assert all(window.kind == "below" for window in windows)
    assert all(window.member_indices.size > 0 for window in windows)
    assert np.array_equal(
        np.sort(np.concatenate([window.member_indices for window in windows])),
        np.arange(support.size))


def test_true_error_budget_is_mass_times_measured_difficulty():
    support = -np.geomspace(1.0, 30.0, 8) + 0.5j
    masses = np.ones(8)
    windows = partition_measure_windows(
        support, masses, np.linspace(0.0, 5.0, 11), window_count=2)
    difficulty = np.array([1.0, 3.0])
    budgets = apportion_true_error(windows, difficulty, 2.0e-4)

    expected_score = np.array(
        [window.delivered_mass for window in windows]) * difficulty
    actual = np.array([budget.absolute_error_budget for budget in budgets])
    assert actual.sum() == pytest.approx(2.0e-4)
    assert actual / actual.sum() == pytest.approx(expected_score / expected_score.sum())


@pytest.mark.parametrize("window_count", [1, 6])
def test_window_count_refuses_outside_small_planner_contract(window_count):
    with pytest.raises(ValueError, match="window_count"):
        partition_measure_windows(
            np.array([-2.0 + 0.2j, 8.0 - 0.2j]), np.ones(2),
            np.linspace(0.0, 5.0, 3), window_count=window_count)
