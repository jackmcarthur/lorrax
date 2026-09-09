"""Shared-pole Σ metadata gates; GPU synthesis gates live in the lane harness."""

import numpy as np
import pytest

from gw.mpa.sigma_windows import (shared_pole_frequencies,
                                  shared_pole_intervals)


def test_shared_pole_partition_keeps_boundary_multiplet_and_ragged_tail():
    poles2 = np.asarray([[1., 4., 4., 9., 1.], [4., 16., 1., 1., 1.],
                         [1., 1., 1., 1., 1.]])
    frequencies = shared_pole_frequencies(
        poles2, np.asarray([4, 2, 0], dtype=np.int64))
    parents = np.arange(3, dtype=np.int64)
    lower = np.tile([0, 2, -np.inf, -np.inf, np.inf, np.inf], (3, 1))
    upper = lower.copy()
    upper[:, :2] = [2, np.inf]
    left = shared_pole_intervals(frequencies, parents, lower)
    right = shared_pole_intervals(frequencies, parents, upper)
    np.testing.assert_array_equal(left, [[0, 3], [0, 1], [0, 0]])
    np.testing.assert_array_equal(right, [[3, 4], [1, 2], [0, 0]])
    # Every active column has exactly one owner; sentinels have none.
    for parent, omega in enumerate(frequencies):
        visits = np.zeros(len(omega), dtype=int)
        for lo, hi in (left[parent], right[parent]):
            visits[lo:hi] += 1
        np.testing.assert_array_equal(visits, np.ones(len(omega), dtype=int))


@pytest.mark.parametrize("poles2,counts", [
    ([[4., 1.]], [2]),              # sortedness cannot be repaired in Σ
    ([[1., 0.]], [1]),              # unsafe padding
    ([[1., np.nan]], [1]),
    ([[1., -1.]], [2]),
    ([[1., 2.]], [3]),
    ([[1., 2.]], [-1]),
])
def test_shared_pole_metadata_refuses_corruption(poles2, counts):
    with pytest.raises(ValueError):
        shared_pole_frequencies(np.asarray(poles2, dtype=np.float64),
                                np.asarray(counts, dtype=np.int64))


def test_shared_pole_selector_refuses_missing_or_duplicate_parent():
    frequencies = (np.asarray([1., 2.]),)
    bounds = np.tile([0, np.inf, -np.inf, -np.inf, np.inf, np.inf], (2, 1))
    for indices in (np.asarray([0, 1]), np.asarray([0, 0])):
        with pytest.raises(ValueError):
            shared_pole_intervals(frequencies, indices, bounds)


def test_shared_pole_census_has_no_frozen_na_ceiling():
    frequencies = shared_pole_frequencies(
        np.asarray([[400.]], dtype=np.float64), np.asarray([1], dtype=np.int64))
    np.testing.assert_array_equal(frequencies[0], [20.])
