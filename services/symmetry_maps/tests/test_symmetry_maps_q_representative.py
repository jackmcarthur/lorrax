"""Contracts for the symmetry service's BGW signed-q representatives."""

import numpy as np
import pytest

from symmetry_maps import (
    bgw_integer_q_to_fractional,
    bgw_signed_q_representative,
)


def test_bgw_signed_q_representative_keeps_positive_half_grid_tie():
    eps = np.finfo(np.float64).eps
    q = np.asarray([
        [0.0, 0.25, 0.5],
        [0.75, 1.0 - eps, 0.5],
    ])
    got = bgw_signed_q_representative(q)
    np.testing.assert_array_equal(
        got,
        np.asarray([[0.0, 0.25, 0.5], [-0.25, -eps, 0.5]]))


def test_fractional_and_integer_doors_share_the_bgw_tie_convention():
    q_int = np.asarray([[0, 2, 3], [3, 1, 2]])
    grid = np.asarray([4, 4, 4])
    np.testing.assert_array_equal(
        bgw_signed_q_representative(q_int / grid),
        bgw_integer_q_to_fractional(q_int, grid),
    )


@pytest.mark.parametrize("bad", [
    np.zeros((2, 2)),
    np.asarray([np.nan, 0.0, 0.0]),
    np.asarray([-0.6, 0.0, 0.0]),
    np.asarray([1.1, 0.0, 0.0]),
])
def test_bgw_signed_q_representative_rejects_nonstored_rows(bad):
    with pytest.raises(ValueError, match="bgw_signed_q_representative"):
        bgw_signed_q_representative(bad)
