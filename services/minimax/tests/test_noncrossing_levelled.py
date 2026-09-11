"""Levelled noncrossing rules: minimal N, optimality certificate, reference values.

Reference numbers: runs/DEV/326_minimax_fit_review_2026-09-11/lanes/noncrossing
(certified reference step 58191974.24; mpmath 30-digit check step 58191974.31).
"""
import numpy as np
import pytest

from minimax import certify_noncrossing, noncrossing_levelled
from minimax import _catalog


@pytest.mark.parametrize("R, eps, N_min", [
    (10.0, 1e-8, 8),
    (100.0, 1e-6, 9),
    (1000.0, 1e-8, 15),
    (1e4, 1e-10, 23),
    (1e6, 1e-6, 15),
])
def test_smallest_levelled_rule(R, eps, N_min):
    tau, w, N, err = noncrossing_levelled(R, eps)
    assert N == N_min
    assert err <= eps
    lower, upper = certify_noncrossing(tau, w, R)
    assert upper == pytest.approx(err, rel=1e-12)
    assert upper / lower - 1.0 < 1e-3 or upper - lower < 1e-13      # levelled: optimal at this N
    assert np.all(tau > 0) and np.all(w > 0)                         # the 1/x minimax is positive
    # N is minimal: the levelled N-1 rule misses eps (its alternation minimum exceeds eps)
    t1, w1, n1, _e1 = noncrossing_levelled(R, 1e-300, N_max=N - 1)
    low1, up1 = certify_noncrossing(t1, w1, R)
    assert n1 == N - 1 and low1 > eps


@pytest.mark.parametrize("R, eps", [(8544.8, 6.072e-7), (28860.0, 4.987e-10), (7231.9, 8.283e-11)])
def test_two_node_step_that_misses_the_basin_recovers(R, eps):
    """Regression: the prototype lost alternation on these requests at an
    N -> N + 2 step (benchmark step 58191974.38)."""
    tau, w, N, err = noncrossing_levelled(R, eps)
    assert err <= eps
    lower, upper = certify_noncrossing(tau, w, R)
    assert upper / lower - 1.0 < 1e-3 or upper - lower < 1e-13
    t1, w1, n1, _e1 = noncrossing_levelled(R, 1e-300, N_max=N - 1)
    assert n1 == N - 1 and certify_noncrossing(t1, w1, R)[0] > eps


def test_matches_30_digit_reference():
    # E_8(10) = 1.71639260855e-9 by a 30-digit Newton-Remez (mpmath)
    tau, w, N, err = noncrossing_levelled(10.0, 1e-300, N_max=8)
    assert N == 8
    assert err == pytest.approx(1.71639260855e-9, rel=1e-6)


def test_certificate_detects_an_unlevelled_rule():
    """Negative control: the shipped R=10, N=7 table meets its bound but is not
    the minimax rule; the certificate must say so (upper/lower ~ 5)."""
    view = _catalog.catalog_view()
    entry = next(e for e in view.entries if e.family == "noncrossing" and e.range_max == 10.0)
    tau, alpha, *_ = _catalog.load_table(entry)
    lower, upper = certify_noncrossing(tau, alpha, 10.0)
    assert upper / lower > 2.0
    t7, w7, n7, e7 = noncrossing_levelled(10.0, 1e-300, N_max=7)
    assert n7 == 7 and e7 < upper / 3.0
