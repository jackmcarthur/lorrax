"""Davidson's ``conv=k/n`` report must count states, not a leading prefix.

``solvers.davidson.davidson`` reported convergence with an inline loop that
walked from state 0 and BROKE at the first unconverged state.  That is a prefix
count wearing a total count's label: a solve with state 0 still moving and
states 1..n-1 converged printed ``conv=0/20``, identical to a solve in which
nothing had converged at all.  The convergence census hit exactly this — its
``--solver davidson`` arm printed ``WARNING: did not converge in 200
iterations. Best: 0/20`` on a run whose eigenvalues were already 1.4 ueV from
the exact 1024-dim dense reference, and the report was read as evidence the
solver had stalled.

THE FIX IS A COUNTER, NOT AN ALGORITHM CHANGE, and section 2 below is what
makes that claim checkable instead of asserted: the only two uses of the count
are ``n_conv == n_eig``, and prefix and total counts agree on that predicate
for every possible convergence pattern.  So the iteration at which Davidson
stops is unchanged, which is what lets a concurrent benchmark lane's numbers
survive this commit.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

import harness  # noqa: F401  (puts src/ on sys.path)

pytest.importorskip("jax")

from solvers.davidson import _count_converged  # noqa: E402


# ---------------------------------------------------------------------------
# 1. THE RED TWIN -- these cases are exactly the ones the prefix count got wrong
# ---------------------------------------------------------------------------

def _prefix_count(conv, n_eig):
    """The OLD rule, verbatim, kept executable so the twin is a real twin."""
    n_conv = 0
    for i in range(n_eig):
        if conv[i]:
            n_conv = i + 1
        else:
            break
    return n_conv


@pytest.mark.parametrize("conv,expected,old", [
    # the census's case: everything converged except the first state
    ([False, True, True, True], 3, 0),
    # one straggler in the middle
    ([True, True, False, True], 3, 2),
    # only the last state left
    ([True, True, True, False], 3, 3),
    # nothing converged -- the two rules agree here, and both say 0
    ([False, False, False, False], 0, 0),
    # everything converged -- the two rules agree here too, and that agreement
    # is what makes the exit test unchanged (section 2)
    ([True, True, True, True], 4, 4),
])
def test_counts_states_not_a_leading_prefix(conv, expected, old):
    conv = np.asarray(conv)
    n_eig = len(conv)
    got = _count_converged(conv, n_eig)
    assert got == expected, (
        f"_count_converged{tuple(conv)} = {got}, expected {expected}")
    if old != expected:
        # This is the twin half: assert the OLD rule really did get it wrong,
        # so the cell above cannot be satisfied by a no-op change.
        assert _prefix_count(conv, n_eig) == old, (
            "the recorded old-rule answer no longer reproduces; the twin has "
            "drifted from the code it is twinning")
        assert _prefix_count(conv, n_eig) != expected, (
            f"the prefix rule already returned {expected} for {tuple(conv)} — "
            f"this case does not discriminate and should not be in the table")


def test_the_census_line_is_now_interpretable():
    """The specific report that was misread, end to end.

    19 of 20 states converged, state 0 still moving.  The old line said 0/20
    and read as a total stall; the new one says 19/20 and reads as one
    straggler.
    """
    conv = np.array([False] + [True] * 19)
    assert _prefix_count(conv, 20) == 0
    assert _count_converged(conv, 20) == 19


def test_respects_n_eig_and_ignores_trailing_states():
    """Only the REQUESTED states count.

    ``conv`` is computed over the ``n_eig`` lowest Ritz values, but nothing
    stops a caller passing a longer array; the count must not silently include
    states the user did not ask for.
    """
    conv = np.array([True, True, False, True, True])
    assert _count_converged(conv, 2) == 2
    assert _count_converged(conv, 3) == 2
    assert _count_converged(conv, 5) == 4


# ---------------------------------------------------------------------------
# 2. THE ALGORITHM IS UNTOUCHED -- exhaustively, not by assertion
# ---------------------------------------------------------------------------

def test_exit_predicate_is_identical_for_every_convergence_pattern():
    """``n_conv == n_eig`` must agree between the old and new rules ALWAYS.

    That predicate is the early return and the print cadence -- the only two
    consumers of the count in ``davidson()``.  If it agrees on every pattern
    then no run can stop at a different iteration than it did before, which is
    the whole safety argument for changing a number a benchmark lane is
    concurrently reading.  Exhaustive over all 2^n patterns up to n=10.
    """
    for n_eig in range(1, 11):
        for pattern in itertools.product([False, True], repeat=n_eig):
            conv = np.asarray(pattern)
            old_fires = (_prefix_count(conv, n_eig) == n_eig)
            new_fires = (_count_converged(conv, n_eig) == n_eig)
            assert old_fires == new_fires, (
                f"exit predicate DIVERGED at n_eig={n_eig} pattern={pattern}: "
                f"old={old_fires} new={new_fires} — this change is no longer "
                f"reporting-only and must not land as such")
            # and the predicate is exactly "all requested states converged"
            assert new_fires == bool(conv.all())


def test_new_rule_is_never_smaller_than_the_old_one():
    """The report can only become MORE optimistic, never less.

    A prefix of converged states is a subset of the converged states, so the
    new number is >= the old one for every pattern.  Anyone comparing a log
    from before this commit with one from after can rely on that direction.
    """
    for n_eig in range(1, 9):
        for pattern in itertools.product([False, True], repeat=n_eig):
            conv = np.asarray(pattern)
            assert _count_converged(conv, n_eig) >= _prefix_count(conv, n_eig)
