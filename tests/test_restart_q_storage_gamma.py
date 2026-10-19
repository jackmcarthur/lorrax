"""restart_q_storage = auto at a q grid that does not reduce resolves full.

At Gamma only (n_q = 1) the IBZ wedge is the whole zone.  Before the fix the
auto arm returned ``ibz`` on an orbit-closed set, and the W0 writer then
refused for want of a pre-unfold capture (run30k try 8, Cr:Al2O3 N_mu 29998).
"""
from types import SimpleNamespace

import pytest

from gw.restart_q_storage import resolve_restart_q_storage


def _resolution(use_ibz=True):
    verdict = SimpleNamespace(closed=True, worst_residual=0.0, tol=1e-5,
                              n_violating=0, n_sym=12)
    return SimpleNamespace(use_ibz=use_ibz, verdict=verdict, reason="")


@pytest.mark.parametrize("n_full", [1, 8])
def test_auto_full_when_q_axis_does_not_reduce(n_full):
    d = resolve_restart_q_storage("auto", _resolution(), context="test",
                                  q_counts=(n_full, n_full))
    assert d.mode == "full" and not d.store_wedge
    assert d.resolution is None
    assert "does not reduce" in d.reason


def test_auto_ibz_when_q_axis_reduces():
    d = resolve_restart_q_storage("auto", _resolution(), context="test",
                                  q_counts=(4, 64))
    assert d.mode == "ibz" and d.store_wedge


def test_auto_without_counts_keeps_previous_behaviour():
    d = resolve_restart_q_storage("auto", _resolution(), context="test")
    assert d.mode == "ibz"


def test_explicit_ibz_unchanged_at_gamma():
    d = resolve_restart_q_storage("ibz", _resolution(), context="test",
                                  q_counts=(1, 1))
    assert d.mode == "ibz"
