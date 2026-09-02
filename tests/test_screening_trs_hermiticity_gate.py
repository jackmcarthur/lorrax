"""``_gate_w``'s Hermiticity check, and the frequency at which it is owed.

WHAT IS BEING PINNED.  ``W_q`` is Hermitian at ``omega = 0`` for any system.
At ``omega = i*omega_p`` it is Hermitian only when time reversal holds:
summed over both particle-hole orientations, a transition of gap ``Delta``
contributes ``[2w Im(P) - 2D Re(P)] / (w^2 + D^2)`` with ``P =
|rho_vc><rho_vc|``, and the first term is real ANTIsymmetric.  It cancels
pairwise between ``k`` and ``-k`` exactly under Theta, is odd in the
magnetisation, and carries the factor ``omega`` so it is identically zero at
``omega = 0``.  (Derivation and a random-state model:
``reports/four_current_head_frequency_audit_2026-09-01`` section F2 --
relative anti-Hermitian part 2.9e-1 at ``z=2i`` without Theta, 2e-16 with
it, 1e-17 at ``z=0`` either way.)

So the gate has to be conditional in exactly one cell of a two-by-two, and
this file is that two-by-two: {omega=0, i*omega_p} x {trs_allowed True,
False}.  Only ``(i*omega_p, False)`` may decline to call a residual a
defect, and even there it must print the number.

WHY THE RED TWIN IS THE TEST.  A gate that has been scoped is
indistinguishable from a gate that has been deleted unless the same input
still fails on the other side of the scope.  Every cell below therefore runs
ONE synthetic ``W`` through BOTH verdicts; the array never changes, only the
measured symmetry verdict does.  ``LORRAX_SANITY=strict`` is the discriminator
because it turns a defect into an exception and leaves a measurement alone.
"""
from __future__ import annotations

import numpy as np
import pytest

from common import sanity
from gw.screening import ScreeningRequest, _gate_w, _trs_verdict

#: omega = i*omega_p, the GN-PPM probe role's frequency (Ry).
PROBE = ScreeningRequest(2.0j, "probe")
STATIC = ScreeningRequest(0.0 + 0.0j, "static")
#: The real-axis HL probe: never Hermitian, never gated, either way.
REAL_AXIS = ScreeningRequest(2.0 + 0.0j, "hl_probe")


class _Log:
    def __init__(self):
        self.lines = []

    def __call__(self, *a, **k):
        self.lines.append(" ".join(str(x) for x in a))

    @property
    def text(self):
        return "\n".join(self.lines)

    @property
    def failures(self):
        return [ln for ln in self.lines if "LORRAX SANITY FAILURE" in ln]


def _w(anti: float) -> np.ndarray:
    """``(nq, mu, mu)`` W whose q=0 tile has anti-Hermitian part ``anti``.

    Only ``W[0]`` is gated, so the other q rows are Hermitian filler that
    keeps the array the production shape rather than a 1x1 scalar.  Real and
    antisymmetric is the SHAPE the TR-odd part actually has -- see the module
    docstring -- so the red twin is the physical component, not an arbitrary
    perturbation.
    """
    tile = np.eye(2, dtype=np.complex128)
    tile[0, 1] += anti
    tile[1, 0] -= anti
    return np.stack([tile, np.eye(2, dtype=np.complex128),
                     np.eye(2, dtype=np.complex128)])


#: 2e-1 relative -- three orders over the 1e-6 gate, and the order of
#: magnitude the random-state model produces at z=2i with Theta broken.
NON_HERMITIAN = _w(0.1)
HERMITIAN = _w(0.0)


def _run(W, req, trs_allowed, strict=True, monkeypatch=None):
    """``_gate_w`` under strict, returning ``(raised, log)``."""
    log = _Log()
    monkeypatch.setenv("LORRAX_SANITY", "strict" if strict else "warn")
    try:
        _gate_w(W, req, print_fn=log, trs_allowed=trs_allowed)
    except sanity.SanityError:
        return True, log
    return False, log


# ---------------------------------------------------------------------------
# Which tree is this file actually testing?
# ---------------------------------------------------------------------------

def test_this_file_is_testing_the_tree_it_was_launched_from():
    """The launcher's banner is not evidence about the import path.

    ``lx run`` announces ``[lx] source tree: $LORRAX_CHECKOUT/src`` and then
    exports ``PYTHONPATH=$LORRAX_ROOT/src`` from the BASE MODULE, so a payload
    can import a different checkout than the one the banner names -- measured
    2026-09-01 on this worktree, where a bare ``lx run`` canary resolved
    ``gw.screening`` under ``lorrax_A`` (KNOWN_SANDBOX_ERRORS.md).  A gate that
    passed under that condition is a statement about somebody else's source.

    So this cell answers the question in-band, from inside the same process
    that runs every other cell in this file, and it fails loudly rather than
    skipping when ``LORRAX_CHECKOUT`` is unset -- an unanswerable provenance
    question reported as a pass is the failure mode it exists to prevent.
    """
    import os

    from gw import screening

    want = os.environ.get("LORRAX_CHECKOUT")
    assert want, ("LORRAX_CHECKOUT is unset, so this file cannot say which "
                  "tree it tested; set it before trusting any verdict here")
    root = os.path.realpath(want) + os.sep
    for mod in (sanity, screening):
        assert os.path.realpath(mod.__file__).startswith(root), (
            f"{mod.__name__} resolved to {mod.__file__}, which is NOT under "
            f"{want} -- this run tested a different checkout")


# ---------------------------------------------------------------------------
# The two-by-two
# ---------------------------------------------------------------------------

def test_probe_role_on_a_magnet_reports_the_residual_and_does_not_refuse(
        monkeypatch):
    raised, log = _run(NON_HERMITIAN, PROBE, False, monkeypatch=monkeypatch)
    assert not raised, "strict refused the TR-odd part of W(i*omega_p)"
    assert log.failures == [], log.text
    assert "hermiticity residual = 2.000e-01" in log.text, log.text
    assert "TR-odd anti-Hermitian part of W" in log.text, log.text
    assert "MEASURED, not gated" in log.text, log.text


def test_the_same_array_refuses_when_time_reversal_was_measured_to_hold(
        monkeypatch):
    """The red twin.  Same W, same frequency, opposite verdict."""
    raised, log = _run(NON_HERMITIAN, PROBE, True, monkeypatch=monkeypatch)
    assert raised, "the gate no longer fails on a TRS deck -- it was deleted"
    assert log.failures, log.text
    assert "index/shard mixing bug" in log.failures[0], log.text


def test_a_hermitian_w_passes_under_both_verdicts(monkeypatch):
    for verdict in (True, False):
        raised, log = _run(HERMITIAN, PROBE, verdict, monkeypatch=monkeypatch)
        assert not raised, f"a Hermitian W was refused at trs={verdict}"
        assert log.failures == [], log.text


def test_omega_zero_is_gated_on_a_magnet_too(monkeypatch):
    """The unconditional half.  The TR-odd part carries a factor omega, so at
    omega = 0 a residual is an index fault for EVERY deck -- and this is the
    role that keeps the Dyson back-solve covered on a magnet, since the probe
    role's W comes off the same solve on the same tiles."""
    raised, log = _run(NON_HERMITIAN, STATIC, False, monkeypatch=monkeypatch)
    assert raised, "the omega=0 Hermiticity gate became conditional"
    assert log.failures, log.text


def test_no_verdict_supplied_refuses_instead_of_selecting_a_trs_branch(
        monkeypatch):
    monkeypatch.setenv("LORRAX_SANITY", "strict")
    with pytest.raises(ValueError, match="W_gate_needs_measured_trs"):
        _gate_w(NON_HERMITIAN, PROBE, trs_allowed=None)


def test_the_real_axis_probe_is_not_gated_under_either_verdict(monkeypatch):
    """A dynamical W obeys Kramers-Kronig and legitimately has W'' != 0.
    Gating that branch would check something false by construction."""
    for verdict in (True, False):
        raised, log = _run(NON_HERMITIAN, REAL_AXIS, verdict,
                           monkeypatch=monkeypatch)
        assert not raised, f"the real-axis probe was gated at trs={verdict}"
        assert log.failures == [], log.text


# ---------------------------------------------------------------------------
# How the gate learns the verdict, and what it does with it
# ---------------------------------------------------------------------------

def test_the_verdict_comes_from_the_measurement_and_nowhere_else():
    """``SymMaps.trs_allowed`` is the load-time density measurement.  A real
    tables object that somehow lacks it must raise rather than default to a
    convenient answer -- the rule ``qgrid_symmetry.qgrid_trs_policy_for``
    states for the same verdict."""
    class _Sym:
        trs_allowed = False

    assert _trs_verdict(_Sym()) is False
    _Sym.trs_allowed = 1                       # numpy/bool0 verdicts coerce
    assert _trs_verdict(_Sym()) is True
    with pytest.raises(ValueError, match="screening_needs_measured_trs"):
        _trs_verdict(None)
    with pytest.raises(ValueError, match="screening_needs_measured_trs"):
        _trs_verdict(object())


def test_the_production_call_sites_all_thread_the_verdict():
    """A keyword with a permissive default is only as good as its callers.
    Every ``_gate_w(`` in the RPA executor must carry ``trs_allowed=``, or a
    role silently reverts to the incumbent behaviour and this file's cells
    stop describing production."""
    import inspect
    import re

    from gw import screening, screening_bse

    src = inspect.getsource(screening.compute_screening)
    calls = re.findall(r"_gate_w\((?:[^()]|\([^()]*\))*\)", src)
    assert len(calls) == 3, f"expected 3 _gate_w call sites, found {calls}"
    for call in calls:
        assert "trs_allowed=" in call, call

    ladder_src = inspect.getsource(screening_bse._gate_w_or_refuse)
    ladder_calls = re.findall(
        r"_gate_w\((?:[^()]|\([^()]*\))*\)", ladder_src)
    assert len(ladder_calls) == 1, ladder_calls
    assert "trs_allowed=" in ladder_calls[0], ladder_calls[0]


def test_there_is_exactly_one_hermiticity_reporter():
    """The measurement path reuses ``report_hermitian_residual``'s ``cause``
    rather than growing a second reporter beside it.  Two of them is how the
    wording, the tolerance and the strict policy drift apart."""
    reporters = [n for n in dir(sanity)
                 if n.startswith("report_") and "hermitian" in n]
    assert reporters == ["report_hermitian_residual"], reporters
