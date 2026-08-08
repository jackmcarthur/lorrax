"""The skip-honesty gate, zeta_loader's half — and it CLOSES distrib_la's gap.

The charter: each machine declares an expected-backend profile, and one gate
per service asserts that reality matches it.  An unexpected skip is a
FAILURE, because a skip reads as "not applicable on this machine" and this
tree has the receipt for what that costs — 2026-08-06, nineteen
ScaLAPACK/SLATE contract cells reported as "19 skipped" beside "0 failed",
suite green, coverage gone.

THE PART THAT IS NEW HERE IS THE PROBE.  lxkit's perlmutter profile carries
four MUST rows, and the fourth says of itself:

    # Not distrib_la's library to probe: phdf5 belongs to slab_io, which is
    # not a service yet.  The row is declared HERE because the promise is
    # the MACHINE's ... distrib_la's gate reports it UNASSERTED; the slab_io
    # retrofit (charter wave 1b) supplies the probe.

``zeta_loader`` is slab_io's client — every ζ data read is a
``SlabIO.read_slab`` — so it is the service that can answer for that row
today, a wave ahead of the retrofit.  :func:`probe_slab_io_target` is the
adapter, and ``test_the_phdf5_row_this_service_can_now_assert`` is
``test_the_rows_this_service_cannot_probe_are_named_not_dropped``'s
inversion: the gap distrib_la registered is closed from this side.

The NEGATIVE half — every skip diffed against the allowlist, plus the
minimum-collected floor — is armed for this whole suite in ``conftest.py``
and runs at ``sessionfinish``.  What lives here is the falsification: the
allowlist checked against the reasons this suite can actually produce, and
red twins for both halves.
"""

from __future__ import annotations

import os
import sys

import pytest
from lxkit.testing import (MachineProfile, MustRow, ObservedSkip, PROFILES,
                           assert_must_probe, assert_skips_match_profile,
                           machine_profile, must_rows_for)

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

import zeta_synth as Z                                         # noqa: E402


# ---------------------------------------------------------------------------
# The probe this service supplies
# ---------------------------------------------------------------------------

#: slab_io's FFI vocabulary.  Declared as DATA so an unknown target is a
#: mistake in the PROFILE (different fix) rather than a missing library.
_SLAB_IO_TARGETS = ("lorrax_phdf5_read", "lorrax_phdf5_write")


def probe_slab_io_target(target: str, platform: str) -> tuple[bool, str]:
    """``(ok, reason)`` for a phdf5 MUST row — the adapter, and its honesty.

    Two questions, ANDed, because they fail for unrelated reasons and the
    profile's failure message quotes this string verbatim:

    1. does this platform's FFI library export the ROW'S OWN target?  Asked
       through ``ffi.common.ffi_loader.probe_target``, whose reason is
       three-way (unknown target / library would not load / loaded but does
       not export) — three states with three different fixes, and
       collapsing them into "unavailable" throws away the only part anybody
       can act on.
    2. can the transport actually run?  ``file_io.slab_io.probe_availability``
       answers that end to end, and it is NOT the same question: it probes
       ``lorrax_phdf5_write`` AND that MPI can bootstrap in this process.
       Handler presence is not capability — on a bare launch with no PMI
       environment Intel MPI aborts inside ``MPIR_pmi_init`` (job 7884926,
       the fastloop's bare P=1 gw stage died at the first collective
       H5Fcreate).  A row that only checked symbol export would report
       phdf5 available on a machine where every ζ read hangs or dies.

    THE ASYMMETRY IS STATED RATHER THAN SMOOTHED: the MUST row names
    ``lorrax_phdf5_read`` and ``probe_availability`` probes the WRITE
    handler.  Both ship from the same ``.so``, so in practice they move
    together — but "in practice" is not a probe, which is why (1) asks
    about the row's own target explicitly and the reason names both.
    """
    if target not in _SLAB_IO_TARGETS:
        return False, (
            f"{target!r} is not a slab_io transport target; slab_io's FFI "
            f"vocabulary is {list(_SLAB_IO_TARGETS)}.  An unknown target is "
            f"a mistake in the machine PROFILE, not a missing library, and "
            f"the fix is to correct the row rather than to rebuild anything.")

    sym_ok, sym_why = True, "not asked"
    try:
        from ffi.common.ffi_loader import probe_target
        sym_ok, sym_why = probe_target(target, platform)
    except Exception as exc:                                   # noqa: BLE001
        sym_ok, sym_why = False, (
            f"ffi.common.ffi_loader is not importable here "
            f"({type(exc).__name__}: {exc})")

    try:
        from file_io.slab_io import probe_availability
    except Exception as exc:                                   # noqa: BLE001
        return False, (
            f"{target!r}: {sym_why}; and file_io.slab_io is not importable "
            f"({type(exc).__name__}: {exc}) — phdf5 lives behind that module "
            f"until the wave-1b retrofit extracts it as a service")
    try:
        ok, stage, reason = probe_availability(platform)
    except Exception as exc:                                   # noqa: BLE001
        return False, (f"{target!r}: {sym_why}; slab_io.probe_availability("
                       f"{platform!r}) raised {type(exc).__name__}: {exc}")
    return (bool(sym_ok) and bool(ok)), (
        f"{target!r}: {sym_why}; slab_io.probe_availability({platform!r}) "
        f"[which probes lorrax_phdf5_write + MPI bootstrap] stage {stage!r}: "
        f"{reason}")


#: The probes this service can answer for.  A row whose service is not in
#: here comes back UNASSERTED, which is the point of the mapping.
_PROBES = {"slab_io": probe_slab_io_target}


# ---------------------------------------------------------------------------
# POSITIVE HALF
# ---------------------------------------------------------------------------

def test_this_machines_declared_backends_are_actually_available():
    """THE GATE.  On a machine that promises phdf5, it is there.

    On Perlmutter the profile names it as a MUST row and BUILD_NOTES pins
    the ``.so`` it lives in (the Aug-7 pair — ``96a6399`` moved
    ``ctx_handle``/``ds_id`` from Attrs into a runtime ``(2,) int64``
    buffer, so the Aug-6 libraries die at "expected 3 but got 4" on every
    read this loader issues).  Anywhere else the profile is ``unknown``,
    which asserts nothing, on purpose: a laptop has no phdf5 FFI and a gate
    that demanded one would train everybody to ignore it.
    """
    profile = machine_profile()
    if profile.asserts_nothing:
        pytest.skip(
            f"machine {profile.machine!r} declares no expected-backend "
            f"profile, so there is nothing to assert — covered by the "
            f"Perlmutter leg, where NERSC_HOST selects the profile that "
            f"names phdf5 (lorrax_phdf5_read, cpu) as a MUST row")
    gaps = assert_must_probe(profile, _PROBES, service="slab_io")
    assert gaps == [], (
        f"service='slab_io' filtered the rows and this service supplies the "
        f"probe, so nothing should have been unassertable; got {gaps}")


def test_the_phdf5_row_this_service_can_now_assert():
    """The INVERSION of distrib_la's registered gap.

    ``test_distrib_la_skip_honesty.py::test_the_rows_this_service_cannot_
    probe_are_named_not_dropped`` asserts that ``phdf5`` comes back as the
    one UNASSERTED row, and says in its own docstring "the slab_io retrofit
    (charter wave 1b) supplies the probe and this assertion inverts".  This
    is the inversion, one wave early, from the service that is slab_io's
    client: with ``_PROBES`` supplying ``slab_io``, phdf5 is no longer a
    gap — and the three rows this service has no library for come back as
    gaps instead, named rather than dropped.

    STUB probes for the rows this service does NOT own: the claim here is
    about WHICH rows are assertable, and running the real probe would make
    the cell also depend on whether this machine has the ``.so`` — measuring
    two things and reporting one.
    """
    profile = machine_profile("perlmutter")
    gaps = assert_must_probe(profile, {"slab_io": lambda t, p: (True, "")})
    names = [r.name for r in gaps]
    assert "phdf5" not in names, (
        f"phdf5 came back UNASSERTED even though this service supplies its "
        f"probe: {gaps}")
    assert names == ["scalapack", "slate", "cusolvermp"], names
    assert {r.service for r in gaps} == {"distrib_la"}


def test_the_phdf5_row_is_the_one_this_service_answers_for():
    """The row's own vocabulary, pinned.

    A probe wired to the wrong target would pass every cell above by
    answering a question nobody asked.  The row's ``target`` is what
    :func:`probe_slab_io_target` dispatches on, so it is asserted here
    against the profile rather than against a literal in the adapter.
    """
    rows = must_rows_for("slab_io", "perlmutter")
    assert len(rows) == 1
    row = rows[0]
    assert (row.name, row.service) == ("phdf5", "slab_io")
    assert row.target == "lorrax_phdf5_read"
    assert row.target in _SLAB_IO_TARGETS
    assert row.platform == "cpu"


def test_the_gate_can_fail():
    """RED TWIN, with THIS service's real probe.

    lxkit's own twin uses a stub probe, which proves the reporting.  This
    one proves the WIRING: a MUST row naming a target slab_io has never
    heard of must make the real adapter say no, and that no must reach the
    failure with its reason intact.  Without this cell, a mis-plumbed
    ``_PROBES`` mapping would make the gate above pass on any machine by
    asserting nothing.

    The unknown-target rung is chosen because it is the one arm of the
    three-way taxonomy that is answerable WITHOUT a library — so this twin
    fires identically on WSL and on Perlmutter, instead of being a check
    that only exists where the ``.so`` is missing.
    """
    fake = MachineProfile(
        machine="perlmutter",
        must=(MustRow("ghost", "slab_io", "lorrax_no_such_target", "cpu"),),
        allowed_skips=(), min_collected=1)
    # ``pytest.fail`` raises ``Failed``, which derives from BaseException and
    # NOT from Exception — ``pytest.raises(Exception)`` does not catch it,
    # and a twin written that way ERRORS instead of passing.
    with pytest.raises(pytest.fail.Exception) as ei:
        assert_must_probe(fake, _PROBES, service="slab_io")
    msg = str(ei.value)
    assert "lorrax_no_such_target" in msg
    assert "DEFECT" in msg
    assert "not a slab_io transport target" in msg          # the taxonomy
    assert "mistake in the machine PROFILE" in msg          # the FIX


def test_the_probe_reports_a_reason_whatever_the_answer_is():
    """The real adapter, on the real row, on THIS machine.

    Asserts nothing about availability — that is the profile's job and this
    machine may legitimately have no phdf5 at all.  What it does assert is
    that the answer always carries an actionable reason: an empty or
    single-word reason is how "unavailable" becomes the only thing anybody
    reads, and the whole three-way taxonomy exists to prevent that.
    """
    ok, reason = probe_slab_io_target("lorrax_phdf5_read", "cpu")
    assert isinstance(ok, bool)
    assert isinstance(reason, str) and len(reason) > 40, reason
    assert "lorrax_phdf5_read" in reason


# ---------------------------------------------------------------------------
# NEGATIVE HALF — the allowlist, checked against reasons this suite emits
# ---------------------------------------------------------------------------

def _conftest_module():
    """This suite's conftest, whatever pytest named it."""
    here = os.path.realpath(os.path.join(_TESTS, "conftest.py"))
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and os.path.realpath(f) == here:
            return mod
    raise AssertionError("this suite's conftest is not in sys.modules")


def _strict_profile():
    """A profile that ASSERTS, for testing the allowlist off Perlmutter.

    The ``unknown`` row asserts nothing, deliberately and correctly — which
    means the WSL leg never exercises the allowlist at all, and an allowlist
    nothing exercises is a comment.  This is a machine that promises no
    backends but DOES judge skips, so the rows can be falsified here.
    """
    return MachineProfile(machine="a-machine-that-judges-skips", must=(),
                          allowed_skips=PROFILES["perlmutter"].allowed_skips,
                          min_collected=1, asserts_nothing=False)


def _reasons_this_suite_can_emit():
    """Every skip reason this suite is capable of producing, with its cell.

    Built by CALLING the probes rather than by copying their strings: a
    reason that drifts from the allowlist row is exactly the failure this
    is for, and a hand-copied literal would drift with it.
    """
    out = []
    ok, why = Z.slab_io_state()
    if not ok:
        out.append(("test_zeta_loader_multiproc.py::test_mu_pad", why))
    ok, why = Z.host_tree_state()
    if not ok:
        out.append(("test_zeta_loader_contract.py::test_open", why))
    # lxkit.require_devices' own wording, which the L-b cells emit verbatim
    # in the full-suite run (where XLA_FLAGS is already inert).
    out.append(("test_zeta_loader_multiproc.py::test_q_window",
                "needs >= 4 devices on platform 'cpu', have 1.  Set "
                "XLA_FLAGS=--xla_force_host_platform_device_count=4 BEFORE "
                "the first jax import, or run the real multi-process leg"))
    out.append(("test_zeta_loader_import_isolation.py::test_x",
                "no lorrax src/ next to this service (standalone install); "
                "the with-monorepo legs need the checkout"))
    return out


def test_every_skip_this_suite_can_emit_is_on_the_allowlist():
    """The negative half, falsified HERE rather than only on Perlmutter.

    ``assert_skips_match_profile`` is PURE on purpose — it takes observed
    skips and raises — so it can be called directly with the reasons this
    suite produces, against a profile that judges.  That makes the WSL leg
    a real test of the allowlist instead of a no-op under the ``unknown``
    row.

    THIS CELL IS MACHINE-SENSITIVE BY DESIGN, and that is not a wart.  The
    reasons come from the live probes and the allowlist comes from
    ``conftest._allowed_for(machine_profile())``, which REMOVES the
    SlabIO-transport row on Perlmutter.  So on Perlmutter WITH the
    BUILD_NOTES FFI pins the probe succeeds, emits no reason, and this
    passes; on Perlmutter WITHOUT them the probe fails, the reason has no
    covering row, and this fails HERE — at the cell, naming the transport —
    rather than only at ``sessionfinish``.  Two signals for one defect, the
    earlier one attached to the thing that measured it.  MEASURED on WSL
    2026-08-07 with ``LX_MACHINE=perlmutter`` forced: this cell and the
    positive-half gate both went red, which is the correct answer for a
    machine claiming a phdf5 it does not have.
    """
    cf = _conftest_module()
    skips = [ObservedSkip(nodeid, why)
             for nodeid, why in _reasons_this_suite_can_emit()]
    assert skips, "this suite can emit no skips at all, so nothing was checked"
    assert_skips_match_profile(skips, _strict_profile(),
                               collected=len(skips) + 10,
                               extra_allowed=cf._EXTRA_ALLOWED)


def test_the_negative_half_can_fail():
    """RED TWIN: a skip nobody wrote a covering leg for turns the session red.

    The reason is deliberately plausible — "the fixture was slow today" is
    the shape a real coverage leak takes, not an obviously bogus string.
    """
    cf = _conftest_module()
    bogus = [ObservedSkip("services/zeta_loader/tests/test_x.py::t",
                          "the fixture was slow today")]
    with pytest.raises(AssertionError) as ei:
        assert_skips_match_profile(bogus, _strict_profile(), collected=20,
                                   extra_allowed=cf._EXTRA_ALLOWED)
    msg = str(ei.value)
    assert "does not allow 1 of the 1 skip(s)" in msg
    assert "NAMING THE LEG" in msg              # the fix, in the message


def test_the_minimum_collected_floor_can_fail():
    """RED TWIN for the floor.  Both halves above are vacuous at zero cells.

    A 38-byte junitxml parses as zero tests and zero failures; ``lx``
    reports pytest rc=5 ("no tests collected") as its own kind of
    not-a-pass for the same reason.  Judge by artifacts.
    """
    with pytest.raises(AssertionError, match="below the floor"):
        assert_skips_match_profile([], _strict_profile(), collected=0)


def test_the_perlmutter_profile_takes_the_transport_row_away():
    """A skip that is a platform fact HERE is a DEFECT on Perlmutter.

    ``conftest._allowed_for`` removes the SlabIO-transport row on the
    machine whose profile declares phdf5 a MUST and whose BUILD_NOTES pins
    the library.  Without that filter, a Perlmutter run where the FFI pins
    were not passed would report every ζ data cell as an honest skip — the
    exact 19-skipped-0-failed shape, on the machine it was written for.
    """
    cf = _conftest_module()
    laptop = [r.why for r in cf._allowed_for(machine_profile("a-laptop"))]
    perl = [r.why for r in cf._allowed_for(machine_profile("perlmutter"))]
    assert any("SlabIO transport" in w for w in laptop)
    assert not any("SlabIO transport" in w for w in perl)
    # The host-tree row survives on both: a standalone install has no
    # file_io wherever it runs.
    assert any("no lorrax host tree" in w for w in perl)


def test_every_allowed_row_names_a_covering_leg():
    """A skip with no covering run is lost coverage wearing a green dot.

    ``AllowedSkip.covered_by`` is required by the type, so this checks the
    thing the type cannot: that the text actually names a LEG (a command, a
    machine, a job) rather than restating the reason.
    """
    cf = _conftest_module()
    for row in cf._ALLOWED:
        assert len(row.covered_by) > 30, row
        assert any(tok in row.covered_by
                   for tok in ("lx run", "lx test", "pytest", "monorepo",
                               "leg L-")), row


# ---------------------------------------------------------------------------
# THE SINGLE-SLOT GATE — this suite must not disarm another service's
# ---------------------------------------------------------------------------

def test_arming_this_gate_did_not_disarm_another_services():
    """Arming is per SCOPE, so two services can both hold a live gate.

    ``lxkit.testing._ARMED`` used to be ONE dict that ``arm_skip_honesty``
    overwrote, so this conftest could not simply call it: whoever loaded last
    replaced the earlier service's scope and allowlist, and the loser's gate
    went silently inert.  This file's conftest carried a whole conditional
    branch and a forked ``pytest_sessionfinish`` because of it, and both said
    in writing that the fix was to make ``_ARMED`` a list of scopes.

    That landed.  So the assertion is now the direct one: THIS suite's
    directory is armed, and arming it took nothing else down.
    """
    import lxkit.testing as lxt

    rows = {r.scope: r for r in lxt.armed_scopes()}
    ours = os.path.realpath(_TESTS)
    assert ours in rows, (
        f"zeta_loader's skip-honesty gate is NOT armed ({sorted(rows)!r}); "
        f"in a full-suite run that means this service's skips are ruled on "
        f"by nobody")
    assert os.path.isdir(rows[ours].scope), (
        f"the armed scope {rows[ours].scope!r} is not a directory; a scope "
        f"that matches no path makes the gate report 'not selected, inert' "
        f"and assert zero")

    # RED TWIN, constructed: arming a SECOND scope must leave ours untouched.
    # Under the old single-slot dict this is exactly what failed.
    mine = rows[ours]
    other = os.path.join(_TESTS, "_arming_probe_dir")
    key = os.path.realpath(other)
    lxt.arm_skip_honesty(lxt.machine_profile("a-laptop"), scope=other)
    try:
        again = {r.scope: r for r in lxt.armed_scopes()}
        assert again.get(ours) == mine, (
            f"arming {other!r} changed zeta_loader's row "
            f"({mine} -> {again.get(ours)}) — services are disarming each "
            f"other again")
    finally:
        lxt._ARMED.pop(key, None)


def test_the_marker_hook_actually_applied(request):
    """``-m zeta_loader`` must select this suite, and the hook is why.

    A module-level ``pytestmark`` in a conftest is silent: the suite
    collects, the marks do not exist, and ``-m zeta_loader`` selects nothing
    while looking like it worked.  Asking the running item for its own
    markers is the measurement that the hook fired.
    """
    names = {m.name for m in request.node.iter_markers()}
    assert "services" in names and "zeta_loader" in names, names
