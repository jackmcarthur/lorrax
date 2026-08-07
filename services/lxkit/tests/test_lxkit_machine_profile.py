"""Skip-honesty: the mechanism, and both halves of it failing.

The charter's rule is that an unexpected skip is a FAILURE, because a skip
reads as "not applicable on this machine" and this tree has the receipt for
what happens when that is false (2026-08-06: nineteen contract cells
reported "19 skipped" beside "0 failed" and the suite looked green).
:mod:`lxkit.testing` implements the rule in two halves and this file shows
BOTH halves returning FALSE, which is the only thing that distinguishes a
gate from a decoration:

* POSITIVE — a synthetic profile whose MUST row names a target that does
  not exist must FAIL, quoting the probe's own three-way reason.
* NEGATIVE — a skip outside the allowlist must FAIL the session, and so
  must a session that collected less than the floor.

The negative half's twin runs a real pytest in a SUBPROCESS, because the
thing under test is a ``sessionfinish`` hook and the only honest way to
observe a session ending red is to end one.

No jax, no pytest plugins, no machine: everything here is either pure
arithmetic over data structures or a child pytest on a two-cell file.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from lxkit.testing import (AllowedSkip, MachineProfile, MustRow, ObservedSkip,
                           PROFILES, assert_must_probe,
                           assert_skips_match_profile, detect_machine,
                           machine_profile)

_TESTS = os.path.dirname(os.path.abspath(__file__))
_LXKIT_SRC = os.path.join(os.path.dirname(_TESTS), "src")


# ---------------------------------------------------------------------------
# The table itself
# ---------------------------------------------------------------------------

def test_perlmutter_declares_the_four_charter_backends():
    """The charter names them: scalapack, slate, cusolvermp, phdf5 MUST
    probe available on Perlmutter.  Pinned by NAME so a row cannot be
    quietly dropped to make a leg green."""
    prof = machine_profile("perlmutter")
    assert {r.name for r in prof.must} == {
        "scalapack", "slate", "cusolvermp", "phdf5"}
    assert not prof.asserts_nothing
    # Each row names the service whose loader owns the target vocabulary.
    assert {r.service for r in prof.must} == {"distrib_la", "slab_io"}
    assert prof.must_for("distrib_la") and prof.must_for("slab_io")


def test_the_unknown_profile_asserts_nothing_and_says_so():
    """A laptop has no ScaLAPACK and never will.  ``unknown`` is a NAMED
    row rather than a fall-through so that "this machine promises nothing"
    is a decision in the table instead of an accident of lookup."""
    prof = machine_profile("unknown")
    assert prof.asserts_nothing and prof.must == ()
    # And it really is inert in both halves.
    assert assert_must_probe(prof, {}) == []
    assert_skips_match_profile(
        [ObservedSkip("t.py::x", "any reason at all")], prof, collected=1)


def test_an_unrecognized_machine_falls_to_unknown_not_to_an_error():
    """A new NERSC host appearing must not turn every service suite red —
    but the profile still carries the name it was asked about, so the
    report says which machine made no promises."""
    prof = machine_profile("some-new-cluster")
    assert prof.asserts_nothing and prof.machine == "some-new-cluster"


@pytest.mark.parametrize("var", ["LX_MACHINE", "NERSC_HOST"])
def test_the_machine_is_detected_from_either_variable(monkeypatch, var):
    for v in ("LX_MACHINE", "NERSC_HOST"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(var, "Perlmutter")          # case-insensitive
    assert detect_machine() == "perlmutter"
    assert machine_profile().machine == "perlmutter"


def test_with_no_variables_set_the_machine_is_unknown(monkeypatch):
    for v in ("LX_MACHINE", "NERSC_HOST"):
        monkeypatch.delenv(v, raising=False)
    assert detect_machine() == "unknown"


def test_lx_machine_wins_over_nersc_host(monkeypatch):
    """A leg must be able to say what it is — including, for the red twins
    below, that it is a machine nobody has ever built."""
    monkeypatch.setenv("NERSC_HOST", "perlmutter")
    monkeypatch.setenv("LX_MACHINE", "unknown")
    assert detect_machine() == "unknown"


def test_every_allowed_skip_names_a_covering_leg():
    """The tests/KNOWN_FAILURES.md discipline, made structural: a skip
    nobody can point at a covering run for is evaporated coverage, so the
    allowlist has nowhere to put one without writing the leg down."""
    for name, prof in PROFILES.items():
        for row in prof.allowed_skips:
            assert row.covered_by.strip(), (name, row)
            assert len(row.covered_by) > 10, (name, row)


# ---------------------------------------------------------------------------
# POSITIVE half
# ---------------------------------------------------------------------------

def _probe_all_ok(target, platform):
    return True, "ok"


def _probe_nothing_there(target, platform):
    return False, (f"unknown target {target!r} for platform {platform!r} "
                   f"(known: lorrax_real_thing)")


def test_the_positive_half_passes_when_every_must_row_probes_available():
    prof = MachineProfile(
        machine="synthetic", must=(MustRow("x", "svc", "t", "cpu"),),
        allowed_skips=(), min_collected=1)
    assert assert_must_probe(prof, {"svc": _probe_all_ok}) == []


def test_the_positive_half_can_fail():
    """RED TWIN.  A synthetic profile row naming a nonexistent target MUST
    fire — and the failure must carry the probe's own reason verbatim,
    because the three ways a target can be unusable (unknown target /
    library would not load / loaded but does not export) have three
    different fixes and "unavailable" names none of them."""
    prof = MachineProfile(
        machine="synthetic",
        must=(MustRow("ghost", "svc", "lorrax_no_such_target", "cpu"),),
        allowed_skips=(), min_collected=1)
    seen = []

    def _fail(msg, **kw):
        seen.append(msg)
        raise AssertionError(msg)

    with pytest.raises(AssertionError) as ei:
        assert_must_probe(prof, {"svc": _probe_nothing_there}, fail=_fail)
    msg = str(ei.value)
    assert "lorrax_no_such_target" in msg
    assert "unknown target" in msg           # the probe's OWN taxonomy
    assert "DEFECT" in msg
    assert "ghost" in msg


def test_a_must_row_with_no_probe_is_reported_unasserted_not_passed():
    """The phdf5 case: distrib_la cannot probe slab_io's library.  The row
    comes back as a GAP rather than passing quietly, which is the whole
    difference between a known hole and a forgotten one."""
    prof = machine_profile("perlmutter")
    gaps = assert_must_probe(prof, {"distrib_la": _probe_all_ok})
    assert [r.name for r in gaps] == ["phdf5"]
    assert gaps[0].service == "slab_io"


def test_the_positive_half_is_inert_for_a_machine_that_promises_nothing():
    assert assert_must_probe(machine_profile("unknown"),
                             {"svc": _probe_nothing_there}) == []


# ---------------------------------------------------------------------------
# NEGATIVE half — the pure core
# ---------------------------------------------------------------------------

_PROF = MachineProfile(
    machine="synthetic",
    must=(),
    allowed_skips=(AllowedSkip("test_ok", "on purpose", "leg Z"),),
    min_collected=2)


def test_an_allowed_skip_passes():
    assert_skips_match_profile(
        [ObservedSkip("f.py::test_ok", "skipped on purpose")],
        _PROF, collected=5)


def test_the_negative_half_can_fail():
    """RED TWIN.  A skip outside the allowlist MUST fire, and the message
    must name the cell and quote its reason — a gate that says only "an
    unexpected skip happened" sends the reader back to the log."""
    with pytest.raises(AssertionError) as ei:
        assert_skips_match_profile(
            [ObservedSkip("f.py::test_ok", "skipped on purpose"),
             ObservedSkip("f.py::test_sneaky", "no GPU here")],
            _PROF, collected=5)
    msg = str(ei.value)
    assert "test_sneaky" in msg and "no GPU here" in msg
    assert "test_ok" not in msg.split("Allowed here")[0]
    assert "NAMING THE LEG" in msg


def test_both_halves_of_an_allowlist_row_must_match():
    """``where`` AND ``why``: a row keyed on the nodeid alone would let any
    future skip in that file through, which is how an allowlist rots."""
    with pytest.raises(AssertionError, match="test_ok"):
        assert_skips_match_profile(
            [ObservedSkip("f.py::test_ok", "for a completely other reason")],
            _PROF, collected=5)


def test_the_minimum_collected_floor_can_fail():
    """RED TWIN for the floor.  Checked FIRST and on purpose: a session
    that collected nothing skips nothing, so every other assertion here
    would pass vacuously.  That is the 38-byte-junitxml failure mode."""
    with pytest.raises(AssertionError, match="below the floor"):
        assert_skips_match_profile([], _PROF, collected=1)


def test_the_floor_is_checked_before_the_allowlist_is_vacuous():
    """The ordering IS the property: an empty session with a disallowed
    skip list would otherwise report the wrong defect."""
    with pytest.raises(AssertionError, match="below the floor"):
        assert_skips_match_profile(
            [ObservedSkip("f.py::x", "unexpected")], _PROF, collected=0)


def test_the_floor_applies_even_when_the_machine_asserts_nothing():
    """``unknown`` promises no backends.  It still cannot promise that a
    run which collected zero tests was a pass."""
    with pytest.raises(AssertionError, match="below the floor"):
        assert_skips_match_profile([], machine_profile("unknown"),
                                   collected=0, min_collected=1)


def test_extra_allowed_rows_compose_with_the_profile():
    assert_skips_match_profile(
        [ObservedSkip("f.py::test_sneaky", "no GPU here")], _PROF,
        collected=5,
        extra_allowed=[AllowedSkip("test_sneaky", "no GPU", "leg GPU")])


# ---------------------------------------------------------------------------
# NEGATIVE half — the pytest wiring, observed ending a real session red
# ---------------------------------------------------------------------------

_CHILD_CONFTEST = '''
import sys
sys.path.insert(0, {src!r})
from lxkit.testing import (AllowedSkip, MachineProfile, arm_skip_honesty,
                           pytest_runtest_logreport, pytest_sessionfinish)

arm_skip_honesty(MachineProfile(
    machine="synthetic", must=(),
    allowed_skips=(AllowedSkip("allowed", "on purpose", "leg Z"),),
    min_collected={floor}))
'''

_CHILD_TEST = '''
import pytest

def test_passes():
    assert True

def test_allowed():
    pytest.skip("skipped on purpose")

{extra}
'''

_DISALLOWED = '''
def test_sneaky():
    pytest.skip("this one is not on the list")
'''


def _run_child(tmp_path, extra="", floor=1):
    (tmp_path / "conftest.py").write_text(
        _CHILD_CONFTEST.format(src=_LXKIT_SRC, floor=floor))
    (tmp_path / "test_child.py").write_text(
        _CHILD_TEST.format(extra=textwrap.dedent(extra)))
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
         str(tmp_path)],
        cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)


def test_an_armed_session_with_only_allowed_skips_stays_green(tmp_path):
    proc = _run_child(tmp_path)
    assert "SKIP-HONESTY GATE FAILED" not in proc.stdout, proc.stdout
    assert proc.returncode == 0, proc.stdout
    # ...and it really did collect and really did skip, so the green is
    # not the vacuous kind.
    assert "1 passed, 1 skipped" in proc.stdout, proc.stdout


def test_the_armed_session_can_fail(tmp_path):
    """RED TWIN for the wiring.  A fake skip outside the allowlist must end
    the SESSION red — the pure core is already twinned above; what this
    adds is that the hook is registered, the reason survives the reporter,
    and the exit status changes."""
    proc = _run_child(tmp_path, extra=_DISALLOWED)
    assert "SKIP-HONESTY GATE FAILED" in proc.stdout, proc.stdout
    assert "test_sneaky" in proc.stdout
    assert "this one is not on the list" in proc.stdout
    assert proc.returncode != 0, proc.stdout


def test_the_armed_session_fails_a_collection_floor_breach(tmp_path):
    proc = _run_child(tmp_path, floor=99)
    assert "SKIP-HONESTY GATE FAILED" in proc.stdout, proc.stdout
    assert "below the floor" in proc.stdout
    assert proc.returncode != 0


def test_an_unarmed_session_asserts_nothing(tmp_path):
    """lxkit's plugin loads in suites that never heard of machine profiles.
    Recording a skip costs a list append; asserting on a session whose
    author declared no profile would make lxkit fail other people's
    suites, which is how a good gate gets deleted."""
    (tmp_path / "conftest.py").write_text(
        f"import sys\nsys.path.insert(0, {_LXKIT_SRC!r})\n"
        "from lxkit.testing import (pytest_runtest_logreport,\n"
        "                           pytest_sessionfinish)\n")
    (tmp_path / "test_child.py").write_text(
        "import pytest\n\n\ndef test_s():\n    pytest.skip('whatever')\n")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
         str(tmp_path)],
        cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    assert "SKIP-HONESTY GATE FAILED" not in proc.stdout
    assert proc.returncode == 0, proc.stdout
