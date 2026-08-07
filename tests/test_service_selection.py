"""The service-tier selection hooks, measured by collecting twice.

``tests/conftest.py`` grew the first collection hooks in this tree
(``--no-services`` / ``--only-service=NAME`` / ``LX_SKIP_SERVICES``).  A
collection hook that is subtly wrong deselects real coverage without
failing anything — the run is smaller, greener and quieter — so the hook
is kept to twenty lines of predicate and the checking is done here, from
the outside: run ``pytest --collect-only`` in a SUBPROCESS with and
without each flag and diff the SETS of node ids.

Sets, not counts.  A count-only comparison passes when a flag drops the
right NUMBER of tests from the wrong place, which is precisely the mistake
a path predicate makes.

WHY A SUBPROCESS.  The thing under test is what happens during collection,
including which conftests load and in what order; ``pytest.main()``
in-process would inherit this session's already-imported plugins and
already-parsed config and would measure something else.

WHY THE `-m` TRAP HAS ITS OWN CELL.  pyproject sets ``addopts = -m 'not
extra'``.  An explicit ``-m`` on the command line REPLACES that rather
than composing with it, so ``-m "not services"`` — the obvious way to
write ``--no-services`` — silently re-enables the whole deselected
``extra`` tier.  The cell below MEASURES that, so the reason these hooks
exist is a fact in the suite rather than a claim in a comment.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SERVICES = _REPO / "services"


def _collect(*args, env_extra=None) -> set[str]:
    """Node ids ``pytest --collect-only`` selects with ``args``."""
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(_REPO / "src"))
    env.update(env_extra or {})
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", *args],
        cwd=str(_REPO), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, timeout=900)
    ids = {ln.strip() for ln in proc.stdout.splitlines()
           if "::" in ln and not ln.startswith(("E ", "  "))}
    assert ids, (
        "collected NO node ids — the instrument is broken, which reads "
        f"identical to 'the flag deselected everything'.\n{proc.stdout}")
    return ids


@pytest.fixture(scope="module")
def baseline():
    return _collect()


def _is_service(nodeid: str) -> bool:
    return nodeid.startswith("services/")


def test_the_default_run_stages_the_service_suites_in(baseline):
    """The charter's "own fast pytests staged into the main suite", as a
    measurement: a plain ``pytest`` must collect service cells."""
    svc = {n for n in baseline if _is_service(n)}
    assert svc, "no services/ tests in the default collection"
    assert any("distrib_la" in n for n in svc)
    assert any("lxkit" in n for n in svc)
    # ...and lorrax's own tests are still there, which is the half a
    # "services are collected" assertion alone cannot see.
    assert {n for n in baseline if not _is_service(n)}


def test_no_services_drops_exactly_the_service_tests(baseline):
    got = _collect("--no-services")
    expected = {n for n in baseline if not _is_service(n)}
    assert got == expected, (
        f"--no-services removed {len(baseline - got - (baseline - expected))} "
        f"non-service test(s) and/or kept "
        f"{len(got - expected)} service test(s)")


def test_lx_skip_services_is_the_same_switch(baseline):
    got = _collect(env_extra={"LX_SKIP_SERVICES": "1"})
    assert got == {n for n in baseline if not _is_service(n)}


@pytest.mark.parametrize("falsy", ["", "0", "false", "no"])
def test_a_falsy_lx_skip_services_changes_nothing(baseline, falsy):
    """An env var that means "off" must not mean "on".  This is the cell
    that catches ``if os.environ.get(...)``: the string "0" is truthy."""
    assert _collect(env_extra={"LX_SKIP_SERVICES": falsy}) == baseline


def test_only_service_keeps_exactly_that_service(baseline):
    got = _collect("--only-service=distrib_la")
    expected = {n for n in baseline
                if n.startswith("services/distrib_la/")}
    assert got == expected
    assert not any("lxkit" in n for n in got)


def test_only_service_with_an_unknown_name_selects_nothing_loudly():
    """A typo must not quietly run the whole suite.

    Two independent things make this loud, and both are asserted because
    each is a different mechanism: pytest's own "no tests collected"
    (rc=5, which `lx` already treats as not-a-pass), and the service's
    minimum-collected floor, which fires FIRST and turns the exit code
    into a plain 1 with a sentence saying why.  Whichever wins, the run is
    non-zero and the output names the emptiness."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "--only-service=no_such_service"],
        cwd=str(_REPO), env={**os.environ,
                             "PYTHONPATH": str(_REPO / "src")},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=900)
    assert proc.returncode != 0, proc.stdout[-2000:]
    assert ("no tests collected" in proc.stdout
            or "below the floor" in proc.stdout), proc.stdout[-2000:]


def test_the_marker_selects_the_same_set_as_the_path_predicate(baseline):
    """``-m distrib_la`` and ``--only-service=distrib_la`` must agree.

    They are two independent mechanisms — a marker applied by the
    service's own conftest, and a path predicate in tests/conftest.py —
    and the ONE thing that would make the marker route useless is the two
    drifting apart.  Leg A2's ``--ignore`` became ``-m``; this is the cell
    that says the ``-m`` means what the path meant.
    """
    assert _collect("-m", "distrib_la") == _collect(
        "--only-service=distrib_la")


def test_a_second_dash_m_would_reenable_the_extra_tier(baseline):
    """THE TRAP, measured.  ``-m 'not services'`` looks like the obvious
    spelling of --no-services and is not equivalent to it: an explicit -m
    replaces pyproject's ``addopts = -m 'not extra'``, so the 26-suite
    extra tier comes back.  If this cell ever fails, pytest changed how
    addopts and -m compose and the hooks can be reconsidered."""
    trap = _collect("-m", "not services")
    extras = trap - baseline
    assert extras, (
        "`-m 'not services'` collected nothing that the default run "
        "deselects — the addopts-override trap this hook exists to avoid "
        "did not reproduce, so re-examine tests/conftest.py's rationale")
    # And it is the extra tier specifically that came back.
    assert _collect("--no-services") != trap


def test_the_hooks_do_not_touch_a_service_only_invocation():
    """``pytest services/distrib_la/tests`` never loads tests/conftest.py
    (it is not on the path to that directory), so the options do not
    exist there.  That is fine — you have already narrowed the run — but
    it must WORK rather than error, because it is how the service suite
    is run standalone and on Perlmutter."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "services/distrib_la/tests"],
        cwd=str(_REPO), env={**os.environ,
                             "PYTHONPATH": str(_REPO / "src")},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "::" in proc.stdout
