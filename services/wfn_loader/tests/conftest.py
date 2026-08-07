"""The wfn_loader suite's own setup: path, devices, markers, decks, outcomes.

Five things, in the order they have to happen.

1. PATH.  Put the in-tree ``wfn_loader`` and its ``lxkit`` foundation on
   ``sys.path``.  Installing the services registers :mod:`lxkit.testing`
   through the ``pytest11`` entry point, but this suite must also run from
   a bare checkout — ``lx`` rewrites the container ``PYTHONPATH`` to
   exactly ``<checkout>/src`` and the Shifter image pip-installs nothing —
   so the harness is imported directly.  ``pytest_plugins`` is not an
   option: pytest honours it only in the ROOTDIR conftest, and this suite
   is collected under the monorepo root.

2. DEVICES, before the first jax import.  See :data:`_XLA_FLAGS`.

3. MARKERS.  Every cell under this directory gets ``services`` and
   ``wfn_loader``, so the main suite can select or deselect the whole
   service without naming paths.

4. DECKS.  The checked-in hostile WFN fixtures, and the ONE place their
   geometry is written down.  See :data:`DECKS`.

5. OUTCOMES.  A service-local collector recording what every cell in this
   directory actually DID, which is what
   ``test_wfn_loader_skip_honesty.py`` rules on.  See :func:`outcomes`.

Why the markers arrive through a HOOK and not ``pytestmark``: a
module-level ``pytestmark`` applies to the module that declares it, and
pytest does not read one out of a conftest.  Writing ``pytestmark = [...]``
here would be silent — the suite would collect, the marks would not exist,
and ``-m wfn_loader`` would select nothing while looking like it worked.
``pytest_collection_modifyitems`` is the mechanism that actually applies,
and ``tests/test_service_selection.py`` measures that it did.

THE ONE THING distrib_la's conftest DOES THAT THIS ONE STILL DOES NOT:
:func:`lxkit.testing.arm_skip_honesty`, and the three hook functions
distrib_la imports beside it.  ``arm_skip_honesty`` writes a
PROCESS-GLOBAL ``_ARMED`` dict (``lxkit/testing.py:801``), so a second
caller does not add a second gate — it REPLACES the first one's scope.
In a full-suite run the conftests load in directory order, so a call here
would land after ``services/distrib_la/tests/conftest.py`` and silently
disarm the gate that found the LORRAX_FFI_HOST_SO restore defect on
2026-08-07.  Importing ``pytest_runtest_logreport`` /
``pytest_sessionfinish`` into a second conftest has the matching problem:
pytest registers hooks per conftest MODULE, so lxkit's recorder would run
twice and each skip would be recorded twice.

So this suite's gate is SERVICE-LOCAL (DESIGN.md DECISION 4, step-1a
adjudication (ii)): the hook below is this file's OWN function writing
this file's OWN dict, and it consumes lxkit's machine-profile vocabulary
READ-ONLY (``machine_profile``, ``MustRow``, ``AllowedSkip``,
``assert_skips_match_profile`` are pure data + pure functions).  Per-scope
arming in lxkit is REGISTERED to the main Fable; lxkit is frozen this wave.
"""

import os
import sys

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))
_REPO = os.path.dirname(_SERVICES)

for _svc in ("lxkit", "wfn_loader"):
    _src = os.path.join(_SERVICES, _svc, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

# ---------------------------------------------------------------------------
# Layer L-b: four emulated devices — but ONLY when this suite is the reason
# the process started.
# ---------------------------------------------------------------------------
# ``--xla_force_host_platform_device_count`` is read by XLA at the FIRST jax
# import in a process and never again, and jax is imported at module scope
# by a great many lorrax test modules.  So this setting is not "on or off";
# it is "did this conftest load before anything imported jax".  Same
# arrangement as ``services/distrib_la/tests/conftest.py``, and copied
# rather than reinvented for the same measured reason:
#
#     pytest services/wfn_loader/tests
#
# makes this an INITIAL conftest (pytest loads the conftests along the path
# of every command-line argument before it imports a single test module),
# nothing has imported jax, the flag takes, and the L-b cells run on a real
# 2x2 of host devices.  In the FULL suite ``testpaths = ["tests",
# "services"]`` collects tests/ first, something there imports jax during
# collection, and by the time this file loads the flag is inert — at which
# point the L-b cells SKIP via ``lxkit.testing.require_devices``, which
# skips and never asserts (tests/KNOWN_FAILURES.md lists 11 cells that
# failed a 1-device leg purely for writing ``assert n_dev >= 4``).
#
# Inert is the CORRECT outcome there: forcing four host devices on the
# whole monorepo run would change the device count every other lorrax test
# sees.  The full-suite set-diff is the check that this stayed true.
#
# ``setdefault``: an explicit XLA_FLAGS from the caller always wins.
_XLA_FLAGS = "--xla_force_host_platform_device_count=4"
if "jax" not in sys.modules:
    _existing = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in _existing:
        os.environ["XLA_FLAGS"] = (
            f"{_existing} {_XLA_FLAGS}".strip() if _existing else _XLA_FLAGS)

# x64 or every ψ cell silently measures complex64 against a complex128
# reference and the bit-identity claims become claims about rounding.
# tests/conftest.py does this for the monorepo suite; a service-only run
# has no tests/conftest.py in its load path.
os.environ.setdefault("JAX_ENABLE_X64", "1")


# ---------------------------------------------------------------------------
# The checked-in hostile decks
# ---------------------------------------------------------------------------
#: Every real WFN.h5 in the monorepo, with the geometry that makes it
#: HOSTILE, written down ONCE.  Source: survey w1_wfn_loader §6.4, which
#: read every header; ``test_the_deck_table_is_true_of_the_file_on_disk``
#: re-measures it from the file rather than trusting this table.
#:
#: ``gnppm`` is the subject of the L-c tier: the survey established it is
#: BYTE-SIZE IDENTICAL to the ``/pscratch`` MoS2 3x3 file the parity
#: harness used to name, whose machine is gone (nrk 9, mnband 82, nspinor
#: 2, ngkmax 1963, ntran 2, coeffs (82,2,17559,2)).  82 % 4 = 2 and
#: min(ngk) = 1917 < 1963, so a 2x2 leg on it exercises BOTH pad axes.
DECKS = {
    #  name:        (relative path,                          nrk, mnband, ngkmax, ngk_min)
    "gnppm":        ("tests/regression/gnppm_debug/WFN.h5",     9, 82, 1963, 1917),
    "cohsex":       ("tests/regression/cohsex_debug/WFNsmall.h5", 4, 150, 780, 749),
    "bispinor":     ("tests/regression/bispinor_debug/WFN.h5",  9, 34, 5545, 5502),
    "si_bse":       ("tests/regression/si_bse_debug/WFN.h5",    8, 62, 588, 537),
    "si_cohsex":    ("tests/regression/si_cohsex_debug/WFN.h5", 8, 62, 588, 537),
}

#: The reason string a deck-dependent cell skips with when the monorepo is
#: not around it.  It contains ``no monorepo`` so that lxkit's UNIVERSAL
#: allowed-skip row matches it, and this service's own gate
#: (``test_wfn_loader_skip_honesty.py``) turns it into a FAILURE on any
#: machine whose profile promises the fixtures.
NO_DECK = ("no monorepo fixture tree beside this service (a standalone "
           "install has no tests/regression/), so the hostile deck {name!r} "
           "at {path!r} is absent — covered by the monorepo run and by leg "
           "L-c (srun -n 4 on the checked-in gnppm deck)")


def deck_path(name: str) -> str | None:
    """Absolute path of a checked-in deck, or ``None`` when it is absent.

    A PLAIN FUNCTION, not a pytest fixture, because the L-c file is also a
    bare ``__main__`` under ``srun`` where no fixture machinery exists.
    """
    rel = DECKS[name][0]
    p = os.path.join(_REPO, rel)
    return p if os.path.exists(p) else None


def _deck_or_skip(name: str) -> str:
    import pytest
    p = deck_path(name)
    if p is None:
        pytest.skip(NO_DECK.format(name=name,
                                   path=os.path.join(_REPO, DECKS[name][0])))
    return p


@pytest.fixture(scope="session")
def repo_root() -> str:
    """The monorepo checkout this service is vendored inside."""
    return _REPO


@pytest.fixture
def gnppm_wfn() -> str:
    """The hostile deck the L-c tier is written against (see :data:`DECKS`)."""
    return _deck_or_skip("gnppm")


@pytest.fixture(params=sorted(DECKS))
def any_deck(request) -> tuple:
    """Every checked-in deck, as ``(name, path, nrk, mnband, ngkmax, ngk_min)``.

    The declared geometry rides along so the cell that re-measures the
    header does not have to import this module by name — three
    directories in this monorepo have a ``conftest``, and which one
    ``sys.modules`` holds under that key depends on collection order.
    """
    name = request.param
    return (name, _deck_or_skip(name)) + tuple(DECKS[name][1:])


# ---------------------------------------------------------------------------
# The service-local outcome collector (DESIGN.md adjudication (ii))
# ---------------------------------------------------------------------------
#: ``nodeid -> (outcome, reason)`` for every CALL-phase report from a cell
#: under this directory, plus the setup-phase skips (a cell skipped by a
#: fixture never reaches the call phase).  This is the raw material the
#: skip-honesty gate rules on; it is a module-global of THIS conftest, so
#: nothing outside ``services/wfn_loader/tests`` can be judged by it and
#: nothing here can clobber distrib_la's gate.
_OUTCOMES: dict = {}
#: Every nodeid this session COLLECTED from this directory.  Separate from
#: ``_OUTCOMES`` on purpose: "was never collected" and "ran and skipped"
#: are different defects, and a gate that could not tell them apart would
#: report a deselected suite as a healthy one.
_COLLECTED: set = set()


def outcomes() -> dict:
    """``{nodeid: (outcome, reason)}`` observed so far in this session."""
    return dict(_OUTCOMES)


def collected() -> frozenset:
    """Every nodeid collected from this directory in this session."""
    return frozenset(_COLLECTED)


@pytest.fixture
def observed_outcomes() -> dict:
    """``{nodeid: (outcome, reason)}`` for this directory, so far.

    A FIXTURE and not an import: a test module doing ``from conftest
    import outcomes`` names a module that three different directories in
    this monorepo also call ``conftest``, and which one ``sys.modules``
    holds depends on collection order.  The fixture channel is the one
    pytest guarantees resolves to THIS directory's conftest.
    """
    return outcomes()


@pytest.fixture
def collected_nodeids() -> frozenset:
    """Every nodeid this session collected from this directory."""
    return collected()


def _here(report_or_item) -> bool:
    for attr in ("path", "fspath"):
        loc = getattr(report_or_item, attr, None)
        if loc:
            return str(loc).startswith(_TESTS + os.sep)
    return False


def pytest_runtest_logreport(report):
    """Record this suite's own outcomes.  THIS FILE'S function, not lxkit's.

    Importing ``lxkit.testing.pytest_runtest_logreport`` here instead would
    register lxkit's recorder a SECOND time (pytest registers hooks per
    conftest module), double-counting every skip in distrib_la's gate.
    Recording into a local dict costs one branch and judges only this
    directory.
    """
    if not _here(report):
        return
    if report.when == "call" or (report.when == "setup" and report.skipped):
        longrepr = getattr(report, "longrepr", None)
        reason = ""
        if isinstance(longrepr, tuple) and len(longrepr) == 3:
            reason = str(longrepr[2])
        elif report.skipped and longrepr is not None:
            reason = str(longrepr)
        if reason.startswith("Skipped: "):
            reason = reason[len("Skipped: "):]
        _OUTCOMES[report.nodeid] = (report.outcome, reason.strip())


def pytest_collection_modifyitems(config, items):
    """Mark everything under this directory ``services`` + ``wfn_loader``.

    Scoped by PATH, not by "every item in the session": this hook is
    called with the whole collection, including lorrax's own tests when
    the full suite runs, and marking those would make ``--no-services``
    deselect the entire suite.
    """
    here = _TESTS + os.sep
    for item in items:
        if str(getattr(item, "fspath", "")).startswith(here):
            item.add_marker(pytest.mark.services)
            item.add_marker(pytest.mark.wfn_loader)
            _COLLECTED.add(item.nodeid)
