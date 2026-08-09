"""The minimax suite's own setup: path, markers, skip-honesty, no leakage.

Four things, in the order they have to happen.

1. PATH.  Put the in-tree ``minimax`` and the ``lxkit`` foundation its
   test tier uses on ``sys.path``.  Installing the services registers
   :mod:`lxkit.testing` through the ``pytest11`` entry point, but this
   suite must also run from a bare checkout — ``lx`` rewrites the
   container ``PYTHONPATH`` to exactly ``<checkout>/src`` and the Shifter
   image pip-installs nothing — so the autouse fixture and the hooks are
   imported directly.  ``pytest_plugins`` is not an option: pytest honours
   it only in the ROOTDIR conftest, and this suite is collected under the
   monorepo root.

   NOTE lxkit is a TEST-time dependency only.  ``import minimax`` does not
   touch it, and the import-isolation cell next door is what measures
   that rather than asserting it.

2. NO x64 PIN AND NO XLA_FLAGS, unlike ``services/vcoul/tests/conftest.py``
   and ``services/distrib_la/tests/conftest.py``.  That is a real
   difference and not an omission: ``import minimax`` never touches jax at
   all.  Every number in this service is host float64 by construction
   because it is numpy, so there is no x32 default to defend against and
   no device count to force.  Forcing either here would change what every
   OTHER lorrax test in a full-suite run sees, for no coverage.

3. MARKERS.  Every cell under this directory gets ``services`` and
   ``minimax``, so the main suite can select or deselect the whole service
   without naming paths.

4. SKIP-HONESTY.  One gate per service (charter), armed here, SCOPED to
   this directory — unscoped it would judge the ~45 device-count skips
   lorrax's own suite emits on a 1-device leg, and a gate that fires on
   other people's business gets disarmed.

Plus one thing this service needs that the others do not: the door
ANNOUNCES, once per distinct request, through :mod:`warnings`.  Those
announcements are process-global state, so a cell that asserts "this
announced" would pass or fail depending on which cell ran first.  The
autouse fixture below resets both announcement registries and the catalog
caches around every cell, which is what makes the announcement assertions
order-independent — and ``test_the_announcement_reset_is_not_a_no_op``
next door is the red twin proving the fixture is doing something.
"""

import os
import sys

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))

for _svc in ("lxkit", "minimax"):
    _src = os.path.join(_SERVICES, _svc, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

from lxkit.testing import (                        # noqa: E402,F401
    AllowedSkip, arm_skip_honesty, gate_state,     # gate_state is autouse
    machine_profile, pytest_configure, pytest_runtest_logreport,
    pytest_sessionfinish,
)

#: Skips this SERVICE is allowed to emit on a machine that declares a
#: profile, on top of the universal rows lxkit ships.
#:
#: SHORT BY DESIGN, and shorter than vcoul's.  minimax has no ``.so`` tier,
#: no GPU tier, no multi-process tier and no jax, so the whole suite runs
#: on a laptop in seconds and there is almost nothing it is allowed to
#: decline to do.  scipy is NOT on this list on purpose: it is a declared
#: optional dependency (the ``solve`` extra), and the quarantine is
#: testable WITHOUT it (the import-isolation arm) and WITH it (the solver
#: arms), so a scipy-shaped skip would be a hole rather than an honest
#: absence.
_ALLOWED = (
    AllowedSkip("", "monorepo wiring",
                "the monorepo run; a standalone install has no lorrax"),
)

#: The directory this gate rules on.
PROFILE = arm_skip_honesty(scope=_TESTS, extra_allowed=_ALLOWED)


@pytest.fixture(autouse=True)
def _fresh_announcements():
    """Announce-once state is process-global; make it cell-local.

    Both directions matter.  Before: a cell that asserts an announcement
    fired must not be silenced by an earlier cell having announced the
    same request.  After: a cell that deliberately triggers a legacy-cache
    or uncertified-solve announcement must not leave that registry primed
    for the next one.
    """
    import minimax                                 # noqa: PLC0415
    from minimax import cache as _cache            # noqa: PLC0415

    minimax.reset_announcements()
    _cache.reset_announcements()
    minimax.clear_caches()
    yield
    minimax.reset_announcements()
    _cache.reset_announcements()
    minimax.clear_caches()


@pytest.fixture()
def isolated_cache(tmp_path, monkeypatch):
    """A disk cache nobody else's run can have written to.

    The WP1 census found a four-month-old entry in the shared ``$HOME``
    cache serving a frozen gate, so a cache cell that used the real
    directory would be testing that machine's history rather than this
    code.
    """
    monkeypatch.setenv("LORRAX_MINIMAX_CACHE_DIR", str(tmp_path / "mmcache"))
    monkeypatch.delenv("LORRAX_DISABLE_MINIMAX_DISK_CACHE", raising=False)
    return tmp_path / "mmcache"


def pytest_collection_modifyitems(config, items):
    """Mark everything under this directory ``services`` + ``minimax``.

    Scoped by PATH, not by "every item in the session": this hook is
    called with the whole collection, including lorrax's own tests when
    the full suite runs, and marking those would make ``--no-services``
    deselect the entire suite.
    """
    here = _TESTS + os.sep
    for item in items:
        if str(getattr(item, "fspath", "")).startswith(here):
            item.add_marker(pytest.mark.services)
            item.add_marker(pytest.mark.minimax)
