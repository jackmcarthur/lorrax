"""The vcoul suite's own setup: path, x64, markers, skip-honesty.

Four things, in the order they have to happen.

1. PATH.  Put the in-tree ``vcoul`` and the ``lxkit`` foundation its
   test tier uses on ``sys.path``.  Installing the services registers
   :mod:`lxkit.testing` through the ``pytest11`` entry point, but this
   suite must also run from a bare checkout — ``lx`` rewrites the
   container ``PYTHONPATH`` to exactly ``<checkout>/src`` and the Shifter
   image pip-installs nothing — so the autouse fixture and the hooks are
   imported directly.  ``pytest_plugins`` is not an option: pytest honours
   it only in the ROOTDIR conftest, and this suite is collected under the
   monorepo root.

   NOTE lxkit is a TEST-time dependency only.  ``import vcoul`` does not
   touch it (``services/vcoul/pyproject.toml`` names it in the ``test``
   extra and nowhere else), and the import-isolation cell next door is
   what measures that rather than asserting it.

2. x64, before the first jax import.  ``wrap_points_to_voronoi`` is a
   jitted float64 kernel and every number in this service is float64; at
   jax's default x32 the door-smoke cell would silently measure float32.
   ``tests/conftest.py`` does this for the monorepo suite, and a
   service-only run never loads that file.

3. MARKERS.  Every cell under this directory gets ``services`` and
   ``vcoul``, so the main suite can select or deselect the whole service
   without naming paths.

4. SKIP-HONESTY.  One gate per service (charter), armed here.

NO XLA_FLAGS AND NO GPU PIN, unlike ``services/distrib_la/tests/
conftest.py``.  That is a real difference and not an omission: vcoul is
host-side arithmetic with no mesh, no ``.so`` and no device count to
force.  There is no L-b tier to give four emulated devices to, so
forcing them here would change the device count every OTHER lorrax test
in a full-suite run sees, for no coverage at all.

Why the markers arrive through a HOOK and not ``pytestmark``: a
module-level ``pytestmark`` applies to the module that declares it, and
pytest does not read one out of a conftest.  Writing ``pytestmark = [...]``
here would be silent — the suite would collect, the marks would not exist,
and ``-m vcoul`` would select nothing while looking like it worked.
``pytest_collection_modifyitems`` is the mechanism that actually applies,
and ``tests/test_service_selection.py`` measures that it did.
"""

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))

for _svc in ("lxkit", "vcoul"):
    _src = os.path.join(_SERVICES, _svc, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

# x64 or every float64 assertion in this suite silently measures float32.
os.environ.setdefault("JAX_ENABLE_X64", "1")

from lxkit.testing import (                        # noqa: E402,F401
    AllowedSkip, arm_skip_honesty, gate_state,     # gate_state is autouse
    machine_profile, pytest_configure, pytest_runtest_logreport,
    pytest_sessionfinish,
)

#: Skips this SERVICE is allowed to emit on a machine that declares a
#: profile, on top of the universal rows lxkit ships.  Every row names the
#: leg that runs the cell instead; a skip nobody can point at a covering
#: run for is evaporated coverage.
#:
#: SHORT BY DESIGN.  vcoul has no ``.so`` tier, no GPU tier and no
#: multi-process tier, so the whole suite runs on a laptop and there is
#: almost nothing it is allowed to decline to do.  scipy is NOT on this
#: list on purpose: it is a declared optional dependency, the
#: announce-or-refuse gate is testable WITHOUT it (the refusal arm) and
#: WITH it (the bit-identity arm), so a scipy-shaped skip would be a
#: hole rather than an honest absence.
_ALLOWED = (
    AllowedSkip("", "monorepo wiring",
                "the monorepo run; a standalone install has no lorrax"),
)


#: The directory this gate rules on.  Scoped, because the allowlist
#: describes THIS service's skips: unscoped, the gate would also judge the
#: ~45 device-count skips lorrax's own suite emits on a 1-device leg, and a
#: gate that fires on other people's business gets disarmed.
PROFILE = arm_skip_honesty(scope=_TESTS, extra_allowed=_ALLOWED)


def pytest_collection_modifyitems(config, items):
    """Mark everything under this directory ``services`` + ``vcoul``.

    Scoped by PATH, not by "every item in the session": this hook is
    called with the whole collection, including lorrax's own tests when
    the full suite runs, and marking those would make ``--no-services``
    deselect the entire suite.
    """
    import pytest
    here = _TESTS + os.sep
    for item in items:
        if str(getattr(item, "fspath", "")).startswith(here):
            item.add_marker(pytest.mark.services)
            item.add_marker(pytest.mark.vcoul)
