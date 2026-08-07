"""The zeta_loader suite's own setup: path, devices, markers.

STUB.  The suite itself is step 2 of this service's process; this file is
the shape it will be collected in, landed with the extraction so the
markers and the path exist from the first commit rather than arriving with
the first test.  Three of distrib_la's four sections are here; the fourth
(SKIP-HONESTY) is deliberately absent and says why below.

1. PATH.  Put the in-tree ``zeta_loader`` and its ``lxkit`` foundation on
   ``sys.path``.  Installing the services registers :mod:`lxkit.testing`
   through the ``pytest11`` entry point, but this suite must also run from
   a bare checkout — ``lx`` rewrites the container ``PYTHONPATH`` to
   exactly ``<checkout>/src`` and the Shifter image pip-installs nothing —
   so the hooks are imported directly.  ``pytest_plugins`` is not an
   option: pytest honours it only in the ROOTDIR conftest, and this suite
   is collected under the monorepo root.

2. DEVICES, before the first jax import.  See :data:`_XLA_FLAGS` below.

3. MARKERS.  Every cell under this directory gets ``services`` and
   ``zeta_loader``, so the main suite can select or deselect the whole
   service without naming paths.

WHY THE SKIP-HONESTY GATE IS NOT ARMED HERE, YET.  Two reasons, both
measurable rather than stylistic:

* :func:`lxkit.testing.arm_skip_honesty` writes a single module-level
  ``_ARMED`` dict.  ``services/distrib_la/tests/conftest.py`` already
  arms it, and in a full-suite run both conftests load — so a second
  ``arm_skip_honesty`` call would OVERWRITE distrib_la's scope and
  allowlist with this service's, silently disarming a gate that is
  currently doing its job.  Arming a second service needs lxkit to hold
  a list rather than a dict, which is an edit to a READ-ONLY package on
  this branch: registered for step 2, not taken here.
* An allowlist is a claim about the skips a suite emits, and this suite
  emits none because it has no cells.  A gate armed over an empty scope
  measures nothing; the honest version arrives with the tests.

Why the markers arrive through a HOOK and not ``pytestmark``: a
module-level ``pytestmark`` applies to the module that declares it, and
pytest does not read one out of a conftest.  Writing ``pytestmark = [...]``
here would be silent — the suite would collect, the marks would not exist,
and ``-m zeta_loader`` would select nothing while looking like it worked.
``pytest_collection_modifyitems`` is the mechanism that actually applies.
"""

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))

for _svc in ("lxkit", "zeta_loader"):
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
# it is "did this conftest load before anything imported jax".  Running the
# service suite BY PATH makes this an initial conftest and the flag takes;
# in the full-suite run ``testpaths = ["tests", "services"]`` collects
# tests/ first, jax is already imported, and the flag is inert — which is
# the CORRECT outcome, not a degradation, because forcing four host devices
# on the whole monorepo run would change the device count every other
# lorrax test sees.  ``setdefault``: an explicit XLA_FLAGS always wins.
_XLA_FLAGS = "--xla_force_host_platform_device_count=4"
if "jax" not in sys.modules:
    _existing = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in _existing:
        os.environ["XLA_FLAGS"] = (
            f"{_existing} {_XLA_FLAGS}".strip() if _existing else _XLA_FLAGS)

# x64 or the complex128 cells silently measure complex64.  tests/conftest.py
# does this for the monorepo suite; a service-only run has no tests/conftest.
os.environ.setdefault("JAX_ENABLE_X64", "1")


def pytest_collection_modifyitems(config, items):
    """Mark everything under this directory ``services`` + ``zeta_loader``.

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
            item.add_marker(pytest.mark.zeta_loader)
