"""The wfn_loader suite's own setup: path, then markers.

Two things, in the order they have to happen.

1. PATH.  Put the in-tree ``wfn_loader`` and its ``lxkit`` foundation on
   ``sys.path``.  Installing the services registers :mod:`lxkit.testing`
   through the ``pytest11`` entry point, but this suite must also run from
   a bare checkout — ``lx`` rewrites the container ``PYTHONPATH`` to
   exactly ``<checkout>/src`` and the Shifter image pip-installs nothing —
   so the harness is imported directly.  ``pytest_plugins`` is not an
   option: pytest honours it only in the ROOTDIR conftest, and this suite
   is collected under the monorepo root.

2. MARKERS.  Every cell under this directory gets ``services`` and
   ``wfn_loader``, so the main suite can select or deselect the whole
   service without naming paths.

Why the markers arrive through a HOOK and not ``pytestmark``: a
module-level ``pytestmark`` applies to the module that declares it, and
pytest does not read one out of a conftest.  Writing ``pytestmark = [...]``
here would be silent — the suite would collect, the marks would not exist,
and ``-m wfn_loader`` would select nothing while looking like it worked.
``pytest_collection_modifyitems`` is the mechanism that actually applies,
and ``tests/test_service_selection.py`` measures that it did.

THREE THINGS distrib_la's conftest DOES THAT THIS ONE DELIBERATELY DOES
NOT, each because copying it would break something that works today:

* ``--xla_force_host_platform_device_count=4`` / ``CUDA_VISIBLE_DEVICES``
  pinning.  Those exist for distrib_la's emulated-mesh and FFI tiers.
  Every cell here runs its assertion in a SUBPROCESS with no jax at all
  (:func:`lxkit.testing.import_isolation`), so there is no device count
  for this suite to have an opinion about — and an inert-looking
  ``os.environ`` write at conftest scope is exactly the kind of knob that
  is inert until the day somebody adds a cell that imports jax.

* ``JAX_ENABLE_X64``.  Same reason: nothing in this suite computes.

* :func:`lxkit.testing.arm_skip_honesty`, and the three hook functions
  distrib_la imports beside it.  ``arm_skip_honesty`` writes a
  PROCESS-GLOBAL ``_ARMED`` dict (``lxkit/testing.py``), so a second
  caller does not add a second gate — it REPLACES the first one's scope.
  In a full-suite run the conftests load in directory order, so a call
  here would land after ``services/distrib_la/tests/conftest.py`` and
  silently disarm the gate that found the LORRAX_FFI_HOST_SO restore
  defect on 2026-08-07.  Importing ``pytest_runtest_logreport`` /
  ``pytest_sessionfinish`` into a second conftest has the matching
  problem: pytest registers hooks per conftest module, so each skip would
  be recorded twice and the gate would run twice.  One gate per service is
  the charter's rule and lxkit cannot express it yet; carrying the gap
  here is better than taking distrib_la's gate down to close it.  The
  fix is an lxkit change (per-scope arming), and it is not this
  extraction's to make.
"""

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))

for _svc in ("lxkit", "wfn_loader"):
    _src = os.path.join(_SERVICES, _svc, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)


def pytest_collection_modifyitems(config, items):
    """Mark everything under this directory ``services`` + ``wfn_loader``.

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
            item.add_marker(pytest.mark.wfn_loader)
