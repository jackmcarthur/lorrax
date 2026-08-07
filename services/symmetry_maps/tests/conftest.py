"""The symmetry_maps suite's own setup: path and markers.  Two things.

1. PATH.  Put the in-tree ``symmetry_maps`` and its ``lxkit`` foundation on
   ``sys.path``.  Installing the services registers :mod:`lxkit.testing`
   through the ``pytest11`` entry point, but this suite must also run from
   a bare checkout — ``lx`` rewrites the container ``PYTHONPATH`` to
   exactly ``<checkout>/src`` and the Shifter image pip-installs nothing —
   so the path is set here.  ``pytest_plugins`` is not an option: pytest
   honours it only in the ROOTDIR conftest, and this suite is collected
   under the monorepo root.

2. MARKERS.  Every cell under this directory gets ``services`` and
   ``symmetry_maps``, so the main suite can select or deselect the whole
   service without naming paths.

Why the markers arrive through a HOOK and not ``pytestmark``: a
module-level ``pytestmark`` applies to the module that declares it, and
pytest does not read one out of a conftest.  Writing ``pytestmark = [...]``
here would be silent — the suite would collect, the marks would not exist,
and ``-m symmetry_maps`` would select nothing while looking like it worked.
``pytest_collection_modifyitems`` is the mechanism that actually applies,
and ``tests/test_service_selection.py`` measures that it did.

WHAT IS DELIBERATELY NOT HERE YET.  The L-b tier's four emulated devices
(``--xla_force_host_platform_device_count=4``, which XLA reads at the
FIRST jax import in a process and never again) and the skip-honesty gate
(``lxkit.testing.arm_skip_honesty``, whose allowlist has to name the skips
this suite actually emits) both belong to the test-suite step and land
with the cells they serve.  Arming a gate over an empty directory would
measure nothing while looking armed; see
``services/distrib_la/tests/conftest.py`` for the shape both take.
"""

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))

for _svc in ("lxkit", "symmetry_maps"):
    _src = os.path.join(_SERVICES, _svc, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

# x64 or the complex128 cells silently measure complex64.  tests/conftest.py
# does this for the monorepo suite; a service-only run has no tests/conftest.
os.environ.setdefault("JAX_ENABLE_X64", "1")


def pytest_collection_modifyitems(config, items):
    """Mark everything under this directory ``services`` + ``symmetry_maps``.

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
            item.add_marker(pytest.mark.symmetry_maps)
