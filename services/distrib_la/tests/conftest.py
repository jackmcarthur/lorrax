"""The distrib_la suite's own setup: path, markers, devices, skip-honesty.

Four things, in the order they have to happen.

1. PATH.  Put the in-tree ``distrib_la`` and its ``lxkit`` foundation on
   ``sys.path``.  Installing the services registers :mod:`lxkit.testing`
   through the ``pytest11`` entry point, but this suite must also run from
   a bare checkout — ``lx`` rewrites the container ``PYTHONPATH`` to
   exactly ``<checkout>/src`` and the Shifter image pip-installs nothing —
   so the autouse fixture and the hooks are imported directly.
   ``pytest_plugins`` is not an option: pytest honours it only in the
   ROOTDIR conftest, and this suite is collected under the monorepo root.

2. DEVICES, before the first jax import.  See :data:`_XLA_FLAGS` below.

3. MARKERS.  Every cell under this directory gets ``services`` and
   ``distrib_la``, so the main suite can select or deselect the whole
   service without naming paths.

4. SKIP-HONESTY.  One gate per service (charter), armed here.

Why the markers arrive through a HOOK and not ``pytestmark``: a
module-level ``pytestmark`` applies to the module that declares it, and
pytest does not read one out of a conftest.  Writing ``pytestmark = [...]``
here would be silent — the suite would collect, the marks would not exist,
and ``-m distrib_la`` would select nothing while looking like it worked.
``pytest_collection_modifyitems`` is the mechanism that actually applies,
and ``tests/test_service_selection.py`` measures that it did.
"""

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))

for _svc in ("lxkit", "distrib_la"):
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
# it is "did this conftest load before anything imported jax".
#
# THAT IS THE BEHAVIOUR WE WANT, and it is worth being explicit about
# because the alternative is a real hazard rather than a stylistic one.
# When someone runs the service suite directly ---
#
#     pytest services/distrib_la/tests
#
# --- this file is an INITIAL conftest (pytest loads the conftests along
# the path of every command-line argument before it imports a single test
# module), nothing has imported jax, the flag takes, and the L-b cells run
# on a real 2x2.  When the FULL suite runs, ``testpaths = ["tests",
# "services"]`` collects tests/ first, some module there imports jax during
# collection, and by the time this file loads the flag is inert.  The L-b
# cells then SKIP — via ``lxkit.testing.require_devices``, which skips and
# never asserts (tests/KNOWN_FAILURES.md lists 11 cells that failed a
# 1-device leg purely for writing ``assert n_dev >= 4``).
#
# Inert is the CORRECT outcome there, not a degradation.  Forcing four host
# devices on the whole monorepo run would change the device count every
# other lorrax test sees: cells that skip below four devices today would
# start running, inside a leg whose reference numbers were measured at one.
# The full-suite set-diff is the check that this stayed true, and it is
# empty in both directions on WSL.
#
# ``setdefault``: an explicit XLA_FLAGS from the caller always wins.
_XLA_FLAGS = "--xla_force_host_platform_device_count=4"
if "jax" not in sys.modules:
    _existing = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in _existing:
        os.environ["XLA_FLAGS"] = (
            f"{_existing} {_XLA_FLAGS}".strip() if _existing else _XLA_FLAGS)

# x64 or the complex128 cells silently measure complex64.  tests/conftest.py
# does this for the monorepo suite; a service-only run has no tests/conftest.
os.environ.setdefault("JAX_ENABLE_X64", "1")

from lxkit.testing import (                        # noqa: E402,F401
    AllowedSkip, arm_skip_honesty, gate_state,     # gate_state is autouse
    machine_profile, pytest_runtest_logreport, pytest_sessionfinish,
)

#: Skips this SERVICE is allowed to emit on a machine that declares a
#: profile, on top of the universal rows lxkit ships.  Every row names the
#: leg that runs the cell instead; a skip nobody can point at a covering
#: run for is evaporated coverage.
_ALLOWED = (
    # The C++ acceptance tier is env-parameterized.  Off Perlmutter there
    # is no pinned .so to check, and the skip says which pin is missing.
    # ON Perlmutter this is NOT allowed — BUILD_NOTES pins the host lib,
    # so a skip there means the pin was not passed and the tier evaporated.
    AllowedSkip("test_so_acceptance", "LORRAX_FFI_HOST_SO",
                "the Perlmutter leg, where BUILD_NOTES.md pins the host .so "
                "(this row is REMOVED by the perlmutter profile: see "
                "_allowed_for)"),
    AllowedSkip("", "no CUDA GPU visible",
                "the GPU leg (`lx test`, one task with every GPU)"),
    AllowedSkip("", "jax cpu backend unavailable",
                "any leg with a working jax"),
    AllowedSkip("", "not built on this machine",
                "the Perlmutter leg with the BUILD_NOTES .so pins; ABSENT "
                "is the honest skip and BUILT-AND-BROKEN is a FAILURE, "
                "which is the distinction lxkit.testing.absent_or_broken "
                "makes and this row does not weaken"),
    AllowedSkip("", "monorepo wiring",
                "the monorepo run; a standalone install has no lorrax"),
)


def _allowed_for(profile):
    """Rows a machine that PROMISES the .so pins does not get to use.

    The env-parameterized C++ tier skipping is fine on a laptop and a
    defect on Perlmutter, where BUILD_NOTES.md pins the library it checks.
    Filtering here rather than writing two lists keeps the reason for the
    difference next to the rows.
    """
    if profile.machine == "perlmutter":
        return tuple(r for r in _ALLOWED
                     if r.where != "test_so_acceptance")
    return _ALLOWED


#: The directory this gate rules on.  Scoped, because the allowlist
#: describes THIS service's skips: unscoped, the gate would also judge the
#: ~45 device-count skips lorrax's own suite emits on a 1-device leg, and a
#: gate that fires on other people's business gets disarmed.  A directory
#: rather than a nodeid prefix — see arm_skip_honesty's docstring; rootdir
#: differs between the monorepo run and the standalone one.
PROFILE = arm_skip_honesty(scope=_TESTS,
                           extra_allowed=_allowed_for(machine_profile()))


def pytest_collection_modifyitems(config, items):
    """Mark everything under this directory ``services`` + ``distrib_la``.

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
            item.add_marker(pytest.mark.distrib_la)
