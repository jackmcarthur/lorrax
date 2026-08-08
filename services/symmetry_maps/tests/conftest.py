"""The symmetry_maps suite's own setup: path, devices, markers, skip-honesty.

Four things, in the order they have to happen.

1. PATH.  Put the in-tree ``symmetry_maps`` and its ``lxkit`` foundation on
   ``sys.path``.  Installing the services registers :mod:`lxkit.testing`
   through the ``pytest11`` entry point, but this suite must also run from
   a bare checkout — ``lx`` rewrites the container ``PYTHONPATH`` to
   exactly ``<checkout>/src`` and the Shifter image pip-installs nothing —
   so the path is set here.  ``pytest_plugins`` is not an option: pytest
   honours it only in the ROOTDIR conftest, and this suite is collected
   under the monorepo root.

2. DEVICES, before the first jax import.  See :data:`_XLA_FLAGS` below.

3. MARKERS.  Every cell under this directory gets ``services`` and
   ``symmetry_maps``, so the main suite can select or deselect the whole
   service without naming paths.

4. SKIP-HONESTY.  One gate per service (charter), armed here.

Why the markers arrive through a HOOK and not ``pytestmark``: a
module-level ``pytestmark`` applies to the module that declares it, and
pytest does not read one out of a conftest.  Writing ``pytestmark = [...]``
here would be silent — the suite would collect, the marks would not exist,
and ``-m symmetry_maps`` would select nothing while looking like it worked.
``pytest_collection_modifyitems`` is the mechanism that actually applies,
and ``tests/test_service_selection.py`` measures that it did.
"""

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))

for _svc in ("lxkit", "symmetry_maps"):
    _src = os.path.join(_SERVICES, _svc, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

# THIS directory too, and explicitly.  ``_profiles`` and ``_deck_stub`` are
# test HELPERS, not cells, and a conftest is imported before pytest has
# decided anything about the test package's own sys.path — under the
# monorepo run the rootdir is the repo, not this directory, so relying on
# pytest's basedir insertion would make the import work standalone and fail
# staged.  Measured writing this file.
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

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
#     pytest services/symmetry_maps/tests
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
# The full-suite set-diff is the check that this stayed true.
#
# COPIED, not shared, from ``services/distrib_la/tests/conftest.py``.  Two
# service conftests both doing ``setdefault`` compose correctly (whichever
# loads first wins and the second is a no-op on an already-set value); a
# shared helper would have to live in lxkit, which is read-only this wave.
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
    arm_skip_honesty, gate_state,                  # gate_state is autouse
    pytest_configure, pytest_runtest_logreport, pytest_sessionfinish,
)

from _profiles import allowed_skips_for, profile_for_this_machine  # noqa: E402

#: The directory this gate rules on.  Scoped, because the allowlist
#: describes THIS service's skips: unscoped, the gate would also judge the
#: ~45 device-count skips lorrax's own suite emits on a 1-device leg, and a
#: gate that fires on other people's business gets disarmed.  A directory
#: rather than a nodeid prefix — see arm_skip_honesty's docstring; rootdir
#: differs between the monorepo run and the standalone one.
PROFILE = arm_skip_honesty(profile_for_this_machine(), scope=_TESTS,
                           extra_allowed=allowed_skips_for(
                               profile_for_this_machine()))


def _invocation_narrowed(config) -> bool:
    """True when the caller asked for PART of this suite.

    ``-k``, ``--deselect``, a ``::nodeid`` argument, or a single test FILE
    under this directory.  Naming the directory (or nothing) is the whole
    suite and is not narrowing.
    """
    if (config.getoption("keyword", "") or "").strip():
        return True
    if config.getoption("deselect", None):
        return True
    for arg in getattr(config, "args", ()) or ():
        head = str(arg).split("::")[0]
        if "::" in str(arg):
            return True
        if os.path.isfile(head) and os.path.realpath(head).startswith(
                _TESTS + os.sep):
            return True
    return False


def pytest_collection_modifyitems(config, items):
    """Mark everything under this directory ``services`` + ``symmetry_maps``,
    and settle the skip-honesty floor now that the invocation is known.

    Scoped by PATH, not by "every item in the session": this hook is
    called with the whole collection, including lorrax's own tests when
    the full suite runs, and marking those would make ``--no-services``
    deselect the entire suite.

    THE FLOOR IS RE-ARMED HERE, not left at its profile value, because the
    floor is a claim about the SUITE and the profile is loaded before
    anyone knows what was asked for.  ``pytest
    services/symmetry_maps/tests/test_symmetry_maps_algebra.py`` collects
    far fewer cells than the suite and is a perfectly ordinary thing to
    run; failing it for "below the floor" would be the gate crying wolf,
    and a gate that cries wolf gets disarmed (which is the failure mode
    this whole mechanism exists to avoid).  A narrowed invocation keeps the
    zero-collected check — the case the floor is actually about — and drops
    the size claim.
    """
    import pytest
    here = _TESTS + os.sep
    for item in items:
        if str(getattr(item, "fspath", "")).startswith(here):
            item.add_marker(pytest.mark.services)
            item.add_marker(pytest.mark.symmetry_maps)
    if _invocation_narrowed(config):
        arm_skip_honesty(PROFILE, scope=_TESTS,
                         extra_allowed=allowed_skips_for(PROFILE),
                         min_collected=1)
