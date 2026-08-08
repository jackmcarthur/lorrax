"""The zeta_loader suite's own setup: path, devices, markers, skip-honesty.

Four things, in the order they have to happen — distrib_la's shape, because
this is the SERVICE FORM and wave-1 Fables conform rather than reinvent.

1. PATH.  Put the in-tree ``zeta_loader`` and its ``lxkit`` foundation on
   ``sys.path``, plus THIS directory (``zeta_synth`` is imported by name by
   every cell and by the multi-rank CLI script, which runs with nothing but
   ``<checkout>/src`` on the path).  Installing the services would register
   :mod:`lxkit.testing` through the ``pytest11`` entry point, but nothing is
   installed here — ``lx`` rewrites the container ``PYTHONPATH`` to exactly
   ``<checkout>/src`` and the Shifter image pip-installs nothing — so the
   hooks are imported directly.  ``pytest_plugins`` is not an option: pytest
   honours it only in the ROOTDIR conftest, and this suite is collected
   under the monorepo root.

2. DEVICES, before the first jax import.  See :data:`_XLA_FLAGS` below.

3. MARKERS.  Every cell under this directory gets ``services`` and
   ``zeta_loader``, so the main suite can select or deselect the whole
   service without naming paths.

4. SKIP-HONESTY.  One gate per service (charter), armed below — and armed
   WITHOUT disarming distrib_la's, which is the part that needed building
   rather than copying.  See THE SINGLE-SLOT GATE.

Why the markers arrive through a HOOK and not ``pytestmark``: a module-level
``pytestmark`` applies to the module that declares it, and pytest does not
read one out of a conftest.  Writing ``pytestmark = [...]`` here would be
silent — the suite would collect, the marks would not exist, and
``-m zeta_loader`` would select nothing while looking like it worked.
``pytest_collection_modifyitems`` is the mechanism that actually applies, and
``test_zeta_loader_skip_honesty.py`` measures that it did.
"""

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))
_REPO = os.path.dirname(_SERVICES)

for _svc in ("lxkit", "zeta_loader"):
    _src = os.path.join(_SERVICES, _svc, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

# ---------------------------------------------------------------------------
# Layer L-b: four emulated devices — but ONLY when this suite is the reason
# the process started.
# ---------------------------------------------------------------------------
# ``--xla_force_host_platform_device_count`` is read by XLA at the FIRST jax
# import in a process and never again, and jax is imported at module scope by
# a great many lorrax test modules.  So this setting is not "on or off"; it is
# "did this conftest load before anything imported jax".
#
# THAT IS THE BEHAVIOUR WE WANT, and it is worth being explicit about because
# the alternative is a real hazard rather than a stylistic one.  When someone
# runs the service suite directly ---
#
#     pytest services/zeta_loader/tests
#
# --- this file is an INITIAL conftest (pytest loads the conftests along the
# path of every command-line argument before it imports a single test module),
# nothing has imported jax, the flag takes, and the L-b cells run on a real
# 2x2 of emulated host devices.  When the FULL suite runs, ``testpaths =
# ["tests", "services"]`` collects tests/ first, some module there imports jax
# during collection, and by the time this file loads the flag is inert.  The
# L-b cells then SKIP — via ``lxkit.testing.require_devices``, which skips and
# never asserts (tests/KNOWN_FAILURES.md lists 11 cells that failed a
# 1-device leg purely for writing ``assert n_dev >= 4``).
#
# Inert is the CORRECT outcome there, not a degradation.  Forcing four host
# devices on the whole monorepo run would change the device count every other
# lorrax test sees: cells that skip below four devices today would start
# running, inside a leg whose reference numbers were measured at one.  The
# full-suite set-diff is the check that this stayed true.
#
# ``setdefault``: an explicit XLA_FLAGS from the caller always wins.
_XLA_FLAGS = "--xla_force_host_platform_device_count=4"
if "jax" not in sys.modules:
    _existing = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in _existing:
        os.environ["XLA_FLAGS"] = (
            f"{_existing} {_XLA_FLAGS}".strip() if _existing else _XLA_FLAGS)

# x64 or the complex128 cells silently measure complex64 — and every ζ payload
# in this suite is complex128 by contract (``read_zeta_G_slab`` states the
# dtype in its own signature).  tests/conftest.py does this for the monorepo
# suite; a service-only run has no tests/conftest.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import lxkit.testing as _lxt                                   # noqa: E402
from lxkit.testing import (                                    # noqa: E402,F401
    AllowedSkip, arm_skip_honesty, assert_skips_match_profile,
    gate_state,                                    # gate_state is autouse
    machine_profile, observed_skips,
)

# ---------------------------------------------------------------------------
# The skips THIS service is allowed to emit, on top of lxkit's universal rows
# ---------------------------------------------------------------------------
# Every row names the leg that runs the cell instead.  A skip nobody can point
# at a covering run for is evaporated coverage wearing a green dot, and the
# allowlist is where that claim is written down.
_ALLOWED = (
    # THE WSL FACT.  The dev box has no phdf5 FFI library at all, so every
    # collective SlabIO read refuses AT OPEN.  ABSENT, not BROKEN: nothing is
    # built here to be broken, and `zeta_synth.slab_io_state` quotes the
    # probe's own stage into the reason so the report says which.  The cells
    # that need the transport are exactly the ones whose 4-rank answers come
    # from leg L-c anyway.
    AllowedSkip("", "SlabIO transport unavailable",
                "leg L-c on Perlmutter: `lx run --cpu -N 1 -n 4 python3 "
                "services/zeta_loader/tests/test_zeta_loader_multiproc.py "
                "--mesh 2x2 --tmpdir $SCRATCH/...`, under the BUILD_NOTES "
                "FFI pins (the Aug-7 pair; the Aug-6 one dies at the "
                "96a6399 signature)"),
    # The header binders and gvec_fft_box live in the host tree until wave
    # 1b extracts them.  A standalone install has no lorrax, so the DATA
    # cells cannot run there; the FORMAT cells (probe/write_g0_mu) have no
    # such row because they have no such dependency, which is the whole
    # claim test_zeta_loader_import_isolation makes.
    AllowedSkip("", "no lorrax host tree",
                "the monorepo run; a standalone install has no file_io/ "
                "for the header binders to come from"),
)


def _allowed_for(profile):
    """Rows a machine that PROMISES the phdf5 handler does not get to use.

    The transport row is a platform fact on a laptop and a DEFECT on
    Perlmutter, where the machine profile declares ``MustRow("phdf5",
    "slab_io", "lorrax_phdf5_read", "cpu")`` and BUILD_NOTES pins the
    library it lives in.  Filtering here rather than writing two lists
    keeps the reason for the difference next to the row — distrib_la's
    ``_allowed_for`` does the same thing for its ``.so`` acceptance tier.
    """
    if profile.machine == "perlmutter":
        return tuple(r for r in _ALLOWED
                     if "SlabIO transport" not in r.why)
    return _ALLOWED


PROFILE = machine_profile()
_EXTRA_ALLOWED = _allowed_for(PROFILE)

# ---------------------------------------------------------------------------
# ONE GATE PER SERVICE, armed under THIS directory
# ---------------------------------------------------------------------------
# This used to be a careful dance around a single-slot ``lxkit.testing._ARMED``:
# arm it only when free, and otherwise re-implement lxkit's
# ``pytest_sessionfinish`` locally so that neither service disarmed the other.
# The registered consolidation wish ("``_ARMED`` should be a LIST of armed
# scopes") landed, so the dance is gone: ``arm_skip_honesty`` registers a row
# under THIS directory and cannot touch anyone else's.
# ``test_zeta_loader_skip_honesty`` asserts that property directly.
from lxkit.testing import (                                    # noqa: E402,F401
    pytest_configure, pytest_runtest_logreport, pytest_sessionfinish,
)

arm_skip_honesty(PROFILE, scope=_TESTS, extra_allowed=_EXTRA_ALLOWED)


def pytest_collection_modifyitems(config, items):
    """Mark everything under this directory ``services`` + ``zeta_loader``.

    Scoped by PATH, not by "every item in the session": this hook is called
    with the whole collection, including lorrax's own tests when the full
    suite runs, and marking those would make ``--no-services`` deselect the
    entire suite.
    """
    import pytest
    here = _TESTS + os.sep
    for item in items:
        if str(getattr(item, "fspath", "")).startswith(here):
            item.add_marker(pytest.mark.services)
            item.add_marker(pytest.mark.zeta_loader)
