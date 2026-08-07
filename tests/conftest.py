"""Shared pytest setup for the LORRAX_A test suite.

JAX must be configured for x64 BEFORE the first ``import jax`` in the
process, otherwise ``jnp.complex128`` silently degrades to complex64
(see jax-ml/jax#current-gotchas).  Pytest collects all test modules
into one process, so the first import wins — set the env here.
"""

import os
import sys as _sys
from pathlib import Path as _Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

# ``harness`` first, and BEFORE the GPU pin below, because the pin's
# decision lives there as a pure function so it can be unit-tested (a
# module-scope side effect in a conftest is otherwise unfalsifiable).
# Safe at this point: harness imports stdlib + numpy and no jax, so
# nothing has initialised a CUDA backend yet.
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import harness  # noqa: E402

# ---------------------------------------------------------------------------
# ONE GPU PER PROCESS.  Not a preference — three separate things require it.
# ---------------------------------------------------------------------------
# Under pytest-xdist each worker takes its own GPU (gw0 → GPU 0, gw1 → GPU
# 1, …) so the e2e regression gates — subprocess launchers that each need
# ONE GPU — run N-wide on an N-GPU node instead of serially on GPU 0.
# WITHOUT a worker id the process takes the FIRST visible device.  Must run
# before the first CUDA/JAX init, which is why it lives at conftest module
# scope.  This OVERRIDES any pre-set CUDA_VISIBLE_DEVICES (SLURM gres sets
# "0,1,2,3" for the task); the mapping goes through the existing list so
# SLURM's device selection is respected.
#
# THE `if _wid.startswith("gw")` GUARD THIS REPLACES WAS A REAL DEFECT, and
# it is worth writing down because it hid behind a green leg.  Three things
# want exactly one visible GPU and only one of them is about xdist:
#
#   1. xdist fan-out (the original reason).
#   2. The 1-GPU-FROZEN REFERENCES.  Every gate subprocess inherits this
#      env; seeing N GPUs makes it build an N-device mesh and compare
#      against numbers measured at one.  That is true whether or not xdist
#      is running, so gating it on the worker id made a NON-xdist run of
#      the same suite a different measurement.
#   3. SLATE REFUSES.  MEASURED, Perlmutter 2026-08-07, step 2:
#          FAILED_PRECONDITION: slate.potrf: blas::get_device_count()=4
#          but JAX one-process-per-GPU model requires exactly 1.
#      EIGHT contract cells (slate potrf / batched_potrf / eigh, plus the
#      bse_setup wiring cell on top of them) died on it in the
#      SERVICE-ONLY leg (`pytest services/distrib_la/tests`), which never
#      loads this file at all.  services/distrib_la/tests/conftest.py does
#      the pinning too — but conditionally, on `"jax" not in sys.modules`,
#      since CUDA_VISIBLE_DEVICES is read once at backend init.  In a
#      full-suite run `testpaths = ["tests", "services"]` collects tests/
#      first, a lorrax module there imports jax during collection, and the
#      service conftest's copy is INERT by the time it loads.  So the
#      service suite's own guard cannot cover the full-suite leg by
#      construction; the only conftest that loads early enough is this one.
#
#      DO NOT REUSE THIS AS THE EXPLANATION OF THE FULL-SUITE LEG.  An
#      earlier revision of this comment said the same eight cells failed
#      "and ONLY in the full-suite -m distrib_la leg".  They did fail
#      there, with a DIFFERENT number — `get_device_count()=0`, at exactly
#      ONE visible device, before and after this pin became unconditional
#      — and a different cause: the two platform .so's share
#      libslate.so.2 / libblaspp.so.2 by SONAME and the host build's
#      blaspp has no CUDA to ask.  Fixed by a load-order rule in both
#      loaders (`_open_cuda_before_host`), measured with dladdr in both
#      legs.  This pin is still all three of the things above; it was
#      never the cause of the eight, and believing it was is how the
#      second cause stayed hidden behind the first for a day.
#
# The service conftest keeps its copy — it is the one that runs when the
# suite is invoked BY PATH, which never loads this file at all.
#
# No-op where there is nothing to pin: unset CUDA_VISIBLE_DEVICES and no
# nvidia-smi (every CPU leg, the WSL box) leaves the environment untouched.
# ---------------------------------------------------------------------------
_pinned = harness.pin_one_gpu(
    os.environ.get("CUDA_VISIBLE_DEVICES"),
    os.environ.get("PYTEST_XDIST_WORKER", ""))
if _pinned is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pinned


# ---------------------------------------------------------------------------
# Session-scoped e2e states (the Tier-1 gates double as prepared state for
# the Tier-2 invariance gates).
#
# ``gnppm_session`` runs the shrunk MoS2 3×3 GN-PPM fixture ONCE, fresh
# (restart = false), and keeps the run dir — including ``tmp/`` with the
# ISDF restart file (isdf_tensors_*.h5) and zeta_q.h5.  The Tier-1 frozen
# gate asserts on this run's outputs; every Tier-2 variant re-runs the
# driver with ``restart = true`` from a COPY of the state (the ζ-fit and
# V_q build — the dominant cost — are not redone).  Copies are mandatory:
# the driver writes W0_qmunu + head scalars back into the restart file
# (gw_output.persist_w0_and_head).
#
# ``gnppm_restart_baseline`` is the canonical restart variant (one-shot,
# freq-debug writers off — the config every other dynamic variant diffs
# against); its equality with the fresh session run IS the
# restart-roundtrip gate.
#
# ``bispinor_session`` is the fresh bispinor GN-PPM run (bispinor restart
# is not yet supported — gw_init.py marks the transverse bundle
# not-restartable — so its Tier-2 pad-flip gate reruns fresh).
#
# Under pytest-xdist each worker builds its own session state (session
# fixtures are per-process); tests stay order-independent and xdist-safe
# because no test mutates a session dir — every variant copies first.
# ---------------------------------------------------------------------------
from types import SimpleNamespace as _NS

import pytest


# ---------------------------------------------------------------------------
# Service-tier selection: --no-services / --only-service=NAME /
# LX_SKIP_SERVICES.  THE FIRST COLLECTION HOOKS IN THIS TREE.
# ---------------------------------------------------------------------------
# The services under services/ ship their own fast suites and are staged
# into this one (charter).  Deselecting them has to go through a hook and
# NOT through a second `-m`, because pyproject sets
#
#     addopts = "-m 'not extra'"
#
# and an explicit `-m` on the command line REPLACES that default instead of
# composing with it.  `pytest -m "not services"` would therefore silently
# re-enable the entire `extra` tier — 26 deselected suites — while looking
# like it had narrowed the run.  A hook composes with addopts by
# construction, since it acts on the items `-m` already selected.
#
# This supersedes the census's leg A2 `--ignore=tests/test_ffi_linalg_
# contract.py`.  That file lives in services/distrib_la/tests now and
# carries the `distrib_la` marker, so the leg that could not run it (a bare
# no-srun launch, where a loadable host .so kills the interpreter at MPI
# init — KNOWN_FAILURES.md) says `--no-services` and the leg that wants it
# says `-m distrib_la`.  Naming a path was always a proxy for naming the
# thing; now the thing has a name.
#
# KEPT MINIMAL ON PURPOSE.  These are the only collection hooks in the
# tree, and a collection hook that is subtly wrong deselects real coverage
# without failing anything — so this is 20 lines of predicate and the
# measurement of what it selected lives in tests/test_service_selection.py,
# which runs pytest --collect-only in a subprocess and DIFFS the sets.

_SERVICES_ROOT = str(_Path(__file__).resolve().parent.parent / "services")


def pytest_addoption(parser):
    group = parser.getgroup("services", "LORRAX service suites")
    group.addoption(
        "--no-services", action="store_true", default=False,
        help="deselect every services/ suite (the staged standalone "
             "service tests).  Composes with addopts; never use a second "
             "-m for this.")
    group.addoption(
        "--only-service", action="store", default="", metavar="NAME",
        help="run ONLY the named service's suite (e.g. --only-service="
             "distrib_la).  Deselects lorrax's own tests and every other "
             "service.")


def _service_of(item) -> str:
    """The service a collected item belongs to, or "" for lorrax's own.

    Keyed on the PATH rather than on the marker: the marker is applied by
    the service's own conftest, so trusting it here would make the
    deselection silently depend on a service having remembered to add it.
    """
    path = str(getattr(item, "fspath", ""))
    if not path.startswith(_SERVICES_ROOT + os.sep):
        return ""
    return path[len(_SERVICES_ROOT) + 1:].split(os.sep)[0]


def pytest_collection_modifyitems(config, items):
    only = (config.getoption("--only-service") or "").strip()
    skip_env = (os.environ.get("LX_SKIP_SERVICES", "") or "").strip()
    no_services = (config.getoption("--no-services")
                   or skip_env not in ("", "0", "false", "no"))
    if not no_services and not only:
        return

    keep, drop = [], []
    for item in items:
        svc = _service_of(item)
        if only:
            (keep if svc == only else drop).append(item)
        else:
            (drop if svc else keep).append(item)
    if drop:
        reason = (f"--only-service={only}" if only else
                  ("--no-services" if config.getoption("--no-services")
                   else f"LX_SKIP_SERVICES={skip_env}"))
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
        config.stash.setdefault(_DESELECT_NOTE, []).append(
            f"{reason}: deselected {len(drop)} service test(s)")


_DESELECT_NOTE = pytest.StashKey[list]()


def pytest_report_header(config):
    """Say it out loud.  A run that quietly dropped a whole tier and
    reported a smaller green number is exactly the shape of the losses
    this tree keeps finding, so the deselection announces itself."""
    only = (config.getoption("--only-service") or "").strip()
    if only:
        return f"services: ONLY {only} (lorrax's own tests deselected)"
    if config.getoption("--no-services") or os.environ.get(
            "LX_SKIP_SERVICES", "").strip() not in ("", "0", "false", "no"):
        return "services: DESELECTED (--no-services / LX_SKIP_SERVICES)"
    return None


def pytest_sessionstart(session):
    """Make the checked-in regression fixtures read-only before anything runs.

    A gate stager that symlinks (rather than copies) a fixture into its run
    dir lets the driver write its OUTPUT through the link and destroy the
    fixture — which happened to
    ``tests/regression/cohsex_debug/sigma_mnk.h5`` on 2026-07-25, silently.
    ``a-w`` turns that into an immediate EACCES.  ``harness.copy_fixture``
    restores owner-write on the run-dir COPY, so nothing legitimate breaks.
    """
    changed = harness.protect_fixtures()
    if changed:
        tw = session.config.get_terminal_writer()
        tw.line(f"[fixtures] made {len(changed)} regression fixture file(s) "
                f"read-only (see harness.protect_fixtures)")


def _run_session_case(tmp_path_factory, case_name, input_name, output_name):
    import pytest as _pytest
    harness.skip_unless_gpu(_pytest)
    case_dir = harness.REG / case_name
    run_dir = harness.copy_fixture(
        case_dir, tmp_path_factory.mktemp(f"{case_name}_session") / case_name)
    res = harness.run_gw_jax(run_dir, input_name)
    if res.returncode != 0:
        _pytest.fail(
            f"{case_name} session run failed.\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
    out = run_dir / output_name
    assert out.exists(), f"session run wrote no {out}"
    return _NS(run_dir=run_dir, input_name=input_name,
               output_name=output_name, stdout=res.stdout)


@pytest.fixture(scope="session")
def gnppm_session(tmp_path_factory):
    """Fresh (restart=false) run of the gnppm fixture; Tier-1 state."""
    return _run_session_case(
        tmp_path_factory, "gnppm_debug", "gnppm_test.in",
        "sigma_diag_gnppm_test.dat")


@pytest.fixture(scope="session")
def gnppm_restart_baseline(gnppm_session, tmp_path_factory):
    """Canonical restart=true variant of the gnppm session state.

    One-shot, freq-debug writers off (historical: the since-removed
    kij_stream mode crashed on the debug writers' None-Σ_c handling; the
    baseline all dynamic variants diff against keeps the same debug-off
    config so existing goldens stay comparable).
    """
    run_dir = harness.copy_fixture(
        harness.REG / "gnppm_debug",
        tmp_path_factory.mktemp("gnppm_restart") / "baseline",
        tmp_from=gnppm_session.run_dir)
    harness.mutate_input(run_dir / "gnppm_test.in", {
        "restart = false": "restart = true",
        "sigma_freq_debug_output = true": "sigma_freq_debug_output = false",
    })
    res = harness.run_gw_jax(run_dir, "gnppm_test.in")
    if res.returncode != 0:
        pytest.fail(
            f"gnppm restart baseline failed.\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
    return _NS(run_dir=run_dir, input_name="gnppm_test.in",
               output_name=gnppm_session.output_name, stdout=res.stdout,
               session=gnppm_session)


@pytest.fixture(scope="session")
def bispinor_session(tmp_path_factory):
    """Fresh run of the bispinor GN-PPM fixture; Tier-1 state."""
    return _run_session_case(
        tmp_path_factory, "bispinor_debug", "bispinor_test.in",
        "sigma_diag_bispinor_test.dat")


@pytest.fixture(scope="session")
def bse_dense_state(gnppm_session, tmp_path_factory):
    """Padded, head-injected (px=py=1) BSE arrays from the gnppm restart.

    Copies the gnppm session run dir (incl. ``tmp/`` restart state) once, then
    loads a 2v2c BSE subset via ``bse_io._load_ring_subset`` — a plain library
    call, no driver subprocess, so the session state is never mutated. MoS2
    3×3×1 ⇒ nk=9 ⇒ N = nc·nv·nk = 36. Shared by the dense-reference gate and
    the trial-stack matvec gate.
    """
    from bse import bse_io

    run_dir = harness.copy_fixture(
        harness.REG / "gnppm_debug",
        tmp_path_factory.mktemp("bse_dense") / "gnppm_debug",
        tmp_from=gnppm_session.run_dir)
    input_path = str(run_dir / "gnppm_test.in")
    restart = bse_io._find_restart_file(input_path)
    return bse_io._load_ring_subset(
        restart, n_val=2, n_cond=2, px=1, py=1, input_file=input_path)


# ---------------------------------------------------------------------------
# Si 4×4×4 sessions.  The production deck is run ONCE and consumed by two
# gates — the self-freeze (``eqp_si_ref.dat``) and the BerkeleyGW anchor
# (``bgw_sigma_hp_noavg.dat``).  Running it twice would double the suite's
# most expensive Si cost for no extra coverage.
#
# Both decks live in the SAME fixture directory and share WFN.h5 / kin_ion.h5 /
# dipole.h5 (26 MB); only the deck, the centroid file and the reference differ.
# That is deliberate — a second directory would duplicate the binaries.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def si_session(tmp_path_factory):
    """Fresh run of the Si production deck (the BerkeleyGW-anchored one)."""
    return _run_session_case(
        tmp_path_factory, "si_cohsex_debug", "cohsex_si_test.in",
        "eqp_si_test.dat")


@pytest.fixture(scope="session")
def si_fast_session(tmp_path_factory):
    """Fresh run of the Si fast deck (20 bands, 144 centroids, ~12 s).

    A separate run dir rather than a restart from ``si_session``: the two decks
    have different band counts and different centroid sets, so they share no
    ISDF state.
    """
    return _run_session_case(
        tmp_path_factory, "si_cohsex_debug", "cohsex_si_fast.in",
        "eqp_si_fast.dat")
