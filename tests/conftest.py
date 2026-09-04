"""Shared pytest setup for the LORRAX_A test suite.

JAX must be configured for x64 BEFORE the first ``import jax`` in the
process, otherwise ``jnp.complex128`` silently degrades to complex64
(see jax-ml/jax#current-gotchas).  Pytest collects all test modules
into one process, so the first import wins — set the env here.
"""

import json as _json
import os
import sys as _sys
from pathlib import Path as _Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

# Pytest imports test modules during collection, before any production driver
# can initialize the runtime.  Seal the same metadata-derived package closure
# once here, while only stdlib modules are loaded; individual test modules do
# not own service-path setup.
from runtime.source_closure import ensure_source_closure as _seal_sources
_seal_sources(print_fn=lambda _line: None)

# Some test modules import JAX during collection, before any driver can enter
# ``runtime.set_default_env``.  Route that early-import corner through the
# same owner as production.  CPU sessions deliberately receive no GPU flag;
# GPU sessions merge the measured default without replacing caller flags.
if "jax" not in _sys.modules:
    from runtime import (_gpu_is_present,
                         set_default_xla_gpu_autotune)       # noqa: E402
    set_default_xla_gpu_autotune(
        platform="gpu" if _gpu_is_present() else "cpu")

# The HDF5 operation journal is off by default in production and in the
# suite.  ``tests/test_h5_journal.py`` enables it explicitly in a private
# directory; no global override is needed here and an operator's explicit
# debugging setting remains visible to other tests.

# ``harness`` first, and BEFORE the GPU pin below, because the pin's
# decision lives there as a pure function so it can be unit-tested (a
# module-scope side effect in a conftest is otherwise unfalsifiable).
# Safe at this point: harness imports stdlib + numpy and no jax, so
# nothing has initialised a CUDA backend yet.
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import fast_gate  # noqa: E402
import harness  # noqa: E402
import known_failure_ledger  # noqa: E402

# ---------------------------------------------------------------------------
# ONE GPU PER PROCESS THAT RUNS TESTS.  Not a preference — three separate
# things require it, and a fourth requires that the xdist CONTROLLER be
# excluded from it.
# ---------------------------------------------------------------------------
# Under pytest-xdist each worker takes its own GPU (gw0 → GPU 0, gw1 → GPU
# 1, …) so the e2e regression gates — subprocess launchers that each need
# ONE GPU — run N-wide on an N-GPU node instead of serially on GPU 0.
# WITHOUT a worker id the process takes the FIRST visible device.  Must run
# before the first CUDA/JAX init; ``pytest_configure`` is that moment (it
# runs before collection, so before any test module — and therefore any
# jax — is imported) and it is also the FIRST moment the controller can be
# told apart from a plain single-process run, which is the whole reason
# the pin lives in a hook instead of at module scope.  This OVERRIDES any
# pre-set CUDA_VISIBLE_DEVICES (SLURM gres sets "0,1,2,3" for the task);
# the mapping goes through the existing list so SLURM's device selection is
# respected.
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
# AND THE CONTROLLER DOES NOT PIN.  MEASURED, Perlmutter 2026-08-07
# (KNOWN_FAILURES B2).  Making the pin unconditional gave it to the xdist
# CONTROLLER too — a process with no worker id, so ``pin_one_gpu`` handed
# it ``devs[0]`` and it wrote CUDA_VISIBLE_DEVICES="0" into its OWN
# environ.  The workers are spawned FROM that environ, so all four
# inherited ``preset="0"``, ``_visible_gpus`` returned a one-element list
# in each of them, ``int(wid[2:]) % 1 == 0`` for all four, and the fan-out
# three lines above quietly stopped existing:
#
#   gw0 gw1 gw2 gw3   xdist arm (6 gnppm-session files)
#   '0' '1' '2' '3'   78ddcee        24 / 24 passed
#   '0' '0' '0' '0'   6920171        11 P / 2 F / 11 E, RESOURCE_EXHAUSTED
#                                    … 19.20 GiB on device ordinal 0
#
# The controller runs no tests, so it has nothing to pin FOR; the only
# effect its write can have is to narrow what its workers inherit.  The
# three reasons above are reasons for a process that COMPUTES, and a
# single-process run is still one of those — which is why the arm is
# "controller", read from pytest-xdist's own signal
# (``harness.is_xdist_controller``), and not "no worker id".
#
# No-op where there is nothing to pin: unset CUDA_VISIBLE_DEVICES and no
# nvidia-smi (every CPU leg, the WSL box) leaves the environment untouched.
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Pin this process's GPU, unless this process is the xdist controller.

    ``config.option.dist`` is ``"no"`` for a plain run and for ``-n 0``,
    and ``PYTEST_XDIST_WORKER`` is set in every worker's environment — so
    the two together separate the three cases the pin has to tell apart.

    THE SESSION'S DEVICE LIST IS READ FIRST, before the pin narrows it, and
    carried in the environment for the mesh runner at the bottom of this
    file.  After the pin there is no way left to ask how many GPUs the node
    had, which is why a ``mesh`` cell could never find four.
    """
    os.environ.setdefault(
        harness.SESSION_DEVICES_ENV,
        ",".join(harness.session_devices(
            os.environ.get("CUDA_VISIBLE_DEVICES"))))
    # And the CALLER's device-shape env, for the same reason and at the same
    # moment: this hook runs before collection, so nothing has leaked yet.
    # See harness.SESSION_JAX_ENV — without this the mesh child inherits a
    # JAX_PLATFORMS=cpu that a test module set at import and runs on emulated
    # devices while claiming the node's GPUs.
    os.environ.setdefault(
        harness.SESSION_JAX_ENV,
        _json.dumps({k: os.environ.get(k) for k in harness.JAX_SHAPE_VARS}))

    # The mesh subprocess IS the widened process.  Pinning it would undo the
    # only thing it was started to do.
    if os.environ.get(harness.MESH_CELL_ENV):
        return

    pinned = harness.pin_one_gpu(
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("PYTEST_XDIST_WORKER", ""),
        controller=harness.is_xdist_controller(
            os.environ.get("PYTEST_XDIST_WORKER", ""),
            getattr(config.option, "dist", "no")))
    if pinned is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = pinned

    # THE SUITE DOES NOT PROMOTE ``load`` TO ``loadgroup`` HERE, and the
    # reason is measured rather than argued.  Promoting it looked like the
    # way to make the ``xdist_group`` the mesh marker adds below concentrate
    # every mesh cell on one worker — one child per module instead of one per
    # (worker, module).  MEASURED on Perlmutter 2026-08-10, nid002460: it did
    # not take.  ``tests/test_charge_zeta_route.py``'s eight cells ran in FOUR
    # separate children (pids 233898/233901/233904/233907, the mesh-child log
    # in the node's TMPDIR), so the group never reached the scheduler.
    #
    # Left out rather than left in, because a mechanism that is asserted and
    # does not work is worse than one that is absent: it makes the next reader
    # believe a cost is paid that is not.  The cost is small and is now
    # stated instead — each child is 4-15 s, and a census pays on the order of
    # fifteen of them.  The marker stays: under an EXPLICIT ``--dist
    # loadgroup`` the grouping does work, and ``_plain_nodeid`` below keeps
    # that arrangement correct.


# ---------------------------------------------------------------------------
# NO TEST MODULE RECONFIGURES THE SESSION AT COLLECTION TIME  (P19, 2026-08-09)
# ---------------------------------------------------------------------------
# Snapshot the environment around COLLECTION -- i.e. around the imports of
# every selected test module, all of which happen before any test runs -- and
# refuse the session if a module left anything behind that a fixture should
# have owned.  The decision is a pure function in ``harness`` so it can be
# unit-tested; the RED TWIN is ``tests/_env_leak_twin.py``, a module that
# leaks on purpose (never auto-collected: it does not match ``test_*.py``),
# driven by ``tests/test_env_leak_gate.py``.
#
# Why this is worth a hook rather than a code-review habit: the failure mode
# is invisible in the run that causes it. The leaking file stays green, and
# the cost lands on a DIFFERENT file, in a DIFFERENT arrangement, as a
# verdict that reverses when someone reproduces it alone.
# ---------------------------------------------------------------------------
_ENV_BEFORE_COLLECTION = None


def pytest_collection(session):
    """Snapshot the env before collection. Returns None: the default
    collection implementation still runs (this hook is firstresult, so a
    non-None return would REPLACE collection rather than observe it)."""
    global _ENV_BEFORE_COLLECTION
    _ENV_BEFORE_COLLECTION = dict(os.environ)
    return None


def pytest_collection_finish(session):
    if _ENV_BEFORE_COLLECTION is None:          # collection was replaced
        return
    offenders = harness.env_collection_offenders(
        _ENV_BEFORE_COLLECTION, dict(os.environ))
    if offenders:
        raise pytest.UsageError(harness.format_env_leak_report(offenders))


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
# `@pytest.mark.mesh(n)` — THE CELLS THE PIN ABOVE USED TO SILENCE  (2026-08-10)
# ---------------------------------------------------------------------------
# THE GAP, measured by the perf fleet: a cell that needs a >=4-device mesh
# CANNOT RUN UNDER THE SUITE.  The pin gives every test process exactly one
# GPU, `jax.device_count()` is therefore 1 in every worker, and the
# `skipif(jax.device_count() < 4)` those cells carried made them SKIP —
# silently, in every census, on every node, no matter how many GPUs the node
# had.  They passed only when someone invoked the file directly with
# `XLA_FLAGS=--xla_force_host_platform_device_count=4`.  So the suite could
# not exercise the four-GPU rule (AGENT_PREAMBLE) even in principle, and a
# per-shard defect is exactly the class of bug that rule exists to catch.
#
# WHY THE FIX IS A PROCESS AND NOT A FIXTURE.  `CUDA_VISIBLE_DEVICES` is read
# ONCE, when the backend is built, and collection imports jax long before any
# fixture runs.  There is no moment in this process that is both after the
# markers are known and before the device list is latched.  A per-cell device
# count is therefore a per-PROCESS fact, and the honest per-cell handle on it
# is a process: a `mesh(n)` cell that finds itself pinned below `n` runs in a
# CHILD that gets the session's full device list, and reports that child's
# verdict as its own.  Nothing widens in this process, so all four reasons the
# pin exists — xdist fan-out, the 1-GPU-frozen references, SLATE's refusal
# above one visible device, and the controller exclusion — are untouched.
#
# The subprocess is not a new idiom here: a dozen test modules in this tree
# already re-launch themselves to get a device count (test_contract_bands,
# test_zeta_mesh_invariance, test_downfold, ...).  What is new is that the
# MARKER, not each module, decides — so the suite can be asked which cells
# want a mesh, and it can be measured that they ran.
#
# ONE CHILD PER MODULE, not per cell: a GPU backend costs seconds to build
# and these modules carry ~40 cells.  Per-cell verdicts survive through
# junit-xml, keyed by nodeid.  The child is serialised node-wide with a lock
# and runs under the `platform` allocator, because under `lx test` four
# pinned workers are already computing on those same cards.
#
# WHAT THIS MARKER IS NOT.  `mesh(n)` is an n-device SINGLE-PROCESS mesh, and
# that is a strictly weaker thing than the production P=n geometry (n
# processes, one device each).  The difference is not academic: the perf fleet
# measured on 2026-08-10 that an in-process multi-device mesh CANNOT execute
# the flat-k cuFFT handler — every in-process 2x2 dies CUFFT_EXEC_FAILED, at
# every size, as an uncatchable SIGABRT — while the multi-process legs are
# fine.  So the child refuses that path outright (`LORRAX_FFT_FFI=0`, see
# harness.mesh_subprocess_env) rather than letting a stray call take the whole
# child down and every verdict in it with it.  A gate that needs the flat-k
# FFT at P=n is a MULTI-PROCESS gate: it belongs in tests/multi_device/, which
# is what that directory is for, and this marker must not be used to pretend
# otherwise.
#
# THE DECISION IS A PURE FUNCTION (`harness.mesh_plan`), same doctrine as the
# pin: a side effect in a conftest that nothing can construct a case for is
# unfalsifiable.  `tests/test_gpu_pinning.py` holds its cases.
# ---------------------------------------------------------------------------

#: Per-module verdicts brought back from a mesh child, keyed module -> {nodeid:
#: (outcome, detail)}.  Cached because every cell of a module shares one child.
_MESH_VERDICTS = pytest.StashKey[dict]()


def _mesh_want(item):
    """The device count this item's ``mesh`` marker asks for, or ``None``."""
    marker = item.get_closest_marker("mesh")
    if marker is None:
        return None
    return int(marker.args[0]) if marker.args else 4


def _devices_now():
    """``(count, platform)`` for THIS process.

    ``jax.device_count()`` and not ``len(CUDA_VISIBLE_DEVICES)``, because a
    direct invocation with ``--xla_force_host_platform_device_count=4`` has
    four devices and no GPUs at all, and that leg must keep working exactly
    as it does today.  The PLATFORM comes along because in the child the
    count alone cannot tell four A100s from four emulated host devices.
    """
    try:
        import jax

        devs = jax.devices()
        return len(devs), (getattr(devs[0], "platform", "") if devs else "")
    except Exception:                                          # noqa: BLE001
        return 0, ""


def _session_real_devices():
    """The session's REAL devices — or none, when the caller asked for a
    platform that is not them.

    A caller who exported ``JAX_PLATFORMS=cpu`` on a GPU node has said what
    they want, and the emulated arm is a legitimate thing to run; that is a
    different fact from a test module having set the same variable at import,
    which is the leak the snapshot exists to exclude.  The snapshot is the
    only place the two can still be told apart.
    """
    caller = _json.loads(os.environ.get(harness.SESSION_JAX_ENV, "{}") or "{}")
    plat = (caller.get("JAX_PLATFORMS") or "").lower()
    if plat and not any(p in plat for p in ("cuda", "gpu", "rocm")):
        return []
    return [d for d in os.environ.get(
        harness.SESSION_DEVICES_ENV, "").split(",") if d]


def _mesh_plan_for(item, want: int):
    have, platform = _devices_now()
    return harness.mesh_plan(
        want, have, _session_real_devices(),
        inner=bool(os.environ.get(harness.MESH_CELL_ENV)),
        platform=platform)


_MESH_GROUP = "lorrax-mesh"


def _plain_nodeid(item) -> str:
    """The item's nodeid as PYTEST spells it, not as xdist may have.

    Under ``--dist loadgroup`` xdist rewrites every grouped item's nodeid to
    ``<nodeid>@<group>`` so its scheduler can read the group back off it.
    That spelling is xdist's, and no pytest can resolve it — handing it to
    the child would collect nothing and turn a whole module red for a
    bookkeeping reason.  Anchored on OUR group name, so a parametrised id
    that happens to contain an ``@`` is untouched.
    """
    nodeid = item.nodeid
    suffix = "@" + _MESH_GROUP
    return nodeid[:-len(suffix)] if nodeid.endswith(suffix) else nodeid


def _mesh_verdict(item, want: int, devices):
    """This cell's outcome, running its module's mesh cells if needed."""
    module = _plain_nodeid(item).split("::", 1)[0]
    cache = item.config.stash.setdefault(_MESH_VERDICTS, {})
    if module not in cache:
        group = sorted({
            _plain_nodeid(it) for it in item.session.items
            if _plain_nodeid(it).split("::", 1)[0] == module
            and _mesh_want(it) is not None})
        res, xml = harness.run_mesh_group(
            group, devices, cwd=str(item.config.rootpath),
            out_dir=_Path(os.environ.get("TMPDIR", "/tmp"))
            / ("lorrax-mesh-%d" % os.getpid()))
        try:
            cache[module] = harness.junit_outcomes(xml, module)
        except Exception as exc:                               # noqa: BLE001
            # A child that died before writing junit takes EVERY verdict in it
            # with it, so the failure has to carry the child's own output and
            # name the one way this is known to happen: an in-process
            # multi-device mesh reaching the flat-k cuFFT handler, which is an
            # uncatchable SIGABRT (perf fleet, 2026-08-10).  The child already
            # refuses that path (harness.mesh_subprocess_env); a negative
            # return code here means something else killed it.
            cache[module] = {"__launch__": (
                "failed",
                f"the mesh child produced no readable junit ({exc}).\n"
                f"rc={res.returncode}"
                + ("  (killed by signal — an in-process multi-device mesh that "
                   "reaches the flat-k FFT aborts uncatchably; such a gate is "
                   "MULTI-PROCESS and belongs in tests/multi_device/)"
                   if res.returncode < 0 else "")
                + f"\n--- stdout ---\n{res.stdout[-4000:]}\n"
                f"--- stderr ---\n{res.stderr[-4000:]}")}
    verdicts = cache[module]
    if "__launch__" in verdicts:
        return verdicts["__launch__"]
    return verdicts.get(
        _plain_nodeid(item),
        ("failed", "the mesh child reported no verdict for this cell — it "
                   "was not collected there.  A cell that cannot be named to "
                   "the child is a cell the suite is not running."))


def pytest_runtest_setup(item):
    """Skip (or refuse) a ``mesh`` cell before its fixtures are built."""
    want = _mesh_want(item)
    if want is None:
        return
    verb, payload = _mesh_plan_for(item, want)
    if verb == "skip":
        pytest.skip(payload)
    if verb == "fail":
        pytest.fail(payload, pytrace=False)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Run a pinned-process ``mesh`` cell in the child that has the devices.

    ``pytest_pyfunc_call`` is pytest's own "replace the call" hook, so the
    cell keeps its normal setup/teardown and its normal report line; only
    the body runs elsewhere.  A ``mesh`` cell is therefore expected to be
    self-contained — it must not lean on a session fixture, since the
    fixture would be built in THIS process and consumed in the other one.
    """
    want = _mesh_want(pyfuncitem)
    if want is None:
        return None
    verb, payload = _mesh_plan_for(pyfuncitem, want)
    if verb != "subprocess":
        return None                       # already on a mesh: run right here
    outcome, detail = _mesh_verdict(pyfuncitem, want, payload)
    pyfuncitem.user_properties.append(("mesh_devices", ",".join(payload)))
    if outcome == "skipped":
        pytest.skip(f"[mesh({want}) child] {detail}")
    if outcome != "passed":
        pytest.fail(
            f"[mesh({want}) child, CUDA_VISIBLE_DEVICES={','.join(payload)}] "
            f"{detail}", pytrace=False)
    return True


def _group_mesh_items(config, items):
    """Put every ``mesh`` cell in one xdist group, so one worker owns them.

    Without this the cells scatter across workers and each worker starts
    its own child for the same module.  ``pytest_configure`` above asks for
    ``loadgroup`` when xdist is active, which is where the group takes
    effect; applied only when xdist is installed, because ``xdist_group``
    is xdist's marker and stamping it without xdist is just a warning on
    every mesh cell of every laptop run.
    """
    if not config.pluginmanager.hasplugin("xdist"):
        return
    for item in items:
        if item.get_closest_marker("mesh") is not None:
            item.add_marker(pytest.mark.xdist_group(_MESH_GROUP))


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
    gate = parser.getgroup("gate", "LORRAX default gate vs census")
    gate.addoption(
        "--census", action="store_true", default=False,
        help="run the FULL census — exactly what a bare `pytest` collected "
             "before 2026-08-09, and what tests/KNOWN_FAILURES.md accounts "
             "for.  `-m census` is the same thing.")


# ---------------------------------------------------------------------------
# THE DEFAULT GATE  (the owner, 2026-08-09)
# ---------------------------------------------------------------------------
#     "really we should run that Si test calculation (granted for all
#      drivers that were touched since last ran) and the tests for the
#      services and have that basically be it."
#
# TWO TIERS, and the split is a SELECTION, not a deletion:
#
#   pytest                 -> the Si end-to-end test calculation for the
#                             drivers this branch touched, plus every
#                             service's own suite.  Minutes.
#   pytest --census        -> byte-for-byte the run `pytest` used to be.
#   pytest -m census       -> the same thing (see `census` below).
#
# WHY THE DESELECTION IS A HOOK AND NOT A SECOND `-m`.  Same reason the
# service deselection above is: pyproject sets `addopts = "-m 'not extra'"`,
# and an explicit `-m` on the command line REPLACES that default rather than
# composing with it.  Spelling the default tier as `-m "not census"` would
# therefore silently re-enable the whole `extra` tier while looking like it
# had narrowed the run.  A hook acts on the items `-m` already selected, so
# it composes by construction.
#
# WHY EVERY CELL CARRIES `census` RATHER THAN ONLY THE ZOO.  So that
# `pytest -m census` means what the owner said it means — EVERYTHING —
# instead of "everything except the Si gates", which is the one shape of
# census that could not account for KNOWN_FAILURES.md.  The marker is
# applied to every collected item EXCEPT those already carrying `extra`,
# which keeps `-m census` collecting exactly the set `-m 'not extra'` did.
#
# WHEN THE GATE STANDS DOWN.  The default tier applies to a BARE `pytest`
# and to nothing else.  Naming paths, `-m`, `-k`, `--census` or
# `LX_CENSUS=1` all mean the caller has already said what they want, and a
# second opinion from this hook would only be a way to lose tests they
# asked for.
def _stamp_census(items):
    """Stamp `census` on every collected cell that is not already `extra`.

    ``extra`` is skipped so that ``-m census`` selects exactly the set
    ``addopts = "-m 'not extra'"`` selects — today's full run, neither 26
    cells short nor 26 cells long.

    THIS RUNS FROM ``pytest_collection_modifyitems``, and it has to.  The
    per-item hook (``pytest_itemcollected``) is dispatched through the
    COLLECTOR's hook proxy, so a conftest under ``tests/`` never sees an
    item under ``services/`` — MEASURED: stamping there marked 2400 of the
    3356 cells and ``-m census`` silently dropped all 930 service cells.
    ``pytest_collection_modifyitems`` is a session-level hook and sees every
    item, which is also why the service deselection below lives there.

    Ordering against pytest's own `-m` filter, which is the other
    implementation of this same hook: ours is ``tryfirst``, so the marker
    exists before ``deselect_by_mark`` reads it.  Without that, ``-m
    census`` would collect NOTHING — a green run of zero tests.
    """
    for item in items:
        if item.get_closest_marker("extra") is None:
            item.add_marker(pytest.mark.census)


def _cmdline_select_flags(args) -> list:
    """``-m`` / ``-k`` SPELLINGS ON THE RAW COMMAND LINE, as (flag, value).

    Raw args and not ``config.option``, because addopts has already been
    folded into the latter: `-m 'not extra'` is there on every run and is
    indistinguishable from a `-m` the caller typed.

    Spelled out rather than done with ``startswith`` on a tuple, because
    ``--maxfail=1`` starts with ``-m`` and a naive prefix test hands the
    whole default gate away to it.  The short forms are the ones that can
    be glued to their value (``-mfoo``, ``-kfoo``); the long forms cannot,
    so ``--`` is disqualifying for them.
    """
    out, i, n = [], 0, len(args)
    while i < n:
        a = args[i]
        nxt = args[i + 1] if i + 1 < n else ""
        if a in ("-m", "--markexpr"):
            out.append(("-m", nxt.strip()))
            i += 2
            continue
        if a == "-k":
            out.append(("-k", nxt.strip()))
            i += 2
            continue
        if a.startswith("--markexpr="):
            out.append(("-m", a.split("=", 1)[1].strip()))
        elif not a.startswith("--") and a.startswith("-m") and len(a) > 2:
            out.append(("-m", a[2:].strip()))
        elif not a.startswith("--") and a.startswith("-k") and len(a) > 2:
            out.append(("-k", a[2:].strip()))
        i += 1
    return out


def _explicit_selection(config) -> str:
    """Why this hook should stand down, or "" to run the default gate."""
    if config.getoption("--census"):
        return "--census"
    if (os.environ.get("LX_CENSUS", "") or "").strip() not in (
            "", "0", "false", "no"):
        return "LX_CENSUS"
    flags = _cmdline_select_flags(
        list(getattr(config.invocation_params, "args", ()) or ()))
    for flag, value in flags:
        # `-m census` is the owner's spelling of the full run.  It needs no
        # special handling to WORK — every non-`extra` cell carries the
        # marker, so pytest's own filter already selects the whole census —
        # it is named here only so the header says CENSUS rather than
        # pretending the fast gate is what stood down.
        if flag == "-m" and value == "census":
            return "-m census"
        return flag
    if getattr(config.option, "file_or_dir", None):
        return "explicit path(s)"
    # The service flags are census instruments: `--no-services` is how the
    # census's no-srun leg runs lorrax's own tests, and `--only-service`
    # names a suite outright.  Both already say what they want.  ALL THREE
    # SPELLINGS, not two: the conftest above documents LX_SKIP_SERVICES and
    # --no-services as one knob, and a knob that means "census minus
    # services" one way and "five cells" the other is the kind of split
    # that only shows up as a suspiciously small green number.
    if config.getoption("--no-services"):
        return "--no-services"
    if (os.environ.get("LX_SKIP_SERVICES", "") or "").strip() not in (
            "", "0", "false", "no"):
        return "LX_SKIP_SERVICES"
    if (config.getoption("--only-service") or "").strip():
        return "--only-service"
    return ""


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


_GATE_NOTE = pytest.StashKey[list]()


def _apply_default_gate(config, items):
    """Narrow a bare ``pytest`` to the default tier.  No-op otherwise."""
    stood_down = _explicit_selection(config)
    drivers, why = fast_gate.selected_drivers(fast_gate.REPO_ROOT)
    roster = fast_gate.smoke_node_ids(drivers)
    undecked = sorted(d for d in drivers if d in fast_gate.UNDECKED)
    note = config.stash.setdefault(_GATE_NOTE, [])
    if stood_down:
        note.append(f"gate: CENSUS ({stood_down}) — full suite, "
                    "KNOWN_FAILURES.md accounting applies")
        return
    note.append(
        "gate: DEFAULT (fast) — Si e2e smoke for "
        + (", ".join(sorted(drivers)) if drivers else "NO drivers")
        + f" [{why}] + every services/ suite.  Full set: pytest --census")
    for d in undecked:
        note.append(f"gate: {d} was touched but has NO runnable in-tree "
                    f"deck — {fast_gate.UNDECKED[d]}")

    keep, drop, hit = [], [], set()
    for item in items:
        nodeid = item.nodeid
        if fast_gate.is_service_item(nodeid):
            keep.append(item)
        elif fast_gate.matches(nodeid, roster):
            keep.append(item)
            hit.add(nodeid.split("[", 1)[0])
        else:
            drop.append(item)

    # A ROSTER ENTRY THAT MATCHES NOTHING IS A SILENTLY EMPTY GATE, which is
    # the one failure mode a fast gate cannot be allowed to have: a renamed
    # cell would leave the driver un-smoked and the run green.  `lx` already
    # refuses pytest's rc=5; this refuses the case rc=5 cannot see, where the
    # SERVICE half still collects and the run looks populated.
    missing = sorted(roster - hit)
    if missing:
        raise pytest.UsageError(
            "tests/fast_gate.py names cells that do not exist:\n    "
            + "\n    ".join(missing)
            + "\n\nThe default gate would run without them and report green. "
              "Fix the roster (or the rename) — do not delete the entry.")
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
        note.append(f"gate: deselected {len(drop)} census cell(s); "
                    f"{len(keep)} selected")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    _stamp_census(items)
    ledger_path = _Path(__file__).resolve().parent / "KNOWN_FAILURES.md"
    try:
        known_xfails = known_failure_ledger.read_known_xfails(ledger_path)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    by_nodeid = {item.nodeid: item for item in items}
    if known_failure_ledger.is_complete_tests_selection(
            getattr(config.option, "file_or_dir", ()), ledger_path.parent):
        missing = known_failure_ledger.missing_nodeids(
            known_xfails, by_nodeid)
        if missing:
            raise pytest.UsageError(
                "tests/KNOWN_FAILURES.md names tests that no longer exist:\n    "
                + "\n    ".join(missing)
                + "\nRemove or update the stale ledger rows before trusting "
                  "this suite verdict.")
    for nodeid, row in known_xfails.items():
        item = by_nodeid.get(nodeid)
        if item is not None:
            item.add_marker(pytest.mark.xfail(
                reason=f"{row['reason']} [owner: {row['owner']}]",
                strict=True))
    _group_mesh_items(config, items)
    _apply_default_gate(config, items)

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


def pytest_report_collectionfinish(config, items):
    """Which tier this run is, and what it narrowed to — AFTER collection.

    Not ``pytest_report_header``: that runs before collection, so it cannot
    say how many cells the gate set aside.  Same doctrine as the service
    deselection above — a run that quietly dropped a tier and reported a
    smaller green number is the exact shape of loss this tree keeps finding.
    """
    return list(config.stash.get(_GATE_NOTE, []))


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
def gnppm_sc_session(gnppm_session, tmp_path_factory):
    """THE self-consistent run.  There is exactly one, and this is it.

    Every other committed deck is ``qp_solver = one_shot_dft``, which is
    how ``sc_on_ibz`` rotted into a crash without turning anything red.
    See ``tests/regression/gnppm_debug/gnppm_sc.in`` for why this fixture
    and not a smaller one — the short version is that the deck must have
    a file wedge and a star wedge of DIFFERENT sizes (9 vs 5 here) and an
    orbit-closed centroid set, and gnppm_debug is the only one with both.

    Restarts from the ``gnppm_session`` state, so it costs one Σ chain
    rather than an ISDF rebuild: ~25 s at P=4 against ~31 s fresh.
    """
    run_dir = harness.copy_fixture(
        harness.REG / "gnppm_debug",
        tmp_path_factory.mktemp("gnppm_sc") / "gnppm_debug",
        tmp_from=gnppm_session.run_dir)
    res = harness.run_gw_jax(run_dir, "gnppm_sc.in")
    if res.returncode != 0:
        pytest.fail(
            f"gnppm SC session run failed.\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
    return _NS(run_dir=run_dir, input_name="gnppm_sc.in",
               output_name="sigma_diag_gnppm_sc.dat", stdout=res.stdout,
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


# ---------------------------------------------------------------------------
# hBN 3×3×2 — the NON-CUBIC COHSEX fixture.  TWO fresh runs, not one.
#
# ``hbn_session`` is the frozen-reference run.  ``hbn_mcavg_false_session`` is
# the NEGATIVE CONTROL: the same deck with the single key
# ``mc_average_vcoul_body`` flipped to false, which must move Σ.
#
# THE SECOND RUN CANNOT BE A RESTART VARIANT, and that is the whole reason it
# costs a second fresh run instead of riding the Tier-2 pattern.  Every Tier-2
# invariance variant re-runs with ``restart = true`` from a copy of the session
# state — which reloads the ALREADY-BUILT V_q, the exact tensor the knob acts
# on.  A from-restart arm would therefore compare the head table against
# itself and measure 0.000 meV: a control that cannot fail, which is the one
# kind of test this tree deletes on sight (charter, falsification doctrine).
#
# Cost: ~33 s each on one A100, measured at the freeze.  That is the price of
# the fixture having a live control at all, and it is the same order as the Si
# production deck's 34.6 s.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def hbn_session(tmp_path_factory):
    """Fresh run of the hBN deck — the frozen-reference run."""
    return _run_session_case(
        tmp_path_factory, "hbn_cohsex_debug", "cohsex_hbn_test.in",
        "eqp_hbn_test.dat")


@pytest.fixture(scope="session")
def hbn_mcavg_false_session(tmp_path_factory):
    """The hBN deck with ``mc_average_vcoul_body = false`` — the control arm.

    Fresh (``restart = false``, as the deck already is), so the mini-BZ head
    average is genuinely rebuilt under the flipped convention.
    """
    import pytest as _pytest
    harness.skip_unless_gpu(_pytest)
    run_dir = harness.copy_fixture(
        harness.REG / "hbn_cohsex_debug",
        tmp_path_factory.mktemp("hbn_mcavg_false") / "hbn_cohsex_debug")
    # ``mutate_input`` asserts the old string is present, so a deck edit that
    # renamed or dropped this key fails here loudly instead of silently
    # producing a second copy of the reference run.
    harness.mutate_input(run_dir / "cohsex_hbn_test.in", {
        "mc_average_vcoul_body = true": "mc_average_vcoul_body = false",
    })
    res = harness.run_gw_jax(run_dir, "cohsex_hbn_test.in")
    if res.returncode != 0:
        pytest.fail(
            f"hBN mc_average_vcoul_body=false arm failed.\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
    out = run_dir / "eqp_hbn_test.dat"
    assert out.exists(), f"control arm wrote no {out}"
    return _NS(run_dir=run_dir, input_name="cohsex_hbn_test.in",
               output_name="eqp_hbn_test.dat", stdout=res.stdout)
