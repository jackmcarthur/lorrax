"""Shared helpers for the e2e regression/invariance gates.

One home for the subprocess runner, output parsers, and fixture-dir
copying that the Tier-1 frozen gates (``test_gw_jax_regression``), the
Tier-2 invariance gates (``test_invariance_gates``), and the session
fixtures in ``conftest.py`` all use.  Not a test module.

Suite architecture (2026-07-09 redesign):

* **Tier 1** — frozen e2e pins, one fresh ``gw.gw_jax`` subprocess per
  fixture (si_cohsex_3d / cohsex / gnppm / bispinor GN-PPM).
* **Tier 2** — self-checking invariances (restart≡fresh, μ-pad flip,
  SC-iter1≡one-shot, fixed-point rotations, IBZ≡full-BZ)
  run as cheap ``restart = true`` variants from a COPY of the Tier-1
  gnppm session state (the ISDF ζ-fit + V_q are not redone).  Each
  variant copies the session ``tmp/`` because the driver WRITES W0 +
  head scalars back into the restart file (``persist_w0_and_head``).
* **Tier 3** — unit tests for what the gates cannot see.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
REG = REPO_ROOT / "tests" / "regression"


def _visible_gpus(preset: str | None, probe) -> list[str]:
    """The device ids this process may use, in the order it may use them."""
    if preset is not None and preset.strip() != "":
        return [d for d in preset.split(",") if d != ""]
    if preset is not None:                       # explicitly masked: ""
        return []
    return [str(i) for i in range(probe())]


def _probe_nvidia_smi() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=10).stdout
        return len(out.strip().splitlines()) if out.strip() else 0
    except Exception:                                          # noqa: BLE001
        return 0


def is_xdist_controller(worker_id: str, dist: str | None) -> bool:
    """True when this process fans tests out and runs NONE of them itself.

    pytest-xdist's OWN predicate, in the two facts a conftest can read:
    a worker has ``PYTEST_XDIST_WORKER`` in its environment, and the
    controller is the process that has no worker id while ``--dist`` is
    something other than ``no`` (``xdist.is_xdist_controller`` is
    ``config.option.dist != "no" and not hasattr(config, "workerinput")``).

    NOT a heuristic on ``sys.argv`` and not "no worker id": a plain
    non-xdist run has no worker id either, and it is going to compute in
    this very process, so it must still be pinned.  ``dist`` is
    ``config.option.dist``, which is ``"no"`` when xdist is absent or was
    given ``-n 0``.
    """
    return not worker_id and str(dist or "no") != "no"


#: Environment variables a test module is ALLOWED to set at import time.
#:
#: The distinction is not "ours vs theirs", it is WHEN THE VALUE IS READ.
#: jax/XLA/CUDA latch these at ``import jax`` — the first import in the
#: process wins, and pytest imports every collected module into ONE process
#: — so a module that needs x64, a CPU platform, or four emulated devices
#: has no choice but to set them before its own import block.  A fixture
#: runs far too late to matter.
#:
#: Everything else — LORRAX_* dials above all — is read at CALL time
#: (``ffi.gate.Gate.enabled`` does a live ``os.environ.get`` per call), so a
#: fixture serves it exactly, and module scope does not: module scope runs
#: at COLLECTION time and never unwinds, which silently reconfigures every
#: other test in the session and makes their verdicts depend on collection
#: order.  That is P19, measured: COMPLETENESS_AUDIT.md, "a gate whose
#: verdict depends on collection scope".
#:
#: An entry here is a CLAIM that the variable cannot be fixtured, and
#: somebody has to defend it — the same convention as
#: ``test_bse_coupling_routes_mesh_invariance._WAIVED_ENCODES``.  The
#: prefixes below were measured, not guessed: they are what a full
#: collection of ``tests/`` actually moves (2026-08-09).
IMPORT_TIME_ENV_PREFIXES = (
    "JAX_",           # JAX_ENABLE_X64, JAX_PLATFORMS — latched at import
    "XLA_",           # XLA_FLAGS — parsed once when the backend is built
    "LIBTPU_",
    "TPU_",           # TPU_SKIP_MDS_QUERY — set by jax itself on import
    "CUDA_",          # CUDA_VISIBLE_DEVICES — the conftest GPU pin
    "NVIDIA_",
    "TF_",            # TF_CPP_MIN_LOG_LEVEL, set by absl/xla on import
    "PYTEST_",        # pytest's own bookkeeping
)

#: Exact names allowed for a DIFFERENT reason than the prefixes above: not
#: "read at import", but "guards something that cannot be undone".
IRREVERSIBLE_ENV_SENTINELS = frozenset({
    # runtime.__init__'s distributed guard, set by test_head_wing_schur's
    # module-scope _init_distributed().  jax.distributed.initialize can be
    # called ONCE per process and has no teardown, so the sentinel that
    # makes a second call a no-op must outlive any fixture: unwinding it
    # would re-arm the very hazard it exists to prevent.  It is an env var
    # rather than a module global on purpose (it has to survive the
    # `python -m gw.gw_jax` re-import path) — see src/runtime/__init__.py.
    "_LORRAX_JAX_DISTRIBUTED_DONE",
})


def env_collection_offenders(before: dict, after: dict) -> list:
    """Environment changes made during collection that a FIXTURE should own.

    Pure function (so it is falsifiable without a pytest session — the same
    reason ``pin_one_gpu`` lives here rather than inline in the conftest).
    Returns a sorted list of ``(name, before_value, after_value)`` for every
    variable that changed across collection and is not import-time-latched;
    ``None`` marks "was not set".  An empty list means collection was inert.
    """
    out = []
    for name in sorted(set(before) | set(after)):
        if (name.startswith(IMPORT_TIME_ENV_PREFIXES)
                or name in IRREVERSIBLE_ENV_SENTINELS):
            continue
        was, now = before.get(name), after.get(name)
        if was != now:
            out.append((name, was, now))
    return out


def format_env_leak_report(offenders: list) -> str:
    """The refusal text for :func:`env_collection_offenders`, with the fix."""
    lines = [
        "TEST MODULE(S) MUTATED os.environ AT COLLECTION TIME.",
        "",
        "Collection imports every selected test module into ONE process "
        "before ANY test runs, and a module-scope write never unwinds. So "
        "each variable below is now set for the WHOLE session, and every "
        "other test's verdict silently depends on whether the module that "
        "set it was collected. That is not a style complaint: it is the "
        "measured cause of P19 (COMPLETENESS_AUDIT.md), where a K^d_B "
        "class gate read 15 passed in the census and 13 failed standalone "
        "on the same tree and the same commit.",
        "",
        "Changed across collection:",
    ]
    for name, was, now in offenders:
        lines.append(f"    {name}: {was!r} -> {now!r}")
    lines += [
        "",
        "FIX: move the write into a fixture that unwinds --",
        "",
        "    @pytest.fixture(autouse=True)",
        "    def _dial(monkeypatch):",
        "        monkeypatch.setenv(NAME, VALUE)",
        "",
        "`autouse` keeps test signatures unchanged, and monkeypatch "
        "restores the previous value after each test, so the pin stops at "
        "this file's boundary. See tests/test_contract_bands.py::"
        "_xla_plan_dial for the worked example.",
        "",
        "If the variable is genuinely latched at IMPORT time (jax/XLA read "
        "it when the backend is built, so no fixture can be early enough), "
        "add its prefix to harness.IMPORT_TIME_ENV_PREFIXES with the reason "
        "-- that list is a claim someone has to defend, not a mute button.",
    ]
    return "\n".join(lines)


def pin_one_gpu(preset: str | None, worker_id: str = "", probe=None,
                *, controller: bool = False):
    """The ONE device this process should see, or ``None`` for "leave it".

    ``preset`` is ``CUDA_VISIBLE_DEVICES`` as the process found it
    (``None`` = unset), ``worker_id`` is ``PYTEST_XDIST_WORKER``.  With a
    worker id the pick fans out across the visible list (``gw2`` -> the
    third one, wrapping); without one it is the FIRST visible device.

    ``controller`` is the xdist CONTROLLER, and it gets ``None`` — never a
    device.  IT RUNS NO TESTS, so it has nothing to pin FOR, and what it
    writes into its own environ is what its workers INHERIT as their
    preset.  MEASURED, Perlmutter 2026-08-07 (KNOWN_FAILURES B2): with the
    controller pinning too, all four workers read ``preset="0"``, every
    ``_visible_gpus`` list was one element long, ``int(wid[2:]) % 1 == 0``
    for all of them, and the fan-out below silently became four sessions
    on one 40 GiB A100 --

        gw0 gw1 gw2 gw3 = '0','1','2','3'   before   24 / 24 passed
        gw0 gw1 gw2 gw3 = '0','0','0','0'   after     7 P / 17 E,
            RESOURCE_EXHAUSTED: Failed to allocate 19.9 GiB on device
            ordinal 0

    -- with this function's own unit tests still green throughout, because
    they construct the preset by hand and never see the controller's write.

    A PURE FUNCTION on purpose.  Its caller is a side effect in
    ``tests/conftest.py`` — it has to run before the first CUDA init,
    which is the one place a test cannot observe — so the DECISION lives
    here where ``tests/test_gpu_pinning.py`` can construct every case,
    including the two that regressed.
    """
    if controller:
        return None
    devs = _visible_gpus(preset, probe or _probe_nvidia_smi)
    if not devs:
        return None
    # ``worker_id[2:].isdigit()`` rather than ``startswith("gw")``: the
    # xdist CONTROLLER sets no worker id at all and other spellings exist
    # ("master"), and a bare ``int(worker_id[2:])`` on one of them raises
    # ValueError out of a conftest at module scope — which pytest reports
    # as a collection error for the entire suite, not as one bad pin.
    tail = worker_id[2:] if worker_id.startswith("gw") else ""
    i = int(tail) % len(devs) if tail.isdigit() else 0
    return devs[i]


def session_devices(preset: str | None, probe=None) -> list:
    """Every device the SESSION may use, read BEFORE ``pin_one_gpu`` narrows it.

    The pin above is the reason a mesh cell cannot simply look at
    ``jax.devices()`` and find four: by the time any test module imports
    jax, this process has already been narrowed to one.  So the list is
    read once, in ``pytest_configure``, and carried in the environment
    (``LORRAX_SESSION_DEVICES``) for the one consumer that needs it — the
    mesh runner below, which spends it on a subprocess.
    """
    return _visible_gpus(preset, probe or _probe_nvidia_smi)


#: The env var that carries :func:`session_devices` past the pin, and the
#: sentinel that marks a process the mesh runner started.  ``LORRAX_`` and
#: not ``CUDA_``: it is read at CALL time, never latched by a backend.
SESSION_DEVICES_ENV = "LORRAX_SESSION_DEVICES"
MESH_CELL_ENV = "LORRAX_MESH_CELL"

#: The CALLER's device-shape environment, snapshotted in ``pytest_configure``
#: and handed to the mesh child verbatim.  MEASURED, Perlmutter 2026-08-10,
#: and it is the difference between a mesh cell and a lie about one.
#:
#: These two are latched at ``import jax``, so a test module that needs a CPU
#: platform or four emulated devices has to set them at MODULE SCOPE — and a
#: module-scope write never unwinds (that is the P19 hazard the collection
#: guard above documents; ``JAX_``/``XLA_`` are on the allowlist precisely
#: because there is nowhere else to put them).  ``tests/test_contract_bands``
#: and ``tests/test_sanity_gates_jax`` both do it.  So by the time ANY test
#: runs in a census, the process environment says ``JAX_PLATFORMS=cpu`` and
#: ``XLA_FLAGS=--xla_force_host_platform_device_count=4``, whatever the
#: caller asked for — and a child started from that environment comes up on
#: FOUR EMULATED HOST DEVICES and passes, on a node with four A100s, while
#: reporting itself as a mesh run.  That is the exact substitution the
#: four-GPU rule exists to forbid, and it is invisible in the verdict.
#:
#: ``pytest_configure`` runs BEFORE collection, so the value read there is
#: the caller's own and no module has spoken yet.  A caller who exported
#: ``JAX_PLATFORMS=cpu`` keeps it; a module that set it during collection
#: does not follow the child.  ``None`` means "the caller had none", and the
#: child gets it UNSET rather than inheriting the leak.
SESSION_JAX_ENV = "LORRAX_SESSION_JAX_ENV"
JAX_SHAPE_VARS = ("JAX_PLATFORMS", "XLA_FLAGS")

#: Platforms that count as "the real devices the session promised".
_REAL_PLATFORMS = frozenset({"gpu", "cuda", "rocm"})


def mesh_plan(want: int, have_here: int, session_devs, *, inner: bool = False,
              platform: str = ""):
    """Where a ``@pytest.mark.mesh(want)`` cell has to run.  PURE.

    Returns ``(verb, payload)`` with ``verb`` one of:

    ``"here"``
        This process already sees ``want`` devices — a direct invocation
        with ``--xla_force_host_platform_device_count``, or the mesh
        subprocess itself.  Run the cell normally; payload is ``None``.
    ``"subprocess"``
        This process is PINNED below ``want`` but the session's node has
        enough.  Payload is the device list to hand a subprocess.  This is
        the case the suite hits under ``lx test``: four GPUs on the node,
        one per xdist worker, so no worker can build a 2x2 by itself.
    ``"skip"``
        The hardware is not there at all (a laptop, a one-GPU leg).
        Payload is the reason string.
    ``"fail"``
        We ARE the mesh subprocess and we still do not have the devices we
        were started for.  Payload is the reason.  A skip here would be a
        lie — the run that was supposed to supply the mesh did not, and a
        green-with-a-skip is exactly how this cell went unexercised for a
        month.  It also bounds the recursion: the subprocess can never
        start another one.

    WHY A SUBPROCESS AND NOT A FIXTURE THAT HANDS BACK THE DEVICES.
    ``CUDA_VISIBLE_DEVICES`` is read ONCE, when the backend is built, and
    the first test module to be collected imports jax.  A fixture runs
    hundreds of items later.  There is no in-process moment between "the
    markers are known" (collection) and "the device list is still
    changeable" (before the first jax import), so a per-cell device set is
    a per-PROCESS fact and the only honest per-cell handle on it is a
    process.  Every other consumer of the pin — the xdist fan-out, the
    1-GPU-frozen references, SLATE's refusal at >1 visible device — keeps
    exactly the pin it has today, because this process never widens.
    """
    devs = list(session_devs or [])
    here = have_here
    # A PROCESS THAT IS NOT ON THE SESSION'S DEVICES HAS NONE OF THEM, however
    # many it can count.  MEASURED, Perlmutter 2026-08-10: several test modules
    # set ``JAX_PLATFORMS=cpu`` and four emulated host devices at MODULE SCOPE
    # (they must — the values latch at ``import jax``) and a module-scope write
    # never unwinds, so after collection EVERY process in a census carries
    # them.  ``jax.device_count()`` then answers 4, and a mesh cell on a node
    # with four A100s runs on four emulated CPU devices and passes.  Four is
    # four: only the platform separates honouring the four-GPU rule from
    # quietly substituting for it.  ``session_devs`` is empty exactly when
    # there is nothing to substitute FOR — no GPUs, or a caller who asked for
    # a host platform on purpose — and then the count is the whole story and
    # emulated legs keep working unchanged.
    if devs and platform not in _REAL_PLATFORMS:
        here = 0
    if here >= want:
        return ("here", None)
    if inner:
        return ("fail",
                f"the mesh child was started for {len(devs)} device(s) and "
                f"jax came up with {have_here} on platform "
                f"'{platform or 'none'}'.  A mesh({want}) cell is not allowed "
                "to be quietly emulated and this is not a skip — check "
                "JAX_PLATFORMS / XLA_FLAGS in the child's environment.")
    if len(devs) >= want:
        return ("subprocess", devs[:want])
    return ("skip",
            f"needs >= {want} devices: this process has {have_here} on "
            f"'{platform or 'none'}' and the session has {len(devs)} visible "
            f"GPU(s).  On a >= {want}-GPU node the suite runs this cell on "
            "the real mesh; off one, run it directly with "
            f"XLA_FLAGS=--xla_force_host_platform_device_count={want}.")


def mesh_subprocess_env(base_env: dict, devices, caller_jax_env=None) -> dict:
    """The environment the mesh subprocess runs under.  PURE.

    Four edits and each one is load-bearing:

    * ``CUDA_VISIBLE_DEVICES`` back to the session's full list — the whole
      point.  The parent's pin stays where it is; only the child widens.
    * ``JAX_PLATFORMS`` / ``XLA_FLAGS`` reset to what the CALLER had, which
      ``pytest_configure`` snapshotted before collection.  See
      :data:`SESSION_JAX_ENV`: without this the child inherits the
      collection-time leak, comes up on four EMULATED host devices, and
      passes while measuring nothing about the mesh it was started for.
    * the xdist worker bookkeeping dropped, so the child is a plain
      single-process run and not a worker that reports to a controller
      that is not listening.
    * ``XLA_PYTHON_CLIENT_ALLOCATOR=platform``.  The child is a CO-TENANT:
      under ``lx test`` four pinned workers are already computing on these
      same cards, and BFC preallocates a fraction of every visible device
      at backend init, so a fifth process that preallocates would OOM the
      node rather than measure anything.  ``platform`` allocates on demand,
      which is right for cells whose arrays are kilobytes — and it is what
      the lx container already exports, so this states the default rather
      than departing from it.  A mesh cell is a CORRECTNESS gate, never a
      timing.
    """
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    env[MESH_CELL_ENV] = "1"
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    # THE FLAT-K FFT FFI IS REFUSED IN THE CHILD, not left to abort.
    # MEASURED by the perf fleet, 2026-08-10: an IN-PROCESS multi-device mesh
    # cannot execute the flat-k cuFFT handler — every in-process 2x2 dies
    # CUFFT_EXEC_FAILED at every size, and it is an uncatchable SIGABRT, while
    # the production multi-PROCESS legs (one process per device) are fine.  A
    # child is exactly an in-process multi-device mesh, so a cell that wandered
    # into `make_flat_k_*` would take the whole child down and every verdict in
    # it with it, attributable to nothing.  `LORRAX_FFT_FFI=0` REFUSES (the XLA
    # flat-k twin was deleted by the 2026-08-01 ruling, so there is nothing to
    # fall back to) — and a refusal names the cell that caused it, which an
    # abort cannot.  A gate that genuinely needs the flat-k FFT at P=n is a
    # MULTI-PROCESS gate and belongs in tests/multi_device/, not behind this
    # marker.
    env["LORRAX_FFT_FFI"] = "0"
    for name in JAX_SHAPE_VARS:
        value = (caller_jax_env or {}).get(name)
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    for k in ("PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT",
              "PYTEST_XDIST_TESTRUNUID", "PYTEST_CURRENT_TEST"):
        env.pop(k, None)
    return env


def junit_outcomes(xml_path, module_relpath: str) -> dict:
    """``{nodeid: (outcome, detail)}`` from a junit-xml the mesh child wrote.

    junit is pytest's own report format, so the child needs no plugin and
    no protocol of ours.  ``xunit2`` drops the ``file`` attribute, so the
    nodeid is rebuilt from the module path we asked for plus whatever
    ``classname`` carries beyond it (a class, when there is one).
    """
    import xml.etree.ElementTree as ET

    mod = module_relpath[:-3].replace(os.sep, ".").replace("/", ".")
    out = {}
    root = ET.parse(str(xml_path)).getroot()
    for case in root.iter("testcase"):
        name = case.get("name", "")
        classname = case.get("classname", "") or mod
        extra = ([p for p in classname[len(mod) + 1:].split(".") if p]
                 if classname.startswith(mod) else [])
        nodeid = "::".join([module_relpath] + extra + [name])
        outcome, detail = "passed", ""
        for child in case:
            tag = child.tag
            if tag in ("failure", "error"):
                outcome = "failed"
                detail = (child.get("message") or "") + "\n" + (child.text or "")
                break
            if tag == "skipped":
                outcome = "skipped"
                detail = child.get("message") or ""
                break
        out[nodeid] = (outcome, detail.strip())
    return out


def run_mesh_group(nodeids, devices, *, cwd, out_dir, timeout=1800):
    """Run ``nodeids`` in ONE subprocess that owns ``devices``.

    ONE process per module rather than one per cell: a GPU backend costs
    seconds to build and these modules carry ~40 cells between them, so
    per-cell processes would put ten minutes of pure jax init on the
    critical path of every census.  Per-cell VERDICTS survive anyway —
    they come back through junit-xml, keyed by nodeid.

    Serialised node-wide with an exclusive lock: under ``lx test`` several
    xdist workers can reach their mesh cells at once, and two children
    each taking every GPU on the node is the co-tenancy failure the pin
    exists to prevent, reintroduced by the fix for it.
    """
    import fcntl
    import json
    import time

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml = out_dir / ("mesh_%d.xml" % os.getpid())
    node_tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    lock_path = node_tmp / ("lorrax-mesh-%s.lock" % os.environ.get("USER", "x"))
    cmd = [sys.executable, "-u", "-m", "pytest", "-q", "--no-header",
           "-p", "no:cacheprovider", f"--junitxml={xml}", *nodeids]
    env = mesh_subprocess_env(
        os.environ, devices,
        json.loads(os.environ.get(SESSION_JAX_ENV, "{}") or "{}"))
    t0 = time.time()
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            res = subprocess.run(cmd, cwd=str(cwd), env=env, timeout=timeout,
                                 capture_output=True, text=True)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    # One line per child, on the node, so a census can be ASKED what its mesh
    # cells cost and how many children it paid for.  Duplicate children for
    # one module mean the xdist grouping did not hold, which is a cost the
    # run should be able to show rather than a thing to infer from wall time.
    try:
        with open(node_tmp / ("lorrax-mesh-%s.log" % os.environ.get("USER", "x")),
                  "a") as log:
            log.write("pid=%d cells=%d rc=%d secs=%.1f devs=%s first=%s\n"
                      % (os.getpid(), len(nodeids), res.returncode,
                         time.time() - t0, ",".join(devices),
                         nodeids[0] if nodeids else "-"))
    except Exception:                                          # noqa: BLE001
        pass
    return res, xml


# Output files never copied from a fixture dir into a run dir.
_FIXTURE_IGNORE = (
    "tmp", "eqp_test.dat", "eqp0_test.dat", "eqp1_test.dat",
    "sigma_diag*.dat", "eqp0.dat", "eqp1.dat", "eqp_g0w0.dat",
    "sigma_mnk.h5", "*_qp.h5", "qp_wfn_rotations.h5",
)


def gpu_available() -> bool:
    try:
        import jax

        return any(getattr(dev, "platform", "") in {"gpu", "cuda"}
                   for dev in jax.devices())
    except Exception:
        return False


def requested_platform() -> str:
    # Default to JAX's native backend selection (typically GPU on test nodes).
    platform = os.environ.get("ISDF_COHSEX_TEST_PLATFORM", "auto").strip().lower()
    valid = {"cpu", "gpu", "cuda", "auto"}
    if platform not in valid:
        raise ValueError(
            f"Invalid ISDF_COHSEX_TEST_PLATFORM={platform!r}. "
            f"Expected one of {sorted(valid)}."
        )
    return platform


def skip_unless_gpu(pytest):
    """Common gate: skip when the requested platform needs a missing GPU."""
    if requested_platform() in {"gpu", "cuda"} and not gpu_available():
        pytest.skip("CUDA GPU not available for the requested platform.")


def copy_fixture(case_dir: Path, run_dir: Path, *, tmp_from: Path = None):
    """Copy a regression fixture dir into a run dir, minus outputs.

    ``tmp_from`` (a previous run dir) additionally copies its ``tmp/``
    (the ISDF restart state) — used by the Tier-2 from-restart variants.
    Each variant needs its OWN copy: the driver mutates the restart file
    in place (``persist_w0_and_head`` writes W0_qmunu + head scalars back).
    """
    shutil.copytree(
        case_dir, run_dir,
        ignore=shutil.ignore_patterns(*_FIXTURE_IGNORE))
    if tmp_from is not None:
        src_tmp = Path(tmp_from) / "tmp"
        assert src_tmp.is_dir(), f"no restart state to copy: {src_tmp}"
        shutil.copytree(src_tmp, run_dir / "tmp")
    # copytree preserves modes, and the fixtures themselves are kept
    # READ-ONLY at rest (see ``protect_fixtures``).  Restore owner-write on
    # the COPY: Tier-2 variants edit their run dir's input file
    # (``mutate_input``) and the driver rewrites tmp/ state in place.
    make_writable(run_dir)
    return run_dir


def make_writable(root: Path) -> None:
    """Give the owner write permission on ``root`` and everything under it."""
    root = Path(root)
    os.chmod(root, os.stat(root).st_mode | stat.S_IWUSR)
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            try:
                os.chmod(p, os.stat(p).st_mode | stat.S_IWUSR)
            except OSError:
                pass


def protect_fixtures(reg_root: Path = None) -> list:
    """Make every regression FIXTURE file read-only; return what it changed.

    Why this exists
    ---------------
    Gates are staged by copying ``tests/regression/<case>/`` into a scratch
    run dir.  On 2026-07-25 one sbatch stager used ``ln -sf`` instead of
    ``cp`` — the driver then wrote its ``sigma_mnk.h5`` output THROUGH the
    symlink and silently destroyed the checked-in fixture.  Nothing failed;
    the corruption was noticed by eye.

    Defence in depth, cheapest layer first:
      1. fixtures are ``a-w`` at rest (this function, called from
         ``conftest.pytest_sessionstart``) — a write through a stray
         symlink now fails loudly with EACCES;
      2. stagers copy, never link (``cp -L`` if the source may be a link);
      3. run-dir copies get owner-write back (:func:`make_writable`).

    The protected set is exactly the **git-tracked** files under
    ``tests/regression/`` — not a filename heuristic.  That distinction
    matters: ``sigma_mnk.h5`` is in ``_FIXTURE_IGNORE`` (the driver writes a
    file of that name, so it is never copied into a run dir) and is ALSO a
    checked-in reference artifact.  It is the file the 2026-07-25 incident
    destroyed. A name-based rule would have skipped precisely the victim.

    Self-healing rather than assertive: it chmods and reports.  A hard
    failure here would strand a fresh clone whose umask left files writable,
    which is every clone.  Outside a git checkout it is a no-op.
    """
    reg_root = Path(reg_root) if reg_root is not None else REG
    if not reg_root.is_dir():
        return []
    try:
        out = subprocess.run(
            ["git", "-C", str(reg_root), "ls-files", "-z", "--full-name", "."],
            capture_output=True, timeout=60)
        if out.returncode != 0:
            return []
        top = subprocess.run(
            ["git", "-C", str(reg_root), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return []
    if not top:
        return []
    changed = []
    for rel in out.stdout.decode().split("\0"):
        if not rel:
            continue
        path = Path(top) / rel
        if path.is_symlink() or not path.is_file():
            continue
        mode = os.stat(path).st_mode
        ro = mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        if ro != mode:
            try:
                os.chmod(path, ro)
                changed.append(str(path))
            except OSError:
                pass
    return changed


def run_gw_jax(run_dir, input_name, platform=None, extra_env=None,
               timeout=900):
    """Run ``python -m gw.gw_jax -i <input_name>`` in run_dir; return the process."""
    if platform is None:
        platform = requested_platform()
    env = os.environ.copy()
    cache_setting = env.get(
        "ISDF_JAX_CACHE_DIR", str(REPO_ROOT / ".pytest_jax_cache"))
    if cache_setting.strip():
        Path(cache_setting).mkdir(parents=True, exist_ok=True)
    env.setdefault("ISDF_JAX_CACHE_DIR", cache_setting)
    # One cache owner: callers testing the native JAX knob may restore it in
    # ``extra_env`` below, but ordinary regression runs exercise LORRAX's
    # agreement/atomic-write path rather than an inherited global cache.
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    env.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    env.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
    env.setdefault("JAX_ENABLE_X64", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if platform == "cpu":
        env["JAX_PLATFORMS"] = "cpu"; env["JAX_PLATFORM_NAME"] = "cpu"
    elif platform in {"gpu", "cuda"}:
        env["JAX_PLATFORMS"] = "cuda,cpu"; env["JAX_PLATFORM_NAME"] = "gpu"
    else:
        env.pop("JAX_PLATFORMS", None); env.pop("JAX_PLATFORM_NAME", None)
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "gw.gw_jax", "-i", input_name],
        cwd=run_dir, env=env, capture_output=True, text=True,
        timeout=timeout, check=False,
    )


def parse_eqp_rows(path: Path, labels=("sigSX", "sigCOH", "sigTOT")) -> np.ndarray:
    """Parse sigma_diag rows → (nrows, 7): kpt, band, 3 Σ columns, VH re/im.

    THE DIRECT-FIELD COLUMN HAS TWO SPELLINGS and this parser accepts both.
    A scalar deck writes ``VH=``; a ``bispinor = true`` deck writes ``Hdir=``,
    the aggregate ``Hdir = V_H + H_T`` of the transverse-Hartree split
    (``file_io/sigma_output.py:138,1141``).  Requiring ``VH=`` alone made
    every bispinor ``sigma_diag.dat`` unparseable here — including the frozen
    ``bispinor_debug`` reference gate, which could only ever have passed on
    its byte-identity fast path and would raise
    ``No Sigma data rows were parsed`` the moment a last-ULP drift sent it to
    the atol comparison.  Column 5 of the returned array is whichever of the
    two the file carries.
    """
    float_re = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    imag_opt = rf"(?:\+\s*{float_re}i)?"  # optional imaginary part
    a, b, c = labels  # COHSEX: sigSX/sigCOH/sigTOT ;  GN-PPM: sigX/sigC/sigXC
    data_re = re.compile(
        rf"n=\s*(\d+)\s+"
        rf"{a}=\s*{float_re}{imag_opt}\s+"
        rf"{b}=\s*{float_re}{imag_opt}\s+"
        rf"{c}=\s*{float_re}{imag_opt}\s+"
        rf"(?:VH|Hdir)=\s*{float_re}{imag_opt}"
    )
    kpt_re = re.compile(r"k-point\s+(\d+)\s*:")

    kpt = -1
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        k_match = kpt_re.search(line)
        if k_match:
            kpt = int(k_match.group(1))
            continue
        m = data_re.search(line)
        if not m:
            continue
        band = int(m.group(1))
        # Groups: 2=A_re, 3=A_im, 4=B_re, 5=B_im, 6=C_re, 7=C_im, 8=VH_re, 9=VH_im
        rows.append([float(kpt), float(band), float(m.group(2)),
                     float(m.group(4)), float(m.group(6)), float(m.group(8)),
                     float(m.group(9)) if m.group(9) else 0.0])
    if not rows:
        raise ValueError(f"No Sigma data rows were parsed from {path}")
    return np.asarray(rows, dtype=np.float64)


#: Header lines a bit-identity comparison must NOT include.
#:
#: ``# Generated by LORRAX`` is the run timestamp — obviously per-run.
#:
#: ``# star_spread_ev`` and its per-band row were added to the wedge writers
#: on 2026-08-15.  They are a DIAGNOSTIC computed by reduction over the
#: full-BZ Σ and printed to 9 significant figures, and on a clean deck their
#: VALUE IS ROUNDOFF.  MEASURED on ``bispinor_debug`` under the μ-pad flip
#: this file's own gate performs (``LORRAX_EXTRA_MU_PAD`` 4 vs 0):
#:
#:     star_spread_ev   1.265016891e-08   (pad 0)
#:                      1.265016181e-08   (pad 4)
#:
#: — a 7.1e-15 eV difference in the 8th significant figure of a 1.3e-8 eV
#: quantity — while **every data row of sigma_diag, eqp0.dat and eqp1.dat is
#: BYTE-IDENTICAL** across the same flip.  Σ really is pad-invariant; a
#: reduction order over roundoff is not, and cannot be.
#:
#: Including these lines therefore broke ``test_mu_pad_flip_invariance_*``
#: with no physics behind it, which is the failure mode a bit-identity check
#: is most vulnerable to: a new header turns a real invariant into an
#: unachievable one.  The DATA is still compared byte for byte, which is what
#: the gate is for.
_NON_REPRODUCIBLE_HEADERS = (
    "# Generated by LORRAX",
    "# star_spread_ev",
    "# star_spread_multiplet_ev",
)


def normalize_dat(text: str) -> str:
    """Drop per-run and roundoff-valued header lines.

    Every numeric byte of the DATA still participates in identity checks;
    see :data:`_NON_REPRODUCIBLE_HEADERS` for what is excluded and the
    measurement that says why.
    """
    return "\n".join(
        ln for ln in text.splitlines()
        if not ln.startswith(_NON_REPRODUCIBLE_HEADERS)
    )


def census_lines(log_text: str) -> tuple:
    """PPM census + adaptive-window signature from a run log — the integer
    quantities that must be exactly invariant under μ-pad flips."""
    windows = tuple(re.findall(
        r'window "(\w+)" \((?:crossing|Laplace)\): (\d+) nodes', log_text))
    m = re.search(r"GN invalid modes: (\d+)/(\d+)", log_text)
    u = re.search(r"unfulfilled=([\d.]+)%", log_text)
    return (windows,
            m.groups() if m else None,
            u.group(1) if u else None)


def eqp_column(path: Path) -> np.ndarray:
    """E_qp column of an eqp0/eqp1.dat file (data rows are 'ik n Edft Eqp';
    k-point header rows are 4 floats — distinguished by '.' in field 0)."""
    vals = []
    for ln in path.read_text().splitlines():
        s = ln.split()
        if len(s) == 4 and not ln.startswith('#') and '.' not in s[0]:
            vals.append(float(s[3]))
    return np.asarray(vals, dtype=np.float64)


def numeric_tokens(path: Path) -> np.ndarray:
    """All numeric tokens of a whitespace-separated .dat file, in order."""
    toks = []
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        for t in line.split():
            try:
                toks.append(float(t))
            except ValueError:
                pass
    return np.asarray(toks, dtype=np.float64)


def mutate_input(path: Path, replacements: dict[str, str], append: str = ""):
    """Apply exact-string replacements to an input file (each must hit)."""
    text = path.read_text()
    for old, new in replacements.items():
        assert old in text, f"{path}: expected {old!r} in input"
        text = text.replace(old, new)
    if append:
        text += "\n" + append + "\n"
    path.write_text(text)


# ---------------------------------------------------------------------------
# BerkeleyGW anchor comparison (the ONE external check in the suite).
#
# Every other gate in this repo compares LORRAX against LORRAX's own frozen
# output.  That catches "the code changed" but is structurally blind to "the
# code drifted away from BerkeleyGW", because BGW never enters the loop.  The
# helpers below put it back in: they read a fixture of literal BGW sigma_hp.log
# columns and line it up with a LORRAX sigma_diag .dat.
#
# COLUMN CONVENTION.  BGW's 14-column sigma_hp.log block
# (Sigma/write_result_hp.f90:88-100) writes
#     CH  = ach + achcor      Sig  = asig + achcor
#     CH` = ach               Sig` = asig
# where ``achcor`` is the STATIC REMAINDER.  LORRAX computes no static
# remainder, so the comparable columns are the PRIMED ones, and the mapping
# below applies NO offset to either side:
#     LORRAX sigSX  == X + SXmX      LORRAX sigCOH == CHp      sigTOT == Sigp
# Comparing against the UNPRIMED CH instead would show a spurious ~367 meV.
# ---------------------------------------------------------------------------

BGW_HP_COLS = ("Emf", "Eo", "X", "SXmX", "CH", "Sig", "KIH", "Eqp0", "Eqp1",
               "CHp", "Sigp", "Eqp0p", "Eqp1p", "Znk")


def parse_bgw_hp_fixture(path: Path):
    """Read a bgw_sigma_hp_*.dat fixture.

    Returns ``(kfrac (nk,3), bands (nb,), {col: (nk, nb)})``.  Rows are
    ``ik kx ky kz n <14 columns>``; ``#`` lines are commentary.
    """
    rows = [ln.split() for ln in Path(path).read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    if not rows:
        raise ValueError(f"no data rows in BGW fixture {path}")
    nk = max(int(r[0]) for r in rows)
    bands = sorted({int(r[4]) for r in rows})
    bpos = {b: j for j, b in enumerate(bands)}
    kfrac = np.zeros((nk, 3))
    data = {c: np.full((nk, len(bands)), np.nan) for c in BGW_HP_COLS}
    for r in rows:
        ik = int(r[0]) - 1
        kfrac[ik] = [float(r[1]), float(r[2]), float(r[3])]
        j = bpos[int(r[4])]
        for c, v in zip(BGW_HP_COLS, r[5:]):
            data[c][ik, j] = float(v)
    for c, arr in data.items():
        if np.isnan(arr).any():
            raise ValueError(f"BGW fixture {path}: column {c} has holes")
    # The fixture must be internally consistent or the parse is wrong.
    resid = np.abs(data["X"] + data["SXmX"] + data["CHp"] - data["Sigp"]).max()
    if resid > 1e-6:
        raise ValueError(
            f"BGW fixture {path} is not self-consistent: "
            f"max|X + SXmX + CHp - Sigp| = {resid:.3e} eV (expected ~1e-9). "
            f"Column order is wrong or the file was edited by hand.")
    return kfrac, np.asarray(bands), data


def _parse_kcrys_blocks(path: Path) -> dict:
    """{kpt index: (kx, ky, kz)} from a sigma_diag .dat's ``# kcrys`` lines."""
    out, ik = {}, -1
    for ln in Path(path).read_text().splitlines():
        m = re.search(r"k-point\s+(\d+)\s*:", ln)
        if m:
            ik = int(m.group(1))
            continue
        if ln.startswith("# kcrys") and ik >= 0:
            out[ik] = tuple(float(x) for x in ln.split()[2:5])
    return out


def _parse_star_spread_header(path: Path):
    """``(per_band (nb,), n_star_members)`` from the writer's header, or None.

    The diagnostic is MEASURED UPSTREAM, on the full-BZ Sigma, against the
    symmetry service's own star labels
    (``gw_output._star_spread_of_sigma_diag``) — it cannot be recomputed
    from this file, because the file is the wedge and unfolding a wedge is
    a gather, which would report 0.000 by construction.

    THE PER-BAND VECTOR IS WHAT THIS RETURNS, deliberately.  The scalar
    ``star_spread_ev`` beside it is the max over the driver's WHOLE sigma
    window, which is a wider question than any given comparison asks: on
    the Si anchor that window is 60 bands and reads 41.34 meV, while the
    16 bands the BerkeleyGW fixture covers read 2.61.  The band scope is
    the consumer's knowledge, so the consumer takes its own max.
    """
    per_band, multiplet, n_members = None, None, None
    for ln in Path(path).read_text().splitlines():
        if not ln.startswith("#") and ln.strip():
            break
        m = re.match(r"#\s*star_spread_multiplet_ev_per_band\s+(.*)$", ln)
        if m:
            multiplet = np.asarray([float(x) for x in m.group(1).split()])
            continue
        m = re.match(r"#\s*star_spread_ev_per_band\s+(.*)$", ln)
        if m:
            per_band = np.asarray([float(x) for x in m.group(1).split()])
            continue
        m = re.match(r"#\s*star_spread_ev\s+(\S+)", ln)
        if m:
            n = re.search(r"over the (\d+)\s+full-BZ k", ln)
            n_members = int(n.group(1)) if n else None
    if per_band is None:
        return None
    return per_band, multiplet, n_members


def compare_to_bgw(output_file: Path, fixture: Path, labels=(
        "sigSX", "sigCOH", "sigTOT")):
    """Deviation of a LORRAX sigma_diag .dat from the BGW anchor, in meV.

    Returns ``{column: (mae, max_abs)}`` plus ``"_star_spread"`` and
    ``"_nstar"``.

    BOTH SIDES ARE THE IRREDUCIBLE WEDGE, JOINED ON THE CRYSTAL
    COORDINATE.  The fixture holds BerkeleyGW's 8 IBZ k with their
    fractional coordinates; since 2026-08-15 LORRAX writes its own wedge
    with a ``# kcrys`` line per block, so the two are matched on the k
    itself — exactly, modulo a lattice vector, with an ambiguous match
    REFUSED rather than resolved.

    This replaced a nearest-match over MEAN-FIELD ENERGY vectors at
    2e-3 eV, whose stated justification was that "the two codes do not
    order k the same way".  True, and irrelevant now that both files name
    their k: an energy fingerprint aliases whenever two stars are
    degenerate across the compared window, and it was one of the
    hand-rolled k-matching sites this branch removes.

    ``_star_spread`` and ``_nstar`` ARE READ FROM THE FILE HEADER, not
    recomputed here.  They are measured in ``gw_output`` on the full-BZ
    Sigma against ``sym.irr_idx_k``, because that is the only place the
    information exists: symmetry-equivalent k carry independently
    computed Sigma before the writer reduces to the wedge, and no
    downstream unfold can bring that back (``star_broadcast`` is a
    gather, so every member would equal its parent and the spread would
    read a fake 0.000).  ``_nstar`` therefore still counts the full-BZ
    k the diagnostic covered — 64 on the Si production deck — and still
    fails the same way if the k assignment collapses.

    ``_star_spread`` IS NOT A SYMMETRY GATE, and must not be read as one.
    It is a max-minus-min over the REAL DIAGONAL ``sigTOT`` values of a
    star's members, so it is structurally blind to the entire time-reversal
    CONJUGATION class: conjugating a Hermitian block leaves its real
    diagonal EXACTLY intact.  That is not a theoretical gap.  27cc885
    measured the wrong ``trs_reference`` at **183.61 eV** against an
    independently computed V_H with "the DIAGONAL left exactly intact", so
    the electron count, hermiticity, the spectrum, the eqp.dat V_H column
    and every diagonal observable were unchanged — which is why nothing
    caught it for a month.  REPRODUCED LIVE on the committed
    ``cohsex_debug/sigma_mnk.h5``: conjugating one time-reversed star member
    moves this metric by **EXACTLY 0.0** (1.2130460739135742 before and
    after, on ``sigma_sx_kij_ev``) while the conjugation relation it broke
    jumps five orders, 6.980e-04 -> 3.992e-01.

    The metric STAYS — it is the right check for what it checks, agreement
    with the BerkeleyGW anchor on the diagonal Sigma each code reports —
    but the off-diagonal question is asked somewhere else, on the full
    matrices where it is answerable: ``tests/test_star_offdiag_gate.py``,
    which gates the TRS conj-pair relation on ``sigma_mnk.h5`` and carries
    the corruption twin that pins the blindness stated here.
    """
    kfrac, bands, bgw = parse_bgw_hp_fixture(fixture)
    nb = bands.size
    rows = parse_eqp_rows(output_file, labels)
    lx = {}
    for r in rows:
        lx.setdefault(int(r[0]), []).append(r)
    lx = {k: np.asarray(v) for k, v in lx.items()}

    kcrys = _parse_kcrys_blocks(output_file)
    if not kcrys:
        raise AssertionError(
            f"{output_file} carries no '# kcrys' lines, so its blocks cannot "
            f"be joined to BerkeleyGW's by k.  A file written before "
            f"2026-08-15 is on the full BZ and anonymous; regenerate it.")

    ref = {labels[0]: bgw["X"] + bgw["SXmX"],
           labels[1]: bgw["CHp"],
           labels[2]: bgw["Sigp"]}
    acc = {c: [] for c in labels}
    for ik in range(kfrac.shape[0]):
        # Exact join on k, modulo a lattice vector.  Not a nearest match:
        # every candidate within the round-trip epsilon of '%13.9f' is
        # collected, and anything other than exactly one is an error.
        hits = []
        for k, kv in kcrys.items():
            d = np.asarray(kv) - kfrac[ik]
            if np.max(np.abs(d - np.rint(d))) < 1e-6:
                hits.append(k)
        if len(hits) != 1:
            raise AssertionError(
                f"BGW anchor: BGW IBZ k{ik + 1} = {kfrac[ik]} matched "
                f"{len(hits)} LORRAX blocks {hits}.  Expected exactly one — "
                f"the two wedges must be the same k-set.  LORRAX blocks: "
                f"{sorted(kcrys.values())}")
        k = hits[0]
        if lx[k].shape[0] < nb:
            raise AssertionError(
                f"LORRAX block {k} has {lx[k].shape[0]} bands, BGW has {nb}")
        for j, c in enumerate(labels):
            acc[c].append(lx[k][:nb, 2 + j] - ref[c][ik])

    out = {}
    for c in labels:
        d = np.concatenate(acc[c]) * 1e3
        out[c] = (float(np.abs(d).mean()), float(np.abs(d).max()))

    hdr = _parse_star_spread_header(output_file)
    if hdr is None:
        raise AssertionError(
            f"{output_file} carries no '# star_spread_ev_per_band' header "
            f"row.  That diagnostic is measured on the full BZ inside "
            f"gw_output before the writer reduces to the wedge; without it "
            f"there is nothing here to report, and recomputing it from a "
            f"wedge file would return a fake 0.000.")
    per_band, multiplet, n_members = hdr
    if per_band.shape[0] < nb:
        raise AssertionError(
            f"star-spread row covers {per_band.shape[0]} bands but this "
            f"comparison spans {nb}; the file's sigma window is narrower "
            f"than the BerkeleyGW fixture.")
    # THE CONSUMER'S OWN SCOPE: the max over exactly the bands compared
    # above, not the driver's whole sigma window.
    out["_star_spread"] = float(per_band[:nb].max()) * 1e3

    # ── THE CUT THIS COMPARISON MAKES, AND WHETHER IT IS CLEAN ────────────
    # ``nb`` is whatever the BerkeleyGW fixture happens to carry, and until
    # 2026-08-15 it was applied with no check at all.  A band boundary that
    # falls INSIDE a degenerate multiplet is not a safe place to truncate:
    # within a multiplet the band index is arbitrary, so a truncated sum is
    # not invariant under the group where a complete one is.  The energies
    # to decide it are in this very file (the ``Eo=`` column), and the rule
    # is the tree's one rule -- ``common.band_degeneracy.boundary_min_gaps``.
    #
    # REPORTED, NOT REFUSED.  This is a diagnostic, the fixture's band count
    # is not ours to move, and on the Si anchor the cut at 16 MEASURES CLEAN
    # anyway.  A refusal here would fail a gate over a property of a
    # reference file rather than of the code under test.
    out["_cut_clean"] = None
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from common.band_degeneracy import (DEGENERACY_TOL_RY,
                                            boundary_min_gaps)
        eo = _parse_eo_column(output_file)
        e_ry = np.asarray([eo[k] for k in sorted(eo)]) / 13.6056980659
        if e_ry.ndim == 2 and e_ry.shape[1] >= nb:
            gaps = boundary_min_gaps(e_ry, is_full_spectrum=False)
            out["_cut_clean"] = bool(nb >= gaps.size
                                     or gaps[nb] > DEGENERACY_TOL_RY)
    except Exception:                                          # noqa: BLE001
        pass                                    # diagnostic only, never fatal

    # ── THE SUBSPACE-INVARIANT TWIN ───────────────────────────────────────
    # ``_star_spread`` is a PER-BAND max-min, and a per-band ``Re Sigma_bb``
    # inside a degenerate multiplet is not a symmetry-invariant quantity:
    # any unitary mixing within the subspace is an equally valid eigenbasis.
    # MEASURED on the Si production deck 2026-08-15, where 60 of 60 bands
    # sit inside a multiplet: per-band 41.338 meV over the full window
    # against 6.734 meV on the multiplet traces, and 2.611 -> 0.593 meV over
    # the 16 bands THIS comparison spans.  Both are reported because they
    # answer different questions; the multiplet one is the one to quote for
    # "is the symmetry broken".
    out["_star_spread_multiplet"] = (
        float(multiplet[:nb].max()) * 1e3 if multiplet is not None else None)
    out["_nstar"] = n_members
    return out


def _parse_eo_column(path: Path) -> dict:
    """{kpt: [Eo per band]} from a sigma_diag .dat (the ``Eo=`` field)."""
    out, ik = {}, -1
    for ln in Path(path).read_text().splitlines():
        m = re.search(r"k-point\s+(\d+)\s*:", ln)
        if m:
            ik = int(m.group(1))
            out[ik] = []
            continue
        m = re.search(r"\bEo=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", ln)
        if m and ik >= 0:
            out[ik].append(float(m.group(1)))
    return {k: v for k, v in out.items() if v}
