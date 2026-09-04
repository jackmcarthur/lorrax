"""One GPU per process — and the case where that used to be skipped.

``tests/conftest.py`` pins ``CUDA_VISIBLE_DEVICES`` at module scope, before
the first CUDA init.  That is the one moment a test cannot observe from
inside the same process, so the DECISION is a pure function in
``tests/harness.pin_one_gpu`` and this file constructs every case.

THE REGRESSION THIS PINS.  The pin used to be inside
``if PYTEST_XDIST_WORKER.startswith("gw"):`` — i.e. it only happened when
pytest-xdist was fanning out.  A NON-xdist run of the same suite on the
same node therefore saw all N GPUs, and three things break there:

  * the e2e gates' subprocesses build an N-device mesh and compare against
    1-GPU-frozen reference numbers;
  * SLATE refuses outright — MEASURED on Perlmutter 2026-08-07,
    ``slate.potrf: blas::get_device_count()=4 but JAX one-process-per-GPU
    model requires exactly 1`` — which killed 8 contract cells in the
    SERVICE-ONLY leg (``pytest services/distrib_la/tests``, which never
    loads this conftest);
  * ``services/distrib_la/tests/conftest.py`` has its own copy of the pin,
    but guarded on ``"jax" not in sys.modules``, and in a full-suite run
    ``testpaths = ["tests", "services"]`` collects ``tests/`` first, some
    module there imports jax during collection, and that copy is inert by
    the time it loads.  It cannot cover the full-suite leg by
    construction; only ``tests/conftest.py`` loads early enough.

CORRECTION, 2026-08-07 (step 4) — READ THIS BEFORE REUSING THE STORY
ABOVE.  An earlier revision of this docstring said the ``=4`` refusal was
what killed 8 cells in the FULL-SUITE ``-m distrib_la`` leg.  That was a
misdiagnosis and it is worth leaving the correction here, because "we
already fixed that" is how a second cause hides behind a first.  The
full-suite leg failed with ``blas::get_device_count()=0`` at EXACTLY ONE
visible device, before AND after this pin became unconditional.  Different
number, different cause: the two platform ``.so``s share ``libslate.so.2``
and ``libblaspp.so.2`` by SONAME, the host build's blaspp has no CUDA and
answers 0, and whichever library is dlopened first wins for both.  The fix
is a load-order rule in both loaders (``_open_cuda_before_host``); the
evidence is ``dladdr`` in both legs.

This pin is still correct and still free, and all three reasons above
stand — it just never was the cause of the eight.  ``test_a_bare_process_
still_gets_pinned`` is the cell that fails on the old code; the rest keep
the fan-out and the no-op cases from regressing in the other direction.

SECOND CORRECTION, 2026-08-07 (KNOWN_FAILURES B2) — WHAT UNCONDITIONAL
COST.  "Without a worker id, take the first visible device" is right for a
process that computes and wrong for exactly one process: the pytest-xdist
CONTROLLER, which has no worker id, runs no tests, and whose environ is
what the workers are SPAWNED FROM.  It took ``devs[0]``, wrote
CUDA_VISIBLE_DEVICES="0" into itself, and every worker then inherited a
one-element device list:

    gw0 gw1 gw2 gw3    the six gnppm-session files under `lx test`
    '0' '1' '2' '3'    78ddcee   24 / 24 passed
    '0' '0' '0' '0'    6920171   11 P / 2 F / 11 E, RESOURCE_EXHAUSTED:
                                 Failed to allocate 19.20 GiB on device
                                 ordinal 0

Every cell in this file stayed GREEN through that, because they all
construct the preset by hand and none of them ever saw the controller's
write.  That is the hole ``test_the_controller_does_not_narrow_what_the_
workers_inherit`` closes: it spawns the real four-worker arm and reads
each worker's own CUDA_VISIBLE_DEVICES back, which is the probe that
caught it.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

import harness


def _probe(n):
    return lambda: n


# ---------------------------------------------------------------------------
#  THE REGRESSION
# ---------------------------------------------------------------------------

def test_a_bare_process_still_gets_pinned():
    """No xdist worker id, four visible GPUs → pin to the first.

    RED ARM: this is exactly what the pre-fix guard returned nothing for.
    """
    assert harness.pin_one_gpu("0,1,2,3", "") == "0"
    assert harness.pin_one_gpu(None, "", probe=_probe(4)) == "0"


def test_the_pin_respects_slurms_selection_not_the_global_index():
    """SLURM hands a SUBSET; the pick indexes into that list, not into the
    node's physical devices.  Picking ``str(i)`` directly would hand a
    process a GPU its cgroup does not contain."""
    assert harness.pin_one_gpu("2,3", "") == "2"
    assert harness.pin_one_gpu("2,3", "gw1") == "3"


# ---------------------------------------------------------------------------
#  …without losing the fan-out it was written for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("wid,want", [
    ("gw0", "0"), ("gw1", "1"), ("gw2", "2"), ("gw3", "3"),
    ("gw4", "0"),                    # more workers than GPUs: wrap
])
def test_xdist_workers_still_fan_out(wid, want):
    assert harness.pin_one_gpu("0,1,2,3", wid) == want


def test_a_non_worker_id_is_not_read_as_a_worker():
    """``master`` (the xdist controller) and any other spelling take the
    first device rather than crashing on ``int(...)``."""
    assert harness.pin_one_gpu("0,1,2,3", "master") == "0"
    assert harness.pin_one_gpu("0,1,2,3", "gw") == "0"


# ---------------------------------------------------------------------------
#  …and the one process that must NOT be pinned: the xdist controller
# ---------------------------------------------------------------------------

def test_the_controller_takes_no_device_at_all():
    """THE B2 REGRESSION, as a pure cell.

    ``devs[0]`` is the right answer for every process that computes and the
    wrong answer for the one that does not: what the controller writes is
    what its workers inherit as their preset, and a one-element preset
    collapses the fan-out.
    """
    assert harness.pin_one_gpu("0,1,2,3", "", controller=True) is None
    assert harness.pin_one_gpu(None, "", probe=_probe(4), controller=True) is None


def test_the_controller_arm_can_fail():
    """The FALSE case: the SAME inputs without the controller flag pin
    ``"0"``.  Without this twin the cell above would pass on any change
    that made ``pin_one_gpu`` return ``None`` for everything."""
    assert harness.pin_one_gpu("0,1,2,3", "", controller=False) == "0"


@pytest.mark.parametrize("worker_id,dist,want", [
    ("",    "load", True),    # the controller: no worker id, fanning out
    ("",    "each", True),
    ("gw0", "load", False),   # a worker: has an id, and it computes
    ("gw3", "load", False),
    ("",    "no",   False),   # a plain single-process run: it computes too
    ("",    None,   False),   # xdist not installed / -p no:xdist
    ("",    "",     False),
])
def test_the_controller_signal_is_xdists_own(worker_id, dist, want):
    """``PYTEST_XDIST_WORKER`` + ``config.option.dist``, which is exactly
    ``xdist.is_xdist_controller``.

    The row that matters most is ``("", "no")``: a non-xdist run has no
    worker id either, and it is going to compute in this very process — so
    "no worker id" alone is NOT the controller test, and using it would
    re-open the leg that ``6920171`` was written to close.
    """
    assert harness.is_xdist_controller(worker_id, dist) is want


def _gpu_ids():
    """The node's real GPUs, or ``[]`` where there is no ``nvidia-smi``.

    Same probe ``harness._probe_nvidia_smi`` uses, and it must not raise:
    the CPU legs and the WSL box have no such binary and this cell has to
    SKIP there, not error.
    """
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=30).stdout
    except Exception:                                          # noqa: BLE001
        return []
    return [ln for ln in out.strip().splitlines() if ln.strip()]


def test_the_controller_does_not_narrow_what_the_workers_inherit(tmp_path):
    """THE PROBE THAT CAUGHT B2, frozen into the suite.

    Every other cell here calls ``pin_one_gpu`` directly and none of them
    could see the regression, because the defect was not in the decision —
    it was in WHICH PROCESS applied it and what the next process inherited.
    Only a real fan-out can answer that, so this one spawns the four-worker
    arm and reads each worker's own ``CUDA_VISIBLE_DEVICES`` back out.

    MEASURED (Perlmutter 2026-08-07): ``'0','1','2','3'`` at ``78ddcee``
    against ``'0','0','0','0'`` at the census HEAD.  The assertion is
    DISTINCTNESS rather than the literal list, because SLURM hands a subset
    and the pick indexes into it.
    """
    pytest.importorskip("xdist")
    gpus = _gpu_ids()
    if len(gpus) < 2:
        pytest.skip(f"needs >=2 real GPUs to fan out over, nvidia-smi -L "
                    f"lists {len(gpus)}")
    n = min(4, len(gpus))

    # THE CONFTEST UNDER TEST, COPIED — not the repo's own tests/ dir.  The
    # arm needs a rootdir whose conftest is this one, and writing a probe
    # file into the checkout to get that would mutate the source tree from
    # inside a test.  ``harness`` comes along because the conftest imports
    # it by path.
    for name in ("conftest.py", "harness.py"):
        (tmp_path / name).write_text(
            (harness.REPO_ROOT / "tests" / name).read_text())
    # EACH WORKER REPORTS THROUGH A FILE, not through stdout.  Under xdist a
    # worker's stdout does not reach the controller (execnet owns it, and -s
    # does not change that) -- measured here first: 16 probe cells passed
    # and the controller's stdout carried none of their prints.  A file per
    # worker is the transport that survives the process boundary, and it is
    # also the only one whose absence is unambiguous.
    reports = tmp_path / "reports"
    reports.mkdir()
    (tmp_path / "test_cvd_probe.py").write_text(textwrap.dedent("""\
        import os
        import pathlib

        import pytest


        @pytest.mark.parametrize("i", range(16))
        def test_probe(i):
            wid = os.environ.get("PYTEST_XDIST_WORKER", "<none>")
            out = pathlib.Path(os.environ["CVD_PROBE_DIR"]) / (wid + ".txt")
            out.write_text(repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
        """))
    # THIS PROCESS IS ITSELF ALREADY PINNED (it is a worker, or a bare run
    # that took device 0), so its own CUDA_VISIBLE_DEVICES is a one-element
    # list and inheriting it would make the arm trivially — and falsely —
    # red.  Hand the child the full list instead; inside a SLURM cgroup
    # ``nvidia-smi -L`` already lists only the devices this task owns, and
    # CUDA numbers them 0..n-1.  The probe imports no jax and allocates
    # nothing, so it cannot disturb a device another worker is using.
    env = {**os.environ,
           "JAX_PLATFORMS": "cpu",
           "CVD_PROBE_DIR": str(reports),
           "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(n))}
    # AND THE CHILD IS A FRESH SESSION, NOT A CONTINUATION OF THIS ONE.
    # When this cell runs under `lx test` it runs INSIDE a worker, whose
    # environ carries PYTEST_XDIST_WORKER=gw2 — and the child's CONTROLLER
    # would inherit that id, fail to recognise itself as a controller, pin
    # devs[2], and hand its own workers a one-element preset.  All four then
    # report '2' and this cell goes red for a reason that is about the
    # fixture, not about the tree.  MEASURED here first, leg A at bb5b5b2:
    # {'gw0': '2', 'gw1': '2', 'gw2': '2', 'gw3': '2'}.  (It is also the
    # defect's own shape, arrived at from the other direction, which is
    # some comfort about what the cell measures.)
    for k in ("PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT",
              "PYTEST_XDIST_TESTRUNUID", "PYTEST_CURRENT_TEST",
              "PYTEST_ADDOPTS"):
        env.pop(k, None)
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "test_cvd_probe.py", "-n", str(n),
         "-p", "no:randomly", "-q"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=900,
        env=env)

    seen = {p.stem: p.read_text().strip() for p in sorted(reports.iterdir())}
    assert len(seen) == n, (
        f"expected {n} workers to report, got {sorted(seen)}.\n"
        f"stdout:\n{res.stdout[-4000:]}\nstderr:\n{res.stderr[-2000:]}")
    devices = sorted(seen.values())
    assert len(set(devices)) == n, (
        f"the {n} xdist workers landed on {len(set(devices))} distinct "
        f"device(s): {seen}.  The fan-out is dead — which is what happens "
        f"when the CONTROLLER pins CUDA_VISIBLE_DEVICES into its own "
        f"environ and the workers inherit a one-element list.  Four gnppm "
        f"sessions on one 40 GiB A100 is RESOURCE_EXHAUSTED, not a slow "
        f"run.")


# ---------------------------------------------------------------------------
#  …and leaves alone what it must
# ---------------------------------------------------------------------------

def test_no_gpus_means_no_pin():
    """Every CPU leg.  Returning ``"0"`` here would hand JAX a device that
    does not exist; returning ``None`` leaves the environment untouched."""
    assert harness.pin_one_gpu(None, "", probe=_probe(0)) is None
    assert harness.pin_one_gpu(None, "gw2", probe=_probe(0)) is None


def test_an_explicit_mask_is_honoured_not_overridden():
    """``CUDA_VISIBLE_DEVICES=""`` means "no GPU, deliberately" — runtime/
    __init__.py reads exactly that spelling to decide a node has no GPU.
    Overriding it would un-mask a device the caller masked on purpose."""
    assert harness.pin_one_gpu("", "") is None
    assert harness.pin_one_gpu("", "gw1") is None


def test_the_probe_is_not_consulted_when_the_env_already_says():
    """A preset list is authoritative; probing nvidia-smi past it would let
    a device outside this process's cgroup back in."""
    def _explode():
        raise AssertionError("probe called despite a preset list")
    assert harness.pin_one_gpu("1", "gw3", probe=_explode) == "1"


# ---------------------------------------------------------------------------
#  …and what the pin costs: the cells that need a MESH  (2026-08-10)
# ---------------------------------------------------------------------------
# The pin's price was a whole class of cell that could not run under the
# suite at all: pinned to one GPU, `jax.device_count()` is 1, and every
# `skipif(device_count() < 4)` fired on every node however many GPUs it had.
# `harness.mesh_plan` is where that is now decided, and these are its cases.
# The decision is a pure function for the same reason `pin_one_gpu` is: its
# caller is a side effect in a conftest, which nothing can construct a case
# for.

def test_a_pinned_worker_on_a_four_gpu_node_gets_a_child_not_a_skip():
    """THE MEASURED GAP.  One device here, four on the node: the old spelling
    skipped, and the suite reported green with the cell never run."""
    assert harness.mesh_plan(4, 1, ["0", "1", "2", "3"], platform="cuda") == (
        "subprocess", ["0", "1", "2", "3"])


def test_a_process_that_already_has_the_mesh_runs_the_cell_itself():
    """The direct invocation — `XLA_FLAGS=--xla_force_host_platform_device_
    count=4 pytest ...` — must keep behaving exactly as it does today, and
    so must the child, which is the same case seen from inside."""
    assert harness.mesh_plan(4, 4, [], platform="cpu") == ("here", None)
    assert harness.mesh_plan(2, 8, [], platform="cpu") == ("here", None)
    assert harness.mesh_plan(4, 4, ["0", "1", "2", "3"],
                             platform="cuda") == ("here", None)


def test_emulated_devices_do_not_stand_in_for_the_nodes_real_ones():
    """The PARENT half of the same defect the child is checked for.

    A census worker carries `JAX_PLATFORMS=cpu` and four emulated host devices
    the moment collection has imported the modules that set them, so
    `jax.device_count()` answers 4 on a node with four A100s.  Counting alone,
    the cell would run right here on the emulated mesh and report a pass that
    says nothing about the hardware.  The session's own device list is what
    breaks the tie."""
    assert harness.mesh_plan(4, 4, ["0", "1", "2", "3"], platform="cpu") == (
        "subprocess", ["0", "1", "2", "3"])


def test_a_node_without_the_devices_still_skips_and_says_so():
    """A laptop, the WSL box, a one-GPU leg.  Skipping is the honest verdict
    there; the reason has to name both counts, or the next reader cannot tell
    "no hardware" from "the pin ate it"."""
    verb, why = harness.mesh_plan(4, 1, ["0"], platform="cuda")
    assert verb == "skip"
    assert "has 1" in why and "session has 1" in why


def test_the_child_never_starts_another_child_and_never_skips():
    """Bounds the recursion, and refuses the one outcome that would rebuild
    the defect: a child that did not get its devices reporting a SKIP would
    put the cell back in the silently-unexercised set, this time with a
    mechanism that looks like it works."""
    verb, why = harness.mesh_plan(
        4, 1, ["0", "1", "2", "3"], inner=True, platform="cpu")
    assert verb == "fail"
    assert "not a skip" in why


def test_a_child_that_came_up_emulated_is_a_failure_not_a_pass():
    """MEASURED, Perlmutter 2026-08-10, and the reason the child is checked on
    its PLATFORM and not only its count.

    ``tests/test_contract_bands`` and ``tests/test_sanity_gates_jax`` set
    ``JAX_PLATFORMS=cpu`` and four emulated host devices at MODULE SCOPE —
    they have to, the values are latched at ``import jax`` — and a module-scope
    write never unwinds.  So in a census every later child inherited them,
    came up with four HOST devices on a node with four A100s, and would have
    passed every mesh cell while measuring the emulated arm.  Four is four:
    only the platform separates the run that honours the four-GPU rule from
    the run that quietly substitutes for it."""
    verb, why = harness.mesh_plan(4, 4, ["0", "1", "2", "3"], inner=True,
                                  platform="cpu")
    assert verb == "fail"
    assert "quietly emulated" in why and "platform 'cpu'" in why
    assert harness.mesh_plan(4, 4, ["0", "1", "2", "3"], inner=True,
                             platform="cuda") == ("here", None)


def test_the_child_env_widens_only_the_child():
    """The parent's pin is not touched — the child gets its own environment —
    and the four edits that make a fifth process safe, and honest, on a busy
    node are all present."""
    base = {"CUDA_VISIBLE_DEVICES": "2", "PYTEST_XDIST_WORKER": "gw2",
            "JAX_ENABLE_X64": "1",
            # what a census's collection leaves behind, which the child must
            # NOT inherit:
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4"}
    env = harness.mesh_subprocess_env(
        base, ["0", "1", "2", "3"],
        {"JAX_PLATFORMS": None, "XLA_FLAGS": None})   # the caller had none
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert base["CUDA_VISIBLE_DEVICES"] == "2"          # caller untouched
    assert env[harness.MESH_CELL_ENV] == "1"
    assert env["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"
    assert "PYTEST_XDIST_WORKER" not in env
    assert env["JAX_ENABLE_X64"] == "1"
    assert "JAX_PLATFORMS" not in env and "XLA_FLAGS" not in env


def test_the_child_refuses_the_flat_k_fft_instead_of_aborting_on_it():
    """MEASURED by the perf fleet, 2026-08-10: an IN-PROCESS multi-device mesh
    cannot execute the flat-k cuFFT handler.  Every in-process 2x2 dies
    CUFFT_EXEC_FAILED at every size, as an UNCATCHABLE SIGABRT, while the
    production multi-process legs are fine.

    The child is exactly an in-process multi-device mesh, so a cell that
    wandered into `make_flat_k_*` would take the child down and every verdict
    in it with it, attributable to nothing.  `LORRAX_FFT_FFI=0` refuses by
    design (the XLA flat-k twin was deleted by the 2026-08-01 ruling), and a
    refusal names the cell that caused it.  A gate that genuinely needs the
    flat-k FFT at P=n is MULTI-PROCESS and belongs in tests/multi_device/."""
    env = harness.mesh_subprocess_env({}, ["0", "1", "2", "3"])
    assert env["LORRAX_FFT_FFI"] == "0"


def test_a_caller_who_asked_for_a_platform_still_gets_it():
    """The snapshot is of the CALLER's environment, not a blanket erase: an
    explicit `JAX_PLATFORMS=cpu` on the command line is a choice, and the
    child honours it.  Only what collection added is dropped."""
    base = {"JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--leaked"}
    env = harness.mesh_subprocess_env(
        base, ["0", "1"], {"JAX_PLATFORMS": "cpu", "XLA_FLAGS": None})
    assert env["JAX_PLATFORMS"] == "cpu"
    assert "XLA_FLAGS" not in env


def test_the_session_device_list_is_read_before_the_pin_narrows_it():
    """`session_devices` is the same reading `pin_one_gpu` starts from, kept
    whole.  After the pin there is nothing left to ask."""
    assert harness.session_devices("0,1,2,3") == ["0", "1", "2", "3"]
    assert harness.session_devices(None, probe=_probe(4)) == [
        "0", "1", "2", "3"]
    assert harness.session_devices("") == []


def test_junit_outcomes_reads_the_childs_verdicts_by_nodeid(tmp_path):
    """The child reports through pytest's own junit-xml, so there is no
    protocol of ours to drift.  xunit2 drops the `file` attribute, so the
    nodeid is rebuilt from the module we asked for — parametrised names and
    class nesting included."""
    xml = tmp_path / "m.xml"
    xml.write_text(
        '<?xml version="1.0"?><testsuites><testsuite name="pytest">'
        '<testcase classname="tests.test_x" name="test_a"/>'
        '<testcase classname="tests.test_x" name="test_b[2-3]">'
        '<failure message="boom">tb</failure></testcase>'
        '<testcase classname="tests.test_x" name="test_c">'
        '<skipped message="no deck"/></testcase>'
        '<testcase classname="tests.test_x.TestK" name="test_d"/>'
        '</testsuite></testsuites>')
    got = harness.junit_outcomes(xml, "tests/test_x.py")
    assert got["tests/test_x.py::test_a"][0] == "passed"
    assert got["tests/test_x.py::test_b[2-3]"][0] == "failed"
    assert "boom" in got["tests/test_x.py::test_b[2-3]"][1]
    assert got["tests/test_x.py::test_c"] == ("skipped", "no deck")
    assert got["tests/test_x.py::TestK::test_d"][0] == "passed"


# ---------------------------------------------------------------------------
#  …and the OTHER thing that decides whether SLATE can see the device
# ---------------------------------------------------------------------------
# One visible GPU is necessary and not sufficient.  ``liblorrax_ffi.so`` and
# ``liblorrax_ffi_host.so`` both carry NEEDED libslate.so.2 / libblaspp.so.2
# out of DIFFERENT builds, ld.so keys a loaded object by SONAME, and the host
# build's blas::get_device_count() is a compiled-in 0 -- so opening the host
# library first gives every CUDA SLATE handler a device count of ZERO at one
# visible device.  ``src/ffi/common/ffi_loader.py`` is the loader that lost
# that race (a module-scope probe_target(FLAT_K_TARGET, "cpu") in
# tests/test_fft_flat_k_numerics.py, at collection), so it is the copy of the
# rule that has to hold.  distrib_la's own copy is covered by
# services/distrib_la/tests/test_distrib_la_contract.py; this is the twin for
# lorrax's, because a rule enforced in two places with a test in one is a rule
# with a hole in it.

def _record_ffi_loader_open_order(monkeypatch, *, disable_rule=False,
                                  present=("CUDA", "cpu"), cuda_capable=True):
    """``ffi_loader.get_lib('cpu')`` with every native step stubbed; returns
    the platform library paths it dlopened, in order.

    ``cuda_capable`` stands in for the process's platform — the cells that
    own the PREDICATE construct its inputs directly (below); these own the
    ORDER."""
    import ctypes
    import pathlib
    from ffi.common import ffi_loader as F

    opened = []

    class _FakeLib:
        def __getattr__(self, name):
            return _FakeLib()

        def __setattr__(self, name, value):
            pass

    def _fake_locate(platform):
        if platform not in present:
            raise FileNotFoundError(f"no {platform} library in this fixture")
        return pathlib.Path("/fixture") / F._PLATFORMS[platform]["so_name"]

    def _fake_cdll(path, mode=0):
        opened.append(str(path))
        return _FakeLib()

    def _fake_open(path, **kwargs):
        return _fake_cdll(path, mode=ctypes.RTLD_GLOBAL), pathlib.Path(path)

    monkeypatch.setattr(F, "_LIBS", {})
    monkeypatch.setattr(F, "_LIB_PATHS", {})
    monkeypatch.setattr(F, "_CUDA_FIRST_TRIED", False)
    monkeypatch.setattr(F, "_locate_so", _fake_locate)
    monkeypatch.setattr(ctypes, "CDLL", _fake_cdll)
    monkeypatch.setattr(F._native, "open_and_attest", _fake_open)
    monkeypatch.setattr(F, "_set_argtypes", lambda lib, platform: None)
    monkeypatch.setattr(F, "_register_ffi_targets", lambda lib, plat: None)
    monkeypatch.setattr(F, "_process_can_use_cuda", lambda: bool(cuda_capable))
    if disable_rule:
        monkeypatch.setattr(F, "_open_cuda_before_host", lambda: None)

    F.get_lib("cpu")
    return opened


def test_lorraxs_loader_opens_the_cuda_library_before_the_host_one(monkeypatch):
    """CUDA-CAPABLE ARM: the load-order rule, in the loader that lost the race.

    RED ARM: disable ``_open_cuda_before_host`` and only the host library is
    opened — which is the process state that produced
    ``blas::get_device_count()=0``.
    """
    opened = _record_ffi_loader_open_order(monkeypatch, cuda_capable=True)
    assert opened == ["/fixture/liblorrax_ffi.so",
                      "/fixture/liblorrax_ffi_host.so"], opened


def test_the_lorrax_loader_open_order_cell_can_fail(monkeypatch):
    """The FALSE case, constructed."""
    opened = _record_ffi_loader_open_order(monkeypatch, disable_rule=True)
    assert opened == ["/fixture/liblorrax_ffi_host.so"], opened


def test_a_cpu_platform_process_opens_only_the_host_library(monkeypatch):
    """CPU-PLATFORM ARM: B1, in the loader that carries it for lorrax.

    A process whose jax platform is cpu has no CUDA SLATE handler for the
    order to protect, and dlopening the CUDA library there brought a second
    libslate/libblaspp AND a second phdf5 into it: ``tests/test_file_io.py``
    on a CPU-platform Perlmutter leg went 42 passed / 1 skipped at the two
    commits before the rule to three failures and ``Fatal Python error:
    Aborted`` at the commit that added it.
    """
    opened = _record_ffi_loader_open_order(monkeypatch, cuda_capable=False)
    assert opened == ["/fixture/liblorrax_ffi_host.so"], (
        f"a CPU-platform process dlopened {opened} — it must open the host "
        f"library and NOTHING else")


def test_the_lorrax_loader_cpu_platform_cell_can_fail(monkeypatch):
    """The FALSE case: same fixture, capability gate forced TRUE, and the
    CUDA library IS opened first.  Without this twin the cell above would
    stay green on any machine with no CUDA library to find — which is every
    WSL leg, i.e. green for a reason unrelated to the rule."""
    opened = _record_ffi_loader_open_order(monkeypatch, cuda_capable=True)
    assert opened == ["/fixture/liblorrax_ffi.so",
                      "/fixture/liblorrax_ffi_host.so"], opened


def test_a_cpu_only_tree_pays_nothing_for_the_rule(monkeypatch):
    """No CUDA library to locate: the host library still loads and the
    refusal is swallowed, so every CPU-only tree is untouched."""
    opened = _record_ffi_loader_open_order(monkeypatch, present=("cpu",))
    assert opened == ["/fixture/liblorrax_ffi_host.so"], opened


@pytest.mark.parametrize("env,devices,want", [
    ({"JAX_PLATFORMS": "cpu"},      True,  False),   # the leg B1 killed
    ({"JAX_PLATFORMS": "cpu,cuda"}, True,  False),
    ({"JAX_PLATFORMS": "cuda,cpu"}, True,  True),
    ({"JAX_PLATFORMS": "gpu"},      True,  True),
    ({},                            True,  True),    # every `lx test` leg
    ({},                            False, False),   # login node, WSL
    ({"CUDA_VISIBLE_DEVICES": ""},  True,  False),
    ({"CUDA_VISIBLE_DEVICES": "0"}, True,  True),
])
def test_the_lorrax_loader_cuda_capability_gate(monkeypatch, env, devices, want):
    """The predicate, every input constructed.

    Both loaders carry their own copy — the service may not import lorrax —
    so both get their own table, and the two tables are the same table.
    ``jax.default_backend()`` is deliberately NOT the signal: it
    INITIALIZES the XLA backend, so asking it inside a loader call would
    let the loader decide the process's platform instead of reading it.
    """
    from ffi.common import ffi_loader as F

    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(F, "_nvidia_device_visible", lambda: bool(devices))
    assert F._process_can_use_cuda() is want
