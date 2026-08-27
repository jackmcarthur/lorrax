"""THE JAX CACHE CONTRACT — one standing gate for both cache-PITA classes.

    the owner, 2026-08-10: "better gates to make sure jax correctly caches
    like 100% of the time and doesn't cause us PITAs."

THE CONTRACT, IN ONE SENTENCE
-----------------------------
Run a driver's Si smoke deck TWICE at P=4 against a fresh cache directory.
On the SECOND run, on EVERY rank:

    xla_compiles == 0    and    vetoed == 0    and    the set of
    persistent-cache keys this rank asked about is IDENTICAL to every
    other rank's.

WHY THOSE THREE AND NOT ONE
---------------------------
Because the campaign produced exactly two failure classes and no single
number sees both (``~/lorrax_bse_perf_2026-08-08/INDEX.md``):

**Class A — VETO.**  The program CANNOT persist.  JAX refuses to write a
persistent entry for any module carrying a host callback
(``jax/_src/compiler.py::_cache_write``), so the entry is never on disk, is
never in the keylist process 0 advertises, and every rank recompiles it on
every run forever.  ``jit__full_run`` carried one ``jax.debug.callback`` and
it cost 1.87 s of XLA compile per warm BSE run for as long as the gate that
put it there existed (FIX_warmcache.md).  Caught by ARM 2.  The sanctioned
fix is the sink pattern (``solvers/lanczos.py::alpha_herm_sink``): move the
scalars out as jit OUTPUTS and do the check on the host.

**Class B — KEY DIVERGENCE.**  Rank-dependent static args, or shapes derived
from them, produce a DIFFERENT PROGRAM PER RANK.  JAX writes cache entries
from process 0 only, so rank 0 hits its key while its peers name keys that
were never written, miss, and compile.  Asymmetric hit/miss across ranks is
the collective-compile deadlock PRECONDITION (FIX_multislice_cachekey.md).
Caught by ARM 3 — and ONLY by ARM 3, which is the reason this file exists:
four ranks that each compiled a private program report exactly the same
``xla_compiles`` and ``vetoed`` as four ranks that shared one.  The counters
are blind to it by construction; the KEY SET is not.

THE RED TWINS ARE NOT OPTIONAL
------------------------------
A green three-armed gate is worth nothing unless each arm is known to be
able to go red, and — because the two classes have different fixes — unless
the arms are known to go red SEPARATELY.  So:

    ``veto``        must take ARM 2 red and leave ARM 3 GREEN
    ``rankstatic``  must take ARM 3 red
    ``clean``       must be green, or neither twin means anything

All three are real four-process programs run against a real persistent cache
(``tests/_cache_contract_probe.py``), never mocks, so this gate also fails if
jax changes its own rules out from under the contract.

WHY ``procs(4)`` AND NOT ``mesh(4)``
------------------------------------
They are different contracts and the difference is the whole subject of this
file.  ``mesh(n)`` — landed 2026-08-10, a6b87fa9 — widens ONE process to n
DEVICES, in a child when the suite's pin has narrowed this one.  That is the
right instrument for a cell that needs a 2x2 device mesh, and it is the
wrong one here: in an n-device single process ``jax.process_count()`` is 1,
so the compile-cache agreement layer is never installed, only process 0
exists to write entries, and ``ArrayImpl._multi_slice`` — the canonical
class-B site — is never even reached.  Every failure this file exists to
catch is invisible to it.  The landed marker says so itself ("NOT a
substitute for a multi-process P=n leg").

So this file declares ``procs(4)``: n real PROCESSES, launched by
``tests/mesh_launch.py`` (srun inside an allocation; otherwise n local
processes wired by an explicit jax coordinator, one GPU each on a whole
node).  A separate marker rather than a redefinition, because the cells that
passed under ``mesh(n)`` depend on what it means.  Every failure message
names the launch mode it got, so a CPU leg can never be quoted as the GPU
one.

UNDECKED DRIVERS ARE NAMED, NOT SKIPPED SILENTLY
------------------------------------------------
Same convention as the default gate: a driver with no runnable in-tree deck
is skipped with ``fast_gate.UNDECKED``'s own reason, so this file says "the
contract says nothing about ``bse.exciton_bands``" out loud rather than
handing back a quiet green over four of seven drivers.
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast_gate                                            # noqa: E402
import mesh_launch                                          # noqa: E402
from harness import REG, copy_fixture                       # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

#: THE WHOLE FILE NEEDS FOUR PROCESSES — see the docstring for why that is
#: not ``mesh(4)``.  A requirement declaration, not a selector: it lets a
#: launcher pick these cells out (``pytest -m procs``) and it makes the
#: four-GPU rule checkable by reading the marker instead of the body.
pytestmark = pytest.mark.procs(4)

#: Generous, because a driver deck is a driver deck.  A hang inside the
#: contract is the very thing it is looking for, so the timeout must be a
#: real bound and not a guess that turns a deadlock into a flake.
_DECK_TIMEOUT_S = 2400
_PROBE_TIMEOUT_S = 600


def _launch_mode():
    return mesh_launch.choose_mode(dict(__import__("os").environ))


def _require_mesh4():
    mode, why = _launch_mode()
    if mode == mesh_launch.NONE:
        pytest.skip(f"no four-process launch available here: {why}")
    return mode, why


# ---------------------------------------------------------------------------
# THE CONTRACT, per driver
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("driver", fast_gate.DRIVERS)
def test_the_cache_contract_holds_warm_at_p4(driver, tmp_path):
    """Two warm runs of ``driver``'s Si smoke deck; the invariant on run 2."""
    if driver in fast_gate.UNDECKED:
        pytest.skip(f"UNDECKED — {fast_gate.UNDECKED[driver]}")
    stages = fast_gate.CONTRACT_DECKS.get(driver)
    assert stages, (
        f"{driver} is neither UNDECKED nor in fast_gate.CONTRACT_DECKS.  "
        f"A driver that is in neither is a driver this contract silently "
        f"says nothing about, which is the one outcome the roster exists "
        f"to prevent.")

    mode, why = _require_mesh4()
    if mode == mesh_launch.LOCAL and not _decks_enabled():
        pytest.skip(
            "the DECK arms need a real four-process leg; here the only "
            f"launch available is {mesh_launch.LOCAL} ({why}).  A CPU mesh "
            "is fine for device-count LOGIC and never substitutes for the "
            "P=4 leg on a GPU path (AGENT_PREAMBLE, the four-GPU rule), so "
            "the deck arms are opt-in off-cluster: set LX_MESH4_DECKS=1.  "
            "The red twins below run in this mode and are not skipped.")

    deck = stages[0]["deck"]
    run_dir = copy_fixture(REG / deck, tmp_path / deck)
    cache_dir = tmp_path / "jax_cache"

    reports = []
    for stage in stages:
        argv = mesh_launch.python_module_argv(stage["module"], *stage["argv"])
        keydirs = []
        for run in (1, 2):                    # run 1 POPULATES, run 2 is warm
            keydir = tmp_path / f"keys_{stage['stage']}_{run}"
            res = mesh_launch.run_mesh4(
                argv, cwd=run_dir, env=_deck_env(cache_dir),
                timeout=_DECK_TIMEOUT_S, mode=mode, keydump_dir=keydir)
            if not res.ok:
                pytest.fail(res.blame(
                    f"{driver}: stage {stage['stage']!r} run {run} of 2 "
                    f"failed before the contract could be measured"))
            keydirs.append(keydir)
        # THE SECOND RUN IS THE MEASUREMENT.  Entries process 0 writes
        # DURING a run are invisible until the NEXT one (jax_compile_cache
        # §2), so run 1 is expected to compile and veto and says nothing;
        # reading it as a failure is FIX_warmcache.md's ops note 3.
        ok, report = mesh_launch.contract_verdict(
            mesh_launch.read_keydumps(keydirs[1]))
        reports.append(f"[{driver} / stage {stage['stage']}]\n{report}")
        if not ok:
            pytest.fail(
                f"THE CACHE CONTRACT IS BROKEN for {driver}, stage "
                f"{stage['stage']} (launch mode {mode}):\n\n"
                + "\n\n".join(reports))
    print("\n".join(reports))


def _decks_enabled() -> bool:
    import os
    return (os.environ.get("LX_MESH4_DECKS", "") or "").strip() not in (
        "", "0", "false", "no")


def _deck_env(cache_dir) -> dict:
    import os
    env = dict(os.environ)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env["ISDF_JAX_CACHE_DIR"] = str(cache_dir)
    # This is a cache-CONTRACT experiment, not production policy: persist even
    # the tiny clean/red probes so run 2 can falsify warm reuse deterministically.
    env["JAX_ENABLE_COMPILATION_CACHE"] = "1"
    env["JAX_COMPILATION_CACHE_MAX_SIZE"] = "-1"
    env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    env["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"]
                               if env.get("PYTHONPATH") else "")
    return env


# ---------------------------------------------------------------------------
# THE RED TWINS
# ---------------------------------------------------------------------------
def _run_probe(kind: str, tmp_path, mode):
    """Two warm P=4 runs of one probe; the verdict on the second."""
    import os
    env = dict(os.environ)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env["ISDF_JAX_CACHE_DIR"] = str(tmp_path / f"cache_{kind}")
    env["JAX_ENABLE_COMPILATION_CACHE"] = "1"
    env["JAX_COMPILATION_CACHE_MAX_SIZE"] = "-1"
    env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    env["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    argv = [sys.executable,
            str(REPO_ROOT / "tests" / "_cache_contract_probe.py"), kind]
    for run in (1, 2):
        keydir = tmp_path / f"keys_{kind}_{run}"
        res = mesh_launch.run_mesh4(argv, cwd=tmp_path, env=env,
                                    timeout=_PROBE_TIMEOUT_S, mode=mode,
                                    keydump_dir=keydir)
        assert res.ok, res.blame(f"probe {kind!r} run {run} did not complete")
    return mesh_launch.contract_verdict(mesh_launch.read_keydumps(keydir))


def test_the_control_probe_is_green(tmp_path):
    """WITHOUT THIS, NEITHER TWIN MEANS ANYTHING.

    ``clean`` is the same shape as both twins — same two shared jits, one
    extra — with nothing rank-dependent and no callback.  If it is not
    green then the harness, not the tree, is what the twins are measuring.
    """
    mode, _ = _require_mesh4()
    ok, report = _run_probe("clean", tmp_path, mode)
    assert ok, f"the CONTROL probe is not green — the harness is broken:\n{report}"


def test_the_veto_arm_goes_red_on_a_host_callback(tmp_path):
    """RED TWIN for ARM 2 — and a green ARM 3, which is the discrimination.

    A ``jax.debug.callback`` inside the jit makes the module unpersistable
    (class A).  Every rank names the SAME key and every rank misses it, so
    a gate that only compared key sets across ranks would call this
    perfectly healthy.  ARM 2 must fail and ARM 3 must not.
    """
    mode, _ = _require_mesh4()
    ok, report = _run_probe("veto", tmp_path, mode)
    assert not ok, (
        "the VETO red twin came back GREEN.  Either the contract's arm 2 "
        f"stopped working or jax now persists modules with host callbacks:\n"
        f"{report}")
    assert "ARM 2" in report and "FAIL" in report.split("ARM 3")[0], (
        f"the veto twin went red on the WRONG ARM:\n{report}")
    assert "ARM 3 (key set identical across ranks): ok" in report, (
        "the veto twin took the SYMMETRY arm red as well.  It must not: a "
        "host callback is rank-symmetric, and a gate that cannot tell class "
        "A from class B will send the next reader to the wrong fix.\n"
        f"{report}")


def test_the_symmetry_arm_goes_red_on_a_rank_static_jit(tmp_path):
    """RED TWIN for ARM 3 — the class-B precondition, reproduced.

    ``jax.process_index()`` as a STATIC argument is ``jit__multi_slice``'s
    defect in four lines: four ranks, four programs, four keys, one shared
    cache directory only process 0 may write to.
    """
    mode, _ = _require_mesh4()
    ok, report = _run_probe("rankstatic", tmp_path, mode)
    assert not ok, (
        f"the RANK-STATIC red twin came back GREEN — arm 3 is not "
        f"measuring anything:\n{report}")
    assert "ARM 3 (key set identical across ranks): FAIL" in report, (
        f"the rank-static twin did not go red on the SYMMETRY arm:\n{report}")
    assert "jit__rank_static" in report, (
        "arm 3 went red but did not NAME the diverging module.  The module "
        "name is the only actionable half of a cache key; a report of bare "
        f"hashes is a report nobody can act on:\n{report}")


# ---------------------------------------------------------------------------
# The roster cannot drift
# ---------------------------------------------------------------------------
def test_every_driver_is_either_decked_or_named_undecked():
    """No driver may be silently absent from the contract."""
    missing = [d for d in fast_gate.DRIVERS
               if d not in fast_gate.CONTRACT_DECKS
               and d not in fast_gate.UNDECKED]
    assert not missing, (
        f"{missing} are in fast_gate.DRIVERS but in neither CONTRACT_DECKS "
        f"nor UNDECKED, so the cache contract says nothing about them and "
        f"does not say so.")
    both = sorted(set(fast_gate.CONTRACT_DECKS) & set(fast_gate.UNDECKED))
    assert not both, f"{both} are both decked and declared undecked"


def test_the_contract_decks_are_the_smoke_decks():
    """CONTRACT_DECKS and SI_SMOKE must cover the same drivers.

    They are two spellings of one roster — node ids for the default gate,
    launches for this one — and a driver that gained a smoke cell without
    gaining a contract deck is a driver whose cache behaviour stopped being
    gated the day it started being tested.
    """
    assert set(fast_gate.CONTRACT_DECKS) == set(fast_gate.SI_SMOKE), (
        f"CONTRACT_DECKS covers {sorted(fast_gate.CONTRACT_DECKS)} but "
        f"SI_SMOKE covers {sorted(fast_gate.SI_SMOKE)}")


def test_every_contract_deck_exists_on_disk():
    """A deck named in the roster that is not in the tree is a silent skip."""
    for driver, stages in fast_gate.CONTRACT_DECKS.items():
        for stage in stages:
            d = REG / stage["deck"]
            assert d.is_dir(), (
                f"{driver} stage {stage['stage']!r} names deck {d}, which "
                f"does not exist")


def test_the_launch_decision_is_falsifiable():
    """``choose_mode`` is pure given its probes, and these are its answers."""
    yes = lambda _n: "/usr/bin/srun"            # noqa: E731
    no = lambda _n: None                        # noqa: E731
    none = lambda: 0                            # noqa: E731
    M = mesh_launch

    # FOUR REAL DEVICES WIN, always: that is the landing-evidence leg.
    assert M.choose_mode({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, no, none)[0] \
        == M.LOCAL_GPU
    # ...even inside a step, which is exactly where a whole-node leg runs.
    assert M.choose_mode(
        {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "SLURM_JOB_ID": "1",
         "SLURM_STEP_ID": "0"}, yes, none)[0] == M.LOCAL_GPU
    # An allocation, on a login node, with no devices here: srun.
    assert M.choose_mode({"SLURM_JOB_ID": "1"}, yes, none)[0] == M.SRUN
    # Nothing at all: the CPU emulation, which is never landing evidence.
    assert M.choose_mode({}, no, none)[0] == M.LOCAL
    # An allocation but no launcher, and no devices: refuse, do not emulate.
    assert M.choose_mode({"SLURM_JOB_ID": "1"}, no, none)[0] == M.NONE
    # Inside a step with too few devices: a nested srun refuses (LX-NESTED),
    # and saying so is cheaper than discovering it as exit 92 four decks in.
    nested = M.choose_mode(
        {"SLURM_JOB_ID": "1", "SLURM_STEP_ID": "0",
         "CUDA_VISIBLE_DEVICES": "0"}, yes, none)
    assert nested[0] == M.NONE and "LX-NESTED" in nested[1]
    # THE SUBSTITUTION THE FOUR-GPU RULE REFUSES: a CPU emulation must never
    # be handed back where four devices exist.
    assert M.choose_mode({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, yes, none)[0] \
        != M.LOCAL


def test_the_gpu_leg_gives_each_process_exactly_one_device():
    """One GPU per process is the production launch shape.

    Four devices in ONE process is the arrangement this whole contract
    cannot see (``jax.process_count()`` is 1 there), so the split has to be
    checked, not assumed.
    """
    gpus = mesh_launch.visible_gpus({"CUDA_VISIBLE_DEVICES": "2,3,5,7"})
    assert gpus == ["2", "3", "5", "7"]
    assert mesh_launch.visible_gpus({"CUDA_VISIBLE_DEVICES": ""}) == []
    assert mesh_launch.visible_gpus({}, lambda: 4) == ["0", "1", "2", "3"]
    # THE PIN MUST NOT HIDE THE NODE.  conftest pins this process to one
    # device before any cell runs; the pre-pin list is what a procs(4) cell
    # has to see, and reading the pinned value instead is what made the
    # whole contract skip itself on a four-A100 node.
    assert mesh_launch.visible_gpus(
        {"CUDA_VISIBLE_DEVICES": "0",
         "LORRAX_SESSION_DEVICES": "0,1,2,3"}) == ["0", "1", "2", "3"]
    assert mesh_launch.choose_mode(
        {"CUDA_VISIBLE_DEVICES": "0", "LORRAX_SESSION_DEVICES": "0,1,2,3",
         "SLURM_JOB_ID": "1", "SLURM_STEP_ID": "0"},
        lambda _n: None, lambda: 0)[0] == mesh_launch.LOCAL_GPU


def test_the_verdict_function_fails_a_missing_rank():
    """A rank that never reported is the deadlock symptom, not a pass."""
    three = [{"proc_idx": i, "n_proc": 4, "xla_compiles": 0, "vetoed": 0,
              "probes": 1, "hits": 1, "keys": ["k-1"]} for i in range(3)]
    ok, report = mesh_launch.contract_verdict(three)
    assert not ok and "ARM 1" in report and "deadlock" in report


def test_the_verdict_function_fails_a_key_set_difference():
    """Same counters on every rank, different programs.  Arm 3 alone."""
    dumps = [{"proc_idx": i, "n_proc": 4, "xla_compiles": 0, "vetoed": 0,
              "probes": 2, "hits": 2, "keys": ["shared-1", f"jit__x-{i}"]}
             for i in range(4)]
    ok, report = mesh_launch.contract_verdict(dumps)
    assert not ok, "four ranks holding four different programs read as green"
    assert "ARM 2" in report and "ARM 2 (warm" in report
    assert "ARM 3 (key set identical across ranks): FAIL" in report
    assert "jit__x" in report, "the diverging module was not named"


def test_srun_line_is_four_ranks_on_one_node():
    line = mesh_launch.srun_argv(["python", "-m", "gw.gw_jax"])
    assert line[:2] == ["srun", "--overlap"]
    assert "-n" in line and line[line.index("-n") + 1] == "4"
    assert line[-3:] == ["python", "-m", "gw.gw_jax"]


@pytest.mark.parametrize("kind", ["clean", "veto", "rankstatic"])
def test_the_probe_kinds_are_all_reachable(kind):
    """The twins exist and are spelled the way the gate spells them."""
    probe = REPO_ROOT / "tests" / "_cache_contract_probe.py"
    assert probe.is_file()
    assert kind in probe.read_text()
    assert shutil.which(sys.executable) or Path(sys.executable).exists()


def test_the_session_device_list_is_read_and_not_re_captured():
    """ONE capture of the pre-pin device list, and this reads it.

    ``tests/conftest.py`` records it as ``LORRAX_SESSION_DEVICES`` for the
    ``mesh(n)`` runner (``harness.session_devices``).  This file needs the
    same fact for a different reason, and takes THEIRS: a second snapshot of
    one fact is the two-copy disease, and the copy that goes stale is always
    the one nobody is looking at.
    """
    import harness
    assert mesh_launch._SESSION_DEVICES_ENV == harness.SESSION_DEVICES_ENV, (
        "mesh_launch reads a different variable than conftest writes")
    conftest_src = (REPO_ROOT / "tests" / "conftest.py").read_text()
    assert conftest_src.count("SESSION_DEVICES_ENV") >= 1
    assert "LORRAX_TEST_VISIBLE_GPUS" not in conftest_src, (
        "a second pre-pin capture is back in conftest")


def test_procs_and_mesh_are_distinct_contracts():
    """``procs(n)`` must not be read as, or silently become, ``mesh(n)``.

    The landed ``mesh(n)`` gives ONE process n devices.  Every defect this
    file gates is invisible there, because ``jax.process_count()`` is 1 and
    the agreement layer that the whole contract measures is never installed.
    Both markers are registered, separately, with their own definitions.
    """
    markers = (REPO_ROOT / "pyproject.toml").read_text()
    assert '"mesh(n):' in markers and '"procs(n):' in markers, (
        "both markers must be registered separately")
    assert "SINGLE-PROCESS" in markers.split('"mesh(n):')[1][:400]
    procs_def = markers.split('"procs(n):')[1][:600]
    assert "real PROCESSES" in procs_def
    assert "process_count() is 1" in procs_def or "jax.process_count" in procs_def
    # And this module declares the process one.  Asserted on the MARKER
    # OBJECTS and on the AST, never on the file's text: this cell has to
    # name the marker it is refusing, so any substring test for
    # ``pytest.mark.mesh`` matches its own body and fails on a correct file.
    marks = pytestmark if isinstance(pytestmark, list) else [pytestmark]
    assert {m.name for m in marks} == {"procs"}, (
        f"module markers are {[m.name for m in marks]}; the cache contract "
        f"must declare procs(n) and not mesh(n) — an n-device single "
        f"process cannot express rank divergence at all")
    assert marks[0].args == (4,), f"procs({marks[0].args}) — want procs(4)"

    tree = ast.parse(Path(__file__).read_text())
    for node in ast.walk(tree):
        for dec in getattr(node, "decorator_list", []):
            fn = dec.func if isinstance(dec, ast.Call) else dec
            assert not (isinstance(fn, ast.Attribute) and fn.attr == "mesh"), (
                f"{getattr(node, 'name', '?')} is decorated mesh(n); a cell "
                f"in this file needs procs(n)")
