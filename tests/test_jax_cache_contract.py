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

WHY P=4 AND NOT "a 2x2 mesh"
----------------------------
``--px 2 --py 2`` under plain pytest is ONE process with four local devices.
``jax.process_count()`` is 1 there, the agreement layer is never installed,
and ``ArrayImpl._multi_slice`` — the canonical class-B site — is never even
reached.  The defect class does not exist below two PROCESSES, so the gate
launches real ones (``tests/mesh_launch.py``), and it says which mode it got
in every failure message so a CPU leg can never be quoted as the GPU one.

UNDECKED DRIVERS ARE NAMED, NOT SKIPPED SILENTLY
------------------------------------------------
Same convention as the default gate: a driver with no runnable in-tree deck
is skipped with ``fast_gate.UNDECKED``'s own reason, so this file says "the
contract says nothing about ``bse.exciton_bands``" out loud rather than
handing back a quiet green over four of seven drivers.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast_gate                                            # noqa: E402
import mesh_launch                                          # noqa: E402
from harness import REG, copy_fixture                       # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

#: THE WHOLE FILE NEEDS FOUR PROCESSES.  ``mesh(4)`` is a requirement
#: declaration, not a selector: it lets a launcher pick these cells out
#: (``pytest -m "mesh"``) and it makes the four-GPU rule checkable by
#: reading the marker instead of reading the body.
pytestmark = pytest.mark.mesh(4)

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
    env["ISDF_JAX_CACHE_DIR"] = str(cache_dir)
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
    env["ISDF_JAX_CACHE_DIR"] = str(tmp_path / f"cache_{kind}")
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
    """``choose_mode`` is a pure function and these are its three answers."""
    which_yes = lambda _n: "/usr/bin/srun"      # noqa: E731
    which_no = lambda _n: None                  # noqa: E731
    assert mesh_launch.choose_mode({"SLURM_JOB_ID": "1"}, which_yes)[0] \
        == mesh_launch.SRUN
    assert mesh_launch.choose_mode({}, which_no)[0] == mesh_launch.LOCAL
    assert mesh_launch.choose_mode({"SLURM_JOB_ID": "1"}, which_no)[0] \
        == mesh_launch.NONE
    # Inside a STEP already: a nested srun refuses (LX-NESTED), and saying
    # so here is cheaper than discovering it as exit 92 four decks in.
    nested = mesh_launch.choose_mode(
        {"SLURM_JOB_ID": "1", "SLURM_STEP_ID": "0"}, which_yes)
    assert nested[0] == mesh_launch.NONE and "NESTED" in nested[1]
    # srun WINS when it is available: the emulation must never quietly
    # stand in for the real thing.
    assert mesh_launch.choose_mode(
        {"SLURM_JOB_ID": "7"}, which_yes)[0] == mesh_launch.SRUN


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
