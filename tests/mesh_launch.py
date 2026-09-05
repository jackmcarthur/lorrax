"""Launch a command as FOUR PROCESSES, or say precisely why it cannot.

WHY A FOUR-PROCESS LAUNCH NEEDS ITS OWN MODULE
----------------------------------------------
The cache contract (``tests/test_jax_cache_contract.py``) is about a defect
class that does not exist below two processes: ranks compiling DIFFERENT
programs and therefore holding different persistent-cache keys.  A
single-process run with four local devices — which is what ``--px 2 --py 2``
gives you under plain ``pytest``, and what most of this suite means by "a
2x2 mesh" — cannot express it at all: ``jax.process_count()`` is 1, the
agreement layer is never installed, and ``ArrayImpl._multi_slice`` is never
even reached.  So the contract needs real processes, and the suite had no
way to ask for them.

THE FOUR-GPU RULE AND THIS MODULE
---------------------------------
``AGENT_PREAMBLE.md``: every GPU verification leg runs at P=4, and emulated
CPU meshes are fine for device-count LOGIC but never substitute for the P=4
leg on a real GPU path.  This module implements both halves and keeps them
apart by NAME, never by accident: :data:`LOCAL_GPU` and :data:`SRUN` are
real four-device legs, :data:`LOCAL` is four local CPU processes, and every
result carries which one it was so a report cannot quote a CPU leg as the
GPU one.

THE DECISION, AND WHY IT IS A PURE FUNCTION
-------------------------------------------
:func:`choose_mode` takes an environment dict and a ``which`` probe and
returns ``(mode, why)``.  It touches nothing.  That is the same convention
as ``harness.pin_one_gpu`` and ``fast_gate.drivers_for_path``: a launcher
decision that can only be observed by launching is a decision no one can
falsify, and every one of this campaign's cache legs was lost or misread at
the launcher rather than in the code (FIX_multislice_cachekey.md §5.1).
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The session's pre-pin device list, recorded by ``tests/conftest.py`` for
#: the ``mesh(n)`` runner.  Named here rather than imported so this module
#: stays importable with no pytest session (it is also driven from scripts).
#: ``tests/test_jax_cache_contract.py`` asserts the two spellings agree.
_SESSION_DEVICES_ENV = "LORRAX_SESSION_DEVICES"

#: Four processes under ``srun`` inside an existing Slurm allocation.  The
#: landing-evidence mode: real devices, real NCCL, real one-GPU-per-process.
SRUN = "srun"

#: Four local processes wired together by an explicit jax coordinator, on
#: the CPU backend.  Device-count LOGIC only — never landing evidence for a
#: GPU path (AGENT_PREAMBLE, the four-GPU rule's unit/CPU clause).
LOCAL = "local-cpu"

#: Four local processes, ONE REAL GPU EACH, on a whole node.
#:
#: THIS IS THE LANDING-EVIDENCE MODE, and it exists because the obvious
#: route does not work here: a whole-node step is already an ``srun`` step,
#: so a cell inside it cannot spawn a nested one, and running pytest on a
#: login node instead loses the container the drivers need.  Four processes
#: pinned to four devices of the node the step already holds is the same
#: measurement with none of that: real devices, real NCCL, real
#: one-GPU-per-process, ``jax.process_count() == 4``.  Launch it with
#: ``lx run -N 1 -G 4 -n 1 -- pytest tests/test_jax_cache_contract.py``.
LOCAL_GPU = "local-gpu"

#: No four-process launch is available here.
NONE = "none"

#: Ranks.  Named rather than spelled 4 in six places, because the whole
#: point of the contract is that the number is the same everywhere.
NPROC = 4


def visible_gpus(env: dict, probe=None) -> list:
    """The GPU ordinals this leg may use, in order.

    ``harness.SESSION_DEVICES_ENV`` FIRST, because under pytest the obvious
    source has already been destroyed: ``tests/conftest.py`` pins this
    process to ONE device before any test runs (three separate things need
    that), so ``CUDA_VISIBLE_DEVICES`` reads "0" on a node that granted
    four.  MEASURED on Perlmutter 2026-08-10, step ``lx-Xg4-154342``: the
    whole contract skipped itself with "no four-process launch available
    here" immediately after printing four A100s.

    THE PRE-PIN LIST IS NOT CAPTURED HERE.  ``pytest_configure`` already
    records it as ``LORRAX_SESSION_DEVICES`` for the ``mesh(n)`` runner
    (``harness.session_devices``), and that is the same fact read at the
    same hook — so this reads THEIRS rather than taking a second snapshot.
    Two captures of one fact is the two-copy disease, and the copy that
    goes stale is always the one nobody is looking at.

    Then ``CUDA_VISIBLE_DEVICES`` — inside a Slurm step it is authoritative
    and ``nvidia-smi`` is not (it enumerates the node, not the cgroup).
    """
    stashed = env.get(_SESSION_DEVICES_ENV)
    if stashed is not None:
        return [d for d in stashed.split(",") if d.strip() != ""]
    preset = env.get("CUDA_VISIBLE_DEVICES")
    if preset is not None:
        return [d for d in preset.split(",") if d.strip() != ""]
    if probe is None:
        probe = _probe_nvidia_smi
    return [str(i) for i in range(probe())]


def _probe_nvidia_smi() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=10).stdout
        return len(out.strip().splitlines()) if out.strip() else 0
    except Exception:                                          # noqa: BLE001
        return 0


def choose_mode(env: dict, which=shutil.which, probe=None) -> tuple[str, str]:
    """``(mode, why)`` for the environment ``env``.  Pure given its probes.

    ``LX_MESH4_MODE`` forces the answer so a leg can PIN what it measured
    instead of inheriting it — the mode is quoted in every failure message
    for the same reason.

    ORDER MATTERS AND IS NOT A PREFERENCE.  A real four-device leg wins
    whenever one is available, because a machine that can run the real thing
    must never quietly hand back the emulation.  That substitution is the
    exact shape the four-GPU rule exists to refuse and it would be invisible
    in a green report.
    """
    forced = (env.get("LX_MESH4_MODE") or "").strip()
    if forced:
        if forced not in (SRUN, LOCAL, LOCAL_GPU, NONE):
            raise ValueError(
                f"LX_MESH4_MODE={forced!r} is not one of "
                f"{[SRUN, LOCAL, LOCAL_GPU, NONE]}")
        return forced, f"LX_MESH4_MODE={forced}"

    gpus = visible_gpus(env, probe)
    if len(gpus) >= NPROC:
        return LOCAL_GPU, (
            f"{len(gpus)} GPUs visible ({','.join(gpus[:NPROC])}) — "
            f"{NPROC} processes, one real device each")

    job = (env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID") or "").strip()
    if job and which("srun"):
        # Inside a STEP already (SLURM_STEP_ID set by srun itself) a nested
        # srun is the launcher's `LX-NESTED` refusal, not a launch.  Only
        # reachable with FEWER than four GPUs visible, since a whole-node
        # step took the branch above.
        if (env.get("SLURM_STEP_ID") or "").strip():
            return NONE, (
                f"inside an srun STEP with only {len(gpus)} GPU(s) visible: "
                f"a nested srun refuses (LX-NESTED) and there are not "
                f"{NPROC} devices here.  Take the whole node — "
                f"`lx run -N 1 -G 4 -n 1 -- pytest ...`")
        return SRUN, f"SLURM_JOB_ID={job} and srun is on PATH"
    if job:
        return NONE, f"SLURM_JOB_ID={job} but no srun on PATH"
    return LOCAL, "no Slurm allocation — four local processes on the CPU backend"


def srun_argv(argv, *, gpus_per_node: int = NPROC) -> list:
    """The ``srun`` line for a four-rank, one-node step.

    ``--overlap`` because the caller is normally already holding the
    allocation through some other step; without it Slurm queues this behind
    itself and the leg reads as a hang.
    """
    return [
        "srun", "--overlap", "-N", "1", "-n", str(NPROC),
        f"--gres=gpu:{gpus_per_node}", "--kill-on-bad-exit=1",
        *argv,
    ]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class Mesh4Result:
    """What a four-process launch did.  Carries its MODE, always."""
    mode: str
    argv: list
    returncodes: list = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    per_rank: list = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.returncodes) and all(rc == 0 for rc in self.returncodes)

    def blame(self, what: str) -> str:
        """A failure message that carries the evidence, not a bare assert."""
        return (
            f"{what}\n"
            f"  mode      : {self.mode}\n"
            f"  argv      : {' '.join(shlex.quote(a) for a in self.argv)}\n"
            f"  returncode: {self.returncodes}\n"
            f"  wall      : {self.wall_s:.1f}s\n"
            f"--- stdout (tail) ---\n{self.stdout[-6000:]}\n"
            f"--- stderr (tail) ---\n{self.stderr[-6000:]}\n")


def run_mesh4(argv, *, cwd, env=None, timeout=1800, mode=None,
              keydump_dir=None) -> Mesh4Result:
    """Run ``argv`` as :data:`NPROC` processes; return a :class:`Mesh4Result`.

    ``keydump_dir`` is passed through as ``LORRAX_JAX_CACHE_KEYDUMP`` so the
    caller does not have to remember the variable's name in two places.
    """
    base = dict(os.environ if env is None else env)
    if keydump_dir is not None:
        base["LORRAX_JAX_CACHE_KEYDUMP"] = str(keydump_dir)
    mode = mode or choose_mode(base)[0]
    t0 = time.monotonic()

    if mode == SRUN:
        line = srun_argv(list(argv))
        proc = subprocess.run(line, cwd=str(cwd), env=base, timeout=timeout,
                              capture_output=True, text=True, check=False)
        return Mesh4Result(mode=mode, argv=line, returncodes=[proc.returncode],
                           stdout=proc.stdout, stderr=proc.stderr,
                           wall_s=time.monotonic() - t0)

    if mode in (LOCAL, LOCAL_GPU):
        port = _free_port()
        gpus = visible_gpus(base) if mode == LOCAL_GPU else []
        procs, outs = [], []
        for i in range(NPROC):
            renv = dict(base)
            # A pytest parent has already initialized its own single-process
            # JAX runtime. Its idempotence sentinel is process-local state
            # encoded in the environment and must not cross this exec
            # boundary: these children are a new four-process world.
            renv.pop("_LORRAX_JAX_DISTRIBUTED_DONE", None)
            renv["JAX_PROCESS_COUNT"] = str(NPROC)
            renv["JAX_PROCESS_INDEX"] = str(i)
            renv["JAX_COORDINATOR_ADDRESS"] = f"127.0.0.1:{port}"
            if mode == LOCAL_GPU:
                # ONE DEVICE PER PROCESS -- the production launch shape, and
                # what makes this a P=4 GPU leg rather than a four-device
                # single process.
                renv["CUDA_VISIBLE_DEVICES"] = gpus[i]
                renv.pop("JAX_PLATFORMS", None)
                renv.pop("JAX_PLATFORM_NAME", None)
            else:
                renv["JAX_PLATFORMS"] = "cpu"
                renv["JAX_PLATFORM_NAME"] = "cpu"
            renv.setdefault("JAX_ENABLE_X64", "1")
            renv.setdefault("PYTHONUNBUFFERED", "1")
            procs.append(subprocess.Popen(
                list(argv), cwd=str(cwd), env=renv, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        deadline = time.monotonic() + timeout
        for i, p in enumerate(procs):
            left = max(1.0, deadline - time.monotonic())
            try:
                o, e = p.communicate(timeout=left)
            except subprocess.TimeoutExpired:
                p.kill()
                o, e = p.communicate()
                e = (e or "") + f"\n[mesh_launch] rank {i} TIMED OUT\n"
            outs.append((p.returncode, o or "", e or ""))
        return Mesh4Result(
            mode=mode, argv=list(argv),
            returncodes=[rc for rc, _, _ in outs],
            stdout="".join(f"\n===== rank {i} stdout =====\n{o}"
                           for i, (_, o, _) in enumerate(outs)),
            stderr="".join(f"\n===== rank {i} stderr =====\n{e}"
                           for i, (_, _, e) in enumerate(outs)),
            per_rank=outs, wall_s=time.monotonic() - t0)

    return Mesh4Result(mode=NONE, argv=list(argv), returncodes=[],
                       wall_s=time.monotonic() - t0)


# ---------------------------------------------------------------------------
# Reading the key dumps back
# ---------------------------------------------------------------------------
def read_keydumps(keydump_dir) -> list:
    """Every rank's dump under ``keydump_dir``, ordered by ``proc_idx``."""
    out = []
    for p in sorted(Path(keydump_dir).glob("rank*_of*.json")):
        out.append(json.loads(p.read_text()))
    return sorted(out, key=lambda d: d.get("proc_idx", 0))


def contract_verdict(dumps, *, expect_ranks: int = NPROC) -> tuple[bool, str]:
    """THE INVARIANT, as one pure function over the dumps.

    Three arms, and the third is the one no counter can express:

    1. every rank present;
    2. ``xla_compiles == 0`` and ``vetoed == 0`` on EVERY rank;
    3. the KEY SET is identical across ranks.

    Returns ``(ok, report)``; ``report`` is what the gate prints either way,
    so a green run leaves the same evidence a red one does.
    """
    lines = []
    ok = True

    if len(dumps) != expect_ranks:
        got = sorted(d.get("proc_idx") for d in dumps)
        return False, (
            f"ARM 1 (all ranks reported): FAIL — {len(dumps)} dump(s) for "
            f"{expect_ranks} ranks; proc_idx present = {got}.  A missing "
            f"dump is a rank that never reached exit, which is the deadlock "
            f"symptom itself — do not read it as a pass.")
    lines.append(f"ARM 1 (all {expect_ranks} ranks reported): ok")

    bad = [(d["proc_idx"], d["xla_compiles"], d["vetoed"]) for d in dumps
           if d.get("xla_compiles", -1) != 0 or d.get("vetoed", -1) != 0]
    if bad:
        ok = False
        lines.append(
            "ARM 2 (warm: xla_compiles=0 vetoed=0 on every rank): FAIL — "
            + ", ".join(f"rank {r}: xla_compiles={c} vetoed={v}"
                        for r, c, v in bad))
    else:
        lines.append("ARM 2 (warm: xla_compiles=0 vetoed=0 on every rank): ok "
                     + ", ".join(f"r{d['proc_idx']}:probes={d['probes']}"
                                 f"/hits={d['hits']}" for d in dumps))

    sets = [set(d.get("keys", ())) for d in dumps]
    union = set().union(*sets) if sets else set()
    common = set.intersection(*sets) if sets else set()
    if union != common:
        ok = False
        lines.append(
            f"ARM 3 (key set identical across ranks): FAIL — "
            f"{len(union - common)} key(s) are not held by every rank.")
        for d, ks in zip(dumps, sets):
            extra = sorted(ks - common)
            if extra:
                lines.append(f"    rank {d['proc_idx']} alone holds "
                             f"{len(extra)}: {extra[:6]}")
        # The module NAME is the actionable half of a key; the digest is
        # not.  Name the modules that diverged so the reader has a grep
        # target rather than a wall of hashes.
        mods = sorted({k.rsplit("-", 1)[0] for k in (union - common)})
        lines.append(f"    diverging module(s): {mods}")
    else:
        lines.append(f"ARM 3 (key set identical across ranks): ok "
                     f"({len(common)} keys, all held by all {expect_ranks})")

    return ok, "\n".join(lines)


def python_module_argv(module: str, *args) -> list:
    """``[python, -m, module, *args]`` with THIS interpreter."""
    return [sys.executable, "-m", module, *[str(a) for a in args]]
