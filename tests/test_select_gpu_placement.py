"""Which device ``select_gpu.sh`` hands a rank.

The script is the last thing between ``srun`` and the user command on every
GPU leg the fleet runs, and until 2026-08-09 its whole body was

    export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}

— the rank's index on the node used directly as a device ordinal.  That is
correct only when the devices the step may touch are numbered 0..n-1 in
local-rank order.  Two things follow, and both are pinned below.

FIRST, the case it got wrong.  When Slurm assigns a step the node's GPUs
2 and 3, the old line pinned ranks 0 and 1 to devices 0 and 1 — devices the
step was not given.  ``test_assignment_not_zero_based`` is the cell that
fails on the old script.

SECOND, and this is the one that cost legs: two concurrent ONE-rank steps
in one allocation are both assigned the same physical GPU.  That was
measured on Perlmutter on 2026-08-09 every way the step can be asked for
(``--overlap`` and not; ``--gres=gpu:1``, ``--gpus=1``,
``--gpus-per-task=1``), and it means the fleet stacked every co-tenant
one-GPU leg onto physical device 0 while the other three idled — the
``Allocator (GPU_0_bfc) ran out of memory`` / ``RESOURCE_EXHAUSTED``
signature in that day's logs, at ``4/16 GPUs free``.  NOTHING READABLE FROM
INSIDE THE STEP distinguishes those legs: Slurm hands both the identical
``SLURM_LOCALID=0``, ``SLURM_STEP_GPUS=0``, ``CUDA_VISIBLE_DEVICES=0``.
Only the scheduler that placed them knows, so ``LORRAX_GPU_DEVICE`` is how
it says so, and the placement cells below are what that hook has to do.

The derivation cells double as the preservation proof: on the 4-rank,
4-GPU step that production actually runs, every rank gets exactly the
device the old line gave it.
"""
from __future__ import annotations

import os
import subprocess

import pytest


SCRIPT = os.path.join(os.path.dirname(__file__), "..", "src", "ffi", "cpp",
                      "select_gpu.sh")


def _select(**env):
    """Run the script with a bare environment and report what it exported.

    Returns ``(returncode, cuda_visible_devices, stderr)``.  The payload is
    a shell that echoes the variable, so this observes what the EXEC'D
    PROCESS sees — the thing the script exists to set — rather than the
    script's own idea of it.
    """
    base = {"LORRAX_SELECT_GPU_QUIET": "1", "PATH": os.environ.get("PATH", "")}
    base.update({k: str(v) for k, v in env.items()})
    proc = subprocess.run(
        ["bash", SCRIPT, "bash", "-c", 'printf "%s" "$CUDA_VISIBLE_DEVICES"'],
        env=base, capture_output=True, text=True, timeout=60)
    return proc.returncode, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# Preservation: the multi-rank step production runs must not move
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("localid", [0, 1, 2, 3])
def test_four_rank_four_gpu_step_is_unchanged(localid):
    """The measured Perlmutter multi-rank step: Slurm assigns 0,1,2,3 and
    each rank must still get its own index, exactly as the old line did."""
    rc, cvd, _ = _select(SLURM_LOCALID=localid,
                         CUDA_VISIBLE_DEVICES="0,1,2,3",
                         SLURM_STEP_GPUS="0,1,2,3")
    assert rc == 0
    assert cvd == str(localid)


def test_one_rank_one_gpu_step_is_unchanged():
    """The measured one-rank step: Slurm assigns a single device and calls
    it 0 inside the step, so the answer is 0 — as before."""
    rc, cvd, _ = _select(SLURM_LOCALID=0, CUDA_VISIBLE_DEVICES="0",
                         SLURM_STEP_GPUS="0")
    assert (rc, cvd) == (0, "0")


# ---------------------------------------------------------------------------
# The derivation the local rank id got wrong
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("localid,expect", [(0, "2"), (1, "3")])
def test_assignment_not_zero_based(localid, expect):
    """RED on the old script.  Given the node's GPUs 2 and 3, rank r must
    take the r-th ASSIGNED device, not device r."""
    rc, cvd, _ = _select(SLURM_LOCALID=localid,
                         CUDA_VISIBLE_DEVICES="2,3",
                         SLURM_STEP_GPUS="2,3")
    assert (rc, cvd) == (0, expect)


def test_cuda_visible_devices_wins_over_the_physical_list():
    """Under cgroup containment Slurm's CUDA_VISIBLE_DEVICES is in the
    numbering the CUDA runtime in this step will use, while
    SLURM_STEP_GPUS is in the node's physical numbering.  Indexing the
    physical list would name a device the step cannot open."""
    rc, cvd, _ = _select(SLURM_LOCALID=1, CUDA_VISIBLE_DEVICES="0,1",
                         SLURM_STEP_GPUS="2,3")
    assert (rc, cvd) == (0, "1")


def test_falls_back_to_the_physical_list_when_cuda_var_is_unset():
    rc, cvd, _ = _select(SLURM_LOCALID=1, SLURM_STEP_GPUS="4,5")
    assert (rc, cvd) == (0, "5")


def test_falls_back_to_job_gpus_when_the_step_list_is_unset():
    rc, cvd, _ = _select(SLURM_LOCALID=2, SLURM_JOB_GPUS="0,1,2,3")
    assert (rc, cvd) == (0, "2")


def test_uuid_assignments_survive():
    """Slurm can name devices by UUID; the list is indexed, not parsed."""
    rc, cvd, _ = _select(SLURM_LOCALID=1,
                         CUDA_VISIBLE_DEVICES="GPU-aaa,GPU-bbb")
    assert (rc, cvd) == (0, "GPU-bbb")


# ---------------------------------------------------------------------------
# Unset: the historical behaviour is the last resort, not the first guess
# ---------------------------------------------------------------------------

def test_no_slurm_assignment_at_all_falls_back_to_local_rank():
    rc, cvd, _ = _select(SLURM_LOCALID=3)
    assert (rc, cvd) == (0, "3")


def test_no_slurm_environment_at_all():
    rc, cvd, _ = _select()
    assert (rc, cvd) == (0, "0")


# ---------------------------------------------------------------------------
# Placement: the hook that de-conflicts co-tenant one-rank legs
# ---------------------------------------------------------------------------

def test_explicit_placement_beats_the_derivation():
    """Two concurrent one-rank legs are INDISTINGUISHABLE from inside the
    step — identical localid and identical Slurm assignment.  Told apart,
    they must land apart."""
    envs = dict(SLURM_LOCALID=0, CUDA_VISIBLE_DEVICES="0,1,2,3",
                SLURM_STEP_GPUS="0,1,2,3")
    leg_a = _select(LORRAX_GPU_DEVICE="0", **envs)
    leg_b = _select(LORRAX_GPU_DEVICE="2", **envs)
    assert leg_a[:2] == (0, "0")
    assert leg_b[:2] == (0, "2")
    assert leg_a[1] != leg_b[1]


def test_explicit_placement_is_indexed_by_rank_for_multi_rank_steps():
    for localid, expect in ((0, "1"), (1, "3")):
        rc, cvd, _ = _select(SLURM_LOCALID=localid, LORRAX_GPU_DEVICE="1,3",
                             CUDA_VISIBLE_DEVICES="0,1,2,3")
        assert (rc, cvd) == (0, expect)


def test_short_explicit_placement_refuses_rather_than_sharing():
    """The two ways to paper over a launcher bug here — reuse the last
    entry, or fall through to the derivation — both end in two ranks
    quietly sharing a device, which is the failure this file is about."""
    rc, _, err = _select(SLURM_LOCALID=2, LORRAX_GPU_DEVICE="0,1")
    assert rc == 2
    assert "LORRAX_GPU_DEVICE" in err and "local rank 2" in err


# ---------------------------------------------------------------------------
# The wrapper contract itself
# ---------------------------------------------------------------------------

def test_it_execs_its_arguments_and_passes_them_through():
    proc = subprocess.run(
        ["bash", SCRIPT, "bash", "-c", 'printf "%s|%s" "$0" "$1"',
         "arg-zero", "arg-one"],
        env={"LORRAX_SELECT_GPU_QUIET": "1", "SLURM_LOCALID": "0",
             "PATH": os.environ.get("PATH", "")},
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0
    assert proc.stdout == "arg-zero|arg-one"


def test_it_announces_its_choice_unless_silenced():
    """Placement being invisible in the logs is a large part of why the
    device-0 stacking ran as long as it did."""
    proc = subprocess.run(
        ["bash", SCRIPT, "true"],
        env={"SLURM_LOCALID": "1", "CUDA_VISIBLE_DEVICES": "2,3",
             "PATH": os.environ.get("PATH", "")},
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0
    assert "select_gpu" in proc.stderr
    assert "CUDA_VISIBLE_DEVICES=3" in proc.stderr


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
