"""Child-launching gates for real 2x2 contracts: dense linalg and SlabIO."""
from __future__ import annotations

import os
import sys
import subprocess

import pytest

import mesh_launch
from core import rank_session


@pytest.mark.procs(4)
def test_real_p4_distrib_la_matrix():
    if rank_session._resolve_proc_count() == 4:
        result = subprocess.run(
            [sys.executable, "tests/core/distrib_la_p4.py", "--mesh", "2x2",
             "--dtypes", "complex128"], cwd=mesh_launch.REPO_ROOT,
            capture_output=True, text=True, timeout=150, check=False,
        )
        result = rank_session.completed(result)
        assert "done: 10 cells ran, 0 failures" in result.stdout
        return
    mode, why = mesh_launch.choose_mode(dict(os.environ))
    if mode in (mesh_launch.NONE, mesh_launch.LOCAL):
        pytest.skip(f"real four-GPU process launch unavailable: {why}")

    root = mesh_launch.REPO_ROOT
    command = (
        sys.executable,
        "tests/core/distrib_la_p4.py",
        "--mesh", "2x2", "--dtypes", "complex128",
    )
    result = mesh_launch.run_mesh4(
        list(command), cwd=root, mode=mode, timeout=150,
    )
    assert result.ok, result.blame("core P4 distrib_la matrix failed")
    assert "done: 10 cells ran, 0 failures" in result.stdout


@pytest.mark.procs(4)
def test_real_p4_slab_io_read_after_failed_write(tmp_path):
    """A write that fails on one rank refuses the next read on every rank."""
    script = "tests/core/slab_io_read_after_write_p4.py"
    marker = "done: read_after_write refused on every rank"
    if rank_session._resolve_proc_count() == 4:
        out = rank_session.ROOT / "slab_io_read_after_write-{}-{}".format(
            os.environ["SLURM_JOB_ID"], os.environ["SLURM_STEP_ID"])
        result = subprocess.run(
            [sys.executable, script, str(out)], cwd=mesh_launch.REPO_ROOT,
            capture_output=True, text=True, timeout=150, check=False,
        )
        result = rank_session.completed(result)
        assert marker in result.stdout
        return
    mode, why = mesh_launch.choose_mode(dict(os.environ))
    if mode in (mesh_launch.NONE, mesh_launch.LOCAL):
        pytest.skip(f"real four-GPU process launch unavailable: {why}")
    result = mesh_launch.run_mesh4(
        [sys.executable, script, str(tmp_path / "out")],
        cwd=mesh_launch.REPO_ROOT, mode=mode, timeout=150,
    )
    assert result.ok, result.blame("core P4 SlabIO read-after-write failed")
    assert marker in result.stdout


@pytest.mark.procs(4)
def test_real_p4_slab_io_many_writable_handles(tmp_path):
    """Three handles' queued collective writes interleave without deadlock.

    Red twin: on a tree whose handles each own a writer thread, the child
    hangs and ``timeout`` fails this cell instead of the job.
    """
    script = "tests/core/slab_io_many_writers_p4.py"
    marker = "read back bit-exact"
    if rank_session._resolve_proc_count() == 4:
        out = rank_session.ROOT / "slab_io_many_writers-{}-{}".format(
            os.environ["SLURM_JOB_ID"], os.environ["SLURM_STEP_ID"])
        result = subprocess.run(
            [sys.executable, script, str(out)], cwd=mesh_launch.REPO_ROOT,
            capture_output=True, text=True, timeout=150, check=False,
        )
        result = rank_session.completed(result)
        assert marker in result.stdout
        return
    mode, why = mesh_launch.choose_mode(dict(os.environ))
    if mode in (mesh_launch.NONE, mesh_launch.LOCAL):
        pytest.skip(f"real four-GPU process launch unavailable: {why}")
    result = mesh_launch.run_mesh4(
        [sys.executable, script, str(tmp_path / "out")],
        cwd=mesh_launch.REPO_ROOT, mode=mode, timeout=150,
    )
    assert result.ok, result.blame("core P4 SlabIO many-writers failed")
    assert marker in result.stdout
