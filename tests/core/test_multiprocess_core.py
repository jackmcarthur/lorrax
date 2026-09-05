"""One child-launching gate for the real 2x2 dense-linalg contracts."""
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
