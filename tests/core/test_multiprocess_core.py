"""One child-launching gate for the real 2x2 dense-linalg contracts."""
from __future__ import annotations

import os
import sys

import pytest

import mesh_launch


@pytest.mark.procs(4)
def test_real_p4_distrib_la_matrix():
    mode, why = mesh_launch.choose_mode(dict(os.environ))
    if mode in (mesh_launch.NONE, mesh_launch.LOCAL):
        pytest.skip(f"real four-GPU process launch unavailable: {why}")

    root = mesh_launch.REPO_ROOT
    command = (
        sys.executable,
        "services/distrib_la/tests/test_distrib_la_multiproc.py",
        "--mesh", "2x2", "--dtypes", "complex128",
    )
    result = mesh_launch.run_mesh4(
        list(command), cwd=root, mode=mode, timeout=150,
    )
    assert result.ok, result.blame("core P4 distrib_la matrix failed")
    assert "done: 10 cells ran, 0 failures" in result.stdout
