"""P=4 GPU contract for ``file_io.host_tile_store`` (child: _host_tile_store_p4.py).

Landing command (one pytest per rank):

    lx run -N 1 -G 4 -n 4 python3 -m pytest tests/test_host_tile_store_p4.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest

import mesh_launch
from core import rank_session

pytestmark = pytest.mark.procs(4)
CHILD = "tests/_host_tile_store_p4.py"
N_CELLS = 9


def test_host_tile_store_p4_round_trip_and_red_twins():
    if rank_session._resolve_proc_count() == 4:
        shared = None
        if rank_session._resolve_proc_id() == 0:
            rank_session.ROOT.mkdir(exist_ok=True)
            shared = tempfile.mkdtemp(prefix="host_tile_store-",
                                      dir=rank_session.ROOT)
        shared = rank_session.exchange(shared)[0]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(mesh_launch.REPO_ROOT / "src") + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        result = subprocess.run(
            [sys.executable, CHILD, "--dir", shared], cwd=mesh_launch.REPO_ROOT,
            env=env, capture_output=True, text=True, timeout=300, check=False)
        result = rank_session.completed(result)
        assert f"host_tile_store P4: {N_CELLS} cells ran, 0 failures" in (
            result.stdout), result.stdout[-4000:]
        return
    mode, why = mesh_launch.choose_mode(dict(os.environ))
    if mode in (mesh_launch.NONE, mesh_launch.LOCAL):
        pytest.skip(f"real four-GPU process launch unavailable: {why}")
    with tempfile.TemporaryDirectory(dir=mesh_launch.REPO_ROOT) as shared:
        result = mesh_launch.run_mesh4(
            [sys.executable, CHILD, "--dir", shared],
            cwd=mesh_launch.REPO_ROOT, mode=mode, timeout=300)
    assert result.ok, result.blame("host_tile_store P4 contract failed")
    assert f"host_tile_store P4: {N_CELLS} cells ran, 0 failures" in result.stdout
