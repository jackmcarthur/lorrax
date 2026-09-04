"""P=4 GPU contract for the cross-rank module-compile refusal."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesh_launch                                           # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.procs(4)


def _require_mesh4():
    mode, why = mesh_launch.choose_mode(dict(os.environ))
    if mode == mesh_launch.NONE:
        pytest.skip(f"no four-process launch available here: {why}")
    return mode


def _run(kind: str, tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["LORRAX_JAX_COMPILE_AGREE_TIMEOUT_S"] = "3"
    env["ISDF_JAX_CACHE_DIR"] = ""
    argv = [sys.executable,
            str(REPO_ROOT / "tests" / "_compile_agreement_probe.py"), kind]
    return mesh_launch.run_mesh4(
        argv, cwd=tmp_path, env=env, timeout=30, mode=_require_mesh4())


def _receipts(stdout: str) -> list[dict]:
    prefix = "COMPILE_AGREEMENT_PROBE "
    return [json.loads(line.split(prefix, 1)[1])
            for line in stdout.splitlines() if prefix in line]


def test_identical_modules_pass_and_report_per_module_overhead(tmp_path):
    result = _run("identical", tmp_path)
    assert result.ok, result.blame("identical P=4 modules were refused")
    receipts = _receipts(result.stdout)
    assert len(receipts) == 4, result.blame("not every rank emitted a receipt")
    assert all(item["process_count"] == 4 for item in receipts)
    assert all(item["checks"] >= 1 for item in receipts)
    assert all("--xla_gpu_autotune_level=0" in item["xla_flags"]
               for item in receipts)
    assert all(item["per_module_overhead_s"] >= 0.0 for item in receipts)
    print(json.dumps(receipts, indent=2, sort_keys=True))


def test_rank_divergent_shape_refuses_before_the_deadline(tmp_path):
    result = _run("divergent", tmp_path)
    assert not result.ok, result.blame(
        "rank-divergent P=4 shape unexpectedly compiled")
    output = result.stdout + "\n" + result.stderr
    assert result.wall_s < 10.0, result.blame(
        "rank-divergent compile did not refuse near the 3-second deadline")
    assert "GATE cross_rank_compile_agreement: REFUSED" in output
    assert "compile_agreement_probe" in output
    for rank in range(4):
        assert f"rank {rank}: key=" in output
