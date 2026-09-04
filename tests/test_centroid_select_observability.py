"""Refusal and phase-receipt contracts for centroid pruning."""
from __future__ import annotations

import os
import subprocess
import sys


def test_select_phase_receipts_name_lower_compile_and_execute(monkeypatch):
    from centroid import pivoted_cholesky as pc

    class _Compiled:
        def __call__(self, *operands):
            return operands[0]

    class _Lowered:
        def compile(self):
            return _Compiled()

    class _Step:
        def lower(self, *operands):
            return _Lowered()

    lines = []
    monkeypatch.setattr(pc.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(pc, "process_rank", lambda: 0)
    got = pc._run_bounded_select(
        _Step(), (17,), time_budget_s=2.0,
        n_candidates=1028, n_groups=25, point_budget=800,
        print_fn=lines.append, start_progress=lambda: None)
    assert got == 17
    for phase in ("lowering", "compilation", "execution"):
        assert any(
            f"phase={phase} state=start" in line for line in lines), lines
        assert any(
            f"phase={phase} state=done" in line for line in lines), lines
    assert "candidates=1028 groups=25 point_budget=800" in lines[0]


def test_select_time_budget_refuses_with_the_stuck_phase():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = r'''
import time
from centroid import pivoted_cholesky as pc

class Step:
    def lower(self, *operands):
        time.sleep(5.0)

pc.process_rank = lambda: 0
pc._run_bounded_select(
    Step(), (1,), time_budget_s=0.05,
    n_candidates=1028, n_groups=25, point_budget=800,
    print_fn=lambda text: print(text, flush=True),
    start_progress=lambda: None)
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(root, "src") + os.pathsep + env.get(
        "PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", code], env=env,
        capture_output=True, text=True, timeout=5)
    combined = out.stdout + out.stderr
    assert out.returncode == 124, combined
    assert "GATE centroid_select_time_budget" in combined
    assert "phase=lowering" in combined
    assert "candidates=1028 groups=25 point_budget=800" in combined


def test_kmeans_cli_declares_a_finite_select_budget():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(root, "src") + os.pathsep + env.get(
        "PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-m", "centroid.kmeans_cli", "--help"],
        env=env, capture_output=True, text=True, timeout=5)
    assert out.returncode == 0, out.stderr
    assert "--prune-time-budget-seconds" in out.stdout
    assert "default 900 s" in out.stdout
