"""Actual spinor-WFN regression for operator-only Galerkin continuation."""
from pathlib import Path
import json
import subprocess
import sys

from tests.hsuite import rank_session

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "hsuite" / "fixture"
CHECK = Path(__file__).with_name("galerkin_selected_source_check.py")


def test_selected_basis_source_matches_retained_source():
    def prepare(_source, target):
        target.mkdir(parents=True)
        return target

    out = rank_session.stage(FIXTURE, prepare)
    result = subprocess.run(
        [sys.executable, str(CHECK), "--wfn", str(FIXTURE / "WFN.h5"),
         "--output", str(out / "parity.json")],
        text=True, capture_output=True, check=False)
    rank_session.completed(result)
    for signature in ("Traceback (most recent call last)",
                      "RESOURCE_EXHAUSTED", "CUDA_ERROR", "MPI_Abort"):
        assert signature not in result.stdout + result.stderr
    rows = json.loads((out / "parity.json").read_text())
    assert [row["nspinor"] for row in rows] == [2, 4]
    for row in rows:
        assert row["basis_relative_error"] < 5e-13
        assert row["operator_relative_error"] < 5e-12
        assert row["metric_relative_error"] < 5e-12
        assert row["source_calls"]["full"] == len(row["ranges"])
        assert row["source_calls"]["selected"] > 0
