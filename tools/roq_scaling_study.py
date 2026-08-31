"""Measure measure-adapted ROQ rank as a delivery interval grows.

This is an offline study driver, not a runtime planner.  It freezes the
DEV-80 measure construction and reuses one weighted snapshot eigensolve for
an ascending integer-rank scan.  Each accepted rank is scored only on the
independent 50-bin validation lattice.

The contour constants come from the Na ROQ anchor recorded by source commit
1426d9f4: conduction uses the real-time resonant contour (260 Ry^-1, rounded
to 20 eV^-1 here), while valence uses -58 degrees and 27 Ry^-1 (rounded to
2 eV^-1).  They are fixed across widths and seeds; this study does not tune
them after seeing a result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg as la


ROOT = Path(__file__).resolve().parents[1]
MINIMAX_SRC = ROOT / "services/minimax/src"
sys.path.insert(0, str(MINIMAX_SRC))

from minimax.reciprocal_fit import (  # noqa: E402
    ReciprocalMeasureProblem,
    delivered_error,
    rule_amplification,
    solve_fixed_time_weights_fast,
)
from minimax.roq_fit import (  # noqa: E402
    RoqGroup,
    _candidates,
    _weighted_subspace,
)


DEFAULT_STUDY = Path(
    "/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/"
    "runs/DEV/80_minimax_delivered_error_toy_20260828"
)
DEFAULT_CATALOG = Path(
    "/pscratch/sd/j/jackm/wt_causaltables_2026-08-31/"
    "services/minimax/src/minimax/minimax_assets/catalog.json"
)
ETA_EV = 0.25
WIDTHS_EV = (2.5, 5.0, 10.0, 20.0)
# These are the production-geometry crossing radii requested in the brief,
# not width / eta.  The toy frequency grids scale with the widths above.
A_OVER_ETA = (12.0, 24.0, 47.0, 94.0)
TARGETS = (1.0e-3, 1.0e-4)
TABLE_ERROR_FOR_TARGET = {1.0e-3: 1.0e-4, 1.0e-4: 1.0e-5}
CONTOURS = {
    "cond": {"angle_deg": 0.0, "horizon": 20.0},
    "val": {"angle_deg": -58.0, "horizon": 2.0},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_study_modules(study: Path):
    """Load the exact DEV-80 builder without copying its implementation."""
    source = study / "tools/run_study.py"
    spec = importlib.util.spec_from_file_location("dev80_run_study", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # run_study carries a historical source pin for its production arm.  Put
    # this worktree back first before any later minimax import.
    sys.path.insert(0, str(MINIMAX_SRC))
    return module, sys.modules["toy_models"]


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _infer_sigma(problem: ReciprocalMeasureProblem) -> int:
    imag = problem.denominators.imag
    if np.min(imag) > 0.0:
        return 1
    if np.max(imag) < 0.0:
        return -1
    raise ValueError("measure crosses causal half-planes")


def build_toy_group(study: Path, seed: int, branch: str, width_ev: float):
    """Build one whole toy branch on the DEV-80 25/50-bin lattices."""
    run_study, toy_models = _load_study_modules(study)
    ensemble_path = study / f"data/toy_ensemble_seed{seed}.npz"
    ensemble = dict(np.load(ensemble_path))
    for half in ("pole_frequency_pos", "pole_frequency_neg"):
        pole = np.asarray(ensemble[half], np.complex128)
        ensemble[half] = pole.real - 1.0j * ((-pole.imag) + ETA_EV)
    terms = toy_models.branch_terms(ensemble, branch)
    fit_cells, fit_mass, val_cells, val_mass = run_study.lattice_measure(
        terms["internal_sum"], terms["mass_abs_residue"], bins_per_axis=25
    )
    frequencies = np.linspace(0.0, float(width_ev), 81)
    fit = ReciprocalMeasureProblem(frequencies, fit_cells, fit_mass)
    validation = ReciprocalMeasureProblem(frequencies, val_cells, val_mass)
    contour = CONTOURS[branch]
    group = RoqGroup(
        f"toy_s{seed}_{branch}_w{width_ev:g}", fit, validation,
        sigma=_infer_sigma(fit), angle_deg=contour["angle_deg"],
        horizon=contour["horizon"],
    )
    return group, ensemble_path


def _fixed_basis_rank_scan(group: RoqGroup, max_rank: int, base_nodes: int):
    """Return every rank through the first 1e-4 validation-lattice pass."""
    scan_started = time.perf_counter()
    basis_started = time.perf_counter()
    candidates, gl = _candidates(group, base_nodes)
    singular, basis = _weighted_subspace(group, candidates, gl, max_rank)
    basis_seconds = time.perf_counter() - basis_started
    rows = []
    for rank in range(4, max_rank + 1):
        rank_started = time.perf_counter()
        pivots = la.qr(basis[:, :rank].T, mode="economic", pivoting=True)[2]
        selected = np.sort(pivots[:rank])
        times = candidates[selected]
        weights, _ = solve_fixed_time_weights_fast(
            group.fit, times, iterations=55, stall_iterations=6
        )
        error, _ = delivered_error(group.validation, times, weights)
        kappa_p99, kappa_max = rule_amplification(
            times, weights, group.validation
        )
        rows.append({
            "rank": rank,
            "max_error": float(np.max(error)),
            "kappa_p99": float(kappa_p99),
            "kappa_max": float(kappa_max),
            "singular_ratio": float(singular[rank - 1] / singular[0]),
            "rank_seconds": time.perf_counter() - rank_started,
        })
        if all(any(row["max_error"] <= target for row in rows)
               for target in TARGETS):
            break
    return rows, basis_seconds, time.perf_counter() - scan_started


def _selected_rows(rows):
    selected = {}
    for target in TARGETS:
        matches = [row for row in rows if row["max_error"] <= target]
        selected[f"{target:.0e}"] = matches[0] if matches else None
    return selected


def _certified_counts(catalog_path: Path, a_over_eta: float):
    catalog = json.loads(catalog_path.read_text())
    tables = [row for row in catalog["tables"]
              if row["family"] == "crossing_causal"]
    answer = {}
    for target, table_error in TABLE_ERROR_FOR_TARGET.items():
        matches = [row for row in tables
                   if row["range_max"] >= a_over_eta
                   and row["error_bound"] <= table_error]
        if not matches:
            answer[f"{target:.0e}"] = None
            continue
        chosen = min(matches, key=lambda row: (row["range_max"],
                                                -row["error_bound"],
                                                row["node_count"]))
        answer[f"{target:.0e}"] = {
            "requested_error": table_error,
            "range_max": chosen["range_max"],
            "node_count": chosen["node_count"],
            "certified_max_error": chosen["max_error"],
            "kappa0": chosen["kappa0"],
            "file": chosen["file"],
        }
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--seed", type=int, choices=range(4), required=True)
    parser.add_argument("--branch", choices=("cond", "val"), required=True)
    parser.add_argument("--width-ev", type=float, choices=WIDTHS_EV,
                        required=True)
    parser.add_argument("--max-rank", type=int, default=180)
    parser.add_argument("--base-nodes", type=int, default=384)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    width_index = WIDTHS_EV.index(args.width_ev)
    group, ensemble_path = build_toy_group(
        args.study, args.seed, args.branch, args.width_ev
    )
    rows, basis_seconds, scan_seconds = _fixed_basis_rank_scan(
        group, args.max_rank, args.base_nodes
    )
    payload = {
        "schema": "roq-scaling-study/v1",
        "method": "fixed-basis ascending integer-rank QDEIM scan",
        "git_head": _git_head(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_environment": {
            key: os.environ.get(key) for key in (
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"
            )
        },
        "source": str(ensemble_path),
        "source_sha256": _sha256(ensemble_path),
        "builder": str(args.study / "tools/run_study.py"),
        "builder_sha256": _sha256(args.study / "tools/run_study.py"),
        "catalog": str(args.catalog),
        "catalog_sha256": _sha256(args.catalog),
        "seed": args.seed,
        "branch": args.branch,
        "width_ev": args.width_ev,
        "eta_ev": ETA_EV,
        "a_over_eta": A_OVER_ETA[width_index],
        "contour": CONTOURS[args.branch],
        "fit_shape": list(group.fit.denominators.shape),
        "validation_shape": list(group.validation.denominators.shape),
        "max_rank": args.max_rank,
        "base_nodes": args.base_nodes,
        "basis_seconds": basis_seconds,
        "scan_seconds": scan_seconds,
        "selected": _selected_rows(rows),
        "certified": _certified_counts(args.catalog,
                                         A_OVER_ETA[width_index]),
        "rank_scan": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "seed", "branch", "width_ev", "a_over_eta", "basis_seconds",
        "scan_seconds", "selected", "certified"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
