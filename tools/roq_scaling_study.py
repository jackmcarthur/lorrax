"""Measure measure-adapted ROQ rank as a delivery interval grows.

This is an offline study driver, not a runtime planner.  It freezes the
DEV-80 measure construction and reuses one weighted snapshot eigensolve for
an ascending rank scan.  Each accepted rank is scored only on the independent
50-bin validation lattice and must also pass the runtime-noise gate.

The controlled family keeps the actual DEV-80 crossing mass and widths.  For
each branch it translates the delivery interval to the nearest support edge
and grows the interval and included measure together.  Translation does not
change the reciprocal kernel.  The causal contour is real time, and its
horizon is the elementary truncation horizon log(1 / target) / eta.  Thus the
only accuracy dials are the target and eta.
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
A_OVER_ETA = (12.0, 24.0, 47.0, 94.0)
WIDTHS_EV = tuple(value * ETA_EV for value in A_OVER_ETA)
TARGETS = (1.0e-3, 1.0e-4)
TABLE_ERROR_FOR_TARGET = {1.0e-3: 1.0e-4, 1.0e-4: 1.0e-5}
RUNTIME_NOISE = 6.0e-8
NOISE_FRACTION = 0.05


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


def build_toy_group(study: Path, seed: int, branch: str, width_ev: float,
                    target: float):
    """Build one translated crossing window on DEV-80 25/50-bin lattices."""
    run_study, toy_models = _load_study_modules(study)
    ensemble_path = study / f"data/toy_ensemble_seed{seed}.npz"
    ensemble = dict(np.load(ensemble_path))
    for half in ("pole_frequency_pos", "pole_frequency_neg"):
        pole = np.asarray(ensemble[half], np.complex128)
        ensemble[half] = pole.real - 1.0j * ((-pole.imag) + ETA_EV)
    terms = toy_models.branch_terms(ensemble, branch)
    raw_internal = terms["internal_sum"]
    if branch == "cond":
        omega_min = float(np.min(raw_internal.real))
        omega_max = omega_min + float(width_ev)
        keep = raw_internal.real <= omega_max
    else:
        omega_max = float(np.max(raw_internal.real))
        omega_min = omega_max - float(width_ev)
        keep = raw_internal.real >= omega_min
    if not np.any(keep):
        raise ValueError(f"empty crossing window for {branch} width {width_ev}")
    fit_cells, fit_mass, val_cells, val_mass = run_study.lattice_measure(
        raw_internal[keep], terms["mass_abs_residue"][keep], bins_per_axis=25
    )
    frequencies = np.linspace(omega_min, omega_max, 81)
    fit = ReciprocalMeasureProblem(frequencies, fit_cells, fit_mass)
    validation = ReciprocalMeasureProblem(frequencies, val_cells, val_mass)
    horizon = float(np.log(1.0 / float(target)) / ETA_EV)
    group = RoqGroup(
        f"toy_s{seed}_{branch}_w{width_ev:g}", fit, validation,
        sigma=_infer_sigma(fit), angle_deg=0.0, horizon=horizon,
    )
    selection = {
        "omega_min_ev": omega_min,
        "omega_max_ev": omega_max,
        "raw_cell_count": int(np.count_nonzero(keep)),
        "raw_cell_fraction": float(np.mean(keep)),
        "raw_mass_fraction": float(
            np.sum(terms["mass_abs_residue"][keep])
            / np.sum(terms["mass_abs_residue"])
        ),
    }
    return group, ensemble_path, selection


def _fixed_basis_rank_scan(group: RoqGroup, target: float, max_rank: int,
                           base_nodes: int, rank_step: int):
    """Bracket the first accepted rank, then check every rank in its bracket."""
    scan_started = time.perf_counter()
    basis_started = time.perf_counter()
    candidates, gl = _candidates(group, base_nodes)
    singular, basis = _weighted_subspace(group, candidates, gl, max_rank)
    basis_seconds = time.perf_counter() - basis_started
    rows_by_rank = {}

    def evaluate(rank):
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
        gate_limit = NOISE_FRACTION * float(target)
        row = {
            "rank": rank,
            "max_error": float(np.max(error)),
            "kappa_p99": float(kappa_p99),
            "kappa_max": float(kappa_max),
            "runtime_noise_bound": float(kappa_p99 * RUNTIME_NOISE),
            "runtime_noise_budget": gate_limit,
            "error_pass": bool(np.max(error) <= target),
            "noise_pass": bool(kappa_p99 * RUNTIME_NOISE <= gate_limit),
            "singular_ratio": float(singular[rank - 1] / singular[0]),
            "rank_seconds": time.perf_counter() - rank_started,
        }
        row["accepted"] = row["error_pass"] and row["noise_pass"]
        rows_by_rank[rank] = row
        print(
            f"rank {rank}: error={row['max_error']:.6e} "
            f"kappa={row['kappa_p99']:.6g} accepted={row['accepted']} "
            f"wall={row['rank_seconds']:.3f}s",
            flush=True,
        )
        return row

    coarse_ranks = list(range(4, max_rank + 1, int(rank_step)))
    if coarse_ranks[-1] != max_rank:
        coarse_ranks.append(max_rank)
    accepted_rank = None
    previous = 3
    for rank in coarse_ranks:
        row = evaluate(rank)
        if row["accepted"]:
            accepted_rank = rank
            break
        previous = rank
    if accepted_rank is not None:
        for rank in range(previous + 1, accepted_rank):
            if evaluate(rank)["accepted"]:
                break
    rows = [rows_by_rank[rank] for rank in sorted(rows_by_rank)]
    matches = [row for row in rows if row["accepted"]]
    selected = min(matches, key=lambda row: row["rank"]) if matches else None
    return (rows, selected, basis_seconds,
            time.perf_counter() - scan_started)


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
    parser.add_argument("--a-over-eta", type=float, choices=A_OVER_ETA,
                        required=True)
    parser.add_argument("--target", type=float, choices=TARGETS, required=True)
    parser.add_argument("--max-rank", type=int, default=180)
    parser.add_argument("--base-nodes", type=int, default=384)
    parser.add_argument("--rank-step", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    width_index = A_OVER_ETA.index(args.a_over_eta)
    width_ev = WIDTHS_EV[width_index]
    group, ensemble_path, selection = build_toy_group(
        args.study, args.seed, args.branch, width_ev, args.target
    )
    rows, selected, basis_seconds, scan_seconds = _fixed_basis_rank_scan(
        group, args.target, args.max_rank, args.base_nodes, args.rank_step
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
        "width_ev": width_ev,
        "eta_ev": ETA_EV,
        "a_over_eta": args.a_over_eta,
        "target": args.target,
        "contour": {"angle_deg": 0.0, "horizon": group.horizon},
        "window_selection": selection,
        "fit_shape": list(group.fit.denominators.shape),
        "validation_shape": list(group.validation.denominators.shape),
        "max_rank": args.max_rank,
        "base_nodes": args.base_nodes,
        "rank_step": args.rank_step,
        "basis_seconds": basis_seconds,
        "scan_seconds": scan_seconds,
        "selected": selected,
        "selected_plan_seconds": (
            None if selected is None
            else basis_seconds + selected["rank_seconds"]
        ),
        "certified": _certified_counts(args.catalog, args.a_over_eta)[
            f"{args.target:.0e}"
        ],
        "rank_scan": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "seed", "branch", "width_ev", "a_over_eta", "target", "basis_seconds",
        "scan_seconds", "selected", "certified"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
