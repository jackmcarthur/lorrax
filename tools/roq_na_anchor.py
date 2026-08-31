"""Replay the frozen Na ROQ anchor and the two crossing-window targets.

The input is the read-only reconstruction exported by the DEV-80 causal
study.  The three-group contours and the 50-node allocation are fixed by
source commit 1426d9f4 and its follow-up measurement: conduction resonant
uses (0 degrees, 260 Ry^-1), conduction tails use (-65 degrees, 85 Ry^-1),
and all valence windows use (-58 degrees, 27 Ry^-1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from minimax import (
    ReciprocalMeasureProblem,
    RoqGroup,
    branch_delivered_error,
    branch_noise_gate,
    delivered_error,
    fit_roq_branch,
    fit_roq_group,
    rule_amplification,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = Path(
    "/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/"
    "runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/"
    "evidence/causal_hankel/na_reconstructed_problems_v1.npz"
)
RUNTIME_NOISE = 6.0e-8
NOISE_FRACTION = 0.05
BASE_NODES = 280


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _problem(saved, indices, validation=False):
    tag = "validation_" if validation else ""
    frequencies = saved[f"p{indices[0]}_frequencies"]
    internal = np.concatenate(
        [saved[f"p{index}_{tag}internal"] for index in indices]
    )
    mass = np.concatenate([saved[f"p{index}_{tag}mass"] for index in indices])
    return ReciprocalMeasureProblem(frequencies, internal, mass)


def _group(saved, name, indices, sigma, angle_deg, horizon):
    return RoqGroup(
        name,
        _problem(saved, indices),
        _problem(saved, indices, validation=True),
        sigma=sigma,
        angle_deg=angle_deg,
        horizon=horizon,
    )


def _fit_rank(group, rank):
    started = time.perf_counter()
    rule = fit_roq_group(
        group, 1.0, ranks=[int(rank)], base_nodes=BASE_NODES
    )
    return rule, time.perf_counter() - started


def _rule_row(rule, fit_seconds):
    return {
        "rank": rule.rank,
        "max_error": rule.max_error,
        "kappa_p99": rule.kappa_p99,
        "kappa_max": rule.kappa_max,
        "singular_ratio": rule.singular_ratio,
        "fit_seconds": fit_seconds,
    }


def _incumbent_aggregate(saved, metadata):
    """Recompute branch errors of the frozen 137-pair production plan."""
    receipt_path = Path(metadata["receipt"])
    with receipt_path.open("rb") as handle:
        receipt = pickle.load(handle)
    eta = float(metadata["eta_ry"])
    rows = []
    for index, row in enumerate(receipt["rows"]):
        branch = "cond" if int(row["branch_index"]) == 0 else "val"
        sigma = 1 if branch == "cond" else -1
        problem = _problem(saved, [index], validation=True)
        times = sigma * np.asarray(row["t"], np.complex128)
        weights = np.asarray(row["alpha"], np.complex128) * np.exp(
            eta * np.asarray(row["t"], np.complex128)
        )
        error, _ = delivered_error(problem, times, weights)
        rows.append({
            "name": str(saved["keys"][index]),
            "problem": problem,
            "times": times,
            "weights": weights,
            "nodes": int(times.size),
            "max_error": float(np.max(error)),
        })

    aggregate = {}
    for branch, indices in (("cond", (0, 1, 2)), ("val", (3, 4, 5))):
        numerator = denominator = 0.0
        for index in indices:
            row = rows[index]
            problem = row["problem"]
            d = problem.denominators
            approximation = (
                np.exp(1j * d[..., None] * row["times"]) @ row["weights"]
            )
            numerator = (
                numerator
                + np.abs(approximation - 1.0 / d) @ problem.cell_masses
            )
            denominator = (
                denominator + (1.0 / np.abs(d)) @ problem.cell_masses
            )
        aggregate[branch] = float(np.max(numerator / denominator))
    return {
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "total_pairs": sum(row["nodes"] for row in rows),
        "window_rows": [
            {key: row[key] for key in ("name", "nodes", "max_error")}
            for row in rows
        ],
        "aggregate_branch_error": aggregate,
    }, receipt


def _first_accepted(group, target, max_rank):
    started = time.perf_counter()
    scanned = []
    selected = None
    for rank in range(4, int(max_rank) + 1):
        rule, fit_seconds = _fit_rank(group, rank)
        row = _rule_row(rule, fit_seconds)
        row["runtime_noise_bound"] = rule.kappa_p99 * RUNTIME_NOISE
        row["runtime_noise_budget"] = NOISE_FRACTION * target
        row["error_pass"] = rule.max_error <= target
        row["noise_pass"] = (
            row["runtime_noise_bound"] <= row["runtime_noise_budget"]
        )
        row["accepted"] = row["error_pass"] and row["noise_pass"]
        scanned.append(row)
        if row["accepted"]:
            selected = row
            break
    return selected, scanned, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    saved = np.load(args.cache)
    metadata = json.loads(str(saved["metadata_json"]))
    incumbent, receipt = _incumbent_aggregate(saved, metadata)
    reports = {
        row["name"]: row
        for branch in receipt["geometry"]["branches"]
        for row in branch["windows"]
    }

    groups = [
        _group(saved, "cond_resonant", [0], 1, 0.0, 260.0),
        _group(saved, "cond_tails", [1, 2], 1, -65.0, 85.0),
        _group(saved, "val_all", [3, 4, 5], -1, -58.0, 27.0),
    ]
    reference_ranks = (25, 13, 12)
    reference_started = time.perf_counter()
    start_rules = []
    reference_rows = []
    for group, rank in zip(groups, reference_ranks):
        rule, fit_seconds = _fit_rank(group, rank)
        start_rules.append(rule)
        reference_rows.append(_rule_row(rule, fit_seconds))
    cond_started = time.perf_counter()
    cond_rules, cond_error = fit_roq_branch(groups[:2], start_rules[:2])
    cond_joint_seconds = time.perf_counter() - cond_started
    cond_gate, cond_kappa = branch_noise_gate(
        groups[:2], cond_rules,
        incumbent["aggregate_branch_error"]["cond"],
    )
    val_error = branch_delivered_error(groups[2:], start_rules[2:])
    val_gate, val_kappa = branch_noise_gate(
        groups[2:], start_rules[2:],
        incumbent["aggregate_branch_error"]["val"],
    )
    reference = {
        "allocation": list(reference_ranks),
        "total_nodes": sum(reference_ranks),
        "group_rows_before_joint_fit": reference_rows,
        "cond_joint_seconds": cond_joint_seconds,
        "total_fit_seconds": time.perf_counter() - reference_started,
        "cond_achieved_error": float(np.max(cond_error)),
        "cond_incumbent_aggregate_error": incumbent[
            "aggregate_branch_error"
        ]["cond"],
        "cond_kappa_p99_effective": cond_kappa,
        "cond_noise_gate": cond_gate,
        "val_achieved_error": float(np.max(val_error)),
        "val_incumbent_aggregate_error": incumbent[
            "aggregate_branch_error"
        ]["val"],
        "val_kappa_p99_effective": val_kappa,
        "val_noise_gate": val_gate,
    }

    crossing_specs = [
        (0, "cond", 1, 0.0, 260.0, "ω≥E_F cond:resonant"),
        (4, "val", -1, -58.0, 27.0, "ω≥E_F val:resonant"),
    ]
    crossing = []
    for index, branch, sigma, angle, horizon, report_name in crossing_specs:
        target = float(reports[report_name]["relative_residual_target"])
        group = _group(
            saved, f"{branch}_crossing", [index], sigma, angle, horizon
        )
        selected, scanned, search_seconds = _first_accepted(
            group, target, max_rank=40
        )
        crossing.append({
            "name": report_name,
            "target": target,
            "contour": {"angle_deg": angle, "horizon": horizon},
            "selected": selected,
            "search_seconds": search_seconds,
            "rank_scan": scanned,
        })

    payload = {
        "schema": "roq-na-anchor/v1",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cache": str(args.cache),
        "cache_sha256": _sha256(args.cache),
        "base_nodes": BASE_NODES,
        "incumbent": incumbent,
        "reference_50_node_replay": reference,
        "crossing_window_targets": crossing,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "reference_50_node_replay": reference,
        "crossing_window_targets": [
            {key: row[key] for key in ("name", "target", "selected",
                                        "search_seconds")}
            for row in crossing
        ],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
