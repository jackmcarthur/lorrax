#!/usr/bin/env python3
"""Time lookup-first fitting on the frozen six-window Na measure.

The archive contains the fitting and refined validation lattices.  The
incumbent targets and achieved residuals below are copied from its companion
``causal_hankel_results.json`` evidence.  The timer excludes imports and file
loading; it includes every catalog lookup, measured score, and the two single
crossing fits.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import warnings

import numpy as np

from gw.mpa import delivered_windows as planner
from minimax import ReciprocalMeasureProblem


DEFAULT_PROBLEMS = Path(
    "/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/"
    "runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/"
    "evidence/causal_hankel/na_reconstructed_problems_v1.npz")

# Source: causal_hankel_results.json beside DEFAULT_PROBLEMS, ``na.incumbent``.
# Values are independently remeasured on the same refined lattices in the NPZ.
_INCUMBENT = (
    (8.078747719098750e-4, 8.034575259862507e-4),
    (2.722503537397882e-4, 2.678331078161917e-4),
    (8.709997535773768e-5, 8.268272943420593e-5),
    (2.687232115805145e-5, 2.245507523444509e-5),
    (1.081294734976452e-3, 1.076877489052870e-3),
    (6.484833977231542e-4, 6.440661517995450e-4),
)


def _problems(path: Path):
    rows = []
    with np.load(path, allow_pickle=False) as saved:
        keys = np.asarray(saved["keys"]).tolist()
        if len(keys) != len(_INCUMBENT):
            raise RuntimeError(
                f"expected {len(_INCUMBENT)} frozen windows, found {len(keys)}")
        for index, (key, incumbent) in enumerate(zip(keys, _INCUMBENT)):
            problem = ReciprocalMeasureProblem(
                saved[f"p{index}_frequencies"],
                saved[f"p{index}_internal"],
                saved[f"p{index}_mass"])
            validation = ReciprocalMeasureProblem(
                saved[f"p{index}_frequencies"],
                saved[f"p{index}_validation_internal"],
                saved[f"p{index}_validation_mass"])
            pole_sign = 1.0 if "cond:" in key else -1.0
            rows.append((key, problem, validation, pole_sign, *incumbent))
    return rows


def _fit(problem, validation, pole_sign, target):
    if planner._window_kind(problem) == "crossing":
        candidates = planner._crossing_table_candidates(
            problem, pole_sign, target, planner.MAX_WINDOW_TAU_PAIRS)
    else:
        candidates = planner._sign_definite_table_candidates(
            problem, target, planner.MAX_WINDOW_TAU_PAIRS)
    for times, weights, evidence in candidates:
        candidate = planner._rule_candidate(
            problem, validation, times, weights, evidence)
        if planner._rule_accepted(candidate["metrics"], target):
            return candidate
    if planner._window_kind(problem) != "crossing":
        raise RuntimeError("the noncrossing catalog family was exhausted")
    return planner._rule_candidate(
        problem, validation,
        *planner._fit_crossing_once(
            problem, pole_sign, target, planner.MAX_WINDOW_TAU_PAIRS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS)
    args = parser.parse_args()
    rows = _problems(args.problems)
    warnings.filterwarnings("ignore", message=r"minimax: served .*")

    started = time.perf_counter()
    fitted = [
        (key, target, incumbent,
         _fit(problem, validation, pole_sign, target))
        for key, problem, validation, pole_sign, target, incumbent in rows
    ]
    wall = time.perf_counter() - started

    print("window | family | nodes | achieved | incumbent | ratio | noise gate")
    for key, target, incumbent, fit in fitted:
        residual, kappa_p99, _peak = fit["metrics"]
        ratio = residual / incumbent
        noise = kappa_p99 * planner.RUNTIME_NOISE_EPSILON
        noise_budget = planner.AMPLIFICATION_NOISE_SAFETY * target
        passed = residual <= target and noise <= noise_budget
        print(
            f"{key} | {fit['evidence']['family']} | {fit['times'].size} | "
            f"{residual:.9g} | {incumbent:.9g} | {ratio:.4f} | "
            f"{noise:.3g}/{noise_budget:.3g} {'PASS' if passed else 'FAIL'}")
        if ratio > 2.0 or not passed:
            raise SystemExit(f"frozen Na accuracy gate failed for {key}")
    print(f"fitting_stage_seconds={wall:.6f} limit=3.000000")
    if wall >= 3.0:
        raise SystemExit("frozen Na fitting-stage wall exceeded 3 seconds")


if __name__ == "__main__":
    main()
