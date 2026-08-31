#!/usr/bin/env python3
"""Audit measure compression against uncompressed Na product pairs.

This is an offline study harness, not a planner path.  It rebuilds selected
Na windows from the frozen histogram export, verifies the production 25/50
lattices bit-for-bit, and fits the same ROQ family against 25-bin, 50-bin,
100-bin, and raw measures.  Every fitted rule is accepted and reported only
by a chunked score on the raw pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import scipy.linalg as la

from minimax import (ReciprocalMeasureProblem, RoqGroup,
                     solve_fixed_time_weights_fast,
                     tail_refined_lattice_measure)
from minimax.roq_fit import _candidates, _weighted_subspace


EXPORT = Path(
    "/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/"
    "reports/tr_broken_gnppm_and_new_minimax_2026-08-28/"
    "hankel_agent_export/na_measures"
)
FROZEN = Path(
    "/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/"
    "runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/"
    "evidence/causal_hankel/na_reconstructed_problems_v1.npz"
)
NOISE = 6.0e-8
NOISE_SHARE = 0.05

# Fixed family from commit 1426d9f4's frozen-Na ROQ evidence: all valence
# windows share the -58 degree contour and 27 Ry^-1 horizon.
CASE_CONFIG = {
    4: {"kind": "crossing", "angle_deg": -58.0, "horizon": 27.0},
    5: {"kind": "sign-definite", "angle_deg": -58.0, "horizon": 27.0},
}


def _raw_window(index: int, states: np.ndarray, poles: np.ndarray,
                registry: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    """Rebuild one raw state-by-pole product measure from the export."""
    spec = registry["windows"][index]
    member = states[f"in_window{index}"] > 0.5
    energy = states["energy_minus_mu_ry"][member]
    mass_name = "cond_mass" if spec["branch"] == "cond" else "val_mass"
    state_mass = states[mass_name][member]
    sum_parts, mass_parts = [], []
    for interval in spec["pole_intervals_used"]:
        for pole in spec["pole_indices"]:
            selected = ((poles["pole"] == pole)
                        & (poles["interval_0shallow_1deep"] == interval))
            if not np.any(selected):
                continue
            cells = (poles["cell_re_ry"][selected]
                     + 1.0j * poles["cell_im_ry_broadened"][selected])
            sum_parts.append((energy[:, None]
                              + spec["pole_sign"] * cells[None, :]).reshape(-1))
            mass_parts.append((state_mass[:, None]
                               * poles["residue_mass"][selected][None, :]
                               ).reshape(-1))
    if not sum_parts:
        raise RuntimeError(f"window {index} has no raw product pairs")
    return spec, np.concatenate(sum_parts), np.concatenate(mass_parts)


def _raw_error(frequencies: np.ndarray, sums: np.ndarray, masses: np.ndarray,
               times: np.ndarray, weights: np.ndarray,
               *, chunk_size: int = 4096) -> float:
    """Exact delivered-error maximum with bounded raw-pair temporaries."""
    numerator = np.zeros(frequencies.size)
    denominator = np.zeros(frequencies.size)
    for start in range(0, sums.size, int(chunk_size)):
        support = sums[start:start + int(chunk_size)]
        mass = masses[start:start + int(chunk_size)]
        d = frequencies[:, None] - support[None, :]
        value = np.zeros(d.shape, dtype=np.complex128)
        for node, coefficient in zip(times, weights):
            value += coefficient * np.exp(1.0j * node * d)
        numerator += np.abs(value - 1.0 / d) @ mass
        denominator += (1.0 / np.abs(d)) @ mass
    return float(np.max(numerator / denominator))


def _raw_amplification(frequencies: np.ndarray, sums: np.ndarray,
                       masses: np.ndarray, times: np.ndarray,
                       weights: np.ndarray,
                       *, chunk_size: int = 4096) -> tuple[float, float]:
    """Production mass-weighted p99/max amplification on raw pairs."""
    kappa_parts, mass_parts = [], []
    for start in range(0, sums.size, int(chunk_size)):
        support = sums[start:start + int(chunk_size)]
        mass = masses[start:start + int(chunk_size)]
        d = frequencies[:, None] - support[None, :]
        value = np.zeros(d.shape, dtype=np.complex128)
        magnitude = np.zeros(d.shape)
        for node, coefficient in zip(times, weights):
            term = coefficient * np.exp(1.0j * node * d)
            value += term
            magnitude += np.abs(term)
        kappa_parts.append(
            (magnitude / np.maximum(np.abs(value), 1.0e-300)).reshape(-1))
        mass_parts.append(np.broadcast_to(mass[None, :], d.shape).reshape(-1))
    kappa = np.concatenate(kappa_parts)
    point_mass = np.concatenate(mass_parts)
    order = np.argsort(kappa, kind="stable")
    cumulative = np.cumsum(point_mass[order])
    position = np.searchsorted(cumulative, 0.99 * cumulative[-1])
    return float(kappa[order[position]]), float(np.max(kappa))


def _measure_variants(sums: np.ndarray, masses: np.ndarray):
    """Return the four fit measures without changing the raw reference."""
    base, base_mass, refined, refined_mass = tail_refined_lattice_measure(
        sums, masses, bins_per_axis=25)
    four, four_mass, _, _ = tail_refined_lattice_measure(
        sums, masses, bins_per_axis=100)
    return {
        "base_25": (base, base_mass),
        "refined_50": (refined, refined_mass),
        "lattice_100": (four, four_mass),
        "raw": (sums, masses),
    }


def _fit_variant(name: str, fit_sums: np.ndarray, fit_masses: np.ndarray,
                 raw_sums: np.ndarray, raw_masses: np.ndarray,
                 frequencies: np.ndarray, target: float, sigma: int,
                 angle_deg: float, horizon: float, max_rank: int,
                 base_nodes: int) -> dict:
    """Find the first integer ROQ rank accepted by the raw-pair gates."""
    started = time.perf_counter()
    problem = ReciprocalMeasureProblem(frequencies, fit_sums, fit_masses)
    group = RoqGroup(name, problem, problem, sigma=sigma,
                     angle_deg=angle_deg, horizon=horizon)
    candidates, quadrature_weights = _candidates(group, base_nodes)
    singular, basis = _weighted_subspace(
        group, candidates, quadrature_weights, max_rank)
    history = []
    accepted = None
    for rank in range(4, int(max_rank) + 1):
        pivots = la.qr(basis[:, :rank].T, mode="economic", pivoting=True)[2]
        times = candidates[np.sort(pivots[:rank])]
        weights, fit_error = solve_fixed_time_weights_fast(
            problem, times, iterations=55, stall_iterations=6)
        raw_error = _raw_error(
            frequencies, raw_sums, raw_masses, times, weights)
        row = {
            "rank": rank,
            "fit_error": float(fit_error),
            "raw_error": raw_error,
            "lattice_to_raw_ratio": float(fit_error / raw_error),
        }
        if raw_error <= target:
            p99, peak = _raw_amplification(
                frequencies, raw_sums, raw_masses, times, weights)
            row.update({
                "raw_kappa_p99": p99,
                "raw_kappa_max": peak,
                "noise_bound": p99 * NOISE,
                "noise_budget": NOISE_SHARE * target,
                "noise_gate": bool(p99 * NOISE <= NOISE_SHARE * target),
            })
            if row["noise_gate"]:
                accepted = dict(row)
                accepted["times_re"] = times.real.tolist()
                accepted["times_im"] = times.imag.tolist()
                accepted["weights_re"] = weights.real.tolist()
                accepted["weights_im"] = weights.imag.tolist()
        history.append(row)
        print(name, rank, f"fit={fit_error:.6e}",
              f"raw={raw_error:.6e}",
              "ACCEPT" if accepted is not None else "", flush=True)
        if accepted is not None:
            break
    return {
        "fit_cells": int(fit_sums.size),
        "fit_mass": float(np.sum(fit_masses)),
        "accepted": accepted,
        "history": history,
        "singular_ratio_at_cap": float(singular[-1] / singular[0]),
        "seconds": float(time.perf_counter() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", nargs="+", type=int, default=[4, 5])
    parser.add_argument("--max-rank", type=int, default=22)
    parser.add_argument("--base-nodes", type=int, default=96)
    arguments = parser.parse_args()

    states = np.genfromtxt(EXPORT / "g_states.csv", delimiter=",", names=True)
    poles = np.genfromtxt(EXPORT / "w_histogram.csv", delimiter=",", names=True)
    registry = json.loads((EXPORT / "windows.json").read_text())
    frozen = np.load(FROZEN)
    result = {
        "label": arguments.label,
        "method": {
            "family": "measure-weighted causal ROQ/QDEIM plus IRLS",
            "rank_scan": [4, arguments.max_rank],
            "base_nodes": arguments.base_nodes,
            "acceptance_reference": "raw state-by-pole pairs",
            "noise": NOISE,
            "noise_share": NOISE_SHARE,
        },
        "windows": [],
    }
    for index in arguments.windows:
        if index not in CASE_CONFIG:
            raise ValueError(f"window {index} has no preregistered fixed family")
        config = CASE_CONFIG[index]
        spec, raw_sums, raw_masses = _raw_window(
            index, states, poles, registry)
        variants = _measure_variants(raw_sums, raw_masses)
        base, base_mass = variants["base_25"]
        refined, refined_mass = variants["refined_50"]
        bit_exact = bool(
            np.array_equal(base, frozen[f"p{index}_internal"])
            and np.array_equal(base_mass, frozen[f"p{index}_mass"])
            and np.array_equal(refined, frozen[f"p{index}_validation_internal"])
            and np.array_equal(refined_mass, frozen[f"p{index}_validation_mass"])
        )
        if arguments.label == "baseline" and not bit_exact:
            raise RuntimeError(f"window {index} did not rebuild bit-exactly")
        frequencies = np.asarray(spec["omega_grid_ry"])
        row = {
            "index": index,
            "name": spec["name"],
            "kind": config["kind"],
            "target": float(spec["relative_residual_target"]),
            "raw_pairs": int(raw_sums.size),
            "raw_mass": float(np.sum(raw_masses)),
            "frozen_base_refined_bit_exact": bit_exact,
            "angle_deg": config["angle_deg"],
            "horizon": config["horizon"],
            "variants": {},
        }
        for name, (fit_sums, fit_masses) in variants.items():
            row["variants"][name] = _fit_variant(
                name, fit_sums, fit_masses, raw_sums, raw_masses,
                frequencies, row["target"], int(spec["sigma_half_plane_sign"]),
                config["angle_deg"], config["horizon"],
                arguments.max_rank, arguments.base_nodes)
        result["windows"].append(row)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
