"""Shared real-pole input recipe and gate vocabulary (DESIGN §§5–6).

Host metadata only: no response matrices, device allocation, backend selection,
chemical-potential solve, or numerical gate implementation lives here. Physical
sample coordinates are Ry; reporting coordinates are eV. Consumers import these
tables instead of copying the thresholds into bank/constructor/store/Sigma code.
"""
from __future__ import annotations

import hashlib
import json
import math

RECIPE_VERSION = "shared_real_pole_v1_r3b"
GATE_VERSION = "shared_real_pole_gates_v1_r3b"
RECEIPT_SCHEMA = "lorrax.shared-real-pole.receipt.v1"

shared_real_pole_v1_r3b = {
    "version": RECIPE_VERSION,
    "height_eta_factor": 4.0,
    "reference_eta_ev": 0.25,
    "active_depth_ev": 15.0,
    "borderline_depth_ev": 25.0,
    "line_break_ev": 12.0,
    "line_low_step_ev": 0.5,
    "line_metal_step_ev": 1.0,
    "line_high_step_ev": 1.0,
    "plasma_margin_ev": 3.5,
    "imaginary_floor_max_ev": 16.0,
    "imaginary_count_epsilon": 1.0e-3,
    "imaginary_min_count": 2,
    "held_line_fractions": (0.25, 0.65),
    "multiplet_relative_tolerance": 1.0e-6,
    "bank_rule_tolerance": 1.0e-8,
    "moment_convention": "S_m = 2 M_(2m+1); physical M1 and M3 only",
    "production": {"direction_cutoff": 1.0e-3, "imaginary_width_fraction": 0.25,
                   "infinity_width_fraction": 0.125, "sigma_tolerance": 1.0e-4},
    "relaxed": {"direction_cutoff": 1.0e-2, "imaginary_width_fraction": 0.125,
                "infinity_width_fraction": 0.0625, "sigma_tolerance": 1.0e-3,
                "line_count": 8, "imaginary_count": 2},
}

# Each entry is (predicate description, threshold); the public table adds name
# and version. Composite checks retain their individual dimensional thresholds.
_GATE_ROWS = {
    "normalized_gram_keep": ("retain gamma/gamma_max strictly above cut", 1.0e-8),
    "normalized_gram_validity": ("gamma_min/gamma_max >= threshold", -1.0e-7),
    "zero_ritz_policy": ("drop lambda <= cutoff only within factor-weight budget",
                         {"lambda_cutoff_ry2": 1.0e-6, "max_dropped_weight_fraction": 1.0e-6}),
    "finite_factors_poles": ("finite complex128 C; finite positive float64 active poles2; int64 K; exact-zero inactive C and positive sentinel", True),
    "passivity": ("V-whitened -Wc(i eta) spectrum in bounds and relative anti-Hermitian part within tolerance",
                  {"eigenvalue_min": -1.0e-10, "eigenvalue_max": 1.0 + 1.0e-8,
                   "antihermitian_relative_max": 1.0e-10}),
    "retained_subspace_moments": ("relative Q_inf-projected M1 and M3 defects after cut and zero policy <= threshold", 1.0e-10),
    "held_w_full_moment_defects": ("held W and full M1/M3 diagnostics with extracted CD8/CD10 values and receipt paths; missing extraction blocks landing", None),
    "representation": ("scalar N_spinor=1 and authenticated TRS allowed", {"nspinor": 1, "trs_allowed": True}),
    "capacity": ("aggregate live bytes per rank including workspace <= threshold * U", 3.0),
    "rule_validity": ("bank and Sigma certificates cover current domains at resolved tolerances", True),
    "sc_rebuild": ("samples, directions, poles, ranks, intervals and rules rebuilt at current bands and occupations", True),
}
shared_real_pole_gates_v1_r3b = {
    name: {"name": name, "predicate": predicate, "threshold": threshold,
           "version": GATE_VERSION}
    for name, (predicate, threshold) in _GATE_ROWS.items()
}


def table_hash(table):
    """SHA256 of a small JSON table with canonical ordering and finite numbers."""
    return hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


RECIPE_HASH = table_hash(shared_real_pole_v1_r3b)
GATE_HASH = table_hash(shared_real_pole_gates_v1_r3b)


def gate_receipt(name, value=None, *, passed=None, reason):
    """Record a consumer-measured gate; missing data can never produce PASS.

    Parameters
    ----------
    name : str
        Key in the canonical gate table.
    value : JSON-compatible scalar, list or dict, optional
        Small measured summaries only; no matrices. Nonfinite numbers refuse.
    passed : bool or None
        Consumer's evaluation of the named predicate, or unmeasured/diagnostic.
    reason : str
        Measurement scope/evidence, or the concrete missing dependency.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("shared-pole receipt requires a nonempty reason")
    if passed is not None and type(passed) is not bool:
        raise TypeError("shared-pole receipt passed must be bool or None")
    json.dumps(value, allow_nan=False)
    row = shared_real_pole_gates_v1_r3b[name]
    status = "NOT_MEASURED" if value is None or passed is None else (
        "PASS" if passed else "FAIL")
    return {"predicate": row["predicate"], "name": name, "version": GATE_VERSION,
            "value": value, "threshold": row["threshold"], "status": status,
            "reason": reason}


def construction_receipt(measurements=None):
    """Complete receipt skeleton, explicitly marking every absent gate unmeasured.

    ``measurements`` maps names to ``gate_receipt`` keyword dictionaries. Dense
    consumers supply measurements; creating this skeleton certifies no physics.
    """
    measurements = {} if measurements is None else measurements
    unknown = set(measurements) - set(shared_real_pole_gates_v1_r3b)
    if unknown:
        raise ValueError(f"unknown shared-pole receipt predicates: {sorted(unknown)}")
    return {"schema": RECEIPT_SCHEMA, "recipe_version": RECIPE_VERSION,
            "recipe_hash": RECIPE_HASH, "gate_version": GATE_VERSION,
            "gate_hash": GATE_HASH,
            "gates": [gate_receipt(name, **measurements.get(
                name, {"reason": "measurement not supplied"}))
                for name in shared_real_pole_gates_v1_r3b]}
