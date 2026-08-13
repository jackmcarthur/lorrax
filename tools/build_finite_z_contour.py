#!/usr/bin/env python3
"""Build a finite-z contour rule for an existing physical request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from minimax import build_finite_z_contour


def _decode_request(path):
    document = json.loads(Path(path).read_text())
    manifest = document["physics_manifest"]
    scale = float.fromhex(manifest["energy_scale_ry"])
    interval = np.asarray(
        [float.fromhex(value) for value in
         manifest["transition_interval_ry"]], dtype=np.float64)
    z = np.asarray([
        complex(float.fromhex(real), float.fromhex(imag))
        for real, imag in manifest["z_values_ry"]], dtype=np.complex128)
    if interval.shape != (2,) or not 0.0 < interval[0] < interval[1]:
        raise ValueError("candidate transition interval is invalid")
    if z.ndim != 1 or z.size == 0 or np.any(z.imag <= 0.0):
        raise ValueError("finite-z request must lie in the upper half-plane")
    if not (scale > 0.0):
        raise ValueError("candidate energy scale must be positive")
    return scale, interval, z, manifest


def _complex_hex(values):
    return [[float(value.real).hex(), float(value.imag).hex()]
            for value in np.asarray(values).ravel()]


def _json_report(rule, scale, interval, z, manifest, target):
    tau_hat, weights_hat, signs = rule.runtime_arguments()
    tau = tau_hat / scale
    weights = weights_hat / scale
    arms = {}
    for name, row in (("resonant", rule.resonant),
                      ("antiresonant", rule.antiresonant)):
        arms[name] = {
            "nodes": int(row.tau.size),
            "frequency_sign": int(row.frequency_sign),
            "contour_angle": float(np.angle(row.contour)),
            "orders": list(map(int, row.orders)),
            "heldout_scaled_error": float(row.heldout_scaled_error),
        }
    return {
        "format": "lorrax-chi-one-sided-sampled-candidate/1",
        "qualification": "sampled-non-production",
        "kernel": "K_z(Delta)=-1/(Delta-z)-1/(Delta+z)",
        "target": float(target),
        "transition_interval_ry": interval.tolist(),
        "z_values_ry": _complex_hex(z),
        "source_batch_id": manifest.get("batch_id"),
        "arms": arms,
        "executed_nodes": rule.executed_nodes,
        "heldout_combined_scaled_error": float(
            rule.heldout_combined_scaled_error),
        "validation_points": int(rule.validation_points),
        "runtime_union": {
            "tau_ry_inverse": _complex_hex(tau),
            "base_weights_ry_inverse": _complex_hex(weights),
            "frequency_sign": signs.tolist(),
            "weight_rows": "broadcast base_weights over requested z rows",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_artifact", type=Path)
    parser.add_argument("--target", type=float, default=1.0e-8)
    parser.add_argument("--design-points", type=int, default=2049)
    parser.add_argument("--validation-points", type=int, default=32769)
    parser.add_argument("--angle-step", type=float, default=0.025)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scale, interval, z, manifest = _decode_request(args.candidate_artifact)
    rule = build_finite_z_contour(
        interval / scale, z / scale, args.target,
        design_points=args.design_points,
        validation_points=args.validation_points,
        angle_step=args.angle_step)
    report = _json_report(rule, scale, interval, z, manifest, args.target)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload)
        print(json.dumps({
            "output": str(args.output),
            "executed_nodes": report["executed_nodes"],
            "heldout_combined_scaled_error": report[
                "heldout_combined_scaled_error"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
