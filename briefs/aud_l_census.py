#!/usr/bin/env python3
"""Matched legacy/current Loewner census on one real W sample store."""

from __future__ import annotations

import argparse
import json
import os
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from gw.mpa import pade_fit


def _legacy_roots(w, x_hat, n, rcond, eig="lapack"):
    """The pre-change one-sided pseudoinverse realization."""
    L, sL = pade_fit._loewner_pencil(w, x_hat, n)
    u, s, vh = jnp.linalg.svd(L, full_matrices=False)
    s_max, s_min = s[0], s[-1]
    s_inv = jnp.where(
        s > rcond * s_max, 1.0 / jnp.where(s > 0, s, 1.0), 0.0)
    L_pinv = vh.conj().T @ (s_inv.astype(L.dtype)[:, None] * u.conj().T)
    X = L_pinv @ sL
    cond = jnp.where(s_min > 0, s_max / s_min, jnp.inf)
    return (pade_fit._eigvals(X, eig), cond, s_max, s_min,
            pade_fit._matrix_backward_error(L, sL, X))


def _sample_indices(n_p: int) -> np.ndarray:
    """Evenly retain n_p points from each line of the 12-pole grid."""
    one_line = np.rint(np.linspace(0, 11, n_p)).astype(np.int64)
    assert len(np.unique(one_line)) == n_p
    return np.concatenate((one_line, 12 + one_line))


def _fit_kernel(n_p: int, legacy: bool):
    def fit(samples, z):
        return pade_fit.fit_mpa_poles_batched(
            samples, z, n_p, rcond=1.0e-13, eig="jax_qr",
            solve="loewner")
    return jax.jit(fit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_file")
    parser.add_argument("output_dir")
    parser.add_argument("--chunk", type=int, default=4096)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    size = int(os.environ.get("SLURM_NTASKS", "1"))
    all_rows = []
    with h5py.File(args.sample_file, "r") as h5:
        data = h5["Wc_qmunu_z"]
        z24 = np.asarray(h5["Wc_qmunu_z__mpa/omega"], np.complex128)
        q_ids = np.arange(data.shape[1])[rank::size]
        for n_p in (8, 10, 12):
            pick = _sample_indices(n_p)
            z = jnp.asarray(z24[pick])
            samples = np.asarray(data[pick][:, q_ids], np.complex128)
            samples = np.moveaxis(samples, 0, -1).reshape(-1, 2 * n_p)
            arm_results = {}
            current_roots = pade_fit._loewner_roots
            for arm, legacy in (("legacy", True), ("current", False)):
                pade_fit._loewner_roots = (
                    _legacy_roots if legacy else current_roots)
                kernel = _fit_kernel(n_p, legacy)
                leaves = []
                started = time.perf_counter()
                for lo in range(0, len(samples), args.chunk):
                    omega, residue, diag = jax.device_get(
                        kernel(jnp.asarray(samples[lo:lo + args.chunk]), z))
                    leaves.append((omega, residue, diag))
                elapsed = time.perf_counter() - started
                omega = np.concatenate([x[0] for x in leaves])
                residue = np.concatenate([x[1] for x in leaves])
                diag = {
                    key: np.concatenate([np.asarray(x[2][key]) for x in leaves])
                    for key in leaves[0][2]
                }
                model = np.asarray(jax.vmap(pade_fit.eval_mpa_model, (0, 0, None))(
                    jnp.asarray(omega), jnp.asarray(residue), z))
                err = np.abs(model - samples)
                scale = max(float(np.max(np.abs(samples))), np.finfo(float).tiny)
                arm_results[arm] = {
                    "cond": diag["cond_pade"],
                    "backward": diag["backward_error"],
                    "n_valid": diag["n_valid"],
                    "omega": omega,
                    "residue": residue,
                    "rel_rms": float(np.sqrt(np.mean(err ** 2)) / scale),
                    "max_rel": float(np.max(err) / scale),
                    "seconds": elapsed,
                }
            pade_fit._loewner_roots = current_roots
            old, new = arm_results["legacy"], arm_results["current"]
            row = {
                "rank": rank, "n_p": n_p, "n_elements": len(samples),
                "legacy_cond_p50": float(np.median(old["cond"])),
                "legacy_cond_p99": float(np.quantile(old["cond"], .99)),
                "legacy_cond_max": float(np.max(old["cond"])),
                "current_cond_p50": float(np.median(new["cond"])),
                "current_cond_p99": float(np.quantile(new["cond"], .99)),
                "current_cond_max": float(np.max(new["cond"])),
                "legacy_rel_rms": old["rel_rms"],
                "current_rel_rms": new["rel_rms"],
                "legacy_max_rel": old["max_rel"],
                "current_max_rel": new["max_rel"],
                "legacy_seconds": old["seconds"],
                "current_seconds": new["seconds"],
                "bit_identical": bool(
                    np.array_equal(old["omega"], new["omega"])
                    and np.array_equal(old["residue"], new["residue"])
                    and all(np.array_equal(old[k], new[k])
                            for k in ("cond", "backward", "n_valid"))),
                "changed_valid_count": int(np.count_nonzero(
                    old["n_valid"] != new["n_valid"])),
                "tiny_residue_elements": int(np.count_nonzero(
                    np.max(np.abs(new["residue"]), axis=1) < 1.0e-10)),
                "near_degenerate_elements": int(np.count_nonzero(
                    np.min(np.where(
                        np.eye(n_p, dtype=bool)[None], np.inf,
                        np.abs(new["omega"][:, :, None]
                               - new["omega"][:, None, :])), axis=(1, 2))
                    < 1.0e-8)),
            }
            all_rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    out = os.path.join(args.output_dir, f"census.rank{rank}.json")
    with open(out, "w", encoding="utf-8") as stream:
        json.dump(all_rows, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
