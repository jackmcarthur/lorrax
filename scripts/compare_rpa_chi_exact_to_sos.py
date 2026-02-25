#!/usr/bin/env python3
"""Compare Casida/GMRES chi(0) columns against SOS+Dyson reference (q=0, static).

This is intended to validate the --rpa path in:
  python -m isdf.bse_isdf.bse_w_exact ... --drive-kind potential --write-kind chi

It reconstructs chi0 from a restart file (SOS) and forms the interacting
chi via the Dyson equation:
  (I - chi0 V) chi = chi0

Then it compares requested columns against the HDF5 output.
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

import jax
from jax.sharding import Mesh

from isdf.bse_isdf.bse_io import _find_restart_file, load_bse_data_from_restart_sharded


def _make_1x1_mesh() -> Mesh:
    devices = jax.devices()
    return Mesh(np.array([devices[0]]).reshape(1, 1), axis_names=("x", "y"))


def _compute_chi_sos_dyson(*, data: dict) -> np.ndarray:
    psi_v = np.asarray(jax.device_get(data["psi_v_X"]))  # (nk, nv, ns, mu)
    psi_c = np.asarray(jax.device_get(data["psi_c_X"]))  # (nk, nc, ns, mu)
    eps_v = np.asarray(jax.device_get(data["eps_v"]))[:, : int(data["n_val"])]
    eps_c = np.asarray(jax.device_get(data["eps_c"]))[:, : int(data["n_cond"])]
    V = np.asarray(jax.device_get(data["V_q0"]))

    nk = int(psi_v.shape[0])
    n_rmu = int(psi_v.shape[3])

    # SOS transition vertex (matches RPA_W0_BUG_GUIDE.md).
    # M[k,c,v,mu] = sum_s conj(psi_v[k,v,s,mu]) * psi_c[k,c,s,mu]
    M = np.einsum("kvsM,kcsM->kcvM", np.conj(psi_v), psi_c)
    DE = eps_c[:, :, None] - eps_v[:, None, :]  # (nk, nc, nv)

    chi0_raw = np.einsum("kcvM,kcv,kcvN->MN", M, 1.0 / DE, np.conj(M))
    chi0 = -(2.0 / float(nk)) * chi0_raw

    I = np.eye(n_rmu, dtype=np.complex128)
    chi = np.linalg.solve(I - chi0 @ V, chi0)
    return chi


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chi columns (Casida exact vs SOS+Dyson)")
    parser.add_argument("-i", "--input", required=True, help="COHSEX input file (for restart lookup)")
    parser.add_argument("--bse", required=True, help="HDF5 output from bse_w_exact (--write-kind chi)")
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--n-cond", type=int, default=4)
    parser.add_argument("--nohead", action="store_true", help="Use headless V if present in restart file.")
    args = parser.parse_args()

    restart_file = _find_restart_file(args.input)
    mesh = _make_1x1_mesh()

    data = load_bse_data_from_restart_sharded(
        restart_file,
        n_val=args.n_val,
        n_cond=args.n_cond,
        mesh_xy=mesh,
        use_nohead=args.nohead,
    )

    chi_ref = _compute_chi_sos_dyson(data=data)

    with h5py.File(args.bse, "r") as h5:
        cols = np.array(h5["columns"][:], dtype=int)
        if "chi" not in h5:
            raise KeyError(f"{args.bse} does not contain dataset 'chi' (did you run --write-kind chi?)")
        chi_cols = np.array(h5["chi"][:])

    print(f"Restart: {restart_file}")
    print(f"Columns: {cols.tolist()}")

    for i, nu in enumerate(cols):
        ref = chi_ref[:, nu]
        got = chi_cols[i]
        rel = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-14)
        max_abs = float(np.max(np.abs(got - ref)))
        print(f"nu={nu:4d}  rel_frob={rel:.3e}  max_abs={max_abs:.3e}")


if __name__ == "__main__":
    main()

