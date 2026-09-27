"""psp/run_dense_h.py — the complete mean-field basis: dense H_k, full eigh, every band.

For each k of a source ``WFN.h5`` the Kohn–Sham Hamiltonian of the QE run is
built as an explicit matrix on the whole plane-wave sphere,

    H_k(sG, s'G') = |k+G|² δ δ' + V_scf(G−G') δ_ss' + Σ_RR' Z_R(G) D^{ss'}_RR' Z*_R'(G'),

with |k+G|² ≤ ecutwfc, s the spinor index (nspinor = 2 under SOC) and
V_scf = V_loc + V_H[ρ] + V_xc[ρ + ρ_core] from the SCF density.  It is
diagonalized completely,

    H_k c_nk = ε_nk c_nk,   n = 1 … nspinor·ngk(k),

and ``min_k nspinor·ngk(k)`` bands are written on the source's k-set,
symmetry and G-lists (``file_io.qp_wfn.write_complete_wfn_h5``): a drop-in
WFN whose band sum is complete.

Each process takes whole k-points (k ≡ rank mod P) and solves them with a
local eigh; nothing is distributed inside one k.  Peak device bytes per k
are ``psp.operator_checks.dense_h_bytes`` = 5·N²·16 with N = nspinor·ngk,
refused against the device budget before any heavy work, so this is a
small-cell route.  The operator is ``psp.dft_operators``' (the Davidson
route's), applied to the unit basis; it is not assembled a second time.

Usage (one rank per GPU):
    lx run ... -- python3 -u -m psp.run_dense_h --save QE.save \\
        --wfn WFN.h5 -o WFN_complete.h5 --sys-dim 3
"""
from __future__ import annotations

from runtime import debug_print, initialize_communicator_stack
RUNTIME = initialize_communicator_stack(print_fn=debug_print)

import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

import distrib_la
from common.collectives import (process_count, process_rank, psum_replicate,
                                single_device_mesh)
from common.gpu_utils import device_budget_bytes, set_device_budget_gb
from file_io import CrystalData
from file_io.qp_wfn import write_complete_wfn_h5
from psp.dft_operators import dense_matrix_k, setup_H_k_from_kvec
from psp.gvec_utils import reorder_to_qe
from psp.operator_checks import validate_dense_h_inputs
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import build_dft_potentials
from wfn_loader import WfnLoader

#: A Hermitian operator: max|H − H†| relative to max|H| above this refuses.
HERMITICITY_TOL = 1e-12


def solve_k(H_k, gvecs_file, nspinor, eigh, nbands):
    """ε and c for one k: the dense H on its (padded) sphere, one full eigh, source G order."""
    if H_k.nG != gvecs_file.shape[0]:
        raise ValueError(
            f"the ecutwfc sphere holds {H_k.nG} G but the source WFN holds "
            f"{gvecs_file.shape[0]}; the .save and the WFN are not one "
            f"calculation.")
    ngkmax = int(H_k.mask.shape[0])
    H, h_pad = dense_matrix_k(H_k.T_diag, H_k.V_scf, H_k.Gx, H_k.Gy, H_k.Gz,
                              H_k.vnl_Z, H_k.vnl_E, H_k.mask, nspinor=nspinor)
    skew = float(jnp.max(jnp.abs(H - jnp.conj(H.T))) / h_pad)
    if not skew <= HERMITICITY_TOL:
        raise RuntimeError(
            f"dense H_k is not Hermitian: max|H - H^H|/||H|| = {skew:.2e} "
            f"> {HERMITICITY_TOL:.0e}.")
    energies, vectors = eigh(H)                    # eigenvectors as columns
    psi = np.asarray(vectors[:, :nbands].T).reshape(nbands, nspinor, ngkmax)
    leak = float(np.max(np.abs(psi[:, :, H_k.nG:]), initial=0.0))
    if not leak <= 1e-10:
        raise RuntimeError(
            f"dense H_k: a physical eigenvector has weight {leak:.1e} on the "
            f"padded G block; the pad did not separate from the spectrum.")
    return (np.asarray(energies[:nbands]), reorder_to_qe(psi, H_k, gvecs_file),
            nspinor * H_k.nG, skew)


def run_dense_h(save_dir, wfn_path, output_path, *, sys_dim, pseudo_dir=None,
                nbands=None):
    rank, nproc = process_rank(), process_count()
    verbose = rank == 0
    mesh = RUNTIME.mesh
    crystal = CrystalData.from_qe_save(save_dir)
    pseudo_dir = pseudo_dir or save_dir
    pseudos = load_pseudopotentials(pseudo_dir)

    with WfnLoader(wfn_path, mesh=mesh) as wfn:
        crystal.validate_against_wfn(wfn)
        nk = int(wfn.nkpts)
        kpoints = np.asarray(wfn.kpoints, dtype=np.float64)
        ngk = np.asarray(wfn.ngk_valid(k="ibz"), dtype=np.int64)
        gvecs = np.asarray(wfn.gvecs(k="ibz"))
        n_basis = int(crystal.nspinor) * ngk
        nb = int(n_basis.min()) if nbands is None else int(nbands)
        validate_dense_h_inputs(
            crystal, pseudos, sys_dim=sys_dim, pseudo_dir=pseudo_dir,
            charge_density_fields=crystal.charge_density_fields(),
            n_basis_max=int(crystal.nspinor) * int(ngk.max()),
            budget_bytes=device_budget_bytes())
        if not 0 < nb <= int(n_basis.min()):
            raise ValueError(
                f"nbands={nb}: the complete basis of the smallest sphere is "
                f"{int(n_basis.min())} states (nspinor·min_k ngk).")
        if verbose:
            print(f"dense H: nk={nk}, nspinor={crystal.nspinor}, N = "
                  f"nspinor·ngk in [{int(n_basis.min())}, {int(n_basis.max())}], "
                  f"writing {nb} bands, {nproc} rank(s)", flush=True)

        rho_val = jnp.asarray(crystal.load_charge_density()[0], jnp.float64)
        V_scf, V_loc, vnl_setup = build_dft_potentials(
            crystal, pseudos, rho_val,
            truncation_2d=crystal.assume_isolated == "2D", verbose=verbose)
        eigh = distrib_la.plan("eigh", single_device_mesh(), backend="off")

        # Round r: rank p solves k = r·P + p.  Every rank sets up every k of
        # the round in the same order (the sphere selection has k-dependent
        # shapes before padding: cross-rank compile agreement) and keeps its
        # own; a rank past nk solves the round's last k and discards it.
        energies = np.zeros((nk, nb))
        coefficients = {}
        ngkmax = int(ngk.max())
        for first in range(0, nk, nproc):
            t0 = time.perf_counter()
            ik = first + rank
            for jk in range(first, min(first + nproc, nk)):
                H_jk = setup_H_k_from_kvec(
                    kpoints[jk], V_scf, vnl_setup, crystal, None,
                    V_loc_r=V_loc, ngkmax=ngkmax)
                if jk == min(ik, nk - 1):
                    H_k, k_run = H_jk, jk
            e_k, c_k, n, skew = solve_k(
                H_k, gvecs[k_run, :ngk[k_run]], int(crystal.nspinor), eigh, nb)
            if ik >= nk:
                continue
            energies[ik], coefficients[ik] = e_k, c_k
            stats = jax.local_devices()[0].memory_stats() or {}
            print(f"  [rank {rank}] k={ik}: N={n} (padded "
                  f"{int(crystal.nspinor) * ngkmax}), "
                  f"{time.perf_counter() - t0:.2f} s, peak "
                  f"{stats.get('peak_bytes_in_use', 0) / 1e9:.3f} GB, "
                  f"skew {skew:.1e}, e0={energies[ik, 0]:.8f} Ry", flush=True)
        energies = psum_replicate(energies, mesh)
        write_complete_wfn_h5(
            output_path, wfn, energies, coefficients, mesh=mesh,
            stamps={"dense_h_source_wfn": os.path.abspath(wfn_path),
                    "dense_h_qe_save": os.path.abspath(save_dir)})
    if verbose:
        print(f"wrote {output_path}: {nk} k x {nb} bands", flush=True)
    return energies


def main():
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Dense H_k on the full G-sphere, complete eigh -> WFN.h5")
    parser.add_argument("--save", required=True, help="QE .save directory")
    parser.add_argument("--wfn", required=True,
                        help="source WFN.h5: k-set, symmetry and G-lists")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--sys-dim", type=int, required=True, choices=(0, 2, 3))
    parser.add_argument("--pseudo-dir", default=None)
    parser.add_argument("--nbands", type=int, default=None,
                        help="bands to write (default: the complete basis)")
    parser.add_argument("--memory-per-device-gb", type=float, default=None)
    args = parser.parse_args()
    if args.memory_per_device_gb:
        set_device_budget_gb(args.memory_per_device_gb)
    run_dense_h(args.save, args.wfn, args.output, sys_dim=args.sys_dim,
                pseudo_dir=args.pseudo_dir, nbands=args.nbands)


if __name__ == "__main__":
    raise SystemExit(main())
