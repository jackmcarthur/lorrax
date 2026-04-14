"""psp/run_nscf.py — NSCF driver: QE .save → Davidson → WFN.h5.

Reads QE .save (crystal + charge density), builds the DFT Hamiltonian,
solves for eigenvalues at all k-points via Davidson, writes BGW-compatible WFN.h5.

2D Coulomb truncation is auto-detected from QE's assume_isolated setting.

Usage:
    module load lorrax
    lxrun python3 -u -m psp.run_nscf --save qe/nscf/silicon.save --nbnd 100 --nk 4 4 4
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import functools
import time

import numpy as np
import jax
import jax.numpy as jnp

from psp.qe_save_reader import CrystalData
from psp.pseudos import load_pseudopotentials
from psp.ionic_gspace import build_ionic_and_core
from psp.dft_operators import (
    build_G_cart, compute_V_H_and_V_xc, build_V_scf,
    setup_H_k_from_kvec, apply_H_k,
)
from psp.gvec_utils import (
    build_master_gvec_list, select_gvecs_for_k,
    compute_ngkmax, reorder_to_qe,
)
from psp.davidson import davidson_k, warmup_jit
from psp.wfn_writer import WFNWriter
import psp.vnl_ops as vnl_ops


# ---------------------------------------------------------------------------
# Sparse-G H|ψ⟩ wrapper
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("_nx", "_ny", "_nz"))
def _apply_H_sparse(psi_G, T, V, Gx, Gy, Gz, Z, E, mask, _nx, _ny, _nz):
    """Sparse-G → sparse-G H|ψ⟩.  Single JIT trace for all k-points."""
    mask_f = mask[None, None, :].astype(psi_G.dtype)
    psi_box = jnp.zeros((*psi_G.shape[:2], _nx, _ny, _nz), dtype=psi_G.dtype)
    psi_box = psi_box.at[:, :, Gx, Gy, Gz].add(psi_G * mask_f)
    return apply_H_k(psi_box, T, V, Gx, Gy, Gz, Z, E, mask)


# ---------------------------------------------------------------------------
# NSCF driver
# ---------------------------------------------------------------------------

def run_nscf(
    crystal: CrystalData,
    pseudos: dict,
    kgrid: tuple[int, int, int],
    nbnd: int,
    output_path: str = "WFN.h5",
    *,
    tol: float = 1e-8,
    verbose: bool = True,
    kpoints_override: np.ndarray | None = None,
    weights_override: np.ndarray | None = None,
):
    """NSCF: build DFT potentials, solve Davidson at each k, write WFN.h5."""
    truncation_2d = crystal.assume_isolated == "2D"
    fft_grid = crystal.fft_grid
    nspinor = crystal.nspinor
    _nx, _ny, _nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
    t0 = time.perf_counter()

    if verbose:
        print(f"NSCF: {nbnd} bands, kgrid={kgrid}, fft={fft_grid}, nspinor={nspinor}")
        if truncation_2d:
            print(f"  2D Coulomb truncation (from assume_isolated='2D')")
        print(f"  GPUs: {len(jax.devices())}")

    # ── k-independent potential ─────────────────────────────────
    V_loc, rho_core, rho_core_G = build_ionic_and_core(
        crystal, pseudos, fft_grid, truncation_2d=truncation_2d)

    rho_val = jnp.asarray(crystal.load_charge_density()[0], dtype=jnp.float64)

    G_cart = build_G_cart(_nx, _ny, _nz,
                          float(crystal.blat) * np.asarray(crystal.bvec, dtype=float))
    V_H, V_xc = compute_V_H_and_V_xc(
        rho_val, rho_core, rho_core_G, G_cart,
        jnp.asarray(crystal.bdot, dtype=jnp.float64),
        jnp.asarray(crystal.bvec, dtype=jnp.float64), crystal.blat,
        truncation_2d=truncation_2d)
    V_scf = build_V_scf(V_loc, V_H, V_xc)

    vnl_setup = vnl_ops.build_vnl_setup(
        crystal, pseudos=pseudos, nspinor=nspinor,
        q_max=float(np.sqrt(float(crystal.ecutwfc))) * 1.01)
    if verbose:
        print(f"  Potentials: {time.perf_counter()-t0:.2f}s")

    # ── k-point grid + G-vector bookkeeping ─────────────────────
    if kpoints_override is not None:
        kpoints = np.asarray(kpoints_override, dtype=np.float64)
        weights = np.asarray(weights_override, dtype=np.float64)
    else:
        kpoints, weights = crystal.build_kgrid(
            nk=kgrid, nosym=True, noinv=True, no_t_rev=True, force_symmorphic=False)
        kpoints = np.asarray(kpoints, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
    nk = len(kpoints)

    G_master, _ = build_master_gvec_list(crystal)
    bdot = np.asarray(crystal.bdot, dtype=float)
    ngkmax = compute_ngkmax(kpoints, bdot, crystal.ecutwfc, fft_grid)
    gvecs_per_k = [select_gvecs_for_k(kpoints[ik], G_master, bdot, crystal.ecutwfc)[0]
                   for ik in range(nk)]
    if verbose:
        print(f"  nk={nk}, ngkmax={ngkmax}")

    # ── JIT warmup ──────────────────────────────────────────────
    t1 = time.perf_counter()
    warmup_jit(ngkmax, nspinor, nbnd)
    H_k0 = setup_H_k_from_kvec(kpoints[0], V_scf, vnl_setup, crystal, None,
                                 V_loc_r=V_loc, ngkmax=ngkmax)
    dummy = jnp.zeros((1, nspinor, ngkmax), dtype=jnp.complex128)
    _apply_H_sparse(dummy, H_k0.T_diag, H_k0.V_scf,
                    H_k0.Gx, H_k0.Gy, H_k0.Gz,
                    H_k0.vnl_Z, H_k0.vnl_E, H_k0.mask, _nx, _ny, _nz)
    if verbose:
        print(f"  JIT warmup: {time.perf_counter()-t1:.2f}s")

    # ── Open output file ────────────────────────────────────────
    writer = WFNWriter(output_path, crystal, kpoints, weights, kgrid, nbnd,
                        gvecs_per_k, nosym=True)

    # ── per-k: build H_k → Davidson → write ────────────────────
    eigenvalues = np.zeros((nk, nbnd))
    t_dav = time.perf_counter()

    for ik in range(nk):
        tk = time.perf_counter()
        H_k = setup_H_k_from_kvec(kpoints[ik], V_scf, vnl_setup, crystal, None,
                                    V_loc_r=V_loc, ngkmax=ngkmax)

        def apply_H(psi_G, _H=H_k):
            return _apply_H_sparse(psi_G, _H.T_diag, _H.V_scf,
                                    _H.Gx, _H.Gy, _H.Gz,
                                    _H.vnl_Z, _H.vnl_E, _H.mask,
                                    _nx, _ny, _nz)

        evals, evecs = davidson_k(
            apply_H, h_diag=H_k.h_diag, nG=ngkmax, nspinor=nspinor,
            n_tgt=nbnd, T_diag=H_k.T_diag, verbose=False, tol=tol)

        eigenvalues[ik] = evals
        writer.write_k(ik, evals, reorder_to_qe(np.asarray(evecs), H_k, gvecs_per_k[ik]))

        if verbose and (ik < 3 or ik == nk - 1 or (ik + 1) % 16 == 0):
            print(f"  k={ik:3d}/{nk}: {time.perf_counter()-tk:.3f}s  "
                  f"evals[0]={evals[0]:.6f} Ry")

    writer.close()
    if verbose:
        dt = time.perf_counter() - t_dav
        print(f"  Davidson + write: {dt:.2f}s ({dt/nk:.3f}s/k)")
        print(f"  Total NSCF: {time.perf_counter()-t0:.2f}s → {output_path}")
    return eigenvalues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LORRAX NSCF: Davidson → WFN.h5")
    parser.add_argument("--save", required=True, help="QE .save directory")
    parser.add_argument("--pseudo_dir", default=None,
                        help="Directory with .upf (default: same as --save)")
    parser.add_argument("--nbnd", type=int, default=100, help="Number of bands")
    parser.add_argument("--nk", type=int, nargs=3, default=[4, 4, 4],
                        help="K-grid dimensions")
    parser.add_argument("-o", "--output", default="WFN.h5", help="Output WFN.h5 path")
    parser.add_argument("--tol", type=float, default=1e-8,
                        help="Davidson convergence tolerance")
    parser.add_argument("--ref_wfn", default=None,
                        help="Reference WFN.h5 to read k-points from (for validation)")
    args = parser.parse_args()

    crystal = CrystalData.from_qe_save(args.save)
    pseudos = load_pseudopotentials(args.pseudo_dir or args.save)

    kpoints_override = weights_override = None
    if args.ref_wfn:
        import h5py
        with h5py.File(args.ref_wfn, "r") as f:
            kpoints_override = f["mf_header/kpoints/rk"][:]
            weights_override = f["mf_header/kpoints/w"][:]

    run_nscf(crystal, pseudos, kgrid=tuple(args.nk), nbnd=args.nbnd,
             output_path=args.output, tol=args.tol,
             kpoints_override=kpoints_override, weights_override=weights_override)


if __name__ == "__main__":
    main()
