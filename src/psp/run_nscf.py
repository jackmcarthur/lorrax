"""psp/run_nscf.py — NSCF driver: QE .save → Davidson → WFN.h5.

Usage with input file:
    lxrun python3 -u -m psp.run_nscf -i nscf.in

Usage with CLI args (backwards-compatible):
    lxrun python3 -u -m psp.run_nscf --save QE.save --nbnd 100 --nk 4 4 4

Input file format (nscf.in):
    [nscf]
    save_dir = qe/nscf/silicon.save
    nbnd = 100
    kgrid = 4 4 4
    nosym = false        # true = full grid, false = IBZ-reduced (default)
    output = WFN.h5
    tol = 1e-8
    charge_from_wfn = false
    wfn_file = WFN_ref.h5
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp

from file_io import CrystalData, WFNWriter
from psp.pseudos import load_pseudopotentials
from psp.ionic_gspace import build_ionic_and_core
from psp.dft_operators import build_G_cart, compute_V_H_and_V_xc, build_V_scf
from psp.h_dft import setup_H_k_from_kvec, make_apply_H
from psp.dft_precond import make_dft_preconditioner, make_pw_init
from psp.gvec_utils import (
    build_master_gvec_list, select_gvecs_for_k,
    compute_ngkmax, reorder_to_qe,
)
from solvers.davidson import davidson, warmup_davidson_jit
import psp.vnl_ops as vnl_ops


def run_nscf(
    crystal: CrystalData,
    pseudos: dict,
    kgrid: tuple[int, int, int],
    nbnd: int,
    output_path: str = "WFN.h5",
    *,
    nosym: bool = False,
    tol: float = 1e-8,
    verbose: bool = True,
    kpoints_override: np.ndarray | None = None,
    weights_override: np.ndarray | None = None,
):
    """NSCF: build DFT potentials, solve Davidson at each k, write WFN.h5.

    By default, the k-grid is reduced to the IBZ using crystal symmetries
    from the QE .save.  Set nosym=True to use the full unreduced grid.
    """
    truncation_2d = crystal.assume_isolated == "2D"
    fft_grid = crystal.fft_grid
    nspinor = crystal.nspinor
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
    nx, ny, nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
    G_cart = build_G_cart(nx, ny, nz,
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

    # ── k-point grid ────────────────────────────────────────────
    if kpoints_override is not None:
        kpoints = np.asarray(kpoints_override, dtype=np.float64)
        weights = np.asarray(weights_override, dtype=np.float64)
    else:
        kpoints, weights = crystal.build_kgrid(
            nk=kgrid, nosym=nosym, noinv=nosym, no_t_rev=nosym)
        kpoints = np.asarray(kpoints, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
    nk = len(kpoints)

    G_master, _ = build_master_gvec_list(crystal)
    bdot = np.asarray(crystal.bdot, dtype=float)
    ngkmax = compute_ngkmax(kpoints, bdot, crystal.ecutwfc, fft_grid)
    gvecs_per_k = [select_gvecs_for_k(kpoints[ik], G_master, bdot, crystal.ecutwfc)[0]
                   for ik in range(nk)]
    if verbose:
        grid_label = "full" if nosym else "IBZ"
        print(f"  nk={nk} ({grid_label}), ngkmax={ngkmax}")

    # ── JIT warmup ──────────────────────────────────────────────
    t1 = time.perf_counter()
    warmup_davidson_jit(nbnd, ngkmax, nspinor)
    H_k0 = setup_H_k_from_kvec(kpoints[0], V_scf, vnl_setup, crystal, None,
                                 V_loc_r=V_loc, ngkmax=ngkmax)
    apply_H0 = make_apply_H(H_k0)
    m_max = 4 * nbnd
    for m in range(nbnd, m_max + nbnd, nbnd):
        apply_H0(jnp.zeros((min(m, m_max), nspinor, ngkmax), dtype=jnp.complex128))
    if verbose:
        print(f"  JIT warmup: {time.perf_counter()-t1:.2f}s")

    # ── Open output file ────────────────────────────────────────
    writer = WFNWriter(output_path, crystal, kpoints, weights, kgrid, nbnd,
                        gvecs_per_k, nosym=nosym)

    # ── per-k: build H_k → Davidson → write ────────────────────
    eigenvalues = np.zeros((nk, nbnd))
    t_dav = time.perf_counter()

    for ik in range(nk):
        tk = time.perf_counter()
        H_k = setup_H_k_from_kvec(kpoints[ik], V_scf, vnl_setup, crystal, None,
                                    V_loc_r=V_loc, ngkmax=ngkmax)
        apply_H = make_apply_H(H_k)
        precond = make_dft_preconditioner(H_k.h_diag)
        init = make_pw_init(H_k.T_diag, nspinor, verbose=False)

        evals, evecs = davidson(
            apply_H, n_eig=nbnd, precond_fn=precond, init_fn=init,
            verbose=False, tol=tol)

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


def main():
    parser = argparse.ArgumentParser(description="LORRAX NSCF: Davidson → WFN.h5")
    parser.add_argument("-i", "--input", default=None, help="Input file (nscf.in)")
    parser.add_argument("--save", default=None, help="QE .save directory")
    parser.add_argument("--pseudo_dir", default=None,
                        help="Directory with .upf (default: same as --save)")
    parser.add_argument("--nbnd", type=int, default=None, help="Number of bands")
    parser.add_argument("--nk", type=int, nargs=3, default=None, help="K-grid dimensions")
    parser.add_argument("--nosym", action="store_true", help="Disable IBZ reduction")
    parser.add_argument("-o", "--output", default=None, help="Output WFN.h5 path")
    parser.add_argument("--tol", type=float, default=None, help="Davidson convergence tol")
    parser.add_argument("--ref_wfn", default=None,
                        help="Reference WFN.h5 for k-points (overrides kgrid)")
    args = parser.parse_args()

    # Input file takes precedence; CLI args override individual fields
    if args.input:
        from psp.nscf_input import read_nscf_input
        inp = read_nscf_input(args.input)
        save_dir = args.save or inp.save_dir
        nbnd = args.nbnd or inp.nbnd
        kgrid = tuple(args.nk) if args.nk else inp.kgrid
        nosym = args.nosym or inp.nosym
        output = args.output or inp.output
        tol = args.tol or inp.tol
    else:
        if not args.save:
            parser.error("Either -i input_file or --save is required")
        save_dir = args.save
        nbnd = args.nbnd or 100
        kgrid = tuple(args.nk) if args.nk else (4, 4, 4)
        nosym = args.nosym
        output = args.output or "WFN.h5"
        tol = args.tol or 1e-8

    crystal = CrystalData.from_qe_save(save_dir)
    pseudos = load_pseudopotentials(args.pseudo_dir or save_dir)

    kpoints_override = weights_override = None
    if args.ref_wfn:
        import h5py
        with h5py.File(args.ref_wfn, "r") as f:
            kpoints_override = f["mf_header/kpoints/rk"][:]
            weights_override = f["mf_header/kpoints/w"][:]

    run_nscf(crystal, pseudos, kgrid=kgrid, nbnd=nbnd,
             output_path=output, nosym=nosym, tol=tol,
             kpoints_override=kpoints_override, weights_override=weights_override)


if __name__ == "__main__":
    main()
