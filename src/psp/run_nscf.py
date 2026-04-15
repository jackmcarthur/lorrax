"""psp/run_nscf.py — NSCF driver: Davidson eigenstates + CJ pseudobands → WFN.h5.

Three modes of operation:

  1. Davidson only (default):
     Solve for nbnd eigenstates → WFN.h5

  2. Davidson + pseudobands:
     Solve for nbnd eigenstates, then fill high-energy spectrum with
     CJ-filtered Ritz pseudobands → WFN.h5 + WFN_pseudobands.h5

  3. Pseudobands only (from existing WFN.h5):
     Read deterministic bands from an existing WFN.h5, build H@ψ from
     the QE .save, and run only the pseudobands stage → WFN_pseudobands.h5.
     Requires the WFN.h5 to have bands exceeding the number of valence electrons.

Usage:
    lxrun python3 -u -m psp.run_nscf -i nscf.in
    lxrun python3 -u -m psp.run_nscf --save QE.save --nbnd 100 --nk 4 4 4
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


# ═══════════════════════════════════════════════════════════════════════
#  Potential construction (shared by all modes)
# ═══════════════════════════════════════════════════════════════════════

def _build_potentials(crystal, pseudos, verbose=True):
    """Build k-independent DFT potentials from crystal + pseudos."""
    truncation_2d = crystal.assume_isolated == "2D"
    fft_grid = crystal.fft_grid
    nspinor = crystal.nspinor
    t0 = time.perf_counter()

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

    return V_scf, V_loc, vnl_setup


# ═══════════════════════════════════════════════════════════════════════
#  K-grid setup (shared)
# ═══════════════════════════════════════════════════════════════════════

def _setup_kgrid(crystal, kgrid, nosym, kpoints_override, weights_override, verbose=True):
    """Build k-point grid and G-vector bookkeeping."""
    if kpoints_override is not None:
        kpoints = np.asarray(kpoints_override, dtype=np.float64)
        weights = np.asarray(weights_override, dtype=np.float64)
    else:
        kpoints, weights = crystal.build_kgrid(
            nk=kgrid, nosym=nosym, noinv=nosym, no_t_rev=nosym)
        kpoints = np.asarray(kpoints, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

    G_master, _ = build_master_gvec_list(crystal)
    bdot = np.asarray(crystal.bdot, dtype=float)
    ngkmax = compute_ngkmax(kpoints, bdot, crystal.ecutwfc, crystal.fft_grid)
    gvecs_per_k = [select_gvecs_for_k(kpoints[ik], G_master, bdot, crystal.ecutwfc)[0]
                   for ik in range(len(kpoints))]

    if verbose:
        label = "full" if nosym else "IBZ"
        print(f"  nk={len(kpoints)} ({label}), ngkmax={ngkmax}")

    return kpoints, weights, ngkmax, gvecs_per_k


# ═══════════════════════════════════════════════════════════════════════
#  Pseudobands helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_flat_matvec(apply_H_batched, nspinor, ngkmax):
    """Wrap batched (batch, nspinor, dim) matvec → flat (nspinor*dim,) → same."""
    def apply_H_flat(x):
        return apply_H_batched(x.reshape(1, nspinor, ngkmax)).reshape(-1)
    return apply_H_flat


def _write_pb_k(writer, ik, pb_result, H_k, gvecs_qe, nspinor, ngkmax):
    """Write one k-point's pseudobands to WFN.h5.

    Reshapes flat (n_total, nspinor*ngkmax) → (n_total, nspinor, ngkmax),
    reorders to QE G-vector convention, and writes eigenvalues + coefficients.
    """
    n_total = pb_result.Phi_out.shape[0]
    # Reshape flat → (n_total, nspinor, ngkmax)
    evecs_3d = pb_result.Phi_out.reshape(n_total, nspinor, ngkmax)
    # Reorder G-vectors from |k+G|²-sorted (Davidson) to QE master order
    coeffs_qe = reorder_to_qe(evecs_3d, H_k, gvecs_qe)
    writer.write_k(ik, pb_result.E_out, coeffs_qe)


# ═══════════════════════════════════════════════════════════════════════
#  Main driver
# ═══════════════════════════════════════════════════════════════════════

def run_nscf(
    crystal: CrystalData,
    pseudos: dict,
    kgrid: tuple[int, int, int],
    nbnd: int,
    output_path: str = "WFN.h5",
    *,
    nosym: bool = False,
    tol: float = 1e-8,
    do_davidson: bool = True,
    do_pseudobands: bool = False,
    pb_k: int = 6,
    pb_M_max: int = 1500,
    pb_F: float = 0.10,
    pb_n_windows: int = 50,
    pb_wfn_input: str | None = None,
    verbose: bool = True,
    kpoints_override: np.ndarray | None = None,
    weights_override: np.ndarray | None = None,
):
    """NSCF driver: Davidson eigenstates and/or CJ pseudobands.

    Modes:
      do_davidson=True,  do_pseudobands=False  → standard NSCF → WFN.h5
      do_davidson=True,  do_pseudobands=True   → NSCF + pseudobands
      do_davidson=False, do_pseudobands=True   → pseudobands from existing WFN.h5
      do_davidson=False, do_pseudobands=False   → error
    """
    if not do_davidson and not do_pseudobands:
        raise ValueError("At least one of do_davidson or do_pseudobands must be True")

    truncation_2d = crystal.assume_isolated == "2D"
    fft_grid = crystal.fft_grid
    nspinor = crystal.nspinor
    t_start = time.perf_counter()

    if verbose:
        mode = []
        if do_davidson:
            mode.append(f"Davidson ({nbnd} bands)")
        if do_pseudobands:
            mode.append(f"pseudobands (k={pb_k}, M={pb_M_max}, {pb_n_windows} windows)")
        print(f"NSCF: {' + '.join(mode)}")
        print(f"  kgrid={kgrid}, fft={fft_grid}, nspinor={nspinor}")
        if truncation_2d:
            print(f"  2D Coulomb truncation (from assume_isolated='2D')")
        print(f"  GPUs: {len(jax.devices())}")

    # ── Build potentials ────────────────────────────────────────
    V_scf, V_loc, vnl_setup = _build_potentials(crystal, pseudos, verbose)

    # ── K-grid ──────────────────────────────────────────────────
    kpoints, weights, ngkmax, gvecs_per_k = _setup_kgrid(
        crystal, kgrid, nosym, kpoints_override, weights_override, verbose)
    nk = len(kpoints)

    # ══════════════════════════════════════════════════════════════
    #  Davidson stage
    # ══════════════════════════════════════════════════════════════

    eigenvalues = None
    all_evecs = None  # (nk, nbnd, nspinor, ngkmax) if needed for pseudobands

    if do_davidson:
        if verbose:
            print("\n── Davidson ──")

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

        writer = WFNWriter(output_path, crystal, kpoints, weights, kgrid, nbnd,
                            gvecs_per_k, nosym=nosym)

        eigenvalues = np.zeros((nk, nbnd))
        if do_pseudobands:
            all_evecs = np.zeros((nk, nbnd, nspinor, ngkmax), dtype=np.complex128)

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
            evecs_np = np.asarray(evecs)
            writer.write_k(ik, evals, reorder_to_qe(evecs_np, H_k, gvecs_per_k[ik]))

            if do_pseudobands:
                all_evecs[ik] = evecs_np

            if verbose and (ik < 3 or ik == nk - 1 or (ik + 1) % 16 == 0):
                print(f"  k={ik:3d}/{nk}: {time.perf_counter()-tk:.3f}s  "
                      f"evals[0]={evals[0]:.6f} Ry")

        writer.close()
        if verbose:
            dt = time.perf_counter() - t_dav
            print(f"  Davidson: {dt:.2f}s ({dt/nk:.3f}s/k) → {output_path}")

    # ══════════════════════════════════════════════════════════════
    #  Pseudobands-only: load deterministic bands from WFN.h5
    # ══════════════════════════════════════════════════════════════

    if not do_davidson and do_pseudobands:
        import h5py

        wfn_path = pb_wfn_input
        if wfn_path is None:
            raise ValueError(
                "do_pseudobands=True without do_davidson requires pb_wfn_input "
                "(path to existing WFN.h5 with deterministic bands)")
        if not os.path.isfile(wfn_path):
            raise FileNotFoundError(f"pb_wfn_input not found: {wfn_path}")

        if verbose:
            print(f"\n── Loading deterministic bands from {wfn_path} ──")

        with h5py.File(wfn_path, "r") as f:
            nk_file = int(f["mf_header/kpoints/nrk"][()])
            nbnd_file = int(f["mf_header/kpoints/mnband"][()])
            eigenvalues = f["mf_header/kpoints/el"][0]  # (nk, nbnd)
            kpoints_file = f["mf_header/kpoints/rk"][:]
            weights_file = f["mf_header/kpoints/w"][:]

            # Validate
            n_occ = int(crystal.nelec) // 2 if nspinor == 1 else int(crystal.nelec)
            if nbnd_file <= n_occ:
                raise ValueError(
                    f"WFN.h5 has {nbnd_file} bands but system has {n_occ} "
                    f"occupied states. Need bands > n_occupied for pseudobands.")

            # Read wavefunction coefficients
            coeffs = f["wfns/coeffs"]  # (nbnd, nspinor, ngk_max, 2)
            all_evecs = np.zeros((nk_file, nbnd_file, nspinor, ngkmax), dtype=np.complex128)
            for ib in range(nbnd_file):
                c = coeffs[ib]  # (nspinor, ngk, 2)
                c_complex = c[:, :, 0] + 1j * c[:, :, 1]
                # Pad to ngkmax if needed
                ngk_file = c_complex.shape[1]
                all_evecs[0, ib, :, :ngk_file] = c_complex

        nbnd = nbnd_file
        kpoints = kpoints_file
        weights = weights_file
        nk = nk_file
        kpoints, weights, ngkmax, gvecs_per_k = _setup_kgrid(
            crystal, kgrid, nosym, kpoints, weights, verbose)

        if verbose:
            print(f"  Loaded {nbnd} bands, {nk} k-points from {wfn_path}")

    # ══════════════════════════════════════════════════════════════
    #  Pseudobands stage
    # ══════════════════════════════════════════════════════════════

    if do_pseudobands:
        from solvers.pseudobands import ritz_pseudobands

        if verbose:
            print(f"\n── Pseudobands (k={pb_k}, M_max={pb_M_max}, "
                  f"{pb_n_windows} windows) ──")

        pb_output = os.path.splitext(output_path)[0] + "_pseudobands.h5"
        dim_flat = nspinor * ngkmax
        t_pb = time.perf_counter()

        # Run k=0 first to determine total band count
        H_k0 = setup_H_k_from_kvec(kpoints[0], V_scf, vnl_setup, crystal, None,
                                     V_loc_r=V_loc, ngkmax=ngkmax)
        apply_H0_flat = _make_flat_matvec(make_apply_H(H_k0), nspinor, ngkmax)
        Phi_det_0 = all_evecs[0].reshape(nbnd, -1)
        E_det_0 = eigenvalues[0, :nbnd] if eigenvalues.ndim == 2 else eigenvalues[:nbnd]

        pb0 = ritz_pseudobands(
            apply_H0_flat, dim=dim_flat,
            Phi_det=Phi_det_0, E_det=E_det_0, E_fermi=0.0,
            k=pb_k, M_max=pb_M_max, F=pb_F,
            n_windows_target=pb_n_windows,
            verbose=verbose, seed=0)

        n_total = pb0.Phi_out.shape[0]
        if verbose:
            print(f"  Total bands per k: {pb0.n_det} det + {pb0.n_pseudo} pseudo = {n_total}")

        # Open WFN_pseudobands.h5 with the total band count
        pb_writer = WFNWriter(pb_output, crystal, kpoints, weights, kgrid, n_total,
                               gvecs_per_k, nosym=nosym)

        # Write k=0 (already computed)
        _write_pb_k(pb_writer, 0, pb0, H_k0, gvecs_per_k[0], nspinor, ngkmax)

        # Remaining k-points
        for ik in range(1, nk):
            tk = time.perf_counter()
            H_k = setup_H_k_from_kvec(kpoints[ik], V_scf, vnl_setup, crystal, None,
                                        V_loc_r=V_loc, ngkmax=ngkmax)
            apply_H_flat = _make_flat_matvec(make_apply_H(H_k), nspinor, ngkmax)
            Phi_det_k = all_evecs[ik].reshape(nbnd, -1)
            E_det_k = eigenvalues[ik, :nbnd] if eigenvalues.ndim == 2 else eigenvalues[:nbnd]

            pb_k = ritz_pseudobands(
                apply_H_flat, dim=dim_flat,
                Phi_det=Phi_det_k, E_det=E_det_k, E_fermi=0.0,
                k=pb_k, M_max=pb_M_max, F=pb_F,
                n_windows_target=pb_n_windows,
                dos_result=pb0.dos,  # reuse KPM DOS from k=0
                verbose=False, seed=ik)

            _write_pb_k(pb_writer, ik, pb_k, H_k, gvecs_per_k[ik], nspinor, ngkmax)

            if verbose and (ik < 3 or ik == nk - 1 or (ik + 1) % 16 == 0):
                print(f"  k={ik:3d}/{nk}: {time.perf_counter()-tk:.1f}s")

        pb_writer.close()
        if verbose:
            print(f"  Pseudobands: {time.perf_counter()-t_pb:.1f}s → {pb_output}")

    if verbose:
        print(f"\nTotal NSCF: {time.perf_counter()-t_start:.1f}s")

    return eigenvalues


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LORRAX NSCF: Davidson + Pseudobands → WFN.h5")
    parser.add_argument("-i", "--input", default=None, help="Input file (nscf.in)")
    parser.add_argument("--save", default=None, help="QE .save directory")
    parser.add_argument("--pseudo_dir", default=None)
    parser.add_argument("--nbnd", type=int, default=None)
    parser.add_argument("--nk", type=int, nargs=3, default=None)
    parser.add_argument("--nosym", action="store_true")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--ref_wfn", default=None,
                        help="Reference WFN.h5 for k-points (overrides kgrid)")
    parser.add_argument("--no-davidson", action="store_true",
                        help="Skip Davidson, use existing WFN.h5 for pseudobands")
    parser.add_argument("--pseudobands", action="store_true",
                        help="Enable pseudobands stage")
    parser.add_argument("--pb-wfn", default=None,
                        help="Existing WFN.h5 for pseudobands-only mode")
    args = parser.parse_args()

    # Defaults
    do_davidson = True
    do_pseudobands = False
    pb_k, pb_M_max, pb_F, pb_n_windows = 6, 1500, 0.10, 50
    pb_wfn_input = None

    if args.input:
        from psp.nscf_input import read_nscf_input
        inp = read_nscf_input(args.input)
        save_dir = args.save or inp.save_dir
        nbnd = args.nbnd or inp.nbnd
        kgrid = tuple(args.nk) if args.nk else inp.kgrid
        nosym = args.nosym or inp.nosym
        output = args.output or inp.output
        tol = args.tol or inp.tol
        do_pseudobands = inp.pseudobands or args.pseudobands
        pb_k = inp.pb_k
        pb_M_max = inp.pb_M_max
        pb_F = inp.pb_F
        pb_n_windows = inp.pb_n_windows
        pb_wfn_input = inp.wfn_file if inp.rho_from_wfn else None
    else:
        if not args.save:
            parser.error("Either -i input_file or --save is required")
        save_dir = args.save
        nbnd = args.nbnd or 100
        kgrid = tuple(args.nk) if args.nk else (4, 4, 4)
        nosym = args.nosym
        output = args.output or "WFN.h5"
        tol = args.tol or 1e-8
        do_pseudobands = args.pseudobands

    if args.no_davidson:
        do_davidson = False
    if args.pb_wfn:
        pb_wfn_input = args.pb_wfn

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
             do_davidson=do_davidson, do_pseudobands=do_pseudobands,
             pb_k=pb_k, pb_M_max=pb_M_max, pb_F=pb_F,
             pb_n_windows=pb_n_windows, pb_wfn_input=pb_wfn_input,
             kpoints_override=kpoints_override, weights_override=weights_override)


if __name__ == "__main__":
    main()
