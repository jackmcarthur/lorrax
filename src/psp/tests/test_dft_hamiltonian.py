#!/usr/bin/env python3
"""
Self-contained validation: DFT Hamiltonian eigenvalues vs QE.

Reads QE .save directory, builds H = T + V_scf + V_NL, diagonalizes
in the NSCF wavefunction basis, compares with QE eigenvalues.

Requires:
  - QE .save directory with data-file-schema.xml + charge-density.hdf5
  - Pseudopotential .upf file
  - WFN.h5 from pw2bgw (for reference eigenvalues + wavefunctions)

Usage (Perlmutter login node, single GPU):
    cd /pscratch/sd/j/jackm/lorrax_sandbox_fresh/runs/Si/00_si_4x4x4_60band
    PYTHONPATH="/global/u2/j/jackm/software/lorrax_bse/src:\
    /global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site" \
    JAX_ENABLE_X64=1 HDF5_USE_FILE_LOCKING=FALSE \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python3 -u -m psp.tests.test_dft_hamiltonian \
        --save qe/scf/silicon.save \
        --pseudo_dir qe/scf \
        --wfn qe/nscf/WFN.h5
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import sys
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from file_io import WFNReader
from common import symmetry_maps, Meta
from common.load_wfns import read_Gvecs_to_devices
from psp.qe_save_reader import CrystalData
from psp.get_DFT_mtxels import (
    load_pseudopotentials, build_atom_pp_assignments,
    build_local_ionic_potential_on_G_total, poisson_potential_from_rhoG,
)
from psp.charge_density import build_core_density, compute_grad_rho_sq, build_V_xc, build_G_cart
from psp.dft_operators import (
    build_V_scf, compute_V_H_and_V_xc, setup_H_k, matrix,
)
import psp.vnl_ops as vnl_ops


def main():
    parser = argparse.ArgumentParser(description="Validate H eigenvalues vs QE")
    parser.add_argument("--save", required=True, help="QE .save directory")
    parser.add_argument("--pseudo_dir", required=True, help="Directory with .upf files")
    parser.add_argument("--wfn", required=True, help="WFN.h5 from pw2bgw")
    parser.add_argument("--sys_dim", type=int, default=3, help="0, 2, or 3")
    parser.add_argument("--nb", type=int, default=60, help="Bands to compare")
    args = parser.parse_args()

    print("=" * 60)
    print("DFT Hamiltonian validation: eig(H) vs QE")
    print("=" * 60)

    # ── Load crystal structure from QE save ──
    crystal = CrystalData.from_qe_save(args.save)
    wfn = WFNReader(args.wfn)
    crystal.validate_against_wfn(wfn)
    pseudos = load_pseudopotentials(args.pseudo_dir)
    assert pseudos, f"No .upf files found in {args.pseudo_dir}"

    nb = min(args.nb, int(wfn.nbands))
    n_occ = int(wfn.nelec)
    truncation_2d = (args.sys_dim == 2)

    # ── Build V_loc ──
    atom_pos = jnp.asarray(crystal.atom_crys, dtype=jnp.float64)
    atom_types_j = jnp.asarray(crystal.atom_types, dtype=jnp.int32)
    assignments = build_atom_pp_assignments(atom_pos, atom_types_j, pseudos)
    sp_tmp = {}
    for ap in assignments:
        if ap.pseudo is None:
            continue
        e = sp_tmp.setdefault(id(ap.pseudo), {"pseudo": ap.pseudo, "positions": []})
        e["positions"].append(np.asarray(ap.position, dtype=float))
    sp_payload = [(e["pseudo"], np.asarray(e["positions"], dtype=float))
                  for e in sp_tmp.values()]

    V_loc_r = jnp.asarray(build_local_ionic_potential_on_G_total(
        assignments=[{"pseudo": ap.pseudo, "position": np.asarray(ap.position, dtype=float)}
                     for ap in assignments],
        species_groups=sp_payload,
        fft_grid=crystal.fft_grid, bdot=crystal.bdot,
        cell_volume=crystal.cell_volume, bvec=crystal.bvec, blat=crystal.blat,
        truncation_2d=truncation_2d,
    ), dtype=jnp.float64)

    # ── Build V_H + V_xc from QE SCF density ──
    rho_r, _ = crystal.load_charge_density()
    rho_val = jnp.asarray(rho_r, dtype=jnp.float64)

    nx, ny, nz = crystal.fft_grid
    B = float(crystal.blat) * np.asarray(crystal.bvec, dtype=float)
    G_cart = build_G_cart(nx, ny, nz, B)

    class _M:
        fft_grid = crystal.fft_grid
        nspinor = crystal.nspinor

    rho_core, rhog_core = build_core_density(crystal, pseudos, _M())

    V_H_r, V_xc_r = compute_V_H_and_V_xc(
        rho_val, rho_core, rhog_core, G_cart,
        jnp.asarray(crystal.bdot, dtype=jnp.float64),
        jnp.asarray(crystal.bvec, dtype=jnp.float64),
        crystal.blat,
    )
    jax.block_until_ready(V_H_r)
    jax.block_until_ready(V_xc_r)

    V_scf = build_V_scf(V_loc_r, V_H_r, V_xc_r)

    # ── VNL setup ──
    sym = symmetry_maps.SymMaps(wfn)
    meta = Meta.from_system(wfn, sym, n_occ, nb - n_occ, nb, 0, bispinor=False)
    vnl_setup = vnl_ops.build_vnl_setup(
        crystal, sym, meta, pseudos, nspinor=crystal.nspinor,
    )

    # ── Load full-BZ wavefunctions ──
    # IMPORTANT: use read_Gvecs_to_devices (full BZ, properly rotated),
    # NOT load_kpoint_fftbox (which takes unfolded indices that don't match
    # the IBZ ordering in WFN.h5 for k >= 3).
    devices = np.array(jax.devices()).reshape(1, -1)
    mesh = Mesh(devices, ["x", "y"])
    global_psi, _ = read_Gvecs_to_devices(wfn, sym, (0, nb), meta, False, mesh)

    # ── IBZ → unfolded mapping ──
    with h5py.File(args.wfn, "r") as f:
        rk_wfn = f["mf_header/kpoints/rk"][:]

    unfolded = np.asarray(sym.unfolded_kpts)
    ibz_map = []
    for i in range(int(wfn.nkpts)):
        k = rk_wfn[i] % 1.0
        k[np.abs(k) < 1e-10] = 0.0
        k[np.abs(k - 1.0) < 1e-10] = 0.0
        u = unfolded % 1.0
        u[np.abs(u) < 1e-10] = 0.0
        u[np.abs(u - 1.0) < 1e-10] = 0.0
        ibz_map.append(int(np.argmin(np.linalg.norm(u - k[None, :], axis=1))))

    ref = np.asarray(wfn.energies)  # (nspin, nk_ibz, nb)

    # ── Diagonalize at each IBZ k-point ──
    print(f"\n{'k':>3} {'unf':>4} {'nG':>5} {'offset(mRy)':>12} {'MAE(mRy)':>10} "
          f"{'gap_err(meV)':>13}")

    max_mae = 0.0
    all_ok = True
    for i_ibz in range(int(wfn.nkpts)):
        i_unf = ibz_map[i_ibz]

        # Build H at the unfolded k-point (correct k-vector for operators)
        H_k = setup_H_k(i_unf, V_scf, vnl_setup, wfn, sym, meta,
                         V_loc_r=V_loc_r)

        # Use the unfolded wavefunction (properly rotated by SymMaps)
        psi_box = jnp.asarray(global_psi[i_unf, :nb])

        H_mn = matrix(psi_box, H_k)
        H_mn = 0.5 * (np.asarray(H_mn) + np.asarray(H_mn).conj().T)
        eigvals = np.sort(np.linalg.eigvalsh(H_mn))

        d = eigvals[:n_occ] - ref[0, i_ibz, :n_occ]
        off = np.mean(d)
        mae = np.mean(np.abs(d - off))
        max_mae = max(max_mae, mae)

        gap = (eigvals[n_occ] - eigvals[n_occ - 1]) * 13605.698
        gap_ref = (ref[0, i_ibz, n_occ] - ref[0, i_ibz, n_occ - 1]) * 13605.698

        ok = mae * 1000 < 0.01  # sub 0.01 mRy
        if not ok:
            all_ok = False
        print(f"{i_ibz:3d} {i_unf:4d} {H_k.nG:5d} {off * 1000:12.4f} "
              f"{mae * 1000:10.4f} {gap - gap_ref:13.4f}")

    print(f"\nMax MAE: {max_mae * 1000:.4f} mRy = {max_mae * 13605.698:.2f} meV")
    if all_ok:
        print("PASS: all k-points match QE to < 0.01 mRy")
    else:
        print("FAIL: some k-points exceed 0.01 mRy tolerance")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
