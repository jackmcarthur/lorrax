"""psp/run_nscf.py — Full NSCF driver: Davidson eigensolver → WFN.h5.

Reads QE .save, builds the DFT Hamiltonian, solves for eigenvalues at
all k-points via Davidson, writes BGW-compatible WFN.h5.

Usage (single GPU):
    module load lorrax
    lxrun python3 -u -m psp.run_nscf \\
        --save qe/nscf/silicon.save --nbnd 100 --nk 4 4 4 -o WFN.h5

Usage (4 GPUs, single process — k-points distributed via mesh):
    srun --gres=gpu:4 -N1 -n1 shifter ... python3 -u -m psp.run_nscf ...
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import functools
import time
import sys

import numpy as np
import jax
import jax.numpy as jnp

from psp.qe_save_reader import CrystalData
from psp.get_DFT_mtxels import load_pseudopotentials
from psp.ionic_gspace import build_ionic_and_core
from psp.charge_density import build_G_cart
from psp.dft_operators import (
    compute_V_H_and_V_xc, build_V_scf, compute_ngkmax,
    setup_H_k_from_kvec, apply_H_k,
)
from psp.davidson import davidson_k, warmup_jit
from psp.wfn_writer import write_wfn_h5
import psp.vnl_ops as vnl_ops


# ---------------------------------------------------------------------------
# G-vector generation in QE master-list order
# ---------------------------------------------------------------------------

def build_master_gvec_list(crystal):
    """Build the master G-vector list sorted by |G|² (QE convention).

    Returns (G_master, G2_master) where G_master is (ng, 3) int and
    G2_master is (ng,) float.
    """
    nx, ny, nz = crystal.fft_grid
    gx = np.fft.fftfreq(int(nx), d=1.0 / int(nx)).astype(int)
    gy = np.fft.fftfreq(int(ny), d=1.0 / int(ny)).astype(int)
    gz = np.fft.fftfreq(int(nz), d=1.0 / int(nz)).astype(int)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing="ij")
    G_all = np.stack([Gx.ravel(), Gy.ravel(), Gz.ravel()], axis=-1)
    bdot = np.asarray(crystal.bdot, dtype=float)
    G2 = np.einsum("gi,ij,gj->g", G_all.astype(float), bdot, G_all.astype(float))
    order = np.argsort(G2)
    return G_all[order].astype(np.int32), G2[order]


def select_gvecs_for_k(kvec, G_master, bdot, ecutwfc):
    """Select G-vectors for one k-point from the master list.

    Preserves master ordering (QE convention). Returns:
    - Gk: (ngk, 3) int — G-vectors for this k
    - mask_master: (ng_master,) bool — which master entries are selected
    """
    kvec = np.asarray(kvec, dtype=float)
    KG = G_master.astype(float) + kvec[None, :]
    KG2 = np.einsum("gi,ij,gj->g", KG, bdot, KG)
    mask = KG2 <= ecutwfc
    return G_master[mask], mask


# ---------------------------------------------------------------------------
# Sparse-G apply_H wrapper (for Davidson)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("_nx", "_ny", "_nz"))
def _apply_H_sparse(psi_G, T, V, Gx, Gy, Gz, Z, E, mask, _nx, _ny, _nz):
    """Sparse-G → sparse-G H|ψ⟩. Single JIT trace for all k-points."""
    mask_f = mask[None, None, :].astype(psi_G.dtype)
    psi_box = jnp.zeros((*psi_G.shape[:2], _nx, _ny, _nz), dtype=psi_G.dtype)
    psi_box = psi_box.at[:, :, Gx, Gy, Gz].add(psi_G * mask_f)
    return apply_H_k(psi_box, T, V, Gx, Gy, Gz, Z, E, mask)


# ---------------------------------------------------------------------------
# Main NSCF driver
# ---------------------------------------------------------------------------

def run_nscf(
    crystal: CrystalData,
    pseudos: dict,
    kgrid: tuple[int, int, int],
    nbnd: int,
    output_path: str = "WFN.h5",
    *,
    truncation_2d: bool = False,
    tol: float = 1e-8,
    verbose: bool = True,
    kpoints_override: np.ndarray | None = None,
    weights_override: np.ndarray | None = None,
):
    """Run NSCF calculation: build potentials, solve Davidson, write WFN.h5.

    Parameters
    ----------
    crystal : CrystalData from QE .save directory
    pseudos : dict from load_pseudopotentials
    kgrid : (nkx, nky, nkz) Monkhorst-Pack grid
    nbnd : number of bands to compute
    output_path : WFN.h5 output file
    truncation_2d : use 2D Coulomb truncation for V_loc
    tol : Davidson convergence tolerance
    """
    fft_grid = crystal.fft_grid
    nspinor = crystal.nspinor
    _nx, _ny, _nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
    t_start = time.perf_counter()

    if verbose:
        print(f"NSCF: {nbnd} bands, kgrid={kgrid}, fft={fft_grid}, nspinor={nspinor}")
        print(f"  GPUs: {len(jax.devices())}")

    # ── Build potentials ──
    t0 = time.perf_counter()
    V_loc_r, rho_core_r, rho_core_G = build_ionic_and_core(
        crystal, pseudos, fft_grid, truncation_2d=truncation_2d)
    rho_r, _ = crystal.load_charge_density()
    rho_val = jnp.asarray(rho_r, dtype=jnp.float64)
    B = float(crystal.blat) * np.asarray(crystal.bvec, dtype=float)
    G_cart = build_G_cart(_nx, _ny, _nz, B)
    V_H_r, V_xc_r = compute_V_H_and_V_xc(
        rho_val, rho_core_r, rho_core_G, G_cart,
        jnp.asarray(crystal.bdot, dtype=jnp.float64),
        jnp.asarray(crystal.bvec, dtype=jnp.float64), crystal.blat)
    V_scf = build_V_scf(V_loc_r, V_H_r, V_xc_r)
    jax.block_until_ready(V_scf)
    if verbose:
        print(f"  Potentials: {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    vnl_setup = vnl_ops.build_vnl_setup(
        crystal, pseudos=pseudos, nspinor=nspinor,
        q_max=float(np.sqrt(float(crystal.ecutwfc))) * 1.01)
    if verbose:
        print(f"  VNL setup: {time.perf_counter()-t0:.2f}s")

    # ── K-points ──
    if kpoints_override is not None:
        kpoints = np.asarray(kpoints_override, dtype=np.float64)
        weights = np.asarray(weights_override, dtype=np.float64)
    else:
        kpoints, weights = crystal.build_kgrid(
            nk=kgrid, nosym=True, noinv=True, no_t_rev=True, force_symmorphic=False)
        kpoints = np.asarray(kpoints, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
    nk = len(kpoints)

    # Master G-vector list (QE ordering)
    G_master, _ = build_master_gvec_list(crystal)
    bdot = np.asarray(crystal.bdot, dtype=float)

    # ngkmax for uniform JIT shapes
    ngkmax = compute_ngkmax(kpoints, bdot, crystal.ecutwfc, crystal.fft_grid)
    if verbose:
        print(f"  nk={nk}, ngkmax={ngkmax}")

    # ── JIT warmup ──
    t0 = time.perf_counter()
    warmup_jit(ngkmax, nspinor, nbnd)
    # Warmup _apply_H_sparse
    H_k0 = setup_H_k_from_kvec(kpoints[0], V_scf, vnl_setup, crystal, None,
                                 V_loc_r=V_loc_r, ngkmax=ngkmax)
    dummy = jnp.zeros((1, nspinor, ngkmax), dtype=jnp.complex128)
    _ = _apply_H_sparse(dummy, H_k0.T_diag, H_k0.V_scf,
                         H_k0.Gx, H_k0.Gy, H_k0.Gz,
                         H_k0.vnl_Z, H_k0.vnl_E, H_k0.mask,
                         _nx, _ny, _nz)
    if verbose:
        print(f"  JIT warmup: {time.perf_counter()-t0:.2f}s")

    # ── Davidson at each k-point ──
    eigenvalues = np.zeros((nk, nbnd))
    gvecs_per_k = []
    coeffs_per_k = []

    t_dav_start = time.perf_counter()
    for ik in range(nk):
        t0 = time.perf_counter()

        # G-vectors in QE master order
        Gk_qe, _ = select_gvecs_for_k(kpoints[ik], G_master, bdot, crystal.ecutwfc)
        ngk = Gk_qe.shape[0]
        gvecs_per_k.append(Gk_qe)

        # Build H_k with ngkmax padding (using master-ordered G-vectors)
        H_k = setup_H_k_from_kvec(kpoints[ik], V_scf, vnl_setup, crystal, None,
                                    V_loc_r=V_loc_r, ngkmax=ngkmax)

        # apply_H closure (reuses single JIT trace)
        def apply_H(psi_G, _H=H_k):
            return _apply_H_sparse(psi_G, _H.T_diag, _H.V_scf,
                                    _H.Gx, _H.Gy, _H.Gz,
                                    _H.vnl_Z, _H.vnl_E, _H.mask,
                                    _nx, _ny, _nz)

        evals, evecs = davidson_k(
            apply_H, h_diag=H_k.h_diag, nG=ngkmax, nspinor=nspinor,
            n_tgt=nbnd, T_diag=H_k.T_diag, verbose=False, tol=tol)

        eigenvalues[ik] = evals

        # Reorder eigenvectors from Davidson's |k+G|²-sorted G-vectors
        # to QE master-ordered G-vectors for WFN.h5
        evecs_np = np.asarray(evecs)  # (nbnd, nspinor, ngkmax) padded

        # Davidson's G-vectors (|k+G|² sorted, padded to ngkmax)
        Gx_dav = np.asarray(H_k.Gx)
        Gy_dav = np.asarray(H_k.Gy)
        Gz_dav = np.asarray(H_k.Gz)
        mask_dav = np.asarray(H_k.mask)

        # Build mapping: for each QE G-vector, find its index in Davidson's list
        # Both are subsets of the FFT grid, so match by Miller indices
        dav_gvecs = np.stack([Gx_dav[mask_dav], Gy_dav[mask_dav], Gz_dav[mask_dav]], axis=-1)
        # Create a dict: (gx,gy,gz) → index in Davidson's unpadded list
        dav_map = {}
        for i in range(dav_gvecs.shape[0]):
            dav_map[tuple(dav_gvecs[i])] = i

        # Reorder: for each QE G-vector, pick the coefficient from Davidson
        coeffs_k = np.zeros((nbnd, nspinor, ngk), dtype=np.complex128)
        evecs_unpadded = evecs_np[:, :, :sum(mask_dav)]  # strip padding
        for ig in range(ngk):
            key = tuple(Gk_qe[ig])
            if key in dav_map:
                coeffs_k[:, :, ig] = evecs_unpadded[:, :, dav_map[key]]

        coeffs_per_k.append(coeffs_k)

        dt = time.perf_counter() - t0
        if verbose and (ik < 3 or ik == nk - 1 or (ik + 1) % 16 == 0):
            print(f"  k={ik:3d}/{nk}: {dt:.3f}s  evals[0]={evals[0]:.6f} Ry")

    t_dav = time.perf_counter() - t_dav_start
    if verbose:
        print(f"  Davidson total: {t_dav:.2f}s ({t_dav/nk:.3f}s/k)")

    # ── Write WFN.h5 ──
    t0 = time.perf_counter()
    write_wfn_h5(
        output_path,
        crystal=crystal,
        kpoints=kpoints,
        weights=weights,
        kgrid=kgrid,
        eigenvalues=eigenvalues,
        gvecs_per_k=gvecs_per_k,
        coeffs_per_k=coeffs_per_k,
    )
    if verbose:
        print(f"  WFN.h5 written: {time.perf_counter()-t0:.2f}s → {output_path}")
        print(f"  Total NSCF: {time.perf_counter()-t_start:.2f}s")

    return eigenvalues


def main():
    parser = argparse.ArgumentParser(description="LORRAX NSCF: Davidson → WFN.h5")
    parser.add_argument("--save", required=True, help="QE .save directory")
    parser.add_argument("--pseudo_dir", default=None, help="Directory with .upf (default: same as --save)")
    parser.add_argument("--nbnd", type=int, default=100, help="Number of bands")
    parser.add_argument("--nk", type=int, nargs=3, default=[4, 4, 4], help="K-grid dimensions")
    parser.add_argument("-o", "--output", default="WFN.h5", help="Output WFN.h5 path")
    parser.add_argument("--sys_dim", type=int, default=3, help="System dimension (2 or 3)")
    parser.add_argument("--tol", type=float, default=1e-8, help="Davidson convergence tolerance")
    parser.add_argument("--ref_wfn", default=None, help="Reference WFN.h5 to read k-points from (for validation)")
    args = parser.parse_args()

    pseudo_dir = args.pseudo_dir or args.save
    crystal = CrystalData.from_qe_save(args.save)
    pseudos = load_pseudopotentials(pseudo_dir)

    # Read k-points from reference WFN if provided (ensures same ordering)
    kpoints_override = None
    weights_override = None
    if args.ref_wfn:
        import h5py
        with h5py.File(args.ref_wfn, "r") as f:
            kpoints_override = f["mf_header/kpoints/rk"][:]
            weights_override = f["mf_header/kpoints/w"][:]

    run_nscf(
        crystal, pseudos,
        kgrid=tuple(args.nk),
        nbnd=args.nbnd,
        output_path=args.output,
        truncation_2d=(args.sys_dim == 2),
        tol=args.tol,
        kpoints_override=kpoints_override,
        weights_override=weights_override,
    )


if __name__ == "__main__":
    main()
