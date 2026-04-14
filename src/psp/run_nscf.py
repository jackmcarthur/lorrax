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
from psp.pseudos import load_pseudopotentials
from psp.ionic_gspace import build_ionic_and_core
from psp.charge_density import build_G_cart
from psp.dft_operators import (
    compute_V_H_and_V_xc, build_V_scf, compute_ngkmax,
    setup_H_k_from_kvec, apply_H_k,
)
from psp.davidson import davidson_k, warmup_jit
from psp.wfn_writer import WFNWriter
import psp.vnl_ops as vnl_ops


# ---------------------------------------------------------------------------
# G-vector generation in QE master-list order
# ---------------------------------------------------------------------------

def build_master_gvec_list(crystal):
    """Build the master G-vector list in QE convention.

    QE ordering: sorted by (|G|², g₁, g₂, g₃) — |G|² primary (with
    eps8 tolerance for shell grouping), lexicographic secondary.
    Range per axis: [-N//2+1, N//2] (matches QE ggen.f90).

    Returns (G_master, G2_master) where G_master is (ng, 3) int and
    G2_master is (ng,) float — filtered to |G|² ≤ ecutrho.
    """
    nx, ny, nz = int(crystal.fft_grid[0]), int(crystal.fft_grid[1]), int(crystal.fft_grid[2])
    gx = np.arange(-nx // 2 + 1, nx // 2 + 1)
    gy = np.arange(-ny // 2 + 1, ny // 2 + 1)
    gz = np.arange(-nz // 2 + 1, nz // 2 + 1)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing="ij")
    G_all = np.stack([Gx.ravel(), Gy.ravel(), Gz.ravel()], axis=-1).astype(np.int32)
    bdot = np.asarray(crystal.bdot, dtype=float)
    G2 = np.einsum("gi,ij,gj->g", G_all.astype(float), bdot, G_all.astype(float))

    # Filter by ecutrho (charge density cutoff)
    mask = G2 <= float(crystal.ecutrho)
    G_f, G2_f = G_all[mask], G2[mask]

    # Sort: primary |G|² (discretized to avoid float issues), secondary lexicographic
    G2_int = np.round(G2_f * 1e8).astype(np.int64)
    order = np.lexsort((G_f[:, 2], G_f[:, 1], G_f[:, 0], G2_int))
    return G_f[order], G2_f[order]


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
    tol: float = 1e-8,
    verbose: bool = True,
    kpoints_override: np.ndarray | None = None,
    weights_override: np.ndarray | None = None,
):
    """Run NSCF calculation: build potentials, solve Davidson, write WFN.h5.

    2D Coulomb truncation is auto-detected from crystal.assume_isolated.
    """
    truncation_2d = crystal.assume_isolated == "2D"
    fft_grid = crystal.fft_grid
    nspinor = crystal.nspinor
    _nx, _ny, _nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
    t_start = time.perf_counter()

    if verbose:
        print(f"NSCF: {nbnd} bands, kgrid={kgrid}, fft={fft_grid}, nspinor={nspinor}")
        if truncation_2d:
            print(f"  2D Coulomb truncation (from assume_isolated='2D')")
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
        jnp.asarray(crystal.bvec, dtype=jnp.float64), crystal.blat,
        truncation_2d=truncation_2d)
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

    # ── Pre-compute per-k G-vectors in QE order ──
    gvecs_per_k = []
    for ik in range(nk):
        Gk_qe, _ = select_gvecs_for_k(kpoints[ik], G_master, bdot, crystal.ecutwfc)
        gvecs_per_k.append(Gk_qe)

    # ── Open WFN.h5 for streaming writes ──
    writer = WFNWriter(output_path, crystal, kpoints, weights, kgrid, nbnd,
                        gvecs_per_k, nosym=True)

    # ── Davidson at each k-point — write coefficients as each finishes ──
    eigenvalues = np.zeros((nk, nbnd))

    t_dav_start = time.perf_counter()
    for ik in range(nk):
        t0 = time.perf_counter()
        Gk_qe = gvecs_per_k[ik]
        ngk = Gk_qe.shape[0]

        # Build H_k with ngkmax padding
        H_k = setup_H_k_from_kvec(kpoints[ik], V_scf, vnl_setup, crystal, None,
                                    V_loc_r=V_loc_r, ngkmax=ngkmax)

        def apply_H(psi_G, _H=H_k):
            return _apply_H_sparse(psi_G, _H.T_diag, _H.V_scf,
                                    _H.Gx, _H.Gy, _H.Gz,
                                    _H.vnl_Z, _H.vnl_E, _H.mask,
                                    _nx, _ny, _nz)

        evals, evecs = davidson_k(
            apply_H, h_diag=H_k.h_diag, nG=ngkmax, nspinor=nspinor,
            n_tgt=nbnd, T_diag=H_k.T_diag, verbose=False, tol=tol)

        eigenvalues[ik] = evals

        # Reorder eigenvectors: Davidson's |k+G|²-sorted → QE master order
        evecs_np = np.asarray(evecs)  # (nbnd, nspinor, ngkmax)
        mask_dav = np.asarray(H_k.mask)
        nG_actual = int(mask_dav.sum())
        Gx_dav = np.asarray(H_k.Gx)[:nG_actual]
        Gy_dav = np.asarray(H_k.Gy)[:nG_actual]
        Gz_dav = np.asarray(H_k.Gz)[:nG_actual]

        # Map (gx,gy,gz) → Davidson index
        dav_gvecs = np.stack([Gx_dav, Gy_dav, Gz_dav], axis=-1)
        dav_map = {tuple(dav_gvecs[i]): i for i in range(nG_actual)}

        # Reindex to QE order
        reorder = np.array([dav_map[tuple(Gk_qe[ig])] for ig in range(ngk)])
        coeffs_k = evecs_np[:, :, reorder]

        # Stream-write this k-point immediately
        writer.write_k(ik, evals, coeffs_k)

        dt = time.perf_counter() - t0
        if verbose and (ik < 3 or ik == nk - 1 or (ik + 1) % 16 == 0):
            print(f"  k={ik:3d}/{nk}: {dt:.3f}s  evals[0]={evals[0]:.6f} Ry")

    writer.close()

    t_dav = time.perf_counter() - t_dav_start
    if verbose:
        print(f"  Davidson + write: {t_dav:.2f}s ({t_dav/nk:.3f}s/k)")
        print(f"  Total NSCF: {time.perf_counter()-t_start:.2f}s → {output_path}")

    return eigenvalues


def main():
    parser = argparse.ArgumentParser(description="LORRAX NSCF: Davidson → WFN.h5")
    parser.add_argument("--save", required=True, help="QE .save directory")
    parser.add_argument("--pseudo_dir", default=None, help="Directory with .upf (default: same as --save)")
    parser.add_argument("--nbnd", type=int, default=100, help="Number of bands")
    parser.add_argument("--nk", type=int, nargs=3, default=[4, 4, 4], help="K-grid dimensions")
    parser.add_argument("-o", "--output", default="WFN.h5", help="Output WFN.h5 path")
    parser.add_argument("--tol", type=float, default=1e-8, help="Davidson convergence tolerance")
    parser.add_argument("--ref_wfn", default=None, help="Reference WFN.h5 to read k-points from (for validation)")
    args = parser.parse_args()

    pseudo_dir = args.pseudo_dir or args.save
    crystal = CrystalData.from_qe_save(args.save)
    pseudos = load_pseudopotentials(pseudo_dir)

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
        tol=args.tol,
        kpoints_override=kpoints_override,
        weights_override=weights_override,
    )


if __name__ == "__main__":
    main()
