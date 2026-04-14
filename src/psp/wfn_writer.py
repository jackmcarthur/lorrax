"""
psp/wfn_writer.py — Write WFN.h5 files compatible with BerkeleyGW / LORRAX.

Constructs a WFN.h5 from a CrystalData object, k-grid, eigenvalues,
and (optionally) wavefunction coefficients from a Davidson solve.

The file format follows the spec in docs/archive/misc/wfn.h5.txt and is
readable by WFNReader without modification.

Usage
-----
    from psp.wfn_writer import write_wfn_h5

    write_wfn_h5(
        "WFN.h5",
        crystal=crystal,
        kpoints=kpts,           # (nk, 3) crystal coords
        weights=weights,        # (nk,)
        kgrid=(4, 4, 4),
        eigenvalues=eigvals,    # (nk, nbands) Ry
        gvecs_per_k=gvecs,      # list of (ngk, 3) per k
        coeffs_per_k=coeffs,    # list of (nbands, nspinor, ngk) per k
    )
"""
from __future__ import annotations

import numpy as np
import h5py


def write_wfn_h5(
    path: str,
    crystal,
    kpoints: np.ndarray,
    weights: np.ndarray,
    kgrid: tuple[int, int, int],
    eigenvalues: np.ndarray,
    gvecs_per_k: list[np.ndarray],
    coeffs_per_k: list[np.ndarray] | None = None,
    *,
    occupations: np.ndarray | None = None,
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Write a complete WFN.h5 file.

    Parameters
    ----------
    path : output file path
    crystal : CrystalData (or WFNReader) — structure + symmetry
    kpoints : (nk, 3) — IBZ k-points in crystal coordinates
    weights : (nk,) — k-weights summing to 1
    kgrid : (3,) — Monkhorst-Pack grid dimensions
    eigenvalues : (nk, nbands) — DFT eigenvalues in Ry
    gvecs_per_k : list of (ngk_i, 3) int arrays — G-vectors per k-point
    coeffs_per_k : list of (nbands, nspinor, ngk_i) complex128, or None
        If None, writes zeros (header-only mode).
    occupations : (nk, nbands), or None to auto-fill from nelec
    shift : MP grid shift
    """
    nk = kpoints.shape[0]
    nbands = eigenvalues.shape[1]
    nspin = crystal.nspin
    nspinor = crystal.nspinor

    # G-vector counts per k
    ngk = np.array([g.shape[0] for g in gvecs_per_k], dtype=np.int32)
    ngkmax = int(ngk.max())
    ngktot = int(ngk.sum())

    # Concatenated G-vectors
    gvecs_cat = np.concatenate(gvecs_per_k, axis=0).astype(np.int32)

    # Concatenated coefficients
    if coeffs_per_k is not None:
        # coeffs_per_k[ik]: (nbands, nspinor, ngk_i) complex128
        # WFN.h5 format: (nbands, nspin*nspinor, ngktot, 2) float64
        coeffs_all = np.zeros((nbands, nspinor, ngktot, 2), dtype=np.float64)
        offset = 0
        for ik in range(nk):
            ng_k = ngk[ik]
            c = coeffs_per_k[ik]  # (nbands, nspinor, ng_k)
            coeffs_all[:, :, offset:offset + ng_k, 0] = c.real
            coeffs_all[:, :, offset:offset + ng_k, 1] = c.imag
            offset += ng_k
    else:
        coeffs_all = np.zeros((nbands, nspinor, ngktot, 2), dtype=np.float64)

    # Occupations
    if occupations is None:
        n_occ = int(crystal.nelec)
        occupations = np.zeros((nk, nbands), dtype=np.float64)
        occupations[:, :n_occ] = 1.0

    # ifmin / ifmax (1-indexed, as per BGW convention)
    ifmin = np.ones((nspin, nk), dtype=np.int32)
    ifmax = np.zeros((nspin, nk), dtype=np.int32)
    for ik in range(nk):
        n_occ_k = int(np.sum(occupations[ik] > 0.5))
        ifmax[0, ik] = n_occ_k

    # Charge-density G-space: G-vectors with |G|² ≤ ecutrho, sorted by |G|²
    nx, ny, nz = crystal.fft_grid
    gx = np.fft.fftfreq(nx, d=1.0 / nx).astype(int)
    gy = np.fft.fftfreq(ny, d=1.0 / ny).astype(int)
    gz = np.fft.fftfreq(nz, d=1.0 / nz).astype(int)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing='ij')
    gspace_components = np.stack([Gx.ravel(), Gy.ravel(), Gz.ravel()], axis=-1)
    bdot = np.asarray(crystal.bdot, dtype=float)
    G2 = np.einsum('gi,ij,gj->g', gspace_components.astype(float), bdot,
                    gspace_components.astype(float))
    # Filter by ecutrho and sort by |G|² (QE/BGW convention)
    ecutrho = float(crystal.ecutrho)
    rho_mask = G2 <= ecutrho
    order = np.argsort(G2[rho_mask])
    gspace_components = gspace_components[rho_mask][order].astype(np.int32)
    ng = gspace_components.shape[0]

    # Eigenvalues: spec says (mnband, nrk, nspin)
    el = np.zeros((nspin, nk, nbands), dtype=np.float64)
    el[0] = eigenvalues

    # Occupations: (nspin, nk, nbands)
    occ = np.zeros((nspin, nk, nbands), dtype=np.float64)
    occ[0] = occupations

    # apos: atom positions in Cartesian, units of alat
    # crystal.atom_crys is in crystal coords; convert via avec
    apos = crystal.atom_crys @ crystal.avec  # (nat, 3)

    # adot: real-space metric
    adot = crystal.avec.T @ crystal.avec * crystal.alat ** 2

    # recvol
    recvol = (2.0 * np.pi) ** 3 / crystal.cell_volume

    # ── Write ──
    with h5py.File(path, 'w') as f:
        # versionnumber, flavor
        f.create_dataset('mf_header/versionnumber', data=1)
        f.create_dataset('mf_header/flavor', data=2)  # complex

        # kpoints group
        kp = 'mf_header/kpoints/'
        f.create_dataset(kp + 'nspin', data=nspin)
        f.create_dataset(kp + 'nspinor', data=nspinor)
        f.create_dataset(kp + 'nrk', data=nk)
        f.create_dataset(kp + 'mnband', data=nbands)
        f.create_dataset(kp + 'ngkmax', data=ngkmax)
        f.create_dataset(kp + 'ecutwfc', data=float(crystal.ecutwfc))
        f.create_dataset(kp + 'kgrid', data=np.array(kgrid, dtype=np.int32))
        f.create_dataset(kp + 'shift', data=np.array(shift, dtype=np.float64))
        f.create_dataset(kp + 'ngk', data=ngk)
        f.create_dataset(kp + 'ifmin', data=ifmin)
        f.create_dataset(kp + 'ifmax', data=ifmax)
        f.create_dataset(kp + 'w', data=weights.astype(np.float64))
        f.create_dataset(kp + 'rk', data=kpoints.astype(np.float64))
        f.create_dataset(kp + 'el', data=el)
        f.create_dataset(kp + 'occ', data=occ)

        # gspace group
        gs = 'mf_header/gspace/'
        f.create_dataset(gs + 'ng', data=ng)
        f.create_dataset(gs + 'ecutrho', data=float(crystal.ecutrho))
        f.create_dataset(gs + 'FFTgrid', data=np.array(crystal.fft_grid, dtype=np.int32))
        f.create_dataset(gs + 'components', data=gspace_components)

        # symmetry group
        sy = 'mf_header/symmetry/'
        f.create_dataset(sy + 'ntran', data=crystal.ntran)
        f.create_dataset(sy + 'cell_symmetry', data=0)  # 0 = cubic
        f.create_dataset(sy + 'mtrx', data=crystal.sym_matrices)
        f.create_dataset(sy + 'tnp', data=crystal.translations)

        # crystal group
        cr = 'mf_header/crystal/'
        f.create_dataset(cr + 'celvol', data=float(crystal.cell_volume))
        f.create_dataset(cr + 'recvol', data=float(recvol))
        f.create_dataset(cr + 'alat', data=float(crystal.alat))
        f.create_dataset(cr + 'blat', data=float(crystal.blat))
        f.create_dataset(cr + 'nat', data=crystal.nat)
        f.create_dataset(cr + 'avec', data=crystal.avec.astype(np.float64))
        f.create_dataset(cr + 'bvec', data=crystal.bvec.astype(np.float64))
        f.create_dataset(cr + 'adot', data=adot.astype(np.float64))
        f.create_dataset(cr + 'bdot', data=crystal.bdot.astype(np.float64))
        f.create_dataset(cr + 'atyp', data=crystal.atom_types.astype(np.int32))
        f.create_dataset(cr + 'apos', data=apos.astype(np.float64))

        # wfns group
        f.create_dataset('wfns/gvecs', data=gvecs_cat)
        f.create_dataset('wfns/coeffs', data=coeffs_all)
