"""psp/gvec_utils.py — G-vector bookkeeping for plane-wave calculations.

Master G-vector list, per-k selection, ngkmax computation,
and Davidson→QE reordering.
"""
from __future__ import annotations

import numpy as np


def build_master_gvec_list(crystal) -> tuple[np.ndarray, np.ndarray]:
    """Master G-vector list in QE convention (ecutrho-filtered).

    Ordering: primary |G|² (discretised for shell grouping), secondary
    lexicographic (g1, g2, g3).  Range per axis: [-N//2+1, N//2].

    Returns (G_master, G2_master): (ng, 3) int32 and (ng,) float64.
    """
    nx, ny, nz = int(crystal.fft_grid[0]), int(crystal.fft_grid[1]), int(crystal.fft_grid[2])
    gx = np.arange(-nx // 2 + 1, nx // 2 + 1)
    gy = np.arange(-ny // 2 + 1, ny // 2 + 1)
    gz = np.arange(-nz // 2 + 1, nz // 2 + 1)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing="ij")
    G_all = np.stack([Gx.ravel(), Gy.ravel(), Gz.ravel()], axis=-1).astype(np.int32)
    bdot = np.asarray(crystal.bdot, dtype=float)
    G2 = np.einsum("gi,ij,gj->g", G_all.astype(float), bdot, G_all.astype(float))

    mask = G2 <= float(crystal.ecutrho)
    G_f, G2_f = G_all[mask], G2[mask]

    G2_int = np.round(G2_f * 1e8).astype(np.int64)
    order = np.lexsort((G_f[:, 2], G_f[:, 1], G_f[:, 0], G2_int))
    return G_f[order], G2_f[order]


def select_gvecs_for_k(kvec, G_master, bdot, ecutwfc):
    """G-vectors for one k-point from the master list.

    Returns (Gk, mask_master): Gk is (ngk, 3) int, mask is (ng_master,) bool.
    """
    kvec = np.asarray(kvec, dtype=float)
    bdot = np.asarray(bdot, dtype=float)
    KG = G_master.astype(float) + kvec[None, :]
    KG2 = np.einsum("gi,ij,gj->g", KG, bdot, KG)
    mask = KG2 <= ecutwfc
    return G_master[mask], mask


def compute_ngkmax(kpoints, bdot, ecutwfc, fft_grid) -> int:
    """Max G-vector count across all k-points (for uniform JIT shapes)."""
    kpoints = np.asarray(kpoints, dtype=np.float64)
    bdot = np.asarray(bdot, dtype=np.float64)
    nx, ny, nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])

    gx = np.fft.fftfreq(nx, d=1.0 / nx).astype(int)
    gy = np.fft.fftfreq(ny, d=1.0 / ny).astype(int)
    gz = np.fft.fftfreq(nz, d=1.0 / nz).astype(int)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing="ij")
    G_all = np.stack([Gx.ravel(), Gy.ravel(), Gz.ravel()], axis=-1).astype(np.float64)

    ngk_max = 0
    for ik in range(len(kpoints)):
        KG = G_all + kpoints[ik][None, :]
        KG_sq = np.einsum("gi,ij,gj->g", KG, bdot, KG)
        ngk_max = max(ngk_max, int(np.sum(KG_sq <= ecutwfc)))
    return ngk_max


def reorder_to_qe(evecs_np, H_k, Gk_qe):
    """Reorder Davidson eigenvectors from |k+G|²-sorted to QE master order.

    Parameters
    ----------
    evecs_np : (nbnd, nspinor, ngkmax) — Davidson eigenvectors
    H_k : HamiltonianK — has .Gx, .Gy, .Gz, .mask
    Gk_qe : (ngk, 3) int — G-vectors in QE ordering for this k

    Returns
    -------
    coeffs_k : (nbnd, nspinor, ngk) — reordered coefficients
    """
    mask_dav = np.asarray(H_k.mask)
    nG_actual = int(mask_dav.sum())
    Gx_dav = np.asarray(H_k.Gx)[:nG_actual]
    Gy_dav = np.asarray(H_k.Gy)[:nG_actual]
    Gz_dav = np.asarray(H_k.Gz)[:nG_actual]

    dav_gvecs = np.stack([Gx_dav, Gy_dav, Gz_dav], axis=-1)
    dav_map = {tuple(dav_gvecs[i]): i for i in range(nG_actual)}

    ngk = Gk_qe.shape[0]
    reorder = np.array([dav_map[tuple(Gk_qe[ig])] for ig in range(ngk)])
    return evecs_np[:, :, reorder]
