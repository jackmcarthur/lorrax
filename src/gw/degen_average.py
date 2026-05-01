"""Degenerate-subspace averaging of diagonal Σ matrix elements.

Mirrors BerkeleyGW's ``Sigma/shiftenergy.f90`` band-averaging (lines
86-122 there): within each contiguous degenerate group of the DFT
eigenvalue spectrum, replace each diagonal Σ value with the group mean.
Default tolerance matches BGW's ``Common/nrtype.f90 :: TOL_Degeneracy``
(1×10⁻⁶ Ry, ≈ 14 µeV).

Motivation
----------
``⟨n|Σ|n⟩`` is basis-dependent inside a degenerate manifold: the QE
diagonaliser picks an arbitrary orthonormal basis of the degenerate
subspace, and individual diagonal elements vary with that choice.  Only
the *trace* (= sum of the diagonal values, = sum of eigenvalues) is
basis-invariant.  When the manifold is an irreducible representation of
the crystal point group, Schur's lemma forces all eigenvalues equal, so
the trace divided by the multiplicity equals each eigenvalue — the
"physical" Σ_X for that manifold.  Averaging recovers this value.

The averaging only affects the diagonal of the output matrix; off-diagonal
elements are preserved unchanged (the BGW convention).
"""
from __future__ import annotations

import numpy as np

# BGW Common/nrtype.f90 :: TOL_Degeneracy = 1.0d-6 (Ry)
TOL_DEGENERACY_RY: float = 1.0e-6


def average_within_degenerate_sets(
    values_kn: np.ndarray,
    energies_kn_ry: np.ndarray,
    tol_ry: float = TOL_DEGENERACY_RY,
) -> np.ndarray:
    """Return a copy of ``values_kn`` with each band replaced by the mean
    over its contiguous degenerate set in ``energies_kn_ry``.

    Parameters
    ----------
    values_kn : np.ndarray, shape (nk, nb), real or complex
        Per-(k, band) diagonal values to be averaged.
    energies_kn_ry : np.ndarray, shape (nk, nb)
        DFT eigenvalues in **Rydberg** (matching BGW's tol convention).
    tol_ry : float
        Energy tolerance in Ry for "same eigenvalue".  Default
        ``TOL_DEGENERACY_RY = 1e-6`` matches BGW.

    Returns
    -------
    out : np.ndarray, shape (nk, nb), same dtype as ``values_kn``
        Group-averaged values.
    """
    out = np.array(values_kn, copy=True)
    e = np.asarray(energies_kn_ry, dtype=np.float64)
    if out.shape != e.shape:
        raise ValueError(
            f"average_within_degenerate_sets: values shape {out.shape} "
            f"!= energies shape {e.shape}"
        )
    nk, nb = out.shape
    for k in range(nk):
        i0 = 0
        for i in range(1, nb):
            if abs(e[k, i] - e[k, i - 1]) >= tol_ry:
                if i - i0 > 1:
                    out[k, i0:i] = out[k, i0:i].mean()
                i0 = i
        if nb - i0 > 1:
            out[k, i0:nb] = out[k, i0:nb].mean()
    return out


def apply_to_matrix_diagonals(
    matrix_knn: np.ndarray,
    energies_kn_ry: np.ndarray,
    tol_ry: float = TOL_DEGENERACY_RY,
) -> np.ndarray:
    """Return a copy of ``(nk, nb, nb)`` matrix with its diagonal averaged
    within each degenerate set.  Off-diagonal entries are preserved.

    Mirrors ``shiftenergy.f90``'s averaging over ``ax``, ``asx``, ``ach``
    (each is a 1-D per-band array; only the diagonal is averaged).
    """
    if matrix_knn.ndim != 3 or matrix_knn.shape[1] != matrix_knn.shape[2]:
        raise ValueError(
            f"apply_to_matrix_diagonals: expected (nk, nb, nb), got "
            f"{matrix_knn.shape}"
        )
    diag = np.diagonal(matrix_knn, axis1=1, axis2=2).copy()
    avg = average_within_degenerate_sets(diag, energies_kn_ry, tol_ry)
    out = np.array(matrix_knn, copy=True)
    nk, nb, _ = out.shape
    idx = np.arange(nb)
    for k in range(nk):
        out[k, idx, idx] = avg[k]
    return out
