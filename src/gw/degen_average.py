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

This module averages only extracted per-band reporting arrays. Full operators
are never modified: diagonal-only changes are not covariant under changes
of basis inside a degenerate manifold.
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
    """Average trailing ``(k, band)`` values over each degenerate set.

    ``values_kn`` may carry any number of leading spectral/component axes;
    only its trailing ``(nk, nb)`` must match ``energies_kn_ry``.  This is the
    one group owner for both ordinary diagonals and output-only C(omega)
    diagonal curves, so C(E_DFT) and its derivative cannot be conditioned by
    different loops.

    Parameters
    ----------
    values_kn : np.ndarray, shape (..., nk, nb), real or complex
        Per-(k, band) values, optionally with leading spectral axes.
    energies_kn_ry : np.ndarray, shape (nk, nb)
        DFT eigenvalues in **Rydberg** (matching BGW's tol convention).
    tol_ry : float
        Energy tolerance in Ry for "same eigenvalue".  Default
        ``TOL_DEGENERACY_RY = 1e-6`` matches BGW.

    Returns
    -------
    out : np.ndarray, shape (..., nk, nb), same dtype as ``values_kn``
        Group-averaged values.
    """
    out = np.array(values_kn, copy=True)
    e = np.asarray(energies_kn_ry, dtype=np.float64)
    if out.ndim < 2 or out.shape[-2:] != e.shape:
        raise ValueError(
            "average_within_degenerate_sets: values trailing shape "
            f"{out.shape[-2:] if out.ndim >= 2 else out.shape} != energies "
            f"shape {e.shape}"
        )
    nk, nb = e.shape
    for k in range(nk):
        i0 = 0
        for i in range(1, nb):
            if abs(e[k, i] - e[k, i - 1]) >= tol_ry:
                if i - i0 > 1:
                    out[..., k, i0:i] = out[..., k, i0:i].mean(
                        axis=-1, keepdims=True)
                i0 = i
        if nb - i0 > 1:
            out[..., k, i0:nb] = out[..., k, i0:nb].mean(
                axis=-1, keepdims=True)
    return out
