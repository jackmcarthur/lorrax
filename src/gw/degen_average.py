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


def average_sigma_components(
    sigma_total,
    sig_sx,
    sig_coh,
    sig_h,
    sig_x,
    sigma_c_at_dft_ev,
    *,
    energies_kn_ry: np.ndarray,
    tol_ry: float,
    mesh_xy,
):
    """BGW-style degenerate-set averaging at the H-build seam.

    Mirrors ``Sigma/shiftenergy.f90``; replaces the previous per-component
    averaging at the writer.  Applied to:

    - ``sigma_total``'s diagonal          → consistent E_qp from eigh
    - ``sig_sx, sig_coh, sig_h, sig_x``   → consistent sigma_diag.dat
    - ``sigma_c_at_dft_ev`` (1-D, or None) → consistent eqp.dat ``sigC``

    Off-diagonals are preserved.  The averaging is cheap (a host loop
    over k of contiguous degeneracy groups) so the redundancy across
    components is not a perf concern.  Matrices come back replicated on
    ``mesh_xy`` (``P(None, None, None)``), matching the post-Σ seam.

    Returns the six inputs, averaged, in the same order.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local

    rep = NamedSharding(mesh_xy, P(None, None, None))

    # Process-local replication, NOT plain ``jax.device_put``: the latter
    # fires JAX's hidden ``assert_equal`` all-gather on a multi-process
    # mesh — P × nk × nb² × 16 B per matrix, six matrices (scorecard
    # AA.1).  Each averaged matrix is a deterministic host function of
    # the (replicated) Σ components, identical on every rank;
    # ``LORRAX_CHECK_REPLICA=1`` re-arms the assertion.
    def _dav(M):
        return device_put_process_local(apply_to_matrix_diagonals(
            np.asarray(M), energies_kn_ry, tol_ry), rep)

    sigma_total = _dav(sigma_total)
    sig_sx, sig_coh, sig_h, sig_x = (
        _dav(sig_sx), _dav(sig_coh), _dav(sig_h), _dav(sig_x))
    if sigma_c_at_dft_ev is not None:
        sigma_c_at_dft_ev = average_within_degenerate_sets(
            np.asarray(sigma_c_at_dft_ev, dtype=np.complex128),
            energies_kn_ry, tol_ry)
    return sigma_total, sig_sx, sig_coh, sig_h, sig_x, sigma_c_at_dft_ev


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
