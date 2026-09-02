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
    sig_h_scalar,
    h_transverse,
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
    - direct-field components             → exact ``V_H + H_T`` total
    - ``sig_sx, sig_coh, sig_h, sig_x``   → consistent sigma_diag.dat
    - ``sigma_c_at_dft_ev`` (1-D, or None) → consistent eqp.dat ``sigC``

    Off-diagonals are preserved.  A replicated input remains replicated.  A
    ``P(None,'x','y')`` input extracts only its ``nk*nb`` diagonal, averages
    that bounded table on host, and writes it back into the existing tiles;
    the ``nk*nb^2`` component is never gathered or replicated.

    Returns the eight inputs, averaged, in the same order.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    import jax
    import jax.numpy as jnp
    from common.collectives import device_put_process_local, gather_to_host
    from gw.qsgw_utils import (
        is_band_sharded_sigma_omega,
        set_band_diag_sharded,
        static_sigma_diag_to_host,
    )

    rep = NamedSharding(mesh_xy, P(None, None, None))
    band_4d = NamedSharding(mesh_xy, P(None, None, "x", "y"))

    def _is_band_sharded_3d(M):
        spec = getattr(getattr(M, "sharding", None), "spec", None)
        if spec is None or getattr(M, "ndim", 0) != 3:
            return False
        spec = tuple(spec) + (None,) * (3 - len(tuple(spec)))
        if spec[1] is None and spec[2] is None:
            return False
        if spec != (None, "x", "y"):
            raise ValueError(
                "static Sigma degeneracy averaging requires canonical "
                "P(None,'x','y') band sharding; got "
                f"PartitionSpec{spec}")
        return True

    def _dav(M):
        # Reuse the dynamic-Sigma diagonal backend by adding a length-one
        # leading axis.  Its shard_map extracts exactly nk*nb values with one
        # psum; the setter changes only diagonal entries in each owned tile.
        if _is_band_sharded_3d(M):
            M4 = jax.lax.with_sharding_constraint(
                jnp.expand_dims(M, axis=0), band_4d)
            if not is_band_sharded_sigma_omega(M4):
                raise ValueError(
                    "band-sharded Sigma lost its matrix-axis layout while "
                    "adding the temporary leading axis")
            diag = static_sigma_diag_to_host(M, mesh_xy)
            averaged = average_within_degenerate_sets(
                diag, energies_kn_ry, tol_ry)
            return set_band_diag_sharded(M4, averaged[None, ...])[0]

        # Process-local replication, NOT plain ``jax.device_put``: the latter
        # fires JAX's hidden ``assert_equal`` all-gather on a multi-process
        # mesh.  Replication is retained only when the input already selected
        # that bounded post-Sigma layout.
        return device_put_process_local(apply_to_matrix_diagonals(
            gather_to_host(M), energies_kn_ry, tol_ry), rep)

    sigma_total = _dav(sigma_total)
    sig_sx, sig_coh, sig_x = _dav(sig_sx), _dav(sig_coh), _dav(sig_x)
    if h_transverse is None:
        # Historical charge-only arithmetic: average the one direct field
        # once, then bind the scalar component to that exact result.
        sig_h = _dav(sig_h)
        sig_h_scalar = sig_h
    else:
        sig_h_scalar = _dav(sig_h_scalar)
        h_transverse = _dav(h_transverse)
        # The aggregate is derived, never independently rounded into a
        # second source of truth.
        sig_h = sig_h_scalar + h_transverse
    if sigma_c_at_dft_ev is not None:
        sigma_c_at_dft_ev = average_within_degenerate_sets(
            np.asarray(sigma_c_at_dft_ev, dtype=np.complex128),
            energies_kn_ry, tol_ry)
    return (sigma_total, sig_sx, sig_coh, sig_h, sig_h_scalar,
            h_transverse, sig_x, sigma_c_at_dft_ev)


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
