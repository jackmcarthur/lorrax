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


def round_band_window_to_closed_shell(
    energies_kn_ry: np.ndarray,
    b_hi: int,
    tol_ry: float = TOL_DEGENERACY_RY,
    direction: str = "down",
) -> int:
    """Round a band-window boundary so it does not split a degenerate shell.

    A window boundary ``b`` (the exclusive upper index of ``[.., b)`` / the
    inclusive lower index of ``[b, ..)``) is **degeneracy-closed** when band
    ``b-1`` and band ``b`` are gap-separated at *every* k:

        min_k( e[k, b] - e[k, b-1] ) > tol_ry .

    Splitting a degenerate multiplet at a window edge makes the transition /
    pair densities built from that window non-covariant under the crystal
    symmetry *exactly* at the high-symmetry k where the multiplet lives — the
    Round-2 root cause of the ISDF ζ-fit tile non-covariance
    (``reports/bse_refactor_map_2026-07-15`` PHASE2_LOG; FINDINGS2 Task 2/3):
    at those k the kept partial multiplet is not an irrep of the little group,
    so ``ζ̃_{α(μ)} ≠ e^{-iG·τ} ζ̃_μ(RᵀG)`` and every downstream V_q/W_q tile
    inherits the defect (amplified by the CCT conditioning).  This returns the
    nearest boundary that keeps whole shells.

    Uses the same contiguous-degenerate-group logic (BGW ``TOL_Degeneracy``,
    ``Common/nrtype.f90``) as :func:`average_within_degenerate_sets`; there is
    no separate multiplet-detector.

    Parameters
    ----------
    energies_kn_ry : np.ndarray, shape (nk, nb)
        DFT eigenvalues in **Rydberg**, over the k-points (and/or spins,
        flattened onto the leading axis) whose covariance the window must
        respect.  IBZ k-points suffice: every full-BZ k is a symmetry image of
        an IBZ parent with identical energies, so the min-over-k gap is the
        same.  To test the top boundary ``b_hi`` the array **must extend above
        it** (band ``b_hi`` must exist); otherwise ``b_hi`` sits at the file
        edge and is reported closed (no neighbour to split).
    b_hi : int
        The boundary to round, in ``[0, nb]``.
    tol_ry : float
        "Same eigenvalue" tolerance in Ry.  Default ``TOL_DEGENERACY_RY``.
    direction : {'down', 'up'}
        ``'down'`` — largest closed ``b <= b_hi`` (drop partial bands off the
        top; the screening-window use).  ``'up'`` — smallest closed
        ``b >= b_hi`` (raise a lower boundary; the degeneracy-gate valence
        bottom).

    Returns
    -------
    b : int
        The rounded, degeneracy-closed boundary.  The edges ``b == 0`` and
        ``b == nb`` are closed by definition (no band across the boundary), so
        the search always terminates.
    """
    e = np.asarray(energies_kn_ry, dtype=np.float64)
    if e.ndim != 2:
        raise ValueError(
            f"round_band_window_to_closed_shell: energies must be (nk, nb), "
            f"got {e.shape}")
    nb = e.shape[1]
    b_hi = int(b_hi)
    if not (0 <= b_hi <= nb):
        raise ValueError(
            f"round_band_window_to_closed_shell: b_hi={b_hi} outside [0, {nb}]")

    def _closed(b: int) -> bool:
        # Edges (b == 0 or b == nb): no band across the boundary → closed.
        if b <= 0 or b >= nb:
            return True
        return float(np.min(e[:, b] - e[:, b - 1])) > tol_ry

    if direction == "down":
        b = b_hi
        while b > 0 and not _closed(b):
            b -= 1
        return b
    if direction == "up":
        b = b_hi
        while b < nb and not _closed(b):
            b += 1
        return b
    raise ValueError(
        f"round_band_window_to_closed_shell: direction must be 'down' or 'up', "
        f"got {direction!r}")


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
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    rep = NamedSharding(mesh_xy, P(None, None, None))

    def _dav(M):
        return jax.device_put(apply_to_matrix_diagonals(
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
