"""Utilities for diagonal Sigma(E) fixed points and QSGW static Sigma_xc construction."""

from __future__ import annotations

import h5py
import numpy as np


def _interp_complex_on_grid(omega_ev: np.ndarray, values_omega: np.ndarray, x_ev: float) -> complex:
    """Linear interpolation with edge clamping for one complex-valued frequency trace."""
    xr = float(np.clip(x_ev, float(omega_ev[0]), float(omega_ev[-1])))
    v_re = np.interp(xr, omega_ev, np.real(values_omega))
    v_im = np.interp(xr, omega_ev, np.imag(values_omega))
    return complex(v_re, v_im)


def solve_diagonal_sigma_fixed_point(
    h0_diag_ev: np.ndarray,
    sigma_omega_diag_ev: np.ndarray,
    omega_ev: np.ndarray,
    *,
    max_iter: int = 80,
    tol_ev: float = 1.0e-6,
    mixing: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve E = h0 + Re Sigma(E) for each (k,n) in the diagonal approximation.

    Parameters
    ----------
    h0_diag_ev
        Base one-body diagonal term (typically diag(kin_ion + V_H)) with shape (nk, nb).
    sigma_omega_diag_ev
        Diagonal Sigma_xc(omega) values with shape (n_omega, nk, nb) in eV.
    omega_ev
        Frequency grid in eV (monotonic increasing).
    """
    omega_ev = np.asarray(omega_ev, dtype=np.float64)
    h0_diag_ev = np.asarray(h0_diag_ev, dtype=np.float64)
    sigma_omega_diag_ev = np.asarray(sigma_omega_diag_ev, dtype=np.complex128)
    if sigma_omega_diag_ev.ndim != 3:
        raise ValueError("sigma_omega_diag_ev must have shape (n_omega, nk, nb).")
    if h0_diag_ev.shape != sigma_omega_diag_ev.shape[1:]:
        raise ValueError(
            f"Shape mismatch: h0_diag_ev={h0_diag_ev.shape} vs sigma_omega_diag_ev={sigma_omega_diag_ev.shape}"
        )

    i0 = int(np.argmin(np.abs(omega_ev)))
    E = h0_diag_ev + np.real(sigma_omega_diag_ev[i0])
    converged = np.zeros_like(E, dtype=bool)
    mix = float(np.clip(mixing, 0.0, 1.0))

    for it in range(max_iter):
        E_new = np.empty_like(E)
        for ik in range(E.shape[0]):
            for ib in range(E.shape[1]):
                sig = _interp_complex_on_grid(omega_ev, sigma_omega_diag_ev[:, ik, ib], E[ik, ib])
                E_new[ik, ib] = h0_diag_ev[ik, ib] + float(np.real(sig))
        E_next = (1.0 - mix) * E + mix * E_new
        diff = np.abs(E_next - E)
        converged = diff < tol_ev
        E = E_next
        if bool(np.all(converged)):
            return E, converged, it + 1
    return E, converged, max_iter


def build_qsgw_sigma_xc(
    sigma_xc_omega_kij_ev: np.ndarray,
    omega_ev: np.ndarray,
    qp_energies_kn_ev: np.ndarray,
    *,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    """Build static Hermitian QSGW Sigma_xc from dynamic Sigma_xc(omega).

    Implements:
      Sigma_xc^QSGW_ij(k) = 1/2 * [ Re Sigma_ij(k, E_i(k)) + Re Sigma_ij(k, E_j(k)) ]
    where Re denotes Hermitian part.

    Note: this uses fixed-point (nonlinear) QP energies when available; we do
    not apply a linearized Z*(Sigma - V_xc) correction in the QSGW utilities.
    """
    sigma = np.asarray(sigma_xc_omega_kij_ev, dtype=np.complex128)
    omega = np.asarray(omega_ev, dtype=np.float64)
    E = np.asarray(qp_energies_kn_ev, dtype=np.float64)
    if sigma.ndim != 4:
        raise ValueError("sigma_xc_omega_kij_ev must have shape (n_omega, nk, nb, nb).")
    n_omega, nk, nb, nb2 = sigma.shape
    if nb != nb2:
        raise ValueError("sigma_xc_omega_kij_ev last two dims must be square.")
    if E.shape != (nk, nb):
        raise ValueError(f"qp_energies_kn_ev must have shape ({nk}, {nb}).")

    sigma_qsgw = np.zeros((nk, nb, nb), dtype=np.complex128)

    omega_lo = float(omega[0])
    omega_hi = float(omega[-1])
    E_clamped = np.clip(E, omega_lo, omega_hi)
    clipped = int(np.count_nonzero(E_clamped != E))

    for ik in range(nk):
        # "Re" in QSGW corresponds to the Hermitian part of Sigma(omega).
        sigma_h_omega = 0.5 * (sigma[:, ik] + np.conj(np.swapaxes(sigma[:, ik], -1, -2)))

        # Interpolate the full Hermitian matrix at each quasiparticle energy E_i(k).
        e_eval = E_clamped[ik]
        idx_hi = np.searchsorted(omega, e_eval, side="left")
        idx_hi = np.clip(idx_hi, 1, n_omega - 1)
        idx_lo = idx_hi - 1
        omega_lo_i = omega[idx_lo]
        omega_hi_i = omega[idx_hi]
        denom = np.where(omega_hi_i > omega_lo_i, omega_hi_i - omega_lo_i, 1.0)
        weight_hi = (e_eval - omega_lo_i) / denom
        weight_lo = 1.0 - weight_hi

        sigma_eval = (
            weight_lo[:, None, None] * sigma_h_omega[idx_lo, :, :]
            + weight_hi[:, None, None] * sigma_h_omega[idx_hi, :, :]
        )

        # A[i,j] = Sigma_ij(E_i), B[i,j] = Sigma_ij(E_j)
        band_idx = np.arange(nb)
        A = sigma_eval[band_idx, band_idx, :]
        B = sigma_eval[band_idx, :, band_idx].T
        sigma_qsgw[ik] = 0.5 * (A + B)

    # Enforce exact Hermiticity against interpolation noise.
    sigma_qsgw = 0.5 * (sigma_qsgw + np.conj(np.swapaxes(sigma_qsgw, -1, -2)))
    if return_diagnostics:
        total_evals = float(nk * nb)
        diagnostics = {
            "n_interp_clipped": float(clipped),
            "frac_interp_clipped": (float(clipped) / total_evals) if total_evals > 0 else 0.0,
            "omega_min_ev": float(omega[0]),
            "omega_max_ev": float(omega[-1]),
        }
        return sigma_qsgw, diagnostics
    return sigma_qsgw


def load_sigma_xc_diag_from_h5(
    sigma_h5_path: str,
    sigma_sx_kij_ev: np.ndarray,
    *,
    sigma_c_dataset: str = "sigma_c_kij_ev",
    k_chunk_size: int = 16,
    band_block_size: int = 128,
) -> np.ndarray:
    """Load diagonal Sigma_xc(omega,k,n) from HDF5 without materializing the full matrix."""

    sigma_sx_diag_ev = np.diagonal(np.asarray(sigma_sx_kij_ev, dtype=np.complex128), axis1=1, axis2=2)

    with h5py.File(sigma_h5_path, "r") as h5:
        dset_c = h5[sigma_c_dataset]
        n_omega, nk, nb, nb2 = dset_c.shape
        if nb != nb2:
            raise ValueError(f"{sigma_c_dataset} must be square in band indices.")
        if sigma_sx_diag_ev.shape != (nk, nb):
            raise ValueError(
                f"sigma_sx_kij_ev diagonal shape {sigma_sx_diag_ev.shape} incompatible with HDF5 shape {(nk, nb)}"
            )

        sigma_xc_diag_ev = np.empty((n_omega, nk, nb), dtype=np.complex128)
        k_chunk = max(1, int(k_chunk_size))
        b_chunk = max(1, int(band_block_size))

        for k0 in range(0, nk, k_chunk):
            k1 = min(k0 + k_chunk, nk)
            for b0 in range(0, nb, b_chunk):
                b1 = min(b0 + b_chunk, nb)
                block = np.asarray(dset_c[:, k0:k1, b0:b1, b0:b1], dtype=np.complex128)
                sigma_c_diag = np.diagonal(block, axis1=2, axis2=3)
                sigma_xc_diag_ev[:, k0:k1, b0:b1] = sigma_c_diag + sigma_sx_diag_ev[None, k0:k1, b0:b1]

    return sigma_xc_diag_ev


def _interp_rows_on_grid(values_omega_ij: np.ndarray, eval_ev: np.ndarray, omega_ev: np.ndarray) -> np.ndarray:
    """Interpolate rows of a frequency-dependent matrix block at row-specific energies.

    Parameters
    ----------
    values_omega_ij
        Shape (n_omega, n_row, n_col).
    eval_ev
        Shape (n_row,), evaluation energy for each row.
    omega_ev
        Monotonic frequency grid.
    """
    eval_ev = np.asarray(eval_ev, dtype=np.float64)
    idx_hi = np.searchsorted(omega_ev, eval_ev, side="left")
    idx_hi = np.clip(idx_hi, 1, len(omega_ev) - 1)
    idx_lo = idx_hi - 1
    omega_lo = omega_ev[idx_lo]
    omega_hi = omega_ev[idx_hi]
    denom = np.where(omega_hi > omega_lo, omega_hi - omega_lo, 1.0)
    weight_hi = (eval_ev - omega_lo) / denom
    weight_lo = 1.0 - weight_hi
    row_idx = np.arange(values_omega_ij.shape[1])
    return (
        weight_lo[:, None] * values_omega_ij[idx_lo, row_idx, :]
        + weight_hi[:, None] * values_omega_ij[idx_hi, row_idx, :]
    )


def build_qsgw_sigma_xc_from_h5(
    sigma_h5_path: str,
    sigma_sx_kij_ev: np.ndarray,
    omega_ev: np.ndarray,
    qp_energies_kn_ev: np.ndarray,
    *,
    sigma_c_dataset: str = "sigma_c_kij_ev",
    output_dataset: str = "sigma_xc_qsgw_kij_ev",
    k_chunk_size: int = 1,
    band_block_size: int = 128,
) -> dict[str, float]:
    """Build static Hermitian QSGW Sigma_xc directly from chunked HDF5 reads.

    This avoids materializing the full dynamic `(omega,k,m,n)` tensor on host memory.
    """

    sigma_sx = np.asarray(sigma_sx_kij_ev, dtype=np.complex128)
    omega = np.asarray(omega_ev, dtype=np.float64)
    E = np.asarray(qp_energies_kn_ev, dtype=np.float64)
    if sigma_sx.ndim != 3:
        raise ValueError("sigma_sx_kij_ev must have shape (nk, nb, nb).")

    nk, nb, nb2 = sigma_sx.shape
    if nb != nb2:
        raise ValueError("sigma_sx_kij_ev must be square in band indices.")
    if E.shape != (nk, nb):
        raise ValueError(f"qp_energies_kn_ev must have shape ({nk}, {nb}).")

    omega_lo = float(omega[0])
    omega_hi = float(omega[-1])
    E_clamped = np.clip(E, omega_lo, omega_hi)
    clipped = int(np.count_nonzero(E_clamped != E))

    with h5py.File(sigma_h5_path, "a") as h5:
        dset_c = h5[sigma_c_dataset]
        n_omega, nk_h5, nb_h5, nb2_h5 = dset_c.shape
        if (nk_h5, nb_h5, nb2_h5) != (nk, nb, nb):
            raise ValueError(
                f"{sigma_c_dataset} shape {(nk_h5, nb_h5, nb2_h5)} incompatible with sigma_sx {sigma_sx.shape}"
            )
        if output_dataset in h5:
            del h5[output_dataset]
        dset_out = h5.create_dataset(
            output_dataset,
            shape=(nk, nb, nb),
            dtype=np.complex128,
            chunks=(max(1, min(int(k_chunk_size), nk)), max(1, min(int(band_block_size), nb)), max(1, min(int(band_block_size), nb))),
        )

        k_chunk = max(1, int(k_chunk_size))
        b_chunk = max(1, int(band_block_size))

        for k0 in range(0, nk, k_chunk):
            k1 = min(k0 + k_chunk, nk)
            for i0 in range(0, nb, b_chunk):
                i1 = min(i0 + b_chunk, nb)
                for j0 in range(0, nb, b_chunk):
                    j1 = min(j0 + b_chunk, nb)
                    block_c_ij = np.asarray(dset_c[:, k0:k1, i0:i1, j0:j1], dtype=np.complex128)
                    block_x_ij = sigma_sx[k0:k1, i0:i1, j0:j1][None, ...]
                    block_xc_ij = block_c_ij + block_x_ij
                    if i0 == j0:
                        block_h_ij = 0.5 * (
                            block_xc_ij + np.conj(np.swapaxes(block_xc_ij, -1, -2))
                        )
                    else:
                        block_c_ji = np.asarray(dset_c[:, k0:k1, j0:j1, i0:i1], dtype=np.complex128)
                        block_x_ji = sigma_sx[k0:k1, j0:j1, i0:i1][None, ...]
                        block_xc_ji = block_c_ji + block_x_ji
                        block_h_ij = 0.5 * (
                            block_xc_ij + np.conj(np.swapaxes(block_xc_ji, -1, -2))
                        )

                    out_block = np.empty((k1 - k0, i1 - i0, j1 - j0), dtype=np.complex128)
                    for kk in range(k1 - k0):
                        eval_rows = E_clamped[k0 + kk, i0:i1]
                        eval_cols = E_clamped[k0 + kk, j0:j1]
                        sigma_h_omega = block_h_ij[:, kk, :, :]
                        row_eval = _interp_rows_on_grid(sigma_h_omega, eval_rows, omega)
                        col_eval = _interp_rows_on_grid(np.swapaxes(sigma_h_omega, 1, 2), eval_cols, omega).T
                        out_block[kk] = 0.5 * (row_eval + col_eval)
                    dset_out[k0:k1, i0:i1, j0:j1] = out_block

    total_evals = float(nk * nb)
    return {
        "n_interp_clipped": float(clipped),
        "frac_interp_clipped": (float(clipped) / total_evals) if total_evals > 0 else 0.0,
        "omega_min_ev": omega_lo,
        "omega_max_ev": omega_hi,
    }


def plot_qp_energy_comparison(
    output_png: str,
    e_ref_kn_ev: np.ndarray,
    e_static_kn_ev: np.ndarray,
    e_dyn0_kn_ev: np.ndarray,
    e_diag_sc_kn_ev: np.ndarray,
) -> str:
    """Create a simple comparison plot of QP energies."""
    import matplotlib.pyplot as plt

    x = np.asarray(e_ref_kn_ev, dtype=np.float64).reshape(-1)
    y_static = np.asarray(e_static_kn_ev, dtype=np.float64).reshape(-1)
    y_dyn0 = np.asarray(e_dyn0_kn_ev, dtype=np.float64).reshape(-1)
    y_diag = np.asarray(e_diag_sc_kn_ev, dtype=np.float64).reshape(-1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(x, y_static, s=10, alpha=0.6, label="Static COHSEX")
    axes[0].scatter(x, y_dyn0, s=10, alpha=0.6, label="Bare X + Sigma_c(0)")
    axes[0].scatter(x, y_diag, s=10, alpha=0.6, label="Diagonal SC Sigma(E)")
    mn = float(min(np.min(x), np.min(y_static), np.min(y_dyn0), np.min(y_diag)))
    mx = float(max(np.max(x), np.max(y_static), np.max(y_dyn0), np.max(y_diag)))
    axes[0].plot([mn, mx], [mn, mx], "k--", lw=1)
    axes[0].set_xlabel("Reference energy (eV)")
    axes[0].set_ylabel("QP energy (eV)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("All (k,n)")

    # k=0 band trend
    b = np.arange(e_ref_kn_ev.shape[1])
    axes[1].plot(b, e_static_kn_ev[0], "-o", ms=3, label="Static COHSEX")
    axes[1].plot(b, e_dyn0_kn_ev[0], "-o", ms=3, label="Bare X + Sigma_c(0)")
    axes[1].plot(b, e_diag_sc_kn_ev[0], "-o", ms=3, label="Diagonal SC Sigma(E)")
    axes[1].set_xlabel("Band index")
    axes[1].set_ylabel("Energy at k=0 (eV)")
    axes[1].set_title("k = 0")
    axes[1].legend(fontsize=8)

    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return output_png
