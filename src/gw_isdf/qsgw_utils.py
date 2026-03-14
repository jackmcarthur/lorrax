"""Utilities for diagonal Sigma(E) fixed points and QSGW static Sigma_xc construction."""

from __future__ import annotations

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
    clipped = 0

    def _interp_matrix_on_grid(values_omega_kij: np.ndarray, x_ev: float) -> np.ndarray:
        nonlocal clipped
        x_lo = float(omega[0])
        x_hi = float(omega[-1])
        x_clamped = float(np.clip(float(x_ev), x_lo, x_hi))
        if x_clamped != float(x_ev):
            clipped += 1
        val_re = np.empty((nb, nb), dtype=np.float64)
        val_im = np.empty((nb, nb), dtype=np.float64)
        for i in range(nb):
            for j in range(nb):
                val_re[i, j] = np.interp(x_clamped, omega, np.real(values_omega_kij[:, i, j]))
                val_im[i, j] = np.interp(x_clamped, omega, np.imag(values_omega_kij[:, i, j]))
        return val_re + 1j * val_im

    for ik in range(nk):
        # "Re" in QSGW corresponds to the Hermitian part of Sigma(omega).
        sigma_h_omega = 0.5 * (sigma[:, ik] + np.conj(np.swapaxes(sigma[:, ik], -1, -2)))
        sigma_eval = []
        for i in range(nb):
            Ei = float(E[ik, i])
            sigma_eval.append(_interp_matrix_on_grid(sigma_h_omega, Ei))
        for i in range(nb):
            for j in range(nb):
                sigma_qsgw[ik, i, j] = 0.5 * (sigma_eval[i][i, j] + sigma_eval[j][i, j])

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
