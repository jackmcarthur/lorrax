"""Scalar GN-PPM head correction for the q=0, G=G'=0 Coulomb head.

In plane waves, rho^{mn}_{q=0}(G=0) = delta_{mn}, so the head of W^c
contributes only to diagonal self-energy elements.  ISDF cannot represent
this exactly, so we handle it as a separate scalar channel outside the
ISDF body path.

The module:
  1. Takes the two scalar head samples W^c_{00}(q=0, omega=0) and
     W^c_{00}(q=0, omega=i*omega_p).
  2. Fits a single-pole GN model: W^c_{00}(omega) = B_h / (omega^2 - Omega_h^2).
  3. Evaluates the diagonal head self-energy correction at requested energies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeadGNParams:
    """Fitted GN parameters for the scalar Coulomb head."""
    omega_h_sq: float   # Omega_h^2 (Ry^2)
    omega_h: float       # Omega_h (Ry), positive
    B_h: float           # Residue numerator (Ry^2 * a.u.)
    R_h: float           # B_h / (2 * Omega_h) (Ry * a.u.)
    wc_head_0: float     # W^c_{00}(omega=0) input (a.u.)
    wc_head_iwp: float   # W^c_{00}(omega=i*omega_p) input (a.u.)
    vc0: float           # v(q->0) bare head (a.u.)
    omega_p: float       # Probe frequency (Ry)


def fit_head_gn(
    vc0: float,
    wcoul0_static: float,
    wcoul0_imfreq: float,
    omega_p_ry: float,
) -> HeadGNParams:
    """Fit a scalar GN plasmon-pole model to the q=0 head.

    Parameters
    ----------
    vc0 : float
        Bare Coulomb head v(q->0) in a.u.
    wcoul0_static : float
        Screened Coulomb head W(q->0, omega=0) in a.u.
    wcoul0_imfreq : float
        Screened Coulomb head W(q->0, omega=i*omega_p) in a.u.
    omega_p_ry : float
        The imaginary probe frequency in Ry.

    Returns
    -------
    HeadGNParams
        Fitted parameters for the scalar head GN model.
    """
    # W^c = W - v  (correlation part of the head)
    w1 = wcoul0_static - vc0       # W^c(omega=0)
    w2 = wcoul0_imfreq - vc0       # W^c(omega=i*omega_p)

    # GN model: W^c(omega) = B_h / (omega^2 - Omega_h^2)
    # At omega=0:        w1 = B_h / (0 - Omega_h^2) = -B_h / Omega_h^2
    # At omega=i*omega_p: w2 = B_h / (-omega_p^2 - Omega_h^2)
    #
    # From the two-frequency GN fit:
    #   Omega_h^2 = (w1 * omega_1^2 - w2 * omega_2^2) / (w1 - w2)
    # where omega_1 = 0, omega_2 = i*omega_p so omega_2^2 = -omega_p^2
    omega_1_sq = 0.0
    omega_2_sq = -(omega_p_ry ** 2)  # (i*omega_p)^2 = -omega_p^2

    denom = w1 - w2
    if abs(denom) < 1e-30:
        # Degenerate case: both head values are the same.
        # Fall back to a zero correction.
        return HeadGNParams(
            omega_h_sq=1.0, omega_h=1.0, B_h=0.0, R_h=0.0,
            wc_head_0=w1, wc_head_iwp=w2, vc0=vc0, omega_p=omega_p_ry,
        )

    omega_h_sq = (w1 * omega_1_sq - w2 * omega_2_sq) / denom

    # B_h = w1 * (omega_1^2 - Omega_h^2) = w1 * (0 - Omega_h^2) = -w1 * Omega_h^2
    B_h = -w1 * omega_h_sq

    # Omega_h must be real and positive for a physical plasmon pole.
    if omega_h_sq <= 0.0:
        # Unphysical pole — the head screening doesn't fit a simple PPM.
        # Use the static value as a fallback (no dynamic correction).
        omega_h = abs(omega_h_sq) ** 0.5 if omega_h_sq != 0 else 1.0
        R_h = B_h / (2.0 * omega_h) if omega_h > 1e-30 else 0.0
        return HeadGNParams(
            omega_h_sq=omega_h_sq, omega_h=omega_h, B_h=B_h, R_h=R_h,
            wc_head_0=w1, wc_head_iwp=w2, vc0=vc0, omega_p=omega_p_ry,
        )

    omega_h = omega_h_sq ** 0.5
    R_h = B_h / (2.0 * omega_h)

    return HeadGNParams(
        omega_h_sq=omega_h_sq, omega_h=omega_h, B_h=B_h, R_h=R_h,
        wc_head_0=w1, wc_head_iwp=w2, vc0=vc0, omega_p=omega_p_ry,
    )


# Ry-to-eV conversion factor
_RY2EV = 13.6056980659


def compute_head_sigma_diagonal(
    head: HeadGNParams,
    energies_dft_ry: np.ndarray,
    occ: np.ndarray,
    cell_volume: float,
) -> np.ndarray:
    """Compute the diagonal head self-energy correction Sigma^{c,head}_{nn}(E_n).

    The GN head self-energy for band n evaluated at E = E_n^DFT:

      Sigma^{c,head}_{nn}(E_n) = (R_h / cell_volume) * [
          f_n / (E_n - eps_n + Omega_h)       (occupied)
        + (1 - f_n) / (E_n - eps_n - Omega_h)  (empty)
      ]

    When evaluated at E = eps_n (DFT eigenvalue), the (E - eps_n) terms
    vanish, giving:

      Sigma^{c,head}_{nn}(eps_n) = (R_h / cell_volume) * [
          f_n / Omega_h          (occupied)
        + (1 - f_n) / (-Omega_h)  (empty)
      ]

    Parameters
    ----------
    head : HeadGNParams
        Fitted head GN parameters.
    energies_dft_ry : np.ndarray
        DFT eigenvalues in Ry, shape (nk, nb).
    occ : np.ndarray
        Occupation numbers (1.0 for occupied, 0.0 for empty), shape (nk, nb).
    cell_volume : float
        Unit cell volume in a.u.^3.

    Returns
    -------
    np.ndarray
        Head self-energy correction in Ry, shape (nk, nb). Diagonal only.
    """
    R_h = head.R_h
    omega_h = head.omega_h

    if abs(R_h) < 1e-30 or abs(omega_h) < 1e-30:
        return np.zeros_like(energies_dft_ry)

    vol_factor = 1.0 / cell_volume

    # Evaluated at E = eps_n (on-shell):
    #   occupied:  R_h / Omega_h
    #   empty:     R_h / (-Omega_h) = -R_h / Omega_h
    # So: sigma_head_nn = (R_h / Omega_h) * (2*f_n - 1) / cell_volume
    # i.e. +R_h/Omega_h for occupied, -R_h/Omega_h for empty
    occ_arr = np.asarray(occ, dtype=np.float64)
    sigma_head = vol_factor * (R_h / omega_h) * (2.0 * occ_arr - 1.0)

    return sigma_head


def compute_head_sigma_at_omega(
    head: HeadGNParams,
    energies_dft_ry: np.ndarray,
    omega_eval_ry: np.ndarray,
    occ: np.ndarray,
    cell_volume: float,
) -> np.ndarray:
    """Compute head self-energy at arbitrary evaluation frequencies.

    Sigma^{c,head}_{nn}(omega) = (1/vol) * R_h * [
        f_n / (omega - eps_n + Omega_h)
      + (1 - f_n) / (omega - eps_n - Omega_h)
    ]

    Parameters
    ----------
    head : HeadGNParams
    energies_dft_ry : np.ndarray
        DFT eigenvalues, shape (nk, nb).
    omega_eval_ry : np.ndarray
        Evaluation frequencies relative to E_F, shape (nk, nb).
    occ : np.ndarray
        Occupation, shape (nk, nb).
    cell_volume : float

    Returns
    -------
    np.ndarray
        Shape (nk, nb), complex.
    """
    R_h = head.R_h
    omega_h = head.omega_h

    if abs(R_h) < 1e-30 or abs(omega_h) < 1e-30:
        return np.zeros_like(energies_dft_ry, dtype=np.complex128)

    vol_factor = 1.0 / cell_volume
    eps = np.asarray(energies_dft_ry, dtype=np.float64)
    omega = np.asarray(omega_eval_ry, dtype=np.float64)
    f = np.asarray(occ, dtype=np.float64)

    # Small imaginary part for pole regularization
    eta = 1e-6

    occ_term = f / (omega - eps + omega_h - 1j * eta)
    emp_term = (1.0 - f) / (omega - eps - omega_h + 1j * eta)

    sigma = vol_factor * R_h * (occ_term + emp_term)
    return sigma


def format_head_diagnostics(head: HeadGNParams, cell_volume: float) -> str:
    """Return a multi-line diagnostic string for the head GN fit."""
    lines = [
        "",
        "-" * 72,
        "  HEAD CORRECTION (scalar GN, separate from ISDF body)",
        "-" * 72,
        f"  v(q→0)             = {head.vc0:12.3f} a.u.",
        f"  W^c(q→0, ω=0)     = {head.wc_head_0:12.3f} a.u.",
        f"  W^c(q→0, ω=iωp)   = {head.wc_head_iwp:12.3f} a.u.  [ωp={head.omega_p:.4f} Ry]",
        f"  Ω_h²               = {head.omega_h_sq:12.6f} Ry²",
        f"  Ω_h                = {head.omega_h:12.6f} Ry  ({head.omega_h * _RY2EV:.6f} eV)",
        f"  B_h                = {head.B_h:12.6f} Ry² · a.u.",
        f"  R_h                = {head.R_h:12.6f} Ry · a.u.",
        f"  R_h / (Ω_h · vol)  = {head.R_h / (head.omega_h * cell_volume):12.6e} (on-shell shift per state, Ry)"
        if abs(head.omega_h) > 1e-30 else
        f"  R_h / (Ω_h · vol)  = 0.0 (degenerate)",
    ]
    return "\n".join(lines)
