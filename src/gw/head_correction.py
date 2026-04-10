"""Scalar GN-PPM head correction for the q=0, G=G'=0 Coulomb head.

In plane waves, rho^{mn}_{q=0}(G=0) = delta_{mn}, so the head of W^c
contributes only to diagonal self-energy elements. ISDF cannot represent
this exactly, so we handle it as a separate scalar diagnostic channel
outside the ISDF body path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeadGNParams:
    """Fitted GN parameters for the scalar Coulomb head."""

    omega_h_sq: float
    omega_h: float
    B_h: float
    R_h: float
    wc_head_0: float
    wc_head_iwp: float
    vc0: float
    omega_p: float


def fit_head_gn(
    vc0: float,
    wcoul0_static: float,
    wcoul0_imfreq: float,
    omega_p_ry: float,
) -> HeadGNParams:
    """Fit a scalar GN pole from two W^c head samples."""

    w1 = wcoul0_static - vc0
    w2 = wcoul0_imfreq - vc0
    omega_2_sq = -(omega_p_ry ** 2)

    denom = w1 - w2
    if abs(denom) < 1.0e-30:
        return HeadGNParams(
            omega_h_sq=1.0,
            omega_h=1.0,
            B_h=0.0,
            R_h=0.0,
            wc_head_0=w1,
            wc_head_iwp=w2,
            vc0=vc0,
            omega_p=omega_p_ry,
        )

    omega_h_sq = -w2 * omega_2_sq / denom
    B_h = -w1 * omega_h_sq

    if omega_h_sq <= 0.0:
        omega_h = abs(omega_h_sq) ** 0.5 if omega_h_sq != 0.0 else 1.0
        R_h = B_h / (2.0 * omega_h) if omega_h > 1.0e-30 else 0.0
        return HeadGNParams(
            omega_h_sq=omega_h_sq,
            omega_h=omega_h,
            B_h=B_h,
            R_h=R_h,
            wc_head_0=w1,
            wc_head_iwp=w2,
            vc0=vc0,
            omega_p=omega_p_ry,
        )

    omega_h = omega_h_sq ** 0.5
    R_h = B_h / (2.0 * omega_h)
    return HeadGNParams(
        omega_h_sq=omega_h_sq,
        omega_h=omega_h,
        B_h=B_h,
        R_h=R_h,
        wc_head_0=w1,
        wc_head_iwp=w2,
        vc0=vc0,
        omega_p=omega_p_ry,
    )


_RY2EV = 13.6056980659


def compute_head_sigma_diagonal(
    head: HeadGNParams,
    energies_dft_ry: np.ndarray,
    occ: np.ndarray,
    cell_volume: float,
) -> np.ndarray:
    """Compute the simple on-shell diagonal head shift."""

    if abs(head.R_h) < 1.0e-30 or abs(head.omega_h) < 1.0e-30:
        return np.zeros_like(energies_dft_ry)
    occ_arr = np.asarray(occ, dtype=np.float64)
    return (head.R_h / (head.omega_h * cell_volume)) * (2.0 * occ_arr - 1.0)


def compute_head_sigma_at_omega(
    head: HeadGNParams,
    energies_dft_ry: np.ndarray,
    omega_eval_ry: np.ndarray,
    occ: np.ndarray,
    cell_volume: float,
) -> np.ndarray:
    """Compute the scalar head self-energy at arbitrary frequencies."""

    if abs(head.R_h) < 1.0e-30 or abs(head.omega_h) < 1.0e-30:
        return np.zeros_like(energies_dft_ry, dtype=np.complex128)

    eps = np.asarray(energies_dft_ry, dtype=np.float64)
    omega = np.asarray(omega_eval_ry, dtype=np.float64)
    f = np.asarray(occ, dtype=np.float64)
    eta = 1.0e-6
    occ_term = f / (omega - eps + head.omega_h - 1j * eta)
    emp_term = (1.0 - f) / (omega - eps - head.omega_h + 1j * eta)
    return (head.R_h / cell_volume) * (occ_term + emp_term)


def format_head_diagnostics(head: HeadGNParams, cell_volume: float) -> str:
    """Return a short multiline diagnostic summary for the scalar head fit."""

    lines = [
        "",
        "-" * 72,
        "  HEAD CORRECTION (scalar GN, separate from ISDF body)",
        "-" * 72,
        f"  v(q→0)             = {head.vc0:12.3f} a.u.",
        f"  W^c(q→0, ω=0)      = {head.wc_head_0:12.3f} a.u.",
        f"  W^c(q→0, ω=iωp)    = {head.wc_head_iwp:12.3f} a.u.  [ωp={head.omega_p:.4f} Ry]",
        f"  Ω_h²               = {head.omega_h_sq:12.6f} Ry²",
        f"  Ω_h                = {head.omega_h:12.6f} Ry  ({head.omega_h * _RY2EV:.6f} eV)",
        f"  B_h                = {head.B_h:12.6f} Ry² · a.u.",
        f"  R_h                = {head.R_h:12.6f} Ry · a.u.",
    ]
    if abs(head.omega_h) > 1.0e-30:
        lines.append(
            f"  R_h / (Ω_h · vol)  = {head.R_h / (head.omega_h * cell_volume):12.6e} (Ry)"
        )
    else:
        lines.append("  R_h / (Ω_h · vol)  = 0.0 (degenerate)")
    return "\n".join(lines)
