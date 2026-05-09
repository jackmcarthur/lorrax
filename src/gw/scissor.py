"""Scissor-shift extrapolation for self-consistent GW.

The dynamic self-energy Σ_c(ω) in LORRAX is computed on a small frequency grid
(typically ±10 eV around E_F) while the full band range used in the QP
Hamiltonian spans tens of eV below E_F to hundreds or thousands of eV above.
Inside the grid we evaluate Σ_xc by interpolating the stored (ω, k, m, n)
tensor; outside the grid we extrapolate the QP correction

    ΔE_nk := E_QP_nk − E_DFT_nk  =  ⟨n| Σ_xc − V_xc^DFT |n⟩_k

by a pair of affine laws, refit at each SC iteration — one for valence bands
(occupied at DFT occupation) and one for conduction:

    ΔE_v(E) = α_v · E + β_v
    ΔE_c(E) = α_c · E + β_c

The scissor is applied at the **eigenvalue level** in the diagonal Σ(E)
fixed-point: out-of-grid bands receive E_QP = E_DFT + (α·E_DFT + β); the
QSGW Σ_xc that enters the QP Hamiltonian then evaluates the dynamic
correlation at this scissor-corrected E_QP.  No matrix-level diagonal-add
is exposed because the post-self-energy plumbing keeps H replicated, so a
plain ``H.at[:, idx, idx].add(diag)`` suffices when the caller wants one.

Per-band classification
-----------------------
A band ``n`` is **in-grid** iff ``E_DFT[k, n]`` lies inside ``[ω_min, ω_max]``
for **every** k.  If any single k for that band lies outside the window the
whole band is treated as out-of-grid — the diagonal Σ(E) fixed-point clipped
Σ_c at the ω-boundary for the offending k, which contaminates the QP
correction at neighbouring k via the band's k-dispersion.  Per-band
classification gives a discontinuity-free scissor: out-of-grid bands receive
the affine law uniformly across k, and the fit itself is restricted to data
from in-grid bands so the contaminated points never enter.

Units and layout
----------------
- Energies: eV.  The caller decides whether to work in absolute or
  Fermi-referenced coordinates; the fit and ``predict`` are domain-agnostic
  as long as inputs at fit time and apply time share the same reference.
- Per-band arrays: ``(nk, nb)``.  Fit / extrapolate are pure NumPy since the
  fit consumes O(nk·nb_σ) scalars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScissorFit:
    """Affine scissor-shift law fit to ``E_QP = α·E_DFT + β`` for
    valence and conduction.

    α is the **stretching factor**:

    - α = 1   ⇔  rigid shift  (E_QP = E_DFT + β)
    - α > 1   ⇔  the band manifold is stretched (gap opens further)
    - α < 1   ⇔  the band manifold is compressed (gap closes)

    Attributes
    ----------
    alpha_v, beta_v_ev : float
        Valence stretching factor and intercept (eV), so
        ``E_QP_v(E) = alpha_v · E + beta_v_ev``.
    alpha_c, beta_c_ev : float
        Analogous for conduction.
    n_fit_v, n_fit_c : int
        Number of (k, n) points used in each fit.
    rmse_v_ev, rmse_c_ev : float
        Fit residual RMSE on the QP correction ΔE = E_QP − E_DFT, in eV.
        0 when the fit has fewer than 2 points.
    """

    alpha_v: float
    beta_v_ev: float
    alpha_c: float
    beta_c_ev: float
    n_fit_v: int
    n_fit_c: int
    rmse_v_ev: float
    rmse_c_ev: float

    def predict(self, E_ev: np.ndarray, valence_mask: np.ndarray) -> np.ndarray:
        """Evaluate the scissor **correction** ΔE = E_QP − E_DFT at each (k, n).

        Returns ``ΔE = (α − 1) · E + β`` so callers can add it to their
        DFT energies to get E_QP — same usage as before, just clearer
        semantics on the stored α.

        Parameters
        ----------
        E_ev : np.ndarray, (nk, nb)
            Input DFT energies, same reference as at fit time.
        valence_mask : np.ndarray, (nk, nb) of bool
            True where the valence law applies; False → conduction.
        """
        E = np.asarray(E_ev, dtype=np.float64)
        vm = np.asarray(valence_mask, dtype=bool)
        delta_v = (self.alpha_v - 1.0) * E + self.beta_v_ev
        delta_c = (self.alpha_c - 1.0) * E + self.beta_c_ev
        return np.where(vm, delta_v, delta_c)

    def summary(self) -> str:
        return (
            f"ScissorFit(val: α={self.alpha_v:+.4f}, β={self.beta_v_ev:+.4f} eV, "
            f"n={self.n_fit_v}, rmse={self.rmse_v_ev:.3f} eV; "
            f"cond: α={self.alpha_c:+.4f}, β={self.beta_c_ev:+.4f} eV, "
            f"n={self.n_fit_c}, rmse={self.rmse_c_ev:.3f} eV)  "
            f"[α=1 ⇔ rigid shift]"
        )


# ---------------------------------------------------------------------------
# Per-band in-grid classification
# ---------------------------------------------------------------------------

def classify_bands_in_grid(
    E_kn_ev: np.ndarray,
    omega_min_ev: float,
    omega_max_ev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(band_in_grid, kn_in_grid_band)``.

    A band is **in-grid** iff ``E_kn`` lies in ``[ω_min, ω_max]`` for every k.

    Parameters
    ----------
    E_kn_ev : np.ndarray, (nk, nb)
        Per-(k, n) energies, same reference as the ω-grid.
    omega_min_ev, omega_max_ev : float
        Σ_c(ω) grid bounds.

    Returns
    -------
    band_in_grid : np.ndarray, (nb,) of bool
        True for bands whose every k lies in the window.
    kn_in_grid_band : np.ndarray, (nk, nb) of bool
        ``band_in_grid`` broadcast to ``(nk, nb)`` for ``np.where`` callers.
    """
    E = np.asarray(E_kn_ev, dtype=np.float64)
    in_window_kn = (E >= float(omega_min_ev)) & (E <= float(omega_max_ev))
    band_in_grid = np.all(in_window_kn, axis=0)
    kn_in_grid_band = np.broadcast_to(
        band_in_grid[None, :], E.shape).astype(bool)
    return band_in_grid, kn_in_grid_band


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def _ols_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Closed-form OLS line fit; returns (slope, intercept, RMSE)."""
    n = int(x.size)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return 0.0, float(y[0]), 0.0
    xm = float(x.mean())
    ym = float(y.mean())
    dx = x - xm
    denom = float(np.dot(dx, dx))
    if denom < 1.0e-30:
        return 0.0, ym, float(np.sqrt(np.mean((y - ym) ** 2)))
    slope = float(np.dot(dx, y - ym) / denom)
    intercept = float(ym - slope * xm)
    resid = y - (slope * x + intercept)
    return slope, intercept, float(np.sqrt(np.mean(resid * resid)))


def fit_scissor(
    E_dft_kn_ev: np.ndarray,
    E_qp_kn_ev: np.ndarray,
    valence_mask_kn: np.ndarray,
    fit_mask_kn: np.ndarray,
) -> ScissorFit:
    """Fit valence / conduction scissor lines to (E_DFT, ΔE) samples.

    **Sort-and-pair** semantics: at each k, both ``E_DFT_kn_ev`` and
    ``E_qp_kn_ev`` are sorted ascending **independently**, then paired
    by sorted index.  The fit pair is

        ``( E_DFT_sorted[k, p],
            E_qp_sorted[k, p] − E_DFT_sorted[k, p] )``

    i.e. the p-th lowest QP eigenvalue at k vs. the p-th lowest DFT
    eigenvalue at k.  This is robust to QP reorderings (where the QSGW
    diagonalisation places eigenvalues in an order that does not match
    the DFT band labels) — band identity is dropped in favour of
    energy-sorted matching, which is the right pairing for a scissor
    fit (a smooth function of E).

    The ``valence_mask_kn`` / ``fit_mask_kn`` are also reordered by
    the **DFT** sort permutation (since "occupied" and "in-window" are
    properties tied to the DFT band each E_DFT_sorted[k, p] came from).

    Parameters
    ----------
    E_dft_kn_ev : np.ndarray, (nk, nb)
        DFT eigenvalues per (k, n).  Pre-shifted by the caller's
        choice of reference (typically E_F-relative).
    E_qp_kn_ev : np.ndarray, (nk, nb), real or complex
        QP eigenvalues per (k, n) at the corresponding state.  Complex
        inputs are reduced to the real part, matching the convention of
        ``solve_diagonal_sigma_fixed_point``.
    valence_mask_kn : np.ndarray, (nk, nb) of bool
        True for occupied bands (at DFT occupation), False for conduction.
    fit_mask_kn : np.ndarray, (nk, nb) of bool
        True where the point is trusted enough to enter the fit —
        typically "E_kn lies inside the Σ(ω) grid".  Applied
        independently within valence and conduction.
    """
    E_dft = np.asarray(E_dft_kn_ev, dtype=np.float64)
    E_qp = np.real(np.asarray(E_qp_kn_ev, dtype=np.complex128))
    vm = np.asarray(valence_mask_kn, dtype=bool)
    fm = np.asarray(fit_mask_kn, dtype=bool)
    if not (E_dft.shape == E_qp.shape == vm.shape == fm.shape):
        raise ValueError(
            f"shape mismatch: E_DFT={E_dft.shape} E_QP={E_qp.shape} "
            f"valence={vm.shape} fit={fm.shape}"
        )

    nk = E_dft.shape[0]
    rows = np.arange(nk)[:, None]
    order_dft = np.argsort(E_dft, axis=1)
    order_qp = np.argsort(E_qp, axis=1)
    E_dft_sorted = E_dft[rows, order_dft]
    E_qp_sorted = E_qp[rows, order_qp]

    # Masks live in DFT-band-identity space; reorder by DFT permutation.
    vm_sorted = vm[rows, order_dft]
    fm_sorted = fm[rows, order_dft]

    mask_v = vm_sorted & fm_sorted
    mask_c = (~vm_sorted) & fm_sorted

    # Fit E_QP = α · E_DFT + β directly so α reads as the stretching factor
    # (α = 1 ⇒ rigid shift).  RMSE is reported on the QP correction
    # ΔE = E_QP − E_DFT for human readability ("the residual error in
    # predicting how much GW shifts each band").
    alpha_v, beta_v, _ = _ols_line(E_dft_sorted[mask_v], E_qp_sorted[mask_v])
    alpha_c, beta_c, _ = _ols_line(E_dft_sorted[mask_c], E_qp_sorted[mask_c])
    resid_v = (E_qp_sorted - E_dft_sorted)[mask_v] - (
        (alpha_v - 1.0) * E_dft_sorted[mask_v] + beta_v)
    resid_c = (E_qp_sorted - E_dft_sorted)[mask_c] - (
        (alpha_c - 1.0) * E_dft_sorted[mask_c] + beta_c)
    rmse_v = float(np.sqrt(np.mean(resid_v * resid_v))) if resid_v.size else 0.0
    rmse_c = float(np.sqrt(np.mean(resid_c * resid_c))) if resid_c.size else 0.0

    return ScissorFit(
        alpha_v=alpha_v, beta_v_ev=beta_v,
        alpha_c=alpha_c, beta_c_ev=beta_c,
        n_fit_v=int(mask_v.sum()), n_fit_c=int(mask_c.sum()),
        rmse_v_ev=rmse_v, rmse_c_ev=rmse_c,
    )


# ---------------------------------------------------------------------------
# Extrapolate to full bandrange
# ---------------------------------------------------------------------------

def extrapolate_delta_e(
    E_kn_ev: np.ndarray,
    delta_e_kn_ev: np.ndarray,
    valence_mask_kn: np.ndarray,
    in_grid_mask_kn: np.ndarray,
    fit: ScissorFit,
) -> np.ndarray:
    """Return ΔE over the full bandrange.

    In-grid bands keep their measured ΔE; out-of-grid bands get
    ``fit.predict(E, valence_mask)``.
    """
    dE = np.real(np.asarray(delta_e_kn_ev, dtype=np.complex128))
    ig = np.asarray(in_grid_mask_kn, dtype=bool)
    return np.where(ig, dE, fit.predict(E_kn_ev, valence_mask_kn))


__all__ = [
    "ScissorFit",
    "classify_bands_in_grid",
    "extrapolate_delta_e",
    "fit_scissor",
]
