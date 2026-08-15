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

The fit is a REDUCTION OVER k and therefore needs k WEIGHTS
----------------------------------------------------------
``fit_scissor`` least-squares over every (k, n) sample, so the answer
depends on how many times each k enters.  On the full BZ every k enters
once; on a symmetry-reduced k-set each star is present once but stands
for ``multiplicity`` full-BZ points.  An unweighted fit on a reduced set
therefore fits a DIFFERENT point cloud and returns a plausible number.

Measured cost of getting this wrong: with the SC loop on the IBZ
(``sc_on_ibz``) the MoS₂ 4×4 deck reduces 16 k to 10 stars, 6 of which
have multiplicity 2.  The unweighted fit made the carried H differ from
the full-BZ arm by 1.67e-02 Ry = 0.23 eV after ONE iteration and eqp0 by
0.386 eV max / 0.037 eV rms after three (jobs 7889373, 7889375), while
Σ itself, the star select/broadcast and the ψ rotation were bit-identical
between the arms — see ``docs/dev/ibz_self_consistency_scaffold.md`` §8.

``k_weights`` is consequently a REQUIRED keyword argument of
``fit_scissor``.  There is no unweighted spelling to forget, and the two
legitimate weight tables have named constructors that state the caller's
claim about its own k-set: :func:`full_bz_k_weights` ("every k is its own
star") and :func:`k_star_weights` ("multiplicities from the map that did
the reduction").  Uniform weights reduce to exactly the previous
arithmetic, bit for bit — asserted in ``tests/test_scissor_weights.py``.

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
    w_fit_v, w_fit_c : float
        Total k-weight carried by those points, ``Σ w_k`` over the fit
        mask.  Equal to ``n_fit`` on the full BZ; on a reduced k-set it is
        the full-BZ sample count the fit stands for.  The two are the
        evidence that an IBZ arm and a full-BZ arm fit the same cloud:
        the point counts differ, these do not.
    rmse_v_ev, rmse_c_ev : float
        Fit residual RMSE on the QP correction ΔE = E_QP − E_DFT, in eV,
        k-weighted like the fit.  0 when the fit has fewer than 2 points.
    """

    alpha_v: float
    beta_v_ev: float
    alpha_c: float
    beta_c_ev: float
    n_fit_v: int
    n_fit_c: int
    rmse_v_ev: float
    rmse_c_ev: float
    w_fit_v: float
    w_fit_c: float

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
            f"n={self.n_fit_v}, w={self.w_fit_v:.0f}, "
            f"rmse={self.rmse_v_ev:.3f} eV; "
            f"cond: α={self.alpha_c:+.4f}, β={self.beta_c_ev:+.4f} eV, "
            f"n={self.n_fit_c}, w={self.w_fit_c:.0f}, "
            f"rmse={self.rmse_c_ev:.3f} eV)  "
            f"[α=1 ⇔ rigid shift; n = samples, w = full-BZ weight]"
        )


def apply_conduction_scissor_to_tail(
    E_dft_kn_ev: np.ndarray,
    fit: ScissorFit,
    *,
    tail_start: int,
    logical_stop: int,
) -> np.ndarray:
    """Apply ``fit`` only to a logical conduction-sum tail.

    The active QP matrix ends at ``tail_start`` (local ``b3``) while the
    physical band sum ends at ``logical_stop`` (local ``b4_user``).  Return
    a copy of the full DFT energy ladder with

    ``E_QP = fit.alpha_c * E_DFT + fit.beta_c_ev``

    on exactly ``[:, tail_start:logical_stop]``.  Active energies before
    ``tail_start`` and mesh-padding slots at/after ``logical_stop`` are
    copied bit-for-bit.  This function owns energies only: wavefunctions
    remain DFT outside the active block, and the bundle constructor that
    consumes the returned ladder owns rebuilding occupations.

    Parameters
    ----------
    E_dft_kn_ev : np.ndarray, (nk, nb_full)
        Full DFT energy ladder in eV, including any padded band slots.
    fit : ScissorFit
        Current active-space fit.  Its conduction law is used regardless
        of energy ordering because this interval is the declared
        conduction-sum tail.
    tail_start, logical_stop : int
        Local half-open interval ``[b3, b4_user)``.
    """
    E = np.asarray(E_dft_kn_ev, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError(
            "apply_conduction_scissor_to_tail: E_dft_kn_ev must have shape "
            f"(nk, nb_full); got {E.shape}.")
    lo = int(tail_start)
    hi = int(logical_stop)
    if not (0 <= lo <= hi <= E.shape[1]):
        raise ValueError(
            "apply_conduction_scissor_to_tail: expected "
            f"0 <= tail_start <= logical_stop <= {E.shape[1]}, got "
            f"tail_start={lo}, logical_stop={hi}.")
    if hi > lo and int(fit.n_fit_c) == 0:
        raise ValueError(
            "apply_conduction_scissor_to_tail: the logical conduction tail "
            f"[{lo}, {hi}) is nonempty but ScissorFit has no trusted "
            "conduction samples.")

    out = E.copy()
    out[:, lo:hi] = (
        float(fit.alpha_c) * E[:, lo:hi] + float(fit.beta_c_ev))
    return out


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
# k-weight tables — the two legitimate ones, each named after its claim
# ---------------------------------------------------------------------------

def full_bz_k_weights(nk: int) -> np.ndarray:
    """Weights for an UNREDUCED k-set: every k is its own star, w ≡ 1.

    Use this only where the k axis really is the full grid.  Everything
    ``compute_sigma_xc`` / ``compute_screening`` produce is on the full BZ
    because Σ is an FFT over the k-grid
    (``docs/dev/ibz_self_consistency_scaffold.md`` §7).
    """
    return np.ones(int(nk), dtype=np.float64)


def k_star_weights(kstar) -> np.ndarray:
    """Star multiplicities for a reduced k-set, in the map's own row order.

    ``kstar`` is a ``symmetry_maps.KStarMap`` (duck-typed on
    ``irr_idx`` and ``select``, so this module keeps no symmetry import).
    ``w[j]`` is the number of full-BZ k that ``select`` collapsed into row
    ``j``: ``np.bincount(irr_idx)`` counted per star, spread back over the
    full BZ, then passed through ``kstar.select`` itself.

    Routing the table through ``select`` rather than re-deriving the row
    order is deliberate — ``star_select`` orders rows by first occurrence
    in ``irr_idx`` (``symmetry_maps.py:1770-1772``), which a second
    implementation here could drift from silently, misaligning weights
    with rows.  Using ``select`` makes that impossible by construction.

    On an identity map every star is a singleton, so this returns ones and
    the caller needs no branch: the full-BZ path stays the same code.
    """
    irr = np.asarray(kstar.irr_idx)
    mult_full = np.bincount(irr)[irr].astype(np.float64)
    w = np.asarray(kstar.select(mult_full), dtype=np.float64)
    return w


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def _wls_line(x: np.ndarray, y: np.ndarray,
              w: np.ndarray) -> tuple[float, float, float]:
    """Closed-form WEIGHTED least-squares line fit; (slope, intercept, RMSE).

    Minimises ``Σ w_i (y_i − a x_i − b)²``.  Written so that ``w ≡ 1``
    reproduces the unweighted arithmetic BIT FOR BIT: ``w * v`` is exact
    for ``w = 1.0``, and the means use ``np.sum(w*x)/np.sum(w)``, which is
    the same pairwise ``add.reduce`` over the same values as ``x.mean()``
    divided by the same float.  Measured over 3000 random (n ≤ 4000)
    problems: 0 mismatches, exact equality of all three returns.  Using
    ``np.dot(w, x)`` for the means instead would go through BLAS ddot and
    is NOT guaranteed to match ``x.mean()``.
    """
    n = int(x.size)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return 0.0, float(y[0]), 0.0
    W = float(np.sum(w))
    xm = float(np.sum(w * x) / W)
    ym = float(np.sum(w * y) / W)
    dx = x - xm
    wdx = w * dx
    denom = float(np.dot(wdx, dx))
    if denom < 1.0e-30:
        return 0.0, ym, float(np.sqrt(np.sum(w * (y - ym) ** 2) / W))
    slope = float(np.dot(wdx, y - ym) / denom)
    intercept = float(ym - slope * xm)
    resid = y - (slope * x + intercept)
    return slope, intercept, float(np.sqrt(np.sum(w * resid * resid) / W))


def fit_scissor(
    E_dft_kn_ev: np.ndarray,
    E_qp_kn_ev: np.ndarray,
    valence_mask_kn: np.ndarray,
    fit_mask_kn: np.ndarray,
    *,
    k_weights: np.ndarray,
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
    k_weights : np.ndarray, (nk,) — REQUIRED, keyword-only
        How many full-BZ k each row stands for.  Build it with
        :func:`full_bz_k_weights` or :func:`k_star_weights`; do not spell
        it inline.  Required rather than defaulted because this fit is a
        reduction over k and an unweighted fit on a reduced k-set returns
        a plausible wrong scissor that no downstream check catches — see
        the module docstring for the measured size (0.386 eV in eqp0).
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
    w_k = np.asarray(k_weights, dtype=np.float64)
    if w_k.shape != (nk,):
        raise ValueError(
            f"fit_scissor: k_weights has shape {w_k.shape}, expected "
            f"({nk},) to match the k axis of E_DFT.  A weight table from a "
            f"different k-set than the energies is the exact failure this "
            f"argument exists to prevent; build it from the map that "
            f"reduced these energies (scissor.k_star_weights) or state "
            f"that they are unreduced (scissor.full_bz_k_weights).")
    if not np.all(np.isfinite(w_k)) or float(w_k.min()) <= 0.0:
        raise ValueError(
            f"fit_scissor: k_weights must be finite and strictly positive; "
            f"got min {float(np.nanmin(w_k)):.6g}, max "
            f"{float(np.nanmax(w_k)):.6g}.  A zero weight drops a star from "
            f"the fit silently.")

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

    # The k-weight rides along per (k, n) sample: every band at k carries
    # the same star multiplicity, so broadcast down the band axis.  The
    # sort permutation is per-k (rows are not reordered), hence no
    # reordering of w_kn is needed.
    w_kn = np.broadcast_to(w_k[:, None], E_dft.shape)

    # Fit E_QP = α · E_DFT + β directly so α reads as the stretching factor
    # (α = 1 ⇒ rigid shift).  RMSE is reported on the QP correction
    # ΔE = E_QP − E_DFT for human readability ("the residual error in
    # predicting how much GW shifts each band").
    w_v = w_kn[mask_v]
    w_c = w_kn[mask_c]
    alpha_v, beta_v, _ = _wls_line(
        E_dft_sorted[mask_v], E_qp_sorted[mask_v], w_v)
    alpha_c, beta_c, _ = _wls_line(
        E_dft_sorted[mask_c], E_qp_sorted[mask_c], w_c)
    # No-information laws.  _wls_line returns (0, 0) on an empty class and
    # (0, y0) on a single sample; as an E_QP = α·E + β scissor those
    # extrapolate every band to ZERO / to a constant — on the metallic
    # sodium deck the Fermi-crossing pair gave the classes no clean
    # samples, every scissored diagonal became exactly 0.0, and eigvalsh's
    # ascending sort interleaved 46 zeros with the two real eigenvalues
    # (the migrating-band snapshots, max|dE| = VBM to six decimals).
    # Zero samples ⇒ identity (keep E_DFT); one sample ⇒ rigid shift.
    for _cls in ("v", "c"):
        _m, _w = (mask_v, w_v) if _cls == "v" else (mask_c, w_c)
        _n = int(_m.sum())
        if _n == 0:
            _a, _b = 1.0, 0.0
        elif _n == 1:
            _a = 1.0
            _b = float((E_qp_sorted - E_dft_sorted)[_m][0])
        else:
            continue
        if _cls == "v":
            alpha_v, beta_v = _a, _b
        else:
            alpha_c, beta_c = _a, _b
    resid_v = (E_qp_sorted - E_dft_sorted)[mask_v] - (
        (alpha_v - 1.0) * E_dft_sorted[mask_v] + beta_v)
    resid_c = (E_qp_sorted - E_dft_sorted)[mask_c] - (
        (alpha_c - 1.0) * E_dft_sorted[mask_c] + beta_c)
    rmse_v = (float(np.sqrt(np.sum(w_v * resid_v * resid_v) / np.sum(w_v)))
              if resid_v.size else 0.0)
    rmse_c = (float(np.sqrt(np.sum(w_c * resid_c * resid_c) / np.sum(w_c)))
              if resid_c.size else 0.0)

    return ScissorFit(
        alpha_v=alpha_v, beta_v_ev=beta_v,
        alpha_c=alpha_c, beta_c_ev=beta_c,
        n_fit_v=int(mask_v.sum()), n_fit_c=int(mask_c.sum()),
        rmse_v_ev=rmse_v, rmse_c_ev=rmse_c,
        w_fit_v=float(w_v.sum()), w_fit_c=float(w_c.sum()),
    )


__all__ = [
    "ScissorFit",
    "apply_conduction_scissor_to_tail",
    "classify_bands_in_grid",
    "fit_scissor",
    "full_bz_k_weights",
    "k_star_weights",
]
