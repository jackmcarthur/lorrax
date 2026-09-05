"""Scissor-shift extrapolation for self-consistent GW.

The dynamic self-energy Σ_c(ω) in LORRAX is computed on a small frequency grid
(typically ±10 eV around E_F) while the full band range used in the QP
Hamiltonian spans tens of eV below E_F to hundreds or thousands of eV above.
Inside the grid we evaluate Σ_xc by interpolating the stored (ω, k, m, n)
tensor; outside the grid we extrapolate the QP correction

    ΔE_nk := E_QP_nk − E_DFT_nk  =  ⟨n| Σ_xc − V_xc^DFT |n⟩_k

by class-dependent laws.  The conduction class keeps the affine law refit at
each SC iteration,

    E_QP,c(E_DFT) = α_c · E_DFT + β_c,

while an out-of-range valence state is kept at its DFT position relative to
the Fermi level,

    E_QP,v − E_F,QP = E_DFT,v − E_F,DFT.

Equivalently every such valence state receives the same rigid displacement
``E_F,QP - E_F,DFT``.  :func:`fit_scissor` still reports a valence affine fit
because it is a general diagnostic and has other consumers, but
:func:`qsgw_out_of_range_energies` deliberately does not use that fit for the
self-consistent active-window valence tail.

Which bands are valence and which are conduction
------------------------------------------------
On an insulator this is the step occupation and there is nothing to decide.
On a METAL it is a decision, and the wrong one is expensive.  The rule this
module implements (owner ruling, 2026-08-16) is:

    valence class   = bands BELOW the LOWEST band that crosses E_F
    conduction class = bands ABOVE the HIGHEST band that crosses E_F
    Fermi-crossing bands belong to NEITHER fit class

A crossing band is not valence data and it is not conduction data: over its
own k-set it is BOTH, so a line fit through it is a line fit through the
Fermi surface.  Those bands are the protected/full-Σ set anyway — nothing
downstream needs a scissor law for them — so excluding them from the fit
costs nothing and removes the only samples that cannot be labelled.

Measured cost of getting this wrong (claim 0212, sodium
``runs/Na/02_soc48b_qsgw_mpa``, 48 bands × 512 full-BZ k, the semicore
window family).  Under the old "occupied at DFT occupation" index mask the
Fermi-crossing Kramers pair (bands 9-10) sat in the VALENCE class:

* ``[-5,+5]`` window: the val fit's only in-window samples WERE the
  crossing pair — n_v = 1024 samples, 100% crossing — giving α = 0.9100,
  β = −0.0015 eV.  Extrapolated to the 2s semicore at E − μ = −53.8 eV that
  law predicts **+4.84 eV** where BerkeleyGW's Eqp0 is **−12.86 eV**:
  wrong by 17.5 eV and wrong in SIGN.  Under the rule above the valence
  class is EMPTY in that window and the law is the identity — which is a
  refusal to extrapolate, and it is more honest than a Fermi-surface-
  anchored line.
* ``[-28,+26]`` window: the val fit mixed the 2p semicore with the same
  crossing pair (8 bands, n_v = 4096) → α = 1.1978.  Under the rule the
  class is the 2p alone.

:func:`classify_scissor_bands` derives the three classes from an
``OccupationState``'s ``f_kn``; the boundary indices it returns are what
both the fit masks and :func:`qsgw_out_of_range_energies` use, so a band is
classified once per iteration and the fit and application cannot disagree.

The scissor is applied at the **eigenvalue level** in the diagonal Σ(E)
fixed-point: out-of-grid conduction bands receive the affine law and
out-of-grid valence bands receive the Fermi displacement above.  The QSGW
Σ_xc that enters the QP Hamiltonian then evaluates the dynamic correlation
at this scissor-corrected E_QP.  No matrix-level diagonal-add is exposed
because the post-self-energy plumbing keeps H replicated, so a plain
``H.at[:, idx, idx].add(diag)`` suffices when the caller wants one.

Per-band classification
-----------------------
A band ``n`` is **in-grid** iff ``E_DFT[k, n]`` lies inside ``[ω_min, ω_max]``
for **every** k.  If any single k for that band lies outside the window the
whole band is treated as out-of-grid — the diagonal Σ(E) fixed-point clipped
Σ_c at the ω-boundary for the offending k, which contaminates the QP
correction at neighbouring k via the band's k-dispersion.  Per-band
classification gives a discontinuity-free scissor: out-of-grid bands receive
their class law uniformly across k (rigid Fermi displacement for valence,
affine fit for conduction), and the fit itself is restricted to data from
in-grid bands so the contaminated points never enter.

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

    def predict(self, E_ev: np.ndarray, valence_mask: np.ndarray,
                *, crossing_mask: np.ndarray | None = None) -> np.ndarray:
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
        crossing_mask : np.ndarray, (nk, nb) of bool, optional
            Bands in NEITHER fit class (the Fermi-crossing set of
            :class:`ScissorBandClasses`).  ΔE is 0 there — the identity, the
            same no-information law an empty class gets: a band we refused
            to FIT is a band we refuse to EXTRAPOLATE.  In practice these
            bands are inside the Σ(ω) window and protected, so the caller's
            ``in_range_mask`` discards this entry anyway; the case where it
            does not is a band that crosses E_F at one k and leaves the
            window at another, and there ``E_QP = E_DFT`` is the honest
            answer.  ``None`` (the default) is the historical two-way
            behaviour, bit for bit.
        """
        E = np.asarray(E_ev, dtype=np.float64)
        vm = np.asarray(valence_mask, dtype=bool)
        delta_v = (self.alpha_v - 1.0) * E + self.beta_v_ev
        delta_c = (self.alpha_c - 1.0) * E + self.beta_c_ev
        out = np.where(vm, delta_v, delta_c)
        if crossing_mask is not None:
            out = np.where(np.asarray(crossing_mask, dtype=bool), 0.0, out)
        return out

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


def qsgw_out_of_range_energies(
    E_dft_kn_ev: np.ndarray,
    fit: ScissorFit,
    valence_mask_kn: np.ndarray,
    *,
    fermi_displacement_ev: float,
    crossing_mask_kn: np.ndarray | None = None,
) -> np.ndarray:
    """Build active-window QSGW candidates for out-of-range diagonals.

    This is the single application policy used by the self-consistent map:

    * valence: ``E_QP = E_DFT + (E_F,QP - E_F,DFT)``;
    * conduction: ``E_QP = fit.alpha_c * E_DFT + fit.beta_c_ev``;
    * Fermi-crossing: ``E_QP = E_DFT`` (neither scissor class).

    The valence fields in ``fit`` are intentionally not read.  They remain
    useful fit diagnostics, but extrapolating a regression obtained from
    higher-lying valence states into a deep out-of-range shell changes that
    shell's binding relative to ``E_F``.  The rigid Fermi displacement keeps
    ``E_band - E_F`` exactly invariant instead.

    This function returns candidates for every entry.  The band-partition
    primitive remains the sole owner of deciding which candidates are used:
    protected/in-range diagonals retain the full QSGW Hamiltonian.
    """
    E = np.asarray(E_dft_kn_ev, dtype=np.float64)
    valence = np.asarray(valence_mask_kn, dtype=bool)
    if E.shape != valence.shape:
        raise ValueError(
            "qsgw_out_of_range_energies: E_DFT and valence mask must have "
            f"the same shape; got {E.shape} and {valence.shape}.")
    shift = float(fermi_displacement_ev)
    if not np.isfinite(shift):
        raise ValueError(
            "qsgw_out_of_range_energies: fermi_displacement_ev must be "
            f"finite; got {shift!r}.")

    valence_qp = E + shift
    conduction_qp = float(fit.alpha_c) * E + float(fit.beta_c_ev)
    out = np.where(valence, valence_qp, conduction_qp)
    if crossing_mask_kn is not None:
        crossing = np.asarray(crossing_mask_kn, dtype=bool)
        if crossing.shape != E.shape:
            raise ValueError(
                "qsgw_out_of_range_energies: E_DFT and crossing mask must "
                f"have the same shape; got {E.shape} and {crossing.shape}.")
        out = np.where(crossing, E, out)
    return out


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
        conduction-sum tail.  An empty conduction class carries the identity
        law from :func:`fit_scissor`, so the tail stays at DFT energy rather
        than being extrapolated from untrusted grid-edge values.
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
    out = E.copy()
    out[:, lo:hi] = (
        float(fit.alpha_c) * E[:, lo:hi] + float(fit.beta_c_ev))
    return out


# ---------------------------------------------------------------------------
# Per-band in-grid classification
# ---------------------------------------------------------------------------

_SC_PAD_BASE_EV = 0.5
_SC_PAD_FRACTION = 0.10


def sc_state_pad_ev(energy_relative_to_mu_ev):
    """Energy drift allowance shared by SC classification and quadrature.

    Parameters
    ----------
    energy_relative_to_mu_ev : array_like
        Host state energies E - mu in eV, of any shape.

    Returns
    -------
    ndarray
        Pad in eV with the input shape: 0.5 + 0.10 * abs(E - mu).
        The constant covers near-Fermi trial motion; the proportional term
        covers the roughly ten-percent spectral stretch measured on Na.
    """
    return _SC_PAD_BASE_EV + _SC_PAD_FRACTION * np.abs(
        np.asarray(energy_relative_to_mu_ev, dtype=np.float64))


def sc_padded_window_ev(lower_ev, upper_ev):
    """Outer energies satisfying E >= lo-pad(E), E <= hi+pad(E).

    Bounds and returned scalars are relative to mu, in eV. Solving these
    inequalities covers the entire hysteresis region, including the extra
    allowance gained as a state moves away from mu.
    """
    lower = float(lower_ev) - _SC_PAD_BASE_EV
    upper = float(upper_ev) + _SC_PAD_BASE_EV
    return (lower / (1.0 + _SC_PAD_FRACTION * np.sign(lower)),
            upper / (1.0 - _SC_PAD_FRACTION * np.sign(upper)))


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
# Three-way val / crossing / cond classification from the occupation table
# ---------------------------------------------------------------------------

#: A cell ``f`` counts as FULL at ``f >= 1 - FRACTIONAL_TOL`` and EMPTY at
#: ``f <= FRACTIONAL_TOL``; anything strictly between is FRACTIONAL.  The
#: tolerance is NOT cosmetic and it is not an "≈ 1" convenience:
#:
#: * MP1 (Methfessel-Paxton order 1) OVERSHOOTS.  ``f_kn`` is never clipped
#:   (``efermi.OccupationState`` says so in its own docstring), so a band a
#:   width or two below μ carries ``f ≈ 1.008`` and one above carries
#:   ``f ≈ −0.008``.  A test written as ``0 < f < 1`` would call the
#:   overshoot cells "not fractional" for the wrong reason and, worse, a
#:   test written as ``f == 1.0`` would call every deep band fractional the
#:   moment the erfc tail stops rounding to exactly 1.0.
#: * The classification must therefore be "which SIDE is this cell on, and
#:   is it saturated there", which is what the two-sided tolerance says.
#:
#: 1e-8 is far outside float64 noise on a saturated erfc tail (which
#: reaches exact 1.0/0.0 by |E − μ| ≈ 6 widths) and far inside any genuine
#: partial occupation (the smallest one the Fermi surface produces is
#: O(width) in energy, i.e. O(0.1) in f).
FRACTIONAL_TOL = 1.0e-8


@dataclass(frozen=True)
class ScissorBandClasses:
    """Which bands are valence, which cross E_F, and which are conduction.

    Held as two BOUNDARY INDICES rather than a per-band label array, because
    the owner's rule is stated in terms of boundaries and because the two
    consumers need the classes at different band widths — the fit sees the
    active window ``nb_active``, the occupation table may be the padded
    parallel-transport manifold ``nb_storage``, and the sum-band tail fit
    sees the full ladder.  Index arithmetic is width-agnostic; a label array
    would have to be padded, and padding a label array is how the wrong
    band gets the wrong class.

    Attributes
    ----------
    valence_stop : int
        Bands ``[0, valence_stop)`` are the valence fit class — everything
        strictly below the lowest Fermi-crossing band.
    conduction_start : int
        Bands ``[conduction_start, nb)`` are the conduction fit class —
        everything strictly above the highest Fermi-crossing band.

    The gap ``[valence_stop, conduction_start)`` is the crossing set, and it
    is exactly the set of bands carrying a fractional cell — the crossing
    bands cannot be non-contiguous on an energy-sorted band axis, and
    :func:`classify_scissor_bands` refuses a table where they are.  So
    ``n_crossing`` is ``conduction_start - valence_stop`` and is not stored
    twice.
    """

    valence_stop: int
    conduction_start: int

    def __post_init__(self):
        if not (0 <= int(self.valence_stop) <= int(self.conduction_start)):
            raise ValueError(
                "ScissorBandClasses: need 0 <= valence_stop <= "
                f"conduction_start; got {self.valence_stop!r}, "
                f"{self.conduction_start!r}")

    @property
    def n_crossing(self) -> int:
        """How many bands are in neither fit class."""
        return int(self.conduction_start) - int(self.valence_stop)

    def masks(self, shape) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(valence_kn, crossing_kn)`` broadcast to ``(nk, nb)``.

        ``valence_kn`` is what ``fit_scissor`` wants as ``valence_mask_kn``
        and ``ScissorFit.predict`` as ``valence_mask``; ``crossing_kn`` is
        what the caller ANDs OUT of its ``fit_mask_kn`` and passes to
        ``predict`` as ``crossing_mask``.  Everything that is neither is
        conduction, so two masks describe three classes.
        """
        nk, nb = int(shape[0]), int(shape[1])
        idx = np.arange(nb)
        val = idx < int(self.valence_stop)
        cross = (~val) & (idx < int(self.conduction_start))
        return (np.broadcast_to(val[None, :], (nk, nb)),
                np.broadcast_to(cross[None, :], (nk, nb)))

    def summary(self) -> str:
        """One line, 1-BASED band labels — the convention the logs use."""
        v = ("none" if self.valence_stop == 0
             else f"1-{self.valence_stop}")
        gap = self.conduction_start - self.valence_stop
        x = ("none" if gap == 0
             else f"{self.valence_stop + 1}-{self.conduction_start}")
        return (f"val = bands {v}, Fermi-crossing = bands {x} "
                f"(n={self.n_crossing}), "
                f"cond = bands {self.conduction_start + 1}+ "
                f"[crossing excluded from both fits]")


def classify_scissor_bands(
    f_kn,
    *,
    frac_tol: float = FRACTIONAL_TOL,
) -> ScissorBandClasses:
    """Derive the valence / crossing / conduction boundaries from ``f_kn``.

    Per BAND, over all k:

    * every cell FULL (``f >= 1 - frac_tol``)   → the band is fully occupied
    * every cell EMPTY (``f <= frac_tol``)      → the band is fully empty
    * anything else — a fractional cell anywhere, or full at one k and empty
      at another — → the band CROSSES E_F.

    Then, per the owner's rule, ``valence_stop`` is the lowest crossing
    band's index and ``conduction_start`` is the highest crossing band's
    index + 1.  With no crossing band at all (an insulator's step table, or
    a "metal" whose k-grid happens to gap) the crossing set is empty and the
    boundaries collapse onto the occupied-band count — which is exactly the
    step mask, so the insulating path is unchanged by construction.

    Refuses if a fully-occupied band lands at or above ``valence_stop``, or
    a fully-empty band below ``conduction_start`` — equivalently, if the
    crossing bands are not contiguous.  With bands energy-sorted within each
    k (every producer here is ``eigvalsh`` or QE, both ascending) that is
    impossible: if band n is full at every k and band m > n crosses, then at
    the k where m is fractional ``E[k,m] >= E[k,n]``, so m cannot be
    saturated-full there and n cannot be above m.  The same argument in
    reverse forbids an empty band below a crossing one.  Tripping the
    refusal therefore means the band axis is not energy-sorted, and a
    boundary-index rule is the wrong rule for that table — better to say so
    than to silently misclassify a shell.

    Note this is a statement about SATURATION, not about the raw value of
    ``f``: MP1 is non-monotonic in E (that is the overshoot), but "which
    side of μ, and saturated there" is monotonic, which is what makes the
    boundary indices well defined at all.

    Parameters
    ----------
    f_kn : array_like, (nk, nb)
        Occupations, NOT clipped — an ``OccupationState.f_kn``.
    frac_tol : float
        Saturation tolerance; see :data:`FRACTIONAL_TOL`.
    """
    f = np.asarray(f_kn, dtype=np.float64)
    if f.ndim != 2 or min(f.shape) < 1:
        raise ValueError(
            f"classify_scissor_bands: f_kn must be a nonempty (nk, nb); "
            f"got shape {f.shape}")
    tol = float(frac_tol)
    if not (0.0 <= tol < 0.5):
        raise ValueError(
            f"classify_scissor_bands: frac_tol must be in [0, 0.5); got {tol!r}")

    nb = int(f.shape[1])
    band_full = np.all(f >= 1.0 - tol, axis=0)
    band_empty = np.all(f <= tol, axis=0)
    band_crossing = ~(band_full | band_empty)

    idx = np.arange(nb)
    if bool(band_crossing.any()):
        v_stop = int(idx[band_crossing].min())
        c_start = int(idx[band_crossing].max()) + 1
    else:
        # No crossing band: the boundaries collapse and the classes are
        # exactly "full" and "empty".  Contiguity is checked below like any
        # other case, so a non-prefix full set refuses here too.
        v_stop = c_start = int(band_full.sum())

    bad_full = np.nonzero(band_full & (idx >= v_stop))[0]
    bad_empty = np.nonzero(band_empty & (idx < c_start))[0]
    if bad_full.size or bad_empty.size:
        raise ValueError(
            "classify_scissor_bands: the occupation table is not consistent "
            "with an energy-sorted band axis, so the below-lowest-crossing / "
            "above-highest-crossing rule cannot label it.  Crossing bands "
            f"(0-based) {np.nonzero(band_crossing)[0].tolist()} put the "
            f"boundaries at [0,{v_stop}) / [{c_start},{nb}), but fully "
            f"occupied bands {bad_full.tolist()} are not below the first and "
            f"fully empty bands {bad_empty.tolist()} are not above the last.")

    return ScissorBandClasses(valence_stop=v_stop, conduction_start=c_start)


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
    conduction_frontier_tol_ev: float | None = None,
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
        True where the valence law applies, False where the conduction law
        does.  On a metal build it from :func:`classify_scissor_bands` —
        "below the lowest Fermi-crossing band" — NOT from an occupied-band
        index cut, which puts crossing bands in the valence class.
    fit_mask_kn : np.ndarray, (nk, nb) of bool
        True where the point is trusted enough to enter the fit —
        "E_kn lies inside the Σ(ω) grid" AND NOT a Fermi-crossing band.
        Applied independently within valence and conduction, so this is the
        mask that makes the crossing set enter neither.  It is a separate
        argument from ``valence_mask_kn`` precisely because a two-valued
        mask cannot express a three-way classification.
    k_weights : np.ndarray, (nk,) — REQUIRED, keyword-only
        How many full-BZ k each row stands for.  Build it with
        :func:`full_bz_k_weights` or :func:`k_star_weights`; do not spell
        it inline.  Required rather than defaulted because this fit is a
        reduction over k and an unweighted fit on a reduced k-set returns
        a plausible wrong scissor that no downstream check catches — see
        the module docstring for the measured size (0.386 eV in eqp0).
    conduction_frontier_tol_ev : float, optional
        Replace the conduction affine regression by a rigid shift fitted to
        the lowest conduction multiplet. Starting at the first clean
        conduction band, adjacent bands join that multiplet while their
        minimum separation over k is at most this tolerance. This is a
        manifold-identification tolerance, not an SC convergence tolerance.
        It makes the energy-only sum-band tail independent of how many higher
        conduction bands happen to lie inside the active QP matrix. ``None``
        preserves the general all-conduction affine fit.
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

    if conduction_frontier_tol_ev is not None and np.any(mask_c):
        tol_ev = float(conduction_frontier_tol_ev)
        if not np.isfinite(tol_ev) or tol_ev < 0.0:
            raise ValueError(
                "fit_scissor: conduction_frontier_tol_ev must be finite and "
                f"non-negative; got {conduction_frontier_tol_ev!r}.")

        # Work in sorted-energy position space, matching the fit itself. A
        # frontier position must be clean conduction data at every k; a
        # Fermi-crossing or partially in-grid position is not evidence for a
        # tail law. Once the first such position is found, extend only across
        # multiplet boundaries. The minimum-over-k rule is the same rule used
        # by common.band_degeneracy: if two bands touch anywhere, cutting
        # between them does not define a global band subspace.
        clean_c_pos = np.all((~vm_sorted) & fm_sorted, axis=0)
        clean_positions = np.flatnonzero(clean_c_pos)
        if clean_positions.size:
            first = int(clean_positions[0])
            stop = first + 1
            while stop < E_dft_sorted.shape[1] and clean_c_pos[stop]:
                min_gap_ev = float(np.min(np.abs(
                    E_dft_sorted[:, stop] - E_dft_sorted[:, stop - 1])))
                if min_gap_ev > tol_ev:
                    break
                stop += 1
            frontier_pos = np.arange(first, stop, dtype=np.int64)
            frontier_mask = np.zeros_like(mask_c)
            frontier_mask[:, frontier_pos] = True
            frontier_mask &= mask_c
            w_frontier = w_kn[frontier_mask]
            correction = (E_qp_sorted - E_dft_sorted)[frontier_mask]
            alpha_c = 1.0
            beta_c = float(
                np.sum(w_frontier * correction) / np.sum(w_frontier))
            mask_c = frontier_mask
            w_c = w_frontier
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
    "FRACTIONAL_TOL",
    "ScissorBandClasses",
    "ScissorFit",
    "apply_conduction_scissor_to_tail",
    "classify_bands_in_grid",
    "classify_scissor_bands",
    "fit_scissor",
    "full_bz_k_weights",
    "k_star_weights",
    "qsgw_out_of_range_energies",
]
