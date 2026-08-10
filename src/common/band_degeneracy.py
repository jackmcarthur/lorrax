"""Degeneracy-safe BSE band-window selection.

THE PROBLEM.  ``--n-val N`` / ``--n-cond M`` pick a band window by COUNTING:
the valence window is ``[n_occ - N, n_occ)`` and the conduction window is
``[n_occ, n_occ + M)``.  Counting knows nothing about the spectrum, so a
window boundary can land *inside* a degenerate multiplet — a Kramers pair
under SOC+TRS, a p-like triplet at Γ, any point-group irrep with dimension
> 1.  Half a multiplet is not a subspace of anything: the BSE built on it is
not the BSE of a symmetry-adapted problem, its exciton multiplets are broken
by an amount that has no convergence parameter, and nothing downstream
reports it.

WHAT IT COSTS, MEASURED.  On the Si 12×12 SOC deck every Kramers pair is
exactly degenerate, and an fH window boundary that cut a pair failed at the
**eV scale** off-grid, producing a spurious "Λ-valley dip" in E₁(Q) that the
delivered exciton band structure faithfully tracked.  Between-pair boundary
min-gaps on that deck ranged from 2194 meV down to **5.9 meV** at the 31|32
boundary (``docs/dev/archive/frontera_campaign/wk_REL/reference/perlmutter/
EXCITON_AND_PERF_SALVAGE.md``).  5.9 meV is the scale this guard has to
resolve, which is why the default tolerance is 1 meV and not BGW's 1e-6 Ry
(14 µeV): 14 µeV separates *exactly* degenerate states, and the question here
is not "are these the same eigenvalue" but "is this boundary safe to cut".

WHAT THIS MODULE DOES.  :func:`resolve_band_window` takes the energies and
the requested counts and returns counts whose boundaries are clean, by
widening OUTWARD (never inward — narrowing would silently drop physics the
user asked for).  Three modes:

``"snap"`` (default)
    Widen to the multiplet boundary and say so, loudly, naming the new
    counts, the offending multiplet and the k at which it is tightest.
``"strict"``
    Raise instead of widening.  For runs where the window is a pinned
    quantity and a silent change of problem size is worse than a stop.
``"off"``
    No check.  Kept because a deliberately half-multiplet window is a
    legitimate *debugging* configuration, and because a guard with no
    off switch gets deleted rather than configured.

WHY OUTWARD.  The multiplet is a subspace; the window must contain all of it
or none of it.  Given a requested count that splits one, the two repairs are
"add the missing partners" and "drop the ones you have".  Adding is the one
that cannot lose a state the user asked for, and it is what the existing
prose in ``exciton_bands`` already tells users to do by hand ("widen the
input's ncond/nband").

WHY A BOUNDARY, NOT A GROUPING.  Degenerate *sets* are formed by BGW's rule
(``Common/nrtype.f90 :: TOL_Degeneracy``): walk the ascending spectrum and
break a group wherever successive eigenvalues differ by more than the
tolerance.  Under that rule the question "does the boundary between band
``b-1`` and band ``b`` split a group?" is answered by ONE number,
``|e[k,b] - e[k,b-1]|``, and a boundary is safe only if it is safe at EVERY
k.  So the whole check is a min over k of a first difference — no grouping,
no transitive closure, no O(nb²) anything.

RELATION TO ``gw.degen_average``.  That module averages Σ *within* degenerate
sets, at BGW's exact-degeneracy tolerance, on the GW side.  Same physical
fact, different question and a different tolerance: it asks "which states must
share a value", this asks "where may I cut".  They are deliberately not
merged; a shared tolerance would be wrong for one of them.
"""
from __future__ import annotations

import numpy as np

#: Ry per eV — the one conversion this module needs.
_EV_PER_RY = 13.6056980659

#: Default "same multiplet" tolerance: **1 meV**, expressed in Ry.
#:
#: Chosen against the 5.9 meV smallest between-pair boundary gap measured on
#: the Si 12×12 SOC deck (module docstring): the tolerance must be well below
#: the smallest gap you are willing to CUT, and well above the numerical
#: noise on eigenvalues that are degenerate by symmetry (~1e-9 Ry on the
#: htransform path).  1 meV sits two orders below the former and six above
#: the latter.  It is NOT BGW's TOL_Degeneracy (1e-6 Ry ≈ 14 µeV), which
#: answers a different question — see the module docstring.
DEGENERACY_TOL_RY: float = 1.0e-3 / _EV_PER_RY      # ≈ 7.3499e-05 Ry

#: The three modes, in the order they appear in the docstring.
MODES = ("snap", "strict", "off")


class BandWindowDegeneracyError(RuntimeError):
    """A window boundary splits a degenerate multiplet, in ``strict`` mode."""


def boundary_min_gaps(enk_ry: np.ndarray) -> np.ndarray:
    """Smallest over k of the gap across each inter-band boundary.

    Parameters
    ----------
    enk_ry : (nk, nb) float array
        Band energies in **Rydberg**, ascending in the band axis at each k.

    Returns
    -------
    (nb + 1,) float array
        Element ``b`` is ``min_k |e[k, b] - e[k, b-1]|`` — the tightest gap
        anywhere in the BZ across the boundary that would separate bands
        ``< b`` from bands ``>= b``.  The two outer boundaries ``b = 0`` and
        ``b = nb`` cut nothing, so they are ``+inf``.
    """
    e = np.asarray(enk_ry, dtype=np.float64)
    if e.ndim != 2:
        raise ValueError(
            f"boundary_min_gaps: expected (nk, nb) energies, got shape {e.shape}")
    nb = e.shape[1]
    gaps = np.full(nb + 1, np.inf, dtype=np.float64)
    if nb >= 2:
        gaps[1:nb] = np.min(np.abs(np.diff(e, axis=1)), axis=0)
    return gaps


def _tightest_k(enk_ry: np.ndarray, b: int) -> tuple[int, float]:
    """``(k, gap)`` at which boundary ``b`` is tightest."""
    e = np.asarray(enk_ry, dtype=np.float64)
    d = np.abs(e[:, b] - e[:, b - 1])
    k = int(np.argmin(d))
    return k, float(d[k])


def resolve_band_window(
    enk_ry: np.ndarray,
    n_occ: int,
    n_val: int,
    n_cond: int,
    *,
    tol_ry: float = DEGENERACY_TOL_RY,
    mode: str = "snap",
    where: str = "band window",
    log=print,
) -> tuple[int, int]:
    """Return ``(n_val, n_cond)`` whose window boundaries split no multiplet.

    Parameters
    ----------
    enk_ry : (nk, nb) float array
        Band energies in **Rydberg**.  Pass the SAME array the window will be
        sliced out of — after any eqp correction, since a QP shift can open or
        close a near-degeneracy that the DFT spectrum did not have.
    n_occ : int
        Index of the first conduction band (the valence/conduction split).
    n_val, n_cond : int
        Requested counts, already clamped to what is available.
    tol_ry : float
        Two bands are "the same multiplet" if they are within this in Ry.
    mode : {"snap", "strict", "off"}
        See the module docstring.
    where : str
        Caller name, quoted in every message so a warning is locatable.
    log : callable
        Where the warning goes.  Defaults to ``print``.

    Returns
    -------
    (n_val, n_cond)
        Unchanged in ``off`` mode, and in the (overwhelmingly common) case
        where the requested boundaries were already clean.

    Raises
    ------
    BandWindowDegeneracyError
        In ``strict`` mode, if either boundary splits a multiplet.
    ValueError
        If ``mode`` is not one of :data:`MODES`.
    """
    if mode not in MODES:
        raise ValueError(
            f"resolve_band_window: mode={mode!r} is not one of {MODES}")
    if mode == "off":
        return int(n_val), int(n_cond)

    e = np.asarray(enk_ry, dtype=np.float64)
    nb = e.shape[1]
    n_occ, n_val, n_cond = int(n_occ), int(n_val), int(n_cond)
    gaps = boundary_min_gaps(e)
    tol = float(tol_ry)

    lo, hi = n_occ - n_val, n_occ + n_cond
    findings: list[str] = []

    # The valence/conduction split itself.  It cannot be repaired by widening
    # — it IS the window's midline — so it is reported and never snapped.  A
    # degeneracy here means the deck has no gap at that k (a metal, or a wrong
    # n_occ), which is a different problem with a different fix.
    if 0 < n_occ < nb and gaps[n_occ] <= tol:
        k, g = _tightest_k(e, n_occ)
        findings.append(
            f"the valence/conduction split at band {n_occ} is itself "
            f"degenerate (gap {g * 1e3 * _EV_PER_RY:.3f} meV at k={k}) — the "
            f"deck is gapless there, or n_occ={n_occ} is wrong.  NOT snapped: "
            f"widening cannot fix the midline of the window.")

    # Valence bottom boundary: widen DOWNWARD (more valence bands).
    lo_new = lo
    while lo_new > 0 and gaps[lo_new] <= tol:
        lo_new -= 1
    # Conduction top boundary: widen UPWARD (more conduction bands).
    hi_new = hi
    while hi_new < nb and gaps[hi_new] <= tol:
        hi_new += 1

    if lo_new != lo:
        k, g = _tightest_k(e, lo)
        findings.append(
            f"the valence boundary at band {lo} cuts a multiplet "
            f"(gap {g * 1e3 * _EV_PER_RY:.3f} meV at k={k}, tol "
            f"{tol * 1e3 * _EV_PER_RY:.3f} meV); the multiplet starts at band "
            f"{lo_new}, so n_val {n_val} -> {n_occ - lo_new}")
    if hi_new != hi:
        k, g = _tightest_k(e, hi)
        findings.append(
            f"the conduction boundary at band {hi} cuts a multiplet "
            f"(gap {g * 1e3 * _EV_PER_RY:.3f} meV at k={k}, tol "
            f"{tol * 1e3 * _EV_PER_RY:.3f} meV); the multiplet ends at band "
            f"{hi_new}, so n_cond {n_cond} -> {hi_new - n_occ}")

    # Ran out of bands with the multiplet still open.  Widening cannot
    # complete it; say so at the same volume rather than returning a window
    # that is still cut but no longer flagged.  (The valence side cannot hit
    # this: boundary 0 is +inf, so the downward walk always terminates on a
    # clean boundary.)
    if hi_new == nb and hi_new != hi and nb >= 2 and gaps[nb - 1] <= tol:
        findings.append(
            f"the conduction multiplet is still open at the TOP of the "
            f"available {nb} bands — the input does not contain the whole "
            f"multiplet.  Widen the input's nband; this window is still cut.")

    if not findings:
        return n_val, n_cond

    n_val_new, n_cond_new = n_occ - lo_new, hi_new - n_occ
    head = (f"[band-window] {where}: requested n_val={n_val} n_cond={n_cond} "
            f"(bands [{lo}, {hi}) of {nb}) is not multiplet-safe.")
    body = "\n".join(f"  - {f}" for f in findings)

    if mode == "strict":
        raise BandWindowDegeneracyError(
            f"{head}\n{body}\n"
            f"  Fix: use --n-val {n_val_new} --n-cond {n_cond_new}, or pass "
            f"--band-degeneracy snap to widen automatically, or "
            f"--band-degeneracy off to proceed on a cut multiplet "
            f"deliberately.")
    log(f"*** {head}")
    for f in findings:
        log(f"***   - {f}")
    if (n_val_new, n_cond_new) != (n_val, n_cond):
        log(f"*** SNAPPED OUTWARD to n_val={n_val_new} n_cond={n_cond_new} "
            f"(bands [{lo_new}, {hi_new}) of {nb}).  The BSE problem is now "
            f"{n_val_new * n_cond_new} pairs per k instead of "
            f"{n_val * n_cond}.  Pass --band-degeneracy strict to make this "
            f"an error, or off to keep the requested counts. ***")
    else:
        # Everything found was un-snappable (the midline, or a multiplet that
        # runs off the end).  Printing "SNAPPED to the same numbers" here
        # would read as a repair that did not happen.
        log(f"*** NOT SNAPPED: n_val={n_val} n_cond={n_cond} kept — nothing "
            f"above can be repaired by widening the window. ***")
    return n_val_new, n_cond_new


def check_band_window(
    enk_ry: np.ndarray,
    b_min: int,
    b_max: int,
    *,
    tol_ry: float = DEGENERACY_TOL_RY,
    mode: str = "snap",
    where: str = "band window",
    log=print,
) -> None:
    """Report-only twin of :func:`resolve_band_window` for a fixed window.

    Used where the window is an OUTPUT SHAPE the caller has already committed
    to — ``bandstructure.bse_setup.compute_wfns_fi``'s ``band_window_fi`` is
    the case in the tree — so widening is not available and the honest move is
    to say the boundary is unsafe and let the caller's own window resolution
    (which does snap) be the thing that fixes it.

    ``mode="strict"`` raises; ``"snap"`` warns (there is nothing to snap here,
    so it degrades to a warning by design); ``"off"`` is silent.
    """
    if mode not in MODES:
        raise ValueError(
            f"check_band_window: mode={mode!r} is not one of {MODES}")
    if mode == "off":
        return
    e = np.asarray(enk_ry, dtype=np.float64)
    gaps = boundary_min_gaps(e)
    nb = e.shape[1]
    bad = []
    for b, name in ((int(b_min), "lower"), (int(b_max), "upper")):
        if 0 < b < nb and gaps[b] <= float(tol_ry):
            k, g = _tightest_k(e, b)
            bad.append(f"{name} boundary at band {b} (gap "
                       f"{g * 1e3 * _EV_PER_RY:.3f} meV at k={k})")
    if not bad:
        return
    msg = (f"[band-window] {where}: window [{b_min}, {b_max}) cuts a "
           f"degenerate multiplet — " + "; ".join(bad) +
           f".  Tolerance {float(tol_ry) * 1e3 * _EV_PER_RY:.3f} meV.")
    if mode == "strict":
        raise BandWindowDegeneracyError(msg)
    log(f"*** {msg} ***")
