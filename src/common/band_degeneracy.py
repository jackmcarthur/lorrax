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

``"strict"`` (default)
    Refuse.  Raise, naming the multiplet, the k at which it is tightest, and
    the counts that would work, so the user re-runs the window they meant.
    Nothing about the calculation changes behind their back.
``"snap"``
    Widen to the multiplet boundary and say so, loudly, naming the new
    counts, the offending multiplet and the k at which it is tightest.  An
    explicit opt-in since 2026-08-10; it was the default before that.
``"off"``
    No check.  Kept because a deliberately half-multiplet window is a
    legitimate *debugging* configuration, and because a guard with no
    off switch gets deleted rather than configured.

WHY STRICT IS THE DEFAULT (the owner, 2026-08-10).  ``snap`` shipped as the
default with the exciton-features landing ``824032b7``, and inside one day it
silently re-windowed two decks that were being used as measurement standards.
On ``si_bse_debug`` it turned the BerkeleyGW-parity deck's requested 4v4c into
4v8c — a 1024-dimension problem became a 2048-dimension one, every one of the
lowest twenty excitons moved, and the gate read as an 0.0906 eV code
regression that no branch had caused (``tests/KNOWN_FAILURES.md``, the anchor
row).  The parity deck's false 28.6 meV "regression" was the same mechanism
seen from the other side.  A widened window is not a repair: it is a
DIFFERENT CALCULATION, and the whole argument for this guard is that a cut
multiplet is not a thing to fix quietly.  So the default now refuses and says
which counts would work, and widening is something a user asks for by name.

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
MODES = ("strict", "snap", "off")

#: The mode every caller gets when nobody says otherwise: **strict**.
#:
#: THE ONE PLACE THIS IS DECIDED.  Every function default and every driver's
#: ``--band-degeneracy`` default reads this name; there is no second literal
#: in the tree, because the way ``snap`` survived a day as an unwanted default
#: was that "the default" was spelled six times in three files and changing it
#: meant finding all six.  Flipping this line flips the flag everywhere.
#:
#: Owner ruling, 2026-08-10 — see "WHY STRICT IS THE DEFAULT" in the module
#: docstring.
DEFAULT_MODE: str = "strict"


class BandWindowDegeneracyError(RuntimeError):
    """A window boundary splits a degenerate multiplet, in ``strict`` mode."""


def boundary_min_gaps(enk_ry: np.ndarray, *,
                      is_full_spectrum: bool) -> np.ndarray:
    """Smallest over k of the gap across each inter-band boundary.

    Parameters
    ----------
    enk_ry : (nk, nb) float array
        Band energies in **Rydberg**, ascending in the band axis at each k.
    is_full_spectrum : bool
        **REQUIRED, no default.**  ``True`` iff ``enk_ry`` is every band the
        mean field carries, so that its outer boundaries really do cut
        nothing.  ``False`` iff it is a WINDOW sliced out of a larger
        spectrum — then the outer boundaries ARE cuts whose gaps this array
        cannot see, and they come back ``nan``.

    Returns
    -------
    (nb + 1,) float array
        Element ``b`` is ``min_k |e[k, b] - e[k, b-1]|`` — the tightest gap
        anywhere in the BZ across the boundary that would separate bands
        ``< b`` from bands ``>= b``.  The two outer boundaries are ``+inf``
        on a full spectrum and ``nan`` on a window.

    WHY THIS ARGUMENT EXISTS, AND WHY IT HAS NO DEFAULT
    ---------------------------------------------------
    This function used to return ``+inf`` at ``b = 0`` and ``b = nb``
    unconditionally, on the reasoning that an outer boundary separates
    nothing.  That is right when the array IS the spectrum and **exactly
    backwards when it is a window**: the window's own edges are the cuts
    somebody made, and reporting ``+inf`` there certifies as safe the one
    thing this module exists to catch.

    MEASURED, 2026-08-15, ``si_cohsex_debug`` — 62 bands in the WFN, deck
    runs ``nband = 60``::

        boundary_min_gaps(sigma window, 60 bands)[60]  ->  +inf   "clean"
        boundary_min_gaps(mean field,   62 bands)[60]  ->  0.000000 meV

    The second is the truth: that edge slices a multiplet, and moving it to
    a clean edge (40: 818 meV, 36: 157 meV) takes every Σ channel's
    within-star spread from ~2 meV to **exactly 0.0000**.  The first is what
    a reader got, and it is why the ζ-window and star-spread analyses both
    reported the sliced edge as safe.

    A DEFAULT WOULD REBUILD THE TRAP.  Same doctrine as ``trs_reference`` in
    ``symmetry_maps``: a choice whose wrong branch is invisible to every
    cheap check does not get a default, because the caller who has not
    thought about it must get a ``TypeError`` rather than a plausible wrong
    answer.  ``nan`` rather than ``0.0`` for the unknowable case for the
    same reason — ``nan > tol`` and ``nan <= tol`` are BOTH False, so a
    window edge can be neither certified clean nor silently called dirty; it
    has to be asked about the full spectrum.

    If you want to know whether a WINDOW's own edges are clean, you cannot
    learn it from the window.  Pass the untruncated ladder and the window
    bounds to :func:`check_band_window`.
    """
    e = np.asarray(enk_ry, dtype=np.float64)
    if e.ndim != 2:
        raise ValueError(
            f"boundary_min_gaps: expected (nk, nb) energies, got shape {e.shape}")
    nb = e.shape[1]
    outer = np.inf if bool(is_full_spectrum) else np.nan
    gaps = np.full(nb + 1, outer, dtype=np.float64)
    if nb >= 2:
        gaps[1:nb] = np.min(np.abs(np.diff(e, axis=1)), axis=0)
    return gaps


def _tightest_k(enk_ry: np.ndarray, b: int) -> tuple[int, float]:
    """``(k, gap)`` at which boundary ``b`` is tightest."""
    e = np.asarray(enk_ry, dtype=np.float64)
    d = np.abs(e[:, b] - e[:, b - 1])
    k = int(np.argmin(d))
    return k, float(d[k])


def snap_cut_to_clean_boundary(
    enk_ry: np.ndarray,
    cut: int,
    *,
    tol_ry: float = DEGENERACY_TOL_RY,
    lo: int = 0,
    hi: int | None = None,
) -> int:
    """Nearest band count ``<= hi`` whose boundary splits no multiplet.

    THE SAME CONSTRAINT :func:`resolve_band_window` ENFORCES, asked as a
    one-sided question.  ``resolve_band_window`` is given a *window* the user
    asked for and may only widen it outward; this is given a single INTERNAL
    sampling point — a band count at which a converging sum is to be
    evaluated — where neither direction loses anything the user requested, so
    the nearest clean boundary is the right answer and staying near the
    requested fraction is what matters.  Both read the same one number per
    boundary, :func:`boundary_min_gaps`; there is no second notion of "clean"
    in the tree.

    Parameters
    ----------
    enk_ry : (nk, nb) float array
        Band energies in **Rydberg**, ascending in the band axis at each k.
        Pass the array the sum will actually be truncated against.
    cut : int
        Requested band count (= boundary index: bands ``< cut`` are kept).
    tol_ry : float
        Two bands are "the same multiplet" within this, in Ry.
    lo, hi : int
        Inclusive search bounds on the returned count.  ``hi`` defaults to
        ``nb``.  ``lo``/``hi`` are themselves returned unchecked when the
        walk reaches them — the caller owns what a degenerate endpoint means.

    Returns
    -------
    int
        The clean boundary nearest ``cut``, ties broken DOWNWARD (fewer
        bands), which keeps a set of ascending cuts from crossing.

    Notes
    -----
    The outer boundaries (0 and ``nb``) are ``+inf`` in
    :func:`boundary_min_gaps` and are therefore always clean — the walk
    terminates.
    """
    e = np.asarray(enk_ry, dtype=np.float64)
    nb = int(e.shape[1])
    hi = nb if hi is None else int(hi)
    lo = int(lo)
    if not (0 <= lo <= hi <= nb):
        raise ValueError(
            f"snap_cut_to_clean_boundary: bad bounds lo={lo} hi={hi} nb={nb}")
    gaps = boundary_min_gaps(e)
    tol = float(tol_ry)
    cut = int(min(max(int(cut), lo), hi))
    for delta in range(0, max(cut - lo, hi - cut) + 1):
        for cand in ((cut - delta), (cut + delta)):     # downward tie-break
            if lo <= cand <= hi and gaps[cand] > tol:
                return int(cand)
    # Unreachable while either endpoint is clean, which boundary_min_gaps
    # guarantees for 0 and nb; a caller-restricted [lo, hi] can in principle
    # contain no clean boundary at all, and that is a refusal, not a guess.
    raise BandWindowDegeneracyError(
        f"snap_cut_to_clean_boundary: no band count in [{lo}, {hi}] has a "
        f"multiplet-clean boundary at tol "
        f"{tol * 1e3 * _EV_PER_RY:.3f} meV (requested {cut}).")


def resolve_band_window(
    enk_ry: np.ndarray,
    n_occ: int,
    n_val: int,
    n_cond: int,
    *,
    tol_ry: float = DEGENERACY_TOL_RY,
    mode: str = DEFAULT_MODE,
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
    mode : {"strict", "snap", "off"}
        See the module docstring.  Defaults to :data:`DEFAULT_MODE`
        (``"strict"``): a cut multiplet stops the run rather than resizing it.
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
        In ``strict`` mode — the default — if either boundary splits a
        multiplet.
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
    # These two read INTERIOR boundaries only (both guard 0 < b < nb), so
    # the outer value cannot change their answers -- and they cannot know
    # what the caller handed them, so they declare the conservative case.
    gaps = boundary_min_gaps(e, is_full_spectrum=False)
    tol = float(tol_ry)

    lo, hi = n_occ - n_val, n_occ + n_cond
    findings: list[str] = []

    # The valence/conduction split itself.  It cannot be repaired by widening
    # — it IS the window's midline — so it is reported and never snapped.  A
    # degeneracy here means the deck has no gap at that k (a metal, or a wrong
    # n_occ), which is a different problem with a different fix.
    unrepairable = False
    if 0 < n_occ < nb and gaps[n_occ] <= tol:
        k, g = _tightest_k(e, n_occ)
        unrepairable = True
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
        unrepairable = True
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
        # The counts line is the whole point of refusing rather than snapping,
        # so it has to be TRUE.  Two of the findings above are not repairable
        # by widening at all — a gapless midline, and a multiplet still open
        # at the top of the input — and on those a "Fix: use --n-val X" line
        # is worse than no line: the user re-runs with the counts it names,
        # gets the same refusal, and concludes the guard is broken.  So the
        # fix line is only a counts line when the counts actually fix it.
        if not unrepairable:
            fix = (f"  Fix: use --n-val {n_val_new} --n-cond {n_cond_new}, or "
                   f"pass --band-degeneracy snap to widen to those counts "
                   f"automatically, or --band-degeneracy off to proceed on a "
                   f"cut multiplet deliberately.")
        else:
            reach = (f"--band-degeneracy snap would reach n_val={n_val_new} "
                     f"n_cond={n_cond_new} and the window would STILL be cut"
                     if (n_val_new, n_cond_new) != (n_val, n_cond) else
                     "--band-degeneracy snap would return the counts you "
                     "asked for")
            fix = (f"  Fix: widening cannot clear everything above "
                   f"({reach}).  What has to change is the INPUT — more bands "
                   f"(nband) above the window, or the right n_occ — not the "
                   f"window counts.  --band-degeneracy off proceeds on a cut "
                   f"multiplet deliberately.")
        raise BandWindowDegeneracyError(f"{head}\n{body}\n{fix}")
    log(f"*** {head}")
    for f in findings:
        log(f"***   - {f}")
    if (n_val_new, n_cond_new) != (n_val, n_cond):
        log(f"*** SNAPPED OUTWARD to n_val={n_val_new} n_cond={n_cond_new} "
            f"(bands [{lo_new}, {hi_new}) of {nb}).  The BSE problem is now "
            f"{n_val_new * n_cond_new} pairs per k instead of "
            f"{n_val * n_cond} — this is NOT the calculation you asked for, "
            f"and you are seeing it because --band-degeneracy snap was passed "
            f"explicitly.  The default (strict) refuses here; off keeps the "
            f"requested counts. ***")
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
    mode: str = DEFAULT_MODE,
    where: str = "band window",
    log=print,
) -> None:
    """Report-only twin of :func:`resolve_band_window` for a fixed window.

    Used where the window is an OUTPUT SHAPE the caller has already committed
    to — ``bandstructure.bse_setup.compute_wfns_fi``'s ``band_window_fi`` is
    the case in the tree — so widening is not available and the honest move is
    to say the boundary is unsafe and let the caller's own window resolution
    (which does snap) be the thing that fixes it.

    ``mode="strict"`` — the default since 2026-08-10 — raises; ``"snap"``
    warns (there is nothing to snap here, so it degrades to a warning by
    design); ``"off"`` is silent.  This twin takes its mode from the SAME
    ``--band-degeneracy`` flag as the resolver, deliberately: one flag that
    means two things depending on which seam it reaches is how a user comes to
    believe a run refused everything unsafe when one seam only whispered.
    """
    if mode not in MODES:
        raise ValueError(
            f"check_band_window: mode={mode!r} is not one of {MODES}")
    if mode == "off":
        return
    e = np.asarray(enk_ry, dtype=np.float64)
    # These two read INTERIOR boundaries only (both guard 0 < b < nb), so
    # the outer value cannot change their answers -- and they cannot know
    # what the caller handed them, so they declare the conservative case.
    gaps = boundary_min_gaps(e, is_full_spectrum=False)
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
