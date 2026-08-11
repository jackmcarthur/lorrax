"""Degeneracy-safe SPECTRAL cuts — the sibling of ``common/band_degeneracy``.

THE PROBLEM, and it is the same one twice.  ``common/band_degeneracy`` guards
a cut through a BAND spectrum: a window boundary that lands inside a
degenerate multiplet keeps half of it, and half a multiplet is not a
point-group-invariant subspace, so nothing built on it is covariant.  This
module guards a cut through a SINGULAR-VALUE or EIGENVALUE spectrum, where
the argument is identical and the consequence is identical.

Every rank truncation in LORRAX has the shape

    keep = { i : σ_i > σ_max · rtol }

(``common/rank_criterion`` derives ``rtol``, and nothing here changes it).
The retained set is a SUBSPACE, and the operator whose spectrum it is —
a charge Gram ``C_q``, a ζ-overlap Gram ``G_S``, ψ-at-centroids — commutes
with the crystal point group when its inputs are orbit- and
degeneracy-closed.  A symmetry that commutes with an operator maps each of
its eigenspaces onto itself and **mixes the members of a degenerate block
freely**.  Cut between whole blocks and the retained span is invariant, so
``C_{Sq} = P C_q P†`` survives the truncation.  Cut THROUGH a block and the
retained span is a symmetry-arbitrary slice of an eigenspace — a different
subspace at ``Sq`` than at ``q``, chosen by round-off — and the k-star
identity is gone for everything downstream.

WHERE THIS CAME FROM.  ``tests/known_failures/2026-08-10-ibz-cascade-vs-\
full-bz-sigma-6x6x6.md`` §6 conjectured exactly this as the cause of a
110 meV IBZ-vs-full-BZ split at Si 6×6×6, and the conjecture was REFUTED
there: the cause was the band window one index over, and the ζ rank cut on
that deck turned out to be whole-star covariant (0 of 16 q-stars carried a
non-constant ``n_keep``).  But it was covariant **by luck** — measured after
the fact, not enforced by anything — and the hazard the section describes is
real and unguarded at every spectral cut in the tree.  This module is the
enforcement the refutation left owing.  Read that file for the whole saga.

WHAT THIS MODULE DOES.  Given a spectrum and a proposed cut, find the
degenerate cluster straddling the cut, if any, and move the cut so that the
cluster is not split.  **The cut moves so that the whole straddled block is
DROPPED** — the keep-fewer direction — on the owner's ruling of 2026-08-10:

    "we just obtain singular values and truncate, and if we're truncating in
     the middle of a block of degenerate singular values we should truncate
     the whole block."

Three modes, spelled the way its sibling spells them:

``"snap"`` (default)
    Close the cut in :data:`DEFAULT_DIRECTION` and say so, loudly, naming the
    cluster's values, its size, and the cut it moved.
``"strict"``
    Refuse, naming the same things and the rank that would work.
``"off"``
    No check.  Kept for the same reason its sibling keeps one: a guard with
    no off switch gets deleted rather than configured.

THE TWO-RULE FAMILY, BECAUSE THE SIBLINGS ROUND OPPOSITE WAYS ON PURPOSE
------------------------------------------------------------------------
Three guards in this tree quantise a user- or criterion-supplied number onto
a symmetry-legal ladder, and a reader who has just met one will assume the
others round the same way.  They do not, and the split is not an
inconsistency — it is which side of the boundary is the safe one for the
quantity being rounded.

**KEPT-SET quantities FLOOR to a symmetric boundary — keep fewer.**  This
module (drop the straddled block) and ``gw/downfold``'s orbit floor (realize
the largest union of whole orbits that does not exceed the requested μ_S)
both round DOWN.  What they round is a retained set, and the failure mode of
rounding a retained set UP is that you admit directions nobody certified: at
a rank cut the straddled block sits AT the rcond boundary, so it is
noise-adjacent by construction and keeping it adds ill-conditioned
directions to the very pseudo-inverse the cut exists to condition.  Flooring
cannot do that.  It is also the strictly safe side of the amplification cap:
dropping the smallest retained values RAISES ``λ_min(kept)``, so ``κ_eff``
can only IMPROVE, and ``common/rank_criterion``'s cap is satisfied by
construction rather than by a slack term (see WHAT THE FLIP DID TO κ_eff).

**BAND WINDOWS include whole multiplets, or refuse.**  ``common/
band_degeneracy`` guards a window boundary in a BAND spectrum, and there
keep-more is the only correct repair, because the window is a statement
about which physical states enter the calculation and half a multiplet is
not a state of anything.  Its default is ``strict`` on the owner's ruling of
2026-08-10 — a widened window is a DIFFERENT CALCULATION (4v4c became 4v8c,
1024 dimensions became 2048, and a gate read 0.0906 eV of regression that no
branch had caused), so it refuses rather than repairing, and
``AGENT_PREAMBLE`` carries the standing rule: **never set ``snap`` to make a
gate pass.**  That guard is untouched by this module and by the ruling above.

The one-line discriminator: **a band window says WHICH STATES exist and
rounds outward; a rank cut says HOW MANY DIRECTIONS are trustworthy and
rounds inward.**

THE TOLERANCE IS RELATIVE-TO-NEIGHBOUR, NOT RELATIVE-TO-σ_max
--------------------------------------------------------------
This is the one place the analogy with ``band_degeneracy`` breaks, and
getting it wrong makes the guard useless rather than merely wrong.  Band
energies live on ONE scale, so an absolute tolerance in Ry works there.
Spectral cuts live at ``σ ≈ σ_max · rtol`` — eight to ten DECADES below the
top of the spectrum.  A tolerance measured against ``σ_max`` would declare
the entire retained tail one enormous cluster.  So two spectral values are
"in the same block" when they agree relative to EACH OTHER:

    same block  ⟺  |σ_i − σ_j| ≤ rtol_deg · max(|σ_i|, |σ_j|)

which is scale-free and therefore equally sharp at every depth.  Clusters
are formed by the same walk ``band_degeneracy`` documents: sort by
magnitude, break wherever successive values fail that test.  The question
this module answers is again a boundary question — "may I cut here?" — and
again it is answered by ONE number, the relative gap at the cut.

:data:`DEFAULT_RTOL` is **1e-6**, and it is bracketed from both sides by
measurements already in the tree.

* FROM ABOVE, the gap it must not swallow.  On the fixed Si 6×6×6 deck
  (``armF``, ``nband=68``) the ζ truncation fires on all 216 q with
  ``λ_min_kept/λ_drop_hi`` as low as **1.46**, i.e. a relative gap at the
  cut of ``1 − 1/1.46 = 0.315``.  1e-6 is five decades below that, so the
  arm whose Σ_x star identity is exactly 0.0000 meV stays SILENT under this
  guard.  That is a gate, not a hope — ``tests/test_spectral_closure.py``
  asserts it on the measured ratio.
* FROM BELOW, the noise floor it must not sink into.  Eigenvalues that are
  equal by symmetry are computed with backward error ``O(ε·σ_max)``, so near
  a cut at ``σ ≈ σ_max·rcond`` their achievable RELATIVE agreement is only
  ``ε/rcond``.  :func:`degeneracy_noise_rtol` computes it: 2.2e-8 at the
  production ``zeta_rcond = 1e-8``, and **2.2e-6 at the 1e-10 the 6×6×6 deck
  used** — above the default.  Every report prints that floor beside the
  tolerance, and says so when the tolerance is below it, because a guard
  looking for agreement finer than the arithmetic can deliver finds nothing
  and reports a clean bill.

WHAT THE FLIP DID TO κ_eff, AND WHY THAT SETTLED THE DIRECTION
--------------------------------------------------------------
The guard landed (1e0d9e23) snapping the cut OUTWARD, and the argument for
that default was a bound: the admitted directions are within ``rtol_deg`` of
ones already retained, so ``κ_eff`` moves by at most ``(1 + rtol_deg)^m``
across an m-member cluster — under one part in 10⁴, measured.  The bound was
correct and it is not what was wrong with the default.  What was wrong is
the SIGN.

Under the ruling the cut drops the block, and the two directions are not
symmetric in their effect on the quantity the truncation exists to control:

* **drop** removes the smallest retained values, so ``λ_min(kept)`` rises and
  ``κ_eff = λ_max/λ_min(kept)`` **falls**.  It is bounded BELOW by 1 and
  above by the old ``κ_eff``: the amplification cap ``κ_eff ≤ 1/rcond`` that
  ``common/rank_criterion`` derives cannot be violated by this move, in any
  direction, at any block size.
* **keep** admits values below the old ``λ_min(kept)``, so ``κ_eff`` **rises**
  — by a bounded and small amount, but through the cap, which is why the
  landed version needed a ``(1+rtol)^m`` slack term in every call site's cap
  assertion in order not to trip its own guard.

So the flip deletes a slack term rather than adding one, and the call sites
assert the cap with **no slack at all** under the default direction.  A
guard that has to widen the invariant it is protecting in order to fit is
worth a second look; this one no longer does.

``keep_block`` remains available per call site, for any site with a MEASURED
reason to differ — see :data:`DIRECTIONS`.  No site in this tree uses it, and
a site that turns out to NEED keep-more to stay correct is a finding to
report rather than a flag to set: it would mean some downstream object
requires the dropped directions, which is a statement about that object's
conditioning and not about this guard.

TWO EXECUTION SURFACES, ONE CRITERION
-------------------------------------
The band-window guard is host-only because band windows are resolved on
host.  Spectral cuts are not: the ζ fit's is inside a jitted kernel that
never brings its eigenvalues to host.  So this module has two faces over one
criterion.

:func:`resolve_spectral_cut` / :func:`cluster_at_cut` are host numpy: full
report, full messages, ``strict`` raises where it is called.

:func:`close_keep_mask` is pure ``jnp``, batched, jit- and vmap-safe, and
returns the moved mask plus the numbers a caller needs to print.  It is a
cumulative-AND over adjacency links rather than a while-loop, so its trip
count does not depend on the data — a REVERSE one running up from the cut
for ``drop_block``, a forward one running down from the top for
``keep_block``, which is the only structural difference between the two.

``strict`` at a device site cannot raise where it fires — a jitted kernel
cannot raise at all, which is the same division of labour
``centroid/pivoted_cholesky`` already documents ("a jitted kernel cannot
raise, so it reports and this refuses").  So a device site that fires under
``strict`` records the finding through a host callback and
:func:`raise_if_pending` refuses at the next host seam, before the result is
consumed.  The alternative — letting one seam refuse and another whisper —
is the failure ``band_degeneracy`` names in its own ``check_band_window``
docstring, and it is not repeated here.

RELATION TO ``common/rank_criterion``
-------------------------------------
That module decides HOW MANY directions to keep and why (a cap on how much
the pseudo-inverse may amplify round-off); this one decides WHERE that many
is allowed to land.  They are orthogonal and deliberately not merged: every
call site here takes the rank ``rank_criterion`` chose and, under the default
direction, returns one that is **no larger**.  So the amplification cap is
still the thing that sizes the retained set, this guard only refuses to let
it stop mid-block, and the cap it was sized by holds afterwards without
slack — ``rank_criterion.violations()`` downstream is measuring a κ_eff this
guard can only have lowered.

THE ONE FAILURE THE DROP DIRECTION HAS AND THE KEEP DIRECTION DOES NOT is
that a block can swallow the entire retained set: if every value from
``σ_max`` down to the cut is one degenerate block, dropping it leaves rank
zero.  That is not a repair and it is never applied silently —
:func:`resolve_spectral_cut` RAISES on it in every mode but ``off``, and the
device face reports a zero count its caller must refuse on.  It means the
spectrum is flat to ``rtol_deg`` across the whole retained range, which is a
statement about the operator and not about the cut.
"""
from __future__ import annotations

import math

__all__ = [
    "DEFAULT_RTOL",
    "DEFAULT_MODE",
    "MODES",
    "DEFAULT_DIRECTION",
    "DIRECTIONS",
    "SpectralClusterError",
    "SpectralBlockEmptiesCut",
    "degeneracy_noise_rtol",
    "cluster_at_cut",
    "resolve_spectral_cut",
    "describe_clean",
    "close_keep_mask",
    "resolve_mode",
    "resolve_direction",
    "note_device_snap",
    "raise_if_pending",
    "pending",
]

#: f64 unit round-off.  Spelled out rather than imported so this module stays
#: numpy-optional at import time, exactly as ``rank_criterion`` does.
_EPS64 = 2.220446049250313e-16

#: Default "same degenerate block" tolerance, RELATIVE TO THE NEIGHBOUR.
#:
#: 1e-6.  Bracketed by measurement from both sides — see "THE TOLERANCE IS
#: RELATIVE-TO-NEIGHBOUR" in the module docstring.  It is NOT
#: ``band_degeneracy.DEGENERACY_TOL_RY`` (an absolute 1 meV) and NOT
#: ``rank_criterion``'s ``rtol`` (the amplification cap's reciprocal); all
#: three answer different questions and sharing one would be wrong for two
#: of them.
DEFAULT_RTOL: float = 1.0e-6

#: The three modes, in the order the docstring introduces them.
MODES = ("snap", "strict", "off")

#: THE ONE PLACE THE DEFAULT IS DECIDED.  Every function default and every
#: call site reads this name — the way ``snap`` survived a day as an unwanted
#: default in the band-window guard was that "the default" was spelled six
#: times in three files.
DEFAULT_MODE: str = "snap"

#: WHICH WAY A STRADDLED BLOCK IS CLOSED.  Orthogonal to :data:`MODES`, which
#: says only whether the guard repairs, refuses, or is absent.
#:
#: ``"drop_block"``
#:     Drop the whole straddled block: the cut moves UP to the block's top
#:     edge and the retained set gets SMALLER.  The owner's ruling of
#:     2026-08-10 and the default — see the module docstring's THE TWO-RULE
#:     FAMILY and WHAT THE FLIP DID TO κ_eff.
#: ``"keep_block"``
#:     Keep the whole straddled block: the cut moves DOWN past the block's
#:     bottom edge and the retained set gets LARGER.  This was the default
#:     that landed at 1e0d9e23 and it is retained as an option ONLY for a
#:     call site with a measured reason to differ.  **No site in this tree
#:     passes it.**  A site that needs it to stay correct is a finding to
#:     report, not a flag to set.
DIRECTIONS = ("drop_block", "keep_block")

#: THE ONE PLACE THE DIRECTION IS DECIDED.  Same rule as :data:`DEFAULT_MODE`:
#: spelled once, read by every function default and every call site.
DEFAULT_DIRECTION: str = "drop_block"

#: NAME of the environment dial the callers read — the seams this guard sits
#: in have no deck key yet, so the mode arrives from the environment.
#:
#: THIS MODULE DOES NOT READ IT.  ``tests/test_layering.py``: "L2 is
#: physics-agnostic mathematics and must be a function of its arguments … if
#: it must come from the environment, the driver reads it and passes it."
#: So the NAME lives here, once, and each call site does the ``os.environ``
#: lookup itself and hands the answer to :func:`resolve_mode`.  That keeps
#: both rules: no literal is duplicated, and this module stays a function of
#: its arguments.
MODE_ENV = "LORRAX_SPECTRAL_CLOSURE"


class SpectralClusterError(RuntimeError):
    """A rank cut falls inside a degenerate block, in ``strict`` mode."""


class SpectralBlockEmptiesCut(SpectralClusterError):
    """Dropping the straddled block would leave rank zero.

    The failure that only the ``drop_block`` direction has: the block
    straddling the cut reaches all the way to ``σ_max``, so there is no
    non-empty closed retained set below the cut.  Raised in every mode but
    ``off``, because a silent rank of zero is not a repair — see the module
    docstring's closing paragraph.
    """


def resolve_mode(explicit=None) -> str:
    """Validate a mode and supply :data:`DEFAULT_MODE` when there is none.

    ``explicit`` is whatever the caller read — typically
    ``os.environ.get(MODE_ENV)``, which is ``None`` when unset.  A
    mis-spelled mode RAISES rather than falling back to ``off``: a guard
    silently disarmed by a typo is worse than no guard, because the log then
    shows a clean run.
    """
    if explicit is None:
        return DEFAULT_MODE
    mode = str(explicit).strip().lower()
    if mode not in MODES:
        raise ValueError(
            f"{MODE_ENV}={mode!r} is not one of {MODES}.  A mis-spelled guard "
            f"mode is not silently 'off'.")
    return mode


def resolve_direction(explicit=None) -> str:
    """Validate a direction and supply :data:`DEFAULT_DIRECTION` when there is none.

    Deliberately NOT readable from the environment.  The mode is a run-time
    dial because a user may want to audit a deck under ``strict``; the
    direction is a RULING, and a ruling that a stray environment variable can
    reverse is not one.  A call site that must differ passes
    ``direction="keep_block"`` in source, where it is reviewable.
    """
    if explicit is None:
        return DEFAULT_DIRECTION
    d = str(explicit).strip().lower()
    if d not in DIRECTIONS:
        raise ValueError(
            f"spectral_closure direction {d!r} is not one of {DIRECTIONS}.  "
            f"The default is {DEFAULT_DIRECTION!r} (drop the straddled "
            f"block), on the owner's ruling of 2026-08-10.")
    return d


def degeneracy_noise_rtol(rcond) -> float:
    """``ε/rcond`` — the finest RELATIVE agreement symmetry-degenerate values can show.

    A Hermitian eigensolver returns ``λ̃ = λ + O(ε·λ_max)``.  Two values that
    are equal by symmetry therefore agree only to ``ε·λ_max`` in ABSOLUTE
    terms, and at the cut — where ``λ ≈ λ_max·rcond`` — that is ``ε/rcond``
    RELATIVE.  A degeneracy tolerance below this line cannot resolve a
    symmetry multiplet from two unrelated neighbours and will report every
    cut clean.  Printed beside the tolerance in every report; it is a
    REFERENCE LINE, never the tolerance itself, for the same reason
    ``rank_criterion.noise_floor_rtol`` is one.
    """
    rc = float(rcond)
    if not (rc > 0.0):
        return float("inf")
    return _EPS64 / rc


def _desc_mag(values):
    """``(magnitudes descending, original indices)`` as plain Python lists.

    Cuts are on ``|σ|``: the charge Gram's spectrum is PSD and ascending out
    of ``eigh``, the transverse CCT's is Hermitian INDEFINITE and cut on
    ``|λ|`` (both signs are physical there), and an SVD is already
    descending.  Sorting by magnitude normalises all three, exactly as
    ``rank_criterion._as_desc_list`` normalises order.
    """
    pairs = sorted(((abs(float(v)), i) for i, v in enumerate(values)),
                   key=lambda p: -p[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def cluster_at_cut(values, n_keep, *, rtol=DEFAULT_RTOL, direction=None):
    """The degenerate block straddling a cut, if there is one.

    Parameters
    ----------
    values : iterable of float
        The spectrum the cut is made in.  Any order; magnitudes are used.
    n_keep : int
        Proposed retained count — the largest ``n_keep`` by magnitude, which
        is what every relative-threshold cut in the tree selects.
    rtol : float
        Two neighbours are in the same block when they agree to this
        RELATIVE to the larger of them.
    direction : str or None
        :data:`DIRECTIONS` member, or ``None`` for :data:`DEFAULT_DIRECTION`.
        BOTH resolutions are always reported; this only says which of them
        ``n_keep_closed`` is.

    Returns
    -------
    dict
        ``n_keep``          the proposal, clamped to the spectrum
        ``n_keep_dropped``  cut moved UP to the block's top edge — the whole
                            straddled block dropped (the ruling's direction)
        ``n_keep_kept``     cut moved DOWN past the block's bottom edge — the
                            whole straddled block retained
        ``n_keep_closed``   whichever of the two ``direction`` selects; equal
                            to ``n_keep`` when the cut already falls in a gap
        ``direction``       the resolved direction, so a log line cannot
                            claim one and report the other
        ``empties``         ``True`` when ``n_keep_dropped == 0`` while the
                            guard fired: the block reaches ``σ_max`` and the
                            drop direction has no non-empty answer
        ``fired``           whether the cut fell inside a block
        ``gap_rel``         relative gap AT the proposed cut (``inf`` when the
                            cut is at an end and slices nothing)
        ``gap_rel_closed``  relative gap at the MOVED cut — the guard's own
                            proof that it landed in a gap
        ``members``         the block's magnitudes, descending, as a list —
                            the whole cluster, so a message can name it
        ``span_rel``        ``(max − min)/max`` over the block: the bound on
                            how far κ_eff can move in either direction
        ``sigma_max``       largest magnitude in the spectrum
        ``last_kept``/``first_dropped`` magnitudes either side of the cut
        ``kappa``/``kappa_closed`` achieved amplification before and after.
                            Under ``drop_block`` the second is ``<=`` the
                            first, always — see the module docstring.

    Nothing here looks for a knee, a plateau or an elbow.  It asks one
    question at one boundary, which is the only question a smooth spectrum
    can answer (``common/rank_criterion``, at length).
    """
    d = resolve_direction(direction)
    mag, _ = _desc_mag(values)
    n = len(mag)
    k = max(0, min(int(n_keep), n))

    def _kappa(kk):
        """``σ_max/σ_min(kept)`` at retained count ``kk`` — the invariant
        ``rank_criterion`` caps, reported here at the MOVED cut so the change
        the guard makes to it is never invisible."""
        if kk <= 0 or not mag or not (mag[0] > 0.0) or not (mag[kk - 1] > 0.0):
            return None
        return mag[0] / mag[kk - 1]

    def _gap_at(c):
        """Relative gap at a cut of ``c``; ``inf`` at either end."""
        if c <= 0 or c >= n:
            return float("inf")
        s = max(mag[c - 1], mag[c])
        return (mag[c - 1] - mag[c]) / s if s > 0.0 else float("inf")

    out = {
        "n_keep": k, "n_keep_dropped": k, "n_keep_kept": k,
        "n_keep_closed": k, "direction": d, "empties": False, "fired": False,
        "gap_rel": float("inf"), "gap_rel_closed": float("inf"),
        "members": [], "span_rel": 0.0,
        "sigma_max": (mag[0] if mag else 0.0),
        "last_kept": (mag[k - 1] if k > 0 else None),
        "first_dropped": (mag[k] if k < n else None),
        "kappa": _kappa(k), "kappa_closed": _kappa(k),
        "n_total": n, "rtol": float(rtol),
    }
    # A cut at either end slices nothing — the same reason
    # ``band_degeneracy.boundary_min_gaps`` makes the outer boundaries +inf.
    if k <= 0 or k >= n:
        return out
    hi, lo = mag[k - 1], mag[k]
    scale = max(hi, lo)
    if not (scale > 0.0):
        # An all-zero tail either side of the cut.  A RELATIVE test is
        # meaningless on it, and ``rank_criterion.violations()`` already
        # refuses a zero/NaN spectrum, so say nothing rather than move the
        # cut through an exactly-null block (which is the mesh pad, and is
        # inert by construction).
        out["gap_rel"] = float("inf")
        return out
    out["gap_rel"] = (hi - lo) / scale
    if out["gap_rel"] > float(rtol):
        return out

    # The cut is inside a block.  Find BOTH its edges with the same
    # successive-difference walk ``band_degeneracy`` documents, with the
    # relative test in place of the absolute one.  Which edge the cut moves
    # to is ``direction``'s business, not the walk's.
    #
    # DOWNWARD, to the block's bottom: the cut that keeps the block whole.
    j = k
    while j < n:
        a, b = mag[j - 1], mag[j]
        s = max(a, b)
        if not (s > 0.0) or (a - b) / s > float(rtol):
            break
        j += 1
    # UPWARD, to the block's top: the cut that DROPS the block whole, which
    # is the default.  Note this walk must reach index 0 and stop there — a
    # block that runs to the top of the spectrum has no closed non-empty cut
    # below it, and that is the ``empties`` case rather than a rank of 0.
    i = k - 1
    while i > 0:
        a, b = mag[i - 1], mag[i]
        s = max(a, b)
        if not (s > 0.0) or (a - b) / s > float(rtol):
            break
        i -= 1
    members = mag[i:j]
    out["fired"] = True
    out["members"] = members
    out["n_keep_dropped"] = i
    out["n_keep_kept"] = j
    out["empties"] = (i == 0)
    closed = i if d == "drop_block" else j
    out["n_keep_closed"] = closed
    out["kappa_closed"] = _kappa(closed)
    # The gap at the NEW cut — the guard's own proof that it landed in one.
    # ``inf`` at either end: off the bottom means the block is still open and
    # the caller must be told; off the top is the ``empties`` case.
    out["gap_rel_closed"] = _gap_at(closed)
    if members and members[0] > 0.0:
        out["span_rel"] = (members[0] - members[-1]) / members[0]
    return out


def _fmt_block(members, head=4, tail=4):
    """The block's values, elided in the middle when it is long."""
    if len(members) <= head + tail:
        return ", ".join(f"{v:.9e}" for v in members)
    return (", ".join(f"{v:.9e}" for v in members[:head]) + ", ... , "
            + ", ".join(f"{v:.9e}" for v in members[-tail:]))


def _message(info, *, where, rcond=None):
    """The loud paragraph both modes print, built once so they cannot drift."""
    k, mem = info["n_keep"], info["members"]
    lo_edge, hi_edge = info["n_keep_dropped"], info["n_keep_kept"]
    closed, d = info["n_keep_closed"], info["direction"]
    # The block occupies sorted positions [lo_edge, hi_edge); the cut at k
    # keeps the ones above it and drops the ones below.
    kept_in_block = k - lo_edge
    floor = ""
    if rcond is not None:
        nf = degeneracy_noise_rtol(rcond)
        floor = (f"\n  - noise floor for this cut: eps/rcond={nf:.2e}, against "
                 f"rtol {info['rtol']:.1e}."
                 + ("  ** THE TOLERANCE IS BELOW THE FLOOR: symmetry-degenerate "
                    "values are not resolvable at this rcond, so a CLEAN "
                    "verdict from this guard would mean little. **"
                    if float(info["rtol"]) < nf else ""))
    if info["kappa"] and info["kappa_closed"]:
        arrow = ("down" if info["kappa_closed"] <= info["kappa"] else "UP")
        kap = (f", so kappa_eff moves {arrow} {info['kappa']:.4e} -> "
               f"{info['kappa_closed']:.4e}"
               + (" — dropping the block can only IMPROVE the amplification "
                  "the cut exists to cap (common/rank_criterion)"
                  if d == "drop_block" else
                  " — keep_block RAISES it; this site opted out of the "
                  "default direction and owes a measured reason"))
    else:
        kap = " (kappa_eff is undefined here — the retained block reaches zero)"
    still_open = ("" if hi_edge < info["n_total"] else
                  "\n  - ** THE BLOCK IS STILL OPEN AT THE BOTTOM OF THE "
                  "SPECTRUM: every value below the cut is inside it, so "
                  "keep_block would retain the whole spectrum. **")
    empties = ("" if not info["empties"] else
               "\n  - ** THE BLOCK REACHES sigma_max: every value ABOVE the "
               "cut is inside it too, so dropping it leaves rank ZERO and "
               "there is no non-empty closed cut.  The spectrum is flat to "
               "rtol across the whole retained range — that is a statement "
               "about the operator, not about the cut. **")
    return (
        f"[spectral-closure] {where}: the rank cut at {k} of {info['n_total']} "
        f"falls INSIDE a degenerate block.\n"
        f"  - the block holds {len(mem)} values [{_fmt_block(mem)}] at sorted "
        f"positions [{lo_edge}, {hi_edge}); the relative gap AT the cut is "
        f"{info['gap_rel']:.3e} <= rtol {info['rtol']:.1e}, so the cut keeps "
        f"{kept_in_block} of them and drops {hi_edge - k}.\n"
        f"  - keeping part of a degenerate block retains a symmetry-ARBITRARY "
        f"slice of an eigenspace: the retained span is a different subspace at "
        f"Sq than at q, chosen by round-off, and the k-star identity fails for "
        f"everything built on it.\n"
        f"  - direction={d}: the cut moves to {closed}, "
        + (f"DROPPING the whole block" if d == "drop_block" else
           f"KEEPING the whole block")
        + f" (relative gap at the new cut {info['gap_rel_closed']:.3e}; the "
        f"other legal cut is {hi_edge if d == 'drop_block' else lo_edge}).  "
        f"The block spans {info['span_rel']:.3e} relative{kap}."
        + still_open + empties + floor)


def describe_clean(info, *, where):
    """The one line a site prints when the guard did NOT fire.

    "No news" and "a good number" must not look alike (preamble measurement
    rule 10): a site that only speaks when it fires leaves a reader unable to
    tell a checked cut from an unchecked one.
    """
    if info["gap_rel"] == float("inf"):
        return (f"    [spectral-closure] {where}: cut at {info['n_keep']} of "
                f"{info['n_total']} slices nothing (it is at an end of the "
                f"spectrum) — exempt.")
    return (f"    [spectral-closure] {where}: cut at {info['n_keep']} of "
            f"{info['n_total']} falls in a gap — relative gap "
            f"{info['gap_rel']:.3e} against rtol {info['rtol']:.1e} "
            f"({info['gap_rel'] / max(info['rtol'], 1e-300):.3g}x the "
            f"tolerance).  No degenerate block is cut.")


def resolve_spectral_cut(values, n_keep, *, rtol=DEFAULT_RTOL, mode=None,
                         direction=None, where="spectral cut", rcond=None,
                         log=print):
    """Return a retained count whose cut does not slice a degenerate block.

    The host face.  ``values`` is the spectrum, ``n_keep`` the rank
    ``common/rank_criterion`` selected.  Returns ``(n_keep_out, info)`` with
    ``info`` exactly :func:`cluster_at_cut`'s dict, so a caller can log the
    numbers whether or not the guard fired — "no news" and "a good number"
    must not look alike (preamble measurement rule 10).

    ``mode`` defaults to :func:`resolve_mode` (``snap``, or the
    ``LORRAX_SPECTRAL_CLOSURE`` env).  ``strict`` raises
    :class:`SpectralClusterError`; ``off`` returns the proposal untouched
    and does not even look.

    ``direction`` defaults to :data:`DEFAULT_DIRECTION` — drop the straddled
    block.  It is a source-level argument with no environment dial; see
    :func:`resolve_direction`.

    RAISES :class:`SpectralBlockEmptiesCut` in ``snap`` as well as ``strict``
    when dropping the block would leave rank zero.  A repair that returns an
    empty basis is not a repair, and the caller cannot tell an empty result
    from a working one by its type.
    """
    m = resolve_mode(mode)
    d = resolve_direction(direction)
    if m == "off":
        # ``rtol = -1`` can never fire (a relative gap is >= 0), so ``off``
        # still returns the FULL dict — the numbers stay available to a log
        # line even when the guard is disarmed, and no caller has to special-
        # case a short dict.
        return int(n_keep), cluster_at_cut(values, n_keep, rtol=-1.0,
                                           direction=d)
    info = cluster_at_cut(values, n_keep, rtol=rtol, direction=d)
    if not info["fired"]:
        return int(info["n_keep"]), info
    msg = _message(info, where=where, rcond=rcond)
    if d == "drop_block" and info["empties"]:
        raise SpectralBlockEmptiesCut(
            msg + f"\n  This is not a mode question: no mode but 'off' can "
                  f"return rank 0 here.  Fix: raise rcond so the cut lands "
                  f"above the flat range, or establish that this operator is "
                  f"genuinely rank-{info['n_keep_kept']} and cut there.")
    if m == "strict":
        raise SpectralClusterError(
            msg + f"\n  Fix: keep {info['n_keep_closed']} instead of "
                  f"{info['n_keep']}, or run with LORRAX_SPECTRAL_CLOSURE=snap "
                  f"to move the cut automatically, or =off to cut through the "
                  f"block deliberately.")
    for line in msg.splitlines():
        log(f"*** {line}")
    log(f"*** {'DROPPED THE BLOCK' if d == 'drop_block' else 'KEPT THE BLOCK'}"
        f": retained rank {info['n_keep']} -> {info['n_keep_closed']}. ***")
    return int(info["n_keep_closed"]), info


# ---------------------------------------------------------------------------
# The device face — pure jnp, batched, jit- and vmap-safe
# ---------------------------------------------------------------------------

def close_keep_mask(values, keep, *, rtol=DEFAULT_RTOL, direction=None):
    """Move a keep-mask off any degenerate block it cuts.  Pure ``jnp``.

    Parameters
    ----------
    values : jnp array ``(..., n)``
        Spectrum, ANY order (the ζ charge route hands us ascending ``eigh``
        output; the transverse route hands us an indefinite spectrum whose
        magnitudes are V-shaped).  Magnitudes are used.
    keep : bool jnp array ``(..., n)``
        The proposed retained set.  Must be "the largest ``n_keep`` by
        magnitude", which is what a relative threshold always selects; it is
        NOT required to be contiguous in the input order.
    rtol : float
        Relative-to-neighbour block tolerance.
    direction : str or None
        :data:`DIRECTIONS` member; ``None`` is :data:`DEFAULT_DIRECTION`.
        Static (a Python string), so the branch below is resolved at trace
        time and neither kernel carries the other.

    Returns
    -------
    ``(keep_out, n_keep, n_keep_out)``
        ``keep_out`` in the INPUT order; the two counts are ``(...)``-shaped
        int arrays so a caller can print or compare them per q.  Under
        ``drop_block``, ``n_keep_out <= n_keep``, and **a caller must refuse
        on ``n_keep_out == 0``** — a jitted kernel cannot raise, so the empty
        case that :func:`resolve_spectral_cut` raises on arrives here as a
        count.

    NO DATA-DEPENDENT TRIP COUNT, in either direction.  The walk that the
    host face writes as a ``while`` is here a cumulative AND over adjacency
    links: sort descending by magnitude and mark each adjacent pair "linked"
    when it passes the relative test.  Then force the links the walk does not
    care about to linked (they are traversed vacuously) and take the running
    AND from the end the walk starts at.  One sort, one cumprod, both
    static-shape.

    * ``drop_block`` walks UP from the cut, so the vacuous links are the ones
      BELOW it (both their endpoints are already dropped) and the AND runs in
      REVERSE.  ``revacc[p]`` is then "every link from p up to the cut is
      unbroken" = "position p is in the block the cut straddles", and every
      such retained position is removed.
    * ``keep_block`` walks DOWN from the top, so the vacuous links are the
      ones INSIDE the kept prefix and the AND runs forward.  ``reach[i]`` is
      "position i is reachable from the top through unbroken links", which
      past the cut means "the block has not ended yet", and every such
      position is admitted.
    """
    import jax.numpy as jnp

    d = resolve_direction(direction)
    v = jnp.asarray(values)
    mag = jnp.abs(v)
    n = mag.shape[-1]
    keep = jnp.asarray(keep, dtype=bool)

    order = jnp.argsort(-mag, axis=-1)
    mag_s = jnp.take_along_axis(mag, order, axis=-1)
    keep_s = jnp.take_along_axis(keep, order, axis=-1)
    n_keep = jnp.sum(keep_s, axis=-1)

    if n < 2:
        return keep, n_keep, n_keep

    hi = mag_s[..., :-1]
    lo = mag_s[..., 1:]
    scale = jnp.maximum(hi, lo)
    # ``scale <= 0`` is an exactly-null pair (the mesh pad, inert by
    # construction) — never "linked", so the walk stops rather than sweeping
    # the whole null tail.
    linked = (scale > 0.0) & ((hi - lo) <= float(rtol) * scale)
    # Link m joins sorted positions m and m+1.
    pos = jnp.arange(n - 1)
    ones = jnp.ones(keep_s.shape[:-1] + (1,), dtype=jnp.int32)

    if d == "keep_block":
        # A link is vacuous when m+1 is already inside the kept prefix.
        chain = linked | ((pos + 1) < n_keep[..., None])
        acc = jnp.cumprod(chain.astype(jnp.int32), axis=-1)
        reach = jnp.concatenate([ones, acc], axis=-1) > 0
        # ``n_keep == 0`` retains nothing, so there is no block to be inside
        # and nothing to admit.  Without this the leading ``ones`` above
        # admits position 0 unconditionally and the device face disagrees
        # with the host's "a cut at an end slices nothing" — a latent
        # disagreement in the landed kernel, found by the two-face sweep when
        # it was extended over both directions.
        keep_s_out = keep_s | (reach & (n_keep > 0)[..., None])
    else:
        # DROP.  A link is vacuous when it lies at or below the cut: link m
        # for m >= n_keep joins two positions the cut already dropped, so
        # traversing it says nothing.  Forcing those to linked lets ONE
        # reverse scan serve every ``n_keep`` without a data-dependent
        # endpoint — the mirror of the ``keep_block`` trick above.
        chain = linked | (pos >= n_keep[..., None])
        racc = jnp.cumprod(chain[..., ::-1].astype(jnp.int32), axis=-1)[..., ::-1]
        # ``in_block[p]`` = AND(chain[p:]) = AND(linked[p : n_keep]) = the cut
        # and p are joined by an unbroken run of links.  Position n-1 is
        # appended as True and never used: it can only be selected below when
        # n_keep == n, which the ``n_keep < n`` factor excludes.
        in_block = jnp.concatenate([racc, ones], axis=-1) > 0
        straddled = (in_block
                     & (jnp.arange(n) < n_keep[..., None])
                     & (n_keep < n)[..., None])
        keep_s_out = keep_s & ~straddled

    inv = jnp.argsort(order, axis=-1)
    keep_out = jnp.take_along_axis(keep_s_out, inv, axis=-1)
    return keep_out, n_keep, jnp.sum(keep_s_out, axis=-1)


# ---------------------------------------------------------------------------
# Deferred refusal, for the seams where the guard fires inside a jit
# ---------------------------------------------------------------------------

_PENDING: list[str] = []


def note_device_snap(where, n_keep, n_keep_out) -> None:
    """Record a device-site firing so a host seam can refuse under ``strict``.

    Called from ``jax.debug.callback`` inside the ζ kernels.  It records; it
    does not raise, because raising out of a host callback inside XLA is not
    a contract this tree relies on.  :func:`raise_if_pending` is the refusal.
    """
    _PENDING.append(
        f"{where}: the rank cut fell inside a degenerate block on at least "
        f"one q — retained rank {int(n_keep)} would have become "
        f"{int(n_keep_out)} under snap ({DEFAULT_DIRECTION}).")


def pending() -> list[str]:
    """The findings recorded so far.  Read-only view, for gates."""
    return list(_PENDING)


def raise_if_pending(where="spectral cut", *, mode=None, log=print) -> None:
    """Refuse, under ``strict``, for any device-site firing recorded since the last call.

    Placed at the first host seam after a jitted truncation, so ``strict``
    means the same thing at a device site as at a host one.  Always clears,
    so a later stage cannot inherit an earlier stage's finding.
    """
    m = resolve_mode(mode)
    found, _PENDING[:] = list(_PENDING), []
    if not found:
        return
    body = "\n".join(f"  - {f}" for f in found)
    if m == "strict":
        raise SpectralClusterError(
            f"[spectral-closure] {where}: {len(found)} spectral cut(s) fell "
            f"inside a degenerate block.\n{body}\n"
            f"  Fix: LORRAX_SPECTRAL_CLOSURE=snap moves each cut off its "
            f"block by DROPPING the block whole (the owner's ruling of "
            f"2026-08-10), which lowers the retained rank and can only lower "
            f"kappa_eff with it; =off cuts through deliberately.")
    for f in found:
        log(f"*** [spectral-closure] {f} ***")
