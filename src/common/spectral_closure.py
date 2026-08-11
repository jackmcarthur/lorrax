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
degenerate cluster straddling the cut, if any, and move the cut OUTWARD past
it — the keep-more direction, the same principle ``band_degeneracy`` widens a
window by.  Three modes, spelled the way its sibling spells them:

``"snap"`` (default here — see WHY SNAP, WHERE ITS SIBLING REFUSES)
    Extend the retained set through the whole cluster and say so, loudly,
    naming the cluster's values, its size, and the cut it moved.
``"strict"``
    Refuse, naming the same things and the rank that would work.
``"off"``
    No check.  Kept for the same reason its sibling keeps one: a guard with
    no off switch gets deleted rather than configured.

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

WHY SNAP, WHERE ITS SIBLING REFUSES
-----------------------------------
``band_degeneracy`` defaults to ``strict`` on an owner ruling (2026-08-10),
because widening a band window is *a different calculation*: 4v4c became
4v8c, a 1024-dimension problem became 2048, and a gate read 0.0906 eV of
regression that no branch had caused.  None of that transfers here, and the
reason is arithmetic rather than taste.

A snapped spectral cut admits directions whose spectral values are within
``rtol_deg`` **of ones already retained**.  The achieved amplification the
truncation exists to bound therefore moves by at most a factor
``(1 + rtol_deg)^m`` across an m-member cluster — at 1e-6 and any physical
multiplet, **under one part in 10⁴**.  Set against ``common/rank_criterion``'s
R19 anchor, where retaining 41 % more rank moved a QP gap from 3.13 eV to
−5049 eV, a spectral snap is not in the same universe of intervention as a
window snap: it is a handful of directions that the criterion could not
distinguish from ones it kept, admitted so that the retained span is a
representation of the point group instead of a slice of one.  Refusing that
by default would be refusing the repair, not protecting the calculation.

``strict`` is one word away for anyone who wants it, the achieved κ_eff
after the snap is reported every time so the change is never invisible, and
``rank_criterion``'s own ``violations()`` still guards the cap independently
downstream.  Owner row: whether ``strict`` should become the default here
too is the owner's call, not this lane's, and the flag is a single constant.

TWO EXECUTION SURFACES, ONE CRITERION
-------------------------------------
The band-window guard is host-only because band windows are resolved on
host.  Spectral cuts are not: the ζ fit's is inside a jitted kernel that
never brings its eigenvalues to host.  So this module has two faces over one
criterion.

:func:`resolve_spectral_cut` / :func:`cluster_at_cut` are host numpy: full
report, full messages, ``strict`` raises where it is called.

:func:`snap_keep_outward` is pure ``jnp``, batched, jit- and vmap-safe, and
returns the moved mask plus the numbers a caller needs to print.  It is a
suffix-AND over adjacency links rather than a while-loop, so its trip count
does not depend on the data.

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
call site here takes the rank ``rank_criterion`` chose and returns one that
is at least as large, so the amplification cap is still the thing that sizes
the retained set and this guard only refuses to let it stop mid-block.
"""
from __future__ import annotations

import math

__all__ = [
    "DEFAULT_RTOL",
    "DEFAULT_MODE",
    "MODES",
    "SpectralClusterError",
    "degeneracy_noise_rtol",
    "cluster_at_cut",
    "resolve_spectral_cut",
    "describe_clean",
    "snap_keep_outward",
    "resolve_mode",
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
#: times in three files.  See "WHY SNAP, WHERE ITS SIBLING REFUSES".
DEFAULT_MODE: str = "snap"

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


def cluster_at_cut(values, n_keep, *, rtol=DEFAULT_RTOL):
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

    Returns
    -------
    dict
        ``n_keep``          the proposal, clamped to the spectrum
        ``n_keep_snapped``  the outward-snapped count (== ``n_keep`` when the
                            cut already falls in a gap)
        ``fired``           whether the cut fell inside a block
        ``gap_rel``         relative gap AT the proposed cut (``inf`` when the
                            cut is at an end and slices nothing)
        ``members``         the block's magnitudes, descending, as a list —
                            the whole cluster, so a message can name it
        ``span_rel``        ``(max − min)/max`` over the snapped extension:
                            the bound on how much κ_eff can move
        ``sigma_max``       largest magnitude in the spectrum
        ``last_kept``/``first_dropped`` magnitudes either side of the cut

    Nothing here looks for a knee, a plateau or an elbow.  It asks one
    question at one boundary, which is the only question a smooth spectrum
    can answer (``common/rank_criterion``, at length).
    """
    mag, _ = _desc_mag(values)
    n = len(mag)
    k = max(0, min(int(n_keep), n))

    def _kappa(kk):
        """``σ_max/σ_min(kept)`` at retained count ``kk`` — the invariant
        ``rank_criterion`` caps, reported here at the SNAPPED cut so the
        change the snap makes to it is never invisible."""
        if kk <= 0 or not mag or not (mag[0] > 0.0) or not (mag[kk - 1] > 0.0):
            return None
        return mag[0] / mag[kk - 1]

    out = {
        "n_keep": k, "n_keep_snapped": k, "fired": False,
        "gap_rel": float("inf"), "gap_rel_snapped": float("inf"),
        "members": [], "span_rel": 0.0,
        "sigma_max": (mag[0] if mag else 0.0),
        "last_kept": (mag[k - 1] if k > 0 else None),
        "first_dropped": (mag[k] if k < n else None),
        "kappa": _kappa(k), "kappa_snapped": _kappa(k),
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
        # refuses a zero/NaN spectrum, so say nothing rather than snap
        # through an exactly-null block (which is the mesh pad, and is inert
        # by construction).
        out["gap_rel"] = float("inf")
        return out
    out["gap_rel"] = (hi - lo) / scale
    if out["gap_rel"] > float(rtol):
        return out

    # The cut is inside a block.  Walk DOWNWARD to its bottom — outward, the
    # keep-more direction, matching the owner's principle.  The walk is the
    # same successive-difference rule ``band_degeneracy`` documents, with the
    # relative test in place of the absolute one.
    j = k
    while j < n:
        a, b = mag[j - 1], mag[j]
        s = max(a, b)
        if not (s > 0.0) or (a - b) / s > float(rtol):
            break
        j += 1
    # And upward, so the message names the WHOLE block rather than the half
    # of it below the cut.  The upward end never moves the cut (those are
    # already kept); it is reported so the reader can see the block's size.
    i = k - 1
    while i > 0:
        a, b = mag[i - 1], mag[i]
        s = max(a, b)
        if not (s > 0.0) or (a - b) / s > float(rtol):
            break
        i -= 1
    members = mag[i:j]
    out["n_keep_snapped"] = j
    out["fired"] = True
    out["members"] = members
    out["kappa_snapped"] = _kappa(j)
    # The gap at the NEW cut — the guard's own proof that it landed in one.
    # ``inf`` when the snap ran off the bottom of the spectrum, which means
    # the block is still open and the caller must be told so.
    if j < n:
        s2 = max(mag[j - 1], mag[j])
        out["gap_rel_snapped"] = ((mag[j - 1] - mag[j]) / s2
                                  if s2 > 0.0 else float("inf"))
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
    k, k2, mem = info["n_keep"], info["n_keep_snapped"], info["members"]
    # The block occupies sorted positions [k2 - len(mem), k2); the cut at k
    # keeps the ones above it and drops the ones below.
    kept_in_block = k - (k2 - len(mem))
    floor = ""
    if rcond is not None:
        nf = degeneracy_noise_rtol(rcond)
        floor = (f"\n  - noise floor for this cut: eps/rcond={nf:.2e}, against "
                 f"rtol {info['rtol']:.1e}."
                 + ("  ** THE TOLERANCE IS BELOW THE FLOOR: symmetry-degenerate "
                    "values are not resolvable at this rcond, so a CLEAN "
                    "verdict from this guard would mean little. **"
                    if float(info["rtol"]) < nf else ""))
    if info["kappa"] and info["kappa_snapped"]:
        kap = (f", so kappa_eff moves {info['kappa']:.4e} -> "
               f"{info['kappa_snapped']:.4e} — NOT an R19-scale rank inflation "
               f"(common/rank_criterion)")
    else:
        kap = " (kappa_eff is undefined here — the retained block reaches zero)"
    still_open = ("" if info["gap_rel_snapped"] != float("inf")
                  or info["n_keep_snapped"] < info["n_total"] else
                  "\n  - ** THE BLOCK IS STILL OPEN AT THE BOTTOM OF THE "
                  "SPECTRUM: every remaining value is inside it, so there is "
                  "no closed cut to snap to and the retained span is the whole "
                  "spectrum. **")
    return (
        f"[spectral-closure] {where}: the rank cut at {k} of {info['n_total']} "
        f"falls INSIDE a degenerate block.\n"
        f"  - the block holds {len(mem)} values [{_fmt_block(mem)}]; the "
        f"relative gap AT the cut is {info['gap_rel']:.3e} <= rtol "
        f"{info['rtol']:.1e}, so the cut keeps {kept_in_block} of them and "
        f"drops {k2 - k}.\n"
        f"  - keeping part of a degenerate block retains a symmetry-ARBITRARY "
        f"slice of an eigenspace: the retained span is a different subspace at "
        f"Sq than at q, chosen by round-off, and the k-star identity fails for "
        f"everything built on it.\n"
        f"  - snapping OUTWARD to {k2} closes the block (relative gap at the "
        f"new cut {info['gap_rel_snapped']:.3e}).  The block spans "
        f"{info['span_rel']:.3e} relative{kap}." + still_open + floor)


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
                         where="spectral cut", rcond=None, log=print):
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
    """
    m = resolve_mode(mode)
    if m == "off":
        # ``rtol = -1`` can never fire (a relative gap is >= 0), so ``off``
        # still returns the FULL dict — the numbers stay available to a log
        # line even when the guard is disarmed, and no caller has to special-
        # case a short dict.
        return int(n_keep), cluster_at_cut(values, n_keep, rtol=-1.0)
    info = cluster_at_cut(values, n_keep, rtol=rtol)
    if not info["fired"]:
        return int(info["n_keep"]), info
    msg = _message(info, where=where, rcond=rcond)
    if m == "strict":
        raise SpectralClusterError(
            msg + f"\n  Fix: keep {info['n_keep_snapped']} instead of "
                  f"{info['n_keep']}, or run with LORRAX_SPECTRAL_CLOSURE=snap "
                  f"to move the cut automatically, or =off to cut through the "
                  f"block deliberately.")
    for line in msg.splitlines():
        log(f"*** {line}")
    log(f"*** SNAPPED OUTWARD: retained rank {info['n_keep']} -> "
        f"{info['n_keep_snapped']}. ***")
    return int(info["n_keep_snapped"]), info


# ---------------------------------------------------------------------------
# The device face — pure jnp, batched, jit- and vmap-safe
# ---------------------------------------------------------------------------

def snap_keep_outward(values, keep, *, rtol=DEFAULT_RTOL):
    """Move a keep-mask outward past any degenerate block it cuts.  Pure ``jnp``.

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

    Returns
    -------
    ``(keep_out, n_keep, n_keep_out)``
        ``keep_out`` in the INPUT order; the two counts are ``(...)``-shaped
        int arrays so a caller can print or compare them per q.

    NO DATA-DEPENDENT TRIP COUNT.  The walk that the host face writes as a
    ``while`` is here a prefix-AND over adjacency links: sort descending by
    magnitude, mark each adjacent pair "linked" when it passes the relative
    test, force every link inside the kept prefix to linked (they are
    already retained, so traversing them is vacuous), and take the running
    AND.  Position ``i`` is retained exactly when every link from the top
    down to it is unbroken — which for ``i`` past the cut means "the block
    has not ended yet".  One sort, one cumprod, both static-shape.
    """
    import jax.numpy as jnp

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
    # construction) — never "linked", so the walk stops rather than snapping
    # the whole null tail in.
    linked = (scale > 0.0) & ((hi - lo) <= float(rtol) * scale)
    # Link m joins sorted positions m and m+1.  It is vacuous when m+1 is
    # already inside the kept prefix.
    pos = jnp.arange(n - 1)
    vacuous = (pos + 1) < n_keep[..., None]
    chain = linked | vacuous
    # ``reach[i]`` = AND of links 0..i-1 = "position i is reachable from the
    # top through unbroken links".  cumprod in the value dtype keeps this in
    # one kernel; the comparison back to bool is exact for 0/1.
    acc = jnp.cumprod(chain.astype(jnp.int32), axis=-1)
    reach = jnp.concatenate(
        [jnp.ones(acc.shape[:-1] + (1,), dtype=jnp.int32), acc], axis=-1) > 0
    keep_s_out = keep_s | reach
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
        f"{int(n_keep_out)} under snap.")


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
            f"  Fix: LORRAX_SPECTRAL_CLOSURE=snap moves each cut outward past "
            f"its block (the change to kappa_eff is bounded by the block's "
            f"relative span, typically under 1e-4); =off cuts through "
            f"deliberately.")
    for f in found:
        log(f"*** [spectral-closure] {f} ***")
