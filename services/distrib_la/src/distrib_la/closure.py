"""Degeneracy-safe RANK CUTS for this service's rank-revealing operations.

A rank-revealing factorization answers "how many directions are worth
keeping?" by cutting a spectrum: the pivot values of a pivoted Cholesky,
``|diag(U)|`` of an LU, the eigenvalues of an ``eigh``.  Every one of those
cuts has the same hazard, and it is not a numerical one.

**A cut that stops INSIDE a degenerate cluster keeps a symmetry-arbitrary
slice of an eigenspace.**  A symmetry that commutes with the factored
operator maps each of its eigenspaces onto itself and mixes the members of
a degenerate block freely, so cutting between whole blocks leaves the
retained span invariant and cutting through one leaves a span that differs
between ``q`` and ``Sq``, chosen by round-off.  Everything built on it
loses the star identity.  This module finds the block straddling a
proposed cut and moves the cut OUTWARD past it — the keep-more direction —
loudly, or refuses.

WHY THERE IS A COPY OF THIS HERE
--------------------------------
The criterion is ``common/spectral_closure`` in the LORRAX monorepo, which
is where it was derived and where its tolerance was bracketed by
measurement.  **This service may not import it.**  Services are
import-isolated from ``src/`` by charter and by gate
(``tests/test_distrib_la_import_isolation.py``: ``import distrib_la`` drags
in NO lorrax package, asserted through ``sys.modules`` AND ``sys.path``),
and the whole worth of that isolation is that it has no exceptions.  The
kernel is ~90 lines of stdlib arithmetic, so a copy is cheap and an import
edge is not.

**The copy is not allowed to drift, and that is a gate rather than a
promise.**  ``tests/test_spectral_closure.py`` in the monorepo — the side
that CAN see both — carries a consistency cell that runs the shared
synthetic spectra through ``common.spectral_closure.cluster_at_cut`` and
through :func:`cluster_at_cut` here and asserts the dicts agree field for
field.  It also pins the ONE difference below, so that difference stays
deliberate instead of becoming the first drift.

THE ONE DIFFERENCE: THE DEFAULT IS ``off`` HERE, AND ``snap`` THERE
-------------------------------------------------------------------
``common/spectral_closure`` defaults to ``snap``, on an argument about
arithmetic: a snapped spectral cut admits directions within ``rtol`` of
ones already retained, so the achieved amplification moves by under one
part in 10⁴ and refusing that by default would be refusing the repair.
That argument is about the criterion and it transfers here unchanged.

What does not transfer is who is allowed to change a route.  This
service's resolution and route semantics are CERTIFIED SURFACE: a resolved
backend name is a promise that every guard passed, and the worst measured
defect in this tree was a silent route change that ran to completion with
``rc=0`` and a QP gap of −161 eV.  A guard that arrives switched on would
change the rank a caller gets back from an operation it already ships,
without that caller asking, which is the same shape of event.  So this
round the guard is **opt-in and OFF by default**, at every entry point,
and turning it on is one kwarg or one environment variable.

**Owner row: whether the default here should follow its sibling to
``snap`` is the owner's call, exactly as the strict-vs-snap row already
open on the monorepo guard is.  It is a single constant,
:data:`DEFAULT_MODE`.**

THE TOLERANCE IS RELATIVE-TO-NEIGHBOUR
--------------------------------------
Two spectral values are "in the same block" when they agree relative to
EACH OTHER::

    same block  <=>  |s_i - s_j| <= rtol_deg * max(|s_i|, |s_j|)

not relative to ``s_max``.  Rank cuts sit eight to ten decades below the
top of the spectrum, and a tolerance measured against ``s_max`` would
declare the whole retained tail one enormous cluster.  Scale-free is
therefore equally sharp at every depth.  :data:`DEFAULT_RTOL` is 1e-6,
bracketed from both sides by measurements recorded in the monorepo module
and in ``tests/known_failures/2026-08-10-spectral-cut-closure.md``; it is
not re-derived here, because a second derivation is a second thing to
drift.

WHAT THIS MODULE IS NOT
-----------------------
It does not decide HOW MANY directions to keep.  A caller's ``rcond`` (or
its own criterion) sizes the retained set; this module only refuses to let
that size land mid-block, and every answer it returns is at least as large
as the one it was given.  It also does not read the environment: the NAME
of the dial lives here once, and the entry points that have a driver
around them do the ``os.environ`` lookup and hand the answer to
:func:`resolve_mode`.  That is ``common/spectral_closure``'s own rule and
it is the same rule this service's ``resolve.py`` states as ZERO
ENVIRONMENT READS.

HOST FACE ONLY, DELIBERATELY
----------------------------
The monorepo module has a second, pure-``jnp`` face because the zeta fit's
cut lives inside a jitted kernel that never brings its eigenvalues to
host.  This service has no such site: :func:`distrib_la.plan.plan` is
eager by construction, a rank decision here is made after a factorization
has returned, and the spectrum is a length-``n`` vector.  So there is one
face, it is host, and ``strict`` raises where it is called — no deferred
refusal, no host callback, nothing to keep in sync.  If a device-side rank
cut ever appears in this service, the ``snap_keep_outward`` half is what
gets copied next, and the consistency cell is where it gets pinned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_RTOL",
    "DEFAULT_MODE",
    "MODES",
    "MODE_ENV",
    "RANK_SPECTRA",
    "SpectralClusterError",
    "RankCut",
    "degeneracy_noise_rtol",
    "cluster_at_cut",
    "resolve_spectral_cut",
    "describe_clean",
    "resolve_mode",
    "mode_from_env",
    "rank_cut",
    "cholesky_pivot_spectrum",
    "lu_rank_spectrum",
]

#: f64 unit round-off.  Spelled out rather than imported so this module
#: stays numpy-optional at import time — the vocabulary and the criterion
#: must be readable on a machine with no jax and no ``.so``, exactly as
#: ``BACKEND_CHOICES`` is.
_EPS64 = 2.220446049250313e-16

#: Default "same degenerate block" tolerance, RELATIVE TO THE NEIGHBOUR.
#: 1e-6.  Identical to ``common.spectral_closure.DEFAULT_RTOL`` and pinned
#: equal to it by the consistency cell — the criterion is shared even
#: though the code is copied.
DEFAULT_RTOL: float = 1.0e-6

#: The three modes, in the order the docstring introduces them.
MODES = ("snap", "strict", "off")

#: THE ONE PLACE THE DEFAULT IS DECIDED, and the one thing about this
#: kernel that deliberately differs from its monorepo sibling.  ``off``:
#: this service's route semantics are certified surface and a guard that
#: arrives switched on would change a shipped operation's answer without
#: the caller asking.  See "THE ONE DIFFERENCE" in the module docstring.
#: OWNER ROW to flip it to ``snap``.
DEFAULT_MODE: str = "off"

#: NAME of the environment dial the entry points read.  THIS MODULE DOES
#: NOT READ IT.
#:
#: It is deliberately NOT ``LORRAX_SPECTRAL_CLOSURE``, the monorepo
#: guard's dial.  Sharing that name would mean a run that armed the zeta
#: fit's guard silently armed this service's too — the caller would get a
#: different rank out of ``cholesky`` because of a variable it set for an
#: unrelated seam.  Two guards, two dials, and the reason is that one of
#: them defaults on and the other defaults off.
MODE_ENV = "LORRAX_DISTRIB_LA_CLOSURE"

#: What the rank-revealing spectrum IS, per op — the sentence a message
#: uses so a reader knows which numbers were cut.  ``eigh`` is here
#: because the criterion is the same one and refusing it would be
#: arbitrary; ``cholesky`` and ``solve_lu`` are the two the feature was
#: asked for.
RANK_SPECTRA = {
    "cholesky": "Cholesky pivot values |diag(L)|^2",
    "solve_lu": "|diag(U)| of the LU factor",
    "eigh": "eigenvalues",
}


class SpectralClusterError(RuntimeError):
    """A rank cut falls inside a degenerate block, in ``strict`` mode.

    ``RuntimeError``, not a service-specific base: this service's contract
    is that its exceptions are ``ValueError`` / ``RuntimeError``, each
    constructible from one string, because ``bandstructure/bse_setup.py``
    re-raises ``type(exc)(_why)`` and a class with a different constructor
    would delete the message it is re-raising.
    """


def resolve_mode(explicit=None) -> str:
    """Validate a mode and supply :data:`DEFAULT_MODE` when there is none.

    A mis-spelled mode RAISES rather than falling back to ``off``: a guard
    silently disarmed by a typo is worse than no guard, because the log
    then shows a clean run.  ``ValueError``, per the service's exception
    contract.
    """
    if explicit is None:
        return DEFAULT_MODE
    mode = str(explicit).strip().lower()
    if mode not in MODES:
        raise ValueError(
            f"{MODE_ENV}={mode!r} is not one of {MODES}.  A mis-spelled "
            f"guard mode is not silently 'off'.")
    return mode


def mode_from_env(environ=None) -> str | None:
    """The mode this environment asks for, or ``None`` when it asks for none.

    THE ONE ``os.environ`` READ IN THIS PACKAGE for the closure, and it is
    here rather than in a caller so the variable's name is looked up in
    exactly one place.  It is still not read at import time and not read
    by the criterion: the entry points call it and pass the answer to
    :func:`resolve_mode`, so every function below stays a function of its
    arguments.  Returns ``None`` for an unset OR empty variable —
    ``FOO=`` is how a shell says "not set" and reading it as a mode would
    raise on a blank.
    """
    if environ is None:
        import os
        environ = os.environ
    raw = environ.get(MODE_ENV)
    if raw is None or not str(raw).strip():
        return None
    return str(raw)


def degeneracy_noise_rtol(rcond) -> float:
    """``eps/rcond`` — the finest RELATIVE agreement degenerate values can show.

    A factorization returns its spectrum with backward error
    ``O(eps * s_max)``, so two values equal by symmetry agree only to
    ``eps * s_max`` in ABSOLUTE terms; at a cut where ``s ~ s_max * rcond``
    that is ``eps/rcond`` RELATIVE.  A degeneracy tolerance below this line
    cannot resolve a symmetry multiplet from two unrelated neighbours and
    will report every cut clean.  Printed beside the tolerance in every
    report; a REFERENCE LINE, never the tolerance itself.
    """
    rc = float(rcond)
    if not (rc > 0.0):
        return float("inf")
    return _EPS64 / rc


def _desc_mag(values):
    """``(magnitudes descending, original indices)`` as plain Python lists.

    Cuts are on ``|s|``: Cholesky pivots are PSD and descending, an
    indefinite eigenspectrum is cut on magnitude with both signs physical.
    Sorting by magnitude normalises all of them.

    REAL input by contract, and that is why this is ``float(v)`` and not a
    complex ``abs``: :func:`lu_rank_spectrum` and
    :func:`cholesky_pivot_spectrum` both return magnitudes, and an
    eigenvalue spectrum is real.  Keeping the body byte-identical to
    ``common.spectral_closure._desc_mag`` is what makes the consistency
    cell a check on the CRITERION rather than on two dialects of it.
    """
    pairs = sorted(((abs(float(v)), i) for i, v in enumerate(values)),
                   key=lambda p: -p[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def cluster_at_cut(values, n_keep, *, rtol=DEFAULT_RTOL):
    """The degenerate block straddling a cut, if there is one.

    Parameters
    ----------
    values : iterable
        The spectrum the cut is made in.  Any order; magnitudes are used.
    n_keep : int
        Proposed retained count — the largest ``n_keep`` by magnitude,
        which is what every relative-threshold cut selects.
    rtol : float
        Two neighbours are in the same block when they agree to this,
        RELATIVE to the larger of them.

    Returns
    -------
    dict
        ``n_keep``          the proposal, clamped to the spectrum
        ``n_keep_snapped``  the outward-snapped count (== ``n_keep`` when
                            the cut already falls in a gap)
        ``fired``           whether the cut fell inside a block
        ``gap_rel``         relative gap AT the proposed cut (``inf`` when
                            the cut is at an end and slices nothing)
        ``members``         the block's magnitudes, descending
        ``span_rel``        ``(max - min)/max`` over the snapped extension
        ``sigma_max``       largest magnitude in the spectrum
        ``last_kept`` / ``first_dropped`` magnitudes either side of the cut
        ``kappa`` / ``kappa_snapped`` ``s_max/s_min(kept)`` before / after

    FIELD-FOR-FIELD IDENTICAL to
    ``common.spectral_closure.cluster_at_cut``, and the monorepo
    consistency cell asserts exactly that.  Nothing here looks for a knee,
    a plateau or an elbow: it asks one question at one boundary.
    """
    mag, _ = _desc_mag(values)
    n = len(mag)
    k = max(0, min(int(n_keep), n))

    def _kappa(kk):
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
    # A cut at either end slices nothing.
    if k <= 0 or k >= n:
        return out
    hi, lo = mag[k - 1], mag[k]
    scale = max(hi, lo)
    if not (scale > 0.0):
        # An all-zero tail either side of the cut.  A RELATIVE test is
        # meaningless on it, and an exactly-null block is the pad, which
        # is inert by construction — say nothing rather than snap through
        # it.  (In this service the pad is the identity/zero fill a
        # non-dividing extent gets before it reaches a distributed
        # factorization, and swallowing it would make the retained rank a
        # function of the DEVICE COUNT.)
        out["gap_rel"] = float("inf")
        return out
    out["gap_rel"] = (hi - lo) / scale
    if out["gap_rel"] > float(rtol):
        return out

    # The cut is inside a block.  Walk DOWNWARD to its bottom — outward,
    # the keep-more direction.
    j = k
    while j < n:
        a, b = mag[j - 1], mag[j]
        s = max(a, b)
        if not (s > 0.0) or (a - b) / s > float(rtol):
            break
        j += 1
    # And upward, so the message names the WHOLE block rather than the
    # half of it below the cut.  The upward end never moves the cut.
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
    # The gap at the NEW cut — the guard's own proof that it landed in
    # one.  ``inf`` when the snap ran off the bottom of the spectrum,
    # which means the block is still open and the caller must be told.
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
    kept_in_block = k - (k2 - len(mem))
    floor = ""
    if rcond is not None:
        nf = degeneracy_noise_rtol(rcond)
        floor = (f"\n  - noise floor for this cut: eps/rcond={nf:.2e}, "
                 f"against rtol {info['rtol']:.1e}."
                 + ("  ** THE TOLERANCE IS BELOW THE FLOOR: degenerate "
                    "values are not resolvable at this rcond, so a CLEAN "
                    "verdict from this guard would mean little. **"
                    if float(info["rtol"]) < nf else ""))
    if info["kappa"] and info["kappa_snapped"]:
        kap = (f", so kappa_eff moves {info['kappa']:.4e} -> "
               f"{info['kappa_snapped']:.4e}")
    else:
        kap = " (kappa_eff is undefined here — the retained block reaches zero)"
    still_open = ("" if info["gap_rel_snapped"] != float("inf")
                  or info["n_keep_snapped"] < info["n_total"] else
                  "\n  - ** THE BLOCK IS STILL OPEN AT THE BOTTOM OF THE "
                  "SPECTRUM: every remaining value is inside it, so there "
                  "is no closed cut to snap to and the retained span is "
                  "the whole spectrum. **")
    return (
        f"[distrib_la-closure] {where}: the rank cut at {k} of "
        f"{info['n_total']} falls INSIDE a degenerate block.\n"
        f"  - the block holds {len(mem)} values [{_fmt_block(mem)}]; the "
        f"relative gap AT the cut is {info['gap_rel']:.3e} <= rtol "
        f"{info['rtol']:.1e}, so the cut keeps {kept_in_block} of them and "
        f"drops {k2 - k}.\n"
        f"  - keeping part of a degenerate block retains a "
        f"symmetry-ARBITRARY slice of an eigenspace: the retained span is "
        f"a different subspace at Sq than at q, chosen by round-off, and "
        f"the k-star identity fails for everything built on it.\n"
        f"  - snapping OUTWARD to {k2} closes the block (relative gap at "
        f"the new cut {info['gap_rel_snapped']:.3e}).  The block spans "
        f"{info['span_rel']:.3e} relative{kap}." + still_open + floor)


def describe_clean(info, *, where):
    """The one line an entry point prints when the guard did NOT fire.

    "No news" and "a good number" must not look alike: a site that only
    speaks when it fires leaves a reader unable to tell a checked cut from
    an unchecked one.
    """
    if info["gap_rel"] == float("inf"):
        return (f"    [distrib_la-closure] {where}: cut at {info['n_keep']} "
                f"of {info['n_total']} slices nothing (it is at an end of "
                f"the spectrum) — exempt.")
    return (f"    [distrib_la-closure] {where}: cut at {info['n_keep']} of "
            f"{info['n_total']} falls in a gap — relative gap "
            f"{info['gap_rel']:.3e} against rtol {info['rtol']:.1e} "
            f"({info['gap_rel'] / max(info['rtol'], 1e-300):.3g}x the "
            f"tolerance).  No degenerate block is cut.")


def resolve_spectral_cut(values, n_keep, *, rtol=DEFAULT_RTOL, mode=None,
                         where="rank cut", rcond=None, log=print):
    """Return a retained count whose cut does not slice a degenerate block.

    Returns ``(n_keep_out, info)`` with ``info`` exactly
    :func:`cluster_at_cut`'s dict, so a caller can log the numbers whether
    or not the guard fired.  ``mode`` defaults to :data:`DEFAULT_MODE`
    (``off`` here); ``strict`` raises :class:`SpectralClusterError`;
    ``off`` returns the proposal untouched and does not even look.

    IDENTICAL to ``common.spectral_closure.resolve_spectral_cut`` for any
    EXPLICIT mode, which is what the consistency cell asserts.  With
    ``mode=None`` the two differ, deliberately and only in the default —
    the cell pins that too.
    """
    m = resolve_mode(mode)
    if m == "off":
        # ``rtol = -1`` can never fire (a relative gap is >= 0), so ``off``
        # still returns the FULL dict — the numbers stay available to a
        # log line even when the guard is disarmed, and no caller has to
        # special-case a short dict.
        return int(n_keep), cluster_at_cut(values, n_keep, rtol=-1.0)
    info = cluster_at_cut(values, n_keep, rtol=rtol)
    if not info["fired"]:
        return int(info["n_keep"]), info
    msg = _message(info, where=where, rcond=rcond)
    if m == "strict":
        raise SpectralClusterError(
            msg + f"\n  Fix: keep {info['n_keep_snapped']} instead of "
                  f"{info['n_keep']}, or run with {MODE_ENV}=snap to move "
                  f"the cut automatically, or {MODE_ENV}=off to cut "
                  f"through the block deliberately.")
    for line in msg.splitlines():
        log(f"*** {line}")
    log(f"*** SNAPPED OUTWARD: retained rank {info['n_keep']} -> "
        f"{info['n_keep_snapped']}. ***")
    return int(info["n_keep_snapped"]), info


# ---------------------------------------------------------------------------
# The rank-revealing surface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankCut:
    """A rank decision that has been through the closure guard.

    Attributes
    ----------
    op
        Which op's spectrum was cut (``cholesky`` / ``solve_lu`` /
        ``eigh``), so a log line can name what the numbers were.
    mode, rtol
        The guard that ran.  ``mode='off'`` means it did not look, and
        :attr:`n_keep` is then exactly :attr:`n_keep_requested`.
    n_keep_requested
        What the ``rcond`` threshold (or the caller) proposed.
    n_keep
        What to actually keep.  **Always >= n_keep_requested** — this
        guard only ever moves a cut outward.
    fired
        Whether the proposed cut fell inside a degenerate block.
    info
        :func:`cluster_at_cut`'s full dict, always present, including
        under ``off`` — "no news" and "a good number" must not look alike.

    The dataclass is frozen for the same reason :class:`FactorToken` is: a
    rank decision that a later frame can edit is a decision nobody owns.
    """

    op: str
    mode: str
    rtol: float
    n_keep_requested: int
    n_keep: int
    fired: bool
    info: dict = field(repr=False)

    @property
    def moved(self) -> int:
        """How many directions the snap added.  ``0`` when it did not fire."""
        return int(self.n_keep) - int(self.n_keep_requested)

    def __repr__(self) -> str:
        return (f"RankCut({self.op}, mode={self.mode}, "
                f"n_keep={self.n_keep_requested}"
                + (f"->{self.n_keep}" if self.moved else "")
                + f" of {self.info.get('n_total', '?')})")

    def describe(self) -> str:
        """One line for a run banner: armed or not, fired or not."""
        if self.mode == "off":
            return (f"    [distrib_la-closure] {self.op}: DISARMED "
                    f"(mode=off) — rank {self.n_keep} was not checked "
                    f"against {RANK_SPECTRA.get(self.op, 'the spectrum')}.  "
                    f"Set {MODE_ENV}=snap|strict, or pass closure= to the "
                    f"plan, to arm it.")
        if not self.fired:
            return describe_clean(self.info, where=self.op)
        return (f"    [distrib_la-closure] {self.op}: ARMED "
                f"(mode={self.mode}, rtol={self.rtol:.1e}) and FIRED — "
                f"rank {self.n_keep_requested} -> {self.n_keep}.")


def _threshold_rank(values, rcond) -> int:
    """``#{i : |v_i| > |v|_max * rcond}`` — the relative-threshold cut.

    The same rule every rank-revealing truncation in this tree uses, and
    the reason this function exists rather than each caller writing it: a
    cut derived one way and closed another way is two criteria, and the
    guard's promise (``n_keep`` is at least what you asked for) is only
    checkable against a proposal it can see.
    """
    mag, _ = _desc_mag(values)
    if not mag or not (mag[0] > 0.0):
        return 0
    cut = mag[0] * float(rcond)
    k = 0
    for v in mag:
        if v > cut:
            k += 1
        else:
            break
    return k


def rank_cut(op: str, values, *, n_keep=None, rcond=None, closure=None,
             rtol=None, where=None, log=print) -> RankCut:
    """Decide a rank in ``values``, with the degeneracy-closure guard.

    THE one place in this package where a rank cut is made, and the reason
    the guard is a feature rather than a caller's responsibility: a cut
    made at three call sites is three places for the closure to be
    forgotten at.

    Parameters
    ----------
    op
        ``'cholesky'``, ``'solve_lu'`` or ``'eigh'`` — what the spectrum
        came out of.  Naming it buys the message; it does not change the
        arithmetic, because the criterion is the same at all three.
    values
        The rank-revealing spectrum: :func:`cholesky_pivot_spectrum`'s
        output, :func:`lu_rank_spectrum`'s, or an eigenvalue vector.  Any
        order; magnitudes are used.
    n_keep, rcond
        The proposal, exactly one of them.  ``rcond`` applies the relative
        threshold ``|v| > |v|_max * rcond``; ``n_keep`` states the count
        directly.
    closure
        ``'snap'`` / ``'strict'`` / ``'off'``, or ``None`` to take the
        environment (:data:`MODE_ENV`) and then :data:`DEFAULT_MODE`,
        which is ``off``.
    rtol
        Block tolerance; :data:`DEFAULT_RTOL` when ``None``.
    where, log
        Message prefix and sink for the loud ``snap`` report.

    Returns
    -------
    :class:`RankCut`

    Raises
    ------
    ValueError
        Unknown ``op``; neither or both of ``n_keep``/``rcond``; a
        mis-spelled mode.
    SpectralClusterError
        Under ``strict``, when the cut falls inside a degenerate block.
    """
    if op not in RANK_SPECTRA:
        raise ValueError(
            f"rank_cut: unknown op {op!r} "
            f"(known: {'|'.join(sorted(RANK_SPECTRA))})")
    if (n_keep is None) == (rcond is None):
        raise ValueError(
            f"rank_cut({op!r}): pass exactly one of n_keep= or rcond=; got "
            f"n_keep={n_keep!r}, rcond={rcond!r}.  A cut needs a proposal "
            f"the guard can be at least as large as, and two proposals is "
            f"two criteria.")
    mode = resolve_mode(mode_from_env() if closure is None else closure)
    tol = DEFAULT_RTOL if rtol is None else float(rtol)
    seq = list(values)
    proposed = (_threshold_rank(seq, rcond) if n_keep is None else int(n_keep))
    label = where or f"{op} rank cut ({RANK_SPECTRA[op]})"
    out, info = resolve_spectral_cut(seq, proposed, rtol=tol, mode=mode,
                                     where=label, rcond=rcond, log=log)
    return RankCut(op=op, mode=mode, rtol=tol, n_keep_requested=proposed,
                   n_keep=int(out), fired=bool(info["fired"]), info=info)


# ---------------------------------------------------------------------------
# What to cut: the rank-revealing spectrum of each factorization
# ---------------------------------------------------------------------------
# Both helpers are pure ``jnp`` on the LAST TWO AXES, so one call serves an
# ``(n, n)`` tile and an ``(nb, n, n)`` stack alike, and either can sit
# inside a caller's own ``jit``.  jax is imported inside the body, not at
# module scope, so this module stays importable with no jax on the machine
# — the same property ``BACKEND_CHOICES`` has and for the same reason.

def cholesky_pivot_spectrum(L):
    """``|diag(L)|**2`` — the pivot values a pivoted Cholesky would take.

    For ``A = L L^H`` the quantity a pivoted Cholesky compares against its
    tolerance at step ``i`` is the Schur-complement diagonal, and with the
    pivot order fixed that is exactly ``|L_ii|^2``.  So the rank-revealing
    spectrum of ANY Cholesky route in this package — ``native``,
    ``native2d``, cuSOLVERMp's batched potrf materialised through
    ``cholesky_handle_to_natural_L``, SLATE's through
    ``SlateLowerL.to_jax_lower`` — is one expression, and the backends can
    then be checked against each other on the cut rather than on a
    per-backend convention.

    SQUARED, not ``|L_ii|``: the pivot lives on the operator's scale, the
    diagonal on its square root, and a RELATIVE-gap criterion is not
    invariant under the square root — two pivots agreeing to 1e-6 have
    diagonals agreeing to 5e-7, so cutting the wrong one of the two moves
    every block boundary by a factor of two in tolerance.  Cut the
    operator's spectrum.
    """
    import jax.numpy as jnp
    d = jnp.diagonal(jnp.asarray(L), axis1=-2, axis2=-1)
    return jnp.abs(d) ** 2


def lu_rank_spectrum(LU):
    """``|diag(U)|`` from a packed LU factor.

    ``getrf`` writes ``U`` on and above the diagonal and ``L``'s strict
    lower part below it, with ``L``'s diagonal an implicit 1, so the
    diagonal of the packed factor IS ``diag(U)`` and no unpacking is
    needed.  ``|U_ii|`` is the standard LU rank-revealing quantity; it is
    NOT squared, because unlike Cholesky's ``L`` it already lives on the
    operator's scale (``A = P L U``, not ``U U^H``).

    THE CAVEAT, stated where it is used rather than in a report nobody
    opens: an unpivoted or partially-pivoted LU is a weaker rank revealer
    than a rank-revealing QR or an SVD, and ``|diag(U)|`` can miss a
    near-null space that a column-pivoted factorization would expose
    (Kahan's matrix is the classical counterexample).  The closure guard
    does not change that and does not claim to: it guarantees the cut you
    make is not inside a degenerate cluster, not that the spectrum you cut
    is the right one to be reading.  Where that distinction matters, cut
    an ``eigh`` spectrum instead.
    """
    import jax.numpy as jnp
    d = jnp.diagonal(jnp.asarray(LU), axis1=-2, axis2=-1)
    return jnp.abs(d)
