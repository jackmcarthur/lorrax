"""``restart_q_storage``: the deck's q-storage decision, taken in ONE place.

WHAT THIS MODULE IS.  The deck key ``restart_q_storage = auto | full | ibz``
and the two functions that turn it into an answer: :func:`closure_for_restart`
— THE SEAM the writer gates on — and :func:`resolve_restart_q_storage`, which
combines the seam's answer with the deck's request and the run's actual q path
and returns a :class:`RestartQStorage` naming the mode and the reason.

WHY A SEAM RATHER THAN A CALL.  The owner's stamp ruling
(DESIGN_symmetry_restart_followup.md, "The stamp architecture", third point:
there is ONE predicate) makes closure a GENERATION-TIME guarantee, verified
where the set is born and carried as a stamp; runtime geometric re-derivation
demotes to the legacy/debug arm.  Phase 2.5 has not landed, so today the
answer comes from the geometric measurement the resolution point already took
— and the whole point of routing it through one named function is that the
swap is ONE LINE in ONE place when the stamp exists.  See
:func:`closure_for_restart` for the exact line.

WHY THE RESOLUTION IS ONE FUNCTION AND NOT A PREDICATE PER TENSOR.  V and W0
must resolve TOGETHER.  ``tagged_arrays.write_restart_state_to_h5`` sizes the
W0 placeholder from ``V_qmunu.shape``, so V's storage decision silently
becomes W0's — a coupling that is fine while both are full-BZ and a trap the
moment they need not be (the rule is written at that site, dbe3b4ec item 3).
One resolution, passed to both writers, is what makes the coupling a decision
somebody took rather than one inherited from an argument's shape.

WHAT ``auto`` COSTS, AND WHO IT MOVES.  ``auto`` is the default and it flips a
deck onto wedge storage exactly when that deck's centroid set is orbit-closed
AND the run's q path already reduced to the IBZ — the two conditions under
which the unfold is the identity the frontier measured.  On a non-closed set
(today's Si production 960-centroid deck: 47 of 48 ops violating) it resolves
``full`` and the bytes are what they were.  It DOES move ``si_bse_debug``,
whose centroid set became orbit-closed at fb046e0c; that deck's on-disk
restart format changes under the default, which is why it is an explicit arm
of the Perlmutter validation script rather than something noticed later.
"""
from __future__ import annotations

import dataclasses

#: Legal values of the ``restart_q_storage`` input key.  Modelled on
#: ``file_io.kin_ion.HARTREE_SOURCES``: a tuple the config parser validates
#: against at PARSE time, so a typo dies on the deck rather than 20 minutes
#: into a 40-node run.
#:
#:   auto  — follow the closure answer: the wedge when the centroid set is
#:           closed and the run's q path reduced, the full BZ otherwise.
#:   full  — preserve today's bytes EXACTLY.  The escape hatch, and the
#:           control arm of the A/B: it does not ask the closure question
#:           at all, so it cannot be changed by the answer.
#:   ibz   — REFUSE unless the wedge is genuinely storable.  For a deck that
#:           believes it is closed and wants to be told when it stops being.
RESTART_Q_STORAGE = ("auto", "full", "ibz")


@dataclasses.dataclass(frozen=True)
class RestartQStorage:
    """The resolved q-storage decision for ONE restart file's tensors.

    ``mode`` is ``"ibz"`` or ``"full"`` and never ``"auto"`` — a record of a
    decision, not of a request; ``requested`` keeps the request beside it so
    a log line or a stamp can say which of the two produced the mode.

    ``resolution`` is the ``QgridSymmetryResolution`` the decision was made
    on, or ``None`` when the deck said ``full`` and the question was never
    asked.  It carries the closure verdict AND the unfold tables, which is
    why it is held rather than a bool: the writer needs both, and a
    re-derivation to get the second is how two answers to one question get
    born.
    """

    mode: str
    requested: str
    resolution: object | None
    reason: str

    @property
    def store_wedge(self) -> bool:
        """The one predicate the writers branch on."""
        return self.mode == "ibz"

    def describe(self) -> str:
        return (f"restart_q_storage={self.requested} -> {self.mode}"
                + (f" ({self.reason})" if self.reason else ""))


def closure_for_restart(resolution):
    """THE SEAM: is the centroid set behind ``resolution`` orbit-closed?

    ONE function, asked by the restart writer and by nothing else, so that
    the owner's stamp ruling lands as a one-line edit instead of a search.

    TODAY: the answer is the GEOMETRIC verdict the resolution point already
    took — ``symmetry_maps.verify_centroid_orbit_closure``, run once per
    centroid set inside ``resolve_qgrid_symmetry`` and carried on
    ``QgridSymmetryResolution.verdict``.  It is re-read here, never
    re-measured: a second measurement of one question is a second answer
    waiting to disagree with the first.

    AFTER PHASE 2.5: the stamped centroid file carries the verdict and its
    hash, and this function reads the STAMP instead — the body below becomes
    ``return resolution.stamp.verdict`` (or the legacy geometric verdict when
    the set is unstamped, which is the transitional arm the design doc's
    fourth point describes).  Nothing else in the restart path changes,
    because nothing else in the restart path asks this question.

    Returns the ``CentroidClosureVerdict``, not a bool, because the refusal
    arm has to name the offending ops and their residuals and a bool cannot.
    """
    # ---- THE ONE LINE THE STAMP SWAP REPLACES -------------------------
    return resolution.verdict


def resolve_restart_q_storage(requested, resolution, *, context: str):
    """Turn the deck key + the closure answer into ONE storage decision.

    Parameters
    ----------
    requested
        The deck's ``restart_q_storage`` value, already normalised by the
        config parser to one of :data:`RESTART_Q_STORAGE`.
    resolution
        The ``QgridSymmetryResolution`` for the centroid set these tensors
        were computed against, or ``None`` when the run never resolved one
        (no symmetry information available at all).  ``full`` is the only
        answer that survives ``None``, and ``ibz`` refuses on it.
    context
        Names the caller in the refusal, e.g. ``"V_q / W0 restart tensors"``.

    Returns
    -------
    RestartQStorage

    Notes
    -----
    THE TWO CONDITIONS, and why ``use_ibz`` is not implied by ``closed``.
    Closure says the μ permutation α EXISTS.  ``use_ibz`` says the run
    ACTUALLY computed on a wedge — a deck whose q-grid the group does not
    reduce (ntran=1, or a Γ-only grid) has a closed centroid set and no
    wedge to store, and ``qirr_store._validate`` would label that file
    ``"full"`` from its shape anyway.  Storing "the wedge" there would be
    storing the full BZ under a different attr, which is the
    shape-versus-attr disagreement the format refuses on read.  So both are
    required, and the ``ibz`` refusal distinguishes them by name.
    """
    req = str(requested).strip().lower()
    if req not in RESTART_Q_STORAGE:
        raise ValueError(
            f"restart_q_storage={requested!r} is not one of "
            f"{RESTART_Q_STORAGE}.  This should have been caught at parse "
            f"time; see gw_config.")

    # ``full`` NEVER ASKS.  That is what makes it the control arm of the
    # A/B: an arm whose bytes could be moved by the closure answer is not a
    # control.  It is also the arm a deck sets when it wants today's file
    # regardless of what the centroid set does next week.
    if req == "full":
        return RestartQStorage(
            mode="full", requested=req, resolution=None,
            reason="the deck asked for full-BZ storage")

    if resolution is None:
        if req == "ibz":
            raise ValueError(
                f"{context}: restart_q_storage=ibz, but this run resolved no "
                f"q-grid symmetry at all — there is no centroid permutation "
                f"and no wedge, so there is nothing to store.  Use auto (which "
                f"falls back to full) or full.")
        return RestartQStorage(
            mode="full", requested=req, resolution=None,
            reason="no q-grid symmetry resolution was taken on this run")

    verdict = closure_for_restart(resolution)
    closed = bool(verdict.closed)
    use_ibz = bool(resolution.use_ibz)

    if req == "ibz":
        # THE REFUSAL ARM.  Loud, and it distinguishes the two ways to fail
        # because they need different fixes: a non-closed set needs
        # regeneration, a non-reduced q path needs nothing at all.
        if not closed:
            raise ValueError(
                f"{context}: restart_q_storage=ibz REFUSES this centroid "
                f"set.  {verdict.describe()}\n"
                f"A wedge stored against a non-closed set has no permutation "
                f"to invert with and is silently unrecoverable.  Regenerate "
                f"the centroid set orbit-closed, or set "
                f"restart_q_storage=auto (which falls back to full and says "
                f"so) or full.")
        if not use_ibz:
            raise ValueError(
                f"{context}: restart_q_storage=ibz, and the centroid set IS "
                f"orbit-closed, but this run's q path did not reduce to a "
                f"wedge ({resolution.reason or 'no reduction available'}).  "
                f"There is no IBZ block to store — the tensor's q axis "
                f"already is the full BZ.  Set restart_q_storage=auto or "
                f"full.")
        return RestartQStorage(
            mode="ibz", requested=req, resolution=resolution,
            reason="the deck asked for wedge storage and the set is closed")

    # ``auto``.
    if closed and use_ibz:
        return RestartQStorage(
            mode="ibz", requested=req, resolution=resolution,
            reason=(f"centroid set is orbit-closed (worst residual "
                    f"{verdict.worst_residual:.3e} at tol {verdict.tol:.1e}) "
                    f"and the q path reduced"))
    if not closed:
        why = (f"centroid set is NOT orbit-closed "
               f"({verdict.n_violating}/{verdict.n_sym} ops violating, "
               f"worst {verdict.worst_residual:.3e})")
    else:
        why = ("the q path did not reduce to a wedge"
               + (f" ({resolution.reason})" if resolution.reason else ""))
    return RestartQStorage(
        mode="full", requested=req, resolution=None, reason=why)


__all__ = ["RESTART_Q_STORAGE", "RestartQStorage", "closure_for_restart",
           "resolve_restart_q_storage"]
