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
#: a tuple the config parser validates
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
    #: The ``PreUnfoldCapture`` for the tensor this decision is about, bound
    #: by :meth:`with_capture` at the writer.  ``None`` on the ``full`` arm
    #: and until the writer takes it — a decision and the array it applies to
    #: are two different things, and pairing them at the writer is what lets
    #: the writer REFUSE the combination "ibz, and nothing captured" instead
    #: of quietly slicing the unfolded tensor.
    capture: object | None = None

    def with_capture(self, capture) -> "RestartQStorage":
        """This decision, bound to the pre-unfold block it describes.

        AND DEMOTED TO ``full`` WHEN THE CAPTURED WEDGE IS THE WHOLE BZ.
        This is where the owner's ruling of 2026-08-08 ~13:20 becomes
        arithmetic instead of a flag: "if symmetries are not to be used, the
        wavefunction file should've been generated with no symmetries."

        A WFN with ``ntran = 1`` takes ``SymMaps``' trivial branch, where
        ``irr_idx_q`` is the identity and the q axis does not reduce.  The
        centroid set of such a deck can still be orbit-closed — trivially,
        under the identity op, and ``hbn_cohsex_debug`` is a REAL example
        whose 12-op point group was recovered from the charge density while
        its WFN stores one op — so the closure question answers "yes" and the
        q-grid resolution reaches ``use_ibz``.  Nothing is wrong with any of
        that; there is simply no wedge, because ``n_q_ibz == n_q_full``.

        Storing "the wedge" there would write the full BZ under a q_irr stamp
        and a table group, which is bytes and metadata for a reduction that
        did not happen — and the format's own table validation would label
        the file ``"full"`` from its SHAPE anyway, so the writer and the
        reader would be describing the same file two different ways.
        Demoting here makes them agree BY CONSTRUCTION: the writer applies
        the reader's own test.
        A no-symmetry deck's restart file is then byte-for-byte the file it
        has always been, with no attrs and no table group, which is what
        makes its red twin an equality rather than a tolerance.

        ``capture is None`` is NOT demoted — that is a plumbing failure, not
        a structural fact, and the writer refuses it by name.
        """
        if capture is not None:
            # Files keep the canonical centroid order; the producer's block
            # is in the run's packed order.  Convert once, here, at the seam.
            capture = capture.canonical()
        if capture is None or self.mode != "ibz":
            return dataclasses.replace(self, capture=capture)
        import numpy as np
        n_q_full = int(np.asarray(capture.irr_idx_q).shape[0])
        n_q_ibz = int(np.asarray(capture.q_irr_frac).shape[0])
        if n_q_ibz < n_q_full:
            return dataclasses.replace(self, capture=capture)
        return dataclasses.replace(
            self, mode="full", capture=capture,
            reason=(f"the q axis does not reduce ({n_q_ibz} IBZ q of "
                    f"{n_q_full} full-BZ q), so there is no wedge to store "
                    f"— the WFN carries no symmetry that reduces this grid"))

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
    wedge to store, and ``symmetry_maps.validate_qirr_tables`` would label
    that file ``"full"`` from its shape anyway.  Storing "the wedge" there
    would be
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


# ---------------------------------------------------------------------------
# The producer capture
# ---------------------------------------------------------------------------
#
# WHY THERE IS A CAPTURE AT ALL.  The design's load-bearing decision is that
# the restart file holds the PRE-UNFOLD IBZ block, so that
# ``unfold(stored) == what the run used`` is an IDENTITY — the same function
# on the same inputs — rather than a property that happens to hold because
# ``sym_idx_q[q_irr_full_idx] == 0`` under an op-selection policy nobody has
# agreed to freeze for this purpose.  That array exists for exactly one
# statement, inside ``v_q_g_flat._compute_V_q_g_flat_one_tile`` and inside
# ``screening``'s W solve, and the writers are three and four call frames
# away in a different module.
#
# WHY NOT THREAD IT THROUGH THE RETURN VALUES.  Both chains return a fixed
# 2-tuple that a dozen call sites and several test suites unpack
# positionally, and the V chain is three layers deep with a separate
# bispinor entry point.  Widening those returns to carry an object that is
# ``None`` on every non-restart run would touch far more code than the
# feature, and every one of those edits is a chance to change the compute
# path — which this feature must not do.
#
# WHAT KEEPS A HAND-OFF SLOT FROM BECOMING A STALE GLOBAL.  Three rules,
# each with a gate:
#
#   * ``take`` REMOVES.  A capture read twice is a second writer silently
#     storing the first one's wedge — the W0 writer storing V's block under
#     W0's name, which reconstructs to plausible, wrong numbers at every q;
#   * a deposit happens on the statement BEFORE the unfold it describes, so
#     the slot never holds an older array than the run's latest compute;
#   * THE WRITER CROSS-CHECKS.  It does not trust the pairing: the tables on
#     the capture are compared against the tables the writer's own
#     resolution holds, and a disagreement REFUSES.  That is what turns "the
#     compute path and the write path agree" from an assumption into a
#     measured fact, and it is the check that would catch a stale capture
#     regardless of how it got there.
#
# :func:`capture_scope` exists on top of that for isolation — a test, or a
# driver that wants a hard boundary, gets a fresh slot and its own teardown.

_SINK: list = [{}]


@dataclasses.dataclass(frozen=True)
class PreUnfoldCapture:
    """The IBZ block a producer held one statement before it unfolded it.

    ``X_ibz`` is the array itself (device-resident, ``P(None,'x','y')``) and
    the five table arrays are THE ONES THE UNFOLD ON THE NEXT LINE USED — not
    a re-derivation, because a table that reconstructs the tensor must be the
    table that deconstructed it.  ``n_rmu_logical`` is the unpadded centroid
    count, the quantity the file stores instead of the device-count-dependent
    padded one (652b731e / SHARDING_RULES §2).

    NO CLOSURE VERDICT TRAVELS HERE.  The writer asks the resolution point
    for it, through the seam, at the moment it decides — and the tables it
    gets back are CROSS-CHECKED against the ones captured here, which turns
    "the compute path and the write path agree" from an assumption into a
    measured fact.  Carrying the verdict on the capture instead would make
    the two agree by construction and check nothing.
    """

    name: str
    X_ibz: object
    n_rmu_logical: int
    q_irr_frac: object
    irr_idx_q: object
    sym_idx_q: object
    sym_perm: object
    L_table: object
    n_sym_spatial: int
    #: The run's packed centroid order the block and its tables are in
    #: (``common.centroid_basis``); ``None`` means canonical already.
    mu_basis: object = None

    def canonical(self) -> "PreUnfoldCapture":
        """This capture in the canonical centroid order the file stores:
        the block unpacked at the I/O seam, its tables un-conjugated, so the
        writer's cross-check still compares the producer's own tables."""
        if self.mu_basis is None:
            return self
        basis = self.mu_basis
        perm, L = basis.unpack_tables(self.sym_perm, self.L_table)
        # The writer strips the pad from the padded tables; give it the
        # canonical suffix-padded form the pre-packed producers always had.
        n_pad = int(basis.n_canonical) - int(basis.n_logical)
        if n_pad:
            import numpy as np
            tail = np.broadcast_to(
                np.arange(basis.n_logical, basis.n_canonical, dtype=perm.dtype),
                (perm.shape[0], n_pad))
            perm = np.concatenate([perm, tail], axis=-1)
            L = np.concatenate(
                [L, np.zeros((L.shape[0], n_pad, 3), dtype=L.dtype)], axis=1)
        return dataclasses.replace(
            self, X_ibz=basis.unpack_operator(self.X_ibz),
            sym_perm=perm, L_table=L, mu_basis=None)

    def tables(self):
        """A ``symmetry_maps.QirrTables`` for this capture.

        THE PADDED TABLES, deliberately, and ``n_rmu_logical`` beside them.
        ``QgridSymmetryResolution.tables()`` returns the UNPADDED pair; the
        producer bakes the μ pad in afterwards and unfolds with the padded
        form, so those are the tables that describe this tensor.  The
        writer strips the pad from both together and asserts the tail
        really was one (652b731e) — which is a free corruption check the
        unpadded tables would have skipped.
        """
        from symmetry_maps import QirrTables
        return QirrTables(
            irr_idx_q=self.irr_idx_q, sym_idx_q=self.sym_idx_q,
            q_irr_frac=self.q_irr_frac, sym_perm=self.sym_perm,
            L_table=self.L_table, n_sym_spatial=int(self.n_sym_spatial))


class _CaptureScope:
    """Context manager: a FRESH hand-off slot, discarded on exit.

    Not the thing that makes the hand-off safe — the cross-check at the
    writer is — but the thing that keeps one test's wedge out of the next
    one's slot, and available to a driver that wants a hard boundary
    around a stage.
    """

    def __enter__(self):
        _SINK.append({})
        return self

    def __exit__(self, *exc):
        _SINK.pop()
        return False


def capture_scope():
    """A fresh capture slot for the duration of a ``with`` block."""
    return _CaptureScope()


def deposit_pre_unfold(name, X_ibz, *, n_rmu_logical,
                       q_irr_frac, irr_idx_q, sym_idx_q, sym_perm, L_table,
                       n_sym_spatial, mu_basis=None):
    """Offer the pre-unfold block for whoever is writing the restart file.

    Called from the unfold sites, which do not know and MUST NOT know
    whether this run persists anything.  Putting the question "am I being
    captured?" inside the compute path is how the closure question ended up
    being asked in three places with three different answers; the producer
    offers, the writer decides.

    Costs one dict store on a run that never writes a restart tensor.
    """
    _SINK[-1][str(name)] = PreUnfoldCapture(
        name=str(name), X_ibz=X_ibz,
        n_rmu_logical=int(n_rmu_logical), q_irr_frac=q_irr_frac,
        irr_idx_q=irr_idx_q, sym_idx_q=sym_idx_q, sym_perm=sym_perm,
        L_table=L_table, n_sym_spatial=int(n_sym_spatial),
        mu_basis=mu_basis)


def take_pre_unfold(name):
    """The capture for ``name``, REMOVED from the scope.  ``None`` if absent.

    Removing is the point: two writers must not both believe they hold the
    wedge, and a second read returning the same object is how the W0 writer
    would end up storing V's block under W0's tables.
    """
    return _SINK[-1].pop(str(name), None)


def peek_pre_unfold(name):
    """Non-destructive read, for assertions and logs only."""
    return _SINK[-1].get(str(name))


def assert_capture_matches(capture, resolution, *, context: str) -> None:
    """REFUSE unless the captured tables are the resolution's own.

    THE CHECK THAT MAKES THE HAND-OFF TRUSTWORTHY.  The capture came out of
    the compute path; the resolution was taken again at the writer.  If they
    describe different centroid sets — a stale slot, a bispinor channel
    crossed, a resolution taken against the wrong ``sym`` — then the file
    would carry tables that do not reconstruct its tensor, and every q would
    come back a permutation of the wrong centroids.  Nothing downstream can
    see that: the shapes agree, the Hermiticity agrees, the spectrum is
    plausible.  It is the same failure shape as the two conjugation bugs,
    reached through the plumbing instead of through the algebra.

    Compared on the UNPADDED tables, because that is the form both sides
    can produce without agreeing about a device count first.
    """
    import numpy as np

    perm_res, L_res = resolution.tables()
    n_log = int(capture.n_rmu_logical)
    perm_cap = np.asarray(capture.sym_perm)[:, :n_log]
    L_cap = np.asarray(capture.L_table)[:, :n_log]
    if not np.array_equal(perm_cap, np.asarray(perm_res)):
        raise ValueError(
            f"{context}: the captured pre-unfold block carries a centroid "
            f"permutation the writer's own q-grid resolution does not "
            f"produce.  The stored wedge and the tables stored beside it "
            f"would describe different centroid sets, and every q would "
            f"reconstruct as a permutation of the wrong centroids — "
            f"silently, because the shapes agree.")
    if not np.array_equal(L_cap, np.asarray(L_res)):
        raise ValueError(
            f"{context}: the captured umklapp wrap table disagrees with the "
            f"writer's resolution.  Dropping or mismatching L is wrong only "
            f"on the q's that wrap, which is the diagonal-preserving, "
            f"off-diagonal-destroying failure shape this area has shipped "
            f"twice.")
    if int(resolution.n_sym_spatial) != int(capture.n_sym_spatial):
        raise ValueError(
            f"{context}: n_sym_spatial {capture.n_sym_spatial} on the "
            f"capture against {resolution.n_sym_spatial} on the resolution; "
            f"the TRS-augmented half would be read at the wrong offset.")


def resolve_restart_q_storage_for_run(config, *, sym, centroid_indices,
                                      fft_grid, print_fn=print,
                                      context="V_q / W0 restart tensors"):
    """The driver's one call: resolve the deck key against THIS run, announce.

    Rank-invariant by construction — a deck key and a centroid file every
    rank reads, with no env override and no probe — so it adds no collective
    and cannot make ranks disagree about the file they are about to write
    together.  That is the same property ``restart_tensor_writes_enabled``
    relies on and it is stated here because a divergence would present as a
    hang inside the collective writer rather than as an error.

    THE RESOLUTION POINT IS ASKED AGAIN HERE, not cached from the compute.
    ``resolve_qgrid_symmetry_tables`` is already called once per V_q pass,
    once per W solve and once per self-consistency iteration — its
    announcement is deduped on the centroid set precisely because the tree
    re-resolves — so one more call is the tree's existing idiom and not a
    new second opinion.  What it buys is that the writer's decision rests on
    a resolution the WRITER took, which is what makes
    :func:`assert_capture_matches` a real check rather than a tautology.
    """
    # ``LORRAX_FORCE_FULL_BZ=1`` BYPASSES THE WEDGE IN COMPUTE, SO THERE IS
    # NO WEDGE TO STORE.  The deposit that offers the pre-unfold block lives
    # inside the IBZ cascade (``v_q_g_flat.py:592``, ``screening.py:479``);
    # the bypass skips that branch entirely, while the closure question this
    # function asks — is the centroid set orbit-closed, does the q grid
    # reduce — is about the DECK and answers "yes" either way.  Resolving
    # ``ibz`` there produces a decision with nothing captured, which
    # ``write_restart_state_to_h5`` refuses by design rather than slicing the
    # unfolded tensor.
    #
    # MEASURED 2026-08-16: with the default at ``auto`` this took
    # ``test_invariance_gates::test_ibz_equals_full_bz`` red — its leg B is a
    # fresh full pipeline under ``LORRAX_FORCE_FULL_BZ=1`` — and it is not a
    # test-only path: any run using the bypass would have died at the writer.
    # The demotion is the same principle as the rest of the key: storage
    # follows what the run actually did.
    requested = str(getattr(config, "restart_q_storage_raw", "auto"))
    from .gw_config import env_bool
    if env_bool("LORRAX_FORCE_FULL_BZ", False) and requested != "ibz":
        requested = "full"
    res = None
    if sym is not None and centroid_indices is not None:
        from .qgrid_symmetry import resolve_qgrid_symmetry_tables
        res = resolve_qgrid_symmetry_tables(
            sym=sym, centroid_indices=centroid_indices, fft_grid=fft_grid,
            context=context)
    decision = resolve_restart_q_storage(requested, res, context=context)
    # ANNOUNCED, ONCE, ON RANK 0 — same discipline as the suppress key.
    # A run whose restart file changed SHAPE and said nothing is a run
    # nobody can tell from one whose centroid set quietly stopped closing.
    import jax
    if jax.process_index() == 0:
        from ffi.gate import announce_once
        announce_once(
            ("restart_q_storage", decision.mode, decision.requested, context),
            f"  [restart_write] {decision.describe()}",
            scope="rank0")
        # THE DEPRECATION, AND ONLY FOR A DECK THAT ASKED.  The key is
        # scheduled for DELETION, not for a better default (owner ruling
        # 2026-08-08 ~13:20), so a deck that names it is building on
        # something with an end date and should be told once.  A deck that
        # does not name it sees nothing: warning on the default path would
        # print this for every deck in the tree, which is how a deprecation
        # notice becomes noise nobody reads — and the default is not what is
        # being deprecated, the KEY is.
        if _deck_named_the_key(config):
            announce_once(
                ("restart_q_storage_deprecated", decision.requested),
                f"\n  *** LORRAX restart_q_storage = "
                f"{decision.requested!r} is set explicitly by this deck, and "
                f"this key is SCHEDULED FOR DELETION (owner ruling "
                f"2026-08-08).\n"
                f"      why:  symmetry never needed a mode switch — the WFN "
                f"file already answers the question.  A deck-level\n"
                f"            tri-state was always a second, weaker source "
                f"of truth about the same fact.\n"
                f"      then: restart storage will simply FOLLOW the WFN's "
                f"own symmetry and both readers will always\n"
                f"            unfold, after which this key goes away "
                f"entirely.  Do not build on it.\n"
                f"      see:  DESIGN_restart_consolidation.md; "
                f"docs/input_reference.md. ***\n",
                scope="rank0")
    return decision


def _deck_named_the_key(config) -> bool:
    """Did the DECK write ``restart_q_storage``, or is this just the default?

    The deprecation above must fire for a deck that pins the key and stay
    silent for one that never mentions it, and those two are
    indistinguishable from the resolved value alone — a deck pinning ``full``
    and a deck saying nothing both arrive here as ``"full"``.  The config
    parser records the raw keys it saw, so the question is asked of that
    rather than inferred from the value.

    Conservative on purpose: a config object that carries no record of its
    own raw keys answers ``False``, because a deprecation notice that fires
    on every run is worse than one that fires on none.
    """
    seen = getattr(config, "raw_input_keys", None)
    if seen is None:
        return False
    try:
        return "restart_q_storage" in seen
    except TypeError:                                    # pragma: no cover
        return False


__all__ = ["RESTART_Q_STORAGE", "RestartQStorage", "closure_for_restart",
           "resolve_restart_q_storage", "PreUnfoldCapture", "capture_scope",
           "deposit_pre_unfold", "take_pre_unfold", "peek_pre_unfold"]
