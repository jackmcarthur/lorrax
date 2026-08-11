"""Restart for the multipole lane: which stage this run may SKIP.

STAGING LOCATION, same as the rest of ``gw/mpa/`` -- see the package
docstring.

WHAT THIS IS FOR.  ``compute_mode = mpa`` reads its whole screening off
disk: ``screening_requests_for(MPA)`` asks the screening stage for
nothing (``gw/screening.py``), so the frequency-dependent W(z) and the
pole field fitted from it are the run's inputs rather than its outputs.
Until this module the deck's ``mpa_fit_file`` had to arrive already
fitted -- ``mpa_pipeline._resolve_fit_store`` refused anything else --
which meant the only route from a W(z) store to a Sigma was a script
outside the drivers.  That is the gap this closes, and it closes it the
way the restart consolidation ruling asks for: the DECISION is taken in
one place, the driver spends a handful of lines on it, and every
refusal is one the format already owned.

THE FILE IS ONE BUNDLE AND THAT IS WHY NO NEW KEY APPEARS HERE.
``mpa_store.allocate_fit_store`` deletes ``Omega_p``, ``B_p``, every
``fit_*`` and the fit ledger, and nothing else -- so a W(omega) tensor
and the poles fitted from it coexist in ONE h5 file, and the deck's
existing ``mpa_fit_file`` names that file.  A deck therefore says where
its multipole screening lives exactly once, and what it finds there
decides what this run has to do:

    poles, finalized      -> STAGE_SIGMA:   skip the fit AND the sweep
    W(omega) only         -> STAGE_FIT:     fit here, skip the sweep
    zeta only             -> STAGE_BUILD_W: sweep and fit, skip the ISDF fit
    neither               -> STAGE_BUILD:   nothing on disk to resume from

TWO KINDS OF FAILURE, AND THEY DO NOT GET THE SAME ANSWER.  This is the
distinction the ladder is built around, so it is stated before the
mechanics:

* an IDENTITY failure REFUSES.  The declared artifact is not of the
  object this run is about -- a different wavefunction, a pole axis
  nobody can name the unit of, a half-written sweep.  That is "you
  pointed at the wrong bundle", which is an error and not a stage
  decision, and no amount of rebuilding makes it right.  These raise.
* a STALENESS failure DROPS one rung, with the mismatched key named on
  one line.  The declared artifact is of the right object but of an
  older version of it -- a different centroid table, a different rank,
  a different frequency grid.  That is the ordinary developer flow ("I
  changed something"), and the correct response is to rebuild exactly
  the stages downstream of what changed.

The failure shape this must be STRUCTURALLY unable to produce is the
one that nearly shipped on 2026-08-10: a store carrying a head derived
from a different basis, reused because nothing compared the basis.
Hence: never silent reuse, and never silent rebuild.  Every rung that
is skipped says so, and every rung that is dropped says which key
dropped it.

WHO OWNS WHICH CHECK.  This module owns the LADDER -- which rung a run
may start on -- and not the identity of the artifacts on it.  The zeta
stage's identity is ``gw_init._zeta_reuse_ok``'s, which already
compares the source WFN, the centroid table's md5, the band windows,
the cutoffs and the solver knobs; it is CALLED and its verdict handed
here as :class:`ZetaState` rather than restated, because a second
implementation of a provenance rule is a second thing to keep in sync.
That split is also what makes the ladder testable without a filesystem:
the decision is a pure function of the states it is handed.

WHAT IT REFUSES, AND WHOSE REFUSAL EACH ONE IS.  Nothing in this module
re-implements a check the store already makes; it CALLS them, so a
store that would be refused at the fit seam is refused here instead of
after the allocation is spent:

* the sampling grid, its hash, the q_irr tables, the table/centroid
  digests, the shape-vs-attr cross-check and the per-frequency
  readiness ledger -- ``mpa_store.read_w_header``;
* ``screening_content`` -- ``mpa_store.require_correlation_part``, the
  one implementation both consumer seams call (the 130 eV refusal);
* ``energy_unit`` -- ``mpa_store.canonical_energy_unit`` on the store's
  own ``omega_units`` stamp, which is what the fit driver hands to
  ``allocate_fit_store`` as the pole axis' unit;
* q completeness -- the ``data_ready`` ledger, all of it, because a
  half-swept W(omega) fits poles against zeros at the missing
  frequencies and the fit reproduces whatever it is handed;
* WFN identity -- :func:`wfn_identity` below, the one fact the store
  layer cannot know and the only check this module adds.

THE WEDGE IS NOT A CASE HERE.  Storage follows the WFN and readers
always unfold (``DESIGN_restart_consolidation.md``): a wedge W(omega)
reaches the fit as a wedge, the fit driver copies its tables beside the
poles, and ``mpa_store.read_pole_slice(..., unfold=True)`` serves the
full zone at Sigma time.  So no branch in this module or its callers
asks which zone the store is on -- the two arms differ only in
``n_q_on_disk``, which every reader below already derives from the
shape.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from file_io import mpa_store

#: The frequency-resolved W dataset's name inside the bundle.  One
#: spelling, here, because the writer, the fit seam and the resume
#: announcement must agree and a literal in three places does not.
W_OMEGA_DATASET = "W_qmunu_omega"

#: A finalized pole field is present: this run integrates Sigma and
#: neither sweeps nor fits.
STAGE_SIGMA = "sigma"
#: A complete W(omega) tensor is present and the poles are not: the fit
#: runs here, off disk, and the screening sweep is skipped.
STAGE_FIT = "fit"
#: A valid zeta/ISDF fit is present and no W(omega) is: the sweep and
#: the fit both run, but the ISDF fit itself is skipped.  This is the
#: rung the "I only changed the solver" iteration lands on.
STAGE_BUILD_W = "build_w"
#: Neither is present.  Nothing to resume from.
STAGE_BUILD = "build"

__all__ = [
    "MpaRestartDecision",
    "SamplingExpectation",
    "STAGE_BUILD",
    "STAGE_BUILD_W",
    "ZetaState",
    "record_zeta_state",
    "take_zeta_state",
    "STAGE_FIT",
    "STAGE_SIGMA",
    "W_OMEGA_DATASET",
    "fit_from_w_omega",
    "resolve_mpa_restart",
    "wfn_identity",
    "write_w_omega",
]


def wfn_identity(wfn) -> dict:
    """``{'wfn_file': realpath, 'wfn_bytes': size}`` -- the zeta rule.

    THE PRECEDENT IS ``gw_init._zeta_fit_provenance`` AND IT IS FOLLOWED
    RATHER THAN RESTATED: the resolved path plus the size, and NOT the
    mtime.  That file's docstring records why, and the reasoning
    transfers unchanged -- copying or restoring a WFN.h5 changes mtime
    without changing content, and a spurious multi-hour re-sweep is a
    worse outcome than the contrived same-path-same-size-different-bytes
    case.  Comparison re-resolves both sides through
    ``gw_init._same_wfn_file``, so a stamp written under one staging
    spelling is still recognised under another.
    """
    path = getattr(wfn, "_filename", None) or ""
    try:
        nbytes = int(os.path.getsize(path)) if path else -1
    except OSError:
        nbytes = -1
    return {"wfn_file": os.path.realpath(path) if path else "",
            "wfn_bytes": nbytes}


@dataclass(frozen=True)
class ZetaState:
    """What the caller found at the declared zeta/ISDF artifact.

    A VERDICT, NOT A PATH, and that is the point.  The zeta reuse rule
    lives in ``gw_init._zeta_reuse_ok`` and already compares the source
    WFN, the centroid table's md5, the band windows, the cutoffs and the
    solver knobs against this run's inputs.  Handing this module the
    ANSWER rather than the file keeps that rule in one place and keeps
    the ladder a pure function of its inputs, which is what lets the
    drop branches be tested without a filesystem.

    ``reason`` names the key that failed when ``valid`` is False -- it is
    printed verbatim, so it should read like ``centroids_md5`` or
    ``n_rmu``, not like a sentence.
    """

    present: bool = False
    valid: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class SamplingExpectation:
    """This run's frequency grid and the deck keys that feed sampling.

    Compared against the stored W(omega)'s own header.  ``omega`` is
    matched EXACTLY -- ``np.array_equal`` and not a tolerance -- because
    a grid that is nearly the same is a different grid: the poles are
    fitted at these abscissae and a shifted sample is a different
    constraint, not a rounding of the same one.

    ``keys`` is whatever the driver declares feeds the sweep (the q set,
    the line heights, the basis rank).  It is a plain dict so a deck can
    grow a sampling key without this module learning its name; anything
    present on both sides and unequal drops the rung and is named.  A key
    the store never stamped is NOT a mismatch -- silence on either side
    means "this store predates the key", which is a reason to be careful
    and not a reason to rebuild, and the same rule the restart lane
    already settled on for its own stamps.
    """

    omega: object = None
    keys: dict | None = None


#: The zeta verdict, recorded where it is TAKEN and read where it is
#: USED.  These are far apart: ``gw_init`` decides reuse deep inside the
#: ISDF path, and the ladder consumes it at the multipole seam, with six
#: call signatures in between that have no other reason to carry it.
#: Threading it through all of them would be a larger change than the
#: ladder it feeds, so it is recorded here instead -- in the module that
#: owns the ladder, so there is still exactly one owner.
#:
#: Reset at the start of every resolve so a second run in one process
#: cannot inherit the first's verdict; a caller that never records gets
#: ``None``, which is the same "cannot say" the parameter already means.
_RECORDED_ZETA: "ZetaState | None" = None


def record_zeta_state(present, valid, reason=None):
    """Called by ``gw_init`` at the moment it decides zeta reuse."""
    global _RECORDED_ZETA
    _RECORDED_ZETA = ZetaState(present=bool(present), valid=bool(valid),
                               reason=reason)
    return _RECORDED_ZETA


def take_zeta_state():
    """Read and clear the recorded verdict.  Clearing is the point.

    A stale verdict is exactly the failure this ladder exists to prevent,
    so the recorder is single-use: a second seam in the same process that
    did not run an ISDF stage gets ``None`` ("cannot say") rather than
    the previous run's answer.
    """
    global _RECORDED_ZETA
    out, _RECORDED_ZETA = _RECORDED_ZETA, None
    return out


def _sampling_mismatch(expect, w_header):
    """The first key on which this run and the stored sweep disagree."""
    if expect is None or w_header is None:
        return None
    want = expect.omega
    if want is not None:
        got = w_header.get("omega")
        if got is None:
            return None
        a = np.asarray(want, dtype=np.complex128).reshape(-1)
        b = np.asarray(got, dtype=np.complex128).reshape(-1)
        if a.shape != b.shape:
            return (f"omega (grid has {b.size} points, this run asks for "
                    f"{a.size})")
        if not np.array_equal(a, b):
            i = int(np.argmax(np.abs(a - b)))
            return (f"omega (first difference at index {i}: stored "
                    f"{b[i]!r}, this run {a[i]!r})")
    prov = dict(w_header.get("provenance", {}) or {})
    for key, mine in sorted((expect.keys or {}).items()):
        if mine is None or key not in prov or prov[key] is None:
            continue
        theirs = prov[key]
        same = (np.array_equal(np.asarray(mine), np.asarray(theirs))
                if isinstance(mine, (list, tuple, np.ndarray))
                else str(mine) == str(theirs))
        if not same:
            return f"{key} (stored {theirs!r}, this run {mine!r})"
    return None


@dataclass(frozen=True)
class MpaRestartDecision:
    """What this run found in the bundle, and therefore what it must do."""

    path: str
    stage: str
    w_name: str = W_OMEGA_DATASET
    #: ``mpa_store.read_w_header`` for the W(omega) tensor, or ``None``.
    w_header: dict | None = None
    #: ``mpa_store.fit_completion_ledger`` for the poles, or ``None``.
    fit_ledger: dict | None = None
    #: One line per rung this run was NOT allowed to start on, each
    #: naming the key that dropped it.  Empty when nothing was stale.
    #: Carried rather than only printed so a test can assert on the
    #: reason instead of scraping a log.
    drops: tuple = ()

    def announce_drops(self, print_fn=print) -> tuple:
        """Say which rungs were refused and why, before the skip line.

        A drop is the half of this decision that is invisible in the
        result: the run simply rebuilds, which looks exactly like a run
        that had nothing to resume.  So each one is stated, with its
        key, in the order the ladder considered them.
        """
        for reason in self.drops:
            print_fn(f"  [mpa restart] NOT resuming: {reason}")
        return self.drops

    def announce(self, print_fn=print) -> str:
        """One rank-0 line naming the skip and the store's provenance.

        A stage that is skipped is invisible in every other way -- the
        run is simply faster -- which is exactly the shape of a restart
        that quietly resumed the wrong file.  So it says which file,
        which grid, which screening object and which WFN.
        """
        line = describe(self)
        print_fn(line)
        return line


def describe(decision: MpaRestartDecision) -> str:
    """The announcement text; separated so a test can read it."""
    base = os.path.basename(decision.path)
    if decision.stage == STAGE_BUILD:
        return (f"  [mpa_restart] {base}: no W(omega) tensor and no pole "
                f"field — nothing to resume from.")
    if decision.stage == STAGE_BUILD_W:
        return (f"  [mpa_restart] {base}: RESUMING from the zeta/ISDF fit "
                f"— the ISDF fit is SKIPPED and the screening sweep (chi0 "
                f"+ Dyson at every z) and the pole fit both run here.")
    hdr = decision.w_header or {}
    prov = hdr.get("provenance", {}) or {}
    wfn = prov.get("wfn_file", "?")
    grid = (f"n_omega={hdr.get('n_omega', '?')} "
            f"grid_hash={str(hdr.get('grid_hash', '?'))[:12]} "
            f"{hdr.get('omega_units', '?')} "
            f"{hdr.get('screening_content', '?')} "
            f"q_storage={hdr.get('q_storage', '?')}")
    if decision.stage == STAGE_FIT:
        return (f"  [mpa_restart] {base}: RESUMING from its W(omega) "
                f"tensor — the screening sweep (chi0 + Dyson at every "
                f"z) is SKIPPED and the pole fit runs here.  {grid} "
                f"wfn={wfn}")
    led = decision.fit_ledger or {}
    # THE W(omega) CLAUSE IS OMITTED WHEN THERE IS NO W(omega), rather
    # than printed with '?' in every slot.  A production fit store is
    # routinely a file of its own -- the samples it was fitted from are
    # 20 GB and get deleted -- and a line of question marks reads as a
    # store that failed to describe itself instead of one that has
    # nothing left to describe.  The digests it was fitted AGAINST are
    # stamped in the fit store itself and refused at the Sigma seam.
    tail = f"; W(omega) {grid} wfn={wfn}" if decision.w_header else (
        "; its W(omega) samples are not in this bundle")
    return (f"  [mpa_restart] {base}: RESUMING from its pole field — the "
            f"screening sweep AND the fit are SKIPPED.  n_p="
            f"{led.get('n_p', '?')} n_q={led.get('n_q', '?')} "
            f"{led.get('energy_unit', '?')} "
            f"{led.get('screening_content', '?')}{tail}")


def _has(path: str, name: str) -> bool:
    """Is ``name`` a member of this h5 file?  Missing file -> False."""
    import h5py

    if not os.path.exists(path):
        return False
    with h5py.File(path, "r") as f:
        return name in f


def resolve_mpa_restart(
    path: str,
    *,
    w_name: str = W_OMEGA_DATASET,
    identity: dict | None = None,
    context: str = "compute_mode = mpa",
    zeta: "ZetaState | None" = None,
    sampling: "SamplingExpectation | None" = None,
) -> MpaRestartDecision:
    """Which stage this bundle lets the run skip.  Refuses a stale one.

    ``identity`` is :func:`wfn_identity` for THIS run.  ``None`` means
    "the caller cannot say", which is not the same as "it matches": the
    check is then skipped and the store's own stamp is still announced,
    so an operator reading the log can see what was resumed.

    ``zeta`` is :class:`ZetaState` -- the verdict of the zeta reuse rule
    the caller already ran.  ``sampling`` is this run's
    :class:`SamplingExpectation`.  Both default to ``None``, meaning
    "the caller cannot say", and both then leave the ladder exactly as
    it was before they existed: this function is a strict extension, so
    a driver that has not been taught the new rungs keeps its old
    behaviour rather than silently gaining a skip.

    The rungs are considered from the DEEPEST down, and the first one
    whose staleness check fails drops to the next with its key recorded
    in :attr:`MpaRestartDecision.drops`.  Identity failures do not drop
    -- they raise, from inside the calls this delegates to.
    """
    drops: list = []
    stale_sampling = False
    def _floor(reasons):
        """The rung to land on when no W(omega) may be resumed."""
        if zeta is None or not zeta.present:
            return MpaRestartDecision(path=path, stage=STAGE_BUILD,
                                      w_name=w_name, drops=tuple(reasons))
        if not zeta.valid:
            return MpaRestartDecision(
                path=path, stage=STAGE_BUILD, w_name=w_name,
                drops=tuple(reasons) + (
                    f"the zeta/ISDF fit, because {zeta.reason} changed; "
                    f"refitting it",))
        return MpaRestartDecision(path=path, stage=STAGE_BUILD_W,
                                  w_name=w_name, drops=tuple(reasons))

    if not os.path.exists(path):
        return _floor(drops)
    w_header = None
    if _has(path, w_name):
        # EVERY FORMAT REFUSAL RUNS HERE, before any stage is skipped:
        # grid hash, table/centroid digests, the shape-vs-attr q_storage
        # cross-check and the per-frequency ledger are all inside this
        # one call, which is why it is not wrapped in a try.
        w_header = mpa_store.read_w_header(path, w_name)
        mpa_store.require_correlation_part(
            w_header.get("screening_content"),
            where=f"{context}: resuming from {os.path.basename(path)}",
            source=f"{path} :: {w_name}")
        mpa_store.canonical_energy_unit(
            w_header.get("omega_units"),
            where=f"{context}: {os.path.basename(path)} :: {w_name}")
        ready = np.asarray(w_header["data_ready"], dtype=bool)
        if not bool(ready.all()):
            missing = [int(i) for i in np.flatnonzero(~ready)]
            raise ValueError(
                f"{context}: {path} :: {w_name} has "
                f"{int(ready.sum())} of {ready.size} frequency slabs "
                f"ready; slabs {missing} are allocated and unwritten.  "
                f"An unwritten slab reads back as zeros of exactly the "
                f"right shape, so a fit resumed against this file would "
                f"fit poles to zero screening at those z and report a "
                f"backward error of 0 for doing it.  Finish the sweep, "
                f"or re-produce the store.")
        # IDENTITY FIRST, AND IT RAISES.  A foreign WFN is not a stale
        # sweep to be redone, it is the wrong bundle: a different WFN
        # with the same centroid count yields a numerically different W
        # that no hash inside the file can distinguish, so there is no
        # safe rung below this one to fall back to.
        _refuse_a_foreign_wfn(w_header.get("provenance", {}) or {},
                              identity, path=path, name=w_name,
                              context=context)
        # STALENESS SECOND, AND IT DROPS.  The sweep is of the right
        # object but of a different sampling of it.
        stale = _sampling_mismatch(sampling, w_header)
        if stale is not None:
            drops.append(f"the W(omega) sweep in {os.path.basename(path)}, "
                         f"because {stale}; re-sweeping")
            w_header = None
            stale_sampling = True
    # A POLE FIELD FITTED FROM A SWEEP THIS RUN CALLS STALE IS STALE
    # TOO, and it is worth saying why this is not obvious from the
    # ladder's shape.  The rungs are otherwise independent -- a bundle
    # may legitimately carry poles and no samples, because the samples
    # are tens of gigabytes and get deleted -- so "poles present" does
    # not normally have to ask anything about W(omega).  It does have to
    # when the W(omega) is BOTH present AND of a different sampling than
    # this run wants: those poles were fitted at abscissae the run is
    # not asking for, and resuming them would answer a question nobody
    # asked while looking exactly like a successful skip.
    if _has(path, mpa_store.MPA_FIT_SUFFIX) and not stale_sampling:
        led = mpa_store.fit_completion_ledger(path)
        if bool(led.get("complete", False)):
            # THE POLE FIELD'S OWN DECLARATIONS, ASKED AT THE DRIVER.
            # The Sigma pass runs both of these itself and would refuse
            # such a store anyway -- but only after ``read_head_poles``
            # and the pass loop have been reached, and only if this
            # bundle's poles get that far.  MEASURED, 2026-08-10: eight
            # of the 48 fit stores on this machine
            # (``si_mpa_0808/_reports/big/mpa_fit*.h5`` and
            # ``mpa_closer_0809``'s) predate the declarations and carry
            # energy_unit=None with screening_content=None, which is a
            # pole axis nobody can say the unit of and a fit nobody can
            # say the object of.  Refusing them here costs a driver
            # nothing and names the file.
            mpa_store.require_correlation_part(
                led.get("screening_content"),
                where=f"{context}: resuming the pole field in "
                      f"{os.path.basename(path)}",
                source=path)
            mpa_store.canonical_energy_unit(
                led.get("energy_unit"),
                where=f"{context}: the pole axis of "
                      f"{os.path.basename(path)}")
            return MpaRestartDecision(path=path, stage=STAGE_SIGMA,
                                      w_name=w_name, w_header=w_header,
                                      fit_ledger=led)
    if w_header is not None:
        return MpaRestartDecision(path=path, stage=STAGE_FIT,
                                  w_name=w_name, w_header=w_header,
                                  drops=tuple(drops))
    return _floor(drops)


def write_w_omega(
    path,
    W_of_z,
    *,
    tables,
    omega,
    sampling,
    closure_verdict,
    identity,
    name: str = W_OMEGA_DATASET,
    omega_line=None,
    screening_content="W_c",
    provenance=None,
):
    """Write a whole W(z) sweep into the bundle, declarations REQUIRED.

    THE ONE SEAM A PRODUCER CALLS, and the reason it exists rather than
    letting each producer call ``allocate_w_omega`` itself is the
    declaration set.  ``allocate_w_omega`` makes ``screening_content``
    optional on purpose -- a producer that has not decided yet may
    allocate first -- and every store that reached a consumer
    undeclared cost the 130 eV in ``require_correlation_part``'s
    refusal.  A driver knows the answer at write time, so this wrapper
    takes it as a REQUIRED argument and stamps the run's WFN identity
    beside it in the same call.  ``identity`` is :func:`wfn_identity`;
    it is what :func:`resolve_mpa_restart` compares on the way back in,
    and a store written without it can be resumed by a run using
    different wavefunctions with nothing able to see it.

    ``W_of_z`` is ``(n_omega, n_q_on_disk, N_mu, N_mu)`` at the LOGICAL
    mu extent -- the wedge when the producer computed on one, the full
    BZ otherwise, per the storage-follows-the-WFN ruling.  No branch
    here asks which: the extent is the array's own.
    """
    if not identity or not str(identity.get("wfn_file", "") or ""):
        raise ValueError(
            "write_w_omega: identity= must carry the run's WFN "
            "(w_restart.wfn_identity(wfn)).  Which wavefunctions a "
            "W(z) sweep came from is the ONE fact the store cannot "
            "hash for itself, so a store written without it can be "
            "resumed by a run using different orbitals and every "
            "digest inside the file will still agree.")
    arr = np.asarray(W_of_z)
    if arr.ndim != 4:
        raise ValueError(
            f"write_w_omega: W_of_z is {arr.shape} (rank {arr.ndim}); "
            f"the frequency axis LEADS, so the write shape is "
            f"(n_omega, n_q, N_mu, N_mu).")
    prov = dict(provenance or {})
    prov.update(identity)
    mpa_store.allocate_w_omega(
        path, name,
        n_omega=int(arr.shape[0]), n_q_on_disk=int(arr.shape[1]),
        n_mu=int(arr.shape[3]), tables=tables, omega=omega,
        sampling=sampling, omega_line=omega_line,
        closure_verdict=closure_verdict,
        screening_content=screening_content, provenance=prov,
        dtype=arr.dtype)
    for i in range(int(arr.shape[0])):
        mpa_store.write_w_slab(path, name, i, arr[i], ready=True)
    return mpa_store.read_w_header(path, name)


def fit_from_w_omega(
    decision: MpaRestartDecision,
    *,
    print_fn=print,
    tile_bytes=None,
    provenance=None,
    report_stream=None,
):
    """Run the pole fit against the resumed W(z), in this process.

    ``n_p`` AND THE ABSCISSAE COME OFF THE STORE, not off a deck key,
    and that is the whole reason no knob appears for either.  The
    sampling protocol stamped beside the tensor says how many poles the
    grid was built to determine (``2*n_p`` samples for ``n_p`` poles,
    ``sample_plan.mpa_plan``), and the abscissae are the file's own
    ``omega`` -- which ``run_fit_driver`` then re-asserts bit-equal
    against what it is handed, so this cannot drift into fitting
    against a re-derived grid.
    """
    from gw.mpa import fit_driver

    hdr = decision.w_header
    if hdr is None:
        raise ValueError(
            "fit_from_w_omega: this decision carries no W(omega) header, "
            "so there is nothing to fit.")
    n_p = int(hdr["sampling"]["n_p"])
    # THE POLES GO INTO THE SAME FILE, AND THAT IS SAID OUT LOUD BEFORE
    # A BYTE MOVES.  The bundle is one file by design, so fitting a
    # resumed W(omega) MUTATES the file the deck named — and the
    # campaign's own W(z) stores are 20 GB artifacts the fleet treats
    # as read-only inputs (``mpa_wcprod_0809/stores/``).  An operator
    # who points ``mpa_fit_file`` at one of those is asking for a fit
    # they probably meant to write elsewhere, so a store this process
    # cannot write is a REFUSAL naming the file rather than an
    # h5py permission traceback three stages in, and a store it CAN
    # write is announced as being written to.
    if not os.access(decision.path, os.W_OK):
        raise PermissionError(
            f"fit_from_w_omega: {decision.path} holds the W(omega) "
            f"samples but is not writable by this process, and the "
            f"poles are written INTO the bundle beside them.  Copy the "
            f"store to the run's own directory and point mpa_fit_file "
            f"there, or fit it once and name the finished bundle.")
    print_fn(
        f"  [mpa_restart] fitting {n_p} poles per element from "
        f"{hdr['n_omega']} stored samples of "
        f"{os.path.basename(decision.path)} :: {decision.w_name} — the "
        f"screening sweep does not run.  The poles are written INTO "
        f"this file, beside the samples they were fitted from.")
    return fit_driver.run_fit_driver(
        decision.path, decision.w_name, decision.path,
        np.asarray(hdr["omega"], dtype=np.complex128), n_p,
        tile_bytes=tile_bytes,
        provenance=dict(provenance or {},
                        **(hdr.get("provenance", {}) or {})),
        report_stream=report_stream)


def _refuse_a_foreign_wfn(prov, identity, *, path, name, context):
    """The ONE check the store layer cannot make for itself.

    Everything else about a W(omega) store is self-describing -- the
    grid hashes itself, the tables digest themselves, the ledger counts
    itself.  Which wavefunctions it was swept from is not: the store
    holds (mu, nu) blocks on a centroid set, and a different WFN with
    the same centroid count produces a numerically different W that no
    hash inside the file can distinguish.  So the writer stamps the
    resolved path and the size (:func:`wfn_identity`) and this compares
    them, through ``gw_init._same_wfn_file`` -- the same two-step
    realpath/inode rule the zeta reuse check applies, called rather than
    copied, so the two cannot disagree about what "the same WFN" means.
    """
    if not identity:
        return
    stamped = str(prov.get("wfn_file", "") or "")
    mine = str(identity.get("wfn_file", "") or "")
    # SILENCE ON EITHER SIDE IS "CANNOT PROVE ANYTHING", NOT "DIFFERENT",
    # and both directions have a real case.  The store's side is the one
    # that matters today: every W(z) store the campaign produced before
    # this stamp existed carries ``producer`` / ``route`` /
    # ``source_store`` / ``why`` and no ``wfn_file`` -- measured, all
    # seven resumable ones on Perlmutter -- and refusing those would
    # make this seam's arrival delete the fleet's resumable inventory.
    # This run's side: ``wfn._filename`` is absent for a caller holding
    # a bundle it built rather than loaded, and ``_same_wfn_file``
    # answers False on an empty path, so without this line such a caller
    # would be refused for a fact nobody asserted.  Only a store that
    # CLAIMS a WFN can contradict a run that CLAIMS one.
    if not stamped or not mine:
        return
    from gw.gw_init import _same_wfn_file

    same, why = _same_wfn_file(
        stamped, mine,
        old_bytes=prov.get("wfn_bytes"),
        new_bytes=identity.get("wfn_bytes"))
    if not same:
        raise ValueError(
            f"{context}: {path} :: {name} was swept from a DIFFERENT "
            f"wavefunction file than this run is using, so its W(omega) "
            f"is not this run's screening.  {why}  Nothing inside the "
            f"store can catch this — the grid, the tables and the "
            f"centroid digest would all still agree — which is why the "
            f"writer stamps the WFN and this seam compares it.  Fix: "
            f"point mpa_fit_file at the bundle swept from this WFN, or "
            f"re-sweep.")
