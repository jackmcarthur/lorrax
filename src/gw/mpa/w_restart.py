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

    poles, finalized      -> STAGE_SIGMA:  skip the fit AND the sweep
    W(omega) only         -> STAGE_FIT:    fit here, skip the sweep
    neither               -> STAGE_BUILD:  nothing on disk to resume from

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
#: Neither is present.  Nothing to resume from.
STAGE_BUILD = "build"

__all__ = [
    "MpaRestartDecision",
    "STAGE_BUILD",
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
class MpaRestartDecision:
    """What this run found in the bundle, and therefore what it must do."""

    path: str
    stage: str
    w_name: str = W_OMEGA_DATASET
    #: ``mpa_store.read_w_header`` for the W(omega) tensor, or ``None``.
    w_header: dict | None = None
    #: ``mpa_store.fit_completion_ledger`` for the poles, or ``None``.
    fit_ledger: dict | None = None

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
) -> MpaRestartDecision:
    """Which stage this bundle lets the run skip.  Refuses a stale one.

    ``identity`` is :func:`wfn_identity` for THIS run.  ``None`` means
    "the caller cannot say", which is not the same as "it matches": the
    check is then skipped and the store's own stamp is still announced,
    so an operator reading the log can see what was resumed.
    """
    if not os.path.exists(path):
        return MpaRestartDecision(path=path, stage=STAGE_BUILD,
                                  w_name=w_name)
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
        _refuse_a_foreign_wfn(w_header.get("provenance", {}) or {},
                              identity, path=path, name=w_name,
                              context=context)
    if _has(path, mpa_store.MPA_FIT_SUFFIX):
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
                                  w_name=w_name, w_header=w_header)
    return MpaRestartDecision(path=path, stage=STAGE_BUILD, w_name=w_name)


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
    if not stamped:
        return
    from gw.gw_init import _same_wfn_file

    same, why = _same_wfn_file(
        stamped, str(identity.get("wfn_file", "") or ""),
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
