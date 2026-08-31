"""GW-side stage helper for ``screening_diagrams = w_bse`` / ``w_rpa_resolvent``.

WHAT THIS MODULE IS.  The other half of the fork in
:func:`gw.screening.compute_screening_model`.  ``w_rpa`` reaches today's
path; ``w_bse`` and ``w_rpa_resolvent`` both reach here, through the SAME
entry point (:func:`compute_screening_ladder`) with ``include_w`` set to
whether the operator carries the ladder's direct rung.  Under ``w_bse``
the W that Sigma consumes is the LADDER-corrected screened Coulomb::

    W(z) - v = v (z - H)^-1 v ,    H carrying the statically screened
                                   direct rung -W(0) in its kernel

Under ``w_rpa_resolvent`` it is the SAME resolvent identity with the rung
parameterized OUT of the ring matvec (``include_w=False`` — see
``bse.bse_ring_comm.build_bse_ring_matvec_full``), i.e. ``H = H_RPA``: a
second route to ``w_rpa``'s own W, useful precisely because it shares
every other moving part with the ladder (restart handoff, TRS/finite-q
machinery it does not need but is built alongside, head resolvent) and so
gates that shared machinery on an operator simple enough to have an
independent reference.  Both are evaluated at exactly the frequencies the
active Sigma ansatz already asks for -- the SAME
:func:`gw.screening.screening_requests_for` plan the RPA path uses.
Nothing about the role set, the Sigma dispatch or any Sigma kernel
changes; only which W satisfies the request does.

THE STAGE IS TWO-STAGE BY CONSTRUCTION, NOT BY ACCIDENT.  The ladder
kernel's ``W_R`` IS the RPA ``W(0)``.  So this module

  1. runs the ordinary RPA static leg (``compute_static_w(role="static")``),
  2. persists it into the ISDF restart file (``persist_w0_and_head``) and
     VERIFIES the file is loadable before anything reads it,
  3. hands that restart path to the BSE ladder facade
     (``bse.w_ladder.compute_wc_qwedge``), which returns head-less
     ``W(z) - v`` bodies on the q wedge,
  4. adds ``v`` back, unfolds the wedge to the full BZ through the same
     ``symmetry_maps.unfold_isdf_operator`` service call the RPA path
     uses, with q/-q rows paired through that service's explicit TRS half
     (the v1 ladder has a TRS gauge but not arbitrary spatial-gauge
     covariance), gates the result, and returns ``{role: W_q}`` in exactly
     the shape and sharding the RPA path returns.

PERSIST-BEFORE-LOAD IS THE ORDERING CONSTRAINT OF THE WHOLE FEATURE, and
it is the reason this helper exists at all rather than a flag on
``compute_screening``.  It is also why the driver's own W0 flush is
suppressed on this branch (``gw.screening.driver_persists_w0``): by the
time the driver reaches its persist call the ``"static"`` role holds the
LADDER W, and stamping that into ``W0_qmunu`` would put a different
operator under a dataset every downstream consumer reads as the RPA
static W.

THE q=0 HEAD RIDES THE SAME RESOLVENT.  The singular macroscopic slot is not
inside ``W_q``: the body facade therefore solves three dipole right-hand
sides on the identical full-window, full-2N operator and returns ``Xi_ab``.
That tensor is already micro-reducible because the finite-G Hartree/ring term
is in the operator.  ``head_correction=full`` restores only the omitted
macroscopic bare-v channel through the mini-BZ reduction; applying the RPA
``Y W Z`` fold to it again is explicitly forbidden as double counting.
``no_local_fields`` retains the direct diagnostic and ``off`` removes the
special Gamma cell.  The separate finite-q ``mc_average_placement`` policy
remains refused under w_bse; a non-None ``head_channel`` reaching here is a
parse-gate bug, not a case to serve.

INSULATORS ONLY, AND THE GATE READS THE MEAN FIELD.
The ladder, its TRS-gauge machinery and every certification behind this
feature assume integer occupations and a gapped D.  The driver infers the
material class from the loaded WFN; this stage independently refuses partial
occupations before any compute (:func:`refuse_fractional_occupations`).
``w_rpa_resolvent`` carries the same occupation gate under rule id
``w_rpa_resolvent_insulators_only``.  Its pair basis uses the identical
band-index cut at ``nelec`` whether or not the rung is in the operator, so
the argument survives even though the ladder's TRS-gauge fix is not applied
on this arm.

WHY THE ``import bse`` CALLS ARE INSIDE THE FUNCTIONS.  Not a level
violation -- ``gw`` and ``bse`` are both L1 and an L1->L1 call is legal
(docs/architecture/layers.md).  It is a PYTHON CYCLE: ``bse`` already
imports ``gw`` helpers at module scope (``gw.head_correction``,
``gw.w_isdf``), so a module-scope ``import bse.w_ladder`` here would make
``gw.screening_bse`` and ``bse`` mutually importable and the winner would
depend on which driver started.  The lazy import also keeps the ladder
machinery physically in ``src/bse/``, where TASTE rule 6's two-sums
exemption lives.

SCALING ENVELOPE (TASTE 8 / INVARIANTS 9), stated before the code.  Cost
per (q_irr, z) is ``GMRES iterations x one ring matvec``; the wedge and
the z-list are both loops, so total cost is
``n_z * n_q_irr * iters * matvec``.  Memory high-water on this side of
the seam is ONE ``(n_q_irr, mu, mu)`` complex128 tile at ``P(None,'x','y')``
per z plus the ``(n_q_full, mu, mu)`` unfolded result -- the same class as
the RPA ``W_q`` this replaces, and the same class the driver already
carries.  Nothing ``N_mu^2`` is ever replicated on a rank: the wedge tile
arrives sharded from the facade and the unfold is the same service call
that already moves the RPA W.  The facade owns the resolvent's own
high-water (the probe block); see ``bse/w_ladder.py``.
"""

from __future__ import annotations

import os

import numpy as np
import jax
from jax.sharding import NamedSharding, PartitionSpec as P

import common.timing as timing

from .screening import (
    ScreeningRequest, _gate_w, compute_static_w, screening_requests_for)


def _resolvent_diagram_name(include_w: bool) -> str:
    """The ``screening_diagrams`` value driving one ladder-facade call.

    ``include_w`` fully determines it: :func:`compute_screening_model`
    routes ONLY ``w_bse`` (``include_w=True``) and ``w_rpa_resolvent``
    (``include_w=False``) through this facade.  One function so every
    message-building site below agrees on the label, instead of a ternary
    repeated at each site -- the arbitrary-choice-under-degeneracy shape
    TASTE.md warns produces exactly this class of drift once two sites
    make the same decision independently.
    """
    return "w_bse" if include_w else "w_rpa_resolvent"


# Block-GMRES convergence for the ladder resolvent.  NOT deck keys in v1,
# deliberately: these are the values ``bse_w_exact``'s CLI has carried
# through every W-resolvent closure measurement on this tree
# (``--gmres-tol 1e-10``; the historical cap was 200), and the wiring-closure
# gate compares against the ~2.5e-9 minimax-quadrature floor, so the solver
# residual has to sit orders BELOW the thing being measured rather than be
# tuned per deck.  A deck key here would be a knob whose only safe setting
# is this one -- the shape the ``slab_io`` key was deleted for.  The caller
# gates on the returned per-column residuals, never on a return code.
#
# WHAT THE NUMBER MEANS CHANGED ON 2026-08-16, and so did the number.
#
# MEANING.  ``_gmres_solve_core`` used to exit on ``||r_k|| <= tol * ||r_0||``
# with ``r_0`` the residual of its PRECONDITIONED start, which on this operator
# is 2-11x ``||b||`` (measured: 11.0x on gnppm_debug q=0 z=0; the Si_scalar arm
# delivered 2.24e-10 at a nominal 1e-10).  It now exits on ``tol * ||b||``, so
# the number below is DELIVERED rather than approached.
#
# VALUE.  1e-10 was inherited from ``bse_w_exact``'s CLI, where it is the
# tolerance a CLOSURE measurement wants.  What production needs is a W good
# enough for the QP energies, and that is a different question with a very
# different answer.  MEASURED 2026-08-16 on the scalar-Si cohsex deck (8 q_irr,
# N_mu=192, ONE z), ladder-stage wall / max GMRES iterations / max |dE_qp| in
# eqp1 against the 1e-10 arm:
#
#   1e-10   60.3 s   16 it   (reference)
#   1e-8    47.3 s   13 it   0.001 ueV
#   1e-7    42.8 s   12 it   0.008 ueV
#   1e-6    38.0 s   10 it   0.086 ueV
#
# i.e. |dE_qp| is LINEAR in the tolerance at ~0.086 ueV per 1e-6, so the
# owner's 1 meV bar is not reached until tol ~ 1e-2 and the QP channel is not
# the binding constraint at all.  What binds is the W the driver ships: at
# 1e-6 the solver's own error sits at the production W gate's own 1e-6
# tolerance class, and one decade of headroom is worth ~5 s here.  Hence 1e-6,
# with the ladder above so the next reader can move it in one line rather than
# re-measure it.  A deck key would still be wrong: this is a numerics floor
# derived from a measurement, not a per-deck preference.
#
# CAP.  The 200 inherited from the small closure fixture is too short for a
# wider transition window.  On scalar Si 4x4x4 at nband=20, the last two
# irreducible q points stop at 200 with true residuals 5.56e-3 and 9.17e-3,
# then converge normally at 240-242 iterations to 9.48e-7 and 9.88e-7 when
# allowed to continue.  Small imaginary shifts do not improve them, the exact
# diagonal has min |z-diag(H)| = 5.12e-2 Ry, and its sampled high-band entries
# match the production matvec to roundoff: this is a cap, not a pole or a bad
# preconditioner.  300 keeps 58 iterations of measured headroom without making
# the cap a deck knob; the returned true-residual and truncation gates still
# decide acceptance.
_GMRES_TOL = 1.0e-6
_GMRES_MAX_ITER = 300

#: The TIGHT tolerance, for gates that measure the ASSEMBLY against a
#: quadrature floor (``tests/test_w_bse_wiring_closure.py``) or against a
#: tight-tol oracle.  Those cells need the solver residual to sit below the
#: thing being measured; the production constant above is chosen for the QP
#: energies instead, which is a different question with a different answer.
_GMRES_TOL_TIGHT = 1.0e-10


def _residual_ceiling(tol: float) -> float:
    """Non-convergence guard for the delivered per-column TRUE residual.

    A MULTIPLE of whatever tolerance was asked for, not an absolute number.
    It exists to catch "converged-looking but truncated" (a column that
    stopped at the iteration cap is truncated, not solved, and its residual
    is the only thing that says so -- ``bse_w_exact.py``'s "TRUNCATED at the
    cap" line makes the same distinction for the CLI), NOT to pin an accuracy:
    pinning one here would silently veto the tolerance the caller chose.
    10x, because with the ``||b||``-relative stopping norm the delivered true
    residual measures ~1.0x tol (9.76e-11 at tol 1e-10, gnppm_debug q=0 z=0),
    so an order of magnitude is headroom rather than slack.
    """
    return 10.0 * float(tol)

#: ``WLadderWedge``'s per-column diagnostics, named exactly as the facade
#: names them.  One spelling, not a list of candidates: the facade is in
#: this tree and a rename there should break HERE, loudly, rather than fall
#: through to a second candidate and quietly stop gating.
_RESIDUAL_FIELDS = ("gmres_resid",)
_ITERATION_FIELDS = ("gmres_iters",)


def _wedge_field(wedge, names, dtype):
    """First present attribute among ``names``, as a numpy array, else None."""
    for name in names:
        value = getattr(wedge, name, None)
        if value is not None:
            return np.asarray(value, dtype=dtype)
    return None


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

#: HOW FAR FROM AN INTEGER AN OCCUPATION MAY SIT AND STILL BE ONE.
#:
#: 1e-6, and both ends of that choice are measured rather than tasteful.
#: BELOW it there is only representation noise: a gapped mean field writes
#: exactly 0.0 / 1.0 (or the file's spin convention's 2.0) into
#: ``mf_header/kpoints/occ``, float64 round-trips through HDF5 exactly, and
#: LORRAX's own writer sets the array from a step function
#: (``file_io/wfn_writer.py``:103).  ABOVE it there is only physics: an
#: occupation within 1e-6 of a step value sits more than 13 kT from E_F for
#: any smearing a GW deck is run at, contributes nothing to a pair basis and
#: cannot be what makes a system metallic.  So the band between the two is
#: empty by construction -- the tolerance never has to adjudicate a close
#: case, which is the property a gate wants and a threshold on a physical
#: quantity usually does not have.
_OCC_INTEGER_TOL = 1e-6


def refuse_fractional_occupations(occs, *, band_lo, band_hi, source,
                                  print_fn=print, diagram_name="w_bse"):
    """Refuse the ladder resolvent on a mean field with PARTIAL occupations.

    ``diagram_name`` names which ``screening_diagrams`` value reached this
    check -- ``"w_bse"`` (the default, preserving every existing caller's
    message byte-for-byte) or ``"w_rpa_resolvent"``.  The rule id is
    ``f"{diagram_name}_insulators_only"`` in both cases: the pair-basis /
    integer-occupation argument below is identical for the rung-carrying
    and rung-free operator (see ``gw_config._W_RPA_RESOLVENT_REFUSALS``'
    own comment on its twin row for what does and does not transfer).

    WHY THIS CHECK IS HERE.  Material class is inferred only after the WFN
    is loaded, so config parsing cannot decide this gate.  This is the first
    resolvent seam that reads the original occupation table and can enforce
    the certification envelope.

    WHY THE WFN'S OWN ``occ`` ARRAY AND NOT ``wfns.occ``.  The bundle's
    occupation array is BUILT here, by ``wavefunction_bundle._build_occ``,
    as a step function -- a band-index cut at ``b2`` or a comparison
    against one E_F.  It is {0, 1} by construction, so a gate reading it
    could never fire, and a gate that cannot fire is worse than no gate:
    it is a green light with a name.  ``mf_header/kpoints/occ`` is what
    the DFT run actually resolved, fractional entries and all, and it is
    the only place in this pipeline where a smeared occupation survives.

    ``band_lo`` / ``band_hi`` are the run's active band window ``[b0, b4)``
    -- the bands the ladder's pair basis and the resolvent's poles are
    built from.  Scoping to it is not a loophole: the window straddles E_F
    by construction (``b2 = nelec`` is inside it), so a partially filled
    band -- which by definition sits AT E_F -- cannot hide above ``b4``.

    ``occs`` is ``(nspin, nk, nb)`` as ``mf_header`` reads it; a 2D
    ``(nk, nb)`` array is accepted for the unit cells.  Returns the worst
    distance-from-integer it saw, so a caller can report the margin.
    """
    occ = np.asarray(occs, dtype=np.float64)
    if occ.ndim == 2:
        occ = occ[None, :, :]
    if occ.ndim != 3:
        raise ValueError(
            f"refuse_fractional_occupations: occupations must be "
            f"(nspin, nk, nb) or (nk, nb); got {occ.shape} from {source}.")
    lo = max(0, int(band_lo))
    hi = min(int(band_hi), int(occ.shape[-1]))
    if hi <= lo:
        raise ValueError(
            f"refuse_fractional_occupations: the active band window "
            f"[{band_lo}, {band_hi}) does not intersect the {occ.shape[-1]} "
            f"bands {source} carries.")
    window = occ[:, :, lo:hi]
    distance = np.abs(window - np.rint(window))
    worst = float(distance.max()) if distance.size else 0.0
    if worst > _OCC_INTEGER_TOL:
        s, k, n = (int(i) for i in
                   np.unravel_index(int(distance.argmax()), distance.shape))
        rule_id = f"{diagram_name}_insulators_only"
        raise NotImplementedError(
            f"GATE {rule_id}: screening_diagrams = {diagram_name} is "
            f"refused on a mean field with partial occupations.\n"
            f"  got:  occupation {float(window[s, k, n])!r} at "
            f"(spin {s}, k-point {k}, band {n + lo}) of {source} -- "
            f"{worst:.3e} from the nearest integer, tolerance "
            f"{_OCC_INTEGER_TOL:g}\n"
            f"  want: integer occupations across the active band window "
            f"[{lo}, {hi}) -- an insulating (gapped) mean field\n"
            f"  fix:  run this system with screening_diagrams = w_rpa, or "
            f"point {diagram_name} at a gapped mean field\n"
            f"  why:  the resolvent operator and every certification this "
            f"feature has are insulator-derived: integer occupations and a "
            f"gapped D throughout.  Partial occupations enter both the "
            f"pair basis (partially blocked transitions the band-index cut "
            f"does not model) and the resolvent poles (transitions at ~0 "
            f"energy), and neither is verified here -- the run would "
            f"produce a complete, plausible W under a diagram set that was "
            f"never checked for it.  The metallic MPA screening/Sigma "
            f"pipeline remains available under screening_diagrams = w_rpa; "
            f"it does not confer fractional-occupation semantics on this "
            f"distinct resolvent operator.  Material class is inferred from "
            f"the loaded WFN, and this stage gate reads those occupations "
            f"directly.\n"
            f"  doc:  docs/input_reference.md '## Screening', "
            f"screening_diagrams.")
    print_fn(
        f"  {diagram_name}: occupations are integer across bands "
        f"[{lo}, {hi}) of {os.path.basename(str(source))} (worst deviation "
        f"{worst:.2e} <= {_OCC_INTEGER_TOL:g}) -- insulating mean field, "
        f"the only kind the resolvent is certified for.")
    return worst


def _refuse_metallic_mean_field(config, meta, *, include_w=True,
                                print_fn=print):
    """Read the run's own WFN occupations and hand them to the gate above.

    BEFORE ANY COMPUTE, beside the restart preconditions and for the same
    reason: it is knowable from inputs the run already has, and learning
    it after the chi0 build costs the expensive part of the run to be told
    something the WFN said all along.

    The read is the mf_header metadata block only -- a few kB of
    ``(nspin, nk, nb)`` doubles, not a coefficient -- through the same
    ``file_io.mf_header`` reader ``WfnLoader`` itself uses, so there is no
    second opinion about how a WFN's occupations are spelled.

    ``include_w`` names which resolvent called (``True`` for ``w_bse``,
    ``False`` for ``w_rpa_resolvent`` -- see :func:`_resolvent_diagram_name`);
    both share this one call site inside :func:`prepare_ladder_restart`.
    """
    from file_io.mf_header import read_mf_header

    diagram_name = _resolvent_diagram_name(include_w)
    path = str(getattr(getattr(config, "paths", None), "wfn_file", "") or "")
    if not path or not os.path.exists(path):
        raise ValueError(
            f"GATE {diagram_name}_insulators_only: "
            f"screening_diagrams = {diagram_name} checks "
            f"the mean field's occupations before it builds anything, and "
            f"the deck's wfn_file ({path!r}) is not readable from this "
            f"stage.  The resolvent is certified for insulators only, so "
            f"the check is not optional; fix the path or run w_rpa.")
    band_lo, band_hi = int(meta.band_edges[0]), int(meta.band_edges[-1])
    return refuse_fractional_occupations(
        read_mf_header(path).occs, band_lo=band_lo, band_hi=band_hi,
        source=path, print_fn=print_fn, diagram_name=diagram_name)


def _refuse_unusable_restart(config, meta, sym, centroid_indices,
                             tensors_filename, *, include_w=True,
                             print_fn=print):
    """Refuse, BEFORE any compute, a run whose restart the ladder cannot read.

    ``include_w`` names which resolvent called (default ``True`` = ``w_bse``,
    preserving every existing caller's rule ids and messages byte-for-byte;
    ``False`` = ``w_rpa_resolvent``, see :func:`_resolvent_diagram_name`).
    All three preconditions below are checks on the SHARED restart handoff
    (``prepare_ladder_restart`` runs identically for both arms), so they
    transfer -- audited, not copied: ``w_rpa_resolvent``'s matvec never
    reads the persisted ``W0_qmunu`` VALUE back (``ensure_W_R(include_W=
    False)`` is a placeholder), but the loader still needs the file to
    carry written ``psi_full_y`` / ``enk_full`` / ``V_qmunu`` datasets,
    which only a ``write_restart_tensors = true`` run produces -- so the
    conclusion (refuse without the writes) holds for a different reason
    than the ladder's own "would read stale/absent W" one.

    Three things have to be true for step 3 to work, and every one of them
    is knowable at the top of the stage.  Discovering them after the chi0
    build would cost the expensive part of the run to learn something the
    deck already said.

    1. ``write_restart_tensors`` must be on and the file must exist.
       ``persist_w0_and_head`` is a silent no-op otherwise -- by design,
       for the RPA path, where nothing reads W0 back -- and the ladder
       would then load a file whose ``W0_ready`` is False and get the BARE
       V fallback banner instead of a screening tensor.  That is the April
       all-zero-screening incident's exact shape
       (``tests/test_bse_w0_ready_gate.py``): plausible output, no W.

    2. The stored q-set must be the FULL BZ.  The sharded BSE loader's
       hyperslab transport REFUSES a q_irr wedge by name
       (``bse_loading.py``, ``_MunuSlabPlan``: the unfold gathers across
       the axes that plan shards on, and the price has never been
       measured).  ``restart_q_storage = auto`` writes the wedge exactly
       when the centroid set is orbit-closed, so this is a live
       combination on a closed deck, not a hypothetical.

    3. ``mc_average_placement`` must be off, i.e. ``head_channel`` is
       None.  Checked by the caller against the object it holds.
    """
    from .gw_output import restart_tensor_writes_enabled
    from .restart_q_storage import resolve_restart_q_storage_for_run

    diagram_name = _resolvent_diagram_name(include_w)
    if not config.do_screened:
        raise ValueError(
            f"GATE {diagram_name}_requires_screening: screening_diagrams = "
            f"{diagram_name} reached the screening stage with do_screened "
            f"= false.  There is no W to correct.  (The parse-time refusal "
            f"covers compute_mode = x_only; this is the legacy-flag twin.)")
    if not tensors_filename:
        raise ValueError(
            f"GATE {diagram_name}_needs_the_restart_path: "
            f"screening_diagrams = {diagram_name} "
            f"persists the RPA W(0) into the ISDF restart file and then "
            f"hands that path to the BSE ladder facade, so it cannot run "
            f"without one.  The caller passed tensors_filename = None -- "
            f"the self-consistency map does this today, which is why "
            f"qp_solver = self_consistent is refused at parse time "
            f"(gw_config, {diagram_name}_self_consistency_unimplemented).")
    if not restart_tensor_writes_enabled(config, tensors_filename):
        raise ValueError(
            f"GATE {diagram_name}_needs_restart_writes: screening_diagrams "
            f"= {diagram_name} requires write_restart_tensors = true.\n"
            f"  got:  write_restart_tensors = false\n"
            f"  want: write_restart_tensors = true\n"
            f"  fix:  set write_restart_tensors = true, or keep "
            f"screening_diagrams = w_rpa\n"
            f"  why:  the facade reads psi/eps/V"
            f"{' and the ladder kernel W_R' if include_w else ''} back out "
            f"of {os.path.basename(tensors_filename)}.  With the writes "
            f"off the persist is a no-op and the loader would fall back to "
            f"BARE V with a warning banner"
            + (" -- an all-zero screening that produces plausible numbers "
               "(tests/test_bse_w0_ready_gate.py)."
               if include_w else
               " (unused by this rung-free operator, but the SAME loader "
               "call also needs the file to carry written psi_full_y / "
               "enk_full / V_qmunu, which write_restart_tensors = false "
               "never produced -- tests/test_bse_w0_ready_gate.py)."))
    if not os.path.exists(tensors_filename):
        raise ValueError(
            f"GATE {diagram_name}_needs_restart_writes: {tensors_filename} "
            f"does not exist, so the W0 persist would be skipped and the "
            f"ladder would read nothing.  A restart = true run reuses "
            f"tensors it did not write; point the deck at the directory "
            f"that has them.")
    decision = resolve_restart_q_storage_for_run(
        config, sym=sym, centroid_indices=centroid_indices,
        fft_grid=getattr(meta, "fft_grid", None), print_fn=print_fn,
        context=f"{diagram_name} ladder restart handoff")
    if decision.store_wedge:
        raise ValueError(
            f"GATE {diagram_name}_needs_full_bz_restart: "
            f"screening_diagrams = {diagram_name} "
            f"requires restart_q_storage = full.\n"
            f"  got:  restart_q_storage resolved to "
            f"{decision.mode!r} (the IBZ q wedge)\n"
            f"  want: a full-BZ W0_qmunu / V_qmunu on disk\n"
            f"  fix:  set restart_q_storage = full in the deck\n"
            f"  why:  the ladder facade loads its kernel through the "
            f"SHARDED BSE loader, whose SlabIO hyperslab transport refuses "
            f"a wedge by name (src/bse/bse_loading.py, _MunuSlabPlan) -- "
            f"the unfold gathers across the mu/nu axes that plan shards "
            f"on, and that cost has never been measured.  Refused here, "
            f"before the chi0 build, rather than inside the loader after "
            f"it.")
    return decision


def _assert_restart_is_loadable(tensors_filename, *, include_w=True,
                                print_fn=print):
    """PRESENCE IS NOT PERSISTENCE -- check the flags, not the datasets.

    ``gw_init`` allocates a full-size ZERO ``W0_qmunu`` unconditionally, so
    a file whose persist never fired still answers "yes" to every presence
    question.  The flags ``tagged_arrays`` stamps (``W0_ready`` on W0,
    ``V_ready`` on V) are the ones that discriminate, and they are exactly
    what ``bse_loading`` gates on before it falls back to bare V with a
    banner.  This makes that fallback UNREACHABLE from this path instead of
    merely unlikely.
    """
    import h5py

    diagram_name = _resolvent_diagram_name(include_w)
    with h5py.File(tensors_filename, "r") as f:
        missing = [k for k in ("W0_qmunu", "V_qmunu", "psi_full_y",
                               "enk_full") if k not in f]
        if missing:
            raise RuntimeError(
                f"{diagram_name}: {tensors_filename} is missing "
                f"{', '.join(missing)} after the W0 persist.  The BSE "
                f"loader needs all four (psi/eps from psi_full_y+enk_full, "
                f"the screening from W0_qmunu, the exchange from V_qmunu "
                f"under load_v_full=True).")
        if not bool(f["W0_qmunu"].attrs.get("W0_ready", False)):
            raise RuntimeError(
                f"{diagram_name}: {tensors_filename} carries W0_qmunu but "
                f"its W0_ready flag is False after the persist -- the "
                f"dataset is the zero placeholder gw_init allocates, not a "
                f"written W.  The loader would print the bare-V fallback "
                f"banner"
                + (" and the ladder would be built on no screening at all."
                   if include_w else
                   " -- harmless for this rung-free operator's OWN matvec, "
                   "but the flag is still the only signal that the persist "
                   "actually ran, so it is checked regardless."))
        if not bool(f["V_qmunu"].attrs.get("V_ready", True)):
            raise RuntimeError(
                f"{diagram_name}: {tensors_filename} says V_ready = False; "
                f"the resolvent needs the full exchange tensor "
                f"(load_v_full=True).")
    print_fn(
        f"  {diagram_name}: restart handoff verified -- "
        f"{os.path.basename(tensors_filename)} carries W0_ready + V_ready "
        f"and the psi/eps datasets the ladder loader reads.")


# ---------------------------------------------------------------------------
# Stage 1+2: the RPA static leg and its persist
# ---------------------------------------------------------------------------

def _gate_w_or_refuse(W, req, *, stage, print_fn=print, kgrid=None,
                      include_w=True):
    """Run ``_gate_w`` as a non-negotiable resolvent-stage refusal.

    ``common.sanity`` defaults to warn-and-continue because its generic gates
    also cover legacy paths.  That policy is unsafe here: every resolvent W
    (``w_bse`` or ``w_rpa_resolvent``) is either the ladder kernel's input or
    the operator Sigma will consume, so a failed invariant would bless
    known-bad physics with rc=0.  Pin strict only for this stage, regardless
    of the process-wide ``LORRAX_SANITY`` default, and restore the
    operator's setting before returning or refusing.

    ``include_w`` defaults ``True`` so every EXISTING direct caller (this
    module's own production call sites did not thread it before this axis
    grew a second value, and the unit tests call this function directly)
    keeps the exact ``w_bse_w_stage_invariants`` rule id and message.
    """
    from common import sanity

    diagram_name = _resolvent_diagram_name(include_w)
    rule_id = f"{diagram_name}_w_stage_invariants"
    previous = os.environ.get("LORRAX_SANITY")
    os.environ["LORRAX_SANITY"] = "strict"
    try:
        _gate_w(W, req, print_fn=print_fn, kgrid=kgrid)
    except sanity.SanityError as exc:
        raise ValueError(
            f"GATE {rule_id}: screening_diagrams = {diagram_name} "
            f"refuses a screened-interaction stage that failed _gate_w.\n"
            f"  got:  {stage}: {exc}\n"
            f"  want: finite W, q=0 hermiticity residual <= 1e-6, and -- "
            f"for omega on the imaginary axis -- q<->-q conjugate "
            f"reciprocity residual <= 1e-5\n"
            f"  fix:  do not consume or publish this run's downstream "
            f"artifacts; fix the named W stage and rerun, or use "
            f"screening_diagrams = w_rpa until the resolvent path is "
            f"repaired\n"
            f"  why:  this W enters the BSE/Sigma physics directly; warning "
            f"and continuing would turn a measured invariant violation into "
            f"plausible rc=0 quasiparticle output\n"
            f"  doc:  docs/input_reference.md '## Screening', "
            f"screening_diagrams; docs/dev/QUALITY_PATTERNS.md #7.") from exc
    finally:
        if previous is None:
            os.environ.pop("LORRAX_SANITY", None)
        else:
            os.environ["LORRAX_SANITY"] = previous

def prepare_ladder_restart(
    wfns, V_q, *, quad, e_ref, sym, centroid_indices, config, meta, mesh_xy,
    tensors_filename, head_resolver, head_channel=None, print_fn=print,
    include_w=True,
):
    """Run the RPA static W(0), persist it, and verify the handoff.

    Returns the RPA ``W(0)`` (full BZ, ``(nq, mu, mu)``, ``P(None,'x','y')``)
    -- the caller keeps it for the closure diagnostics; the ladder itself
    reads its copy back off disk, which is the point of the persist.

    THE PERSIST IS UNCONDITIONAL ON THIS BRANCH, INCLUDING UNDER
    ``compute_mode = mpa``.  The driver's own flush skips MPA because
    ``persist_w0_and_head``'s ``{0, probe}`` head grid is not MPA's sample
    set -- a real refusal, kept.  What this branch writes is not MPA's
    sample set either: it is the ONE static RPA W(0) the ladder kernel
    needs, whose head grid is ``{0}`` and nothing else.  So the call opts
    into the static-only head grid BY NAME (``static_head_only=True``)
    rather than the refusal being relaxed.

    ``include_w`` names which resolvent will consume this restart --
    ``True`` (default) for ``w_bse``, ``False`` for ``w_rpa_resolvent`` --
    and is used ONLY to label the preconditions and gates below; the RPA
    static leg computed and persisted here is identical either way (both
    arms need the SAME restart infrastructure, see module docstring).
    """
    from .gw_output import persist_w0_and_head

    diagram_name = _resolvent_diagram_name(include_w)
    if head_channel is not None:
        raise ValueError(
            f"GATE {diagram_name}_head_placement_unimplemented: a "
            f"head_channel reached the {diagram_name} stage helper, which "
            f"means mc_average_placement != off got past the parse-time "
            f"refusal in gw_config.refuse_unsupported_screening_diagrams.  "
            f"v1 policy is that the q=0 head/wing channel stays RPA and "
            f"the resolvent replaces the body only; a post-solve head "
            f"rescale is a second opinion about the same channel.  Fix the "
            f"parse gate, do not serve this here.")

    # THE TWO PRECONDITION BLOCKS, in cost order.  The occupation gate runs
    # FIRST because it is the cheapest read in the stage (an mf_header
    # metadata block) and because it is the one that decides whether this
    # SYSTEM is one the ladder is certified for at all -- a question that
    # outranks whether the restart handoff is wired.
    _refuse_metallic_mean_field(config, meta, include_w=include_w,
                                print_fn=print_fn)
    _refuse_unusable_restart(config, meta, sym, centroid_indices,
                             tensors_filename, include_w=include_w,
                             print_fn=print_fn)

    W0_rpa = compute_static_w(
        wfns, V_q, quad, e_ref=e_ref, sym=sym,
        centroid_indices=centroid_indices, config=config, meta=meta,
        mesh_xy=mesh_xy, role="static", head_channel=None)
    with timing.section("W.gate", announce=True,
                        label="W[static] (RPA, ladder kernel) "
                              "finiteness + hermiticity gate"):
        _gate_w_or_refuse(
            W0_rpa, ScreeningRequest(0.0 + 0.0j, "static"),
            stage="RPA input W[static] for the ladder kernel",
            print_fn=print_fn, kgrid=tuple(meta.kgrid), include_w=include_w)

    rpa_iteration_head = None
    from .gw_config import HeadCorrection
    if config.head.correction is HeadCorrection.FULL:
        from .qsgw_head import (
            build_dft_head_response, finalize_iteration_head_samples)
        static_request = ScreeningRequest(0.0 + 0.0j, "static")
        direct = build_dft_head_response(
            wfns, np.asarray([0.0j]), input_dir=config.input_dir,
            mesh=mesh_xy, wfn=head_resolver.wfn,
            meta=meta, config=config)
        rpa_iteration_head = finalize_iteration_head_samples(
            direct, wfn=head_resolver.wfn, meta=meta,
            config=config, mesh=mesh_xy, requests=[static_request],
            W_by_role={"static": W0_rpa})
    with timing.section("gw_jax.persist_w0"):
        persist_w0_and_head(
            W0_rpa, tensors_filename=tensors_filename,
            head_resolver=head_resolver, iteration_head=rpa_iteration_head,
            config=config, meta=meta,
            mesh_xy=mesh_xy, sym=sym, centroid_indices=centroid_indices,
            static_head_only=True, print_fn=print_fn)
    _assert_restart_is_loadable(tensors_filename, include_w=include_w,
                                print_fn=print_fn)
    return W0_rpa


# ---------------------------------------------------------------------------
# Stage 3: the ladder facade
# ---------------------------------------------------------------------------

def _ladder_wedge(tensors_filename, z_list_ry, mesh_xy, *, input_file,
                  include_w=True, print_fn=print, gmres_tol=None,
                  config=None, meta=None, wfn=None):
    """One call into ``bse.w_ladder``, with the residual gate on its output.

    ``gmres_tol`` defaults to the PRODUCTION constant :data:`_GMRES_TOL`.
    Gates that measure the assembly against a quadrature floor pass
    :data:`_GMRES_TOL_TIGHT` instead — the solver residual has to sit below
    the thing being measured, and the production tolerance is chosen for the
    QP energies rather than for a closure cell (see the constants' comment).

    The import is function-level; see the module docstring for why (Python
    cycle, not a layer rule).

    ``input_file`` is the GW deck.  The facade needs it in addition to the
    restart path because the irreducible q wedge comes from ``SymMaps``,
    which is built from the WFN the DECK names and which no restart file
    records; it is an ADDITIVE keyword on the frozen contract and the
    facade refuses by name without it.  Supplied from
    ``config.input_file``, so the ladder reads the same deck the driver ran.
    """
    from bse.w_ladder import compute_wc_qwedge   # noqa: PLC0415 (cycle)

    if not input_file:
        raise ValueError(
            "GATE w_bse_needs_the_deck_path: the ladder facade resolves the "
            "irreducible q wedge from SymMaps, built from the WFN named by "
            "the GW DECK, and no restart file records which deck that was.  "
            "This config carries no input_file, which means it was built by "
            "hand rather than parsed (LorraxConfig.from_input_file sets it). "
            "Pass a parsed config, or keep screening_diagrams = w_rpa.")
    z = np.asarray(z_list_ry, dtype=np.complex128)
    tol = _GMRES_TOL if gmres_tol is None else float(gmres_tol)
    ceiling = _residual_ceiling(tol)
    # THREAD THE FACADE'S BOUNDED-MEMORY CONTROL.  ``compute_wc_qwedge``
    # has always taken a public ``probe_chunk``, but until 2026-08-22 this
    # facade never passed it, so every production run solved the whole
    # padded mu^2 tile in one block — a 77.83-GiB single allocation on the
    # fully relativistic LiF 666-centroid deck (JID 57288835), against a
    # measured gate-preserving probe_chunk=64 discriminator on the same
    # deck (JID 57280453).  The deck key is ``ladder_probe_chunk``
    # (ScreeningConfig); 0 keeps the whole-basis block, bit-identical.
    # The kernel's contract requires a multiple of the mesh 'y' extent
    # (the reduce-scatter snapshot tiles the probe axis over 'y'), so a
    # positive deck value is ROUNDED UP here — a placement change, never a
    # value change — and announced with its per-block granularity.
    probe_chunk = None
    _pc_deck = int(getattr(getattr(config, "screening", None),
                           "ladder_probe_chunk", 0) or 0)
    if _pc_deck > 0:
        _py = int(mesh_xy.devices.shape[1])
        probe_chunk = ((_pc_deck + _py - 1) // _py) * _py
        print_fn(
            f"  w_bse: ladder_probe_chunk={_pc_deck}"
            + (f" rounded up to {probe_chunk} (mesh 'y'={_py} multiple)"
               if probe_chunk != _pc_deck else "")
            + f" — probe columns solved {probe_chunk} per block instead "
              f"of the whole padded basis in one allocation")
    with timing.section("gw_jax.w_ladder", announce=True,
                        label=f"W ladder resolvent ({z.size} z, "
                              f"include_w={include_w})"):
        head_kwargs = {}
        if config is not None:
            from .gw_config import HeadCorrection
            if config.head.correction is HeadCorrection.FULL:
                if meta is None or wfn is None:
                    raise ValueError(
                        "head_correction=full requires meta and wfn at the "
                        "ladder facade so the resolvent normalization is "
                        "defined")
                from bse.head_resolvent import head_prefactor
                head_kwargs = {
                    "head_dipole_path": os.path.join(
                        config.input_dir, "dipole.h5"),
                    "head_n_occ": int(meta.nelec),
                    "head_pref": head_prefactor(
                        float(meta.cell_volume), int(meta.nk_tot),
                        int(wfn.nspin), int(meta.nspinor_wfnfile)),
                }
        wedge = compute_wc_qwedge(
            tensors_filename, z, mesh_xy, include_w=include_w,
            gmres_tol=tol, gmres_max_iter=_GMRES_MAX_ITER,
            probe_chunk=probe_chunk,
            input_file=input_file, **head_kwargs)
    resid = _wedge_field(wedge, _RESIDUAL_FIELDS, np.float64)
    iters = _wedge_field(wedge, _ITERATION_FIELDS, np.int64)
    # AN ABSENT RESIDUAL IS A REFUSAL, NOT A SKIPPED CHECK.  "no residuals
    # to look at" and "every residual is fine" must not produce the same
    # behaviour here (QUALITY_PATTERNS addendum: the observable has to
    # discriminate), and the facade contract obliges the caller to gate on
    # residuals rather than on a return code.
    if resid is None:
        raise AttributeError(
            f"w_bse: the ladder facade returned no per-column GMRES "
            f"residuals (looked for {', '.join(_RESIDUAL_FIELDS)}).  The "
            f"caller is contractually required to gate on them, so their "
            f"absence is refused rather than treated as convergence.")
    worst = float(np.max(resid)) if resid.size else float("nan")
    # An iteration count equal to the cap is truncation, not convergence,
    # and the two are indistinguishable downstream.
    hottest = int(np.max(iters)) if iters is not None and iters.size else -1
    truncated = hottest >= _GMRES_MAX_ITER
    print_fn(
        f"  w_bse: ladder GMRES max residual {worst:.2e} "
        f"(tol {tol:.0e}), max iters {hottest}/{_GMRES_MAX_ITER}"
        + ("  TRUNCATED AT THE CAP" if truncated else ""))
    if not (worst <= ceiling) or truncated:
        raise RuntimeError(
            f"w_bse: the ladder resolvent did not converge -- max "
            f"per-column GMRES residual {worst:.3e} (ceiling "
            f"{ceiling:.0e})"
            + (", and at least one column stopped AT the "
               f"{_GMRES_MAX_ITER}-iteration cap" if truncated else "")
            + ".  A truncated column returns a finite, plausible tile; "
              "it is refused here rather than fitted downstream.")
    return wedge


def _finalize_ladder_head(wedge, *, config, meta, head_resolver, print_fn=print):
    """Install the micro-reducible resolvent head without a second fold."""
    from .gw_config import HeadCorrection
    if config.head.correction is not HeadCorrection.FULL:
        return None
    result = getattr(wedge, "head_result", None)
    if result is None:
        raise RuntimeError(
            "head_correction=full requested w_bse, but the ladder facade "
            "returned no q->0 resolvent tensor")
    worst = float(np.max(result.resids))
    hottest = int(np.max(result.iters))
    if not worst <= _residual_ceiling(_GMRES_TOL) or hottest >= _GMRES_MAX_ITER:
        raise RuntimeError(
            "w_bse q->0 head resolvent did not converge: max true residual "
            f"{worst:.3e}, iterations {hottest}/{_GMRES_MAX_ITER}")
    xi = np.asarray(result.xi, dtype=np.complex128)
    xi_long = 0.5 * (xi + np.swapaxes(xi, -1, -2))
    print_fn(
        "  w_bse head: using micro-reducible BSE resolvent exactly once "
        f"(max residual {worst:.2e}, max |Xi-Xi^T|/|Xi| "
        f"{float(np.max(result.asym)):.2e}, dipole/operator Delta mismatch "
        f"{float(result.delta_mismatch):.2e}); no Schur refold.")
    from .qsgw_head import IterationHeadSamples, head_samples_from_s
    samples = head_samples_from_s(
        xi_long, result.z, wfn=head_resolver.wfn,
        meta=meta, config=config, response_kind="micro_reducible",
        source_prefix="bse_resolvent_micro")
    resolved = IterationHeadSamples(
        omegas=tuple(complex(z) for z in result.z), samples=samples,
        sigma_energies_ry=np.empty((0, 0), dtype=np.float64),
        sigma_occupations=np.empty((0, 0), dtype=np.float64),
        efermi_ry=0.0)
    head_resolver.install_samples(samples)
    return resolved


def _assert_wedge_matches_run(wedge, sym):
    """The facade's q wedge must BE this run's q wedge, not merely look like it.

    Both sides derive it from ``SymMaps``, but they derive it from
    different entry points (the facade from the WFN it opens, the GW leg
    from the ``sym`` object the driver built), and the unfold tables this
    module applies are the GW leg's.  If the two orderings ever disagreed
    the add-back and the unfold would each be individually correct and the
    composition silently wrong -- the finite-q class of bug that was
    invisible at q=0 (KNOWN_FAILURES:1248).  So it is compared, not assumed.
    """
    got = np.asarray(getattr(wedge, "q_irr_kgrid_int"), dtype=np.int64)
    want = np.asarray(sym.q_irr_kgrid_int, dtype=np.int64)
    if got.shape != want.shape or not np.array_equal(got, want):
        raise ValueError(
            f"w_bse: the ladder facade's q wedge does not match this run's. "
            f"facade {got.shape} vs SymMaps {want.shape}; first "
            f"disagreement at index "
            f"{int(np.argmax(np.any(got != want, axis=-1))) if got.shape == want.shape else 0}. "
            f"The unfold tables applied below are this run's, so a "
            f"different ordering would misplace every q.")
    idx_got = np.asarray(getattr(wedge, "q_irr_full_idx"), dtype=np.int64)
    idx_want = np.asarray(sym.q_irr_full_idx, dtype=np.int64)
    if idx_got.shape != idx_want.shape or not np.array_equal(idx_got, idx_want):
        raise ValueError(
            f"w_bse: the facade's q_irr_full_idx {idx_got.tolist()[:8]}... "
            f"disagrees with this run's {idx_want.tolist()[:8]}...  That "
            f"table is what slices V onto the wedge below, so a mismatch "
            f"would add the wrong v to every body.")


# ---------------------------------------------------------------------------
# Stage 4: + v, mu-pad, unfold
# ---------------------------------------------------------------------------

def _assert_mu_width(tile, mu_target, *, where):
    """The ladder tile's mu extent IS this run's padded extent — checked, not fixed.

    NO STRIP-THEN-RE-PAD ROUND TRIP (design amendment, 2026-08-15).  The
    facade returns tiles on the PADDED extent because a logical-extent tile
    cannot carry ``P(None, None, 'x', 'y')`` when ``n_rmu`` does not divide
    the mesh (399 on a 2x2 — measured, JID 57064957), and this side already
    lives there: ``V_q``, the ``sym_perm`` unfold tables and the production
    ``W_q`` are all ``padded_mu_extent`` wide.  The assembly and the unfold
    therefore run at padded width from end to end, and the only place a
    physical ``n_rmu`` sub-tile is ever cut is a HOST view at the comparison
    boundary (the closure gate) — never on a sharded device array, where
    slicing a ``P('x','y')`` axis from 400 to 399 is precisely the
    un-expressible shape that started this.

    The equality is an invariant and not a coincidence: both sides call
    ``runtime.padding.padded_mu_extent(n_rmu, px*py)`` — the loader at
    ``bse_loading.py`` (``n_rmu_pad``) and the GW leg through the sym
    tables.  So a mismatch means those two disagree about the mesh, and
    padding over it here would hide which one is wrong.
    """
    n_mu = int(tile.shape[-1])
    if n_mu != int(mu_target):
        raise ValueError(
            f"GATE w_bse_mu_width_mismatch ({where}): the ladder tile is "
            f"{n_mu} wide and this run's padded mu extent is {mu_target}.\n"
            f"  why:  both are runtime.padding.padded_mu_extent(n_rmu, "
            f"px*py) — the BSE loader's n_rmu_pad and the GW leg's sym "
            f"tables — so they cannot differ unless the two legs saw "
            f"different meshes or different centroid counts.\n"
            f"  fix:  not here.  Padding over the gap would put the ladder "
            f"body under the wrong unfold tables, which is the finite-q "
            f"class of bug that is invisible at q=0.")
    return tile


from .qgrid_symmetry import qgrid_trs_policy_for


def _assemble_full_bz_w(wc_wedge, V_q, *, sym, centroid_indices, meta,
                        mesh_xy, label, print_fn=print):
    """``Wc(z)`` on the wedge -> ``W(z)`` on the full BZ.

    ``+ v`` first (the facade returns ``W - v`` bodies), then the SAME
    ``unfold_isdf_operator`` service call ``compute_static_w`` makes, with
    the SAME geometry tables from ``_resolve_ibz_q_list``.  The q-axis
    time-reversal decision — pair coherence and the fixed-q projector — is
    the SHARED policy object (``gw.qgrid_symmetry.qgrid_trs_policy_for``),
    because bare V, RPA W and ladder W must use one q-grid realization and
    because the decision is taken from the shared TRS verdict, not assumed
    here. This is not a second unfold: one service, one set of
    centroid/phase tables, and one convention for the umklapp phase and TRS
    conjugation.
    """
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import slice_q_full_to_ibz, unfold_isdf_operator

    from .v_q_g_flat import _resolve_ibz_q_list

    (_, q_irr_frac, full_to_irr_idx, full_to_irr_sym, sym_perm, L_table,
     use_ibz) = _resolve_ibz_q_list(
        sym=sym, centroid_indices=centroid_indices,
        kgrid=tuple(meta.kgrid), fft_grid=tuple(meta.fft_grid),
        context=f"W[{label}] ladder wedge -> full BZ unfold")
    if not use_ibz:
        raise ValueError(
            "GATE w_bse_needs_an_orbit_closed_wedge: the ladder computes "
            "on the symmetry-reduced q wedge, so the run's centroid set "
            "must be orbit-closed for the unfold back to the full BZ to "
            "exist.  This deck's set is not (the same condition that drops "
            "the RPA Dyson solve to the full BZ).  Use a closed centroid "
            "set, or keep screening_diagrams = w_rpa.")

    _nat = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    mu_target = int(np.asarray(sym_perm).shape[-1])
    V_wedge = slice_q_full_to_ibz(V_q, sym.q_irr_full_idx, out_sharding=_nat)
    W_wedge = _assert_mu_width(
        wc_wedge, mu_target, where=f"W[{label}] wedge -> full BZ") + V_wedge
    n_sym_spatial = int(np.asarray(sym_perm).shape[0]) // 2
    policy = qgrid_trs_policy_for(
        sym=sym, irr_idx_q=full_to_irr_idx, sym_idx_q=full_to_irr_sym,
        kgrid=tuple(meta.kgrid), n_sym_spatial=n_sym_spatial,
        context=f"W[{label}] ladder")
    ladder_unfold_sym = policy.unfold_sym_idx
    cov = policy.measure_covariance(
        W_wedge, q_irr_frac=q_irr_frac, q_irr_full_idx=sym.q_irr_full_idx,
        sym_mats_k=sym.sym_mats_k, sym_perm=sym_perm, L_table=L_table)
    W_wedge, removed = policy.project_fixed_q(W_wedge, sym.q_irr_full_idx)
    from common import sanity
    sanity.report_parent_covariance(
        f"W[{label}] ladder IBZ parents", cov, removed=removed,
        print_fn=print_fn)
    with timing.section("W.unfold_to_full_bz", announce=True,
                        label=f"W[{label}] ladder IBZ -> full-BZ unfold "
                              f"({int(W_wedge.shape[0])} q -> "
                              f"{int(meta.nk_tot)} q)"):
        W_q = unfold_isdf_operator(
            W_wedge,
            irr_idx=full_to_irr_idx, sym_idx=ladder_unfold_sym,
            sym_perm=sym_perm, L_table=L_table, q_irr_frac=q_irr_frac,
            mesh_xy=mesh_xy, n_sym_spatial=n_sym_spatial)
        del W_wedge
        W_q.block_until_ready()
    return W_q


# ---------------------------------------------------------------------------
# The stage entry points
# ---------------------------------------------------------------------------

def compute_screening_ladder(
    mode, wfns, V_q, *, quad, e_ref, sym, centroid_indices, config, meta,
    mesh_xy, tensors_filename, head_resolver, head_channel=None,
    static_only=False, print_fn=print, include_w=True,
):
    """``{role: W_q}`` from the resolvent, for the non-MPA modes.

    Same return contract as :func:`gw.screening.compute_screening`:
    ``(nq_full, mu, mu)`` complex128 at ``NamedSharding(mesh_xy,
    P(None,'x','y'))``, keyed by the SAME role labels
    ``screening_requests_for`` produced.

    ``include_w`` selects the operator this call solves for:
    ``True`` (default) is the LADDER kernel, i.e. ``screening_diagrams =
    w_bse``; ``False`` builds the RPA operator inside the same resolvent
    machinery instead (the rung parameterized out --
    ``bse.bse_ring_comm.build_bse_ring_matvec_full(..., include_W=False)``
    -- one matvec builder, not a second one), i.e. ``screening_diagrams =
    w_rpa_resolvent``.  :func:`gw.screening.compute_screening_model` is
    the ONLY caller and sets it from ``diagrams`` directly.  Both arms
    reproduce their respective reference to the minimax-quadrature floor
    (``tests/test_bse_w_ladder_identities.py``,
    ``tests/test_w_bse_wiring_closure.py``) -- the RPA arm was a
    wiring-closure-only knob before this axis grew a second member; it is
    now itself a served ``screening_diagrams`` value.
    """
    diagram_name = _resolvent_diagram_name(include_w)
    requests = screening_requests_for(mode, config)
    if static_only:
        requests = [r for r in requests if r.role == "static"]
    if not requests:
        # X_ONLY is refused at parse time under both resolvent diagrams;
        # MPA goes through make_ladder_wc_source under w_bse (refused at
        # parse time under w_rpa_resolvent -- see
        # gw_config._W_RPA_RESOLVENT_REFUSALS).  Reaching here with no
        # request means a mode whose plan is empty asked for a resolvent W
        # anyway.
        raise ValueError(
            f"compute_screening_ladder: compute_mode = "
            f"{getattr(mode, 'value', mode)} declares no screening "
            f"requests, so there is no frequency at which to evaluate the "
            f"{diagram_name} W.  MPA's shared frequency walk goes through "
            f"gw.screening_bse.make_ladder_wc_source instead.")

    prepare_ladder_restart(
        wfns, V_q, quad=quad, e_ref=e_ref, sym=sym,
        centroid_indices=centroid_indices, config=config, meta=meta,
        mesh_xy=mesh_xy, tensors_filename=tensors_filename,
        head_resolver=head_resolver, head_channel=head_channel,
        print_fn=print_fn, include_w=include_w)

    # THE z-LIST COMES FROM THE ROLE PLAN, not from a second table.
    # cohsex -> [0]; gn_ppm -> [0, i*omega_p].  One source of truth for
    # "which W does this Sigma need" stays where it is (screening.py).
    z_list = [complex(r.omega_ry) for r in requests]
    for r in requests:
        if abs(complex(r.omega_ry).real) > 0.0:
            raise NotImplementedError(
                f"GATE {diagram_name}_hl_ppm_broadening_unimplemented: "
                f"role {r.role!r} wants W at the REAL frequency "
                f"{complex(r.omega_ry)!r}, and (z - H)^-1 on the real axis "
                f"needs a broadening policy the resolvent does not have.  "
                f"Refused at parse time for hl_ppm; restated here so a new "
                f"real-axis role cannot inherit an answer.")

    wedge = _ladder_wedge(
        tensors_filename, z_list, mesh_xy,
        input_file=getattr(config, "input_file", ""),
        include_w=include_w, print_fn=print_fn,
        config=config, meta=meta, wfn=head_resolver.wfn)
    _assert_wedge_matches_run(wedge, sym)
    _finalize_ladder_head(
        wedge, config=config, meta=meta, head_resolver=head_resolver,
        print_fn=print_fn)
    wc = wedge.wc
    if int(wc.shape[0]) != len(z_list):
        raise ValueError(
            f"{diagram_name}: the facade returned {int(wc.shape[0])} "
            f"z-slabs for a {len(z_list)}-frequency request.")

    W_by_role: dict[str, jax.Array] = {}
    for i, req in enumerate(requests):
        W_q = _assemble_full_bz_w(
            wc[i], V_q, sym=sym, centroid_indices=centroid_indices,
            meta=meta, mesh_xy=mesh_xy, label=req.role, print_fn=print_fn)
        # THE SAME GATE THE RPA PATH RUNS, at the same tolerances.
        # Hermiticity and W_q = conj(W_{-q}) are EXPECTED to hold for the
        # resolvent on a TRS deck at omega in {0} u iR; if this fires it is
        # evidence about the operator, not an inconvenience.  Do not
        # loosen it (design section 4).
        with timing.section("W.gate", announce=True,
                            label=f"W[{req.role}] ({diagram_name}) "
                                  f"finiteness + hermiticity gate"):
            _gate_w_or_refuse(
                W_q, req, stage=f"assembled {diagram_name} W[{req.role}]",
                print_fn=print_fn, kgrid=tuple(meta.kgrid),
                include_w=include_w)
        W_by_role[req.role] = W_q
    return W_by_role


def make_ladder_wc_source(
    wfns, V_q, *, quad, e_ref, sym, centroid_indices, config, meta, mesh_xy,
    tensors_filename, head_resolver, head_channel=None, print_fn=print,
):
    """The ``wc_source`` seam MPA's ``build_mpa_fit`` calls instead of its Dyson.

    Returns a callable with ``mpa.model._solve_wc``'s signature
    ``(sample_path, V, z, q_idx, meta, mesh_xy, dyson_solver=None)``.  Same
    per-z, per-wedge-q ``Wc(z)`` slabs written into the same SlabIO sample
    store; the fit, the store lifecycle and the pole consumer are
    untouched.  What changes is only where the slab came from.

    The RPA static leg + persist runs ONCE, on first call, because MPA's
    sample plan is not known until ``build_mpa_fit`` has built it -- and
    because the ladder kernel needs the same ``W(0)`` whatever the plan
    turns out to be.
    """
    from file_io import mpa_store
    # The dataset name is READ FROM the module that owns the store rather
    # than re-typed here: two spellings of one HDF5 key is the shadow-
    # accounting shape (QUALITY_PATTERNS #3).
    from .mpa.model import _WC

    state = {"prepared": False}

    def _wc_from_ladder(sample_path, V, z, q_idx, meta_, mesh_,
                        dyson_solver=None):
        z_all = np.asarray(z, dtype=np.complex128)
        if not state["prepared"]:
            prepare_ladder_restart(
                wfns, V_q, quad=quad, e_ref=e_ref, sym=sym,
                centroid_indices=centroid_indices, config=config, meta=meta,
                mesh_xy=mesh_xy, tensors_filename=tensors_filename,
                head_resolver=head_resolver, head_channel=head_channel,
                print_fn=print_fn)
            state["prepared"] = True
        wedge = _ladder_wedge(
            tensors_filename, z_all, mesh_,
            input_file=getattr(config, "input_file", ""),
            include_w=True, print_fn=print_fn,
            config=config, meta=meta_, wfn=head_resolver.wfn)
        _assert_wedge_matches_run(wedge, sym)
        iteration_head = _finalize_ladder_head(
            wedge, config=config, meta=meta_, head_resolver=head_resolver,
            print_fn=print_fn)
        shape = (z_all.size, q_idx.size, meta_.n_rmu, meta_.n_rmu)
        mu_target = int(V.shape[-1])
        for index in range(z_all.size):
            Wc = _assert_mu_width(
                wedge.wc[index], mu_target,
                where=f"MPA Wc slab z[{index}]")
            Wc.block_until_ready()
            mpa_store.write_w_slab_collective(
                sample_path, _WC, index, Wc, mesh_xy=mesh_,
                global_shape=shape)
            del Wc
        return iteration_head

    return _wc_from_ladder


__all__ = [
    "compute_screening_ladder",
    "make_ladder_wc_source",
    "prepare_ladder_restart",
    "refuse_fractional_occupations",
]
