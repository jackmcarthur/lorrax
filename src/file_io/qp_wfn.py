"""QP wavefunction rotation matrix I/O.

Two writers live here:

* :func:`write_qp_rotations_h5` — small ``(U, E_qp)`` companion file used
  by tools that want to apply the QP rotation themselves.
* :func:`write_qp_wfn_h5` — full BGW-compatible ``WFN.h5`` with ψ already
  rotated and energies replaced.  This is the canonical "QP WFN" output
  consumed by downstream BSE / restart paths that just want a WFN.h5
  drop-in replacement.

WHICH k-SET ``qp_wfn_rotations.h5`` IS STORED ON
------------------------------------------------------------------------
Historically: the full BZ, always, with nothing on the file saying so.
Now: the FILE WEDGE (``wfn.kpoints``, ``sym.nk_red`` rows — the k-set
``kirr_to_kfull`` already addressed) when the writer can PROVE the
reduction loses nothing, and the full BZ otherwise.

The stamping model is ``kin_ion.h5``'s and the constants are ITS
constants, imported rather than re-spelled: ``k_storage`` /
``k_storage_version`` / ``n_sym_spatial`` per dataset, the two unfold
tables beside the arrays, and **a dataset with no ``k_storage`` attr read
as ``"full"``**.  One stamp contract for three files is the point; a
second copy here would be a second place for the version and the refusals
to drift.

WHY THE PROOF, AND WHY IT IS NOT OPTIONAL.  ``U_mnk`` is NOT a scalar
operator like ``kin_ion``.  It is a stack of EIGENVECTORS, defined up to a
phase and — inside a degenerate multiplet — up to a unitary mixing, so
"the star relation holds for this quantity" is a statement about HOW THIS
RUN PRODUCED IT, not about the physics:

* Self-consistent runs with ``sc_on_ibz`` (the default) diagonalise on the
  wedge and broadcast, so the full-BZ rows ARE gathers and the wedge form
  is exact.
* The one-shot path runs an independent ``eigh`` at every full-BZ k
  (``gw_jax.py``), so its off-wedge rows are a different gauge and
  discarding them WOULD lose information.

Both write this file.  Rather than encode which caller is which — a rule
that decays the moment a third producer appears — :func:`write_qp_rotations_h5`
performs the round trip on the arrays in hand and keeps the wedge only if
it reproduces them.  ``"auto"`` therefore means "the wedge, if this run's
own numbers say the wedge is enough", and a file that carries the stamp is
a file whose reconstruction was checked by the process that wrote it.

Nothing here weakens the absent-means-full rule: an old file, a
hand-written test file and a full-BZ file written today are all read
verbatim, because the discriminator is an attribute no old writer wrote.
"""
import json
import os

import numpy as np
import h5py

from .kin_ion import (
    IRR_IDX_DATASET,
    K_STORAGE_ATTR,
    K_STORAGE_FULL,
    K_STORAGE_IBZ,
    K_STORAGE_VALUES,
    K_STORAGE_VERSION,
    K_STORAGE_VERSION_ATTR,
    N_SYM_SPATIAL_ATTR,
    SYM_IDX_DATASET,
    broadcast_ibz_to_full_bz,
)
from .sigma_output import SIGMA_EVAL_PROVENANCE_ATTR

#: Legal values of the ``qp_rotations_k_storage`` input key.  Same three
#: words as ``restart_q_storage`` (``gw.restart_q_storage.RESTART_Q_STORAGE``)
#: so a deck author learns one vocabulary, not two:
#:
#:   auto — store the file wedge IF the round trip reproduces the full-BZ
#:          arrays exactly; the full BZ otherwise, saying which array and
#:          by how much.
#:   full — the old bytes, unconditionally.  Does not ask the question.
#:   ibz  — REFUSE rather than fall back, for a run that believes its rows
#:          are gathers and wants to be told the day they stop being.
QP_ROTATIONS_K_STORAGE = ("auto", "full", "ibz")

#: The datasets that MOVE onto the wedge — the physics arrays, and only
#: those.  Each is star-invariant in the sense the gather needs: a member of
#: a star holds the same matrix (or the same energies) as its parent.
QP_ROT_K_DATASETS = ("U_mnk", "E_qp_nk_hartree", "E_qp_nk_rydberg")

#: ``kpoints_crys`` and ``kirr_to_kfull`` STAY ON THE FULL BZ, always, and
#: that is not an oversight.
#:
#: ``broadcast_ibz_to_full_bz`` is a GATHER: every member of a star receives
#: its parent's row verbatim.  For an operator that commutes with the
#: symmetry, that IS the right answer.  For the k-VECTORS it is not, because
#: k is the one quantity in this file that the symmetry operation changes —
#: a gather would hand every member of a star its parent's coordinates.
#: MEASURED on ``si_cohsex_debug``: ``max|Δ| = 7.500000e-01``, and NOT a
#: reciprocal-lattice vector, so no modulo-G reading rescues it.
#:
#: They are also 1,536 and 32 bytes on that deck against 3.5 MB of ``U_mnk``,
#: so there is nothing to win.  Leaving them alone keeps the file's own
#: coordinate table and index table meaning exactly what they always meant,
#: which is what lets the two in-tree consumers unfold and then index as
#: before.
QP_ROT_FULL_BZ_DATASETS = ("kpoints_crys", "kirr_to_kfull")

#: Small, non-k-reduced datasets that define which Hamiltonian the physics
#: arrays belong to.  Consumers of both ``U_mnk`` and ``E_qp`` must read
#: these through :func:`read_qp_rotations_artifact`; opening the HDF5 file a
#: second time in each driver would create another metadata contract.
QP_ROT_METADATA_DATASETS = ("band_range", "kpoints_crys", "kgrid")

#: Source mean-field identity for the DFT-band basis in which ``U_mnk`` is
#: expressed.  These are root attributes because they do not carry a k axis
#: and must survive either full-BZ or wedge storage unchanged.
QP_ROT_WFN_FINGERPRINT_SCHEME_ATTR = "source_wfn_fingerprint_scheme"
QP_ROT_WFN_FINGERPRINT_ATTR = "source_wfn_fingerprint"


def _require_wfn_fingerprint(value, *, where: str) -> str:
    """Validate the canonical owner's lowercase SHA-256 representation."""
    fingerprint = str(value).strip()
    if (len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)):
        raise ValueError(
            f"{where} must be a 64-digit lowercase hexadecimal SHA-256, "
            f"got {value!r}.")
    return fingerprint


#: Root attribute :func:`write_qp_wfn_h5` stamps on its output, and the ONLY
#: content-based way to tell a QP WFN.h5 from a mean-field one.  A QP WFN's ψ
#: and E are a matched pair — the rotated orbitals carry the eigenvalues that
#: produced the rotation — and a consumer that then applies a second,
#: DFT-band-labelled ``eqp1.dat`` ladder discards the canonical ones and
#: relabels rotated orbitals with someone else's band ordering (measured on
#: the MoS2 run-82 parent smoke, JID 57269074 step .128; deck
#: ``exciton_parent_smoke.in:9`` selects ``WFN_qp.h5`` and the wrapper passes
#: ``--eqp eqp1.dat``).  Until this stamp existed the only discriminator was
#: the FILENAME, which is not a fact about the contents.
#:
#: ABSENT MEANS UNVERIFIABLE, NOT MEAN-FIELD — the same reading the
#: ``k_storage`` stamps above take.  Every WFN.h5 written before this stamp,
#: and every one written by BerkeleyGW or ``pw2bgw``, carries nothing here.
QP_WFN_ATTR = "qp_wfn_scheme"

#: Versioned so a future change of what "rotated" means is a different word
#: rather than the same word meaning something else.
QP_WFN_SCHEME = "lorrax-qp-wfn-v1"

#: Additive method provenance shared by the compact rotations file and the
#: matched WFN.  These facts describe how the stored E/U pair was obtained;
#: they do not change the v1 matched-pair contract above.  The Sigma
#: evaluation spelling is imported from ``sigma_output``, which owns that
#: vocabulary for ``sigma_mnk.h5`` as well.
QP_SOLVER_ATTR = "qp_solver"
QP_ENERGY_DEFINITION_ATTR = "qp_energy_definition"


def _qp_provenance_attrs(*, qp_solver=None, qp_energy_definition=None,
                         sigma_eval_provenance=None) -> dict[str, str]:
    """Validate and return the all-or-none additive QP provenance attrs."""
    values = {
        QP_SOLVER_ATTR: qp_solver,
        QP_ENERGY_DEFINITION_ATTR: qp_energy_definition,
        SIGMA_EVAL_PROVENANCE_ATTR: sigma_eval_provenance,
    }
    present = [name for name, value in values.items() if value is not None]
    if present and len(present) != len(values):
        missing = [name for name, value in values.items() if value is None]
        raise ValueError(
            "QP artifact provenance is all-or-none; got "
            f"{present}, missing {missing}.")
    return ({name: str(value) for name, value in values.items()}
            if present else {})


#: Small top-level dataset carried by a restart bundle to say which WFN
#: supplied its matched ``psi_full_y`` / ``enk_full`` state.  The payload is
#: JSON owned and parsed in this module; restart I/O only transports the
#: opaque bytes through its incumbent SlabIO metadata path.
QP_STATE_SOURCE_DATASET = "qp_state_source_provenance"
QP_STATE_SOURCE_SCHEMA = 1


def _wedge_reduction(payload, kirr_to_kfull, star_tables):
    """``(reduced_payload, worst_by_name)`` — the round trip, MEASURED.

    Reduces every array in ``payload`` to the file wedge by taking the
    ``kirr_to_kfull`` rows, unfolds each straight back through the single
    adapter, and returns the reduced arrays beside the worst absolute
    deviation of the reconstruction from the array it came from.

    It is deliberately the WHOLE round trip and not a cheaper equivalence
    check.  What the caller needs to know is not "do these rows look like a
    star" but "will the reader that unfolds this file get back the bytes I
    am about to discard", and the only statement of that is the reader's own
    composition, run forwards.

    ``reduce`` is a plain row take rather than
    ``symmetry_maps.reduce_full_bz_to_file_wedge`` because the table is
    already in hand — ``kirr_to_kfull`` IS ``sym.kirr_fullids``, which is
    what that function selects by — and the writer has no ``SymMaps``.  The
    UNFOLD, which is the half with a convention in it, goes through
    ``kin_ion.broadcast_ibz_to_full_bz`` like every other unfold in the
    tree.
    """
    rows = np.asarray(kirr_to_kfull, dtype=np.int64)
    irr_idx_k, sym_idx_k, n_sym_spatial = star_tables
    reduced, worst = {}, {}
    for name, arr in payload.items():
        if arr is None:
            continue
        full = np.asarray(arr)
        red = full[rows]
        back = np.asarray(broadcast_ibz_to_full_bz(
            red, irr_idx_k, sym_idx_k, n_sym_spatial))
        reduced[name] = red
        worst[name] = (float(np.max(np.abs(back - full)))
                       if full.size else 0.0)
    return reduced, worst


def write_qp_rotations_h5(
    filepath: str,
    U_mnk: np.ndarray,
    E_qp_nk: np.ndarray,
    band_start: int,
    band_stop: int,
    kpoints_crys: np.ndarray,
    nkx: int,
    nky: int,
    nkz: int,
    kpoints_reduced: np.ndarray = None,
    kirr_to_kfull: np.ndarray = None,
    k_storage: str = "full",
    star_tables=None,
    source_wfn=None,
    print_fn=None,
    qp_solver=None,
    qp_energy_definition=None,
    sigma_eval_provenance=None,
):
    """Write QP rotation matrices and eigenvalues to HDF5 file.
    
    This file can be used to postprocess WFN.h5 → WFN_qp.h5 by rotating
    the G-vector coefficients and replacing eigenvalues.
    
    Args:
        filepath: Output path for the h5 file
        U_mnk: Unitary matrices (nk, nb, nb) where U[k,m,n] = <m_DFT|n_QP>
               To rotate coefficients: c_qp_n(G) = Σ_m U[k,m,n] c_dft_m(G)
        E_qp_nk: QP eigenvalues (nk, nb) in Hartree atomic units
        band_start: First band index (0-based) included in the calculation
        band_stop: One past last band index included
        kpoints_crys: Full k-mesh in crystal coordinates (nk, 3)
        nkx, nky, nkz: k-mesh dimensions
        kpoints_reduced: Reduced k-points from WFN.h5 (nk_red, 3), optional
        kirr_to_kfull: Mapping from reduced k-point index to full zone index, optional
        k_storage: one of :data:`QP_ROTATIONS_K_STORAGE`.  ``"full"`` (the
               default, and what every caller got before this argument
               existed) writes the arrays exactly as handed over.  ``"auto"``
               and ``"ibz"`` reduce them to the FILE WEDGE and stamp the
               file, and both need ``kirr_to_kfull`` and ``star_tables``.
        star_tables: ``(irr_idx_k, sym_idx_k, n_sym_spatial)`` — the tables
               the reader unfolds with, written INTO the file beside the
               arrays.  A table that lives elsewhere is a table that
               silently decays when anything upstream is regenerated.
        source_wfn: loaded mean-field WFN whose DFT-band basis defines
               ``U_mnk``.  Required: the writer computes the artifact's
               identity through :func:`common.parallel_transport.wfn_fingerprint`.
        print_fn: where the storage decision is announced, or ``None``.
        qp_solver, qp_energy_definition, sigma_eval_provenance: optional,
               all-or-none run provenance for the stored E/U pair.  Legacy
               callers may omit all three; absence means unverifiable.

    For postprocessing WFN.h5 → WFN_qp.h5:
        1. Load WFN.h5 coefficients for bands [band_start:band_stop]
        2. For each k-point k:
           c_qp[n, G] = Σ_m U[k, m, n] * c_dft[m, G]  (matrix form: c_qp = U^T @ c_dft)
        3. Replace eigenvalues with E_qp_nk (convert to Rydberg if needed)
        4. Write rotated coefficients back to WFN_qp.h5

    WHAT MOVES AND WHAT DOES NOT.  Only :data:`QP_ROT_K_DATASETS` is
    reduced.  :data:`QP_ROT_FULL_BZ_DATASETS` — ``kpoints_crys`` and
    ``kirr_to_kfull`` — stay on the full BZ and keep their exact old
    values and meaning, because the unfold is a GATHER and k is the one
    quantity in this file the symmetry operation changes.  So a consumer
    reads the physics arrays through :func:`read_qp_rotations_full_bz`
    and then indexes them by full-BZ k exactly as it always did.
    """
    provenance = _qp_provenance_attrs(
        qp_solver=qp_solver,
        qp_energy_definition=qp_energy_definition,
        sigma_eval_provenance=sigma_eval_provenance)
    if k_storage not in QP_ROTATIONS_K_STORAGE:
        raise ValueError(
            f"write_qp_rotations_h5: k_storage={k_storage!r} is none of "
            f"{QP_ROTATIONS_K_STORAGE}.")

    say = print_fn if print_fn is not None else (lambda *_a, **_k: None)
    if source_wfn is None:
        raise ValueError(
            "write_qp_rotations_h5 requires source_wfn: U_mnk is labelled "
            "in that WFN's DFT-band basis and an unstamped future artifact "
            "cannot be authenticated by a consumer.")
    from common.parallel_transport import (
        WFN_FINGERPRINT_SCHEME,
        wfn_fingerprint,
    )
    source_wfn_scheme = WFN_FINGERPRINT_SCHEME
    source_wfn_fingerprint = _require_wfn_fingerprint(
        wfn_fingerprint(source_wfn),
        where="write_qp_rotations_h5 canonical WFN fingerprint")
    payload = {
        "U_mnk": np.asarray(U_mnk),
        "E_qp_nk_hartree": np.asarray(E_qp_nk),
        "E_qp_nk_rydberg": np.asarray(E_qp_nk) * 2.0,
    }
    stored = K_STORAGE_FULL
    kirr_full_bz = (None if kirr_to_kfull is None
                    else np.asarray(kirr_to_kfull, dtype=np.int32))

    if k_storage != "full":
        # EVERY WAY THE REQUEST CANNOT BE HONOURED, NAMED SEPARATELY.  A
        # single "cannot reduce" would make the two very different causes
        # — no tables, and tables that do not reconstruct — look alike, and
        # only one of them is a reason to regenerate anything.
        missing = [n for n, v in (("kirr_to_kfull", kirr_to_kfull),
                                  ("star_tables", star_tables)) if v is None]
        if missing:
            raise ValueError(
                f"write_qp_rotations_h5: k_storage={k_storage!r} needs "
                f"{missing} and got None.  The wedge form is only writable "
                f"by a caller that also hands over the tables the reader "
                f"unfolds with; there is no re-derivation here, because a "
                f"table that reconstructs the tensor must be the table that "
                f"deconstructed it.")
        nk_full = int(payload["U_mnk"].shape[0])
        irr = np.asarray(star_tables[0], dtype=np.int32)
        if irr.size != nk_full:
            raise ValueError(
                f"write_qp_rotations_h5: irr_idx_k describes {irr.size} "
                f"full-BZ k but the arrays carry {nk_full} rows — the "
                f"tables and the arrays are not the same calculation.")
        # THE READER'S OWN CONSISTENCY CONDITION, CHECKED AT THE WRITER.
        # ``kin_ion.read_star_map`` refuses a file whose stored k extent does
        # not equal ``irr_idx_k.max() + 1``, and that can fail here without
        # the round trip noticing: the round trip only reads rows the tables
        # POINT AT, so a file-wedge row that is never an orbit parent — the
        # register's ``cohsex_debug`` case, where row 1 is the time-reverse of
        # row 2 — is reconstructed fine and still leaves a table the reader
        # will not accept.  Refusing here is the difference between a writer
        # that cannot produce an unreadable file and one that merely usually
        # does not.
        n_star = int(irr.max(initial=-1)) + 1
        nk_red = int(np.asarray(kirr_full_bz).size)
        reasons = []
        if n_star != nk_red:
            reasons.append(
                f"the file wedge has {nk_red} rows but irr_idx_k names only "
                f"{n_star} distinct parents, so {nk_red - n_star} stored k "
                f"are never an orbit parent and file_io.kin_ion.read_star_map "
                f"would refuse the file — it cannot tell that from a "
                f"truncated slab")
        reduced, worst = _wedge_reduction(payload, kirr_full_bz, star_tables)
        bad = {n: d for n, d in worst.items() if d != 0.0}
        if bad:
            reasons.append(
                "the full-BZ rows are not the unfold of the wedge rows ("
                + ", ".join(f"{n} max|Δ| = {d:.6e}"
                            for n, d in sorted(bad.items())) + ")")
        if reasons:
            detail = "; ".join(reasons)
            if k_storage == "ibz":
                raise ValueError(
                    f"write_qp_rotations_h5: k_storage='ibz' was asked for "
                    f"and the wedge form is not writable — {detail}.  Storing "
                    f"it would discard rows no reader can rebuild.  U_mnk is "
                    f"a stack of EIGENVECTORS — defined up to a phase, and up "
                    f"to a unitary mixing inside a degenerate multiplet — so "
                    f"a round-trip failure is the expected answer for a run "
                    f"whose off-wedge rows came from their own eigh rather "
                    f"than from a broadcast.  Use k_storage='auto' to fall "
                    f"back to full-BZ storage.")
            say(f"  qp_wfn_rotations: k_storage='auto' -> FULL BZ; {detail}.")
        else:
            stored = K_STORAGE_IBZ
            payload = reduced
            say(f"  qp_wfn_rotations: k axis REDUCED to the file wedge, "
                f"{nk_full} -> {len(kirr_full_bz)} rows; the round trip "
                f"reproduces every dataset exactly (max|Δ| = 0).")

    with h5py.File(filepath, 'w') as f:
        # Main data
        f.create_dataset('U_mnk', data=payload["U_mnk"], dtype=np.complex128)
        f.create_dataset('E_qp_nk_hartree', data=payload["E_qp_nk_hartree"], dtype=np.float64)
        f.create_dataset('E_qp_nk_rydberg', data=payload["E_qp_nk_rydberg"], dtype=np.float64)  # Also save in Ry

        # Metadata
        f.create_dataset('band_range', data=np.array([band_start, band_stop], dtype=np.int32))
        f.create_dataset('kpoints_crys', data=kpoints_crys, dtype=np.float64)
        f.create_dataset('kgrid', data=np.array([nkx, nky, nkz], dtype=np.int32))

        # Optional: reduced k-points and mapping for easy WFN.h5 lookup
        if kpoints_reduced is not None:
            f.create_dataset('kpoints_reduced', data=kpoints_reduced, dtype=np.float64)
        if kirr_to_kfull is not None:
            f.create_dataset('kirr_to_kfull', data=kirr_to_kfull, dtype=np.int32)

        # ---- the k-basis declaration -------------------------------------
        # Only on the wedge arm.  A full-BZ file is left EXACTLY as it was,
        # attrs included, so "absent means full" keeps its meaning and this
        # change cannot be detected downstream of a `full` run at all.
        if stored == K_STORAGE_IBZ:
            irr_idx_k, sym_idx_k, n_sym_spatial = star_tables
            f.create_dataset(IRR_IDX_DATASET,
                             data=np.asarray(irr_idx_k, dtype=np.int32))
            f.create_dataset(SYM_IDX_DATASET,
                             data=np.asarray(sym_idx_k, dtype=np.int32))
            for name in QP_ROT_K_DATASETS:
                d = f[name]
                d.attrs[K_STORAGE_ATTR] = K_STORAGE_IBZ
                d.attrs[K_STORAGE_VERSION_ATTR] = K_STORAGE_VERSION
                d.attrs[N_SYM_SPATIAL_ATTR] = int(n_sym_spatial)
                d.attrs['nk_full'] = int(np.asarray(irr_idx_k).size)

        # Attributes for documentation
        f.attrs['description'] = (
            'QP rotation data for transforming DFT wavefunctions to QP basis. '
            'U_mnk[k,m,n] = <m_DFT|n_QP>. '
            'To rotate: c_qp[n,G] = sum_m U[k,m,n] * c_dft[m,G] (i.e. c_qp = U^T @ c_dft)'
        )
        f.attrs['energy_units'] = 'E_qp_nk_hartree in Hartree, E_qp_nk_rydberg in Rydberg'
        f.attrs['band_convention'] = '0-based indexing; bands [band_start, band_stop) were computed'
        for name, value in provenance.items():
            f.attrs[name] = value
        f.attrs[QP_ROT_WFN_FINGERPRINT_SCHEME_ATTR] = source_wfn_scheme
        f.attrs[QP_ROT_WFN_FINGERPRINT_ATTR] = source_wfn_fingerprint
        if kirr_to_kfull is not None:
            f.attrs['mapping_description'] = (
                'kirr_to_kfull[ik_red] gives the index into kpoints_crys/U_mnk/E_qp_nk '
                'for the reduced k-point ik_red from WFN.h5'
            )
    return stored


# ---------------------------------------------------------------------------
# Reading it back on the full BZ
# ---------------------------------------------------------------------------

def qp_rotations_k_storage(h5_path: str, dataset: str = "U_mnk") -> str:
    """``"ibz"`` or ``"full"`` — what ``dataset`` of this file is stored on.

    Absent attr means :data:`~file_io.kin_ion.K_STORAGE_FULL`, which is what
    makes every file written before this format keep its exact meaning.
    """
    with h5py.File(h5_path, "r") as f:
        if dataset not in f:
            raise KeyError(f"Dataset {dataset!r} missing from {h5_path}")
        stored = str(f[dataset].attrs.get(K_STORAGE_ATTR, K_STORAGE_FULL))
    if stored not in K_STORAGE_VALUES:
        raise ValueError(
            f"{os.path.basename(h5_path)}: {dataset}.{K_STORAGE_ATTR} is "
            f"{stored!r}, neither {K_STORAGE_IBZ!r} nor {K_STORAGE_FULL!r}.")
    return stored


def read_qp_rotations_full_bz(h5_path: str, datasets=None) -> dict:
    """``qp_wfn_rotations.h5``'s k-indexed arrays, ON THE FULL BZ.

    THE unfolding option the wedge form owes its consumers, and the reason
    the wedge form is safe to write at all: anything that wants the array
    the old writer produced calls this and gets it, wedge-stored file or
    not.  A full-BZ file is read verbatim — the unfold is not attempted,
    because the tables are not there and the rows are not stars.

    Reuses ``kin_ion.read_star_map`` for the stamp contract rather than
    re-implementing it, so the version number, the table names and every
    refusal have ONE definition across ``kin_ion.h5``, ``sigma_mnk.h5`` and
    this file.
    """
    from .kin_ion import read_star_map
    names = tuple(datasets) if datasets is not None else QP_ROT_K_DATASETS
    star = read_star_map(h5_path, names[0])
    out = {}
    with h5py.File(h5_path, "r") as f:
        for name in names:
            if name not in f:
                continue
            arr = np.asarray(f[name][()])
            out[name] = (arr if star is None
                         else np.asarray(broadcast_ibz_to_full_bz(arr, *star)))
    return out


def read_qp_rotations_artifact(h5_path: str) -> dict:
    """Read one complete ``qp_wfn_rotations.h5`` Hamiltonian artifact.

    The physics arrays are unfolded through
    :func:`read_qp_rotations_full_bz`, so wedge and full-BZ storage retain
    one meaning.  The small identity datasets are read here as part of the
    same public format contract rather than independently in every physics
    driver.

    Returns ``U_mnk`` and ``E_qp_nk_rydberg`` on the full BZ together with
    ``band_range``, ``kpoints_crys``, ``kgrid`` and the optional legacy/source
    WFN fingerprint pair.  A partial artifact is refused: a rotation without
    its matched eigenvalues, band labels or k-set cannot define
    ``H_QP = U diag(E_QP) U^H``; one fingerprint attribute without the other
    cannot define which identity scheme was used.
    """
    path = os.fspath(h5_path)
    arrays = read_qp_rotations_full_bz(
        path, datasets=("U_mnk", "E_qp_nk_rydberg"))
    missing = [name for name in ("U_mnk", "E_qp_nk_rydberg")
               if name not in arrays]
    with h5py.File(path, "r") as h5:
        missing.extend(name for name in QP_ROT_METADATA_DATASETS
                       if name not in h5)
        if missing:
            raise ValueError(
                f"{os.path.basename(path)} is not a complete QP rotation "
                f"artifact; missing {sorted(set(missing))}.")
        arrays.update({
            "band_range": np.asarray(h5["band_range"][()], dtype=np.int64),
            "kpoints_crys": np.asarray(
                h5["kpoints_crys"][()], dtype=np.float64),
            "kgrid": np.asarray(h5["kgrid"][()], dtype=np.int64),
        })
        has_scheme = QP_ROT_WFN_FINGERPRINT_SCHEME_ATTR in h5.attrs
        has_fingerprint = QP_ROT_WFN_FINGERPRINT_ATTR in h5.attrs
        if has_scheme != has_fingerprint:
            raise ValueError(
                f"{os.path.basename(path)} has an incomplete source-WFN "
                "identity: fingerprint and scheme attributes must appear "
                "together.")
        if has_fingerprint:
            def _text(value):
                return value.decode("ascii") if isinstance(value, bytes) \
                    else str(value)
            scheme = _text(h5.attrs[QP_ROT_WFN_FINGERPRINT_SCHEME_ATTR])
            fingerprint = _text(h5.attrs[QP_ROT_WFN_FINGERPRINT_ATTR])
            fingerprint = _require_wfn_fingerprint(
                fingerprint,
                where=(f"{os.path.basename(path)} source-WFN fingerprint"))
        else:
            scheme = fingerprint = None
        arrays["source_wfn_fingerprint_scheme"] = scheme
        arrays["source_wfn_fingerprint"] = fingerprint
    return arrays


def authenticate_qp_rotations_source_wfn(
        artifact: dict, source_wfn, *, artifact_path: str) -> str:
    """Require that one QP rotation artifact names its exact DFT basis.

    Every consumer that applies ``U_mnk`` calls this owner after
    :func:`read_qp_rotations_artifact`.  The comparison deliberately uses
    the already-loaded WFN and the sole repository fingerprint service; a
    filename, k-grid, or array shape is not a DFT-band-basis identity.
    """
    name = os.path.basename(os.fspath(artifact_path))
    scheme = artifact.get("source_wfn_fingerprint_scheme")
    fingerprint = artifact.get("source_wfn_fingerprint")
    if scheme is None or fingerprint is None:
        raise ValueError(
            f"{name} has no authenticated source-WFN fingerprint.  "
            "Regenerate the QP rotation artifact through the canonical "
            "writer with the mean-field WFN that defines U_mnk; matching "
            "filenames, k grids, and band counts do not prove the DFT-band "
            "basis identity.")
    from common.parallel_transport import (
        WFN_FINGERPRINT_SCHEME,
        wfn_fingerprint,
    )
    if scheme != WFN_FINGERPRINT_SCHEME:
        raise ValueError(
            f"{name} source-WFN fingerprint scheme {scheme!r} does not "
            "match the installed canonical scheme "
            f"{WFN_FINGERPRINT_SCHEME!r}.")
    expected = _require_wfn_fingerprint(
        wfn_fingerprint(source_wfn),
        where="installed canonical WFN fingerprint")
    if fingerprint != expected:
        raise ValueError(
            f"{name} was produced from a different mean-field WFN "
            f"(source fingerprint {fingerprint}, selected WFN fingerprint "
            f"{expected}).  U_mnk is expressed in the source WFN's DFT-band "
            "basis and may not be applied by shape.")
    return expected


# ---------------------------------------------------------------------------
# Full WFN.h5 with rotated ψ + replaced energies
# ---------------------------------------------------------------------------

def write_qp_wfn_h5(
    output_path: str,
    wfn,                                    # WFNReader (also serves as `crystal` for the writer)
    U_kmn: np.ndarray,                      # (nk, nb_active, nb_active)  ⟨DFT_m | QP_n⟩
    enk_active_qp_ry: np.ndarray,           # (nk, nb_active)             E_QP for active block, Ry
    band_start: int,
    band_stop: int,
    *,
    enk_full_base_ry: np.ndarray | None = None,
    qp_solver=None,
    qp_energy_definition=None,
    sigma_eval_provenance=None,
) -> None:
    """Write a BGW-compatible WFN.h5 with QP-rotated ψ and replaced energies.

    For each k:
      * Bands ``[band_start, band_stop)`` (the "active" block):
          ``c_qp[n, s, G] = Σ_m U[k, m, n] · c_dft[m, s, G]``
          ``E[n] ← enk_active_qp_ry[k, n - band_start]``
      * All other coefficients remain DFT.  Their energies default to DFT,
        or may be supplied through ``enk_full_base_ry`` when a caller owns
        an explicit energy-only extrapolation such as the SC sum-band tail.

    The ``wfn`` argument is a :class:`~wfn_loader.WfnLoader` and
    is reused as the ``crystal`` source for :class:`WFNWriter` (it
    exposes the same ``nspin``, ``nspinor``, ``nelec``, ``ecutwfc``,
    ``ecutrho``, ``fft_grid``, ``avec``, ``bdot``, … attributes the
    writer reads).

    Notes
    -----
    Symmetry & k-mesh: the rotation is on the irreducible-k WFN —
    output is on the same irreducible-k grid as the input, with the
    ``mtrx`` / ``tnp`` blocks copied through.  Symmetry-equivalent
    full-zone wavefunctions are reconstructed by downstream consumers
    (BSE, etc.) via the same maps as for the input WFN.

    Spinors: handled identically per (k, s) — the rotation is in band
    space and does not mix spinor components.

    Occupations: ``ifmin`` / ``ifmax`` are inherited from the source
    WFN via :class:`WFNWriter` (which sets ``ifmax = nelec`` on every
    k).  Safe whenever the SC iteration preserves the overall
    valence/conduction band ordering — which holds for insulators with
    QP shifts smaller than the gap.  For metals or near-gap-closure
    systems, an SC update can permute occupied vs. empty bands; the
    output ``occ`` array would then be wrong and would need to be
    recomputed from the QP energies + a fresh midgap E_F before
    handing the file to a downstream consumer that trusts ``occ``.
    """
    provenance = _qp_provenance_attrs(
        qp_solver=qp_solver,
        qp_energy_definition=qp_energy_definition,
        sigma_eval_provenance=sigma_eval_provenance)

    from .wfn_writer import WFNWriter

    nb_active = int(band_stop - band_start)
    if U_kmn.shape != (wfn.nkpts, nb_active, nb_active):
        raise ValueError(
            f"write_qp_wfn_h5: U shape {U_kmn.shape} inconsistent with "
            f"(nk={wfn.nkpts}, nb_active={nb_active}).")
    if enk_active_qp_ry.shape != (wfn.nkpts, nb_active):
        raise ValueError(
            f"write_qp_wfn_h5: enk_active_qp_ry shape "
            f"{enk_active_qp_ry.shape} inconsistent with "
            f"(nk={wfn.nkpts}, nb_active={nb_active}).")

    # All-band IBZ coefficients + per-k G-vectors via the unified loader.
    # The output writer is already k-streamed, so the input must be too:
    # materialising ``(nk, nbands, ns, ngkmax)`` first made this optional
    # end-of-run artifact a whole-WFN device allocation after Sigma.  Keep
    # exactly one raw IBZ row live and rotate it on the host.
    if enk_full_base_ry is None:
        enk_full_ry = np.array(
            wfn.energies[0], dtype=np.float64).copy()  # (nk, nbands)
    else:
        base = np.asarray(enk_full_base_ry, dtype=np.float64)
        expected = (int(wfn.nkpts), int(wfn.nbands))
        if base.shape != expected:
            raise ValueError(
                "write_qp_wfn_h5: enk_full_base_ry shape "
                f"{base.shape} inconsistent with {expected}.")
        enk_full_ry = base.copy()
    enk_full_ry[:, band_start:band_stop] = np.asarray(
        enk_active_qp_ry, dtype=np.float64)

    # The top-level WfnLoader carries the device mesh; ``.load`` is a
    # collective on the phdf5 backend, so it MUST be called by every rank
    # even though only rank-0 writes the file.  We open a fresh
    # mesh-less WfnLoader (eager backend) here so the rank-0 write does
    # not need the other ranks at all — qp_wfn is a one-shot
    # end-of-run dump and the re-slurp cost is paid once.
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import IBZRows, WfnLoader
    with WfnLoader(wfn.path) as loader:
        gvecs_full = loader.gvecs(k="ibz")                     # (nk, ngkmax, 3)
        ngk_v = loader.ngk_valid(k="ibz")                      # (nk,)
        gvecs_per_k = [gvecs_full[ik, : int(ngk_v[ik])]
                       for ik in range(int(wfn.nkpts))]

        with WFNWriter(
            output_path, wfn,
            kpoints=np.asarray(wfn.kpoints, dtype=np.float64),
            weights=np.asarray(wfn.kweights, dtype=np.float64),
            kgrid=tuple(int(x) for x in wfn.kgrid),
            nbands=int(wfn.nbands),
            gvecs_per_k=gvecs_per_k,
            nosym=False,
            shift=tuple(float(x) for x in wfn.shift),
        ) as writer:
            for ik in range(int(wfn.nkpts)):
                n = int(ngk_v[ik])
                psi_k = loader.load(
                    bands=(0, int(wfn.nbands)),
                    k=IBZRows((ik,)), sharding=None)
                c_all_dft = np.asarray(psi_k)[0, :, :, :n].copy()
                del psi_k
                c_active_dft = c_all_dft[band_start:band_stop]
                c_all_dft[band_start:band_stop] = np.einsum(
                    "mn,msg->nsg", U_kmn[ik], c_active_dft,
                    optimize=True)
                writer.write_k(ik, enk_full_ry[ik], c_all_dft)

    # THE FILE SAYS WHAT IT IS.  ψ and E in here are a MATCHED PAIR: the
    # rotated orbitals carry the QP eigenvalues that produced the rotation,
    # and applying a second, DFT-band-labelled QP ladder on top of them
    # (``exciton_bands --eqp``) silently discards the canonical ones and
    # relabels the rotated orbitals with someone else's ordering.  Nothing on
    # disk could tell a consumer that this WFN.h5 is not mean-field, so the
    # only discriminator available was the filename.  Stamped here, in the one
    # writer, using the same "an absent attr means the old thing" reading the
    # k_storage stamps above use.
    with h5py.File(str(output_path), "a") as h5:
        h5.attrs[QP_WFN_ATTR] = QP_WFN_SCHEME
        h5.attrs["qp_wfn_band_start"] = int(band_start)
        h5.attrs["qp_wfn_band_stop"] = int(band_stop)
        h5.attrs["qp_wfn_source"] = str(getattr(wfn, "path", "") or "")
        for name, value in provenance.items():
            h5.attrs[name] = value



def read_qp_wfn_stamp(path) -> dict | None:
    """Is ``path`` a LORRAX QP WFN?  ``None`` when the file does not say.

    Returns the stamp written by :func:`write_qp_wfn_h5` — ``scheme``,
    ``band_start``, ``band_stop``, ``source`` and optional method provenance
    — or ``None``.

    THREE OUTCOMES, AND ONLY ONE OF THEM IS "NO".  A missing file, an
    unreadable one, and one with no stamp all return ``None``, which means
    **unverifiable**: BerkeleyGW's ``pw2bgw`` output, every WFN.h5 written
    before this stamp, and a QP WFN produced by some other tool are
    indistinguishable here, and a consumer must not read ``None`` as proof
    that the file is mean-field.  A consumer may only use a POSITIVE answer
    to refuse; the absence licenses nothing (``TASTE.md``: an absence is a
    claim about what was searched).

    An unrecognised scheme string is returned as-is rather than mapped onto
    the current one — a caller comparing it against
    :data:`QP_WFN_SCHEME` can then say "this file was written by a different
    version" instead of silently accepting it.
    """
    try:
        with h5py.File(str(path), "r") as h5:
            raw = h5.attrs.get(QP_WFN_ATTR)
            if raw is None:
                return None
            scheme = raw.decode() if isinstance(raw, bytes) else str(raw)
            def _get(name, default=None):
                v = h5.attrs.get(name, default)
                if isinstance(v, bytes):
                    return v.decode()
                return v
            return {
                "scheme": scheme,
                "band_start": (None if _get("qp_wfn_band_start") is None
                               else int(_get("qp_wfn_band_start"))),
                "band_stop": (None if _get("qp_wfn_band_stop") is None
                              else int(_get("qp_wfn_band_stop"))),
                "source": _get("qp_wfn_source", "") or "",
                "qp_solver": _get(QP_SOLVER_ATTR),
                "qp_energy_definition": _get(QP_ENERGY_DEFINITION_ATTR),
                "sigma_eval_provenance": _get(
                    SIGMA_EVAL_PROVENANCE_ATTR),
            }
    except (OSError, KeyError):
        return None


def _validate_qp_state_source(record, *, path: str) -> dict:
    """Return one canonical restart source-state record or refuse it."""
    from common.parallel_transport import WFN_FINGERPRINT_SCHEME

    try:
        fingerprint = record["wfn_fingerprint"]
        stamp = record["qp_wfn_stamp"]
        valid = (set(record) == {
            "schema", "wfn_fingerprint_scheme", "wfn_fingerprint",
            "qp_wfn_stamp"} and isinstance(fingerprint, str)
            and len(fingerprint) == 64 and int(fingerprint, 16) >= 0
            and record.get("schema") == QP_STATE_SOURCE_SCHEMA
            and record.get("wfn_fingerprint_scheme")
            == WFN_FINGERPRINT_SCHEME)
        valid = valid and (stamp is None or (
            isinstance(stamp, dict) and set(stamp) == {
                "scheme", "band_start", "band_stop", "source"}))
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{path}: invalid {QP_STATE_SOURCE_DATASET} record")
    return record


def read_qp_state_source_provenance(path) -> dict | None:
    """Read a restart source-state record; ``None`` means legacy/unproven."""
    try:
        with h5py.File(str(path), "r") as h5:
            if QP_STATE_SOURCE_DATASET not in h5:
                return None
            raw = h5[QP_STATE_SOURCE_DATASET][()]
        payload = raw if isinstance(raw, bytes) else np.asarray(raw).tobytes()
        record = json.loads(payload.decode("utf-8", "strict").rstrip("\x00"))
    except (OSError, KeyError):
        return None
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(
            f"{path}: {QP_STATE_SOURCE_DATASET} is not valid UTF-8 JSON") \
            from exc
    return _validate_qp_state_source(record, path=str(path))


def qp_state_source_provenance(wfn) -> dict:
    """Describe the already-loaded WFN which supplied restart ``psi/E``."""
    from common.parallel_transport import bind_wfn_fingerprint
    return qp_state_source_provenance_from_binding(
        wfn, wfn_fingerprint_binding=bind_wfn_fingerprint(wfn))


def qp_state_source_provenance_from_binding(
        wfn, *, wfn_fingerprint_binding) -> dict:
    """Describe ``wfn`` from a canonical digest bound to this exact object."""
    from common.parallel_transport import (
        WFN_FINGERPRINT_SCHEME,
        fingerprint_from_binding,
    )
    source_path = getattr(wfn, "path", None)
    if not source_path:
        raise ValueError("QP restart provenance requires a path-bearing WFN")
    return {
        "schema": QP_STATE_SOURCE_SCHEMA,
        "wfn_fingerprint_scheme": WFN_FINGERPRINT_SCHEME,
        "wfn_fingerprint": _require_wfn_fingerprint(
            fingerprint_from_binding(wfn_fingerprint_binding, wfn),
            where="QP restart provenance bound WFN fingerprint"),
        "qp_wfn_stamp": read_qp_wfn_stamp(source_path),
    }


def encode_qp_state_source_provenance(record) -> np.ndarray:
    """Opaque scalar bytes for the restart writer's existing metadata path."""
    record = _validate_qp_state_source(record, path="restart writer")
    text = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return np.asarray(text.encode("utf-8"), dtype="S")


def _qp_state_source_from_path(path) -> dict:
    """Fingerprint one WFN through the canonical loader and hash owner."""
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader

    wfn = WfnLoader(path)
    try:
        return qp_state_source_provenance(wfn)
    finally:
        close = getattr(wfn, "close", None)
        if close is not None:
            close()


def _refuse_conflicting_qp_state_sources(
        *, wfn_path: str, eqp_file: str | None = None,
        qp_rotations_file: str | None = None,
        state_artifact_path: str | None = None,
        where: str = "QP-state consumer",
        _selected_source=None) -> dict | None:
    """Implement the QP-state refusal and return an authenticated record.

    A positively stamped QP WFN already contains the matched rotated orbitals
    and eigenvalues.  Applying either a DFT-labelled diagonal eqp ladder or a
    second rotation artifact changes that Hamiltonian while preserving every
    array shape.  Conversely, a mean-field/unverifiable WFN may consume one
    explicit QP source when no restart carrier also needs association.  At a
    restart join, missing legacy provenance refuses an external QP source
    because its band labels cannot be proved to index the stored ``psi/E``.

    Association is established only by explicit arguments and content
    stamps.  In particular, this owner never infers a relationship from a
    filename or sibling directory.  A rotation consumer separately requires
    its source-WFN fingerprint through
    :func:`authenticate_qp_rotations_source_wfn`; when
    ``state_artifact_path`` is supplied here, the restart carrier's canonical
    content fingerprint is compared with the selected WFN.  A missing legacy
    restart record remains usable only without an external or positive QP
    description.

    ``_selected_source`` is an internal lazy callback used by the GW restart
    owner.  Its result is still built by this module from an opaque canonical
    WFN binding; no digest value crosses the caller API.
    """
    if eqp_file and qp_rotations_file:
        raise ValueError(
            "The diagonal eqp override and QP rotation artifact are mutually "
            "exclusive state descriptions: the first changes energies in "
            "the WFN's current DFT band LABELS, while the second supplies "
            "the matched U_mnk,E_qp for H_QP = U diag(E_qp) U^H.  Select "
            "exactly one.")
    external = eqp_file or qp_rotations_file
    wfn_stamp = read_qp_wfn_stamp(wfn_path)
    stamp = wfn_stamp
    authenticated_restart_source = None

    if state_artifact_path is not None:
        provenance = read_qp_state_source_provenance(state_artifact_path)
        if provenance is None:
            if external or wfn_stamp is not None:
                requested = (f"diagonal eqp override {eqp_file}"
                             if eqp_file else
                             (f"QP rotation artifact {qp_rotations_file}"
                              if qp_rotations_file else
                              f"stamped QP WFN {wfn_path}"))
                raise ValueError(
                    f"{where}: {requested} cannot be combined with "
                    f"{state_artifact_path}: that restart predates "
                    f"{QP_STATE_SOURCE_DATASET} and cannot prove which WFN "
                    "supplied its matched psi_full_y/enk_full state.  "
                    "Regenerate the restart from the selected WFN.  A legacy "
                    "restart remains usable only without an external or "
                    "positively stamped QP state.")
            return

        selected = (
            _qp_state_source_from_path(wfn_path)
            if _selected_source is None else _selected_source())
        stamp = provenance["qp_wfn_stamp"]
        if provenance["wfn_fingerprint"] != selected["wfn_fingerprint"]:
            raise ValueError(
                f"{where}: restart {state_artifact_path} and selected WFN "
                f"{wfn_path} have different canonical content fingerprints. "
                "They do not carry the same psi/E representation; regenerate "
                "the restart from the selected WFN.")
        if stamp != selected["qp_wfn_stamp"]:
            raise ValueError(
                f"{where}: restart {state_artifact_path} has a torn QP-state "
                f"record for {wfn_path}: its content fingerprint matches but "
                "its QP-WFN stamp does not.")
        authenticated_restart_source = provenance

    if not external:
        return authenticated_restart_source
    if stamp is None:
        return authenticated_restart_source
    requested = (f"diagonal eqp override {eqp_file}"
                 if eqp_file else
                 f"QP rotation artifact {qp_rotations_file}")
    band_where = ""
    if stamp.get("band_stop") is not None:
        band_where = (
            f", bands [{stamp['band_start']}, {stamp['band_stop']}) "
            f"rotated from {stamp['source'] or 'an unrecorded WFN'}")
    version_note = ""
    if stamp["scheme"] != QP_WFN_SCHEME:
        version_note = (
            f"  That stamp is not {QP_WFN_SCHEME!r}: the file was written "
            "by a different version of the QP writer, which is still a "
            "positive QP-state identification rather than permission to "
            "stack another source.")
    fix = ("Fix: drop --eqp.  To run a mean-field WFN with diagonal QP "
           "corrections instead, select the mean-field WFN and keep --eqp."
           if eqp_file else
           "Fix: drop --qp-rotations and use this QP WFN by itself, or "
           "select the mean-field WFN before applying the rotation artifact.")
    raise ValueError(
        f"{requested} cannot be applied to {wfn_path}: it is already a "
        f"LORRAX QP WFN (stamp {stamp['scheme']!r}{band_where}).  Its "
        "wavefunctions ARE the QP orbitals and its energies ARE the matched "
        "QP eigenvalues.  A second eqp ladder is written against DFT band "
        "LABELS, and a second U rotates the state twice; either operation "
        "silently changes the represented Hamiltonian with the right "
        f"shapes.  {fix}{version_note}")


def refuse_conflicting_qp_state_sources(
        *, wfn_path: str, eqp_file: str | None = None,
        qp_rotations_file: str | None = None,
        state_artifact_path: str | None = None,
        where: str = "QP-state consumer") -> None:
    """Refuse two explicit descriptions of one quasiparticle state.

    This established public guard deliberately remains verdict-only.  The GW
    restart orchestrator uses
    :func:`authenticate_restart_qp_state_source_for_wfn` when it also needs
    the exact record and bound loaded-WFN identity used by the comparison.
    """
    _refuse_conflicting_qp_state_sources(
        wfn_path=wfn_path, eqp_file=eqp_file,
        qp_rotations_file=qp_rotations_file,
        state_artifact_path=state_artifact_path, where=where)


def authenticate_restart_qp_state_source_for_wfn(
        *, wfn, state_artifact_path: str,
        where: str = "QP-state consumer"):
    """Authenticate one restart and return ``(record, WFN binding)``.

    A legacy restart returns ``(None, None)`` without scanning the WFN.  A
    stamped restart scans the exact already-loaded WFN once, compares that
    bound identity with the record, and returns the same opaque binding for
    receipt construction.  The digest string itself never enters caller
    control.
    """
    wfn_path = getattr(wfn, "path", None)
    if not wfn_path:
        raise ValueError(
            f"{where}: selected WFN has no path, so restart psi/E "
            "compatibility cannot be checked")
    binding_box = []

    def selected_source():
        from common.parallel_transport import bind_wfn_fingerprint
        binding = bind_wfn_fingerprint(wfn)
        binding_box.append(binding)
        return qp_state_source_provenance_from_binding(
            wfn, wfn_fingerprint_binding=binding)

    record = _refuse_conflicting_qp_state_sources(
        wfn_path=wfn_path, state_artifact_path=state_artifact_path,
        where=where, _selected_source=selected_source)
    binding = binding_box[0] if binding_box else None
    if (record is None) != (binding is None):
        raise AssertionError(
            "restart source authentication returned a torn host binding")
    return record, binding
