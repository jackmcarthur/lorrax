"""Deck-driven shared real-pole screening (DESIGN §3, rulings15–16).

The response, constructor and persistence owners implement their own stages.
This helper authenticates the current map and passes handles between stages:
files, or the device-resident bank and SC model when they fit. Samples,
directions and poles are rebuilt across changed maps; the SC owner may retain
time nodes after current-domain certification.
"""
from pathlib import Path
import dataclasses
from functools import partial
import hashlib
import json
import shutil
import time

import numpy as np
from common import timing


def _print_eigh_stack_routes(print_fn):
    """The route distrib_la chose for each new eigh stack shape, one line each."""
    import distrib_la
    for line in distrib_la.new_stack_routes():
        print_fn("  distrib_la " + line)


def _json(value):
    def encode(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, complex):
            return dict(real=x.real, imag=x.imag)
        from file_io.shared_pole_store import ResidentSectorModel
        if isinstance(x, ResidentSectorModel):
            return str(x)
        raise TypeError(type(x).__name__)
    return json.dumps(value, default=encode, sort_keys=True, allow_nan=False)


def _authenticated_constructor_resume(root, identity, recipe, *, photon=False, coulomb_sha256=None):
    """Authenticate a complete producer bank before preserving it.

    This is deliberately narrower than restart: it resumes only the missing
    constructor after the producer receipts and store validator bind the exact
    current map identity, resolved recipe, response convention and final commit.
    Photon moments are in the bank receipt; scalar moments have a second receipt,
    and both bind the bare V digest ``coulomb_sha256`` (``_coulomb_resource``).
    Any other partial directory follows the existing remove-and-rebuild path,
    including a bank built against another V digest (another P, older code).
    """
    from file_io.shared_pole_store import validate_shared_pole_bank

    receipt_paths = ((root / 'bank_receipt.json',) if photon else
                     (root / 'bank_receipt.json', root / 'moments_receipt.json'))
    bank_path = root / 'bank.h5'
    required = (*receipt_paths, bank_path)
    if not all(path.is_file() for path in required):
        return False
    try:
        bank_receipt = json.loads(receipt_paths[0].read_text())
        moments_receipt = None if photon else json.loads(receipt_paths[1].read_text())
    except (OSError, ValueError):
        return False
    if bank_receipt.get('identity') != identity or bank_receipt.get('completion') is not True:
        return False
    if photon:
        if (bank_receipt.get('stage') != 'photon'
                or bank_receipt.get('bank_complete') is not True):
            return False
    elif (moments_receipt.get('identity') != identity
          or moments_receipt.get('completion') is not True
          or moments_receipt.get('bank_complete') is not True
          or bank_receipt.get('coulomb_identity') != moments_receipt.get('coulomb_identity')
          or (bank_receipt.get('coulomb_identity') or {}).get('sha256') != coulomb_sha256):
        return False
    try:
        header = validate_shared_pole_bank(
            bank_path, expected_identity=identity, mesh_xy=None,
            require_complete=True, expected_recipe=recipe)
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return ('photon_layout' in header) == photon


# What the sector constructor publishes into a map directory.
_SECTOR_OUTPUTS = ('sectors.json', 'CC.h5', 'TT.h5', 'CT_C.h5', 'CT_T.h5')


def _published_sector_handle(root, identity):
    """The sector handle an interrupted run published for this map, or None.

    A run killed after its constructor leaves ``sectors.json``. Its handle is
    reused when the manifest binds this map's identity (label, WFN, energies,
    occupations, centroids, recipe_hash) and every model and the constant it
    names is a file in this directory; Sigma's manifest validation then
    authenticates each digest on every rank. Models that were device-resident
    died with the run, so that manifest gives None and the caller rebuilds. A
    manifest of another identity refuses: it is not this map's private state.
    """
    manifest = root / 'sectors.json'
    try:
        header = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return None
    if header.get('identity') != identity:
        raise ValueError(f'GATE shared_pole_output: sector manifest at {manifest} binds another '
                         'map, recipe or WFN; use a fresh run directory')
    paths = [row.get('path') for row in header.get('sectors', {}).values()]
    paths.append(header.get('constant', {}).get('path'))
    own = root.resolve()
    if len(paths) != 5 or not all(isinstance(p, str) and Path(p).parent == own
                                  and Path(p).is_file() for p in paths):
        return None
    return dict(path=str(manifest.resolve()), identity=header['identity'],
                digest=header['digest'], representation=header['representation'],
                sectors=header['sectors'], constant=header['constant'])


def _mark_photon_head(handle, config):
    """Name the direct four-current Γ head on a photon handle when heads are on."""
    from .gw_config import HeadCorrection, uses_direct_bispinor_shared_pole_head
    if config.head.correction is HeadCorrection.OFF:
        return
    if not uses_direct_bispinor_shared_pole_head(config):
        raise ValueError("GATE shared_pole_photon_head: unsupported photon Γ policy")
    handle["direct_photon_head"] = "first_order_cc_ct_tc_tt"


def shared_pole_identity(wfns, meta, *, label, wfn, binding, centroid_indices,
                         charge_zeta_identity=None):
    """Bind current bands and the authenticated physical charge fit.

    SC supplies the current Hamiltonian/rotation identity from its owner;
    a DFT fingerprint alone cannot authenticate rotated wavefunctions.
    Array hashes here cover only replicated small energy/occupation tables.
    """
    from common.parallel_transport import fingerprint_from_binding, wfn_fingerprint
    from file_io.wfn_basis import centroid_table_fingerprint_scheme, centroid_table_md5
    from .response_bank import response_weights
    # The same canonical physical points own sampling, symmetry and restart
    # identity. In particular, fractional points must never pass through the
    # historical FFT-index cast: distinct sub-grid positions would collide.
    basis = meta.mu_basis
    coordinates = np.asarray(basis.canonical_indices)
    if not np.array_equal(np.asarray(centroid_indices), coordinates):
        raise ValueError("GATE shared_pole_centroids: supplied points differ from the canonical basis")
    coordinate_kind = basis.coordinate_kind
    scheme = centroid_table_fingerprint_scheme(coordinate_kind)
    if coordinate_kind == "fft_indices":
        # Preserve the exact legacy identity bytes and dictionary protocol.
        centroid_identity = hashlib.sha256(np.asarray(coordinates, np.int64).tobytes()).hexdigest()
    else:
        centroid_identity = centroid_table_md5(coordinates, coordinate_kind=coordinate_kind)
    census = response_weights(wfns, meta)[-1]
    source = wfn_fingerprint(wfn) if binding is None else fingerprint_from_binding(binding, wfn)
    state = getattr(meta, "shared_pole_state_identity", None)
    if str(label).startswith("sc_"):
        if not isinstance(state, dict) or any(not isinstance(state.get(k), str)
                or not state[k] for k in ("hamiltonian", "wavefunctions")):
            raise ValueError("GATE shared_pole_state_identity: current SC Hamiltonian/rotation identity missing")
    else:
        state = dict(wavefunctions=source, hamiltonian=hashlib.sha256(
            (source + census["energy_sha256"]).encode()).hexdigest())
    recipe = meta.shared_pole_recipe
    identity = dict(iteration_id=str(label), hamiltonian=state["hamiltonian"],
        wavefunctions=state["wavefunctions"], energies=census["energy_sha256"],
        occupations=census["occupation_sha256"],
        centroids=centroid_identity,
        recipe_hash=recipe["recipe_hash"], gate_hash=recipe["gate_hash"])
    if coordinate_kind != "fft_indices":
        identity.update(centroid_coordinate_kind=coordinate_kind,
                        centroid_fingerprint_scheme=scheme)
    if str(label).startswith("sc_"):
        # The SC state labels do not name the WFN (bind_shared_pole_sc_identity),
        # so an SC map's bank, model and sector manifest bind the source WFN's
        # fingerprint here (the dipole provenance's): a rerun reuses them only
        # on the same WFN, never on energies alone.
        identity["wfn"] = source
    if charge_zeta_identity is not None:
        # gw_init owns all fit semantics (including augmentation and endpoint
        # weights). Transport its opaque receipt, never rederive them here.
        from file_io.tagged_arrays import _encode_charge_zeta_identity
        _encode_charge_zeta_identity(charge_zeta_identity)
        identity.update(charge_zeta_identity_scheme=charge_zeta_identity["scheme"],
                        charge_zeta_identity=charge_zeta_identity["digest"])
    return identity


def _coulomb_resource(value, meta, sym, mesh_xy):
    """The bare V parents' resource: the on-device operator, named by its token and digest.

    ``value`` is the packed bare V (a ``QirrOperator`` on the run's q wedge,
    or a full-q face [Q,mu_p,mu_p]), Ry; its wedge rows are the parents.  The
    response owner roots the parents straight from this operator
    (``response_bank._coulomb_roots``); no copy is staged on disk (54.6 GB and
    41 s at Ni 20^3 when it was).  The identity is a device digest of the
    packed parents, the same on every rank, so a bank's receipts and a
    constructor resume bind the operator's values.  The packed layout follows
    the mesh, so a bank built at another P does not match and is rebuilt.
    """
    from symmetry_maps import QirrOperator
    from .response_bank import _operator_token, operator_digest
    basis = meta.mu_basis
    qids = np.asarray(sym.q_irr_full_idx, np.int64)
    op = QirrOperator.of(value)
    if op.n_full != meta.nk_tot or tuple(op.values.shape[1:]) != (basis.n_packed, basis.n_packed):
        raise ValueError("GATE shared_pole_coulomb: expected the packed bare V")
    # The parents' rows are one array for the run (the wedge's own values, or
    # one gather of a whole-zone operator), so the held roots and the digest
    # are reused by every SC map.
    held = _PARENT_ROWS.get("V")
    if held is None or held[0] is not op.values or held[1] != tuple(qids.tolist()):
        held = _PARENT_ROWS["V"] = (op.values, tuple(qids.tolist()), op.at_rows(qids))
    rows = held[2]
    return dict(path=None, dataset=None, basis="canonical", q_irr_full_idx=qids.tolist(),
                sha256=operator_digest(rows, mesh_xy), operator=_operator_token(rows))


#: The bare V of this run and its parent rows (one entry).
_PARENT_ROWS: dict = {}


def _bank_residence(meta, config, *, mesh_xy, sym, root, label, photon, mu_bases=None):
    """Keep this map's bank on the devices when it fits, else in host memory.

    The bank is written once and read back once by the constructor; the file
    exists only because the producer is frequency-major and the constructor
    parent-major. It stays resident when (1) the resident payload R and one
    complete read copy fit in half of the available device budget (R4's
    q-local margin), and (2) the constructor's own route admission, with R
    live, still selects local parents, so residency never changes the route.
    ``photon`` is the photon layout of a full bispinor bank (else None); its
    sector constructor route must likewise be unchanged. Otherwise the payload
    goes to the host tier when it takes at most half of this process's
    host budget (``host_bytes_per_process``), else to one file per rank
    (both SlabIO's streamed tier); the devices then hold only the span being
    read. The shared scratch file is left for a W export (``write_w`` re-reads
    the bank after the constructor), a requested distributed layout, and a
    per-rank file the disk or quota refuses on any rank.
    Returns ``(payload or None, receipt)``; a device payload's R is reserved
    (``receipt["stage"]``) and the caller keeps it in ``ledger.live_stages``
    until it is released.
    """
    from common.gpu_utils import host_bytes_per_process
    from file_io.shared_pole_store import ResidentBankPayload, shared_pole_bank_payload_bytes
    from .gw_config import linalg_resolution
    from .shared_pole_constructor import constructor_route

    from .shared_pole_directions import line_width_bounds
    ledger = meta.shared_pole_capacity
    ordered = bool(photon) or not bool(sym.trs_allowed)
    nq = len(np.asarray(sym.q_irr_full_idx))
    R = shared_pole_bank_payload_bytes(meta, recipe=meta.shared_pole_recipe,
                                       ordered=ordered, nq=nq, mesh_xy=mesh_xy,
                                       line_widths=line_width_bounds(meta, mesh_xy=mesh_xy,
                                           photon_bases=None if photon is None else mu_bases),
                                       photon_extent=None if photon is None else photon.packed_extent,
                                       photon_bases=None if photon is None else mu_bases)
    receipt = dict(residence="file", payload_bytes_per_rank=R,
                   payload_bytes_total=R * int(mesh_xy.size))
    if config.debug.write_w or linalg_resolution(
            {"linalg": config.backend.linalg}).layout != "local":
        receipt["reason"] = "write_w export or distributed layout"
        return None, receipt
    carrier = meta.mu_basis.n_canonical if photon is None else photon.packed_extent
    bank_label = str(root / "bank.h5")

    def host(reason):
        receipt["half_host_budget_bytes_per_rank"] = int(host_bytes_per_process()) // 2
        if R > receipt["half_host_budget_bytes_per_rank"]:
            # Read once by the same ranks in the same layout: one file per rank
            # (SlabIO's streamed tier), not the shared bank.h5.
            receipt.update(residence="rank_file", reason=reason + "; payload exceeds half the "
                           "host budget; per-rank files")
            return ResidentBankPayload(mesh_xy, carrier=carrier, label=bank_label,
                                       memory_kind="file", root=root), receipt
        receipt.update(residence="host", reason=reason + "; host tier")
        return ResidentBankPayload(mesh_xy, carrier=carrier, label=bank_label,
                                   memory_kind="host", root=root), receipt

    both = ledger.preview(resident_bytes_per_rank=2 * R, workspace_bytes_per_rank=0,
                          concurrent_with=())
    receipt["half_budget_bytes_per_rank"] = both["available_device_bytes_per_rank"] // 2
    if both["aggregate_bytes_per_rank"] > receipt["half_budget_bytes_per_rank"]:
        return host("payload and one read copy exceed half the device budget")
    if photon is not None:
        from .shared_pole_sectors import sector_execution
        route = lambda upstream: sector_execution(meta, config, mu_bases, nq,
                                                  mesh_xy=mesh_xy, upstream=upstream)[0]
        without = route(())
    stage = f"bank_resident.{label}"
    ledger.reserve(stage, resident_bytes_per_rank=R, workspace_bytes_per_rank=0,
                   concurrent_with=())
    if photon is not None:
        # The sector constructor must keep the layout it would have chosen
        # without the payload live, so residency never changes the route.
        execution, wanted = route((stage,)), without
    else:
        execution, wanted = constructor_route(
            meta, config, meta.shared_pole_recipe, mesh_xy=mesh_xy, ledger=ledger,
            upstream=(stage,), ordered=ordered, odd_moments=ordered, nq=nq)[0], "local"
    if execution != wanted:
        return host("constructor would change route with the payload live")
    receipt.update(residence="device", stage=stage,
                   reason="payload, one read copy and the unchanged constructor route fit")
    return ResidentBankPayload(mesh_xy, carrier=carrier, label=bank_label), receipt


#: The device-resident scalar model of this process's latest SC map (at most
#: one). It lives from its constructor through the map's head and Sigma and,
#: on the accepted final map, the W0 persist; the next map's entry
#: (``sc_iteration.gw_iteration_map``, before the wavefunction rotation and
#: the Hartree rebuild, which no planner prices it against) or the end of the
#: SC run releases it (:func:`release_resident_model`).
_RESIDENT_MODEL: list = []


def release_resident_model():
    """Release the latest SC map's device-resident scalar or CC sector model, if any."""
    while _RESIDENT_MODEL:
        _RESIDENT_MODEL.pop().release()


def hold_resident_model(model, meta, nbytes, *, stage):
    """Keep an SC map's resident CC sector model live, as the scalar model is.

    ``mpa.sector_sigma.compute_sector_sigma`` releases TT and CT after Sigma
    and hands CC here: the accepted final map's W0 persist reads it
    (``mpa.sector_sigma.sector_static_wc``).  Its ``nbytes`` stay reserved
    as ``stage``; :func:`release_resident_model` releases it.
    """
    ledger = meta.shared_pole_capacity
    ledger.reserve(stage, resident_bytes_per_rank=int(nbytes),
                   workspace_bytes_per_rank=0, concurrent_with=ledger.live_stages)
    ledger.live_stages = (*ledger.live_stages, stage)
    _RESIDENT_MODEL.append(model)


def _scalar_model_residence(meta, nq, width, *, mesh_xy, root, identity):
    """Keep an SC map's scalar model on the devices when it fits.

    The model file of an SC map has three readers in the same process: the
    head, Sigma and, on the accepted final map, the W0 persist
    (``gw.mpa.sigma.shared_pole_static_wc``); no later run reads it (SC
    scratch never serves a restart). It stays resident under the sector
    models' admission (``shared_pole_store.admit_resident_model``): R, its
    bytes at the stored K extent's bound (``model_column_bound`` of the
    constructor's column width), and one copy fit in half of the device
    budget beside the live stages. The constructor has finished when this is
    asked, so its route cannot change; Sigma and the W0 persist price R, so
    their results match the file route bit for bit where their panel schedule
    is unchanged. Otherwise model.h5 is written as before. Returns ``(model or None, receipt)``; a resident R is reserved as
    ``receipt["stage"]`` and the model is held for :func:`release_resident_model`.
    """
    from file_io.shared_pole_store import (ResidentSectorModel, admit_resident_model,
                                           model_column_bound)
    ledger = meta.shared_pole_capacity
    R = ResidentSectorModel.payload_bytes(mesh_xy, nq, meta.mu_basis.n_canonical,
                                          model_column_bound(meta, width))
    receipt = admit_resident_model(
        ledger, R, f"scalar_model.{identity['iteration_id']}.{len(ledger.entries)}",
        ledger.live_stages)
    if receipt["residence"] != "device":
        return None, receipt
    model = ResidentSectorModel(mesh_xy, label=str(root / "model.h5"))
    _RESIDENT_MODEL.append(model)
    return model, receipt


def _release_bank_file(path):
    """Unlink this run's link to a committed map's file-tier bank.

    At production shape the bank is the run's largest file:
    N_q (2 N_dense + N_moments) N_mu^2 16 B plus the line panels, 1.26e12 B at
    Fe 20^3 (1062 parents, 1796 centroids). A resume directory's bank may be
    hard-linked from another attempt; only this link goes, and the link count
    is returned with the bytes so the receipt says whether space was freed.
    Returns ``None`` when there is no file.  Rank 0 alone looks: a per-rank
    ``exists()`` raced its unlink, and a late rank skipped the release while
    the others waited in it (the Na SC hang on a cold cache, CPU P4).
    """
    import os
    from common.collectives import rank0_transaction

    def unlink():
        if not os.path.exists(path):
            return None
        stat = os.stat(path)
        os.unlink(path)
        return dict(bytes=int(stat.st_size), links=int(stat.st_nlink))
    return rank0_transaction(path, stage="shared_pole.bank.release", write=unlink,
                             return_value=True)


#: The per-map scratch generation ``screen_shared_poles`` creates under an SC
#: label (bank, Coulomb staging, constant and receipts).
_MANAGED_SCRATCH = r"sc_[0-9]{4}_shared_pole"
#: Set once this process has swept earlier processes' streamed stores (prepare_output).
_SWEPT: list = []


def retain_iteration_scratch(run_dir, label, *, print_fn=print):
    """Collectively keep only map ``label``'s shared-pole scratch generation.

    Each SC map screens into its own ``sc_NNNN_shared_pole/``, and a file-tier
    bank there is the payload of ``shared_pole_bank_payload_bytes`` (about
    0.27 TB on Fe 8^3), so without this the scratch grows linearly in maps.  The caller invokes it
    only after the current map's model has been built, consumed by Sigma and
    passed the Sigma gates, so a failure keeps the last usable generation, and
    the current one survives convergence for constructor resume.  Only exact
    managed names are eligible; scanning rather than removing only ``N-1``
    also clears stale later maps of a longer earlier run.  This is
    ``gw.mpa.model.retain_iteration_artifacts``'s rule, through the same
    removal owner.  Run-lifetime artifacts (the photon route's static
    reference) live beside the generations, never inside one.
    """
    import os
    from .qsgw_utils import remove_managed

    root = os.path.abspath(os.fspath(run_dir))
    removed = remove_managed(
        root, _MANAGED_SCRATCH, keep=[os.path.join(root, f"{label}_shared_pole")],
        barrier_tag=f"shared_pole.scratch.retain.{label}", print_fn=print_fn)
    if removed:
        print_fn(f"  shared-pole scratch: retained {label}; discarded: "
                 + ", ".join(sorted(removed)))
    return tuple(sorted(removed))


def _shared_pole_tables(meta, sym, basis):
    """Build raw-parent tables from the authenticated centroid coordinates."""
    from symmetry_maps import (QirrTables, centroid_source_map_and_wrap,
                               bgw_integer_q_to_fractional)
    perm, wraps = centroid_source_map_and_wrap(
        basis.canonical_indices, sym.sym_matrices, sym.translations,
        np.asarray(meta.fft_grid), extend_trs=True,
        required_rows=np.asarray(sym.sym_idx_q),
        coordinate_kind=basis.coordinate_kind)
    qt = QirrTables(irr_idx_q=sym.irr_idx_q, sym_idx_q=sym.sym_idx_q,
        q_irr_frac=bgw_integer_q_to_fractional(
            sym.q_irr_kgrid_int, (meta.nkx, meta.nky, meta.nkz)),
        sym_perm=perm, L_table=wraps, n_sym_spatial=len(sym.sym_matrices))
    return dict(qirr=qt, q_irr_full_idx=np.asarray(sym.q_irr_full_idx, np.int64), sym=sym)


def screen_shared_poles(wfns, V_q, meta, config, *, mesh_xy, sym,
                        centroid_indices, run_dir, label, wfn,
                        wfn_fingerprint_binding, tensors_filename, occupation_state, print_fn,
                        head_resolver=None, mpa_plan=None, iteration_head_response=None,
                        material_class=None, wfns_transverse=None,
                        bispinor_v_q_path=None, mu_bases=None,
                        photon_g0_vectors=None, photon_head_cache=None,
                        photon_head_state=None, charge_zeta_identity=None):
    """Build current W; only one-shot models may use ISDF restart membership.

    SC labels own separate map scratch. ``restart`` may restore the invariant
    ISDF basis, but never skips the current response or W construction.
    The fitting catalogue supplies its path-independent two-string receipt;
    a model cannot authenticate different physical vertices from the same WFN.
    """
    if charge_zeta_identity is None:
        raise ValueError("GATE shared_pole_charge_fit: authenticated charge-zeta "
                         "identity missing; regenerate the ISDF fit in a fresh run variant")
    source_wfn = None
    from .gw_config import uses_full_bispinor_shared_pole
    photon = uses_full_bispinor_shared_pole(config)
    if photon and int(meta.nspinor) != 4:
        raise ValueError("GATE shared_pole_sectors: full_shared_pole requires four-spinor metadata")
    photon_layout = None
    if photon:
        from .photon_layout import PhotonBasisLayout
        if wfns_transverse is None or bispinor_v_q_path is None or mu_bases is None:
            raise ValueError("GATE shared_pole_sectors: both current-map endpoint bundles and photon V are required")
        photon_layout = PhotonBasisLayout.from_centroid_extents(
            mu_bases[0].n_logical, mu_bases[1].n_logical, mesh_xy)
    if config.debug.write_w or config.write_poles:
        source_wfn = getattr(wfn, "path", None)
        if source_wfn is None or not str(source_wfn).strip():
            raise ValueError("GATE shared_pole_output: write_w/write_poles require the source WFN path before screening")
    # A time-reversal-broken metal runs on this route (METAL 2026-09-16): the
    # ordered bank keeps both particle-hole orientations with the odd channel
    # (8d3ad5fe) and its kernels return the physical orientation FT_q[chi]
    # (86307cff). GATE mpa_ordered_metal stays on the MPA route, which fits one
    # residue with no odd channel (gw.mpa.model._require_metal_time_reversal).
    if material_class == "metal" and not bool(getattr(sym, "trs_allowed", True)):
        print_fn("  shared-pole screening: time-reversal-broken METAL; ordered bank "
                 "(both orientations, odd channel) with fractional occupations")
    with timing.section("spole.screening_setup"):
        from file_io.shared_pole_store import initialize_shared_pole_bank
        from file_io.tagged_arrays import register_shared_pole_restart_member
        from .shared_pole_recipe import shared_pole_restart_handle
        from .shared_pole_constructor import construct_shared_poles
        from .w_isdf import produce_w_bank, compute_response_moments

        started = time.monotonic()
        recipe, ledger = meta.shared_pole_recipe, meta.shared_pole_capacity
        # This is the top-level map boundary. All upstream V/psi carriers are
        # inherited; no newly allocated bank/constructor arrays exist yet.
        # (The SC map entry already released the previous map's resident model.)
        ledger.live_stages = ()
        if occupation_state is not None:
            from common.collectives import replicate_to_mesh
            occupations = np.asarray(occupation_state.f_kn, np.float64)
            if occupations.shape != wfns.occ.shape:
                raise ValueError("GATE shared_pole_occupations: supplied current state does not match carrier")
            wfns = dataclasses.replace(wfns, occ=replicate_to_mesh(occupations, mesh_xy))
            if photon:
                wfns_transverse = dataclasses.replace(wfns_transverse, occ=wfns.occ)
        identity = shared_pole_identity(wfns, meta, label=label, wfn=wfn,
            binding=wfn_fingerprint_binding, centroid_indices=centroid_indices,
            charge_zeta_identity=charge_zeta_identity)
        sc_scratch = str(label).startswith("sc_")
        if config.restart and tensors_filename is not None and not sc_scratch and not photon:
            handle = shared_pole_restart_handle(tensors_filename,
                expected_identity=identity, meta=meta, mesh_xy=mesh_xy, print_fn=print_fn)
            if handle is not None:
                result = dict(shared_pole=handle)
                from .gw_config import HeadCorrection
                if config.head.correction is not HeadCorrection.OFF:
                    from .shared_pole_head import build_shared_pole_head
                    head, iteration_head = build_shared_pole_head(
                        handle, None, V_q, wfns, meta, config, mesh_xy=mesh_xy, wfn=wfn,
                        response=iteration_head_response, head_resolver=head_resolver,
                        plan=mpa_plan, material_class=material_class, occupation_state=occupation_state)
                    result.update(mpa_head=head, iteration_head=iteration_head)
                # The export is written only once the map is complete, so an
                # export on disk always implies a finished head.  Under SC
                # this branch is unreachable (``sc_scratch`` skips restart).
                if config.debug.write_w or config.write_poles:
                    from file_io.shared_pole_store import export_shared_pole_outputs
                    with timing.section("spole.outputs"):
                        export_shared_pole_outputs(handle, meta=meta, config=config,
                            mesh_xy=mesh_xy, source_wfn=source_wfn,
                            run_dir=run_dir, label=label, print_fn=print_fn,
                            tables=_shared_pole_tables(meta, sym, meta.mu_basis))
                return result
        root = Path(run_dir).resolve() / (str(label) + "_shared_pole")
        # The bare V parents (a collective device digest, before the rank-0 resume
        # decision): a complete bank resumes only when built against these values.
        coulomb = None if photon else _coulomb_resource(V_q, meta, sym, mesh_xy)
        from common.collectives import rank0_transaction
        from file_io.commit_state import assert_committed

        def prepare_output():
            # Only rank zero reads the small completion marker, on the compute
            # node. The transaction owner broadcasts any refusal to every rank.
            # An interrupted run's directory keeps what authenticates for this
            # map and loses the rest: a published sector handle is reused, a
            # complete bank resumes the constructor, anything else is rebuilt.
            import h5py
            from file_io.slab_io import remove_stale_streamed_banks
            # A per-rank streamed store is unlinked as soon as it is opened, so its bytes
            # die with the process, SIGKILL included; only older code, or a kill between
            # open and unlink, leaves files, so the first map of a process removes every
            # generation's leftovers before it creates any.
            stale = remove_stale_streamed_banks(
                [root / "streamed_bank"] if _SWEPT else
                [p / "streamed_bank" for p in Path(run_dir).resolve().glob("*_shared_pole")])
            _SWEPT.append(True)
            if stale:
                print_fn("WARNING shared-pole output: removed streamed stores left by an earlier "
                         "process: " + " ".join(stale))
            model = root / "model.h5"
            if photon and (root / 'sectors.json').exists():
                if _published_sector_handle(root, identity) is not None:
                    print_fn(f"shared-pole output: authenticated sector manifest retained at {root}; reusing it")
                    return "published"
            complete, bound = False, None
            if model.exists():
                try:
                    with h5py.File(model, "r") as h5:
                        assert_committed(h5, path=model)
                        complete = "final_commit" in h5
                        raw = h5["header_json"][()] if complete else b"{}"
                    bound = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw)).get("identity")
                except (OSError, ValueError, KeyError, AttributeError):
                    complete = False
            if complete and sc_scratch and not photon and bound == identity:
                # An SC map's committed model of this very map (the file tier
                # persists it): Sigma reads it; nothing upstream is rebuilt.
                print_fn(f"shared-pole output: committed model of this map retained at {model}; reusing it")
                return "committed"
            if complete:
                print_fn(f"shared-pole output: complete model retained at {model}; refusing rebuild")
                raise ValueError(f"GATE shared_pole_output: complete model {model}; use its compatible restart member or a fresh run directory")
            if _authenticated_constructor_resume(root, identity, recipe, photon=photon,
                                                 coulomb_sha256=None if photon else coulomb['sha256']):
                print_fn(f"shared-pole output: authenticated complete bank retained at {root}; resuming constructor")
                stale = [name for name in (_SECTOR_OUTPUTS if photon else ())
                         if (root / name).exists()]
                for name in stale:
                    (root / name).unlink()
                if stale:
                    print_fn(f"WARNING shared-pole output: removed {' '.join(stale)} from {root}; "
                             "rebuilding this map's constructor")
                return True
            if root.exists():
                removed = ' '.join(sorted(path.name for path in root.iterdir()))
                print_fn(f"WARNING shared-pole output: removed partial directory {root} "
                         f"({removed}); rebuilding this map")
                shutil.rmtree(root)
            else:
                print_fn(f"shared-pole output: creating new directory {root}")
            root.mkdir(parents=True)
            return False

        resume_constructor = rank0_transaction(
            root, stage="shared_pole.prepare_output", write=prepare_output,
            return_value=True)
        if resume_constructor == "published":
            from common.collectives import agree_io_error
            handle, error = None, None
            try:
                handle = _published_sector_handle(root, identity)
                if handle is None:
                    raise ValueError(f"GATE shared_pole_output: sector manifest at {root} changed during reuse")
            except (OSError, ValueError) as exc:
                error = exc
            agree_io_error(error, path=root / 'sectors.json', stage='shared_pole.published_handle')
            _mark_photon_head(handle, config)
            return dict(shared_pole=handle)
        tables = _shared_pole_tables(meta, sym, meta.mu_basis)
    if resume_constructor == "committed":
        coulomb = None
    with timing.section("spole.bank_setup"):
        resident, residence = (None, dict(residence="file", reason=(
            "committed model reused" if resume_constructor == "committed" else "authenticated resume")))
        if not resume_constructor:
            resident, residence = _bank_residence(meta, config, mesh_xy=mesh_xy, sym=sym,
                                                  root=root, label=label,
                                                  photon=photon_layout if photon else None,
                                                  mu_bases=mu_bases)
        print_fn(f"shared-pole bank residence: {residence['residence']}; "
                 f"{residence['payload_bytes_per_rank'] / 2**30:.3f} GiB/rank, "
                 f"{residence['payload_bytes_total'] / 2**30:.2f} GiB total; {residence['reason']}"
                 if 'payload_bytes_per_rank' in residence else
                 f"shared-pole bank residence: file; {residence['reason']}")
        if "stage" in residence:
            ledger.live_stages = (residence["stage"],)
        bank = dict(path=str(root / "bank.h5") if resident is None else resident,
                    identity=identity, tables=tables, coulomb=coulomb, root=str(root))
        if photon:
            bank.update(photon_layout=photon_layout, mu_bases=mu_bases,
                        bispinor_v_q_path=bispinor_v_q_path,
                        sector_tables=(tables, _shared_pole_tables(
                            meta, sym, mu_bases[1])))
        if not resume_constructor:
            initialize_shared_pole_bank(bank["path"], meta=meta, tables=tables,
                recipe=recipe, identity=identity, mesh_xy=mesh_xy,
                **(dict(photon_layout=photon_layout, mu_bases=mu_bases) if photon else {}))
            if not getattr(bank["path"], "fits", True):
                # The per-rank files could not reserve their bytes on some rank
                # (agreed on every rank): the shared scratch file, as before.
                bank["path"].release()
                bank["path"], resident = str(root / "bank.h5"), None
                residence = dict(residence, residence="file", reason="per-rank files refused "
                                 "by the filesystem (capacity); shared scratch file")
                print_fn(f"shared-pole bank residence: file; {residence['reason']}")
                initialize_shared_pole_bank(bank["path"], meta=meta, tables=tables,
                    recipe=recipe, identity=identity, mesh_xy=mesh_xy,
                    **(dict(photon_layout=photon_layout, mu_bases=mu_bases) if photon else {}))
        receipts = dict(identity=identity)
        if resume_constructor is True:
            receipts['bank'] = json.loads((root / 'bank_receipt.json').read_text())
            if not photon:
                receipts['moments'] = json.loads((root / 'moments_receipt.json').read_text())
            receipts['constructor_resume'] = dict(
                status='AUTHENTICATED_COMPLETE_BANK', source=str(root / 'bank.h5'))
        recorded = set()

        def record(stage, receipt):
            # EVERY RANK LEAVES THIS CALL THE SAME WAY.  A bare
            # ``process_index() == 0`` write raises on rank 0 alone (quota,
            # EIO on purge-eligible scratch) while ranks 1..P-1 walk into
            # the next collective and hang to walltime (INVARIANTS 21).
            # ``rank0_transaction`` does the same serial write and
            # broadcasts its verdict, exactly as the sibling write above.
            receipts[stage] = receipt
            recorded.add(stage)
            path = root / (stage + "_receipt.json")
            rank0_transaction(
                path, stage=f"shared_pole.receipt.{stage}",
                write=lambda: path.write_text(_json(receipt) + "\n"))
            print_fn(f"shared-pole {stage}: completion={receipt.get('completion', receipt.get('status'))}; "
                     f"seconds={receipt.get('seconds', {})}")
    with timing.section("spole.bank"):
        if resume_constructor:
            print_fn('shared-pole bank: not produced; '
                     + ('committed model reused' if resume_constructor == 'committed'
                        else 'authenticated complete producer artifact reused'))
        elif photon:
            from .response_bank import compute_photon_bank
            record("bank", compute_photon_bank(wfns, wfns_transverse, meta, config,
                mesh_xy=mesh_xy, sym=sym, mu_bases=mu_bases, layout=photon_layout,
                occupation_state=occupation_state, sample_plan=recipe, bank_io=bank,
                wfn=wfn, photon_g0_vectors=photon_g0_vectors,
                wfn_fingerprint_binding=wfn_fingerprint_binding,
                photon_head_cache=photon_head_cache,
                photon_head_state=photon_head_state, print_fn=print_fn))
        else:
            record("bank", produce_w_bank(wfns, meta, config, mesh_xy=mesh_xy,
                sym=sym, sample_plan=recipe, bank_io=bank, print_fn=print_fn))
    if not photon and not resume_constructor:
        with timing.section("spole.moments", announce=True,
                            label="shared-pole response moments"):
            record("moments", compute_response_moments(wfns, meta, config,
                mesh_xy=mesh_xy, sym=sym, bank_io=bank))
        moments_seconds = receipts["moments"].get("seconds", {})
        print_fn(f"Response quadrature: moments {receipts['moments'].get('correlation_count', 0)} "
                 f"correlations in {len(receipts['moments'].get('q_batches', ()))} q batch(es) "
                 f"of <= {receipts['moments'].get('q_width', 0)} parents; "
                 f"{moments_seconds.get('total', 0.0):.2f} s ("
                 + " ".join(f"{name}={moments_seconds[name]:.2f}" for name in
                            ("correlations", "coulomb", "dyson", "diagnostics", "io")
                            if name in moments_seconds) + ")")
    # The constructor owns scratch reads, actual pencil planning and the
    # final writer. It must query its own native workspace at the actual R.
    # W/dW and M1/M3 are distinct keyed datasets in the same scratch file.
    if resume_constructor == "committed":
        from file_io.shared_pole_store import validate_shared_pole_model
        model_path = root / "model.h5"
        result = dict(model=str(model_path), status="COMMITTED_MODEL_REUSED",
                      model_header=validate_shared_pole_model(
                          model_path, expected_identity=identity, mesh_xy=mesh_xy,
                          capacity=ledger),
                      model_residence=dict(residence="file", reason="committed model reused"))
    elif photon:
        from .shared_pole_sectors import construct_sector_poles
        result = construct_sector_poles(bank, meta, config,
            mesh_xy=mesh_xy, output=str(root / "model.h5"))
        from .shared_pole_execution import route_summary
        rows = result.get("execution") or ()
        if rows:
            routes = [(row["sector"], row) for row in rows] + [("CT", rows[0]["joint"])]
            mode = "face" if any(row["mode"] == "face" for _, row in routes) else "local"
            print_fn(f"Shared-pole sector constructor: {mode} route; " + "; ".join(
                f"{name} {route_summary(row['mode'], row)}" for name, row in routes))
            _print_eigh_stack_routes(print_fn)
    else:
        # An SC map keeps its model on the devices when it fits; an export
        # (write_w, write_poles) reads model.h5 and a one-shot registers it
        # as a restart member, so those write it.  On the bank's file tier
        # the model is persisted too: a rerun reuses it instead of the bank,
        # which is released below.
        model_rule = None
        if (sc_scratch and resident is not None
                and not (config.debug.write_w or config.write_poles)):
            model_rule = partial(_scalar_model_residence, meta, mesh_xy=mesh_xy,
                                 root=root, identity=identity)
        with timing.section("spole.constructor", announce=True,
                            label="shared-pole constructor"):
            result = construct_shared_poles(bank, bank, meta, config,
                mesh_xy=mesh_xy, output=str(root / "model.h5"), residence=model_rule)
        walls = dict(result.get("seconds", {}))
        rounds = walls.pop("rounds", 0)
        from .shared_pole_execution import route_summary
        route = result.get("execution")
        print_fn("Shared-pole constructor: "
                 + ("" if route is None else route_summary(route["mode"], route) + "; ")
                 + f"{rounds} round(s); seconds "
                 + " ".join(f"{name}={value:.2f}" for name, value in
                            sorted(walls.items(), key=lambda item: -item[1])))
        _print_eigh_stack_routes(print_fn)
    if resident is not None:
        # The model is committed; the constructor was the bank's last reader.
        resident.release()
        ledger.live_stages = ()
    elif not photon and not config.debug.write_w:
        # The file tier likewise: model.h5 is committed, and only a write_w
        # export (and the photon Sigma's constant) would read the bank again.
        released = _release_bank_file(root / "bank.h5")
        if released is not None:
            residence = dict(residence, released=released)
            print_fn(f"shared-pole bank: scratch file released after the constructor "
                     f"({released['bytes'] / 2**30:.2f} GiB, {released['links']} link(s))")
    receipts["bank_residence"] = residence
    with timing.section("spole.screening_finalize", announce=True,
                        label="shared-pole receipts and head"):
        record("constructor", result)
        models = result["model_residence"]
        if "payload_bytes_per_rank" in models:
            print_fn(f"shared-pole {'sector models' if photon else 'model'}: {models['residence']}; "
                     f"{models['payload_bytes_per_rank'] / 2**30:.3f} GiB/rank"
                     f"{' bound' if photon else ''}; {models['reason']}")
        header = None if photon else result["model_header"]
        handle = (result['handle'] if photon else
                  dict(path=result["model"], identity=identity,
                       digest=header["digest"], K=list(header["K"])))
        if not photon and models.get("stage"):
            # Sigma and the final-map W0 persist read it; the next map releases it.
            handle["model_stage"] = models["stage"]
        if photon and sc_scratch and handle.get("model_stage"):
            # Sigma releases TT and CT; CC stays for the final-map W0 persist.
            handle["hold_charge_model"] = True
        # Device-resident sector models stay live until Sigma releases them.
        ledger.live_stages = (handle['model_stage'],) if handle.get('model_stage') else ()
        result = dict(shared_pole=handle)
        from .gw_config import HeadCorrection
        if photon:
            _mark_photon_head(handle, config)
        elif config.head.correction is not HeadCorrection.OFF:
            from .shared_pole_head import build_shared_pole_head
            head, iteration_head = build_shared_pole_head(
                handle, header, V_q, wfns, meta, config, mesh_xy=mesh_xy, wfn=wfn,
                response=iteration_head_response, head_resolver=head_resolver,
                plan=mpa_plan, material_class=material_class, occupation_state=occupation_state)
            result.update(mpa_head=head, iteration_head=iteration_head)
            record("head", head)
        if (config.debug.write_w or config.write_poles) and not photon:
            from file_io.shared_pole_store import export_shared_pole_outputs
            with timing.section("spole.outputs"):
                receipts["outputs"] = export_shared_pole_outputs(handle, meta=meta,
                    config=config, mesh_xy=mesh_xy, source_wfn=source_wfn,
                    run_dir=run_dir, label=label, tables=tables, print_fn=print_fn)
        # The member lives in the restart file, so it is registered exactly when
        # that file is written (one owner: gw_output.restart_tensor_writes_enabled).
        from .gw_output import restart_tensor_writes_enabled
        if (tensors_filename is not None and not sc_scratch and not photon
                and restart_tensor_writes_enabled(config, tensors_filename)):
            receipts["restart_member"] = register_shared_pole_restart_member(
                tensors_filename, handle["path"], expected_identity=identity,
                mesh_xy=mesh_xy, capacity=ledger)
        receipts["seconds"] = time.monotonic() - started
        receipts["handle"] = handle
        summary = root / "construction_receipt.json"
        # A recorded stage is already in its own ``<stage>_receipt.json``; the
        # summary names that file instead of encoding the stage a second time.
        compact = {key: (dict(file=key + "_receipt.json") if key in recorded else value)
                   for key, value in receipts.items()}
        rank0_transaction(
            summary, stage="shared_pole.construction_receipt",
            write=lambda: summary.write_text(_json(compact) + "\n"))
        return result
