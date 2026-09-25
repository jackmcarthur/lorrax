"""Deck-driven shared real-pole screening (DESIGN §3, rulings15–16).

The response, constructor and persistence owners implement their own stages.
This helper authenticates the current map and passes only disk handles between
stages. Samples, directions and poles are rebuilt across changed maps;
the SC owner may retain time nodes after current-domain certification.
"""
from pathlib import Path
import dataclasses
import hashlib
import json
import shutil
import time

import jax
import numpy as np
from common import timing
from jax.sharding import PartitionSpec as P


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


def _authenticated_constructor_resume(root, identity, recipe, *, photon=False):
    """Authenticate a complete producer bank before preserving it.

    This is deliberately narrower than restart: it resumes only the missing
    constructor after the producer receipts and store validator bind the exact
    current map identity, resolved recipe, response convention and final commit.
    Photon moments are in the bank receipt; scalar moments have a second receipt.
    Any other partial directory follows the existing remove-and-rebuild path.
    """
    from file_io.shared_pole_store import validate_shared_pole_bank

    receipt_paths = ((root / 'bank_receipt.json',) if photon else
                     (root / 'bank_receipt.json', root / 'moments_receipt.json'))
    bank_path = root / 'bank.h5'
    coulomb_path = root / 'coulomb.h5'
    required = (*receipt_paths, bank_path) if photon else (*receipt_paths, bank_path, coulomb_path)
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
        reference = bank_receipt.get('static_reference', {})
        if (bank_receipt.get('stage') != 'photon'
                or bank_receipt.get('bank_complete') is not True
                or reference.get('identity') != identity
                or not isinstance(reference.get('path'), str)
                or Path(reference['path']).resolve() != bank_path.resolve()):
            return False
    elif (moments_receipt.get('identity') != identity
          or moments_receipt.get('completion') is not True
          or moments_receipt.get('bank_complete') is not True
          or bank_receipt.get('coulomb_identity') != moments_receipt.get('coulomb_identity')):
        return False
    try:
        header = validate_shared_pole_bank(
            bank_path, expected_identity=identity, mesh_xy=None,
            require_complete=True, expected_recipe=recipe)
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return ('photon_layout' in header) == photon


def shared_pole_identity(wfns, meta, *, label, wfn, binding, centroid_indices):
    """Bind logical current energies/occupations and their wavefunction source.

    SC supplies the current Hamiltonian/rotation identity from its owner;
    a DFT fingerprint alone cannot authenticate rotated wavefunctions.
    Array hashes here cover only replicated small energy/occupation tables.
    """
    from common.parallel_transport import fingerprint_from_binding, wfn_fingerprint
    from .response_bank import response_weights
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
    return dict(iteration_id=str(label), hamiltonian=state["hamiltonian"],
        wavefunctions=state["wavefunctions"], energies=census["energy_sha256"],
        occupations=census["occupation_sha256"],
        centroids=hashlib.sha256(np.asarray(centroid_indices, np.int64).tobytes()).hexdigest(),
        recipe_hash=recipe["recipe_hash"], gate_hash=recipe["gate_hash"])


def _coulomb_resource(value, meta, sym, mesh_xy, path):
    """Stage bounded canonical V parents via centroid and SlabIO owners.

    ``value`` is the incumbent packed full-q face [Q,mu_p,mu_p], Ry.
    New conversion/transport buffers are reserved before each allocation.
    Only one parent is unpacked at a time; no full-q canonical copy exists.
    """
    from file_io.slab_io import SlabIO
    from .response_bank import _reserve, resource_digest
    basis = meta.mu_basis
    qids = np.asarray(sym.q_irr_full_idx, np.int64)
    if value.shape != (meta.nk_tot, basis.n_packed, basis.n_packed):
        raise ValueError("GATE shared_pole_coulomb: expected incumbent packed full-q face")
    kernel = jax.jit(lambda v: basis.unpack_operator(v, spec=P(None, "x", "y")))
    shape = (1, basis.n_packed, basis.n_packed)
    operand = jax.ShapeDtypeStruct(shape, value.dtype, sharding=value.sharding)
    executable = kernel.lower(operand).compile()
    stats = executable.memory_analysis()
    if stats is None:
        raise ValueError("GATE shared_pole_coulomb: conversion memory unavailable")
    _reserve(meta, "coulomb_staging", stats.argument_size_in_bytes,
             stats.output_size_in_bytes + stats.temp_size_in_bytes)
    with SlabIO(path, mode="w", mesh=mesh_xy) as io:
        for iq, q in enumerate(qids):
            canonical = executable(value[int(q):int(q)+1])
            io.write_slab("V_canonical_qwedge", canonical, offset=(iq, 0, 0),
                          global_shape=(len(qids), basis.n_canonical, basis.n_canonical),
                          valid_shape=(1, basis.n_logical, basis.n_logical))
            io.sync_writes()
            del canonical
    return dict(path=str(path), dataset="V_canonical_qwedge", basis="canonical",
                q_irr_full_idx=qids.tolist(), sha256=resource_digest(path))


def _bank_residence(meta, config, *, mesh_xy, sym, root, label, photon, mu_bases=None):
    """Keep this map's bank on the devices when it fits, else in pinned host memory.

    The bank is written once and read back once by the constructor; the file
    exists only because the producer is frequency-major and the constructor
    parent-major. It stays resident when (1) the resident payload R and one
    complete read copy fit in half of the available device budget (R4's
    q-local margin), and (2) the constructor's own route admission, with R
    live, still selects local parents, so residency never changes the route.
    ``photon`` is the photon layout of a full bispinor bank (else None); its
    sector constructor route must likewise be unchanged. Otherwise the payload
    goes to the pinned host tier when it takes at most half of this process's
    host budget (``host_bytes_per_process``); the devices then hold only the
    span being read, as on the file route. The scratch file is left for a
    payload host memory cannot hold, a W export (``write_w`` re-reads the
    bank after the constructor) and a requested distributed layout.
    Returns ``(payload or None, receipt)``; a device payload's R is reserved
    (``receipt["stage"]``) and the caller keeps it in ``ledger.live_stages``
    until it is released.
    """
    from common.gpu_utils import host_bytes_per_process
    from file_io.shared_pole_store import ResidentBankPayload, shared_pole_bank_payload_bytes
    from .gw_config import linalg_resolution
    from .shared_pole_constructor import constructor_route

    ledger = meta.shared_pole_capacity
    ordered = bool(photon) or not bool(sym.trs_allowed)
    nq = len(np.asarray(sym.q_irr_full_idx))
    R = shared_pole_bank_payload_bytes(meta, recipe=meta.shared_pole_recipe,
                                       ordered=ordered, nq=nq, mesh_xy=mesh_xy,
                                       photon_extent=None if photon is None else photon.packed_extent)
    receipt = dict(residence="file", payload_bytes_per_rank=R,
                   payload_bytes_total=R * int(mesh_xy.size))
    if config.debug.write_w or linalg_resolution(
            {"linalg": config.backend.linalg}).layout != "local":
        receipt["reason"] = "write_w export or distributed layout"
        return None, receipt
    carrier = meta.mu_basis.n_canonical if photon is None else photon.packed_extent
    bank_label = str(root / "bank.h5")

    def pinned(reason):
        receipt["half_host_budget_bytes_per_rank"] = int(host_bytes_per_process()) // 2
        if R > receipt["half_host_budget_bytes_per_rank"]:
            receipt["reason"] = reason + "; payload exceeds half the host budget"
            return None, receipt
        receipt.update(residence="pinned_host", reason=reason + "; pinned host tier")
        return ResidentBankPayload(mesh_xy, carrier=carrier, label=bank_label,
                                   memory_kind="pinned_host"), receipt

    both = ledger.preview(resident_bytes_per_rank=2 * R, workspace_bytes_per_rank=0,
                          concurrent_with=())
    receipt["half_budget_bytes_per_rank"] = both["available_device_bytes_per_rank"] // 2
    if both["aggregate_bytes_per_rank"] > receipt["half_budget_bytes_per_rank"]:
        return pinned("payload and one read copy exceed half the device budget")
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
            upstream=(stage,), ordered=ordered, odd_moments=ordered, mirrored=ordered,
            nq=nq)[0], "local"
    if execution != wanted:
        return pinned("constructor would change route with the payload live")
    receipt.update(residence="device", stage=stage,
                   reason="payload, one read copy and the unchanged constructor route fit")
    return ResidentBankPayload(mesh_xy, carrier=carrier, label=bank_label), receipt


#: The per-map scratch generation ``screen_shared_poles`` creates under an SC
#: label (bank, Coulomb staging, constant and receipts).
_MANAGED_SCRATCH = r"sc_[0-9]{4}_shared_pole"


def retain_iteration_scratch(run_dir, label, *, pinned=(), print_fn=print):
    """Collectively keep only map ``label``'s shared-pole scratch generation.

    Each SC map screens into its own ``sc_NNNN_shared_pole/``, and a file-tier
    bank there is ``16 N_q (f_s N_s + f_m) N_mu^2`` bytes (0.87 TB on Fe 8^3),
    so without this the scratch grows linearly in maps.  The caller invokes it
    only after the current map's model has been built, consumed by Sigma and
    passed the Sigma gates, so a failure keeps the last usable generation, and
    the current one survives convergence for constructor resume.  Only exact
    managed names are eligible; scanning rather than removing only ``N-1``
    also clears stale later maps of a longer earlier run.  This is
    ``gw.mpa.model.retain_iteration_artifacts``'s rule, through the same
    removal owner.

    ``pinned`` paths are run-lifetime artifacts that live inside an earlier
    generation, and their generation is kept too: the photon route's static
    reference, the immutable initial contact every later map freezes, is map
    0's ``photon_static_reference.h5`` for a resident bank and map 0's
    ``bank.h5`` itself for a file-tier bank.
    """
    import os
    import re
    from .qsgw_utils import remove_managed

    root = os.path.abspath(os.fspath(run_dir))
    keep = [os.path.join(root, f"{label}_shared_pole")]
    for path in pinned:
        generation = os.path.dirname(os.path.abspath(os.fspath(path)))
        if (os.path.dirname(generation) == root
                and re.fullmatch(_MANAGED_SCRATCH, os.path.basename(generation))):
            keep.append(generation)
    removed = remove_managed(
        root, _MANAGED_SCRATCH, keep=keep,
        barrier_tag=f"shared_pole.scratch.retain.{label}", print_fn=print_fn)
    if removed:
        print_fn(f"  shared-pole scratch: retained {label}; discarded: "
                 + ", ".join(sorted(removed)))
    return tuple(sorted(removed))


def _shared_pole_tables(meta, sym, centroid_indices):
    """Build raw-parent tables through the canonical symmetry service."""
    from symmetry_maps import (QirrTables, centroid_source_map_and_wrap,
                               bgw_integer_q_to_fractional)
    perm, wraps = centroid_source_map_and_wrap(
        np.asarray(centroid_indices), sym.sym_matrices, sym.translations,
        np.asarray(meta.fft_grid), extend_trs=True)
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
                        photon_static_reference=None,
                        photon_g0_vectors=None, photon_head_cache=None,
                        photon_head_rotation=None):
    """Build current W; only one-shot models may use ISDF restart membership.

    SC labels own separate map scratch. ``restart`` may restore the invariant
    ISDF basis, but never skips the current response or W construction.
    """
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
            binding=wfn_fingerprint_binding, centroid_indices=centroid_indices)
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
                            tables=_shared_pole_tables(meta, sym, centroid_indices))
                return result
        root = Path(run_dir).resolve() / (str(label) + "_shared_pole")
        from common.collectives import rank0_transaction
        from file_io.commit_state import assert_committed

        def prepare_output():
            # Only rank zero reads the small completion marker, on the compute
            # node. The transaction owner broadcasts any refusal to every rank.
            import h5py
            model = root / "model.h5"
            if photon and (root/'sectors.json').exists():
                raise ValueError(f'GATE shared_pole_output: immutable sector manifest exists at {root}; use a fresh run directory')
            complete = False
            if model.exists():
                try:
                    with h5py.File(model, "r") as h5:
                        assert_committed(h5, path=model)
                        complete = "final_commit" in h5
                except (OSError, ValueError):
                    complete = False
            if complete:
                print_fn(f"shared-pole output: complete model retained at {model}; refusing rebuild")
                raise ValueError(f"GATE shared_pole_output: complete model {model}; use its compatible restart member or a fresh run directory")
            if _authenticated_constructor_resume(root, identity, recipe, photon=photon):
                print_fn(f"shared-pole output: authenticated complete bank retained at {root}; resuming constructor")
                return True
            if root.exists():
                print_fn(f"shared-pole output: removing partial directory {root} and rebuilding")
                shutil.rmtree(root)
            else:
                print_fn(f"shared-pole output: creating new directory {root}")
            root.mkdir(parents=True)
            return False

        resume_constructor = rank0_transaction(
            root, stage="shared_pole.prepare_output", write=prepare_output,
            return_value=True)
        tables = _shared_pole_tables(meta, sym, centroid_indices)
    with timing.section("spole.coulomb_staging"):
        if resume_constructor:
            saved_bank_receipt = json.loads((root / 'bank_receipt.json').read_text())
            coulomb = (None if photon else dict(saved_bank_receipt['coulomb_identity'],
                           path=str(root / 'coulomb.h5')))
        else:
            coulomb = (None if photon else
                       _coulomb_resource(V_q, meta, sym, mesh_xy, root / "coulomb.h5"))
    with timing.section("spole.bank_setup"):
        resident, residence = (None, dict(residence="file", reason="authenticated resume"))
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
                    identity=identity, tables=tables, coulomb=coulomb)
        if photon:
            bank.update(photon_layout=photon_layout, mu_bases=mu_bases,
                        static_reference=photon_static_reference,
                        bispinor_v_q_path=bispinor_v_q_path,
                        sector_tables=(tables, _shared_pole_tables(
                            meta, sym, mu_bases[1].canonical_indices)))
        if not resume_constructor:
            initialize_shared_pole_bank(bank["path"], meta=meta, tables=tables,
                recipe=recipe, identity=identity, mesh_xy=mesh_xy,
                **(dict(photon_layout=photon_layout, mu_bases=mu_bases) if photon else {}))
        receipts = dict(identity=identity)
        if resume_constructor:
            receipts['bank'] = json.loads((root / 'bank_receipt.json').read_text())
            if not photon:
                receipts['moments'] = json.loads((root / 'moments_receipt.json').read_text())
            receipts['constructor_resume'] = dict(
                status='AUTHENTICATED_COMPLETE_BANK', source=str(root / 'bank.h5'))
        def record(stage, receipt):
            # EVERY RANK LEAVES THIS CALL THE SAME WAY.  A bare
            # ``process_index() == 0`` write raises on rank 0 alone (quota,
            # EIO on purge-eligible scratch) while ranks 1..P-1 walk into
            # the next collective and hang to walltime (INVARIANTS 21).
            # ``rank0_transaction`` does the same serial write and
            # broadcasts its verdict, exactly as the sibling write above.
            receipts[stage] = receipt
            path = root / (stage + "_receipt.json")
            rank0_transaction(
                path, stage=f"shared_pole.receipt.{stage}",
                write=lambda: path.write_text(_json(receipt) + "\n"))
            print_fn(f"shared-pole {stage}: completion={receipt.get('completion', receipt.get('status'))}; "
                     f"seconds={receipt.get('seconds', {})}")
    with timing.section("spole.bank"):
        if resume_constructor:
            print_fn('shared-pole bank: authenticated complete producer artifact reused')
        elif photon:
            from .response_bank import compute_photon_bank
            record("bank", compute_photon_bank(wfns, wfns_transverse, meta, config,
                mesh_xy=mesh_xy, sym=sym, mu_bases=mu_bases, layout=photon_layout,
                occupation_state=occupation_state, sample_plan=recipe, bank_io=bank,
                wfn=wfn, photon_g0_vectors=photon_g0_vectors,
                wfn_fingerprint_binding=wfn_fingerprint_binding,
                photon_head_cache=photon_head_cache,
                photon_head_rotation=photon_head_rotation, print_fn=print_fn))
        else:
            record("bank", produce_w_bank(wfns, meta, config, mesh_xy=mesh_xy,
                sym=sym, sample_plan=recipe, bank_io=bank, print_fn=print_fn))
    if not photon and not resume_constructor:
        with timing.section("spole.moments"):
            record("moments", compute_response_moments(wfns, meta, config,
                mesh_xy=mesh_xy, sym=sym, bank_io=bank))
    # The constructor owns scratch reads, actual pencil planning and the
    # final writer. It must query its own native workspace at the actual R.
    # W/dW and M1/M3 are distinct keyed datasets in the same scratch file.
    if photon:
        from .shared_pole_sectors import construct_sector_poles
        result = construct_sector_poles(bank, meta, config,
            mesh_xy=mesh_xy, output=str(root / "model.h5"))
    else:
        result = construct_shared_poles(bank, bank, meta, config,
            mesh_xy=mesh_xy, output=str(root / "model.h5"))
    if resident is not None:
        # The model is committed; the constructor was the bank's last reader.
        resident.release()
        ledger.live_stages = ()
    receipts["bank_residence"] = residence
    with timing.section("spole.screening_finalize"):
        record("constructor", result)
        if photon:
            models = result["model_residence"]
            print_fn(f"shared-pole sector models: {models['residence']}; "
                     f"{models['payload_bytes_per_rank'] / 2**30:.3f} GiB/rank bound; {models['reason']}")
        header = None if photon else result["model_header"]
        handle = (result['handle'] if photon else
                  dict(path=str(root / "model.h5"), identity=identity,
                       digest=header["digest"], K=list(header["K"])))
        # Device-resident sector models stay live until Sigma releases them.
        ledger.live_stages = (handle['model_stage'],) if handle.get('model_stage') else ()
        result = dict(shared_pole=handle)
        if photon:
            result['photon_static_reference'] = receipts['bank']['static_reference']
        from .gw_config import HeadCorrection
        if config.head.correction is not HeadCorrection.OFF:
            if photon:
                from .gw_config import uses_direct_bispinor_shared_pole_head
                if not uses_direct_bispinor_shared_pole_head(config):
                    raise ValueError("GATE shared_pole_photon_head: unsupported photon Γ policy")
                handle["direct_photon_head"] = "first_order_cc_ct_tc_tt"
            else:
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
        if tensors_filename is not None and not sc_scratch and not photon:
            receipts["restart_member"] = register_shared_pole_restart_member(
                tensors_filename, handle["path"], expected_identity=identity,
                mesh_xy=mesh_xy, capacity=ledger)
        receipts["seconds"] = time.monotonic() - started
        receipts["handle"] = handle
        summary = root / "construction_receipt.json"
        rank0_transaction(
            summary, stage="shared_pole.construction_receipt",
            write=lambda: summary.write_text(_json(receipts) + "\n"))
        return result
