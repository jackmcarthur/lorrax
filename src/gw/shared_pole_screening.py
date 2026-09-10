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
from common import timing
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P


def _json(value):
    def encode(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, complex):
            return dict(real=x.real, imag=x.imag)
        raise TypeError(type(x).__name__)
    return json.dumps(value, default=encode, sort_keys=True, allow_nan=False)


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
    from .response_bank import _reserve, _resource_hash
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
            canonical.block_until_ready()
            io.write_slab("V_canonical_qwedge", canonical, offset=(iq, 0, 0),
                          global_shape=(len(qids), basis.n_canonical, basis.n_canonical),
                          valid_shape=(1, basis.n_logical, basis.n_logical))
            io.sync_writes()
            del canonical
    stat = path.stat()
    digest = np.zeros(32, np.uint8)
    if jax.process_index() == 0:
        digest[:] = np.frombuffer(bytes.fromhex(_resource_hash(
            str(path), stat.st_size, stat.st_mtime_ns)), np.uint8)
    digest = multihost_utils.broadcast_one_to_all(digest)
    return dict(path=str(path), dataset="V_canonical_qwedge", basis="canonical",
                q_irr_full_idx=qids.tolist(), sha256=bytes(np.asarray(digest)).hex())


def screen_shared_poles(wfns, V_q, meta, config, *, mesh_xy, sym,
                        centroid_indices, run_dir, label, wfn,
                        wfn_fingerprint_binding, tensors_filename, occupation_state, print_fn,
                        head_resolver=None, mpa_plan=None, iteration_head_response=None,
                        material_class=None):
    """Build current W; only one-shot models may use ISDF restart membership.

    SC labels own separate map scratch. ``restart`` may restore the invariant
    ISDF basis, but never skips the current response or W construction.
    """
    timing.fence("spole.screening_setup")
    with timing.section("spole.screening_setup"):
        from symmetry_maps import (QirrTables, centroid_source_map_and_wrap,
                                   bgw_integer_q_to_fractional)
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
        identity = shared_pole_identity(wfns, meta, label=label, wfn=wfn,
            binding=wfn_fingerprint_binding, centroid_indices=centroid_indices)
        sc_scratch = str(label).startswith("sc_")
        if config.restart and tensors_filename is not None and not sc_scratch:
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
                return result
        root = Path(run_dir).resolve() / (str(label) + "_shared_pole")
        from common.collectives import rank0_transaction
        from file_io.commit_state import assert_committed

        def prepare_output():
            # Only rank zero reads the small completion marker, on the compute
            # node. The transaction owner broadcasts any refusal to every rank.
            import h5py
            model = root / "model.h5"
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
            if root.exists():
                print_fn(f"shared-pole output: removing partial directory {root} and rebuilding")
                shutil.rmtree(root)
            else:
                print_fn(f"shared-pole output: creating new directory {root}")
            root.mkdir(parents=True)

        rank0_transaction(root, stage="shared_pole.prepare_output", write=prepare_output)
        qids = np.asarray(sym.q_irr_full_idx, np.int64)
        grid = (meta.nkx, meta.nky, meta.nkz)
        perm, wraps = centroid_source_map_and_wrap(
            np.asarray(centroid_indices), sym.sym_matrices, sym.translations,
            np.asarray(meta.fft_grid), extend_trs=True)
        qt = QirrTables(irr_idx_q=sym.irr_idx_q, sym_idx_q=sym.sym_idx_q,
            q_irr_frac=bgw_integer_q_to_fractional(sym.q_irr_kgrid_int, grid),
            sym_perm=perm, L_table=wraps, n_sym_spatial=len(sym.sym_matrices))
        tables = dict(qirr=qt, q_irr_full_idx=qids, sym=sym)
    timing.fence("spole.coulomb_staging")
    with timing.section("spole.coulomb_staging"):
        coulomb = _coulomb_resource(V_q, meta, sym, mesh_xy, root / "coulomb.h5")
    timing.fence("spole.bank_setup")
    with timing.section("spole.bank_setup"):
        bank = dict(path=str(root / "bank.h5"), identity=identity,
                    tables=tables, coulomb=coulomb)
        initialize_shared_pole_bank(bank["path"], meta=meta, tables=tables,
            recipe=recipe, identity=identity, mesh_xy=mesh_xy)
        receipts = dict(identity=identity)
        def record(stage, receipt):
            receipts[stage] = receipt
            if jax.process_index() == 0:
                (root / (stage + "_receipt.json")).write_text(_json(receipt) + "\n")
            print_fn(f"shared-pole {stage}: completion={receipt.get('completion', receipt.get('status'))}; "
                     f"seconds={receipt.get('seconds', {})}")
    timing.fence("spole.bank")
    with timing.section("spole.bank"):
        record("bank", produce_w_bank(wfns, meta, config, mesh_xy=mesh_xy,
            sym=sym, sample_plan=recipe, bank_io=bank))
    timing.fence("spole.moments")
    with timing.section("spole.moments"):
        record("moments", compute_response_moments(wfns, meta, config,
            mesh_xy=mesh_xy, sym=sym, bank_io=bank))
    # The constructor owns scratch reads, actual pencil planning and the
    # final writer. It must query its own native workspace at the actual R.
    # W/dW and M1/M3 are distinct keyed datasets in the same scratch file.
    result = construct_shared_poles(bank, bank, meta, config,
        mesh_xy=mesh_xy, output=str(root / "model.h5"))
    timing.fence("spole.screening_finalize")
    with timing.section("spole.screening_finalize"):
        record("constructor", result)
        header = result["model_header"]
        handle = dict(path=str(root / "model.h5"), identity=identity,
                      digest=header["digest"], K=list(header["K"]))
        ledger.live_stages = ()
        result = dict(shared_pole=handle)
        from .gw_config import HeadCorrection
        if config.head.correction is not HeadCorrection.OFF:
            from .shared_pole_head import build_shared_pole_head
            head, iteration_head = build_shared_pole_head(
                handle, header, V_q, wfns, meta, config, mesh_xy=mesh_xy, wfn=wfn,
                response=iteration_head_response, head_resolver=head_resolver,
                plan=mpa_plan, material_class=material_class, occupation_state=occupation_state)
            result.update(mpa_head=head, iteration_head=iteration_head)
            record("head", head)
        if tensors_filename is not None and not sc_scratch:
            receipts["restart_member"] = register_shared_pole_restart_member(
                tensors_filename, handle["path"], expected_identity=identity,
                mesh_xy=mesh_xy, capacity=ledger)
        receipts["seconds"] = time.monotonic() - started
        receipts["handle"] = handle
        if jax.process_index() == 0:
            (root / "construction_receipt.json").write_text(_json(receipts) + "\n")
        return result
