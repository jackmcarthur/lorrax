"""Deck-driven shared real-pole screening (DESIGN §3, rulings15–16).

The response, constructor and persistence owners implement their own stages.
This helper authenticates the current map and passes only disk handles between
stages. No sample, direction, pole or quadrature is reused across changed maps.
"""
from pathlib import Path
import dataclasses
import hashlib
import json
import time

import jax
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
                        wfn_fingerprint_binding, tensors_filename, occupation_state, print_fn):
    """Build/reuse one immutable current-map model and return its small handle."""
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
    from .response_bank import response_weights
    census = response_weights(wfns, meta)[-1]
    if census["occupation_sha256"] != recipe["census"]["occupation_sha256"]:
        raise ValueError("GATE shared_pole_occupations: bank state differs from resolved census")
    identity = shared_pole_identity(wfns, meta, label=label, wfn=wfn,
        binding=wfn_fingerprint_binding, centroid_indices=centroid_indices)
    if config.restart and tensors_filename is not None:
        handle = shared_pole_restart_handle(tensors_filename,
            expected_identity=identity, meta=meta, mesh_xy=mesh_xy, print_fn=print_fn)
        if handle is not None:
            return dict(shared_pole=handle)
    root = Path(run_dir).resolve() / (str(label) + "_shared_pole")
    # A failed transaction is evidence, not permission to overwrite its files.
    exists = multihost_utils.broadcast_one_to_all(np.asarray(root.exists() if jax.process_index() == 0 else False))
    if bool(exists):
        raise ValueError(f"GATE shared_pole_output: {root} exists; use a fresh run directory or a compatible restart member")
    if jax.process_index() == 0:
        root.mkdir(parents=True)
    multihost_utils.sync_global_devices("shared-pole-output-directory")
    qids = np.asarray(sym.q_irr_full_idx, np.int64)
    grid = (meta.nkx, meta.nky, meta.nkz)
    perm, wraps = centroid_source_map_and_wrap(
        np.asarray(centroid_indices), sym.sym_matrices, sym.translations,
        np.asarray(meta.fft_grid), extend_trs=True)
    qt = QirrTables(irr_idx_q=sym.irr_idx_q, sym_idx_q=sym.sym_idx_q,
        q_irr_frac=bgw_integer_q_to_fractional(sym.q_irr_kgrid_int, grid),
        sym_perm=perm, L_table=wraps, n_sym_spatial=len(sym.sym_matrices))
    tables = dict(qirr=qt, q_irr_full_idx=qids, sym=sym)
    coulomb = _coulomb_resource(V_q, meta, sym, mesh_xy, root / "coulomb.h5")
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
    record("bank", produce_w_bank(wfns, meta, config, mesh_xy=mesh_xy,
        sym=sym, sample_plan=recipe, bank_io=bank))
    record("moments", compute_response_moments(wfns, meta, config,
        mesh_xy=mesh_xy, sym=sym, bank_io=bank))
    # The constructor owns scratch reads, actual pencil planning and the
    # final writer. It must query its own native workspace at the actual R.
    result = construct_shared_poles(bank, bank, meta, config,
        mesh_xy=mesh_xy, output=str(root / "model.h5"))
    record("constructor", result)
    header = result["model_header"]
    handle = dict(path=str(root / "model.h5"), identity=identity,
                  digest=header["digest"], K=list(header["K"]))
    ledger.live_stages = ()
    if tensors_filename is not None:
        receipts["restart_member"] = register_shared_pole_restart_member(
            tensors_filename, handle["path"], expected_identity=identity,
            mesh_xy=mesh_xy, capacity=ledger)
    receipts["seconds"] = time.monotonic() - started
    receipts["handle"] = handle
    if jax.process_index() == 0:
        (root / "construction_receipt.json").write_text(_json(receipts) + "\n")
    return dict(shared_pole=handle)
