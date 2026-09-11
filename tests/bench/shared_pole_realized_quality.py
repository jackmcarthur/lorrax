"""Read-only physical-quality audit of a frozen shared-pole map.

Authenticates the original bank/model/V, then explicitly requests the current
operator realization in an in-memory metadata copy. Reads one parent and one
held field at a time through the store owner. No completed artifact is changed.
Every numerical matrix/factor stays on all P processors; only scalar errors,
spectral extrema, identities and capacity receipts reach the host report.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-directory", type=Path, required=True)
    parser.add_argument("--centroids", type=Path, required=True,
                        help="original logical fractional centroid table")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from gw import gw_jax as driver
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la
    from common.centroid_basis import PackedCentroidBasis
    from common.units import RYD_TO_EV
    from file_io import shared_pole_store as store
    from file_io.slab_io import SlabIO
    from gw.qgrid_symmetry import shared_pole_operator_realizer
    from gw.shared_pole_constructor import _sample_point, shared_pole_operator_passivity
    from gw.shared_pole_recipe import (CapacityLedger, shared_real_pole_v1_r3b,
                                       shared_real_pole_gates_v1_r3b as gates)
    from gw.response_bank import _reserve
    from gw.w_isdf import response_coulomb_powers

    mesh = driver.RUNTIME.mesh
    assert jax.process_count() == jax.device_count() == 4
    root = args.map_directory.resolve()
    paths = {name: root/(name+".h5") for name in ("model", "bank", "coulomb")}
    original = store._read_header(paths["model"])
    grid = np.asarray(original["grid"])
    fft = np.asarray(original["fft_grid"])
    fractional = np.loadtxt(args.centroids)
    indices = (np.rint(fractional*fft[None]).astype(np.int32) % fft[None]).astype(np.int32)
    # Canonical identity packing needs no re-created SymMaps object. The
    # authenticated model still supplies every physical operation to Pi_G.
    basis = PackedCentroidBasis.build(indices, None, fft, mesh, identity=True)
    meta = SimpleNamespace(mu_basis=basis, nspinor=1, nk_tot=int(np.prod(grid)),
        n_rmu=basis.n_logical, n_rmu_padded=basis.n_packed,
        nkx=int(grid[0]), nky=int(grid[1]), nkz=int(grid[2]))
    ledger = meta.shared_pole_capacity = CapacityLedger(meta, mesh_xy=mesh)
    ledger.live_stages = ()
    from runtime.padding import padded_axis
    width = padded_axis(max(1, original["Kmax"]), mesh, name="shared_pole_K",
                        specs=((P(None, "x", None, "y"), 3),)).carrier
    factor_bytes = 16*basis.n_packed*width//mesh.size
    digest_rows = min(mesh.size, original["n_q_irr"])
    digest_columns = max(1, (original["Kmax"]+mesh.shape["y"]-1)//mesh.shape["y"])
    digest_panel_bytes = 16*digest_rows*basis.n_canonical*digest_columns//mesh.shape["x"]
    if max(factor_bytes, digest_panel_bytes) > ledger.U_bytes_per_rank:
        raise ValueError("audit factor/read panel exceeds logical U before allocation")
    header = store.validate_shared_pole_model(paths["model"],
        expected_identity=original["identity"], mesh_xy=mesh, capacity=ledger)
    bank = store.validate_shared_pole_bank(paths["bank"],
        expected_identity=header["identity"], mesh_xy=mesh, require_complete=True)
    bank_receipt = json.loads((root/"bank_receipt.json").read_text())
    if bank_receipt["identity"] != header["identity"]:
        raise ValueError("audit bank receipt identity differs from authenticated model")
    coulomb = bank_receipt["coulomb_identity"]
    if Path(coulomb["path"]).resolve() != paths["coulomb"]:
        raise ValueError("audit Coulomb receipt does not name this frozen map")
    descriptor = dict(path=str(paths["bank"]), identity=header["identity"], coulomb=coulomb)
    requested = deepcopy(header)
    realization = shared_real_pole_v1_r3b["operator_realization"]
    requested["recipe"]["operator_realization"] = realization
    recipe = bank["recipe"]
    eta = float(recipe["eta_ev"])/RYD_TO_EV
    face = NamedSharding(mesh, P(None, "x", "y"))
    scalar = NamedSharding(mesh, P())
    n = basis.n_packed
    eig = distrib_la.plan("eigh", mesh, n=n, backend="distributed")
    def mm(a, b, **kwargs):
        return distrib_la.matmul(a, b, mesh=mesh, backend="distributed", **kwargs)
    @jax.jit
    def passive(wc, inverse):
        return shared_pole_operator_passivity(wc, inverse, matmul=mm,
                                              eigh=eig.batched, gates=gates)
    def arrays_bytes(values):
        return sum(int(a.size*a.dtype.itemsize//mesh.size)
                   if a.ndim > 2 else int(a.size*a.dtype.itemsize)
                   for a in jax.tree.leaves(values))
    compiled_rows = []
    def admitted(kernel, operands, stage, native=0):
        executable = kernel.lower(*operands).compile()
        memory = executable.memory_analysis()
        if memory is None:
            raise ValueError("audit compiled memory unavailable")
        hlo = executable.as_text()
        extents = [int(np.prod([int(v) for v in shape.split(",") if v]))
                   for shape in re.findall(r"c128\[([0-9,]*)\]", hlo)]
        largest = 16*max(extents, default=0)
        if largest > ledger.U_bytes_per_rank:
            raise ValueError(f"audit matrix/factor carrier {largest} exceeds logical U")
        _reserve(meta, stage, 0, memory.output_size_in_bytes+memory.temp_size_in_bytes+native)
        compiled_rows.append(dict(stage=stage, arguments=memory.argument_size_in_bytes,
            outputs=memory.output_size_in_bytes, temporaries=memory.temp_size_in_bytes,
            native_workspace_bytes=native, max_complex_array_bytes_per_rank=largest))
        return executable
    def host_checks(checks):
        return {name: np.asarray(value).tolist() for name, value in checks.items()}
    rows = []
    for iq, full_q in enumerate(header["q_irr_full_idx"]):
        ledger.live_stages = ()
        with SlabIO(paths["model"], mode="r", mesh=mesh) as io:
            factors, poles, counts = store.read_shared_pole_matrix(io, (iq, iq+1), meta=meta, header=header)
        model_stage, _ = _reserve(meta, "audit_model", arrays_bytes((factors, poles, counts)))
        ledger.live_stages = (model_stage,)
        h, inverse, coulomb_check = response_coulomb_powers(
            meta, {"linalg": "distributed"}, mesh_xy=mesh, bank_io=descriptor, q_span=(iq, iq+1))
        del h
        inverse_stage, _ = _reserve(meta, "audit_inverse", arrays_bytes(inverse))
        ledger.live_stages = (model_stage, inverse_stage)
        realize = shared_pole_operator_realizer(meta, requested,
            q_full_idx=np.array([full_q]), mesh_xy=mesh)
        @jax.jit(out_shardings=(face, face))
        def evaluate(s, b, lam, count, derivative):
            weights = jnp.where(jnp.arange(lam.shape[-1])[None] < count[:, None], 1/(s-lam), 0)
            weights = jnp.where(derivative, -weights**2, weights)
            raw = mm(b*weights[:, None], b, transb="C")
            partner = jax.lax.with_sharding_constraint(raw.swapaxes(-1, -2), face)
            physical, _ = realize(raw, partner)
            return raw, physical
        @jax.jit(out_shardings=scalar)
        def quality(raw, physical, reference):
            partner = jax.lax.with_sharding_constraint(reference.swapaxes(-1, -2), face)
            target, _ = realize(reference, partner)
            norm = jnp.maximum(jnp.linalg.norm(reference), jnp.finfo(jnp.float64).tiny)
            return jnp.array([jnp.linalg.norm(raw-reference)/norm,
                              jnp.linalg.norm(physical-reference)/norm,
                              jnp.linalg.norm(physical-raw)/norm,
                              jnp.linalg.norm(target-reference)/norm,
                              jnp.linalg.norm(physical-target)/jnp.maximum(
                                  jnp.linalg.norm(target), jnp.finfo(jnp.float64).tiny)])
        controls = (jnp.asarray(-eta**2, jnp.complex128), factors, poles, counts, jnp.asarray(False))
        native_gemm = distrib_la.workspace_bytes_per_rank(eig, "gemm",
            (factors.shape, (1, factors.shape[-1], n)), np.complex128)
        build = admitted(evaluate, controls, "audit_evaluation", native_gemm)
        raw, physical = build(*controls)
        del controls
        values_stage, _ = _reserve(meta, "audit_evaluated_values", arrays_bytes((raw, physical)))
        ledger.live_stages = (model_stage, inverse_stage, values_stage)
        native_passivity = (distrib_la.workspace_bytes_per_rank(eig, "gemm", ((1,n,n),)*2, np.complex128)
                           + distrib_la.workspace_bytes_per_rank(eig, "eigh", ((1,n,n),), np.complex128))
        check = admitted(passive, (physical, inverse), "audit_passivity", native_passivity)
        raw_passivity, projected_passivity = host_checks(check(raw, inverse)), host_checks(check(physical, inverse))
        row = dict(parent=iq, full_q=int(full_q), counts=np.asarray(counts).tolist(),
                   eta_ry=eta, raw_passivity=raw_passivity, projected_passivity=projected_passivity,
                   coulomb=coulomb_check, held=[])
        del raw, physical
        ledger.live_stages = (model_stage, inverse_stage)
        quality_kernel = None
        with SlabIO(paths["bank"], mode="r", mesh=mesh) as io:
            for sample_id in recipe["held_ids"]:
                s = _sample_point(recipe, int(sample_id))**2
                for field in ("Wc", "dWc_ds"):
                    samples = store.read_shared_pole_bank(io, (iq, iq+1), meta=meta, header=bank,
                        sample_span=(int(sample_id), int(sample_id)+1), fields=(field,))
                    reference = samples[field][:, 0]
                    reference_stage, _ = _reserve(meta, "audit_reference", arrays_bytes(reference))
                    ledger.live_stages = (model_stage, inverse_stage, reference_stage)
                    memory = build.memory_analysis()
                    _reserve(meta, "audit_held_evaluation", 0,
                             memory.output_size_in_bytes+memory.temp_size_in_bytes+native_gemm)
                    raw, physical = build(jnp.asarray(s, jnp.complex128), factors, poles, counts,
                                          jnp.asarray(field == "dWc_ds"))
                    values_stage, _ = _reserve(meta, "audit_held_values", arrays_bytes((raw, physical)))
                    ledger.live_stages = (model_stage, inverse_stage, reference_stage, values_stage)
                    if quality_kernel is None:
                        quality_kernel = admitted(quality, (raw, physical, reference), "audit_quality")
                    result = np.asarray(quality_kernel(raw, physical, reference)).tolist()
                    row["held"].append(dict(sample_id=int(sample_id), field=field,
                        z_ry=dict(real=_sample_point(recipe, int(sample_id)).real,
                                  imag=_sample_point(recipe, int(sample_id)).imag),
                        **dict(zip(("raw_relative", "projected_relative", "model_removed_relative",
                                    "reference_removed_relative", "projected_target_relative"), result))))
                    del samples, reference, raw, physical
                    ledger.live_stages = (model_stage, inverse_stage)
        rows.append(row)
        if jax.process_index() == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            with (args.output/"parents.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            print("REALIZED_QUALITY "+json.dumps(dict(parent=iq, full_q=int(full_q),
                projected_passivity=projected_passivity)), flush=True)
        del factors, poles, counts, inverse
    report = dict(status="MEASURED", job_step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"],
        scope=__doc__, source_map=str(root), source_model_digest=header["digest"],
        source_identity=header["identity"], source_recipe_hash=header["recipe_hash"],
        source_operator_realization=header["recipe"].get("operator_realization"),
        requested_operator_realization=realization,
        explicit_override_scope="test-only in-memory realization request after original artifact authentication",
        source_bank_commit=bank["final_commit"], coulomb_identity=coulomb,
        centroid_sha256=hashlib.sha256(args.centroids.read_bytes()).hexdigest(),
        projected_passivity_all=all(all(row["projected_passivity"]["passivity"]) for row in rows),
        rows=rows, compiled=compiled_rows, capacity=ledger.receipt())
    if jax.process_index() == 0:
        (args.output/"result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print("REALIZED_QUALITY_COMPLETE "+json.dumps(dict(
            projected_passivity_all=report["projected_passivity_all"], parents=len(rows))), flush=True)


if __name__ == "__main__":
    main()
