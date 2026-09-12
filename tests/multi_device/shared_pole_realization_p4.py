"""Packed physical-model realization and complex Gamma-head integration.

The independent oracle transforms canonical factor columns before packing.
It covers parent and child q selections, local and nonlocal endpoint maps,
causal complex weights, changed numerical inputs, callable reuse and recipe
identity/refusal. This is a planted P4 integration gate, not an SC QP verdict.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from gw import gw_jax as driver
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from file_io import shared_pole_store as store
    from gw.qgrid_symmetry import shared_pole_operator_realizer
    from gw.shared_pole_head import _realized_gamma_body
    from gw.shared_pole_recipe import shared_real_pole_v1_r3b, RECIPE_HASH, table_hash

    mesh = driver.RUNTIME.mesh
    assert jax.process_count() == jax.device_count() == 4
    helpers = runpy.run_path("tests/test_shared_pole_store.py")
    face = P(None, "x", "y")
    rng = np.random.default_rng(2026091047)
    def put(value, spec=face):
        value = np.asarray(value)
        return jax.make_array_from_callback(value.shape, NamedSharding(mesh, spec),
                                            lambda index: value[index])
    def error(value, reference):
        return float(jnp.max(jnp.abs(value-put(reference))))
    cases = []
    for identity_layout in (False, True):
        meta, tables, recipe, identity = helpers["_sigma_fixture"](
            mesh, identity_layout=identity_layout)
        header = store._metadata(meta, tables, recipe, identity)
        basis = meta.mu_basis
        def pack_matrix(value):
            return basis.pack_host(basis.pack_host(value, axis=1), axis=2)
        def oracle(factors, poles, counts, s, qids):
            qfrac = np.stack(np.unravel_index(qids, header["grid"]), axis=1)/np.asarray(header["grid"])
            rotations = np.asarray(header["operations"]["rotation"])
            qt = header["qirr"]
            perm, wraps = np.asarray(qt["sym_perm"]), np.asarray(qt["L_table"])
            result = np.zeros((len(qids), basis.n_logical, basis.n_logical), complex)
            for q, point in enumerate(qfrac):
                operations = [row for row in header["operations"]["authorized_rows"]
                              if np.max(abs(rotations[row]@point-point-
                                            np.rint(rotations[row]@point-point))) < 1e-12]
                weights = np.where(np.arange(poles.shape[-1]) < counts[q], 1/(s-poles[q]), 0)
                for row in operations:
                    columns = np.exp(2j*np.pi*(wraps[row]@point))[:, None]*factors[q, perm[row]]
                    if header["operations"]["antiunitary"][row]:
                        columns = columns.conj()
                    result[q] += (columns*weights[None, :])@columns.conj().T/len(operations)
            return pack_matrix(result)
        factors = rng.normal(size=(3, basis.n_logical, 4)) + 1j*rng.normal(size=(3, basis.n_logical, 4))
        poles = np.broadcast_to(np.array([.3, .8, 1.7, 2.2]), (3, 4)).copy()
        counts = np.array([3, 4, 2])
        packed_factors = basis.pack_host(factors, axis=1)
        s = (1+.5j)**2
        weights = np.where(np.arange(4)[None, :] < counts[:, None], 1/(s-poles), 0)
        raw = np.einsum("qmk,qk,qnk->qmn", packed_factors, weights, packed_factors.conj())
        for label, qids in (("parents", np.array(header["q_irr_full_idx"])),
                            ("children", np.array([2, 3, 7]))):
            realize = shared_pole_operator_realizer(meta, header, q_full_idx=qids, mesh_xy=mesh)
            assert realize is shared_pole_operator_realizer(meta, deepcopy(header), q_full_idx=qids, mesh_xy=mesh)
            expected = oracle(factors, poles, counts, s, qids)
            got, partner = realize(put(raw), put(raw.swapaxes(-1, -2)))
            twice, _ = realize(got, partner)
            changed, _ = realize(2*put(raw), 2*put(raw.swapaxes(-1, -2)))
            row = dict(identity_layout=identity_layout, selection=label,
                       oracle_error=error(got, expected),
                       transpose_error=error(partner, expected.swapaxes(-1, -2)),
                       idempotence_error=float(jnp.max(abs(twice-got))),
                       changed_input_error=float(jnp.max(abs(changed-2*got))),
                       original_defect=float(np.max(abs(raw-expected))))
            assert max(row[key] for key in ("oracle_error", "transpose_error", "idempotence_error", "changed_input_error")) < 2e-11, row
            assert row["original_defect"] > .1
            cases.append(row)
        gamma = shared_pole_operator_realizer(meta, header, q_full_idx=np.array([0]), mesh_xy=mesh)
        evaluate = _realized_gamma_body(mesh, gamma)
        assert evaluate is _realized_gamma_body(mesh, gamma)
        # Asymmetric V proves that the physical correction, not total V+Wc,
        # is projected. This matters even when fitted V has a small defect.
        v = pack_matrix(np.diag(np.linspace(1., 2., basis.n_logical))[None].astype(complex))
        operands = (put(packed_factors[:1]), put(poles[:1], P(None, "y")),
                    put(counts[:1], P()), put(v))
        errors = []
        before = None
        for scale, z in ((1., 1+.5j), (1.2, .7j), (.8, 0j), (1.1, .9+2j)):
            current = (scale*operands[0], *operands[1:])
            got = evaluate(put(np.array(z*z), P()), *current)
            expected = v + oracle(scale*factors[:1], poles[:1], counts[:1], z*z, np.array([0]))
            errors.append(error(got, expected))
            if before is None:
                before = evaluate._cache_size()
                compiled = evaluate.lower(put(np.array(z*z), P()), *current).compile()
                assert compiled.memory_analysis() is not None
                if jax.process_index() == 0:
                    args.output.mkdir(parents=True, exist_ok=True)
                    (args.output/f"head_{identity_layout}.hlo").write_text(compiled.as_text())
            assert evaluate._cache_size() == before
        assert max(errors) < 2e-11, errors
        cases.append(dict(identity_layout=identity_layout, selection="Gamma head",
                          sample_errors=errors, executable_specializations=before))
    refusals = []
    for label, mutate in (
            ("legacy_missing", lambda h: h["recipe"].pop("operator_realization")),
            ("unknown_version", lambda h: h["recipe"].update(operator_realization="unknown")),
            ("untyped_operations", lambda h: h["operations"].update(typing_source="")),
            ("wrong_antiunitary_typing", lambda h: h["operations"].update(antiunitary=[False]*12)),
            ("unmeasured_representation", lambda h: h.update(representation="unknown"))):
        bad = deepcopy(header)
        mutate(bad)
        try:
            shared_pole_operator_realizer(meta, bad, q_full_idx=np.array([0]), mesh_xy=mesh)
        except ValueError:
            refusals.append(label)
        else:
            raise AssertionError("missing refusal: "+label)
    for qids in (np.array([0, 0]), np.array([-1]), np.array([9]), np.array([.5])):
        try:
            shared_pole_operator_realizer(meta, header, q_full_idx=qids, mesh_xy=mesh)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid q selection accepted")
    legacy_recipe = dict(recipe)
    legacy_recipe.pop("operator_realization")
    legacy_header = store._metadata(meta, tables, legacy_recipe, identity)
    assert legacy_header["recipe_hash"] != header["recipe_hash"]
    legacy_table = dict(shared_real_pole_v1_r3b)
    legacy_table.pop("operator_realization")
    assert table_hash(legacy_table) != RECIPE_HASH
    # The evaluated-operator seam must retain exactly the raw factor gate's
    # spectrum/thresholds, while rejecting both overscreening and non-Hermiticity.
    import distrib_la
    from gw.shared_pole_constructor import shared_pole_passivity, shared_pole_operator_passivity
    from gw.shared_pole_recipe import shared_real_pole_gates_v1_r3b as gates
    eig = distrib_la.plan("eigh", mesh, n=8, backend="distributed")
    def mm(a, b, **kwargs):
        return distrib_la.matmul(a, b, mesh=mesh, backend="distributed", **kwargs)
    inverse = put(np.eye(8, dtype=complex)[None])
    model = (inverse*.2, put(np.ones((1, 8)), P()), put(np.ones((1, 8), bool), P()))
    wc = -inverse*.04/(1+.25**2)
    raw_checks = jax.jit(lambda model, inverse: shared_pole_passivity(
        model, inverse, eta_ry=.25, matmul=mm, eigh=eig.batched, gates=gates))(model, inverse)
    operator_checks = jax.jit(lambda wc, inverse: shared_pole_operator_passivity(
        wc, inverse, matmul=mm, eigh=eig.batched, gates=gates))
    checks = operator_checks(wc, inverse)
    for name in raw_checks:
        np.testing.assert_allclose(np.asarray(checks[name]), np.asarray(raw_checks[name]), atol=1e-12)
    assert bool(jnp.all(checks["passivity"]))
    assert not bool(jnp.all(operator_checks(-2*inverse, inverse)["passivity"]))
    perturbation = np.zeros((1, 8, 8), complex)
    perturbation[0, 0, 1] = .01j
    assert not bool(jnp.all(operator_checks(wc+put(perturbation), inverse)["passivity"]))
    report = dict(status="PASS", job_step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"],
                  scope=__doc__, cases=cases, metadata_refusals=refusals,
                  operator_passivity="raw spectrum parity; upper-bound and anti-Hermitian red twins refused",
                  current_recipe_hash=RECIPE_HASH,
                  realization=recipe["operator_realization"])
    if jax.process_index() == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
