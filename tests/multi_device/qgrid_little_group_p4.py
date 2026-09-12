"""Independent scalar residue/group oracle and all-P projection shape proof.

Run through the site's four-rank runtime. The glide (x,y)->(-x,y+1/2)
has nonlocal centroid permutations and nontrivial Bloch phases. The three
parents exercise unitary-only, antiunitary-only and full stabilizers.
"""
import argparse
import json
import os
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from gw import gw_jax as driver
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from symmetry_maps import project_little_group_operator
    from symmetry_maps.qgrid_trs import _little_group_operations

    mesh = driver.RUNTIME.mesh
    assert jax.process_count() == jax.device_count() == 4
    assert int(mesh.shape["x"]) == int(mesh.shape["y"]) == 2
    points = np.array([(x/4, y/4, 0.) for x in range(4) for y in range(4)])
    pulled = points.copy()
    pulled[:, 0] *= -1
    pulled[:, 1] -= .5
    lattice = np.floor(pulled).astype(np.int32)
    source = np.array([np.flatnonzero(np.all(
        np.isclose(points, p), axis=1))[0] for p in pulled-lattice])
    eye = np.eye(3, dtype=np.int32)
    reflection = np.diag([-1, 1, 1]).astype(np.int32)
    rotations = np.stack((eye, reflection, -eye, -reflection))
    perm = np.stack((np.arange(16), source, np.arange(16), source)).astype(np.int32)
    wraps = np.stack((np.zeros_like(lattice), lattice,
                      np.zeros_like(lattice), lattice))
    q = np.array([[.25, 0., 0.], [0., .25, 0.], [0., .5, 0.]])
    ids = np.array([16, 4, 8], dtype=np.int32)
    stabilizers = ([0, 3], [0, 1], [0, 1, 2, 3])
    metadata = dict(q_full_idx=ids, q_irr_frac=q, sym_mats_k=rotations,
                    sym_perm=perm, L_table=wraps,
                    active_symmetry_rows=np.arange(4, dtype=np.int32),
                    kgrid=(4, 4, 4), n_sym_spatial=2, mesh=mesh)
    assert [rows.tolist() for rows in _little_group_operations(
        ids, kgrid=(4, 4, 4), sym_mats_k=rotations,
        active_symmetry_rows=np.arange(4, dtype=np.int32))] == list(map(list, stabilizers))
    rng = np.random.default_rng(2026091041)
    factors = rng.normal(size=(3, 16, 3)) + 1j*rng.normal(size=(3, 16, 3))
    weights = np.array([.31+.73j, -.2+.41j, 1.2-.67j])
    residues = np.einsum("qmp,qnp->pqmn", factors, factors.conj())
    original = np.einsum("p,pqmn->qmn", weights, residues)

    # Independent oracle applies the real-space glide directly to residue
    # columns; it does not call the service's action or projector helpers.
    averaged_residues = np.zeros_like(residues)
    for parent, operations in enumerate(stabilizers):
        for operation in operations:
            phase = np.exp(2j*np.pi*(wraps[operation] @ q[parent]))
            columns = phase[:, None]*factors[parent, perm[operation]]
            if operation >= 2:
                columns = columns.conj()
            averaged_residues[:, parent] += np.einsum(
                "mp,np->pmn", columns, columns.conj())/len(operations)
    expected = np.einsum("p,pqmn->qmn", weights, averaged_residues)
    sh = NamedSharding(mesh, P(None, "x", "y"))
    def put(value):
        return jax.make_array_from_callback(value.shape, sh, lambda index: value[index])
    def error(value, reference):
        return float(jnp.max(jnp.abs(value-put(reference))))
    value, partner = put(original), put(original.swapaxes(-1, -2))
    def project(a, at):
        return project_little_group_operator(a, transposed_partner=at, **metadata)
    executable = jax.jit(project).lower(value, partner).compile()
    average, average_t = executable(value, partner)
    result = dict(oracle_error=error(average, expected),
                  transpose_error=error(average_t, expected.swapaxes(-1, -2)),
                  original_defect=float(np.max(np.abs(original-expected))))
    twice, _ = executable(average, average_t)
    result["idempotence_error"] = float(jnp.max(jnp.abs(twice-average)))
    wrong, _ = executable(value, value.conj())
    result["wrong_time_conjugation_error"] = error(wrong, expected)
    result["residue_oracle_error"] = 0.
    result["residue_min_eigenvalue"] = float(np.linalg.eigvalsh(averaged_residues).min())
    for pole in range(3):
        residue = put(residues[pole])
        projected, _ = executable(residue, put(residues[pole].swapaxes(-1, -2)))
        result["residue_oracle_error"] = max(
            result["residue_oracle_error"], error(projected, averaged_residues[pole]))
    # Changing numerical inputs must not retrieve cached values.
    scaled, _ = executable(2*value, 2*partner)
    result["changed_input_error"] = float(jnp.max(jnp.abs(scaled-2*average)))
    for name in ("oracle_error", "transpose_error", "idempotence_error",
                 "residue_oracle_error", "changed_input_error"):
        assert result[name] < 3e-12, (name, result[name])
    assert result["original_defect"] > .1
    assert result["wrong_time_conjugation_error"] > .1
    assert result["residue_min_eigenvalue"] > -3e-12
    refusals = []
    for name, changed in (
            ("unauthorized_row", dict(active_symmetry_rows=np.array([0, 4]))),
            ("q_mismatch", dict(q_irr_frac=q+.125)),
            ("nonpermutation", dict(sym_perm=np.zeros_like(perm))),
            ("missing_identity", dict(active_symmetry_rows=np.array([1, 3])))):
        try:
            project_little_group_operator(value, transposed_partner=partner,
                                          **(metadata | changed))
        except ValueError:
            refusals.append(name)
        else:
            raise AssertionError("missing metadata refusal: "+name)
    hlo = executable.as_text()
    local_bound = int(np.prod(original.shape)//4)
    extents = [int(np.prod([int(x) for x in shape.split(",") if x]))
               for shape in re.findall(r"c128\[([0-9,]*)\]", hlo)]
    result.update(max_complex_array_elements=max(extents),
                  matrix_tile_elements=local_bound,
                  all_to_all=bool(re.search(r"\ball-to-all\(", hlo)),
                  all_gather=bool(re.search(r"\ball-gather\(", hlo)),
                  while_loop=bool(re.search(r"\bwhile\(", hlo)),
                  metadata_refusals=refusals,
                  output_shardings=[str(x.sharding.spec) for x in (average, average_t)])
    assert max(extents) <= local_bound, result
    assert result["all_to_all"] and result["while_loop"] and not result["all_gather"], result
    result.update(status="PASS", job_step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"])
    if jax.process_index() == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"projection.hlo").write_text(hlo)
        (args.output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
