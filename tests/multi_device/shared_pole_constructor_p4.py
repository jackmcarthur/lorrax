"""P4 constructor gates against an independent planted latent measure.

Run through lx with four ranks; each rank executes every check and only the
replicated receipt is written by rank zero. Both resolved dense policies are
tested on a mesh whose x and y axes exceed one. No campaign bank is opened.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import subprocess


def run_checks(mesh):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_constructor import (
        assemble_shared_pole_pencil, reduce_shared_pole_pencil,
        apply_shared_pole_zero_policy, sort_shared_pole_columns,
        shared_pole_passivity,
        retained_moment_identity,
        _direction_states,
    )
    from gw.shared_pole_recipe import shared_real_pole_gates_v1_r3b as gates

    assert jax.process_count() == 4
    assert int(mesh.shape["x"]) > 1 and int(mesh.shape["y"]) > 1
    face = NamedSharding(mesh, P(None, "x", "y"))
    replicated = NamedSharding(mesh, P())

    def placed(a, sharding=face):
        a = np.asarray(a)
        return jax.make_array_from_callback(a.shape, sharding, lambda index: a[index])

    def relative(got, expected):
        target = placed(expected)
        return float(jnp.linalg.norm(got - target) / jnp.linalg.norm(target))

    rng = np.random.default_rng(320)
    # Unequal ranks, complex non-real residue off-diagonals, and an exact
    # duplicate role at a self-conjugate sample all exercise real block cases.
    c = .03 * (rng.normal(size=(2, 8, 9)) + 1j * rng.normal(size=(2, 8, 9)))
    c[0, :, 7:] = 0
    poles = np.broadcast_to(np.linspace(.2, 2.0, 9), (2, 9)).copy()
    m1 = c @ c.conj().swapaxes(-1, -2) / 2
    m3 = (c * poles[:, None, :]) @ c.conj().swapaxes(-1, -2) / 2
    nodes = [-.08, -.08, .6 + .3j, .6 - .3j, 1.4 + .5j, 1.4 - .5j]
    states, latent = [], []
    for s in nodes:
        q = rng.normal(size=(2, 8, 2)) + 1j * rng.normal(size=(2, 8, 2))
        x = (c.conj().swapaxes(-1, -2) @ q) / (s - poles)[:, :, None]
        dx = -x / (s - poles)[:, :, None]
        states.append((s, placed(q), placed(c @ x), placed(c @ dx)))
        latent.append(x)
    qi = rng.normal(size=(2, 8, 2)) + 1j * rng.normal(size=(2, 8, 2))
    infinity = tuple(placed(a) for a in (qi, m1 @ qi, m3 @ qi))
    latent.append(c.conj().swapaxes(-1, -2) @ qi)
    x = np.concatenate(latent, axis=-1)
    exact = (x.conj().swapaxes(-1, -2) @ x,
             x.conj().swapaxes(-1, -2) @ (poles[:, :, None] * x), c @ x)
    rows = []
    models = []
    for layout in ("local", "distributed"):
        resolution = linalg_resolution({"linalg": layout})
        mm = partial(distrib_la.matmul, mesh=mesh, backend="auto",
                     batched_route=resolution.batched_route)
        pencil = assemble_shared_pole_pencil(states, infinity, matmul=mm)
        errors = [relative(a, b) for a, b in zip(pencil, exact)]
        assert max(errors) < 1e-12, (layout, errors)
        rows.append(dict(name="all_block_types", layout=layout, status="PASS", errors=errors))
        side = pencil[0].shape[-1]
        eig = distrib_la.plan("eigh", mesh, n=side,
                              backend=resolution.eigh_backend,
                              batched_route=resolution.batched_route)
        active = placed(np.ones((2, side), dtype=bool), replicated)
        model, diagnostics, coefficients = reduce_shared_pole_pencil(
            pencil, active, eigh=eig.batched, matmul=mm, gates=gates)
        for name in ("gram_diagonal_positive", "gram_valid", "retained_metric_positive"):
            assert bool(jnp.all(diagnostics[name])), (layout, name)
        model, zero = apply_shared_pole_zero_policy(model, gates=gates)
        assert bool(jnp.all(zero["zero_policy"]))
        selector = np.broadcast_to(np.eye(side, dtype=np.complex128)[:, -2:], (2, side, 2))
        retained = retained_moment_identity(pencil, coefficients, model, placed(selector), matmul=mm)
        assert max(float(jnp.max(v)) for v in retained.values()) < 1e-10, retained
        model, order = sort_shared_pole_columns(model, mesh_xy=mesh)
        factor, t, mask = model
        counts = np.asarray(jnp.sum(mask, axis=-1)).tolist()
        assert counts == [7, 9], (layout, counts)
        assert np.asarray(jnp.sum(mask, axis=-1, dtype=jnp.int64)).dtype == np.int64
        assert bool(jnp.all(mask == (jnp.arange(side)[None, :] < jnp.sum(mask, axis=-1)[:, None])))
        assert bool(jnp.all(jnp.where(mask[:, None, :], True, factor == 0)))
        assert bool(jnp.all(jnp.where(mask, True, t == 1)))
        expected_poles = np.ones((2, side))
        expected_poles[0, :7], expected_poles[1, :9] = poles[0, :7], poles[1]
        pole_error = float(jnp.max(jnp.abs(t - placed(expected_poles, replicated))))
        assert pole_error < 1e-10, pole_error
        moment_errors = [relative(mm(factor, factor, transb="C"), 2 * m1),
                         relative(mm(factor * t[:, None, :], factor, transb="C"), 2 * m3)]
        assert max(moment_errors) < 1e-10, moment_errors
        rows.append(dict(name="ritz_moments_ragged_sorted", layout=layout, status="PASS",
                         K=counts, pole_max_absolute_ry2=pole_error,
                         moment_relative=moment_errors, eig_plan=eig.describe(),
                         retained_moment_relative={k: np.asarray(v).tolist() for k, v in retained.items()},
                         permutation=np.asarray(order).tolist()))
        pe = distrib_la.plan("eigh", mesh, n=8, backend=resolution.eigh_backend,
                             batched_route=resolution.batched_route)
        inverse_v = placed(np.broadcast_to(np.eye(8, dtype=np.complex128), (2, 8, 8)))
        passive = shared_pole_passivity(model, inverse_v, eta_ry=.02,
                                       matmul=mm, eigh=pe.batched, gates=gates)
        assert bool(jnp.all(passive["passivity"]))
        red = shared_pole_passivity((factor * 100, t, mask), inverse_v,
                                   eta_ry=.02, matmul=mm, eigh=pe.batched, gates=gates)
        assert not bool(jnp.any(red["passivity"]))
        rows.append(dict(name="passivity_red_twin", layout=layout, status="PASS",
                         positive_max=np.asarray(passive["passivity_max"]).tolist(),
                         red_max=np.asarray(red["passivity_max"]).tolist()))
        # A line and imaginary role share z=i*h. The relative near-cut
        # doublet straddles both the singular cutoff and the requested width.
        sample = placed(-np.diag([.8, .10000001, .09999999, .01, .008, .006, .004, .002])[None].astype(np.complex128))
        reads, admissions = [], []
        def read_once(sample_id, retained_states):
            reads.append(sample_id)
            return sample, sample * .01
        recipe = dict(fit_ids=[0], held_ids=[], distinct_id=[0, 0],
                      z_ry=[.2j, .2j], role=[0, 1], held=[False, False],
                      direction_cutoff=.125, imaginary_width=2,
                      multiplet_relative_tolerance=1e-6)
        svd = distrib_la.plan("eigh", mesh, n=16, backend=resolution.eigh_backend,
                              batched_route=resolution.batched_route)
        selected, selected_masks, roles = _direction_states(
            read_once, recipe, eigh_plan=pe, svd_plan=svd, matmul=mm,
            column_extent=lambda width: 2*((width+1)//2), logical_n=8,
            admit=admissions.append, infinity_carrier=2)
        assert reads == [0] and len(selected) == 1
        selected, selected_masks, roles = selected[0], selected_masks[0], roles[0]
        assert len(selected) == 2
        assert [role["width"] for role in roles] == [3, 3], roles
        projector = np.diag([1., 1., 1., 0., 0., 0., 0., 0.])[None]
        selection_errors = [relative(mm(state[1], state[1], transb="C"), projector)
                            for state in selected]
        assert max(selection_errors) < 1e-10, selection_errors
        assert all(int(jnp.sum(mask)) == 3 for mask in selected_masks)
        assert admissions == [2, 6, 10]
        rows.append(dict(name="directions_multiplet_dedup", layout=layout, status="PASS",
                         reads=reads, roles=roles, projector_relative=selection_errors))
        models.append(model)

    # Gate on the actual zero-policy implementation, not a duplicated predicate.
    zero_c = np.zeros((2, 8, 4), dtype=np.complex128)
    zero_c[:, 0] = [1e-6, 1e-6, 1., 1.]
    zero_t = placed(np.broadcast_to([-1e-8, 5e-7, 2e-6, 1.], (2, 4)), replicated)
    zero_mask = placed(np.ones((2, 4), dtype=bool), replicated)
    _, good = apply_shared_pole_zero_policy((placed(zero_c), zero_t, zero_mask), gates=gates)
    _, bad = apply_shared_pole_zero_policy((placed(np.ones_like(zero_c)), zero_t, zero_mask), gates=gates)
    assert bool(jnp.all(good["zero_policy"])) and not bool(jnp.any(bad["zero_policy"]))
    rows.append(dict(name="zero_weight_red_twin", status="PASS",
                     benign_fraction=np.asarray(good["dropped_factor_weight_fraction"]).tolist(),
                     red_fraction=np.asarray(bad["dropped_factor_weight_fraction"]).tolist()))
    assert len(rows) == 9
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from runtime import initialize_communicator_stack, finalize_process
    initialize_communicator_stack()
    import jax
    from common.collectives import resolve_mesh
    rows = run_checks(resolve_mesh())
    result = dict(status="PASS", checks=rows, expected_checks=9,
                  jobid=os.environ.get("SLURM_JOB_ID"), stepid=os.environ.get("SLURM_STEP_ID"),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  scope="P4 planted constructor kernels, both dense plans; no production bank, storage or campaign parity")
    if jax.process_index() == 0:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, allow_nan=False), flush=True)
    finalize_process()


if __name__ == "__main__":
    main()
