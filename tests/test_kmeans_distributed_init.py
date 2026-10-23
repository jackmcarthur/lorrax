"""Compute-node gate for grid-sharded seeding and a physical Lloyd step.

The independent NumPy oracle sums every logical point. The memory guard
also rejects the otherwise numerically equivalent generic-gather twin.
"""
import json
import re
import sys


def main():
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from centroid import distribution as dist
    from centroid import kmeans_isdf as km
    from common.collectives import gather_to_host, prepare_mesh
    mesh = prepare_mesh(stack.mesh)
    side, nc = 7, 9
    physical = side**3
    rho = (1.+np.sin(np.arange(physical)*.137)**2).reshape(side, side, side)
    positions, weights, tag = dist.place_grid(rho, mesh, ('x', 'y'))
    axis = jnp.linspace(0, 1, side, endpoint=False, dtype=jnp.float64)
    old_positions = jnp.stack(jnp.meshgrid(axis, axis, axis, indexing='ij'),
                              axis=-1).reshape(-1, 3)
    physical_points = np.asarray(old_positions)
    grid = np.asarray(gather_to_host(positions))
    np.testing.assert_array_equal(grid[:physical], physical_points)
    assert np.all(grid[physical:] == 0)
    placed_weights = np.asarray(gather_to_host(weights))
    np.testing.assert_array_equal(placed_weights[:physical], rho.reshape(-1))
    assert np.all(placed_weights[physical:] == 0)
    tied = jax.jit(lambda: jnp.where(
        (jnp.arange(tag.carrier) == physical//3)
        | (jnp.arange(tag.carrier) == physical-1), 1., 0.),
        out_shardings=weights.sharding)()
    chosen = dist.grid_argmax(positions, tied, mesh, ('x', 'y'))
    np.testing.assert_array_equal(np.asarray(gather_to_host(chosen)),
                                  physical_points[physical//3])
    key = jax.random.PRNGKey(504)
    offsets_np = np.asarray([(i, j, k) for i in range(-2, 3)
                             for j in range(-2, 3) for k in range(-2, 3)])

    def numpy_distance(points, center, metric, inversion):
        centers = [center, (-center) % 1.] if inversion else [center]
        best = np.full(len(points), np.inf)
        for image in centers:
            delta = points-image
            delta -= np.round(delta)
            values = delta[:, None, :]+offsets_np[None, :, :]
            best = np.minimum(best, np.einsum('pmi,ij,pmj->pm', values,
                                              metric, values).min(axis=1))
        return best

    seed_checks = []
    for skew, inversion in ((False, False), (True, False), (True, True)):
        metric = np.asarray([[1., .3, .05], [.3, 1.4, .1], [.05, .1, 2.]]) if skew else np.diag([1., 1.4, 2.])
        offsets = jnp.asarray(km.build_min_image_offsets(metric))
        syms = (jnp.stack((jnp.eye(3, dtype=jnp.int32), -jnp.eye(3, dtype=jnp.int32)))
                if inversion else None)
        tau = jnp.zeros((2, 3)) if inversion else None
        baseline = km.kmeans_pp_init(old_positions, jnp.asarray(rho.reshape(-1)),
            jnp.asarray(metric), nc, key, offsets=offsets, Rinv=syms, tau=tau)
        actual = km.kmeans_pp_init(positions, weights, jnp.asarray(metric), nc, key,
            offsets=offsets, Rinv=syms, tau=tau, n_logical=physical,
            mesh=mesh, mesh_axis=('x', 'y'))
        actual = np.asarray(gather_to_host(actual))
        np.testing.assert_array_equal(actual, np.asarray(baseline))
        rng, minimum, reference = key, np.ones(physical), []
        for number in range(nc):
            rng, sub = jax.random.split(rng)
            uniforms = np.asarray(jax.random.uniform(sub, (physical,), dtype=jnp.float64,
                minval=np.finfo(np.float64).tiny, maxval=1.-np.finfo(np.float64).tiny))
            scores = np.log(rho.reshape(-1))+np.log(np.maximum(minimum, np.finfo(float).tiny))
            index = int(np.argmax(scores-np.log(-np.log(uniforms))))
            point = physical_points[index]
            if inversion:
                point = min((tuple(point % 1.), tuple((-point) % 1.)))
                point = np.asarray(point)
            reference.append(point)
            distance = numpy_distance(physical_points, point, metric, inversion)
            minimum = distance if number == 0 else np.minimum(minimum, distance)
        np.testing.assert_array_equal(actual, np.asarray(reference))
        seed_checks.append(dict(skew=skew, inversion=inversion,
                                padding=tag.carrier-physical, exact=True))

    # One literal Lloyd update, independent nearest-image and weighted sums.
    avec = np.diag([1., 1.3, 1.7])
    probability = jnp.asarray(rho.reshape(-1))
    probability /= jnp.sum(probability)
    initial_index = np.asarray(jax.random.choice(key, physical, shape=(nc,),
                                                 p=probability, replace=False))
    initial = physical_points[initial_index]
    delta = physical_points[:, None, :]-initial[None, :, :]
    delta -= np.round(delta)
    metric = avec@avec.T
    distances = np.einsum('pci,ij,pcj->pc', delta, metric, delta)
    labels = distances.argmin(axis=1)
    expected = initial.copy()
    for centroid in range(nc):
        picked = labels == centroid
        if np.any(picked):
            expected[centroid] += np.sum(rho.reshape(-1)[picked, None]
                *delta[picked, centroid], axis=0)/rho.reshape(-1)[picked].sum()
    expected %= 1.
    returned_labels, centers, steps, movement = km.weighted_kmeans_jax(
        avec, rho, nc, max_steps=1, tolerance=0., seed=504,
        mesh=mesh, mesh_axis=('x', 'y'), init_method='random')
    actual = np.asarray(gather_to_host(centers))
    np.testing.assert_allclose(actual, expected, rtol=0., atol=3e-15)
    labels = np.asarray(gather_to_host(returned_labels))
    assert np.all(labels[physical:] == -1)
    assert steps == 1 and np.isfinite(movement)

    # Zero-weight logical points precede zero-weight padding in weighted
    # top-k ties, even when every logical point must be returned.
    sparse = np.zeros_like(rho)
    sparse.reshape(-1)[123] = 1.
    _, sparse_centers, _, _ = km.weighted_kmeans_jax(
        avec, sparse, physical, max_steps=0, seed=504,
        mesh=mesh, mesh_axis=('x', 'y'), init_method='random')
    sparse_centers = np.asarray(gather_to_host(sparse_centers))
    expected_index = np.asarray(jax.random.choice(key, physical,
        shape=(physical,), p=jnp.asarray(sparse.reshape(-1)), replace=False))
    np.testing.assert_array_equal(sparse_centers, physical_points[expected_index])

    # A large collective is forbidden even when its final answer is exact.
    # The generic positions[argmax] twin previously gathered every point.
    def large_collectives(executable):
        found = []
        for line in executable.as_text().splitlines():
            if ' all-gather(' not in line and ' all-reduce(' not in line:
                continue
            shape = re.search(r'= (?:f64|s64|s32|u32)\[([0-9,]*)\]', line)
            if shape:
                dims = tuple(int(x) for x in shape.group(1).split(',') if x)
                if dims in ((tag.carrier,), (tag.carrier, 3)):
                    found.append(line)
        return found

    metric = jnp.asarray(np.diag([1., 1.4, 2.]))
    offsets = jnp.zeros((1, 3), jnp.int32)
    good = km.kmeans_pp_init.lower(positions, weights, metric, nc, key,
        offsets=offsets, n_logical=physical, mesh=mesh,
        mesh_axis=('x', 'y')).compile()
    assert not large_collectives(good)
    if mesh.size > 1:
        bad = km.kmeans_pp_init.lower(positions, weights, metric, nc, key,
            offsets=offsets, n_logical=physical).compile()
        # The GPU partitioner can optimize this generic gather already.
        # Keep the observed CPU regression as the negative control.
        if jax.default_backend() == "cpu":
            assert large_collectives(bad), 'generic-gather twin did not exercise the guard'
    if jax.process_index() == 0:
        print('KMEANS_DISTRIBUTED_INIT PASS '+json.dumps(dict(
            ranks=jax.process_count(), seed_checks=seed_checks,
            lloyd_max_abs=float(np.max(np.abs(actual-expected))),
            compiled=str(good.memory_analysis()))), flush=True)
    runtime.finalize_process(0)
    return 0


def test_compute_node_only():
    import pytest
    pytest.skip('run as a script on a compute node at P4/P16/P64')


if __name__ == '__main__':
    sys.exit(main())
