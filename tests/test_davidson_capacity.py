"""Near/full-spectrum Davidson: bounded storage and true residual convergence."""
def initial_subspace():
    from bse.bse_davidson_helpers import init_bse_subspace
    return init_bse_subspace


def solve_case(mesh, *, padded=False, roots=None, tolerance=1e-10):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    from common.collectives import gather_to_host
    from solvers.davidson_fixed import plan_davidson, CONVERGED, NO_DIRECTIONS
    jax.config.update("jax_enable_x64", True)
    side_x, side_y = int(mesh.shape["x"]), int(mesh.shape["y"])
    nc = ((3+side_x-1)//side_x)*side_x if padded else 2*side_x
    nv = ((3+side_y-1)//side_y)*side_y if padded else 2*side_y
    lc, lv, nk = (3, 3, 2) if padded else (nc, nv, 2)
    shape = (nc, nv, nk)
    indices = np.arange(nc*nv*nk).reshape(shape)[:lc, :lv].reshape(-1)
    physical, dimension = len(indices), nc*nv*nk
    roots = physical if roots is None else int(roots)
    rng = np.random.default_rng(34)
    a = rng.normal(size=(physical, physical)) + 1j*rng.normal(size=(physical, physical))
    h = (a+a.conj().T)*0.03 + np.diag(np.linspace(1., 2., physical))
    dense = np.eye(dimension, dtype=np.complex128)*1000
    dense[np.ix_(indices, indices)] = h
    sh = NamedSharding(mesh, P(None, "x", "y", None))
    matrix = device_put_process_local(dense, NamedSharding(mesh, P()))
    eps_c = np.full((nk, nc), 1000.)
    eps_v = np.full((nk, nv), -1000.)
    eps_c[:, :lc] = 0.4 + np.arange(lc)*0.04
    eps_v[:, :lv] = -0.1 - np.arange(lv)*0.03
    start = initial_subspace()(jnp.asarray(eps_c), jnp.asarray(eps_v), roots,
        n_cond=lc, n_val=lv, mesh=mesh, sharding=sh)

    def apply(data, rows):
        return (rows.reshape(rows.shape[0], -1) @ data.T).reshape(rows.shape)

    def precondition(data, residual, values, _vectors):
        diagonal = jnp.real(jnp.diag(data)).reshape(shape)
        denom = diagonal[None]-values[:, None, None, None]+1e-3
        return residual/jnp.where(jnp.abs(denom)<1e-12, 1e-12, denom)

    plan = plan_davidson(apply, precondition, n_eig=roots, capacity=10*roots,
                         vector_shape=shape, vector_sharding=sh)
    assert plan.capacity == min(10*roots, dimension)
    assert plan.workspace_specs["projected"].shape == (plan.capacity, plan.capacity)
    # Random starts are deterministic, but their iteration count is not a
    # solver contract. Keep the true-residual tolerance for either RNG stream.
    values, vectors, info = plan.solve(matrix, start, tolerance, 60, 0)
    values, vectors, info = jax.tree.map(gather_to_host, (values, vectors, info))
    status = int(info.status)
    expected_status = NO_DIRECTIONS if tolerance < 1e-25 else CONVERGED
    assert status == expected_status, (status, info)
    assert int(info.active_size) <= plan.capacity
    if expected_status == CONVERGED:
        np.testing.assert_allclose(values, np.linalg.eigvalsh(h)[:roots], atol=2e-11, rtol=2e-11)
        actual_residual = np.linalg.norm(dense @ vectors.reshape(roots, -1).T
                                       - vectors.reshape(roots, -1).T*values[None], axis=0)
        assert np.max(actual_residual) < 3e-9
        if roots == physical:
            assert int(info.iterations) == 1
    else:
        assert np.max(info.residuals) > 0
        assert int(info.matvecs) == roots  # no correction H/store past the full basis
    pad = np.ones(dimension, bool)
    pad[indices] = False
    assert np.max(np.abs(vectors.reshape(roots, -1)[:, pad]), initial=0.) < 1e-13
    return dict(roots=roots, physical=physical, dimension=dimension,
        capacity=plan.capacity, iterations=int(info.iterations), status=status,
        max_residual=float(np.max(info.residuals)))


def run_cases(mesh):
    from solvers.davidson_fixed import plan_davidson
    out = [solve_case(mesh), solve_case(mesh, roots=6),
           solve_case(mesh, tolerance=1e-30)]
    if int(mesh.shape["x"]) <= 3 and int(mesh.shape["y"]) <= 3:
        out.extend([solve_case(mesh, padded=True), solve_case(mesh, padded=True, roots=15)])
    try:
        plan_davidson(lambda data, x: x, lambda data, r, e, x: r,
                      n_eig=4, capacity=7, vector_shape=(10,))
    except ValueError:
        pass
    else:
        raise AssertionError("undersized partial-spectrum capacity was accepted")
    return out


def test_full_near_full_and_padded_davidson_capacity():
    import jax
    import numpy as np
    from jax.sharding import Mesh
    for side in (1, 2):
        if len(jax.devices("cpu")) >= side*side:
            run_cases(Mesh(np.asarray(jax.devices("cpu")[:side*side]).reshape(side, side), ("x", "y")))


def test_full_physical_seed_has_exact_identity_gram_and_partial_seed_keeps_support():
    import jax
    import numpy as np
    import pytest
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.collectives import gather_to_host
    jax.config.update("jax_enable_x64", True)
    devices = jax.devices("cpu")
    side = 2 if len(devices) >= 4 else 1
    mesh = Mesh(np.asarray(devices[:side*side]).reshape(side, side), ("x", "y"))
    sh = NamedSharding(mesh, P(None, "x", "y", None))
    nc = nv = 4
    ec = np.tile([.3, .4, .5, 1000.], (2, 1))
    ev = np.tile([-.1, -.2, -.3, -1000.], (2, 1))
    init = initial_subspace()
    full = np.asarray(gather_to_host(init(ec, ev, 18, n_cond=3, n_val=3,
        n_random=5, seed=89, mesh=mesh, sharding=sh)))
    np.testing.assert_array_equal(full.reshape(18, -1) @ full.reshape(18, -1).conj().T, np.eye(18))
    assert np.count_nonzero(full) == 18
    assert np.all(full[:, 3:] == 0) and np.all(full[:, :, 3:] == 0)
    # Lowest-energy unit rows retain their physical meaning. Gaussian bits now
    # come from the distributed JAX owner, rather than a replicated NumPy bank.
    partial = gather_to_host(init(ec, ev, 7, n_cond=3, n_val=3,
        n_random=5, seed=89, mesh=mesh, sharding=sh))
    picks = np.lexsort((np.arange(18), (ec[:, :3, None]-ev[:, None, :3]).reshape(-1)))[:2]
    k, c, v = np.unravel_index(picks, (2, 3, 3))
    expected = np.zeros((2, nc, nv, 2), np.complex128)
    expected[np.arange(2), c, v, k] = 1
    np.testing.assert_array_equal(partial[:2], expected)
    np.testing.assert_allclose(np.linalg.norm(partial.reshape(7, -1), axis=1), 1., atol=3e-15)
    assert np.all(partial[:, 3:] == 0) and np.all(partial[:, :, 3:] == 0)
    with pytest.raises(ValueError, match="logical transition space"):
        init(ec, ev, 19, n_cond=3, n_val=3)


if __name__ == "__main__":
    import json
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    out = run_cases(stack.mesh)
    print(f"r{jax.process_index()} DAVIDSON_CAPACITY PASS " + json.dumps(out), flush=True)
    runtime.finalize_process(0)
