"""Snapped duplicate repair must preserve exact whole-orbit point budgets."""
import numpy as np

from centroid.kmeans_isdf import _restore_orbit_pool


def _numpy_group_pc(G, labels, active, budget):
    """Independent scalar recurrence using only the already-built prefix."""
    residual = G.diagonal().real.copy()
    floor = np.sqrt(np.finfo(float).eps)*residual.max()
    factor = np.zeros((len(G), budget), complex)
    selected = []
    admitted = set()
    available = active.copy()
    taken = []
    committed = 0
    while committed < budget:
        choices = []
        for group in np.unique(labels[active]):
            rows = np.flatnonzero((labels == group) & active)
            if group in admitted or len(rows) > budget-committed:
                continue
            score = np.where(residual[rows] > floor, residual[rows], 0).mean()
            choices.append((score, -group, rows))
        if not choices:
            break
        _, neg_group, rows = max(choices, key=lambda item: item[:2])
        admitted.add(-neg_group)
        committed += len(rows)
        for _ in rows:
            candidates = np.flatnonzero((labels == -neg_group) & available)
            p = candidates[np.argmax(residual[candidates])]
            j = len(selected)
            take = residual[p] > floor
            if take:
                col = (G[:, p]-factor[:, :j]@factor[p, :j].conj())/np.sqrt(residual[p])
                col[p] = np.sqrt(residual[p])
                factor[:, j] = col
                residual = np.maximum(residual-np.abs(col)**2, 0)
            selected.append(p)
            taken.append(take)
            available[p] = False
            residual[p] = 0
    return np.array(selected), np.count_nonzero(taken), factor


def test_grouped_complex_pc_full_rank_and_exhausted_tail():
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.pivoted_cholesky import make_sharded_group_block_pivoted_cholesky_select

    mesh = Mesh(np.array(jax.local_devices()[:1]), ("x",))
    labels = np.repeat(np.arange(7), [1, 2, 3, 4, 5, 4, 3])
    labels = np.r_[labels, 7, 7].astype(np.int32)
    active = np.arange(24) < 22
    for feature_rank in (3, 22):
        rng = np.random.default_rng(200+feature_rank)
        features = rng.normal(size=(22, feature_rank))+1j*rng.normal(size=(22, feature_rank))
        G = np.zeros((24, 24), complex)
        G[:22, :22] = features@features.conj().T
        expected_piv, expected_rank, expected_L = _numpy_group_pc(G, labels, active, 20)
        kernel = make_sharded_group_block_pivoted_cholesky_select(
            mesh, 24, 20, 7, mesh_axis="x")
        result = kernel(
            jax.device_put(G, NamedSharding(mesh, P("x", None))),
            jax.device_put(labels, NamedSharding(mesh, P("x"))),
            jax.device_put(active, NamedSharding(mesh, P("x"))))
        piv, L, rank, d_final, d_taken, trace, info = jax.device_get(result)
        selected = piv[piv >= 0]
        assert int(rank) == expected_rank
        assert np.array_equal(selected[:rank], expected_piv[:rank])
        assert len(selected) <= 20 and len(selected) == len(np.unique(selected))
        assert np.all(selected < 22)
        for group in np.unique(labels[selected]):
            assert np.sum(labels[selected] == group) == np.sum((labels == group) & active)
        np.testing.assert_allclose(L, expected_L, rtol=1e-11, atol=1e-11)
        remainder = G-L@L.conj().T
        assert np.linalg.eigvalsh(remainder).min() > -1e-10
        assert np.all(np.isfinite(d_final)) and np.all(np.isfinite(trace))
        assert np.max(np.abs(L[:, d_taken == 0]), initial=0) == 0
        if feature_rank == 3:
            np.testing.assert_allclose(remainder, 0, atol=1e-11)


def test_duplicate_pool_repair_closure_density_order_and_grid_capacity():
    # A nonsymmorphic glide: (x,y,z) -> (x+1/2,-y,z). Construct the
    # independent integer-grid orbits directly, without the symmetry service.
    grid = (8, 4, 2)
    all_idx = np.array(list(np.ndindex(grid)))
    flat = np.arange(len(all_idx))
    image = all_idx.copy()
    image[:, 0] = (image[:, 0]+4) % 8
    image[:, 1] = (-image[:, 1]) % 4
    partner = np.ravel_multi_index(image.T, grid)
    canonical = np.minimum(flat, partner)
    weight = (1+canonical % 11).reshape(grid).astype(float)
    Rinv = np.stack((np.eye(3), np.diag([1, -1, 1]))).astype(int)
    tau = np.array([[0., 0., 0.], [.5, 0., 0.]])
    seed = all_idx[[0, partner[0]]]

    result = _restore_orbit_pool(seed, grid, weight, Rinv, tau, 7)
    selected = np.ravel_multi_index(result.T, grid)
    assert len(selected) == len(np.unique(selected)) == 6
    assert set(selected) == set(partner[selected])
    assert set(np.ravel_multi_index(seed.T, grid)) <= set(selected)
    available = np.unique(canonical[canonical != canonical[0]])
    order = sorted(available, key=lambda r: (-weight.flat[r], r))
    assert set(np.unique(canonical[selected])) == {canonical[0], *order[:2]}
    assert np.array_equal(result, _restore_orbit_pool(seed, grid, weight, Rinv, tau, 7))

    full = _restore_orbit_pool(seed, grid, weight, Rinv, tau, 1000)
    assert len(full) == len(all_idx)
    assert np.array_equal(np.sort(np.ravel_multi_index(full.T, grid)), flat)
    assert _restore_orbit_pool(seed, grid, weight, Rinv, tau, 2) is seed


def test_literal_complex_pc_full_rank_and_exhausted_tail():
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.pivoted_cholesky import make_sharded_pivoted_cholesky_select

    mesh = Mesh(np.array(jax.local_devices()[:1]), ("x",))
    active = np.arange(24) < 22
    for feature_rank in (3, 22):
        rng = np.random.default_rng(901+feature_rank)
        features = rng.normal(size=(22, feature_rank))+1j*rng.normal(size=(22, feature_rank))
        G = np.zeros((24, 24), complex)
        G[:22, :22] = features@features.conj().T
        expected_piv, expected_rank, expected_L = _numpy_group_pc(
            G, np.arange(24), active, 20)
        kernel = make_sharded_pivoted_cholesky_select(
            mesh, 24, 20, mesh_axis="x")
        piv, L, rank, d_final, d_taken, trace, info = jax.device_get(kernel(
            jax.device_put(G, NamedSharding(mesh, P("x", None))), None,
            jax.device_put(active, NamedSharding(mesh, P("x")))))
        assert int(rank) == expected_rank
        assert np.array_equal(piv[:rank], expected_piv[:rank])
        assert len(np.unique(piv)) == 20 and np.all((piv >= 0) & (piv < 22))
        np.testing.assert_allclose(L, expected_L, rtol=1e-11, atol=1e-11)
        assert np.max(np.abs(L[:, d_taken == 0]), initial=0) == 0
        remainder = G-L@L.conj().T
        assert np.linalg.eigvalsh(remainder).min() > -1e-10
        assert np.all(np.isfinite(d_final)) and np.all(np.isfinite(trace))
        if feature_rank == 3:
            np.testing.assert_allclose(remainder, 0, atol=1e-11)


def test_legacy_group_pc_dependent_labels_and_independent_factor():
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.pivoted_cholesky import make_sharded_pivoted_cholesky_select

    mesh = Mesh(np.array(jax.local_devices()[:1]), ("x",))
    labels = np.r_[np.repeat(np.arange(6), [3, 2, 3, 4, 5, 5]), -1, -1].astype(np.int32)
    active = np.arange(24) < 22
    for feature_rank in (3, 22):
        rng = np.random.default_rng(147+feature_rank)
        features = rng.normal(size=(22, feature_rank))+1j*rng.normal(size=(22, feature_rank))
        G = np.zeros((24, 24), complex)
        G[:22, :22] = features@features.conj().T
        # Independent NumPy recurrence: select the largest active residual,
        # deflate one column, then remove every row of that label.
        d = np.diag(G).real.copy()
        available = active.copy()
        factor = np.zeros((24, 5), complex)
        floor = np.sqrt(np.finfo(float).eps)*d.max()
        chosen, took = [], []
        for j in range(5):
            candidates = np.flatnonzero(available)
            p = candidates[np.argmax(d[candidates])]
            take = d[p] > floor
            if take:
                col = (G[:, p]-factor[:, :j]@factor[p, :j].conj())/np.sqrt(d[p])
                col[p] = np.sqrt(d[p])
                factor[:, j] = col
                d = np.maximum(d-np.abs(col)**2, 0)
            chosen.append(p)
            took.append(take)
            available[labels == labels[p]] = False
        expected_rank = sum(took)
        kernel = make_sharded_pivoted_cholesky_select(mesh, 24, 5, mesh_axis="x")
        piv, L, rank, _, d_taken, _, _ = jax.device_get(kernel(
            jax.device_put(G, NamedSharding(mesh, P("x", None))),
            jax.device_put(labels, NamedSharding(mesh, P("x"))),
            jax.device_put(active, NamedSharding(mesh, P("x")))))
        assert int(rank) == expected_rank
        assert np.array_equal(piv[:rank], np.asarray(chosen)[:rank])
        assert len(np.unique(labels[piv])) == len(piv) == 5
        assert np.all((piv >= 0) & (piv < 22))
        np.testing.assert_allclose(L, factor, rtol=1e-11, atol=1e-11)
        assert np.max(np.abs(L[:, d_taken == 0]), initial=0) == 0
