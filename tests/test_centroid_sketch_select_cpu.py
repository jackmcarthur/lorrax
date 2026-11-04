"""The sketched centroid select: exact construction, mesh independence, budget.

(a) With Omega the identity on the pair index (the exhaustive sketch), Y^H Y
    equals the q=0 candidate Gram that ``isdf.gram_q0_tiled_from_psi_sm``
    assembles from the same WFN faces, nested and separate left windows, two
    spinor components and unequal k weights.
(b) A 2x2 emulated CPU mesh and a 1x1 mesh build the same Gaussian sketch and
    select identical pivots.
(c) The delivered set is a union of whole orbits within the point budget,
    no remaining orbit fits the leftover budget, and with one orbit per block
    and gamma = 1 the pivot order is the group-block pivoted Cholesky order
    on Y^H Y.  Transverse mode refuses.

CPU only; needs four virtual devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4``).
"""
import numpy as np
import pytest


def _mesh(shape):
    import jax
    from jax.sharding import Mesh
    world = int(np.prod(shape))
    if len(jax.devices()) < world:
        pytest.skip(f"needs {world} virtual CPU devices")
    return Mesh(np.asarray(jax.devices()[:world]).reshape(shape), ("x", "y"))


def _faces(mesh, psi):
    """Loader-layout faces of host ``psi (nk, nb, ns, M)``: Y and conjugate X."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from runtime.padding import padded_mu_extent
    nk, nb, ns, M = psi.shape
    m_pad = padded_mu_extent(M, mesh)
    y = np.zeros((nk, nb, ns, m_pad), np.complex128)
    y[..., :M] = psi
    x = y.conj().transpose(0, 3, 1, 2)
    return (jax.device_put(y, NamedSharding(mesh, P(None, None, None, "y"))),
            jax.device_put(x, NamedSharding(mesh, P(None, "x", None, None))))


def _random_psi(rng, nk, nb, ns, M):
    return rng.standard_normal((nk, nb, ns, M)) + 1j * rng.standard_normal((nk, nb, ns, M))


def _sketch(mesh, psi_r, psi_l, *, left, weights, n_rows, factors, seed=7):
    import jax
    from centroid.sketch_select import _zero_kernel, sketch_from_faces
    from runtime.padding import padded_mu_extent
    r_y, _ = _faces(mesh, psi_r)
    l_y = None if psi_l is None else _faces(mesh, psi_l)[0]
    nk = psi_r.shape[0]
    Y = _zero_kernel(mesh, n_rows, padded_mu_extent(psi_r.shape[3], mesh))()
    Y = sketch_from_faces(
        Y, r_y, l_y, mesh, sqrt_w=np.sqrt(weights), k_ids=np.arange(nk) + 3,
        k_pos=np.arange(nk), key_data=np.asarray(jax.random.key_data(
            jax.random.key(seed))), left_slice=left, factors=factors)
    return np.asarray(Y)[0, :, :psi_r.shape[3]]


def test_exhaustive_sketch_reproduces_the_candidate_gram():
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    jax.config.update("jax_enable_x64", True)
    from centroid.pivoted_cholesky import gram_tile_width
    from isdf import gram_q0_tiled_from_psi_sm
    mesh = _mesh((2, 2))
    rng = np.random.default_rng(3)
    nk, ns, nb_r, M = 2, 2, 6, 10
    weights = np.asarray([0.3, 0.7])
    psi_r = _random_psi(rng, nk, nb_r, ns, M)
    for left, psi_l in (((1, 4), None), (None, _random_psi(rng, nk, 3, ns, M))):
        nb_l = (left[1] - left[0]) if left is not None else psi_l.shape[1]
        Y = _sketch(mesh, psi_r, psi_l, left=left, weights=weights,
                    n_rows=nk * nb_l * nb_r, factors="exhaustive")
        r_y, r_x = _faces(mesh, psi_r)
        l_y, l_x = (None, None) if psi_l is None else _faces(mesh, psi_l)
        m_pad = int(r_y.shape[3])
        G0 = jax.device_put(np.zeros((m_pad, m_pad), np.complex128),
                            NamedSharding(mesh, P("x", "y")))
        G = gram_q0_tiled_from_psi_sm(
            G0, l_x, l_y, r_x, r_y, jnp.asarray(weights), mesh_xy=mesh,
            tile_width=gram_tile_width(nk, ns, m_pad, divisor=2, x_shards=2,
                                       y_shards=2),
            gamma_mode="charge", left_bands=left)
        G = np.asarray(G)[:M, :M]
        np.testing.assert_allclose(Y.conj().T @ Y, G, rtol=0,
                                   atol=1e-12 * np.abs(G).max())
        # The Gaussian sketch is unbiased for the same Gram: a loose check
        # that the conjugation convention is the Gram's, not its transpose.
        Yg = _sketch(mesh, psi_r, psi_l, left=left, weights=weights,
                     n_rows=4000, factors="gaussian")
        rel = np.abs(Yg.conj().T @ Yg - G).max() / np.abs(G).max()
        assert rel < 0.15, rel


def _orbits(rng, M):
    sizes, total = [], 0
    while total < M:
        s = int(min(rng.integers(1, 5), M - total))
        sizes.append(s)
        total += s
    gid = np.repeat(np.arange(len(sizes)), sizes)
    return gid[rng.permutation(M)].astype(np.int32), len(sizes)


def _select(mesh, Y_host, gid, n_groups, budget, n_slots, gamma):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from centroid.sketch_select import CAND, sketch_block_select
    from runtime.padding import padded_mu_extent
    r, M = Y_host.shape
    y = np.zeros((1, r, padded_mu_extent(M, mesh)), np.complex128)
    y[0, :, :M] = Y_host
    Y = jax.device_put(y, NamedSharding(mesh, P(None, None, CAND)))
    return sketch_block_select(Y, mesh, group_id=gid, n_groups=n_groups,
                               point_budget=budget, n_slots=n_slots, gamma=gamma)


def test_sketch_select_is_mesh_independent_orbit_closed_and_greedy():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from centroid.sketch_select import prune_candidates_by_sketch, sketch_rows_default
    from common.pivoted_cholesky import group_block_pivoted_cholesky_select
    mesh4, mesh1 = _mesh((2, 2)), _mesh((1, 1))
    rng = np.random.default_rng(11)
    nk, ns, nb_r, M, budget = 2, 2, 8, 37, 20
    weights = np.asarray([0.5, 0.5])
    psi_r = _random_psi(rng, nk, nb_r, ns, M)
    n_rows = sketch_rows_default(budget)
    Y4 = _sketch(mesh4, psi_r, None, left=(0, 4), weights=weights,
                 n_rows=n_rows, factors="gaussian")
    Y1 = _sketch(mesh1, psi_r, None, left=(0, 4), weights=weights,
                 n_rows=n_rows, factors="gaussian")
    np.testing.assert_allclose(Y4, Y1, rtol=0, atol=1e-12 * np.abs(Y1).max())

    gid, n_groups = _orbits(rng, M)
    sizes = np.bincount(gid, minlength=n_groups)
    out4 = _select(mesh4, Y4, gid, n_groups, budget, 3, 0.7)
    out1 = _select(mesh1, Y4, gid, n_groups, budget, 3, 0.7)
    np.testing.assert_array_equal(out4[0], out1[0])
    assert out4[1] == out1[1]
    np.testing.assert_allclose(out4[2], out1[2], rtol=0, atol=1e-10 * out1[7])

    piv = out4[0]
    picked = np.unique(gid[piv])
    assert piv.size == np.unique(piv).size == int(sizes[picked].sum()) <= budget
    np.testing.assert_array_equal(np.sort(piv), np.flatnonzero(np.isin(gid, picked)))
    leftover = budget - piv.size
    unpicked = np.setdiff1d(np.arange(n_groups), picked)
    assert unpicked.size == 0 or sizes[unpicked].min() > leftover
    assert np.all(out4[2][piv] == 0.0)

    # One orbit per block and no cut: the group-block Cholesky order on Y^H Y.
    greedy = _select(mesh4, Y4, gid, n_groups, budget, 1, 1.0)[0]
    G = jnp.asarray(Y4.conj().T @ Y4)
    ref = np.asarray(group_block_pivoted_cholesky_select(
        G, budget, jnp.asarray(gid), n_groups=n_groups)[0])
    np.testing.assert_array_equal(greedy, ref[ref >= 0])

    with pytest.raises(ValueError, match="charge Gram only"):
        prune_candidates_by_sketch(
            None, None, np.zeros((4, 3), np.int64), 2, mesh4,
            band_range_left=(0, 1), band_range_right=(0, 1),
            gamma_mode="transverse")
