"""The sketched centroid select: exact construction, mesh independence, budget.

(a) With Omega the identity on the pair index (the exhaustive sketch), Y^H Y
    equals the q=0 pair Gram assembled in numpy from the same WFN faces: the
    charge Gram with nested and separate left windows, two spinor components
    and unequal k weights, and the stacked-current Gram sum_i Z_i^H Z_i of
    Z_i = <psi^L| alpha^i |psi^B> on four-component faces.
(b) A 2x2 emulated CPU mesh and a 1x1 mesh build the same Gaussian sketch and
    select identical pivots.
(c) The delivered set is a union of whole orbits within the point budget,
    no remaining orbit fits the leftover budget, and with one orbit per block
    and gamma = 1 the pivot order is the greedy group-block pivoted Cholesky
    order on Y^H Y.
(d) The reported residuals carry the shrink correction: after t directions
    the Gaussian sketch keeps (r'-t)/r' of a column's true residual, and the
    select reads d r'/(r'-t).  Over the unpicked columns the corrected median
    is within the selection bias of the exact residual on the true Gram (the
    columns left behind are those the sketch happened to underestimate:
    about -17% at r'-t = 64 here, against -49% uncorrected).

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


def _cand_face(mesh, psi):
    """The loader's ``faces='cand'`` layout of host ``psi (nk, nb, ns, M)``."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from centroid.sketch_select import CAND
    from runtime.padding import padded_mu_extent
    nk, nb, ns, M = psi.shape
    y = np.zeros((nk, nb, ns, padded_mu_extent(M, mesh)), np.complex128)
    y[..., :M] = psi
    return jax.device_put(y, NamedSharding(mesh, P(None, None, None, CAND)))


def _random_psi(rng, nk, nb, ns, M):
    return rng.standard_normal((nk, nb, ns, M)) + 1j * rng.standard_normal((nk, nb, ns, M))


def _sketch(mesh, psi_r, psi_l, *, left, weights, n_rows, factors, seed=7,
            gamma_mode="charge"):
    import jax
    from centroid.sketch_select import _zero_kernel, sketch_from_faces, sketch_row_carrier
    from runtime.padding import padded_mu_extent
    nk = psi_r.shape[0]
    Y = _zero_kernel(mesh, sketch_row_carrier(n_rows, mesh),
                     padded_mu_extent(psi_r.shape[3], mesh))()
    Y = sketch_from_faces(
        Y, _cand_face(mesh, psi_r), None if psi_l is None else _cand_face(mesh, psi_l),
        mesh, n_rows=n_rows, sqrt_w=np.sqrt(weights), k_ids=np.arange(nk) + 3,
        k_pos=np.arange(nk), key_data=np.asarray(jax.random.key_data(
            jax.random.key(seed))), left_slice=left, factors=factors,
        gamma_mode=gamma_mode)
    Y = np.asarray(Y)
    assert not Y[0, n_rows:].any()
    return Y[0, :n_rows, :psi_r.shape[3]]


def _numpy_gram(psi_l, psi_r, weights, vertices):
    """``sum_v Z_v^H Z_v``, ``Z_v(a; k,l,b) = sqrt(w_k) sum_ab conj(psi^L_la) V_ab psi^B_bb``."""
    G = 0
    for V in vertices:
        Z = np.einsum("k,klsa,st,kbta->klba", np.sqrt(weights), psi_l.conj(), V, psi_r)
        Z = Z.reshape(-1, psi_r.shape[3])
        G = G + Z.conj().T @ Z
    return G


def test_exhaustive_sketch_reproduces_the_pair_grams():
    import jax
    jax.config.update("jax_enable_x64", True)
    from common.gamma_matrices import gammas
    mesh = _mesh((2, 2))
    rng = np.random.default_rng(3)
    nk, nb_r, M = 2, 6, 10
    weights = np.asarray([0.3, 0.7])
    cases = [("charge", 2, (1, 4), None), ("charge", 2, None, 3),
             ("transverse", 4, (1, 4), None), ("transverse", 4, None, 3)]
    for mode, ns, left, nb_sep in cases:
        psi_r = _random_psi(rng, nk, nb_r, ns, M)
        psi_l = None if nb_sep is None else _random_psi(rng, nk, nb_sep, ns, M)
        legs = psi_r[:, left[0]:left[1]] if psi_l is None else psi_l
        vertices = ([np.eye(ns)] if mode == "charge"
                    else [np.asarray(gammas[i]) for i in (1, 2, 3)])
        G = _numpy_gram(legs, psi_r, weights, vertices)
        n_v = len(vertices)
        Y = _sketch(mesh, psi_r, psi_l, left=left, weights=weights,
                    n_rows=n_v * nk * legs.shape[1] * nb_r, factors="exhaustive",
                    gamma_mode=mode)
        np.testing.assert_allclose(Y.conj().T @ Y, G, rtol=0,
                                   atol=1e-12 * np.abs(G).max(), err_msg=mode)
        # The Gaussian sketch is unbiased for the same Gram: a loose check
        # that the conjugation convention is the Gram's, not its transpose.
        Yg = _sketch(mesh, psi_r, psi_l, left=left, weights=weights,
                     n_rows=4000, factors="gaussian", gamma_mode=mode)
        rel = np.abs(Yg.conj().T @ Yg - G).max() / np.abs(G).max()
        assert rel < 0.15, (mode, rel)


def _orbits(rng, M):
    sizes, total = [], 0
    while total < M:
        s = int(min(rng.integers(1, 5), M - total))
        sizes.append(s)
        total += s
    gid = np.repeat(np.arange(len(sizes)), sizes)
    return gid[rng.permutation(M)].astype(np.int32), len(sizes)


def _numpy_group_order(G, gid, budget):
    """Greedy group-block pivoted Cholesky: best mean residual among groups that
    fit, every member pivoted largest residual first (the reference order)."""
    floor = np.sqrt(np.finfo(float).eps) * np.real(np.diag(G)).max()
    d = np.real(np.diag(G)).copy()
    L = np.zeros((len(G), 0), complex)
    order, used, left = [], np.zeros(len(G), bool), int(budget)
    sizes = np.bincount(gid)
    while True:
        fits = [g for g in np.unique(gid[~used]) if sizes[g] <= left]
        if not fits:
            return np.asarray(order)
        score = [np.where(d[gid == g] > floor, d[gid == g], 0).sum() / sizes[g] for g in fits]
        g = fits[int(np.argmax(score))]
        members = list(np.flatnonzero(gid == g))
        while members:
            p = max(members, key=lambda m: d[m])
            members.remove(p)
            if d[p] > floor:
                col = (G[:, p] - L @ L[p].conj()) / np.sqrt(d[p])
                col[used] = 0
                col[p] = np.sqrt(d[p])
                L = np.concatenate([L, col[:, None]], axis=1)
                d = d - np.abs(col) ** 2
            d[p] = 0
            used[p] = True
            order.append(p)
        left -= sizes[g]


def _select(mesh, Y_host, gid, n_groups, budget, n_slots, gamma):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from centroid.sketch_select import CAND, sketch_block_select, sketch_row_carrier
    from runtime.padding import padded_mu_extent
    r, M = Y_host.shape
    y = np.zeros((1, sketch_row_carrier(r, mesh), padded_mu_extent(M, mesh)), np.complex128)
    y[0, :r, :M] = Y_host
    Y = jax.device_put(y, NamedSharding(mesh, P(None, None, CAND)))
    return sketch_block_select(Y, mesh, group_id=gid, n_groups=n_groups,
                               point_budget=budget, n_slots=n_slots, gamma=gamma,
                               n_rows=r)


def test_sketch_select_is_mesh_independent_orbit_closed_and_greedy():
    import jax
    jax.config.update("jax_enable_x64", True)
    from centroid.sketch_select import prune_candidates_by_sketch, sketch_rows_default
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

    # One orbit per block and no cut: the greedy group-block order on Y^H Y.
    greedy = _select(mesh4, Y4, gid, n_groups, budget, 1, 1.0)[0]
    np.testing.assert_array_equal(greedy, _numpy_group_order(Y4.conj().T @ Y4, gid, budget))

    with pytest.raises(ValueError, match="four-component bispinor"):
        prune_candidates_by_sketch(
            None, None, np.zeros((4, 3), np.int64), 2, mesh4,
            band_range_left=(0, 1), band_range_right=(0, 1),
            gamma_mode="transverse")


def test_sketch_residuals_are_unbiased():
    import jax
    jax.config.update("jax_enable_x64", True)
    from centroid.sketch_select import sketch_rows_default
    mesh = _mesh((2, 2))
    rng = np.random.default_rng(5)
    nk, ns, nb_r, M, budget = 2, 2, 8, 96, 40
    weights = np.asarray([0.5, 0.5])
    psi_r = _random_psi(rng, nk, nb_r, ns, M)
    left = (0, 4)
    G = _numpy_gram(psi_r[:, left[0]:left[1]], psi_r, weights, [np.eye(ns)])
    n_rows = sketch_rows_default(budget)
    Y = _sketch(mesh, psi_r, None, left=left, weights=weights,
                n_rows=n_rows, factors="gaussian", seed=19)
    piv, rank, d_final, *_ = _select(mesh, Y, np.arange(M, dtype=np.int32), M,
                                     budget, 4, 0.7)
    rest = np.setdiff1d(np.arange(M), piv)
    S = np.asarray(piv)
    true = (np.real(np.diag(G))[rest] - np.real(np.einsum(
        "as,sa->a", G[np.ix_(rest, S)],
        np.linalg.solve(G[np.ix_(S, S)], G[np.ix_(S, rest)]))))
    ratio = np.median(d_final[rest] / true)
    shrink = (n_rows - rank) / n_rows
    assert rank == budget
    assert 0.75 < ratio < 1.18, (ratio, shrink)
    assert abs(np.log(ratio)) < 0.5 * abs(np.log(shrink)), (ratio, shrink)
