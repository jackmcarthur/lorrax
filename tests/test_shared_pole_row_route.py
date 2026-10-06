"""The row route (parents over x, matrices over y) against the local round (CPU 2x2 mesh).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
A synthetic ordered pencil: four parents, each a full-rank positive real-pole model
W(z) = C diag(1/(z mu - 1)) C^H (k = n poles, so the half-pencil Gram is nonsingular)
with its exact moments M_k = sum_j c_j c_j^H mu_j^-(k+1), sampled at two imaginary
supports and their mirrors. The local round program
(one whole parent per rank, the numerical control of every route) and the row
program (``row_reduce_round``: XLA-partitioned products, rank-local eighs
through one all_to_all over y) must agree on the retained poles and factors to
round-off, and the row eigh must agree with the local solver.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _model(rng, n, k):
    mu = np.sort(rng.uniform(0.5, 4.0, k))                       # inverse poles, Ry^-2
    c = rng.normal(size=(n, k)) + 1j * rng.normal(size=(n, k))
    return mu, c / np.sqrt(n)


def _w(mu, c, z):
    return (c * (1 / (z * mu - 1))[None, :]) @ np.conj(c.T)


def _dw(mu, c, z):
    return (c * (-mu / (z * mu - 1) ** 2)[None, :]) @ np.conj(c.T)


def _moment(mu, c, k):
    return (c * (mu ** -(k + 1))[None, :]) @ np.conj(c.T)


def _orthonormal(rng, n, w):
    q, _ = np.linalg.qr(rng.normal(size=(n, w)) + 1j * rng.normal(size=(n, w)))
    return q


def _pencil(mesh, seed=0, n=24, k=24, width=4, iw=2, q=4):
    """States, infinity panels and tables of ``q`` parents in the ordered paired layout."""
    from gw.shared_pole_directions import port_extent
    from gw.shared_pole_local import round_tables
    rng = np.random.default_rng(seed)
    extent = port_extent(mesh)
    nodes = [0.7j, 1.9j]
    models = [_model(rng, n, k) for _ in range(q)]
    directions = [[_orthonormal(rng, n, width) for _ in nodes] for _ in range(q)]
    states = []
    for sign in (1, -1):
        for a, z in enumerate(nodes):
            zz = sign * z
            qs = np.stack([directions[p][a] for p in range(q)])
            os_ = np.stack([_w(*models[p], zz) @ directions[p][a] for p in range(q)])
            ds = np.stack([_dw(*models[p], zz) @ directions[p][a] for p in range(q)])
            states.append((np.full(q, zz, np.complex128), qs, os_, ds))
    moments = [np.stack([_moment(*models[p], kk) for p in range(q)]) for kk in range(4)]
    qi = np.stack([np.linalg.eigh((moments[1][p] + np.conj(moments[1][p].T)) / 2)[1][:, -iw:] for p in range(q)])
    infinity = (qi, *(np.einsum("pij,pjk->pik", m, qi) for m in moments))
    counts = np.full((q, len(states)), width)
    widths = [extent(width)] * len(states)
    tables = round_tables(counts, widths, [s[0][0] for s in states], [iw] * q, extent(iw),
                          column_extent=extent, ordered=True, odd_moments=True)
    return states, infinity, tables, models


def _pad(a, w):
    return np.pad(a, ((0, 0), (0, 0), (0, w - a.shape[-1])))


def test_row_route_matches_local_round():
    mesh = _mesh()
    from gw.shared_pole_directions import port_extent
    from gw.shared_pole_execution import face_ritz_carrier, row_reduce_round, row_sharding
    from gw.shared_pole_local import BATCH, reduce_round
    from gw.shared_pole_capacity import _local_eigenplan
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1
    extent = port_extent(mesh)
    states, infinity, tables, models = _pencil(mesh)
    q = len(models)
    width, iw = extent(4), extent(2)
    gram_keep = shared_real_pole_gates_ordered_v1["normalized_gram_keep"]["sector_threshold"]
    budget = 8
    batch = NamedSharding(mesh, P(BATCH))
    face = NamedSharding(mesh, P(None, "x", "y"))

    def place(sharding):
        put = lambda a: jax.device_put(np.asarray(a), sharding)
        return ([(s[0], put(_pad(s[1], width)), put(_pad(s[2], width)), put(_pad(s[3], width))) for s in states],
                tuple(put(_pad(a, iw)) for a in infinity))
    side = int(tables["active"].shape[-1])
    local_states, local_infinity = place(batch)
    local = reduce_round(local_states, local_infinity, tables, real=q, mesh_xy=mesh,
                         native_eigh=_local_eigenplan(mesh, side).native_fn, ordered=True,
                         odd_moments=True, keep_budget=budget, retain_span=True, gram_keep=gram_keep)
    row_states, row_infinity = place(face)
    row = row_reduce_round(row_states, row_infinity, tables, mesh=mesh, ordered=True, odd_moments=True,
                           keep_budget=budget, retain_span=True, gram_keep=gram_keep,
                           carrier=face_ritz_carrier(mesh, budget))
    # The synthetic model is not a physical ordered-pencil source (its paired Schur
    # complement is indefinite), so the gates refuse it on both routes alike; the
    # test is that the two routes compute the same reduction.
    ld, rd = (jax.tree.map(np.asarray, r[3][0]) for r in (local, row))
    for key in ("gram_spectrum_relative", "gram_min_relative", "paired_min_relative",
                "retained_metric_relative", "infinite_weight_fraction"):
        assert np.allclose(ld[key], rd[key], rtol=1e-9, atol=1e-12), key
    for key in ("retained_rank", "paired_rank", "positive_count", "negative_count",
                "gram_valid", "retained_metric_positive", "orientation_paired"):
        assert np.array_equal(ld[key], rd[key]), key
    assert np.all(ld["gram_diagonal_positive"]) and np.all(ld["retained_metric_positive"])
    lb, lp, la = (np.asarray(a) for a in local[0])
    rb, rp, ra = (np.asarray(a) for a in row[0])
    assert row[0][0].sharding.spec == row_sharding(mesh).spec
    assert np.array_equal(la, ra) and np.all(la.sum(axis=-1) > 0)
    assert np.allclose(np.where(la, lp, 0), np.where(ra, rp, 0), rtol=1e-8, atol=1e-10)
    # Factors agree up to a phase per column: compare the model's b b^H.
    for p in range(q):
        lw = lb[p] @ np.conj(lb[p].T)
        rw = rb[p] @ np.conj(rb[p].T)
        assert np.allclose(lw, rw, rtol=1e-7, atol=1e-9 * max(1.0, np.abs(lw).max()))
    # The signed model and the coefficient map agree the same way.
    lc, lm, lr = (np.asarray(a) for a in local[1])
    rc, rm, rr = (np.asarray(a) for a in row[1])
    assert np.array_equal(lr, rr) and np.allclose(np.where(lr, lm, 0), np.where(rr, rm, 0), rtol=1e-8, atol=1e-10)
    ly, ry = np.asarray(local[4]), np.asarray(row[4])
    assert ly.shape == ry.shape
    for p in range(q):
        assert np.allclose(ly[p] @ np.conj(ly[p].T), ry[p] @ np.conj(ry[p].T), rtol=1e-7, atol=1e-9)


def test_row_eigh_matches_local():
    mesh = _mesh()
    from gw.shared_pole_execution import row_eigh, row_sharding
    rng = np.random.default_rng(1)
    for q in (2, 4):
        a = rng.normal(size=(q, 12, 12)) + 1j * rng.normal(size=(q, 12, 12))
        a = (a + np.conj(np.swapaxes(a, -1, -2))) / 2
        a[:, :3, :] = 0
        a[:, :, :3] = 0                                              # exact zero rows: deflated
        w, v = jax.jit(row_eigh(mesh))(jax.device_put(a, row_sharding(mesh)))
        w, v = np.asarray(w), np.asarray(v)
        assert w.shape == (q, 12) and v.shape == (q, 12, 12)
        # The deflated solver reports the zero rows as exact zero eigenvalues inside the
        # ascending spectrum of the zero-padded matrix.
        live = np.linalg.eigvalsh(a[:, 3:, 3:])
        padded = np.sort(np.concatenate((live, np.zeros((q, 3))), axis=-1), axis=-1)
        assert np.allclose(np.sort(w, axis=-1), padded, atol=1e-12)
        assert np.allclose(np.einsum("qij,qjk->qik", a, v), v * w[:, None, :], atol=1e-12)
