"""The constructor's host traffic paths on the CPU 2x2 mesh (seconds).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
``host_tree`` reads a mixed pytree back bit for bit in one transfer; a round's
tables reach the devices in one placement and split exactly; the stacked
partner eigensystems (``_round_partner_directions_many``) give each pair the
arrays and widths of its own call; a Coulomb span is one program for every
start row; a replicated constant is placed once.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def test_host_tree_is_bit_exact():
    from gw.shared_pole_local import host_tree
    mesh = _mesh()
    rep = NamedSharding(mesh, P())
    rng = np.random.default_rng(0)
    tree = {"b": np.array([True, False, True]), "i": np.arange(5, dtype=np.int32),
            "l": np.array([2 ** 40 + 3, -7], np.int64), "f": rng.normal(size=(2, 3)),
            "c": rng.normal(size=(4,)) + 1j * rng.normal(size=(4,)), "s": np.float64(np.pi),
            "e": np.zeros((0, 3))}
    placed = jax.tree.map(lambda a: jax.device_put(a, rep), tree)
    got = host_tree((placed, "host-leaf"))
    assert got[1] == "host-leaf"
    for k, v in tree.items():
        g = got[0][k]
        assert g.dtype == np.asarray(v).dtype and g.shape == np.shape(v), k
        assert np.asarray(v).tobytes() == g.tobytes(), k


def test_round_operands_split_exactly():
    from gw.shared_pole_local import BATCH, round_operands
    mesh = _mesh()
    rng = np.random.default_rng(1)
    tables = dict(own=np.zeros(4), points=rng.normal(size=(4, 6)) + 1j * rng.normal(size=(4, 6)),
                  active=rng.random((4, 9)) > 0.5, order=rng.integers(0, 40, size=(4, 6)).astype(np.int32))
    live, points, active, order = round_operands(mesh, tables, 3)
    assert all(a.sharding.spec == P(BATCH) for a in (live, points, active, order))
    assert np.array_equal(np.asarray(live), np.arange(4) < 3) and np.asarray(live).dtype == bool
    assert np.asarray(points).tobytes() == tables["points"].tobytes()
    assert np.array_equal(np.asarray(active), tables["active"]) and np.asarray(active).dtype == bool
    assert np.array_equal(np.asarray(order), tables["order"]) and np.asarray(order).dtype == np.int32


def _partner_case(rng, slots=4, n=24, r=8):
    """(Q, O) pairs of a round: orthonormal Q, O = W Q with an odd component outside span(Q)."""
    pairs = []
    for e in range(3):
        qs, os_ = [], []
        for s in range(slots):
            a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            q, _ = np.linalg.qr(a[:, :r])
            w = a @ a.conj().T / n
            scale = 0.0 if (e == 1 and s == 2) else 1.0        # one slot of one pair: O in span(Q)
            o = q @ (q.conj().T @ (w @ q)) + scale * 1e-2 * (w @ q - q @ (q.conj().T @ (w @ q)))
            qs.append(q); os_.append(o)
        pairs.append((np.stack(qs), np.stack(os_)))
    return pairs


def test_stacked_partner_directions_match_each_pair():
    import distrib_la
    from gw.shared_pole_directions import (_round_kernels, _round_partner_directions,
                                           _round_partner_directions_many, port_extent)
    from gw.shared_pole_local import BATCH
    mesh = _mesh()
    rng = np.random.default_rng(2)
    put = lambda a: jax.device_put(a, NamedSharding(mesh, P(BATCH)))
    pairs = [(put(q), put(o)) for q, o in _partner_case(rng)]
    k = _round_kernels(mesh, "batch")
    plan = distrib_la.plan("eigh", mesh, n=int(pairs[0][0].shape[1]), backend="off",
                           batched_route="batch_reshard")
    kw = dict(real=3, kernels=k, eigh_plan=plan, column_extent=port_extent(mesh), tol=1e-6)
    many = _round_partner_directions_many(pairs, 1e-3, **kw)
    for (q, o), (d, widths) in zip(pairs, many):
        d0, widths0 = _round_partner_directions(q, o, 1e-3, **kw)
        assert widths == widths0
        assert d.shape == d0.shape and d.sharding.spec == d0.sharding.spec
        # The same eigenvectors up to the eigensolver's rounding (CPU LAPACK, stacked or not);
        # a column may differ by a phase, so compare the projectors of the retained columns.
        a, b = np.asarray(d), np.asarray(d0)
        np.testing.assert_allclose(a @ np.conj(np.swapaxes(a, -1, -2)),
                                   b @ np.conj(np.swapaxes(b, -1, -2)), atol=1e-10)


def test_span_rows_one_program_for_every_start():
    from gw.response_bank import _span_rows, _span_start
    mesh = _mesh()
    x = jax.device_put(np.arange(10 * 4 * 4, dtype=np.complex128).reshape(10, 4, 4),
                       NamedSharding(mesh, P(None, "x", "y")))
    before = _span_rows.cache_info().currsize
    for a in (0, 3, 6):
        got = _span_rows(mesh, 4)(x, _span_start(mesh, a))
        assert np.array_equal(np.asarray(got), np.asarray(x)[a:a + 4])
        assert got.sharding.spec == P(None, "x", "y")
    assert _span_rows.cache_info().currsize == before + 1


def test_replicated_places_each_value_once():
    from gw.shared_pole_directions import replicated
    mesh = _mesh()
    put = replicated(mesh)
    a, b = put(np.int32(7)), replicated(mesh)(np.int32(7))
    assert a is b and int(a) == 7
    assert put(np.int32(8)) is not a
    big = np.zeros(1024, np.complex128)                       # larger than the reuse bound
    assert put(big) is not put(big)
