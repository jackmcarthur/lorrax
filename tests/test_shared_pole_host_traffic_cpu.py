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


def _zero_tailed_round(rng, counts, widths, rows=12):
    """Panels whose Q columns are nonzero up to each slot's count and exact zeros past it."""
    slots = counts.shape[0]
    fields = []
    for k in range(3):
        panels = []
        for a, w in enumerate(widths):
            p = rng.normal(size=(slots, rows, w)) + 1j * rng.normal(size=(slots, rows, w))
            p = np.where(np.arange(w)[None, None, :] >= counts[:, a][:, None, None], 0, p)
            panels.append(p)
        fields.append(panels)
    return fields


def test_pack_round_equals_host_tables_and_pack():
    """The device-built tables and packed columns equal round_tables + round_operands + pack_panels."""
    from gw.shared_pole_local import BATCH, pack_panels, pack_round, round_operands, round_tables
    mesh = _mesh()
    rng = np.random.default_rng(5)
    ladder = lambda w: 0 if int(w) == 0 else -(-int(w) // 2) * 2
    widths = [4, 8, 4, 8]
    counts = rng.integers(0, 5, size=(4, 4))
    counts[3] = 0                                            # a synthetic slot (real = 3) selects nothing
    nodes = [1 + 2j, 1 - 2j, -1 - 2j, -1 + 2j]
    fields = _zero_tailed_round(rng, counts, widths)
    qi = rng.normal(size=(4, 12, 6)) + 0j
    infinity_counts = np.array([3, 6, 1, 0])
    qi = np.where(np.arange(6)[None, None, :] >= infinity_counts[:, None, None], 0, qi)
    tables = round_tables(counts, widths, nodes, infinity_counts, 6, column_extent=ladder,
                          ordered=True, odd_moments=True)
    put = lambda a: jax.device_put(a, NamedSharding(mesh, P(BATCH)))
    dev = tuple([put(a) for a in field] for field in fields)
    live, points, active, packed = pack_round(dev, widths, nodes, (put(qi),), 6,
                                              extent=tables["extent"], halves=2, blocks=tables["blocks"],
                                              real=3, mesh_xy=mesh, column_extent=ladder)
    want = round_operands(mesh, tables, 3)
    for got, ref in zip((live, points, active), want[:3]):
        assert got.sharding.spec == P(BATCH)
        assert np.asarray(got).dtype == np.asarray(ref).dtype
        assert np.array_equal(np.asarray(got), np.asarray(ref))
    ref_pack = pack_panels(dev, tables["order"], widths, mesh_xy=mesh, layout="batch")
    for g, r in zip(packed, ref_pack):
        assert np.asarray(g).tobytes() == np.asarray(r).tobytes()


def test_device_constants_and_node_scales():
    """device_index / sample_points come from compiled constants; the in-kernel scales equal the
    host's np.complex128 products bit for bit."""
    from gw.shared_pole_directions import _node_scale, device_index, sample_points
    mesh = _mesh()
    assert int(device_index(mesh, 0)) == 0 and int(device_index(mesh, 70)) == 70
    assert device_index(mesh, 5) is device_index(mesh, 5)
    recipe = dict(distinct_id=[0, 1, 1, 2], z_ry=[0.3 + 1.7j, -0.2 + 0.9j, -0.2 + 0.9j, 0 + 2.5j])
    table = sample_points(mesh, recipe)
    assert np.asarray(table).tolist() == [0.3 + 1.7j, -0.2 + 0.9j, 2.5j]
    for sid, z in enumerate(np.asarray(table)):
        for scale, want in (((2, False), np.complex128(2 * z)), ((2, True), np.complex128(2 * np.conj(z))),
                            ((-2, False), np.complex128(-2 * z)), ((-2, True), np.complex128(-2 * np.conj(z))),
                            (None, np.complex128(1.0))):
            got = jax.jit(lambda t, j: _node_scale(t, j, scale))(table, device_index(mesh, sid))
            assert np.asarray(got).tobytes() == want.tobytes(), (sid, scale)


def test_store_pole_rows_equals_host_join():
    from gw.shared_pole_local import store_pole_rows
    mesh = _mesh()
    rng = np.random.default_rng(9)
    rep = NamedSharding(mesh, P())
    blocks = [rng.normal(size=(4, 7)), rng.normal(size=(4, 7))]
    shapes = ((4, 5), (2, 3))
    order = (5, 0, 2, 4, 1, 3)
    host = np.concatenate([np.pad(b[:r, :k], ((0, 0), (0, 6 - k)), constant_values=1.0)
                           for b, (r, k) in zip(blocks, shapes)])[list(order)]
    got = store_pole_rows(mesh, shapes, 6, order)(*(jax.device_put(b, rep) for b in blocks))
    assert np.asarray(got).tobytes() == host.tobytes()
