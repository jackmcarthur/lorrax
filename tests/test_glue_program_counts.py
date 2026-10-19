"""Glue that builds no program per call (CPU 2x2 mesh).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
Each pattern is counted by JAX's own lowering event: a sharded zero array
lowers once per shard shape, the centroid-conversion price compiles nothing,
the face masks and member-row gather are one cached program each, the cylinder
tables and small host gathers touch no device, and every value is the one the
replaced code produced.
"""
import numpy as np
import pytest
import jax
import jax.monitoring
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

_EVENTS = []
jax.monitoring.register_event_duration_secs_listener(lambda event, *_, **__: _EVENTS.append(event))


def _lowerings():
    return sum(e == "/jax/core/compile/jaxpr_to_mlir_module_duration" for e in _EVENTS)


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _basis(mesh, groups):
    from common.centroid_basis import PackedCentroidBasis
    from common.grouped_layout import build_square_grouped_shard_layout
    from runtime.padding import padded_mu_extent
    groups = np.asarray(groups)
    return PackedCentroidBasis(
        mesh_xy=mesh, layout=build_square_grouped_shard_layout(groups, (2, 2)),
        canonical_indices=np.arange(groups.size, dtype=np.int32),
        n_canonical=int(padded_mu_extent(groups.size, mesh)))


def test_sharded_zero_sites_lower_once_per_shape():
    from isdf.zeta_mubatch import _zero_accumulators
    from gw.response_bank import _group_zeros
    from file_io.shared_pole_store import _zeros
    mesh = _mesh()
    calls = (lambda: _zero_accumulators(mesh, 4, 8, 6, debug_m=True),
             lambda: (_group_zeros(mesh, (3, 2, 8, 8)),),
             lambda: (_zeros(mesh, P(None, "x", None, "y"), (2, 8, 1, 4)),))
    for call in calls:
        first = call()
        before = _lowerings()
        for _ in range(4):
            again = call()
        assert _lowerings() == before
        for a, b in zip(first, again):
            assert a.sharding == b.sharding and a.dtype == jnp.complex128
            assert not np.any(np.asarray(b))
            assert all(s.data.shape == a.sharding.shard_shape(a.shape) for s in b.addressable_shards)


def test_conversion_price_compiles_nothing_and_bounds_the_compiled_kernel():
    from file_io.shared_pole_store import _conversion_bytes
    mesh = _mesh()
    basis = _basis(mesh, [0] * 9 + [1] * 4)           # packed 20 > canonical 16
    assert basis.n_packed != basis.n_canonical
    cases = [((5, 16, 3, 12), P(None, "x", None, "y"), False, False, 1),
             ((5, 20, 3, 12), P(None, "x", None, "y"), True, False, 1),
             ((2, 20, 2, 8), P(None, "y", None, "x"), True, False, 1),
             ((4, 4, 20), P(None, "x", "y"), True, False, 2),
             ((6, 20, 20), P(None, "x", "y"), True, True, 1),
             ((6, 16, 16), P(None, "x", "y"), False, True, 1)]
    before = _lowerings()
    prices = [_conversion_bytes(basis, s, spec, unpack=u, operator=o, axis=a)
              for s, spec, u, o, a in cases]
    assert _lowerings() == before
    for (shape, spec, unpack, operator, axis), price in zip(cases, prices):
        kernel = (basis._operator_kernel(spec, unpack) if operator
                  else basis._axis_kernel(axis, spec, unpack))
        stats = kernel.lower(jax.ShapeDtypeStruct(
            shape, jnp.complex128, sharding=NamedSharding(mesh, spec))).compile().memory_analysis()
        compiled = (stats.argument_size_in_bytes, stats.output_size_in_bytes,
                    stats.temp_size_in_bytes - stats.alias_size_in_bytes)
        assert price[:2] == compiled[:2], (shape, price, compiled)
        assert compiled[2] <= price[2] <= 1.6 * compiled[2] + 8192, (shape, price, compiled)


def test_face_masks_are_one_program_and_match_the_eager_masks():
    from file_io.shared_pole_store import _live_face, _live_poles
    mesh = _mesh()
    rng = np.random.default_rng(3)
    counts = np.array([3, 0, 4], np.int64)
    mask = rng.random(8) > 0.3
    for spec in (P(None, "x", None, "y"), P(None, "y", None, "x")):
        b = jax.device_put(rng.normal(size=(3, 8, 2, 4)) + 1j * rng.normal(size=(3, 8, 2, 4)),
                           NamedSharding(mesh, spec))
        active = jnp.arange(4)[None, :] < jnp.asarray(counts)[:, None]
        want = jnp.where(active[:, None, None, :] & jnp.asarray(mask)[None, :, None, None], b, 0.0)
        got = _live_face(b, counts, mask)
        assert np.array_equal(np.asarray(got), np.asarray(want))
        assert got.sharding.is_equivalent_to(b.sharding, 4)
        before = _lowerings()
        _live_face(b, counts[::-1].copy(), ~mask)
        assert _lowerings() == before
    poles = jnp.asarray(rng.random((3, 4)))
    assert np.array_equal(np.asarray(_live_poles(poles, counts)),
                          np.where(np.arange(4)[None, :] < counts[:, None], np.asarray(poles), 1.0))


def test_member_rows_is_one_program_per_shape():
    from gw.response_bank import _MemberRows
    mesh = _mesh()
    rng = np.random.default_rng(4)
    host = rng.normal(size=(6, 5, 4, 4)) + 1j * rng.normal(size=(6, 5, 4, 4))
    carry = jax.device_put(host, NamedSharding(mesh, P(None, None, "x", "y")))
    view = _MemberRows(carry, 2)
    rows = np.array([4, 0, 2])
    assert np.array_equal(np.asarray(view[1, rows]), host[3, rows])
    before = _lowerings()
    for i in range(4):
        assert np.array_equal(np.asarray(view[i, rows[::-1]]), host[2 + i, rows[::-1]])
    assert _lowerings() == before


def test_cylinder_tables_are_host_and_place_every_sphere_slot():
    from common.wfn_transforms import psi_cylinder_tables
    rng = np.random.default_rng(5)
    grid, axis, ngk = (6, 5, 8), 2, 40
    n_rtot = int(np.prod(grid))
    nk = 3
    g_index = np.empty((nk, ngk), np.int32)
    for k in range(nk):
        cells = rng.choice(n_rtot, size=ngk - 5 - k, replace=False)
        g_index[k] = np.concatenate([cells, n_rtot + np.arange(ngk - cells.size)])
    before = len(_EVENTS)
    ci, cax, pfc = psi_cylinder_tables(g_index, grid, axis, ngkmax=ngk)
    assert len(_EVENTS) == before and all(isinstance(t, np.ndarray) for t in (ci, cax, pfc))
    valid = g_index < n_rtot
    x, y, z = np.unravel_index(np.where(valid, g_index, 0), grid)
    plane = x * grid[1] + y
    assert np.array_equal(cax, np.unique(z[valid]))
    cols = np.unique(plane[valid])
    assert np.array_equal(np.flatnonzero(pfc < cols.size), cols)
    want = np.full((nk, cols.size, cax.size), ngk, np.int32)
    for k, g in zip(*np.nonzero(valid)):
        want[k, np.searchsorted(cols, plane[k, g]), np.searchsorted(cax, z[k, g])] = g
    assert np.array_equal(ci, want)


def test_local_potential_ifft_is_one_program_and_bitwise():
    from psp.radial.build_projectors_qe import _real_ifftn
    rng = np.random.default_rng(6)
    v = rng.normal(size=(6, 5, 4)) + 1j * rng.normal(size=(6, 5, 4))
    want = jnp.real(jnp.fft.ifftn(jnp.asarray(v), norm="ortho"))
    got = _real_ifftn(v)
    assert got.dtype == jnp.float64 and np.array_equal(np.asarray(got), np.asarray(want))
    before = _lowerings()
    _real_ifftn(2 * v)
    assert _lowerings() == before


def test_small_host_gathers_take_the_host_control_store(monkeypatch):
    import ffi.common.broadcast as broadcast
    from common import collectives
    sent = []
    monkeypatch.setattr(collectives, "process_count", lambda: 3)
    monkeypatch.setattr(broadcast, "reduce_bytes_to_all", lambda buf, *, key, reduce, max_bytes: (
        sent.append((key, buf.nbytes)) or reduce([buf, buf, buf])))
    before = len(_EVENTS)
    budgets = collectives.all_gather_processes(np.asarray(41.5, np.float64))
    assert budgets.dtype == np.float64 and np.array_equal(budgets, [41.5] * 3)
    rows = collectives.all_gather_processes(np.arange(4, dtype=np.int64).reshape(2, 2), tiled=True)
    assert rows.shape == (6, 2) and np.array_equal(rows[2:4], [[0, 1], [2, 3]])
    receipt = collectives.all_gather_processes(np.frombuffer(b"ok", np.uint8))
    assert receipt.dtype == np.uint8 and receipt.shape == (3, 2)
    assert len(_EVENTS) == before and len({key for key, _ in sent}) == 3
    gathered = []
    from jax.experimental import multihost_utils
    monkeypatch.setattr(multihost_utils, "process_allgather",
                        lambda x, tiled=False: gathered.append(x.shape) or np.stack([np.asarray(x)] * 3))
    big = np.zeros(collectives._HOST_GATHER_BYTES // 8, np.float64)
    assert collectives.all_gather_processes(big).shape == (3,) + big.shape
    assert gathered == [big.shape] and len(sent) == 3
