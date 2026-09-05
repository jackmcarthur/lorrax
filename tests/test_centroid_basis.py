"""The one in-memory centroid order and its two I/O-seam conversions.

``common.centroid_basis.PackedCentroidBasis`` packs whole symmetry orbits per
shard; files stay canonical.  On a 2x2 emulated CPU mesh: pack/unpack round
trips on an X-sharded, a Y-sharded and a flat ('x','y')-sharded axis, both
operator axes, host tables and permutations; the identity basis is a no-op.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.centroid_basis import PackedCentroidBasis


def _mesh_2x2():
    if len(jax.devices()) < 4:
        pytest.skip("needs four emulated CPU devices")
    return Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))


def _sym_swap_xy():
    identity = np.eye(3, dtype=np.int32)
    swap_xy = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int32)
    return SimpleNamespace(
        sym_matrices=np.stack([identity, swap_xy]),
        translations=np.zeros((2, 3), dtype=np.float64))


def _basis(mesh, n=7):
    # 7 centroids on a 4x4x1 grid: three swap-xy pairs and one fixed point,
    # so the packed order differs from the canonical one and carries pads.
    cents = np.asarray(
        [[1, 0, 0], [0, 1, 0], [2, 0, 0], [0, 2, 0], [3, 1, 0], [1, 3, 0],
         [2, 2, 0]], dtype=np.int32)[:n]
    return PackedCentroidBasis.build(cents, _sym_swap_xy(), (4, 4, 1), mesh)


def test_build_packs_orbits_and_reports_extents():
    mesh = _mesh_2x2()
    basis = _basis(mesh)
    assert not basis.is_identity
    assert basis.n_logical == 7
    assert basis.n_canonical == 8            # round_up(7, 4)
    assert basis.n_packed % 4 == 0 and basis.n_packed >= basis.n_canonical
    ax = basis.layout.axis
    # every orbit sits inside one X shard
    owner = np.arange(basis.n_packed) // ax.shard_size
    for g in range(ax.n_groups):
        rows = np.flatnonzero(ax.packed_group_id == g)
        assert len(set(owner[rows])) == 1
    packed = basis.packed_indices
    assert packed.shape == (basis.n_packed, 3)
    np.testing.assert_array_equal(
        basis.unpack_host(packed, axis=0), basis.canonical_indices)


@pytest.mark.parametrize("spec,axis", [
    (P(None, 'x', 'y'), 1),
    (P(None, 'x', 'y'), 2),
    (P(None, ('x', 'y'), None), 1),
    (P(None, 'x', None, 'y'), 3),
])
def test_pack_unpack_axis_round_trips_without_all_gather(spec, axis):
    mesh = _mesh_2x2()
    basis = _basis(mesh)
    rng = np.random.default_rng(7)
    shape = [3, 8, 8] if len(spec) == 3 else [2, 4, 1, 8]
    shape[axis] = basis.n_canonical
    canonical = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    # canonical pad slots are zero by contract
    pad = [slice(None)] * len(shape)
    pad[axis] = slice(basis.n_logical, None)
    canonical[tuple(pad)] = 0.0
    dev = jax.device_put(jnp.asarray(canonical), NamedSharding(mesh, spec))

    packed = jax.block_until_ready(basis.pack_axis(dev, axis))
    assert packed.shape[axis] == basis.n_packed
    assert packed.sharding.spec == spec
    # packed == host pack of the logical rows; pads exactly zero
    logical = np.take(canonical, np.arange(basis.n_logical), axis=axis)
    np.testing.assert_array_equal(
        np.asarray(packed), basis.pack_host(logical, axis=axis))
    back = jax.block_until_ready(basis.unpack_axis(packed, axis))
    np.testing.assert_array_equal(np.asarray(back), canonical)

    hlo = jax.jit(lambda a: basis.unpack_axis(a, axis, spec=spec)).lower(
        packed).compiler_ir(dialect="hlo").as_hlo_text().lower()
    assert "all-gather(" not in hlo
    assert "all-to-all(" in hlo


def test_operator_round_trip_and_tables():
    mesh = _mesh_2x2()
    basis = _basis(mesh)
    rng = np.random.default_rng(11)
    n = basis.n_canonical
    op = rng.normal(size=(2, n, n)) + 0j
    op[:, basis.n_logical:, :] = 0.0
    op[:, :, basis.n_logical:] = 0.0
    dev = jax.device_put(jnp.asarray(op), NamedSharding(mesh, P(None, 'x', 'y')))
    packed = basis.pack_operator(dev)
    host = basis.pack_host(basis.pack_host(
        op[:, :basis.n_logical, :basis.n_logical], axis=1), axis=2)
    np.testing.assert_array_equal(np.asarray(packed), host)
    np.testing.assert_array_equal(
        np.asarray(basis.unpack_operator(packed)), op)

    # tables: canonical permutations conjugate into the packed order and back
    from symmetry_maps import centroid_source_map_and_wrap
    sym = _sym_swap_xy()
    perm, L = centroid_source_map_and_wrap(
        basis.canonical_indices, sym.sym_matrices, sym.translations,
        np.asarray((4, 4, 1), dtype=np.int32), extend_trs=True)
    perm_p, L_p = basis.pack_tables(perm, L)
    assert perm_p.shape == (perm.shape[0], basis.n_packed)
    perm_c, L_c = basis.unpack_tables(perm_p, L_p)
    np.testing.assert_array_equal(perm_c, perm)
    np.testing.assert_array_equal(L_c, L)


def test_identity_basis_is_a_no_op():
    mesh = _mesh_2x2()
    cents = np.asarray([[1, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=np.int32)
    basis = PackedCentroidBasis.build(
        cents, _sym_swap_xy(), (4, 4, 1), mesh, identity=True)
    assert basis.is_identity
    assert basis.n_packed == basis.n_canonical == 4
    np.testing.assert_array_equal(basis.packed_indices[:3], cents)
    dev = jax.device_put(jnp.ones((2, 4, 4)), NamedSharding(mesh, P(None, 'x', 'y')))
    assert basis.pack_operator(dev) is dev
    assert basis.unpack_operator(dev) is dev
    # a set that is not orbit-closed also falls back to the identity layout
    open_set = PackedCentroidBasis.build(
        cents[:1], _sym_swap_xy(), (4, 4, 1), mesh)
    assert open_set.is_identity


@pytest.mark.parametrize("spec,axis", [
    (P(None, 'x', 'y'), 1),
    (P(None, 'x', 'y'), 2),
    (P(None, ('x', 'y'), None), 1),
])
def test_extent_change_round_trips_when_packed_exceeds_canonical(spec, axis):
    """Orbits [3, 3, 2] on a 2x2 mesh: canonical 8, packed 12 (one shard
    holds 5 rows padded to 6).  The rank-local prefix pad/crop at the seam
    is live here, unlike the equal-extent case."""
    from common.grouped_layout import build_square_grouped_shard_layout
    mesh = _mesh_2x2()
    layout = build_square_grouped_shard_layout(
        np.asarray([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int32), (2, 2))
    assert layout.axis.n_padded == 12
    cents = np.stack([np.arange(8), np.zeros(8), np.zeros(8)], axis=1).astype(np.int32)
    basis = PackedCentroidBasis(
        mesh_xy=mesh, layout=layout, canonical_indices=cents, n_canonical=8)
    assert basis.n_packed == 12 and basis.n_canonical == 8 and not basis.is_identity
    rng = np.random.default_rng(3)
    shape = [3, 8, 8]
    shape[axis] = 8
    canonical = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    dev = jax.device_put(jnp.asarray(canonical), NamedSharding(mesh, spec))
    packed = jax.block_until_ready(basis.pack_axis(dev, axis))
    assert packed.shape[axis] == 12
    np.testing.assert_array_equal(
        np.asarray(packed), basis.pack_host(canonical, axis=axis))
    back = jax.block_until_ready(basis.unpack_axis(packed, axis))
    np.testing.assert_array_equal(np.asarray(back), canonical)

@pytest.mark.parametrize("extra", [0, 4, 8])
def test_file_carrier_can_be_smaller_equal_or_larger_than_runtime(monkeypatch, extra):
    """Physical operators and Dyson solves survive either I/O extent change."""
    from gw.w_isdf import solve_w

    mesh = _mesh_2x2()
    monkeypatch.setenv("LORRAX_EXTRA_MU_PAD", str(extra))
    cycle = np.asarray([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int32)
    sym = SimpleNamespace(
        sym_matrices=np.stack([np.eye(3, dtype=np.int32), cycle, cycle @ cycle]),
        translations=np.zeros((3, 3)))
    cents = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]])
    basis = PackedCentroidBasis.build(cents, sym, (4, 4, 4), mesh)
    assert basis.n_packed == 8
    assert basis.n_canonical == 4 + extra
    assert basis.solve_axis.logical == basis.solve_axis.carrier == 8
    # The object owns its table even if the caller reuses the input buffer.
    cents[:] = 0
    assert not basis.canonical_indices.flags.writeable
    assert np.any(basis.canonical_indices)

    canonical = np.zeros((4, basis.n_canonical, basis.n_canonical), complex)
    canonical[:, np.arange(4), np.arange(4)] = 0.1
    sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    packed = basis.pack_operator(jax.device_put(canonical, sharding))
    np.testing.assert_array_equal(np.asarray(basis.unpack_operator(packed)), canonical)
    assert np.count_nonzero(np.asarray(packed)[:, ~basis.active_mask, :]) == 0
    assert np.count_nonzero(np.asarray(packed)[:, :, ~basis.active_mask]) == 0
    meta = SimpleNamespace(nk_tot=4, nspin=1, nspinor_wfnfile=1,
                           n_rmu=4, mu_basis=basis,
                           mu_solve_extent=basis.solve_axis.logical)
    W = solve_w(packed, 2 * packed, meta, mesh, dyson_solver="local")
    # W = V/(1 - pref V chi), pref = 2/sqrt(nq) = 1 for four q rows.
    expected = canonical / (1 - 0.1 * 0.2)
    np.testing.assert_allclose(np.asarray(basis.unpack_operator(W)), expected,
                               rtol=1e-13, atol=1e-15)
    hlo = jax.jit(lambda x: basis.pack_operator(x, spec=sharding.spec)).lower(
        jax.device_put(canonical, sharding)).compile().as_text().lower()
    assert 'all-gather(' not in hlo

    flat = jax.device_put(canonical, NamedSharding(mesh, P(None, ('x', 'y'), None)))
    np.testing.assert_array_equal(np.asarray(basis.unpack_axis(
        basis.pack_axis(flat, 1), 1)), canonical)


def test_single_device_mesh_converts_without_collectives():
    # P = 1: every axis is replicated (spec P()), the permutation is local.
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))
    # canonical order interleaves the orbits, so even one shard permutes
    cents = np.asarray([[1, 0, 0], [2, 0, 0], [3, 1, 0], [2, 2, 0],
                        [0, 1, 0], [0, 2, 0], [1, 3, 0]], dtype=np.int32)
    basis = PackedCentroidBasis.build(cents, _sym_swap_xy(), (4, 4, 1), mesh)
    assert not basis.is_identity
    canonical = np.zeros((3, basis.n_canonical), dtype=np.complex128)
    canonical[:, :basis.n_logical] = np.arange(3 * basis.n_logical).reshape(
        3, -1) + 1.0
    dev = jax.device_put(jnp.asarray(canonical), NamedSharding(mesh, P()))
    packed = jax.block_until_ready(basis.pack_axis(dev, 1))
    np.testing.assert_array_equal(
        np.asarray(packed), basis.pack_host(canonical[:, :basis.n_logical], axis=1))
    back = basis.unpack_axis(packed, 1)
    np.testing.assert_array_equal(np.asarray(back), canonical)
    op = jnp.einsum('im,in->mn', dev, dev.conj())
    np.testing.assert_array_equal(
        np.asarray(basis.unpack_operator(basis.pack_operator(op))), np.asarray(op))
