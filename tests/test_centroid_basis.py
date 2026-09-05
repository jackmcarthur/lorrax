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
