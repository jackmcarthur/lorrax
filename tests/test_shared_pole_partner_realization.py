"""The constructor's partner realization on packed face tiles equals the canonical unfold.

``gw.qgrid_symmetry.shared_pole_partner_realizer`` runs the Sigma unfold body on a stack of packed
face tiles [S, mu_X, nu_Y] on a 2x2 mesh. The reference is ``symmetry_maps.unfold_operator_local``
on the canonical (unpacked) tables on one device; packing and the face split must commute with the
realization exactly. Geometry: the store's triangular C3 x inversion fixture (orbit-packed basis,
wraps that differ across centroids). The physics of the unfold convention at complex frequency is
``tests/test_shared_pole_symmetric_partner.py``. RED TWIN: a different unitary row misses.
"""
import numpy as np
import pytest

from test_shared_pole_store import _sigma_fixture, _test_mesh


def _header(meta, tables, recipe, identity):
    from file_io import shared_pole_store as store
    return store._metadata(meta, tables, recipe, identity)


def test_packed_partner_realization_equals_the_canonical_unfold():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.shard_map import shard_map
    from gw.qgrid_symmetry import shared_pole_partner_realizer
    from symmetry_maps import unfold_operator_local

    mesh = _test_mesh()
    meta, tables, recipe, identity = _sigma_fixture(mesh)
    header = _header(meta, tables, recipe, identity)
    basis, qt = meta.mu_basis, header["qirr"]
    n = basis.n_logical
    rng = np.random.default_rng(917)
    stack = rng.normal(size=(2, n, n)) + 1j * rng.normal(size=(2, n, n))
    perm, wraps = np.asarray(qt["sym_perm"], np.int32), np.asarray(qt["L_table"], np.float64)
    one = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))

    def canonical(row, parent):
        body = shard_map(lambda a: unfold_operator_local(
            a, irr_idx=np.arange(2), sym_idx=np.full(2, row), q_irr_frac=np.repeat(np.asarray(qt["q_irr_frac"])[parent][None], 2, 0),
            left_local_perm=perm, left_L_table=wraps, right_local_perm=perm, right_L_table=wraps,
            n_sym_spatial=int(qt["n_sym_spatial"])), mesh=one, in_specs=P(None, "x", "y"), out_specs=P(None, "x", "y"),
            check_vma=False)
        return np.asarray(jax.jit(body)(jnp.asarray(stack)))

    packed = basis.pack_host(basis.pack_host(stack, axis=1), axis=2)
    face = jax.make_array_from_callback(packed.shape, NamedSharding(mesh, P(None, "x", "y")), lambda i: packed[i])
    realize = shared_pole_partner_realizer(meta, header, mesh_xy=mesh)
    nsp = int(qt["n_sym_spatial"])
    worst, twin = 0.0, np.inf
    for parent in (1, 2):
        for row in (3, 4):                               # inversion x C3^k: unitary, permutes and wraps
            got = basis.unpack_host(basis.unpack_host(np.asarray(realize(face, row, parent)), axis=1), axis=2)
            want = canonical(row, parent)
            assert np.linalg.norm(want - stack) > 0.1 * np.linalg.norm(stack)
            worst = max(worst, float(np.max(np.abs(got - want))))
            twin = min(twin, float(np.max(np.abs(got - canonical(5, parent)))))
    assert worst < 1e-14, worst
    assert twin > 1e-2, twin
    with pytest.raises(ValueError, match="GATE minus_q_partner.*antiunitary"):
        realize(face, nsp + 1, 1)
