"""GW adapter contracts for raw-parent centroid operators."""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from gw.centroid_k_unfold import build_centroid_k_unfold_plan


def _mesh_2x2():
    if len(jax.devices()) < 4:
        pytest.skip("needs four emulated CPU devices")
    return Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))


def _symmetry_fixture():
    identity = np.eye(3, dtype=np.int32)
    swap_xy = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                         dtype=np.int32)

    def spinor_action(rows, *, nspinor):
        rows = np.asarray(rows)
        return np.broadcast_to(
            np.eye(nspinor, dtype=np.complex128),
            rows.shape + (nspinor, nspinor)).copy()

    return SimpleNamespace(
        sym_matrices=np.stack([identity, swap_xy]),
        translations=np.zeros((2, 3), dtype=np.float64),
        irr_idx_k=np.asarray([0, 0, 1], dtype=np.int32),
        sym_idx_k=np.asarray([0, 1, 0], dtype=np.int32),
        unfolded_kpts=np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        kirr_fullids=np.asarray([0, 2], dtype=np.int32),
        spinor_action=spinor_action,
    )


def test_plan_packs_raw_parent_faces_and_unfolds_their_operator():
    mesh = _mesh_2x2()
    centroids = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        dtype=np.int32)
    plan = build_centroid_k_unfold_plan(
        _symmetry_fixture(), centroids, (2, 2, 1), mesh,
        nspinor=1,
        parent_k_frac=np.asarray([[0.0, 0.0, 0.0],
                                  [0.5, 0.0, 0.0]]),
    )
    assert plan.n_parent == 2
    assert plan.n_full == 3
    assert plan.n_centroid_packed % 4 == 0

    rng = np.random.default_rng(22)
    psi_nmu_np = (
        rng.normal(size=(2, 4, 1, 4))
        + 1j * rng.normal(size=(2, 4, 1, 4)))
    psi_mun_np = psi_nmu_np.transpose(0, 2, 3, 1)
    with mesh:
        psi_nmu, psi_mun = plan.pack_face_pair(
            jnp.asarray(psi_nmu_np), jnp.asarray(psi_mun_np))
        parent_op = jnp.einsum(
            'ksmn,kntv->ksmtv', psi_mun, jnp.conj(psi_nmu),
            optimize=True)
        full_op = plan.unfold_operator(parent_op)

    packed_nmu = np.asarray(psi_nmu)
    packed_mun = np.asarray(psi_mun)
    np.testing.assert_allclose(
        plan.layout.axis.unpack_host(packed_nmu, axis=3), psi_nmu_np)
    np.testing.assert_allclose(
        plan.layout.axis.unpack_host(packed_mun, axis=2), psi_mun_np)

    parent = np.asarray(parent_op)
    expected = np.empty_like(np.asarray(full_op))
    for child, (parent_row, sym_row) in enumerate(
            zip(plan.irr_idx, plan.sym_idx)):
        perm = plan.sym_perm[int(sym_row)]
        transported = np.take(parent[int(parent_row)], perm, axis=1)
        expected[child] = np.take(transported, perm, axis=3)
    np.testing.assert_allclose(np.asarray(full_op), expected, rtol=2e-13,
                               atol=2e-13)


def test_parent_scalar_rows_do_not_masquerade_as_parent_wavefunctions():
    mesh = _mesh_2x2()
    plan = build_centroid_k_unfold_plan(
        _symmetry_fixture(),
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        (2, 2, 1), mesh, nspinor=1)
    full_energy = jnp.asarray([[2.0], [2.0], [7.0]])
    np.testing.assert_array_equal(
        np.asarray(plan.parent_rows(full_energy)), [[2.0], [7.0]])
