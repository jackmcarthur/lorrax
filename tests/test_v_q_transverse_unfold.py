"""Tests for ``_unfold_v_q_ij_ibz_to_full`` (Phase D).

The transverse Coulomb kernel is a rank-2 Cartesian tensor; under any
orthogonal sym ``R``::

    v_ij(R K) = R_ia R_jb v_ab(K)

So the bilinear ``V_q^{ij}(μ, ν)`` unfolds with TWO contributions:
* centroid double-permute on (μ, ν), same as the scalar case;
* polarization mixing ``R·V·Rᵀ`` on the (i, j) indices.

These tests cover:

1. Identity sym: output equals input.
2. ``π/2`` rotation about z: (V_xx, V_yy, V_xy, V_yx) get mixed
   exactly as a tensor would.
3. Inversion (-I): V_ij unchanged on Cartesian (rotation matrix
   squares to identity in the bilinear).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import pytest

jax.config.update("jax_enable_x64", True)

from gw.v_q_tile import _unfold_v_q_ij_ibz_to_full


@pytest.fixture
def single_device_mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))


def _make_v_q_ij(n_q_ibz=1, n_rmu=4, seed=0xABCD):
    rng = np.random.default_rng(seed)
    arr = (rng.standard_normal((n_q_ibz, 3, 3, n_rmu, n_rmu))
           + 1j * rng.standard_normal((n_q_ibz, 3, 3, n_rmu, n_rmu)))
    return jnp.asarray(arr.astype(np.complex128))


# ---------------------------------------------------------------------------

def test_identity_sym_passthrough(single_device_mesh):
    """ntran=1: identity rotation, identity centroid perm; output ≡ input."""
    n_rmu = 4
    V_in = _make_v_q_ij(n_q_ibz=1, n_rmu=n_rmu)
    full_to_irr_idx = np.array([0], dtype=np.int32)
    full_to_irr_sym = np.array([0], dtype=np.int32)
    sym_perm = np.arange(n_rmu, dtype=np.int32)[None, :]   # (1, n_rmu) identity
    R_cart = np.eye(3, dtype=np.float64)[None, :, :]       # (1, 3, 3) identity

    V_out = _unfold_v_q_ij_ibz_to_full(
        V_in,
        full_to_irr_idx=full_to_irr_idx,
        full_to_irr_sym=full_to_irr_sym,
        sym_perm=sym_perm,
        R_cart=R_cart,
        mesh_xy=single_device_mesh,
    )
    np.testing.assert_allclose(
        np.asarray(V_out), np.asarray(V_in),
        atol=1e-12, rtol=1e-12,
    )


def test_inversion_is_polarization_identity(single_device_mesh):
    """Under R = −I, R·V·Rᵀ = V (the two −1 factors cancel in the bilinear)."""
    n_rmu = 4
    V_in = _make_v_q_ij(n_q_ibz=1, n_rmu=n_rmu)
    # 2 syms: identity + inversion.  Inversion maps the single IBZ q to
    # itself (q = 0 is its own image under -I), so we pick a "full" BZ
    # of size 2 where q[0] = IBZ q (identity), q[1] = IBZ q (inversion).
    full_to_irr_idx = np.array([0, 0], dtype=np.int32)
    full_to_irr_sym = np.array([0, 1], dtype=np.int32)
    sym_perm = np.tile(np.arange(n_rmu, dtype=np.int32)[None, :], (2, 1))
    R_cart = np.stack([np.eye(3), -np.eye(3)], axis=0).astype(np.float64)

    V_out = _unfold_v_q_ij_ibz_to_full(
        V_in,
        full_to_irr_idx=full_to_irr_idx,
        full_to_irr_sym=full_to_irr_sym,
        sym_perm=sym_perm,
        R_cart=R_cart,
        mesh_xy=single_device_mesh,
    )
    # Both q's should equal the input row (identity AND inversion give same V).
    np.testing.assert_allclose(
        np.asarray(V_out[0]), np.asarray(V_in[0]),
        atol=1e-12, rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(V_out[1]), np.asarray(V_in[0]),
        atol=1e-12, rtol=1e-12,
        err_msg="V_q under R=-I should match V_q under identity"
    )


def test_pi_over_two_z_rotation_mixes_xy_components(single_device_mesh):
    """90° rotation about z: V_xx ↔ V_yy and V_xy ↔ −V_yx etc."""
    n_rmu = 3
    V_in = _make_v_q_ij(n_q_ibz=1, n_rmu=n_rmu)

    # R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]] (Rₓ̂ → ŷ, ŷ → -x̂, ẑ → ẑ).
    R = np.array([[0.0, -1.0, 0.0],
                  [1.0,  0.0, 0.0],
                  [0.0,  0.0, 1.0]], dtype=np.float64)
    R_cart = np.stack([np.eye(3), R], axis=0)
    sym_perm = np.tile(np.arange(n_rmu, dtype=np.int32)[None, :], (2, 1))
    full_to_irr_idx = np.array([0, 0], dtype=np.int32)
    full_to_irr_sym = np.array([0, 1], dtype=np.int32)

    V_out = _unfold_v_q_ij_ibz_to_full(
        V_in,
        full_to_irr_idx=full_to_irr_idx,
        full_to_irr_sym=full_to_irr_sym,
        sym_perm=sym_perm,
        R_cart=R_cart,
        mesh_xy=single_device_mesh,
    )

    # Compute reference by hand: V' = R V Rᵀ on the (i,j) axes.
    V_ref = np.einsum(
        'ia,jb,abmn->ijmn',
        R, R, np.asarray(V_in[0]),
        optimize=True,
    )
    np.testing.assert_allclose(
        np.asarray(V_out[1]), V_ref,
        atol=1e-12, rtol=1e-12,
    )


def test_padded_mu_pad_rows_are_passthrough(single_device_mesh):
    """μ pad rows (logical < padded) are identity under inv_perm padding."""
    n_rmu_logical = 4
    n_rmu_padded = 8
    rng = np.random.default_rng(0xFADE)
    V_in_full = (rng.standard_normal((1, 3, 3, n_rmu_padded, n_rmu_padded))
                  + 1j * rng.standard_normal((1, 3, 3, n_rmu_padded, n_rmu_padded))
                ).astype(np.complex128)
    # Zero out pad rows in input (the writer's contract).
    V_in_full[:, :, :, n_rmu_logical:, :] = 0
    V_in_full[:, :, :, :, n_rmu_logical:] = 0
    V_in = jnp.asarray(V_in_full)

    # Logical-sized sym_perm; helper pads to n_rmu_padded with identity.
    sym_perm = np.arange(n_rmu_logical, dtype=np.int32)[None, :]
    R_cart = np.eye(3, dtype=np.float64)[None, :, :]
    full_to_irr_idx = np.array([0], dtype=np.int32)
    full_to_irr_sym = np.array([0], dtype=np.int32)

    V_out = _unfold_v_q_ij_ibz_to_full(
        V_in,
        full_to_irr_idx=full_to_irr_idx,
        full_to_irr_sym=full_to_irr_sym,
        sym_perm=sym_perm,
        R_cart=R_cart,
        mesh_xy=single_device_mesh,
    )
    np.testing.assert_allclose(
        np.asarray(V_out), np.asarray(V_in),
        atol=1e-12, rtol=1e-12,
    )
