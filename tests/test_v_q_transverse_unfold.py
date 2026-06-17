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
from common.symmetry_maps import unfold_v_q_bispinor_lorentz


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


@pytest.mark.skipif(len(jax.devices()) < 4,
                    reason="needs a ≥4-device 2×2 mesh to make P() vs "
                           "P(None,'x','y') a real (crash-reproducing) distinction")
def test_lorentz_unfold_accepts_replicated_screened_tiles():
    """``unfold_v_q_bispinor_lorentz`` must accept BOTH input shardings.

    Bare G-flat TT tiles arrive ``P(None,'x','y')``; screened supermatrix-W
    tiles arrive fully replicated ``P()``.  The inner Lorentz-mix jit has
    explicit ``in_shardings=P(None,None,None,'x','y')``, so the replicated
    screened case crashed (ValueError, symmetry_maps.py:562) until the input
    was constrained.  Verify both shardings unfold to the same hand-computed
    ``Σ_ab R[a,i]R[b,j] V^{ab}`` reference.
    """
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2),
                axis_names=('x', 'y'))
    n_q, n_rmu, n_sym = 6, 8, 3
    rng = np.random.default_rng(0xB12)
    tiles = {(i, j): jnp.asarray(
                (rng.standard_normal((n_q, n_rmu, n_rmu))
                 + 1j * rng.standard_normal((n_q, n_rmu, n_rmu))).astype(np.complex128))
             for i in (1, 2, 3) for j in (1, 2, 3)}

    def Rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    # (2·n_sym, 3, 3): spatial + TRS halves carry the same spatial R.
    R_half = np.stack([np.eye(3), Rz(2*np.pi/3), Rz(4*np.pi/3)], axis=0)
    R_tab = np.concatenate([R_half, R_half], axis=0)
    sym_idx = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)

    # Reference: V'^{ij}[q] = Σ_ab R[s(q),a,i] R[s(q),b,j] V^{ab}[q].
    ref = {}
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            acc = np.zeros((n_q, n_rmu, n_rmu), complex)
            for a in (1, 2, 3):
                for b in (1, 2, 3):
                    for q in range(n_q):
                        R = R_tab[sym_idx[q]]
                        acc[q] += R[a-1, i-1] * R[b-1, j-1] * np.asarray(tiles[(a, b)][q])
            ref[(i, j)] = acc

    def run(spec):
        sh = NamedSharding(mesh, spec)
        put = {k: jax.device_put(v, sh) for k, v in tiles.items()}
        return unfold_v_q_bispinor_lorentz(
            put, sym_idx=sym_idx, R_proper_table=R_tab, mesh_xy=mesh)

    out_rep = run(P())                       # screened: replicated (used to crash)
    out_shd = run(P(None, 'x', 'y'))         # bare: x,y-sharded
    for k in ref:
        np.testing.assert_allclose(np.asarray(out_rep[k]), ref[k], atol=1e-11)
        np.testing.assert_allclose(np.asarray(out_shd[k]), ref[k], atol=1e-11)


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
