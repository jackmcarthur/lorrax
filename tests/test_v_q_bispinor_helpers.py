"""Transverse Coulomb projector in the live per-q V-builder.

``gw.v_q_bispinor._make_per_q_v_builder_for_tile`` applies the rank-2 Cartesian
projector to the bare ``v(q+G)``:
    CC (μ_L=ν_L=0):        bare v(q+G)          (real, ≥0)
    TT diagonal (i=j):     v · (1 − K̂_i²)
    TT off-diagonal (i≠j): v · (−K̂_i K̂_j)
Checked via the TT/CC ratio, so the test is independent of
``compute_v_q_per_G``'s absolute normalisation.

(Replaces the old tests of the deleted r-space ``_make_v_per_G_for_tile`` /
``_make_K_cart_on_sphere_fn`` twins, removed with the r-space tile subsystem
on 2026-07-02.)
"""
import numpy as np
import pytest


def _case():
    # bvec = I so K_cart = q + G directly (einsum('ba,qbg->qag', I, ·) = ·).
    bvec = np.eye(3)
    q = np.array([[0.1, 0.2, 0.0]])                # (n_q=1, 3); avoid K=0
    # gvec_components: (n_q, xyz, ngk) — 3 G-vectors (1,0,0),(0,1,0),(2,1,0).
    G = np.array([[[1.0, 0.0, 2.0],
                   [0.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0]]])
    return bvec, q, G


def _build(mu_L, nu_L):
    from src.gw.v_q_bispinor import _make_per_q_v_builder_for_tile
    bvec, q, G = _case()
    b = _make_per_q_v_builder_for_tile(
        mu_L=mu_L, nu_L=nu_L, bvec=bvec, cell_volume=1.0,
        sys_dim=3, vcoul_cutoff_ry=None)
    return np.asarray(b(q, G))


def _K_and_K2():
    bvec, q, G = _case()
    K = q[:, :, None] + G                          # (1, xyz, ngk); bvec=I
    K2 = np.sum(K * K, axis=1)                      # (1, ngk)
    return K, K2


def test_cc_tile_is_bare_v_real():
    v = _build(0, 0)
    assert np.allclose(v.imag, 0.0)
    assert np.all(np.abs(v) > 0)                   # nonzero at K≠0


def test_tt_diagonal_applies_1_minus_khat2():
    v_cc = _build(0, 0)
    v_11 = _build(1, 1)                            # i=j=0
    K, K2 = _K_and_K2()
    khat_x2 = K[:, 0] ** 2 / K2
    np.testing.assert_allclose(v_11, v_cc * (1.0 - khat_x2), rtol=1e-10, atol=1e-12)


def test_tt_offdiagonal_applies_minus_khat_ij():
    v_cc = _build(0, 0)
    v_12 = _build(1, 2)                            # i=0, j=1
    K, K2 = _K_and_K2()
    khat_xy = K[:, 0] * K[:, 1] / K2
    np.testing.assert_allclose(v_12, v_cc * (-khat_xy), rtol=1e-10, atol=1e-12)


def test_invalid_tt_indices_raise():
    from src.gw.v_q_bispinor import _make_per_q_v_builder_for_tile
    with pytest.raises(ValueError):
        _make_per_q_v_builder_for_tile(
            mu_L=4, nu_L=1, bvec=np.eye(3), cell_volume=1.0,
            sys_dim=3, vcoul_cutoff_ry=None)
