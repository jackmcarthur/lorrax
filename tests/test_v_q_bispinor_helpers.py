"""Unit tests for the bookkeeping in :mod:`gw.v_q_bispinor`.

CPU-only.  These exercise the static metadata (which tiles are unique,
which are zero-by-gauge, which are Hermitian-redundant) and the small
helpers that don't need a GPU or a real V_q kernel run.

The end-to-end orchestrator + reader tests live in
``tests/test_v_q_bispinor_orchestrator.py`` (GPU-required).
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

import jax
import jax.numpy as jnp


def test_unique_tiles_count():
    from src.gw.v_q_bispinor import UNIQUE_TILES
    # 1 CC + 3 TT diagonal + 3 TT off-diagonal upper = 7
    assert len(UNIQUE_TILES) == 7
    assert (0, 0) in UNIQUE_TILES
    for i in (1, 2, 3):
        assert (i, i) in UNIQUE_TILES
    for (i, j) in [(1, 2), (1, 3), (2, 3)]:
        assert (i, j) in UNIQUE_TILES


def test_zero_tiles_count():
    from src.gw.v_q_bispinor import ZERO_TILES
    # 6 = 3 + 3, the (0,i) and (i,0) cross terms killed by Coulomb gauge.
    assert len(ZERO_TILES) == 6
    for i in (1, 2, 3):
        assert (0, i) in ZERO_TILES
        assert (i, 0) in ZERO_TILES


def test_hermitian_pairs_count_and_orientation():
    from src.gw.v_q_bispinor import HERMITIAN_PAIRS, UNIQUE_TILES
    # Exactly 3 redundant tiles paired with (1,2), (1,3), (2,3).
    assert set(HERMITIAN_PAIRS.values()) == {(1, 2), (1, 3), (2, 3)}
    assert set(HERMITIAN_PAIRS.keys()) == {(2, 1), (3, 1), (3, 2)}
    # Companions must be in UNIQUE_TILES (otherwise reader would fail).
    for redundant, companion in HERMITIAN_PAIRS.items():
        assert companion in UNIQUE_TILES
        assert redundant not in UNIQUE_TILES


def test_no_overlap_between_categories():
    from src.gw.v_q_bispinor import UNIQUE_TILES, ZERO_TILES, HERMITIAN_PAIRS
    unique = set(UNIQUE_TILES)
    redundant = set(HERMITIAN_PAIRS.keys())
    zero = set(ZERO_TILES)
    # All 16 (μ_L, ν_L) blocks accounted for, exactly once.
    all_blocks = unique | redundant | zero
    assert len(all_blocks) == 16
    assert len(unique) + len(redundant) + len(zero) == 16
    # No overlap.
    assert unique.isdisjoint(redundant)
    assert unique.isdisjoint(zero)
    assert redundant.isdisjoint(zero)


def test_tile_dataset_name_roundtrip():
    from src.gw.v_q_bispinor import tile_dataset_name
    assert tile_dataset_name(0, 0) == "V_qmunu_CC"
    assert tile_dataset_name(1, 1) == "V_qmunu_TT_11"
    assert tile_dataset_name(1, 2) == "V_qmunu_TT_12"
    assert tile_dataset_name(2, 3) == "V_qmunu_TT_23"
    assert tile_dataset_name(3, 3) == "V_qmunu_TT_33"


def test_make_v_per_G_for_tile_cc_is_identity():
    """For (0, 0) the projector is unity; v_per_G_fn must equal the
    base callable (no projector multiplication)."""
    from src.gw.v_q_bispinor import _make_v_per_G_for_tile

    base = lambda q: jnp.full((q.shape[0], 5), 7.0 + 0.0j, dtype=jnp.complex128)
    Kc = lambda q: jnp.zeros((q.shape[0], 5, 3), dtype=jnp.float64)

    fn = _make_v_per_G_for_tile(
        base_v_per_G_fn=base, K_cart_batch_fn=Kc, mu_L=0, nu_L=0,
    )
    # CC closure should BE base (returned as-is).
    assert fn is base


def test_make_v_per_G_for_tile_tt_diagonal_applies_projector():
    """For (i, i) the projector is (1 − K̂_i²); given K aligned with
    axis i, K̂_i = 1 → projector = 0 → v_per_G = 0."""
    from src.gw.v_q_bispinor import _make_v_per_G_for_tile

    # Mock: 2 q-points, 1 G each.  K_cart = (1, 0, 0) at both — i.e.
    # K is purely along axis 1.
    base = lambda q: jnp.full((q.shape[0], 1), 5.0 + 0.0j, dtype=jnp.complex128)
    Kc = lambda q: jnp.broadcast_to(
        jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        (q.shape[0], 1, 3),
    )

    fn = _make_v_per_G_for_tile(
        base_v_per_G_fn=base, K_cart_batch_fn=Kc, mu_L=1, nu_L=1,
    )
    out = fn(np.zeros((2, 3), dtype=np.float64))
    # K̂ along axis 1 → (1 − K̂_1²) = 0 → output = 0.
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-12)


def test_make_v_per_G_for_tile_tt_offdiagonal_applies_minus_khat_ij():
    """For (i, j) i≠j the projector is −K̂_i K̂_j.  K aligned with axis 1
    → K̂_1 = 1, K̂_2 = 0 → projector for (1, 2) = 0."""
    from src.gw.v_q_bispinor import _make_v_per_G_for_tile

    base = lambda q: jnp.full((q.shape[0], 1), 5.0 + 0.0j, dtype=jnp.complex128)
    Kc = lambda q: jnp.broadcast_to(
        jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        (q.shape[0], 1, 3),
    )

    fn = _make_v_per_G_for_tile(
        base_v_per_G_fn=base, K_cart_batch_fn=Kc, mu_L=1, nu_L=2,
    )
    out = fn(np.zeros((1, 3), dtype=np.float64))
    np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-12)

    # Diagonal direction K = (1, 1, 0)/√2 → K̂_1 K̂_2 = 1/2 → projector = −1/2 → v = −2.5
    Kc2 = lambda q: jnp.broadcast_to(
        jnp.array([1.0, 1.0, 0.0], dtype=jnp.float64) / np.sqrt(2),
        (q.shape[0], 1, 3),
    )
    fn2 = _make_v_per_G_for_tile(
        base_v_per_G_fn=base, K_cart_batch_fn=Kc2, mu_L=1, nu_L=2,
    )
    out2 = fn2(np.zeros((1, 3), dtype=np.float64))
    np.testing.assert_allclose(np.asarray(out2.real), -2.5, atol=1e-12)


def test_make_v_per_G_rejects_invalid_indices():
    from src.gw.v_q_bispinor import _make_v_per_G_for_tile
    base = lambda q: jnp.zeros((q.shape[0], 1), dtype=jnp.complex128)
    Kc = lambda q: jnp.zeros((q.shape[0], 1, 3), dtype=jnp.float64)
    with pytest.raises(ValueError):
        _make_v_per_G_for_tile(
            base_v_per_G_fn=base, K_cart_batch_fn=Kc, mu_L=4, nu_L=1,
        )


def test_K_cart_on_sphere_matches_manual_arithmetic():
    """Build K_cart via the helper and compare to a direct (G + q) · b
    computation on a small sphere index list."""
    from src.gw.v_q_bispinor import _make_K_cart_on_sphere_fn

    fft_grid = (4, 4, 4)
    bvec = np.diag([2.0, 3.0, 5.0]).astype(np.float64)
    sphere_idx = jnp.asarray([0, 5, 10], dtype=jnp.int32)

    fn = _make_K_cart_on_sphere_fn(sphere_idx, fft_grid, bvec)
    qvec = jnp.asarray([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]], dtype=jnp.float64)
    K = np.asarray(fn(qvec))
    assert K.shape == (2, 3, 3)
    # K[0] - K[1] should equal q[0] · b broadcast over the sphere, since
    # both share the same G_int_sph and only differ by the q shift.
    np.testing.assert_allclose(
        K[0] - K[1],
        np.broadcast_to(np.asarray(qvec[0]) @ bvec, K[0].shape),
        atol=1e-12,
    )
