"""Tests for the IBZ→full-BZ V_q unfold helpers in ``gw.v_q_tile``.

V_q is a pure centroid-axis double-permute under {S | τ} (eq. 3 of the
zeta_ibz design report).  These tests verify the unfold against an
independently-built reference.

We don't run a full V_q kernel — the unfold is geometric only, so we
construct synthetic ``V_q_ibz`` arrays and the sym tables that would
come out of ``SymMaps.find_irreducible_qpoints`` + ``compute_centroid_sym_perm``.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import pytest

from common.symmetry_maps import unfold_v_q
from gw.v_q_tile import _unfold_g0_ibz_to_full


def _single_device_mesh() -> Mesh:
    """1×1 mesh — single-rank pytest doesn't need sharded fanout."""
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                axis_names=('x', 'y'))


def test_unfold_v_q_identity_sym_is_noop():
    """With only the identity sym, IBZ == full BZ → unfold is identity."""
    mesh = _single_device_mesh()
    n_rmu = 5
    nq = 4

    V_ibz = jnp.asarray(np.random.default_rng(0).standard_normal(
        (nq, n_rmu, n_rmu)) + 0j)
    V_ibz = jax.device_put(V_ibz, NamedSharding(mesh, P(None, 'x', 'y')))

    full_to_irr_idx = np.arange(nq, dtype=np.int32)
    full_to_irr_sym = np.zeros(nq, dtype=np.int32)
    sym_perm = np.arange(n_rmu, dtype=np.int32)[None, :]   # identity

    V_full = unfold_v_q(
        V_ibz,
        irr_idx=full_to_irr_idx,
        sym_idx=full_to_irr_sym,
        sym_perm=sym_perm,
        mesh_xy=mesh,
    )
    np.testing.assert_array_equal(np.asarray(V_full), np.asarray(V_ibz))


def test_unfold_v_q_under_permutation():
    """Two sym ops, one centroid permutation: verify V_full[q, π(μ), π(ν)]
    == V_ibz[i_irr(q), μ, ν] for every q in the full BZ that uses sym 1."""
    mesh = _single_device_mesh()
    n_rmu = 6
    n_qpt_irr = 2
    n_q_full = 5

    rng = np.random.default_rng(seed=7)
    V_ibz_np = rng.standard_normal((n_qpt_irr, n_rmu, n_rmu)).astype(np.complex128)
    V_ibz_np += 1j * rng.standard_normal((n_qpt_irr, n_rmu, n_rmu))
    V_ibz = jax.device_put(jnp.asarray(V_ibz_np),
                            NamedSharding(mesh, P(None, 'x', 'y')))

    # Build a concrete permutation π for sym 1: μ -> (μ+2) mod n_rmu.
    perm_1 = np.array([(m + 2) % n_rmu for m in range(n_rmu)], dtype=np.int32)
    sym_perm = np.stack([np.arange(n_rmu, dtype=np.int32), perm_1], axis=0)

    # 5 full-BZ q's; alternating sym 0 / sym 1, alternating IBZ index.
    full_to_irr_idx = np.array([0, 0, 1, 1, 0], dtype=np.int32)
    full_to_irr_sym = np.array([0, 1, 0, 1, 1], dtype=np.int32)

    V_full = np.asarray(unfold_v_q(
        V_ibz,
        irr_idx=full_to_irr_idx,
        sym_idx=full_to_irr_sym,
        sym_perm=sym_perm,
        mesh_xy=mesh,
    ))
    assert V_full.shape == (n_q_full, n_rmu, n_rmu)

    inv_perm_1 = np.argsort(perm_1)

    # Verify each full-BZ q against eq. 3 of the design report:
    #   V_full[q, μ', ν'] = V_ibz[i_irr, inv_perm[s, μ'], inv_perm[s, ν']]
    for iq in range(n_q_full):
        s = int(full_to_irr_sym[iq])
        ir = int(full_to_irr_idx[iq])
        if s == 0:
            expected = V_ibz_np[ir]
        else:
            expected = V_ibz_np[ir][np.ix_(inv_perm_1, inv_perm_1)]
        np.testing.assert_allclose(V_full[iq], expected, atol=0.0,
            err_msg=f"q={iq} (s={s}, ir={ir})")


def test_unfold_v_q_round_trip_against_eq3():
    """Build a closed-orbit sym group (identity + inversion), construct
    V_ibz at the IBZ wedge, unfold to full BZ, then verify the eq. 3
    identity:  V_full[Sq, π(μ), π(ν)] = V_full[q, μ, ν]  (i.e. V_q is
    sym-covariant after unfold)."""
    mesh = _single_device_mesh()
    n_rmu = 4

    # 2×2×1 kgrid; inversion sym → IBZ has 3 q's (Γ + 2 others; one
    # boundary point at the M-like corner is self-inverse).
    # We just construct the tables by hand.
    full_to_irr_idx = np.array([0, 1, 1, 2], dtype=np.int32)
    full_to_irr_sym = np.array([0, 0, 1, 0], dtype=np.int32)
    # Inversion sym swaps μ ↔ (n_rmu-1-μ) (any concrete closed perm).
    perm_inv = np.array([3, 2, 1, 0], dtype=np.int32)
    sym_perm = np.stack([np.arange(n_rmu, dtype=np.int32), perm_inv], axis=0)

    rng = np.random.default_rng(seed=11)
    V_ibz_np = (rng.standard_normal((3, n_rmu, n_rmu))
                + 1j * rng.standard_normal((3, n_rmu, n_rmu)))
    V_ibz = jax.device_put(jnp.asarray(V_ibz_np),
                            NamedSharding(mesh, P(None, 'x', 'y')))

    V_full = np.asarray(unfold_v_q(
        V_ibz,
        irr_idx=full_to_irr_idx,
        sym_idx=full_to_irr_sym,
        sym_perm=sym_perm,
        mesh_xy=mesh,
    ))

    # Eq. 3:  V_full[Sq, π_s(μ), π_s(ν)] = V_full[q, μ, ν]
    # Pair: q=1 (sym 0) and q=2 (sym 1).  Both have IBZ idx 1.
    q_a, q_b = 1, 2
    # q_b's eq.3-mapped values from q_a via perm_inv:
    expected_q_b = V_full[q_a][np.ix_(perm_inv, perm_inv)]
    np.testing.assert_allclose(V_full[q_b], expected_q_b,
                                atol=0.0, rtol=0.0)


def test_unfold_g0_identity_sym_is_noop():
    mesh = _single_device_mesh()
    n_rmu = 5
    nq = 3
    g0_ibz = jnp.asarray(np.arange(nq * n_rmu, dtype=np.complex128).reshape(
        nq, n_rmu))
    g0_ibz = jax.device_put(g0_ibz, NamedSharding(mesh, P(None, 'x')))

    g0_full = _unfold_g0_ibz_to_full(
        g0_ibz,
        irr_idx=np.arange(nq, dtype=np.int32),
        sym_idx=np.zeros(nq, dtype=np.int32),
        sym_perm=np.arange(n_rmu, dtype=np.int32)[None, :],
        mesh_xy=mesh,
    )
    np.testing.assert_array_equal(np.asarray(g0_full), np.asarray(g0_ibz))


def test_unfold_g0_permutes_mu_axis():
    mesh = _single_device_mesh()
    n_rmu = 4

    perm_1 = np.array([1, 0, 3, 2], dtype=np.int32)
    sym_perm = np.stack([np.arange(n_rmu, dtype=np.int32), perm_1], axis=0)
    full_to_irr_idx = np.array([0, 0], dtype=np.int32)
    full_to_irr_sym = np.array([0, 1], dtype=np.int32)
    inv_perm_1 = np.argsort(perm_1)

    rng = np.random.default_rng(seed=3)
    g0_ibz_np = (rng.standard_normal((1, n_rmu))
                 + 1j * rng.standard_normal((1, n_rmu)))
    g0_ibz = jax.device_put(jnp.asarray(g0_ibz_np),
                              NamedSharding(mesh, P(None, 'x')))

    g0_full = np.asarray(_unfold_g0_ibz_to_full(
        g0_ibz,
        irr_idx=full_to_irr_idx,
        sym_idx=full_to_irr_sym,
        sym_perm=sym_perm,
        mesh_xy=mesh,
    ))
    np.testing.assert_array_equal(g0_full[0], g0_ibz_np[0])
    # q=1 uses sym 1 → output[μ] = ibz[inv_perm_1[μ]]
    np.testing.assert_allclose(g0_full[1], g0_ibz_np[0, inv_perm_1])
