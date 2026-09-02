"""First-principles gates for the user-ruled GN 0.2% tail policy.

The inputs are analytic, ordered, and non-random.  These gates own only the
representation-level ruling: exact tail budgets/ties, the preserved static
identity, changed high-frequency moment, and reduced fitted-pole support.
This is not strict BGW pole parity.  The dynamic window planner is the shared
MPA route and is gated separately.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.minimax_screening import (
    GN_PPM_EXTREME_TAIL_DIVISOR,
    _coarsen_gn_ppm_extreme_tails,
)
from symmetry_maps import q_negation_index


def _run_policy(
    omega: np.ndarray,
    Wc0: np.ndarray,
    fallback: float = 2.0,
    valid: np.ndarray | None = None,
):
    omega_j = jnp.asarray(omega, dtype=jnp.float64)
    Wc0_j = jnp.asarray(Wc0, dtype=jnp.complex128)
    B_j = -0.5 * Wc0_j * omega_j
    valid_j = jnp.asarray(
        np.ones(omega.shape, dtype=bool) if valid is None else valid)
    q_neg_j = jnp.asarray(q_negation_index((omega.shape[0], 1, 1)))
    out = _coarsen_gn_ppm_extreme_tails(
        omega_j, B_j, valid_j, Wc0_j, q_neg_j,
        jnp.asarray(fallback, dtype=jnp.float64),
        tail_divisor=GN_PPM_EXTREME_TAIL_DIVISOR,
    )
    return tuple(np.asarray(jax.device_get(x)) for x in out)


def test_extreme_tails_reduce_range_and_preserve_static_wc0():
    # 1024 valid lanes -> exact budget 2 on each side.  Each tail is one
    # complete same-q transpose orbit, so both candidates survive closure.
    i, j = np.indices((32, 32))
    omega = (1.0 + 4.0 * (i + j) / 62.0)[None, ...]
    omega[0, 0, 1] = omega[0, 1, 0] = 1.0e-12
    omega[0, 2, 3] = omega[0, 3, 2] = 1.0e9
    Wc0 = (-2.0 + 0.01 * (i + j))[None, ...].astype(np.complex128)
    B_before = -0.5 * Wc0 * omega

    (omega_after, B_after, n_low, n_high,
     omega_min_after, omega_max_after,
     anchor) = _run_policy(omega, Wc0)

    assert int(n_low) == int(n_high) == 2
    assert float(anchor) == 2.0
    tail = np.zeros(omega.shape, dtype=bool)
    tail[0, 0, 1] = tail[0, 1, 0] = True
    tail[0, 2, 3] = tail[0, 3, 2] = True
    np.testing.assert_array_equal(omega_after[~tail], omega[~tail])
    np.testing.assert_array_equal(B_after[~tail], B_before[~tail])
    assert float(omega_min_after) == 1.0
    assert float(omega_max_after) == 5.0

    # The chosen invariant is the exact static fitted observable.
    np.testing.assert_allclose(
        -2.0 * B_after / omega_after, Wc0, rtol=2.0e-16, atol=2.0e-16)
    # A nontrivial clamp cannot preserve the 1/z^2 coefficient 2 B Omega.
    assert np.all((2.0 * B_after * omega_after)[tail]
                  != (2.0 * B_before * omega)[tail])

    # The live support presented to the one shared MPA planner is bounded.
    assert float(omega.min()) < 1.0e-10
    assert float(omega.max()) > 1.0e8
    assert (float(omega_min_after), float(omega_max_after)) == (1.0, 5.0)


def test_boundary_degeneracy_is_not_split_or_over_budget():
    # Three diagonal singleton orbits share the lower boundary while the
    # budget is two, so all stay.  One exact upper transpose pair is changed.
    i, j = np.indices((32, 32))
    omega = (1.0 + 4.0 * (i + j) / 62.0)[None, ...]
    omega[0, 0, 0] = omega[0, 1, 1] = omega[0, 2, 2] = 0.1
    omega[0, 30, 31] = omega[0, 31, 30] = 200.0
    Wc0 = np.full(omega.shape, -2.0 + 0.5j, dtype=np.complex128)
    omega_after, _B, n_low, n_high, *_ = _run_policy(omega, Wc0)

    assert int(n_low) == 0
    assert int(n_high) == 2
    np.testing.assert_array_equal(
        omega_after[0, (0, 1, 2), (0, 1, 2)],
        omega[0, (0, 1, 2), (0, 1, 2)])
    assert omega_after[0, 30, 31] == omega_after[0, 31, 30] == 2.0


def test_subbudget_is_byte_exact_and_invalid_extrema_do_not_count():
    # 499 valid lanes have zero budget even though 30 invalid/padded lanes
    # take the carrier over 500 and contain the global numerical extrema.
    omega = np.ones(23 * 23, dtype=np.float64)
    omega[:499] = np.linspace(1.0, 5.0, 499)
    omega[499:] = 10.0
    omega[499:501] = [1.0e-300, 1.0e300]
    omega = omega.reshape(1, 23, 23)
    valid = np.r_[np.ones(499, dtype=bool), np.zeros(30, dtype=bool)]
    valid = valid.reshape(1, 23, 23)
    Wc0 = np.full(omega.shape, -2.0 + 0.5j, dtype=np.complex128)
    B_before = -0.5 * Wc0 * omega

    (omega_after, B_after, n_low, n_high,
     omega_min_after, omega_max_after, anchor) = _run_policy(
         omega, Wc0, valid=valid)

    np.testing.assert_array_equal(omega_after.view(np.uint64),
                                  omega.view(np.uint64))
    np.testing.assert_array_equal(B_after.view(np.uint64),
                                  B_before.view(np.uint64))
    assert int(n_low) == int(n_high) == 0
    assert float(omega_min_after) == 1.0
    assert float(omega_max_after) == 5.0
    assert np.isnan(anchor)


def test_one_ulp_partner_boundary_keeps_both_physical_orbits_dynamic():
    # 3*20^2 valid lanes -> budget 2.  The low candidate contains only the
    # smaller same-q partners; the high candidate contains only the q-side
    # partners.  Orbit closure must reject both partial selections.
    omega = np.full((3, 20, 20), 2.0, dtype=np.float64)
    low, low_next = 0.1, np.nextafter(0.1, np.inf)
    omega[1, 0, 1] = omega[2, 0, 1] = low
    omega[1, 1, 0] = omega[2, 1, 0] = low_next
    high, high_prev = 100.0, np.nextafter(100.0, -np.inf)
    omega[1, 2, 3] = omega[1, 3, 2] = high
    omega[2, 2, 3] = omega[2, 3, 2] = high_prev
    Wc0 = np.full(omega.shape, -2.0, dtype=np.complex128)

    omega_after, B_after, n_low, n_high, *_ = _run_policy(omega, Wc0)

    np.testing.assert_array_equal(omega_after.view(np.uint64),
                                  omega.view(np.uint64))
    np.testing.assert_array_equal(B_after.view(np.uint64),
                                  (-0.5 * Wc0 * omega).view(np.uint64))
    assert int(n_low) == int(n_high) == 0


def test_offdiagonal_conjugate_extrema_remain_hermitian():
    # 32^2 valid lanes -> budget two per side.  Put each tail on one exact
    # off-diagonal conjugate pair; a selector that split matrix partners would
    # make either Omega or B non-Hermitian here.
    i, j = np.indices((32, 32))
    omega = (1.0 + 4.0 * (i + j) / 62.0).astype(np.float64)
    omega[0, 1] = omega[1, 0] = 1.0e-12
    omega[2, 3] = omega[3, 2] = 1.0e9
    Wc0 = (
        -2.0 + 0.01 * (i + j) + 0.02j * (i - j)
    ).astype(np.complex128)
    np.testing.assert_array_equal(Wc0, Wc0.T.conj())
    np.testing.assert_array_equal(omega, omega.T)

    omega_after, B_after, n_low, n_high, *_ = _run_policy(
        omega[None, ...], Wc0[None, ...])
    omega_after = omega_after[0]
    B_after = B_after[0]

    assert int(n_low) == int(n_high) == 2
    np.testing.assert_array_equal(omega_after, omega_after.T)
    np.testing.assert_array_equal(B_after, B_after.T.conj())
    np.testing.assert_allclose(
        -2.0 * B_after / omega_after, Wc0, rtol=2.0e-16, atol=2.0e-16)


@pytest.mark.mesh(4)
def test_tail_reduction_preserves_a_real_2x2_sharding():
    if len(jax.devices()) < 4:
        pytest.skip("requires four real devices")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    i, j = np.indices((32, 32))
    omega = (1.0 + 4.0 * (i + j) / 62.0)[None, ...]
    omega[0, 0, 1] = omega[0, 1, 0] = 1.0e-12
    omega[0, 2, 3] = omega[0, 3, 2] = 1.0e9
    Wc0 = np.full(omega.shape, -2.0 + 0.5j, dtype=np.complex128)
    omega_j = jax.device_put(omega, sharding)
    Wc0_j = jax.device_put(Wc0, sharding)
    B_j = -0.5 * Wc0_j * omega_j
    valid_j = jax.device_put(np.ones(omega.shape, dtype=bool), sharding)

    omega_after, B_after, n_low, n_high, *_ = (
        _coarsen_gn_ppm_extreme_tails(
            omega_j, B_j, valid_j, Wc0_j,
            jnp.asarray(q_negation_index((1, 1, 1))),
            jnp.asarray(2.0, dtype=jnp.float64),
            tail_divisor=GN_PPM_EXTREME_TAIL_DIVISOR,
        ))
    jax.block_until_ready((omega_after, B_after))

    assert tuple(omega_after.sharding.spec) == (None, "x", "y")
    assert tuple(B_after.sharding.spec) == (None, "x", "y")
    assert int(n_low) == int(n_high) == 2
