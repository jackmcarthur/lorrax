"""First-principles gates for the GN fitted-pole 0.2% tail policy.

The inputs are analytic, ordered, and non-random.  These gates own only the
representation-level ruling: exact tail budgets/ties, the static identity,
and the dynamic-range reduction seen by the existing minimax pane planner.
They do not duplicate a Sigma or minimax kernel.
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
from gw.ppm_windows import _plan_sign_definite_omega_panes


def _run_policy(omega: np.ndarray, Wc0: np.ndarray, fallback: float = 2.0):
    omega_j = jnp.asarray(omega, dtype=jnp.float64)
    Wc0_j = jnp.asarray(Wc0, dtype=jnp.complex128)
    B_j = -0.5 * Wc0_j * omega_j
    valid_j = jnp.ones(omega.shape, dtype=bool)
    out = _coarsen_gn_ppm_extreme_tails(
        omega_j, B_j, valid_j, Wc0_j,
        jnp.asarray(fallback, dtype=jnp.float64),
        tail_divisor=GN_PPM_EXTREME_TAIL_DIVISOR,
    )
    return tuple(np.asarray(jax.device_get(x)) for x in out)


def test_extreme_tails_reduce_range_and_preserve_static_wc0():
    # 1000 valid lanes -> exact budget 2 on each side.
    central = np.linspace(1.0, 5.0, 996, dtype=np.float64)
    omega = np.concatenate(([1.0e-12, 2.0e-12], central, [1.0e8, 1.0e9]))
    Wc0 = (
        np.linspace(-3.0, -1.0, omega.size)
        + 1j * np.linspace(0.25, 0.75, omega.size)
    ).astype(np.complex128)
    B_before = -0.5 * Wc0 * omega

    (omega_after, B_after, n_low, n_high,
     omega_min_after, omega_max_after,
     _lower_boundary, _upper_boundary, anchor) = _run_policy(omega, Wc0)

    assert int(n_low) == int(n_high) == 2
    assert float(anchor) == 2.0
    np.testing.assert_array_equal(omega_after[2:-2], omega[2:-2])
    np.testing.assert_array_equal(B_after[2:-2], B_before[2:-2])
    assert float(omega_min_after) == 1.0
    assert float(omega_max_after) == 5.0

    # The chosen invariant is the exact static fitted observable.
    np.testing.assert_allclose(
        -2.0 * B_after / omega_after, Wc0, rtol=2.0e-16, atol=2.0e-16)
    # A nontrivial clamp cannot preserve the 1/z^2 coefficient 2 B Omega.
    changed = np.r_[0:2, omega.size - 2:omega.size]
    assert np.all((2.0 * B_after * omega_after)[changed]
                  != (2.0 * B_before * omega)[changed])

    # The canonical exact pane owner sees the intended cost collapse; the
    # tail policy itself does not grow a second minimax implementation.
    mask = jnp.ones(omega.shape, dtype=bool)
    raw_panes = _plan_sign_definite_omega_panes(
        Omega_q=jnp.asarray(omega), base_mask_B=mask,
        mask_B_count=omega.size,
        mask_B_min=float(omega.min()), mask_B_max=float(omega.max()),
        E_min=0.25, E_max=4.0, omega_max=1.0)
    reduced_panes = _plan_sign_definite_omega_panes(
        Omega_q=jnp.asarray(omega_after), base_mask_B=mask,
        mask_B_count=omega.size,
        mask_B_min=float(omega_min_after),
        mask_B_max=float(omega_max_after),
        E_min=0.25, E_max=4.0, omega_max=1.0)
    assert len(raw_panes) > 1
    assert len(reduced_panes) == 1


def test_boundary_degeneracy_is_not_split_or_over_budget():
    # The lower boundary group has three equal values but the budget is two:
    # all three stay untouched.  The distinct upper pair is coarsened.
    omega = np.concatenate((
        np.full(3, 0.1),
        np.linspace(1.0, 5.0, 995, dtype=np.float64),
        np.asarray([100.0, 200.0]),
    ))
    Wc0 = np.full(omega.shape, -2.0 + 0.5j, dtype=np.complex128)
    omega_after, _B, n_low, n_high, *_ = _run_policy(omega, Wc0)

    assert int(n_low) == 0
    assert int(n_high) == 2
    np.testing.assert_array_equal(omega_after[:3], omega[:3])
    assert np.all(omega_after[-2:] == 2.0)


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
    omega = np.concatenate((
        np.asarray([1.0e-12, 2.0e-12]),
        np.linspace(1.0, 5.0, 996),
        np.asarray([1.0e8, 1.0e9]),
    )).reshape(1, 20, 50)
    Wc0 = np.full(omega.shape, -2.0 + 0.5j, dtype=np.complex128)
    omega_j = jax.device_put(omega, sharding)
    Wc0_j = jax.device_put(Wc0, sharding)
    B_j = -0.5 * Wc0_j * omega_j
    valid_j = jax.device_put(np.ones(omega.shape, dtype=bool), sharding)

    omega_after, B_after, n_low, n_high, *_ = (
        _coarsen_gn_ppm_extreme_tails(
            omega_j, B_j, valid_j, Wc0_j,
            jnp.asarray(2.0, dtype=jnp.float64),
            tail_divisor=GN_PPM_EXTREME_TAIL_DIVISOR,
        ))
    jax.block_until_ready((omega_after, B_after))

    assert tuple(omega_after.sharding.spec) == (None, "x", "y")
    assert tuple(B_after.sharding.spec) == (None, "x", "y")
    assert int(n_low) == int(n_high) == 2
