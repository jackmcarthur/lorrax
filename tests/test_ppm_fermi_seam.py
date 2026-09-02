"""Regression algebra for the GN/HL-PPM Fermi-frequency seam.

The incumbent executor mixed two different target functions at ``omega=0``:
the crossing half used the HGL-regularized reciprocal while the sign-definite
half used the ordinary reciprocal.  A wide omega grid raised ``xi`` enough
that those two targets had different one-sided limits.  The box owner instead
fits the same causal denominator on both halves.
"""

import numpy as np

from gw.sigma_box_plan import _box_for_window
from minimax import G_hgl


def _hgl_reciprocal(value, xi):
    value = np.asarray(value, np.float64)
    return np.sign(value) * G_hgl(np.abs(value) / xi) / xi


def _legacy_half_limits(e_cond, h_val, pole, r_plus, r_minus, xi):
    """Return the exact targets used on the old positive/negative halves."""
    s_cond = e_cond + pole
    s_val = h_val + pole
    positive = -r_plus * _hgl_reciprocal(s_cond, xi) + r_minus / s_val
    negative = -r_plus / s_cond + r_minus * _hgl_reciprocal(s_val, xi)
    return positive, negative


def _causal_sigma(omega, e_cond, h_val, pole, r_plus, r_minus, eta):
    """One-pole causal denominator represented by the shared box route."""
    return (
        r_plus / (omega - e_cond - pole + 1j * eta)
        + r_minus / (omega + h_val + pole - 1j * eta)
    )


def test_legacy_mixed_targets_have_a_seam_even_when_odd_residue_is_zero():
    """Red twin: HGL/exact mixing itself, not ``D``, creates the jump."""
    b = np.asarray([[1.0, 0.2 + 0.1j], [0.2 - 0.1j, 0.8]])
    pole = np.asarray([[1.5, 1.7], [1.7, 1.4]])
    e_cond = np.asarray([[0.7, 0.9], [0.9, 1.1]])
    h_val = np.asarray([[0.6, 0.8], [0.8, 1.0]])

    positive, negative = _legacy_half_limits(
        e_cond, h_val, pole, b, b, xi=8.571428571428571)

    assert np.max(np.abs(positive - negative)) > 0.5


def test_box_causal_target_has_one_fermi_limit_for_both_residue_arms():
    """The production target is smooth through zero for ``D=0`` and ``D!=0``."""
    b = np.asarray([[1.0, 0.2 + 0.1j], [0.2 - 0.1j, 0.8]])
    d = np.asarray([[0.12, -0.04j], [0.04j, -0.07]])
    pole = np.asarray([[1.5, 1.7], [1.7, 1.4]])
    e_cond = np.asarray([[0.7, 0.9], [0.9, 1.1]])
    h_val = np.asarray([[0.6, 0.8], [0.8, 1.0]])
    eta = 8.571428571428571
    step = 1.0e-5

    for odd in (np.zeros_like(d), d):
        minus = _causal_sigma(
            -step, e_cond, h_val, pole, b + odd, b - odd, eta)
        zero = _causal_sigma(
            0.0, e_cond, h_val, pole, b + odd, b - odd, eta)
        plus = _causal_sigma(
            step, e_cond, h_val, pole, b + odd, b - odd, eta)
        np.testing.assert_allclose(
            plus - 2.0 * zero + minus, 0.0, rtol=0.0, atol=1.0e-10)


def test_box_geometry_uses_the_same_physical_denominator_on_both_halves():
    """Both frequency halves reach the one denominator-box owner by sign."""
    step, eta, state, pole = 1.0e-5, 0.63, 0.7, 1.5
    pole_stats = ((pole, pole, 0.0, 0.0),)

    for pole_sign in (+1.0, -1.0):
        _, negative_support, _ = _box_for_window(
            np.asarray([-step]), np.asarray([state]),
            pole_stats, pole_sign, eta)
        _, positive_support, _ = _box_for_window(
            np.asarray([step]), np.asarray([state]),
            pole_stats, pole_sign, eta)
        expected_negative = -step - pole_sign * (state + pole)
        expected_positive = step - pole_sign * (state + pole)
        np.testing.assert_array_equal(
            negative_support, (expected_negative, expected_negative))
        np.testing.assert_array_equal(
            positive_support, (expected_positive, expected_positive))
        np.testing.assert_allclose(
            positive_support[0] - negative_support[0],
            2.0 * step,
            rtol=0.0,
            atol=8.0 * np.finfo(np.float64).eps,
        )
