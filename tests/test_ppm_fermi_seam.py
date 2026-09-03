"""Regression algebra for the GN/HL-PPM Fermi-frequency seam.

The legacy executor selects the HGL-regularized target on the crossing half
and the ordinary reciprocal on the other half.  At the requested
``xi = eta = 0.25 eV`` those targets already agree at the toy's Fermi limit
to below ``1e-3 eV^-1``.  The fixed four-xi crossing padding additionally
keeps every ordinary-reciprocal cell outside the near-axis region.
"""

import numpy as np
import pytest

from minimax import G_hgl


def _hgl_reciprocal(value, xi):
    value = np.asarray(value, np.float64)
    return np.sign(value) * G_hgl(np.abs(value) / xi) / xi


def _half_limits(e_cond, h_val, pole, r_plus, r_minus, xi, padding_factor):
    """Return the mixed targets after applying the HGL crossing shell."""
    s_cond = e_cond + pole
    s_val = h_val + pole
    crossing_limit = (np.inf if padding_factor is None
                      else float(padding_factor) * xi)
    cond_target = np.where(
        np.abs(s_cond) <= crossing_limit,
        _hgl_reciprocal(s_cond, xi),
        1.0 / s_cond,
    )
    val_target = np.where(
        np.abs(s_val) <= crossing_limit,
        _hgl_reciprocal(s_val, xi),
        1.0 / s_val,
    )
    positive = -r_plus * cond_target + r_minus / s_val
    negative = -r_plus / s_cond + r_minus * val_target
    return positive, negative


def _toy():
    b = np.asarray([[1.0, 0.2 + 0.1j], [0.2 - 0.1j, 0.8]])
    pole = np.asarray([[1.5, 1.7], [1.7, 1.4]])
    e_cond = np.asarray([[0.7, 0.9], [0.9, 1.1]])
    h_val = np.asarray([[0.6, 0.8], [0.8, 1.0]])
    return e_cond, h_val, pole, b


def _assert_fermi_mismatch_below(limit, *, xi, padding_factor):
    e_cond, h_val, pole, residue = _toy()
    positive, negative = _half_limits(
        e_cond, h_val, pole, residue, residue, xi, padding_factor)
    mismatch = float(np.max(np.abs(positive - negative)))
    assert mismatch < limit, (
        f"Fermi target mismatch {mismatch:.12e} eV^-1 is not below "
        f"{limit:.12e} eV^-1 at xi={xi:.12e} eV")
    return mismatch


def test_requested_xi_and_padding_close_the_fermi_seam():
    """The named shell keeps this toy's denominators on one plain target."""
    from gw.ppm_windows import HGL_CROSSING_PADDING_FACTOR

    mismatch = _assert_fermi_mismatch_below(
        1.0e-3,
        xi=0.25,
        padding_factor=HGL_CROSSING_PADDING_FACTOR,
    )
    assert mismatch == 0.0

    # Even without the shell, requested xi already reaches the seam report's
    # 0.25 eV level.  The shell removes rather than conceals that residual.
    unpadded = _assert_fermi_mismatch_below(
        1.0e-3, xi=0.25, padding_factor=None)
    assert unpadded == pytest.approx(3.8505266699129237e-4, rel=2.0e-13)


def test_inflated_xi_without_padding_is_a_red_twin():
    """Negative control: the shipped wide-grid xi fails the seam criterion."""
    with pytest.raises(AssertionError, match="Fermi target mismatch"):
        _assert_fermi_mismatch_below(
            1.0e-3,
            xi=8.571428571428571,
            padding_factor=None,
        )


def test_four_xi_padding_excludes_near_axis_denominators_from_laplace_cells():
    """Every non-crossing cell starts at the named four-xi boundary."""
    from gw.ppm_windows import (
        HGL_CROSSING_PADDING_FACTOR,
        plan_hgl_crossing_cells,
    )

    xi = 0.25
    plan = plan_hgl_crossing_cells(
        omega_abs=np.asarray([0.0, 0.5]),
        E_A=np.asarray([[0.2, 0.7]]),
        base_mask_A=np.asarray([[True, True]]),
        regularization_width_ry=xi,
        edge_factor=HGL_CROSSING_PADDING_FACTOR,
        omega_cluster_gap_ry=1.0,
        omega_max_span_ry=0.1,
    )
    padding = HGL_CROSSING_PADDING_FACTOR * xi
    assert HGL_CROSSING_PADDING_FACTOR == 4.0
    for cell in plan.cells:
        if cell.kind == "positive":
            closest = cell.omega_lo - cell.e_max - cell.b_hi
            assert closest >= padding * (1.0 - 8.0 * np.finfo(float).eps)
        elif cell.kind == "negative":
            closest = cell.omega_hi - cell.e_min - cell.b_lo
            assert closest <= -padding * (1.0 - 8.0 * np.finfo(float).eps)
