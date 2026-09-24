"""Energies outside the sampled omega grid take Sigma(omega=0); no policy flag exists.

Owner rule 2026-09-22: the omega grid only needs to cover the manifold of interest; any
state outside it (semicore, high conduction) uses the static Sigma_mnk(omega=0).  This
replaced the endpoint clamp, the clamp/mask/refuse policy and LORRAX_OMEGA_OUT_OF_RANGE.
"""
from __future__ import annotations

import numpy as np
import pytest

from gw import qsgw_utils
from gw.qsgw_utils import interp_along_omega, omega_coverage


def _cube():
    omega = np.linspace(-2.0, 2.0, 9)                      # E_F-relative, contains 0
    rng = np.random.default_rng(0)
    return omega, rng.normal(size=(omega.size, 3, 4)) + 1j * rng.normal(size=(omega.size, 3, 4))


def test_outside_the_grid_is_the_omega_zero_value():
    omega, vals = _cube()
    e = np.array([[-5.0, 0.3, 7.0, -2.0]] * 3)
    out = interp_along_omega(vals, omega, e)
    i0 = int(np.argmin(np.abs(omega)))
    np.testing.assert_allclose(out[:, 0], vals[i0, :, 0])   # below the grid -> Sigma(0)
    np.testing.assert_allclose(out[:, 2], vals[i0, :, 2])   # above the grid -> Sigma(0)
    np.testing.assert_allclose(out[:, 3], vals[0, :, 3])    # the endpoint itself is inside


def test_inside_the_grid_is_linear_interpolation():
    omega, vals = _cube()
    e = np.full((3, 4), 0.25)                                # halfway between 0.0 and 0.5
    j = int(np.searchsorted(omega, 0.25))
    np.testing.assert_allclose(interp_along_omega(vals, omega, e), 0.5 * (vals[j - 1] + vals[j]))


def test_grid_without_omega_zero_refuses():
    omega = np.linspace(1.0, 3.0, 5)
    with pytest.raises(ValueError, match="omega = 0"):
        interp_along_omega(np.zeros((5, 1, 1)), omega, np.zeros((1, 1)))


def test_the_count_line_names_the_fallback():
    omega, vals = _cube()
    lines = []
    interp_along_omega(vals, omega, np.array([[-5.0, 0.0, 0.1, 9.0]] * 3), context="t", print_fn=lines.append)
    assert len(lines) == 1 and "6 of 12" in lines[0] and "Sigma(omega=0)" in lines[0]
    assert omega_coverage(omega, np.array([[-5.0, 0.0]]))[1] == 1


def test_no_policy_machinery_is_left():
    for name in ("OUT_OF_RANGE_POLICIES", "resolve_out_of_range_policy", "_OUT_OF_RANGE_ENV"):
        assert not hasattr(qsgw_utils, name), name


def test_z_factor_probes_the_continuous_sigma_at_the_grid_edge():
    """Planted Sigma_c = 1 - 0.3*omega on [-10, 10] eV: Z = 1/1.3 everywhere
    the derivative is sampled, including within dE of the edge; the probes
    never take the off-grid Sigma(0) value (review B5 gave Z < 0 there)."""
    from gw.eqp_bgw import compute_z_factor_from_omega_grid
    omega = np.linspace(-10.0, 10.0, 81)
    e = np.array([[9.0, 9.6, 9.9, 10.0, 10.4]])
    sigma = (1.0 - 0.3 * omega)[:, None, None] * np.ones((1, *e.shape))
    _, z = compute_z_factor_from_omega_grid(
        sigma_c_omega_diag_ev=sigma.astype(complex), omega_rel_ev=omega,
        e_dft_rel_ev=e, dE_ev=0.5)
    np.testing.assert_allclose(z, 1.0 / 1.3, rtol=1e-12)
