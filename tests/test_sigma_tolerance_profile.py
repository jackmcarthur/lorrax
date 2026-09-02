"""Per-state-mass profile regressions for the MPA denominator rule."""

import numpy as np

from gw.sigma_tolerance_profile import (
    build_tolerance_profile,
    profile_grid,
    state_max_mass,
)


def _single_pole_histogram(u_nodes, v_nodes, *, a, gamma, eta):
    histogram = np.zeros((u_nodes.size, v_nodes.size), np.float64)
    iu = np.argmin(np.abs(u_nodes - np.log1p(a / eta)))
    iv = np.argmin(np.abs(v_nodes - np.log1p(gamma / eta)))
    histogram[iu, iv] = 1.0
    return histogram


def test_low_mass_fermi_state_is_not_diluted_by_other_states():
    """Duplicating heavy states elsewhere cannot erase one E_F mass row."""
    eta = 0.1
    u_nodes, v_nodes = profile_grid(8.0, 1.0, eta, 1.0e-4)
    histogram = _single_pole_histogram(
        u_nodes, v_nodes, a=1.0, gamma=0.0, eta=eta)
    d_fermi = np.asarray([-1.0 + 1j * eta])

    reference = state_max_mass(
        d_fermi, np.asarray([0.0, 5.0]), np.asarray([0.01, 1.0]),
        np.asarray([0.0]), 1.0, histogram, u_nodes, v_nodes, eta,
        eps=1.0e-4)
    duplicated = state_max_mass(
        d_fermi, np.asarray([0.0] + [5.0] * 200),
        np.asarray([0.01] + [1.0] * 200), np.asarray([0.0]), 1.0,
        histogram, u_nodes, v_nodes, eta, eps=1.0e-4)
    np.testing.assert_allclose(duplicated, reference, rtol=0.0, atol=0.0)
    assert reference[0] > 0.0

    rho, _digest, _report = build_tolerance_profile(
        (-7.0, 1.0, eta, 1.1), "crossing", 1.0,
        np.asarray([0.0] + [5.0] * 200),
        np.asarray([0.01] + [1.0] * 200), np.asarray([0.0]),
        histogram, u_nodes, v_nodes, eta, eps=1.0e-4)
    # A summed-state normalization would suppress this by another factor of
    # roughly 200.  The max-over-states profile retains its own 1% mass.
    assert rho(d_fermi)[0] / eta > 5.0e-3
