"""Per-state-mass profile regressions for the MPA denominator rule."""

import numpy as np

from gw.sigma_tolerance_profile import (
    build_tolerance_profile,
    per_state_bin_currency,
    profile_grid,
    state_max_mass,
)


def _single_pole_histogram(u_nodes, v_nodes, *, a, gamma, eta):
    histogram = np.zeros((u_nodes.size, v_nodes.size), np.float64)
    iu = np.argmin(np.abs(u_nodes - a))
    iv = np.argmin(np.abs(v_nodes - np.log1p(gamma / eta)))
    histogram[iu, iv] = 1.0
    return histogram


def test_low_mass_state_obeys_the_uniform_per_state_bound():
    """The 1/B_n budget protects a light state and sums to the uniform bar."""
    eta, eps = 0.25, 1.0e-4
    # State 0 is heavy and concentrated in bin 0.  State 1 is arbitrarily
    # light and owns only bin 1.  A global-peak normalization would dilute
    # state 1 by 1e8; the per-state total cancels that scale exactly.
    mass = np.asarray([
        [100.0, 1.0, 1.0],
        [0.0, 1.0e-6, 0.0],
    ])
    rho = per_state_bin_currency(mass, eta)
    tau = np.divide(
        eps, rho, out=np.full_like(rho, np.inf), where=rho > 0.0)
    delivered_bounds = np.sum(mass * tau[None, :], axis=1)
    uniform_bounds = np.sum(mass, axis=1) * eps / eta

    assert rho[1] >= eta  # the light state's sole bin is fully protected
    np.testing.assert_array_less(
        delivered_bounds, np.nextafter(uniform_bounds, np.inf))


def test_low_mass_fermi_shift_is_not_diluted_by_other_states():
    """Duplicating heavy states elsewhere cannot erase one E_F shift."""
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
    # Per-state normalization cancels the 1% absolute state mass entirely.
    assert rho(d_fermi)[0] / eta > 0.5
