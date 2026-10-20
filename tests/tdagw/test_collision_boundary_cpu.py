"""Host-only Pauli boundary oracle: degenerate blocks, gauge and trace (seconds)."""
import numpy as np
import pytest

from tdagw.collision import boundary_diagnostics


def test_boundary_diagnostics_against_degenerate_outward_oracle():
    """Positive diagonal expectations and zero trace can hide outward blocks."""
    gamma = 2.0 * np.eye(4)
    injection = np.array([[1., 2., 0., 0.], [2., 1., 0., 0.],
                          [0., 0., 1., -2.], [0., 0., -2., 1.]])
    rho = np.diag([0., 0., 1., 1.])
    original = [a.copy() for a in (gamma, injection, rho)]
    got = boundary_diagnostics(gamma, injection, rho, occupation_tolerance=1e-12)
    assert got["empty_count"] == got["filled_count"] == 2
    for key in ("min_injection", "min_blocking", "empty_min_injection",
                "filled_min_blocking"):
        np.testing.assert_allclose(got[key], -1.0, atol=1e-13)
    np.testing.assert_allclose(got["trace_rate"], 0.0, atol=1e-13)
    assert got["rho_min"] == 0.0 and got["rho_max"] == 1.0
    # Independent exact held solution: gamma=2I gives e^-2t rho + (1-e^-2t)D/2.
    step = 1e-3
    evolved = np.exp(-2 * step) * rho + (-np.expm1(-2 * step) / 2) * injection
    eig = np.linalg.eigvalsh(evolved)
    assert eig[0] < 0.0 and eig[-1] > 1.0
    assert np.all(np.diag(injection) >= 0)  # A diagonal-only test misses this.
    assert np.all(np.diag(gamma - injection) >= 0)
    for before, after in zip(original, (gamma, injection, rho)):
        np.testing.assert_array_equal(before, after)

    rng = np.random.default_rng(20261020)
    unitary, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    rotate = lambda a: unitary @ a @ unitary.conj().T
    rotated = boundary_diagnostics(*(rotate(a) for a in (gamma, injection, rho)),
                                   occupation_tolerance=1e-12)
    for key in got:
        np.testing.assert_allclose(rotated[key], got[key], atol=2e-14)

    # Leading indices and absent subspaces retain their own diagnostics.
    batch = boundary_diagnostics(np.stack((gamma, gamma)),
                                np.stack((injection, injection)),
                                np.stack((rho, 0.5 * np.eye(4))),
                                occupation_tolerance=1e-12)
    np.testing.assert_array_equal(batch["empty_count"], [2, 0])
    np.testing.assert_array_equal(batch["filled_count"], [2, 0])
    assert np.isnan(batch["empty_min_injection"][1])
    assert np.isnan(batch["filled_min_blocking"][1])
    near = rho + 2e-7 * np.eye(4)
    loose = boundary_diagnostics(gamma, injection, near, occupation_tolerance=1e-6)
    tight = boundary_diagnostics(gamma, injection, near, occupation_tolerance=1e-8)
    assert loose["empty_count"] == loose["filled_count"] == 2
    assert tight["empty_count"] == tight["filled_count"] == 0
    nonhermitian = injection.astype(complex) + 0.1j * np.eye(4)
    defect = boundary_diagnostics(gamma, nonhermitian, rho, occupation_tolerance=1e-12)
    np.testing.assert_allclose(defect["injection_hermiticity_error"], 0.2, atol=1e-14)
    assert defect["gamma_hermiticity_error"] == defect["rho_hermiticity_error"] == 0.0
    for invalid in (-1.0, 0.5, np.nan):
        with pytest.raises(ValueError, match="occupation_tolerance"):
            boundary_diagnostics(gamma, injection, rho, occupation_tolerance=invalid)


def test_positive_boson_finite_eta_is_inward_and_number_conserving_not_stationary():
    """Literal retarded columns, not an imposed equilibrium counterterm."""
    from tests._full_microscopic_oracle import retarded_reference_pairs

    gap, omega, coupling_squared = 0.7, 0.9, 0.13
    energy = np.array([[0., gap]])
    rho = np.diag([1., 0.])[None].astype(complex)
    vertices = np.array([[0., 1.], [1., 0.]])[None, ..., None]
    poles = np.array([omega, -omega], complex)
    residues = np.array([coupling_squared, -coupling_squared])[:, None, None]
    rates = []
    dagger = lambda a: a.conj().swapaxes(-1, -2)
    for eta in (0.2, 0.02, 1e-9):
        particle, hole = retarded_reference_pairs(
            vertices, energy, rho, energy, poles, residues, eta)
        rp, rh = particle[1], hole[1]
        gamma = 1j * (rp + rh - dagger(rp + rh))
        injection = 1j * (rh - dagger(rh))
        collision = injection - 0.5 * (gamma @ rho + rho @ gamma)
        rate = 2 * coupling_squared * eta / ((gap + omega)**2 + eta**2)
        rates.append(rate)
        np.testing.assert_allclose(collision[0], np.diag([-rate, rate]),
                                   rtol=2e-14, atol=1e-20)
        got = boundary_diagnostics(gamma, injection, rho, occupation_tolerance=1e-12)
        np.testing.assert_allclose(got["trace_rate"], 0., atol=1e-17)
        np.testing.assert_allclose(got["empty_min_injection"], rate, rtol=2e-14)
        np.testing.assert_allclose(got["filled_min_blocking"], rate, rtol=2e-14)
        assert np.linalg.norm(collision) > 0.0
    assert rates[2] < rates[1] < rates[0]
    assert rates[2] < 1e-9
    np.testing.assert_allclose(rates[2] / 1e-9,
                               2 * coupling_squared / (gap + omega)**2, rtol=1e-14)
    assert 2 * coupling_squared * 0.0 / (gap + omega)**2 == 0.0


def test_exact_held_flow_can_drift_despite_zero_fresh_trace_and_positive_boundaries():
    gamma = np.diag([1., 2.])
    injection = np.diag([0.8, 0.7])
    rho_star = 0.5 * np.eye(2)
    fresh = boundary_diagnostics(gamma, injection, rho_star, occupation_tolerance=1e-12)
    np.testing.assert_allclose(fresh["trace_rate"], 0., atol=1e-15)
    assert fresh["min_injection"] > 0.0 and fresh["min_blocking"] > 0.0
    time = 0.1
    stationary_diagonal = np.diag(injection) / np.diag(gamma)
    rho = np.diag(stationary_diagonal + (0.5 - stationary_diagonal)
                  * np.exp(-np.diag(gamma) * time))
    held = boundary_diagnostics(gamma, injection, rho, occupation_tolerance=1e-12)
    np.testing.assert_allclose(held["trace_rate"], -np.trace(gamma @ (rho-rho_star)),
                               atol=1e-15)
    assert held["trace_rate"] > 0.0 and np.trace(rho) > np.trace(rho_star)
