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
