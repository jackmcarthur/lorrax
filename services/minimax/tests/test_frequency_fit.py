import numpy as np
import pytest

from minimax.frequency_fit import (
    fit_shared_sine_rule,
    positive_complex_chebyshev,
    resolvent_system,
    shared_sine_system,
)


def test_positive_complex_chebyshev_recovers_positive_weights():
    basis = np.asarray([
        [1.0 + 0.2j, 0.3 - 0.1j],
        [0.4 - 0.3j, 1.2 + 0.5j],
        [0.8 + 0.7j, 0.2 + 0.4j],
    ])
    expected = np.asarray([0.7, 0.25])
    fit = positive_complex_chebyshev(basis, basis @ expected,
                                     n_directions=16)
    assert np.all(fit.weights >= 0.0)
    np.testing.assert_allclose(basis @ fit.weights, basis @ expected,
                               atol=2.0e-8, rtol=0.0)


def test_resolvent_system_refuses_inadmissible_contour():
    with pytest.raises(ValueError, match="inadmissible"):
        resolvent_system(np.asarray([-1.0 + 0.1j]), 1.0 + 0.0j,
                          np.asarray([0.1, 1.0]))


def test_positive_resolvent_fit_on_real_interval():
    d = np.geomspace(0.5, 8.0, 200)
    t = np.geomspace(1.0e-3, 20.0, 80)
    basis, target = resolvent_system(d, 1.0 + 0.0j, t)
    fit = positive_complex_chebyshev(basis, target, n_directions=8)
    assert np.max(np.abs(basis @ fit.weights - target)) < 3.0e-3


def test_shared_sine_system_matches_exact_kernel():
    delta = np.asarray([0.7, 1.4])
    z = np.asarray([0.3 + 0.2j, 0.8 + 0.5j])
    t = np.asarray([0.1, 0.5, 2.0])
    basis, exact = shared_sine_system(delta, z, t, scale=z.imag)
    assert basis.shape == (delta.size * z.size, t.size)
    reference = (-2.0 * delta[None, :]
                 / (delta[None, :] ** 2 - z[:, None] ** 2)
                 * z.imag[:, None])
    np.testing.assert_allclose(exact.reshape(z.size, delta.size), reference)


def test_shared_sine_fit_reduces_dense_error_and_reports_executed_rank():
    z = np.asarray([0.4 + 0.5j, 1.2 + 0.5j, 0.0 + 2.0j])
    candidate = np.geomspace(1.0e-3, 35.0, 96)
    result = fit_shared_sine_rule(
        z, candidate, (0.3, 3.0), row_scale=z.imag,
        solve_points=81, validation_points=2048, exchange_rounds=1,
        n_directions=12)
    assert result.sine_rank < candidate.size
    assert result.executed_contour_nodes == 2 * result.sine_rank
    assert result.sampled_max_error < 2.0e-2
    assert np.all(result.weights >= 0.0)
