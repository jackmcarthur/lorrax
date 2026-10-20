"""Small-matrix gates for the future nonequilibrium Keldysh screening bank."""

import numpy as np
import pytest
import scipy.integrate as sint

from tdagw import keldysh_rational as kr


def _direct_cauchy(system, z):
    """Direct real-frequency reference for C_-[F](z)."""
    value, error = sint.quad_vec(
        lambda omega: 1j * kr.evaluate(system, omega) / (2.0 * np.pi * (z - omega)),
        -np.inf, np.inf, epsabs=2e-10, epsrel=2e-10, limit=1000)
    assert error < 2e-8
    return value


def _random_residues(seed, n_pole, n):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_pole, n, n)) + 1j * rng.normal(size=(n_pole, n, n))


def test_sharp_and_series_product_are_pointwise_exact():
    residues = _random_residues(4, 2, 2)
    h = kr.pole_residue_realization([-0.7 - 0.4j, 1.1 - 0.25j], residues,
                                    constant=np.array([[1.2, 0.3j], [-0.3j, 0.8]]))
    g = kr.pole_residue_realization([0.2 - 0.35j], _random_residues(7, 1, 2))
    z = -0.31 + 0.73j
    np.testing.assert_allclose(
        kr.evaluate(kr.sharp(h), z), kr.evaluate(h, np.conj(z)).conj().T,
        rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        kr.evaluate(kr.multiply(h, g), z), kr.evaluate(h, z) @ kr.evaluate(g, z),
        rtol=3e-14, atol=3e-14)


def test_causal_projection_recovers_lower_half_plane_part():
    lower = kr.pole_residue_realization(
        [-0.8 - 0.3j, 0.9 - 0.45j], _random_residues(9, 2, 2))
    spectrum = kr.add(lower, kr.scale(kr.sharp(lower), -1.0))
    projected = kr.causal_part(spectrum)
    for z in (-1.2 + 0.6j, 0.1 + 0.9j, 1.4 + 0.2j):
        np.testing.assert_allclose(kr.evaluate(projected, z), kr.evaluate(lower, z),
                                   rtol=2e-13, atol=2e-13)


def test_retarded_feedback_matches_dyson_solve():
    pret = kr.pole_residue_realization(
        [-0.9 - 0.2j, 0.7 - 0.3j], 0.04 * _random_residues(31, 2, 2))
    v = np.array([[1.1, 0.08j], [-0.08j, 0.7]])
    wr = kr.screen_retarded(pret, v)
    identity = np.eye(2)
    for z in (-0.4 + 0.7j, 0.8 + 0.5j):
        reference = np.linalg.solve(identity - v @ kr.evaluate(pret, z), v)
        np.testing.assert_allclose(kr.evaluate(wr, z), reference,
                                   rtol=3e-14, atol=3e-14)


def test_screened_lesser_matches_direct_frequency_projection_with_repeated_poles():
    # W^R and P^< deliberately share a pole.  Their product contains a repeated
    # lower-half-plane pole, which the state-space projection must retain.
    shared = -0.35 - 0.28j
    wr = kr.pole_residue_realization(
        [shared, 1.0 - 0.4j], _random_residues(12, 2, 2),
        constant=np.array([[1.4, 0.2j], [-0.2j, 0.9]]))
    pless = kr.pole_residue_realization(
        [shared, -1.1 - 0.37j], _random_residues(18, 2, 2))
    phat = kr.add(pless, kr.scale(kr.sharp(pless), -1.0))
    full = kr.multiply(kr.multiply(wr, phat), kr.sharp(wr))
    wless = kr.screened_lesser(wr, pless)
    for z in (-0.6 + 0.8j, 0.7 + 0.55j):
        np.testing.assert_allclose(kr.evaluate(wless, z), _direct_cauchy(full, z),
                                   rtol=2e-9, atol=2e-9)


def test_half_filled_noise_survives_without_retarded_correlation_poles():
    # rho=f I makes the retarded occupation differences vanish, but the lesser
    # transition noise is nonzero.  A retarded-only pole bank would lose it.
    v = np.array([[1.3, 0.2], [0.2, 0.8]], dtype=np.complex128)
    wr = kr.pole_residue_realization([], np.zeros((0, 2, 2)), constant=v)
    pless = kr.pole_residue_realization(
        [-0.75 - 0.22j], np.array([[[0.3j, 0.08], [-0.04, 0.2j]]]))
    wless = kr.screened_lesser(wr, pless)
    assert wless[0].shape[0] > 0
    for z in (-0.2 + 0.5j, 0.9 + 0.7j):
        np.testing.assert_allclose(kr.evaluate(wless, z),
                                   v @ kr.evaluate(pless, z) @ v.conj().T,
                                   rtol=2e-13, atol=2e-13)


def test_causal_projection_refuses_undeclared_lossless_limit_and_direct_term():
    lossless = kr.pole_residue_realization([0.5], np.ones((1, 1, 1)))
    with pytest.raises(ValueError, match="declared regulator"):
        kr.causal_part(lossless)
    constant = kr.pole_residue_realization([], np.zeros((0, 1, 1)),
                                           constant=np.ones((1, 1)))
    with pytest.raises(ValueError, match="strictly proper"):
        kr.causal_part(constant)
