"""Physical sign, gauge, finite-T and kinetic-balance checks."""
import numpy as np
import jax.numpy as jnp

from common.bispinor_init import lift_to_4spinor, apply_dirac_velocity_to_ket
from psp.dft_operators import momentum_matrix_k
from psp.orbital_response import orbital_moments, orbital_magnetization
from psp.orbital_magnetization import orbital_pieces_at_k, MU_B_PREFACTOR


def fixture():
    rng = np.random.default_rng(73)
    r = rng.normal(size=(3, 5, 5)) + 1j * rng.normal(size=(3, 5, 5))
    r = (r + r.conj().swapaxes(-1, -2)) / 2
    e = np.array([-1.2, -0.4, 0.1, 0.5, 1.3])
    v = 1j * (e[:, None] - e[None, :]) * r
    return r, e, v


def test_electron_moment_matches_bound_state_angular_momentum():
    r, e, v = fixture()
    moment, _, flags = orbital_moments(v, e)
    # p=v/2 in Ry units; electron moment/mu_B=-<r cross p>.
    angular = np.stack(tuple(np.diag(r[a] @ v[b] - r[b] @ v[a]).real / 2
                              for a, b in ((1, 2), (2, 0), (0, 1))))
    np.testing.assert_allclose(moment, -angular, atol=2e-14)
    assert not np.any(flags)


def test_finite_temperature_trace_and_energy_gauge():
    _, e, v = fixture()
    mu, width = 0.17, 0.08
    moment, berry, _ = orbital_moments(v, e)
    f = 1 / (1 + np.exp((e - mu) / width))
    expected = np.sum(moment * f + berry * width * np.logaddexp(0, (mu-e)/width), axis=-1)
    actual = orbital_magnetization(v, e, mu_ry=mu, width_ry=width)
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    np.testing.assert_allclose(actual, orbital_magnetization(
        v, e+91, mu_ry=mu+91, width_ry=width), atol=2e-12)
    extreme = orbital_magnetization(v, e, mu_ry=100, width_ry=1e-8)
    assert np.isfinite(extreme).all()
    pa, pb = orbital_pieces_at_k(v, e, 2, 1e-8)
    legacy = MU_B_PREFACTOR * np.imag((pa - 2*(-0.2)*pb).sum(axis=(-2,-1)))
    np.testing.assert_allclose(legacy, orbital_magnetization(
        v, e, mu_ry=-0.2, width_ry=0), atol=2e-14)


def test_band_phase_and_degenerate_trace():
    _, e, v = fixture()
    phase = np.exp(1j * np.arange(5)**2)
    phased = phase.conj()[None, :, None] * v * phase[None, None, :]
    np.testing.assert_allclose(orbital_moments(phased, e)[0],
                               orbital_moments(v, e)[0], atol=2e-14)
    e[1] = e[0]
    moment, _, flags = orbital_moments(v, e)
    np.testing.assert_array_equal(flags, [True, True, False, False, False])
    assert np.isfinite(moment).all()


def test_raw_dirac_same_k_current_is_exactly_pauli_kinetic_velocity():
    rng = np.random.default_rng(95)
    psi = rng.normal(size=(7,2,23)) + 1j*rng.normal(size=(7,2,23))
    g = rng.integers(-3,4,size=(23,3))
    k = np.array([0.125,0.25,-0.125])
    b = np.array([[1.1,0.1,0], [0.2,1.3,0.1], [0,0.2,0.9]])
    lifted = lift_to_4spinor(jnp.asarray(psi[None]), jnp.asarray(g[None]),
                            jnp.asarray(k[None]), jnp.asarray(b))[0]
    action = apply_dirac_velocity_to_ket(lifted)
    current = jnp.einsum('msg,ansg->amn', lifted.conj(), action)
    expected = momentum_matrix_k(jnp.asarray(psi), jnp.asarray(g),
                                  jnp.asarray(k), jnp.asarray(b))
    np.testing.assert_allclose(current, expected, atol=5e-12, rtol=3e-14)
