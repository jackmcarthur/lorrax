"""Small independent time-integral gates for a general Keldysh Sigma bank."""
import numpy as np
from scipy.integrate import quad_vec

from tdagw import keldysh_rational as kr
from tdagw import snapshot_collision as sc


def _dag(a):
    return np.conj(np.swapaxes(a, -1, -2))


def test_column_operator_matches_direct_positive_time_integral(monkeypatch):
    rng = np.random.default_rng(713)
    nk, na, ni, rank = 2, 2, 3, 2
    vertices = rng.normal(size=(nk, na, ni, rank)) + 1j * rng.normal(size=(nk, na, ni, rank))
    density = np.empty((nk, ni, ni), complex)
    for k in range(nk):
        u, _ = np.linalg.qr(rng.normal(size=(ni, ni)) + 1j * rng.normal(size=(ni, ni)))
        density[k] = u @ np.diag([0.15, 0.5, 0.85]) @ u.conj().T
    energy = np.array([[-0.6, 0.1, 0.7], [-0.5, 0.2, 0.9]])
    external = np.array([[-0.4, 0.8], [-0.3, 1.0]])
    poles = np.array([-0.8 - 0.3j, 0.65 - 0.4j])
    less_residue = 0.07 * (rng.normal(size=(2, rank, rank))
                            + 1j * rng.normal(size=(2, rank, rank)))
    ret_residue = 0.1 * (rng.normal(size=(2, rank, rank))
                          + 1j * rng.normal(size=(2, rank, rank)))
    less = kr.pole_residue_realization(poles, less_residue)
    ret = kr.pole_residue_realization(poles, ret_residue)
    calls = []
    original = sc.rational_factors

    def count_factor(system, **options):
        calls.append(system[3].shape)
        return original(system, **options)

    monkeypatch.setattr(sc, "rational_factors", count_factor)
    eta = 0.27
    got = sc.q_sigma_operators(vertices, energy, density, external,
                               dict(retarded=ret, lesser=less), eta=eta)
    assert calls == [(2 * rank, rank)]  # One common response/noise bank.

    def integrand(t):
        wl = -1j * np.einsum("p,prs->rs", np.exp(-1j * poles * t), less_residue)
        wg = -1j * np.einsum("p,prs->rs", np.exp(-1j * poles * t),
                              less_residue + ret_residue)
        output = np.empty((2, nk, na, na), complex)
        for k in range(nk):
            phase = np.diag(np.exp(-1j * energy[k] * t))
            gl, gg = 1j * density[k] @ phase, -1j * (np.eye(ni) - density[k]) @ phase
            for branch, (g, w) in enumerate(((gl, wl), (gg, wg))):
                sigma_t = 1j * np.einsum("amr,mn,rs,bns->ab", vertices[k], g, w,
                                         vertices[k].conj(), optimize=True)
                output[branch, k] = (sigma_t * np.exp(1j * external[k] * t)[None]
                                     * np.exp(-eta * t))
        return output

    sigma, error = quad_vec(integrand, 0.0, np.inf, epsabs=1e-10, epsrel=1e-10)
    assert error < 1e-8
    ain, aout = -1j * sigma[0], 1j * sigma[1]
    np.testing.assert_allclose(got["ain"], ain, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(got["aout"], aout, rtol=2e-10, atol=2e-10)
    m = -1j * (ain + aout)
    np.testing.assert_allclose(got["m_h"], 0.5 * (m + _dag(m)), atol=2e-10)
    np.testing.assert_allclose(got["gamma"], 1j * (m - _dag(m)), atol=2e-10)
    np.testing.assert_allclose(got["injection"], ain + _dag(ain), atol=2e-10)


def test_active_destinations_leave_frozen_terms_coherent():
    rng = np.random.default_rng(117)
    vertex = rng.normal(size=(1, 2, 3, 1)) + 1j * rng.normal(size=(1, 2, 3, 1))
    density = np.diag([1.0, 0.7, 0.3])[None].astype(complex)
    energy, ext = np.array([[-1.2, -0.2, 0.6]]), np.array([[-0.2, 0.6]])
    ret = kr.pole_residue_realization([0.8 - 0.1j], np.array([[[0.2]]]))
    less = kr.pole_residue_realization([-0.8 - 0.1j], np.array([[[0.1]]]))
    bank = dict(retarded=ret, lesser=less)
    full = sc.q_sigma_operators(vertex, energy, density, ext, bank, eta=0.1)
    active = sc.q_sigma_operators(vertex, energy, density, ext, bank,
                                  eta=0.1, active=slice(1, 3))
    direct = sc.q_sigma_operators(vertex[:, :, 1:], energy[:, 1:], density[:, 1:, 1:],
                                  ext, bank, eta=0.1)
    np.testing.assert_allclose(active["m_h"], full["m_h"])
    np.testing.assert_allclose(active["gamma"], direct["gamma"])
    np.testing.assert_allclose(active["injection"], direct["injection"])
    assert np.max(np.abs(full["gamma"] - active["gamma"])) > 1e-3


def test_general_factors_reconstruct_signed_nonnormal_bank():
    A = np.array([[-0.4 - 0.2j, 0.35], [0.0, 0.7 - 0.3j]])
    B = np.array([[1.0, 0.2j], [0.3, -0.4]])
    C = np.array([[0.1j, 0.7], [-0.2, 0.4j]])
    bank = (A, B, C, np.zeros((2, 2)))
    factors = sc.rational_factors(bank)
    for z in [-0.6 + 0.2j, 0.5 + 0.4j]:
        got = (factors["left"] / (z - factors["poles"])[None]) @ factors["right"]
        np.testing.assert_allclose(got, kr.evaluate(bank, z), rtol=1e-12, atol=1e-12)


def test_joint_union_keeps_noise_only_poles_and_exact_greater_sum():
    ret = kr.pole_residue_realization([0.6 - 0.2j],
        np.array([[[0.2, 0.1j], [-0.05j, -0.1]]]))
    less = kr.pole_residue_realization([-0.3 - 0.1j],
        np.array([[[0.04j, -0.02], [0.03, 0.07j]]]))
    factors = sc.joint_rational_factors(ret, less)
    assert np.min(abs(factors["poles"] - (-0.3 - 0.1j))) < 1e-12
    for z in [-0.4 + 0.2j, 0.2 + 0.4j]:
        denominator = (z - factors["poles"])[None]
        rc = (factors["left_retarded"] / denominator) @ factors["right"]
        rl = (factors["left_lesser"] / denominator) @ factors["right"]
        rg = (factors["left_greater"] / denominator) @ factors["right"]
        np.testing.assert_allclose(rc, kr.evaluate(ret, z), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(rl, kr.evaluate(less, z), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(rg, rc + rl, rtol=1e-14, atol=1e-14)
