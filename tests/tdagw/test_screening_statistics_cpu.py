"""Passive Gaussian screening statistics references, not a production fix.

The tests distinguish correlated Gibbs, isolated adiabatic, and bare-bath
preparations. They do not define a general coherent/gainful snapshot closure.
Energies use hbar=k_B=1; all matrices are tiny, unsharded host arrays.
References: Kubo, Rep. Prog. Phys. 29, 255 (1966), sections 6-7,
doi:10.1088/0034-4885/29/1/306; Martinez-Tibaduiza et al., arXiv:2010.08081,
section II (instantaneous versus initial oscillator basis).
"""

import numpy as np
import pytest
import scipy.integrate as sint
import scipy.linalg as sla

from tdagw import keldysh_rational as kr


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_correlated_thermal_rpa_dyson_covariance_and_kms(channels, temperature):
    """Independent oscillator diagonalization and rational feedback agree."""
    delta = np.array([1.0, 1.7])[:channels]
    g = np.array([0.5, 0.35])[:channels]
    bare = np.array([[1.0, 0.35], [0.35, 0.85]])[:channels, :channels]
    coupling = np.diag(np.sqrt(2 * g * delta))
    potential = np.diag(delta**2) + coupling.T @ bare @ coupling
    omega2, modes = sla.eigh(potential)
    omega = np.sqrt(omega2)
    n = np.zeros(channels) if temperature == 0 else 1 / np.expm1(omega / temperature)
    if channels == 2:
        assert np.linalg.norm(potential @ np.diag(delta) - np.diag(delta) @ potential) > 0.1

    # Gibbs covariance of H=(p.p+q.V.q)/2, in original transition coordinates.
    qq = (modes * ((n + 0.5) / omega)) @ modes.T
    pp = (modes * ((n + 0.5) * omega)) @ modes.T
    sigma = sla.block_diag(qq, pp)
    zero, identity = np.zeros_like(bare), np.eye(channels)
    generator = np.block([[zero, identity], [-potential, zero]])
    symplectic = np.block([[zero, identity], [-identity, zero]])
    np.testing.assert_allclose(generator @ sigma + sigma @ generator.T, 0, atol=2e-14)
    assert sla.eigvalsh(sigma + 0.5j * symplectic).min() > -2e-14
    if temperature == 0:
        # Pure Gaussian vacuum, generally correlated in bare coordinates.
        np.testing.assert_allclose(sigma @ symplectic @ sigma, symplectic / 4, atol=2e-14)

    bare_residues = np.array([np.diag(np.eye(channels)[j] * g[j]) for j in range(channels)])
    pret = kr.pole_residue_realization(
        np.concatenate((delta, -delta)), np.concatenate((bare_residues, -bare_residues)))
    screened = kr.screen_retarded(pret, bare)
    output = bare @ coupling @ modes
    residues = np.einsum("aj,bj,j->jab", output, output, 1 / (2 * omega))
    poles = np.concatenate((omega, -omega))
    wr = kr.pole_residue_realization(poles, np.concatenate((residues, -residues)))
    wl = kr.pole_residue_realization(
        poles, np.concatenate((n[:, None, None] * residues, (n + 1)[:, None, None] * residues)))
    wg = kr.pole_residue_realization(
        poles, np.concatenate(((n + 1)[:, None, None] * residues, n[:, None, None] * residues)))
    for z in (-0.4 + 0.7j, 1.2 + 0.35j, 2.1 + 0.8j):
        direct = bare + bare @ coupling @ sla.solve(z*z*identity - potential, coupling.T @ bare)
        np.testing.assert_allclose(kr.evaluate(screened, z), direct, atol=3e-14)
        np.testing.assert_allclose(bare + kr.evaluate(wr, z), direct, atol=3e-14)
        np.testing.assert_allclose(kr.evaluate(wg, z) - kr.evaluate(wl, z), kr.evaluate(wr, z), atol=3e-14)
    # Covariance and summed lesser weights are independent constructions.
    np.testing.assert_allclose(bare @ coupling @ qq @ coupling.T @ bare,
                               np.einsum("j,jab->ab", 2*n + 1, residues), atol=3e-14)
    for residue in residues:
        assert sla.eigvalsh(residue).min() > -2e-14
    expected = np.zeros(channels) if temperature == 0 else np.exp(-omega / temperature)
    np.testing.assert_allclose(n / (n + 1), expected, atol=2e-15)


@pytest.mark.parametrize("gamma", [0.1, 0.001])
@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_frequency_dependent_quantum_kms_noise_screens_to_fdt(gamma, temperature):
    """Colored equilibrium noise obeys Dyson FDT at every declared width.

    Fixed bare Bose residue weights broadened into Lorentzians do not. The
    exact finite-T colored spectrum is generally not a finite rational bank.
    """
    frequency = np.array([-2.0, -1.4, -0.4, 0.4, 1.4, 2.0])
    delta, g = 1.0, 0.5
    pret = g * (1 / (frequency-delta+1j*gamma) - 1 / (frequency+delta+1j*gamma))
    wr = 1 / (1 - pret)
    ap, aw = -2 * pret.imag, -2 * wr.imag  # i(PR-PA), i(WR-WA).
    n = (np.where(frequency < 0, -1.0, 0.0) if temperature == 0
         else 1 / np.expm1(frequency / temperature))
    lesser = abs(wr)**2 * n * ap
    greater = abs(wr)**2 * (n + 1) * ap
    assert lesser.min() >= 0 and greater.min() >= 0
    np.testing.assert_allclose(lesser, n * aw, atol=2e-13)
    np.testing.assert_allclose(greater - lesser, aw, atol=2e-13)
    if temperature == 0:
        assert np.all(lesser[frequency > 0] == 0)
    else:
        np.testing.assert_allclose(lesser / greater, np.exp(-frequency / temperature), atol=2e-13)
    nb = 0.0 if temperature == 0 else 1 / np.expm1(delta / temperature)
    broadened_bare = 2*gamma*g*((nb+1)/((frequency+delta)**2+gamma**2)
                              + nb/((frequency-delta)**2+gamma**2))
    wrong_lesser = abs(wr)**2 * broadened_bare
    relative_error = np.linalg.norm(wrong_lesser-lesser) / np.linalg.norm(lesser)
    assert relative_error > 0.01


def test_bare_white_noise_lyapunov_equals_quench_not_correlated_ground_state():
    """Identify the old finite-lag residue through independent state evolution."""
    delta, g, gamma = 1.0, 0.5, 1e-6
    omega = np.sqrt(delta**2 + 2*g*delta)
    initial = np.diag([1/(2*delta), delta/2])
    generator = np.array([[0.0, 1.0], [-omega**2, 0.0]])
    drift = generator - gamma*np.eye(2)
    sigma_white = sla.solve_continuous_lyapunov(drift, -2*gamma*initial)
    # Time averaging the unitary sudden-quench covariance is independent of
    # both the Lyapunov solve and the closed occupation expression below.
    evolved = []
    for t in np.arange(32)*np.pi/(32*omega):
        evolution = sla.expm(generator*t)
        evolved.append(evolution @ initial @ evolution.T)
    dephased = np.mean(evolved, axis=0)
    np.testing.assert_allclose(sigma_white, dephased, atol=2e-7)
    nwhite = (sigma_white[1, 1]/omega + omega*sigma_white[0, 0])/2 - 0.5
    nquench = (omega-delta)**2/(4*delta*omega)
    assert nwhite == pytest.approx(nquench, abs=2e-15)
    retarded_residue = g*delta/omega
    wrong_lesser = g*(omega-delta)**2/(4*omega**2)
    assert retarded_residue*nwhite == pytest.approx(wrong_lesser, abs=2e-15)
    assert wrong_lesser == pytest.approx(0.010723304703363122)
    # An independently declared correlated covariance changes the noise input,
    # and the same Lyapunov algorithm then recovers the correct vacuum.
    thermal = np.diag([1/(2*omega), omega/2])
    diffusion = -(drift @ thermal + thermal @ drift.T)
    recovered = sla.solve_continuous_lyapunov(drift, -diffusion)
    np.testing.assert_allclose(recovered, thermal, atol=3e-10)


def test_slow_isolated_switch_preserves_occupations_not_final_gibbs_temperature():
    """Numerical Gaussian evolution creates correlations, without thermalizing."""
    delta, omega, temperature, duration = 1.0, np.sqrt(2.0), 0.8, 80.0
    nbare = 1 / np.expm1(delta / temperature)
    ngibbs = 1 / np.expm1(omega / temperature)
    initial = (2*nbare+1)*np.diag([1/(2*delta), delta/2])

    def rhs(t, flat):
        switch = 0.5*(1-np.cos(np.pi*t/duration))
        generator = np.array([[0.0, 1.0], [-(delta**2 + switch), 0.0]])
        sigma = flat.reshape(2, 2)
        return (generator @ sigma + sigma @ generator.T).ravel()

    solution = sint.solve_ivp(rhs, (0.0, duration), initial.ravel(), rtol=2e-10, atol=2e-12)
    assert solution.success
    final = solution.y[:, -1].reshape(2, 2)
    nfinal = (final[1, 1]/omega + omega*final[0, 0])/2 - 0.5
    assert nfinal == pytest.approx(nbare, abs=2e-8)
    assert abs(nfinal - ngibbs) > 0.19
    assert np.linalg.det(final) == pytest.approx(np.linalg.det(initial), abs=2e-8)


def test_half_filled_thermal_continuation_keeps_noise_when_response_vanishes():
    """Retain d/beta as beta->0; setting d=0 first loses finite noise."""
    delta, g = 1.0, 0.5
    for beta in (0.1, 1e-3, 1e-6):
        difference = np.tanh(beta*delta/4)  # levels +/-Delta/2, mu=0.
        fupper = 1/(1+np.exp(beta*delta/2))
        np.testing.assert_allclose(difference/np.expm1(beta*delta), fupper**2, atol=2e-16)
        omega = np.sqrt(delta**2 + 2*g*difference*delta)
        retarded_residue = g*difference*delta/omega
        lesser_residue = retarded_residue/np.expm1(beta*omega)
    assert retarded_residue < 2e-7
    assert lesser_residue == pytest.approx(g/4, abs=2e-7)
    # Continuous finite-width colored-noise limit, including off resonance.
    frequency, gamma = np.array([-1.3, -0.4, 0.4, 1.3]), 0.1
    base = g*(1/(frequency-delta+1j*gamma) - 1/(frequency+delta+1j*gamma))
    finite_noise = difference*(-2*base.imag)/np.expm1(beta*frequency)
    denominator = ((frequency-delta)**2+gamma**2)*((frequency+delta)**2+gamma**2)
    limiting_noise = 2*gamma*g*delta**2/denominator
    np.testing.assert_allclose(finite_noise, limiting_noise, rtol=7e-7)
    assert limiting_noise.min() > 0
