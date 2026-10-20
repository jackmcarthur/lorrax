"""Scientific limits of stationary screening statistics, using tiny CPU matrices.

These characterize approximations; they do not certify collision positivity or
implement a physical statistical completion of the snapshot screening model.
"""

import numpy as np


def test_stationary_endpoint_spectrum_can_be_indefinite_for_physical_density():
    # C(t>0)=rho exp(-i H t-gamma t), C(-t)=C(t)^dagger.  Its full
    # spectrum is i[rho R-R^dagger rho], R=(omega-H+i gamma)^-1.
    delta, omega = 1.0, 0.5
    hamiltonian = np.diag([0.0, delta])
    for coherence in (0.25, 0.5):
        rho = np.array([[0.5, coherence], [coherence, 0.5]])
        assert np.linalg.eigvalsh(rho).min() >= 0.0
        assert np.linalg.eigvalsh(np.eye(2) - rho).min() >= 0.0
        for gamma in (0.01, 0.05, 0.5, 2.0):
            resolvent = np.linalg.inv((omega + 1j * gamma) * np.eye(2) - hamiltonian)
            spectrum = 1j * (rho @ resolvent - resolvent.conj().T @ rho)
            numerator = gamma**2 - coherence**2 * (4 * gamma**2 + delta**2)
            denominator = ((omega**2 + gamma**2)
                           * ((omega - delta)**2 + gamma**2))
            np.testing.assert_allclose(np.linalg.det(spectrum), numerator / denominator,
                                       rtol=2e-13, atol=2e-13)
            # PSD iff |c| <= gamma/sqrt(4 gamma^2+Delta^2).  A pure
            # coherent state c=1/2 fails for every finite positive gamma.
            assert (np.linalg.eigvalsh(spectrum).min() >= 0.0) == (numerator >= 0.0)
    gamma, coherence = 0.05, 0.5
    bound = gamma / np.sqrt(4 * gamma**2 + delta**2)
    assert coherence > bound


def test_exact_frozen_two_time_kernel_preserves_coherences_and_positivity():
    # Equal-time rho is prescribed at T=0; all internal times are <=T.
    # The exact noninteracting kernel C(t,s)=U(t,T) rho U(s,T)^dagger
    # is a block Gram matrix, including noncommuting coherences.
    energies = np.array([0.0, 1.0])
    rho = 0.5 * np.ones((2, 2))
    times = np.array([-1.4, -0.7, 0.0])
    unitaries = [np.diag(np.exp(-1j * energies * t)) for t in times]
    exact = np.block([[ut @ rho @ us.conj().T for us in unitaries]
                      for ut in unitaries])
    np.testing.assert_allclose(exact, exact.conj().T, atol=2e-15)
    assert np.linalg.eigvalsh(exact).min() > -2e-15
    assert np.linalg.norm(exact[:2, :2] - rho) > 0.5
    np.testing.assert_allclose(exact[-2:, -2:], rho, atol=2e-15)

    stationary_blocks = []
    for t in times:
        row = []
        for s in times:
            lag = t - s
            positive = rho @ np.diag(np.exp(-1j * energies * abs(lag)))
            row.append(positive if lag >= 0 else positive.conj().T)
        stationary_blocks.append(row)
    stationary = np.block(stationary_blocks)
    # The upper-endpoint row agrees exactly, but declaring every other
    # endpoint to carry the same rho destroys the positive block kernel.
    np.testing.assert_allclose(stationary[-2:], exact[-2:], atol=2e-15)
    assert np.linalg.eigvalsh(stationary).min() < -0.1

    # Wick charge fluctuations also form a PSD two-time kernel.  Their
    # variances vary with central time and cannot be held stationary.
    vertex = np.array([[1.0, 1.0], [1.0, 0.0]])
    vertices = [u.conj().T @ vertex @ u for u in unitaries]
    charge = np.array([[np.trace(rho @ vt @ (np.eye(2) - rho) @ vs)
                        for vs in vertices] for vt in vertices])
    assert np.linalg.eigvalsh(charge).min() > -2e-15
    assert np.ptp(charge.diagonal().real) > 0.5


def test_reference_kms_plus_delta_bubble_fails_for_weak_diagonal_excitation():
    # Candidate completion: N=N_eq,KMS+(P<_rho-P<_eq)_full,
    # W<_full=W^R N W^A. At T=0 and omega>0 the KMS reference lesser
    # source is zero. A genuine diagonal density diag(1-p,p) has bare
    # lesser transition weights g(1-p)^2 and g p^2 at -Delta,+Delta.
    delta, coupling, gamma, omega = 1.0, 0.5, 0.05, 0.5
    dminus = (omega - delta)**2 + gamma**2
    dplus = (omega + delta)**2 + gamma**2
    linear_source = -4 * coupling * gamma / dplus
    omega_eq = np.sqrt(delta**2 + 2 * coupling * delta)
    linear_residue = -coupling * (omega_eq - delta)**2 / (2 * omega_eq**2)
    for population in (1e-2, 1e-4, 1e-6):
        rho = np.diag([1 - population, population])
        assert np.linalg.eigvalsh(rho).min() > 0.0
        source = 2 * coupling * gamma * (
            (-2 * population + population**2) / dplus + population**2 / dminus)
        response = coupling * (1 - 2 * population) * (
            1 / (omega - delta + 1j * gamma) - 1 / (omega + delta + 1j * gamma))
        screened = source / abs(1 - response)**2
        assert source < 0 and screened < 0
        # The violation is first order in an arbitrarily weak excitation,
        # rather than an untested high-population or coherence effect.
        np.testing.assert_allclose(source / population,
                                   linear_source + 2 * coupling * gamma * population
                                   * (1 / dplus + 1 / dminus), rtol=2e-15)

        collective = np.sqrt(delta**2 + 2 * coupling * delta * (1 - 2 * population))
        # Near the collective pole the negative source is amplified.  Its
        # signed spectral residue remains negative as gamma -> 0.
        residue_limit = coupling / (4 * collective**2) * (
            (-2 * population + population**2) * (collective - delta)**2
            + population**2 * (collective + delta)**2)
        assert residue_limit < 0
    np.testing.assert_allclose(source / population, linear_source, rtol=1e-5)
    np.testing.assert_allclose(residue_limit / population, linear_residue, rtol=2e-5)
