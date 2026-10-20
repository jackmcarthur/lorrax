"""Retarded-only reference branches and full-density screening checks."""
import numpy as np
import pytest

from tdagw import keldysh_rational as kr
from tdagw import snapshot_screening as ss


def test_reference_branches_match_equilibrium_pole_formula_and_keep_signed_gain():
    poles = np.array([-.8 - .04j, 1.2 - .03j])
    residues = np.array([[-.3], [-.2]]).reshape(2, 1, 1)
    model = kr.retarded_reference_branches(kr.pole_residue_realization(poles, residues))
    assert model["signed_gain_pole_count"] == 1
    for z in (.2 + .3j, -.4 + .7j):
        np.testing.assert_allclose(kr.evaluate(model["lesser"], z), [[.3 / (z - poles[0])]])
        np.testing.assert_allclose(kr.evaluate(model["greater"], z), [[-.2 / (z - poles[1])]])
        np.testing.assert_allclose(kr.evaluate(model["greater"], z) - kr.evaluate(model["lesser"], z),
                                   kr.evaluate(model["correlation"], z))


def test_reference_midpoint_zero_frequency_split_and_causality_gate():
    system = kr.pole_residue_realization([-.1j], np.array([[[.2j]]]))
    model = kr.retarded_reference_branches(system)
    assert model["zero_frequency_count"] == 1
    np.testing.assert_allclose(kr.evaluate(model["lesser"], .2j), -.5 * kr.evaluate(system, .2j))
    with pytest.raises(ValueError, match="snapshot_reference_causality"):
        kr.retarded_reference_branches(kr.pole_residue_realization([.4 + .02j], np.ones((1, 1, 1))))


def test_growing_causal_reference_uses_convergent_sigma_laplace_height():
    from scipy.integrate import quad_vec
    pole = .4 + .02j
    residue = np.array([[.3, .1j], [-.1j, -.2]])
    system = kr.pole_residue_realization([pole], residue[None])
    model = kr.retarded_reference_branches(system, evaluation_height=.1)
    assert model["maximum_pole_growth"] == pytest.approx(.02)
    assert model["laplace_margin"] == pytest.approx(.08)
    np.testing.assert_allclose(model["poles"], np.full(model["poles"].shape, pole))
    z = .13 + .1j
    actual, error = quad_vec(lambda t: -1j * residue * np.exp(1j * (z - pole) * t),
                             0, np.inf, epsabs=1e-11, epsrel=1e-11)
    assert error < 1e-9
    np.testing.assert_allclose(actual, kr.evaluate(model["correlation"], z), atol=1e-10)
    for height in (0, .019, .02):
        with pytest.raises(ValueError, match="snapshot_reference_causality"):
            kr.retarded_reference_branches(system, evaluation_height=height)


def test_retarded_reference_checks_supports_above_feedback_growth():
    def sampler(z):
        p = (.04j / (np.asarray(z) + .01j))[:, None, None]
        return np.stack((np.zeros_like(p), p))
    fit = np.linspace(-1, 1, 20) + .2j
    checks = np.linspace(-.9, .9, 19) + .3j
    model = ss.retarded_reference_snapshot(sampler, fit, checks, collision_eta=.1,
                                           fit_tolerance=1e-9)
    assert model["receipt"]["reference_maximum_pole_growth_ry"] == pytest.approx(.03)
    assert model["receipt"]["reference_laplace_margin_ry"] == pytest.approx(.07)
    with pytest.raises(ValueError, match="snapshot_reference_support_height"):
        ss.retarded_reference_snapshot(sampler, fit.real + .02j,
            checks.real + .025j, collision_eta=.1, fit_tolerance=1e-9)


def test_retarded_reference_discards_noise_only_data_by_declared_ansatz():
    def sampler(z):
        noise = (0.2 / (np.asarray(z) + .3 + .04j))[:, None, None]
        return np.stack((noise, noise))
    fit = np.linspace(-1, 1, 20) + .3j
    model = ss.retarded_reference_snapshot(sampler, fit, fit + .2j)
    for z in fit:
        np.testing.assert_allclose(kr.evaluate(model["correlation"], z), 0)
        np.testing.assert_allclose(kr.evaluate(model["lesser"], z), 0)
    assert model["receipt"]["noise_check_kind"] == "not_applicable_retarded_reference"


def test_full_density_retarded_fit_keeps_coherence_and_crossed_partner():
    rng = np.random.default_rng(9)
    energy = np.array([[-.4, .6], [-.3, .7], [-.5, .5]])
    rho = np.array([[[.8, .03j], [-.03j, .2]], [[.75, .02], [.02, .25]],
                    [[.85, -.01j], [.01j, .15]]])
    v = .1 * (rng.normal(size=(3, 2, 2, 1)) + 1j * rng.normal(size=(3, 2, 2, 1)))
    partner = np.array([2, 0, 1])
    fit = np.linspace(-2, 2, 48) + .3j
    check = np.linspace(-1.9, 1.9, 19) + .43j
    def sampler(z):
        return ss.polarization_from_vertices(v, energy, rho, partner, z, adiabatic_ry=.1)
    model = ss.retarded_reference_snapshot(sampler, fit, check, max_order=24, fit_tolerance=1e-7)
    vm = np.empty_like(v)
    for k in range(3):
        vm[partner[k]] = v[k].swapaxes(0, 1).conj()
    def negative_sampler(z):
        return ss.polarization_from_vertices(vm, energy, rho, np.argsort(partner), z, adiabatic_ry=.1)
    negative = ss.retarded_reference_snapshot(negative_sampler, fit, check,
        representative=model, fit_tolerance=1e-7)
    for z in check:
        np.testing.assert_allclose(kr.evaluate(negative["lesser"], z),
            kr.evaluate(ss.mirror(kr.add(model["correlation"], model["lesser"])), z), atol=1e-10)
    diagonal = rho * np.eye(2)[None]
    no_coherence = ss.polarization_from_vertices(v, energy, diagonal, partner, check, adiabatic_ry=.1)
    full = sampler(check)
    assert np.linalg.norm((full[1] - full[0]) - (no_coherence[1] - no_coherence[0])) > 1e-5
