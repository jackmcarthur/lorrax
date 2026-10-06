"""Independent charge-kernel oracles and the neutral-residual counterexample."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from isdf.augmentation import (
    compensation_basis,
    local_basis_fourier,
    mixed_coulomb_tile,
    onsite_coulomb_tile,
    radial_coulomb_matrix,
    radial_coulomb_potential,
)


def radial_rule(n, R):
    x, w = leggauss(n)
    return (x + 1) * R / 2, w * R / 2


def test_uniform_sphere_against_closed_form_and_refinement():
    # Q=1 sphere: (rho|v|rho)=6/(5R), with NO classical self-energy half.
    R = 1.7
    errors = []
    for n in (128, 256, 512):
        r, w = radial_rule(n, R)
        rho = np.full((1, 1, n), 3 / (np.sqrt(4 * np.pi) * R**3))
        got = radial_coulomb_matrix(rho, r, w, [0])[0, 0]
        errors.append(abs(got - 6 / (5 * R)))
        potential = radial_coulomb_potential(rho[0, 0], r, w, 0)
        analytic = np.sqrt(4 * np.pi) * (3 * R**2 - r**2) / (2 * R**3)
        np.testing.assert_allclose(potential, analytic, atol=2e-4, rtol=0)
    assert errors[1] < errors[0] / 3
    assert errors[2] < errors[1] / 3
    assert errors[-1] < 5e-6


def test_compensation_has_all_retained_complex_moments():
    r, w = radial_rule(96, 2.2)
    rng = np.random.default_rng(247)
    rho = rng.normal(size=(3, 4, len(r))) + 1j * rng.normal(size=(3, 4, len(r)))
    degrees = np.array([0, 1, 1, 2])
    comp, moments = compensation_basis(rho, r, w, degrees, support_radius=2.2)
    for j, l in enumerate(degrees):
        residual = (rho[:, j] - comp[:, j]) @ (w * r ** (l + 2))
        np.testing.assert_allclose(residual, 0, atol=5e-14)
        np.testing.assert_allclose(comp[:, j] @ (w * r ** (l + 2)), moments[:, j])
    K = radial_coulomb_matrix(rho, r, w, degrees)
    np.testing.assert_allclose(K, K.conj().T, rtol=0, atol=5e-12)
    assert np.linalg.eigvalsh(K).min() > 0


def test_fourier_bessel_matches_cartesian_gaussian_and_its_dipole():
    # Independent closed forms fix normalization, complex Y_lm orientation,
    # origin translation, K=0 and the -i Fourier sign at once.
    r, w = radial_rule(160, 9)
    a = 0.83
    k = np.array([[0, 0, 0], [0.4, -0.3, 0.6], [-0.7, 0.5, -0.2]])
    center = np.array([0.6, -0.2, 1.3])
    base = (np.pi / a) ** 1.5 * np.exp(-np.sum(k*k, axis=1) / (4*a))
    translation = np.exp(-1j * (k @ center))
    rho0 = (np.sqrt(4*np.pi) * np.exp(-a*r*r))[None, None, :]
    actual0 = local_basis_fourier(rho0, r, w, [(0, 0)], k, center_cart=center)[0]
    np.testing.assert_allclose(actual0, base * translation, atol=1e-12, rtol=1e-12)
    rhop = (-np.sqrt(8*np.pi/3) * r * np.exp(-a*r*r))[None, None, :]
    actualp = local_basis_fourier(rhop, r, w, [(1, 1)], k, center_cart=center)[0]
    expectedp = -1j * (k[:, 0] + 1j*k[:, 1]) / (2*a) * base * translation
    np.testing.assert_allclose(actualp, expectedp, atol=1e-12, rtol=1e-12)


def test_zero_monopole_does_not_eliminate_smooth_cross_term():
    # h00=1-5r²/3 has Q00=0, but cos(Kx) penetrates its sphere.
    # Compare the integral of its LOCAL potential with Fourier 4pi*h(K)/K².
    from scipy.special import spherical_jn

    r, w = radial_rule(512, 1)
    h = 1 - 5*r*r/3
    assert abs(h @ (w*r*r)) < 2e-14
    K = 2.3
    potential = radial_coulomb_potential(h, r, w, 0)
    direct = np.sqrt(4*np.pi) * (potential @ (w*r*r*spherical_jn(0, K*r)))
    hG = local_basis_fourier(h[None, None], r, w, [(0, 0)],
                            [[K, 0, 0]], center_cart=[0, 0, 0])[0, 0]
    fourier = 4*np.pi/K**2 * hG
    assert abs(direct) > 0.02
    np.testing.assert_allclose(direct, fourier, atol=8e-6, rtol=0)
    # An implementation omitting the residual term would return zero here.
    s = np.ones((1, 1, 1), complex)
    d = np.array([[[hG]]])
    result = np.asarray(mixed_coulomb_tile(s, d, np.zeros_like(s),
                                         np.array([[4*np.pi/K**2]])))
    reference = 4*np.pi/K**2 + 2*fourier.real
    np.testing.assert_allclose(result[0, 0, 0], reference, atol=1e-12)


def test_mixed_tile_and_onsite_are_complete_for_a_planted_local_basis():
    rng = np.random.default_rng(19)
    s = rng.normal(size=(2, 3, 5)) + 1j*rng.normal(size=(2, 3, 5))
    coeff = rng.normal(size=(2, 3, 2)) + 1j*rng.normal(size=(2, 3, 2))
    delta_ft = rng.normal(size=(2, 5)) + 1j*rng.normal(size=(2, 5))
    comp_ft = rng.normal(size=(2, 5)) + 1j*rng.normal(size=(2, 5))
    v = np.abs(rng.normal(size=(2, 5)))
    d = np.einsum('qmi,ig->qmg', coeff, delta_ft)
    g = np.einsum('qmi,ig->qmg', coeff, comp_ft)
    kg = np.einsum('ig,jg->ij', comp_ft.conj(), comp_ft)
    kd = np.array([[3.1, 0.2j], [-0.2j, 2.6]])
    got = np.asarray(mixed_coulomb_tile(s, d, g, v))
    got = got + np.asarray(onsite_coulomb_tile(coeff, kd, kg))
    ref = np.empty_like(got)
    # Pair-by-pair independent oracle is confined to this tiny test.
    for q in range(2):
        for m in range(3):
            for n in range(3):
                ref[q, m, n] = sum(v[q, G] * (
                    s[q, m, G].conj()*s[q, n, G]
                    + s[q, m, G].conj()*d[q, n, G]
                    + d[q, m, G].conj()*s[q, n, G]
                    + g[q, m, G].conj()*g[q, n, G]) for G in range(5))
                ref[q, m, n] += coeff[q, m].conj() @ (kd-kg) @ coeff[q, n]
    np.testing.assert_allclose(got, ref, atol=2e-13, rtol=1e-13)
    np.testing.assert_allclose(got, got.conj().transpose(0, 2, 1), atol=2e-13)


@pytest.mark.parametrize('r,w,l', [([0, 1], [1, 1], 0),
                                 ([1, 0.5], [1, 1], 0),
                                 ([1, 2], [1, -1], 0),
                                 ([1, 2], [1, 1], -1)])
def test_invalid_radial_data_refuses(r, w, l):
    with pytest.raises(ValueError):
        radial_coulomb_potential([1, 1], r, w, l)
