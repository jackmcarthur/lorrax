"""Analytic Fourier/Pauli oracles for the atom-local normalized RKB carrier."""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import gamma

from common.bispinor_init import HALFALPHA
from psp.augmentation_spinors import (
    build_normalized_radial_cache,
    evaluate_normalized_delta,
    normalized_delta_fourier,
    reconstruction_overlap,
    spinor_function_labels,
    spinor_spherical_harmonic,
)


def rule(n, endpoint):
    x, w = leggauss(n)
    return (x+1)*endpoint/2, w*endpoint/2


@pytest.fixture(scope="module")
def gaussian_basis():
    a = 0.73
    l = np.array([0, 1, 1, 2, 2])
    k = np.array([-1, 1, -2, 2, -3])
    r, wr = rule(180, 9)
    radial = r[:, None]**l * np.exp(-a*r*r)[:, None]
    K, wk = rule(180, 16)
    cache = build_normalized_radial_cache(radial, r, wr, l, k, K, wk,
                                          np.linspace(0, 10, 1001))
    return a, l, k, r, wr, radial, cache


def test_spinor_angular_phase_and_channel_orthogonality():
    z, wz = leggauss(18)
    phi = np.arange(40)*2*np.pi/40
    directions = np.stack(np.broadcast_arrays(
        np.sqrt(1-z*z)[:, None]*np.cos(phi),
        np.sqrt(1-z*z)[:, None]*np.sin(phi), z[:, None]), axis=-1).reshape(-1, 3)
    weights = np.repeat(wz, len(phi))*2*np.pi/len(phi)
    channels = []
    for k in (-1, 1, -2, 2, -3):
        for m2 in range(-2*abs(k)+1, 2*abs(k), 2):
            omega = spinor_spherical_harmonic(k, m2, directions)
            partner = spinor_spherical_harmonic(-k, m2, directions)
            x, y, zc = directions.T
            acted = np.stack((zc*omega[:, 0]+(x-1j*y)*omega[:, 1],
                              (x+1j*y)*omega[:, 0]-zc*omega[:, 1]), axis=-1)
            np.testing.assert_allclose(acted, -partner, atol=2e-15, rtol=0)
            channels.append(omega)
    angular = np.asarray(channels)
    gram = np.einsum('ips,jps,p->ij', angular.conj(), angular, weights)
    # 1440 scalar terms are summed per element; allow their float64 rounding.
    np.testing.assert_allclose(gram, np.eye(len(channels)), atol=2e-14, rtol=0)


def test_gaussian_fourier_normalization_translation_and_isometry(gaussian_basis):
    a, l, k, r, wr, radial, _ = gaussian_basis
    vectors = np.array([[0, 0, 0], [0.8, -0.3, 0.7], [-0.4, 1.2, 0.2], [6, 3, -4]])
    center = np.array([0.5, -0.2, 0.8])
    got = normalized_delta_fourier(radial, r, wr, l, k, vectors, center_cart=center)
    K = np.linalg.norm(vectors, axis=1)
    labels = spinor_function_labels(l, k)
    for row, (i, m2) in enumerate(labels):
        # Closed-form Gaussian Hankel transform, independent of the radial code.
        A = np.sqrt(np.pi)*K**l[i]/(2**(l[i]+2)*a**(l[i]+1.5))*np.exp(-K*K/(4*a))
        omega = spinor_spherical_harmonic(k[i], m2, vectors).T
        source = 4*np.pi*(-1j)**l[i]*omega*A*np.exp(-1j*(vectors @ center))
        x, y, z = vectors.T
        lower = HALFALPHA*np.stack((z*source[0]+(x-1j*y)*source[1],
                                    (x+1j*y)*source[0]-z*source[1]))
        expected = np.concatenate((source, lower))/(np.sqrt(1+HALFALPHA**2*K*K))
        np.testing.assert_allclose(got[row], expected, atol=8e-14, rtol=3e-12)
        np.testing.assert_allclose(np.sum(abs(got[row])**2, axis=0),
                                   np.sum(abs(source)**2, axis=0), atol=2e-12, rtol=2e-14)


def test_inverse_hankel_carrier_has_physical_radial_derivative(gaussian_basis):
    _, _, k, _, _, _, cache = gaussian_basis
    r = cache['radius'][1:]
    expected = 1j*HALFALPHA*(cache['dlarge_R_dr'][1:]
                            +(k+1)*cache['large_R'][1:]/r[:, None])
    np.testing.assert_allclose(cache['small_R'][1:], expected, atol=2e-15, rtol=2e-11)
    # This catches the l -> l_small angular derivative and its phase separately.
    points = np.array([[0.3, -0.2, 0.5], [0.7, 0.4, -0.8], [1.2, -0.3, 0.2]])
    step = 2e-5
    gradients = []
    for axis in range(3):
        offset = np.zeros(3); offset[axis] = step
        gradients.append((evaluate_normalized_delta(cache, points+offset)[:, :2]
                          -evaluate_normalized_delta(cache, points-offset)[:, :2])/(2*step))
    dx, dy, dz = gradients
    sigma_p = -1j*HALFALPHA*np.stack((dz[:, 0]+dx[:, 1]-1j*dy[:, 1],
                                    dx[:, 0]+1j*dy[:, 0]-dz[:, 1]), axis=1)
    actual = evaluate_normalized_delta(cache, points)[:, 2:]
    np.testing.assert_allclose(actual, sigma_p, atol=5e-11, rtol=2e-7)


def test_real_space_parseval_for_every_j_resolved_radial_channel(gaussian_basis):
    a, l, _, _, _, _, cache = gaussian_basis
    from scipy.interpolate import CubicHermiteSpline

    r, w = rule(260, 10)
    large = CubicHermiteSpline(cache['radius'], cache['large_R'], cache['dlarge_R_dr'], axis=0)(r)
    small = CubicHermiteSpline(cache['radius'], cache['small_R'], cache['dsmall_R_dr'], axis=0)(r)
    got = (w*r*r) @ (abs(large)**2+abs(small)**2)
    expected = gamma(l+1.5)/(2*(2*a)**(l+1.5))
    np.testing.assert_allclose(got, expected, atol=4e-10, rtol=2e-9)


def test_lift_commutes_with_reconstruction_sum_and_zero(gaussian_basis):
    _, _, _, r, w, radial, _ = gaussian_basis
    vectors = np.array([[0.3, 0.2, 0.7], [1.4, -0.2, -0.3]])
    # Two radial functions of the same angular channel, arbitrary complex c.
    pair = np.stack((radial[:, 0], radial[:, 0]*(1-0.4*r*r)), axis=1)
    c = np.array([0.7-0.3j, -0.2+0.8j])
    parts = normalized_delta_fourier(pair, r, w, [0, 0], [-1, -1], vectors).reshape(2, 2, 4, 2)
    direct = normalized_delta_fourier((pair @ c)[:, None], r, w, [0], [-1], vectors)
    np.testing.assert_allclose(direct, np.einsum('i,imsg->msg', c, parts), atol=2e-14, rtol=2e-14)
    zero = normalized_delta_fourier(np.zeros((len(r), 1)), r, w, [0], [-1], vectors)
    np.testing.assert_array_equal(zero, 0)


def test_full_overlap_diagnostic_keeps_smooth_residual_cross_terms():
    rng = np.random.default_rng(20)
    smooth = rng.normal(size=(3, 4, 7))+1j*rng.normal(size=(3, 4, 7))
    delta = 0.1*(rng.normal(size=smooth.shape)+1j*rng.normal(size=smooth.shape))
    w = np.arange(1, 8)/20
    got = reconstruction_overlap(smooth, smooth+delta, w)
    expected = ((smooth.conj()*w).reshape(3, -1) @ delta.reshape(3, -1).T
                +(delta.conj()*w).reshape(3, -1) @ smooth.reshape(3, -1).T
                +(delta.conj()*w).reshape(3, -1) @ delta.reshape(3, -1).T)
    np.testing.assert_allclose(got, expected, atol=5e-15, rtol=0)
    np.testing.assert_allclose(got, got.conj().T, atol=3e-15, rtol=0)
    assert np.max(abs(got)) > 0.05


@pytest.mark.parametrize('k,m', [(0, 1), (-1, 0), (-1, 3), (1.2, 1)])
def test_invalid_spinor_quantum_numbers_refuse(k, m):
    with pytest.raises(ValueError):
        spinor_spherical_harmonic(k, m, [[0, 0, 1]])


def test_cache_extrapolation_refuses_instead_of_cutting_tail(gaussian_basis):
    cache = gaussian_basis[-1]
    with pytest.raises(ValueError, match="extend and converge its tail"):
        evaluate_normalized_delta(cache, [[11, 0, 0]])
