"""Independent full-Gram, carrier-commutation and distributed rotation oracles."""
from runtime import initialize_communicator_stack
RUNTIME = initialize_communicator_stack()

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P
from numpy.polynomial.legendre import leggauss

from psp.augmentation_spinors import (
    _lift_cartesian, atomic_pauli_fourier, build_pauli_fourier_cache,
    evaluate_pauli_fourier_cache, spinor_function_labels, spinor_spherical_harmonic,
)
from psp.reconstruction_overlap import (
    atomic_delta_gram, atomic_delta_overlap_table, build_delta_radial_cache,
    reconstruction_gram, reconstruction_gram_correction, lowdin_factor, rotate_band_rows,
)


def atomic_data():
    z, w = leggauss(100)
    r, wr = (z+1)*4.5, w*4.5
    ell, kappa = np.array([0, 1, 1, 1]), np.array([-1, 1, -2, -2])
    R = r[:, None]**ell*np.exp(-r[:, None]**2*np.array([.6, .7, .8, 1.1]))
    return dict(r=r, weights_dr=wr, l=ell, kappa=kappa, delta_u=r[:, None]*R)


def test_raw_delta_gram_matches_independent_spatial_angular_quadrature():
    data = atomic_data()
    z, wz = leggauss(8)
    phi = np.arange(16)*np.pi/8
    directions = np.stack(np.broadcast_arrays(
        np.sqrt(1-z*z)[:, None]*np.cos(phi),
        np.sqrt(1-z*z)[:, None]*np.sin(phi), z[:, None]), axis=-1).reshape(-1, 3)
    angular_weights = np.repeat(wz, len(phi))*np.pi/8
    labels = spinor_function_labels(data['l'], data['kappa'])
    functions = []
    for opf, mj in labels:
        angle = spinor_spherical_harmonic(data['kappa'][opf], mj, directions)
        functions.append((data['delta_u'][:, opf]/data['r'])[:, None, None]*angle[None])
    waves = np.asarray(functions)
    direct = np.einsum('irps,jrps,r,p->ij', waves.conj(), waves,
                       data['weights_dr']*data['r']**2, angular_weights)
    np.testing.assert_allclose(atomic_delta_gram(data), direct, atol=3e-14, rtol=3e-14)


def test_generic_fourier_cache_and_delta_overlap_convention():
    data = atomic_data()
    r, w = data['r'], data['weights_dr']
    K = np.array([[0., 0., 0.], [.7, -.3, .2], [2.1, .9, -.4], [4., 2., 1.]])
    center, volume = np.array([.3, -.2, .1]), 75.
    R = data['delta_u']/r[:, None]
    cache = build_pauli_fourier_cache(R, r, w, data['l'], data['kappa'],
                                       momentum_max=8, momentum_points=4097)
    direct = atomic_pauli_fourier(R, r, w, data['l'], data['kappa'], K, center_cart=center)
    np.testing.assert_allclose(evaluate_pauli_fourier_cache(cache, K, center_cart=center),
                               direct, atol=2e-11, rtol=2e-11)
    delta_cache = build_delta_radial_cache(data, momentum_max=8, momentum_points=4097)
    raw = atomic_delta_overlap_table(data, K, center_cart=center, cell_volume=volume,
                                     radial_cache=delta_cache)
    np.testing.assert_allclose(raw, direct.conj()/np.sqrt(volume), atol=3e-12, rtol=3e-11)
    # A canonical lift obtains r(K) without restating its normalization formula.
    probe = np.zeros((1, 2, len(K)), complex)
    probe[:, 0] = 1
    canonical_r = _lift_cartesian(probe, K)[0, 0]
    normalized = atomic_delta_overlap_table(data, K, center_cart=center,
                                             cell_volume=volume, normalized_rkb_source=True,
                                             radial_cache=delta_cache)
    np.testing.assert_allclose(normalized*canonical_r, raw, atol=2e-14, rtol=2e-14)
    with pytest.raises(ValueError, match='cached range'):
        evaluate_pauli_fourier_cache(cache, np.array([[8.01, 0, 0]]))
    changed = dict(data, delta_u=data['delta_u']*1.0001)
    with pytest.raises(ValueError, match='source mismatch'):
        atomic_delta_overlap_table(changed, K, center_cart=center, cell_volume=volume,
                                    radial_cache=delta_cache)
    with pytest.raises(ValueError, match='interpolation exceeds'):
        build_pauli_fourier_cache(R, r, w, data['l'], data['kappa'],
                                  momentum_max=8, momentum_points=8,
                                  relative_tolerance=0, absolute_tolerance=0)


def paired_fields():
    rng = np.random.default_rng(14329)
    source = (rng.normal(size=(2, 6, 2, 64))+1j*rng.normal(size=(2, 6, 2, 64)))/12
    delta = (rng.normal(size=(3, 2, 64))+1j*rng.normal(size=(3, 2, 64)))/30
    dual = (rng.normal(size=(3, 2, 64))+1j*rng.normal(size=(3, 2, 64)))/25
    c = np.einsum('isg,pnsg->pni', dual.conj(), source)
    d = np.einsum('isg,pnsg->pni', delta.conj(), source)
    B = np.einsum('isg,jsg->ij', delta.conj(), delta)
    corrected = source+np.einsum('pni,isg->pnsg', c, delta)
    S0 = np.einsum('pnsg,pmsg->pnm', source.conj(), source)
    return source, delta, c, d, B, corrected, S0


def test_full_metric_includes_nonidentity_source_and_residual_cross_terms():
    source, delta, c, d, B, corrected, S0 = paired_fields()
    gram = reconstruction_gram(S0, c, d, B)
    direct = np.einsum('pnsg,pmsg->pnm', corrected.conj(), corrected)
    np.testing.assert_allclose(gram, direct, atol=8e-15, rtol=3e-14)
    assert np.max(abs(S0-np.eye(6))) > .1
    incomplete = S0+np.einsum('pni,ij,pmj->pnm', c.conj(), B, c)
    assert np.max(abs(incomplete-direct)) > .01
    np.testing.assert_allclose(reconstruction_gram_correction(c*0, d, B), 0, atol=0)


def test_complex_lowdin_orientation_padding_and_lift_commutation():
    source, delta, c, d, B, corrected, S0 = paired_fields()
    metric = reconstruction_gram(S0, c, d, B)
    padded = np.pad(metric, ((0, 0), (0, 2), (0, 2)))
    receipt = lowdin_factor(padded, physical_bands=6)
    A = receipt['inverse_sqrt']
    np.testing.assert_allclose(A[:, 6:, 6:], np.broadcast_to(np.eye(2), (2, 2, 2)), atol=0)
    np.testing.assert_allclose(A[:, :6, 6:], 0, atol=0)
    np.testing.assert_allclose(receipt['gram'], padded, atol=0)
    assert np.max(receipt['factor_isometry_error']) < 5e-14
    carrier = np.pad(corrected, ((0, 0), (0, 2), (0, 0), (0, 0)))
    got = np.asarray(rotate_band_rows(carrier, A))
    expected = np.einsum('pmn,pmsg->pnsg', A, carrier)
    np.testing.assert_allclose(got, expected, atol=5e-15, rtol=3e-14)
    np.testing.assert_allclose(got[:, 6:], 0, atol=0)
    newS = np.einsum('pnsg,pmsg->pnm', got[:, :6].conj(), got[:, :6])
    np.testing.assert_allclose(newS, np.broadcast_to(np.eye(6), (2, 6, 6)), atol=8e-14)
    # Wrong conjugation of the band factor is a material negative control.
    wrong = np.einsum('pmn,pmsg->pnsg', A.conj(), carrier)
    assert np.max(abs(wrong-expected)) > .02
    rng = np.random.default_rng(827)
    K = rng.normal(size=(64, 3))*14
    for parent in range(2):
        lifted = _lift_cartesian(carrier[parent], K)
        after = np.asarray(rotate_band_rows(lifted[None], A[parent:parent+1]))[0]
        before = _lift_cartesian(got[parent], K)
        np.testing.assert_allclose(after, before, atol=3e-15, rtol=3e-14)


def test_unresolved_nonpositive_nonhermitian_and_nonzero_padding_refuse():
    for value in (np.diag([1., -.1]), np.diag([1., 1e-18])):
        with pytest.raises(ValueError, match='nonpositive or numerically unresolved'):
            lowdin_factor(value, physical_bands=2)
    with pytest.raises(ValueError, match='not Hermitian'):
        lowdin_factor(np.array([[1., .1j], [0., 1.]]), physical_bands=2)
    with pytest.raises(ValueError, match='padded'):
        lowdin_factor(np.eye(3), physical_bands=2)
    with pytest.raises(ValueError, match='atomic Gram block'):
        reconstruction_gram_correction(np.zeros((2, 3)), np.zeros((2, 4)), np.eye(3))


def test_p4_band_rotation_source_coefficients_and_sample_layouts():
    if jax.device_count() != 4:
        pytest.skip('explicit four-device distribution oracle')
    _, _, c, _, _, corrected, _ = paired_fields()
    S = np.einsum('pnsg,pmsg->pnm', corrected.conj(), corrected)
    A = lowdin_factor(np.pad(S, ((0, 0), (0, 2), (0, 2))), physical_bands=6)['inverse_sqrt']
    source = np.pad(corrected, ((0, 0), (0, 2), (0, 0), (0, 0)))
    coeff = np.pad(c, ((0, 0), (0, 2), (0, 0)))
    factor = jax.device_put(A, NamedSharding(RUNTIME.mesh, P()))
    for host, spec, axis, conjugated in (
            (source, P(None, None, None, ('x', 'y')), 1, False),
            (coeff, P(None, ('x', 'y'), None), 1, False),
            (source, P(None, 'x', None, 'y'), 1, False),
            (source.conj().transpose(0, 2, 3, 1), P(None, None, 'x', 'y'), -1, True)):
        sharding = NamedSharding(RUNTIME.mesh, spec)
        distributed = jax.device_put(host, sharding)
        rotate = jax.jit(lambda value, a: rotate_band_rows(value, a,
                         band_axis=axis, conjugated=conjugated), out_shardings=sharding)
        actual = rotate(distributed, factor)
        expected = np.asarray(rotate_band_rows(host, A, band_axis=axis, conjugated=conjugated))
        assert actual.sharding == sharding
        for shard in actual.addressable_shards:
            np.testing.assert_allclose(np.asarray(shard.data), expected[shard.index],
                                       atol=5e-15, rtol=3e-14)
