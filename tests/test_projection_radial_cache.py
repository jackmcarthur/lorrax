"""Species radial transform reuse against uncached, complex j-resolved duals."""
import numpy as np
import pytest

from psp.augmented_samples import atomic_projection_table, build_projection_radial_cache


def _data():
    x, wx = np.polynomial.legendre.leggauss(96)
    r, w = 2*(x+1), 2*wx
    return dict(r=r, weights_dr=w, l=np.array([0, 0, 1, 1]),
        kappa=np.array([-1, -1, 1, -2]), ps_u=np.column_stack((
        r*np.exp(-r*r), r*(1-.4*r*r)*np.exp(-.6*r*r)*(1+.3j),
        r*r*np.exp(-.8*r*r)*(1-.2j), r*r*np.exp(-.9*r*r))))


def test_cached_duals_match_direct_at_independent_cartesian_momenta():
    data = _data()
    cache = build_projection_radial_cache(data, momentum_max=4., momentum_points=2049)
    rng = np.random.default_rng(119920)
    k = rng.normal(size=(513, 3))
    k *= rng.uniform(0., 3.9, size=(513, 1))/np.linalg.norm(k, axis=1)[:, None]
    k[0] = 0.
    options = dict(center_cart=(.7, -.3, .2), cell_volume=120.)
    direct = atomic_projection_table(data, k, **options)
    cached = atomic_projection_table(data, k, radial_cache=cache, **options)
    assert np.max(np.abs(cached-direct)) < 2e-11
    assert cache['maximum_scaled_error'] < 1e-10
    assert np.all(np.isfinite(cached))


def test_cache_authenticates_source_and_refuses_extrapolation():
    data = _data()
    cache = build_projection_radial_cache(data, momentum_max=4., momentum_points=2049)
    altered = dict(data, ps_u=data['ps_u']*1.001)
    options = dict(center_cart=(0., 0., 0.), cell_volume=120., radial_cache=cache)
    with pytest.raises(ValueError, match="cache source or momentum range mismatch"):
        atomic_projection_table(altered, np.array([[1., 0., 0.]]), **options)
    with pytest.raises(ValueError, match="cache source or momentum range mismatch"):
        atomic_projection_table(data, np.array([[4.01, 0., 0.]]), **options)


def test_coarse_cache_cannot_claim_strict_interpolation_accuracy():
    with pytest.raises(ValueError, match="interpolation exceeds tolerance"):
        build_projection_radial_cache(_data(), momentum_max=4., momentum_points=8,
                                      relative_tolerance=1e-12, absolute_tolerance=1e-14)
