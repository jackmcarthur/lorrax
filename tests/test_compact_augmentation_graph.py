"""Cartesian derivative, support, Fourier-graph and served-metric oracles."""
import numpy as np
import pytest

from runtime import bootstrap
bootstrap()

from common.bispinor_init import HALFALPHA
from psp.augmentation_spinors import (COMPACT_GRAPH_FIELD_MODEL, compact_graph_taper,
    evaluate_normalized_delta, evaluate_normalized_radials, spinor_function_labels)


def compact_cache():
    radius = np.linspace(0., 3., 121)
    kappa = np.asarray((-1, -2, 1, -3, 2))
    ell = np.where(kappa < 0, -kappa-1, kappa)
    f = radius[:, None]**ell[None]*np.exp(-radius[:, None]**2)
    df = -2*radius[:, None]*f
    positive = radius > 0
    df[positive] += ell[None]*f[positive]/radius[positive, None]
    df[0, ell == 1] = 1.
    return dict(radius=radius, ell=ell, kappa=kappa, large_R=f.astype(complex),
        dlarge_R_dr=df.astype(complex), small_R=np.zeros_like(f, dtype=complex),
        dsmall_R_dr=np.zeros_like(f, dtype=complex), field_model=COMPACT_GRAPH_FIELD_MODEL,
        taper_start=1.7, support_radius=2.6, half_alpha=float(HALFALPHA))


def cartesian_lower(cache, points, step):
    # Independent five-point Cartesian derivative of the served upper spinor.
    derivatives = []
    for axis in np.eye(3):
        samples = [evaluate_normalized_delta(cache, points+offset*step*axis)[:, :2]
                   for offset in (-2, -1, 1, 2)]
        derivatives.append((samples[0]-8*samples[1]+8*samples[2]-samples[3])/(12*step))
    dx, dy, dz = derivatives
    return -1j*HALFALPHA*np.stack((dz[:, 0]+dx[:, 1]-1j*dy[:, 1],
        dx[:, 0]+1j*dy[:, 0]-dz[:, 1]), axis=1)


def test_compact_lower_is_cartesian_gradient_including_taper_and_lower_f():
    cache = compact_cache()
    points = np.asarray(((.23, .13, .11), (.41, -.32, .21), (2.13, .19, -.17), (2.48, .23, .12)))
    actual = evaluate_normalized_delta(cache, points)[:, 2:]
    expected = cartesian_lower(cache, points, 2e-5)
    np.testing.assert_allclose(actual, expected, rtol=4e-7, atol=3e-12)
    assert np.max(abs(actual[spinor_function_labels(cache['ell'], cache['kappa'])[:, 0] == 3])) > 1e-6
    # Omitting the derivative of the window demonstrably fails in its annulus.
    from scipy.interpolate import CubicHermiteSpline
    r = np.linalg.norm(points, axis=1)
    spline = CubicHermiteSpline(cache['radius'], cache['large_R'], cache['dlarge_R_dr'], axis=0)
    w, dw = compact_graph_taper(r, cache['taper_start'], cache['support_radius'])
    f, df = spline(r), spline(r, 1)
    wrong = 1j*HALFALPHA*w[:, None]*(df+(cache['kappa'][None]+1)*f/r[:, None])
    _, right = evaluate_normalized_radials(cache, r)
    assert np.max(abs(right-wrong)) > 1e-4
    assert np.max(abs(dw)) > 0


def test_support_has_no_surface_contact_and_origin_limits_are_finite():
    cache = compact_cache()
    at_origin = evaluate_normalized_delta(cache, np.zeros((1, 3)))
    assert np.isfinite(at_origin).all()
    labels = spinor_function_labels(cache['ell'], cache['kappa'])
    assert np.count_nonzero(at_origin[cache['kappa'][labels[:, 0]] != 1, 2:]) == 0
    assert np.max(abs(at_origin[cache['kappa'][labels[:, 0]] == 1, 2:])) > 0
    outside = evaluate_normalized_delta(cache, np.asarray(((2.6, 0., 0.), (100., 0., 0.))))
    assert np.count_nonzero(outside) == 0
    points = np.asarray(((2.6, 0., 0.),))
    for h in (1e-4, 5e-5):
        assert np.max(abs(cartesian_lower(cache, points, h))) < 3e-10
    uncut = {key: value for key, value in cache.items()
             if key not in ('field_model', 'taper_start', 'support_radius', 'half_alpha')}
    assert np.max(abs(evaluate_normalized_delta(uncut, points)[:, :2])) > .001
    for bad in (dict(cache, field_model='hard_mask'), dict(cache, half_alpha=2*HALFALPHA),
                dict(cache, taper_start=2.7)):
        with pytest.raises(ValueError, match='descriptor'):
            evaluate_normalized_delta(bad, points)


def test_served_graph_metric_and_fourier_match_one_physical_field(tmp_path):
    from isdf.atomic_moments import (build_served_overlap_cache, served_overlap_table,
        write_served_moment_cache, load_served_moment_cache)
    from psp.augmentation_spinors import _lift_cartesian
    cache = compact_cache()
    controls = dict(support_radius=2.6, momentum_max=4., momentum_points=1025)
    first = build_served_overlap_cache(cache, **controls)
    refined = build_served_overlap_cache(cache, **controls, source_quad_order=12)
    assert first['source_quad_order'] == 10
    # Both leg diagnostics use amplitude-relative error, not error/tolerance.
    scales = np.max(abs(first['lower']['radial']), axis=0)
    absolute, relative = (first['lower'][name] for name in
                          ('maximum_absolute_error', 'maximum_scaled_error'))
    assert absolute/scales.max() <= relative*(1+1e-14)
    assert relative <= absolute/scales.min()*(1+1e-14)
    assert relative < 1e-8
    np.testing.assert_allclose(first['B'], refined['B'], rtol=2e-14, atol=2e-15)
    with pytest.raises(ValueError, match='GL order'):
        build_served_overlap_cache(cache, **controls, source_quad_order=9)
    K = np.asarray(((0., 0., 0.), (.3, -.8, .4), (1.1, .4, -1.2), (2.9, -.1, .3)))
    field = served_overlap_table(first, K, center_cart=(.2, -.4, .7), cell_volume=17.).conj()*np.sqrt(17.)
    precursor = field[:, :2]*np.sqrt(1+HALFALPHA**2*np.sum(K*K, axis=1))[None, None]
    np.testing.assert_allclose(_lift_cartesian(precursor, K), field, rtol=3e-14, atol=2e-15)
    path = tmp_path/'graph.npz'
    write_served_moment_cache(path, first, normalized_cache_sha256='a'*64)
    restored = load_served_moment_cache(path, normalized_cache_sha256='a'*64, support_radius=2.6)
    np.testing.assert_array_equal(served_overlap_table(restored, K, center_cart=(.2, -.4, .7), cell_volume=17.),
                                  served_overlap_table(first, K, center_cart=(.2, -.4, .7), cell_volume=17.))
    with pytest.raises(ValueError, match='identity'):
        load_served_moment_cache(path, normalized_cache_sha256='b'*64, support_radius=2.6)
