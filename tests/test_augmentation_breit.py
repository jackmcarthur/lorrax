"""Independent static transverse geometry gates; no fitting-route admission."""
import numpy as np
import pytest

from isdf.augmentation_breit import (
    build_two_moment_compensation, compensation_curl_field_tile,
    compensation_poisson_gradient_maps, current_radial_moments,
    curl_angular_maps, curl_poisson_field_tile, exterior_curl_coefficients,
    evaluate_two_moment_compensation, static_compensation_bilinear,
    radial_poisson_gradient_maps, static_transverse_bilinear,
    static_transverse_geometry, two_moment_compensation_radial_fourier,
    two_moment_current_fourier, vector_harmonic_basis,
    static_transverse_block_factors,
)


def _labels(maximum):
    return np.asarray([(l, m) for l in range(maximum+1) for m in range(-l, l+1)])


def _grid(count=32):
    return (np.arange(1, count+1)/count)**3


def _fourier_energy(current, geometry, momentum_max, *, radial_values=None):
    """Direct Bessel transform and Cartesian P angular integral, without curl."""
    from scipy.special import sph_harm_y, spherical_jn

    lm = geometry['angular']['lm']
    radial = geometry['radial']
    r, wr = radial['quadrature_radius'], radial['quadrature_weights_dr']
    values = radial_values
    if values is None:
        values = np.einsum('qn,bin->biq', radial['interpolation_map'], current)
        for index, (l, _) in enumerate(lm):
            degree = np.where(radial['degrees'] == l)[0][0]
            values[:, index, :radial['origin_row_count']] *= radial['origin_factors'][degree]
    x, w = np.polynomial.legendre.leggauss(12)
    edges = np.linspace(0, momentum_max, int(momentum_max/2)+1)
    K = ((edges[:-1, None]+edges[1:, None])/2
         + (edges[1:]-edges[:-1])[:, None]*x/2).ravel()
    wK = ((edges[1:]-edges[:-1])[:, None]*w/2).ravel()
    transform = np.empty((len(K), 3, len(lm)), dtype=complex)
    for l in np.unique(lm[:, 0]):
        members = lm[:, 0] == l
        bessel = spherical_jn(int(l), K[:, None]*r[None])
        transform[:, :, members] = (-1j)**int(l)*np.einsum(
            'kq,q,biq->kbi', bessel, wr*r*r, values[:, members])
    maximum = int(lm[:, 0].max())
    cosine, weight = np.polynomial.legendre.leggauss(maximum+6)
    phi = np.arange(4*maximum+20)*2*np.pi/(4*maximum+20)
    theta = np.arccos(cosine)
    direction = np.stack(np.broadcast_arrays(
        np.sqrt(1-cosine[:, None]**2)*np.cos(phi),
        np.sqrt(1-cosine[:, None]**2)*np.sin(phi), cosine[:, None]), -1).reshape(-1, 3)
    Y = np.asarray([sph_harm_y(l, m, theta[:, None], phi).ravel() for l, m in lm])
    projector = np.eye(3)[None]-direction[:, :, None]*direction[:, None, :]
    angular = np.einsum('is,s,sbc,js->bicj', Y.conj(),
                       np.repeat(weight, len(phi))*2*np.pi/len(phi), projector, Y)
    return (2/np.pi)*np.einsum('kbi,bicj,kcj,k->', transform.conj(), angular,
                               transform, wK).real


def test_discontinuous_monopole_includes_surface_contact_and_exterior():
    geometry = static_transverse_geometry(_grid(8), _labels(0), support_radius=1)
    current = np.zeros((3, 1, 8)); current[2] = 1
    inside = static_transverse_bilinear(current, current, geometry, include_exterior=False)
    whole = static_transverse_bilinear(current, current, geometry)
    np.testing.assert_allclose(inside, 2/135, atol=2e-15)
    np.testing.assert_allclose(whole-inside, 2/27, atol=2e-15)
    np.testing.assert_allclose(whole, 4/45, atol=2e-15)
    # Interior div J is zero; dropping its surface contact would yield 2/15.
    assert abs(whole-2/15) > .04


def test_longitudinal_current_jump_has_pointwise_null_curl():
    r = _grid(12); geometry = static_transverse_geometry(r, _labels(1), support_radius=1)
    normal = curl_angular_maps(_labels(0))['normal']
    current = np.zeros((3, 4, len(r)), complex)
    # grad[(r²-1)Y00/2] inside, scalar continuous and zero outside.
    current[:, 1:] = normal[:, 1:, 0, None]*r
    field = curl_poisson_field_tile(current, geometry, quadrature_slice=slice(None))
    assert np.max(np.abs(field)) < 3e-14
    assert np.max(np.abs(exterior_curl_coefficients(current, geometry))) < 3e-15
    assert abs(static_transverse_bilinear(current, current, geometry)) < 1e-27


def test_poisson_and_charge_owners_share_the_same_physical_interpolant():
    from isdf.augmentation import radial_coulomb_metric_interpolated

    r = _grid(); lm = _labels(5)
    poisson = radial_poisson_gradient_maps(r, lm[:, 0], support_radius=1.03)
    coulomb = radial_coulomb_metric_interpolated(r, lm[:, 0], support_radius=1.03)
    for key in ('quadrature_radius', 'quadrature_weights_dr', 'interpolation_map', 'origin_factors'):
        np.testing.assert_allclose(poisson[key], coulomb[key], atol=0, rtol=0)
    np.testing.assert_allclose(poisson['moments0'], coulomb['moments'], atol=2e-15, rtol=2e-14)
    rng = np.random.default_rng(741)
    samples = rng.normal(size=len(r))
    geometry = static_transverse_geometry(r, _labels(0), support_radius=1.03)
    current = np.zeros((3, 1, len(r))); current[2, 0] = samples
    expected = (2/3)*(samples@coulomb['metric'][0]@samples)/(4*np.pi)
    np.testing.assert_allclose(static_transverse_bilinear(current, current, geometry), expected,
                               atol=3e-14, rtol=3e-13)


def test_complex_current_positive_curl_matches_independent_fourier_projector():
    rng = np.random.default_rng(51821); r = _grid(); lm = _labels(2)
    geometry = static_transverse_geometry(r, lm, support_radius=1, quadrature_order=32)
    amplitudes = rng.normal(size=(3, len(lm)))+1j*rng.normal(size=(3, len(lm)))
    current = amplitudes[:, :, None]*r[None, None]**lm[None, :, 0, None]*(1-r*r)**6
    geometric = static_transverse_bilinear(current, current, geometry)
    reference = _fourier_energy(current, geometry, 160)
    coarse = _fourier_energy(current, geometry, 80)
    assert abs(reference-coarse) < 2e-8
    np.testing.assert_allclose(geometric, reference, rtol=2e-7, atol=2e-9)


def test_batched_bilinear_is_hermitian_positive_and_independent_of_tile_size():
    rng = np.random.default_rng(12); r = _grid(12)
    geometry = static_transverse_geometry(r, _labels(2), support_radius=1)
    current = rng.normal(size=(4, 3, 9, len(r)))+1j*rng.normal(size=(4, 3, 9, len(r)))
    gram = static_transverse_bilinear(current[:, None], current[None], geometry, quadrature_tile=7)
    np.testing.assert_allclose(gram, gram.conj().T, atol=2e-14)
    assert np.linalg.eigvalsh(gram).min() > 0
    np.testing.assert_allclose(gram, static_transverse_bilinear(
        current[:, None], current[None], geometry, quadrature_tile=113), atol=2e-14, rtol=2e-14)


def test_matched_multipoles_cancel_exterior_but_signed_onsite_need_not_be_positive():
    r = _grid(); geometry = static_transverse_geometry(r, _labels(0), support_radius=1)
    moment = geometry['radial']['moments0'][0]
    delta = np.zeros((3, 1, len(r))); delta[2] = 1
    compensation = np.zeros_like(delta); compensation[2, 0] = (1-r*r)**6
    compensation *= (moment@delta[2, 0])/(moment@compensation[2, 0])
    np.testing.assert_allclose(exterior_curl_coefficients(delta, geometry),
                               exterior_curl_coefficients(compensation, geometry), atol=2e-15)
    full = (static_transverse_bilinear(delta, delta, geometry)
            - static_transverse_bilinear(compensation, compensation, geometry))
    interior = (static_transverse_bilinear(delta, delta, geometry, include_exterior=False)
                - static_transverse_bilinear(compensation, compensation, geometry, include_exterior=False))
    np.testing.assert_allclose(full, interior, atol=2e-15)
    assert full.real < 0


def test_two_radial_moments_have_independent_exact_monomial_values():
    r = np.linspace(.001, 1, 20)
    maps = radial_poisson_gradient_maps(r, np.arange(9), support_radius=1.07)
    # The origin convention differs from r^s if s!=L; include it explicitly.
    for index, l in enumerate(maps['degrees']):
        for s in range(4):
            for extra, key in ((0, 'moments0'), (2, 'moments2')):
                exponent = l+2+extra+s
                expected = (1.07**(exponent+1)-r[0]**(exponent+1))/(exponent+1)
                expected += r[0]**(l+3+extra+s)/(2*l+3+extra)
                np.testing.assert_allclose(maps[key][index]@r**s, expected, atol=3e-15, rtol=3e-13)


def test_geometry_rejects_missing_shells_bad_support_and_invalid_current():
    with pytest.raises(ValueError, match='complete canonical'):
        curl_angular_maps(np.asarray([[0, 0], [1, 0]]))
    with pytest.raises(ValueError, match='invalid compact'):
        static_transverse_geometry(_grid(), _labels(0), support_radius=.9)
    geometry = static_transverse_geometry(_grid(), _labels(0), support_radius=1)
    with pytest.raises(ValueError, match='Cartesian current'):
        static_transverse_bilinear(np.zeros((2, 1, 32)), np.zeros((2, 1, 32)), geometry)
    with pytest.raises(ValueError, match='positive integer'):
        static_transverse_bilinear(np.zeros((3, 1, 32)), np.zeros((3, 1, 32)), geometry, quadrature_tile=0)


def test_analytic_duals_match_both_physical_moments_by_direct_quadrature():
    R = 2.6; compensation = build_two_moment_compensation(np.arange(7), support_radius=R)
    x, w = np.polynomial.legendre.leggauss(192)
    r, wr = (x+1)*R/2, w*R/2
    g = evaluate_two_moment_compensation(compensation, r)
    for row, l in enumerate(compensation['degrees']):
        measured = np.asarray([np.einsum('ds,s->d', g[row], wr*r**(l+2+extra))
                               for extra in (0, 2)])
        np.testing.assert_allclose(measured, np.eye(2), atol=1e-13, rtol=1e-13)
    assert compensation['condition_number'].max() < 200


def test_sonine_transforms_equal_direct_bessel_integrals_and_have_correct_zero_limit():
    from scipy.special import spherical_jn

    R = 2.6; compensation = build_two_moment_compensation(np.arange(7), support_radius=R)
    K = np.asarray([0., 1e-10, 1e-5, .0004, .2, 1.2, 12., 75.])
    x, w = np.polynomial.legendre.leggauss(256)
    r, wr = (x+1)*R/2, w*R/2
    profiles = evaluate_two_moment_compensation(compensation, r)
    analytic = two_moment_compensation_radial_fourier(compensation, K)
    for row, l in enumerate(compensation['degrees']):
        reference = np.einsum('ds,ks,s->dk', profiles[row],
            spherical_jn(int(l), K[:, None]*r), wr*r*r)
        np.testing.assert_allclose(analytic[row], reference, atol=6e-14, rtol=3e-10)
    np.testing.assert_allclose(analytic[0, :, 0], [1., 0.], atol=3e-15)
    np.testing.assert_allclose(analytic[1:, :, 0], 0., atol=0)


def test_analytic_compensation_metric_matches_independent_fourier_without_interpolating_g():
    rng = np.random.default_rng(901); lm = _labels(2)
    geometry = static_transverse_geometry(_grid(), lm, support_radius=1, quadrature_order=32)
    compensation = build_two_moment_compensation(lm[:, 0], support_radius=1)
    moments = rng.normal(size=(3, len(lm), 2))+1j*rng.normal(size=(3, len(lm), 2))
    maps = compensation_poisson_gradient_maps(compensation, geometry['radial']['quadrature_radius'])
    profiles = evaluate_two_moment_compensation(compensation, geometry['radial']['quadrature_radius'])
    values = np.einsum('bid,idq->biq', moments, profiles[lm[:, 0]])
    reference = _fourier_energy(None, geometry, 160, radial_values=values)
    measured = static_compensation_bilinear(moments, moments, compensation, geometry, field_maps=maps)
    np.testing.assert_allclose(measured, reference, atol=2e-8, rtol=2e-9)
    np.testing.assert_allclose(compensation_curl_field_tile(moments, compensation, geometry,
        quadrature_slice=slice(30, 61)), compensation_curl_field_tile(moments, compensation, geometry,
        quadrature_slice=slice(30, 61), field_maps=maps), atol=2e-13, rtol=3e-13)


def test_delta_and_analytic_compensation_exterior_cancel_with_two_authentic_moments():
    geometry = static_transverse_geometry(_grid(8), _labels(0), support_radius=1)
    compensation = build_two_moment_compensation([0], support_radius=1)
    delta = np.zeros((3, 1, 8)); delta[2] = 1
    moments = current_radial_moments(delta, geometry)
    np.testing.assert_allclose(moments[2, 0], [1/3, 1/5], atol=1e-15)
    whole = (static_transverse_bilinear(delta, delta, geometry)
             - static_compensation_bilinear(moments, moments, compensation, geometry))
    inside = (static_transverse_bilinear(delta, delta, geometry, include_exterior=False)
              - static_compensation_bilinear(moments, moments, compensation, geometry, include_exterior=False))
    np.testing.assert_allclose(whole, inside, atol=3e-15)
    assert whole.real < 0


def test_current_fourier_uses_physical_monopole_and_center_bloch_phase():
    compensation = build_two_moment_compensation([0], support_radius=2.6)
    moments = np.asarray([[[.7+.2j, 1.2]], [[-.3j, .2]], [[.4, -.1]]])
    K = np.asarray([[0., 0., 0.], [.2, -.7, 1.3], [2.1, .1, -.2]])
    center = np.asarray([.3, -.1, 2.])
    origin = two_moment_current_fourier(moments, _labels(0), compensation, K)
    translated = two_moment_current_fourier(moments, _labels(0), compensation, K, center_cart=center)
    np.testing.assert_allclose(translated, origin*np.exp(-1j*K@center)[None], atol=2e-15)
    np.testing.assert_allclose(origin[:, 0], np.sqrt(4*np.pi)*moments[:, 0, 0], atol=3e-15)


def test_compensation_cache_binding_and_invalid_inputs_refused():
    geometry = static_transverse_geometry(_grid(8), _labels(0), support_radius=1)
    compensation = build_two_moment_compensation([0], support_radius=1)
    maps = compensation_poisson_gradient_maps(compensation, geometry['radial']['quadrature_radius'])
    other = build_two_moment_compensation([0], support_radius=1, power=7)
    with pytest.raises(ValueError, match='bind this geometry'):
        compensation_curl_field_tile(np.zeros((3, 1, 2)), other, geometry,
            quadrature_slice=slice(1, 3), field_maps=maps)
    with pytest.raises(ValueError, match='invalid two-moment'):
        build_two_moment_compensation([0], support_radius=1, power=1)
    with pytest.raises(ValueError, match='nonnegative'):
        two_moment_compensation_radial_fourier(compensation, np.asarray([-1.]))


def _block_energy(left, right, blocks, *, moments_left=None, moments_right=None):
    U=blocks['basis']['transform']
    def transform(values):
        return np.einsum('bk,...br->...kr',U.conj(),values.reshape(*values.shape[:-3],-1,values.shape[-1]))
    lhs,rhs=transform(left),transform(right)
    ml=transform(moments_left) if moments_left is not None else None
    mr=transform(moments_right) if moments_right is not None else None
    result=np.zeros(np.broadcast_shapes(left.shape[:-3],right.shape[:-3]),complex)
    for sector in blocks['sectors']:
        for index in sector['indices_by_M']:
            apply=lambda f,v:np.einsum('fr,...r->...f',f,v[...,index,:].reshape(*v.shape[:-2],-1))
            a,b=apply(sector['delta_factor'],lhs),apply(sector['delta_factor'],rhs)
            result+=np.einsum('...f,...f->...',a.conj(),b)
            if ml is not None:
                a,b=apply(sector['compensation_factor'],ml),apply(sector['compensation_factor'],mr)
                result-=np.einsum('...f,...f->...',a.conj(),b)
    return result


def test_vector_harmonic_closure_keeps_longitudinal_and_outer_edge_sectors():
    lm=_labels(4);geometry=static_transverse_geometry(_grid(12),lm,support_radius=1)
    comp=build_two_moment_compensation(lm[:,0],support_radius=1)
    blocks=static_transverse_block_factors(geometry,comp)
    assert blocks['basis']['unitary_error']<4e-14
    assert blocks['angular_forbidden_error']<4e-14
    assert blocks['angular_M_covariance_error']<4e-14
    for J,L in ((0,1),(4,3),(5,4)):
        assert any(s['J']==J and np.array_equal(s['Ls'],[L]) for s in blocks['sectors'])
    assert blocks['retained_columns']==3*len(lm)*12
    assert blocks['retained_compensation_columns']==3*len(lm)*2
    assert blocks['no_cutoff']
    with pytest.raises(ValueError,match='complete canonical'):
        vector_harmonic_basis(np.asarray([[0,0],[1,0]]))


def test_complex_vector_blocks_match_independent_cartesian_positive_field_gram():
    rng=np.random.default_rng(4831);lm=_labels(3)
    geometry=static_transverse_geometry(_grid(12),lm,support_radius=1)
    comp=build_two_moment_compensation(lm[:,0],support_radius=1)
    blocks=static_transverse_block_factors(geometry,comp)
    current=rng.normal(size=(4,3,len(lm),12))+1j*rng.normal(size=(4,3,len(lm),12))
    direct=static_transverse_bilinear(current[:,None],current[None],geometry,include_exterior=False)
    got=_block_energy(current[:,None],current[None],blocks)
    np.testing.assert_allclose(got,direct,atol=2e-13,rtol=2e-13)
    np.testing.assert_allclose(got,got.conj().T,atol=2e-13)
    assert np.linalg.eigvalsh(got).min()>0


def test_vector_blocks_match_signed_analytic_compensation_without_discarding_rank():
    rng=np.random.default_rng(8832);lm=_labels(2)
    geometry=static_transverse_geometry(_grid(12),lm,support_radius=1)
    comp=build_two_moment_compensation(lm[:,0],support_radius=1)
    blocks=static_transverse_block_factors(geometry,comp)
    current=rng.normal(size=(3,3,len(lm),12))+1j*rng.normal(size=(3,3,len(lm),12))
    moments=current_radial_moments(current,geometry)
    direct=(static_transverse_bilinear(current[:,None],current[None],geometry,include_exterior=False)
            -static_compensation_bilinear(moments[:,None],moments[None],comp,geometry,include_exterior=False))
    got=_block_energy(current[:,None],current[None],blocks,moments_left=moments[:,None],moments_right=moments[None])
    np.testing.assert_allclose(got,direct,atol=4e-13,rtol=3e-13)


def test_vector_Jzero_null_and_boundary_jump_constants_have_no_origin_artifact():
    lm=_labels(1);geometry=static_transverse_geometry(_grid(8),lm,support_radius=1)
    comp=build_two_moment_compensation(lm[:,0],support_radius=1)
    blocks=static_transverse_block_factors(geometry,comp)
    Jzero=next(s for s in blocks['sectors'] if s['J']==0)
    assert np.max(abs(Jzero['delta_factor']))<3e-14
    current=np.zeros((3,4,8),complex);current[2,0]=1
    np.testing.assert_allclose(_block_energy(current,current,blocks),2/135,atol=2e-15)
