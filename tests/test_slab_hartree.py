"""Independent Fourier probes of the two slab Hartree mixed adjoints.

Only one local side is nonzero in each probe. The reference integrates the
compact polynomial and the compensation profile directly with Gauss--Bessel
quadrature; it does not use the production radial maps or Coulomb tables.
"""
from types import SimpleNamespace

import numpy as np
import pytest


def _slab_fixture():
    """Return a translated compact sphere and a physical 8-cubed FFT cell."""
    lengths = np.asarray([10., 10., 12.])
    reciprocal = np.diag(2*np.pi/lengths)
    grid = (8, 8, 8)
    wfn = SimpleNamespace(fft_grid=grid, cell_volume=float(np.prod(lengths)),
                          blat=1., bvec=reciprocal,
                          bdot=reciprocal@reciprocal.T)
    radius = .75
    r = np.r_[radius*1e-5, np.linspace(radius/15, radius, 15)]
    center = np.asarray([[.9, -.7, .35]])
    options = dict(radius=r, weights_dr=np.full(len(r), radius/len(r)),
                   lm=np.asarray([[0, 0]]), centers_cart=center,
                   support_radius=radius, minimum_atom_image_distance=10.,
                   interpolation_degree=5, quadrature_order=16,
                   fourier_points=1025, sys_dim=2)
    delta = (1-(r/radius)**2)**2
    x = np.arange(grid[0])[:, None, None]
    return wfn, options, delta, x


def _independent_sphere_transform(momentum, radius, epsilon):
    """Fourier transform of Y00*(D + epsilon*g0), in density-volume units."""
    from scipy.special import beta, spherical_jn
    z, w = np.polynomial.legendre.leggauss(128)
    r, dr = radius*(z+1)/2, radius*w/2
    delta = (1-(r/radius)**2)**2
    g0 = (1-(r/radius)**2)**6/(radius**3*beta(1.5, 7.)/2)
    return np.sqrt(4*np.pi)*np.dot(dr*r*r*(delta+epsilon*g0),
                                  spherical_jn(0, float(momentum)*r))


def _independent_slab_v(momentum, wfn):
    """Public slab convention, independently evaluated for an in-plane mode."""
    if momentum == 0:
        return 0.
    half_height = np.pi/wfn.bvec[2, 2]
    return 8*np.pi*(1-np.exp(-abs(momentum)*half_height))/(wfn.cell_volume*momentum**2)


def _functional_action(operand, rows, columns, ps, delta, exact):
    from isdf.atomic_hartree import charge_hartree_functional
    functional = charge_hartree_functional(operand)
    smooth = np.einsum('visxyz,uxyz,vjsxyz->uvij', rows.conj(),
                       np.asarray(functional['smooth_potential']), columns)
    ni, nj = delta.shape[1:3]
    raw = np.concatenate((delta.reshape(ni*nj, -1), ps.reshape(ni*nj, -1),
                          exact.reshape(ni*nj, -1)), axis=-1)
    raw *= operand['volume']/np.prod(operand['fft_grid'])
    local = (raw@np.asarray(functional['local_response']).T).T
    return smooth+local.reshape(len(operand['source_delta']), 1, ni, nj)


@pytest.mark.parametrize('epsilon', [0., .013])
def test_compact_source_to_complex_smooth_receiving_has_direct_fourier_limit(epsilon):
    from isdf.atomic_hartree import prepare_charge_hartree, make_charge_hartree_tile
    wfn, options, delta, x = _slab_fixture()
    radial_m0 = 8*options['support_radius']**3/105
    exact = np.asarray([[radial_m0+epsilon]])
    local = delta[None, None, None]
    zero = np.zeros_like(local)
    options['electron_count'] = np.sqrt(4*np.pi)*exact[:, 0]
    operand = prepare_charge_hartree(wfn, np.zeros((1, *wfn.fft_grid)), zero,
                                     local, exact, **options)
    np.testing.assert_allclose(operand['source_epsilon'], epsilon, atol=2e-12)
    np.testing.assert_array_equal(operand['smooth_neutral_response'], zero)
    modes = np.asarray([0, 1, -1])
    phases = np.exp(1j*np.asarray([.31, -.83, .47]))
    rows = np.zeros((1, 3, 1, *wfn.fft_grid), complex)
    for i, mode in enumerate(modes):
        rows[0, i, 0] = phases[i]*np.broadcast_to(
            np.exp(2j*np.pi*mode*x/wfn.fft_grid[0]), wfn.fft_grid)/np.sqrt(np.prod(wfn.fft_grid))
    pairs = np.zeros((1, 3, 3, 1, 1, len(delta)), complex)
    m0 = np.zeros((1, 3, 3, 1), complex)
    expected = np.zeros((1, 1, 3, 3), complex)
    g = wfn.bvec[0, 0]
    for i, mi in enumerate(modes):
        for j, mj in enumerate(modes):
            momentum = (mj-mi)*g
            expected[0, 0, i, j] = (phases[i].conjugate()*phases[j]
                *np.exp(1j*momentum*options['centers_cart'][0, 0])
                *_independent_slab_v(momentum, wfn)
                *_independent_sphere_transform(abs(momentum), options['support_radius'], epsilon))
    contract = make_charge_hartree_tile(operand)
    matrix, body, local_terms, mean, _ = map(np.asarray, contract(rows, rows, pairs, pairs, m0))
    np.testing.assert_allclose(matrix, expected, rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(matrix, matrix.swapaxes(-1, -2).conj(), atol=2e-13)
    np.testing.assert_array_equal(matrix, body)
    assert np.max(abs(local_terms)) == 0 and np.max(abs(mean)) == 0
    np.testing.assert_allclose(_functional_action(operand, rows, rows, pairs, pairs, m0), matrix,
                               rtol=2e-13, atol=2e-13)
    rectangular = np.asarray(contract(rows[:, :2], rows, pairs[:, :2], pairs[:, :2], m0[:, :2])[0])
    np.testing.assert_allclose(rectangular, expected[:, :, :2], rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize('epsilon', [0., .013])
def test_smooth_cosine_source_to_compact_receiving_retains_neutral_adjoint(epsilon):
    from isdf.atomic_hartree import prepare_charge_hartree, make_charge_hartree_tile
    wfn, options, delta, x = _slab_fixture()
    q, amplitude, angle = 3., .2, .27
    rho = np.broadcast_to(q/wfn.cell_volume*(1+amplitude*np.cos(2*np.pi*x/8+angle)),
                          wfn.fft_grid)[None]
    zero = np.zeros((1, 1, 1, len(delta)), complex)
    options['electron_count'] = q
    operand = prepare_charge_hartree(wfn, rho, zero, zero, np.zeros((1, 1)), **options)
    h = np.asarray([[1., .23+.37j, -.12j], [.23-.37j, .8, .41], [.12j, .41, -.4]])
    td = h[None, :, :, None, None, None]*delta[None, None, None, None, None]
    ps = np.zeros_like(td)
    exact = h[None, :, :, None]*(8*options['support_radius']**3/105+epsilon)
    boxes = np.zeros((1, 3, 1, *wfn.fft_grid), complex)
    g = wfn.bvec[0, 0]
    expected_scale = (_independent_slab_v(g, wfn)*q*amplitude
        *np.cos(angle+g*options['centers_cart'][0, 0])
        *_independent_sphere_transform(g, options['support_radius'], epsilon))
    expected = h[None, None]*expected_scale
    contract = make_charge_hartree_tile(operand)
    matrix, body, local, mean, _ = map(np.asarray, contract(boxes, boxes, ps, td, exact))
    np.testing.assert_allclose(matrix, expected, rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(matrix, matrix.swapaxes(-1, -2).conj(), atol=2e-13)
    assert np.max(abs(local[1])) > 1e-6
    assert np.max(abs(local[[0, 2, 3]])) == 0 and np.max(abs(mean)) == 0
    # Exact-M0 enrichment changes C only; the neutral D-M[D]*g0 adjoint is unchanged.
    un_enriched = h[None, :, :, None]*(8*options['support_radius']**3/105)
    _, body0, local0, _, _ = map(np.asarray, contract(boxes, boxes, ps, td, un_enriched))
    np.testing.assert_array_equal(local, local0)
    np.testing.assert_allclose(matrix-expected, 0., atol=2e-11)
    assert epsilon == 0 or np.max(abs(body-body0)) > 1e-5
    np.testing.assert_allclose(_functional_action(operand, boxes, boxes, ps, td, exact), matrix,
                               rtol=2e-13, atol=2e-13)
    rectangular = np.asarray(contract(boxes[:, :2], boxes, ps[:, :2], td[:, :2], exact[:, :2])[0])
    np.testing.assert_allclose(rectangular, expected[:, :, :2], rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize('invalid', ['support', 'tilted_cell', 'bulk_operator', 'kernel_owner'])
def test_slab_hartree_refuses_wrong_support_or_kernel(invalid):
    from isdf.atomic_hartree import prepare_charge_hartree, make_charge_hartree_tile
    wfn, options, delta, x = _slab_fixture()
    rho = np.zeros((1, *wfn.fft_grid))
    local = np.zeros((1, 1, 1, len(delta)), complex)
    options['electron_count'] = 0.
    if invalid == 'support':
        options['support_radius'] = 3.1
        with pytest.raises(ValueError, match='support|half-height|slab'):
            prepare_charge_hartree(wfn, rho, local, local, np.zeros((1, 1)), **options)
    elif invalid == 'tilted_cell':
        wfn.bvec = wfn.bvec.copy()
        wfn.bvec[2, 0] = .1
        wfn.bdot = wfn.bvec@wfn.bvec.T
        with pytest.raises(ValueError, match='slab|normal|orthogonal'):
            prepare_charge_hartree(wfn, rho, local, local, np.zeros((1, 1)), **options)
    else:
        operand = prepare_charge_hartree(wfn, rho, local, local, np.zeros((1, 1)), **options)
        if invalid == 'bulk_operator':
            operand = dict(operand, operator='ordinary_3D_periodic_full_FFT_G0_zero')
        else:
            operand = dict(operand, kernel=dict(operand['kernel'], owner='vcoul.Periodic3D'))
        with pytest.raises(ValueError, match='own full-FFT|kernel binding'):
            make_charge_hartree_tile(operand)
