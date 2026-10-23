"""The shared slab rule must resolve screening and certify both averages."""
from types import SimpleNamespace

import numpy as np
import pytest


def _hexagonal_cell():
    from vcoul import CoulombGeometry
    real = np.array([[6., 0., 0.], [-3., 3.*np.sqrt(3.), 0.], [0., 0., 23.]])
    return CoulombGeometry(2*np.pi*np.linalg.inv(real).T, np.linalg.det(real))


def _polar_reference(geometry, screening):
    """Independent adaptive polar integral over a regular hexagonal cell."""
    from scipy.integrate import quad
    # The regular mini-lattice WS hexagon has apothem |b1|/(2 Nk).
    apothem = np.linalg.norm(geometry.bvec[0])/6.
    area = 2*np.sqrt(3.)*apothem**2
    zc = np.pi/geometry.bvec[2, 2]

    def angular(phi):
        radius = apothem/np.cos(phi)

        def radial(r):
            if r == 0.:
                return 8*np.pi*zc
            numerator = -np.expm1(-zc*r)
            return 8*np.pi*numerator/(r*(1+8*np.pi*screening*numerator))

        return quad(radial, 0., radius, epsabs=1e-11, epsrel=1e-12)[0]

    return 6*quad(angular, -np.pi/6, np.pi/6, epsabs=1e-10, epsrel=1e-12)[0]/area


def test_shared_slab_rule_resolves_screened_radial_scale():
    """Strong screening agrees with independent polar integration, not a Duffy clone."""
    import jax
    from vcoul import get_kernel, slab_minibz_photon_cubature, slab_minibz_graded_photon_cubature
    from vcoul.minibz import minibz_coulomb_moment
    jax.config.update('jax_enable_x64', True)
    geometry = _hexagonal_cell()
    kernel, grid = get_kernel(2), (3, 3, 1)
    certificates = []
    bare, screened = kernel.q0_average(geometry, grid,
        S_cart=np.diag([-1., -1., 0.]), certificate_fn=certificates.append)
    np.testing.assert_allclose(np.asarray(bare), _polar_reference(geometry, 0.), rtol=2e-12)
    np.testing.assert_allclose(np.asarray(screened), _polar_reference(geometry, 1.), rtol=2e-12)
    assert certificates[0].final_error_ratio <= 1.

    receipt = slab_minibz_photon_cubature(kernel, geometry, grid)
    direct = slab_minibz_graded_photon_cubature(kernel, geometry, grid)
    assert receipt.orders == direct.orders
    for actual, other in zip(receipt.chunks, direct.chunks):
        np.testing.assert_array_equal(actual.q_cart, other.q_cart)
        np.testing.assert_array_equal(actual.sample_weight, other.sample_weight)
        np.testing.assert_array_equal(actual.D_raw, other.D_raw)
    chunk = receipt.chunks[-1]
    n = chunk.physical_count
    weight, q, v = chunk.sample_weight[:n], chunk.q_cart[:n], chunk.D_raw[:n, 0, 0]
    tt = np.einsum('q,qab->ab', weight, -chunk.D_raw[:n, 1:, 1:])/sum(weight)
    np.testing.assert_allclose(kernel.q0_average_transverse_tensor(geometry, grid), tt, rtol=1e-14)
    moment = np.einsum('q,qa,qb->ab', weight*v, q, q)/sum(weight*v)
    np.testing.assert_allclose(minibz_coulomb_moment(geometry.bvec, grid, is_2d=True),
                               moment, rtol=1e-14, atol=1e-16)


@pytest.mark.parametrize('bare_bad', [True, False])
def test_slab_certificate_refuses_either_unconverged_final_average(bare_bad):
    """A constant screened value cannot hide a divergent bare final pair, or vice versa."""
    from vcoul.slab_2d import Slab2D
    receipt = SimpleNamespace(orders=(8, 12, 16), physical_counts=(1, 1, 1),
                              polygon_vertices=np.zeros((6, 2)))
    final = (16, 1.1 if bare_bad else 1., 2. if bare_bad else 2.1)
    with pytest.raises(ValueError, match='slab_q0_polygon_not_converged'):
        Slab2D._require_q0_ladder_converged(receipt, [(8, 1., 2.), (12, 1., 2.), final])
