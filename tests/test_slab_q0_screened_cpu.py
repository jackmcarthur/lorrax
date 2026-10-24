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
    receipt = SimpleNamespace(orders=(16, 24, 32), physical_counts=(1, 1, 1),
                              polygon_vertices=np.zeros((6, 2)))
    final = (32, 1.1 if bare_bad else 1., 2. if bare_bad else 2.1)
    with pytest.raises(ValueError, match='slab_q0_polygon_not_converged'):
        Slab2D._require_q0_ladder_converged(receipt, [(16, 1., 2.), (24, 1., 2.), final])


def test_shared_slab_rule_resolves_actual_near_real_head_derivative():
    """The old radial pair misses a physical slope; the shared rule matches polar quadrature."""
    import json
    from pathlib import Path
    from scipy.integrate import quad
    from vcoul import CoulombGeometry,get_kernel,scalar_resolvent_head
    from vcoul.minibz import (_slab_minibz_wigner_seitz_polygon,
                             _slab_minibz_graded_rule,_photon_D_raw)
    from vcoul.resolvent_head import _chunk_fields
    from vcoul.slab_2d import _Q0_LADDER_ATOL as atol,_Q0_LADDER_RTOL as rtol
    def decode(v):
        if isinstance(v,dict):
            return complex(v['real'],v['imag']) if set(v)=={'real','imag'} else {k:decode(x) for k,x in v.items()}
        return [decode(x) for x in v] if isinstance(v,list) else v
    row=decode(json.loads((Path(__file__).parent/'data/slab_head_response_actions.json').read_text()))
    geometry=CoulombGeometry(np.asarray(row['geometry']['bvec']),row['geometry']['cell_volume'])
    response={k:({n:np.asarray(a) for n,a in value.items()} if k=='moments' else np.asarray(value))
              for k,value in row['response'].items()}
    kernel,grid=get_kernel(2),tuple(row['kgrid'])
    polygon,_,area=_slab_minibz_wigner_seitz_polygon(geometry.bvec,grid)
    old=[]
    for order in (12,16):
        q,weights,*_=_slab_minibz_graded_rule(polygon,area,order)
        bare,_=_photon_D_raw(q,kind='slab',zc=kernel.truncation_half_height(geometry))
        fields=_chunk_fields(q,bare[:,0,0],weights,response)
        old.append(fields['dWc_ds'][0]/weights.sum())
    old_ratio=abs(old[1]-old[0])/(atol+rtol*max(map(abs,old)))
    assert old_ratio>1.5  # Actual original failure1.964, without altering its tolerance.
    fields=scalar_resolvent_head(kernel,geometry,grid,response)
    assert fields['quadrature']['orders']==[16,24,32]

    # Independent polar integration over the exact Bragg-plane sectors:
    # no provider polygon, Duffy map, Gauss nodes or _chunk_fields here.
    # This actual cell is slightly strained; assuming a regular hexagon
    # would itself introduce a relative domain error of about6e-10.
    b0,b1=geometry.bvec[:2,:2]/np.asarray(grid[:2])[:,None]
    planes=np.array([b0,b1,b1-b0,-b0,-b1,b0-b1])
    planes=planes[np.argsort(np.arctan2(planes[:,1],planes[:,0]))]
    offsets=np.sum(planes**2,axis=1)/2
    vertices=np.array([np.linalg.solve(planes[[i,(i+1)%6]],offsets[[i,(i+1)%6]]) for i in range(6)])
    assert np.all(vertices@planes.T<=offsets+1e-14)
    polar_area=.5*abs(np.sum(vertices[:,0]*np.roll(vertices[:,1],-1)-vertices[:,1]*np.roll(vertices[:,0],-1)))
    breaks=np.r_[0.,np.sort(np.mod(np.arctan2(vertices[:,1],vertices[:,0]),2*np.pi)),2*np.pi]
    zc=kernel.truncation_half_height(geometry)
    xi,dxi=response['Xi'][0],response['dXi_ds'][0]
    def integrate(derivative):
        def angle(phi):
            u=np.array([np.cos(phi),np.sin(phi),0.])
            projection=planes@u[:2]
            radius=min(h/d for h,d in zip(offsets,projection) if d>0)
            s=u@xi@u;ds=u@dxi@u
            def radial(r):
                if r==0:return 0j
                bare=8*np.pi*(-np.expm1(-zc*r))/r**2
                denominator=1-bare*r*r*s
                return r*bare*bare*r*r*(ds/denominator**2 if derivative else s/denominator)
            return complex(quad(lambda r:radial(r).real,0,radius,epsabs=1e-10,epsrel=1e-12)[0],
                           quad(lambda r:radial(r).imag,0,radius,epsabs=1e-10,epsrel=1e-12)[0])
        total=0j
        for lo,hi in zip(breaks[:-1],breaks[1:]):
            total+=complex(quad(lambda t:angle(t).real,lo,hi,epsabs=1e-9,epsrel=1e-11)[0],
                           quad(lambda t:angle(t).imag,lo,hi,epsabs=1e-9,epsrel=1e-11)[0])
        return total/polar_area
    for key,derivative in [('Wc',False),('dWc_ds',True)]:
        np.testing.assert_allclose(fields[key][0],integrate(derivative),rtol=2e-11,atol=1e-9)
