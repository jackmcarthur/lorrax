"""Scalar adapters preserve the physical head and reuse its common owners."""
import numpy as np
import pytest


def _measure(z,omega,residue):
    s=np.asarray(z)**2
    weight=2*omega*residue
    return (np.sum(weight/(s[...,None]-omega**2),axis=-1),
            -np.sum(weight/(s[...,None]-omega**2)**2,axis=-1))


def test_scalar_shared_ritz_preserves_positive_poles_and_both_moment_kinds():
    import jax
    from gw.tangential_head_scalar import scalar_head_ritz
    jax.config.update('jax_enable_x64',True)
    omega=np.array([.07,.23,.9,2.7])
    residue=np.array([.03,.17,.8,.3])
    z=np.r_[0j,1j*np.geomspace(.015,8,8),.13+.04j,.48+.17j]
    values,slopes=_measure(z,omega,residue)
    result=scalar_head_ritz(z,values,slopes,moment_m1=np.sum(omega*residue),
                           moment_m3=np.sum(omega**3*residue))
    actual_o,actual_b=result['Omega_p'],result['B_p']
    assert np.all(actual_o.real>0) and np.all(actual_o.imag==0)
    assert np.all(actual_b.real>=0) and np.all(actual_b.imag==0)
    held=np.r_[1j*np.geomspace(.001,30,31),np.linspace(.02,3.2,47)+.019j]
    reference,dreference=_measure(held,omega,residue)
    actual,dactual=_measure(held,actual_o,actual_b)
    np.testing.assert_allclose(actual,reference,rtol=3e-9,atol=3e-10)
    np.testing.assert_allclose(dactual,dreference,rtol=3e-8,atol=3e-9)
    np.testing.assert_allclose(np.sum(actual_b/actual_o**2),np.sum(residue/omega**2),rtol=3e-10)
    np.testing.assert_allclose(np.sum(2*actual_o*actual_b),np.sum(2*omega*residue),rtol=3e-10)
    # The exact static and inverse-frequency derivative constraints do not
    # arise from a residue modification: the entire held response agrees.
    assert result['provenance']['solve_mode']=='shared-Hermite-Ritz-scalar-v1'


def test_scalar_shared_ritz_refuses_nonphysical_gram_and_invalid_actions():
    import jax
    from gw.tangential_head_scalar import scalar_head_ritz
    jax.config.update('jax_enable_x64',True)
    z=1j*np.array([.05,.2,.5,1.3])
    omega,residue=np.array([.2,.8]),np.array([1.,-.3])
    values,slopes=_measure(z,omega,residue)
    with pytest.raises(ValueError,match='GATE scalar_head_'):
        scalar_head_ritz(z,values,slopes,moment_m1=np.sum(omega*residue),
                         moment_m3=np.sum(omega**3*residue))
    with pytest.raises(ValueError,match='matching nonempty'):
        scalar_head_ritz(z,values,slopes[:-1],moment_m1=1,moment_m3=1)
    with pytest.raises(ValueError,match='GATE scalar_head_keep'):
        scalar_head_ritz(z,values,slopes,moment_m1=1,moment_m3=1,gram_keep=1e-3)


def test_scalar_shared_ritz_refuses_nonfinite_retained_identity(monkeypatch):
    import jax
    import jax.numpy as jnp
    from gw import tangential_head_scalar as owner
    jax.config.update('jax_enable_x64',True)
    real_factory=owner._scalar_ritz_program
    def factory(cut):
        real=real_factory(cut)
        def inject(*args):
            model,diagnostics,zero,retained=real(*args)
            retained=dict(retained)
            retained[next(iter(retained))]=jnp.array([jnp.nan])
            return model,diagnostics,zero,retained
        return inject
    monkeypatch.setattr(owner,'_scalar_ritz_program',factory)
    z=np.array([0j,.2j,.7j])
    value,slope=_measure(z,np.array([.3]),np.array([.5]))
    with pytest.raises(ValueError,match='GATE scalar_head_retained_moments'):
        owner.scalar_head_ritz(z,value,slope,moment_m1=.15,moment_m3=.0135)


def test_scalar_cell_actions_match_independent_rank_one_completed_hamiltonian():
    import jax
    from vcoul import CoulombGeometry,get_kernel,scalar_resolvent_head,slab_minibz_photon_cubature
    jax.config.update('jax_enable_x64',True)
    real=np.array([[6.,0,0],[-3.,3*np.sqrt(3),0],[0,0,23.]])
    geometry=CoulombGeometry(2*np.pi*np.linalg.inv(real).T,np.linalg.det(real))
    kernel,grid=get_kernel(2),(3,3,1)
    omega,residue=np.array([.3,1.2]),np.array([.02,.03])
    weights=2*omega*residue
    z=np.array([0j,.2j,.31+.12j])
    value,slope=_measure(z,omega,residue)
    xy=np.diag([1.,1.,0])
    response=dict(Xi=value[:,None,None]*xy,dXi_ds=slope[:,None,None]*xy,
                  moments=dict(M1=.5*weights.sum()*xy,M3=.5*np.sum(weights*omega**2)*xy))
    result=scalar_resolvent_head(kernel,geometry,grid,response)
    chunk=slab_minibz_photon_cubature(kernel,geometry,grid).chunks[-1]
    count=chunk.physical_count
    q=chunk.q_cart[:count];v=chunk.D_raw[:count,0,0];w=chunk.sample_weight[:count]
    q2=np.sum(q[:,:2]**2,axis=1)
    # For each q, macro completion is an independent Hermitian rank-one
    # update T_q=diag(omega²)+v(q) q² cc^T, not a clone of its scalar formula.
    c=np.sqrt(weights)
    h=np.diag(omega**2)[None]+(v*q2)[:,None,None]*np.outer(c,c)
    eigenvalues,eigenvectors=np.linalg.eigh(h)
    endpoints=np.einsum('qij,i->qj',eigenvectors,c)
    spectral=(v*v*q2)[:,None]*endpoints**2
    exact=np.array([np.sum(w[:,None]*spectral/(point**2-eigenvalues))/sum(w) for point in z])
    exact_d=np.array([-np.sum(w[:,None]*spectral/(point**2-eigenvalues)**2)/sum(w) for point in z])
    np.testing.assert_allclose(result['Wc'],exact,rtol=2e-13,atol=3e-12)
    np.testing.assert_allclose(result['dWc_ds'],exact_d,rtol=2e-13,atol=3e-12)
    np.testing.assert_allclose(result['M1'],.5*np.sum(w[:,None]*spectral)/sum(w),rtol=2e-13)
    np.testing.assert_allclose(result['M3'],.5*np.sum(w[:,None]*spectral*eigenvalues)/sum(w),rtol=2e-13)
    for index in range(len(z)):
        bare,total=kernel.q0_average(geometry,grid,S_cart=response['Xi'][index])
        np.testing.assert_allclose(result['Wc'][index],complex(total-bare),rtol=2e-13,atol=3e-12)
