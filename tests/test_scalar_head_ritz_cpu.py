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
    from gw.tangential_head_scalar import scalar_head_certificate
    observations=dict(z=held,Wc=reference,dWc_ds=dreference,vc0=4*abs(values[0]),
        M1=np.sum(omega*residue),M3=np.sum(omega**3*residue))
    certificate=scalar_head_certificate(result,observations,eta_ry=.02,
        static_value=values[0],inverse_m1=np.sum(residue/omega**2))
    assert np.all(certificate['passivity']['passivity'])
    assert max(certificate['sample_value_relative'])<3e-9
    assert max(certificate['sample_slope_relative'])<3e-8
    assert certificate['static_relative']<3e-10
    assert certificate['inverse_m1_relative']<3e-10
    with pytest.raises(ValueError,match='GATE scalar_head_passivity'):
        scalar_head_certificate(result,dict(observations,vc0=.1*abs(values[0])),eta_ry=.02,
            static_value=values[0],inverse_m1=np.sum(residue/omega**2))


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


def test_scalar_pole_agreement_handles_different_rank_counts_without_host_payload(monkeypatch):
    import jax
    import common.collectives as collectives
    from gw.tangential_head_scalar import _shared_poles
    counts=np.array([[2],[3],[1]],np.int64)
    packets=np.array([[[.2,.8,1.],[.1,.2,0.]],
                      [[.3,.7,.9],[.2,.1,.05]],
                      [[.4,1.,1.],[.3,0.,0.]]],complex)
    calls=[]
    def gather(value):
        calls.append(value)
        if len(calls)==1:
            assert not isinstance(value,jax.Array)
            return counts
        assert isinstance(value,jax.Array)  # Physics uses the existing device collective.
        assert value.shape==(2,3)
        return packets
    monkeypatch.setattr(collectives,'all_gather_processes',gather)
    model=dict(Omega_p=np.array([.4+0j]),B_p=np.array([.3+0j]),provenance={})
    actual=_shared_poles(model,object())
    np.testing.assert_array_equal(actual['Omega_p'],packets[0,0,:2])
    np.testing.assert_array_equal(actual['B_p'],packets[0,1,:2])
    assert actual['provenance']['rank_local_counts']==[2,3,1]


def test_scalar_pencil_refuses_other_rank_failure_before_payload_agreement(monkeypatch):
    import jax
    import common.collectives as collectives
    from gw import tangential_head_scalar as owner
    jax.config.update('jax_enable_x64',True)
    real_execute=owner._execute
    monkeypatch.setattr(owner,'_execute',lambda program,args,**kw:
        real_execute(program,args,mesh=None,side=kw['side'],label=kw['label']))
    def gather(flags):
        # This process's physical pencil passes, another process fails.
        assert np.asarray(flags).dtype==bool and np.all(flags)
        return np.array([[True],[False]])
    monkeypatch.setattr(collectives,'all_gather_processes',gather)
    z=np.array([0j,.2j,.7j]);value,slope=_measure(z,np.array([.3]),np.array([.5]))
    with pytest.raises(ValueError,match='GATE scalar_head_gram_diagonal_positive'):
        owner.scalar_head_ritz(z,value,slope,moment_m1=.15,moment_m3=.0135,mesh=object())


def test_scalar_program_admits_new_buffers_beside_current_prepared_payload(monkeypatch):
    import jax
    import jax.numpy as jnp
    import distrib_la
    import common.gpu_utils as gpu
    import runtime.aot_memory as aot
    from lxkit import memory
    from gw.tangential_head_scalar import _execute
    jax.config.update('jax_enable_x64',True)
    program=jax.jit(lambda x:x*2)
    args=(jnp.arange(3,dtype=jnp.float64),)
    expected_new=aot.compiled_new_bytes(program.lower(*args).compile())+7
    seen=[]
    monkeypatch.setattr(gpu,'device_budget_bytes',lambda:1000)
    monkeypatch.setattr(memory,'room',lambda **kw:64)
    monkeypatch.setattr(memory,'book',lambda *a,**kw:None)
    monkeypatch.setattr(distrib_la,'local_eigh_workspace',lambda *a:7)
    def admit(stage,price,room):
        seen.append((price,room))
    monkeypatch.setattr(memory,'admit',admit)
    output,priced=_execute(program,args,mesh=object(),side=3,label='test')
    np.testing.assert_array_equal(output,args[0]*2)
    assert priced==expected_new
    # The run has 1000 total bytes, but only64 remain beside the current
    # operator/RHS/body and already materialized scalar args. No double charge.
    assert len(seen)==1 and seen[0][1]==64


def test_scalar_pencil_reuses_shared_single_matrix_dispatch_policy(monkeypatch):
    import jax
    import distrib_la
    from gw import tangential_head_scalar as owner
    jax.config.update('jax_enable_x64',True)
    real=distrib_la.eigh_local
    shapes=[]
    def record(a):
        shapes.append(a.shape)
        return real(a)
    monkeypatch.setattr(distrib_la,'eigh_local',record)
    owner._scalar_ritz_program.cache_clear()
    owner._scalar_check_program.cache_clear()
    z=np.array([0j,.2j,.7j]);value,slope=_measure(z,np.array([.3]),np.array([.5]))
    model=owner.scalar_head_ritz(z,value,slope,moment_m1=.15,moment_m3=.0135)
    owner.scalar_head_certificate(model,dict(z=z,Wc=value,dWc_ds=slope,vc0=4.,M1=.15,M3=.0135),
        eta_ry=.02,static_value=value[0],inverse_m1=.5/.3**2)
    assert len(shapes)>=3 and all(len(shape)==2 for shape in shapes)


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


def test_bulk_draw_identity_and_actions_are_fixed_between_frequency_batches(monkeypatch):
    import jax
    import vcoul.resolvent_head as owner
    from vcoul import CoulombGeometry,get_kernel,scalar_resolvent_head
    jax.config.update('jax_enable_x64',True)
    original=owner.iter_minibz_photon_samples
    def reduced_draw(kernel,geometry,kgrid,**kwargs):
        # The real provider/seed/folding route, with a small test-only count;
        # no production sampler policy or additional environment setting.
        return original(kernel,geometry,kgrid,nsamples=256,qmc_reps=2,**kwargs)
    monkeypatch.setattr(owner,'iter_minibz_photon_samples',reduced_draw)
    geometry=CoulombGeometry(np.eye(3)*2*np.pi/10.,1000.)
    omega,residue=np.array([.3,1.2]),np.array([.002,.003])
    z=np.array([0j,.2j,.31+.12j]);value,slope=_measure(z,omega,residue)
    response=dict(Xi=value[:,None,None]*np.eye(3),dXi_ds=slope[:,None,None]*np.eye(3),
        moments=dict(M1=np.sum(omega*residue)*np.eye(3),M3=np.sum(omega**3*residue)*np.eye(3)))
    kernel=get_kernel(3);grid=(2,2,2)
    full=scalar_resolvent_head(kernel,geometry,grid,response)
    subset=dict(response,Xi=response['Xi'][1:],dXi_ds=response['dXi_ds'][1:])
    repeated=scalar_resolvent_head(kernel,geometry,grid,subset)
    np.testing.assert_array_equal(full['Wc'][1:],repeated['Wc'])
    np.testing.assert_array_equal(full['dWc_ds'][1:],repeated['dWc_ds'])
    assert full['M1']==repeated['M1'] and full['M3']==repeated['M3']
    assert full['quadrature']==repeated['quadrature']
    for index in range(len(z)):
        bare,total=kernel.q0_average(geometry,grid,S_cart=response['Xi'][index],
                                    method='auto',nsamples=256,qmc_reps=2)
        np.testing.assert_allclose(full['Wc'][index],complex(total-bare),rtol=3e-13,atol=3e-12)
    with pytest.raises(ValueError,match='GATE scalar_resolvent_head_sphere'):
        scalar_resolvent_head(kernel,geometry,grid,response,analytic_sphere=True)
