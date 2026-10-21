"""Independent positive-REAL-time audit of the user's full-matrix MGKBA.

The oracle builds G</>, W</>, Sigma</> and their integrals explicitly; it
does not call a production branch/contraction/resolvent helper. Native setup
fixtures supply only physical orbitals, meshes and a signed complex W bank.
"""
import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from tests.test_tdagw_full_microscopic_cpu import _native_port_fixture, _reciprocal_banks


def _literal_time(psi,energy,rho,banks,eta,*,left_phase=False,negative_q=False,transpose_w=False,row_shell=False,order=768):
    """User G< = i Psi rho exp(-iE*t) Psi†, Sigma< = i G< W< / Nq."""
    nodes,weights=np.polynomial.legendre.leggauss(order)
    times=(nodes+1.)*150.
    weights=weights*150.*np.exp(-eta*times)
    nk,nb=energy.shape
    ain=np.zeros((nk,nb,nb),complex)
    aout=np.zeros_like(ain)
    for q in range(nk):
        poles,residues=banks[(-q)%nk if negative_q else q]
        if transpose_w:
            residues=residues.swapaxes(-1,-2)
        hp=np.where((poles.real<0)[:,None,None],-residues,0.)
        pp=np.where((poles.real>0)[:,None,None],residues,0.)
        wp=-1j*np.einsum('tl,luv->tuv',np.exp(-1j*times[:,None]*poles),pp)
        wh=-1j*np.einsum('tl,luv->tuv',np.exp(-1j*times[:,None]*poles),hp)
        for k in range(nk):
            partner=(k-q)%nk
            orbitals=psi[partner]
            phase=np.exp(-1j*times[:,None]*energy[partner])
            def green(density):
                if left_phase:
                    return np.einsum('msu,tm,mn,nrv->tusvr',orbitals,phase,density,orbitals.conj())
                return np.einsum('msu,mn,tn,nrv->tusvr',orbitals,density,phase,orbitals.conj())
            gl=1j*green(rho[partner])
            gg=-1j*green(np.eye(nb)-rho[partner])
            # These are the literal scalar pointwise W/spin Green products,
            # followed by the independent external orbital projection.
            sl=1j*np.einsum('asu,tusvr,tuv,brv->tab',psi[k].conj(),gl,wh,psi[k])/nk
            sg=1j*np.einsum('asu,tusvr,tuv,brv->tab',psi[k].conj(),gg,wp,psi[k])/nk
            shell=np.exp(1j*times[:,None]*energy[k])
            fold='t,tab,ta->ab' if row_shell else 't,tab,tb->ab'
            ain[k]+=-1j*np.einsum(fold,weights,sl,shell)
            aout[k]+=1j*np.einsum(fold,weights,sg,shell)
    qmat=np.eye(nb)[None]-rho
    dagger=lambda x:x.conj().swapaxes(-1,-2)
    raw=ain@qmat+qmat@dagger(ain)-aout@rho-rho@dagger(aout)
    return ain,aout,raw


@pytest.mark.parametrize('density_case',['coherent','projector'])
def test_native_and_device_collision_match_literal_positive_time_user_kernel(tmp_path,monkeypatch,density_case,record_property):
    from common.units import RYD_TO_EV
    from ffi import fft
    from file_io.shared_pole_store import write_general_shared_pole_model
    from gw import ppm_sigma as ps
    from tdagw import collision as coll
    from tdagw import propagate as tp

    case=_native_port_fixture(monkeypatch)
    mesh,rng,nk,nmu=(case[k] for k in ('mesh','rng','nk','nmu'))
    banks=_reciprocal_banks(rng,nk,nmu)
    width=4*nmu
    left=np.stack([np.concatenate(list(r),axis=1) for _,r in banks])[:,case['perm'],None]
    right=np.broadcast_to(np.tile(np.eye(nmu),(1,4)),(nk,nmu,width))[:,case['perm'],None]
    poles=np.stack([np.repeat(p,nmu) for p,_ in banks])
    put=lambda a:jax.device_put(np.ascontiguousarray(a,complex),NamedSharding(mesh,P(None,'x',None,'y')))
    identity=dict(iteration_id='literal-time',hamiltonian='H',energies='E',occupations='rho',
                  wavefunctions='psi',centroids='mu')
    path=str(tmp_path/'literal_time.h5')
    header=write_general_shared_pole_model(path,put(left),put(right),
        jax.device_put(poles,NamedSharding(mesh,P())),np.full(nk,width),q_span=(0,nk),
        meta=case['meta'],tables=case['tables'],recipe=dict(accuracy='test'),receipts=dict(identity=identity),kmax=width)
    source=ps.shared_pole_window_source(dict(path=path,identity=identity,digest=header['digest'],
        K=header['K'],representation='ordered_complex'),case['meta'],efermi_ry=.071,linalg='distributed')
    rotations=[];density=[]
    for _ in range(nk):
        u,_=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
        rotations.append(u)
        density.append(np.array([[.64,.13+.08j],[.13-.08j,.31]]))
    rotations,rho=np.asarray(rotations),np.asarray(density)
    if density_case=='projector':
        rho[:]=np.diag([1.,0.])
    energy=case['energy'][:,1:]
    psi=np.einsum('kmsu,kmi->kisu',case['psi'][:,1:],rotations)
    config=case['fixture']._config(np.array([-2.,2.]))
    config.sigma.regularization_ev=.14*RYD_TO_EV
    config.sigma.quadrature_eps=2e-7
    window=dict(b_lo=1,U=rotations,e=energy,rho=rho,f=rho.diagonal(axis1=-2,axis2=-1).real)
    with fft.xla_reference():
        result=ps.window_sigma_c(case['wfns'],source,window,ps.window_sigma_plan(config,window_only=True),
            meta=case['meta'],mesh_xy=mesh,internal='active',pairing='vertex',branches=True,print_fn=lambda *a,**k:None)
    ain,aout,expected=_literal_time(psi,energy,rho,banks,.14)
    alternate=_literal_time(psi,energy,rho,banks,.14,order=512)
    quad_error=max(float(np.max(abs(a-b))) for a,b in zip((ain,aout,expected),alternate))
    assert quad_error<1e-11
    rp=result['active_branch_pair']['cond'][1]
    rh=result['active_branch_pair']['val'][0].conj().swapaxes(-1,-2)
    np.testing.assert_allclose(1j*rp,aout,rtol=1e-5,atol=3e-8)
    np.testing.assert_allclose(1j*rh,ain,rtol=1e-5,atol=3e-8)
    gamma,injection,mh=map(np.asarray,coll.branch_operators(result['active_branch_pair']['cond'],
        result['active_branch_pair']['val']))
    actual=np.asarray(coll.collision(gamma,injection,rho))-1j*(mh@rho-rho@mh)
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=6e-8)
    # Verify the actual k-sharded propagation program does not subtract the
    # equilibrium collision in raw mode, and adds MH exactly separately.
    padded=int(mesh.size)*int(np.ceil(nk/mesh.size))
    pack=lambda a:tp.place_window_blocks(a[None],mesh,padded)
    req=tp._place(tp._pad_k(rho,0,padded),mesh,0)
    weights=tp._place(np.pad(np.full(nk,1/nk),(0,padded-nk)),mesh,0)
    device=tp.collision_program(mesh,raw=True)(pack(gamma),pack(injection),pack(rho),req,
        pack(gamma),pack(injection),weights)
    rate=np.asarray(jax.jit(coll.rate)(device['diss'],pack(np.zeros_like(rho))))[0,:nk]
    np.testing.assert_allclose(rate-1j*(mh@rho-rho@mh),expected,rtol=2e-5,atol=6e-8)
    errors=dict(ain_max_abs=float(np.max(abs(1j*rh-ain))),aout_max_abs=float(np.max(abs(1j*rp-aout))),
        raw_C_max_abs=float(np.max(abs(actual-expected))),
        device_program_raw_C_max_abs=float(np.max(abs(rate-1j*(mh@rho-rho@mh)-expected))),
        time_512_vs_768_max_abs=quad_error,negative_controls={})
    if density_case=='coherent':
        # A wrong phase side, an extra q reversal or endpoint transpose are
        # detectable; this fixture does not commute rho past the propagator.
        for wrong in (dict(left_phase=True),dict(negative_q=True),dict(transpose_w=True),dict(row_shell=True)):
            bad=_literal_time(psi,energy,rho,banks,.14,**wrong)
            error=max(float(np.linalg.norm(bad[i]-value)) for i,value in enumerate((ain,aout,expected)))
            assert error>1e-4
            errors['negative_controls'][next(iter(wrong))]=error
    for key,value in errors.items():
        record_property(key,value)


def test_coarse_q_real_lattice_tile_is_literal_minus_q_average(record_property):
    """Independent FFT identity for the exact 1/Nq coarse Sigma quadrature."""
    from gw.ppm_tau_kernel import sigma_lattice_tile
    rng=np.random.default_rng(83071)
    fine,coarse=(6,3,1),(3,3,1)
    nk,nq=int(np.prod(fine)),int(np.prod(coarse))
    g=rng.normal(size=(nk,2,2))+1j*rng.normal(size=(nk,2,2))
    w=rng.normal(size=(nq,2,2))+1j*rng.normal(size=(nq,2,2))
    wr=np.fft.ifftn(w.reshape(*coarse,2,2),axes=(0,1,2),norm='ortho').reshape(nq,2,2)
    with jax.enable_x64(True):
        tiled=np.asarray(sigma_lattice_tile(wr,fine,coarse))
    gr=np.fft.ifftn(g.reshape(*fine,2,2),axes=(0,1,2),norm='ortho')
    actual=np.fft.fftn(gr*tiled.reshape(*fine,2,2),axes=(0,1,2),norm='ortho').reshape(nk,2,2)/np.sqrt(nk)
    expected=np.zeros_like(actual)
    for ki,k in enumerate(np.ndindex(fine)):
        for qi,q in enumerate(np.ndindex(coarse)):
            shift=np.asarray(q)*(np.asarray(fine)//np.asarray(coarse))
            partner=np.ravel_multi_index(tuple((np.asarray(k)-shift)%fine),fine)
            expected[ki]+=g[partner]*w[qi]/nq
    error=float(np.max(abs(actual-expected)))
    np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=2e-14)
    record_property('coarse_q_convolution_max_abs',error)


def test_passive_three_state_actual_native_fft_has_outward_filled_boundary(tmp_path,monkeypatch,record_property):
    """Real ISDF density ports realize a closed Hermitian transition channel.

    M=V diag(g) V† is realized by Psi[a,mu]=V[a,mu]* and the PSD bosonic
    residue g g^T. One exactly dark empty coordinate pads the active mesh.
    Identical k/q copies cancel the exact 1/Nq normalization.
    This is a prescribed frozen passive W, not a self-consistent RPA proof.
    """
    from common.units import RYD_TO_EV
    from ffi import fft
    from file_io.shared_pole_store import write_general_shared_pole_model
    from gw import ppm_sigma as ps
    from tdagw import collision as coll

    case=_native_port_fixture(monkeypatch)
    mesh,nk,nmu=(case[k] for k in ('mesh','nk','nmu'))
    channel=np.array([[0.,0.,1.],[0.,0.,1.],[1.,1.,0.]])
    eigen,vectors=np.linalg.eigh(channel)
    g=np.r_[eigen,0.]
    psi=np.zeros((nk,4,2,nmu),complex)
    psi[:,:3,0,:3]=vectors.conj()
    np.testing.assert_allclose(np.einsum('asu,u,bsu->ab',psi[0,:3].conj(),g,psi[0,:3]),channel,atol=5e-16)
    energy=np.broadcast_to([-1.,-.5,1.,3.],(nk,4)).copy()
    rho=np.broadcast_to(np.diag([1.,1.,0.,0.]),(nk,4,4)).copy()
    monkeypatch.setattr(case['fixture'],'NBF',4)
    wfns,meta=case['fixture']._gw_bundle(mesh,psi[...,case['perm']],energy,rho.diagonal(axis1=-2,axis2=-1).real,
                                       case['wfns'].green_parent.plan)
    meta.mu_basis=case['meta'].mu_basis
    meta.shared_pole_capacity=case['meta'].shared_pole_capacity
    b=np.outer(g,g)
    banks=[(np.array([1.,-1.],complex),np.array([b,-b])) for _ in range(nk)]
    width=2*nmu
    left=np.stack([np.concatenate(list(r),axis=1) for _,r in banks])[:,case['perm'],None]
    right=np.broadcast_to(np.tile(np.eye(nmu),(1,2)),(nk,nmu,width))[:,case['perm'],None]
    poles=np.broadcast_to(np.repeat([1.,-1.],nmu),(nk,width)).astype(complex)
    put=lambda a:jax.device_put(np.ascontiguousarray(a,complex),NamedSharding(mesh,P(None,'x',None,'y')))
    identity=dict(iteration_id='passive-three-state',hamiltonian='H',energies='E',occupations='P',
                  wavefunctions='ISDF-channel',centroids='four')
    path=str(tmp_path/'passive_three_state.h5')
    header=write_general_shared_pole_model(path,put(left),put(right),
        jax.device_put(poles,NamedSharding(mesh,P())),np.full(nk,width),q_span=(0,nk),
        meta=meta,tables=case['tables'],recipe=dict(accuracy='test'),receipts=dict(identity=identity),kmax=width)
    source=ps.shared_pole_window_source(dict(path=path,identity=identity,digest=header['digest'],
        K=header['K'],representation='ordered_complex'),meta,efermi_ry=0.,linalg='distributed')
    config=case['fixture']._config(np.array([-2.,2.]))
    config.sigma.regularization_ev=.2*RYD_TO_EV
    config.sigma.quadrature_eps=1e-9
    window=dict(b_lo=0,U=np.broadcast_to(np.eye(4),(nk,4,4)),e=energy,rho=rho,
                f=rho.diagonal(axis1=-2,axis2=-1).real)
    with fft.xla_reference():
        result=ps.window_sigma_c(wfns,source,window,ps.window_sigma_plan(config,window_only=True),
            meta=meta,mesh_xy=mesh,internal='active',pairing='vertex',branches=True,print_fn=lambda *a,**k:None)
    ain,aout,expected=_literal_time(psi,energy,rho,banks,.2)
    gamma,injection,mh=map(np.asarray,coll.branch_operators(result['active_branch_pair']['cond'],
                                                        result['active_branch_pair']['val']))
    actual=np.asarray(coll.collision(gamma,injection,rho))-1j*(mh@rho-rho@mh)
    error=float(np.max(abs(actual-expected)))
    np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=2e-9)
    blocking=np.linalg.eigvalsh((gamma-injection)[:,:2,:2])
    np.testing.assert_allclose(blocking[:,0],-.031543335232515776,atol=2e-9)
    assert np.linalg.eigvalsh(actual[:,:2,:2]).max()>.0315
    assert np.max(abs(np.trace(actual,axis1=-2,axis2=-1)))<2e-9
    record_property('passive_three_state_raw_C_max_abs',error)
    record_property('passive_three_state_filled_min',float(blocking.min()))
    record_property('passive_three_state_fresh_trace_max_abs',float(np.max(abs(np.trace(actual,axis1=-2,axis2=-1)))))
