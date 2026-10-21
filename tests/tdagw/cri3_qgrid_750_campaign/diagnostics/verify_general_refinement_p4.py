"""Bounded P4 native refinement bindings; NOT a material fit admission.

Actual native current-site chi/Dyson, epsilon/LU, residue panels, and a
separate exact signed rational-pencil oracle execute. Host lifecycle tests
cover injected 0/1/2-round fit decisions; this proof does not inject a fit
into a physical microscopic model or claim a trajectory.
"""
from __future__ import annotations


def refinement_stage(c):
    import time
    from pathlib import Path
    import numpy as np
    import jax.numpy as jnp
    from types import SimpleNamespace
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import gather_to_host
    from ffi import ffi_dial_key
    from gw.greens_function_kernel import density_left_endpoint
    from gw.ppm_sigma import fresh_shared_pole_capacity
    from gw.response_bank import _bank_execution,response_algebra
    from gw.shared_pole_execution import face_matmul
    from gw.shared_pole_pencil import tile_join,join_vectors
    from gw.shared_pole_constructor import collective_general_pencil
    from gw.shared_pole_screening import (_general_current_site_samples,
        _general_dielectric_basic,_general_dielectric_inverse,_general_refinement_programs,
        _general_refinement_provider_receipt,_general_frequency_programs)
    from gw.shared_pole_refinement import (general_refinement_candidates,
        general_refinement_preflight,general_refinement_targets,evaluate_general_refinement)
    from verify_common_coherent_chi_p4 import bank_config
    import gw.shared_pole_refinement as refinement_owner
    import tdagw.snapshot_runtime as runtime_owner
    root=Path(__file__).resolve().parents[4]
    origins={m.__name__:str(Path(m.__file__).resolve()) for m in (refinement_owner,runtime_owner)}
    if any(root not in Path(path).parents for path in origins.values()):
        raise ValueError('refinement/runtime source module origin mismatch')

    mesh=c['mesh'];grid=c['grid'];qgrid=c['coarse'];nq=int(np.prod(qgrid));mu=c['mu']
    put=c['put'];face=NamedSharding(mesh,P(None,'x','y'));rep=NamedSharding(mesh,P())
    host=lambda value:np.asarray(gather_to_host(value))
    coords=np.indices(qgrid).reshape(3,-1).T
    minus=np.ravel_multi_index(((-coords)%qgrid).T,qgrid)
    q=1;partner=int(minus[q]);assert q!=partner
    config=bank_config();gamma=.03;eta=.1;p=.4+.6j
    meta=SimpleNamespace(nkx=8,nky=8,nkz=1,nk_tot=c['nk'],nq_tot=nq,
        nspinor=c['ns'],nspinor_wfnfile=2,nspin=1,n_rmu=mu,qgrid=qgrid,
        qgrid_response_window='fejer')
    meta=fresh_shared_pole_capacity(meta,mesh);meta.shared_pole_capacity.at_phase()
    price=meta.shared_pole_capacity.reserve('general_refinement_software_fixture',
        resident_bytes_per_rank=16*nq*mu*mu*80//mesh.size,
        workspace_bytes_per_rank=16*(2*nq+8)*mu*mu//mesh.size)
    meta.shared_pole_capacity.live_stages=(price['stage'],)
    execution=dict(seconds={},memory=[],compiled=[])
    execute=_bank_execution(meta,mesh,execution,config)
    H=put(np.broadcast_to(.15*np.eye(mu,dtype=np.complex128),(nq,mu,mu)),P(None,'x','y'))
    dyson,_,_,algebra=response_algebra(meta,config,mesh_xy=mesh,n=mu,ordered=True)
    mm=face_matmul(mesh);endpoint=density_left_endpoint(mesh,c['pm'].shape)
    density=np.broadcast_to(np.diag([1.,.3,.7,0.]),(c['nk'],c['nb'],c['nb'])).astype(complex).copy()
    density[:,1,2]=.12j;density[:,2,1]=-.12j
    occupations=np.broadcast_to([1.,.3,.7,0.],(c['nk'],c['nb'])).copy()
    ph=endpoint(c['pm'],put(density,P()))
    physical=dict(energy=c['energy'],rho=density,occupations=occupations,
        active=slice(1,3),response_active=slice(1,3),stop=c['nb'],nb=c['nb'],gamma=gamma,
        recipe=dict(bank_rule_tolerance=1e-8),label='native-refinement-software',
        pm=c['pm'],pn=c['pn'],ph=ph,energy_d=c['ed'],mesh_xy=mesh,grid=grid,qgrid=qgrid,
        nk=c['nk'],nmu=mu,ns=c['ns'],layout='face',response_window='fejer',
        response_partition='blocks',query_q_ids=np.arange(nq),plan=c['plan'],
        execute_response=execute,dyson=dyson,H=H)
    clocks={};sample_calls=[]
    def begin(name):clocks[name]=time.monotonic()
    def end(name,operands=None):
        import jax
        if operands is not None:jax.block_until_ready(operands)
        clocks.pop(name)
    def sample(sites,*,values=True):
        sample_calls.append([complex(z) for z in sites])
        return _general_current_site_samples(physical,np.asarray(sites),scalar_cache={},
            begin=begin,end=end,phase_start=clocks,evaluate_dyson=values)
    initial=sample([1.2j]);old=initial['values'][0]
    # Synthetic observable suspect residue has EXACTLY orthogonal L/R.
    lf=np.zeros((1,mu,2),complex);rf=lf.copy()
    lf[0,0,0]=1+1j;rf[0,1,0]=2-1j
    ld,rd=put(lf,P(None,'x','y')),put(rf,P(None,'x','y'))
    norms,panels=_general_refinement_programs(mesh,nq,0,q,int(mesh.shape['y']))
    ln,rn=execute(norms,(ld,rd),'direct')
    norm=float(host(ln*rn)[0,0]);assert norm>0
    metadata=[dict(poles_ry=np.array([p]),residue_norm_bound=np.array([norm if i==q else 0.]))
        for i in range(nq)]
    candidates=general_refinement_candidates(metadata,minus,eta,state_digest='native-fixture',round_index=0)
    assert len(candidates)==1 and candidates[0]['q']==q
    supports=[dict(sample_index=0,z_ry=1.2j,side='R',panel_id='original')]
    general_refinement_preflight(candidates,np.array([1.2j]),state_digest='native-fixture',
        eta_ry=eta,existing_supports=supports)
    memory_start=len(execution['memory']);at_p=sample([p],values=False)
    raw=at_p['values'][0]
    operators=dict(dyson=dyson,H=H[q:q+1],matmul=mm,face=face,rep=rep,mask=np.ones(mu,bool))
    record=dict(q=q,z_ry=[p.real,p.imag],at_requested_site=True,state_digest='native-fixture',
        gamma_ry=gamma,prefactor=float(algebra['prefactor']))
    basic=_general_dielectric_basic(raw[q:q+1],operators,record)
    witness=_general_dielectric_inverse(raw[q:q+1],operators,basic)
    plan=general_refinement_targets(candidates,[witness],np.array([1.2j]),
        state_digest='native-fixture',gamma_ry=gamma,eta_ry=eta,existing_supports=supports)
    lp,rp,lm,rm=execute(panels,(ld,rd,put(minus,P())),'direct')
    for actual,index,expected in ((lp,q,lf),(rp,q,rf),(lm,partner,lf.conj()),(rm,partner,rf.conj())):
        h=host(actual);want=np.zeros_like(h);v=expected[0,:,0];want[index,:,0]=v/np.linalg.norm(v)
        np.testing.assert_allclose(h,want,rtol=1e-14,atol=1e-14)
    state=dict(sites=np.array([1.2j]),values=[old],fit_positions=[0],held_positions=[],
        supports=supports,state_digest='native-fixture',refinement_rounds=0)
    receipt_context=dict(state_digest='native-fixture',recipe=physical['recipe'],response_execution=execution)
    def provider(sites):
        missing=[z for z in sites if z!=p];new=sample(missing)
        values={complex(z):v for z,v in zip(missing,new['values'])}
        values[p]=dyson.value(H,raw)
        receipt=_general_refinement_provider_receipt(receipt_context,sites,
            at_p['rules']+new['rules'],memory_start)
        return [values[complex(z)] for z in sites],receipt
    extended=evaluate_general_refinement(state,plan,evaluate_current=provider,
        crossing=lambda value:jnp.conj(value[minus]))
    assert extended['values'][0] is old and len(sample_calls)==3
    assert all(1.2j not in sites for sites in sample_calls[1:])

    # Separate exact nonnormal signed model: actual new panels enter the
    # real native L/R pencil, with full-port old tangents ensuring rank.
    rng=np.random.default_rng(6743)
    a=rng.normal(size=(nq,mu,1))+1j*rng.normal(size=(nq,mu,1))
    b=rng.normal(size=(nq,mu,1))+1j*rng.normal(size=(nq,mu,1))
    residue=a@b.conj().swapaxes(-1,-2);pole=.7-.04j
    def exact(z):return residue/(z-pole)-residue[minus].conj()/(z+pole.conjugate())
    oldq=put(np.broadcast_to(np.eye(mu,dtype=np.complex128),(nq,mu,mu)),P(None,'x','y'))
    left=[(.2+.4j,oldq),(p+.025*(1+1j),lp),(-p.conjugate()+.025*(-1+1j),lm)]
    right=[(.35+.3j,oldq),(p,rp),(-p.conjugate(),rm)]
    def join(rows,is_left):
        zs=join_vectors([np.full(v.shape[-1],z) for z,v in rows],int(mesh.shape['y']))
        actions=[]
        for z,v in rows:
            w=put(exact(z),P(None,'x','y'))
            actions.append(mm(v,w,transa='C') if is_left else mm(w,v))
        return put(zs,P()),tile_join([v for z,v in rows],-1,face),tile_join(actions,-2 if is_left else -1,face)
    ml,ql,il=join(left,True);mr,qr,wr=join(right,False)
    pencil,_,_,_=_general_frequency_programs(mesh,face.spec,ffi_dial_key())
    e,h,binp,cout=execute(pencil,(ml,mr,ql,il,qr,wr),'direct')
    maximum=0.
    for iq in (q,partner):
        model=collective_general_pencil(host(e[iq]),host(h[iq]),max_order=8,
            rank_tolerance=1e-10,frequency_scale_ry=2.)
        l=host(cout[iq])@model['output_map'];r=host(binp[iq]).conj().T@model['input_map']
        for z in (.31+.12j,-.44+.17j,1.1+.08j):
            got=(l/(z-model['poles_ry']))@r.conj().T
            error=float(np.linalg.norm(got-exact(z)[iq])/np.linalg.norm(exact(z)[iq]))
            maximum=max(maximum,error);assert error<1e-8
    assert not np.allclose(exact(.31+.12j)[q],exact(-.31-.12j)[q])
    return dict(actual_native_chi_dyson=True,actual_exact_point_epsilon_LU=witness,
        additional_module_origins=origins,
        actual_admitted_residue_panels=True,orthogonal_left_right=True,distinct_q_crossing=True,
        original_sample_object_preserved=True,new_physical_sites=3,
        exact_signed_toy_full_W_held_relative_error=maximum,
        synthetic_suspect_not_a_physical_fit_admission=True,execution=execution)


if __name__=='__main__':
    from verify_complex_box_chi_p4 import main
    raise SystemExit(main(refinement_stage))
