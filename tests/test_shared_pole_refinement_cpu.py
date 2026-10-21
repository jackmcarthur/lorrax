"""Bounded refinement contracts only; no native/GPU execution claim."""
import numpy as np
import pytest

from gw.shared_pole_refinement import (general_refinement_candidates,
    authenticate_false_pole,general_refinement_targets,append_general_refinement_samples,
    evaluate_general_refinement,general_refinement_preflight)


def bank(p,n):
    return dict(poles_ry=np.asarray(p,complex),residue_norm_bound=np.asarray(n,float))


def candidate():
    return general_refinement_candidates([bank([.4+.6j,-.4+.6j,.9+.8j],[1,1,0])],
        np.array([0]),.1,state_digest='current',round_index=0)[0]


def witness(c):
    return dict(q=c['q'],z_ry=[c['z_ry'].real,c['z_ry'].imag],state_digest='current',
        gamma_ry=.02,at_requested_site=True,additional_collision_eta_ry=0.,
        inverse_bounds=dict(resolved=True,dimension=2,padding_defect=0.,
            residual_spectral_upper_bound=1e-12,sigma_min_lower=.03))


def original_supports():
    return [dict(sample_index=0,side='R',panel_id='original:0')]


def test_candidates_bound_rounds_observability_and_crossing_without_even_frequency_assumption():
    banks=[bank([.3+.4j,-.3+.4j,.8+.7j],[1,1,0]),
           bank([.9+.8j,-.2+.6j],[1e-100,4]),bank([-.9+.8j,.2+.6j],[1e-100,4])]
    rows=general_refinement_candidates(banks,np.array([0,2,1]),.1,state_digest='current',round_index=1)
    assert len(rows)==2 and [r['q'] for r in rows]==[1,1]
    assert rows[0]['residue_norm_bound']==1e-100  # No weak-pole cutoff.
    assert all(r['status']=='NEEDS_EXACT_POINT_DIELECTRIC_WITNESS' for r in rows)
    assert len(general_refinement_candidates([banks[0]],np.array([0]),.1,
        state_digest='current',round_index=0))==1
    with pytest.raises(ValueError,match='bounded round'):
        general_refinement_candidates(banks,np.array([0,2,1]),.1,state_digest='current',round_index=2)
    low=[bank([.4+.08j],[1])]
    assert not general_refinement_candidates(low,np.array([0]),.1,state_digest='current',round_index=0)
    assert general_refinement_candidates(low,np.array([0]),.1,state_digest='current',round_index=0,
                                         support_height_ry=.05)[0]['required_height_ry']==.05


@pytest.mark.parametrize('change', ['offset','old_state','gamma','padding','unresolved','zero_bound','extra_eta'])
def test_offset_W_agreement_and_unresolved_or_stale_epsilon_are_not_false_pole_proofs(change):
    c=candidate();w=witness(c)
    if change=='offset':w['z_ry'][0]+=.001
    elif change=='old_state':w['state_digest']='old'
    elif change=='gamma':w['gamma_ry']=.03
    elif change=='padding':w['inverse_bounds']['padding_defect']=.1
    elif change=='unresolved':w['inverse_bounds']['resolved']=False
    elif change=='zero_bound':w['inverse_bounds']['sigma_min_lower']=0.
    else:w['additional_collision_eta_ry']=.001
    with pytest.raises(ValueError,match='nonsingularity NOT established'):
        authenticate_false_pole(c,w,state_digest='current',gamma_ry=.02)


def test_actual_native_inverse_bound_schema_has_no_invented_padding_boolean():
    from gw.response_bank import dielectric_inverse_bounds
    c=candidate();w=witness(c)
    w['inverse_bounds']=dielectric_inverse_bounds(3.,2.,1e-14,2,padding_defect=0.)
    assert 'padding_decoupled' not in w['inverse_bounds']
    proof=authenticate_false_pole(c,w,state_digest='current',gamma_ry=.02)
    assert proof['classification']=='COMPUTED_DIELECTRIC_NONSINGULAR_AT_FITTED_POINT_ONLY'
    assert not proof['physical_kernel_pole_excluded'] and not proof['global_stability_proved']


def test_geometry_preflight_needs_no_fake_witness_but_cannot_evaluate_or_append():
    c=candidate();supports=original_supports()
    geometry=general_refinement_preflight([c],[.3j],existing_supports=supports,
        state_digest='current',eta_ry=.1)
    authenticated=general_refinement_targets([c],[witness(c)],[.3j],existing_supports=supports,
        state_digest='current',eta_ry=.1,gamma_ry=.02)
    assert geometry['proofs']==[] and geometry['authentication_required']
    assert geometry['target_geometry_signature']==authenticated['target_geometry_signature']
    assert geometry['support_descriptor_signature']==authenticated['support_descriptor_signature']
    state=dict(state_digest='current',sites=[.3j],values=[np.zeros((1,2,2),complex)],
        fit_positions=[0],held_positions=[],supports=supports)
    with pytest.raises(ValueError,match='authentication required BEFORE'):
        evaluate_general_refinement(state,geometry,evaluate_current=lambda sites:pytest.fail('unauthenticated probe'),
            crossing=lambda value:value.conj())
    with pytest.raises(ValueError,match='authentication required BEFORE'):
        append_general_refinement_samples(state,geometry,[np.zeros((1,2,2),complex)]*3,
            crossing=lambda value:value.conj())


def test_geometry_collision_is_detected_without_querying_or_authenticating_epsilon(monkeypatch):
    import gw.shared_pole_refinement as owner
    c=candidate();other=dict(c,z_ry=c['z_ry']+.025*(1+1j),q=1,partner=2)
    monkeypatch.setattr(owner,'authenticate_false_pole',lambda *a,**k:pytest.fail('geometry asked for a witness'))
    with pytest.raises(ValueError,match='collision|confluence'):
        general_refinement_preflight([c,other],[.3j],existing_supports=original_supports(),
            state_digest='current',eta_ry=.1)


def test_current_incremental_samples_keep_original_operands_roles_and_crossing_association():
    c=candidate();old_sites=np.array([.3j,.2+.1j])
    plan=general_refinement_targets([c],[witness(c)],old_sites,
        state_digest='current',gamma_ry=.02,eta_ry=.1,existing_supports=original_supports())
    assert plan['status']=='PLANNED_NOT_EXECUTED_NOT_ADMITTED'
    assert plan['new_microscopic_target_count']==3
    assert [t['role'] for t in plan['targets']]==['fit','fit','fit','fit','held','held']
    rng=np.random.default_rng(318)
    old=[rng.normal(size=(3,2,2)).astype(complex) for _ in old_sites]
    new=[rng.normal(size=(3,2,2)).astype(complex) for _ in range(3)]
    minus=np.array([0,2,1]);calls=[]
    def crossing(value):
        calls.append(value)
        return value[minus].conj()
    state=dict(state_digest='current',sites=old_sites,values=old,fit_positions=[0],held_positions=[1],
        supports=original_supports())
    extended=append_general_refinement_samples(state,plan,new,crossing=crossing)
    assert extended['values'][0] is old[0] and extended['values'][1] is old[1]
    assert extended['values'][2] is new[0] and extended['values'][4] is new[1]
    assert calls[0] is new[0] and calls[1] is new[1]
    np.testing.assert_array_equal(extended['values'][3],new[0][minus].conj())
    np.testing.assert_array_equal(extended['values'][5],new[1][minus].conj())
    assert extended['fit_positions']==[0,2,3,4,5] and extended['held_positions']==[1,6,7]
    assert extended['supports'][0] is state['supports'][0]
    assert [s['side'] for s in plan['supports']]==['R','R','L','L']
    np.testing.assert_array_equal(extended['sites'][:2],old_sites)
    assert extended['status']=='CURRENT_SAMPLES_EXTENDED_FIT_AND_ADMISSION_STILL_REQUIRED'
    assert len(state['values'])==2  # No mutation of the cached old bank.


def test_existing_held_target_is_not_promoted_and_wrong_current_values_refuse():
    c=candidate()
    with pytest.raises(ValueError,match='collision'):
        general_refinement_targets([c],[witness(c)],[c['z_ry']],
            state_digest='current',gamma_ry=.02,eta_ry=.1,existing_supports=original_supports())
    plan=general_refinement_targets([c],[witness(c)],[.3j],
        state_digest='current',gamma_ry=.02,eta_ry=.1,existing_supports=original_supports())
    state=dict(state_digest='current',sites=[.3j],values=[np.zeros((1,2,2),complex)],
        fit_positions=[0],held_positions=[],supports=original_supports())
    with pytest.raises(ValueError,match='shape/dtype'):
        append_general_refinement_samples(state,plan,[np.zeros((1,3,3),complex)]*3,
            crossing=lambda value:value.conj())
    state['refinement_rounds']=2
    with pytest.raises(ValueError,match='round/value count'):
        append_general_refinement_samples(state,plan,[np.zeros((1,2,2),complex)]*3,
            crossing=lambda value:value.conj())


def test_incremental_provider_receives_only_new_sites_and_must_bind_certified_current_order():
    c=candidate();old=[np.zeros((1,2,2),complex)]
    state=dict(state_digest='current',sites=[.3j],values=old,fit_positions=[0],held_positions=[],
        supports=original_supports())
    plan=general_refinement_targets([c],[witness(c)],state['sites'],
        state_digest='current',gamma_ry=.02,eta_ry=.1,existing_supports=state['supports'])
    calls=[]
    def provider(sites):
        calls.append(sites.copy())
        assert not np.any(sites==.3j)
        return [np.full((1,2,2),i+1j) for i in range(len(sites))],dict(
            state_digest='current',sites_ry=sites,native_capacity_admitted=True,scalar_rules_certified=True)
    result=evaluate_general_refinement(state,plan,evaluate_current=provider,crossing=lambda value:value.conj())
    assert len(calls)==1 and len(calls[0])==3
    assert result['values'][0] is old[0]
    assert result['status']=='CURRENT_SAMPLES_EXTENDED_FIT_AND_ADMISSION_STILL_REQUIRED'
    def stale(sites):
        values,receipt=provider(sites);receipt['sites_ry']=sites[::-1]
        return values,receipt
    with pytest.raises(ValueError,match='native target/certificate/admission'):
        evaluate_general_refinement(state,plan,evaluate_current=stale,crossing=lambda value:value.conj())
    state['state_digest']='stale'
    before=len(calls)
    with pytest.raises(ValueError,match='BEFORE evaluation'):
        evaluate_general_refinement(state,plan,evaluate_current=provider,crossing=lambda value:value.conj())
    assert len(calls)==before


def test_residue_directions_keep_independent_complex_left_right_and_exact_dark_columns(monkeypatch):
    import jax.numpy as jnp
    from gw.shared_pole_refinement import residue_aligned_directions
    # Exercise the helper's algebra only, without compiling a native program.
    for name in ('sqrt','sum','where'):
        monkeypatch.setattr(jnp,name,getattr(np,name))
    l=np.array([[[1+2j,0],[3-1j,0]],[[0,0],[0,0]]])
    r=np.array([[[4-1j,0],[-2+3j,0]],[[0,0],[0,0]]])
    norm=lambda v:np.sqrt(np.sum(abs(v)**2,axis=-2,keepdims=True))
    ql,qr=residue_aligned_directions(l,r,0,port_norm=norm)
    np.testing.assert_allclose(np.linalg.norm(ql[0]),1.)
    np.testing.assert_allclose(np.linalg.norm(qr[0]),1.)
    assert not np.allclose(ql,qr)
    np.testing.assert_array_equal(ql[1],0)
    np.testing.assert_array_equal(qr[1],0)
    zl,zr=residue_aligned_directions(l,r,1,port_norm=norm)
    np.testing.assert_array_equal(zl,0);np.testing.assert_array_equal(zr,0)


@pytest.mark.parametrize('real', [0.,-.025])
@pytest.mark.parametrize('q,partner', [(0,0),(1,2)])
def test_pure_imaginary_training_deduplicates_samples_not_same_side_mirror_panels(real,q,partner):
    c=dict(candidate(),z_ry=real+.6j,q=q,partner=partner)
    plan=general_refinement_targets([c],[witness(c)],[.3j],existing_supports=original_supports(),
        state_digest='current',gamma_ry=.02,eta_ry=.1)
    assert len(plan['targets'])==5 and plan['new_microscopic_target_count']==3
    assert len(plan['supports'])==4
    pure=[s for s in plan['supports'] if s['z_ry'].real==0]
    assert len(pure)==2 and pure[0]['sample_index']==pure[1]['sample_index']
    assert pure[0]['side']==pure[1]['side']
    assert pure[0]['panel_id']!=pure[1]['panel_id']
    assert [s['q'] for s in pure]==[q,partner]
    assert pure[1]['direction_crossing'] and pure[1]['mirror_from']==pure[0]['panel_id']
    left={s['z_ry'] for s in plan['supports'] if s['side']=='L'}
    right={s['z_ry'] for s in plan['supports'] if s['side']=='R'}
    assert not left&right
    held={t['z_ry'] for t in plan['targets'] if t['role']=='held'}
    assert not held&(left|right) and len(held)==2


def test_shared_same_role_sites_keep_all_candidate_panels_without_duplicate_chi():
    c=candidate();other=dict(c,q=1,partner=2,pole_index=7)
    plan=general_refinement_targets([c,other],[witness(c),witness(other)],[.3j],
        existing_supports=original_supports(),state_digest='current',gamma_ry=.02,eta_ry=.1)
    assert plan['new_microscopic_target_count']==3 and len(plan['targets'])==6
    assert len(plan['supports'])==8
    assert len({s['panel_id'] for s in plan['supports']})==8
    assert {s['candidate_q'] for s in plan['supports']}=={0,1}
    assert len({s['sample_index'] for s in plan['supports']})==4
    assert plan['candidate_sites'][0]['held']==plan['candidate_sites'][1]['held']
    assert plan['candidate_sites'][1]['pole_index']==7


@pytest.mark.parametrize('collision', ['LR','mirrorLR','fitheld'])
def test_multipole_role_and_side_collisions_refuse_before_provider(collision):
    c=candidate();delta=.025
    z=c['z_ry']+delta*((-1+2j) if collision=='fitheld' else (1+1j))
    if collision=='mirrorLR':z=-z.conjugate()
    other=dict(c,z_ry=z,q=1,partner=2)
    with pytest.raises(ValueError,match='collision|confluence'):
        general_refinement_targets([c,other],[witness(c),witness(other)],[.3j],
            existing_supports=original_supports(),state_digest='current',gamma_ry=.02,eta_ry=.1)


def test_old_implicit_mirror_and_near_confluent_supports_are_checked_at_actual_frequencies():
    c=dict(candidate(),z_ry=-.7+.2j)
    original=[dict(sample_index=0,side='L',panel_id='original:mirror',
        z_ry=-.7+.2j,sample_crossing=True,direction_crossing=True)]
    with pytest.raises(ValueError,match='original sample/support collision'):
        general_refinement_targets([c],[witness(c)],[.7+.2j],existing_supports=original,
            state_digest='current',gamma_ry=.02,eta_ry=.1)
    c=candidate();left=c['z_ry']+.025*(1+1j)
    with pytest.raises(ValueError,match='collision|confluence'):
        general_refinement_targets([c],[witness(c)],[left+4*np.finfo(float).eps],
            existing_supports=original_supports(),state_digest='current',gamma_ry=.02,eta_ry=.1)


def test_nearby_held_is_not_claimed_independent_of_another_candidates_training():
    c=candidate();other=dict(c,z_ry=c['z_ry']+.025-.05j+4*np.finfo(float).eps,q=1,partner=2)
    with pytest.raises(ValueError,match='independent held/training collision'):
        general_refinement_targets([c,other],[witness(c),witness(other)],[.3j],
            existing_supports=original_supports(),state_digest='current',gamma_ry=.02,eta_ry=.1)


@pytest.mark.parametrize('change', ['roles','target_index','target_crossing','support_side','old_mirror',
                                  'candidate_q','pole_index','mirror_from','direction_crossing','old_q',
                                  'candidate_map','held_geometry'])
def test_changed_support_or_sample_graph_fails_before_expensive_provider(change):
    c=candidate();old=np.zeros((1,2,2),complex)
    state=dict(state_digest='current',sites=[.3j],values=[old],fit_positions=[0],held_positions=[],
        supports=original_supports())
    state['supports'][0]['q']=0
    plan=general_refinement_targets([c],[witness(c)],state['sites'],existing_supports=state['supports'],
        state_digest='current',gamma_ry=.02,eta_ry=.1)
    if change=='roles':state.update(fit_positions=[],held_positions=[0])
    elif change=='target_index':plan['targets'][0]['sample_index']=99
    elif change=='target_crossing':plan['targets'][1]['crossing_from']=1
    elif change=='support_side':plan['supports'][0]['side']='L'
    elif change=='old_mirror':state['supports'][0]['sample_crossing']=True;state['supports'][0]['z_ry']=.3+.3j
    elif change=='candidate_q':plan['supports'][0]['candidate_q']=2
    elif change=='pole_index':plan['supports'][0]['pole_index']=99
    elif change=='mirror_from':plan['supports'][1]['mirror_from']='other:pole'
    elif change=='direction_crossing':plan['supports'][1]['direction_crossing']=False
    elif change=='old_q':state['supports'][0]['q']=1
    elif change=='candidate_map':plan['candidate_sites'][0]['right']=plan['candidate_sites'][0]['left']
    else:plan['targets'][-2]['z_ry']+=.001
    with pytest.raises(ValueError,match='BEFORE evaluation|collision|support'):
        evaluate_general_refinement(state,plan,
            evaluate_current=lambda sites:pytest.fail('provider ran before graph validation'),crossing=lambda v:v.conj())


def test_same_side_complex_crossing_is_not_a_left_right_swap():
    l=np.array([1,1j])/np.sqrt(2);r=np.array([1,-1j])/np.sqrt(2)
    w=np.outer(l,r.conj())/(.7+.4j);mirror=w.conj()
    assert np.linalg.norm(mirror@r.conj())>1
    assert np.linalg.norm(l@mirror)>1  # (conj(l)).H Wmirror
    np.testing.assert_allclose(r@mirror,0.,atol=1e-15)
    np.testing.assert_allclose(mirror@l.conj(),0.,atol=1e-15)


def test_two_rounds_keep_every_previous_sample_and_support_object_then_stop():
    state=dict(state_digest='current',sites=[.3j],values=[np.zeros((1,2,2),complex)],
        fit_positions=[0],held_positions=[],supports=original_supports())
    for round_index,z in enumerate((.4+.6j,.9+.8j)):
        before_values=list(state['values']);before_supports=list(state['supports'])
        c=dict(candidate(),z_ry=z,round_index=round_index)
        plan=general_refinement_targets([c],[witness(c)],state['sites'],existing_supports=state['supports'],
            state_digest='current',gamma_ry=.02,eta_ry=.1)
        state=append_general_refinement_samples(state,plan,
            [np.full((1,2,2),i+1j) for i in range(plan['new_microscopic_target_count'])],crossing=lambda v:v.conj())
        assert all(a is b for a,b in zip(state['values'],before_values))
        assert all(a is b for a,b in zip(state['supports'],before_supports))
    assert state['refinement_rounds']==2
    with pytest.raises(ValueError,match='BEFORE evaluation'):
        evaluate_general_refinement(state,plan,evaluate_current=lambda s:pytest.fail('third round'),crossing=lambda v:v.conj())


def test_explicit_sides_assemble_nonconfluent_asymmetric_signed_pole_oracle(monkeypatch):
    import gw.shared_pole_pencil as pencil
    from gw.shared_pole_constructor import reduce_general_pencil
    monkeypatch.setattr(pencil,'jnp',np)
    c=candidate();old=np.array([.3j]);supports=original_supports()
    plan=general_refinement_targets([c],[witness(c)],old,existing_supports=supports,
        state_digest='current',gamma_ry=.02,eta_ry=.1)
    sites=np.r_[old,[t['z_ry'] for t in plan['targets']]]
    l=np.array([1,1j])/np.sqrt(2);r=np.array([1,-1j])/np.sqrt(2)
    residue=np.outer(l,r.conj());pole=.7-.1j
    def value(z):return residue/(z-pole)-residue.conj()/(z+pole.conjugate())
    records=supports+plan['supports'];left=[];right=[]
    for s in records:
        z=sites[s['sample_index']]
        direction=np.array([1.,0.]) if s['panel_id'].startswith('original') else (
            r if s['residue_factor']=='right' else l)
        if s.get('direction_crossing'):direction=direction.conj()
        if s['side']=='L':left.append((z,direction,direction.conj()@value(z)))
        else:right.append((z,direction,value(z)@direction))
    mu=np.array([s[0] for s in left]);lam=np.array([s[0] for s in right])
    assert np.min(abs(mu[:,None]-lam[None]))>1e-3
    mm=lambda a,b,transa='N':(a.conj().T if transa=='C' else a)@b
    e,a,b,out=pencil.general_tangential_pencil(
        (mu,np.stack([s[1] for s in left],axis=1),np.stack([s[2] for s in left])),
        (lam,np.stack([s[1] for s in right],axis=1),np.stack([s[2] for s in right],axis=1)),matmul=mm)
    model=reduce_general_pencil(e,a,max_order=2,rank_tolerance=1e-10)
    for z in list(sites)+[.13+.17j,-.37+.29j]:
        fitted=((out@model['output_map'])/(z-model['poles_ry']))@model['input_map'].conj().T@b
        np.testing.assert_allclose(fitted,value(z),rtol=1e-11,atol=1e-11)
