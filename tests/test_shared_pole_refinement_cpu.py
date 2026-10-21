"""Bounded refinement contracts only; no native/GPU execution claim."""
import numpy as np
import pytest

from gw.shared_pole_refinement import (general_refinement_candidates,
    authenticate_false_pole,general_refinement_targets,append_general_refinement_samples,
    evaluate_general_refinement)


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


def test_current_incremental_samples_keep_original_operands_roles_and_crossing_association():
    c=candidate();old_sites=np.array([.3j,.2+.1j])
    plan=general_refinement_targets([c],[witness(c)],old_sites,
        state_digest='current',gamma_ry=.02,eta_ry=.1)
    assert plan['status']=='PLANNED_NOT_EXECUTED_NOT_ADMITTED'
    assert plan['new_microscopic_target_count']==2
    assert [t['role'] for t in plan['targets']]==['fit','fit','held','held']
    rng=np.random.default_rng(318)
    old=[rng.normal(size=(3,2,2)).astype(complex) for _ in old_sites]
    new=[rng.normal(size=(3,2,2)).astype(complex) for _ in range(2)]
    minus=np.array([0,2,1]);calls=[]
    def crossing(value):
        calls.append(value)
        return value[minus].conj()
    state=dict(state_digest='current',sites=old_sites,values=old,fit_positions=[0],held_positions=[1])
    extended=append_general_refinement_samples(state,plan,new,crossing=crossing)
    assert extended['values'][0] is old[0] and extended['values'][1] is old[1]
    assert extended['values'][2] is new[0] and extended['values'][4] is new[1]
    assert calls[0] is new[0] and calls[1] is new[1]
    np.testing.assert_array_equal(extended['values'][3],new[0][minus].conj())
    np.testing.assert_array_equal(extended['values'][5],new[1][minus].conj())
    assert extended['fit_positions']==[0,2,3] and extended['held_positions']==[1,4,5]
    np.testing.assert_array_equal(extended['sites'][:2],old_sites)
    assert extended['status']=='CURRENT_SAMPLES_EXTENDED_FIT_AND_ADMISSION_STILL_REQUIRED'
    assert len(state['values'])==2  # No mutation of the cached old bank.


def test_existing_held_target_is_not_promoted_and_wrong_current_values_refuse():
    c=candidate()
    with pytest.raises(ValueError,match='collision'):
        general_refinement_targets([c],[witness(c)],[c['z_ry']],
            state_digest='current',gamma_ry=.02,eta_ry=.1)
    plan=general_refinement_targets([c],[witness(c)],[.3j],
        state_digest='current',gamma_ry=.02,eta_ry=.1)
    state=dict(state_digest='current',sites=[.3j],values=[np.zeros((1,2,2),complex)],
        fit_positions=[0],held_positions=[])
    with pytest.raises(ValueError,match='shape/dtype'):
        append_general_refinement_samples(state,plan,[np.zeros((1,3,3),complex)]*2,
            crossing=lambda value:value.conj())
    state['refinement_rounds']=2
    with pytest.raises(ValueError,match='round/value count'):
        append_general_refinement_samples(state,plan,[np.zeros((1,2,2),complex)]*2,
            crossing=lambda value:value.conj())


def test_incremental_provider_receives_only_new_sites_and_must_bind_certified_current_order():
    c=candidate();old=[np.zeros((1,2,2),complex)]
    state=dict(state_digest='current',sites=[.3j],values=old,fit_positions=[0],held_positions=[])
    plan=general_refinement_targets([c],[witness(c)],state['sites'],
        state_digest='current',gamma_ry=.02,eta_ry=.1)
    calls=[]
    def provider(sites):
        calls.append(sites.copy())
        assert not np.any(sites==.3j)
        return [np.full((1,2,2),i+1j) for i in range(len(sites))],dict(
            state_digest='current',sites_ry=sites,native_capacity_admitted=True,scalar_rules_certified=True)
    result=evaluate_general_refinement(state,plan,evaluate_current=provider,crossing=lambda value:value.conj())
    assert len(calls)==1 and len(calls[0])==2
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
