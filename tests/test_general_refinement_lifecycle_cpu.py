"""Host protocol fixture for actual orchestration; NOT native/GPU proof.

Only fit, microscopic sampling, dielectric witness and native dispatch are
replaced by declared tiny NumPy providers. The real bounded lifecycle,
planning, sample append, admission-receipt checks and cache policy execute.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def lifecycle(monkeypatch, *, bad_rounds=1, final_error=0., admission='PASS',
              scalar_error=1e-10, inverse_resolved=True, collision=False,run_dir):
    import gw.shared_pole_screening as owner
    import common.collectives as collectives
    import jax.numpy as jnp
    monkeypatch.setattr(collectives,'gather_to_host',lambda a:a)
    monkeypatch.setattr(collectives,'device_put_process_local',lambda a,sh:np.asarray(a))
    monkeypatch.setattr(jnp,'conj',np.conj)
    events=[];sampled=[];fits=[];raw_values={};dyson_inputs=[]
    old=np.ones((1,2,2),complex)
    initial=.4+.6j if collision else .3j
    state=dict(state_digest='current',sites=np.array([initial]),values=[old],
        fit_positions=[0],held_positions=[],supports=[dict(sample_index=0,z_ry=initial,
            side='R',panel_id='original:0')])
    ledger=SimpleNamespace(live_stages=('source',))
    ledger.at_phase=lambda:events.append(('capacity_phase',))
    def reserve(name,**price):
        events.append(('reserve',name,price));return dict(stage=name,**price)
    ledger.reserve=reserve
    def execute(program,args,mode):
        events.append(('execute',mode))
        current['response_execution']['memory'].append(dict(device_budget_status=admission))
        return program(*args)
    def dyson(h,raw):dyson_inputs.append(raw);return raw*2
    current=dict(meta=SimpleNamespace(mu_basis=SimpleNamespace(active_mask=np.ones(2,bool))),
        mesh_xy=SimpleNamespace(shape={'y':1},size=1),nq=1,nmu=2,qgrid=(1,1,1),
        eta=.1,gamma=.02,state_digest='current',label='fixture',run_dir=str(run_dir),
        H=np.eye(2)[None],mm=None,face=None,rep=None,capacity=ledger,tolerance=.02,
        dyson=SimpleNamespace(value=dyson),sample_current=dict(physical_token=object()),
        response_execution=dict(memory=[]),recipe=dict(general_gates={},bank_rule_tolerance=1e-8),
        print_fn=lambda *a:None,fit_basis=dict(spaces={0:'old-general'},equilibrium_column_spaces_v1={0:'EQ'}),
        begin=lambda name:events.append(('begin',name)),end=lambda name,op=None:events.append(('end',name)))
    def fit(context,samples,*,defer_admission):
        assert defer_admission and samples['values'][0] is old
        attempt=len(fits);fits.append(samples)
        if attempt:assert not context['fit_basis']['spaces']
        context['fit_basis']['spaces'][0]='current-general'
        assert context['fit_basis']['equilibrium_column_spaces_v1']=={0:'EQ'}
        pole=(.4+.6j if attempt==0 else .9+.8j) if attempt<bad_rounds else .4-.1j
        factors=np.array([[[1.+1j],[2.-1j]]])
        checks=[dict(q=0,support=i,held=i in samples['held_positions'],error=final_error)
                for i in range(len(samples['sites']))]
        return dict(banks={0:(factors,factors*2,np.array([pole]))},checks=checks,receipts=[],
            nl=2+attempt*2,nr=2+attempt*2)
    def programs(mesh,nq,pole_index,q,width):
        def norms(l,r):return np.linalg.norm(l,axis=-2),np.linalg.norm(r,axis=-2)
        def panels(l,r,minus):
            l=l[:,:,:1];r=r[:,:,:1]
            return l,r,l[minus].conj(),r[minus].conj()
        return norms,panels
    def sample(physical,sites,*,evaluate_dyson=True,**kwargs):
        assert physical is current['sample_current']
        events.append(('sample',))
        sites=np.asarray(sites,complex);sampled.append((sites.copy(),evaluate_dyson))
        assert not np.any(sites==initial)
        current['response_execution']['memory'].append(dict(device_budget_status=admission))
        values=[]
        for z in sites:
            value=np.full((1,2,2),z,complex);raw_values[complex(z)]=value
            values.append(dyson(current['H'],value) if evaluate_dyson else value)
        return dict(values=values,rules=[dict(sampled_error=scalar_error)])
    def epsilon(raw,operators,record,*,root,index,strict):
        assert strict and raw.shape==(1,2,2)
        events.append(('epsilon',record['q'],tuple(record['z_ry'])))
        return dict(witness=dict(record,additional_collision_eta_ry=0.,inverse_bounds=dict(
            resolved=inverse_resolved,dimension=2,padding_defect=0.,
            residual_spectral_upper_bound=1e-14,sigma_min_lower=.03)))
    current['execute_response']=execute
    current['refresh_dyson']=lambda:(events.append(('refresh_dyson',)) or current['dyson'])
    monkeypatch.setattr(owner,'_general_current_fit',fit)
    monkeypatch.setattr(owner,'_general_refinement_programs',programs)
    monkeypatch.setattr(owner,'_general_current_site_samples',sample)
    monkeypatch.setattr(owner,'_save_general_dielectric_probe',epsilon)
    return owner,current,state,events,sampled,fits,raw_values,dyson_inputs


@pytest.mark.parametrize('rounds',[0,1,2])
def test_actual_lifecycle_only_new_sites_exact_raw_reuse_and_explicit_sides(monkeypatch,tmp_path,rounds):
    owner,current,state,events,calls,fits,raw,dyson=lifecycle(monkeypatch,bad_rounds=rounds,run_dir=tmp_path)
    old=state['values'][0];old_support=state['supports'][0]
    fitted,extended=owner._general_current_refinement(current,state)
    assert len(fits)==rounds+1 and len(fitted['refinement_receipts'])==rounds
    assert extended['values'][0] is old and extended['supports'][0] is old_support
    assert current['fit_basis']['equilibrium_column_spaces_v1']=={0:'EQ'}
    assert all(a is b for before,after in zip(fits,fits[1:]) for a,b in zip(before['values'],after['values']))
    requested=[complex(z) for sites,flag in calls for z in sites]
    assert len(requested)==3*rounds and len(set(requested))==len(requested)
    assert [flag for sites,flag in calls]==[False,True]*rounds
    refreshes=[i for i,event in enumerate(events) if event[0]=='refresh_dyson']
    assert len(refreshes)==rounds
    for i in refreshes:
        assert events[i-1][0]=='reserve'
        assert events[i-1][1].endswith('_workspace')
        assert next(event[0] for event in events[i+1:] if event[0] in ('sample','refresh_dyson'))=='sample'
    for sites,flag in calls:
        if not flag:assert any(a is raw[complex(sites[0])] for a in dyson)
    added=extended['supports'][1:]
    assert [s['side'] for s in added]==['R','R','L','L']*rounds
    for report in fitted['refinement_receipts']:
        plan=report['plan']
        assert not plan['authentication_required']
        assert plan['max_new_microscopic_targets']==6
    # The real phase wrapper records each fit separately, never overwriting
    # the initial or preceding round's planning/stream attribution.
    begin=[e[1] for e in events if e[0]=='begin']
    assert len(begin)==len(set(begin))
    assert len(list(tmp_path.glob('fixture_general_refinement/round_*.json')))==rounds


def test_two_round_cap_refuses_without_third_microscopic_refresh(monkeypatch,tmp_path):
    owner,current,state,events,calls,fits,*_=lifecycle(monkeypatch,bad_rounds=3,run_dir=tmp_path)
    with pytest.raises(ValueError,match='two-round current refinement exhausted'):
        owner._general_current_refinement(current,state)
    assert len(fits)==3 and sum(len(sites) for sites,flag in calls)==6


@pytest.mark.parametrize('failure',[dict(final_error=.03),dict(admission='FAIL'),
    dict(scalar_error=2e-8),dict(inverse_resolved=False)])
def test_failed_actual_gate_is_not_turned_into_refinement_success(monkeypatch,tmp_path,failure):
    owner,current,state,events,calls,fits,*_=lifecycle(monkeypatch,**failure,run_dir=tmp_path)
    with pytest.raises(ValueError,match='held_w|provider|nonsingularity NOT established'):
        owner._general_current_refinement(current,state)


def test_geometry_preflight_refuses_before_any_new_chi_or_epsilon(monkeypatch,tmp_path):
    owner,current,state,events,calls,fits,*_=lifecycle(monkeypatch,collision=True,run_dir=tmp_path)
    with pytest.raises(ValueError,match='collision'):
        owner._general_current_refinement(current,state)
    assert not calls and not [e for e in events if e[0]=='epsilon']


def test_actual_caller_selects_only_coherent_native_member_and_keeps_defaults():
    from gw.shared_pole_screening import _general_refinement_selected
    for definite in (False,True):
        for direct in (None,dict(q=0)):
            assert not _general_refinement_selected({},definite=definite,direct_request=direct)
            assert _general_refinement_selected(dict(general_refinement=True),definite=definite,
                direct_request=direct)==(not definite and direct is None)
    with pytest.raises(ValueError,match='boolean'):
        _general_refinement_selected(dict(general_refinement='yes'),definite=False,direct_request=None)
    import tdagw.snapshot_runtime as runtime
    tree=ast.parse(Path(runtime.__file__).read_text())
    assert any(isinstance(n,ast.keyword) and n.arg=='general_refinement'
               and isinstance(n.value,ast.Constant) and n.value.value is True for n in ast.walk(tree))


@pytest.mark.parametrize('definite,explicit',[(False,True),(False,False),(True,True)])
def test_actual_fit_invalidates_only_incompatible_coherent_frequency_axes(definite,explicit):
    import gw.shared_pole_screening as owner
    tree=ast.parse(Path(owner.__file__).read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_general_current_fit')
    branch=next(n for n in ast.walk(function) if isinstance(n,ast.If)
                and 'del spaces[' in ast.unparse(n))
    compatible=dict(left=np.ones((3,2)),right=np.ones((4,2)))
    stale=dict(left=np.ones((7,2)),right=np.ones((8,2)))
    spaces={0:compatible,1:stale}
    scope=dict(definite=definite,samples=dict(supports=[]) if explicit else {},
               spaces=spaces,nl=3,nr=4)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),owner.__file__,'exec'),scope)
    assert spaces[0] is compatible
    if not definite and explicit:assert 1 not in spaces
    else:assert spaces[1] is stale
