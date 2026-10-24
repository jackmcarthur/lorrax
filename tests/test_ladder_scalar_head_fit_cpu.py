"""The tangential-only head publishes a measured shared positive model."""
import numpy as np
import pytest


def _recipe():
    points=[.3+.15j,.8+.15j,1.5+.15j,.55+.15j,.4j]
    return dict(eta_ev=.25,bank_rule_tolerance=1e-8,direction_cutoff=1e-3,
        accuracy='production',fit_ids=[0,1,2],held_ids=[3,4],
        z_ry=points,distinct_id=list(range(len(points))))


def _oracle():
    omega=np.array([.07,.23,.9,2.7]);residue=np.array([.03,.17,.8,.3])
    seen=[]
    def evaluate(z):
        seen.extend(map(complex,z))
        s=z[:,None]**2
        return dict(z=z,Wc=np.sum(2*omega*residue/(s-omega**2),axis=1),
            dWc_ds=-np.sum(2*omega*residue/(s-omega**2)**2,axis=1),
            vc0=12.,M1=np.sum(omega*residue),M3=np.sum(omega**3*residue))
    return evaluate,seen,omega,residue


def test_ladder_head_fit_has_physical_measure_and_independent_certificates():
    import jax
    from gw.ladder_head_fit import fit_ladder_head_actions,HEAD_FIT_POLICY
    jax.config.update('jax_enable_x64',True)
    evaluate,seen,omega,residue=_oracle()
    result,state=fit_ladder_head_actions(evaluate,_recipe(),[0j,.4+.2j,1j,2+.2j],gap=.12)
    assert len(seen)==len(set(seen))
    assert result['provenance']['head_fit_policy']==HEAD_FIT_POLICY
    assert not set(result['provenance']['fit_z']) & set(result['provenance']['held_z'])
    assert result['diagnostics']['certificate']['static_relative']<=1e-10
    assert result['diagnostics']['certificate']['inverse_m1_relative']<=1e-8
    assert result['provenance']['moment_convergence']<=1e-8
    assert result['completion'] is True
    points=np.linspace(.005,3.4,71)+.013j
    expected=evaluate(points)['Wc']
    o,b=result['Omega_p'],result['B_p']
    actual=np.sum(2*o*b/(points[:,None]**2-o**2),axis=1)
    np.testing.assert_allclose(actual,expected,rtol=2e-8,atol=2e-8)
    np.testing.assert_allclose(np.sum(b/o**2),np.sum(residue/omega**2),rtol=1e-8)
    assert state['cache'][0j]['Wc'].real<0


def test_ladder_head_fit_refuses_operator_drift_across_observation_batches():
    import jax
    from gw.ladder_head_fit import fit_ladder_head_actions
    jax.config.update('jax_enable_x64',True)
    evaluate,*_= _oracle()
    calls=[]
    def changed(z):
        row=evaluate(z)
        if calls:
            row['M1']*=1.01
        calls.append(len(z))
        return row
    with pytest.raises(ValueError,match='GATE ladder_head_operator: changed M1'):
        fit_ladder_head_actions(changed,_recipe(),[0j,.4+.2j,1j,2+.2j],gap=.12)


def test_head_held_role_survives_coincident_legacy_fit_and_gap_midpoint():
    import jax
    from common.units import RYD_TO_EV
    from gw.ladder_head_fit import fit_ladder_head_actions
    jax.config.update('jax_enable_x64',True)
    recipe=_recipe()
    coincident=.4+.2j
    midpoint=.5*complex(.12,recipe['eta_ev']/RYD_TO_EV)
    recipe['z_ry'][3:]=[coincident,midpoint]
    evaluate,*_=_oracle()
    model,_=fit_ladder_head_actions(evaluate,recipe,[0j,coincident,1j,2+.2j],gap=.12)
    assert coincident not in model['provenance']['fit_z']
    assert coincident in model['provenance']['held_z']
    assert midpoint not in model['provenance']['fit_z']
    assert midpoint in model['provenance']['held_z']


def test_exact_multimode_head_refines_failed_held_sites_and_honors_site_cap(monkeypatch):
    import jax
    from gw.ladder_head_fit import fit_ladder_head_actions
    from gw.tangential_recipe import shared_real_pole_v1_r3b
    jax.config.update('jax_enable_x64',True)
    omega=np.geomspace(.025,2.7,40);residue=np.full_like(omega,.001)
    def evaluate(z):
        s=z[:,None]**2;weight=2*omega*residue
        return dict(z=z,Wc=np.sum(weight/(s-omega**2),axis=1),
            dWc_ds=-np.sum(weight/(s-omega**2)**2,axis=1),vc0=5.,
            M1=np.sum(omega*residue),M3=np.sum(omega**3*residue))
    model,_=fit_ladder_head_actions(evaluate,_recipe(),[0j,.4+.2j,1j,2+.2j],gap=.12)
    assert model['provenance']['promotions']
    assert not set(model['provenance']['fit_z']) & set(model['provenance']['held_z'])
    z=np.linspace(.005,.5,41)+.018j
    o,b=model['Omega_p'],model['B_p']
    actual=np.sum(2*o*b/(z[:,None]**2-o**2),axis=1)
    np.testing.assert_allclose(actual,evaluate(z)['Wc'],rtol=2e-3,atol=2e-6)
    # A smaller inherited site budget must refuse, rather than publishing
    # the first unresolved positive model or fitting its held data silently.
    monkeypatch.setitem(shared_real_pole_v1_r3b,'production',
        dict(shared_real_pole_v1_r3b['production'],sample_cap=2))
    with pytest.raises(ValueError,match='GATE ladder_head_refinement_cap'):
        fit_ladder_head_actions(evaluate,_recipe(),[0j,.4+.2j,1j,2+.2j],gap=.12)


def test_certified_head_wiring_preserves_current_body_identity_without_mpa_refit(monkeypatch):
    import dataclasses
    from types import SimpleNamespace as NS
    from gw.qsgw_head import IterationHeadSamples
    from gw.head_correction import HeadSample,HeadResponseKind
    from gw.gw_config import HeadCorrection
    from gw.ladder_head_fit import HEAD_FIT_POLICY
    from gw.tangential_head import build_tangential_head
    import gw.mpa.model as mpa
    def forbidden(*args,**kwargs):
        raise AssertionError('a certified shared head must not be residue-refitted')
    monkeypatch.setattr(mpa,'fit_head_samples',forbidden)
    identity=dict(wavefunctions='WFN',screening_diagrams='w_bse',ladder_static_w_sha256='rungA')
    model=dict(Omega_p=np.array([.2+0j]),B_p=np.array([.1+0j]),completion=True,
        provenance=dict(head_fit_policy=HEAD_FIT_POLICY,head_identity=dict(
            head_fit_policy=HEAD_FIT_POLICY,ladder_static_w_sha256='rungA')))
    sample=HeadSample(2+0j,1+0j,'physical',0j,response_kind=HeadResponseKind.MICRO_REDUCIBLE)
    response=IterationHeadSamples((0j,),(sample,),np.empty((0,0)),np.empty((0,0)),0.,model)
    wfns=NS(enk=np.arange(6.).reshape(2,3),occ=np.ones((2,3)),slices=NS(sigma=slice(1,3)))
    cfg=NS(head=NS(uses_bgw_metal_q0shift=False,correction=HeadCorrection.FULL),
           screening=NS(diagrams='w_bse'))
    meta=NS(tangential_recipe=dict(census=dict(mu_ry=.1)))
    kwargs=dict(mesh_xy=None,wfn=None,response=response,head_resolver=None,plan=None,
                material_class='insulator',occupation_state=None)
    head,current=build_tangential_head(dict(identity=identity,digest='body'),None,None,wfns,meta,cfg,**kwargs)
    np.testing.assert_array_equal(head['Omega_p'],model['Omega_p'])
    assert head['provenance']['head_identity']['wavefunctions']=='WFN'
    np.testing.assert_array_equal(current.sigma_energies_ry,wfns.enk[:,1:])
    stale=dict(model,provenance=dict(model['provenance'],head_identity=dict(
        model['provenance']['head_identity'],ladder_static_w_sha256='other')))
    kwargs['response']=dataclasses.replace(response,scalar_model=stale)
    with pytest.raises(ValueError,match='GATE tangential_head_identity'):
        build_tangential_head(dict(identity=identity,digest='body'),None,None,wfns,meta,cfg,**kwargs)


def test_captured_actions_select_inverse_reference_without_ill_conditioned_grid_fit():
    """Measured contribution selection meets all budgets on real captured actions."""
    import json
    from pathlib import Path
    import jax
    from gw.ladder_head_fit import _inverse_reference_site
    from gw.qsgw_head import head_quadrature_rule
    from gw.tangential_head_scalar import scalar_head_ritz,scalar_head_certificate
    jax.config.update('jax_enable_x64',True)
    def decode(v):
        if isinstance(v,dict):
            return complex(v['real'],v['imag']) if set(v)=={'real','imag'} else {k:decode(x) for k,x in v.items()}
        return [decode(x) for x in v] if isinstance(v,list) else v
    raw=decode(json.loads((Path(__file__).parent/'data/ladder_head_conditioning_actions.json').read_text()))
    obs=raw['observations'];z=np.asarray(obs['z']);values=np.asarray(obs['Wc']);slopes=np.asarray(obs['dWc_ds'])
    quadrature=set(map(complex,np.r_[*[1j*head_quadrature_rule(n)[0] for n in (24,48,96)]]))
    fit={complex(p) for p in z[:raw['fit_prefix']] if complex(p) not in quadrature}
    held=set(map(complex,z))-fit
    xi,weights=head_quadrature_rule(raw['reference_order']);points=tuple(map(complex,1j*xi))
    promoted=[]
    for _ in range(6):
        ids=np.array([i for i,p in enumerate(z) if complex(p) in fit])
        model=scalar_head_ritz(z[ids],values[ids],slopes[ids],moment_m1=obs['M1'],moment_m3=obs['M3'],gram_keep=1e-12)
        certificate=scalar_head_certificate(model,obs,eta_ry=raw['eta_ry'],static_value=raw['static_value'],inverse_m1=raw['inverse_m1'])
        errors=np.maximum(certificate['sample_value_relative'],certificate['sample_slope_relative'])
        if certificate['inverse_m1_relative']<=1e-8 and np.max(errors)<=1e-3:
            break
        if certificate['inverse_m1_relative']>1e-8:
            point,contribution=_inverse_reference_site(obs,certificate,(points,xi,weights),held,agree=np.asarray)
            assert point in points and contribution>0
        else:
            # An independently failing held slope still cannot be published.
            # This component check uses other captured held observations;
            # the full-controller oracle above checks newly solved guards.
            i=max((i for i,p in enumerate(z) if complex(p) in held),key=lambda i:errors[i])
            assert errors[i]>1e-3
            point=complex(z[i])
        assert point in held
        # The selected observation was never fitted before this failed
        # inverse certificate. No pole/residue constraint is applied.
        held.remove(point);fit.add(point);promoted.append(point)
    else:
        raise AssertionError('captured inverse reference did not converge')
    assert promoted and held
    assert max(certificate['sample_value_relative'])<=1e-3
    assert max(certificate['sample_slope_relative'])<=1e-3
    assert certificate['static_relative']<=1e-10
    assert certificate['inverse_m1_relative']<=1e-8
    assert all(max(v)<=1e-8 for row in certificate['high_frequency'].values() for v in row.values())
    assert len(fit)<raw['fit_prefix']//2  # Observed but redundant reference grid was not stuffed into FIT.
