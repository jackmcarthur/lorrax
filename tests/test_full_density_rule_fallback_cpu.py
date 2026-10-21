"""Actual FA planner refusal and certified existing time-panel fallback."""
import numpy as np
import pytest


def test_actual_44band_fa_remote_group_uses_certified_existing_time_panels():
    from common.units import RYD_TO_EV
    from gw.response_bank import full_density_sample_rules
    interval=(.0016086047044431284,5.589443665102052)
    sites=np.array([1.175977882391852j,.5154394035458149j])
    gamma=.05/RYD_TO_EV
    rules=full_density_sample_rules(np.array([[0.,interval[1]]]),sites,
        gamma_ry=gamma,tolerance=1e-8,interval_ry=interval,paired=True,held={})
    assert sum(len(r['times']) for r in rules)==152
    assert all(r['construction']=='continuum_real_time_after_ritz_refusal' for r in rules)
    assert all('remote Ritz certificate failed through 64 nodes:' in r['ritz_refusal'] for r in rules)
    assert max(r['sampled_error'] for r in rules)<1e-8
    assert max(r['coefficient_mass'] for r in rules)<5000.
    lo,hi=rules[0]['interval_ry']
    assert lo==pytest.approx(.0008043023522215642)
    assert hi==pytest.approx(5.590247967454273)
    d=np.unique(np.r_[np.geomspace(lo,hi,300),np.linspace(lo,hi,400)])
    actual=np.zeros((2,2,len(d)),complex);derivative=actual.copy()
    z=sites+1j*gamma
    for rule in rules:
        for side in (0,1):
            t=rule['times'].conj() if side else rule['times']
            phase=np.exp(-(d[:,None]-rule['reference_ry'])*t).T
            weight=rule['weights'][side]
            actual[side]+=weight@phase
            derivative[side]+=(weight*rule['times'][None]/(2*z[:,None]))@phase
    exact=np.stack([1/(z[:,None]-d),-1/(z[:,None]+d)])
    dexact=np.stack([-1/(2*z[:,None]*(z[:,None]-d)**2),
                     1/(2*z[:,None]*(z[:,None]+d)**2)])
    assert np.max(abs(actual-exact)*z.imag[None,:,None])<1e-8
    assert np.max(abs(derivative-dexact)*z.imag[None,:,None]**3)<1e-8


def test_successful_remote_rule_is_unchanged_and_does_not_enter_fallback(monkeypatch):
    import minimax
    from gw.response_bank import full_density_sample_rules
    real=minimax.response_laplace_rule
    seen=[]
    def remote(*a,**kw):
        result=real(*a,**kw);seen.append(result);return result
    monkeypatch.setattr(minimax,'response_laplace_rule',remote)
    monkeypatch.setattr(minimax,'response_bank_rule',lambda *a,**k:pytest.fail('unexpected fallback'))
    rules=full_density_sample_rules(np.array([[0.,5.]]),np.array([.2+.4j]),
        gamma_ry=.03,interval_ry=(2.,5.),paired=True)
    assert len(seen)==len(rules)==1
    np.testing.assert_array_equal(rules[0]['times'],seen[0]['t'])
    exact=-np.stack([seen[0]['projection_value']+seen[0]['odd_projection_value'],
                     seen[0]['projection_value']-seen[0]['odd_projection_value']])
    np.testing.assert_array_equal(rules[0]['weights'],exact)
    assert rules[0]['construction']=='continuum_noncrossing_laplace'
    assert 'ritz_refusal' not in rules[0]


@pytest.mark.parametrize('failure',[RuntimeError('other native failure'),
    ValueError('invalid domain'),ArithmeticError('Loss of positivity')])
def test_only_named_certificate_exhaustion_can_fallback(monkeypatch,failure):
    import minimax
    from gw.response_bank import full_density_sample_rules
    def fail(*a,**kw): raise failure
    monkeypatch.setattr(minimax,'response_laplace_rule',fail)
    monkeypatch.setattr(minimax,'response_bank_rule',lambda *a,**k:pytest.fail('unexpected fallback'))
    with pytest.raises(ValueError,match=str(failure)):
        full_density_sample_rules(np.array([[0.,5.]]),np.array([.2+.4j]),
            gamma_ry=.03,interval_ry=(2.,5.),paired=True)
