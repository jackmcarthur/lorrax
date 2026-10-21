"""Small metadata/provider contracts; not native or material admission."""
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest


def test_current_identity_binds_U_rho_V_and_never_retains_ephemeral_faces(monkeypatch):
    from gw.shared_pole_screening import _general_refinement_identity
    import gw.response_bank as bank
    import common.contract_bands as bands
    calls=[]
    monkeypatch.setattr(bands,'merge_spin_centroid',lambda value,*axes:value)
    def digest(value,mesh,*,cache):
        assert cache is False
        calls.append(value)
        return hashlib.sha256(memoryview(value)).hexdigest()
    monkeypatch.setattr(bank,'operator_digest',digest)
    current=dict(energy=np.array([[-1.,1.]]),rho=np.array([np.diag([1.,0.])],complex),
        snapshot=dict(U_active=np.array([np.eye(2)],complex),active_slice=slice(0,2)),
        pm=np.ones((1,1,2,2),complex),V=np.array([np.eye(2)],complex),mesh_xy=object(),
        meta=SimpleNamespace(mu_basis=SimpleNamespace(canonical_indices=np.array([[0,0,0],[1,0,0]]),
            layout=SimpleNamespace(axis=SimpleNamespace(packed_to_canonical=np.array([0,1]))))),
        grid=(1,1,1),qgrid=(1,1,1),response_window='sharp',gamma=.02,eta=.04,
        recipe=dict(source_recipe_hash='native-source',recipe_hash='current-method'))
    baseline,receipt=_general_refinement_identity(current)
    for value in (current['snapshot']['U_active'],current['rho'],current['V']):
        saved=value.copy();value.flat[0]+=1e-10
        assert _general_refinement_identity(current)[0]!=baseline
        value[:]=saved
    assert _general_refinement_identity(current)[0]==baseline
    assert all(value is current['pm'] or value is current['V'] for value in calls)
    assert receipt['active_bands']==[0,2]


@pytest.mark.parametrize('status,error,accepted',[
    ('PASS',1e-9,True),('FAIL',1e-9,False),('PASS',2e-8,False),('PASS',np.nan,False)])
def test_provider_uses_actual_native_and_public_rule_receipts(status,error,accepted):
    from gw.shared_pole_screening import _general_refinement_provider_receipt
    current=dict(response_execution=dict(memory=[dict(device_budget_status='PASS'),
        dict(device_budget_status=status)]),recipe=dict(bank_rule_tolerance=1e-8),state_digest='physical')
    rules=[dict(sampled_error=error)]
    if not accepted:
        with pytest.raises(ValueError,match='actual native admission'):
            _general_refinement_provider_receipt(current,[.2+.3j],rules,1)
    else:
        receipt=_general_refinement_provider_receipt(current,[.2+.3j],rules,1)
        assert receipt['native_dispatch_count']==1 and receipt['native_capacity_admitted'] is True
        assert receipt['scalar_rules_certified'] is True


def test_round_phases_keep_local_stream_clock_and_unique_global_names():
    from gw.shared_pole_screening import _general_refinement_phases
    events=[]
    current=dict(begin=lambda name:events.append(('begin',name)),
        end=lambda name,operands:events.append(('end',name,operands)))
    begin,end,clock=_general_refinement_phases(current,'round_01_new')
    begin('stream');assert set(clock)=={'stream'}
    end('stream','runtime-faces');assert not clock
    assert events==[('begin','round_01_new_stream'),('end','round_01_new_stream','runtime-faces')]
