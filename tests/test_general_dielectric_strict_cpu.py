"""Strict scalar witness lifecycle only; no physical/native qualification."""
import pytest


def test_strict_inverse_failure_propagates_after_basic_without_failure_collective(tmp_path,monkeypatch):
    import gw.shared_pole_screening as owner
    import common.collectives as collectives
    calls=[]
    monkeypatch.setattr(owner,'_general_dielectric_basic',lambda *a:dict(q=3,admitted=False))
    failure=ArithmeticError('native inverse unresolved')
    def inverse(*a):raise failure
    monkeypatch.setattr(owner,'_general_dielectric_inverse',inverse)
    def save(path,*,stage,write):calls.append(stage);write()
    monkeypatch.setattr(collectives,'rank0_transaction',save)
    with pytest.raises(ArithmeticError) as caught:
        owner._save_general_dielectric_probe(None,{},dict(q=3),root=tmp_path,index=0,strict=True)
    assert caught.value is failure
    assert calls==['shared_general.dielectric_basic']
    assert (tmp_path/'q_003_dielectric_00_basic.json').exists()
    assert not (tmp_path/'q_003_dielectric_00_inverse.json').exists()


def test_legacy_diagnostic_records_inverse_error_but_strict_success_returns_same_witness(tmp_path,monkeypatch):
    import gw.shared_pole_screening as owner
    import common.collectives as collectives
    monkeypatch.setattr(owner,'_general_dielectric_basic',lambda *a:dict(q=0,admitted=False))
    monkeypatch.setattr(collectives,'rank0_transaction',lambda path,stage,write:write())
    def fail(*a):raise ArithmeticError('original diagnostic failure')
    monkeypatch.setattr(owner,'_general_dielectric_inverse',fail)
    old=owner._save_general_dielectric_probe(None,{},dict(q=0),root=tmp_path/'old',index=0)
    assert set(old)=={'dielectric_basic_file','dielectric_inverse_file'}
    witness=dict(q=0,inverse_bounds=dict(resolved=True,sigma_min_lower=.1))
    monkeypatch.setattr(owner,'_general_dielectric_inverse',lambda *a:witness)
    new=owner._save_general_dielectric_probe(None,{},dict(q=0),root=tmp_path/'new',index=0,strict=True)
    assert new['witness']['inverse_bounds'] is witness['inverse_bounds']
