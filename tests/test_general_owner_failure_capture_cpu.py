"""Canonical failure metadata and bounded host-frequency I/O only."""
import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_canonical_failure_has_machine_readable_owner_on_every_rank(monkeypatch):
    import common.collectives as cc
    from gw.shared_pole_constructor import _share_general_pencil
    rank=[0];saved={}
    monkeypatch.setattr(cc,'process_rank',lambda:rank[0])
    def gather(a):
        if rank[0]==0:
            saved[a.ndim]=a.copy()
        root=saved[a.ndim]
        return np.stack((root,np.zeros_like(root)))
    monkeypatch.setattr(cc,'all_gather_processes',gather)
    errors=[]
    for r in (0,1):
        rank[0]=r
        with pytest.raises(ValueError,match='GATE shared_general_owner: owner_rank=0') as caught:
            _share_general_pencil(None,ValueError('live H-null') if r==0 else None,
                owner_rank=0,price={},shape=(3,3))
        errors.append(caught.value)
        assert caught.value.canonical_host_failure is True
        assert caught.value.owner_rank==0 and caught.value.reduced_shape==(3,3)
    assert str(errors[0])==str(errors[1])


def test_failed_parent_capture_uses_owner_not_q0_no_overwrite_and_no_numerical_gather(tmp_path,monkeypatch):
    import common.collectives as cc
    from gw.shared_pole_screening import _save_equilibrium_owner_failure
    calls=[]
    def transaction(path,*,stage,write):
        calls.append(stage);write()
    monkeypatch.setattr(cc,'rank0_transaction',transaction)
    monkeypatch.setattr(cc,'gather_to_host',lambda *a,**k:pytest.fail('failure capture must not gather'))
    pencils=[(np.eye(3,dtype=complex)*i,np.eye(3,dtype=complex)*(i+1)) for i in (1,7)]
    grams=[np.eye(3,dtype=complex)*i for i in (2,8)]
    error=ValueError('GATE shared_general_owner: owner_rank=1, live H-null')
    error.canonical_host_failure=True;error.owner_rank=1;error.reduced_shape=(3,3)
    capture=dict(root=tmp_path,retained_storage_bytes=1024,storage_bound_bytes=100000,
        state_units=np.ones(3),sites=np.array([.1j]),metadata=dict(gamma_ry=.02,recipe_hash='current'))
    original=tmp_path/'q_000_equilibrium_pencil.npz';original.write_bytes(b'original')
    path=_save_equilibrium_owner_failure(pencils,grams,[0,7],error,capture)
    assert path.endswith('q_007_equilibrium_failure.npz')
    with np.load(path,allow_pickle=False) as data:
        np.testing.assert_array_equal(data['G'],pencils[1][0])
        np.testing.assert_array_equal(data['H'],pencils[1][1])
        np.testing.assert_array_equal(data['output_gram'],grams[1])
        meta=json.loads(str(data['metadata_json']))
    assert meta['q']==7 and meta['owner_rank']==1 and meta['recipe_hash']=='current'
    assert not meta['admitted'] and not meta['full_microscopic_host_payload']
    assert original.read_bytes()==b'original'
    assert calls==['shared_equilibrium.failed_owner_capture']
    with pytest.raises(FileExistsError):
        _save_equilibrium_owner_failure(pencils,grams,[0,7],error,capture)


def test_uncoordinated_failure_and_unpriced_storage_never_enter_transaction(tmp_path,monkeypatch):
    import common.collectives as cc
    from gw.shared_pole_screening import _save_equilibrium_owner_failure
    monkeypatch.setattr(cc,'rank0_transaction',lambda *a,**k:pytest.fail('no collective I/O allowed'))
    arrays=[(np.eye(2,dtype=complex),np.eye(2,dtype=complex))]
    capture=dict(root=tmp_path,retained_storage_bytes=0,storage_bound_bytes=1,
        state_units=np.ones(2),sites=np.array([.1j]),metadata={})
    with pytest.raises(ValueError,match='synchronized canonical'):
        _save_equilibrium_owner_failure(arrays,[arrays[0][0]],[0],ValueError('rank-local'),capture)
    error=ValueError('owner');error.canonical_host_failure=True;error.owner_rank=0;error.reduced_shape=(2,2)
    with pytest.raises(ValueError,match='storage bound'):
        _save_equilibrium_owner_failure(arrays,[arrays[0][0]],[0],error,capture)
