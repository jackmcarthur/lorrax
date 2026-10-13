"""Explicit Slab current policy tests; no production current admission."""
import copy
import numpy as np
import pytest
from tests.test_static_breit_body_heads import physical_head_fixture,authenticate,rehash


def test_slab_policy_reuses_public_owner_and_distinct_geometry():
    from isdf.atomic_breit import SLAB_BODY_HEAD_SCHEMA,_current_point_body_tile
    from vcoul import get_kernel,v_qG_table
    f=physical_head_fixture(sys_dim=2);p=f['policy'];authenticate(f)
    assert p['schema']==SLAB_BODY_HEAD_SCHEMA and p['sys_dim']==2
    assert p['slab_half_height']==get_kernel(2).truncation_half_height(f['geometry'])
    B=np.asarray([[1,1j,0],[1,-1j,0],[0,0,np.sqrt(2)]])/np.sqrt(2)
    pairs=tuple((a,b) for a in range(3) for b in range(a,3))
    prefactor=-8*np.pi/f['volume']
    got=_current_point_body_tile(p,[2,0,3,1],0,8,pairs,B,prefactor,sys_dim=2)
    for row,q in enumerate([2,0,3,1]):
        if q==3:
            assert not got[row].any();continue
        n=f['counts'][q];K=f['K'][q,:n];length=np.linalg.norm(K,axis=1)
        u=np.divide(K,length[:,None],out=np.zeros_like(K),where=length[:,None]>0)
        cart=np.eye(3)-u[:,:,None]*u[:,None,:]
        projected=np.einsum('ai,gij,bj->gab',B,cart,B.conj())
        scalar=v_qG_table(get_kernel(2),f['q'][q:q+1],f['G'][q:q+1,:,:n],
            geometry=f['geometry'],vcoul_cutoff_ry=p['cutoff_ry'])[0]
        for i,(a,b) in enumerate(pairs):
            np.testing.assert_allclose(got[row,i,:n],-scalar*projected[:,a,b],atol=2e-15,rtol=2e-14)
        assert not got[row,:,n:].any()


@pytest.mark.parametrize('key,value', [('sys_dim',3),('slab_half_height',5.1)])
def test_rehashed_slab_geometry_change_refuses(key,value):
    f=physical_head_fixture(sys_dim=2);p=copy.deepcopy(f['policy']);p[key]=value
    with pytest.raises(ValueError,match='dimensionality or truncation'):authenticate(f,rehash(p))


def test_bulk_default_is_explicitly_same_policy():
    f=physical_head_fixture();g=physical_head_fixture(sys_dim=3)
    assert f['policy']['payload_sha256']==g['policy']['payload_sha256']
    assert 'sys_dim' not in f['policy'] and 'slab_half_height' not in f['policy']
