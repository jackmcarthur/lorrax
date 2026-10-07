"""Independent public scalar mini-BZ heads for static three-current bodies."""
import copy
import numpy as np
import pytest

def physical_head_fixture(*,near_ties=False,embedded_gamma=False):
    from vcoul import CoulombGeometry,get_kernel,head_slot_table,v_qG_table
    from vcoul.minibz import build_v_head_miniBZ_fn_3d
    from isdf.atomic_breit import static_breit_body_head_slots
    volume=1000.;bvec=(2*np.pi/10)*np.eye(3)
    if near_ties:bvec[0,1]=1e-10*bvec[1,1]
    geometry=CoulombGeometry(bvec=bvec,cell_volume=volume)
    q=np.asarray([[0,0,0],[.5,0,0],[.5,.5,0]],float)
    base=np.asarray([[0,0,0],[-1,0,0],[0,-1,0],[-1,-1,0],[1,0,0],[0,1,0],[0,0,1]],np.int32)
    G=np.broadcast_to(base.T,(3,3,7)).copy();counts=np.asarray([7,5,6],np.int32)
    for i,n in enumerate(counts):G[i,:,n:]=0  # Inert padding deliberately mimics a head tie.
    K=(G.transpose(0,2,1)+q[:,None])@bvec
    active=np.arange(7)[None]<counts[:,None];cutoff=2.
    body=active&(np.einsum('qgi,qgi->qg',K,K)<=cutoff)
    callback=build_v_head_miniBZ_fn_3d((6,6,6),bvec,volume,nmc=512,seed=42)
    gamma=None
    if embedded_gamma:
        import hashlib,json
        from pathlib import Path
        import gw.v_q_bispinor as owner
        from vcoul import COULOMB_GAUGE_TT_SIGN
        tensor=COULOMB_GAUGE_TT_SIGN*owner._tt_head_tensor(
            bvec=bvec,cell_volume=volume,sys_dim=3,kgrid=(6,6,6))/volume
        gamma=dict(cartesian_ry=tensor,kgrid=np.asarray([6,6,6],np.int32),
            source_receipt=json.dumps(dict(owner='gw.v_q_bispinor._tt_head_tensor',
                owner_sha256=hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest(),
                head_rule=owner.V_QMUNU_HEAD_RULE,analytic_sphere=True,
                units='signed_physical_cartesian_Ry'),sort_keys=True))
    policy=static_breit_body_head_slots(q_frac=q,gvec_components=G,ngk_per_q=counts,
        geometry=geometry,vcoul_cutoff_ry=cutoff,v_head_fn=callback,gamma_head=gamma)
    averaged=np.zeros((3,7));bare=np.zeros_like(averaged);tables=[]
    for i,n in enumerate(counts):
        args=dict(geometry=geometry,vcoul_cutoff_ry=cutoff)
        averaged[i,:n]=v_qG_table(get_kernel(3),q[i:i+1],G[i:i+1,:,:n],v_head_fn=callback,**args)[0]
        bare[i,:n]=v_qG_table(get_kernel(3),q[i:i+1],G[i:i+1,:,:n],**args)[0]
        tables.append(head_slot_table(get_kernel(3),q[i:i+1],G[i:i+1,:,:n],v_head_fn=callback,**args))
    return dict(policy=policy,q=q,G=G,K=K,counts=counts,volume=volume,
        geometry=geometry,active=active,body=body,averaged=averaged,bare=bare,tables=tables,callback=callback,
        gamma=gamma)

def authenticate(fixture,policy=None):
    from isdf.atomic_breit import _authenticate_body_head_slots
    f=fixture
    return _authenticate_body_head_slots(f['policy'] if policy is None else policy,
        f['K'],f['counts'],f['volume'],f['body'])

def rehash(policy):
    from isdf.atomic_breit import _body_head_digest
    policy['payload_sha256']=_body_head_digest(policy)
    return policy

def test_public_physical_head_factory_and_ties():
    f=physical_head_fixture();p=f['policy'];selected=authenticate(f)
    assert p['mult'].tolist()==[0,2,4]
    assert selected.sum(axis=1).tolist()==[0,2,4]
    assert np.flatnonzero(selected[1]).tolist()==[0,1]
    assert np.flatnonzero(selected[2]).tolist()==[0,1,2,3]
    assert not selected[0].any() and p['v_avg'][0]==0
    assert not np.any(selected&~f['active'])
    assert abs(f['averaged']-f['bare']).max()>1e-5
    for i,n in enumerate(f['counts']):
        t=f['tables'][i];np.testing.assert_array_equal(p['mult'][i:i+1],t.mult)
        np.testing.assert_allclose(p['v_avg'][i:i+1],t.v_avg,rtol=0,atol=0)
        if i:
            wanted=np.flatnonzero(selected[i])
            direct=f['callback'](f['K'][i,wanted]).mean()
            np.testing.assert_allclose(p['v_avg'][i],direct,rtol=0,atol=0)
            np.testing.assert_allclose(f['averaged'][i,wanted],direct,rtol=0,atol=0)
    assert f['averaged'][0,0]==0  # The separate measured current Gamma head is absent here.

def test_public_near_but_unequal_ties_are_admitted():
    """Public tolerance ties average distinct bare values, not only the minimum."""
    f=physical_head_fixture(near_ties=True)
    selected=authenticate(f)
    slots=np.flatnonzero(selected[2]);assert slots.tolist()==[0,1,2,3]
    values=f['bare'][2,slots]
    assert np.ptp(values)>1e-12  # Meaningful regression, not an exactly degenerate tie.
    np.testing.assert_allclose(f['policy']['v_bare'][2],values.mean(),rtol=2e-14,atol=0.)
    np.testing.assert_allclose(f['policy']['v_bare'][2],f['tables'][2].v_bare[0],rtol=0.,atol=0.)

def test_canonical_embedded_current_gamma_is_separate_from_scalar_heads():
    f=physical_head_fixture(embedded_gamma=True);p=f['policy'];selected=authenticate(f)
    assert p['gamma_placement']=='canonical_embedded_physical_current'
    assert not selected[0].any() and p['v_avg'][0]==0
    np.testing.assert_array_equal(p['gamma_head_cartesian'][0],f['gamma']['cartesian_ry'])
    assert not p['gamma_head_cartesian'][1:].any()
    assert np.trace(p['gamma_head_cartesian'][0]).real<0
    assert p['gamma_source_receipt']==f['gamma']['source_receipt']

@pytest.mark.parametrize('change',('double_placement','away_from_Gamma','missing_source','bad_grid'))
def test_canonical_gamma_rebound_policy_refusals(change):
    f=physical_head_fixture(embedded_gamma=True);p=copy.deepcopy(f['policy'])
    if change=='double_placement':p['gamma_placement']='separate_measured_current'
    elif change=='away_from_Gamma':p['gamma_head_cartesian'][1]=p['gamma_head_cartesian'][0]
    elif change=='missing_source':p['gamma_source_receipt']=''
    else:p['gamma_kgrid'][0]=3
    with pytest.raises(ValueError):authenticate(f,rehash(p))

@pytest.mark.parametrize('change',('two_physical_Gamma_slots','near_but_nonzero_Gamma','noninteger_grid'))
def test_canonical_gamma_factory_requires_the_actual_unique_slot(change):
    from isdf.atomic_breit import static_breit_body_head_slots
    f=physical_head_fixture(embedded_gamma=True)
    G=f['G'].copy();q=f['q'].copy();gamma=copy.deepcopy(f['gamma'])
    if change=='two_physical_Gamma_slots':G[0,:,-1]=0
    elif change=='near_but_nonzero_Gamma':q[0,0]=3e-14
    else:gamma['kgrid']=np.asarray([6.,6.,6.])
    with pytest.raises(ValueError):
        static_breit_body_head_slots(q_frac=q,gvec_components=G,ngk_per_q=f['counts'],
            geometry=f['geometry'],vcoul_cutoff_ry=2.,v_head_fn=f['callback'],gamma_head=gamma)

@pytest.mark.parametrize('change',('missing_tie','duplicate_tie','Gamma_double_count','wrong_count',
    'wrong_cell','wrong_cutoff','wrong_bare_scale','wrong_q','wrong_G','wrong_bvec','stale_identity'))
def test_physical_head_refusal_even_with_rebound_hash(change):
    f=physical_head_fixture();p=copy.deepcopy(f['policy'])
    if change=='missing_tie':p['mask'][2,3]=0;p['mult'][2]=3
    elif change=='duplicate_tie':p['sel'][2,3]=p['sel'][2,2]
    elif change=='Gamma_double_count':p['sel'][0,0]=0;p['mask'][0,0]=1;p['mult'][0]=1;p['v_avg'][0]=1.
    elif change=='wrong_count':p['ngk_per_q'][1:]=p['ngk_per_q'][1:][::-1]
    elif change=='wrong_cell':p['cell_volume']*=1.01
    elif change=='wrong_cutoff':p['cutoff_ry']=.2
    elif change=='wrong_bare_scale':p['v_bare'][1]*=1.1
    elif change=='wrong_q':p['q_frac'][1,0]+=.1
    elif change=='wrong_G':p['gvec_components'][1,0,0]+=1
    elif change=='wrong_bvec':p['bvec'][0,0]*=1.01
    else:p['payload_sha256']='0'*64
    if change!='stale_identity':rehash(p)
    with pytest.raises(ValueError):authenticate(f,p)

def test_physical_head_mutation_without_rehash():
    f=physical_head_fixture();p=copy.deepcopy(f['policy']);p['v_avg'][1]*=1.01
    with pytest.raises(ValueError,match='missing or changed'):authenticate(f,p)
