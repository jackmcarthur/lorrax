"""Literal f-channel current survives its own compressor without changing charge."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest


def artifact(current_lmax=7):
    value = dict(angular=dict(lebedev_order=23,lmax=6,orthogonality_tolerance=1e-12),
        tables={24:dict(l=np.asarray([0,1,2,3]),kappa=np.asarray([-1,-2,-3,-4]))})
    if current_lmax is not None:
        value['current_angular'] = dict(lmax=current_lmax)
    return value


def test_current_support_is_explicit_and_does_not_change_charge():
    from gw.isdf_augmentation import _current_angular_support
    value=artifact();before=deepcopy(value['angular'])
    control,contract,identity=_current_angular_support(value)
    assert control['lmax']==7 and contract['minimum_current_lmax']==7
    assert value['angular']==before and contract['charge_lmax']==6
    assert contract['shared_endpoint_rule'] is True and len(identity)==64
    assert contract['support_scope']=='atomic_channel_products_only'
    assert contract['smooth_cross_convergence']=='not_certified'
    with pytest.raises(ValueError,match='current_angular_support'):
        _current_angular_support(artifact(None))
    # The unchanged default still covers a represented spd current family.
    old=artifact(None);old['tables'][24]=dict(l=np.asarray([0,1,2]),kappa=np.asarray([-1,-2,-3]))
    assert _current_angular_support(old)[0]['lmax']==6


def test_current_support_refuses_an_arbitrary_nonzero_kappa():
    from gw.isdf_augmentation import _current_angular_support
    value=artifact();value['tables'][24]['kappa'][-1]=7
    with pytest.raises(ValueError,match='invalid relativistic kappa'):
        _current_angular_support(value)


@pytest.mark.parametrize('control',[dict(lmax=True),dict(lmax=7.0),dict(lmax=-1),
    dict(lmax=7,lebedev_order=23),dict(lmax=12)])
def test_current_extent_cannot_silently_change_the_endpoint_rule(control):
    from gw.isdf_augmentation import _current_angular_control
    value=artifact();value['current_angular']=control
    with pytest.raises(ValueError):
        _current_angular_control(value)


@pytest.mark.parametrize('mutation',['identity','extent','harmonic_rows'])
def test_current_support_refuses_changed_live_handoff(mutation):
    from gw.isdf_augmentation import _current_angular_support,_authenticate_current_angular_support
    _,contract,identity=_current_angular_support(artifact())
    lm=np.asarray([(l,m) for l in range(8) for m in range(-l,l+1)],np.int32)
    state=dict(angular_support_contract=contract,angular_support_identity=identity,lm=lm)
    assert _authenticate_current_angular_support(state)==identity
    if mutation=='identity':state['angular_support_identity']='0'*64
    elif mutation=='extent':state['angular_support_contract']['current_lmax']=6
    else:state['lm']=lm[:-1]
    with pytest.raises(ValueError,match='support|harmonic rows'):
        _authenticate_current_angular_support(state)


def literal_f_current(directions):
    """Independent normalized angular j=m=7/2, kappa=-4 spinor."""
    from scipy.special import sph_harm_y
    theta=np.arccos(directions[:,2]);phi=np.arctan2(directions[:,1],directions[:,0])
    psi=np.zeros((4,len(directions)),complex)
    psi[0]=sph_harm_y(3,3,theta,phi)
    psi[2]=-1j*sph_harm_y(4,3,theta,phi)/3
    psi[3]=1j*np.sqrt(8/9)*sph_harm_y(4,4,theta,phi)
    sigma=np.asarray([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])
    return (np.einsum('sd,ast,td->ad',psi[:2].conj(),sigma,psi[2:])
           +np.einsum('sd,ast,td->ad',psi[2:].conj(),sigma,psi[:2]))


def test_literal_L7_current_through_existing_compressor_and_radial_storage():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from gw import isdf_augmentation as stage
    from common.staged_reshard import face_to_batch_reshard
    value=artifact()
    directions,weights,charge_lm,charge_Y,_=stage._orbit_angular_quadrature(value['angular'],np.eye(3)[None])
    control,_,_=stage._current_angular_support(value)
    current_lm,current_Y,_=stage._angular_harmonics(control,directions,weights)
    current=literal_f_current(directions)
    weighted=lambda a:float(np.sqrt(np.sum(abs(a)**2*weights)))
    charge_coeff=(current*weights)@charge_Y.conj().T
    current_coeff=(current*weights)@current_Y.conj().T
    assert weighted(current-charge_coeff@charge_Y)/weighted(current)>0.01
    assert weighted(current-current_coeff@current_Y)/weighted(current)<2e-12
    assert len(charge_lm)==49 and len(current_lm)==64
    # Permute physical rows to exercise canonical direction/radius buckets;
    # the current compressor must retain L7 even with a q ghost and padding.
    nq,nmu,rp,nr=2,3,2,5;nang=len(weights)
    logical=rp*nang;permutation=np.arange(logical)[::-1]
    packed=np.r_[permutation,-1,-1]
    axis=SimpleNamespace(packed_to_canonical=packed,active_mask=packed>=0,n_logical=logical)
    plan=SimpleNamespace(layout=SimpleNamespace(axis=axis))
    # Current Cartesian component is selected by mu; both q and radius
    # carry independent amplitudes to detect a harmonic/radius transpose.
    canonical=np.einsum('q,ad,r->qard',np.asarray([1.,3.]),current,np.asarray([1.,2.]))
    canonical=canonical.reshape(nq,nmu,logical)
    raw=np.zeros((nq,nmu,len(packed)),complex);raw[...,:logical]=canonical[...,permutation]
    mesh=Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1,1),('x','y'))
    raw=jax.device_put(raw,NamedSharding(mesh,P(None,'x','y')))
    compressor,nf,_=stage._compress_rhs_kernel(mesh,plan,1,len(current_lm),rp,current_Y.conj()*weights)
    selector,store=stage._rhs_storage_kernels(mesh,np.arange(nq),nq+1,1,len(current_lm),nr,rp,nf)
    packet=face_to_batch_reshard(mesh)(compressor(selector(raw)))
    zero=jnp.zeros((nq+1,nmu,len(current_lm)*nr),jnp.complex128,
        device=NamedSharding(mesh,P(('x','y'),None,None)))
    result=np.asarray(store(zero,packet,jnp.int32(1))).reshape(nq+1,nmu,len(current_lm),nr)
    expected=np.zeros_like(result)
    expected[:nq,:,:,1:3]=np.einsum('q,ah,r->qahr',np.asarray([1.,3.]),current_coeff,np.asarray([1.,2.]))
    np.testing.assert_allclose(result,expected,rtol=2e-12,atol=2e-13)
    assert np.max(abs(result[-1]))==0
    assert np.linalg.norm(result[:,:,current_lm[:,0]==7])>1e-3
