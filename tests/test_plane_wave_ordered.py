"""Independent directed finite-q density and fractional Lehmann controls."""
import jax
import numpy as np
import pytest
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P

from common.gvec_fft_box import build_sphere_box_index
from gw.plane_wave_lehmann import (OrderedLehmannPair,GammaLehmannResponse,
                                  transition_vertices,gamma_transition_vertices)


@pytest.fixture
def mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),("x","y"))


def tile(x,mesh):
    return jax.device_put(np.asarray(x,np.complex128),NamedSharding(mesh,P(("x","y"),None,None,None)))


def plant(mesh,*,spinor=1):
    rng=np.random.default_rng(271004)
    ea=np.asarray([[-.8,.4],[-.55,.2]])
    eb=np.asarray([[-1.2,.1,1.1,9.],[-.7,-.1,.7,9.]])
    de=ea[None]-eb.T[:,:,None]
    fd=lambda e:1/(1+np.exp((e-.13)/.31))
    df=fd(ea)[None]-fd(eb.T)[:,:,None]
    valid=np.ones((4,2,2),bool);valid[3]=False;valid[0,0,1]=False
    v=rng.normal(size=(4,2,2,4))+1j*rng.normal(size=(4,2,2,4))
    v[~valid]=0;v[...,3]=0
    de[~valid]=np.nan;df[~valid]=np.inf
    bank=OrderedLehmannPair(tile(v,mesh),de,df,valid,mesh=mesh,
        cell_volume=11.,n_k=512,n_spinor=spinor,physical_g_count=3,panel_bytes=4096)
    return bank,v,de,df,valid


def oracle(v,de,df,valid,z,scale):
    value=np.zeros((len(z),1,v.shape[-1],v.shape[-1]),complex)
    slope=np.zeros_like(value)
    for b,k,a in np.argwhere(valid):
        density=v[b,k,a];outer=np.outer(density,density.conj())
        for j,site in enumerate(z):
            den=site+de[b,k,a]
            value[j,0]+=df[b,k,a]*outer/den
            slope[j,0]+=-df[b,k,a]*outer/(2*site*den*den)
    return value*scale,slope*scale


@pytest.mark.parametrize("spinor,spin_factor",[(1,2.),(2,1.)])
def test_directed_fractional_complex_response_matches_literal_sum(mesh,spinor,spin_factor):
    bank,v,de,df,valid=plant(mesh,spinor=spinor)
    z=np.asarray([.7+.25j,-.3+.5j,0.+.8j])
    expected,ds=oracle(v,de,df,valid,z,spin_factor/(11*512))
    got,slope=bank.evaluate(z,with_derivative=True)
    np.testing.assert_allclose(got,expected,rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(slope,ds,rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(bank.evaluate(z),expected,rtol=3e-12,atol=3e-12)
    assert np.linalg.norm(expected-expected.swapaxes(-1,-2))>1e-6
    assert np.linalg.norm(expected-expected.swapaxes(-1,-2).conj())>1e-6
    assert np.max(np.abs(np.asarray(got)[...,3,:]))==0.
    assert np.max(np.abs(np.asarray(got)[...,:,3]))==0.
    assert bank.receipt["full_k_normalization"]==512
    assert bank.receipt["physical_pair_count"]==int(valid.sum())


def test_squared_frequency_derivative_and_actual_reverse_tile(mesh):
    bank,v,de,df,valid=plant(mesh)
    z=.7+.25j;s=z*z;h=2e-6
    _,ds=bank.evaluate([z],with_derivative=True)
    samples=np.asarray(bank.evaluate(np.sqrt([s+h,s-h])))
    np.testing.assert_allclose((samples[0]-samples[1])/(2*h),np.asarray(ds)[0],rtol=2e-9,atol=2e-9)
    # A general-q reverse channel needs its own actual density; this witness
    # would reject inferring it from this channel by an adjoint or G-negation.
    reverse=v.copy();reverse[...,0]*=2+1j;reverse[...,1]*=-.7+.3j
    partner=OrderedLehmannPair(tile(reverse,mesh),-de,-df,valid,mesh=mesh,
        cell_volume=11.,n_k=512,physical_g_count=3,panel_bytes=4096)
    expected,_=oracle(reverse,-de,-df,valid,[z],2/(11*512))
    np.testing.assert_allclose(partner.evaluate([z]),expected,rtol=3e-12,atol=3e-12)
    inferred,_=oracle(v[..., [0,2,1,3]].conj(),-de,-df,valid,[z],2/(11*512))
    assert np.linalg.norm(expected-inferred)>1e-5


def test_gamma_paired_owner_preserves_legacy_and_both_explicit_terms(mesh):
    rng=np.random.default_rng(710052)
    g=np.asarray([[0,0,0],[1,0,0],[-1,0,0]],np.int32)
    ea=np.asarray([[-2.,-1.],[-1.9,-.9]])
    eb=np.asarray([[.3,.8,1.7,1e100],[.4,1e100,1.8,-1e100]])
    live=np.asarray([[True,True,True,False],[True,False,True,False]])
    valid=np.broadcast_to(live.T[:,:,None],(4,2,2)).copy()
    v=rng.normal(size=(4,2,2,4))+1j*rng.normal(size=(4,2,2,4))
    v[~valid]=0;v[...,3]=0
    de=np.where(valid,ea[None]-eb.T[:,:,None],0.)
    df=valid.astype(float)
    legacy=GammaLehmannResponse(tile(v,mesh),ea,eb,live,gvecs=g,mesh=mesh,cell_volume=11.,panel_bytes=4096)
    directed=OrderedLehmannPair(tile(v,mesh),de,df,valid,mesh=mesh,cell_volume=11.,physical_g_count=3,panel_bytes=4096)
    z=np.asarray([.7+.25j,0.+.8j])
    a,ad=oracle(v,de,df,valid,z,2/(11*2))
    b,bd=oracle(v[..., [0,2,1,3]].conj(),-de,-df,valid,z,2/(11*2))
    value,slope=directed.evaluate_gamma_pair(z,gvecs=g,with_derivative=True)
    np.testing.assert_allclose(value,a+b,rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(slope,ad+bd,rtol=3e-12,atol=3e-12)
    old,old_ds=legacy.evaluate(z,with_derivative=True)
    np.testing.assert_allclose(old,value,rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(old_ds,slope,rtol=3e-12,atol=3e-12)


@pytest.mark.parametrize("kind",["invalid_pair","g_ghost"])
def test_zero_weight_never_authorizes_a_nonzero_ghost(mesh,kind):
    _,v,de,df,valid=plant(mesh)
    if kind=="invalid_pair":v[3,0,0,0]=1e-30
    else:v[0,0,0,3]=1e-30
    with pytest.raises(ValueError,match="nonzero native-pair or retained-G ghost"):
        OrderedLehmannPair(tile(v,mesh),de,df,valid,mesh=mesh,cell_volume=11.,physical_g_count=3)


@pytest.mark.parametrize("kind",["valid_dtype","shape","physical_nan","df_range","nk","ng"])
def test_physical_metadata_refusals(mesh,kind):
    _,v,de,df,valid=plant(mesh);kw=dict(n_k=512,physical_g_count=3)
    if kind=="valid_dtype":valid=valid.astype(int)
    elif kind=="shape":de=de[:2]
    elif kind=="physical_nan":de[0,0,0]=np.nan
    elif kind=="df_range":df[0,0,0]=1.001
    elif kind=="nk":kw["n_k"]=1
    elif kind=="ng":kw["physical_g_count"]=5
    with pytest.raises(ValueError):
        OrderedLehmannPair(tile(v,mesh),de,df,valid,mesh=mesh,cell_volume=11.,**kw)


def full_bloch(coeff,g,k,grid):
    coordinates=np.indices(grid).reshape(3,-1).T/np.asarray(grid)
    result=np.zeros((coeff.shape[0],coeff.shape[1],len(coordinates)),complex)
    for i,mode in enumerate(g):
        result+=coeff[...,i,None]*np.exp(2j*np.pi*(coordinates@(mode+k)))[None,None]
    return (result/np.sqrt(np.prod(grid))).reshape(1,coeff.shape[0],coeff.shape[1],*grid)


@pytest.mark.parametrize("conjugated",[False,True])
def test_finite_q_phase_removal_preserves_integer_wrap_and_spin_trace(mesh,conjugated):
    rng=np.random.default_rng(100572)
    grid=(8,4,4);g=np.asarray([[-1,0,0],[0,0,0],[1,0,0]],np.int32)
    modes=np.asarray([[i,0,0] for i in range(-3,4)],np.int32)
    left=rng.normal(size=(2,2,3))+1j*rng.normal(size=(2,2,3))
    right=rng.normal(size=(4,2,3))+1j*rng.normal(size=(4,2,3))
    left/=np.sqrt(np.sum(abs(left)**2,axis=(1,2),keepdims=True))
    right/=np.sqrt(np.sum(abs(right)**2,axis=(1,2),keepdims=True))
    q=np.asarray([.25,0.,0.])
    ka=np.asarray([-.375 if conjugated else .375,.125,0.])
    kb=np.asarray([.375 if conjugated else -.375,.125,0.])
    wrap=np.rint((ka-kb if conjugated else kb-ka)-q).astype(int)
    assert np.array_equal(wrap,[-1,0,0])
    a=jax.device_put(full_bloch(left,g,ka,grid),NamedSharding(mesh,P(None,None,None,None,None,None)))
    b=jax.device_put(full_bloch(right,g,kb,grid),NamedSharding(mesh,P(None,("x","y"),None,None,None,None)))
    index=build_sphere_box_index([modes],grid,8)
    actual=transition_vertices(a,b,index,mesh=mesh,q_frac=q,left_k_frac=ka,right_k_frac=kb,
        left_replication_bound_bytes=a.size*a.dtype.itemsize,conjugate_density=conjugated)
    expected=np.zeros((4,2,8),complex)
    for j,mode in enumerate(modes):
        for ia,ga in enumerate(g):
            for ib,gb in enumerate(g):
                frequency=(ga-gb if conjugated else gb-ga)+wrap
                if np.array_equal(frequency,mode):
                    for target in range(2):
                        for intermediate in range(4):
                            expected[intermediate,target,j]+=np.sum(
                                left[target,:,ia]*right[intermediate,:,ib].conj() if conjugated
                                else left[target,:,ia].conj()*right[intermediate,:,ib])
    np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12)
    assert np.max(abs(expected[...,0]))>1e-3  # the retained wrap is observable.
    assert np.max(abs(np.asarray(actual)[...,7]))==0.
    assert actual.sharding==NamedSharding(mesh,P(("x","y"),None,None))
    with pytest.raises(ValueError,match="integer reciprocal wrap"):
        transition_vertices(a,b,index,mesh=mesh,q_frac=q+[.1,0,0],left_k_frac=ka,right_k_frac=kb,
            left_replication_bound_bytes=a.size*a.dtype.itemsize,conjugate_density=conjugated)
    with pytest.raises(ValueError,match="explicit replication bound"):
        transition_vertices(a,b,index,mesh=mesh,q_frac=q,left_k_frac=ka,right_k_frac=kb,
            left_replication_bound_bytes=a.size*a.dtype.itemsize-1,conjugate_density=conjugated)


def test_directed_gamma_density_default_matches_legacy(mesh):
    rng=np.random.default_rng(105911);grid=(4,4,4)
    a=rng.normal(size=(1,2,1,*grid))+1j*rng.normal(size=(1,2,1,*grid))
    b=rng.normal(size=(1,4,1,*grid))+1j*rng.normal(size=(1,4,1,*grid))
    a=jax.device_put(a,NamedSharding(mesh,P(None,None,None,None,None,None)))
    b=jax.device_put(b,NamedSharding(mesh,P(None,("x","y"),None,None,None,None)))
    g=np.asarray([[0,0,0],[1,0,0],[-1,0,0]],np.int32)
    index=build_sphere_box_index([g],grid,4);bound=a.size*a.dtype.itemsize
    old=gamma_transition_vertices(a,b,index,mesh=mesh,occupied_replication_bound_bytes=bound)
    new=transition_vertices(a,b,index,mesh=mesh,q_frac=[0.,0.,0.],left_k_frac=[.25,0.,0.],
        right_k_frac=[.25,0.,0.],left_replication_bound_bytes=bound)
    np.testing.assert_allclose(new,old,rtol=3e-12,atol=3e-12)
