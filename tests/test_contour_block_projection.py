"""Independent ordered finite-G target blocks and causal external transposes."""
import numpy as np
import pytest
import jax
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from common.collectives import gather_to_host
from gw import contour_reference as cd


@pytest.fixture
def mesh():
    devices=np.asarray(jax.devices())
    width=2 if jax.process_count()==4 and len(devices)>=4 else 1
    return Mesh(devices[:width*width].reshape(width,width),('x','y'))


def face(a,mesh):
    return jax.device_put(np.asarray(a,np.complex128),NamedSharding(mesh,P(None,'x','y')))


def literal(operator,pairs,prefactor):
    """Explicit finite-G bilinear contraction, independent of GEMM/reduction."""
    out=np.zeros(pairs.shape[:3]+(pairs.shape[2],),complex)
    for k in range(len(pairs)):
        w=operator[0 if len(operator)==1 else k]
        for m in range(pairs.shape[1]):
            for a in range(pairs.shape[2]):
                for b in range(pairs.shape[2]):
                    for i in range(pairs.shape[3]):
                        for j in range(pairs.shape[3]):
                            out[k,m,a,b]+=pairs[k,m,a,i].conj()*w[i,j]*pairs[k,m,b,j]*prefactor
    return out


def plant():
    rng=np.random.default_rng(3763)
    pairs=rng.normal(size=(3,4,4,4))+1j*rng.normal(size=(3,4,4,4))
    w=rng.normal(size=(3,4,4))+1j*rng.normal(size=(3,4,4))
    pairs[:,:,2:]=0
    return pairs,w


@pytest.mark.parametrize('broadcast',[False,True])
def test_full_block_retains_literal_complex_offdiagonals_and_diagonal_defaults(mesh,broadcast):
    pairs,w=plant()
    if broadcast:w=w[:1]
    rows=pairs.reshape(3,16,4);pref=1/(512*137.)
    product,block=cd.project_interaction_block(face(w,mesh),face(rows,mesh),mesh=mesh,n_targets=4,
        prefactor=pref,scalar_replication_bound_bytes=3*4*4*4*16)
    expected=literal(w,pairs,pref)
    np.testing.assert_allclose(block,expected,atol=2e-15,rtol=3e-13)
    assert product.sharding==NamedSharding(mesh,P(None,'x','y'))
    assert block.sharding==NamedSharding(mesh,P())
    p0,diagonal=cd.project_interaction_diagonal(face(w,mesh),face(rows,mesh),mesh=mesh,
        prefactor=pref,scalar_replication_bound_bytes=3*16*16)
    np.testing.assert_array_equal(gather_to_host(product),gather_to_host(p0))
    np.testing.assert_allclose(np.diagonal(np.asarray(block),axis1=-2,axis2=-1).reshape(3,16),
        diagonal,atol=2e-15,rtol=3e-13)
    assert np.max(abs(expected[:,:,0,1]))>1e-5
    assert np.max(abs(expected-expected.swapaxes(-1,-2)))>1e-5
    np.testing.assert_array_equal(np.asarray(block)[:,:,2:,:],0.)
    np.testing.assert_array_equal(np.asarray(block)[:,:,:,2:],0.)


@pytest.mark.parametrize('convention',['retarded','time_ordered_fractional'])
def test_asymmetric_qminus_block_needs_external_transpose_before_full_pole_integral(mesh,convention):
    rng=np.random.default_rng(3764);perm=np.array([2,0,1,3]);omega=.9;eta=.18;pref=.04
    v=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2));v[3]=0
    u=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2));u[3]=0
    rp=.003*v@v.conj().T;rm=.004*u@u.conj().T
    pairs=rng.normal(size=(1,4,4,4))+1j*rng.normal(size=(1,4,4,4))
    pairs[:,2:]=0;pairs[:,:,2:]=0;pairs[:,:,:,3]=0
    reversed_pairs=pairs[...,perm].conj()
    energy=np.array([.3,-.2,-100.,100.]);f=np.array([1.,.25,.3,.7]);valid=np.array([1,1,0,0],bool)
    evaluations=np.array([-.7,0.,.3,.7])
    x=np.broadcast_to((evaluations[:,None]-energy)[None,None],(1,4,4,4))
    fh=f.reshape(1,1,1,4);mask=valid.reshape(1,1,1,4)
    active,weight=cd._residue_weights(x,fh,None,mask)
    crossings=np.unique(abs(x[active&(weight!=0.)]));nodes,weights=cd.imaginary_rule(192,eta,scale=omega)
    sites=np.concatenate(([1j*eta],1j*nodes,crossings+1j*eta))
    q=-rp[None]/(sites-omega)[:,None,None]+rm[None]/(sites+omega)[:,None,None]
    minus=(-rm.T[None]/(sites-omega)[:,None,None]+rp.T[None]/(sites+omega)[:,None,None])[:,perm][:,:,perm]
    def project(w,p):
        repeated=np.broadcast_to(p,(len(w),)+p.shape[1:]).reshape(len(w),16,4)
        _,b=cd.project_interaction_block(face(w,mesh),face(repeated,mesh),mesh=mesh,n_targets=4,
            prefactor=pref,scalar_replication_bound_bytes=len(w)*4*4*4*16)
        return np.asarray(b).transpose(0,2,3,1)[:,None]
    values=project(q,pairs)
    raw_partner=project(minus,reversed_pairs)
    # The actual -q density roles reverse target endpoints. The projection
    # is transposed only on those endpoints, at the same complex frequency.
    partners=raw_partner.swapaxes(-3,-2)
    z=1j*eta;ds=(rp/(z-omega)**2-rm/(z+omega)**2)/(2*z)
    derivative=project(ds[None],pairs)[0]
    kwargs=dict(eta=eta,band_valid=mask,analytic_convention=convention)
    anchor,cp,cm,beta=cd.anchor_part(values[0],derivative,x,fh,**kwargs)
    result=anchor.copy();wrong=anchor.copy()
    for i,(n,w) in enumerate(zip(nodes,weights),1):
        term=cd.imag_remainder_node(values[i],n,w,x,fh,cp,cm,beta,**kwargs)
        result+=term;wrong+=term
    for i,node in enumerate(crossings,1+len(nodes)):
        result+=cd.real_residue_node(values[i],partners[i],x,fh,node,**kwargs)
        wrong+=cd.real_residue_node(values[i],raw_partner[i],x,fh,node,**kwargs)
    pp=literal(rp[None],pairs,pref).transpose(0,2,3,1)
    pm=literal(rm[None],pairs,pref).transpose(0,2,3,1)
    expected=np.zeros((1,4,4,len(evaluations)),complex)
    sign=-1 if convention=='time_ordered_fractional' else 1
    for m in (0,1):
        expected+=pm[...,m,None]*f[m]/(evaluations-energy[m]+omega+sign*1j*eta)
        expected+=pp[...,m,None]*(1-f[m])/(evaluations-energy[m]-omega+1j*eta)
    np.testing.assert_allclose(result,expected,atol=2e-7*np.max(abs(expected)),rtol=0.)
    assert np.max(abs(wrong-expected))/np.max(abs(expected))>1e-3
    assert np.max(abs(raw_partner-partners))>.001


@pytest.mark.parametrize('count',[True,0,-1,4.,3])
def test_invalid_or_incomplete_target_groups_refuse(mesh,count):
    pairs,w=plant()
    with pytest.raises(ValueError,match='target count|complete internal groups'):
        cd.project_interaction_block(face(w,mesh),face(pairs.reshape(3,16,4),mesh),mesh=mesh,
            n_targets=count,prefactor=1.,scalar_replication_bound_bytes=3072)


@pytest.mark.parametrize('bound',[True,3072.,3071,-1])
def test_exact_block_replication_bound_refuses_before_projection(mesh,bound):
    pairs,w=plant()
    with pytest.raises(ValueError,match='replication bound'):
        cd.project_interaction_block(face(w,mesh),face(pairs.reshape(3,16,4),mesh),mesh=mesh,
            n_targets=4,prefactor=1.,scalar_replication_bound_bytes=bound)
