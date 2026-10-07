"""Independent finite-G ordered correlation projections and resource guards."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P

from common.units import RYD_TO_EV
from gw.contour_reference import project_interaction_diagonal
from gw import contour_reference as cd


@pytest.fixture
def mesh():
    devices=np.asarray(jax.devices())
    # The native distributed matmul communicator has one process per cell;
    # lx test's one-process/four-device mesh is not that execution geometry.
    if jax.process_count()==4 and devices.size>=4:
        return Mesh(devices[:4].reshape(2,2),('x','y'))
    return Mesh(devices[:1].reshape(1,1),('x','y'))


def face(value,mesh):
    return jax.device_put(np.asarray(value,np.complex128),NamedSharding(mesh,P(None,'x','y')))


def plant(batch=3):
    rng=np.random.default_rng(718345)
    pair=rng.normal(size=(batch,2,4))+1j*rng.normal(size=(batch,2,4))
    operator=rng.normal(size=(batch,4,4))+1j*rng.normal(size=(batch,4,4))
    return pair,operator


def independent_projection(operator,pair,prefactor):
    """Literal independently ordered finite-G scalar sum, no matrix kernel."""
    value=np.zeros(pair.shape[:2],np.complex128)
    for b in range(pair.shape[0]):
        interaction=operator[0 if operator.shape[0]==1 else b]
        for target in range(pair.shape[1]):
            for left in range(pair.shape[2]):
                for right in range(pair.shape[2]):
                    value[b,target]+=pair[b,target,left].conj()*interaction[left,right]*pair[b,target,right]*prefactor
    return value


@pytest.mark.parametrize('broadcast',[False,True])
def test_nonhermitian_ordered_projection_matches_independent_sum(mesh,broadcast):
    pair,operator=plant()
    if broadcast:operator=operator[:1]
    prefactor=1/(512*137.)
    product,value=project_interaction_diagonal(face(operator,mesh),face(pair,mesh),
        mesh=mesh,prefactor=prefactor,scalar_replication_bound_bytes=96)
    expected=independent_projection(operator,pair,prefactor)
    np.testing.assert_allclose(value,expected,atol=2e-15,rtol=3e-13)
    np.testing.assert_allclose(product,np.matmul(pair.conj(),operator),atol=3e-12,rtol=3e-13)
    assert product.sharding==NamedSharding(mesh,P(None,'x','y'))
    assert value.sharding==NamedSharding(mesh,P())
    # Each plausible conjugation/transpose mistake has an independent witness.
    wrongs=[independent_projection(operator.swapaxes(-1,-2),pair,prefactor),
            independent_projection(operator.swapaxes(-1,-2).conj(),pair,prefactor),
            independent_projection(operator,pair.conj(),prefactor),
            independent_projection(operator,pair,prefactor*137.),
            -expected]
    assert all(np.max(abs(wrong-expected))>1e-5 for wrong in wrongs)


def test_gamma_reverse_vertices_are_conjugated_at_negative_g(mesh):
    forward,operator=plant();negative=np.array([0,2,1,3])
    pair=forward[...,negative].conj()
    _,actual=project_interaction_diagonal(face(-operator,mesh),face(pair,mesh),
        mesh=mesh,prefactor=1/(512*137.),scalar_replication_bound_bytes=96)
    expected=independent_projection(-operator,pair,1/(512*137.))
    np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=2e-15)
    wrong=independent_projection(-operator,forward.conj(),1/(512*137.))
    assert np.max(abs(expected-wrong))>1e-5
    # The CD helper consumes minusWc; changing its sign changes the observable.
    positive=independent_projection(operator,pair,1/(512*137.))
    np.testing.assert_allclose(actual,-positive,rtol=3e-13,atol=2e-15)


def test_isdf_legacy_arithmetic_and_plane_wave_volume_unit_conversion(mesh):
    pair,operator=plant();nk=512;volume=137.
    _,isdf=project_interaction_diagonal(face(-operator,mesh),face(pair,mesh),
        mesh=mesh,prefactor=1/nk,scalar_replication_bound_bytes=96)
    # Frozen reference project() used this exact equation; NumPy is an
    # independent implementation of its formerly local matmul/reduction.
    legacy=np.sum(np.matmul(pair.conj(),-operator)*pair,axis=-1)/nk
    np.testing.assert_allclose(isdf,legacy,atol=3e-13,rtol=3e-13)
    _,pw=project_interaction_diagonal(face(-operator,mesh),face(pair,mesh),
        mesh=mesh,prefactor=1/(nk*volume),scalar_replication_bound_bytes=96)
    expected_ev=independent_projection(-operator,pair,1/(nk*volume))*RYD_TO_EV
    np.testing.assert_allclose(np.asarray(pw)*RYD_TO_EV,expected_ev,rtol=3e-13,atol=3e-14)
    np.testing.assert_allclose(np.asarray(isdf)/volume,pw,rtol=3e-13,atol=3e-14)


@pytest.mark.parametrize('convention',['time_ordered_fractional','retarded'])
def test_gamma_pw_full_pole_integral_uses_reciprocal_partner_not_plain_conjugation(mesh,convention):
    """A physical ordered Lehmann field, independently integrated after projection.

    In reciprocal coordinates the real-space transpose is P_-G W^T P_-G.
    Its scalar contraction is W projected with conjugate(pair[-G]). Both
    spectral residues are PSD/Hermitian, yet generally distinct at Γ when
    time reversal is absent. No fitted response/model supplies the oracle.
    """
    rng=np.random.default_rng(1020226);negative=np.array([0,2,1,3])
    vectors=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2))
    # Logical G=(0,+g,-g), fourth slot is a carrier ghost. The Γ charge
    # body has a zero G0 row/column; no planted head is smuggled into Wc.
    vectors[[0,3]]=0
    positive=.002*vectors@vectors.conj().T
    negative_residue=positive.T[np.ix_(negative,negative)]
    pair=rng.normal(size=(1,2,4))+1j*rng.normal(size=(1,2,4))
    pair[...,3]=0
    reverse=pair[...,negative].conj()
    eta=.25/RYD_TO_EV;omega=.4;prefactor=1/(512*137.)
    queries=np.array([-.6,-.4,-.1,0.,.1,.4,.6])
    x=np.broadcast_to(queries[None,None,:,None],(1,1,len(queries),2))
    f=np.broadcast_to(np.array([1.,.25])[None,None,None,:],x.shape)
    u,w=cd.imaginary_rule(192,eta,scale=omega)
    crossing=np.unique(abs(x)[np.where(x<0,f,1-f)!=0])
    z=np.concatenate(([1j*eta],1j*u,crossing+1j*eta))
    field=(-positive[None]/(z-omega)[:,None,None]
           +negative_residue[None]/(z+omega)[:,None,None])
    repeated=np.broadcast_to(pair,(len(z),2,4))
    _,sample=project_interaction_diagonal(face(field,mesh),face(repeated,mesh),mesh=mesh,
        prefactor=prefactor,scalar_replication_bound_bytes=len(z)*2*16)
    _,partner=project_interaction_diagonal(face(field,mesh),
        face(np.broadcast_to(reverse,repeated.shape),mesh),mesh=mesh,
        prefactor=prefactor,scalar_replication_bound_bytes=len(z)*2*16)
    _,wrong=project_interaction_diagonal(face(field,mesh),face(repeated.conj(),mesh),mesh=mesh,
        prefactor=prefactor,scalar_replication_bound_bytes=len(z)*2*16)
    anchor_slope=(positive/(1j*eta-omega)**2-negative_residue/(1j*eta+omega)**2)/(2j*eta)
    _,derivative=project_interaction_diagonal(face(anchor_slope[None],mesh),face(pair,mesh),
        mesh=mesh,prefactor=prefactor,scalar_replication_bound_bytes=32)
    sample=np.asarray(sample)[:,None,None,None,:]
    partner=np.asarray(partner)[:,None,None,None,:]
    wrong=np.asarray(wrong)[:,None,None,None,:]
    derivative=np.asarray(derivative)[0][None,None,None,:]
    kwargs=dict(eta=eta,analytic_convention=convention)
    anchor,cp,cm,beta=cd.anchor_part(sample[0],derivative,x,f,**kwargs)
    remainder=np.zeros_like(anchor)
    for i,(ui,wi) in enumerate(zip(u,w),1):
        remainder+=cd.imag_remainder_node(sample[i],ui,wi,x,f,cp,cm,beta,**kwargs)
    total=anchor+remainder;bad=total.copy()
    for i,node in enumerate(crossing,1+len(u)):
        total+=cd.real_residue_node(sample[i],partner[i],x,f,node,**kwargs)
        bad+=cd.real_residue_node(sample[i],wrong[i],x,f,node,**kwargs)
    # Literal finite-G spectral projections are independent of matmul and CD.
    rp=independent_projection(positive[None],pair,prefactor)[0].real
    rm=independent_projection(negative_residue[None],pair,prefactor)[0].real
    sheet=-1 if convention=='time_ordered_fractional' else 1
    expected=np.sum(rp[None,None,None,:]*(1-f)/(x-omega+1j*eta)
        +rm[None,None,None,:]*f/(x+omega+sheet*1j*eta),axis=-1)[:,:,None,:]
    assert np.max(abs(total-expected))/np.max(abs(expected))<2e-7
    assert np.max(abs(bad-expected))/np.max(abs(expected))>1e-3
    assert np.max(abs(positive-positive.T))>.001
    assert np.max(abs(positive-negative_residue))>.001


@pytest.mark.parametrize('prefactor',[0.,-1.,np.nan,np.inf])
def test_bad_physical_prefactor_refuses(mesh,prefactor):
    pair,operator=plant()
    with pytest.raises(ValueError,match='positive finite physical prefactor'):
        project_interaction_diagonal(face(operator,mesh),face(pair,mesh),mesh=mesh,
            prefactor=prefactor,scalar_replication_bound_bytes=96)


def test_scalar_replication_bound_is_checked_before_projection(mesh):
    pair,operator=plant()
    with pytest.raises(ValueError,match='explicit replication bound'):
        project_interaction_diagonal(face(operator,mesh),face(pair,mesh),mesh=mesh,
            prefactor=1.,scalar_replication_bound_bytes=95)


@pytest.mark.parametrize('operand',['interaction','pair'])
def test_replicated_operator_or_pair_refuses(mesh,operand):
    pair,operator=plant();args=dict(interaction=face(operator,mesh),pair=face(pair,mesh))
    args[operand]=jax.device_put(operator if operand=='interaction' else pair,NamedSharding(mesh,P()))
    with pytest.raises(ValueError,match='complex128 all-P'):
        project_interaction_diagonal(**args,mesh=mesh,prefactor=1.,scalar_replication_bound_bytes=96)


@pytest.mark.parametrize('case',['nonsquare','wrong_G','wrong_batch','real_dtype','rank'])
def test_incompatible_projected_geometry_or_dtype_refuses(mesh,case):
    pair,operator=plant()
    if case=='nonsquare':operator=operator[:,:,:2]
    elif case=='wrong_G':pair=pair[:,:,:2]
    elif case=='wrong_batch':operator=operator[:2]
    left=face(operator,mesh);right=face(pair,mesh)
    if case=='real_dtype':left=jax.device_put(np.real(operator),NamedSharding(mesh,P(None,'x','y')))
    elif case=='rank':left=jax.device_put(operator[0],NamedSharding(mesh,P('x','y')))
    with pytest.raises(ValueError,match='CD projection'):
        project_interaction_diagonal(left,right,mesh=mesh,prefactor=1.,scalar_replication_bound_bytes=96)
