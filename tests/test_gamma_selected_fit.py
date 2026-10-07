"""Small independent conditioning/layout controls for the opt-in Γ fit.

CPU4 single-process fixtures do not certify the distributed GPU eigensolver
or the full native archive. Those are separate physical parity/resource legs.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import require_full_mesh
from isdf import cplus
from isdf.zeta_mubatch import ZStore
from runtime.padding import padded_axis
from gw.isdf_fitting import _gamma_c_from_parent_faces, fit_zeta_to_h5


@pytest.fixture(scope='module')
def mesh():
    if len(jax.devices()) != 4:
        pytest.skip('explicit four-device source fixture required')
    return Mesh(np.asarray(jax.devices()).reshape(2,2),('x','y'))


def plant():
    rng=np.random.default_rng(728)
    q,_=np.linalg.qr(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))
    lam=np.array([1e-12,1e-12,1.,2.,3.,4.,6.,8.],np.float64)
    z=rng.normal(size=(1,8,4))+1j*rng.normal(size=(1,8,4))
    return lam,q.astype(np.complex128),z.astype(np.complex128)


def test_eigenpair_conditioning_matches_independent_known_inverse():
    lam,v,z=plant()
    b=cplus.factor_from_eigensystem(jnp.asarray(lam[None]),jnp.asarray(v[None]),
        rcond=1e-8,rank_log=False,n_log=8)
    expected=(v[:,2:]/lam[2:])@v[:,2:].conj().T@z[0]
    np.testing.assert_allclose(np.asarray(cplus.apply(b,jnp.asarray(z)))[0],expected,atol=3e-13,rtol=3e-13)


def test_default_factor_and_application_literal_parity():
    lam,v,z=plant();c=(v*lam)@v.conj().T
    a=jnp.asarray(c[None]);w,q=jnp.linalg.eigh(a)
    expected=cplus.factor_from_eigensystem(w,q,rcond=1e-8,rank_log=False,n_log=8)
    got=cplus.factor(a,rcond=1e-8,rank_log=False,n_log=8)
    np.testing.assert_array_equal(np.asarray(got),np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(cplus.apply(got,jnp.asarray(z))),
        np.asarray(got@(jnp.conj(jnp.swapaxes(got,-1,-2))@jnp.asarray(z))))


@pytest.mark.parametrize('lshape,vshape',[( (1,7),(1,8,8)),((1,8),(1,7,8)),((8,),(1,8,8))])
def test_incompatible_eigenpair_metadata_refuses(lshape,vshape):
    with pytest.raises(ValueError,match='eigenvectors'):
        cplus.factor_from_eigensystem(jnp.ones(lshape),jnp.ones(vshape),rcond=1e-8,rank_log=False,n_log=8)


def test_face_application_complex_adjoint_not_transpose(mesh):
    lam,v,z=plant();b=v*np.where(lam>1e-8*lam[-1],1/np.sqrt(lam),0.)
    face=NamedSharding(mesh,P(None,'x','y'))
    got=cplus.apply(jax.device_put(b[None],face),jax.device_put(z,face),mesh_xy=mesh,panel_bytes=256)
    expected=b@(b.conj().T@z[0])
    np.testing.assert_allclose(np.asarray(got)[0],expected,atol=3e-13,rtol=3e-13)
    assert got.sharding == face
    assert np.max(abs(b@(b.T@z[0])-expected))>1e-2


def test_all_global_mesh_guard_and_replicated_face_refusal(mesh):
    assert require_full_mesh(mesh) is mesh
    subset=Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),('x','y'))
    with pytest.raises(ValueError,match='global processor'):
        require_full_mesh(subset)
    with pytest.raises(ValueError,match='face'):
        cplus.apply(jax.device_put(np.eye(8)[None].astype(np.complex128),NamedSharding(mesh,P())),
            jax.device_put(np.ones((1,8,4),np.complex128),NamedSharding(mesh,P(None,'x','y'))),mesh_xy=mesh)


@pytest.mark.parametrize('pfs',[
    np.arange(8,dtype=np.int32),np.array([7,0,5,2,3,6,1,4],np.int32),
    np.array([7,0,-1,2,3,6,1,-1],np.int32)])
def test_host_raw_store_face_rank_batch_packing_and_zero_holes(mesh,pfs):
    store=ZStore(mesh=mesh,q_axis=padded_axis(1,4,name='test q'),mu_pad=8,
        g_axis=padded_axis(4,4,name='test G'),b=4,placement='host',packed_from_slot=pfs,n_batch=2)
    expected=np.arange(1*8*4,dtype=np.float64).reshape(1,8,4).astype(np.complex128)*(1+.2j)
    for dev in mesh.local_devices:
        p=list(mesh.devices.flat).index(dev)
        for beta in range(2):store._host[dev.id][0,0,beta,0]=expected[0,beta*4+p]
    got=store.read_tile(0,face=True)
    wanted=expected[:,np.clip(pfs,0,None)]*(pfs>=0)[None,:,None]
    np.testing.assert_array_equal(np.asarray(got),wanted)
    assert got.sharding == NamedSharding(mesh,P(None,'x','y'))
    store.close()


def test_gamma_gram_bounded_children_against_literal_density_sum(mesh):
    rng=np.random.default_rng(17)
    psi=(rng.normal(size=(2,4,1,4))+1j*rng.normal(size=(2,4,1,4))).astype(np.complex128)
    full=np.array([1,0,1,0],np.int32)
    tables=dict(irr_idx=full,sym_idx=np.zeros(4,np.int32),
        spin_action_full=np.ones((4,1,1),np.complex128),n_sym_spatial=1)
    plan=SimpleNamespace(nspinor=1,n_sym_spatial=1,n_centroid_packed=4,
        wavefunction_unfold_tables=lambda:tables,
        unfold_face=lambda a,**kw:jnp.take(a,kw['tables']['irr_idx'],axis=0))
    y=jax.device_put(psi,NamedSharding(mesh,P(None,'x',None,'y')))
    m=jax.device_put(psi.conj().transpose(0,2,3,1),NamedSharding(mesh,P(None,None,'x','y')))
    got=_gamma_c_from_parent_faces(m,y,plan,mesh,panel_k=2)
    want=np.zeros((4,4),np.complex128)
    for k in full:
        projector=np.einsum('nsu,nsv->uv',psi[k].conj(),psi[k])
        want+=projector*projector.conj()
    np.testing.assert_allclose(np.asarray(got)[0],want,atol=2e-12,rtol=3e-13)
    assert got.sharding == NamedSharding(mesh,P(None,'x','y'))


@pytest.mark.parametrize('selection',[[True],[0.],[1],[0,1],[],np.array([2**63],np.uint64)])
def test_selected_q_requires_exact_gamma_identity(selection):
    fake=np.zeros((1,1,1,1),np.complex128)
    with pytest.raises(ValueError,match='selected_q'):
        fit_zeta_to_h5(wfn=None,meta=SimpleNamespace(nspinor=1),sym=None,
            centroid_indices=None,mesh_xy=None,output_files={0:'unused'},
            k_unfold_plan=SimpleNamespace(n_parent=1,n_centroid_packed=1),
            psi_nmu_parent=fake,psi_mun_parent=fake,mubatch_plan=object(),selected_q=selection)
