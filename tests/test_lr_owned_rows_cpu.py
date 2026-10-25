"""Long-range W rows: owned-column fits and sharded carrier conversion."""
from types import SimpleNamespace
import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()
import jax
import jax.numpy as jnp
import numpy as np
import h5py
import pytest
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
jax.config.update('jax_enable_x64',True)


def mesh_for(side):
    if jax.device_count()<side*side:
        pytest.skip('requires nine virtual CPU devices')
    return Mesh(np.array(jax.devices()[:side*side]).reshape(side,side),('x','y'))


@pytest.mark.parametrize('side,target',[(2,14),(3,15)])
def test_lr_rows_fit_only_owned_columns_and_add_to_legal_face(monkeypatch,side,target):
    from bse import bse_densify as B
    from file_io import restart_bundle as R
    from gw import head_correction as HC,lr_channels as LC
    from vcoul import lr_channels,CoulombGeometry
    mesh=mesh_for(side)
    logical,carrier=13,-(-13//(side*side))*(side*side)
    grid=(3,3,3)
    q=np.stack(np.meshgrid(*[np.arange(3)/3 for _ in range(3)],indexing='ij'),-1).reshape(-1,3)
    q-=np.round(q)
    rng=np.random.default_rng(831+side)
    r=rng.random((logical,3))
    physical=rng.normal(size=(27,1,logical))+1j*rng.normal(size=(27,1,logical))
    source=np.pad(physical,((0,0),(0,0),(0,carrier-logical)))
    ft=jax.device_put(source,NamedSharding(mesh,P(None,None,('x','y'))))
    polar=(-.2*np.sum(q*q,axis=1))[:,None,None].astype(complex)
    ch=lr_channels(q,bvec=np.eye(3),sys_dim=3,n_z=0)
    payload=dict(Ft=ft,P=polar,v=ch.v,q_frac=q,gvec=ch.gvec,n_z=0,
        sys_dim=3,head_kernel='point',n_mu_logical=logical,tables=SimpleNamespace(q_irr_frac=q))
    monkeypatch.setattr(R,'read_lr_channels',lambda *a,**kw:payload)
    monkeypatch.setattr(B,'_unfold_lr_channels',lambda *a,**kw:(ft,polar,ch.v,q[:,None],q))
    monkeypatch.setattr(HC,'resolve_head_S_cart',lambda *a,**kw:(-.2*np.eye(3),'literal fixture'))
    original_fit=LC.fit_model
    widths,empty_blocks=[],[]
    def owned_fit(qc,Ft,*a,**kw):
        widths.append(Ft.shape[-1])
        assert Ft.shape[-1]<=carrier//(side*side)
        result=original_fit(qc,Ft,*a,**kw)
        if not np.any(Ft):
            assert result[1]['relres_F']==0
            empty_blocks.append(True)
        return result
    monkeypatch.setattr(LC,'fit_model',owned_fit)
    original_array=np.asarray
    def no_global(value,*a,**kw):
        if isinstance(value,jax.Array) and isinstance(value.sharding,NamedSharding) and value.ndim>=2:
            raise AssertionError('global LR field copied to host')
        return original_array(value,*a,**kw)
    monkeypatch.setattr(np,'asarray',no_global)
    wfn=SimpleNamespace(blat=1.,bvec=np.eye(3),cell_volume=1.)
    A,D,Af,Df=B._srlr_terms('unused',wfn,None,SimpleNamespace(sys_dim=3),{},
        coarse_grid=grid,fine_grid=grid,n_rmu=logical,n_rmu_pad=target,
        mesh_xy=mesh,input_file='unused',centroid_frac=r)
    assert len(widths)==len(ft.addressable_shards)
    if side>=3:
        assert empty_blocks
    assert A.sharding.is_equivalent_to(NamedSharding(mesh,P(None,None,('x','y'))),3)
    # Literal global model, after the guarded production fit has completed.
    model,_=original_fit(q,physical,polar,r_mu=r*(2*np.pi),S_cart=-.2*np.eye(3),
        c0=0,sys_dim=3,r_fit=LC.WINDOW_OUTER*.5)
    qn=np.linalg.norm(q,axis=1)
    on=(LC.window(qn,LC.WINDOW_INNER*.5,LC.WINDOW_OUTER*.5)>0)&(qn>1e-9)
    wanted=np.zeros_like(source)
    wanted[on,:,:logical]=LC.model_rows(model,q[on])
    wanted[qn<=1e-9]=source[qn<=1e-9]
    np.testing.assert_allclose(original_array(A),wanted,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(original_array(Af),wanted,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(D,Df,rtol=1e-12,atol=1e-12)
    # Actual face conversion executable, including source16/18 -> face14/15.
    from runtime import aot_memory
    original_price=aot_memory.compiled_new_bytes
    seen=[]
    def actual_quote(exe):
        hlo=exe.as_text()
        for n in (carrier,target):
            assert f'c128[27,1,{n}]' not in hlo,hlo
        seen.append(hlo)
        return original_price(exe)
    monkeypatch.setattr(aot_memory,'compiled_new_bytes',actual_quote)
    W=np.zeros((target,target,*grid),complex)
    value=B._add_rank_lr(jax.device_put(W,NamedSharding(mesh,P('x','y',None,None,None))),A,D,mesh,1.)
    term=np.einsum('qam,qab,qbn->mnq',wanted[...,:target].conj(),D,wanted[...,:target]).reshape(W.shape)
    np.testing.assert_allclose(original_array(value),term,rtol=1e-12,atol=1e-12)
    assert len(seen)==2


@pytest.mark.parametrize('side',[2,3])
def test_lr_restart_reads_only_owned_mu_hyperslabs(tmp_path,monkeypatch,side):
    from file_io.restart_bundle import read_lr_channels
    mesh=mesh_for(side)
    path=tmp_path/'lr.h5'
    rng=np.random.default_rng(149)
    Ft=rng.normal(size=(3,2,13))+1j*rng.normal(size=(3,2,13))
    with h5py.File(path,'w') as f:
        f.create_dataset('W0_qmunu',shape=(13,13,1,1,1),dtype=complex)
        g=f.create_group('W0_lr')
        for key,value in dict(Ft=Ft,P=np.zeros((3,2,2),complex),gvec=np.zeros((3,2,3),int),v=np.ones((3,2)),q_frac=np.zeros((3,3))).items():
            g.create_dataset(key,data=value)
        g.attrs.update(n_z=0,sys_dim=3,head_kernel='point')
    original=h5py.Dataset.__getitem__
    def selected(ds,key):
        if ds.name=='/W0_lr/Ft' and key==():
            raise AssertionError('full LR restart form-factor host read')
        return original(ds,key)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',selected)
    value=read_lr_channels(path,mesh=mesh)
    expected=np.pad(Ft,((0,0),(0,0),(0,-(-13//(side*side))*(side*side)-13)))
    np.testing.assert_array_equal(np.asarray(value['Ft']),expected)
    assert value['n_mu_logical']==13
    assert value['Ft'].sharding.is_equivalent_to(NamedSharding(mesh,P(None,None,('x','y'))),3)
