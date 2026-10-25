"""Static LR producer and restart use distributed rows, exact Schur equations."""
from types import SimpleNamespace
import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()
import jax
import numpy as np
import pytest
import h5py
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
jax.config.update('jax_enable_x64',True)


def mesh_for(side):
    if jax.device_count()<side*side:
        pytest.skip('needs nine virtual CPU devices')
    return Mesh(np.array(jax.devices()[:side*side]).reshape(side,side),('x','y'))


@pytest.mark.parametrize('side,packed',[(2,False),(3,False),(2,True),(3,True)])
def test_device_producer_matches_literal_schur_and_sharded_writer(tmp_path,monkeypatch,side,packed):
    from gw import lr_channels as LC,w_isdf,restart_q_storage
    from file_io.tagged_arrays import write_lr_channels_to_h5
    from runtime import aot_memory
    mesh=mesh_for(side)
    nq,nl,logical=3,2,13
    mu=-(-logical//(side*side))*(side*side)
    rng=np.random.default_rng(789+side)
    F=.02*(rng.normal(size=(nq,nl,logical))+1j*rng.normal(size=(nq,nl,logical)))
    X=.02*(rng.normal(size=(nq,logical,logical))+1j*rng.normal(size=(nq,logical,logical)))
    chi=-X@X.conj().transpose(0,2,1)
    v=np.array([[0.,.2],[.1,.3],[.4,.05]])
    V=.5*np.eye(logical)[None]+np.einsum('qam,qa,qan->qmn',F.conj(),v,F)
    W=np.linalg.solve(np.eye(logical)[None]-V@chi,V)
    fchi=F@chi;chif=chi@F.conj().transpose(0,2,1)
    G=F+fchi@W;Q=G@chif
    a=np.eye(nl)[None]+Q*v[:,None,:]
    expected=np.linalg.solve(a,G)
    expectedP=np.linalg.solve(a,Q);expectedP=.5*(expectedP+expectedP.conj().transpose(0,2,1))
    row=NamedSharding(mesh,P(None,None,('x','y')))
    face=NamedSharding(mesh,P(None,'x','y'))
    pad=lambda a:np.pad(a,((0,0),(0,0),(0,mu-logical)))
    fdev=jax.device_put(pad(F),row)
    cdev=jax.device_put(np.pad(chi,((0,0),(0,mu-logical),(0,mu-logical))),face)
    wdev=jax.device_put(np.pad(W,((0,0),(0,mu-logical),(0,mu-logical))),face)
    basis=None
    if packed:
        from common.centroid_basis import PackedCentroidBasis
        from common.grouped_layout import build_square_grouped_shard_layout
        groups=np.repeat(np.arange(7),[3,3,2,2,1,1,1])
        layout=build_square_grouped_shard_layout(groups,(side,side))
        basis=PackedCentroidBasis(mesh,layout,np.zeros((logical,3),np.int32),mu)
        cdev=basis.pack_operator(cdev)
        wdev=basis.pack_operator(wdev)
    q=np.array([[0.,0.,0.],[.2,0.,0.],[-.2,0.,0.]])
    ch=SimpleNamespace(v=v,gvec=np.zeros((nq,nl,3),np.int32))
    monkeypatch.setattr(LC,'read_channel_rows',lambda *a,**kw:(fdev,ch,0,'point',logical))
    monkeypatch.setattr(w_isdf,'_w_solve_pref_scalar',lambda meta:1.)
    captured={}
    monkeypatch.setattr(restart_q_storage,'deposit_lr_channels',lambda p:captured.update(p))
    from common.collectives import rank0_transaction
    def fixture():
        tmp_path.joinpath('tmp').mkdir(exist_ok=True)
        tmp_path.joinpath('tmp/zeta_q.h5').touch()
    rank0_transaction(tmp_path,stage='lr.producer.test.fixture',write=fixture)
    config=SimpleNamespace(input_dir=str(tmp_path),head=SimpleNamespace(mc_average_vcoul_body=False))
    meta=SimpleNamespace(sys_dim=3,mu_basis=basis,kgrid=(3,1,1))
    original_asarray=np.asarray
    def no_global(a,*args,**kw):
        if isinstance(a,jax.Array) and isinstance(a.sharding,NamedSharding) and a.ndim>=2:
            raise AssertionError('global LR producer row field copied to host')
        return original_asarray(a,*args,**kw)
    monkeypatch.setattr(np,'asarray',no_global)
    original_quote=aot_memory.compiled_new_bytes
    quotes=[]
    def quote(exe):
        text=exe.as_text()
        channel_carrier=-(-nl//side)*side
        for width in {mu,mu if basis is None else basis.n_packed}:
            for channels in {nl,channel_carrier}:
                for shape in ((nq,channels,width),(nq,width,channels),(nq*channels,width),(width,nq*channels)):
                    assert 'c128['+','.join(map(str,shape))+']' not in text,text
        quotes.append(text)
        return original_quote(exe)
    monkeypatch.setattr(aot_memory,'compiled_new_bytes',quote)
    state=LC.begin(cdev,config=config,meta=meta,sym=None,centroid_indices=None,
        wedge=({'q_irr_frac':q},),mesh_xy=mesh)
    LC.finish(state,wdev)
    assert len(quotes)==4
    assert captured['Ft'].sharding.is_equivalent_to(row,3)
    for shard in captured['Ft'].addressable_shards:
        np.testing.assert_allclose(original_asarray(shard.data),pad(expected)[shard.index],rtol=3e-14,atol=3e-14)
    np.testing.assert_allclose(captured['P'],expectedP,rtol=3e-14,atol=3e-14)
    assert captured['n_rmu_logical']==logical
    path=tmp_path/'restart.h5'
    write_lr_channels_to_h5(path,**captured)
    with h5py.File(path) as f:
        np.testing.assert_allclose(f['W0_lr/Ft'][()],expected,rtol=3e-14,atol=3e-14)
        np.testing.assert_allclose(f['W0_lr/P'][()],expectedP,rtol=3e-14,atol=3e-14)
        assert f['W0_lr/Ft'].shape==(nq,nl,logical)


@pytest.mark.parametrize('side',[1,2,3])
def test_bare_channel_reader_requests_only_owned_mu_columns(monkeypatch,side):
    from gw import lr_channels as LC
    import zeta_loader
    mesh=mesh_for(side)
    logical,nq,ng=13,3,3
    carrier=-(-logical//(side*side))*(side*side)
    rng=np.random.default_rng(793)
    values=rng.normal(size=(nq,logical,ng))+1j*rng.normal(size=(nq,logical,ng))
    gcomp=np.zeros((nq,3,ng),np.int64);gcomp[:,0]=[-1,0,1]
    reads=[]
    class OwnedReader:
        bvec=np.eye(3);blat=1.;cell_volume=1.;n_rmu=logical
        gvec_components=gcomp;ngk_per_q=np.array([ng]*nq)
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read_zeta_G_local(self,key):
            qi,mu,gs=key
            assert mu.stop-mu.start<=carrier//(side*side)
            reads.append(mu)
            return values[qi,mu][:,gs]
    monkeypatch.setattr(zeta_loader,'ZetaLoader',OwnedReader)
    q=np.array([[0.,0.,0.],[.2,0.,0.],[-.2,0.,0.]])
    result,ch,nz,kernel,nmu=LC.read_channel_rows('owned fixture',q,sys_dim=3,
        kgrid=(3,1,1),mc_average_vcoul_body=False,mesh_xy=mesh)
    wanted=np.pad(values[:,:,1][:,None],((0,0),(0,0),(0,carrier-logical)))
    for shard in result.addressable_shards:
        np.testing.assert_array_equal(np.asarray(shard.data),wanted[shard.index])
    assert result.sharding.is_equivalent_to(NamedSharding(mesh,P(None,None,('x','y'))),3)
    assert nmu==logical and nz==0 and kernel=='point'
    assert reads
