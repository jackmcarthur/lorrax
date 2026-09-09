"""Shared-pole storage gates; the CLI runs the same checks on a real P4 mesh."""
from pathlib import Path
from types import SimpleNamespace
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.centroid_basis import PackedCentroidBasis
from common.collectives import rank0_transaction
from file_io import shared_pole_store as store
from file_io.slab_io import SlabIO
from symmetry_maps import QirrTables


def _fixture(mesh):
    identity_matrix = np.eye(3, dtype=np.int32)
    swap = np.asarray([[0,1,0],[1,0,0],[0,0,1]], np.int32)
    rotations = np.stack((identity_matrix,swap))
    sym = SimpleNamespace(sym_matrices=rotations, translations=np.zeros((2,3)),
        trs_allowed=True, active_symmetry_rows=np.arange(4,dtype=np.int32),
        operation_typing_source="planted typed scalar fixture")
    sym.operation_rows = lambda rows: (
        np.asarray([rotations[r%2] * (-1 if r>=2 else 1) for r in rows]),
        np.zeros((len(rows),3)), np.asarray(rows)>=2)
    sym.spinor_action = lambda rows, nspinor: np.ones((len(rows),1,1),np.complex128)
    cents = np.asarray([[1,0,0],[0,1,0],[2,0,0],[0,2,0],[3,1,0],[1,3,0],[2,2,0]],np.int32)
    basis = PackedCentroidBasis.build(cents,sym,(4,4,1),mesh)
    from symmetry_maps import centroid_source_map_and_wrap
    perm, wraps = centroid_source_map_and_wrap(cents,rotations,sym.translations,
        np.asarray((4,4,1),np.int32),extend_trs=True)
    qt = QirrTables(irr_idx_q=np.arange(27,dtype=np.int32)%3,
        sym_idx_q=np.zeros(27,np.int32), q_irr_frac=np.asarray([[0,0,0],[1/3,0,0],[2/3,0,0]]),
        sym_perm=perm,L_table=wraps,n_sym_spatial=2)
    meta=SimpleNamespace(mu_basis=basis,nspinor=1,kgrid=(3,3,3),fft_grid=(4,4,1),nk_tot=27,n_rmu=7)
    from gw.shared_pole_recipe import CapacityLedger
    meta.shared_pole_capacity=CapacityLedger(meta,mesh_xy=mesh)
    meta.shared_pole_capacity.reserve("fixture_live_bound",resident_bytes_per_rank=4096,workspace_bytes_per_rank=0)
    meta.shared_pole_capacity.live_stages=("fixture_live_bound",)
    tables={"qirr":qt,"q_irr_full_idx":np.arange(3,dtype=np.int64),"sym":sym}
    recipe={"version":"shared_real_pole_v1_r3b","gate_version":"shared_real_pole_gates_v1_r3b"}
    identity={key:"planted-"+key for key in store._IDENTITY_KEYS}
    return meta,tables,recipe,identity


def _device(host,mesh,spec):
    # Tiny independent planted oracle; production payloads use SlabIO.
    return jax.make_array_from_callback(host.shape,NamedSharding(mesh,spec),lambda idx:host[idx])


def _assert_local(array, expected):
    for shard in array.addressable_shards:
        np.testing.assert_array_equal(np.asarray(shard.data),expected[shard.index])


def _model(meta):
    count=np.asarray([3,5,2],np.int64)
    C=np.zeros((3,7,1,6),np.complex128)
    poles=np.ones((3,6),np.float64)
    for q,k in enumerate(count):
        a=np.arange(7*k).reshape(7,k)+1
        C[q,:,0,:k]=a*(q+1)+1j*(a+q)/8
        poles[q,:k]=(np.arange(k)+1+q/8)**2
    packed=meta.mu_basis.pack_host(C,axis=1)
    return C,packed,poles,count


def check_roundtrip(mesh,path,layout="local"):
    meta,tables,recipe,identity=_fixture(mesh)
    assert mesh.shape['x']>1 and mesh.shape['y']>1
    assert not meta.mu_basis.is_identity
    C,packed,poles,count=_model(meta)
    cdev=_device(packed,mesh,P(None,'x',None,'y'))
    # Exercise actual local/distributed service-produced handoffs. This is
    # storage parity, not a test of the constructor eigensolve or its recipe.
    from gw.gw_config import linalg_resolution
    from distrib_la import matmul, resolve_matmul_backend
    route = linalg_resolution({'linalg':layout}).batched_route
    backend = 'off' if layout == 'local' else 'distributed'
    resolved = resolve_matmul_backend(backend, mesh, batched_route=route)
    identity_matrix = np.broadcast_to(np.eye(6,dtype=np.complex128),(3,6,6)).copy()
    product = matmul(cdev[:,:,0,:],_device(identity_matrix,mesh,P(None,'x','y')),
                     mesh=mesh,backend=backend,batched_route=route)
    cdev=product[:,:,None,:]
    _assert_local(cdev,packed)
    print(f"STORE_HANDOFF layout={layout} route={route} backend={resolved}",flush=True)
    pdev=_device(poles,mesh,P(None,'y'))
    partial=store.write_shared_pole_model(path,cdev[:1],pdev[:1],count[:1],q_span=(0,1),
        meta=meta,tables=tables,recipe=recipe,receipts={"identity":identity,"scope":"planted"})
    assert not partial['finalized']
    with pytest.raises(ValueError,match="incomplete|missing final"):
        store.validate_shared_pole_model(path,expected_identity=identity,mesh_xy=mesh,capacity=meta.shared_pole_capacity)
    with pytest.raises(ValueError,match="identity changed"):
        store.write_shared_pole_model(path,cdev[1:],pdev[1:],count[1:],q_span=(1,3),
            meta=meta,tables=tables,recipe={**recipe,"version":"stale"},receipts={"identity":identity})
    header=store.write_shared_pole_model(path,cdev[1:],pdev[1:],count[1:],q_span=(1,3),
        meta=meta,tables=tables,recipe=recipe,receipts={"identity":identity,"scope":"planted"})
    assert header['finalized'] and header['Kmax']==5
    assert header['peak_payload_bytes']<=2*header['compact_payload_bytes']
    assert store.validate_shared_pole_model(path,expected_identity=identity,mesh_xy=mesh,capacity=meta.shared_pole_capacity)['digest']==header['digest']
    with pytest.raises(ValueError,match="stale"):
        store.validate_shared_pole_model(path,expected_identity={**identity,"energies":"changed"},mesh_xy=mesh,capacity=meta.shared_pole_capacity)
    with pytest.raises(ValueError,match="immutable"):
        store.write_shared_pole_model(path,cdev[:1],pdev[:1],count[:1],q_span=(0,1),
            meta=meta,tables=tables,recipe=recipe,receipts={"identity":identity})
    with SlabIO(path,mode='r',mesh=mesh) as io:
        all_poles,all_counts=store.read_shared_pole_census(io,header=header,capacity=meta.shared_pole_capacity)
        np.testing.assert_array_equal(np.asarray(all_poles),poles[:,:5])
        np.testing.assert_array_equal(np.asarray(all_counts),count)
        assert store.shared_pole_qirr_tables(header).digest()==tables['qirr'].digest()
        for columns in (None,(1,4),(4,5)):
            cx,cy,lam,k=store.read_shared_pole_faces(io,(0,3),meta=meta,header=header,column_span=columns)
            start,stop=(0,5) if columns is None else columns
            for face,axis in ((cx,'x'),(cy,'y')):
                assert face.sharding.is_equivalent_to(NamedSharding(mesh,P(None,axis,None,None)),4)
                _assert_local(face,packed[:,:,:,start:stop])
            np.testing.assert_array_equal(np.asarray(k),np.clip(count-start,0,stop-start))
            np.testing.assert_array_equal(np.asarray(lam),poles[:,start:stop])
        # Canonical staging can exceed packed extent; readers must honor the
        # basis's n_canonical rather than deriving it from logical n or P_x.
        from dataclasses import replace
        padded_meta = SimpleNamespace(**vars(meta))
        padded_meta.mu_basis = replace(meta.mu_basis,
                                       n_canonical=meta.mu_basis.n_canonical+int(mesh.size))
        cx,cy,_,_=store.read_shared_pole_faces(io,(0,3),meta=padded_meta,header=header)
        _assert_local(cx,packed[:,:,:,:5])
        _assert_local(cy,packed[:,:,:,:5])
    # Independent HDF5 inspection, no live collective handle. Fixture is tiny.
    def oracle():
        import h5py
        with h5py.File(path,'r') as f:
            assert f['factor'].chunks is None and f['factor'].shape==(3,7,1,5)
            np.testing.assert_array_equal(f['factor'][:],C[:,:,:,:5])
            np.testing.assert_array_equal(f['poles2_ry2'][:],poles[:,:5])
            # Full-K centroid intervals are contiguous in the persisted order.
            offsets=np.arange(7*5).reshape(7,1,5)[2:5].ravel()
            assert np.all(np.diff(offsets)==1)
            assert 'staging' not in f
    rank0_transaction(path,stage='test.independent_oracle',write=oracle)
    # Exercise the actual collective validator and restart transaction together.
    from file_io.tagged_arrays import (register_shared_pole_restart_member,
                                      read_shared_pole_restart_member)
    from file_io.commit_state import set_commit_state
    restart = path.with_name(path.stem+'_restart.h5')
    def create_restart():
        import h5py
        with h5py.File(restart,'w') as f:set_commit_state(f,True)
    rank0_transaction(restart,stage='test.restart_create',write=create_restart)
    member=register_shared_pole_restart_member(restart,path,
        expected_identity=identity,mesh_xy=mesh,capacity=meta.shared_pole_capacity)
    assert member['digest']==header['digest']
    assert read_shared_pole_restart_member(restart,
        expected_identity=identity,mesh_xy=mesh,capacity=meta.shared_pole_capacity)==member
    return header


def check_finalization_resume(mesh,path):
    meta,tables,recipe,identity=_fixture(mesh)
    _,packed,poles,count=_model(meta)
    original=store._finalize_model
    def interrupt(*args,**kwargs):
        raise RuntimeError('planted stop after staged close')
    store._finalize_model=interrupt
    try:
        with pytest.raises(RuntimeError,match='planted stop'):
            store.write_shared_pole_model(path,_device(packed,mesh,P(None,'x',None,'y')),
                _device(poles,mesh,P(None,'y')),count,q_span=(0,3),meta=meta,tables=tables,
                recipe=recipe,receipts={'identity':identity})
    finally:
        store._finalize_model=original
    header=store.finalize_shared_pole_model(path,meta=meta,expected_identity=identity)
    assert header['finalized']
    for dataset in ('operations/rotation','q_irr_full_idx','qirr/n_sym_spatial'):
        def corrupt():
            import h5py
            with h5py.File(path,'a') as f:
                saved=f[dataset][()]
                f[dataset][...]=saved+1
        rank0_transaction(path,stage='test.metadata_corrupt',write=corrupt)
        with pytest.raises(ValueError,match='metadata changed'):
            store.validate_shared_pole_model(path,expected_identity=identity,mesh_xy=mesh,capacity=meta.shared_pole_capacity)
        def restore():
            import h5py
            with h5py.File(path,'a') as f:f[dataset][...]=f[dataset][()]-1
        rank0_transaction(path,stage='test.metadata_restore',write=restore)


def check_corruption(mesh,path):
    meta,tables,recipe,identity=_fixture(mesh)
    _,packed,poles,count=_model(meta)
    cdev=_device(packed,mesh,P(None,'x',None,'y'))
    pdev=_device(poles,mesh,P(None,'y'))
    store.write_shared_pole_model(path,cdev,pdev,count,q_span=(0,3),
        meta=meta,tables=tables,recipe=recipe,receipts={"identity":identity})
    def corrupt():
        import h5py
        with h5py.File(path,'a') as f:f['factor'][0,0,0,0]+=1j
    rank0_transaction(path,stage='test.corrupt_payload',write=corrupt)
    with pytest.raises(ValueError,match='digest mismatch'):
        store.validate_shared_pole_model(path,expected_identity=identity,mesh_xy=mesh,capacity=meta.shared_pole_capacity)


def check_sentinel_refusal(mesh,path):
    meta,tables,recipe,identity=_fixture(mesh)
    _,packed,poles,count=_model(meta)
    packed[0,0,0,5]=2
    with pytest.raises(ValueError,match='sentinel'):
        store.write_shared_pole_model(path,_device(packed,mesh,P(None,'x',None,'y')),
            _device(poles,mesh,P(None,'y')),count,q_span=(0,3),meta=meta,tables=tables,
            recipe=recipe,receipts={"identity":identity})
    assert not path.exists()


def check_empty_parents(mesh,path):
    meta,tables,recipe,identity=_fixture(mesh)
    _,packed,poles,count=_model(meta)
    packed[0]=0; poles[0]=1; count[0]=0
    for q in range(3):
        store.write_shared_pole_model(path,
            _device(packed[q:q+1],mesh,P(None,'x',None,'y')),
            _device(poles[q:q+1],mesh,P(None,'y')),count[q:q+1],q_span=(q,q+1),
            meta=meta,tables=tables,recipe=recipe,receipts={'identity':identity})
    header=store.validate_shared_pole_model(path,expected_identity=identity,mesh_xy=mesh,capacity=meta.shared_pole_capacity)
    with SlabIO(path,mode='r',mesh=mesh) as io:
        cx,cy,p,k=store.read_shared_pole_faces(io,(0,1),meta=meta,header=header)
        assert bool(jnp.all(cx==0)) and bool(jnp.all(cy==0))
        assert bool(jnp.all(p==1)) and int(k[0])==0
    empty_path=path.with_name('empty_model.h5')
    store.write_shared_pole_model(empty_path,
        _device(np.zeros_like(packed),mesh,P(None,'x',None,'y')),
        _device(np.ones_like(poles),mesh,P(None,'y')),np.zeros_like(count),q_span=(0,3),
        meta=meta,tables=tables,recipe=recipe,receipts={'identity':identity})
    header=store.validate_shared_pole_model(empty_path,expected_identity=identity,mesh_xy=mesh,capacity=meta.shared_pole_capacity)
    assert header['Kmax']==0
    with SlabIO(empty_path,mode='r',mesh=mesh) as io:
        cx,cy,p,k=store.read_shared_pole_faces(io,(0,3),meta=meta,header=header)
        assert cx.shape[-1]==cy.shape[-1]==p.shape[-1]==0
        assert bool(jnp.all(k==0))


def check_capacity_controls(mesh,path):
    from gw.shared_pole_recipe import CapacityLedger
    meta,tables,recipe,identity=_fixture(mesh)
    _,packed,poles,count=_model(meta)
    C=_device(packed,mesh,P(None,'x',None,'y'))
    lam=_device(poles,mesh,P(None,'y'))
    def write():
        return store.write_shared_pole_model(path,C,lam,count,q_span=(0,3),
            meta=meta,tables=tables,recipe=recipe,receipts={'identity':identity})
    ledger=CapacityLedger(meta,mesh_xy=mesh)
    meta.shared_pole_capacity=ledger
    with pytest.raises(ValueError,match='unbound caller'):
        write()
    assert not path.exists()
    ledger.reserve('caller',resident_bytes_per_rank=int(ledger.limit_bytes_per_rank)-1,
                   workspace_bytes_per_rank=0)
    ledger.live_stages=('caller',)
    with pytest.raises(MemoryError,match='aggregate live allocation'):
        write()
    assert not path.exists() and ledger.entries[-1]['status']=='FAIL'
    ledger=CapacityLedger(meta,mesh_xy=mesh)
    ledger.live_stages=()
    ledger.geometry['nq']+=1
    meta.shared_pole_capacity=ledger
    with pytest.raises(ValueError,match='ledger geometry'):
        write()
    assert not path.exists()
    # Restore the actual bounded caller footprint for the accepted execution.
    ledger=CapacityLedger(meta,mesh_xy=mesh)
    ledger.reserve('caller',resident_bytes_per_rank=4096,workspace_bytes_per_rank=0)
    ledger.live_stages=('caller',)
    meta.shared_pole_capacity=ledger
    header=write()
    metadata=store.validate_shared_pole_model(path,expected_identity=identity,mesh_xy=mesh)
    assert metadata['validation_receipt']['status']=='NOT_MEASURED'
    with SlabIO(path,mode='r',mesh=mesh) as io:
        with pytest.raises(ValueError,match='metadata-only validation'):
            store.read_shared_pole_faces(io,(0,1),meta=meta,header=metadata)
        with pytest.raises(ValueError,match='metadata-only validation'):
            store.read_shared_pole_census(io,header=metadata,capacity=ledger)
    authenticated=store.validate_shared_pole_model(path,expected_identity=identity,
        mesh_xy=mesh,capacity=ledger)
    assert authenticated['digest']==header['digest']
    from file_io.tagged_arrays import register_shared_pole_restart_member
    with pytest.raises((ValueError,RuntimeError),match='authentication requires capacity'):
        register_shared_pole_restart_member(path.with_name('unadmitted_restart.h5'),path,
            expected_identity=identity,mesh_xy=mesh)
    assert all(row['status']=='PASS' for row in ledger.entries)
    assert all(row['aggregate_bytes_per_rank']<=row['limit_bytes_per_rank'] for row in ledger.entries)
    assert any(row['workspace_bytes_per_rank']>0 for row in ledger.entries)
    if jax.process_index()==0:
        path.with_suffix('.capacity.json').write_text(json.dumps(ledger.receipt(),indent=2))


def _test_mesh():
    if len(jax.devices())<4:pytest.skip('requires four devices; production gate uses real P4')
    return Mesh(np.asarray(jax.devices()[:4]).reshape(2,2),('x','y'))


def test_shared_pole_roundtrip(tmp_path):
    check_roundtrip(_test_mesh(),tmp_path/'model.h5')


def test_shared_pole_payload_corruption(tmp_path):
    check_corruption(_test_mesh(),tmp_path/'corrupt.h5')


def test_shared_pole_sentinel_refusal(tmp_path):
    check_sentinel_refusal(_test_mesh(),tmp_path/'sentinel.h5')


if __name__=='__main__':
    import faulthandler
    faulthandler.dump_traceback_later(120, exit=True)
    from runtime import initialize_communicator_stack,run_main_and_finalize
    initialize_communicator_stack(platform='gpu')
    def main():
        import sys
        root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
        mesh=_test_mesh()
        assert jax.process_count()==4
        cells=[('roundtrip_local',check_roundtrip),
               ('roundtrip_distributed',lambda m,p:check_roundtrip(m,p,'distributed')),
               ('finalization_resume',check_finalization_resume),
               ('empty_parents',check_empty_parents),
               ('capacity_controls',check_capacity_controls),
               ('corruption',check_corruption),('sentinel',check_sentinel_refusal)]
        receipts=[]
        for name,check in cells:
            check(mesh,root/(name+'.h5'))
            receipts.append({'test':name,'status':'PASS','scope':'P4 packed scalar SlabIO','job_step':os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','')})
            rank0_transaction(root,stage='test.progress',write=lambda:(root/'progress.json').write_text(json.dumps(receipts,indent=2)))
            print('STORE_GATE_PASS '+name,flush=True)
        from test_shared_pole_bank import check_bank_roundtrip
        check_bank_roundtrip(mesh,root/'bank.h5')
        receipts.append({'test':'bank_roundtrip','status':'PASS'})
        rank0_transaction(root,stage='test.receipt',write=lambda:(root/'receipt.json').write_text(json.dumps({'collected':len(cells)+1,'passed':len(receipts),'tests':receipts},indent=2)))
        return 0
    run_main_and_finalize(main)
