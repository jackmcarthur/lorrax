"""P4 nonlocal packed-map fallback; forced tiny panels, not capacity acceptance."""
from pathlib import Path
import argparse,json,os,runpy


def main(rt):
    import jax
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from common.centroid_basis import PackedCentroidBasis
    from file_io import shared_pole_store as store
    from file_io.slab_io import SlabIO
    from symmetry_maps import QirrTables,centroid_source_map_and_wrap
    from gw.mpa.sigma import (_shared_pole_w_synthesis,_shared_pole_panel_cost,
                              _shared_pole_panel_tables)
    from gw.mpa.sigma_windows import shared_pole_frequencies
    import pytest
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True)
    mesh=rt.mesh
    assert jax.process_count()==jax.device_count()==4 and mesh.shape['x']>1 and mesh.shape['y']>1
    assert pytest.main(['-q','-p','no:cacheprovider','tests/test_shared_pole_sigma_capacity.py',
        'tests/test_grouped_layout.py::test_nonclosed_group_partition_refuses_a_fake_local_gather',
        f'--junitxml={a.output}/metadata_rank{jax.process_index()}.xml'])==0
    h=runpy.run_path('tests/test_shared_pole_store.py')
    meta,tables,recipe,identity=h['_fixture'](mesh)
    cents=meta.mu_basis.canonical_indices[[0,2,4,6,1,3,5]]
    sym=tables['sym']
    meta.mu_basis=PackedCentroidBasis.build(cents,sym,meta.fft_grid,mesh,identity=True)
    meta.kgrid=(6,1,1)
    perm,wraps=centroid_source_map_and_wrap(cents,sym.sym_matrices,sym.translations,np.asarray(meta.fft_grid),extend_trs=True)
    # Planted lattice wraps make the q-dependent endpoint phase observable.
    wraps=np.asarray(wraps).copy();wraps[1,:,0]=np.arange(7)%2;wraps[3]=wraps[1]
    irr=np.array([0,1,2,0,1,2],np.int32);ops=np.array([0,0,0,2,3,1],np.int32)
    q=np.array([[.125,0,0],[.25,0,0],[.375,0,0]])
    tables['qirr']=QirrTables(irr_idx_q=irr,sym_idx_q=ops,q_irr_frac=q,
                              sym_perm=perm,L_table=wraps,n_sym_spatial=2)
    with pytest.raises(ValueError,match='moves canonical row'):
        meta.mu_basis.layout.axis.pack_permutations_host(perm)
    packed_perm=meta.mu_basis.layout.axis.pack_permutations_host(perm,require_local=False)
    C,packed,poles,counts=h['_model'](meta)
    put=lambda x,spec:h['_device'](np.asarray(x),mesh,spec)
    path=a.output/'model.h5'
    header=store.write_shared_pole_model(path,put(packed,P(None,'x',None,'y')),
        put(poles,P(None,'y')),counts,q_span=(0,3),meta=meta,tables=tables,recipe=recipe,
        receipts={'identity':identity,'scope':'planted nonlocal endpoint algebra'})
    plan=_shared_pole_panel_tables(meta,header,(0,3),mesh_xy=mesh)
    assert not plan['certificates']['x']['is_local'] and not plan['certificates']['y']['is_local']
    b,c=1,2
    cost=_shared_pole_panel_cost(meta,header,b,c,mesh_xy=mesh,local=False)
    forced=dict(status='PASS',parent_capacity=b,column_capacity=c,endpoint_budgets=cost['endpoint_budgets'])
    omega=np.sqrt(poles);selected=(omega>1)&(omega<=4)&(np.arange(6)[None,:]<counts[:,None])
    weights=np.where(selected,np.exp(-1j*(omega-.6)*(.7+.2j))/(2*omega),0)
    expected=[]
    for parent,op in zip(irr,ops):
        factor=C[parent,perm[op],0,:]*np.exp(2j*np.pi*(wraps[op]@q[parent]))[:,None]
        if op>=2:factor=factor.conj()
        expected.append((factor*weights[parent])@factor.conj().T)
    expected=meta.mu_basis.pack_host(meta.mu_basis.pack_host(np.asarray(expected),axis=1),axis=2)
    args=(None,None,put(np.arange(3,dtype=np.int32),P()),
          put(np.tile([1,4,-np.inf,-np.inf,np.inf,np.inf],(3,1)),P()),
          put(np.ones(3),P()),put(np.asarray(.6),P()),put(np.asarray(.7+.2j),P()))
    with SlabIO(path,mode='r',mesh=mesh) as io:
        build=_shared_pole_w_synthesis(io,meta,header,shared_pole_frequencies(poles,counts),forced,mesh_xy=mesh)
        got=build(*args)
        error=float(jax.numpy.max(jax.numpy.abs(got-put(expected,P(None,'x','y')))))
        assert error<1e-10,error
    report=dict(status='PASS',job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        dense_error=error,default_nonlocal_refusal='PASS',packed_map=packed_perm.tolist(),
        route_cost=cost,scope='P4 real canonical reader and forced q1/K2 nonlocal action; scalar metadata tests',
        aggregate_capacity='NOT_MEASURED; forced execution fixture is not an admitted production schedule')
    (a.output/f'receipt_rank{jax.process_index()}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    rt=initialize_communicator_stack();run_main_and_finalize(lambda:main(rt))
