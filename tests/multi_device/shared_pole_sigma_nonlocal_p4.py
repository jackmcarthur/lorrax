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
                              _shared_pole_panel_tables,_shared_pole_fixed_q_policy)
    from gw.mpa.sigma_windows import shared_pole_frequencies
    import pytest
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True)
    mesh=rt.mesh
    assert jax.process_count()==jax.device_count()==4 and mesh.shape['x']>1 and mesh.shape['y']>1
    assert pytest.main(['-q','-p','no:cacheprovider','tests/test_shared_pole_sigma_capacity.py',
        'tests/test_grouped_layout.py::test_nonclosed_group_partition_refuses_a_fake_local_gather',
        f'--junitxml={a.output}/metadata_rank{jax.process_index()}.xml'])==0
    h=runpy.run_path('tests/test_shared_pole_store.py')
    meta,tables,recipe,identity=h['_sigma_fixture'](mesh,identity_layout=True)
    qt=tables['qirr'];perm,wraps=qt.sym_perm,qt.L_table
    irr,q=qt.irr_idx_q,qt.q_irr_frac
    ops=np.asarray([0,0,6,2,4,0,8,6,10])
    with pytest.raises(ValueError,match='moves canonical row'):
        meta.mu_basis.layout.axis.pack_permutations_host(perm)
    packed_perm=meta.mu_basis.layout.axis.pack_permutations_host(perm,require_local=False)
    C,packed,poles,counts=h['_model'](meta)
    C=C[...,:4];poles=poles[:,:4];counts=np.minimum(counts,4)
    packed=meta.mu_basis.pack_host(C,axis=1)
    put=lambda x,spec:h['_device'](np.asarray(x),mesh,spec)
    path=a.output/'model.h5'
    header=store.write_shared_pole_model(path,put(packed,P(None,'x',None,'y')),
        put(poles,P(None,'y')),counts,q_span=(0,3),meta=meta,tables=tables,recipe=recipe,
        receipts={'identity':identity,'scope':'planted nonlocal endpoint algebra'})
    plan=_shared_pole_panel_tables(meta,header,(0,3),mesh_xy=mesh)
    assert np.array_equal(_shared_pole_fixed_q_policy(header).unfold_sym_idx,ops)
    assert not plan['certificates']['x']['is_local'] and not plan['certificates']['y']['is_local']
    b,c=1,2
    cost=_shared_pole_panel_cost(meta,header,b,c,mesh_xy=mesh,local=False)
    forced=dict(status='PASS',parent_capacity=b,column_capacity=c,endpoint_budgets=cost['endpoint_budgets'])
    omega=np.sqrt(poles);selected=(omega>1)&(omega<=4)&(np.arange(4)[None,:]<counts[:,None])
    weights=np.where(selected,np.exp(-1j*(omega-.6)*(.7+.2j))/(2*omega),0)
    oracle_helpers = runpy.run_path('tests/multi_device/shared_pole_dense_oracle.py')
    expected = oracle_helpers['full_q_operator'](C, weights, tables, ops)
    expected=meta.mu_basis.pack_host(meta.mu_basis.pack_host(expected,axis=1),axis=2)
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
