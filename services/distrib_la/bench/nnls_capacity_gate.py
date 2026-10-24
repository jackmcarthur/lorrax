"""Actual NNLS capacity, backup, cache and independent-value controls."""
import json,os,runpy
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import runtime
stack=runtime.initialize_communicator_stack()
import jax,numpy as np
from jax.sharding import NamedSharding,PartitionSpec as P
from lxkit import memory
from common.gpu_utils import resolve_device_budget_gb
from common.collectives import gather_to_host,prepare_mesh
from common.jax_compile_cache import compile_cache_stats
import distrib_la.nonnegative as nn
import distrib_la.workspace as ws
from distrib_la import plan
from distrib_la.resolve import resolve_backend
resolve_device_budget_gb(linalg='distributed')
mesh=prepare_mesh(stack.mesh);memory.set_nmu(2)
original=nn._lower;rows=[];executions=[];force=dict(local=False,both=False)
def lower(fn,operands,mesh,label,**kwargs):
    run,price=original(fn,operands,mesh,label,**kwargs)
    linear=kwargs.get('linear');backend=linear.backend if linear else 'plain'
    rows.append(dict(backend=backend,price=price.bytes(),terms=price.terms(),whole=price.whole))
    if linear and (force['both'] or force['local'] and backend=='one_device'):
        price=replace(price,xla=memory.Term(memory.room()+1,source='forced capacity control'),
                      vendor=memory.Term(),outside=memory.Term())
    def call(*args):
        executions.append(backend)
        return run(*args)
    call.backend=backend
    return call,price
nn._lower=lower
def inputs():
    a=np.array([[2.,.3],[.3,1.5]]);truth=np.array([1.,.5]);b=a@truth
    aa=jax.make_array_from_callback((2,2),NamedSharding(mesh,P('x','y')),lambda ix:a[ix])
    bb=jax.make_array_from_callback((2,),NamedSharding(mesh,P('x')),lambda ix:b[ix])
    return aa,bb,truth
a,b,truth=inputs();x=nn.solve_nonnegative_quadratic(a,b,mesh=mesh,n_logical=2)
np.testing.assert_allclose(np.asarray(gather_to_host(x)),truth,rtol=1e-12,atol=1e-12)
assert [r['backend'] for r in rows if r['backend']!='plain']==['one_device'],rows
fast_rows=list(rows);rows.clear();executions.clear();force['local']=True
a,b,truth=inputs();x=nn.solve_nonnegative_quadratic(a,b,mesh=mesh,n_logical=2)
np.testing.assert_allclose(np.asarray(gather_to_host(x)),truth,rtol=1e-12,atol=1e-12)
assert 'cusolvermp' in executions and 'one_device' not in executions,executions
backup_rows=list(rows);force.update(local=False,both=True)
p1=plan('solve_lu',mesh,backend='one_device',n=2,batched_route='auto',budget_bytes=1<<30)
p2=plan('solve_lu',mesh,backend='one_device',n=2,batched_route='auto',budget_bytes=2<<30)
before=compile_cache_stats();bundle=nn._programs(mesh,2,2,p1)
assert nn._programs(mesh,2,2,p2) is bundle
assert compile_cache_stats()['compiles']==before['compiles']
h,c,_=inputs();free=jax.make_array_from_callback((2,),NamedSharding(mesh,P('x')),lambda ix:np.ones(2,bool)[ix])
admit=memory.admit
def eligible(stage,routes,room,**kw):
    assert len(routes)==1 and routes[0][0]=='distributed face',routes
    return routes[0][0]
with patch.object(memory,'admit',eligible):
    runner,*_=nn._principal(mesh,2,2,p1,(h,c,free),bundle)
    assert runner.backend=='cusolvermp'
force['both']=False
assert resolve_backend('solve_lu','distributed',mesh,n=4096,room=1<<20)=='cusolvermp'
with patch.object(ws,'mesh_platform',return_value='cpu'),patch('distrib_la.loader.get_lib',side_effect=AssertionError('CUDA loaded on CPU')):
    assert ws.local_lu_workspace(object(),2,np.dtype('float64'))==0
for n,dtype in ((0,'float64'),(1<<31,'float64'),(2,'int64')):
    try:ws.local_lu_workspace(mesh,n,np.dtype(dtype))
    except ValueError:pass
    else:raise AssertionError((n,dtype))
row=dict(rank=jax.process_index(),fast=fast_rows,backup=backup_rows,
    local_workspace=ws.local_lu_workspace(mesh,2,np.dtype('float64')),
    nmu_scale_completeness=True,lazy_backup=True,forced_backup_exact=True,
    both_over_excludes_local=True,room_equivalent_factory_reuse=True,cpu_cuda_free=True)
(Path(os.environ['PROBE_OUT'])/f'capacity.rank{jax.process_index()}.json').write_text(json.dumps(row,indent=2)+'\n')
nn._lower=original
runpy.run_path(os.environ['PROBE_SOURCE']+'/tests/test_nonnegative_quadratic.py',run_name='__main__')
