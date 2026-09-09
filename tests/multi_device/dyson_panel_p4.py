"""Bounded Dyson GEMM: P4/P16 numerics, sample reuse, HLO and timing receipt."""
import json
from functools import partial
import os
from pathlib import Path
import sys
import time

from runtime import initialize_communicator_stack, finalize_process
initialize_communicator_stack()
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import resolve_mesh
from distrib_la import panel_matmul
from common.shard_map import shard_map


def main():
    mesh = resolve_mesh()
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    rank = jax.process_index()
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    sample_face = NamedSharding(mesh, P(None, None, 'x', 'y'))
    rng = np.random.default_rng(417)
    q, n, ns = 2, 24, 3
    a = rng.normal(size=(q, n, n)) + 1j*rng.normal(size=(q, n, n))
    b = rng.normal(size=(q, ns, n, n)) + 1j*rng.normal(size=(q, ns, n, n))
    put = lambda x, spec: jax.make_array_from_callback(x.shape, spec, lambda ix: x[ix])
    aa, bb = put(a, face), put(b, sample_face)
    budget = 16*q*(n//mesh.shape['x'])*(n//mesh.shape['y'])
    fn = jax.jit(lambda a,b: panel_matmul(a,b,mesh=mesh,panel_bytes=budget))
    exe = fn.lower(aa,bb).compile()
    got = exe(aa,bb)
    want = put(a[:,None] @ b, sample_face)
    err = float(jnp.linalg.norm(got-want)/jnp.linalg.norm(want))
    assert err < 1e-12, err
    single = panel_matmul(aa, put(b[:,0],face), mesh=mesh,panel_bytes=budget)
    error_single = float(jnp.linalg.norm(single-want[:,0])/jnp.linalg.norm(want[:,0]))
    assert error_single < 1e-12, error_single
    # A wrong conjugation must be visible to this complex non-Hermitian test.
    red = float(jnp.linalg.norm(got-put(a.conj()[:,None]@b,sample_face))/jnp.linalg.norm(want))
    assert red > 0.1, red
    jax.block_until_ready(got)
    times=[]
    for _ in range(5):
        start=time.perf_counter();jax.block_until_ready(exe(aa,bb));times.append(time.perf_counter()-start)
    memory=exe.memory_analysis()
    row=dict(status='PASS',job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        rank=rank,mesh=dict(mesh.shape),relative_error=err,single_error=error_single,
        conjugation_red=red,panel_budget_bytes=budget,seconds=times,
        compiled_memory={key:getattr(memory,key) for key in ['argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes']},
        device_memory=jax.local_devices()[0].memory_stats())
    # Reference is exactly the retired full-row/full-column A-build. Kept
    # only in this controlled A/B gate, never as a production route.
    @partial(shard_map, mesh=mesh, in_specs=(P(None,'x','y'),P(None,'x','y')),
             out_specs=P(None,'x','y'),check_vma=False)
    def gathered(a,b):
        left=jax.lax.all_gather(a,'y',axis=2,tiled=True)
        right=jax.lax.all_gather(b,'x',axis=1,tiled=True)
        return left@right

    # Production-sized deterministic tile callbacks avoid a replicated
    # all-parent host matrix and deliberately exercise complex orientations.
    nq, nn = 8, 368
    shape=(nq,nn,nn)
    def values(ix,phase):
        axes=[np.arange(s.start or 0,s.stop if s.stop is not None else size)
              for s,size in zip(ix,shape)]
        q0,i,j=axes[0][:,None,None],axes[1][None,:,None],axes[2][None,None,:]
        return (np.sin(i*.03+j*.07+q0+phase)+1j*np.cos(i*.09-j*.05+phase)).astype(np.complex128)/nn
    av=jax.make_array_from_callback(shape,face,lambda ix:values(ix,0.1))
    bv=jax.make_array_from_callback(shape,face,lambda ix:values(ix,0.3))
    budget=16*nq*(nn//mesh.shape['x'])*(nn//mesh.shape['y'])
    candidates={'before':jax.jit(gathered),'after':jax.jit(lambda a,b:panel_matmul(a,b,mesh=mesh,panel_bytes=budget))}
    bench={};outputs={}
    for label,fun in candidates.items():
        compiled=fun.lower(av,bv).compile()
        outputs[label]=compiled(av,bv);jax.block_until_ready(outputs[label])
        elapsed=[]
        for _ in range(5):
            start=time.perf_counter();jax.block_until_ready(compiled(av,bv));elapsed.append(time.perf_counter()-start)
        mem=compiled.memory_analysis()
        folder=root/label;folder.mkdir(exist_ok=True)
        (folder/f'panel_rank{rank}.after_optimizations.txt').write_text(compiled.as_text())
        bench[label]=dict(seconds=elapsed,compiled_memory={key:getattr(mem,key) for key in
            ['argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes']})
    bench['relative_error']=float(jnp.linalg.norm(outputs['after']-outputs['before'])/jnp.linalg.norm(outputs['before']))
    assert bench['relative_error']<1e-12,bench
    row['benchmark']=bench
    # Exercise the actual changed Dyson owner, including logical padding.
    from gw.w_isdf import _get_w_solve_fn_distributed
    logical=20
    v=np.zeros((q,n,n),np.complex128);chi=v.copy()
    v[:,:logical,:logical]=np.eye(logical)[None]+.001*a[:,:logical,:logical]
    chi[:,:logical,:logical]=.001*b[:,0,:logical,:logical]
    # Poisoned padding may not enter either the A build or the RHS.
    v[:,logical:,:]=1e3;v[:,:,logical:]=1e3
    chi[:,logical:,:]=1e3;chi[:,:,logical:]=1e3
    solve=_get_w_solve_fn_distributed(mesh,q,n,logical)
    result=solve(put(v,face),put(chi,face),jnp.asarray(1.+0j))
    expected=np.zeros_like(v)
    expected[:,:logical,:logical]=np.linalg.solve(np.eye(logical)[None]-v[:,:logical,:logical]@chi[:,:logical,:logical],v[:,:logical,:logical])
    row['dyson_relative_error']=float(jnp.linalg.norm(result-put(expected,face))/jnp.linalg.norm(result))
    assert row['dyson_relative_error']<1e-12,row['dyson_relative_error']
    (root/f'receipt_rank{rank}.json').write_text(json.dumps(row,indent=2))
    print('ARCH1_PANEL',json.dumps(row),flush=True)

main()
finalize_process()
