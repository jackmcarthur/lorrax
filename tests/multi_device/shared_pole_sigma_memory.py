"""Real-shape W admission/peak and matched full-k inherited Sigma compilation.

No physical Na accuracy claim: dimensions are Na, factors are planted, and
reader transport is replaced by on-device panel generation. Canonical I/O is
covered by the separate store gate. All arrays stay distributed on compute.
"""
from pathlib import Path
from functools import partial
import argparse,json,os
from types import SimpleNamespace as NS


def main(rt):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.grouped_layout import identity_square_grouped_shard_layout
    from common.collectives import device_put_process_local
    from gw.shared_pole_recipe import CapacityLedger
    from gw.mpa.sigma import (_shared_pole_memory_schedule,_shared_pole_w_synthesis,
                              _shared_pole_inherited_peak)
    import file_io.shared_pole_store as store
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--profile',action='store_true');a=p.parse_args();a.output.mkdir(exist_ok=True)
    mesh=rt.mesh;side=mesh.shape['x'];assert mesh.shape['y']==side and side in (2,4)
    assert jax.process_count()==jax.device_count()==side*side
    n,m,Q,b,K=896,(912 if side==2 else 960),512,29,1664
    layout=identity_square_grouped_shard_layout(n,m,(side,side))
    meta=NS(nk_tot=Q,nspinor=1,n_rmu=n,mu_basis=NS(n_packed=m,layout=layout,active_mask=layout.axis.active_mask))
    ledger=CapacityLedger(meta,mesh_xy=mesh);meta.shared_pole_capacity=ledger;ledger.live_stages=()
    header=dict(n_q_full=Q,n_q_irr=b,n_mu_logical=n,nspinor=1,Kmax=K,
        qirr=dict(irr_idx_q=np.arange(Q,dtype=np.int32)%b,sym_idx_q=np.zeros(Q,np.int32),
                  sym_perm=np.arange(n,dtype=np.int32)[None,:],L_table=np.zeros((1,n,3),np.int32),
                  q_irr_frac=np.zeros((b,3)),n_sym_spatial=1))
    schedule=_shared_pole_memory_schedule(meta,header,mesh_xy=mesh)
    freq=[np.linspace(.1,10,K)]*b
    def panel(_io,span,*,meta,header,column_span=None):
        lo,hi=span;c0,c1=column_span or (0,K)
        shape=(hi-lo,m,1,c1-c0)
        specs=(P(None,'x',None,None),P(None,'y',None,None),P(),P())
        @partial(jax.jit,out_shardings=tuple(NamedSharding(mesh,s) for s in specs))
        def create():
            C=jnp.broadcast_to((jnp.arange(m)<n)[None,:,None,None],shape).astype(jnp.complex128)/np.sqrt(K)
            poles=jnp.broadcast_to(jnp.asarray(freq[0][c0:c1]**2),(hi-lo,c1-c0))
            return C,C,poles,jnp.full(hi-lo,c1-c0,jnp.int64)
        return create()
    import runtime.aot_memory as memory_owner
    original_peak=memory_owner.aot_kernel_peak_bytes
    hlo_rows=[]
    def capture_peak(compiled):
        import hashlib,re
        text=compiled.as_text()
        path=a.output/f'body_rank{jax.process_index()}_{len(hlo_rows)}.hlo'
        path.write_text(text)
        collectives=[line for line in text.splitlines() if re.search(
            r'\b(all-gather|all-reduce|all-to-all|collective-permute|reduce-scatter)\(',line)]
        assert not collectives,collectives
        hlo_rows.append(dict(path=str(path),sha256=hashlib.sha256(text.encode()).hexdigest(),collectives=collectives))
        return original_peak(compiled)
    original=store.read_shared_pole_faces;store.read_shared_pole_faces=panel
    memory_owner.aot_kernel_peak_bytes=capture_peak
    try:build=_shared_pole_w_synthesis(None,meta,header,freq,schedule,mesh_xy=mesh)
    finally:
        store.read_shared_pole_faces=original
        memory_owner.aot_kernel_peak_bytes=original_peak
    put=lambda x:device_put_process_local(np.asarray(x),NamedSharding(mesh,P()))
    args=(None,None,put(np.arange(b,dtype=np.int32)),put(np.tile([0,np.inf,-np.inf,-np.inf,np.inf,np.inf],(b,1))),put(np.zeros(b,bool)),put(.6),put(.7+.2j))
    W=build(*args);W.block_until_ready()
    before=jax.local_devices()[0].memory_stats()
    if a.profile:
        import ctypes
        cuda=ctypes.CDLL('libcudart.so.13');assert cuda.cudaProfilerStart()==0
        for _ in range(25):
            del W
            W=build(*args);W.block_until_ready()
        assert cuda.cudaProfilerStop()==0
    stats=jax.local_devices()[0].memory_stats()
    output_bytes=16*Q*m*m//(side*side)
    measured={key:int(stats[key]) for key in ('bytes_in_use','peak_bytes_in_use','bytes_limit') if key in stats}
    from jax.experimental import multihost_utils
    peaks=np.asarray(multihost_utils.process_allgather(np.asarray(measured['peak_bytes_in_use'],np.int64)))
    measured['maximum_over_ranks']=int(peaks.max())
    if measured['maximum_over_ranks']<=ledger.limit_bytes_per_rank:
        ledger.record_measured_peak(measured['maximum_over_ranks'],reason=(
            'Isolated real-shape planted W process, maximum JAX allocator peak over ranks; '
            'includes the one inherited output W, a conservative new-object bound; '
            'reader is on-device generator, spatial kernel is compiled only after this peak'))
    # Compile the SAME full-k geometry through both inherited entry routes.
    def abstract(shape,dtype,spec):return jax.ShapeDtypeStruct(shape,dtype,sharding=NamedSharding(mesh,spec))
    x=abstract((Q,1,m,88),np.complex128,P(None,None,'x','y'))
    y=abstract((Q,88,1,m),np.complex128,P(None,'x',None,'y'))
    energy=abstract((Q,88),np.float64,P());scalar=abstract((),np.float64,P());tau=abstract((),np.complex128,P())
    inherited=_shared_pole_inherited_peak((x,y,y,x,energy,energy,None,None,None,None,None,scalar,scalar,tau),meta,
        mesh_xy=mesh,kgrid=(8,8,8),brackets=None,pack_brackets=False,
        face_kwargs=dict(layout='face',face_shape=(Q,88,m,1),face_band_extent=88))
    report=dict(status='PASS',job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        geometry=dict(Q=Q,n=n,m=m,K=K,parents=b,px=side,py=side),schedule=schedule,
        W_shape=list(W.shape),W_bytes_per_rank=output_bytes,device_memory=measured,synthesis_hlo=hlo_rows,
        device_peak_in_U=measured.get('peak_bytes_in_use',0)/ledger.U_bytes_per_rank,
        inherited_sigma_peak=inherited,capacity=ledger.receipt(),
        scope='Na dimensions, planted C, on-device reader surrogate; local W admission and actual allocator peak, full-k inherited compiler lower-bound comparison; not raw-parent/full-driver peak')
    (a.output/f'receipt_rank{jax.process_index()}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','job_step','device_peak_in_U')}),flush=True)


if __name__=='__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    rt=initialize_communicator_stack();run_main_and_finalize(lambda:main(rt))
