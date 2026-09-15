"""Actual distributed Dyson owner; low-rank analytic oracle, all matrices tiled."""
from pathlib import Path
import hashlib,json,os,sys,time
from runtime import initialize_communicator_stack,finalize_process
initialize_communicator_stack()
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding,PartitionSpec as P
from common.collectives import resolve_mesh,gather_to_host
import gw.w_isdf as owner
mesh=resolve_mesh();q,n=(29,896) if mesh.size==16 else (8,368)
r=Path(sys.argv[1]);r.mkdir(parents=True,exist_ok=True)
face=NamedSharding(mesh,P(None,'x','y'));shape=(q,n,n)
i=np.arange(n);u=np.sin(i*.07)+1j*np.cos(i*.11);v=np.cos(i*.13)+1j*np.sin(i*.17)
overlap=np.vdot(v,u)/n
# V=I, chi_q=alpha_q u v^H/n, so W=(I-chi)^-1 exactly by Sherman-Morrison.
def tile(ix,kind):
    axes=[np.arange(s.start or 0,s.stop if s.stop is not None else size) for s,size in zip(ix,shape)]
    qi,ri,ci=axes
    identity=np.broadcast_to(ri[None,:,None]==ci[None,None,:],(len(qi),len(ri),len(ci))).astype(np.complex128)
    if kind=='V':return identity
    alpha=(.01+.0001*qi)[:,None,None]
    chi=alpha*u[ri][None,:,None]*v[ci].conj()[None,None,:]/n
    return chi if kind=='chi' else identity+chi/(1-alpha*overlap)
def put(kind):return jax.make_array_from_callback(shape,face,lambda ix:tile(ix,kind))
V=put('V');chi=put('chi');jax.block_until_ready((V,chi))
fn=owner._get_w_solve_fn_distributed(mesh,q,n,n)
start=time.perf_counter();W=fn(V,chi,jnp.asarray(1.+0j));W.block_until_ready();wall=time.perf_counter()-start
peak=jax.local_devices()[0].memory_stats()
want=put('W')
error=float(np.asarray(gather_to_host(jnp.linalg.norm(W-want)/jnp.linalg.norm(want))))
assert error<1e-12,error
row=dict(status='PASS',job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],rank=jax.process_index(),mesh=dict(mesh.shape),q=q,n=n,owner=str(Path(owner.__file__)),source_sha256=hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest(),cold_owner_seconds=wall,device_memory_after_solve_before_oracle=peak,relative_error=error,scope='Fresh-process actual distributed Dyson solve, V=I plus complex non-Hermitian rank-one chi analytic inverse; production-sized all-q faces, not a physical sample bank.')
(r/f'receipt_rank{jax.process_index()}.json').write_text(json.dumps(row,indent=2))
if jax.process_index()==0:print(json.dumps(row),flush=True)
finalize_process()
