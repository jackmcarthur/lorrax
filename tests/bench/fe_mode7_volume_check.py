"""Native scalar mode7 timing, authentic Fe k metadata and matched local volume.

Centroid actions and numerical fields are synthetic; this isolates convolution
cost and does not certify the production W synthesis or band projector.
"""
from runtime import initialize_communicator_stack,run_main_and_finalize,rank0_print
R=initialize_communicator_stack()
import argparse,json,time
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from jax.sharding import NamedSharding,PartitionSpec as P
from common.collectives import rank0_transaction
from common.grouped_layout import identity_square_grouped_shard_layout
from file_io import WFNReader
from gw.centroid_k_unfold import CentroidKUnfoldPlan
from common.fft_helpers import make_kconv_klead_unfold
ap=argparse.ArgumentParser();ap.add_argument('--wfn',required=True);ap.add_argument('--schema',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
def main():
 w=WFNReader(a.wfn,mesh=R.mesh,qe_schema=a.schema);s=w.symmetry();kg=tuple(map(int,w.kgrid));np_=int(w.nkpts);m=600;ns=2;nk=int(np.prod(kg));nsp=int(w.ntran)
 assert kg==(20,20,20) and np_==1062 and not np.any(np.asarray(s.sym_idx_k)>=nsp)
 plan=CentroidKUnfoldPlan(R.mesh,identity_square_grouped_shard_layout(m,m,(2,2)),np.asarray(s.irr_idx_k),np.asarray(s.sym_idx_k),np.tile(np.arange(m),(2*nsp,1)),np.zeros((2*nsp,m,3)),np.asarray(s.unfolded_kpts)[np.asarray(s.kirr_fullids)],np.asarray(s.spinor_action(s.sym_idx_k,nspinor=ns)),nsp,ns,parent_full_rows=np.asarray(s.kirr_fullids,dtype=np.int32))
 conv=make_kconv_klead_unfold(R.mesh,kg,plan.unfold_load_tables(),store_rows=plan.parent_full_rows,mult=-1/np.sqrt(nk))
 fn=jax.jit(lambda g,v:conv(g,None,v,conj_partner=True))
 gs=NamedSharding(R.mesh,P(None,'x',None,'y',None));vs=NamedSharding(R.mesh,P(None,'x','y'))
 ga=jax.ShapeDtypeStruct((np_,m,ns,m,ns),np.complex128,sharding=gs);va=jax.ShapeDtypeStruct((nk,m,m),np.complex128,sharding=vs)
 t=time.monotonic();exe=fn.lower(ga,va).compile();ma=exe.memory_analysis();peak=ma.argument_size_in_bytes+ma.output_size_in_bytes+ma.temp_size_in_bytes-ma.alias_size_in_bytes
 row=dict(grid=kg,parents=np_,local_mu=300,fe_equivalent_mesh=36,antiunitary_rows=0,source_scope='native mode7, authenticated Fe k/spin metadata; synthetic identity centroid actions and fields; excludes synthesis/projector',argument_bytes=ma.argument_size_in_bytes,output_bytes=ma.output_size_in_bytes,temp_bytes=ma.temp_size_in_bytes,peak_bytes=peak,compile_s=time.monotonic()-t)
 rank0_print(json.dumps(row),flush=True)
 if peak>65_000_000_000:raise ValueError('GATE benchmark_compiled_capacity: native mode7 proxy exceeds65GB before allocation')
 g=jax.jit(lambda:jnp.full(ga.shape,1+0.1j,dtype=jnp.complex128),out_shardings=gs)();v=jax.jit(lambda:jnp.full(va.shape,.2+0.3j,dtype=jnp.complex128),out_shardings=vs)();g.block_until_ready();v.block_until_ready()
 elapsed=[]
 for i in range(3):
  t=time.monotonic();out=exe(g,v);out.block_until_ready();elapsed.append(time.monotonic()-t);out.delete()
 row['dispatch_seconds']=elapsed;row['projected_1111_tau_seconds']=1111*float(np.median(elapsed[1:]));rank0_print(json.dumps(row),flush=True)
 rank0_transaction(a.output,stage='native Fe mode7 matched volume',write=lambda:Path(a.output).write_text(json.dumps(row,indent=2)+'\n'));g.delete();v.delete();w.close();return 0
run_main_and_finalize(main)
