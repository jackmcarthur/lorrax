"""Compile only: production-sized local pair tiles; no large buffers allocated."""
from runtime import initialize_communicator_stack, run_main_and_finalize, rank0_print
R=initialize_communicator_stack()
import argparse,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding,PartitionSpec as P
from common.shard_map import shard_map
from common.collectives import rank0_transaction
from ffi.fft import make_fused_conv_kparent,make_fused_conv_kplane
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()

def main():
 mesh=R.mesh;nk=8000;np_=1062;rows=[]
 for ns,m in [(2,452),(4,300)]:
  rng=np.random.default_rng(ns);coefs=rng.normal(size=(nk,ns*ns,ns*ns))+1j*rng.normal(size=(nk,ns*ns,ns*ns))
  def local(D,E,C):
   irr=jnp.arange(nk,dtype=jnp.int32)%np_;sym=jnp.zeros(nk,jnp.int32)
   perm=jnp.arange(m,dtype=jnp.int32)[None,:];wrap=jnp.zeros((1,m,3),jnp.float64)
   tables=(irr,sym,perm,perm,wrap,wrap,jnp.zeros((np_,3),jnp.float64),sym,C,C)
   call=make_fused_conv_kparent(mesh,(20,20,20),ns,(m,m),perm_l=np.arange(ns),phase_l=np.ones(ns),perm_r=np.arange(ns),phase_r=np.ones(ns))
   return call(D,E,tables)
  spec=P(None,None,'x',None,'y');sh=NamedSharding(mesh,spec)
  f=jax.jit(shard_map(local,mesh=mesh,in_specs=(spec,spec,P()),out_specs=P(None,'x','y'),check_vma=False))
  arg=jax.ShapeDtypeStruct((np_,ns,m*2,ns,m*2),jnp.complex128,sharding=sh)
  coef=jax.ShapeDtypeStruct(coefs.shape,jnp.complex128,sharding=NamedSharding(mesh,P()))
  t=time.monotonic();stats=f.lower(arg,arg,coef).compile().memory_analysis()
  row=dict(kind='parent',ns=ns,local_mu=m,local_nu=m,production_equivalent_ranks=16 if ns==2 else 36,argument_bytes=stats.argument_size_in_bytes,output_bytes=stats.output_size_in_bytes,temp_bytes=stats.temp_size_in_bytes,alias_bytes=stats.alias_size_in_bytes,compile_s=time.monotonic()-t,scope='shape-only CUDA AOT figure; excludes surrounding producer/live WFNs; no device allocations of large placeholders')
  rows.append(row);rank0_print(json.dumps(row),flush=True)
 for ns,c,g,p_ in [(2,6,1,729),(4,452,156,14)]:
  call=make_fused_conv_kplane(mesh,(20,20,20),ns,perm_l=np.arange(ns),phase_l=np.ones(ns),perm_r=np.arange(ns),phase_r=np.ones(ns))
  f=jax.jit(call)
  D=jax.ShapeDtypeStruct((nk,g,ns,2*c,ns,p_),jnp.complex128)
  F=jax.ShapeDtypeStruct((nk,g,p_),jnp.complex128)
  t=time.monotonic();stats=f.lower(D,F).compile().memory_analysis()
  row=dict(kind='plane',ns=ns,local_c=c,g=g,p=p_,argument_bytes=stats.argument_size_in_bytes,output_bytes=stats.output_size_in_bytes,temp_bytes=stats.temp_size_in_bytes,alias_bytes=stats.alias_size_in_bytes,compile_s=time.monotonic()-t,scope='rank-local CUDA AOT figure; stress plane may exceed card in inputs/output; only temp figure tests gather bound')
  rows.append(row);rank0_print(json.dumps(row),flush=True)
 rank0_transaction(a.output,stage='pair memory compile-only',write=lambda:Path(a.output).write_text(json.dumps(rows,indent=2)+'\n'))
 return 0
run_main_and_finalize(main)
