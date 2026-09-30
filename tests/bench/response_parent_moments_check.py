"""Selected parent retarded/KMS route: old full-spin equation and large AOT price."""
from runtime import initialize_communicator_stack,run_main_and_finalize,rank0_print
R=initialize_communicator_stack()
import argparse,json,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import jax,jax.numpy as jnp
from jax.sharding import NamedSharding,PartitionSpec as P
from common.collectives import device_put_process_local,gather_to_host,rank0_transaction
from common.grouped_layout import identity_square_grouped_shard_layout
from gw.centroid_k_unfold import CentroidKUnfoldPlan
from gw.w_isdf import _get_chi_fractional_contour_kernel_face as factory
from ffi.fft import chi_unfold_scratch_bytes
ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
def plan(kg,np_,m,ns,anti):
 nk=int(np.prod(kg));rng=np.random.default_rng(913+ns)
 sym=(np.arange(nk)%2).astype(np.int32) if anti else np.zeros(nk,np.int32)
 perm=np.stack([np.arange(m),np.concatenate([np.arange(m//2)[::-1],np.arange(m//2,m)[::-1]])]).astype(np.int32)
 L=rng.integers(-1,2,(2,m,3)).astype(float)
 U=np.broadcast_to(np.eye(ns,dtype=np.complex128),(nk,ns,ns)).copy()
 theta=.23;U[1::2,0,0]=np.cos(theta);U[1::2,1,1]=np.cos(theta);U[1::2,0,1]=np.sin(theta);U[1::2,1,0]=-np.sin(theta)
 return CentroidKUnfoldPlan(R.mesh,identity_square_grouped_shard_layout(m,m,(2,2)),np.arange(nk,dtype=np.int32)%np_,sym,perm,L,rng.uniform(-.4,.4,(np_,3)),U,1,ns)
def main():
 receipts=[];mesh=R.mesh;rep=lambda n:NamedSharding(mesh,P(*([None]*n)))
 for ns in (2,4):
  kg=(20,20,20);np_,m,nb=4,4,4;pl=plan(kg,np_,m,ns,True);rng=np.random.default_rng(22+ns)
  psi=rng.normal(size=(np_,nb,ns,m))+1j*rng.normal(size=(np_,nb,ns,m));psi/=np.sqrt(nb*ns*m)
  pm=device_put_process_local(np.conj(psi).transpose(0,2,3,1),NamedSharding(mesh,P(None,None,'x',None)))
  pn=device_put_process_local(psi,NamedSharding(mesh,P(None,None,None,'y')))
  en=rng.uniform(-.5,.8,(np_,nb));f=1/(1+np.exp(en*3));u=1-f
  for ordered in (False,True):
   for mode in ('retarded','kms_static'):
    kw=dict(k_unfold_plan=pl,layout='axis',selected_q=(0,1,23),pair_mode=mode,ordered=ordered)
    new=factory(mesh,kg,1,(np_,nb,m,ns),**kw)
    with patch('gw.w_isdf._chi_door_serves',return_value=False):old=factory(mesh,kg,1,(np_,nb,m,ns),**kw)
    times=np.array([0.,.17]) if mode=='retarded' else np.array([0.,.3])
    projection=np.array([[.7+.2j,-.2+.1j]])
    cases=[(f*en,-1j*u*en**2),(f*en**2,u)] if mode=='retarded' else [(np.ones_like(f),np.ones_like(u))]
    for lo,hi in cases:
     ref=np.array(.11) if mode=='retarded' else np.array([3.,.11])
     args=tuple(device_put_process_local(v,rep(np.ndim(v))) for v in (times,projection))+(pm,pn,device_put_process_local(en,rep(2)),device_put_process_local(lo,rep(2)),device_put_process_local(hi,rep(2)),device_put_process_local(ref,rep(np.ndim(ref))))
     t=time.monotonic();nv=new(*args);ov=old(*args);nv.block_until_ready();ov.block_until_ready()
     n=np.asarray(gather_to_host(nv));o=np.asarray(gather_to_host(ov));err=float(np.max(abs(n-o))/max(np.max(abs(o)),1e-30));assert err<3e-12,err
     bad=list(args);bad[6]=bad[6]*1.01;negative=float(np.max(abs(np.asarray(gather_to_host(new(*bad)))-n))/max(np.max(abs(n)),1e-30))
     if mode=='retarded':assert negative>.005
     row=dict(ns=ns,ordered=ordered,mode=mode,upper_complex=bool(np.iscomplexobj(hi)),relative_error=err,negative_weight=negative,wall_s=time.monotonic()-t);receipts.append(row);rank0_print(json.dumps(row),flush=True)
 # Compile only: P4 mu600 has the SAME 300x300 local tile as Fe mu1800/P36.
 kg=(20,20,20);np_,m,nb,ns=1062,600,180,2;pl=plan(kg,np_,m,ns,False)
 for mode,q in [('retarded',tuple(range(131))),('kms_static',(0,))]:
  fn=factory(mesh,kg,1,(np_,nb,m,ns),k_unfold_plan=pl,layout='axis',selected_q=q,pair_mode=mode,ordered=True)
  sd=lambda shape,dtype,sh:jax.ShapeDtypeStruct(shape,dtype,sharding=sh)
  args=(sd((1,),np.float64,rep(1)),sd((1,1),np.complex128,rep(2)),sd((np_,ns,m,nb),np.complex128,NamedSharding(mesh,P(None,None,'x',None))),sd((np_,nb,ns,m),np.complex128,NamedSharding(mesh,P(None,None,None,'y'))),sd((np_,nb),np.float64,rep(2)),sd((np_,nb),np.complex128,rep(2)),sd((np_,nb),np.complex128,rep(2)),sd(() if mode=='retarded' else (2,),np.float64,rep(0 if mode=='retarded' else 1)))
  t=time.monotonic();exe=fn.lower(*args).compile();ma=exe.memory_analysis();scratch=chi_unfold_scratch_bytes(kg,ns,np_*300*300*ns*ns*16,optin=166912)
  row=dict(mode=mode,aot_only=True,local_mu=300,fe_equivalent_mesh=36,selected_q=len(q),argument_bytes=ma.argument_size_in_bytes,output_bytes=ma.output_size_in_bytes,temp_bytes=ma.temp_size_in_bytes,alias_bytes=ma.alias_size_in_bytes,foreign_scratch=scratch,new_bytes=ma.output_size_in_bytes+ma.temp_size_in_bytes-ma.alias_size_in_bytes+scratch,wall_s=time.monotonic()-t);receipts.append(row);rank0_print(json.dumps(row),flush=True)
 rank0_transaction(a.output,stage='parent moments parity',write=lambda:Path(a.output).write_text(json.dumps(receipts,indent=2)+'\n'))
 return 0
run_main_and_finalize(main)
