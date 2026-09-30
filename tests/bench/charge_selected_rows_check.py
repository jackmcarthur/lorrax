"""Selected scalar parent rows: native full-grid oracle and no-field AOT."""
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
ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--aot-only',action='store_true');a=ap.parse_args()
def plan(kg,np_,m,anti):
 nk=int(np.prod(kg));rng=np.random.default_rng(913);sym=(np.arange(nk)%2).astype(np.int32)
 U=np.array([[np.cos(.23),1j*np.sin(.23)],[1j*np.sin(.23),np.cos(.23)]])
 return CentroidKUnfoldPlan(R.mesh,identity_square_grouped_shard_layout(m,m,(2,2)),np.arange(nk,dtype=np.int32)%np_,sym,np.stack([np.arange(m),np.r_[np.arange(m//2)[::-1],np.arange(m//2,m)[::-1]]]),rng.integers(-1,2,(2,m,3)).astype(float),rng.uniform(-.4,.4,(np_,3)),np.broadcast_to(U,(nk,2,2)),1 if anti else 2,2)
def main():
 rows=[];rep=lambda n:NamedSharding(R.mesh,P(*([None]*n)))
 if a.aot_only:
  from file_io import WFNReader
  from symmetry_maps import q_negation_index
  base=Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Fe/73_soc80_20nscf_closed_buffer_20260930/qe')
  w=WFNReader(str(base/'WFN.h5'),mesh=R.mesh,qe_schema=str(base/'data-file-schema.xml'));s=w.symmetry();kg=tuple(map(int,w.kgrid));np_,m,nb=1062,904,180
  pl=CentroidKUnfoldPlan(R.mesh,identity_square_grouped_shard_layout(m,m,(2,2)),np.asarray(s.irr_idx_k),np.asarray(s.sym_idx_k),np.tile(np.arange(m),(2*int(w.ntran),1)),np.zeros((2*int(w.ntran),m,3)),np.asarray(s.unfolded_kpts)[np.asarray(s.kirr_fullids)],np.asarray(s.spinor_action(s.sym_idx_k,nspinor=2)),int(w.ntran),2,parent_full_rows=np.asarray(s.kirr_fullids,dtype=np.int32))
  neg=np.asarray(q_negation_index(kg));parents=np.asarray(s.kirr_fullids);q=tuple(dict.fromkeys(parents.tolist()+neg[parents].tolist()));assert len(q)==2120,len(q)
  fn=factory(R.mesh,kg,2,(np_,nb,m,2),k_unfold_plan=pl,layout='axis',selected_q=q,pair_mode='direct',ordered=True,bank_carry=True)
  sd=lambda sh,dt,sp:jax.ShapeDtypeStruct(sh,dt,sharding=NamedSharding(R.mesh,sp))
  import minimax
  cap=minimax.RESPONSE_NODE_CAPACITY
  args=(sd((cap,),np.complex128,P()),sd((2,2,cap),np.complex128,P()),sd((np_,2,m,nb),np.complex128,P(None,None,'x',None)),sd((np_,nb,2,m),np.complex128,P(None,None,None,'y')),sd((np_,nb),np.float64,P()),sd((np_,nb),np.complex128,P()),sd((np_,nb),np.complex128,P()),sd((2,),np.float64,P()),sd((2,len(q),m,m),np.complex128,P(None,None,'x','y')))
  t=time.monotonic();exe=fn.lower(*args).compile();ma=exe.memory_analysis()
  from runtime.aot_memory import aot_kernel_peak_bytes
  priced=aot_kernel_peak_bytes(exe)
  row=dict(scope='native Fe unitary metadata, identity proxy centroids, P4 local452 proxy P16 physical1808; no field allocation',parents=np_,grid=kg,selected_q=len(q),physical_m=1808,proxy_m=m,bands=nb,local_mu=452,argument_bytes=ma.argument_size_in_bytes,output_bytes=ma.output_size_in_bytes,temp_bytes=ma.temp_size_in_bytes,alias_bytes=ma.alias_size_in_bytes,resident_increment=int(priced.resident_increment),carry_bytes=2*len(q)*452**2*16,compile_s=time.monotonic()-t)
  rows.append(row);rank0_print(json.dumps(row),flush=True);rank0_transaction(a.output+'.hlo',stage='selected charge HLO',write=lambda:Path(a.output+'.hlo').write_text(exe.as_text()));w.close()
 else:
  kg=(4,4,4);np_,m,nb=3,8,6;rng=np.random.default_rng(84)
  for anti in (False,True):
   pl=plan(kg,np_,m,anti);psi=rng.normal(size=(np_,nb,2,m))+1j*rng.normal(size=(np_,nb,2,m));psi/=np.sqrt(nb*2*m)
   for layout in ('axis','face'):
    pm=device_put_process_local(np.conj(psi).transpose(0,2,3,1),NamedSharding(R.mesh,P(None,None,'x',None if layout=='axis' else 'y')))
    pn=device_put_process_local(psi,NamedSharding(R.mesh,P(None,None if layout=='axis' else 'x',None,'y')))
    en=rng.uniform(-.5,.8,(np_,nb));f=1/(1+np.exp(en*3));u=(1-f)*(1+.2j)
    for ordered in (False,True):
     for mode in ('retarded','kms_static','direct'):
      kw=dict(k_unfold_plan=pl,layout=layout,selected_q=(0,1,23),pair_mode=mode,ordered=ordered,bank_carry=mode=='direct')
      new=factory(R.mesh,kg,2,(np_,nb,m,2),**kw)
      with patch('gw.w_isdf._chi_door_serves',return_value=False):old=factory(R.mesh,kg,2,(np_,nb,m,2),**kw)
      times=np.array([.13+.07j,.27-.12j]) if mode=='direct' else np.array([.0,.17]);proj=np.array([[.7+.2j,-.2+.1j],[.1-.1j,.3+.04j]])
      if mode=='direct':proj=np.stack([proj,np.conj(proj)*.4])
      ref=np.array([.11,.07]) if mode=='direct' else np.array([3.,.11]) if mode=='kms_static' else np.array(.11)
      args=tuple(device_put_process_local(v,rep(np.ndim(v))) for v in (times,proj))+(pm,pn)+tuple(device_put_process_local(v,rep(np.ndim(v))) for v in (en,f,u,ref))
      if mode=='direct':
       carry=lambda:device_put_process_local(np.zeros((2,3,m,m),complex),NamedSharding(R.mesh,P(None,None,'x','y')))
       nv=new(*args,carry());ov=old(*args,carry())
      else:nv=new(*args);ov=old(*args)
      n=np.asarray(gather_to_host(nv));o=np.asarray(gather_to_host(ov));err=float(np.max(abs(n-o))/max(np.max(abs(o)),1e-30));assert err<3e-12,(anti,layout,ordered,mode,err)
      row=dict(antiunitary=anti,layout=layout,ordered=ordered,mode=mode,relative_error=err);rows.append(row);rank0_print(json.dumps(row),flush=True)
  # A wrong minus-q selection must be detected by this magnetic complex fixture.
  bad=factory(R.mesh,kg,2,(np_,nb,m,2),**dict(kw,selected_q=(0,63,41)))
  b=np.asarray(gather_to_host(bad(*args,carry())));negative=float(np.max(abs(b-n))/max(np.max(abs(n)),1e-30));assert negative>1e-4,negative;rows.append(dict(negative_minus_q_error=negative))
 rank0_transaction(a.output,stage='selected charge evidence',write=lambda:Path(a.output).write_text(json.dumps(rows,indent=2)+'\n'));return 0
run_main_and_finalize(main)
