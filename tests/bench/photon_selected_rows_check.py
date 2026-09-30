"""Typed selected photon rows against the incumbent full-k gamma equation."""
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
from gw.photon_layout import PhotonFamilies,PhotonBasisLayout
from gw.w_isdf import _get_chi_fractional_contour_kernel_face as factory
ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--aot-only',action='store_true');ap.add_argument('--donated-direct',action='store_true');a=ap.parse_args()
def plan(kg,np_,m):
 nk=int(np.prod(kg));rng=np.random.default_rng(913+m);sym=(np.arange(nk)%2).astype(np.int32)
 perm=np.stack([np.arange(m),np.concatenate([np.arange(m//2)[::-1],np.arange(m//2,m)[::-1]])]).astype(np.int32)
 L=rng.integers(-1,2,(2,m,3)).astype(float)
 theta=.23;U=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]],complex)
 spin=np.zeros((nk,4,4),complex);spin[:,:2,:2]=U;spin[:,2:,2:]=np.where(sym==0,1.,-1.)[:,None,None]*U
 return CentroidKUnfoldPlan(R.mesh,identity_square_grouped_shard_layout(m,m,(2,2)),np.arange(nk,dtype=np.int32)%np_,sym,perm,L,rng.uniform(-.4,.4,(np_,3)),spin,1,4)
def main():
 receipts=[];mesh=R.mesh;rep=lambda n:NamedSharding(mesh,P(*([None]*n)))
 if a.aot_only:
  kg=(20,20,20);np_,nb=1062,180;extents=(600,600)
  plans=tuple(plan(kg,np_,m) for m in extents)
  layout=PhotonBasisLayout.from_centroid_extents(*extents,mesh,packed=True)
  photon=PhotonFamilies(plans,layout,layout)
  sd=lambda shape,dtype,spec:jax.ShapeDtypeStruct(shape,dtype,sharding=NamedSharding(mesh,spec))
  modes=[('direct',tuple(range(128)))] if a.donated_direct else [('retarded',tuple(range(128))),('kms_static',(0,)),('direct',tuple(range(128)))]
  for mode,q in modes:
   fn=factory(mesh,kg,1 if mode!='direct' else 2,(np_,nb,layout.packed_extent,4),layout='face',selected_q=q,pair_mode=mode,ordered=True,vertex=photon,bank_carry=a.donated_direct)
   shape=(2,2,1) if mode=='direct' else (1,1)
   refshape=(2,) if mode!='retarded' else ()
   args=(sd((1,),np.complex128 if mode=='direct' else np.float64,P()),sd(shape,np.complex128,P()),tuple(sd((np_,4,m,nb),np.complex128,P(None,None,'x','y')) for m in extents),tuple(sd((np_,nb,4,m),np.complex128,P(None,'x',None,'y')) for m in extents),sd((np_,nb),np.float64,P()),sd((np_,nb),np.complex128,P()),sd((np_,nb),np.complex128,P()),sd(refshape,np.float64,P()))
   if a.donated_direct:args+=(sd((2,len(q),layout.packed_extent,layout.packed_extent),np.complex128,P(None,None,'x','y')),)
   t=time.monotonic();exe=fn.lower(*args).compile();ma=exe.memory_analysis()
   row=dict(mode=mode,aot_only=True,bank_carry=a.donated_direct,fe_equivalent_mesh=36,local_mu=300,selected_q=len(q),argument_bytes=ma.argument_size_in_bytes,output_bytes=ma.output_size_in_bytes,temp_bytes=ma.temp_size_in_bytes,alias_bytes=ma.alias_size_in_bytes,new_bytes=ma.output_size_in_bytes+ma.temp_size_in_bytes-ma.alias_size_in_bytes,wall_s=time.monotonic()-t);receipts.append(row);rank0_print(json.dumps(row),flush=True)
  rank0_transaction(a.output,stage='selected photon AOT',write=lambda:Path(a.output).write_text(json.dumps(receipts,indent=2)+'\n'))
  return 0
 kg=(4,4,4);np_,nb=3,4;extents=(4,8);plans=tuple(plan(kg,np_,m) for m in extents)
 layout=PhotonBasisLayout.from_centroid_extents(*extents,mesh,packed=True)
 photon=PhotonFamilies(plans,layout,layout)
 rng=np.random.default_rng(77);pm=[];pn=[]
 for m in extents:
  psi=rng.normal(size=(np_,nb,4,m))+1j*rng.normal(size=(np_,nb,4,m));psi/=np.sqrt(nb*4*m)
  pm.append(device_put_process_local(np.conj(psi).transpose(0,2,3,1),NamedSharding(mesh,P(None,None,'x','y'))))
  pn.append(device_put_process_local(psi,NamedSharding(mesh,P(None,'x',None,'y'))))
 en=rng.uniform(-.5,.8,(np_,nb));f=1/(1+np.exp(en*3));u=1-f
 for mode in ('retarded','kms_static','direct'):
  kw=dict(layout='face',selected_q=(0,1,23),pair_mode=mode,ordered=True,vertex=photon)
  new=factory(mesh,kg,1,(np_,nb,layout.packed_extent,4),**kw)
  with patch('gw.w_isdf._photon_selected_rows_serve',return_value=False):old=factory(mesh,kg,1,(np_,nb,layout.packed_extent,4),**kw)
  times=np.array([0.,.17]) if mode=='retarded' else np.array([0.,.3]) if mode=='kms_static' else np.array([.13+.07j,.27-.12j])
  projection=np.array([[.7+.2j,-.2+.1j]])
  if mode=='direct':projection=np.stack([projection,np.conj(projection)*.4])
  cases=[(f*en,-1j*u*en**2),(f*en**2,u)] if mode=='retarded' else [(np.ones_like(f),np.ones_like(u))] if mode=='kms_static' else [(f,u)]
  for lo,hi in cases:
   ref=np.array([3.,.11]) if mode=='kms_static' else np.array([.11,.07]) if mode=='direct' else np.array(.11)
   args=tuple(device_put_process_local(v,rep(np.ndim(v))) for v in (times,projection))+(tuple(pm),tuple(pn),device_put_process_local(en,rep(2)),device_put_process_local(lo,rep(2)),device_put_process_local(hi,rep(2)),device_put_process_local(ref,rep(np.ndim(ref))))
   t=time.monotonic();nv=new(*args);ov=old(*args);nv.block_until_ready();ov.block_until_ready()
   n=np.asarray(gather_to_host(nv));o=np.asarray(gather_to_host(ov));err=float(np.max(abs(n-o))/max(np.max(abs(o)),1e-30))
   row=dict(mode=mode,upper_complex=bool(np.iscomplexobj(hi)),relative_error=err,wall_s=time.monotonic()-t);rank0_print(json.dumps(row),flush=True);receipts.append(row);assert err<3e-12,err
 from common.gamma_matrices import gamma_perm_phase_host
 def wrong_phase(channel):
  perm,phase=gamma_perm_phase_host(channel)
  return perm,phase*(1.01 if channel==2 else 1.)
 with patch('common.gamma_matrices.gamma_perm_phase_host',side_effect=wrong_phase):
  bad=factory(mesh,kg,1,(np_,nb,layout.packed_extent,4),**kw)
 bn=np.asarray(gather_to_host(bad(*args)));negative=float(np.max(abs(bn-n))/max(np.max(abs(n)),1e-30))
 assert negative>1e-4,negative
 receipts.append(dict(negative_gamma2_phase=negative));rank0_print(json.dumps(receipts[-1]),flush=True)
 rank0_transaction(a.output,stage='selected photon parity',write=lambda:Path(a.output).write_text(json.dumps(receipts,indent=2)+'\n'))
 return 0
run_main_and_finalize(main)
