"""Native full-spin reference versus bounded Dirac-quarter CCT, P4."""
from runtime import initialize_communicator_stack, run_main_and_finalize, rank0_print
R = initialize_communicator_stack()
import argparse, json
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from functools import partial
from jax.sharding import NamedSharding, PartitionSpec as P
from common.shard_map import shard_map
from common.collectives import device_put_process_local, gather_to_host, rank0_transaction
from common.contract_bands import merge_spin_centroid
from common.grouped_layout import identity_square_grouped_shard_layout
from common.gamma_matrices import gamma_perm_phase, gamma_perm_phase_host
from gw.centroid_k_unfold import CentroidKUnfoldPlan
from distrib_la import gemm_plan
from ffi.fft import make_fused_conv_kparent
from isdf.core import c_q_from_psi_sm, _parent_conv_tables, _parent_conv_vertices
ap=argparse.ArgumentParser();ap.add_argument('--output', required=True);a=ap.parse_args()
def reference(pm,pn,wl,wr,plan,kg,gemm,l,r):
 mu=pm.shape[2];ns=4;nl=mu//2
 pair=make_fused_conv_kparent(R.mesh,kg,ns,(nl,nl),perm_l=np.arange(ns),phase_l=np.ones(ns),perm_r=np.arange(ns),phase_r=np.ones(ns),centroid_major=True)
 @partial(shard_map,mesh=R.mesh,in_specs=(P(None,None,'x',None,'y'),)*2,out_specs=P(None,'x','y'),check_vma=False)
 def tail(dl,dr):
  tables=_parent_conv_tables(plan,plan.centroid_local_perm,plan.L_table,nl,nl)
  return pair(dl,dr,_parent_conv_vertices(tables,gamma_perm_phase(l),gamma_perm_phase(r)))
 @jax.jit
 def run(pm,pn,wl,wr):
  def proj(w):
   A=merge_spin_centroid(pm,1,2)*w[None,None,:]
   B=merge_spin_centroid(jnp.conj(pn),2,3)
   return jnp.transpose(gemm(A,B).reshape(pm.shape[0],mu,4,mu,4),(0,2,1,4,3))
  return tail(proj(wl),proj(wr))
 return run(pm,pn,wl,wr)
def main():
 rows=[];rng=np.random.default_rng(916)
 for kg,anti in (((3,3,3),False),((3,3,3),True),((20,20,20),True)):
  nk=int(np.prod(kg));np_=5;mu=8;nb=6;sym=np.arange(nk,dtype=np.int32)%2
  U=np.array([[np.cos(.31),1j*np.sin(.31)],[1j*np.sin(.31),np.cos(.31)]])
  spin=np.zeros((nk,4,4),complex);spin[:,:2,:2]=U;par=np.where(sym==0,1.,-1.);spin[:,2:,2:]=par[:,None,None]*U
  perm=np.stack([np.arange(mu),np.r_[np.arange(mu//2)[::-1],np.arange(mu//2,mu)[::-1]]])
  plan=CentroidKUnfoldPlan(R.mesh,identity_square_grouped_shard_layout(mu,mu,(2,2)),np.arange(nk,dtype=np.int32)%np_,sym,perm,rng.integers(-1,2,(2,mu,3)).astype(float),rng.uniform(-.4,.4,(np_,3)),spin,1 if anti else 2,4)
  gemm=gemm_plan(R.mesh,m=mu*4,n=mu*4,k=nb,nq=np_,dtype=np.complex128,layout='face',warmup=False)
  z=lambda sh:rng.normal(size=sh)+1j*rng.normal(size=sh)
  pm=device_put_process_local(z((np_,4,mu,nb)),NamedSharding(R.mesh,P(None,None,'x','y')))
  pn=device_put_process_local(z((np_,nb,4,mu)),NamedSharding(R.mesh,P(None,'x',None,'y')))
  wl=device_put_process_local(rng.uniform(-.4,1.,nb),NamedSharding(R.mesh,P()))
  wr=device_put_process_local(rng.uniform(-.4,1.,nb),NamedSharding(R.mesh,P()))
  for l,r in ((0,0),(1,1),(2,2),(3,3),(1,2),(2,1),(0,3),(3,0)):
   kw=dict(k_unfold_plan=plan,kgrid=kg,mesh_xy=R.mesh,gemm=gemm,gamma_L=l,gamma_R=r)
   nv=c_q_from_psi_sm(pm,pn,wl,wr,**kw);ov=reference(pm,pn,wl,wr,plan,kg,gemm,l,r)
   n=np.asarray(gather_to_host(nv));o=np.asarray(gather_to_host(ov));err=float(np.max(abs(n-o))/max(np.max(abs(o)),1e-30))
   assert err<3e-12,(kg,anti,l,r,err)
   rows.append(dict(grid=kg,antiunitary=anti,left=l,right=r,relative_error=err));rank0_print(json.dumps(rows[-1]),flush=True)
 # The typed action is mandatory, not silently approximated by quarters.
 import dataclasses
 broken=spin.copy();broken[:,0,2]=.01
 bad=dataclasses.replace(plan,spin_action_full=broken)
 try: c_q_from_psi_sm(pm,pn,wl,wr,k_unfold_plan=bad,kgrid=kg,mesh_xy=R.mesh,gemm=gemm,gamma_L=2,gamma_R=2)
 except ValueError as e: assert 'GATE dirac_halves' in str(e)
 else: raise AssertionError('off-diagonal Dirac action was accepted')
 rows.append(dict(off_diagonal_action_refused=True))
 rank0_transaction(a.output,stage='native current Gram quarters',write=lambda:Path(a.output).write_text(json.dumps(rows,indent=2)+'\n'))
 return 0
run_main_and_finalize(main)
