"""Four-rank resident/staged pair parity, including native20³ Ns2/Ns4 loads."""
from runtime import initialize_communicator_stack, run_main_and_finalize, rank0_print
R = initialize_communicator_stack()
import argparse,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding,PartitionSpec as P
from common.shard_map import shard_map
from common.collectives import device_put_process_local,gather_to_host,rank0_transaction
from ffi.fft import (make_fused_conv_kparent,_staged_kparent,_parent_open_spin,
                     _plan_pair_tail,_staged_pair_ffts,conv_kpair_scale,pair_resident_refusal,
                     make_fused_conv_kpair,make_fused_conv_kplane)
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()

def main():
 mesh=R.mesh;receipts=[]
 for kg in [(3,2,2),(20,20,20)]:
  nk=int(np.prod(kg))
  for ns in [2,4]:
   rng=np.random.default_rng(31+ns);mu,nu=(35,67) if nk==12 else (2,3)
   shape=(4,ns,mu*2,ns,nu*2)
   L=rng.normal(size=shape)+1j*rng.normal(size=shape);D=L+.2j
   irr=np.arange(nk,dtype=np.int32)%4;sym=np.arange(nk,dtype=np.int32)%2
   lp=np.stack([np.arange(mu),np.arange(mu)[::-1]]).astype(np.int32)
   rp=np.stack([np.arange(nu),np.roll(np.arange(nu),1)]).astype(np.int32)
   lv=rng.integers(-1,2,(2,mu,3)).astype(float);rv=rng.integers(-1,2,(2,nu,3)).astype(float)
   q=rng.random((4,3));trs=(np.arange(nk)%3==0).astype(np.int32)
   coef=rng.normal(size=(nk,ns,ns,ns,ns))+1j*rng.normal(size=(nk,ns,ns,ns,ns))
   tables=tuple(jnp.asarray(x) for x in [irr,sym,lp,rp,lv,rv,q,trs,coef.reshape(nk,ns*ns,ns*ns),(.7+.3j)*coef.reshape(nk,ns*ns,ns*ns)])
   pl=np.arange(ns)[::-1];pr=np.roll(np.arange(ns),1);hl=np.array([1,1j,-1,-1j][:ns]);hr=np.array([-1j,1,-1,1j][:ns])
   scale=conv_kpair_scale('forward',nk,1)
   staged=_staged_kparent(mesh,kg,ns,pl,hl,pr,hr,scale)
   native=make_fused_conv_kparent(mesh,kg,ns,(mu,nu),perm_l=pl,phase_l=hl,perm_r=pr,phase_r=hr)
   fft,ifft=_staged_pair_ffts(mesh,kg)
   def reference(X,Y):
    A=_parent_open_spin(X,tables,False);B=_parent_open_spin(Y,tables,True)
    # Independent full-spin equation on the small fixture, with the same native
    # Fourier door. This deliberately builds full banks only in this test.
    l=jnp.conj(ifft(A));r=ifft(B);z=jnp.zeros((nk,mu,nu),jnp.complex128)
    for x in range(ns):
     for y in range(ns):z=z+hl[x]*hr[y]*l[:,x,:,:,y]*r[:,pl[x],:,:,pr[y]]
    return fft(z)*scale
   def check(X,Y):
    u=staged(X,Y,tables);v=reference(X,Y);w=native(X,Y,tables)
    e=jnp.linalg.norm(u-v)/jnp.linalg.norm(v);f=jnp.linalg.norm(u-w)/jnp.linalg.norm(w)
    # All resident pair doors must use exactly the same algebra at20³.
    A=_parent_open_spin(X,tables,False);B=_parent_open_spin(Y,tables,True)
    pair=make_fused_conv_kpair(mesh,kg,perm_l=pl,phase_l=hl,perm_r=pr,phase_r=hr)
    pairgot=pair(A.reshape(kg+(ns,mu,nu,ns)),B.reshape(kg+(ns,mu,nu,ns)))
    ep=jnp.linalg.norm(pairgot.reshape(nk,mu,nu)-v)/jnp.linalg.norm(v)
    # A plane's phase and L/R split on load, compared to the explicit pair input.
    plane=make_fused_conv_kplane(mesh,kg,ns,perm_l=pl,phase_l=hl,perm_r=pr,phase_r=hr)
    phase=jnp.exp(.2j*jnp.arange(nk))[:,None,None]*jnp.ones((nk,1,nu))
    raw=jnp.moveaxis(jnp.concatenate([jnp.conj(A),jnp.conj(B)],axis=2),-1,3)
    raw=raw[:,None,:,:,:,:]/phase[:,:,None,None,None,:]
    planeout=plane(raw,phase)
    eg=jnp.linalg.norm(planeout-v)/jnp.linalg.norm(v)
    # A changed transport coefficient must be visible in the real input.
    bad=list(tables);bad[8]=bad[8]*1.01
    neg=jnp.linalg.norm(staged(X,Y,tuple(bad))-u)/jnp.linalg.norm(u)
    return jnp.stack([e,f,ep,eg,neg])
   sm=jax.jit(shard_map(check,mesh=mesh,in_specs=(P(None,None,'x',None,'y'),)*2,out_specs=P('x','y'),check_vma=False))
   # Scalars differ by local tile, so retain a tile-indexed receipt through collectives.
   def localcheck(X,Y):return check(X,Y)[None,None,:]
   sm=jax.jit(shard_map(localcheck,mesh=mesh,in_specs=(P(None,None,'x',None,'y'),)*2,out_specs=P('x','y',None),check_vma=False))
   sh=NamedSharding(mesh,P(None,None,'x',None,'y'))
   X=device_put_process_local(L,sh);Y=device_put_process_local(D,sh)
   t=time.monotonic();got=np.asarray(gather_to_host(sm(X,Y)));assert np.max(got[:,:,:4])<3e-12,got;assert np.min(got[:,:,4])>.005,got
   row=dict(kgrid=kg,ns=ns,local_shape=[mu,nu],max_relative_error=float(got[:,:,:4].max()),negative_relative_min=float(got[:,:,4].min()),wall_s=time.monotonic()-t,resident_refusal=pair_resident_refusal(kg),compiled_memory=str(sm.lower(X,Y).compile().memory_analysis()))
   receipts.append(row);rank0_print(json.dumps(row),flush=True)
 rank0_transaction(a.output,stage='pair capacity parity',write=lambda:Path(a.output).write_text(json.dumps(receipts,indent=2)+'\n'))
 return 0
run_main_and_finalize(main)
