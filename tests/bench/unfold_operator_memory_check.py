"""P4 semantic and compile-only large-carrier checks for bounded q unfold."""
from runtime import initialize_communicator_stack,run_main_and_finalize,rank0_print
R=initialize_communicator_stack()
import argparse,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding,PartitionSpec as P
from common.collectives import device_put_process_local,gather_to_host,rank0_transaction
from symmetry_maps.maps import _get_unfold_isdf_operator_jit
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()

def factory(mesh,packed,rule,nfull,npar,nl,nr,cap):
 rng=np.random.default_rng(29);idx=np.arange(nfull,dtype=np.int32)%npar;sym=np.arange(nfull,dtype=np.int32)%2
 if packed:
  fl=np.concatenate([np.arange(nl//2)[::-1]+z for z in (0,nl//2)])
  fr=np.concatenate([np.roll(np.arange(nr//2),1)+z for z in (0,nr//2)])
 else:fl=np.roll(np.arange(nl),3);fr=np.arange(nr)[::-1]
 lp=np.stack([np.arange(nl),fl]).astype(np.int32);rp=np.stack([np.arange(nr),fr]).astype(np.int32)
 ll=lp%(nl//2) if packed else None;rl=rp%(nr//2) if packed else None
 L=rng.integers(-1,2,(2,nl,3)).astype(float);Q=rng.random((npar,3));RR=rng.integers(-1,2,(2,nr,3)).astype(float)
 args=dict(V_q_shape=(npar,nl,nr),fwd_perm_arr=lp,fwd_perm_right_arr=rp,idx_arr=idx,sym_arr=sym,L_arr=L,L_right_arr=RR,q_irr_arr=Q,trs_mask_arr=sym>=1,logical_left=nl-2 if nl<100 else nl,logical_right=nr-3 if nr<100 else nr,n_sym_spatial=1,mesh_xy=mesh,left_local_perm_arr=ll,right_local_perm_arr=rl,trs_rule=rule,q_tile_bytes=cap)
 return _get_unfold_isdf_operator_jit(**args),args

def main():
 mesh=R.mesh;out=[];rng=np.random.default_rng(73)
 for packed in [False,True]:
  for rule in ['conj','pair_transpose']:
   f,kw=factory(mesh,packed,rule,137,5,12,20,2048)
   ref,_=factory(mesh,packed,rule,137,5,12,20,1<<30)
   v=rng.normal(size=(5,12,20))+1j*rng.normal(size=(5,12,20));pair=rng.normal(size=(5,20,12))+1j*rng.normal(size=(5,20,12))
   vv=device_put_process_local(v,NamedSharding(mesh,P(None,'x','y')));pp=device_put_process_local(pair,NamedSharding(mesh,P(None,'y','x')))
   args=(vv,pp) if rule=='pair_transpose' else (vv,)
   got=np.asarray(gather_to_host(f(*args)));direct=np.asarray(gather_to_host(ref(*args)))
   expected=[]
   for i,s in enumerate(kw['sym_arr']):
    j=kw['idx_arr'][i];base=pair[j].T if rule=='pair_transpose' and s else v[j]
    x=base[kw['fwd_perm_arr'][s][:,None],kw['fwd_perm_right_arr'][s][None,:]]
    lm=np.exp(2j*np.pi*(kw['L_arr'][s]@kw['q_irr_arr'][j]));rn=np.exp(2j*np.pi*(kw['L_right_arr'][s]@kw['q_irr_arr'][j]))
    if rule=='pair_transpose' and s:x=lm.conj()[:,None]*x*rn[None,:]
    else:
     x=lm[:,None]*x*rn.conj()[None,:]
     if rule=='conj' and s:x=x.conj()
    x[10:,:]=0;x[:,17:]=0;expected.append(x)
   expected=np.asarray(expected);e=float(np.linalg.norm(got-expected)/np.linalg.norm(expected));d=float(np.linalg.norm(got-direct)/np.linalg.norm(direct));assert e<1e-12 and d<1e-12,(e,d)
   # Wrong physical umklapp wrap must change the invariant result.
   bad=dict(kw);bad['L_arr']=kw['L_arr'].copy();bad['L_arr'][0,0,0]+=1
   wrong=np.asarray(gather_to_host(_get_unfold_isdf_operator_jit(**bad)(*args)));neg=float(np.linalg.norm(wrong-got)/np.linalg.norm(got));assert neg>.001,neg
   row=dict(packed=packed,trs_rule=rule,q_tile_bytes=2048,n_full=137,numpy_relative_error=e,direct_relative_error=d,negative_wrap_relative_error=neg);out.append(row);rank0_print(json.dumps(row),flush=True)
 # Abstract production-equivalent local452²: no large arrays allocated.
 f,kw=factory(mesh,True,'conj',8000,1062,904,904,256<<20)
 x=jax.ShapeDtypeStruct((1062,904,904),jnp.complex128,sharding=NamedSharding(mesh,P(None,'x','y')))
 t=time.monotonic();stats=f.lower(x).compile().memory_analysis();row=dict(kind='abstract20gridP16equivalent',local_mu=452,argument_bytes=stats.argument_size_in_bytes,output_bytes=stats.output_size_in_bytes,temp_bytes=stats.temp_size_in_bytes,alias_bytes=stats.alias_size_in_bytes,compile_s=time.monotonic()-t);assert stats.temp_size_in_bytes<5e9,stats;out.append(row);rank0_print(json.dumps(row),flush=True)
 rank0_transaction(a.output,stage='bounded unfold semantic and memory',write=lambda:Path(a.output).write_text(json.dumps(out,indent=2)+'\n'))
 return 0
run_main_and_finalize(main)
