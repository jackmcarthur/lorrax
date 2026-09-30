"""P4 mode11 pair-chunk seams, independent FFT reference, and minimum admission."""
from pathlib import Path
import argparse
import json
import time
from runtime import initialize_communicator_stack
rt=initialize_communicator_stack(platform='gpu')
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import gather_to_host, device_put_process_local
from symmetry_maps import unfold_load_tables
from ffi.fft import make_kconv_chi_unfold, chi_unfold_scratch_bytes
from runtime import aot_memory
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out
out.mkdir(parents=True,exist_ok=True)
mesh=rt.mesh;assert mesh.size==4
kg=(20,20,20);nk=np.prod(kg);np_=5;m=6;n=10;results=[]
for ns in (2,4):
    spec=P(None,'x',None,'y',None);sh=NamedSharding(mesh,spec)
    shape=(np_,m,ns,n,ns)
    def host(seed):
        i=np.arange(np.prod(shape)).reshape(shape)
        return np.sin(i*.013+seed)+1j*np.cos(i*.017-seed)
    @jax.jit(out_shardings=sh)
    def make(seed):
        i=jnp.arange(np.prod(shape),dtype=jnp.float64).reshape(shape)
        return jnp.sin(i*.013+seed)+1j*jnp.cos(i*.017-seed)
    gv,gc,gvt,gct=[make(float(s)) for s in (1,2,3,4)]
    row=(np.arange(nk)%np_).astype(np.int32);trs=((np.arange(nk)//7)%2).astype(np.int32)
    perm_l=np.stack([np.arange(m),np.concatenate([np.arange(m//2)[::-1],
        np.arange(m//2,m)[::-1]])]).astype(np.int32)
    perm_r=np.stack([np.arange(n),np.concatenate([np.roll(np.arange(n//2),1),
        np.roll(np.arange(n//2,n),1)])]).astype(np.int32)
    rng=np.random.default_rng(801+ns)
    wraps_l=rng.integers(-1,2,size=(2,m,3));wraps_r=rng.integers(-1,2,size=(2,n,3))
    kparent=rng.uniform(-.4,.4,size=(np_,3))
    spin=np.broadcast_to(np.eye(ns,dtype=np.complex128),(nk,ns,ns)).copy()
    # A nontrivial unitary spin action at alternate k rows.
    theta=.31
    spin[1::2,0,0]=np.cos(theta);spin[1::2,1,1]=np.cos(theta)
    spin[1::2,0,1]=np.sin(theta);spin[1::2,1,0]=-np.sin(theta)
    tables=unfold_load_tables(irr_idx=row,sym_idx=trs,sym_perm=perm_l,
        L_table=wraps_l,k_irr_frac=kparent,spin_action_full=spin,n_sym_spatial=1,
        mesh_xy=mesh,right_sym_perm=perm_r,right_L_table=wraps_r,trs_rule='pair_transpose')
    alpha_h=np.array([.7+.2j,-.3+.1j]);alpha=device_put_process_local(alpha_h,NamedSharding(mesh,P(None)))
    aspec=P(None,None,'x','y');ash=NamedSharding(mesh,aspec)
    @jax.jit(out_shardings=ash)
    def zero():return jnp.zeros((2,nk,m,n),jnp.complex128)
    pair_bytes=int(nk)*2*ns*ns*16;local_pairs=(m//2)*(n//2)
    assert chi_unfold_scratch_bytes(kg,ns,10**12,optin=166912)==1<<30
    assert chi_unfold_scratch_bytes(kg,ns,1,optin=166912)==pair_bytes
    assert chi_unfold_scratch_bytes((2,2,2),ns,10**12,optin=166912)==0
    wide=make_kconv_chi_unfold(mesh,kg,tables,n_out=2,complete=False,scratch_bytes=pair_bytes*local_pairs)
    tiny=make_kconv_chi_unfold(mesh,kg,tables,n_out=2,complete=False,scratch_bytes=pair_bytes*2+99)
    default=make_kconv_chi_unfold(mesh,kg,tables,n_out=2,complete=False)
    timings={}
    for conj_src in (False,True):
        partners=() if conj_src else (gvt,gct)
        values={}
        for name,fn in [('wide',wide),('two_pair_tail',tiny),('default',default)]:
            fn(zero(),gv,gc,alpha,*partners).block_until_ready()
            started=time.perf_counter()
            val=fn(zero(),gv,gc,alpha,*partners);val.block_until_ready()
            timings[name]=time.perf_counter()-started
            values[name]=gather_to_host(val)
        vp,cp=(np.conj(host(1)),np.conj(host(2))) if conj_src else (host(3),host(4))
        unfolded=[]
        for G,Gp in [(host(1),vp),(host(2),cp)]:
            field=np.where(trs[:,None,None,None,None],Gp[row],G[row])
            field=np.take_along_axis(field,perm_l[trs][:,:,None,None,None],axis=1)
            field=np.take_along_axis(field,perm_r[trs][:,None,None,:,None],axis=3)
            pl=np.exp(2j*np.pi*np.einsum('kc,kxc->kx',kparent[row],wraps_l[trs]))
            pr=np.exp(2j*np.pi*np.einsum('kc,kyc->ky',kparent[row],wraps_r[trs]))
            pl=np.where(trs[:,None],np.conj(pl),pl)
            pr=np.where(trs[:,None],pr,np.conj(pr))
            field=field*pl[:,:,None,None,None]*pr[:,None,None,:,None]
            field=np.einsum('kac,kxcyd,kbd->kxayb',spin,field,np.conj(spin))
            f=np.fft.ifftn(field.reshape(kg+(m,ns,n,ns)),axes=(0,1,2),norm='ortho')
            unfolded.append(f.reshape((nk,m,ns,n,ns)))
        chi=np.einsum('kxayb,kxayb->kxy',np.conj(unfolded[1]),unfolded[0])
        reference=alpha_h[:,None,None,None]*chi[None]
        norm=max(float(np.max(np.abs(reference))),1e-30)
        errors={name:float(np.max(np.abs(v-reference))/norm) for name,v in values.items()}
        seam=max(float(np.max(np.abs(values['wide']-v))/norm) for v in values.values())
        assert max(errors.values())<2e-12,errors
        assert seam<2e-13,seam
        negative=np.max(np.abs(reference-reference*1.01))/norm;assert negative>.009
        results.append(dict(ns=ns,conj_src=conj_src,local_pairs=local_pairs,chunk_pairs=2,
            final_chunk_pairs=1,per_pair_bytes=pair_bytes,default_budget=chi_unfold_scratch_bytes(
                kg,ns,np_*local_pairs*ns*ns*16,optin=166912),relative_errors=errors,
            seam_relative=seam,timing_seconds=timings.copy(),negative_control=float(negative)))
# Pure memory-door probe: actual over-budget minima must refuse even below analytic.
original=aot_memory.compiled_new_bytes
try:
    for got,analytic,room,refuse in [(11,12,10,True),(11,9,10,True),(9,12,10,False),(10,10,10,False)]:
        aot_memory.compiled_new_bytes=lambda *a,**k:got
        try:
            aot_memory.check_chunk(1,build=lambda c:object(),compiled=object(),fixed=analytic,
                                  per_unit=0,room=room,stage='minimum-negative-control')
            assert not refuse
        except MemoryError as exc:
            assert refuse and 'GATE compiled_chunk_capacity' in str(exc)
finally:aot_memory.compiled_new_bytes=original
if jax.process_index()==0:
    (out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
    print('CHI SCRATCH CHUNKS PASS',json.dumps(results),flush=True)
