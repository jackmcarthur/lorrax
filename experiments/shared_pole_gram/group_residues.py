"""P4 q0 feasibility of grouped Hermitian residues, using canonical bank owners.

A fixed symmetric mask acts in V-whitened coordinates. Undo that congruence
before ranking physical residues; a mask can increase their matrix ranks.
"""
from pathlib import Path
import argparse,json,os,subprocess,time
import numpy as np
import banks
from varpro import basis


def main(group_dir, fits, out, data_banks, moment_bank):
    from runtime import initialize_communicator_stack
    initialize_communicator_stack()
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import resolve_mesh,barrier
    from distrib_la import plan,gemm_plan
    from file_io.slab_io import SlabIO
    assert jax.process_count()==4
    mesh=resolve_mesh();face=NamedSharding(mesh,P('x','y'));stack=NamedSharding(mesh,P(None,'x','y'));rep=NamedSharding(mesh,P())
    eig=plan('eigh',mesh,backend='distributed',n=896,batched_route='auto')
    ej=jax.jit(lambda a:eig((a+a.conj().T)/2),in_shardings=face,out_shardings=(rep,face))
    product=gemm_plan(mesh,m=896,k=896,n=896,nq=1,dtype=jnp.complex128,backend='distributed')
    mm=lambda a,b:product(a[None],b[None])[0]
    root=jax.jit(lambda u,e:mm(u*jnp.sqrt(e)[None,:],jax.lax.with_sharding_constraint(u.conj().T,face)),out_shardings=face)
    invroot=jax.jit(lambda u,e:mm(u/jnp.sqrt(e)[None,:],jax.lax.with_sharding_constraint(u.conj().T,face)),out_shardings=face)
    @jax.jit
    def rows(v,w):
        return jax.lax.scan(lambda _,a:(None,mm(mm(v,a),v)),None,w)[1]
    adj=lambda a:jnp.swapaxes(a.conj(),-1,-2)
    channel=jax.jit(lambda a:jnp.concatenate(((a+adj(a))/2,(a-adj(a))/(2j))),out_shardings=stack)
    reconstruct=jax.jit(lambda c,y,m:jnp.einsum('i,iab->ab',c,y)*m,out_shardings=face)
    physical=jax.jit(lambda v,a:mm(mm(v,a),v),out_shardings=face)
    add=jax.jit(lambda total,c,a:total+c*a,out_shardings=face)
    add_rows=jax.jit(lambda total,c,a:total+c[:,None,None]*a[None],out_shardings=stack)
    norm=jax.jit(lambda a:jnp.sqrt(jnp.sum(jnp.abs(a)**2)),out_shardings=rep)
    anti=jax.jit(lambda a:jnp.linalg.norm(a-a.conj().T)/jnp.maximum(jnp.linalg.norm(a),1e-300),in_shardings=face,out_shardings=rep)
    out.mkdir(parents=True,exist_ok=True)
    grouping=json.loads((group_dir/'grouping.json').read_text())
    assert banks.sha(grouping['mask_path'])==grouping['mask_sha256']
    assert banks.sha(moment_bank)=='f68a710c1831587afa4b9be642bddaf67e0c674b3433fa56e068129aa41fd2d1'
    owner=banks.low_owner(mesh);low,broad,z,zh=banks.metadata(data_banks)
    w,wh,v,paths=banks.load(0,mesh,ej,owner,broad,low['data_banks'])
    eigen,u=ej(v);assert float(eigen[0])>0
    vh,vi=root(u,eigen),invroot(u,eigen)
    white=rows(vi,w)
    with SlabIO(moment_bank,mode='r',mesh=mesh) as io:
        moment=[io.read_slab(f'q00/parent/{name}',partition_spec=P('x','y')) for name in ('M0','Mm1','M1')]
    targets=jax.jit(lambda *a:jnp.stack(a),out_shardings=stack)(*moment)
    y=jax.jit(lambda a,b:jnp.concatenate((a,b)),out_shardings=stack)(channel(white),rows(vi,targets))
    with SlabIO(grouping['mask_path'],mode='r',mesh=mesh) as io:labels=io.read_slab('labels',partition_spec=P('x','y'))
    job=os.getenv('SLURM_JOB_ID')+'.'+os.getenv('SLURM_STEP_ID');records=[]
    for p in (8,16,24,32):
        started=time.perf_counter();fit_records=[json.loads((fits/f'group{g}/p{p:02d}.json').read_text()) for g in range(2)]
        receipt=dict(q=0,p_per_group=p,job_step=job,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                     scope='q0 only; grouped physical residue ranks and four physical low held matrices',grouping=grouping,input_paths=paths)
        if any(d['status']!='FIT_COMPLETE' for d in fit_records):
            receipt.update(status='REFUSED_FIT',reasons=[d.get('error') for d in fit_records]);records.append(receipt)
            if jax.process_index()==0:
                with (out/f'p{p:02d}.json').open('x') as stream:json.dump(receipt,stream,indent=2)
            continue
        total=jax.jit(lambda:jnp.zeros((896,896),jnp.complex128),out_shardings=face)()
        prediction=jax.jit(lambda:jnp.zeros((4,896,896),jnp.complex128),out_shardings=stack)()
        spectra=[];poles_all=[];anti_max=0.
        for g in range(2):
            model_path=fits/f'group{g}/p{p:02d}.npz'
            if banks.sha(model_path)!=fit_records[g]['export_sha256']:
                raise ValueError('Grouped model export hash mismatch')
            with np.load(model_path) as model:coef=model['row_map'];poles=model['poles_ry']
            assert coef.shape in ((p,2*len(z)),(p,2*len(z)+3))
            mask=jax.jit(lambda a:a==g,out_shardings=face)(labels)
            for k,pole in enumerate(poles):
                wr=reconstruct(jnp.asarray(coef[k]),y[:coef.shape[1]],mask)
                residue=physical(vh,wr)
                anti_max=max(anti_max,float(anti(residue)))
                ev,_=ej(residue);spectra.append(np.asarray(ev));poles_all.append(pole)
                total=add(total,float(-basis(np.array([.25j/banks.EV]),np.array([pole]))[0,0].real),wr)
                prediction=add_rows(prediction,jnp.asarray(basis(zh[:4],np.array([pole]))[:,0]),wr)
        ev,_=ej(total);ev=np.asarray(ev);response_anti=float(anti(total));pred=rows(vh,prediction)
        held_error=float(norm(pred-wh[:4])/norm(wh[:4]));spectra=np.asarray(spectra);poles_all=np.asarray(poles_all)
        ranks={str(t):int(sum(np.sum(abs(a)>t*np.max(abs(a))) for a in spectra)) for t in [1e-2,1e-3,1e-4]}
        receipt.update(status='MEASURED',J=len(np.unique(poles_all)),K_by_threshold=ranks,
            compact_signed_bytes={t:16*896*k+24*k for t,k in ranks.items()},
            direct_group_entry_bytes=int(16*p*sum(grouping['Ng_full_matrix_entries'])),incumbent_16_n2_bytes=16*16*896**2,
            held_025_physical_relative=held_error,physical_residue_antihermitian_max=anti_max,
            passivity_untruncated=dict(min_eigenvalue=float(ev[0]),max_eigenvalue=float(ev[-1]),antihermitian_relative=response_anti,nu_ev=.25,status='PASS' if ev[0]>=-1e-6 and ev[-1]<=1+1e-6 and response_anti<=1e-10 else 'FAIL'),
            fits_step_converged=all(d['diagnostics']['success'] for d in fit_records),
            damping_fraction_gt_0p1_ev=float(np.mean(-poles_all.imag*banks.EV>.1)),max_center_ev=float(np.max(poles_all.real)*banks.EV),
            cond_phi_by_group=[d['diagnostics']['cond_phi'] for d in fit_records],seconds=time.perf_counter()-started)
        records.append(receipt)
        if jax.process_index()==0:
            with (out/f'p{p:02d}.json').open('x') as stream:json.dump(receipt,stream,indent=2)
            with (out/f'p{p:02d}_eigen.npz').open('xb') as stream:np.savez(stream,eigenvalues=spectra,poles_ry=poles_all)
        print('GROUP_RESIDUE',p,ranks,held_error,flush=True)
    barrier('group-residues-complete')
    if jax.process_index()==0:
        with (out/'receipt.json').open('x') as stream:json.dump(dict(status='COMPLETE',job_step=job,records=records),stream,indent=2)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--groups',type=Path,required=True);ap.add_argument('--fits',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--data-bank',type=Path,action='append',required=True);ap.add_argument('--moment-bank',type=Path,required=True)
    a=ap.parse_args();main(a.groups,a.fits,a.out,a.data_bank,a.moment_bank)
