"""All-P physical Hermitian residue ranks and assembled imaginary-axis bounds."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
import numpy as np
import banks
from varpro import basis


MOMENT_SHA = 'f68a710c1831587afa4b9be642bddaf67e0c674b3433fa56e068129aa41fd2d1'
MOMENT_NAMES = ('M0','Mm1','M1')


def main(fits, out, data_banks=None, moment_bank=None):
    assert os.getenv('SLURM_JOB_ID'), 'Compute only'
    from runtime import initialize_communicator_stack
    initialize_communicator_stack()
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import resolve_mesh, barrier
    from distrib_la import plan, gemm_plan
    from file_io.slab_io import SlabIO
    assert jax.process_count()==4
    mesh=resolve_mesh(); face=NamedSharding(mesh,P('x','y')); rep=NamedSharding(mesh,P())
    stack=NamedSharding(mesh,P(None,'x','y'))
    eig=plan('eigh',mesh,backend='distributed',n=896,batched_route='auto')
    ej=jax.jit(lambda a:eig((a+a.conj().T)/2),out_shardings=(rep,face))
    gp=gemm_plan(mesh,m=896,k=896,n=896,nq=1,dtype=jnp.complex128,backend='distributed')
    mm=lambda a,b:gp(a[None],b[None])[0]
    invroot=jax.jit(lambda u,e:mm(u/jnp.sqrt(e)[None,:],jax.lax.with_sharding_constraint(u.conj().T,face)),out_shardings=face)
    white=jax.jit(lambda v,a:mm(mm(v,a),v),out_shardings=face)
    anti_norm=jax.jit(lambda a:jnp.linalg.norm(a-a.conj().T)/jnp.maximum(jnp.linalg.norm(a),1e-300),out_shardings=rep)
    reconstruct=jax.jit(lambda coef,w:jnp.einsum('i,iab->ab',coef[:w.shape[0]],(w+jnp.swapaxes(w.conj(),-1,-2))/2)+jnp.einsum('i,iab->ab',coef[w.shape[0]:2*w.shape[0]],(w-jnp.swapaxes(w.conj(),-1,-2))/(2j)),out_shardings=face)
    add_moments=jax.jit(lambda residue,coef,targets:residue+jnp.einsum('i,iab->ab',coef,targets),
                        out_shardings=face)
    accumulate=jax.jit(lambda a,c,b:a+c*b,out_shardings=face)
    owner=banks.low_owner(mesh); low,broad,z,zh=banks.metadata(data_banks)
    data_specs=low.get('data_banks')
    if moment_bank is not None and banks.sha(moment_bank)!=MOMENT_SHA:
        raise ValueError('SHIFT parent moment bank SHA256 mismatch')
    out.mkdir(parents=True,exist_ok=True)
    job=os.getenv('SLURM_JOB_ID')+'.'+os.getenv('SLURM_STEP_ID','?')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    records=[]
    for q in range(29):
        w,wh,v,paths=banks.load(q,mesh,ej,owner,broad,data_specs)
        if w.shape!=(len(z),896,896) or wh.shape!=(len(zh),896,896):
            raise ValueError('Sample shapes disagree with authenticated bank metadata')
        targets=None
        if moment_bank is not None:
            with SlabIO(moment_bank,mode='r',mesh=mesh) as io:
                moment_rows=[io.read_slab(f'q{q:02d}/parent/{name}',partition_spec=P('x','y'))
                             for name in MOMENT_NAMES]
            if any(a.shape!=(896,896) or a.dtype!=jnp.complex128 for a in moment_rows):
                raise ValueError('SHIFT physical moment shape/dtype mismatch')
            targets=jax.jit(lambda *a:jnp.stack(a),out_shardings=stack)(*moment_rows)
        ev,u=ej(v); vi=invroot(u,ev)
        for p in (8,16,24,32):
            start=time.perf_counter(); src=fits/f'q{q:02d}_p{p:02d}.npz'
            fit_receipt=json.loads(src.with_suffix('.json').read_text())
            if fit_receipt['status']!='FIT_COMPLETE' or banks.sha(src)!=fit_receipt['export_sha256']:
                raise ValueError(f'Fit is incomplete or unauthenticated: {src}')
            with np.load(src,allow_pickle=False) as f:
                poles=f['poles_ry'].copy(); coef=f['row_map'].copy()
                assert np.array_equal(z,f['z_train_ry'])
                kind=str(f['kind'].item()) if 'kind' in f.files else 'unconstrained'
                names=f['moment_names'].tolist() if 'moment_names' in f.files else []
            if kind!=fit_receipt.get('kind','unconstrained'):
                raise ValueError('Fit NPZ/receipt functional kind mismatch')
            extra=3 if kind=='moment_constrained' else 0
            if kind not in ('unconstrained','moment_constrained') or coef.shape!=(p,2*len(z)+extra):
                raise ValueError('Unsupported fit kind or physical row-map shape')
            if extra:
                target_receipt=fit_receipt.get('moment_target') or {}
                if targets is None or names!=list(MOMENT_NAMES) or target_receipt.get('sha256')!=MOMENT_SHA:
                    raise ValueError('Moment fit requires the authenticated matching parent M0/Mm1/M1 bank')
                if target_receipt.get('identity',{}).get('kind')!='parent_dA':
                    raise ValueError('Moment fit targets are not parent dA moments')
            if data_specs:
                upstream=fit_receipt.get('upstream_receipt',{}).get('input_paths',[])
                for item in paths:
                    if item.get('coordinate')=='canonical_physical_Wc':
                        if not any(old.get('sha256')==item['sha256'] and
                                   old.get('receipt_sha256')==item['receipt_sha256'] for old in upstream):
                            raise ValueError('Reconstruction DATA bank differs from fitted DATA bank')
            total=jax.jit(lambda:jnp.zeros((896,896),jnp.complex128),out_shardings=face)()
            eigen=[]; retained={t:[] for t in (1e-2,1e-3,1e-4)}
            # Only p8's tau1e-3 model is exported for an initial failed-model score.
            factors=[]; signs=[]; frequencies=[]
            for k in range(p):
                residue=reconstruct(jnp.asarray(coef[k]),w)
                if extra:
                    # The exported map already carries all Ry bandwidth and
                    # moment scalings. Apply raw physical moments directly;
                    # neither another Coulomb congruence nor rescaling belongs here.
                    residue=add_moments(residue,jnp.asarray(coef[k,-3:]),targets)
                total=accumulate(total,float(-basis(np.array([.25j/banks.EV]),poles)[0,k].real),residue)
                lam,vec=ej(residue); lam=np.asarray(lam); eigen.append(lam)
                scale=np.max(abs(lam))
                for tau in retained:
                    retained[tau].append(int(np.sum(abs(lam)>tau*scale)))
                if p==8:
                    ids=np.flatnonzero(abs(lam)>1e-3*scale)
                    # Pad columns for XY face storage, excluding padding in metadata.
                    count=len(ids); padded=((count+mesh.shape['y']-1)//mesh.shape['y'])*mesh.shape['y']
                    idx=np.pad(ids,(0,padded-count)); amp=np.pad(np.sqrt(abs(lam[ids])),(0,padded-count))
                    factor=jax.jit(lambda a,i,b:a[:,i]*b[None,:],out_shardings=face)(vec,jnp.asarray(idx),jnp.asarray(amp))
                    factors.append(factor); signs.extend(np.r_[np.sign(lam[ids]),np.zeros(padded-count)].tolist());frequencies.extend([poles[k]]*padded)
            eigs=np.asarray(eigen)
            response=white(vi,total); pv,_=ej(response); pv=np.asarray(pv)
            anti=float(anti_norm(response))
            record=dict(q=q,p=p,J=p,r=None,K_by_threshold={str(t):sum(ranks) for t,ranks in retained.items()},
                node_ranks={str(t):ranks for t,ranks in retained.items()},
                compact_signed_storage_bytes={str(t):int(896*sum(ranks)*16+sum(ranks)*24) for t,ranks in retained.items()},
                incumbent_16_n2_bytes=16*896*896*16,
                min_residue_eigenvalue=float(eigs.min()), max_residue_eigenvalue=float(eigs.max()),
                worst_relative_negative=float(max(max(0,-x.min())/max(np.max(abs(x)),1e-300) for x in eigs)),
                damping_fraction_gt_0p1_ev=float(np.mean(-poles.imag*banks.EV>.1)),
                max_center_ev=float(np.max(poles.real)*banks.EV),
                passivity_untruncated=dict(status='PASS' if pv[0]>=-1e-6 and pv[-1]<=1+1e-6 and anti<=1e-10 else 'FAIL',
                    min_eigenvalue=float(pv[0]),max_eigenvalue=float(pv[-1]),antihermitian_relative=anti,nu_ev=.25),
                input_fit=str(src),input_sha256=banks.sha(src),job_step=job,source=source,
                fit_diagnostics_path=str(src.with_suffix('.json')),seconds=time.perf_counter()-start)
            record.update(kind=kind,input_paths=paths,ntrain=len(z),nheld=len(zh),
                          values_used=fit_receipt.get('values_used',len(z)),
                          row_map_units='physical Hermitian residues from unweighted physical H/A and optional raw Ry moments')
            if extra:
                record.update(moment_bank=str(moment_bank),moment_sha256=MOMENT_SHA,
                              moment_names=list(MOMENT_NAMES),moment_target=fit_receipt['moment_target'])
            record['real_snapped_control']='REFUSED_INDEFINITE' if record['worst_relative_negative']>1e-3 else ('REFUSED_WINDOW' if record['max_center_ev']>149.7645217482975 else 'ELIGIBLE_PSD_PROJECTION')
            if p==8:
                factor=jax.jit(lambda *a:jnp.concatenate(a,axis=1),out_shardings=face)(*factors)
                signs=np.asarray(signs); frequencies=np.asarray(frequencies)
                sign_device=jnp.asarray(signs); omega_device=jnp.asarray(frequencies)
                right=jax.jit(lambda a,s:a*s[None,:],out_shardings=face)(factor,sign_device)
                jax.block_until_ready((factor,right,sign_device,omega_device))
                model=out/f'q{q:02d}_p08_tau1e-3.h5'
                with SlabIO(model,mode='w',mesh=mesh) as io:
                    io.create_dataset('left',shape=factor.shape,dtype=np.complex128);io.write_slab('left',factor)
                    io.sync_writes()
                    io.create_dataset('right',shape=factor.shape,dtype=np.complex128);io.write_slab('right',right)
                    io.sync_writes()
                    io.create_dataset('omega_ry',shape=frequencies.shape,dtype=np.complex128)
                    io.write_slab('omega_ry',omega_device)
                    io.sync_writes()
                    io.create_dataset('residue_sign',shape=signs.shape,dtype=np.float64)
                    io.write_slab('residue_sign',sign_device)
                    io.sync_writes()
                    io.write_attr('q_parent',int(low['parent_qrows'][q]));io.write_attr('qslot',q)
                    io.write_attr('scope',np.bytes_('signed Hermitian residue model; tau1e-3 truncation; no passivity repair'))
                    io.write_attr('fit_kind',np.bytes_(kind))
                    if extra:
                        io.write_attr('moment_sha256',np.bytes_(MOMENT_SHA))
                record['export_path']=str(model);record['export_K_with_padding']=len(signs)
                record['export_actual_factor_bytes']=int(2*896*len(signs)*16)
                record['passivity_scope']='untruncated model only; exported truncation needs independent EVAL receipt'
            if jax.process_index()==0:
                with (out/f'q{q:02d}_p{p:02d}_eigen.npz').open('xb') as f:np.savez(f,eigenvalues=eigs,poles_ry=poles)
                with (out/f'q{q:02d}_p{p:02d}.json').open('x') as f:json.dump(record,f,indent=2)
            records.append(record)
            print(f'RESIDUE q{q:02d} p{p} K={record["K_by_threshold"]} passivity={record["passivity_untruncated"]} secs={record["seconds"]:.2f}',flush=True)
    barrier('residue-ranks-complete')
    if jax.process_index()==0:
        (out/'receipt.json').write_text(json.dumps(dict(status='COMPLETE',job_step=job,records=records),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fits',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--data-bank',type=Path,action='append',default=None,
                    help='Repeatable COMPLETE DATA physical bank; same order used to produce the fit Gram')
    ap.add_argument('--moment-bank',type=Path,default=None,
                    help='Authenticated SHIFT parent M0/Mm1/M1 bank, required for moment-constrained fits')
    args=ap.parse_args();main(args.fits,args.out,args.data_bank,args.moment_bank)
