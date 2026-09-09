"""Measure saved tangent directions' physical M1/M3 trace capture on CPU.

This is an offline diagnostic, not a constructor or a Sigma acceptance gate.
For each orthonormal panel Q, tr(Q.H M Q)/tr(M) is captured moment mass.
The normalized combination M1/tr(M1)+M3/tr(M3) defines the reference ranking.
Independent support captures must never be summed (their subspaces overlap).
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sandbox',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Compute-node HDF5 only'
    import h5py
    import numpy as np
    from spectral_counts import cumulative_count, close_cut
    f=a.sandbox/'runs/frequency_integration_sandbox'
    sources={'Na':f/'308_carrier_incremental_krylov_20260907/299_singular_cutoff/spectra',
             'Si':f/'313_si_control_20260908/46_tau_spectra'}
    a.output.mkdir(parents=True,exist_ok=True)
    job=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
    for dataset,root in sources.items():
        meta=json.loads((root/'result.json').read_text())
        moment_meta=json.loads(Path(meta['moments']).read_text())
        for rec in meta['rows']:
            slot=rec['qslot'];qdir=root/f'q{slot:02d}'
            moment=moment_meta['per_q'][slot]
            assert digest(moment['path'])==moment['sha256']
            assert digest(qdir/'directions.h5')==rec['directions_sha256']
            assert digest(qdir/'spectra.npz')==rec['spectra_sha256']
            with h5py.File(moment['path'],'r') as stream:
                matrices={k:np.asarray(stream[k]) for k in ['M1','M3']}
            # Signed moment audit, performed independently of direction ranking.
            moment_checks={}
            for k,m in matrices.items():
                assert np.linalg.norm(m-m.conj().T) <= 1e-10*np.linalg.norm(m)
                vals=np.linalg.eigvalsh(m)
                assert vals[0] >= -1e-10*vals[-1]
                moment_checks[k]=dict(trace=float(np.trace(m).real),min_relative=float(vals[0]/vals[-1]))
                matrices[k]=m/np.trace(m).real
            rows=[];weights={};max_orth=0
            with h5py.File(qdir/'directions.h5','r') as stream,np.load(qdir/'spectra.npz') as spectra:
                for kind in ['line','imaginary_response']:
                    for support in range(len(stream[kind])):
                        Q=np.asarray(stream[kind][support]);values=spectra[kind][support]
                        orth=float(np.linalg.norm(Q.conj().T@Q-np.eye(Q.shape[1]),ord=2))
                        assert orth<1e-8, ('invalid saved directions',dataset,slot,kind,support,orth)
                        max_orth=max(max_orth,orth)
                        capture=np.array([np.sum(Q.conj()*(m@Q),axis=0).real for m in matrices.values()])
                        assert capture.min() > -1e-10
                        weights[f'{kind}_{support}']=capture
                        criteria=[('relative_1e-3',close_cut(values,int(np.sum(values>.001*values[0])))),
                                  ('relative_1e-2',close_cut(values,int(np.sum(values>.01*values[0])))),
                                  ('fixed_ceil_n4',close_cut(values,(rec['n']+3)//4))]
                        for eps in [.01,.001,.0001]:
                            for power,name in [(1,'nuclear'),(2,'frobenius_squared')]:
                                criteria.append((f'{name}_eps{eps:g}',cumulative_count(values,eps,power)))
                        for name,count in criteria:
                            measured=count<=Q.shape[1]
                            mass=capture[:,:count].sum(axis=1) if measured else [None,None]
                            rows.append(dict(dataset=dataset,parent=slot,kind=kind,support=support,criterion=name,rank=count,
                                M1_capture=mass[0],M3_capture=mass[1],status='MEASURED' if measured else 'REFUSE_SAVED_CAP',job_step=job))
                        # The moment ranking closes W multiplets as groups, retaining gauge invariance.
                        ends=[j for j in range(1,Q.shape[1]+1) if j==len(values) or
                              abs(values[j-1]-values[j])>1e-6*max(abs(values[j-1]),abs(values[j]))]
                        starts=[0]+ends[:-1]
                        blocks=[(lo,hi,capture[:,lo:hi].sum(axis=1)) for lo,hi in zip(starts,ends)]
                        blocks.sort(key=lambda b:float(b[2].sum()/(b[1]-b[0])),reverse=True)
                        for eps in [.01,.001,.0001]:
                            total=np.zeros(2);count=0
                            for lo,hi,weight in blocks:
                                total+=weight;count+=hi-lo
                                if np.all(total>=1-eps):break
                            rows.append(dict(dataset=dataset,parent=slot,kind=kind,support=support,
                                criterion=f'moment_joint_eps{eps:g}',rank=count,M1_capture=total[0],M3_capture=total[1],
                                status='MEASURED' if np.all(total>=1-eps) else 'REFUSE_SAVED_CAP',job_step=job))
            np.savez(a.output/f'{dataset}_q{slot:02d}_weights.npz',**weights)
            out=dict(status='COMPLETE',job_step=job,dataset=dataset,parent=slot,moment_checks=moment_checks,
                     direction_orthogonality=max_orth,rows=rows,source_spectra=str(qdir),
                     spectra_sha256=rec['spectra_sha256'],directions_sha256=rec['directions_sha256'],
                     moment_path=moment['path'],moment_sha256=moment['sha256'],
                     scope='Saved direction trace capture; no new W/model/Sigma; no sum across supports.')
            (a.output/f'{dataset}_q{slot:02d}.json').write_text(json.dumps(out,indent=2)+'\n')
            print(json.dumps(dict(dataset=dataset,parent=slot,status='COMPLETE')),flush=True)
    paths=[a.output/f'{dataset}_q{slot:02d}.json' for dataset,nq in [('Na',29),('Si',8)] for slot in range(nq)]
    assert all(p.exists() for p in paths)
    (a.output/'result.json').write_text(json.dumps(dict(status='COMPLETE',job_step=job,parents=len(paths),artifacts=[str(p) for p in paths]),indent=2)+'\n')


if __name__=='__main__':main()
