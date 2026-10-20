#!/usr/bin/env python3
"""Read-only numerical diagnosis of captured SMALL general Loewner pencils.

This rank study NEVER admits a production bank, removes poles, or identifies
physical gain. No microscopic matrix or wavefunction is gathered here. Run
with an explicit time limit for a material-sized capture.

Required NPZ: E[l,r], A[l,r]. Preferred frequency-only physical metric:
GC[r,r]=C.H C, GB[l,l]=B B.H; these have NO microscopic axes. Optional
probe_sites_ry[s], probe_mixed_J[s,r,l]=C.H W_exact B.H,
probe_exact_norm_sq[s] allow exact full-W probe defects without gathering W.
Alternatively, optional SMALL transfer actions:
B[l,input], C[output,r], sites_ry[s], target[s,output,input],
C1[output,input], C2[output,input]. B/C must be supplied together. Without
them, residue and transfer norms are DESCRIPTOR-COORDINATE diagnostics, not
physical W. Optional canonical maps: poles_ry, output_map, input_map.
All additional capture metadata stays in the original file, whose SHA is saved.

IMPORTANT: every map/Gram analysis below concerns the RAW UNSYMMETRIZED
descriptor transfer C(zE-A)^-1B. For self-negative q, the production bank is
the separate half-weight mixture of this transfer and its crossing mirror.
GC/GB alone do not determine interference with that mirror. Final-bank
sample/moment errors in the producer JSON are separate authoritative checks;
these raw-transfer norms must not be labeled final-bank errors or admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import scipy
import scipy.linalg as la


def finite(value):
    if isinstance(value, dict):
        return {str(k):finite(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, (np.integer,np.bool_)):
        return value.item()
    if isinstance(value, (float,np.floating)):
        return float(value) if np.isfinite(value) else str(value)
    if isinstance(value, (complex,np.complexfloating)):
        return dict(real=finite(value.real),imag=finite(value.imag))
    return value


def spectrum_summary(s):
    s=np.asarray(s,float)
    return dict(values=s,maximum=float(s[0]) if s.size else 0.,
                minimum=float(s[-1]) if s.size else 0.,
                condition=float(s[0]/s[-1]) if s.size and s[-1]>0 else float('inf'))


def transfer_summary(model, b, c, sites, *, target=None, c1=None, c2=None):
    p=model['poles_ry'];left=c@model['output_map'];right=b.conj().T@model['input_map']
    weights=np.linalg.norm(left,axis=0)*np.linalg.norm(right,axis=0)
    # Rank-one residue Frobenius norms are invariant under modal phase and
    # reciprocal left/right rescaling; no Hermiticity/PSD assumption enters.
    top=np.argsort(p.imag)[::-1][:min(12,len(p))]
    result=dict(maximum_growth_ry=float(p.imag.max()) if len(p) else None,
        pole_count=len(p),growing_count=int(np.count_nonzero(p.imag>0)),
        positive_real_count=int(np.count_nonzero(p.real>0)),
        negative_real_count=int(np.count_nonzero(p.real<0)),
        residue_norm_sum=float(weights.sum()),
        largest_growth=[dict(index=int(i),pole_ry=p[i],residue_frobenius=float(weights[i]),
            output_norm=float(np.linalg.norm(left[:,i])),input_norm=float(np.linalg.norm(right[:,i]))) for i in top])
    values=[];records=[]
    for j,z in enumerate(sites):
        terms=weights/abs(z-p)
        value=(left/(z-p)[None,:])@right.conj().T
        norm=float(np.linalg.norm(value));values.append(value)
        entry=dict(site_ry=z,transfer_frobenius=norm,
            modal_triangle_bound=float(terms.sum()),
            cancellation_ratio=float(terms.sum()/max(norm,np.finfo(float).tiny)),
            minimum_pole_distance_ry=float(np.min(abs(z-p))) if len(p) else None,
            laplace_margin_ry=float(z.imag-p.imag.max()) if len(p) else None)
        if target is not None:
            error=float(np.linalg.norm(value-target[j]));ref=float(np.linalg.norm(target[j]))
            entry.update(absolute_target_defect=error,relative_target_defect=error/max(ref,np.finfo(float).tiny))
        records.append(entry)
    result['probes']=records
    moment1=left@right.conj().T;moment2=(left*p[None,:])@right.conj().T
    for name,actual,expected in [('C1',moment1,c1),('C2',moment2,c2)]:
        row=dict(frobenius=float(np.linalg.norm(actual)))
        if expected is not None:
            row['absolute_defect']=float(np.linalg.norm(actual-expected))
            row['relative_defect']=row['absolute_defect']/max(float(np.linalg.norm(expected)),np.finfo(float).tiny)
        result[name]=row
    return result,np.asarray(values)


def _sqrt_roundoff(value, bound):
    """Expose squared-norm roundoff; never repair a physical state/model."""
    value=complex(value);tol=256*np.finfo(float).eps*max(float(bound),np.finfo(float).tiny)
    if value.real < -tol or abs(value.imag)>tol:
        raise ValueError(f'Unresolved Gram norm: square={value}, roundoff bound={tol}')
    return float(np.sqrt(max(0.,value.real))),dict(raw_squared_norm=value,roundoff_bound=tol)


def gram_transfer_summary(model, gc, gb, sites, *, probe_sites=None, mixed=None, exact_sq=None):
    """Physical full-port norms from frequency-only metrics, no port gather."""
    p=model['poles_ry'];o=model['output_map'];i=model['input_map']
    gl=o.conj().T@gc@o;gr=i.conj().T@gb@i
    h=gl*gr.T
    residue=[]
    for k in range(len(p)):
        weight,_=_sqrt_roundoff(h[k,k],abs(gl[k,k])*abs(gr[k,k]))
        residue.append(weight)
    residue=np.asarray(residue)
    top=np.argsort(p.imag)[::-1][:min(12,len(p))]
    result=dict(metric='RAW UNSYMMETRIZED full-port transfer norms from frequency-only GC,GB; NOT final self-negative mixture',
        maximum_growth_ry=float(p.imag.max()) if len(p) else None,
        pole_count=len(p),growing_count=int(np.count_nonzero(p.imag>0)),
        positive_real_count=int(np.count_nonzero(p.real>0)),negative_real_count=int(np.count_nonzero(p.real<0)),
        residue_norm_sum=float(residue.sum()),
        largest_growth=[dict(index=int(k),pole_ry=p[k],residue_frobenius=float(residue[k])) for k in top])
    records=[]
    for z in sites:
        coeff=1/(z-p);bound=float(np.sum(abs(coeff)*residue))**2
        norm,roundoff=_sqrt_roundoff(coeff.conj()@h@coeff,bound)
        records.append(dict(site_ry=z,transfer_frobenius=norm,
            modal_triangle_bound=np.sqrt(bound),cancellation_ratio=np.sqrt(bound)/max(norm,np.finfo(float).tiny),
            laplace_margin_ry=float(z.imag-p.imag.max()) if len(p) else None,**roundoff))
    result['probes']=records
    result['exact_probe_defects']=[]
    if mixed is not None:
        for z,j,norm_exact_sq in zip(probe_sites,mixed,exact_sq):
            coeff=1/(z-p);bound=float(np.sum(abs(coeff)*residue))**2
            fitted_sq=coeff.conj()@h@coeff
            inner=coeff.conj()@np.diag(o.conj().T@j@i)
            error_sq=fitted_sq+norm_exact_sq-2*inner.real
            error,roundoff=_sqrt_roundoff(error_sq,bound+abs(norm_exact_sq)+2*abs(inner))
            result['exact_probe_defects'].append(dict(site_ry=z,absolute_target_defect=error,
                relative_target_defect=error/max(np.sqrt(norm_exact_sq),np.finfo(float).tiny),
                exact_frobenius=float(np.sqrt(norm_exact_sq)),**roundoff))
    result['moments']={}
    for name,coeff in [('C1',np.ones(len(p))),('C2',p)]:
        norm,roundoff=_sqrt_roundoff(coeff.conj()@h@coeff,float(np.sum(abs(coeff)*residue))**2)
        result['moments'][name]=dict(frobenius=norm,**roundoff)
    return result


def analyze(path, *, max_order, tolerances=(1e-8,1e-9,1e-10,1e-11),
            max_workspace_bytes=1024**3, probe_sites=None, basis='union', record=None):
    from gw.shared_pole_constructor import reduce_general_pencil, general_pencil_workspace_bytes
    started=time.perf_counter();path=Path(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as store:
        if not {'E','A'}<=set(store.files):
            raise ValueError('Capture must contain E,A; do not silently guess orientations')
        e,a=np.asarray(store['E'],complex),np.asarray(store['A'],complex)
        optional={k:np.asarray(store[k]) for k in
            ('B','C','sites_ry','target','C1','C2','poles_ry','output_map','input_map',
             'GC','GB','probe_sites_ry','probe_mixed_J','probe_exact_norm_sq') if k in store.files}
        keys=list(store.files)
    if e.ndim!=2 or a.shape!=e.shape or min(e.shape)<1 or not np.isfinite(e).all() or not np.isfinite(a).all():
        raise ValueError('Finite, nonempty compatible E,A required')
    nl,nr=e.shape;price=general_pencil_workspace_bytes(e.shape)+sum(v.nbytes for v in optional.values())
    if price>max_workspace_bytes:
        raise ValueError(f'Bounded analysis refuses estimated {price} > {max_workspace_bytes} bytes')
    if max_order<1 or any(not 0<t<1 for t in tolerances):
        raise ValueError('Explicit positive order and numerical tolerances in (0,1) required')
    if basis not in ('union','descriptor'):
        raise ValueError('Diagnostic basis must be union or descriptor')
    if ('B' in optional)!=('C' in optional):
        raise ValueError('B,C must be supplied together; no partial physical-transfer claims')
    actions='B' in optional
    grams='GC' in optional or 'GB' in optional
    if grams and not {'GC','GB'}<=optional.keys():
        raise ValueError('GC and GB must be supplied together')
    if grams and (optional['GC'].shape!=(nr,nr) or optional['GB'].shape!=(nl,nl)):
        raise ValueError('Frequency GC[r,r],GB[l,l] dimensions do not match descriptor')
    probe_keys={'probe_sites_ry','probe_mixed_J','probe_exact_norm_sq'}
    if probe_keys&optional.keys() and not probe_keys<=optional.keys():
        raise ValueError('All three frequency-only exact probe fields must be supplied together')
    if probe_keys<=optional.keys():
        count=len(optional['probe_sites_ry'])
        if not grams or optional['probe_mixed_J'].shape!=(count,nr,nl) or optional['probe_exact_norm_sq'].shape!=(count,):
            raise ValueError('Exact probe metric requires compatible GC/GB, J[s,r,l] and norm_sq[s]')
        if np.any(optional['probe_exact_norm_sq']<0):raise ValueError('Exact probe squared norm must be nonnegative')
    # Without physical metrics only first eight descriptor coordinates are
    # probed. Do not construct huge identity actions or call them physical W.
    b=optional.get('B',np.eye(nl,min(nl,8),dtype=complex));c=optional.get('C',np.eye(min(nr,8),nr,dtype=complex))
    if b.ndim!=2 or c.ndim!=2 or b.shape[0]!=nl or c.shape[1]!=nr:
        raise ValueError('B[l,input] and C[output,r] dimensions do not match captured descriptor')
    if b.size+c.size>2_000_000:
        raise ValueError('Small action-only analysis refuses more than 2,000,000 B/C entries')
    sites=np.asarray(optional.get('sites_ry',probe_sites if probe_sites is not None else
        [-.5+.2j,0.+.2j,.5+.2j]),complex)
    target=optional.get('target')
    if target is not None and (not actions or target.shape!=(len(sites),c.shape[0],b.shape[1])):
        raise ValueError('Target requires explicit B,C and matching [site,output,input] shape')
    scale=max(float(np.linalg.norm(e)),float(np.linalg.norm(a)))
    if scale==0:
        raise ValueError('Zero descriptor has no rank/growth issue to diagnose')
    svd_start=time.perf_counter()
    if basis=='descriptor':
        y,se,xh=la.svd(e,full_matrices=False,check_finite=False)
        sy=sx=se
    else:
        y,sy,_=la.svd(np.concatenate((e,a),axis=1)/scale,full_matrices=False,check_finite=False)
        _,sx,xh=la.svd(np.concatenate((e,a),axis=0)/scale,full_matrices=False,check_finite=False)
        se=la.svdvals(e,check_finite=False)
    reducer_path=Path(reduce_general_pencil.__code__.co_filename).resolve()
    report=dict(scope='DIAGNOSTIC numerical-rank study; not an accepted physical model or evidence of physical gain',
        input=str(path.resolve()),input_sha256=digest,input_keys=keys,shape=e.shape,
        basis=basis,basis_method='two-sided SVD(E)' if basis=='descriptor' else 'two-sided horizontal/vertical union SVD(E,A)',
        analysis_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        production_reducer_file=str(reducer_path),production_reducer_sha256=hashlib.sha256(reducer_path.read_bytes()).hexdigest(),
        max_order=max_order,max_workspace_bytes=max_workspace_bytes,estimated_workspace_bytes=price,
        transfer_scope='RAW UNSYMMETRIZED full-port norms via frequency-only GC/GB' if grams else
            'captured SMALL physical actions' if actions else
            'DESCRIPTOR COORDINATES ONLY (first <=8): no physical W residues or errors available',
        bank_scope='All reconstructed maps describe RAW C(zE-A)^-1B. GC/GB cannot recover crossing-mirror interference. Final self-negative bank sample/moment errors require the separate producer JSON.',
        norms=dict(E_frobenius=np.linalg.norm(e),A_frobenius=np.linalg.norm(a),common_scale=scale),
        descriptor_singular_values=spectrum_summary(se),
        left_union_singular_values=spectrum_summary(sy) if basis=='union' else None,
        right_union_singular_values=spectrum_summary(sx) if basis=='union' else None,
        union_and_descriptor_svd_seconds=time.perf_counter()-svd_start,
        versions=dict(numpy=np.__version__,scipy=scipy.__version__),
        thread_environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        status='RUNNING_DIAGNOSTIC_NOT_ADMITTED',studies=[])
    def checkpoint():
        report['analysis_seconds']=time.perf_counter()-started
        if record is not None:
            record(finite(report))
    checkpoint()
    try:
        from threadpoolctl import threadpool_info
        report['loaded_blas_threadpools']=threadpool_info()
    except ImportError:
        report['loaded_blas_threadpools']='threadpoolctl unavailable; environment is not proof of actual loaded pool size'
    baseline=None
    for tol in tolerances:
        n=min(max_order,int(np.count_nonzero(sy>tol*sy[0])),int(np.count_nonzero(sx>tol*sx[0])))
        row=dict(rank_tolerance=tol,selected_order=n,E_numerical_rank_at_same_relative_tolerance=int(np.count_nonzero(se>tol*se[0])))
        try:
            x=xh[:n].conj().T;yy=y[:,:n]
            er=yy.conj().T@e@x;ar=yy.conj().T@a@x
            row['projected_E_singular_values']=spectrum_summary(la.svdvals(er,check_finite=False))
            homogeneous=la.eigvals(ar,er,homogeneous_eigvals=True,check_finite=False)
            row['homogeneous_beta_magnitudes']=np.abs(homogeneous[1])
            model=reduce_general_pencil(e,a,max_order=max_order,rank_tolerance=tol,
                max_workspace_bytes=max_workspace_bytes,fixed_spaces=dict(right=x,left=yy))
            summary,values=transfer_summary(model,b,c,sites,target=target,
                c1=optional.get('C1'),c2=optional.get('C2'))
            row.update(status='reduced-for-diagnostic-only',receipt=model['receipt'],transfer=summary)
            if grams:
                try:
                    row['physical_transfer']=gram_transfer_summary(model,optional['GC'],optional['GB'],sites,
                        probe_sites=optional.get('probe_sites_ry'),mixed=optional.get('probe_mixed_J'),
                        exact_sq=optional.get('probe_exact_norm_sq'))
                except (ValueError,la.LinAlgError) as exc:
                    row['physical_transfer']=dict(status='unresolved_metric',error=str(exc))
            if baseline is None:baseline=values
            else:row['absolute_transfer_change_from_first_tolerance']=np.linalg.norm(values-baseline,axis=(1,2))
        except (ValueError,la.LinAlgError) as exc:
            row.update(status='refused',error=str(exc))
        report['studies'].append(row)
        checkpoint()
    if {'poles_ry','output_map','input_map'}<=optional.keys():
        model={k:optional[k] for k in ('poles_ry','output_map','input_map')}
        report['captured_canonical_model']=transfer_summary(model,b,c,sites,target=target,
            c1=optional.get('C1'),c2=optional.get('C2'))[0]
        if grams:
            try:
                report['captured_canonical_physical_model']=gram_transfer_summary(model,optional['GC'],optional['GB'],sites,
                    probe_sites=optional.get('probe_sites_ry'),mixed=optional.get('probe_mixed_J'),
                    exact_sq=optional.get('probe_exact_norm_sq'))
            except (ValueError,la.LinAlgError) as exc:
                report['captured_canonical_physical_model']=dict(status='unresolved_metric',error=str(exc))
    report['status']='COMPLETE_DIAGNOSTIC_NOT_ADMITTED'
    checkpoint()
    return finite(report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--max-order',required=True,type=int)
    parser.add_argument('--tolerances',default='1e-8,1e-9,1e-10,1e-11')
    parser.add_argument('--basis',choices=('union','descriptor'),default='union')
    parser.add_argument('--max-workspace-gib',type=float,default=1.)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(dict(status='RUNNING_DIAGNOSTIC_NOT_ADMITTED',phase='before_input_and_svd',
                       input=str(args.input),basis=args.basis),handle,indent=2)
        handle.write('\n')
    def record(report):
        # This invocation created the destination exclusively above. Keep
        # completed rank studies if a later factorization hits the outer cap.
        with tempfile.NamedTemporaryFile('w',dir=args.output.parent,
                prefix=args.output.name+'.',suffix='.pending',delete=False) as handle:
            json.dump(report,handle,indent=2,allow_nan=False)
            handle.write('\n')
            temporary=handle.name
        os.replace(temporary,args.output)
    report=analyze(args.input,max_order=args.max_order,
        tolerances=tuple(map(float,args.tolerances.split(','))),
        max_workspace_bytes=int(args.max_workspace_gib*1024**3),basis=args.basis,record=record)
    print(json.dumps(dict(output=str(args.output),seconds=report['analysis_seconds'],
        studies=[dict(tolerance=r['rank_tolerance'],order=r['selected_order'],status=r['status']) for r in report['studies']]),indent=2))


if __name__=='__main__':main()
