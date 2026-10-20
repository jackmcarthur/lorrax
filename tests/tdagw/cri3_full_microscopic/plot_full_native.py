"""Plot authenticated native B/C trajectories and reused static A benchmarks.

No calculation, interpolation, extrapolation, damping fit or eigenvalue clipping.
Missing data refuse by default. --allow-partial admits ONLY checkpoint-written
samples and labels incomplete runs; comparisons use exact common time prefixes.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

FS_AU=1000./24.188843265857
LABELS=dict(A='A: static TD-aGW (different coherent kernel)',
            B='B: full microscopic, frozen W',C='C: full microscopic, refreshed W')
COLORS=dict(A='0.35',B='tab:blue',C='tab:orange')


def _receipts(out,tag):
    candidates=list(out.parent.glob('runs_*.json'))+[out.parent/'runs.json']
    matches=[]
    for path in candidates:
        if not path.exists():
            continue
        data=json.loads(path.read_text())
        for row in data.get('runs',[]):
            if row.get('tag')==tag:
                matches.append((path,data,row))
    return matches


def _native_auth(out):
    """Authenticate every observed native screening receipt, not plot titles."""
    found=[]
    def visit(x):
        if isinstance(x,dict):
            if x.get('construction')=='full-microscopic-general-shared-pole':
                if (x.get('n_mu')!=978 or x.get('n_q')!=36 or x.get('screening_bands')!=160
                        or x.get('spatial_projection') is not False
                        or x.get('screening_density')!='full_matrix'):
                    raise ValueError('Nonmatching microscopic screening receipt in '+str(out))
                found.append(dict(nmu=978,nq=36,screening_bands=160,spatial_projection=False))
            for y in x.values():
                visit(y)
        elif isinstance(x,list):
            for y in x:
                visit(y)
    paths=sorted((out/'campaign_diagnostics').glob('snapshot_*.json'))
    if not paths:
        paths=sorted((out/'accepted_snapshots').glob('snapshot_*.json'))
    for path in paths:
        visit(json.loads(path.read_text()))
    if not found:
        raise ValueError('Missing authenticated978mu/36q/160/fullrho native receipt: '+str(out))
    return dict(native_receipts_checked=len(found),microscopic=found[0])


def read_run(out,tag,mode,tf_fs,allow_partial=False,source_tree=None,active_window=(128,132)):
    out=Path(out)
    lo,hi=map(int,active_window)
    nb=hi-lo
    if not 0<=lo<130<hi<=160:
        raise ValueError('Expected active window containing occupied and empty source bands')
    path=out/'tdgw.h5'
    if not path.exists():
        raise FileNotFoundError('Missing actual native dataset: '+str(path))
    records=_receipts(out,tag)
    if mode!='A':
        auth=_native_auth(out)
        if not records:
            raise ValueError('Missing native campaign run receipt: '+str(out))
        if source_tree and any(data.get('source_tree')!=source_tree for _,data,_ in records):
            raise ValueError('Native source tree disagrees with manifest: '+str(out))
    else:
        auth=dict(scope='reused static TD-aGW benchmark; NOT native collision baseline')
    completed=any(row.get('status')=='COMPLETE' or
        (mode=='A' and row.get('status') is None and row.get('accepted',0)>=round(tf_fs*1000/25))
        for _,_,row in records)
    with h5py.File(path,'r') as f:
        cp=f['tdgw_data/checkpoint']
        n=int(cp['it_next'][()]);end=float(cp['tchkp'][()])/FS_AU
        td=f['tdgw_data/time_domain']
        datasets=('polarization','Efield','carrier_density','rho_full')
        if n<=0 or any(n>len(td[name]['data']) or n>len(td[name]['times']) for name in datasets):
            raise ValueError('Invalid checkpoint-written sample count: '+str(path))
        t=np.asarray(td['polarization/times'][:n])/FS_AU
        if any(not np.array_equal(t,np.asarray(td[name]['times'][:n])/FS_AU) for name in datasets):
            raise ValueError('Native datasets have different exact time axes: '+str(path))
        if not np.isfinite(t).all() or np.any(np.diff(t)<=0) or t[-1]>end+1e-7:
            raise ValueError('Invalid actual native time coverage: '+str(path))
        p=np.asarray(td['polarization/data'][:n]).real.reshape(n,-1)
        field=np.asarray(td['Efield/data'][:n]).real.reshape(n,-1)
        carrier=np.asarray(td['carrier_density/data'][:n]).real.reshape(n,-1)
        rho=np.asarray(td['rho_full/data'][:n])
        weights=np.asarray(f['tdgw_header/wfn_header/w_k'])
        kpts=np.asarray(f['tdgw_header/wfn_header/kpts'])
        bands=f['tdgw_header/wfn_header/bands']
        if tuple(int(bands[k][()]) for k in ('start','stop','step'))!=(lo+1,hi+1,1):
            raise ValueError('Native header does not match manifest active window: '+str(path))
        if rho.shape!=(n,len(weights),nb,nb) or weights.shape!=(36,) or not np.isclose(weights.sum(),1.):
            raise ValueError('Expected complete36k manifest-window density and normalized weights: '+str(path))
    if end>tf_fs+1e-6:
        raise ValueError('Checkpoint extends beyond requested manifest final time: '+str(path))
    complete=bool(completed and abs(end-tf_fs)<1e-6 and len(t)==round(tf_fs/.05)
                  and np.allclose(t,np.arange(len(t))*.05,rtol=0.,atol=1e-7))
    if not complete and not allow_partial:
        raise ValueError('Run not authenticated complete; use --allow-partial explicitly: '+str(out))
    if not all(np.isfinite(x).all() for x in (p,field,carrier,rho,weights,kpts)):
        raise ValueError('Nonfinite actual observables: '+str(path))
    adj=rho.conj().swapaxes(-1,-2)
    eigen=np.linalg.eigvalsh(.5*(rho+adj))
    trace=np.einsum('k,tkii->t',weights,rho).real
    purity=np.einsum('k,tkij,tkji->t',weights,rho,rho).real
    data=dict(time_fs=t,polarization=p,field=field,carrier=carrier,trace=trace,purity=purity,
        eigen_min=eigen.min(axis=(1,2)),eigen_max=eigen.max(axis=(1,2)),kpts=kpts,weights=weights)
    report=dict(tag=tag,mode=mode,label=LABELS[mode],path=str(path),
        status='COMPLETE' if complete else 'PARTIAL_NOT_COMPLETED',samples=n,
        first_sample_fs=float(t[0]),last_actual_sample_fs=float(t[-1]),checkpoint_fs=end,
        requested_tf_fs=tf_fs,completion_receipts=[str(p) for p,_,_ in records],authentication=auth,
        trace_initial=float(trace[0]),trace_final=float(trace[-1]),trace_max_drift=float(abs(trace-trace[0]).max()),
        purity_initial=float(purity[0]),purity_final=float(purity[-1]),
        active_zero_based=[lo,hi],
        purity_definition=f'sum_k w_k Tr(rho_k^2), active{nb}, no clipping',
        rho_min=float(eigen.min()),rho_max=float(eigen.max()),hermiticity_max=float(abs(rho-adj).max()),
        polarization_units='native atomic units',field_units='native atomic units',carrier_units='native e/cell')
    return report,data


def exact_common(items,allow_partial=False,check_field=True):
    """Require exact matching native samples; optional cropping, NEVER interpolation."""
    n=min(len(x['time_fs']) for x in items)
    if not allow_partial and any(len(x['time_fs'])!=n for x in items):
        raise ValueError('Different native coverage; explicit --allow-partial required')
    for item in items[1:]:
        for name in ('time_fs','kpts','weights'):
            left=items[0][name][:n] if name=='time_fs' else items[0][name]
            right=item[name][:n] if name=='time_fs' else item[name]
            if not np.array_equal(left,right):
                raise ValueError('Exact '+name+' mismatch; refusing interpolation/reordering')
        if check_field and not np.array_equal(items[0]['field'][:n],item['field'][:n]):
            raise ValueError('Drive datasets differ at identical native times')
    return n


def gap_mixing(out,tf_limit,active_window=(128,132)):
    """Optional instantaneous Hermitian MATERIAL H, excluding the drive; not an occupation gap."""
    lo,hi=map(int,active_window)
    nb,nv=hi-lo,130-lo
    if not 0<nv<nb:
        raise ValueError('Material-gap diagnostic requires valence and conduction states')
    rows=[]
    for path in sorted((Path(out)/'campaign_diagnostics').glob('snapshot_*.npz')):
        with np.load(path) as z:
            if not all(k in z for k in ('U','E_ry','time_fs')):
                continue
            t=float(z['time_fs'])
            if t>tf_limit+1e-7:
                continue
            u,e=np.asarray(z['U']),np.asarray(z['E_ry'])
            if u.shape!=(1,36,nb,nb) or e.shape!=(1,36,nb):
                raise ValueError('Unsupported snapshot basis shape: '+str(path))
            h=(u[0]*e[0,:,None,:])@u[0].conj().swapaxes(-1,-2)
            energies,vectors=np.linalg.eigh(h)
            # Tr(Pc_reference Pv_instantaneous)/n_valence, subspace gauge-invariant.
            mixing=np.sum(abs(vectors[:,nv:,:nv])**2,axis=(1,2))/nv
            from common.units import RYD_TO_EV
            rows.append(dict(time_fs=t,
                indirect_gap_ev=float(energies[:,nv].min()-energies[:,nv-1].max())*RYD_TO_EV,
                minimum_direct_gap_ev=float(np.min(energies[:,nv]-energies[:,nv-1]))*RYD_TO_EV,
                valence_reference_conduction_leakage_kmean=float(mixing.mean()),
                valence_reference_conduction_leakage_kmax=float(mixing.max()),
                vc_material_H_max_mev=float(abs(h[:,:nv,nv:]).max())*RYD_TO_EV*1e3))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest')
    p.add_argument('--a-root',default='',help='local existing static A7/A10 output root')
    p.add_argument('--a-zero',default='',help='actual A0 output directory, required for --subtract-zero')
    p.add_argument('--fields',default='7e9,1e10')
    p.add_argument('--out',required=True,help='NEW figures/report directory')
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--subtract-zero',action='store_true',help='observable P-P0 ONLY; never modifies dynamics')
    a=p.parse_args()
    manifest=Path(a.manifest).resolve();root=manifest.parent
    plan=json.loads(manifest.read_text());fields=list(map(float,a.fields.split(',')))
    active_window=tuple(plan['microscopic_space']['active_zero_based'])
    if len(fields)!=2 or any(x<=0 for x in fields):
        p.error('Exactly two positive fields are required for nonlinear overlays')
    out=Path(a.out).resolve()
    if out.exists():
        raise FileExistsError('Choose a NEW plot output directory: '+str(out))
    static_root=a.a_root or plan['static_A']['root']
    if not static_root:
        raise ValueError('No matched static A is available for this window; supply actual --a-root')
    aroot=Path(static_root)
    results={};arrays={};run_out={}
    for amp in fields:
        for mode,arm in (('A','off'),('B','joint_frozen'),('C','joint')):
            tag=f'E{amp:g}_{arm}'
            if mode!='A' and not any(r['tag']==tag and r['arm']==arm for r in plan['runs']):
                raise ValueError('Missing required native manifest leg: '+tag)
            directory=(aroot if mode=='A' else root)/tag
            results[amp,mode],arrays[amp,mode]=read_run(directory,tag,mode,plan['tf_fs'],
                a.allow_partial,plan['source_tree'] if mode!='A' else None,active_window)
            run_out[amp,mode]=directory
    zero={}
    if a.subtract_zero:
        if not a.a_zero:
            raise ValueError('--subtract-zero requires explicit actual --a-zero as well as B0/C0')
        for mode,arm in (('A','off'),('B','joint_frozen'),('C','joint')):
            directory=Path(a.a_zero) if mode=='A' else root/f'E0_{arm}'
            rr,zero[mode]=read_run(directory,f'E0_{arm}',mode,plan['tf_fs'],a.allow_partial,
                plan['source_tree'] if mode!='A' else None,active_window)
            if np.max(abs(zero[mode]['field']))!=0:
                raise ValueError('Control is not exactly zero field: '+str(directory))
            results[0.,mode]=rr
    common={}
    for amp in fields:
        data=[arrays[amp,m] for m in ('A','B','C')]
        n=exact_common(data,a.allow_partial)
        if zero:
            n=min(n,exact_common(data+list(zero.values()),a.allow_partial,check_field=False))
        common[amp]=n
    # Cross-field normalized comparisons also require exact time samples, but
    # naturally NOT identical drive amplitudes.
    exact_common([arrays[f,m] for f in fields for m in ('A','B','C')],a.allow_partial,check_field=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    suffix='P-P0 (observable only)' if zero else 'P (raw observable)'
    subtitle=' | PARTIAL: actual checkpoint-written coverage only' if any(
        r['status']!='COMPLETE' for r in results.values()) else ' | 10fs integration; samples normally end9.95fs'
    fig,axes=plt.subplots(2,4,figsize=(17,7),squeeze=False)
    comparison=[]
    observed={}
    for i,amp in enumerate(fields):
        n=common[amp];t=arrays[amp,'B']['time_fs'][:n]
        for mode in ('A','B','C'):
            d=arrays[amp,mode];pol=d['polarization'][:n].copy()
            if zero:
                pol-=zero[mode]['polarization'][:n]
            observed[amp,mode]=pol
            label=LABELS[mode]+(' [PARTIAL]' if results[amp,mode]['status']!='COMPLETE' else '')
            axes[i,0].plot(t,pol[:,0],label=label,color=COLORS[mode])
            axes[i,1].plot(t,np.linalg.norm(pol,axis=1),label=mode,color=COLORS[mode])
        axes[i,2].plot(t,arrays[amp,'B']['field'][:n,0],color='k',label='identical native Ex, A/B/C')
        scale=float(np.max(np.linalg.norm(observed[amp,'B'],axis=1)))
        if scale==0:
            raise ValueError('B polarization peak is zero; normalized difference undefined')
        delta=(observed[amp,'C']-observed[amp,'B'])/scale
        axes[i,3].plot(t,delta[:,0],color='tab:red',label='signed(Cx-Bx)/max|P_B|')
        axes[i,3].axhline(0,color='0.7',lw=.6)
        comparison.append(dict(field_v_m=amp,comparison_samples=n,first_fs=float(t[0]),last_fs=float(t[-1]),
            normalization_B_vector_peak_au=scale,signed_x_difference_peak=float(np.max(abs(delta[:,0]))),
            relative_vector_waveform_l2=float(np.linalg.norm(observed[amp,'C']-observed[amp,'B'])/
                max(np.linalg.norm(observed[amp,'B']),np.finfo(float).tiny))))
        for j,label in enumerate((suffix+' x (au)',suffix+' vector magnitude (au)','Ex (native au)',
                                  'signed deltaPx / B vector peak')):
            axes[i,j].set(xlabel='time (fs)',ylabel=label,title=f'{amp:g} V/m')
            axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=6)
    fig.suptitle('Authenticated full-microscopic B/C; reused static A'+subtitle,fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95))
    health,ha=plt.subplots(2,4,figsize=(17,7),squeeze=False)
    for i,amp in enumerate(fields):
        for mode in ('A','B','C'):
            d=arrays[amp,mode];t=d['time_fs'];color=COLORS[mode]
            ha[i,0].plot(t,d['carrier'][:,0],color=color,label=mode)
            ha[i,1].plot(t,d['purity']-d['purity'][0],color=color,label=mode)
            ha[i,2].plot(t,d['trace']-d['trace'][0],color=color,label=mode)
            ha[i,3].plot(t,d['eigen_min'],color=color,label=mode+' min')
            ha[i,3].plot(t,d['eigen_max']-1,color=color,ls='--',label=mode+' max-1')
        for j,label in enumerate(('native electron carrier (/cell)','delta sum w Tr rho^2',
                                   'delta active electrons (/cell)','rho min and max-1 (no clipping)')):
            ha[i,j].set(xlabel='time (fs)',ylabel=label,title=f'{amp:g} V/m')
            ha[i,j].grid(alpha=.2);ha[i,j].legend(fontsize=7)
    health.suptitle('Raw carrier / purity / trace / actual occupation health'+subtitle,fontsize=11)
    health.tight_layout(rect=(0,0,1,.95))
    nonlinear,na=plt.subplots(2,3,figsize=(14,7),squeeze=False)
    nonlinear_n=min(common.values())
    for j,mode in enumerate(('A','B','C')):
        for amp in fields:
            n=nonlinear_n;d=arrays[amp,mode];t=d['time_fs'][:n];relative=amp/1e10
            na[0,j].plot(t,observed[amp,mode][:n,0]/relative,label=f'{amp:g} V/m')
            na[1,j].plot(t,d['carrier'][:n,0]/relative**2,label=f'{amp:g} V/m')
        na[0,j].set(title=LABELS[mode],ylabel=suffix+'x / (E0/1e10)',xlabel='time(fs)')
        na[1,j].set(ylabel='raw carrier / (E0/1e10)^2',xlabel='time(fs)')
        for ax in na[:,j]:ax.grid(alpha=.2);ax.legend(fontsize=8)
    nonlinear.suptitle('Nonlinear field-normalized overlays; not damping fits'+subtitle,fontsize=11)
    nonlinear.tight_layout(rect=(0,0,1,.95))
    gaps={f'{amp:g}_{mode}':gap_mixing(run_out[amp,mode],results[amp,mode]['last_actual_sample_fs'],active_window)
          for amp in fields for mode in ('B','C')}
    gapfig=None
    if any(gaps.values()):
        gapfig,ga=plt.subplots(1,2,figsize=(11,4))
        for key,rows in gaps.items():
            if not rows:continue
            ga[0].plot([r['time_fs'] for r in rows],[r['indirect_gap_ev'] for r in rows],label=key)
            ga[1].plot([r['time_fs'] for r in rows],
                [r['valence_reference_conduction_leakage_kmean'] for r in rows],label=key)
        ga[0].set(ylabel='instantaneous material indirect gap(eV)',xlabel='time(fs)')
        ga[1].set(ylabel='mean Tr(Pc_ref Pv_inst)/n_valence',xlabel='time(fs)')
        for ax in ga:ax.grid(alpha=.2);ax.legend(fontsize=7)
        gapfig.suptitle('Material H=U diag(E) U†; excludes drive; NOT occupation/optical gap')
        gapfig.tight_layout()
    report=dict(status='PLOTS_OF_ACTUAL_DATA_ONLY',manifest=str(manifest),source_tree=plan['source_tree'],
        allow_partial=a.allow_partial,zero_subtraction='observable P-P0 ONLY' if zero else 'none',
        A_scope=LABELS['A'],B_scope=LABELS['B'],C_scope=LABELS['C'],
        no_interpolation=True,no_damping_fit=True,no_eigenvalue_clipping=True,
        metrics=[dict(field_v_m=amp,**r) for (amp,mode),r in results.items()],
        matched_screening_comparison=comparison,gap_mixing_optional=gaps,
        nonlinear_common_samples=nonlinear_n,
        caveats=['A differs in coherent functional; A/B waveform change alone does not isolate collision damping',
                 'B/C changes include real/offdiagonal Sigma feedback, not only dissipative attenuation',
                 'Numerical and physical convergence are not inferred from a completed trajectory',
                 'Coverage is actual samples, never a synthetic tail to10fs'])
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig.savefig(out/'polarization_field_and_signed_screening_difference.png',dpi=160)
    health.savefig(out/'carrier_purity_trace_occupations.png',dpi=160)
    nonlinear.savefig(out/'nonlinear_field_normalized_overlays.png',dpi=160)
    if gapfig is not None:gapfig.savefig(out/'optional_material_gap_mixing.png',dpi=160)
    print(json.dumps(dict(status=report['status'],output=str(out),comparisons=comparison),indent=2))


if __name__=='__main__':
    main()
