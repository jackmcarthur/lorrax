"""Replay the saved t0 active BODY before any GW map or W construction.

Four- or sixteen-rank diagnostic only. Native initialization and the authenticated accepted
SC INPUT supply the full750 orbitals; the committed general W is read verbatim.
Only active44 intermediate columns run. No head, virtual MH, tail, fixed point,
collision-law change, or propagation is performed.
Optional spectral mode samples full active retarded matrices at selected
bands' actual energies at one diagnostic k; it performs no on-shell sweep.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time


def density_digest(occupations, rho, *, b_lo, screening_stop):
    """Hash the producer's full complex rho with one small band block live."""
    import numpy as np
    occupations=np.asarray(occupations)
    digest=hashlib.sha256()
    for f,r in zip(occupations,rho):
        block=np.zeros((len(f),len(f)),np.complex128)
        np.fill_diagonal(block,f)
        block[b_lo:b_lo+len(r),b_lo:b_lo+len(r)]=r
        block[screening_stop:,:]=0
        block[:,screening_stop:]=0
        digest.update(memoryview(block))
    return digest.hexdigest()


def load_projector_snapshot(path):
    """This diagnostic is valid only for the exact undriven t0 projector."""
    import numpy as np
    with np.load(path,allow_pickle=False) as saved:
        arrays={key:np.asarray(saved[key]) for key in
                ('U','E_ry','rho_in_basis','Gamma_ry','D_ry','C_raw_ry','time_fs')}
    u,r=arrays['U'],arrays['rho_in_basis']
    if (u.shape!=(64,44,44) or arrays['E_ry'].shape!=(64,44) or r.shape!=u.shape
            or float(arrays['time_fs'])!=0. or not all(np.isfinite(a).all() for a in arrays.values())
            or not np.array_equal(u,np.broadcast_to(np.eye(44),u.shape))
            or not np.array_equal(r,np.broadcast_to(np.diag([1.]*30+[0.]*14),r.shape))):
        raise ValueError('GATE same_W_rate_replay: exact t0 identity U and binary diagonal 30/14 projector required')
    return arrays


def collision_statistics(gamma,injection,rho):
    """Unchanged branch-operator collision and uniform full64 trace statistic."""
    import numpy as np
    from tdagw.collision import collision,boundary_diagnostics
    c=collision(gamma,injection,rho)
    outgoing=-float(np.trace(gamma@rho,axis1=-2,axis2=-1).real.mean())
    incoming=float(np.trace(injection,axis1=-2,axis2=-1).real.mean())
    flux=.5*(abs(incoming)+abs(outgoing))
    boundary=boundary_diagnostics(gamma,injection,rho,occupation_tolerance=1e-8)
    return c,dict(trace_rate_ry=incoming+outgoing,body_incoming_ry=incoming,
        body_outgoing_ry=outgoing,body_flux_ry=flux,
        body_trace_relative=abs(incoming+outgoing)/flux if flux else 0.,
        empty_min_injection_ry=float(np.min(boundary['empty_min_injection'])),
        filled_min_blocking_ry=float(np.min(boundary['filled_min_blocking'])))


def snapshot_energy(prepared,snapshot):
    """The native evaluator's exact instantaneous frozen endpoint shifts."""
    energy=prepared['energy_padded'].copy();e=snapshot['E_ry']
    energy[:,:100]+=(e[:,0]-energy[:,100])[:,None]
    energy[:,144:750]+=(e[:,-1]-energy[:,143])[:,None]
    energy[:,100:144]=e
    return energy


def evaluate_rate_body(source,prepared,snapshot,model,*,quadrature_eps,omega_ry=None,print_fn=print):
    """Run one active-only native sweep, on shell or at explicit relative Ry."""
    import numpy as np
    from gw.ppm_sigma import shared_pole_window_source,window_sigma_plan,window_sigma_c
    from tdagw.collision import branch_operators
    energy=snapshot_energy(prepared,snapshot)
    e=snapshot['E_ry'];rho=snapshot['rho_in_basis'];u=snapshot['U']
    origin=prepared['frequency_origin_ry']
    handle=shared_pole_window_source(model,source['meta'],efermi_ry=origin,linalg=source['linalg'])
    plan=window_sigma_plan(source['config'],head=None,window_only=True)
    plan['sigma_cfg']=replace(plan['sigma_cfg'],quadrature_eps=float(quadrature_eps))
    window=dict(b_lo=100,U=u,e=e,rho=rho,f=rho.diagonal(axis1=-2,axis2=-1).real,e_frozen=energy)
    raw=window_sigma_c(source['wfns'],handle,window,plan,meta=source['meta'],
        mesh_xy=prepared['mesh'],branches=True,pairing='vertex',internal='active',
        omega_ry=omega_ry,print_fn=print_fn)
    if omega_ry is not None:
        return dict(raw=raw)
    pair=raw['active_branch_pair']
    gamma,injection,_=branch_operators(pair['cond'],pair['val'])
    collision,statistics=collision_statistics(gamma,injection,rho)
    return dict(gamma=gamma,injection=injection,collision=collision,statistics=statistics,raw=raw)


def spectral_probe_matrices(raw,snapshot,*,bands,k,origin):
    """Select one k of R[frequency,k,44,44] and diagnose K=i(R-R†), Ry.

    Every frequency is the selected band's actual absolute energy minus
    the common numerical origin. No matrix is clipped or spectrally repaired.
    """
    import numpy as np
    energy=np.asarray(snapshot['E_ry'])
    bands=np.asarray(bands,np.int64)
    omega=energy[k,bands]-float(origin)
    if not np.array_equal(np.asarray(raw['omega_ry']),omega):
        raise ValueError('GATE same_W_spectral_probe: actual selected shell frequencies changed')
    payload=dict(omega_relative_ry=omega,omega_absolute_ry=energy[k,bands],
        spectral_bands=bands,spectral_k=np.asarray(k,np.int64))
    report=dict(spectral_k=int(k),bands_active_relative=bands.tolist(),
        omega_relative_ry=omega.tolist(),omega_absolute_ry=energy[k,bands].tolist(),branches={})
    shape=(len(bands),len(energy),energy.shape[1],energy.shape[1])
    for space in ('cond','val'):
        values=np.asarray(raw['retarded_omega'][space])
        if values.shape!=shape or not np.isfinite(values).all():
            raise ValueError('GATE same_W_spectral_probe: finite full active branch matrices required')
        r=values[:,k].copy()
        width=1j*(r-r.conj().swapaxes(-1,-2))
        norm=np.linalg.norm(width,axis=(-2,-1))
        skew=np.linalg.norm(width-width.conj().swapaxes(-1,-2),axis=(-2,-1))
        eig=np.linalg.eigvalsh(width)
        payload['retarded_'+space+'_ry']=r
        payload['K_'+space+'_ry']=width
        payload['K_'+space+'_eigenvalues_ry']=eig
        report['branches'][space]=dict(minimum_eigenvalue_ry=eig[:,0].tolist(),
            maximum_eigenvalue_ry=eig[:,-1].tolist(),
            negative_eigenvalue_count=np.sum(eig<0.,axis=-1).tolist(),
            hermiticity_frobenius_ry=skew.tolist(),
            hermiticity_relative_frobenius=np.divide(skew,norm,out=np.zeros_like(skew),where=norm!=0).tolist())
    return payload,report


def parse_args(argv=None):
    """Validate bounded diagnostic modes before importing the native runtime."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup-deck',required=True)
    parser.add_argument('--accepted-deck',required=True)
    parser.add_argument('--frames',required=True)
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--snapshot-json')
    parser.add_argument('--model',required=True)
    parser.add_argument('--model-digest',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--quadrature-eps',type=float,nargs='+')
    parser.add_argument('--spectral-bands',type=int,nargs='+',help='Unique active-relative band indices (0..43); spectral-only sweep')
    parser.add_argument('--spectral-k',type=int,default=0,help='Full-grid diagnostic k row (0..63)')
    args=parser.parse_args(argv)
    spectral=args.spectral_bands is not None
    if args.quadrature_eps is None:
        args.quadrature_eps=[1e-8] if spectral else [1e-4,1e-6]
    if (Path(args.out).exists() or any(not 0<x<1 for x in args.quadrature_eps)
            or len(args.quadrature_eps)>3):
        parser.error('fresh output and one to three finite relative quadrature targets in (0,1) required')
    if spectral and (len(args.quadrature_eps)!=1 or not 0<=args.spectral_k<64
            or any(not 0<=b<44 for b in args.spectral_bands)
            or len(set(args.spectral_bands))!=len(args.spectral_bands)):
        parser.error('spectral probe needs one eps, unique active-relative bands 0..43 and k 0..63')
    if not spectral and (args.snapshot_json is None or args.spectral_k!=0):
        parser.error('rate replay requires --snapshot-json; --spectral-k requires --spectral-bands')
    return args


def main(argv=None):
    args=parse_args(argv)
    # Bootstrap the native process before frame I/O imports NumPy/BLAS.
    from gw import gw_jax,sc_iteration
    from tests.tdagw.cri3_qgrid_750_campaign import native_frozen_map_probe as frozen
    arrays,frame=frozen.load_accepted_frame(args.frames)
    policy=frozen.accepted_response_policy(frame,args.accepted_deck)
    snapshot=load_projector_snapshot(args.snapshot)
    baseline=None if args.spectral_bands is not None else json.loads(Path(args.snapshot_json).read_text())
    from common.collectives import rank0_transaction,device_put_process_local,gather_to_host
    from common.parallel_transport import fingerprint_from_binding
    from common.contract_bands import merge_spin_centroid
    from file_io.shared_pole_store import _read_header_once,validate_general_shared_pole_model
    from gw.ppm_sigma import fresh_shared_pole_capacity
    from gw.response_bank import operator_digest
    from tdagw.snapshot_runtime import prepare_native_source
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding,PartitionSpec as P
    out=Path(args.out).resolve();finished=[];started=time.monotonic()
    original=sc_iteration.run_self_consistency
    def intercept(initial,inputs,**kwargs):
        if jax.process_count() not in (4,16):
            raise ValueError('GATE same_W_rate_replay: four or sixteen ranks required')
        say=inputs.print_fn
        frozen.validate_callback_reference(frame,inputs.meta.qgrid,policy,inputs.config.qgrid_response_window)
        if (fingerprint_from_binding(inputs.wfn_fingerprint_binding,inputs.wfn)!=frame['wfn_fingerprint']
                or not np.array_equal(inputs.meta.mu_basis.canonical_indices,arrays['centroid_canonical_indices'])
                or not np.allclose(inputs.sym.unfolded_kpts,arrays['kpoints_full_frac'],rtol=0,atol=1e-12)
                or float(inputs.config.sigma.regularization_ev)!=.25):
            raise ValueError('GATE same_W_rate_replay: source WFN/centroids/k rows/bulk eta mismatch')
        mesh=inputs.mesh_xy;face=NamedSharding(mesh,P(None,'x','y'));rep=NamedSharding(mesh,P())
        put=lambda a,sh=rep:device_put_process_local(np.asarray(a),sh)
        current=sc_iteration.rotate_wavefunctions(inputs.wfns_dft,put(arrays['U_active'],face),
            enk_active_new=put(arrays['energy_padded_ry'][:,:160]),enk_base=put(arrays['energy_padded_ry']),
            occupations=put(arrays['occupation_padded']),efermi=None,active_slice=slice(0,160),mesh_xy=mesh)
        if not np.array_equal(gather_to_host(current.enk),arrays['energy_padded_ry']):
            raise ValueError('GATE same_W_rate_replay: current source changed accepted full750 energies')
        meta=fresh_shared_pole_capacity(copy.copy(inputs.meta),mesh)
        source=dict(wfns=current,meta=meta,config=inputs.config,V_q=inputs.V_q,frame=frame,
            linalg=inputs.config.backend.linalg,efermi_ry=0.)
        prepared=prepare_native_source(source,mesh)
        if prepared['nb']!=750 or prepared['nmu_logical']!=3000 or prepared['qgrid']!=(4,4,1):
            raise ValueError('GATE same_W_rate_replay: exact full750/3000mu/own q4 source required')
        energy=snapshot_energy(prepared,snapshot)
        header=_read_header_once(args.model,stage='same_W_rate_replay/header')
        stop=min(int(meta.b_id_4_chi_user),int(current.slices.b4_logical))
        pm=current.green_parent.psi_mun
        pm=jax.jit(lambda a:merge_spin_centroid(a*(jnp.arange(a.shape[-1])<stop)[None,None,None],1,2),
            out_shardings=face)(pm)
        identity=dict(iteration_id=header['identity']['iteration_id'],
            energies=hashlib.sha256(energy.tobytes()).hexdigest(),
            occupations=density_digest(prepared['occupations_padded'],snapshot['rho_in_basis'],b_lo=100,screening_stop=stop),
            hamiltonian=hashlib.sha256(energy.tobytes()+snapshot['U'].tobytes()).hexdigest(),
            wavefunctions=operator_digest(pm,mesh,cache=False),
            centroids=hashlib.sha256(np.asarray(meta.mu_basis.canonical_indices).tobytes()).hexdigest(),
            electronic_grid=[8,8,1],response_grid=[4,4,1],qgrid_response_window='fejer')
        del pm
        header=validate_general_shared_pole_model(args.model,expected_identity=identity,mesh_xy=mesh,
            capacity=meta.shared_pole_capacity)
        if header['digest']!=args.model_digest:raise ValueError('GATE same_W_rate_replay: committed W digest mismatch')
        meta.shared_pole_recipe=header['recipe']
        model=dict(path=args.model,identity=identity,digest=header['digest'],K=header['K'],representation='ordered_complex')
        rank0_transaction(out,stage='same_W_rate_replay/output',write=lambda:out.mkdir(parents=True,exist_ok=False))
        rows=[];previous=None;baseline_stats=None
        spectral=args.spectral_bands is not None
        if not spectral:
            baseline_c,baseline_stats=collision_statistics(snapshot['Gamma_ry'],snapshot['D_ry'],snapshot['rho_in_basis'])
            if not np.allclose(baseline_c,snapshot['C_raw_ry'],rtol=0,atol=2e-15):
                raise ValueError('GATE same_W_rate_replay: saved collision does not reconstruct from saved Gamma/D')
        for eps in args.quadrature_eps:
            start=time.monotonic()
            omega=(snapshot['E_ry'][args.spectral_k,args.spectral_bands]-prepared['frequency_origin_ry']
                   if spectral else None)
            result=evaluate_rate_body(source,prepared,snapshot,model,quadrature_eps=eps,
                omega_ry=omega,print_fn=say)
            if spectral:
                payload,row=spectral_probe_matrices(result['raw'],snapshot,bands=args.spectral_bands,
                    k=args.spectral_k,origin=prepared['frequency_origin_ry'])
            else:
                delta=result['collision']-baseline_c
                row=dict(**result['statistics'],
                    collision_difference_from_saved_max_ry=float(np.max(abs(delta))),
                    collision_difference_from_saved_per_k_frobenius_max_ry=float(np.max(np.linalg.norm(delta,axis=(-2,-1)))),
                    trace_difference_from_saved_ry=result['statistics']['trace_rate_ry']-baseline_stats['trace_rate_ry'])
                # Producer guard denominator: Frobenius norms, not scattering flux.
                saved_norm_scale=abs(float(baseline['fresh_trace_rate_ry']))/float(baseline['fresh_trace_relative'])
                row['residual_over_fixed_saved_full_norm_scale']=abs(row['trace_rate_ry'])/saved_norm_scale
                if previous is not None:
                    row['collision_change_from_previous_max_ry']=float(np.max(abs(result['collision']-previous)))
                previous=result['collision']
                payload=dict(gamma_body_ry=result['gamma'],injection_body_ry=result['injection'],C_body_ry=result['collision'])
                for branch,value in result['raw']['retarded_pair'].items():payload['retarded_'+branch]=value
            row.update(quadrature_eps=eps,seconds=time.monotonic()-start,
                native_receipt=result['raw']['native_receipt'])
            tag=f'eps_{eps:.0e}'
            rank0_transaction(out/(tag+'.npz'),stage='same_W_rate_replay/operators',
                write=lambda:np.savez(out/(tag+'.npz'),**payload))
            rows.append(row)
        report=dict(status=('SAME_W_ACTIVE_SPECTRAL_BODY_PROBE_COMPLETE_NOT_DYNAMICS' if spectral else
                            'SAME_W_ACTIVE_RATE_BODY_REPLAY_COMPLETE_NOT_DYNAMICS'),
            source_frame=frame['frame'],snapshot_sha256=hashlib.sha256(Path(args.snapshot).read_bytes()).hexdigest(),
            W_digest=header['digest'],frequency_origin_ry=prepared['frequency_origin_ry'],
            active_intermediates=[100,144],full_source_bands=750,centroids=3000,
            head_included=False,virtual_MH_included=False,tail_included=False,screening_rebuilt=False,
            baseline_full_statistics=baseline_stats,rows=rows,seconds=time.monotonic()-started,
            scope=('Exact saved t0 projector and W; full44 active-body matrices at selected actual energies '
                   'at one k. Both branches retarded, K=i(R-Rdagger) without clipping, roots or closure. '
                   'Headless, active-intermediate only; no all-energy positivity, full-Sigma/MH, '
                   'collision guard admission, propagation or physical convergence claim.' if spectral else
                  'Exact t0 projector only: compare collision/empty injection/filled blocking. '
                  'Headless Gamma/D/Sigma are not full physical quantities; body_flux_ry is a gain/loss trace scale, '
                  'not the physical empty-gain/filled-loss scattering flux. '
                  'No full-Sigma/MH, collision guard admission, propagation or physical convergence claim.'))
        rank0_transaction(out/'receipt.json',stage='same_W_rate_replay/receipt',
            write=lambda:(out/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'))
        finished.append(True)
        raise StopIteration('SAME_W_RATE_REPLAY_COMPLETE')
    sc_iteration.run_self_consistency=intercept
    try:
        try:gw_jax.main(['-i',str(Path(args.setup_deck).resolve())])
        except StopIteration as exc:
            if str(exc)!='SAME_W_RATE_REPLAY_COMPLETE' or not finished:raise
        if not finished:raise RuntimeError('native source initialization did not reach replay boundary')
    finally:sc_iteration.run_self_consistency=original


if __name__=='__main__':main()
