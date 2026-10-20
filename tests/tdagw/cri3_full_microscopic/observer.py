"""Campaign-only native small-table capture and fail-closed exploratory guards."""
import json
import time
import traceback
from pathlib import Path

import jax
import numpy as np


def install(outdir, *, start, budget_s, tf_fs, refresh_fs, storage_gib, limits):
    from common.collectives import gather_to_host
    from file_io.commit_state import agree_io_refusal
    from tdagw import stages, snapshot_runtime as sr
    original_factory, original_snapshot, original_write = (
        sr.make_native_snapshot_evaluator, sr.snapshot_collision, stages._write_chunk)
    root = Path(outdir)/'campaign_diagnostics'
    rank0 = jax.process_index() == 0
    evaluated, health, reference = [], [], {}
    fs_au = 1000./24.188843265857
    from tdagw.tdgw_io import HA_IN_EV
    from common.units import RYD_TO_EV
    ry_ha = RYD_TO_EV/HA_IN_EV

    def host(x):
        return np.asarray(gather_to_host(x)) if isinstance(x,jax.Array) else np.asarray(x)

    def save(name, data):
        error=None
        if rank0:
            try:
                root.mkdir(parents=True,exist_ok=True)
                (root/name).write_text(json.dumps(data,indent=2,default=str)+'\n')
            except OSError as exc:
                error=exc
        agree_io_refusal(error,path=root/name,stage='campaign.native.json')

    def arrays(name, **data):
        error=None
        if rank0:
            try:
                root.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(root/name,**data)
            except OSError as exc:
                error=exc
        agree_io_refusal(error,path=root/name,stage='campaign.native.npz')

    def resources(stage):
        # All ranks derive the same collective decision; root alone stats files.
        from jax.experimental import multihost_utils
        size = 0
        error=None
        if rank0:
            try:
                for path in Path(outdir).rglob('*'):
                    if path.is_file():
                        size += path.stat().st_size
            except OSError as exc:
                error=exc
        agree_io_refusal(error,path=outdir,stage='campaign.native.storage_stat')
        size = int(np.asarray(multihost_utils.broadcast_one_to_all(np.int64(size),is_source=rank0)))
        elapsed = time.perf_counter()-start
        # Root time is broadcast as well, avoiding rank-divergent refusal.
        elapsed = float(np.asarray(multihost_utils.broadcast_one_to_all(np.float64(elapsed),is_source=rank0)))
        receipt = dict(stage=stage,elapsed_s=elapsed,wall_budget_s=budget_s,
                       bytes=size,storage_budget_bytes=int(storage_gib*2**30),
                       storage_scope='this new leg only; no deletion, no shared source mutation')
        save('resources.json',receipt)
        if elapsed > budget_s or size > storage_gib*2**30:
            raise ValueError('GATE campaign_resources: explicit wall/storage budget reached; no continuation')
        return receipt

    def checked_density(rho, weights, stage):
        rho=host(rho)
        if not np.isfinite(rho).all():
            raise ValueError('GATE campaign_density: nonfinite density')
        adj=rho.conj().swapaxes(-1,-2)
        eig=np.linalg.eigvalsh(.5*(rho+adj))
        trace=np.einsum('k,tk->t',weights,np.trace(rho,axis1=-2,axis2=-1).real)
        row=dict(stage=stage,rho_min=float(eig.min()),rho_max=float(eig.max()),
                 hermiticity_max=float(abs(rho-adj).max()),trace=trace.tolist(),
                 trace_error_electrons_per_cell=float(abs(trace-reference['trace']).max()))
        health.append(row);save('checkpoint_health.json',health)
        if (row['rho_min'] < limits['rho_min'] or row['rho_max'] > limits['rho_max']
                or row['trace_error_electrons_per_cell'] > limits['trace_error_electrons_per_cell']
                or row['hermiticity_max'] > 1e-9):
            arrays('invalid_density.npz',rho=rho)
            raise ValueError('GATE campaign_density_bounds: saved state exceeds declared exploratory guard; no clipping')

    def factory(source,prepared,options):
        evaluate,cache=original_factory(source,prepared,options)
        def observed(basis,**kwargs):
            resources('before_native_evaluate')
            row=evaluate(basis,**kwargs)
            evaluated.append((basis,row));del evaluated[:-48]
            resources('after_native_evaluate')
            return row
        return observed,cache

    def snapshot(*args,**kwargs):
        try:
            built=original_snapshot(*args,**kwargs)
        except Exception:
            save('t0_refusal.json',dict(error=traceback.format_exc()))
            raise
        ops=args[1] if len(args)>1 else kwargs['ops']
        nk=len(built['bundle_of'])
        weights=host(ops.w_k)[:nk];req=host(ops.rho_eq)[:nk]
        reference.update(weights=weights,trace=float(np.einsum('k,k->',weights,np.trace(req,axis1=-2,axis2=-1).real)))
        save('exploratory_guards.json',dict(**limits,clipping=False,trace_repair=False,
            precision_claim=False,reference_active_electrons_per_cell=reference['trace']))
        save('table_conventions.json',dict(
            U_E_rho='instantaneous active basis, bundle k order; U columns in fixed reference orbital gauge',
            Gamma_D_MH='fixed reference orbital gauge, TD k order (already indexed by bundle_of)',
            raw_branch_tables='instantaneous active basis, BUNDLE k order; NOT indexed by bundle_of',
            retarded_pair='both particle and hole RETARDED; axis0 row-shell then column-shell',
            compatible_branch_pair='particle retarded; hole advanced swap+adjoint for branch_operators',
            active_branch_pair='active internal bands only; full branch_pair and retarded_pair include all virtuals',
            bundle_of='TD row i reads BUNDLE row bundle_of[i]',
            held_M_H_ha='fixed reference orbital gauge, TD k order; installed inner-solve INPUT',
            energy_units='explicit suffix Ry or Ha; E_ry Ry; time_fs fs',
            time_coverage='10fs integration normally emits200 samples through9.95fs; checkpoint may reach10fs'))
        arrays('raw_equilibrium_collision.npz',C_raw_eq_ha=host(built['c_eq']),
            hHF_eq_ry=built['state']['hf_reference_ry'],one_body_counterterm_ry=built['state']['one_body_counterterm_ry'])

        def accepted():
            state=built['state'];bases=state['basis_last'];index=int(state['n'])
            data=dict(U=np.stack([b['U'] for b in bases]),E_ry=np.stack([b['e'] for b in bases]),
                rho_in_basis=np.stack([b['rho'] for b in bases]),bundle_of=built['bundle_of'],
                time_fs=float(state['t_last'])/fs_au,held_M_H_ha=host(state['snapshot_held'])[:,:nk],
                M_H_eq_ry=host(built['eq']['m_h'])[built['bundle_of']])
            rows=[]
            for basis in bases:
                row=next((r for b,r in reversed(evaluated) if b is basis),None)
                if row is None:
                    raise ValueError('GATE campaign_capture: accepted native table provenance unavailable')
                rows.append(row)
            for key,name in [('gamma','Gamma_ha'),('injection','D_ha'),('m_h','M_H_current_ry')]:
                data[name]=np.stack([host(r[key])[built['bundle_of']] for r in rows])
                if name.endswith('_ha'):
                    data[name]*=ry_ha
            # Full complex retarded small tables, not merely Hermitian MH.
            for bank in ('retarded_pair','active_branch_pair','branch_pair'):
                for branch in ('cond','val'):
                    data[bank+'_'+branch+'_ry']=np.stack([host(r['raw'][bank][branch]) for r in rows])
            arrays(f'snapshot_{index:04d}.npz',**data)
            save(f'snapshot_{index:04d}.json',state['history'][-1])
            save('accepted_history.json',state['history'])
            rec=state['history'][-1]
            rel=np.max(np.abs(np.asarray(rec['number_residual_rel'])))
            absolute=np.max(np.abs(np.asarray(rec['number_residual_per_fs'])))
            if rel > limits['fresh_trace_relative'] and absolute > limits['fresh_trace_per_fs']:
                raise ValueError('GATE campaign_fresh_number: reciprocal active raw flux cancellation failed')
        accepted()
        refresh=built['refresh']
        def observed_refresh(*arguments,**keywords):
            try:
                update=refresh(*arguments,**keywords)
                if update is not None:
                    accepted()
                    checked_density(host(arguments[1])[:,:nk],weights,'refresh')
                    resource=resources('accepted_refresh')
                    if built['state']['n']==1:
                        rec=built['state']['history'][-1]
                        elapsed=resource['elapsed_s']
                        remaining=max(0,int(np.ceil((tf_fs-rec['t_fs'])/refresh_fs)))
                        predicted=elapsed+remaining*rec['wall_s']
                        from jax.experimental import multihost_utils
                        predicted=float(np.asarray(multihost_utils.broadcast_one_to_all(
                            np.float64(predicted),is_source=rank0)))
                        save('first_refresh_cost.json',dict(elapsed_s=elapsed,first_refresh_s=rec['wall_s'],
                            estimated_total_s=predicted,hard_budget_s=budget_s,
                            caveat='pilot extrapolation, later fixed-point iterations can increase cost'))
                        if predicted > budget_s:
                            raise ValueError('GATE campaign_estimated_wall: accepted pilot saved; 10fs exceeds explicit budget')
                return update
            except Exception:
                arrays('refused_refresh_density.npz',rho=host(arguments[1]),time_au=float(arguments[0]))
                save('refused_refresh.json',dict(error=traceback.format_exc(),
                    last_accepted=int(built['state']['n']),snapshot_failure=built['state'].get('snapshot_failure')))
                raise
        built['refresh']=observed_refresh
        return built

    def write(outs,system,i0,i1,samples,state_fn,t_state,fingerprint):
        state=state_fn()
        original_write(outs,system,i0,i1,samples,lambda:state,t_state,fingerprint)
        checked_density(host(state[0])[:,:system.nk],reference['weights'],f'checkpoint {float(t_state)/fs_au:.5f}fs')
        resources('checkpoint')

    sr.make_native_snapshot_evaluator=factory
    sr.snapshot_collision=snapshot
    stages._write_chunk=write
    def restore():
        sr.make_native_snapshot_evaluator=original_factory
        sr.snapshot_collision=original_snapshot
        stages._write_chunk=original_write
    return restore
