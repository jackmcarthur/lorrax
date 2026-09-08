"""Stage an immutable frozen-carrier control or synthetic finite-width sweep.

Reuse authenticated bound runs; preserve every G, FD, symmetry and Sigma owner.
The synthetic arm changes only the parent W synthesis and its spectral masks.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

CHECKOUT = Path(__file__).resolve().parents[1]
BASE = '043fabe83d167fe52eae92d7c2d10a66b5e6df1f'
S = Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14')
EVAL = S/'runs/DEV/152_shared_pole_push_2026-09-07/exchange/eval'
TEMPLATE = S/'runs/frequency_integration_sandbox/303_eval_damped_scoring_20260907/sweep_cost/sweep_Kfull'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Frozen seam changed: '+old[:100])
    return text.replace(old, new)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gamma-ev', type=float, default=0.)
    p.add_argument('--export', type=Path, help='Canonical EVAL all-q input; preserve left/right residues')
    p.add_argument('--allow-anticausal', action='store_true')
    p.add_argument('--signed', action='store_true',
                   help='Announced opt-in: real-pole signs dataset, active entries exactly +/-1')
    args = p.parse_args()
    if not os.getenv('SLURM_JOB_ID') or args.output.exists():
        raise RuntimeError('Compute-only staging into a new directory')
    sys.path.insert(0, str(EVAL))
    from residue_kernel import width_diagnostic
    from windowed_damped_kernel import W_CERT_RY, fourier_components
    width = width_diagnostic(np.array([(5.-1j*args.gamma_ev)/13.605693122994]),
                             allow_anticausal=args.allow_anticausal)
    if args.export and args.gamma_ev:
        raise ValueError('Export widths are physical inputs; do not override gamma')
    def git(*a):
        return subprocess.check_output(['git',*a],cwd=CHECKOUT,text=True).strip()
    commit = git('rev-parse','HEAD')
    assert not git('status','--porcelain'), 'Require clean committed source'
    changed = git('diff','--name-only',BASE,commit).splitlines()
    allowed = ['tools/audit_damped_carrier.py','tools/windowed_damped_kernel.py',
               'tools/verify_windowed_kernel.py','tools/prepare_windowed_sweep.py']
    if set(changed)-set(allowed):
        raise RuntimeError('Source differs outside EVAL2 tools')
    original = args.source_run.resolve()
    inputs = json.loads((original/'inputs.json').read_text())
    template = json.loads((TEMPLATE/'inputs.json').read_text())
    has_signs = 'signs' in inputs['model']['datasets']
    if has_signs != args.signed:
        raise ValueError('Signed dataset requires --signed; --signed requires a signs dataset')
    if args.signed and (args.gamma_ev or args.export):
        raise ValueError('Signed exponential carrier requires real-pole factors')
    # Receipt/model authentication, including the original real source pins.
    for record in (inputs['model'], inputs['generalized_sigma_owner'], inputs['geometry']):
        assert sha(record['path']) == record['sha256']
    out = args.output.resolve();out.mkdir()
    for sub in ('scripts','tmp','logs','evidence','artifacts'):
        (out/sub).mkdir()
    for name in ('CONTRACT.md','cohsex.in'):
        shutil.copy2(original/name,out/name)
    for name in ('zeta_q.h5','isdf_tensors_896.h5'):
        shutil.copy2(original/'tmp'/name,out/'tmp'/name)
    shutil.copy2(original/'scripts/smooth_fd_binding.py',out/'scripts/smooth_fd_binding.py')
    inputs['smooth_fd_binding']['path'] = str(out/'scripts/smooth_fd_binding.py')
    # Relocate the established wrapper, retaining runtime cleanliness,
    # ancestry and protected production-file authentication.
    adapter = (TEMPLATE/'scripts/run_tau200_sigma.py').read_text()
    old_owner = TEMPLATE/'scripts/run229_empty_view_owner.py'
    assert sha(old_owner) == 'fab8e2235d5382741702ac94f2c5812fd93bd69ffcac10c51a7cd696b41f238e'
    owner = old_owner.read_text()
    table_receipt = None
    general = bool(args.export)
    hermitian = True
    stage_receipt = None
    if general:
        import h5py
        # One canonical reader/factorization owner for every lane.
        sys.path.insert(0, str(TEMPLATE.parents[1]/'allq'))
        import stage_model
        model_path, stage_path = stage_model.stage(args.export, out/'staged',
            allow_anticausal=args.allow_anticausal)
        stage_receipt = json.loads(stage_path.read_text())
        hermitian = stage_receipt['hermitian_residues']
        shape = stage_receipt['shape']
        # The frozen carrier reads C. Convert L to C exactly once here; its
        # unchanged normalization restores L. Keep R as its own slab.
        carrier = out/'carrier.h5'
        with h5py.File(model_path, 'r') as src, h5py.File(carrier, 'x') as dst:
            omega = np.asarray(src['omega_ry'])
            mask = np.asarray(src['factor_mask'], bool)
            width = width_diagnostic(omega[mask], allow_anticausal=args.allow_anticausal)
            if np.any(omega.real[mask] > W_CERT_RY):
                raise ValueError('Export exceeds physical W_cert; no clipping')
            if not hermitian and np.any(omega.imag[mask] == 0):
                raise ValueError('Real non-Hermitian residues require an unimplemented PV kernel')
            dst.create_dataset('factor', shape=shape, dtype=np.complex128)
            dst.create_dataset('right', shape=shape, dtype=np.complex128)
            for q in range(shape[0]):
                dst['factor'][q] = np.asarray(src['left'][q])*np.sqrt(2*omega[q].real)[None,:]
                dst['right'][q] = src['right'][q]
            dst['poles2'] = omega.real**2
            dst['factor_mask'] = mask.astype(np.int32)
            dst['pole_id'] = np.where(mask, np.arange(shape[2]), -1)
            dst['retained_rank'] = mask.sum(axis=1).astype(np.int32)
        inputs['model'].update(path=str(carrier),sha256=sha(carrier),factor_shape=shape,
            freeze_receipt_path=str(stage_path),freeze_receipt_sha256=sha(stage_path),
            datasets={key:key for key in ('factor','poles2','factor_mask','pole_id','retained_rank')})
    if args.gamma_ev or general:
        import h5py
        started = time.monotonic()
        if not general:
            with h5py.File(inputs['model']['path'],'r') as f:
                names = inputs['model']['datasets']
                poles2 = np.asarray(f[names['poles2']])
                mask = np.asarray(f[names['factor_mask']],bool)
            omega = np.sqrt(np.where(mask,poles2,1.))-1j*args.gamma_ev/13.605693122994
        if np.any(omega.real[mask] > W_CERT_RY):
            raise RuntimeError('Synthetic centers exceed physical W_cert')
        geom = json.loads(Path(inputs['geometry']['path']).read_text())
        windows = geom['schedules']['primary/owner3']
        table = np.lib.format.open_memmap(out/'phase_table.npy',mode='w+',
            dtype=np.complex128,shape=(447,)+(() if hermitian else (2,))+omega.shape)
        keys=[];offset=0
        for window in windows:
            label=window['name']
            rec=inputs['run252']['source_rule_sha256']['447:'+label]
            assert sha(rec['path']) == rec['sha256']
            with np.load(rec['path']) as f:
                raw=f['times']
            times=np.conj(raw) if window['space']=='val' else raw
            lo,hi=window['pole_interval_ry'];hi=min(hi,W_CERT_RY)
            for t in times:
                value,dispersive=fourier_components(t,omega,lo,hi,
                    allow_anticausal=args.allow_anticausal)
                if hermitian:
                    table[offset]=np.where(mask,value,0j)
                else:
                    # H L + A(-D) = R(L-i(-D))/2 + R^dagger(L+i(-D))/2.
                    table[offset,0]=np.where(mask,(value-1j*dispersive)/2,0j)
                    table[offset,1]=np.where(mask,(value+1j*dispersive)/2,0j)
                keys.append(dict(window=label,time=[float(t.real),float(t.imag)],index=offset))
                offset+=1
            print('tabulated',label,offset,flush=True)
        assert offset==447
        table.flush();del table
        table_receipt=dict(path=str(out/'phase_table.npy'),sha256=sha(out/'phase_table.npy'),
            shape=[447]+([] if hermitian else [2])+list(omega.shape),keys=keys,W_cert_ev=149.7645,
            gamma_ev=args.gamma_ev,normalization='none',seconds=time.monotonic()-started,
            width=width,kernel_sha256=sha(CHECKOUT/'tools/windowed_damped_kernel.py'))
        (out/'phase_table.json').write_text(json.dumps(table_receipt,indent=2)+'\n')
        # Every damped mode overlaps every spectral interval. The boson
        # partition now applies to the scalar integral, not to pole centers.
        owner=replace_once(owner,'low_selected = mask_host & (pole_host <= low_edge)',
                           'low_selected = mask_host.copy()')
        owner=replace_once(owner,'high_selected = mask_host & (pole_host > low_edge) & (pole_host <= high_edge)',
                           'high_selected = mask_host.copy()')
        owner=replace_once(owner,'pole_owner = np.asarray(pole_owner, bool) & mask_host',
                           'pole_owner = mask_host.copy()')
        begin=owner.index('    @partial(jax.jit, out_shardings=parent_operator_sharding)\n    def synthesize_parents')
        end=owner.index('    fixed_parent = ',begin)
        owner=owner[:begin]+'''    # Run309 finite physical spectral intervals, no mass renormalization.
    phase_meta = json.loads((RUN / "phase_table.json").read_text())
    if collective_sha(RUN / "phase_table.npy") != phase_meta["sha256"]:
        raise RuntimeError("Run309 phase-table drift")
    phase_table = np.load(RUN / "phase_table.npy", mmap_mode="r")
    phase_keys = {(row["window"], complex(*row["time"])): row["index"]
                  for row in phase_meta["keys"]}
    owner_windows = {}
    @partial(jax.jit, out_shardings=parent_operator_sharding)
    def phased_matmul(left, right_, phase, e_ref_b, tau):
        phase = phase * jnp.exp(1j * e_ref_b * tau)
        return distrib_la.matmul(left * phase[:, None, :], right_, mesh=mesh,
            backend="off", batched_route="batch_reshard")
    def synthesize_parents(left, right_, omega, owner, e_ref_b, tau):
        key = (owner_windows[id(owner)], complex(np.asarray(tau)))
        # Only one small [29,K] phase slice enters the device per call. The
        # read-only 447-node table stays in host mmap, never a jit argument.
        phase = replicate_to_mesh(np.asarray(phase_table[phase_keys[key]]), mesh)
        return phased_matmul(left, right_, phase, e_ref_b, tau)

'''+owner[end:]
        owner=replace_once(owner,'        return {\n            "branch": branch, "e_call": e_call,',
                           '        result = {\n            "branch": branch, "e_call": e_call,')
        owner=replace_once(owner,'\n    schedule_operands = {}',
            '\n        owner_windows[id(result["pole_owner"])] = window["name"]\n'
            '        owner_windows[id(result["full_pole_owner"])] = window["name"]\n'
            '        return result\n\n    schedule_operands = {}')
        owner=owner.replace('b=C/sqrt(2Omega); b exp(-iOmega*t) b^H',
                            'Run309: unnormalized finite spectral interval; b=C/sqrt(2Omega_center)')
        if general:
            owner=replace_once(owner,'        poles2 = reader.read_slab(',
                '        right_factor = reader.read_slab("right", shape=(NPARENT, NMU_LOGICAL, kmax),\n'
                '            partition_spec=P(None, "x", "y"))\n        poles2 = reader.read_slab(')
            owner=replace_once(owner,'    right = jax.jit(',
                '    right_factor = meta.mu_basis.pack_axis(right_factor, 1, spec=P(None, "x", "y"))\n'
                '    right = jax.jit(')
            owner=replace_once(owner,'out_shardings=parent_factor_sharding)(b)',
                'out_shardings=parent_factor_sharding)(right_factor)')
            # Every mode contributes to both intervals; never regenerate the
            # right factor from the left in the old compact-view shortcut.
            begin=owner.index('    view_bounds = {')
            end=owner.index('    view_selected = ',begin)
            owner=owner[:begin]+'    view_bounds = {label: (0, kmax) for label in ("low", "high", "full")}\n'+owner[end:]
            owner=replace_once(owner, '        if label == "full":',
                '        if lo == 0 and hi == kmax:')
            if not hermitian:
                old='''        phase = phase * jnp.exp(1j * e_ref_b * tau)
        return distrib_la.matmul(left * phase[:, None, :], right_, mesh=mesh,
            backend="off", batched_route="batch_reshard")'''
                new='''        phase = phase * jnp.exp(1j * e_ref_b * tau)
        direct = distrib_la.matmul(left * phase[0, :, None, :], right_, mesh=mesh,
            backend="off", batched_route="batch_reshard")
        mirror_left = jnp.conj(jnp.swapaxes(right_, -1, -2))
        mirror_right = jnp.conj(jnp.swapaxes(left, -1, -2))
        mirror = distrib_la.matmul(mirror_left * phase[1, :, None, :], mirror_right,
            mesh=mesh, backend="off", batched_route="batch_reshard")
        return direct + mirror'''
                owner=replace_once(owner,old,new)
    if args.signed:
        # W(t) = B diag(s exp(-i Omega t)) B^H. Keep boolean ownership
        # through the partition checks, then encode signed weights. Compact
        # and full views receive identical weights; B and B^H are unchanged.
        owner=replace_once(owner, '        poles2 = reader.read_slab(',
            '        signs = reader.read_slab(datasets["signs"], shape=(NPARENT, kmax),\n'
            '            partition_spec=P(None, None))\n        poles2 = reader.read_slab(')
        owner=replace_once(owner, '    safe_poles2 = ',
            '    signs_host = np.asarray(gather_to_host(signs))\n'
            '    if not np.all(np.isin(signs_host, [-1, 1])):\n'
            '        raise ValueError("Signed carrier requires signs exactly +/-1, including padding")\n'
            '    print("[EVAL2 signed carrier] active negative columns",\n'
            '          int(np.sum((signs_host < 0) & mask_host)), flush=True)\n'
            '    safe_poles2 = ')
        owner=replace_once(owner,
            '            owner, jnp.exp(-1j * (omega - e_ref_b) * tau),',
            '            owner != 0, owner * jnp.exp(-1j * (omega - e_ref_b) * tau),')
        owner=replace_once(owner,
            '"pole_owner": replicate_to_mesh(compact_owner, mesh),',
            '"pole_owner": replicate_to_mesh(compact_owner * signs_host[:, lo:hi], mesh),')
        owner=replace_once(owner,
            '"full_pole_owner": replicate_to_mesh(pole_owner, mesh),',
            '"full_pole_owner": replicate_to_mesh(pole_owner * signs_host, mesh),')
        inputs['eval2_signed_carrier'] = dict(enabled=True,
            convention='B diag(signs exp(-i Omega tau)) B^H; C=B sqrt(2 Omega)',
            absent_signs='unchanged frozen owner arithmetic')
    owner_path=out/'scripts/run229_eval2_owner.py';owner_path.write_text(owner)
    start=adapter.index('OWNER = Path(');end=adapter.index('SMOOTH = ',start)
    adapter=adapter[:start]+f'OWNER = Path({str(owner_path)!r})\nOWNER_SHA256 = {sha(owner_path)!r}\n'+adapter[end:]
    (out/'scripts/run_tau200_sigma.py').write_text(adapter)
    binding=dict(template['source_binding'])
    binding.update(checkout=str(CHECKOUT),base_commit=BASE,actual_commit=commit,
                   allowed_changed_paths=allowed,actual_changed_paths=changed)
    for name,digest in binding['protected_sha256'].items():
        assert sha(CHECKOUT/name)==digest
    inputs.update(source_binding=binding,adapter_sha256=sha(out/'scripts/run_tau200_sigma.py'))
    inputs['eval2_spectral_window']=dict(gamma_ev=args.gamma_ev,table=table_receipt,
        W_cert_ev=149.7645,normalization='none',scope='general adjoint-mirrored export' if general else 'synthetic Hermitian residues only',
        stage_receipt=stage_receipt,hermitian_rank_one_shortcut=hermitian,
        allow_anticausal=args.allow_anticausal,width=width)
    (out/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    receipt=dict(status='PREPARED',job_step=os.environ['SLURM_JOB_ID']+'.'+os.getenv('SLURM_STEP_ID','?'),
        source_run=str(original),source_input_sha256=sha(original/'inputs.json'),source_commit=commit,
        input_sha256=sha(out/'inputs.json'),owner_sha256=sha(owner_path),gamma_ev=args.gamma_ev,
        spectral_mask='all active modes per finite interval' if args.gamma_ev else 'unchanged real-pole owner',
        phase_table=table_receipt)
    (out/'prepare_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (out/'manifest.yaml').write_text('run_id: '+out.name+'\nsystem: Na\nplatform: perlmutter\nsteps:\n  prepare: {state: complete}\n  sweep: {state: pending}\n')
    print('PREPARED',out,flush=True)


if __name__ == '__main__':
    main()
