"""Prepare native full-microscopic matched trajectories; never launch a job."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def toml(data):
    lines = []
    def value(x):
        if isinstance(x, bool):
            return str(x).lower()
        if isinstance(x, list):
            return '[' + ', '.join(value(y) for y in x) + ']'
        return json.dumps(x) if isinstance(x, str) else repr(x)
    def table(name, body):
        if name:
            lines.append('[' + name + ']')
        lines.extend(k + ' = ' + value(v) for k, v in body.items() if not isinstance(v, dict))
        lines.append('')
        for k, v in body.items():
            if isinstance(v, dict):
                table(name + '.' + k if name else k, v)
    table('', data)
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, help='new campaign directory on launch machine')
    p.add_argument('--source', required=True, help='exact full microscopic source_eta010.in')
    p.add_argument('--ref', required=True, help='unchanged 516/07 GN-PPM TD bundle directory')
    p.add_argument('--source-tree', required=True)
    p.add_argument('--launch-root', default='', help='absolute campaign path on launch machine; defaults to --out')
    p.add_argument('--launch-source', default='', help='copied identical source deck on launch machine')
    p.add_argument('--fields', default='0,7e9,1e10')
    p.add_argument('--refresh-fs', type=float, default=.5)
    p.add_argument('--tf-fs', type=float, default=10.)
    p.add_argument('--support-sites-ev', default='')
    p.add_argument('--active-window', choices=('2v2c', '30v14c'), default='2v2c',
                   help='propagated window only; screening retains all160 source bands')
    p.add_argument('--leg-budget-s', type=float, default=3600.)
    p.add_argument('--storage-gib', type=float, default=64.)
    a = p.parse_args()
    if re.fullmatch('[0-9a-f]{40}', a.source_tree) is None:
        p.error('--source-tree must be the complete immutable 40-character tree hash')
    if min(a.refresh_fs, a.tf_fs, a.leg_budget_s, a.storage_gib) <= 0:
        p.error('cadence, final time and explicit budgets must be positive')
    lo, hi = (128, 132) if a.active_window == '2v2c' else (100, 144)
    root, ref, source = Path(a.out).resolve(), Path(a.ref).resolve(), Path(a.source).resolve()
    launch_root = Path(a.launch_root).resolve() if a.launch_root else root
    launch_source = str(Path(a.launch_source).resolve()) if a.launch_source else str(source)
    if (root / 'manifest.json').exists():
        raise FileExistsError('Refusing existing manifest; choose a fresh campaign directory')
    source_text = source.read_text()
    compact = '\n'.join(line.split('#')[0].strip().replace(' ', '').lower() for line in source_text.splitlines())
    for required in ('number_bands=160', 'sigma_w_model=shared_pole', 'sigma_regularization_ev=0.1'):
        if required not in compact.splitlines():
            raise ValueError('Source does not authenticate required setting: ' + required)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for field in map(float, a.fields.split(',')):
        for arm in ('joint_frozen', 'joint'):
            tag = f'E{field:g}_{arm}'
            out = launch_root / tag
            pulse = (f'{field!r} * np.exp(-((t - 3000.0) / 1000.0)**2) '
                     '* np.cos(0.07 * t / 24.188843265857)')
            deck = dict(control=dict(calc_type='COHSEX',
                timesteps=dict(t0=0., tf=a.tf_fs*1000., dt_measure=50., ntcheck=5),
                files=dict(outdir=str(out))),
                system=dict(ibndmin=lo+1, ibndmax=hi, ivbm=130, geometry='slab', esmear=.05,
                    depopulation=0., dephasing=0., Efunc=dict(E_x=pulse,E_y=0.,E_z=0.)),
                measurements=dict(types=['Efield','polarization','carrier_density','density_matrix'],
                    **{'3':dict(dataset_name='rho_full',brange1=[lo+1,hi+1],brange2=[lo+1,hi+1])}),
                lorrax=dict(method='cfm4',dt=25.,n_corr=2,chunk=5,stiff_modes=0,
                    system='bundle',bundle_input=str(ref/'oneshot.in'),dipole_file=str(ref/'dipole.h5'),
                    model='isdf',collision='raw',collision_source=launch_source,collision_screening=arm,
                    collision_statistics='retarded_reference',collision_hamiltonian='hf',
                    collision_spm_chi0_active=[88,144],
                    collision_basis='instantaneous',collision_hermitian='dynamic',
                    collision_internal_pairing='vertex',collision_rate_intermediates='active',
                    collision_in_scattering='full',collision_refresh_fs=a.refresh_fs,
                    collision_refresh_df=0.,collision_adiabatic_ev=.05,collision_fit_tolerance=.02,
                    collision_snapshot_steps=32,collision_snapshot_tol_ev=1e-4,
                    collision_snapshot_mixing=.7,collision_snapshot_acceleration='anderson'))
            if a.support_sites_ev:
                deck['lorrax']['collision_spm_support_sites_ev'] = a.support_sites_ev
            path = root/(tag+'.toml')
            path.write_text(toml(deck))
            rows.append(dict(tag=tag,field_v_m=field,arm=arm,deck=str(launch_root/path.name),out=str(out)))
    manifest = dict(status='PREPARED_NOT_AUTHORIZED_OR_RUN',source=launch_source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_tree=a.source_tree,
        source_recipe='native full-map shared-pole; no prototype rank/order/sample controls',
        microscopic_space=dict(nmu=978,nq=36,virtual_bands=160,active_zero_based=[lo,hi],
                               response_partition_zero_based=[88,144]),
        active_window_status='PREPARED_ONLY; no leakage, cost or convergence admission',
        density_output_bytes_per_trial=math.ceil(a.tf_fs/.05)*36*(hi-lo)**2*16,
        hamiltonian='absolute hHF[fullrho]+MH[rho,H], converged inner solve; NOT reference anchored',
        statistics='retarded_reference',collision='raw; no equilibrium subtraction or trace repair',
        gamma_ev=.05,eta_ev=.1,tf_fs=a.tf_fs,refresh_fs=a.refresh_fs,
        cadence_status='exploratory underconverged; half-fs is not an accuracy certificate',
        dt_as=25.,sample_as=50.,pulse_ha=.07,pulse_center_fs=3.,pulse_width_fs=1.,
        nominal_intensity_w_cm2={'7e9':6.50332588799e12,'1e10':1.32720936490e13},
        leg_budget_s=a.leg_budget_s,storage_gib=a.storage_gib,
        common_source_storage_gib=a.storage_gib,
        health=dict(rho_min=-1e-3,rho_max=1.001,trace_error_electrons_per_cell=1e-3,
            fresh_trace_relative=1e-8,fresh_trace_per_fs=1e-8),
        static_A=dict(scope='static TD-aGW, NOT same coherent functional as B/C; exact active-window match required',
            root=('/home/jackm/mgkba_campaign_20261019/coherent_strong_calibration_2v2c'
                  if a.active_window == '2v2c' else ''),
            remote_root=('/pscratch/sd/j/jackm/cs_claude/coll_stab/mgkba_20261019/coherent_strong_calibration_2v2c'
                         if a.active_window == '2v2c' else ''),
            tags=['E7e+09_off','E1e+10_off'],
            status='existing2v2c only' if a.active_window == '2v2c' else 'NO_MATCHED_STATIC_A_YET'),runs=rows)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
