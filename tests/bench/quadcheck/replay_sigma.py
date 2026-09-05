"""Matched diagnostic of retained E4; coordinator-authorized affine input.

Use the production SC map, tau contraction, Hermitianization and scissor.
Only diagnostic providers (input eigenpair, read-only W and retained rules)
are replaced in this process. No source or retained store is mutated.
"""
from pathlib import Path
import os
import sys
import json
import re
import hashlib
import gc
import subprocess
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[3]
SANDBOX = Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14')
SOURCE = SANDBOX/'runs/Na/12_sc_observables_eta05_2026-09-04/arm_E_eta05_rcrop_metric/qsgw'
EVIDENCE = SANDBOX/'runs/Na/14_quadcheck_2026-09-05'
OUT = EVIDENCE/'replay'
TARGET = 'ω≥E_F cond:pole_tail'
MU = 0.09667275484535012

# Driver import owns startup, before JAX or numerical module imports.
from gw import gw_jax
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from common.units import RYD_TO_EV
from common.collectives import gather_to_host, process_rank, barrier
from common.collectives import device_put_process_local
from gw import sc_iteration as sc
from gw import sigma_box_plan as box
from gw.mpa import sigma as mpa_sigma
from gw import qsgw_head
from gw.efermi import OccupationState, mp1_occupations, assert_fixed_n
from gw.dynamic_sigma import extract_sigma_diag_logical
from gw.qsgw_utils import interp_along_omega

def print(*values, **kwargs):
    if process_rank() == 0:
        sys.__stdout__.write(' '.join(str(v) for v in values)+'\n')
        sys.__stdout__.flush()

ARM = 'retained'
RECEIPTS = {}
for line in (SOURCE/'evidence/armE/gw.rank0000.log').read_text().splitlines():
    if 'SC fixed window:' in line:
        name = line.split('SC fixed window: ')[1].split(': n_tau=')[0]
        RECEIPTS.setdefault(name, []).append(line)
RECEIPTS = {name: rows[4] for name, rows in RECEIPTS.items() if len(rows) > 4}
RULES = {}
for path in (SOURCE/'tmp/sigma_quadrature_rules').glob('*.npz'):
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}
    for conjugate in (False, True):
        t, w = data['times'], data['weights']
        if conjugate:
            t, w = -np.conj(t), np.conj(w)
        digest = hashlib.sha256(np.ascontiguousarray(t).tobytes()+np.ascontiguousarray(w).tobytes()).hexdigest()[:16]
        RULES[digest] = (data, t, w, str(path))


def retained_fits(specs, eta, **kwargs):
    """Restore exact accepted call-4 nodes on the current tuple partition."""
    fits = []
    assert {s['name'] for s in specs} == set(RECEIPTS)
    for spec in specs:
        print('SPEC_DEBUG',spec['name'],'box_eV',np.asarray(spec['box'])*RYD_TO_EV,'state_eV',np.min(spec['states'])*RYD_TO_EV,np.max(spec['states'])*RYD_TO_EV,'freq_eV',np.min(spec['frequencies'])*RYD_TO_EV,np.max(spec['frequencies'])*RYD_TO_EV)
        receipt = RECEIPTS[spec['name']]
        digest = re.search(r'nodes=(\w+)', receipt)[1]
        data, t, w, path = RULES[digest]
        physical = np.array([float(x) for x in re.search(r' box=\((.*?)\) eV', receipt)[1].split(',')])/RYD_TO_EV
        error = float(np.max(abs(np.asarray(spec['box'])-physical))*RYD_TO_EV)
        assert error < 1e-7, (spec['name'], 'support differs from call4', error, 'actual_eV', np.array(spec['box'])*RYD_TO_EV, 'expected_eV', physical*RYD_TO_EV)
        rule_box = tuple(data['box'])
        noise_amp = float(data['roundoff_amplification'])
        relative = bool(data['relative'])
        sup = float(data['sup_error'])
        if spec['name'] == TARGET and ARM != 'retained':
            path = str(EVIDENCE/'rule_audit'/('physical_120s.npz' if ARM == 'accurate24' else 'contour_gl24.npz'))
            with np.load(path) as f:
                t, w = f['times'], f['weights']
                rule_box = tuple(f['box'])
                relative = bool(f['relative'])
                sup = float(f['sup_error'] if 'sup_error' in f else f['measured_sup'])
            assert np.all(np.isfinite(t)) and np.all(np.isfinite(w))
            cloud = box.box_samples(*rule_box, per_unit=8., n_im=48)
            rho = np.abs(cloud) if relative else rule_box[2]
            noise_amp = box.rule_roundoff_amplification(t,w,cloud,rho)
            assert noise_amp*box._RUNTIME_NOISE_EPSILON <= 5e-6
        growth = box._factor_growth(t, spec['pole_sign'], spec['states'], spec['pole_stats'], spec['E_ref_A'], spec['E_ref_B'])
        assert max(growth) <= box._FACTOR_GROWTH_CAP
        fit = dict(times=t, weights=w, node_count=len(t), rule_box=rule_box,
                   relative=relative, sup_error=sup, kappa_max=float(data['kappa_max']),
                   theta_deg=float(data['theta_deg']), rank=int(data['rank']), seconds=0.,
                   cache_status='diagnostic:'+path, factor_growth=growth,
                   noise_bound=noise_amp*box._RUNTIME_NOISE_EPSILON, noise_budget=5e-6, roundoff_amplification=noise_amp,
                   node_digest=hashlib.sha256(np.ascontiguousarray(t).tobytes()+np.ascontiguousarray(w).tobytes()).hexdigest()[:16],
                   cache_write_warning=None, cache_lookup_warnings=(), one_line='retained E4 diagnostic; noise receipt in original log')
        fits.append(fit)
        if process_rank() == 0:
            print('REPLAY_RULE', ARM, spec['name'], 'support_error_ev', error, 'nodes',len(t),'digest',fit['node_digest'], flush=True)
    return fits, []

box.fit_sigma_box_specs = retained_fits

# Retained head is consumed from the fit by Sigma dispatch. No new head/body
# screening is built; the static terms are held identical between arms.
sc.load_head_velocity_source = lambda *a, **k: None
qsgw_head.build_dft_head_response = lambda *a, **k: None


def approximate_head_match(attrs, state, **kwargs):
    """Explicit diagnostic authorization, not an authenticated restart claim."""
    assert attrs['occ_hash'] == 'e33cfdca02aa7439'
    assert float(attrs['mu_ry']) == MU == state.mu_ry
    if process_rank() == 0:
        print('AFFINE_OCCUPATION_AUTHORIZATION retained_hash=',attrs['occ_hash'],
              'reconstructed_hash=',state.occ_hash,'input_rotation_residual_ev=1.4572556481210558e-10',flush=True)
    return 'coordinator_authorized_affine_diagnostic'

mpa_sigma.assert_head_body_occupation_match = approximate_head_match
ORIGINAL_COMPUTE = sc.compute_sigma_xc


def compute_and_write(*args, **kwargs):
    if process_rank() == 0:
        np.savez(OUT/'sigma_input_debug.npz',enk=np.asarray(kwargs['wfns'].enk),occ=np.asarray(kwargs['occupation_state'].f_kn),mu=kwargs['occupation_state'].mu_ry,omega=kwargs['config'].omega_grid_ev,threshold=kwargs['config'].mpa.occupation_window_threshold)
    # The full-head SC initializer normally supplies these per-map arrays.
    # W/head itself is retained; only its current-basis consumer table is rebuilt.
    kwargs['iteration_head'] = qsgw_head.IterationHeadSamples(
        omegas=(), samples=(), sigma_energies_ry=np.asarray(kwargs['wfns'].enk),
        sigma_occupations=np.asarray(kwargs['occupation_state'].f_kn), efermi_ry=MU)
    assert kwargs['static_head_terms'] is not None
    kwargs['write_sigma_omega_h5'] = True
    kwargs['input_dir'] = str(OUT/ARM)
    return ORIGINAL_COMPUTE(*args, **kwargs)

sc.compute_sigma_xc = compute_and_write


def replay(state_init, inputs, **kwargs):
    global ARM
    with np.load(EVIDENCE/'reconstruction/approximate_input_map0004_NOT_AUTHENTICATED.npz') as f:
        k = f['kpoints']
        index = np.ravel_multi_index((np.rint(np.mod(k,1)*8).astype(int)%8).T, (8,8,8))
        from symmetry_maps import KStarMap
        labels = np.asarray(inputs.sym.irr_idx_k)[index]
        assert len(np.unique(labels)) == len(index) == 29
        stars = KStarMap(inputs.sym.irr_idx_k, inputs.sym.sym_idx_k, int(inputs.wfn.ntran), labels=labels)
        eh = stars.broadcast(f['E_input_ev']/RYD_TO_EV)
        uh = np.load(SOURCE/'sc_history/rotation_iter0004.npy')
    mesh = inputs.mesh_xy
    ks = sc._kstar(inputs)
    eloop = eh if ks.is_identity else ks.select(eh)
    uloop = uh if ks.is_identity else ks.select(uh)
    ep = device_put_process_local(eloop, NamedSharding(mesh,P(None,None)))
    up = device_put_process_local(uloop, NamedSharding(mesh,P(None,'x','y')))
    # H=U diag(E) U^dagger; retain the full stored gauge and canonical star energies.
    hp = sc._rotate_to_dft_basis(ep[:,:,None]*jnp.eye(eh.shape[1])[None], up, mesh=mesh)
    sc._sc_eigh_bands = lambda *a, **k: (ep, up)
    occupations = mp1_occupations(device_put_process_local(eh, NamedSharding(mesh,P(None,None))), MU, 0.01, clamp_tol=inputs.config.occupation_clamp_tol)
    occ = OccupationState(occupations,MU,'mp1',0.01,9.)
    # Rounded affine input shifts N by 1.212e-10 at the retained mu.
    # Diagnostic tolerance only; keep mu and occupations identical in all arms.
    assert_fixed_n(occ,np.full(512,1/512),state_capacity=float(inputs.wfn.occupation_state_capacity),atol=1e-8)
    sc._solve_head_occupations = lambda *a, **k: (occ,None)
    from gw.vcoul import compute_q0_averages
    from gw.head_correction import compute_static_head_terms
    # Bare vc0 is geometric and independent of the screened S tensor.
    # Only sigma_x_diag is consumed in MPA; its screened head comes from W4.
    vc0, _ = compute_q0_averages(
        inputs.wfn, jnp.asarray(1.,dtype=jnp.float64), inputs.meta,
        analytic_sphere=bool(getattr(inputs.config.head,'analytic_q0_sphere',inputs.config.head.head_minibz_average)))
    static_head = compute_static_head_terms(
        vc0=complex(vc0), wcoul0_static=complex(vc0), occ=occupations,
        cell_volume=inputs.meta.cell_volume, nk_tot=inputs.meta.nk_tot,
        source='diagnostic bare-X geometry; dynamic screened head is retained E4')
    print('REPLAY_HEAD', 'vc0',complex(vc0),'occupation_hash',occ.occ_hash,'dynamic_fit',str(SOURCE/'tmp/mpa/mpa_fit_sc_0004.h5'))
    inputs = replace(inputs, static_head_terms=static_head, parallel_transport=None, fixed_dft_head_response=None,
                     screening_model_fn=lambda *a, **k: {'mpa_fit': str(SOURCE/'tmp/mpa/mpa_fit_sc_0004.h5')},
                     fixed_quadrature_session=None, print_fn=print)
    initial = replace(state_init,H_qp_dft=hp,iteration=4)
    if process_rank() == 0:
        np.savez(OUT/'input_debug.npz',E_full_ev=eh*RYD_TO_EV,occ=np.asarray(occupations),indices=index)
    if process_rank() == 0:
        print('REPLAY_INPUT',json.dumps(dict(energy_shape=list(eh.shape),mu_ry=MU,occ_hash=occ.occ_hash,
              nelec=float(np.asarray(occupations).sum()/512*float(inputs.wfn.occupation_state_capacity)),
              source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              ranks=jax.process_count(),devices=jax.device_count(),mesh=str(mesh))),flush=True)
    for ARM in ('retained','accurate24','dense_contour384'):
        (OUT/ARM).mkdir(parents=True,exist_ok=True)
        if process_rank() == 0:
            print('REPLAY_START',ARM,flush=True)
        result = sc.gw_iteration_map(initial,inputs)
        sigma = result.outputs.sigma_result
        diag = extract_sigma_diag_logical(sigma.sigma_c_omega_kij_ry,mesh,band_axis=sigma.sigma_band_axis)
        relative = eloop*RYD_TO_EV-MU*RYD_TO_EV
        shell = interp_along_omega(diag*RYD_TO_EV,np.asarray(inputs.config.omega_grid_ev),relative,
                                  out_of_range='clamp',context='diagnostic E4 on-shell',print_fn=print)
        h = gather_to_host(result.H_qp_dft)
        sx = gather_to_host(sigma.sigma_xc_kij_ry)
        if not ks.is_identity:
            h, sx, shell = ks.broadcast(h), ks.broadcast(sx), ks.broadcast(shell)
        relative = eh*RYD_TO_EV-MU*RYD_TO_EV
        if process_rank() == 0:
            np.savez(OUT/ARM/'observables.npz',H_dft_ev=h*RYD_TO_EV,
                     sigma_xc_qp_ev=sx*RYD_TO_EV,sigma_on_shell_ev=shell,E_input_ev=eh*RYD_TO_EV,
                     covered=(relative>=inputs.config.omega_grid_ev[0])&(relative<=inputs.config.omega_grid_ev[-1]),
                     protected=result.partition.protected_mask,in_range=result.partition.in_range_mask)
            print('REPLAY_DONE',ARM,flush=True)
        barrier('quadcheck.'+ARM,print_fn=print)
        del result,sigma,diag,shell,h,sx
        gc.collect()
    from compare_replay import compare
    compare(OUT,SOURCE,write=process_rank()==0)
    barrier('quadcheck.analysis',print_fn=print)
    raise SystemExit(0)

sc.run_self_consistency = replay
if __name__ == '__main__':
    gw_jax.main(['-i',str(OUT/'replay.in')])
