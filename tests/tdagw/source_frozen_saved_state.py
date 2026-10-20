"""Verification-only full microscopic ordered Sigma with SOURCE-FROZEN W.

This is NOT production joint_frozen, does not call the general W builder,
does not solve the new Hamiltonian fixed point, and propagates no trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import runtime


def evaluate_source_frozen_snapshot(source, prepared, basis, *, b_lo, model, print_fn=print):
    """Use current full-rho lines and one frozen source bank in the native executor."""
    import numpy as np
    from common.collectives import gather_to_host
    from gw.ppm_sigma import shared_pole_window_source, window_sigma_c, window_sigma_plan
    from tdagw.collision import branch_operators

    U, e, rho = (np.asarray(basis[k]) for k in ("U", "e", "rho"))
    start, stop, nb = int(b_lo), int(b_lo)+e.shape[1], int(prepared["nb"])
    if (not 0 <= start < stop <= nb or e.shape[0] != prepared['nk']
            or U.shape != (*e.shape, e.shape[-1]) or rho.shape != U.shape
            or not all(np.isfinite(a).all() for a in (U, e, rho))):
        raise ValueError('GATE source_frozen_snapshot: invalid active tables or source extent')
    if (np.max(abs(U.conj().swapaxes(-1,-2)@U-np.eye(e.shape[-1]))) > 1e-7
            or np.max(abs(rho-rho.conj().swapaxes(-1,-2))) > 1e-8):
        raise ValueError('GATE source_frozen_snapshot: nonunitary basis or non-Hermitian density')
    energy = np.asarray(gather_to_host(source["wfns"].enk)).copy()
    energy[:, :start] += (e[:, 0]-energy[:, start])[:, None]
    energy[:, stop:nb] += (e[:, -1]-energy[:, stop-1])[:, None]
    energy[:, start:stop] = e
    handle = shared_pole_window_source(model, source["meta"],
        efermi_ry=prepared["efermi_ry"], linalg=source.get("linalg"))
    window = dict(b_lo=start, U=U, e=e, rho=rho,
        f=np.diagonal(rho, axis1=-2, axis2=-1).real, e_frozen=energy)
    plan = window_sigma_plan(source["config"], head=source.get("head"), window_only=True)
    raw = window_sigma_c(source["wfns"], handle, window, plan, meta=source["meta"],
        mesh_xy=prepared["mesh"], branches=True, pairing="vertex", print_fn=print_fn)
    active, full = raw["active_branch_pair"], raw["branch_pair"]
    gamma, injection, _ = branch_operators(active["cond"], active["val"])
    _, _, mh = branch_operators(full["cond"], full["val"])
    Uh = U.conj().swapaxes(-1, -2)
    return dict(gamma=U@np.asarray(gamma)@Uh, injection=U@np.asarray(injection)@Uh,
                m_h=U@np.asarray(mh)@Uh, raw=raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--b-lo', type=int, default=128)
    parser.add_argument('--trial', type=int, default=0)
    args = parser.parse_args()
    stack = runtime.initialize_communicator_stack()
    import jax
    import numpy as np
    from common.collectives import gather_to_host
    from common.units import RYD_TO_EV
    from file_io.commit_state import agree_io_refusal
    from gw.ppm_sigma import fresh_shared_pole_capacity
    from gw.shared_pole_screening import _shared_pole_tables
    from gw.shared_pole_reference import source_frozen_ordered_model
    from tdagw import dynsig, snapshot_runtime as sr
    from tdagw.collision import boundary_diagnostics

    if jax.process_count() != 4:
        raise ValueError('GATE source_frozen_verification: requires four ranks')
    rank0 = jax.process_index() == 0
    say = print if rank0 else (lambda *a, **k: None)
    root = Path(args.out).resolve()
    error = None
    if rank0:
        try:
            if root.exists():
                raise FileExistsError(str(root))
            root.mkdir(parents=True)
        except OSError as exc:
            error = exc
    agree_io_refusal(error, path=root, stage='source_frozen/output')
    with np.load(args.snapshot) as saved:
        basis = {key: np.asarray(saved[name][args.trial]) for key, name in
                 (("U", "U"), ("e", "E_ry"), ("rho", "rho_in_basis"))}
        saved_time = float(saved['time_fs'])
    report = dict(status='running', control='SOURCE-FROZEN EQUILIBRIUM W',
        refreshed_screening=False, physical_time_propagated_fs=0,
        source=str(Path(args.source).resolve()), snapshot=str(Path(args.snapshot).resolve()),
        source_input_sha256=hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
        snapshot_sha256=hashlib.sha256(Path(args.snapshot).read_bytes()).hexdigest(),
        saved_time_fs=saved_time, commit=os.environ.get('LEG_HEAD'),
        source_tree=os.environ.get('LEG_SOURCE_TREE'),
        archive_sha256=os.environ.get('LEG_SOURCE_ARCHIVE_SHA256'), walls_s={})
    def save():
        if rank0:
            (root/'receipt.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
    def timed(name, fn):
        report['phase'] = name; save()
        start = time.perf_counter()
        try:
            return fn()
        finally:
            report['walls_s'][name] = time.perf_counter()-start
            say(f'[SOURCE-FROZEN W verification] {name}: {report["walls_s"][name]:.3f}s', flush=True)
            save()
    started = time.perf_counter()
    rc = 0
    try:
        source = timed('source', lambda: dynsig.gw_window_source(args.source, stack.mesh,
            workdir=str(root/'driver'), screening=True, say=say))
        prepared = timed('prepare', lambda: sr.prepare_native_source(source, stack.mesh))
        meta = fresh_shared_pole_capacity(source['meta'], stack.mesh)
        tables = _shared_pole_tables(meta, source['screening_inputs']['kwargs']['sym'],
                                     meta.mu_basis.canonical_indices)
        model = timed('exact_source_adapter', lambda: source_frozen_ordered_model(
            source['poles']['model'], meta, mesh_xy=stack.mesh, tables=tables,
            path=str(root/'source_frozen_ordered'/'model.h5')))
        report['adapter'] = model['source_frozen_receipt']; save()
        result = timed('native_ordered_sigma', lambda: evaluate_source_frozen_snapshot(
            source, prepared, basis, b_lo=args.b_lo, model=model, print_fn=say))
        U, rho = basis['U'], basis['rho']
        rho_fixed = U@rho@U.conj().swapaxes(-1,-2)
        gamma, injection = result['gamma'], result['injection']
        collision = injection-.5*(gamma@rho_fixed+rho_fixed@gamma)
        outgoing = -np.trace(gamma@rho_fixed, axis1=-2,axis2=-1).real.mean()
        incoming = np.trace(injection,axis1=-2,axis2=-1).real.mean()
        flux = .5*(abs(outgoing)+abs(incoming))
        arrays = dict(gamma=gamma, injection=injection, m_h=result['m_h'],
                      rho_fixed=rho_fixed, C_raw_ry=collision, **basis)
        boundary = boundary_diagnostics(gamma, injection, rho_fixed, occupation_tolerance=1e-8)
        arrays.update({f'boundary_{key}': value for key, value in boundary.items()})
        for name in ('retarded_pair','branch_pair','active_branch_pair'):
            for branch, value in result['raw'][name].items():
                arrays[f'{name}_{branch}'] = (np.asarray(gather_to_host(value))
                    if isinstance(value, jax.Array) else np.asarray(value))
        report.update(status='complete_fixed_snapshot_SOURCE_FROZEN_not_refreshed_or_dynamics',
            native_receipt=result['raw']['native_receipt'],
            microscopic_space=dict(nk=prepared['nk'], nmu=prepared['nmu_logical'],
                                   virtual_bands=prepared['nb'], active_bands=[args.b_lo,args.b_lo+rho.shape[-1]]),
            trace_collision_ry=float(incoming+outgoing), flux_ry=float(flux),
            trace_residual_over_flux=float(abs(incoming+outgoing)/flux) if flux else 0.,
            input_density_bounds=[float(np.linalg.eigvalsh(rho).min()),float(np.linalg.eigvalsh(rho).max())],
            min_injection_mev=float(np.linalg.eigvalsh(injection).min())*RYD_TO_EV*1e3,
            min_blocking_mev=float(np.linalg.eigvalsh(gamma-injection).min())*RYD_TO_EV*1e3,
            min_gamma_mev=float(np.linalg.eigvalsh(gamma).min())*RYD_TO_EV*1e3,
            positivity_claim=False)
        if rank0:
            np.savez(root/'operators.npz', **arrays)
    except Exception:
        report.update(status='refused', error=traceback.format_exc()); rc=1
        say(report['error'], flush=True)
    report['total_wall_s'] = time.perf_counter()-started
    report['active_gpu_hours'] = report['total_wall_s']*4/3600
    save()
    runtime.finalize_process(rc)


if __name__ == '__main__':
    main()
