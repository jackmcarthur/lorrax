"""Bounded native P4 complex-contour chi proof; not a material timing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback


def frequency_literal(psi, energy, left, right, coordinates, grid, sites, gamma):
    """Independent two resolvents, with no quadrature or production FFT."""
    import numpy as np
    lm = np.einsum('kasr,kab->kbsr', psi, left)
    rm = np.einsum('kasr,kab->kbsr', psi, right)
    result = []
    for q in coordinates:
        partner = np.ravel_multi_index(((coordinates-q) % grid).T, grid)
        vertex = np.einsum('kbsr,kdsr->kbdr', psi, psi[partner].conj())
        forward = np.einsum('kbsr,kdsr->kbdr', lm, rm[partner].conj())
        reverse = np.einsum('kbsr,kdsr->kbdr', rm, lm[partner].conj())
        gap = energy[:, :, None]-energy[partner, None, :]
        result.append(np.array([np.einsum('kbdm,kbdn,kbd->mn',
            forward-reverse, vertex.conj(), 1/(z+1j*gamma-gap))
            /np.sqrt(len(psi)) for z in sites]))
    return np.asarray(result)


def response_stage(c):
    import numpy as np
    from types import SimpleNamespace
    from jax.sharding import PartitionSpec as P
    from common.collectives import gather_to_host
    from gw.greens_function_kernel import density_left_endpoint
    from gw.ppm_sigma import fresh_shared_pole_capacity
    from gw.response_bank import _bank_execution, full_density_sample_rules
    from gw.w_isdf import full_density_response_stream
    from verify_common_coherent_chi_p4 import bank_config
    from verify_fejer_p4_38dbea1 import literal_crop

    mesh, grid, coarse = (c[k] for k in ('mesh', 'grid', 'coarse'))
    nk, nb, mu, ns = (c[k] for k in ('nk', 'nb', 'mu', 'ns'))
    sites, gamma = np.array([-.17+.12j, .42+.18j]), .03
    center = np.zeros((nk, nb, nb), complex); center[:, 1:3, 1:3] = np.eye(2)
    filled = np.zeros_like(center); filled[:, 0, 0] = 1
    empty = np.zeros_like(center); empty[:, 3, 3] = 1
    rng = np.random.default_rng(423)
    u, _ = np.linalg.qr(rng.normal(size=(nk, 2, 2))+1j*rng.normal(size=(nk, 2, 2)))
    rho = filled.copy()
    rho[:, 1:3, 1:3] = (u*np.array([.21, .79])[None, None])@u.conj().swapaxes(-1, -2)
    endpoint = density_left_endpoint(mesh, c['pm'].shape)
    host = lambda a: np.asarray(gather_to_host(a))
    maximum, calls, node_calls = 0., 0, 0
    executions, records = [], []
    for qgrid, window in ((grid, 'sharp'), (coarse, 'fejer')):
        nq = int(np.prod(qgrid))
        meta = SimpleNamespace(nkx=8, nky=8, nkz=1, nk_tot=nk, nq_tot=nq,
            nspinor=ns, n_rmu=mu, qgrid=qgrid, qgrid_response_window=window)
        meta = fresh_shared_pole_capacity(meta, mesh)
        meta.shared_pole_capacity.at_phase()
        row = meta.shared_pole_capacity.reserve('complex_box_software_carry',
            resident_bytes_per_rank=16*nq*len(sites)*mu*mu//mesh.size,
            workspace_bytes_per_rank=0)
        meta.shared_pole_capacity.live_stages = (row['stage'],)
        execution = dict(seconds={}, memory=[], compiled=[])
        execute = _bank_execution(meta, mesh, execution, bank_config())
        for current in range(2):
            energy = c['energy']+.03*current*np.sin(np.arange(nk*nb).reshape(nk, nb))
            density = rho.copy()
            if current:
                density[:, 1:3, 1:3] = np.eye(2)-density[:, 1:3, 1:3]
            af, au = density*center, center-density*center
            pairs = dict(full=(np.eye(nb)[None]-density, density), FF=(empty, filled),
                         FA=(au, filled), AF=(empty, af), AA=(au, af))
            for name, (left, right) in pairs.items():
                support = [np.any(a != 0, axis=1) for a in (left, right)]
                bounds = tuple(v for mask in support
                    for v in (energy[mask].min(), energy[mask].max()))
                bands = np.arange(nb)[None]
                ranges = np.array([[np.min(np.where(mask, bands, nb), axis=1),
                    np.max(np.where(mask, bands+1, 0), axis=1)] for mask in support])
                rules = full_density_sample_rules(energy, sites, gamma_ry=gamma,
                    interval_ry=(bounds[0]-bounds[3], bounds[1]-bounds[2]),
                    paired=True, values_only=True)
                l = endpoint(c['pm'], c['put'](left, P()))
                r = endpoint(c['pm'], c['put'](right, P()))
                actual = None
                for rule in rules:
                    stream = full_density_response_stream(mesh, grid, (nk, nb, mu, ns),
                        q_ids=range(nq), n_outputs=len(sites), k_unfold_plan=c['plan'],
                        execute=execute, qgrid=qgrid, response_window=window,
                        rule_reference_ry=rule['reference_ry'], pair_bounds_ry=bounds,
                        pair_band_ranges=ranges)
                    value = host(stream(c['put'](rule['times'], P()), c['put'](rule['weights'], P()),
                        l, c['pn'], r, c['put'](energy, P())))
                    actual = value if actual is None else actual+value
                    node_calls += rule['active_count']
                expected = frequency_literal(c['psi'], energy, left, right,
                    c['coords'], grid, sites, gamma)
                if qgrid != grid:
                    expected = literal_crop(expected, grid, qgrid)
                np.testing.assert_allclose(actual, expected, rtol=4e-7, atol=3e-10, err_msg=name)
                maximum = max(maximum, float(np.max(abs(actual-expected)))); calls += 1
                records.append(dict(current=current, block=name, qgrid=list(qgrid),
                    nodes=sum(v['active_count'] for v in rules), panels=len(rules),
                    node_carrier_slots=sum(len(v['times']) for v in rules),
                    max_bound=max(v['sampled_error'] for v in rules),
                    max_coefficient_mass=max(v['coefficient_mass'] for v in rules)))
        assert execution['compiled'] and execution['memory']
        executions.append(execution)
    return dict(max_literal_error=maximum, current_inputs=2, paired_calls=calls,
        total_node_calls=node_calls, typed_identity_parent=True, native_mode11=True,
        panels=['full', 'FF', 'FA', 'AF', 'AA'], rules=records, execution=executions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-wall-seconds', type=float, default=180.)
    args = parser.parse_args()
    if not 0 < args.max_wall_seconds <= 180 or os.environ.get('LEG_HEAD') != args.source_revision:
        raise ValueError('exact source revision and <=180 second cap required')
    watchdog = threading.Timer(args.max_wall_seconds, lambda: os._exit(124))
    watchdog.daemon = True; watchdog.start()
    import gw.gw_jax as driver
    import runtime
    import jax
    import gw.w_isdf, gw.response_bank
    from common.collectives import rank0_transaction
    from ffi.fft import kconv_backend
    import verify_fejer_p4_38dbea1 as fixture_owner
    import verify_common_coherent_chi_p4 as common_owner
    root = Path(args.source_root).resolve()
    modules = (driver, gw.w_isdf, gw.response_bank, fixture_owner, common_owner)
    origins = {m.__name__: str(Path(m.__file__).resolve()) for m in modules}
    if any(root not in Path(p).parents for p in origins.values()):
        raise ValueError('source module origin mismatch')
    if (jax.process_count() != 4 or jax.local_device_count() != 1
            or driver.RUNTIME.mesh.devices.shape != (2, 2)
            or any(d.platform != 'gpu' for d in jax.devices()) or not jax.config.x64_enabled
            or kconv_backend(driver.RUNTIME.mesh, (8, 8, 1)) != 'mathdx'):
        raise ValueError('requires native CUDA P4 2x2 FP64; no reference override')
    out = Path(args.out).resolve()
    rank0_transaction(out, stage='complex_box/new', write=lambda: out.mkdir(parents=True, exist_ok=False))
    started = time.monotonic()
    try:
        result = response_stage(fixture_owner.fixture(driver.RUNTIME.mesh))
    except BaseException:
        print(f'Complex box rank {jax.process_index()} failed:\n'+traceback.format_exc(), flush=True)
        raise
    report = dict(status='PASS_SOFTWARE_ONLY_NOT_MATERIAL', source_revision=args.source_revision,
        module_origins=origins, harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_archive_sha256=os.environ.get('LEG_SOURCE_ARCHIVE_SHA256'),
        elapsed_seconds=time.monotonic()-started, max_wall_seconds=args.max_wall_seconds,
        active_GPUh_cap=4*args.max_wall_seconds/3600, result=result)
    def write():
        tmp = out/'receipt.tmp.json'; tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        os.replace(tmp, out/'receipt.json')
    rank0_transaction(out, stage='complex_box/pass', write=write)
    watchdog.cancel(); runtime.finalize_process(0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
