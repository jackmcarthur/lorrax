"""Bounded P4 device-pencil proof, optionally replaying a frequency-only cache.

This does not certify material W/Sigma accuracy or a driven trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time


def device_stage(c):
    import jax
    import json
    import numpy as np
    from types import SimpleNamespace
    from jax.sharding import PartitionSpec as P
    from common.collectives import gather_to_host
    from gw.ppm_sigma import fresh_shared_pole_capacity
    from gw.shared_pole_constructor import reduce_equilibrium_pencil_device
    from gw.shared_pole_execution import face_matmul

    mesh, put = c['mesh'], c['put']
    host = lambda a: np.asarray(gather_to_host(a))
    mm = face_matmul(mesh)
    meta = SimpleNamespace(nkx=8, nky=8, nkz=1, nk_tot=64, nq_tot=16,
        nspinor=2, n_rmu=4, qgrid=(4, 4, 1), qgrid_response_window='fejer')
    meta = fresh_shared_pole_capacity(meta, mesh)
    capacity = meta.shared_pole_capacity
    capacity.at_phase()
    rng = np.random.default_rng(20261021)
    x = rng.normal(size=(4, 8))+1j*rng.normal(size=(4, 8))
    coupling = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    signature = np.diag([1., 1., -1., -1.])
    g = x.conj().T @ signature @ x
    maximum = 0.
    reports = []
    held = None
    gamma = .037
    for current in range(2):
        energy = np.diag(np.array([.6, 1.2, .8, 1.7])+.13*current)
        output = (1+.05*current)*coupling @ x
        gh, hh, nh = [put(a[None], P(None, 'x', 'y')) for a in
            (g, x.conj().T @ energy @ x, output.conj().T @ output)]
        model = reduce_equilibrium_pencil_device(gh, hh, nh, mesh_xy=mesh,
            max_order=8, rank_tolerance=1e-8, gamma_ry=gamma,
            fixed_spaces=held, capacity=capacity)
        poles = model['poles_ry']
        assert len(poles) == 4
        od = put(output[None], P(None, 'x', 'y'))
        left = host(mm(od, model['output_map']))[0, :, :len(poles)]
        right = host(mm(od, model['input_map']))[0, :, :len(poles)]
        for z in (.31+.04j, -.43+.2j, 1.5+.08j):
            exact = (1+.05*current)**2*coupling @ np.linalg.solve(
                (z+1j*gamma)*signature-energy, coupling.conj().T)
            value = (left/(z-poles)) @ right.conj().T
            error = np.linalg.norm(value-exact)/np.linalg.norm(exact)
            maximum = max(maximum, float(error))
            assert error < 2e-9
        if held is not None:
            assert model['spaces']['right'] is held['right']
        held = model['spaces']
        assert model['receipt']['host_frequency_matrix_bytes'] == 0
        reports.append(model['receipt'])
    result = dict(tiny_current_inputs=2, max_tiny_complex_transfer_error=maximum,
        exact_held_space=True, native_receipts=reports,
        material_W_accuracy_certified=False, material_Sigma_accuracy_certified=False)
    if _PENCIL is not None:
        # This is an explicit replay of an already saved frequency-only
        # forensic cache, not a gather in the production screening path.
        path = Path(_PENCIL).resolve()
        if path.stat().st_size > 256*1024**2:
            raise ValueError('frequency replay exceeds its 256 MiB input bound')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024**2), b''):
                digest.update(block)
        if digest.hexdigest() != _PENCIL_SHA256:
            raise ValueError('frequency replay digest mismatch')
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json']))
            cached_tolerance = float(metadata['rank_tolerance'])
            cached_gamma = float(metadata['gamma_ry'])
            arrays = [put(np.asarray(data[key])[None], P(None, 'x', 'y'))
                      for key in ('G', 'H', 'output_gram')]
        for value in arrays:
            value.block_until_ready()
        cached = []
        for phase in ('cold', 'warm'):
            start = time.monotonic()
            model = reduce_equilibrium_pencil_device(*arrays, mesh_xy=mesh,
                max_order=arrays[0].shape[-1], rank_tolerance=cached_tolerance,
                gamma_ry=cached_gamma, capacity=capacity)
            jax.block_until_ready((model['output_map'], model['input_map']))
            cached.append(dict(phase=phase, wall_seconds=time.monotonic()-start,
                               receipt=model['receipt']))
        result['frequency_only_replay'] = dict(path=str(path), sha256=digest.hexdigest(),
            source_rank_tolerance=cached_tolerance, gamma_ry=cached_gamma, trials=cached,
            scope='runtime and retained-space checks only; no original full-W/Sigma targets')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pencil')
    parser.add_argument('--pencil-sha256')
    args, rest = parser.parse_known_args()
    if bool(args.pencil) != bool(args.pencil_sha256):
        parser.error('--pencil and --pencil-sha256 must be supplied together')
    _PENCIL, _PENCIL_SHA256 = args.pencil, args.pencil_sha256
    sys.argv[1:] = rest
    from verify_complex_box_chi_p4 import main
    raise SystemExit(main(device_stage))
