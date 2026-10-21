"""Bounded cached-q0 production-owner check, never full-W admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--source-identity', required=True)
    parser.add_argument('--constructor-sha256', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--max-order', required=True, type=int)
    parser.add_argument('--max-order-provenance', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-wall-seconds', type=float, default=180.)
    args = parser.parse_args()
    if not 0 < args.max_wall_seconds <= 180 or args.max_order < 1:
        raise ValueError('positive order and wall cap <=180 seconds required')
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    peak = [0]

    def guard():
        while True:
            rows = Path('/proc/self/status').read_text().splitlines()
            rss = next(int(row.split()[1])*1024 for row in rows if row.startswith('VmRSS:'))
            peak[0] = max(peak[0], rss)
            if rss > 4*1024**3 or time.monotonic()-started > args.max_wall_seconds:
                print('CACHED_PRODUCTION_GUARD', rss, time.monotonic()-started, flush=True)
                os._exit(124)
            time.sleep(.2)

    threading.Thread(target=guard, daemon=True).start()
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    source = Path(args.source_root).resolve()
    constructor = source/'src/gw/shared_pole_constructor.py'
    if hashlib.sha256(constructor.read_bytes()).hexdigest() != args.constructor_sha256:
        raise ValueError('exact reviewed production constructor SHA mismatch')
    sys.path.insert(0, str(source/'src'))
    from runtime.source_closure import ensure_source_closure
    ensure_source_closure()
    import numpy as np
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    from gw.shared_pole_constructor import reduce_general_pencil, EQUILIBRIUM_METRIC_SOLVER

    def mark(label):
        print('CACHED_PRODUCTION', label, f'{time.monotonic()-started:.3f}s', flush=True)

    path = Path(args.input)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(part)
    if digest.hexdigest() != args.input_sha256:
        raise ValueError('capture SHA mismatch')
    with np.load(path, allow_pickle=False) as record:
        metadata = json.loads(str(record['metadata_json']))
        g, h, output_gram = (np.asarray(record[key], complex) for key in ('G', 'H', 'output_gram'))
        units = np.asarray(record['state_units'], float)
    if (metadata['q'] != 0 or not metadata['equilibrium']['eligible']
            or metadata['representation'] != 'equilibrium-definite-same-state'
            or metadata['state_selection'] != 'principal-rrqr-v1'):
        raise ValueError('authenticated captured equilibrium-q0 premise required')
    structure = dict(kind='equilibrium_definite', gamma_ry=float(metadata['gamma_ry']),
                     state_selection=metadata['state_selection'], state_units=units,
                     frequency_scale_ry=float(metadata['frequency_scale_ry']),
                     metric_solver=EQUILIBRIUM_METRIC_SOLVER)
    result = dict(status='CACHED_Q0_PRODUCTION_CHECK_STARTED_NOT_W_ADMISSION',
                  source_identity=args.source_identity, source_root=str(source),
                  constructor_sha256=args.constructor_sha256,
                  input_sha256=digest.hexdigest(), input_path=str(path), q=0,
                  original_shape=list(g.shape), max_order=args.max_order,
                  max_order_provenance=args.max_order_provenance,
                  rank_tolerance=float(metadata['rank_tolerance']),
                  metric_solver=structure['metric_solver'],
                  host_workspace_bound_bytes=4*1024**3,
                  full_original_RRQR=True, selected_index_monkeypatch=False,
                  failed_material_owner_unknown=True,
                  no_microscopic_port_payload=True, no_full_W_admission=True,
                  GPU_hours=0, local_dense_analysis=False)
    mark('ORIGINAL-PRODUCTION-OWNER-BEGIN-full-RRQR')
    owner_started = time.monotonic()
    rc = 0
    try:
        model = reduce_general_pencil(g, h, output_gram=output_gram, structure=structure,
                                     max_order=args.max_order,
                                     rank_tolerance=float(metadata['rank_tolerance']),
                                     max_workspace_bytes=4*1024**3, fixed_spaces=None)
        result.update(status='CACHED_Q0_PRODUCTION_REDUCER_PASS_NOT_W_ADMISSION',
                      reducer_receipt=model['receipt'])
        poles = np.asarray(model['poles_ry'])
        result['pole_diagnostics'] = dict(count=int(poles.size),
            finite=bool(np.isfinite(poles).all()),
            max_imaginary_ry=float(np.max(poles.imag)) if poles.size else None,
            min_real_ry=float(np.min(poles.real)) if poles.size else None,
            max_real_ry=float(np.max(poles.real)) if poles.size else None,
            max_lag_imaginary_defect_ry=float(np.max(abs(poles.imag+structure['gamma_ry']))) if poles.size else 0.)
        result['map_shapes'] = {key:list(np.shape(model[key])) for key in ('input_map', 'output_map')}
        result['scope'] = ('Exact cached q0 frequency-pencil production reduction only; no full-port '
                           'held W, other-parent, collision, positivity or trajectory admission')
    except Exception as error:
        rc = 1
        traceback.print_exc()
        result.update(status='CACHED_Q0_PRODUCTION_REDUCER_REFUSED',
                      error_type=type(error).__name__, error=str(error),
                      scope='Original production-owner refusal preserved; no retry or gate waiver')
    result.update(owner_seconds=time.monotonic()-owner_started,
                  seconds=time.monotonic()-started, peak_rss_bytes=peak[0],
                  BLAS_threads=1, max_wall_seconds=args.max_wall_seconds, returncode=rc)

    def encode(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.size > 4096:
                raise ValueError('unexpected large array in scalar reducer receipt')
            return value.tolist()
        raise TypeError(type(value).__name__)

    with output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False, default=encode)
        handle.write('\n')
    mark('COMPLETE-rc-'+str(rc))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
