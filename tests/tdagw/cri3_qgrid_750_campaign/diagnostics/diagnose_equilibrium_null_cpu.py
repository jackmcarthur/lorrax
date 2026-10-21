"""Bounded CPU-only q0 captured-pencil diagnostics; NEVER changes a model.

The failed canonical owner in the material batch is UNKNOWN. A q0 result
does not establish the failure of any other parent. No GPU/device work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--input-sha256', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-wall-seconds', type=float, default=180.)
    a = p.parse_args()
    if not 0 < a.max_wall_seconds <= 180:
        raise ValueError('CPU diagnostic cap must be <=180 seconds')
    output = Path(a.out)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic(); peak = [0]
    def guard():
        while True:
            rss = 0
            for row in Path('/proc/self/status').read_text().splitlines():
                if row.startswith('VmRSS:'):
                    rss = int(row.split()[1])*1024
            peak[0] = max(peak[0], rss)
            if rss > 4*1024**3 or time.monotonic()-started > a.max_wall_seconds:
                print('CPU_DIAGNOSTIC_GUARD', rss, time.monotonic()-started, flush=True)
                os._exit(124)
            time.sleep(.2)
    threading.Thread(target=guard, daemon=True).start()
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    source = Path(a.source_root).resolve()
    sys.path.insert(0, str(source/'src'))
    from runtime.source_closure import ensure_source_closure
    ensure_source_closure()
    import numpy as np
    import scipy.linalg as la
    from threadpoolctl import threadpool_limits
    from gw.shared_pole_constructor import _equilibrium_state_subset, general_pencil_workspace_bytes
    threadpool_limits(limits=1)
    def mark(name):
        print('CPU_DIAGNOSTIC', name, f'{time.monotonic()-started:.3f}s', flush=True)
    path = Path(a.input)
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(8*1024**2), b''):
            sha.update(part)
    if sha.hexdigest() != a.input_sha256:
        raise ValueError('capture SHA mismatch')
    mark('load')
    with np.load(path, allow_pickle=False) as record:
        metadata = json.loads(str(record['metadata_json']))
        g, h, n = (np.asarray(record[k], complex) for k in ('G', 'H', 'output_gram'))
        units = np.asarray(record['state_units'], float)
    if metadata['q'] != 0 or g.shape != h.shape or n.shape != g.shape or g.shape[0] != g.shape[1]:
        raise ValueError('requires exact q0 square G/H/outputGram capture')
    size = len(g); eps = np.finfo(float).eps; rounding = 128*size*eps
    rtol = float(metadata['rank_tolerance']); scale = float(metadata['frequency_scale_ry'])
    if general_pencil_workspace_bytes(g.shape) > 4*1024**3:
        raise ValueError('owner host workspace exceeds4GiB bound')
    asymmetry = {name: float(la.norm(m-m.conj().T)/max(la.norm(m), np.finfo(float).tiny))
                 for name, m in (('G', g), ('H', h), ('output_gram', n))}
    if max(asymmetry.values()) > rounding:
        raise ValueError('input differs from owner Hermiticity admission')
    g, h, n = ((m+m.conj().T)*.5 for m in (g, h, n))
    mark('exact-owner-RRQR-BEGIN')
    selected, selection = _equilibrium_state_subset(g, h, n, units=units,
        frequency_scale=scale, rank_tolerance=rtol, rounding=rounding)
    mark('exact-owner-RRQR-END')
    hp, gp, op = (m[np.ix_(selected, selected)] for m in (h, g, n))
    full_norm_g, full_norm_n = float(la.norm(g)), float(la.norm(n))
    def norm_scaled(matrix, d):
        total = 0.
        for start in range(0, len(d), 128):
            block = d[start:start+128, None]*matrix[start:start+128]*d[None]
            total += float(la.norm(block))**2
        return np.sqrt(total)
    def metric_case(name, d):
        mark(name+'-BEGIN')
        ds = d[selected]
        metric = ds[:, None]*hp*ds[None]
        ev, u = la.eigh((metric+metric.conj().T)*.5, check_finite=False)
        hs = max(float(np.max(abs(ev))), np.finfo(float).tiny)
        cut = max(rounding, rtol)*hs
        keep = ev > cut; null = ds[:, None]*u[:, ~keep]
        full_g_null = g[:, selected]@null
        output_loss = float(la.norm(null.conj().T@op@null)) if null.shape[1] else 0.
        g_loss = float(la.norm(full_g_null))
        # Compare the SAME null subspace in original Euclidean coordinates.
        # Congruence may rescale ds*u, so its raw loss alone is not invariant.
        null_span = la.qr(null, mode='economic', check_finite=False)[0] if null.shape[1] else null
        span_g_loss = float(la.norm(g[:, selected]@null_span))
        span_output_loss = float(la.norm(null_span.conj().T@op@null_span)) if null.shape[1] else 0.
        threshold_g = (rtol+rounding)*max(full_norm_g, np.finfo(float).tiny)
        threshold_o = (rtol**2+rounding)*max(full_norm_n, np.finfo(float).tiny)
        eigen_residual = float(la.norm(metric@u-u*ev[None]))
        orthogonality = float(la.norm(u.conj().T@u-np.eye(len(ev))))
        # Diagnostic residual/roundoff scale, NOT an admitted operator-error
        # certificate and NOT a new eigenvalue/drop threshold.
        backward_scale = eigen_residual+128*len(ev)*eps*float(la.norm(metric))
        whitening = None
        if keep.any():
            z = ds[:, None]*u[:, keep]/np.sqrt(ev[keep])[None]
            whitening = float(la.norm(z.conj().T@hp@z-np.eye(int(keep.sum()))))
        result = dict(name=name, selected_count=len(ev), H_min=float(ev[0]), H_max=float(ev[-1]),
            H_scale=hs, owner_cut=cut, owner_rank_above_cut=int(keep.sum()), null_count=int((~keep).sum()),
            H_roundoff_cut=rounding*hs, positive_above_roundoff_count=int(np.count_nonzero(ev>rounding*hs)),
            eigen_residual_fro=eigen_residual, eigenvector_orthogonality_fro=orthogonality,
            diagnostic_backward_scale=backward_scale,
            H_min_over_backward_scale=float(ev[0]/max(backward_scale, np.finfo(float).tiny)),
            positive_above_backward_scale_count=int(np.count_nonzero(ev>backward_scale)),
            first_eigenvalues=ev[:20].tolist(), last_eigenvalues=ev[-5:].tolist(),
            owner_full_G_null_norm=g_loss, owner_full_G_null_threshold=threshold_g,
            owner_full_G_null_relative=g_loss/max(full_norm_g, np.finfo(float).tiny),
            owner_output_null_loss=output_loss, owner_output_null_threshold=threshold_o,
            owner_output_null_relative=output_loss/max(full_norm_n, np.finfo(float).tiny),
            original_normalized_nullspan_G_norm=span_g_loss,
            original_normalized_nullspan_G_relative=span_g_loss/max(full_norm_g,np.finfo(float).tiny),
            original_normalized_nullspan_output_loss=span_output_loss,
            original_normalized_nullspan_output_relative=span_output_loss/max(full_norm_n,np.finfo(float).tiny),
            original_normalized_nullspan_exceeds_owner_loss_allowances=bool(
                span_g_loss>threshold_g or span_output_loss>threshold_o),
            selected_only_G_null_relative=float(la.norm(gp@null))/max(float(la.norm(gp)), np.finfo(float).tiny),
            congruent_full_G_null_relative=float(la.norm(d[:, None]*full_g_null))/max(norm_scaled(g,d),np.finfo(float).tiny),
            congruent_output_null_relative=output_loss/max(norm_scaled(n,d),np.finfo(float).tiny),
            retained_physical_whitening_error=whitening,
            scale_min=float(ds.min()),scale_max=float(ds.max()),scale_condition=float(ds.max()/ds.min()),
            predicted_metric_negative=bool(ev[0]<-rounding*hs),
            predicted_live_null_refusal=bool(g_loss>threshold_g or output_loss>threshold_o),
            interpretation='DIAGNOSTIC ONLY: no source arrays, cuts, model, or admission changed')
        mark(name+'-END')
        return result
    cases = [metric_case('actual_owner_unscaled_principal', np.ones(size)),
             metric_case('same_finite_infinity_unit_congruence', units)]
    diagonal = h.diagonal().real
    if np.all(diagonal[selected] > 0):
        diagonal_scale = np.ones(size)
        diagonal_scale[selected] = 1/np.sqrt(diagonal[selected])
        cases.append(metric_case('same_positive_selected_diagonal_H_congruence', diagonal_scale))
    else:
        cases.append(dict(name='same_positive_selected_diagonal_H_congruence',
            skipped='nonpositive selected actual H diagonal'))
    result = dict(status='CPU_Q0_DIAGNOSTIC_ONLY_NOT_MODEL_ADMISSION', q=0,
        failed_material_owner_unknown=True, failed_batch=[0,1,2,4,5,6],
        input_path=str(path),input_sha256=sha.hexdigest(),source_root=str(source),
        constructor_sha256=hashlib.sha256((source/'src/gw/shared_pole_constructor.py').read_bytes()).hexdigest(),
        shape=list(g.shape),rank_tolerance=rtol,rounding=rounding,frequency_scale_ry=scale,
        unit_min=float(units.min()),unit_max=float(units.max()),
        Hermiticity_relative=asymmetry, selection=selection,cases=cases,
        seconds=time.monotonic()-started,peak_rss_bytes=peak[0],blas_threads=1,
        max_wall_seconds=a.max_wall_seconds,rss_limit_bytes=4*1024**3,
        scope='q0 capture may not be the failed parent; no tolerance, PSD, pole or physical-data modification')
    result['normalization_note'] = ('Raw owner-formula losses and original-normalized QR null-span losses '
        'are both reported. Selected-diagonal scaling is one outside selected coordinates; '
        'congruent full-norm diagnostics therefore use that explicitly stated extension.')
    with output.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    mark('COMPLETE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
