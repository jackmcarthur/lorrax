"""Bounded remote-CPU all-selected H-congruence diagnostic, NOT admission.

q0 may not be the failed material owner. No rank/eigenvalue truncation,
PSD repair, tolerance change, microscopic port data, or GPU execution.
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
        raise ValueError('remote CPU wall cap must be <=180 seconds')
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
                print('CHOLESKY_DIAGNOSTIC_GUARD', rss, time.monotonic()-started, flush=True)
                os._exit(124)
            time.sleep(.2)
    threading.Thread(target=guard, daemon=True).start()
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    source = Path(a.source_root).resolve()
    sys.path.insert(0, str(source/'src'))
    from runtime.source_closure import ensure_source_closure
    ensure_source_closure()
    import numpy as np
    import scipy.linalg as la
    from threadpoolctl import threadpool_limits
    from gw.shared_pole_constructor import _equilibrium_state_subset
    threadpool_limits(limits=1)
    def mark(name):
        print('CHOLESKY_DIAGNOSTIC', name, f'{time.monotonic()-started:.3f}s', flush=True)
    def gamma(k, eps):
        x = np.longdouble(k)*eps
        if x >= 1:
            raise ValueError('roundoff envelope outside finite gamma_k regime')
        return x/(1-x)
    ext = np.finfo(np.longdouble); fp = np.finfo(float)
    if ext.nmant <= fp.nmant:
        raise ValueError('extended residual products require more precision than binary64')
    def norm_ext(x):
        # Upward error envelope for rounded modulus-square/sum/sqrt.
        v = np.asarray(x, np.clongdouble)
        total = np.sum(v.real*v.real+v.imag*v.imag, dtype=np.longdouble)
        return np.sqrt(total)/(1-gamma(8*v.size+8, ext.eps))
    def extended_product(x, y, label):
        out = np.empty((x.shape[0],y.shape[1]), np.clongdouble)
        for start in range(0,len(x),64):
            out[start:start+64] = x[start:start+64]@y
            mark(label+f'-rows-{min(start+64,len(x))}')
        return out
    path = Path(a.input); digest = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(part)
    if digest.hexdigest() != a.input_sha256:
        raise ValueError('capture SHA mismatch')
    with np.load(path, allow_pickle=False) as record:
        metadata = json.loads(str(record['metadata_json']))
        g,h,n = (np.asarray(record[k],complex) for k in ('G','H','output_gram'))
        units = np.asarray(record['state_units'],float)
        sites = np.asarray(record['sites_ry'],complex)
    if metadata['q'] != 0 or g.shape != h.shape or n.shape != g.shape:
        raise ValueError('exact q0 square capture required')
    size = len(g); rounding = 128*size*fp.eps
    asymmetry = {name:float(la.norm(m-m.conj().T)/max(la.norm(m),fp.tiny))
        for name,m in (('G',g),('H',h),('output_gram',n))}
    if max(asymmetry.values()) > rounding:
        raise ValueError('owner Hermiticity premise fails')
    g,h,n = ((m+m.conj().T)*.5 for m in (g,h,n))
    mark('exact-owner-RRQR-BEGIN')
    selected, selection = _equilibrium_state_subset(g,h,n,units=units,
        frequency_scale=float(metadata['frequency_scale_ry']),
        rank_tolerance=float(metadata['rank_tolerance']),rounding=rounding)
    mark('exact-owner-RRQR-END')
    hp,gp,op = (m[np.ix_(selected,selected)] for m in (h,g,n))
    if np.any(hp.diagonal().real <= 0):
        raise ValueError('positive selected H diagonal required; no floor')
    d = 1/np.sqrt(hp.diagonal().real); count=len(selected)
    dx = d.astype(np.longdouble)
    # Exact target is the common diagonal congruence of owner-normalized H.
    # Extended arithmetic residual includes its own operation envelope.
    hx = dx[:,None]*np.asarray(hp,np.clongdouble)*dx[None]
    hs = np.asarray(hx,np.complex128)
    hs = (hs+hs.conj().T)*.5
    mark('Cholesky-BEGIN')
    lower = la.cholesky(hs,lower=True,check_finite=False)
    inverse = la.solve_triangular(lower,np.eye(count),lower=True,check_finite=False)
    mark('Cholesky-END')
    lx = np.asarray(lower,np.clongdouble); bx=np.asarray(inverse,np.clongdouble)
    ll = extended_product(lx,lx.conj().T,'extended-LLH')
    bl = extended_product(bx,lx,'extended-BL')
    err = hx-ll; inverse_err=bl-np.eye(count,dtype=np.clongdouble)
    def product_majorant_fro(x,y):
        # Nonnegative binary64 product, inflated for its summation roundoff.
        majorant = np.abs(x)@np.abs(y)
        return norm_ext(majorant)/(1-gamma(4*x.shape[1]+8,fp.eps))
    h_round = gamma(8,ext.eps)*norm_ext(hx)
    factor_round = gamma(8*count+8,ext.eps)*product_majorant_fro(lower,lower.conj().T)
    factor_round += gamma(8,ext.eps)*(norm_ext(hx)+norm_ext(ll))
    delta = norm_ext(err)+factor_round+h_round
    inverse_round = gamma(8*count+8,ext.eps)*product_majorant_fro(inverse,lower)
    inverse_round += gamma(8,ext.eps)*(norm_ext(bl)+np.sqrt(np.longdouble(count)))
    r = norm_ext(inverse_err)+inverse_round
    bnorm = norm_ext(bx)
    singular_lower = max(np.longdouble(0),1-r)/bnorm
    positive_lower = singular_lower**2-delta
    spd = dict(target='D H_selected D of exact recorded owner-normalized H entries',
        factor_residual_fro=float(norm_ext(err)),factor_residual_upper=float(delta),
        inverse_BL_residual_fro=float(norm_ext(inverse_err)),inverse_BL_residual_upper=float(r),
        inverse_fro_upper=float(bnorm),L_singular_min_lower=float(singular_lower),
        H_scaled_eigen_min_lower=float(positive_lower),
        certified_computed_H_positive=bool(r<1 and positive_lower>0),
        scope='floating-operation residual envelope; NOT physical sample-error or W certificate')
    mark('SPD-WITNESS-END')
    z = d[:,None]*inverse.conj().T
    gs = d[:,None]*gp*d[None]
    metric_raw = inverse@gs@inverse.conj().T
    metric = (metric_raw+metric_raw.conj().T)*.5
    mu,vectors = la.eigh(metric,check_finite=False)
    physical = z@vectors
    original_white = z.conj().T@hp@z
    balanced_white = inverse@hs@inverse.conj().T
    generalized = gp@physical-(hp@physical)*mu[None]
    generalized_den = max(la.norm(gp@physical)+la.norm((hp@physical)*mu[None]),fp.tiny)
    full_generalized = g[:,selected]@physical-(h[:,selected]@physical)*mu[None]
    full_den = max(la.norm(g[:,selected]@physical)+la.norm((h[:,selected]@physical)*mu[None]),fp.tiny)
    mu_cut = rounding*max(float(np.max(abs(mu))),fp.tiny)
    finite = abs(mu)>mu_cut
    null_mode = physical[:,~finite]
    null_output = float(la.norm(null_mode.conj().T@op@null_mode)) if null_mode.shape[1] else 0.
    metrics = dict(all_selected_states=count,rank_tolerance_H_cut_applied=False,
        original_physical_whitening_fro=float(la.norm(original_white-np.eye(count))),
        original_whitening_passes_existing_1e_9=bool(la.norm(original_white-np.eye(count))<=1e-9),
        balanced_whitening_fro=float(la.norm(balanced_white-np.eye(count))),
        G_metric_relative_antihermiticity=float(la.norm(metric_raw-metric_raw.conj().T)/max(la.norm(metric_raw),fp.tiny)),
        selected_original_generalized_residual_relative=float(la.norm(generalized)/generalized_den),
        full_original_Galerkin_residual_relative=float(la.norm(full_generalized)/full_den),
        mu_min_abs=float(np.min(abs(mu))),mu_max_abs=float(np.max(abs(mu))),
        current_owner_mu_cut=mu_cut,unresolved_mu_count=int(np.count_nonzero(~finite)),
        unresolved_mu_original_output_loss=null_output,
        scope='metrics measured, no failed check hidden or changed')
    transfer=[]
    if np.all(finite):
        modes = physical/np.sqrt(abs(mu))[None]
        signed = modes*np.sign(mu)[None]
        nev,nu = la.eigh(op,check_finite=False)
        resolved = nev>rounding*max(float(np.max(abs(nev))),fp.tiny)
        output_factor = np.sqrt(nev[resolved])[:,None]*nu[:,resolved].conj().T
        for zeta in sites[[0,len(sites)//2,-1]]+1j*float(metadata['gamma_ry']):
            direct = la.solve(zeta*gp-hp,np.eye(count),assume_a='gen',check_finite=False)
            modal = (modes/(zeta-1/mu)[None])@signed.conj().T
            direct_output=output_factor@direct@output_factor.conj().T
            delta_output=output_factor@(modal-direct)@output_factor.conj().T
            transfer.append(dict(zeta_ry=[float(zeta.real),float(zeta.imag)],
                coefficient_relative=float(la.norm(modal-direct)/max(la.norm(direct),fp.tiny)),
                roundoff_resolved_output_gram_relative=float(la.norm(delta_output)/max(la.norm(direct_output),fp.tiny)),
                direct_solve_residual_relative=float(la.norm((zeta*gp-hp)@direct-np.eye(count))/np.sqrt(count)),
                scope='selected-pencil coefficient/roundoff-resolved OGram diagnostic, NOT native held full-W'))
    result=dict(status='CPU_Q0_CHOLESKY_PROTOTYPE_NOT_MODEL_ADMISSION',q=0,
        failed_material_owner_unknown=True,failed_batch=[0,1,2,4,5,6],
        input_sha256=digest.hexdigest(),input_path=str(path),source_root=str(source),
        constructor_sha256=hashlib.sha256((source/'src/gw/shared_pole_constructor.py').read_bytes()).hexdigest(),
        original_shape=list(g.shape),selection=selection,selected_indices=selected.tolist(),
        Hermiticity_relative=asymmetry,extended_mantissa_bits=int(ext.nmant),
        binary64_mantissa_bits=int(fp.nmant),extended_epsilon=float(ext.eps),
        SPD_witness=spd,metrics=metrics,transfer_checks=transfer,
        seconds=time.monotonic()-started,peak_rss_bytes=peak[0],blas_threads=1,
        max_wall_seconds=a.max_wall_seconds,rss_limit_bytes=4*1024**3,
        scope='all selected states retained; no pole clipping, H floor, tolerance waiver, production mutation or admission')
    with output.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
    mark('COMPLETE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
