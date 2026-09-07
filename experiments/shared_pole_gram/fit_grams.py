"""Host-only fitting of Run306 small Grams; no full matrices or HDF5 reads.

Training rows are [0:Nt,Nt+Nh:2*Nt+Nh] in each 2*(Nt+Nh) channel Gram.
Saved row maps act on unweighted [H_train; A_train], in this order.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

# The campaign's existing measure.py uses sibling imports. Pin this directory
# for both direct-script and python -m invocation without importing JAX.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import varpro
from measure import spectrum


ORDER = Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/152_shared_pole_push_2026-09-07/exchange/order/weight.npz')
ORDER_SHA = '27eb75153dc49849e6c5f3d27cbff24514f7ff28a0f705bc864be34c1b82b0f2'
EV = 13.605693122994


def train_indices(bank):
    """Indices of unweighted Htrain then Atrain in a full train/held Gram."""
    nt, nh = len(bank['z']), len(bank['zh'])
    return np.r_[np.arange(nt), np.arange(nt)+nt+nh]


def sha(path):
    """SHA256 of one explicitly named file."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def authenticated_weight():
    """Read authenticated ORDER frequency weights, never regenerate them."""
    if not ORDER.is_file():
        raise FileNotFoundError(f'ORDER weight missing: {ORDER}')
    if sha(ORDER) != ORDER_SHA:
        raise ValueError(f'ORDER weight hash mismatch: {ORDER}')
    receipt = json.loads(ORDER.with_name('receipt.json').read_text())
    if receipt.get('weight_sha256') != ORDER_SHA:
        raise ValueError('ORDER receipt does not authenticate the pinned weight')
    with np.load(ORDER, allow_pickle=False) as bank:
        omega = bank['omega_ev'].copy()
        weight = bank['weight_ev_minus2'].copy()
    if np.any(np.diff(omega) <= 0) or np.any(weight < 0):
        raise ValueError('ORDER grid must increase and weights must be nonnegative')
    return omega, weight, receipt


def line_weights(z, omega, weight):
    """Trapezoid times ORDER weight; each distinct height has equal total loss.

    Frequency is Re(z) in eV. Equal-frequency duplicates on one line split
    that node's trapezoid weight, avoiding accidental line double counting.
    """
    z = np.asarray(z)
    result = np.zeros(z.size)
    heights = np.unique(np.round(z.imag * EV, 9))
    for height in heights:
        indices = np.flatnonzero(np.round(z.imag * EV, 9) == height)
        x, inverse, counts = np.unique(z[indices].real * EV, return_inverse=True, return_counts=True)
        trapezoid = np.ones(x.size)
        if x.size > 1:
            trapezoid[0], trapezoid[-1] = (x[1]-x[0])/2, (x[-1]-x[-2])/2
            trapezoid[1:-1] = (x[2:]-x[:-2])/2
        sampled = np.interp(np.abs(x), omega, weight, left=0., right=0.)
        node_loss = trapezoid * sampled
        if node_loss.sum() <= 0:
            raise ValueError(f'No positive authenticated weight on height {height} eV')
        result[indices] = node_loss[inverse] / counts[inverse] / node_loss.sum() / heights.size
    return result


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def write_json(path, data):
    """Exclusive creation: completed output can never be overwritten."""
    with Path(path).open('x') as stream:
        json.dump(_json_value(data), stream, indent=2, allow_nan=False)
        stream.write('\n')


def errors_from_gram(z_all, indices, row_map, poles, gram, weights):
    """Relative Frobenius error from Tr(L G L^T), with no n-dependent arrays.

    L consists of real and imaginary prediction rows minus the corresponding
    stored H/A rows. Linear Coulomb congruence commutes with this same map,
    so either whitened or physical channel Gram can be scored directly.
    """
    nall = len(z_all)
    nt = row_map.shape[1]//2
    if row_map.shape[1] != 2*nt or gram.shape != (2*nall,2*nall):
        raise ValueError('Residual row-map/Gram shape mismatch')
    selection = np.eye(2*nall)[np.r_[np.arange(nt),np.arange(nt)+nall]]
    phi = varpro.basis(z_all[indices], poles)
    prediction = np.vstack((phi.real, phi.imag)) @ row_map @ selection
    reference = np.eye(2*nall)[np.r_[indices, indices+nall]]
    residual = prediction - reference
    row_error = np.einsum('ij,jk,ik->i', residual, gram, residual)
    row_norm = np.diag(gram)[np.r_[indices, indices+nall]]
    w = np.r_[weights, weights]
    denominator = float(w @ row_norm)
    numerator = float(w @ row_error)
    if denominator <= 0:
        raise ValueError('Zero or negative reference norm in error diagnostic')
    if numerator < -1e-10 * denominator:
        raise ValueError('Materially negative Gram residual norm')
    return {'relative_frobenius': float(np.sqrt(max(numerator, 0.) / denominator)),
            'squared_error': numerator, 'squared_reference': denominator,
            'negative_roundoff_clamped': bool(numerator < 0),
            'rows': int(len(indices))}


def line_errors(bank, fitted, omega, weight):
    """Report each train/held height, in whitened and physical coordinates."""
    z_all = np.r_[bank['z'], bank['zh']]
    result = {}
    nt = len(bank['z'])
    for split, base in [('train', np.arange(nt)), ('held', np.arange(nt,len(z_all)))]:
        for height in np.unique(np.round(z_all[base].imag * EV, 9)):
            indices = base[np.round(z_all[base].imag * EV, 9) == height]
            loss = line_weights(z_all[indices], omega, weight)
            label = f'{split}_height_{height:.9f}_ev'
            result[label] = {}
            for coordinate, key in [('white', 'channel_gram'), ('physical', 'physical_channel_gram')]:
                result[label][coordinate] = {
                    mode: errors_from_gram(z_all, indices, fitted['row_map'], fitted['poles_ry'],
                                           bank[key], selected)
                    for mode, selected in [('uniform', np.ones(indices.size)), ('order_trapezoid', loss)]}
    return result


def load_gram(path):
    """Refuse missing or mismatched banks before starting any fit."""
    if not path.is_file():
        raise FileNotFoundError(f'Gram input missing (production may still be running): {path}')
    with np.load(path, allow_pickle=False) as data:
        bank = {key: data[key].copy() for key in data.files}
    for key in ('z','zh'):
        if key not in bank or bank[key].ndim != 1 or not bank[key].size:
            raise ValueError(f'{path}: missing/empty/nonvector {key}')
    nt, nh = len(bank['z']), len(bank['zh'])
    nc = 2*(nt+nh)
    shapes = {'z': (nt,), 'zh': (nh,), 'channel_gram': (nc,nc),
              'physical_channel_gram': (nc,nc), 'complex_gram': (nt,nt), 'sketches': (nt,5)}
    for key, shape in shapes.items():
        if key not in bank or bank[key].shape != shape or not np.all(np.isfinite(bank[key])):
            raise ValueError(f'{path}: invalid {key}, expected finite shape {shape}')
    return bank


def provenance():
    """Record source and Slurm job.step even for small host computations."""
    root = Path(__file__).resolve().parents[2]
    return {'job_step': os.getenv('SLURM_JOB_ID', 'login-cpu')+'.'+os.getenv('SLURM_STEP_ID', 'none'),
            'rank': int(os.getenv('SLURM_PROCID', '0')), 'source_tree': str(root),
            'source_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
            'source_sha256': {p.name: sha(p) for p in [Path(__file__), Path(varpro.__file__)]},
            'weight_path': str(ORDER), 'weight_sha256': ORDER_SHA,
            'weight_rule': 'trapezoid times ORDER; training heights normalized equally; held rows excluded',
            'row_map_contract': 'unweighted Htrain[0:Nt], Atrain[Nt+Nh:2*Nt+Nh]; residues in Ry',
            'scope': 'Hermitian-residue damped subcase only; no general complex residues, residue ranks, passivity, or Sigma score'}


def rank_summary(gram_dir, out_dir):
    """Write all-q ORDER-weighted original complex-Gram spectral rank table."""
    for name in ('order_rank_summary.json', 'order_rank_table.md'):
        if (out_dir/name).exists():
            raise FileExistsError(f'Immutable output exists: {out_dir/name}')
    omega, weight, receipt = authenticated_weight()
    records = []
    for q in range(29):
        path = gram_dir / f'q{q:02d}.npz'
        bank = load_gram(path)
        loss = line_weights(bank['z'], omega, weight)
        scopes = {}
        groups = [(f'height_{h:.9f}_ev', np.flatnonzero(np.round(bank['z'].imag*EV,9) == h))
                  for h in np.unique(np.round(bank['z'].imag*EV,9))]
        groups.append(('combined', np.arange(len(bank['z']))))
        for name, indices in groups:
            sw = np.sqrt(loss[indices])
            scopes[name] = spectrum(bank['complex_gram'][np.ix_(indices, indices)] * sw[:,None] * sw[None,:])
        records.append({'q': q, 'input_path': str(path), 'input_sha256': sha(path), 'scopes': scopes})
    out_dir.mkdir(parents=True, exist_ok=True)
    origin = provenance()
    origin['scope'] = 'All 29 q original V-whitened complex Gram, training only; no pole fit or residue-rank claim'
    write_json(out_dir/'order_rank_summary.json', {'provenance': origin, 'order_receipt': receipt, 'records': records})
    lines = ['# ORDER-weighted complex Gram ranks', '',
             'All 29 q, training only. Cutoffs apply to singular amplitudes; no fit acceptance is claimed.', '',
             '|q|scope|rank 1e-2|rank 1e-3|rank 1e-4|', '|---|---|---|---|---|']
    for record in records:
        for name, entry in record['scopes'].items():
            ranks = entry['rank_amplitude']
            lines.append(f"|{record['q']}|{name}|{ranks['0.01']}|{ranks['0.001']}|{ranks['0.0001']}|")
    with (out_dir/'order_rank_table.md').open('x') as stream:
        stream.write('\n'.join(lines)+'\n')


def warm_start(directory, q, p, bank, loss):
    """Authenticate a warm seed and refuse changes to the fixed fit functional.

    Poles alone are seeded; every frequency and width remains an optimizer
    variable. Training/held grids, weights, p, and solver implementation must
    match. This verifies a fixed functional, not physical SC convergence.
    """
    path = directory/f'q{q:02d}_p{p:02d}.npz'
    record_path = path.with_suffix('.json')
    record = json.loads(record_path.read_text())
    if record['status'] != 'FIT_COMPLETE' or record['q'] != q or record['p'] != p:
        raise ValueError(f'Warm model is not a completed matching q/p: {record_path}')
    if sha(path) != record['export_sha256']:
        raise ValueError(f'Warm export hash mismatch: {path}')
    if record['provenance']['source_sha256']['varpro.py'] != sha(varpro.__file__):
        raise ValueError('Warm-start solver changed; cannot claim identical functional')
    if record['provenance']['weight_sha256'] != ORDER_SHA:
        raise ValueError('Warm-start ORDER functional changed')
    old_gram_path = Path(record['input_path'])
    if sha(old_gram_path) != record['input_sha256']:
        raise ValueError(f'Warm input Gram hash mismatch: {old_gram_path}')
    old_bank = load_gram(old_gram_path)
    with np.load(path, allow_pickle=False) as old:
        if (not np.array_equal(old['z_train_ry'], bank['z']) or
                not np.array_equal(old['weights_loss'], loss) or
                not np.array_equal(old_bank['zh'], bank['zh']) or
                old['poles_ry'].shape != (p,)):
            raise ValueError('Warm-start grid, weights, held grid, or p changed; fixed functional refused')
        poles = old['poles_ry'].copy()
    return poles, {'path': str(path), 'sha256': sha(path), 'receipt_path': str(record_path),
                   'receipt_sha256': sha(record_path), 'grid_weights_solver_match': True,
                   'pole_variables': 'all frequencies and widths remain free'}


def main(gram_dir, out_dir, check=False, warm_dir=None):
    """Fit q slots distributed by SLURM_PROCID stride SLURM_NTASKS."""
    rank, tasks = int(os.getenv('SLURM_PROCID', '0')), int(os.getenv('SLURM_NTASKS', '1'))
    if tasks < 1 or not 0 <= rank < tasks:
        raise ValueError('Invalid Slurm rank geometry')
    omega, weight, receipt = authenticated_weight()
    origin = provenance()
    slots = list(range(rank, 29, tasks))
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir/f'receipt_rank{rank:02d}.json').exists():
        raise FileExistsError(f'Immutable rank receipt exists in {out_dir}')
    for q in slots:
        if not (gram_dir/f'q{q:02d}.npz').is_file():
            raise FileNotFoundError(f'Missing assigned Gram q{q:02d} in {gram_dir}')
        for p in (8,16,24,32):
            if warm_dir is not None:
                for suffix in ('npz','json'):
                    if not (warm_dir/f'q{q:02d}_p{p:02d}.{suffix}').is_file():
                        raise FileNotFoundError(f'Warm-start q{q:02d} p{p:02d} {suffix} missing in {warm_dir}')
            for suffix in ('npz', 'json'):
                path = out_dir/f'q{q:02d}_p{p:02d}.{suffix}'
                if path.exists():
                    raise FileExistsError(f'Immutable output exists; use a new experiment: {path}')
    check_result = varpro.synthetic_check() if check else None
    statuses = []
    for q in slots:
        path = gram_dir/f'q{q:02d}.npz'
        bank = load_gram(path)
        loss = line_weights(bank['z'], omega, weight)
        training = train_indices(bank)
        gram = bank['channel_gram'][np.ix_(training, training)]
        upstream_path = path.with_suffix('.json')
        upstream = json.loads(upstream_path.read_text()) if upstream_path.is_file() else {'receipt_missing': str(upstream_path)}
        for p in (8,16,24,32):
            stem = out_dir/f'q{q:02d}_p{p:02d}'
            record = {'q': q, 'p': p, 'provenance': origin, 'order_receipt': receipt,
                      'input_path': str(path), 'input_sha256': sha(path), 'upstream_receipt': upstream}
            try:
                print(f'FIT q{q:02d} p={p} rank={rank} start', flush=True)
                initial = None
                if warm_dir is not None:
                    initial, seed_receipt = warm_start(warm_dir, q, p, bank, loss)
                    record['warm_start'] = seed_receipt
                fitted = varpro.fit(bank['z'], loss, gram, bank['sketches'], p, initial=initial)
                errors = line_errors(bank, fitted, omega, weight)
                with stem.with_suffix('.npz').open('xb') as stream:
                    np.savez(stream, poles_ry=fitted['poles_ry'], row_map=fitted['row_map'],
                             z_train_ry=bank['z'], z_held_ry=bank['zh'], weights_loss=loss,
                             train_channel_indices=training)
                record.update(status='FIT_COMPLETE', diagnostics={k:v for k,v in fitted.items() if k != 'row_map'},
                              line_errors=errors, export_path=str(stem.with_suffix('.npz')),
                              export_sha256=sha(stem.with_suffix('.npz')))
            except Exception as exc:
                record.update(status='FAILED', error=str(exc), traceback=traceback.format_exc())
            write_json(stem.with_suffix('.json'), record)
            statuses.append({'q': q, 'p': p, 'status': record['status'], 'path': str(stem.with_suffix('.json'))})
            print(f"FIT q{q:02d} p={p} rank={rank} {record['status']}", flush=True)
    write_json(out_dir/f'receipt_rank{rank:02d}.json', {'provenance': origin,
               'status': 'COMPLETE' if all(x['status']=='FIT_COMPLETE' for x in statuses) else 'COMPLETE_WITH_FAILURES',
               'synthetic_check': check_result, 'records': statuses})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gram', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--synthetic-check', action='store_true')
    parser.add_argument('--warm-start', type=Path, help='Matching q/p exports; grid/weights/solver equality required')
    parser.add_argument('--summary-only', action='store_true', help='Emit all-q original-Gram ORDER rank tables only')
    args = parser.parse_args()
    if args.summary_only:
        rank_summary(args.gram, args.out)
    else:
        main(args.gram, args.out, args.synthetic_check, args.warm_start)
