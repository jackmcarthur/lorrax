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
import constrained_varpro
from measure import spectrum


ORDER = Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/152_shared_pole_push_2026-09-07/exchange/order/weight.npz')
ORDER_SHA = '27eb75153dc49849e6c5f3d27cbff24514f7ff28a0f705bc864be34c1b82b0f2'
EV = 13.605693122994
MOMENT_NAMES = ['M0', 'Mm1', 'M1']
MOMENT_IDENTITY = {'kind': 'parent_dA', 'names': MOMENT_NAMES, 'units': 'Ry',
                   'coordinate': 'physical moments in the same V-whitened Hermitian channels'}


def train_indices(bank):
    """Indices of unweighted Htrain then Atrain in a full train/held Gram."""
    nt, nh = len(bank['z']), len(bank['zh'])
    return np.r_[np.arange(nt), np.arange(nt)+nt+nh]


def read_rowplan(path):
    """Read one fixed list of original training indices, with its byte hash."""
    if path is None:
        return None
    raw = Path(path).read_bytes()
    indices = json.loads(raw)
    if (not isinstance(indices, list) or not indices or
            any(type(index) is not int for index in indices) or
            len(set(indices)) != len(indices)):
        raise ValueError('Training rowplan must be a nonempty JSON list of unique integers')
    return {'indices': indices, 'path': str(Path(path).resolve()),
            'sha256': hashlib.sha256(raw).hexdigest()}


def selected_rows(bank, rowplan):
    """Validate original training indices; held rows can never enter the fit."""
    nt = len(bank['z'])
    selected = np.arange(nt) if rowplan is None else np.asarray(rowplan['indices'], dtype=int)
    if np.any(selected < 0) or np.any(selected >= nt):
        raise ValueError(f'Training rowplan indices must lie in [0,{nt}); held rows are forbidden')
    scales = observation_scales(bank)
    value_rows = selected[scales[selected]==0]
    if (rowplan is not None or np.any(scales)) and len(value_rows)>40:
        raise ValueError('Training rowplan exceeds 40 value rows; derivative rows count separately')
    for row in selected[scales[selected]>0]:
        if np.count_nonzero(bank['z'][value_rows]==bank['z'][row]) != 1:
            raise ValueError('Each selected derivative needs exactly one selected value twin')
    return selected


def observation_scales(bank, held=False):
    '''Validate explicit physical h_Ry markers; absent metadata means values.'''
    grid = bank['zh' if held else 'z']
    key = 'held_derivative_scale_ry' if held else 'derivative_scale_ry'
    scales = np.asarray(bank.get(key,np.zeros(len(grid))),float)
    if scales.shape!=grid.shape or not np.all(np.isfinite(scales)) or np.any(scales<0):
        raise ValueError('Invalid observation derivative scales')
    if np.any((scales>0)&(scales!=grid.imag)):
        raise ValueError('Hermite scale must equal the sampled positive height in Ry')
    return scales


def value_count(bank, selected):
    return int(np.count_nonzero(observation_scales(bank)[selected]==0))


def fitting_weights(bank, selected, omega, weight, mode="equal_lines"):
    """Recompute quadrature on selected nodes; retain zero weights elsewhere."""
    loss = np.zeros(len(bank['z']))
    scales = observation_scales(bank)
    values = selected[scales[selected]==0]
    loss[values] = line_weights(bank['z'][values], omega, weight, mode)
    # h*dW/dz has W units. Its twin inherits the same loss weight, no new dial.
    for row in selected[scales[selected]>0]:
        twin = values[bank['z'][values]==bank['z'][row]]
        if len(twin)!=1:
            raise ValueError('Derivative weights require one selected value twin')
        loss[row] = loss[twin[0]]
    return loss


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


def line_weights(z, omega, weight, mode="equal_lines"):
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
        node_loss = trapezoid * (np.ones_like(sampled) if mode == "quadrature_height" else sampled)
        if node_loss.sum() <= 0:
            raise ValueError(f'No positive authenticated weight on height {height} eV')
        if mode == 'equal_lines':
            result[indices] = node_loss[inverse] / counts[inverse] / node_loss.sum() / heights.size
        elif mode in ('kernel_height', 'quadrature_height'):
            result[indices] = node_loss[inverse] / counts[inverse] / height**2
        else:
            raise ValueError(f'Unknown experimental weighting: {mode}')
    return result / result.sum()


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


def errors_from_gram(z_all, indices, row_map, poles, gram, weights, derivative_scale_ry=None):
    """Relative Frobenius error from Tr(L G L^T), with no n-dependent arrays.

    L consists of real and imaginary prediction rows minus the corresponding
    stored H/A rows. Linear Coulomb congruence commutes with this same map,
    so either whitened or physical channel Gram can be scored directly.
    """
    nall = len(z_all)
    extra = gram.shape[0]-2*nall
    nt = (row_map.shape[1]-extra)//2
    if extra not in (0,3) or row_map.shape[1] != 2*nt+extra or gram.shape != (2*nall+extra,2*nall+extra):
        raise ValueError('Residual row-map/Gram shape mismatch')
    source_rows = np.r_[np.arange(nt),np.arange(nt)+nall,np.arange(extra)+2*nall]
    selection = np.eye(2*nall+extra)[source_rows]
    scales = None if derivative_scale_ry is None else derivative_scale_ry[indices]
    phi = varpro.basis(z_all[indices], poles, derivative_scale_ry=scales)
    evaluation_map = np.vstack((phi.real, phi.imag)) @ row_map
    prediction = evaluation_map @ selection
    reference = np.eye(2*nall+extra)[np.r_[indices, indices+nall]]
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
    # Fixed-pole data amplification only; optimized pole motion is not included.
    amplification = np.linalg.svd(evaluation_map, compute_uv=False)[0]
    return {'relative_frobenius': float(np.sqrt(max(numerator, 0.) / denominator)),
            'squared_error': numerator, 'squared_reference': denominator,
            'negative_roundoff_clamped': bool(numerator < 0),
            'evaluation_map_operator_2norm': float(amplification),
            'evaluation_map_scope': ('unweighted real H/A output from unweighted original Htrain/Atrain'
                                     + (' and raw Ry M0/Mm1/M1' if extra else '') + '; fixed-pole map only'),
            'rows': int(len(indices))}


def line_errors(bank, fitted, omega, weight, selected=None):
    """Report each train/held height, in whitened and physical coordinates."""
    z_all = np.r_[bank['z'], bank['zh']]
    result = {}
    scales = np.r_[observation_scales(bank),observation_scales(bank,held=True)]
    nt = len(bank['z'])
    constrained = fitted['row_map'].shape[1] == 2*nt+3
    coordinates = ([('white','moment_channel_gram'),('physical','physical_moment_channel_gram')]
                   if constrained else [('white','channel_gram'),('physical','physical_channel_gram')])
    selected = np.arange(nt) if selected is None else selected
    omitted = np.setdiff1d(np.arange(nt), selected)
    for split, base in [('train', selected), ('validation_unselected_train', omitted),
                        ('held', np.arange(nt,len(z_all)))]:
        for derivative in (False,True):
            rows = base[(scales[base]>0)==derivative]
            for height in np.unique(np.round(z_all[rows].imag * EV, 9)):
                indices = rows[np.round(z_all[rows].imag * EV, 9) == height]
                loss = line_weights(z_all[indices], omega, weight)
                label = f'{split}_height_{height:.9f}_ev' + ('_derivative' if derivative else '')
                result[label] = {}
                for coordinate, key in coordinates:
                    result[label][coordinate] = {
                        mode: errors_from_gram(z_all, indices, fitted['row_map'], fitted['poles_ry'],
                                               bank[key], selected, scales)
                        for mode, selected in [('uniform', np.ones(indices.size)), ('order_trapezoid', loss)]}
    return result


def load_gram(path, require_moments=False):
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
    if require_moments:
        shapes.update(moment_channel_gram=(nc+3,nc+3),physical_moment_channel_gram=(nc+3,nc+3))
    for key, shape in shapes.items():
        if key not in bank or bank[key].shape != shape or not np.all(np.isfinite(bank[key])):
            raise ValueError(f'{path}: invalid {key}, expected finite shape {shape}')
    observation_scales(bank)
    observation_scales(bank,held=True)
    return bank


def provenance():
    """Record source and Slurm job.step even for small host computations."""
    root = Path(__file__).resolve().parents[2]
    return {'job_step': os.getenv('SLURM_JOB_ID', 'login-cpu')+'.'+os.getenv('SLURM_STEP_ID', 'none'),
            'rank': int(os.getenv('SLURM_PROCID', '0')), 'source_tree': str(root),
            'source_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
            'source_sha256': {p.name: sha(p) for p in [Path(__file__), Path(varpro.__file__),
                                                      Path(constrained_varpro.__file__)]},
            'weight_path': str(ORDER), 'weight_sha256': ORDER_SHA,
            'weight_rule': 'trapezoid times ORDER; training heights normalized equally; held rows excluded',
            'row_map_contract': 'unweighted Htrain[0:Nt], Atrain[Nt+Nh:2*Nt+Nh]; residues in Ry',
            'scope': 'Hermitian-residue damped subcase only; no general complex residues, residue ranks, passivity, or Sigma score'}


def rank_summary(gram_dir, out_dir, training_indices=None):
    """Write all-q ORDER-weighted original complex-Gram spectral rank table."""
    for name in ('order_rank_summary.json', 'order_rank_table.md'):
        if (out_dir/name).exists():
            raise FileExistsError(f'Immutable output exists: {out_dir/name}')
    omega, weight, receipt = authenticated_weight()
    rowplan = read_rowplan(training_indices)
    records = []
    for q in range(29):
        path = gram_dir / f'q{q:02d}.npz'
        bank = load_gram(path)
        selected = selected_rows(bank, rowplan)
        loss = fitting_weights(bank, selected, omega, weight)
        scopes = {}
        selected_heights = np.round(bank['z'][selected].imag*EV,9)
        groups = [(f'height_{h:.9f}_ev', selected[selected_heights == h])
                  for h in np.unique(selected_heights)]
        groups.append(('combined', selected))
        for name, indices in groups:
            sw = np.sqrt(loss[indices])
            scopes[name] = spectrum(bank['complex_gram'][np.ix_(indices, indices)] * sw[:,None] * sw[None,:])
        records.append({'q': q, 'input_path': str(path), 'input_sha256': sha(path), 'scopes': scopes,
                        'values_used': value_count(bank,selected), 'selected_training_indices': selected.tolist()})
    out_dir.mkdir(parents=True, exist_ok=True)
    origin = provenance()
    origin['scope'] = 'All 29 q original V-whitened complex Gram, training only; no pole fit or residue-rank claim'
    origin['training_rowplan'] = rowplan
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


def moment_receipt(upstream):
    """Identify parent moments; refuse a known frozen-parent perturbed input."""
    digest = upstream.get('moment_sha256','')
    scope = upstream.get('moment_scope','')
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest.lower()):
        raise ValueError('Moment-constrained fit needs a producer-authenticated moment SHA256')
    if 'parent' not in scope.lower() or 'defect' in scope.lower():
        raise ValueError('Moment constraints require identified parent dA moments, not defects')
    censuses = [item['band_energy_census'] for item in upstream.get('input_paths',[])
                if isinstance(item,dict) and 'band_energy_census' in item]
    if 'unchanged parent' in scope.lower() and any(census.get('amplitude_mev',0)!=0 for census in censuses):
        raise ValueError('Perturbed DATA has frozen parent moments; dependent SC moments must be regenerated')
    return {'identity': MOMENT_IDENTITY, 'path': upstream.get('moment_bank'), 'sha256': digest,
            'scope': scope, 'band_energy_after_sha256': [census.get('after_sha256') for census in censuses]}


def warm_start(directory, q, p, bank, loss, selected=None, moment_constrained=False, current_moments=None):
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
    old_kind = record.get('kind','unconstrained')
    kind = 'moment_constrained' if moment_constrained else 'unconstrained'
    if old_kind != kind:
        raise ValueError('Warm-start constrained/unconstrained functional changed')
    if moment_constrained:
        if record['provenance']['source_sha256'].get('constrained_varpro.py') != sha(constrained_varpro.__file__):
            raise ValueError('Warm-start constrained solver changed')
        old_moments = record.get('moment_target')
        if old_moments is None or current_moments is None or old_moments['identity']!=current_moments['identity']:
            raise ValueError('Warm-start moment identity/order/units changed')
        if (old_moments['band_energy_after_sha256'] != current_moments['band_energy_after_sha256']
                and old_moments['sha256'] == current_moments['sha256']):
            raise ValueError('Band energies changed but parent moment artifact stayed frozen')
    if record['provenance']['weight_sha256'] != ORDER_SHA:
        raise ValueError('Warm-start ORDER functional changed')
    old_gram_path = Path(record['input_path'])
    if sha(old_gram_path) != record['input_sha256']:
        raise ValueError(f'Warm input Gram hash mismatch: {old_gram_path}')
    old_bank = load_gram(old_gram_path,require_moments=moment_constrained)
    selected = np.arange(len(bank['z'])) if selected is None else selected
    with np.load(path, allow_pickle=False) as old:
        old_selected = (old['selected_training_indices'] if 'selected_training_indices' in old.files
                        else np.arange(len(old['z_train_ry'])))
        if (not np.array_equal(observation_scales(old_bank),observation_scales(bank)) or
                not np.array_equal(observation_scales(old_bank,True),observation_scales(bank,True)) or
                not np.array_equal(old['z_train_ry'], bank['z']) or
                not np.array_equal(old['weights_loss'], loss) or
                not np.array_equal(old_selected, selected) or
                not np.array_equal(old_bank['zh'], bank['zh']) or
                old['poles_ry'].shape != (p,)):
            raise ValueError('Warm-start grid, selected rows, weights, held grid, or p changed; fixed functional refused')
        poles = old['poles_ry'].copy()
    return poles, {'path': str(path), 'sha256': sha(path), 'receipt_path': str(record_path),
                   'receipt_sha256': sha(record_path), 'grid_weights_solver_match': True,
                   'kind':kind, 'moment_identity_match':bool(moment_constrained),
                   'moment_data_policy':'Dependent moment values may change; constraint identities remain fixed',
                   'pole_variables': 'all frequencies and widths remain free'}


def main(gram_dir, out_dir, check=False, warm_dir=None, training_indices=None, moment_constrained=False, weighting="equal_lines"):
    """Fit q slots distributed by SLURM_PROCID stride SLURM_NTASKS."""
    rank, tasks = int(os.getenv('SLURM_PROCID', '0')), int(os.getenv('SLURM_NTASKS', '1'))
    if tasks < 1 or not 0 <= rank < tasks:
        raise ValueError('Invalid Slurm rank geometry')
    omega, weight, receipt = authenticated_weight()
    rowplan = read_rowplan(training_indices)
    origin = provenance()
    origin['training_rowplan'] = rowplan
    solver = constrained_varpro if moment_constrained else varpro
    kind = 'moment_constrained' if moment_constrained else 'unconstrained'
    moment_names = MOMENT_NAMES if moment_constrained else []
    extra = 3 if moment_constrained else 0
    origin['kind'] = kind
    origin['weighting'] = weighting
    if moment_constrained:
        origin['row_map_contract'] += '; then raw physical Ry M0,Mm1,M1 coordinates'
    slots = list(range(rank, 29, tasks))
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir/f'receipt_rank{rank:02d}.json').exists():
        raise FileExistsError(f'Immutable rank receipt exists in {out_dir}')
    for q in slots:
        if not (gram_dir/f'q{q:02d}.npz').is_file():
            raise FileNotFoundError(f'Missing assigned Gram q{q:02d} in {gram_dir}')
        for p in (8,16,24,32):
            # Missing/refused warm models are recorded per q/p inside the fit loop.
            for suffix in ('npz', 'json'):
                path = out_dir/f'q{q:02d}_p{p:02d}.{suffix}'
                if path.exists():
                    raise FileExistsError(f'Immutable output exists; use a new experiment: {path}')
    check_result = solver.synthetic_check() if check else None
    statuses = []
    for q in slots:
        path = gram_dir/f'q{q:02d}.npz'
        bank = load_gram(path,require_moments=moment_constrained)
        selected = selected_rows(bank, rowplan)
        loss = fitting_weights(bank, selected, omega, weight, weighting)
        training = train_indices(bank)
        nt = len(bank['z'])
        selected_channels = np.r_[selected, selected+nt]
        fit_training = training[selected_channels]
        if moment_constrained:
            fit_training = np.r_[fit_training,2*(nt+len(bank['zh']))+np.arange(3)]
        gram_name = 'moment_channel_gram' if moment_constrained else 'channel_gram'
        gram = bank[gram_name][np.ix_(fit_training, fit_training)]
        upstream_path = path.with_suffix('.json')
        upstream = json.loads(upstream_path.read_text()) if upstream_path.is_file() else {'receipt_missing': str(upstream_path)}
        current_moments = moment_receipt(upstream) if moment_constrained else None
        for p in (8,16,24,32):
            stem = out_dir/f'q{q:02d}_p{p:02d}'
            record = {'q': q, 'p': p, 'provenance': origin, 'order_receipt': receipt,
                      'input_path': str(path), 'input_sha256': sha(path), 'upstream_receipt': upstream,
                      'values_used': value_count(bank,selected), 'selected_training_indices': selected.tolist(),
                      'rowplan_sha256': rowplan['sha256'] if rowplan is not None else None,
                      'training_values_available': value_count(bank,np.arange(nt)), 'held_validation_values': int(np.count_nonzero(observation_scales(bank,True)==0)),
                      'unselected_training_validation_values': value_count(bank,np.setdiff1d(np.arange(nt),selected)),
                      'derivative_rows_used':len(selected)-value_count(bank,selected),
                      'within_40_fitted_values': bool(value_count(bank,selected) <= 40), 'kind':kind,
                      'moment_names':moment_names, 'moment_target':current_moments}
            try:
                print(f'FIT q{q:02d} p={p} rank={rank} start', flush=True)
                initial = None
                if warm_dir is not None:
                    initial, seed_receipt = warm_start(warm_dir, q, p, bank, loss, selected,
                                                       moment_constrained,current_moments)
                    record['warm_start'] = seed_receipt
                fitted = solver.fit(bank['z'][selected], loss[selected], gram,
                                    bank['sketches'][selected], p, initial=initial,
                                    derivative_scale_ry=observation_scales(bank)[selected])
                expanded_map = np.zeros((p,2*nt+extra), dtype=fitted['row_map'].dtype)
                output_columns = np.r_[selected_channels,2*nt+np.arange(extra)]
                expanded_map[:,output_columns] = fitted['row_map']
                fitted['row_map'] = expanded_map
                fitted.update(values_used=value_count(bank,selected), selected_training_indices=selected.tolist(),
                              rowplan_sha256=rowplan['sha256'] if rowplan is not None else None)
                errors = line_errors(bank, fitted, omega, weight, selected)
                with stem.with_suffix('.npz').open('xb') as stream:
                    np.savez(stream, poles_ry=fitted['poles_ry'], row_map=fitted['row_map'],
                             z_train_ry=bank['z'], z_held_ry=bank['zh'], weights_loss=loss,
                             derivative_scale_ry=observation_scales(bank),
                             held_derivative_scale_ry=observation_scales(bank,True),
                             train_channel_indices=training, selected_training_indices=selected,
                             kind=np.asarray(kind),moment_names=np.asarray(moment_names,dtype='U3'),
                             rowplan_sha256=np.asarray(rowplan['sha256'] if rowplan is not None else ''))
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
    parser.add_argument('--training-indices', type=Path,
                        help='Fixed JSON list of at most 40 original training rows; held rows forbidden')
    parser.add_argument('--moment-constrained',action='store_true',
                        help='Experiment variant: enforce parent M0,Mm1,M1 from extended Grams')
    parser.add_argument('--weighting', choices=['equal_lines','kernel_height','quadrature_height'], default='equal_lines',
                        help='Experimental line loss; height variants use trapezoid/height^2 with/without ORDER')
    parser.add_argument('--summary-only', action='store_true', help='Emit all-q original-Gram ORDER rank tables only')
    args = parser.parse_args()
    if args.summary_only:
        if args.moment_constrained:
            parser.error('--summary-only reports original sample ranks; omit --moment-constrained')
        rank_summary(args.gram, args.out, args.training_indices)
    else:
        main(args.gram, args.out, args.synthetic_check, args.warm_start, args.training_indices,args.moment_constrained,args.weighting)
