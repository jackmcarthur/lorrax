"""Audit cumulative tangential ranks from authenticated saved O(n) spectra.

Spectra are normalized by their largest value; scale cancels in all fractions.
The imaginary-response arrays contain singular values, so their nuclear mass
is a trace proxy conditional on -W(iu) being PSD, not a signed-eigenvalue audit.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def close_cut(values, count, tolerance=1e-6):
    """Include an entire adjacent relative-gap multiplet at the boundary."""
    while 0 < count < len(values):
        a, b = values[count-1:count+1]
        if abs(a-b) > tolerance * max(abs(a), abs(b)):
            break
        count += 1
    return count


def cumulative_count(values, epsilon, power):
    """Capture (1-epsilon) of sum(sigma**power), then close the multiplet."""
    weight = values**power
    target = (1-epsilon)*weight.sum()
    count = min(len(values), int(np.searchsorted(np.cumsum(weight), target))+1)
    assert weight[:count].sum() >= target*(1-1e-14)
    assert count == 1 or weight[:count-1].sum() < target*(1+1e-14)
    return close_cut(values, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sandbox', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    f = args.sandbox/'runs/frequency_integration_sandbox'
    sources = {
        'Na_sparse': f/'308_carrier_incremental_krylov_20260907/299_singular_cutoff/spectra',
        'Na_dense': f/'308_carrier_incremental_krylov_20260907/299_singular_cutoff/dense_spectra',
        'Si': f/'313_si_control_20260908/46_tau_spectra',
    }
    rows, provenance = [], []
    for dataset, root in sources.items():
        receipt = json.loads((root/'result.json').read_text())
        assert receipt['status'] == 'COMPLETE'
        expected = 8 if dataset == 'Si' else 29
        assert len(receipt['rows']) == expected
        for q in receipt['rows']:
            path = root/f"q{q['qslot']:02d}/spectra.npz"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == q['spectra_sha256']
            provenance.append(dict(dataset=dataset, parent=q['qslot'], path=str(path), sha256=digest,
                                   job_step=q['jobid']+'.'+q['stepid'], bank=receipt['bank']))
            with np.load(path) as data:
                for kind, axis in [('line', 'line_ev'), ('imaginary_response', 'imaginary_ev')]:
                    spectra = data[kind]
                    assert spectra.shape == (len(q[axis]), q['n'])
                    for support, (energy, values) in enumerate(zip(q[axis], spectra)):
                        assert np.isfinite(values).all() and np.all(values >= 0)
                        assert np.all(np.diff(values) <= 1e-14) and values[0] > 0
                        base = dict(dataset=dataset, parent=q['qslot'], q_full=q['q_full'], kind=kind,
                                    support=support, energy_ev=energy, n=q['n'],
                                    job_step=q['jobid']+'.'+q['stepid'], spectra_path=str(path))
                        criteria = [('relative_1e-3', close_cut(values, int(np.sum(values > .001*values[0])))),
                                    ('relative_1e-2', close_cut(values, int(np.sum(values > .01*values[0])))),
                                    ('fixed_ceil_n4', close_cut(values, (q['n']+3)//4))]
                        for eps in [.1,.03,.01,.003,.001,.0001,.00001,.000001]:
                            for power,name in [(1,'nuclear'),(2,'frobenius_squared')]:
                                criteria.append((f'{name}_eps{eps:g}', cumulative_count(values, eps, power)))
                        for criterion,count in criteria:
                            rows.append(dict(**base, criterion=criterion, rank=count,
                                discarded_nuclear=float(values[count:].sum()/values.sum()),
                                discarded_frobenius_squared=float((values[count:]**2).sum()/(values**2).sum())))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'per_support.csv').open('w') as out:
        writer=csv.DictWriter(out, fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    summary=[]
    for dataset in sources:
        for scope in ['q0','finite_q']:
            for kind in ['line','imaginary_response']:
                selected=[r for r in rows if r['dataset']==dataset and r['kind']==kind and (r['parent']==0)==(scope=='q0')]
                for criterion in dict.fromkeys(r['criterion'] for r in selected):
                    group=[r for r in selected if r['criterion']==criterion]
                    summary.append(dict(dataset=dataset,scope=scope,kind=kind,criterion=criterion,
                        supports=len(group),rank_min=min(r['rank'] for r in group),rank_max=max(r['rank'] for r in group),
                        rank_sum=sum(r['rank'] for r in group),
                        nuclear_min=min(r['discarded_nuclear'] for r in group),nuclear_max=max(r['discarded_nuclear'] for r in group),
                        frobenius_squared_min=min(r['discarded_frobenius_squared'] for r in group),
                        frobenius_squared_max=max(r['discarded_frobenius_squared'] for r in group)))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Saved-spectrum tangential-rank census','',
        'CPU postprocessing of authenticated saved spectra; no new GPU measurement. ',
        'Imaginary nuclear values are PSD-conditional trace proxies. Small tails inherit the W†W producer’s precision limit.',
        'Rank sums count each saved support once; they exclude conjugate copies and infinity and are not final model K.', '',
        '| Dataset | Scope | Role | Criterion | Rank range | Rank sum | Discarded nuclear % (range) | Discarded Frobenius² % (range) |',
        '|---|---|---|---|---|---:|---|---|']
    keep={'relative_1e-3','relative_1e-2','fixed_ceil_n4','nuclear_eps0.01','nuclear_eps0.001','frobenius_squared_eps0.001','frobenius_squared_eps1e-05'}
    for r in summary:
        if r['criterion'] not in keep:continue
        lines.append(f"| {r['dataset']} | {r['scope']} | {r['kind']} | {r['criterion']} | {r['rank_min']}–{r['rank_max']} | {r['rank_sum']} | {100*r['nuclear_min']:.6g}–{100*r['nuclear_max']:.6g} | {100*r['frobenius_squared_min']:.6g}–{100*r['frobenius_squared_max']:.6g} |")
    lines+=['','Job.step and hashed input paths for every support: per_support.csv / provenance.json.']
    (args.output/'table.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASS_SAVED_SPECTRA',spectra_files=len(provenance),table_rows=len(rows),output=str(args.output))))


if __name__ == '__main__':
    main()
