"""Compare HDF5 payload bytes and attributes, allowing only the commit receipt.

Usage: python tests/bench/compare_io_artifacts.py BEFORE AFTER REPORT.json
Scans only the two named run directories and their immediate tmp directories.
"""
import json
from pathlib import Path
import sys

import h5py
import numpy as np

RECEIPT = 'lorrax_io_committed'


def same_bytes(left, right):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == 'O':
        return a.tolist() == b.tolist()
    return a.tobytes() == b.tobytes()


def compare(before, after):
    differences = []
    datasets, compared_bytes = 0, 0
    with h5py.File(before) as a, h5py.File(after) as b:
        names_a, names_b = [], []
        a.visit(names_a.append)
        b.visit(names_b.append)
        names_a = set(names_a) - {RECEIPT}
        names_b = set(names_b) - {RECEIPT}
        if names_a != names_b:
            differences.append({'objects': [sorted(names_a - names_b), sorted(names_b - names_a)]})
        for name in [''] + sorted(names_a & names_b):
            x, y = a[name or '/'], b[name or '/']
            if set(x.attrs) != set(y.attrs):
                differences.append({'attributes': name})
            for attr in set(x.attrs) & set(y.attrs):
                if not same_bytes(x.attrs[attr], y.attrs[attr]):
                    differences.append({'attribute_values': f'{name}@{attr}'})
            if not isinstance(x, h5py.Dataset):
                continue
            datasets += 1
            if x.shape != y.shape or x.dtype != y.dtype:
                differences.append({'geometry': name})
                continue
            selections = ([()] if not x.shape else
                          [slice(i, i + 16) for i in range(0, x.shape[0], 16)])
            for selection in selections:
                left, right = x[selection], y[selection]
                compared_bytes += np.asarray(left).nbytes
                if not same_bytes(left, right):
                    differences.append({'payload': name, 'selection': str(selection)})
                    break
        if RECEIPT in b and int(b[RECEIPT][0]) != 1:
            differences.append({'commit_state': 'not complete'})
    return {'datasets': datasets, 'compared_bytes': compared_bytes, 'differences': differences}


if __name__ == '__main__':
    before, after, report_path = map(Path, sys.argv[1:])
    results = {}
    for folder in (Path('.'), Path('tmp')):
        left = {p.name for p in (before / folder).glob('*.h5')}
        right = {p.name for p in (after / folder).glob('*.h5')}
        assert left == right, (folder, left, right)
        for name in sorted(left):
            relative = folder / name
            results[str(relative)] = compare(before / relative, after / relative)
    assert results, 'No HDF5 artifacts were collected'
    report_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    raise SystemExit(int(any(row['differences'] for row in results.values())))
