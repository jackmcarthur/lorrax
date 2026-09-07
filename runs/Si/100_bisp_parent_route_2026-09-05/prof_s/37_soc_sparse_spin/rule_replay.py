"""Replay copied certificates while bounding roundoff-sized containment-edge drift."""
from pathlib import Path
import json
import numpy as np
from gw import sigma_box_plan
original = sigma_box_plan._rule_cache_lookup

def lookup(directory, box, eps, relative, **options):
    result = original(directory, box, eps, relative, **options)
    if result[0] is not None:
        return result
    candidates = []
    for path in Path(directory).glob('*.npz'):
        with np.load(path) as data:
            old = data['box']
            adjusted = np.array([max(box[0], old[0]), min(box[1], old[1]),
                                 max(box[2], old[2]), min(box[3], old[3])])
            shift = float(np.max(np.abs(adjusted-np.array(box))))
            if shift > 1e-10 or adjusted[0] > adjusted[1] or adjusted[2] > adjusted[3]:
                continue
            hit = original(directory, tuple(adjusted), eps, relative, **options)
            if hit[0] is not None:
                candidates.append((hit[0][0].node_count, shift, hit))
    if not candidates:
        raise RuntimeError(f'Copied schedule does not cover box {box}')
    _, shift, result = min(candidates, key=lambda row: row[:2])
    print(f'[fixed-rule-edge] shift={shift:.17g} Ry rule={result[0][1]}', flush=True)
    return result
sigma_box_plan._rule_cache_lookup = lookup
