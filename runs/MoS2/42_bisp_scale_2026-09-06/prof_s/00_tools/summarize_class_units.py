"""Group explicit class/block timing receipts by the two endpoint extents."""
from pathlib import Path
from collections import defaultdict
import argparse, json, statistics
p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
a = p.parse_args()
groups = defaultdict(list)
for line in (a.run/'boundary.jsonl').read_text().splitlines():
    row = json.loads(line)
    shapes = row['options']['shapes']
    groups[tuple(s[-1] for s in shapes)].append(row)
result = []
for extents, rows in groups.items():
    cold = [r['host_ms'] for r in rows if r['compiles']]
    warm = [r['host_ms'] for r in rows if not r['compiles']]
    result.append(dict(extents=extents, calls=len(rows),
        cold_median_ms=statistics.median(cold) if cold else None,
        warm_median_ms=statistics.median(warm) if warm else None,
        compiler_seconds=sum(r['compile_s'] for r in rows)))
(a.run/'class_units.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
