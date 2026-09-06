"""Summarize the lane's explicit boundary receipts, keeping nested totals separate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
rows = [json.loads(line) for line in (args.run/'boundary.jsonl').read_text().splitlines()]
groups = defaultdict(list)
for row in rows:
    key = (row['label'], row.get('term', ''),
           row.get('options', {}).get('q0_only', False))
    groups[key].append(row)
summary = []
for (label, term, head), values in groups.items():
    cold = [v['host_ms'] for v in values if v['compiles']]
    warm = [v['host_ms'] for v in values if not v['compiles']]
    summary.append(dict(label=label, term=term, head=head, calls=len(values),
        compiles=sum(v['compiles'] for v in values),
        compile_s=sum(v['compile_s'] for v in values),
        host_s=sum(v['host_ms'] for v in values)/1000,
        cold_calls=len(cold), cold_median_ms=statistics.median(cold) if cold else None,
        warm_calls=len(warm), warm_median_ms=statistics.median(warm) if warm else None))
result = dict(scope='synchronized host boundaries; outer stage is nested, never sum it with children', groups=summary)
(args.run/'boundary_summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
