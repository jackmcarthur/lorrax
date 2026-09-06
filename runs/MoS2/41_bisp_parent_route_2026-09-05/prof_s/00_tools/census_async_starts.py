"""Supplement the canonical census with async starts it currently omits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
pattern = re.compile(r'(?<![%\w-])(all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute)(-start)?\(')
modules = []
for path in sorted((args.run/'xla_dump_rank0').glob('*gpu_after_optimizations.txt')):
    counts, evidence = Counter(), []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        code = line.split(', metadata=', 1)[0]
        for kind, start in pattern.findall(code):
            counts[kind] += 1
            evidence.append(dict(line=number, op=kind+start, instruction=code.strip()))
    modules.append(dict(file=path.name, collectives=dict(counts), instructions=evidence))
output = dict(scope='Optimized instruction census; async starts counted once, done instructions excluded. Complements the unchanged canonical analyzer.', modules=modules)
(args.run/'async_collectives.json').write_text(json.dumps(output, indent=2)+'\n')
print(f'{len(modules)} optimized modules; {sum(bool(m["collectives"]) for m in modules)} contain collectives')
