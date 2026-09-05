#!/usr/bin/env python3
"""Join the sandbox HLO analyzer with native rank-local Nsight CSV tables.

Static HLO counts exclude async done instructions. Dynamic device kernel
counts include communication hidden inside vendor-backed distributed GEMMs.
Projected spans and kernel sums are different currencies and must not be added.
Only named files in the supplied analyzer manifest are read.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def census(run):
    summary = json.loads((run/'xla_dump_rank0/hlo_summary.json').read_text())
    projected = read_csv(run/'stats_nvtx_gpu_proj_sum.csv')
    kernels = read_csv(run/'stats_nvtx_kern_sum.csv')
    results = []
    modules = summary['modules']
    if isinstance(modules, list):
        modules = {m['module']:m for m in modules}
    for name, mod in modules.items():
        number = int(re.search(r'module_(\d+)', name)[1])
        hlo = mod.get('hlo') or {}
        path = Path(hlo.get('path', ''))
        if not path.is_file():
            path = run/'xla_dump_rank0'/path.name
        if not path.is_file():
            continue
        source = path.read_text(errors='replace')
        counts = Counter(c['op'].removesuffix('-start') for c in hlo.get('collectives', []))
        # The sandbox analyzer currently omits async all-reduce-start.
        supplemental = re.findall(r'^\s*(?:ROOT )?%[^\n=]+ = [^\n]+? all-reduce-start\(', source, re.M)
        counts['all-reduce'] += len(supplemental)
        key = f'program_id={number}#'
        spans = [r for r in projected if key in r['Range']]
        device = [r for r in kernels if key in r['NVTX Range']]
        family = Counter()
        for r in device:
            if 'nccl' in r['Kernel Name'].lower():
                family[r['Kernel Name']] += int(r['Kern Inst'])
        results.append(dict(module=name, peak_bytes=(mod.get('memory') or {}).get('total_bytes'),
            static_collectives={k:v for k,v in counts.items() if v},
            source_files=re.findall(r'^\d+ "([^"\n]+\.py)"', source, re.M),
            source_functions=re.findall(r'^\d+ "([^"\n]+)"', source.split('FunctionNames',1)[-1].split('FileLocations',1)[0], re.M),
            projected=spans, kernels=device, nccl_kernel_counts=dict(family),
            hlo_collectives=hlo.get('collectives', [])))
    log_path = run/'driver_rank0.log'
    log = log_path.read_text(errors='replace') if log_path.exists() else ''
    host = dict(
        stages={n.strip():float(t) for n,t in re.findall(r'^  (.+?)\s{2,}([0-9.]+)\s+[0-9.]+%',log,re.M)},
        z_build_solve_ms=[list(map(int,p)) for p in re.findall(r'z_q_build=(\d+)ms solve=(\d+)ms',log)],
        tile_fit_write_total_ms=[list(map(int,p)) for p in re.findall(r'fit=(\d+)ms write=(\d+)ms total=(\d+)ms',log)],
        arena_gb=re.findall(r'GPU high-water mark: ([0-9.]+) GB',log),
        compile_receipt=re.findall(r'\[compile-cache\].*summary:.*',log),
        rules=re.findall(r'n_tau=(\d+), nodes=(\w+), cache=([^,]+)',log))
    return dict(host=host, run=str(run), instrument='Nsight native NVTX projections/kernel sums + sandbox analyze_hlo_dump.py', modules=results,
                thunk_kernels=[r for r in kernels if r['NVTX Range'].startswith('TSL:Thunk:')],
                thunk_projected=[r for r in projected if r['Range'].startswith('TSL:Thunk:')])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args=parser.parse_args()
    data=census(args.run)
    args.out.write_text(json.dumps(data,indent=2)+'\n')
    rows=['| module | calls | projected median ms | static collectives | source functions |',
          '|---|---:|---:|---|---|']
    for m in data['modules']:
        p=m['projected']
        if not m['static_collectives'] and not m['nccl_kernel_counts']:
            continue
        rows.append('| '+m['module']+' | '+(p[0]['Range Instances'] if p else '—')+' | '+(f"{float(p[0]['Proj Med (ns)'])/1e6:.3f}" if p else '—')+' | '+str(m['static_collectives'])+' | '+', '.join(m['source_functions'])+' |')
    args.out.with_suffix('.md').write_text('\n'.join(rows)+'\n')

if __name__=='__main__':
    main()
