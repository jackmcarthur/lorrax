"""Reuse PERF2's census on immutable driver logs through named analysis links."""
import importlib.util
import json
import sys
from pathlib import Path

sandbox=Path(__file__).resolve().parents[3]
parser=sandbox/'tmp/worktrees/wt_psi_irr_perf2_codex_20260905/tools/profile_collective_census.py'
spec=importlib.util.spec_from_file_location('perf2_census',parser)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
paths=[sandbox/'runs/MoS2/41_bisp_parent_route_2026-09-05/prof_zw'/name for name in ('06_parent_static','07_fixed_static')]
paths += [Path(p) for p in json.loads(Path('remaining_paths.json').read_text())]
paths += [Path(p) for p in sys.argv[1:]]
result={}
for run in paths:
    view=Path('baseline_analysis')/run.name; view.mkdir(parents=True,exist_ok=True)
    (view/'xla_dump_rank0').mkdir(exist_ok=True)
    (view/'xla_dump_rank0/hlo_summary.json').write_text('{"modules":{}}')
    log=view/'driver_rank0.log'
    if not log.exists(): log.symlink_to(run/'driver.rank0.log')
    result[run.name]=module.census(view)['host']
Path('baseline_analysis/host.json').write_text(json.dumps(result,indent=2)+'\n')
for name,data in result.items():
    print(name, data['stages'])
    print(data['compile_receipt'][-1])
    print('rules',data['rules'])
