"""Apply the incumbent PERF2 parser to the six-by-six lane's immutable logs."""
import importlib.util,json
from pathlib import Path
s=Path(__file__).resolve().parents[3];root=s/'runs/MoS2/42_bisp_scale_2026-09-06/prof_zw';parser=s/'tmp/worktrees/wt_psi_irr_perf2_codex_20260905/tools/profile_collective_census.py'
spec=importlib.util.spec_from_file_location('census',parser);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rows={}
for run in root.glob('[0-9][0-9]_*'):
 if not (run/'gwjax.out').is_file():continue
 view=Path('six_analysis')/run.name;view.mkdir(parents=True,exist_ok=True);(view/'xla_dump_rank0').mkdir(exist_ok=True);(view/'xla_dump_rank0/hlo_summary.json').write_text('{"modules":{}}')
 log=view/'driver_rank0.log'
 if not log.exists():log.symlink_to(run/'driver.rank0.log')
 rows[run.name]=module.census(view)['host']
Path('six_analysis/host.json').write_text(json.dumps(rows,indent=2)+'\n')
for name,row in rows.items():print(name,row['stages'],row['compile_receipt'][-1:] )
