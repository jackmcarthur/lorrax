"""Collect native Green-gate receipts and run the canonical HLO census on an indexed capture."""
from pathlib import Path
import argparse,json,re,shutil,subprocess
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run.resolve()
w=Path(__file__).resolve().parents[5]
s=Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14')
parsed=json.loads(subprocess.check_output(['python3',str(s/'tools/parse_lorrax_sigma_run.py'),str(r/'gwjax.out')],text=True))
boundaries=json.loads(subprocess.check_output(['python3',str(Path(__file__).with_name('analyze_boundaries.py')),str(r)],text=True))
modules=[json.loads(x) for x in (r/'compile_modules.jsonl').read_text().splitlines()]
hlo=r/'xla_dump_rank0/module_tau.jit__tau.gpu_after_optimizations.txt'
hlo_text=hlo.read_text()
analysis=r/'green_hlo_analysis';analysis.mkdir(exist_ok=True)
shutil.copy2(hlo,analysis/'module_0000.jit__tau.gpu_after_optimizations.txt')
subprocess.run(['python3',str(s/'tools/hlo/analyze_hlo_dump.py'),str(analysis),'--top','3'],check=True,stdout=subprocess.DEVNULL)
subprocess.run(['python3',str(w/'runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/00_tools/census_async_starts.py'),str(r)],check=True,stdout=subprocess.DEVNULL)
result=dict(step=[x for x in (r/'driver.1.log').read_text().splitlines() if '[lx] step ' in x and ' exit ' in x], stages=parsed,boundaries=boundaries,
 compile_events=len(modules),compiler_seconds=sum(x['seconds'] for x in modules),tau_compile=[x for x in modules if x['name']=='jit__tau'],
 custom_calls=dict(__import__('collections').Counter(re.findall(r'custom_call_target="([^"]+)"',hlo_text))),
 collectives=json.loads((r/'async_collectives.json').read_text()),memory=(r/'tau_memory.txt').read_text(),
 capture_note='module_0000 is the analysis index of the single as_text capture, not an original XLA module ordinal')
(r/'green_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('tau_compile','stages','collectives')},indent=2))
