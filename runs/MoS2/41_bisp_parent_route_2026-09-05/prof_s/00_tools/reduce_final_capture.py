"""Reduce final native captures through the same canonical and async-corrected tools."""
import argparse
import json
from pathlib import Path
import re
import subprocess

p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
p.add_argument('--static',action='store_true')
a=p.parse_args()
here=Path(__file__).parent
subprocess.run(['python3','/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tools/hlo/analyze_hlo_dump.py',str(a.run/'xla_dump_rank0'),'--top','12'],check=True,stdout=subprocess.DEVNULL)
subprocess.run(['python3',str(here/'census_async_starts.py'),str(a.run)],check=True)
subprocess.run(['python3',str(here/'analyze_boundaries.py'),str(a.run)],check=True,stdout=subprocess.DEVNULL)
hlo=json.loads((a.run/'xla_dump_rank0/hlo_summary.json').read_text())
async_rows=json.loads((a.run/'async_collectives.json').read_text())['modules']
rows=[]
for name,module in hlo['modules'].items():
 if not (name.endswith('.jit__tau') or name.endswith('.jit_contract_block')): continue
 n=int(re.search(r'module_(\d+)',name)[1])
 occurrence=1
 # Static ordered CC,CT,TC,TT classes have 3,9,9,27 calls.
 if a.static:
  cores=sorted(k for k in hlo['modules'] if k.endswith('.jit_contract_block'))
  occurrence=[1,3,3,9][cores.index(name)]
 subprocess.run(['python3',str(here/'extract_nsys_unit.py'),str(a.run),str(n),'--occurrence',str(occurrence)],check=True)
 collective=next(r for r in async_rows if r['file'].startswith(name+'.'))
 rows.append(dict(module=name,peak_bytes=module['memory']['total_bytes'],collectives=collective['collectives'],unit=json.loads((a.run/f'unit_{n}_{occurrence}.json').read_text())))
result=dict(run=str(a.run),step=[l for l in (a.run/'driver.1.log').read_text().splitlines() if '[lx] step ' in l],units=rows,boundaries=json.loads((a.run/'boundary_summary.json').read_text()))
(a.run/'final_capture_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
