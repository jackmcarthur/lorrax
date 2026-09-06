"""Gate a completed candidate and register its evidence through the sandbox tools."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('baseline', type=Path)
parser.add_argument('candidate', type=Path)
parser.add_argument('description')
parser.add_argument('sources', nargs='+')
args = parser.parse_args()
here, run = Path(__file__).parent, args.candidate
subprocess.run(['python3', str(here/'gate_candidate.py'), str(args.baseline), str(run)], check=True)
summary = {}
if (run/'boundary.jsonl').exists():
    subprocess.run(['python3', str(here/'analyze_boundaries.py'), str(run)], check=True, stdout=subprocess.DEVNULL)
    summary = json.loads((run/'boundary_summary.json').read_text())
step = next(line for line in (run/'driver.1.log').read_text().splitlines()
            if '[lx] step ' in line and 'exit 0' in line)
body = '**'+args.description+'** On branch perf/bisp-prof-s-2026-09-06, unmerged. '+(run/'sigma_rows_ab.txt').read_text().strip()+' Both EQP files pass tolerance0.'
static = next((g for g in summary.get('groups', []) if g['label']=='sigma_static_stage'), None)
if static:
    body += f' Static caller {static["host_s"]:.3f}s; {static["compiles"]} compilation events, {static["compile_s"]:.3f}s compiler work.'
tool = runpy.run_path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/tools/claims_append.py')
tool['main'].__globals__['CLAIMS'] = Path('CLAIMS.md')
sys.argv = ['claims_append.py', '--evidence', 'JID57966610 '+step, '--artifact', str(run/'sigma_rows_ab.txt'), body]
output = io.StringIO()
with contextlib.redirect_stdout(output):
    assert tool['main']() == 0
number = int(output.getvalue().strip())
claim = Path(f'claims/{number:04d}.md')
claim.write_text(body+'\n\n'+step+'\n'+str(run)+'\n'+json.dumps(summary, indent=2)+'\n')
report = Path('reports/bisp_prof_s_2026-09-06/report.md')
report.write_text(report.read_text()+'\n'+body+'\n\nEvidence: '+str(run)+'; '+step+'.\n')
manifest = run/'manifest.yaml'
manifest.write_text(manifest.read_text().replace('state: pending', 'state: complete')+'completion_receipt: "'+step+'"\n')
paths = [*args.sources, 'CLAIMS.md', str(claim), str(report)]
for pattern in ['*.sh', '*.py', 'manifest.yaml', 'eqp0.dat', 'eqp1.dat', 'sigma_diag.dat', 'gwjax.out', 'boundary*.json*', 'source.diff', 'source_head.txt', 'eqp*.dat_ab.log', 'sigma_rows_ab.txt']:
    paths.extend(str(path) for path in run.glob(pattern))
subprocess.run(['git', 'add', '-f', '--', *paths], check=True)
print(body, flush=True)
