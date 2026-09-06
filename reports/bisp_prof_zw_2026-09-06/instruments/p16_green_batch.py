"""Measure pinned baseline then accepted source on the authorized shared P16 pool."""
from pathlib import Path
import json,subprocess
from prepare_6x6 import prepare,w,root,scale
paths=('src/gw/w_isdf.py','src/gw/photon_layout.py')
saved={p:(w/p).read_bytes() for p in paths}
status=[]
def run(name,baseline=False):
 r=prepare(name,w,scale/'02_P_static_P4_fresh',restart=True,reuse=root/'47_P6_fresh_before',ranks=16)
 if baseline:
  with (r/'manifest.yaml').open('a') as f:f.write('production_pin: 71ae0bde\nmeasurement_order: retrospective_pinned_baseline\n')
 with (r/'batch_console.log').open('w') as out:rc=subprocess.call(['bash',str(r/'runner.sh'),'57982945'],cwd=r,stdout=out,stderr=subprocess.STDOUT)
 status.append({'run':str(r),'returncode':rc});Path('p16_green_status.json').write_text(json.dumps(status,indent=2)+'\n')
 return r,rc
try:
 for p in paths:(w/p).write_bytes(subprocess.check_output(['git','-C',str(w),'show',f'71ae0bde:{p}']))
 subprocess.run(['git','-C',str(w),'diff','--exit-code','71ae0bde','--','src','services'],check=True)
 before,rc=run('66_P16_cached_before',True)
finally:
 for p,data in saved.items():(w/p).write_bytes(data)
 Path('p16_source_restored.txt').write_text('Restored accepted production source bytes.\n')
if rc==0:
 after,rc=run('67_P16_cached_streamed')
 if rc==0:
  gate=subprocess.call(['python3','gate_candidate.py',str(before),str(after)])
  status[-1]['gate']=gate
 Path('p16_green_status.json').write_text(json.dumps(status,indent=2)+'\n')
