"""Compare every fresh canonical zeta file with the sandbox's dataset parser."""
from pathlib import Path
import json,subprocess
s=Path(__file__).resolve().parents[5]
# This snapshot executes from the named prof_zw run directory.
root=Path.cwd().parent
rows=[]
for name in ('zeta_q.h5','zeta_q_mu1.h5','zeta_q_mu2.h5','zeta_q_mu3.h5'):
 a=root/'52_F6_fresh_control/tmp'/name;b=root/'47_P6_fresh_before/tmp'/name
 with open(name+'.comparison.log','w') as out:
  rc=subprocess.run(['python3',str(s/'tools/compare_zeta_h5.py'),str(a),str(b),'--rtol','0'],stdout=out,stderr=subprocess.STDOUT).returncode
 rows.append({'file':name,'returncode':rc})
Path('comparison_status.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows),flush=True)
