"""Record completed receipts without promoting failed or unlaunched variants."""
from pathlib import Path
import hashlib,json,re
from prepare_6x6 import root
rows=[]
for run in sorted(root.glob('[0-9][0-9]_*')):
 logs=list(run.glob('lx_attempt*.log'));text='\n'.join(p.read_text(errors='replace') for p in logs);steps=re.findall(r'\[lx\] step (\S+) exit (\d+)',text)
 status='complete' if steps and steps[-1][1]=='0' else 'failed' if steps else 'pending'
 if not steps and (run/'queue_disposition.txt').exists():status='skipped'
 if run.name=='00_environment':continue
 row={'run':run.name,'state':status,'steps':steps,'eqp0':(run/'eqp0.dat').exists(),'eqp1':(run/'eqp1.dat').exists(),'source_checks_failed':[]}
 for log in run.glob('source.rank*.txt'):
  if 'FAILED' in log.read_text():row['source_checks_failed'].append(log.name)
 checks=run/'input_checksums.json'
 if checks.exists():row['input_checksum_mismatch']=[name for name,digest in json.loads(checks.read_text()).items() if hashlib.sha256((run/name).read_bytes()).hexdigest()!=digest]
 manifest=run/'manifest.yaml'
 if manifest.exists() and status!='pending':
  data=manifest.read_text().replace('state: pending',f'state: {status}').replace('state: running',f'state: {status}')
  manifest.write_text(data)
 rows.append(row)
Path('six_run_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:print(row)
