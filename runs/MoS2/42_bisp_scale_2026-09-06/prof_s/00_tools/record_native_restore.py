"""Retain one warm native producer per class and its optimized collective/memory census."""
from pathlib import Path
import argparse,json,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run
c=json.loads((r/'async_collectives.json').read_text());rows=[]
for cls,f in zip(('CC','CT','TC','TT'),sorted((r/'xla_dump_rank0').glob('*.jit_restore.*memory-usage-report.txt'))):
 program=int(f.name.split('.')[0].split('_')[1])
 subprocess.run([sys.executable,str(Path(__file__).with_name('extract_nsys_unit.py')),str(r),str(program),'--occurrence','1'],check=True,stdout=subprocess.DEVNULL)
 u=json.loads((r/f'unit_{program}_1.json').read_text())
 m=next(m for m in c['modules'] if m['file']==f.name.replace('-memory-usage-report',''))
 rows.append(dict(shape_class=cls,program=program,projected_ms=u['projected_ms'],kernel_sum_ms=u['kernel_sum_ms'],memory=f.read_text().splitlines()[0],explicit_collectives=m['collectives']))
(r/'restore_units.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
