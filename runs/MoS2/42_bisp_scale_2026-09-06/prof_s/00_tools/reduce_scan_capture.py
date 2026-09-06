"""Reduce canonical HLO and native launch-correlated units for the four Sigma classes."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
p.add_argument('--scan',action='store_true')
a=p.parse_args();r=a.run
census=json.loads((r/'async_collectives.json').read_text())
name='jit_contract_class' if a.scan else 'jit_contract_block'
files=sorted((r/'xla_dump_rank0').glob(f'module_*.{name}.*memory-usage-report.txt'))
assert len(files)==4,files
rows=[]
for cls,f,occ in zip(('CC','CT','TC','TT'),files,((1,1,1,1) if a.scan else (1,3,3,9))):
 program=int(f.name.split('.')[0].split('_')[1])
 subprocess.run([sys.executable,str(Path(__file__).with_name('extract_nsys_unit.py')),str(r),str(program),'--occurrence',str(occ)],check=True,stdout=subprocess.DEVNULL)
 u=json.loads((r/f'unit_{program}_{occ}.json').read_text())
 groups={'GEMM':0.,'NCCL':0.,'FFT':0.,'other_fused':0.}
 for k in u['kernels']:
  label=k['name'].lower()
  key='NCCL' if 'nccl' in label else 'GEMM' if ('gemm' in label or 'cutlass' in label) else 'FFT' if 'fft' in label else 'other_fused'
  groups[key]+=k['sum_ms']
 m=next(m for m in census['modules'] if f'module_{program:04d}.' in m['file'])
 rows.append(dict(shape_class=cls,program=program,occurrence=occ,projected_ms=u['projected_ms'],kernel_sum_ms=u['kernel_sum_ms'],kernel_classes_ms=groups,memory=f.read_text().splitlines()[0],explicit_collectives=m['collectives']))
(r/'selected_units.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
