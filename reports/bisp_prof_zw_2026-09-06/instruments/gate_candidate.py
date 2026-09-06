"""Run the canonical printed-digit and photon-sector identity gates."""
from pathlib import Path
import subprocess
import sys
s=Path(__file__).resolve().parents[3]; w=s/'tmp/worktrees/wt_bisp_prof_zw_codex_20260906'
a,b=map(Path,sys.argv[1:]); passed=True
for name in ('eqp0','eqp1'):
    with (b/f'{name}_ab.txt').open('w') as out:
        p=subprocess.run(['python3',str(s/'tools/eqp_ab.py'),str(a/f'{name}.dat'),str(b/f'{name}.dat'),'--tol-uev','0'],stdout=out,stderr=subprocess.STDOUT)
    print(name,p.returncode,(b/f'{name}_ab.txt').read_text()); passed &= p.returncode==0
rows=[]
for run in (a,b):
    result=subprocess.check_output(['python3',str(s/'tools/sigma_diag_rows.py'),str(run/'sigma_diag.dat'),'--lorrax-root',str(w),'--labels','sigCC,sigTT,sigCT'],text=True)
    rows.append(result)
(b/'sectors.csv').write_text(rows[1]); (b/'sectors_baseline.csv').write_text(rows[0]); (b/'sector_gate.txt').write_text(f'exact={rows[0]==rows[1]}\n')
print('sectors exact',rows[0]==rows[1]); passed &= rows[0]==rows[1]
raise SystemExit(0 if passed else 1)
