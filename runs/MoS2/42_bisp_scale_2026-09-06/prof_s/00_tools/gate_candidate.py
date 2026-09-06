"""Run the canonical EQP comparison and compare complete printed Sigma rows."""
from pathlib import Path
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('baseline', type=Path)
parser.add_argument('candidate', type=Path)
args = parser.parse_args()
sandbox = Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14')
for name in ('eqp0.dat', 'eqp1.dat'):
    result = subprocess.run(['python3', str(sandbox/'tools/eqp_ab.py'),
        str(args.baseline/name), str(args.candidate/name), '--tol-uev', '0'],
        capture_output=True, text=True)
    (args.candidate/(name+'_ab.log')).write_text(result.stdout+result.stderr)
    print(result.stdout, end='')
    if result.returncode:
        raise SystemExit(result.returncode)
before = [row for row in (args.baseline/'sigma_diag.dat').read_text().splitlines() if 'n=' in row]
after = [row for row in (args.candidate/'sigma_diag.dat').read_text().splitlines() if 'n=' in row]
assert before and before == after, 'Complete printed Sigma rows differ.'
receipt = f'PASS: {len(before)} complete printed state rows identical, including sigCC/sigTT/sigCT.\n'
(args.candidate/'sigma_rows_ab.txt').write_text(receipt)
print(receipt, end='')
