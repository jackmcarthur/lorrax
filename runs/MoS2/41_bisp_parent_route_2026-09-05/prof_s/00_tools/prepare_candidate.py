"""Copy one completed control into an immutable candidate measurement directory."""
from pathlib import Path
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('baseline', type=Path)
parser.add_argument('candidate', type=Path)
parser.add_argument('--boundaries', action='store_true')
parser.add_argument('--hlo', action='store_true')
args = parser.parse_args()
source, target = args.baseline.resolve(), args.candidate.resolve()
worktree = Path(__file__).resolve().parents[5]
target.mkdir()
for name in ('cohsex.in', 'rankwrap.sh', 'run.sh', 'dipole.h5'):
    if (source/name).exists():
        shutil.copy2(source/name, target/name)
shutil.copytree(source/'tmp', target/'tmp')
(target/'cohsex.in').write_text((target/'cohsex.in').read_text().replace(str(source), str(target)))
manifest = (source/'manifest.yaml').read_text().replace(source.name, target.name)
manifest = '\n'.join(line for line in manifest.splitlines()
                     if not line.startswith('completion_receipt:'))
manifest = manifest.replace('state: complete', 'state: pending')
(target/'manifest.yaml').write_text(manifest+'\ncandidate_source: source.diff and source_head.txt\n')
driver = f'''#!/bin/bash
set -euo pipefail
git -C {worktree} rev-parse HEAD > source_head.txt
git -C {worktree} diff HEAD -- src services tests > source.diff
export LORRAX_DEBUG_PRINT=1
'''
if args.hlo:
    driver += 'export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=$PWD/xla_dump_rank${SLURM_PROCID}"\n'
if args.boundaries:
    instrument = (Path(__file__).parent.parent/'14_P_host_boundary/profile_driver.py').read_text()
    (target/'profile_driver.py').write_text(instrument)
    driver += 'exec python3 -u profile_driver.py -i cohsex.in > driver.rank${SLURM_PROCID}.log 2>&1\n'
else:
    # Preserve the production driver invocation from the original wrapper.
    driver += (source/'driver.sh').read_text().split('export LORRAX_DEBUG_PRINT=1\n', 1)[1]
(target/'driver.sh').write_text(driver)
(target/'driver.sh').chmod(0o755)
print(target)
