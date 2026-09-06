"""Prepare a private candidate run with exact source and input receipts."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

sandbox = Path(__file__).resolve().parents[3]
worktree = sandbox/'tmp/worktrees/wt_bisp_prof_zw_codex_20260906'
source = Path(sys.argv[1]); dest = source.parent/sys.argv[2]
dest.mkdir()
for name in ('cohsex.in','dipole.h5','rankwrap.sh','runner.sh'):
    if (source/name).exists():
        shutil.copy2(source/name,dest/name)
shutil.copytree(source/'tmp',dest/'tmp')
files = ('src/gw/w_isdf.py','src/isdf/core.py','tests/test_photon_chi_vertices.py')
checksums = '\n'.join(hashlib.sha256((worktree/p).read_bytes()).hexdigest()+'  '+str(worktree/p) for p in files)+'\n'
(dest/'source.sha256').write_text(checksums)
(dest/'source.diff').write_text(subprocess.check_output(['git','-C',str(worktree),'diff','HEAD','--','src','services','tests'],text=True))
head = subprocess.check_output(['git','-C',str(worktree),'rev-parse','HEAD'],text=True).strip()
manifest = f'''run_id: {dest.name}
system: {source.parts[-4]}
pipeline: lorrax_only
platform: perlmutter
variant_of: {source}
reuse_from_parent: [cohsex.in, dipole.h5, tmp]
source: {{checkout: {worktree}, base_commit: {head}, diff: source.diff, checksums: source.sha256}}
allocator: BFC@0.85
geometry: {{nodes: 1, gpus_per_node: 4, ranks: 4}}
steps:
  00_lorrax: {{state: pending}}
'''
(dest/'manifest.yaml').write_text(manifest)
(dest/'driver.sh').write_text(f'''#!/bin/bash
set -euo pipefail
sha256sum -c source.sha256 > source.rank${{SLURM_PROCID}}.txt
export LORRAX_DEBUG_PRINT=1
python3 -u -m gw.gw_jax -i cohsex.in > driver.rank${{SLURM_PROCID}}.log 2>&1
test -s eqp0.dat
test -s eqp1.dat
''')
(dest/'driver.sh').chmod(0o755)
receipt = {}
for path in sorted((dest/'tmp').rglob('*')):
    if path.is_file():
        receipt[str(path.relative_to(dest))] = hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
(dest/'input_checksums.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(dest)
