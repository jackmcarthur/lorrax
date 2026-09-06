"""Copy matched P/F inputs and write launchers without submitting jobs."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

W = Path(__file__).resolve().parents[5]
S = W.parents[2]
F = S / 'tmp/worktrees/wt_main_de8dcfbc_fixed'
P_PIN = '9f569c4bf75bad40e4f5895946874b4c503e4410'
F_PIN = 'e1559a071e244b4f049c924781b668d9e1560739'
CASES = [
    ('Si', '100', 'cohsex', '66_parent_cohsex_binding', '49_fixed_main_cohsex'),
    ('Si', '100', 'gn', '56_parent_gn', '48_fixed_main_gn'),
    ('MoS2', '41', 'full_static', '73_parent_full_static', '11_fixed_main_full_static'),
    ('MoS2', '41', 'packed_bare', '74_parent_packed_bare', '12_fixed_main_packed_bare'),
    ('MoS2', '41', 'dynamic_eps5', '80_parent_dynamic_eps5', '79_fixed_main_dynamic_eps5'),
]
for index, (system, number, label, parent, fixed) in enumerate(CASES):
    base = Path('runs') / system / f'{number}_bisp_parent_route_2026-09-05'
    donor = S / base / parent
    for offset, (arm, name, source, pin) in enumerate([
            ('P', parent, W, P_PIN), ('F', fixed, F, F_PIN)]):
        dest = W / base / 'prof_s' / f'{index*2+offset+1:02d}_{arm}_{label}_baseline'
        dest.mkdir(parents=True, exist_ok=False)
        template = S / base / name
        shutil.copytree(donor / 'tmp', dest / 'tmp')
        deck = (template / 'cohsex.in').read_text()
        deck = deck.replace(str(template), str(dest))
        (dest / 'cohsex.in').write_text(deck)
        for extra in ['dipole.h5', 'parallel_transport.h5']:
            if (template / extra).exists():
                shutil.copy2(template / extra, dest / extra)
        wrapper = (template / 'rankwrap.sh').read_text()
        wrapper = '\n'.join('W='+str(source) if l.startswith('W=') else l
                            for l in wrapper.splitlines())+'\n'
        (dest / 'rankwrap.sh').write_text(wrapper)
        (dest / 'driver.sh').write_text(f'''#!/bin/bash
set -euo pipefail
git -C {source} rev-parse HEAD
git -C {source} diff --exit-code {pin} -- src services > source.diff
export LORRAX_DEBUG_PRINT=1
exec python3 -u -m gw.gw_jax -i cohsex.in > driver.rank${{SLURM_PROCID}}.log 2>&1
''')
        (dest / 'run.sh').write_text('''#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
JID=${1:?Pass an explicitly authorized allocation JID}
export LX_BASE_MODULE=lorrax_A
for attempt in 1 2; do
 test ! -e "driver.$attempt.log" || { echo "Attempt exists; make a new variant."; exit 1; }
 lx run --jid "$JID" --wait 1800 -N 1 -G 4 -n 4 -- ./rankwrap.sh ./driver.sh > "driver.$attempt.log" 2>&1
 rc=$?
 if [ "$rc" = 0 ]; then
  test -s eqp0.dat && test -s eqp1.dat && test -s sigma_diag.dat && grep -Eq '\\[lx\\] step .* exit 0' "driver.$attempt.log"
  exit $?
 fi
 [ "$rc" = 98 ] || exit "$rc"
done
exit 98
''')
        for script in ['rankwrap.sh', 'driver.sh', 'run.sh']:
            (dest / script).chmod(0o755)
        receipts = []
        # Bound traversal to this copied run's tmp, never a shared top-level root.
        for file in sorted((dest / 'tmp').rglob('*')):
            if file.is_file():
                h = hashlib.sha256()
                with file.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024*1024), b''):
                        h.update(chunk)
                receipts.append(f'{h.hexdigest()}  {file.relative_to(dest)}')
        (dest / 'inputs.sha256').write_text('\n'.join(receipts)+'\n')
        (dest / 'manifest.yaml').write_text(f'''run_id: {dest.name}
system: {system}
pipeline: lorrax_only
platform: perlmutter
variant_of: {template}
reuse_from_parent: [{donor}/tmp]
overrides: {{debug_print: true, matched_tmp_donor: P}}
source:
  checkout: {source}
  commit: {pin}
  based_on: {pin}
allocator: BFC@0.85
geometry: {{nodes: 1, gpus: 4, ranks: 4}}
steps:
  00_lorrax: {{state: pending}}
''')
        print(dest.relative_to(W))
