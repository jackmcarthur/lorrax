"""Prepare immutable paired baseline inputs without launching either arm."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

worktree = Path(__file__).resolve().parents[5]
sandbox = worktree.parents[2]
run_root = Path(__file__).resolve().parent.parent
source_run = sandbox / "runs/MoS2/41_bisp_parent_route_2026-09-05/73_parent_full_static"
control = sandbox / "tmp/worktrees/wt_main_de8dcfbc_fixed"
arms = (("02_parent_static", worktree, "9f569c4b"),
        ("03_fixed_static", control, "e1559a07"))


def git(root, *args):
    """Read the source identity without changing its checkout."""
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def digest(path):
    """Hash copied inputs without interpreting their HDF5 format."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


for name, source, pin in arms:
    dest = run_root / name
    dest.mkdir(exist_ok=False)
    shutil.copy2(source_run / "cohsex.in", dest / "cohsex.in")
    shutil.copytree(source_run / "tmp", dest / "tmp")
    shutil.copy2(source_run / "dipole.h5", dest / "dipole.h5")
    checksums = {str(p.relative_to(dest)): digest(p)
                 for p in sorted((dest / "tmp").rglob("*")) if p.is_file()}
    (dest / "input_checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    wrapper = (source_run / "rankwrap.sh").read_text().splitlines()
    wrapper = [f"W={source}" if line.startswith("W=") else line for line in wrapper]
    (dest / "rankwrap.sh").write_text("\n".join(wrapper) + "\n")
    checks = "\n".join(
        f'test "$(git -C {source} rev-parse HEAD:{part})" = {git(source, "rev-parse", pin + ":" + part)}'
        for part in ("src", "services"))
    (dest / "driver.sh").write_text(f'''#!/bin/bash
set -euo pipefail
{checks}
test -z "$(git -C {source} diff -- src services)"
git -C {source} rev-parse HEAD > source.rank${{SLURM_PROCID}}.txt
export LORRAX_DEBUG_PRINT=1
python3 -u -m gw.gw_jax -i cohsex.in > driver.rank${{SLURM_PROCID}}.log 2>&1
test -s eqp0.dat
test -s eqp1.dat
''')
    (dest / "runner.sh").write_text(f'''#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
JID=${{1:?Pass an explicitly authorized pool JID}}
test ! -e lx_attempt1.log
export LX_BASE_MODULE=lorrax_A
for attempt in 1 2; do
  set +e
  lx run --jid "$JID" --wait 1800 -N 1 -G 4 -n 4 ./rankwrap.sh ./driver.sh > "lx_attempt${{attempt}}.log" 2>&1
  rc=$?
  set -e
  if [ "$rc" = 0 ]; then
    test -s eqp0.dat && test -s eqp1.dat
    grep -Eq '\\[lx\\] step .*exit 0' "lx_attempt${{attempt}}.log"
    exit 0
  fi
  # Retry only a pre-launch pool-expiry refusal on the same authorized JID.
  if [ "$rc" != 98 ] || [ "$attempt" = 2 ]; then exit "$rc"; fi
done
''')
    for script in ("rankwrap.sh", "driver.sh", "runner.sh"):
        (dest / script).chmod(0o755)
        subprocess.run(["bash", "-n", str(dest / script)], check=True)
    (dest / "manifest.yaml").write_text(f'''run_id: MoS2_bisp_prof_zw_{name}_2026-09-06
system: MoS2
pipeline: lorrax_only
platform: perlmutter
variant_of: 73_parent_full_static
reuse_from_parent: [cohsex.in, tmp, dipole.h5]
overrides: {{debug_print: true, source: {pin}}}
source:
  checkout: {source}
  commit: {git(source, 'rev-parse', 'HEAD')}
  production_pin: {git(source, 'rev-parse', pin)}
allocator: BFC@0.85
geometry: {{nodes: 1, gpus_per_node: 4, ranks: 4}}
steps:
  00_lorrax: {{state: pending}}
''')
    print(dest)
