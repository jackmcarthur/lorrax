"""Restore the pinned P bytes for matched controls, then restore the gated tip."""
from pathlib import Path
import json
import subprocess

s=Path(__file__).resolve().parents[3];w=s/'tmp/worktrees/wt_bisp_prof_zw_codex_20260906'
files=('src/gw/w_isdf.py','src/isdf/core.py')
originals={name:(w/name).read_bytes() for name in files}
rows=[]
try:
    for name in files:
        (w/name).write_bytes(subprocess.check_output(['git','-C',str(w),'show','9f569c4b:'+name]))
    diff=subprocess.check_output(['git','-C',str(w),'diff','9f569c4b','--','src','services'])
    assert not diff, 'Pinned P source/services differ beyond the two owned files'
    for run in map(Path,json.loads(Path('common_control_paths.json').read_text())):
        result=subprocess.run(['./runner.sh','57966610'],cwd=run)
        rows.append({'run':str(run),'returncode':result.returncode})
        Path('common_control_status.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(run.name,result.returncode,flush=True)
finally:
    for name,content in originals.items():(w/name).write_bytes(content)
    Path('source_restored.txt').write_text('Restored the gated source bytes after matched controls.\n')
raise SystemExit(any(row['returncode'] for row in rows))
