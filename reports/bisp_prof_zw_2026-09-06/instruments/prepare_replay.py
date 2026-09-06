"""Adapt PERF2's guarded single-rule replay to the recorded parent windows."""
import ast
import json
from pathlib import Path
import re
import shutil

s=Path(__file__).resolve().parents[3]
pairs=[('Si/100_bisp_parent_route_2026-09-05','13_parent_gn','14_fixed_gn','32_fixed_gn_replay'),('MoS2/41_bisp_parent_route_2026-09-05','19_parent_dynamic','20_fixed_dynamic','33_fixed_dynamic_replay')]
for system,pname,fname,name in pairs:
    root=s/'runs'/system/'prof_zw';parent=root/pname;fixed=root/fname;dest=root/name;dest.mkdir()
    for file in ('rankwrap.sh','driver.sh','runner.sh','manifest.yaml'):
        shutil.copy2(fixed/file,dest/file)
    for file in ('cohsex.in','dipole.h5'):
        if (parent/file).exists():shutil.copy2(parent/file,dest/file)
    shutil.copytree(parent/'tmp',dest/'tmp')
    rows=re.findall(r'n_tau=(\d+), nodes=(\w+), cache=hit:([^,]+), box=(\([^\n]+?\)) Ry', (parent/'driver.rank0.log').read_text())
    assert len(rows) in (6,8)
    boxes=[]
    for i,(nodes,digest,filename,box) in enumerate(rows):
        selected=dest/'replay_rules'/str(i);selected.mkdir(parents=True)
        shutil.copy2(parent/'tmp/sigma_quadrature_rules'/filename,selected/filename)
        boxes.append(ast.literal_eval(box))
    (dest/'recorded_windows.json').write_text(json.dumps(rows,indent=2)+'\n')
    reference=next((s/'runs/Si/99_psi_irr_zeta_2026-09-05/perf2').glob('09*/rule_replay.py'))
    source=reference.read_text();start=source.index('_boxes = ');end=source.index('\ndef replay',start)
    source=source[:start]+'_boxes = '+repr(boxes)+source[end:]
    (dest/'rule_replay.py').write_text(source)
    (dest/'replay_driver.py').write_text('import rule_replay\nfrom gw import gw_jax\nfrom runtime import run_main_and_finalize\nrun_main_and_finalize(gw_jax.main)\n')
    driver=(dest/'driver.sh').read_text().replace('python3 -u -m gw.gw_jax','python3 -u replay_driver.py');(dest/'driver.sh').write_text(driver)
    manifest=(dest/'manifest.yaml').read_text().replace(fname,name)
    manifest+='\npurpose: guarded PERF2 replay of parent selected rules with unchanged certificates and acceptance gates\n';(dest/'manifest.yaml').write_text(manifest)
    print(dest)
