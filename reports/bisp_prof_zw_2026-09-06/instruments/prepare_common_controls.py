"""Copy common immutable rule certificates for matched pinned-source controls."""
from pathlib import Path
import hashlib
import json
import shutil

s=Path(__file__).resolve().parents[3];w=s/'tmp/worktrees/wt_bisp_prof_zw_codex_20260906'
selection=json.loads((s/'runs/MoS2/41_bisp_parent_route_2026-09-05/prof_zw/31_cpu_zeta/common_rule_selection.json').read_text())
reference=next((s/'runs/Si/99_psi_irr_zeta_2026-09-05/perf2').glob('09*/rule_replay.py')).read_text()
paths=[]
for pname,fname,arms in [('13_parent_gn','14_fixed_gn',('38_parent_gn_common','39_fixed_gn_common')),('19_parent_dynamic','20_fixed_dynamic',('40_parent_dynamic_common','41_fixed_dynamic_common'))]:
    chosen=[r for r in selection if Path(r['parent']).name==pname]
    assert len(chosen) in (6,8) and all(r['selected'] for r in chosen)
    parent=Path(chosen[0]['parent']);fixed=Path(chosen[0]['fixed'])
    for index,name in enumerate(arms):
        source=parent if index==0 else fixed;dest=parent.parent/name;dest.mkdir()
        for file in ('rankwrap.sh','driver.sh','runner.sh','manifest.yaml'):
            shutil.copy2(source/file,dest/file)
        for file in ('cohsex.in','dipole.h5'):
            if (parent/file).exists():shutil.copy2(parent/file,dest/file)
        shutil.copytree(parent/'tmp',dest/'tmp')
        for row in chosen:
            folder=dest/'replay_rules'/str(row['window']);folder.mkdir(parents=True)
            path=Path(row['selected']);shutil.copy2(path,folder/path.name)
        text=reference;a=text.index('_boxes = ');b=text.index('\ndef replay',a)
        text=text[:a]+'_boxes = '+repr([r['pbox'] for r in chosen])+text[b:]
        if index==0:
            text=text.replace('*, noise_amplification_cap):','*, noise_amplification_cap, reduction_steps=None):')
            text=text.replace('noise_amplification_cap=noise_amplification_cap)','noise_amplification_cap=noise_amplification_cap, reduction_steps=reduction_steps)')
        (dest/'rule_replay.py').write_text(text)
        (dest/'replay_driver.py').write_text('import rule_replay\nfrom gw import gw_jax\nfrom runtime import run_main_and_finalize\nrun_main_and_finalize(gw_jax.main)\n')
        driver=(dest/'driver.sh').read_text().replace('python3 -u -m gw.gw_jax','python3 -u replay_driver.py')
        if index==0:
            start=driver.index('test "$(git');end=driver.index('export LORRAX_DEBUG_PRINT=1')
            driver=driver[:start]+f'git -C {w} diff 9f569c4b -- src services > source.rank${{SLURM_PROCID}}.diff\ntest ! -s source.rank${{SLURM_PROCID}}.diff\ngit -C {w} rev-parse HEAD > source.rank${{SLURM_PROCID}}.txt\n'+driver[end:]
        (dest/'driver.sh').write_text(driver)
        manifest=(dest/'manifest.yaml').read_text().replace(source.name,name)
        manifest+='\npurpose: matched pinned production sources with common unchanged certificates; all original guards retained\n';(dest/'manifest.yaml').write_text(manifest)
        hashes={str(path.relative_to(dest)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (dest/'replay_rules').glob('*/*.npz')}
        (dest/'common_rules.sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
        paths.append(str(dest))
Path('common_control_paths.json').write_text(json.dumps(paths,indent=2)+'\n')
print(paths)
