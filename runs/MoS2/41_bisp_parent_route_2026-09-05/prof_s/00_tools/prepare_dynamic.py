"""Prepare P/F captures replaying one common certified rule per measured window."""
from pathlib import Path
import json
import re
import shutil
from gw import sigma_box_plan

W=Path(__file__).resolve().parents[5]
for system,root,pname,fname,pout,fout,eps in [
 ('Si','100','03_P_gn_baseline','04_F_gn_baseline','13_P_gn_profile','14_F_gn_profile',1e-4),
 ('MoS2','41','09_P_dynamic_eps5_baseline','10_F_dynamic_eps5_baseline','20_P_dynamic_profile','21_F_dynamic_profile',1e-5)]:
 base=W/'runs'/system/f'{root}_bisp_parent_route_2026-09-05/prof_s'
 logs=[(base/n/'driver.rank0.log').read_text() for n in (pname,fname)]
 pattern=r'n_tau=(\d+), nodes=(\w+), cache=([^,]+), box=\(([^)]+)\)'
 receipts=[[dict(n=int(n),digest=d,cache=c,box=[float(x) for x in b.split(',')])
            for n,d,c,b in re.findall(pattern,log)] for log in logs]
 assert receipts[0] and len(receipts[0])==len(receipts[1])
 selected=[]
 # Select a certificate that contains both actual domains, without changing it.
 for left,right in zip(*receipts):
  box=[min(left['box'][0],right['box'][0]),max(left['box'][1],right['box'][1]),
       min(left['box'][2],right['box'][2]),max(left['box'][3],right['box'][3])]
  candidates=[]
  for donor in [pname,fname]:
   result,warnings=sigma_box_plan._rule_cache_lookup(str(base/donor/'tmp/sigma_quadrature_rules'),
        box,eps,box[0]>0 or box[1]<0,noise_amplification_cap=0.05*eps/6e-8)
   if result is not None: candidates.append((result[0].node_count,base/donor/'tmp/sigma_quadrature_rules'/result[1]))
  assert candidates, ('No common certified rule',system,box)
  count,path=min(candidates,key=lambda item:item[0])
  selected.append(dict(box=box,count=count,path=str(path)))
 for donor,name in [(pname,pout),(fname,fout)]:
  src,out=base/donor,base/name;out.mkdir()
  for item in ['cohsex.in','rankwrap.sh','run.sh','dipole.h5']:
   if (src/item).exists(): shutil.copy2(src/item,out/item)
  shutil.copytree(base/pname/'tmp',out/'tmp')
  for i,row in enumerate(selected):
   target=out/'replay_rules'/str(i);target.mkdir(parents=True)
   shutil.copy2(row['path'],target/Path(row['path']).name)
  (out/'replay.json').write_text(json.dumps(selected,indent=2)+'\n')
  (out/'manifest.yaml').write_text((src/'manifest.yaml').read_text().replace(donor,name)+'instrument: nsys_rank0_matched_certified_replay\n')
  (out/'cohsex.in').write_text((out/'cohsex.in').read_text().replace(str(src),str(out)))
  static=W/'runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/11_P_full_static_profile'
  payload=(static/'driver.sh').read_text()
  prefix=(src/'driver.sh').read_text().split('export LORRAX_DEBUG_PRINT')[0]
  (out/'driver.sh').write_text(prefix+payload[payload.index('export LORRAX_DEBUG_PRINT'):])
  (out/'driver.sh').chmod(0o755)
  instrument=(W/'runs/MoS2/41_bisp_parent_route_2026-09-05/prof_s/00_tools/profile_driver.py').read_text()
  additions='''from gw import ppm_tau_kernel
from gw.mpa import sigma as mpa_sigma
_tau_factory = ppm_tau_kernel.get_shared_sigma_tau_kernel
def tau_factory(*args, **kwargs):
    kernel = _tau_factory(*args, **kwargs)
    wrapped = measured(kernel, 'sigma_tau', {})
    if hasattr(kernel, 'lower'):
        wrapped.lower = kernel.lower
    return wrapped
ppm_tau_kernel.get_shared_sigma_tau_kernel = tau_factory
mpa_sigma.get_shared_sigma_tau_kernel = tau_factory
'''
  instrument='from gw import gw_jax\nimport rule_replay\n'+instrument.replace('def captured_main():',additions+'\ndef captured_main():')
  (out/'profile_driver.py').write_text(instrument)
  shutil.copy2(Path(__file__).parent/'rule_replay.py',out/'rule_replay.py')
  print(out, 'common nodes',sum(row['count'] for row in selected),flush=True)
