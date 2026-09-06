"""Copy immutable scale inputs into this lane's guarded profiling runs."""
from pathlib import Path
import hashlib,json,shutil,subprocess
s=Path(__file__).resolve().parents[3];w=s/'tmp/worktrees/wt_bisp_prof_zw_codex_20260906';f=s/'tmp/worktrees/wt_main_de8dcfbc_fixed';scale=s/'runs/MoS2/42_bisp_scale_2026-09-06';root=scale/'prof_zw';root.mkdir(exist_ok=True)
def prepare(name,source,template,*,restart=False,reuse=None,profile=None,ranks=4):
 r=root/name;r.mkdir()
 for filename in ('cohsex.in','centroids_charge.txt','centroids_current.txt','dipole.h5','kin_ion.h5','Mo.upf','S.upf'):
  shutil.copy2(template/filename,r/filename)
 for filename in ('WFN.h5','MoS2.save'):(r/filename).symlink_to((template/filename).resolve())
 if reuse:shutil.copytree(reuse/'tmp',r/'tmp')
 else:(r/'tmp').mkdir()
 deck=(r/'cohsex.in').read_text().replace('restart = false','restart = true' if restart else 'restart = false')
 (r/'cohsex.in').write_text(deck)
 wrapper=(scale/'02_P_static_P4_fresh/rankwrap.sh').read_text().replace(str(s/'tmp/worktrees/wt_bisp_scale_codex_20260906'),str(source));(r/'rankwrap.sh').write_text(wrapper)
 pin=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip();diff=subprocess.check_output(['git','-C',str(source),'diff','--','src','services','tests'],text=True);(r/'source.diff').write_text(diff)
 tracked=subprocess.check_output(['git','-C',str(source),'ls-files','src','services'],text=True).splitlines()
 with (r/'source.sha256').open('w') as out:
  for path in tracked:
   p=source/path
   if p.is_file():out.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p)+'\n')
 (r/'input_checksums.json').write_text(json.dumps({name:hashlib.sha256((r/name).read_bytes()).hexdigest() for name in ('cohsex.in','centroids_charge.txt','centroids_current.txt','dipole.h5','kin_ion.h5')},indent=2)+'\n')
 command='python3 -u -m gw.gw_jax -i cohsex.in'
 if profile:
  shutil.copy2(profile,r/'profile_driver.py');command='python3 -u profile_driver.py -i cohsex.in'
 payload=f'#!/bin/bash\nset -euo pipefail\n{command} > driver.rank${{SLURM_PROCID}}.log 2>&1\n'
 if profile:
  nsys='/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/bin/nsys'
  payload=f'''#!/bin/bash
set -euo pipefail
export XLA_FLAGS="${{XLA_FLAGS:-}} --xla_dump_to=$PWD/xla_dump_rank${{SLURM_PROCID}}"
if [ "$SLURM_PROCID" = 0 ] && [ -x {nsys} ]; then
 {nsys} profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node --sample=none --cpuctxsw=none -t cuda,nvtx,osrt -o nsys_rank0 {command} > driver.rank0.log 2>&1
 {nsys} stats --report nvtx_gpu_proj_sum,nvtx_kern_sum,cuda_gpu_kern_sum --format csv --output stats nsys_rank0.nsys-rep > nsys_stats.log 2>&1
else
 {command} > driver.rank${{SLURM_PROCID}}.log 2>&1
fi
'''
 (r/'payload.sh').write_text(payload)
 (r/'driver.sh').write_text(f'#!/bin/bash\nset -euo pipefail\nsha256sum -c source.sha256 > source.rank${{SLURM_PROCID}}.txt\nexport LORRAX_DEBUG_PRINT=1\n./payload.sh\ntest -s eqp0.dat\ntest -s eqp1.dat\n')
 runner=(s/'runs/MoS2/41_bisp_parent_route_2026-09-05/prof_zw/26_after_units/runner.sh').read_text().replace('-N 1 -G 4 -n 4',f'-N {ranks//4} -G 4 -n {ranks}')
 (r/'runner.sh').write_text(runner)
 for p in r.glob('*.sh'):p.chmod(0o775)
 (r/'manifest.yaml').write_text(f'''run_id: {name}
system: MoS2
pipeline: lorrax_only
platform: perlmutter
variant_of: {template}
reuse_from_parent: [WFN.h5, MoS2.save, centroids_charge.txt, centroids_current.txt, dipole.h5, kin_ion.h5]
source: {{checkout: {source}, commit: {pin}, diff: source.diff, checksums: source.sha256}}
geometry: {{nodes: {ranks//4}, gpus_per_node: 4, ranks: {ranks}}}
allocator: BFC@0.85
pool: 57982945
steps:
  00_lorrax: {{state: pending}}
''')
 return r
if __name__=='__main__':
 paths=[prepare('47_P6_fresh_before',w,scale/'02_P_static_P4_fresh'),prepare('48_F6_fresh',f,scale/'03_F_static_P4_fresh')]
 Path('six_baseline_paths.json').write_text(json.dumps(list(map(str,paths)),indent=2)+'\n')
