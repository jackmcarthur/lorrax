"""Copy a common-basis deck and caches into an isolated phase-three leg."""
from pathlib import Path
import argparse,shutil
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('target',type=Path);p.add_argument('--profile',action='store_true');a=p.parse_args()
s,t=a.source.resolve(),a.target.resolve();w=Path(__file__).resolve().parents[5];t.mkdir()
base=w/'runs/MoS2/42_bisp_scale_2026-09-06/prof_s/16_P_restore_dynamic'
for n in ['cohsex.in','rule_replay.py','replay.json']:
 if (s/n).exists():shutil.copy2(s/n,t/n)
for n in ['tmp','replay_rules']:
 if (s/n).exists():shutil.copytree(s/n,t/n)
for n in ['WFN.h5','kin_ion.h5','dipole.h5','centroids_charge.txt','centroids_current.txt']:
 if (s/n).exists():(t/n).symlink_to((s/n).resolve())
for n in ['rankwrap.sh','run.sh']:shutil.copy2(base/n,t/n)
driver=(base/'driver.sh').read_text()
(t/'run_driver.py').write_text(('import rule_replay\n' if (t/'rule_replay.py').exists() else '')+'from gw import gw_jax\nfrom runtime import run_main_and_finalize\nrun_main_and_finalize(gw_jax.main)\n')
driver=driver.replace('-m gw.gw_jax','run_driver.py')
if a.profile:
 shutil.copy2(Path(__file__).with_name('tau_profile.py'),t/'run_driver.py')
 driver=driver[:driver.index('exec python3')]+'''export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=$PWD/xla_dump_rank${SLURM_PROCID}"
NSYS=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/bin/nsys
if [ "$SLURM_PROCID" = 0 ]; then
 "$NSYS" profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-flush-interval=1000 --cuda-graph-trace=node --sample=none --cpuctxsw=none -t cuda,nvtx,osrt -o nsys_rank0 python3 -u run_driver.py -i cohsex.in > driver.rank0.log 2>&1
 "$NSYS" stats --report nvtx_gpu_proj_sum,nvtx_kern_sum,cuda_gpu_kern_sum --format csv --output stats nsys_rank0.nsys-rep > nsys_stats.log 2>&1
else
 python3 -u run_driver.py -i cohsex.in > driver.rank${SLURM_PROCID}.log 2>&1
fi
'''
shutil.copy2(Path(__file__).with_name('compile_receipts.py'),t/'compile_receipts.py')
(t/'run_driver.py').write_text('import compile_receipts\n'+(t/'run_driver.py').read_text())
(t/'driver.sh').write_text(driver);(t/'driver.sh').chmod(0o755)
(t/'manifest.yaml').write_text(f'run_id: {t.name}\nsystem: {"Si" if "/Si/" in str(t) else "MoS2"}\npipeline: gwjax\nplatform: perlmutter\nvariant_of: {s}\nreused: copied zeta and rule caches; immutable input links\npool: 57988457\ngeometry: P4 one rank per GPU\nsteps:\n  sigma:\n    state: pending\n')
print(t)
