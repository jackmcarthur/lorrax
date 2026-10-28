#!/bin/bash
#SBATCH -J mos2hbn_gw60k
#SBATCH -A m2651_g
#SBATCH -q premium
#SBATCH -C gpu&hbm80g
#SBATCH -N 9
#SBATCH --gpus-per-node=4
#SBATCH -t 01:00:00
#SBATCH -o /global/cfs/cdirs/m4598/jackm/mos2_hbn/4-gw60k/slurm-gw-%j.out
# Owner-approved scale test, GW only: gn_ppm one-shot at N_mu = 60000, number_bands 9010 (spinor;
# 9000 splits a 0.73 meV pair), k 2x2x1, P = 36 = 6 x 6 on 9 nodes (owner choice; corrected planner 427e6e45c: zeta C assembly 49.2 GB/rank binds). Centroids, kin_ion and dipole
# come from prep60k.sbatch (same checkout); this job refuses if any is missing. Premium over 4
# nodes: the owner submits it from their terminal (lx-budget guard).
set -uo pipefail
R=/global/cfs/cdirs/m4598/jackm/mos2_hbn/4-gw60k; cd $R
for f in centroids.txt kin_ion.h5 dipole.h5 WFN.h5; do [ -s $f ] || { echo "MISSING $f: run prep60k.sbatch first"; exit 1; }; done
export LORRAX=/global/cfs/cdirs/m4598/jackm/lorrax_main_1027
export LORRAX_CHECKOUT=$LORRAX LORRAX_ROOT=$LORRAX PYTHONPATH=$LORRAX/src
PY=$LORRAX/.venv/bin/python; export PATH=$(dirname $PY):$PATH
source $LORRAX/config/perlmutter/gpu_env.sh
export MPICH_GPU_SUPPORT_ENABLED=0
[ "$(git -C $LORRAX rev-parse HEAD)" = "$(cat lorrax_head.txt)" ] || { echo "checkout moved since prep"; exit 1; }
srun --overlap -N 9 --ntasks-per-node=1 -c 1 --mem=0 bash -c 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 15 > gpu_mem_gw_$(hostname).csv' & SMI=$!
s=$(date +%s)
srun -N 9 -n 36 --ntasks-per-node=4 -c 16 --threads-per-core=1 --cpu-bind=threads --gpus-per-node=4 \
    $LORRAX/src/ffi/cpp/select_gpu.sh $PY -u -m gw.gw_jax -i gnppm.in > gnppm.out 2>&1
echo "gnppm rc=$? wall=$(( $(date +%s)-s ))s" | tee -a stages.txt
kill $SMI 2>/dev/null
sacct -j $SLURM_JOB_ID -n -o jobid%18,jobname%14,elapsed,maxrss,maxrssnode > sacct_gw.txt
echo GW60K_DONE; date
