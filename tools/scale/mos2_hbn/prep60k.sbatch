#!/bin/bash
#SBATCH -J mos2hbn_prep60k
#SBATCH -A m2651_g
#SBATCH -q premium
#SBATCH -C gpu&hbm80g
#SBATCH -N 4
#SBATCH --gpus-per-node=4
#SBATCH -t 01:00:00
#SBATCH -o /global/cfs/cdirs/m4598/jackm/mos2_hbn/4-gw60k/slurm-prep-%j.out
# Preprocessing for the N_mu = 60000 / 9010-band gn_ppm GW, from a fresh origin/main clone on CFS:
# build the FFI pair, then kmeans (60000), kin_ion, dipole at P = 16. Each stage is its own srun
# step (sacct MaxRSS per step); an overlapping nvidia-smi step logs device memory.
set -uo pipefail
R=/global/cfs/cdirs/m4598/jackm/mos2_hbn/4-gw60k; cd $R
export LORRAX=/global/cfs/cdirs/m4598/jackm/lorrax_main_1027
export LORRAX_CHECKOUT=$LORRAX LORRAX_ROOT=$LORRAX PYTHONPATH=$LORRAX/src
PY=$LORRAX/.venv/bin/python; export PATH=$(dirname $PY):$PATH
git -C $LORRAX rev-parse HEAD > lorrax_head.txt
t() { local n=$1; shift; local s=$(date +%s); "$@" > $n.out 2>&1; local rc=$?
      echo "$n rc=$rc wall=$(( $(date +%s)-s ))s" | tee -a stages.txt; return $rc; }
if [ ! -s $LORRAX/src/ffi/cpp/build/liblorrax_ffi.so ]; then
  t ffi_build srun -N 1 -n 1 -c 64 --gres=none bash -c "cd $LORRAX && bash config/perlmutter/build_ffi_host.sh && bash config/perlmutter/build_ffi_cuda.sh" || exit 1
fi
source $LORRAX/config/perlmutter/gpu_env.sh
export MPICH_GPU_SUPPORT_ENABLED=0
L=(srun -N 4 -n 16 -c 16 --threads-per-core=1 --cpu-bind=threads --gpus-per-node=4
   $LORRAX/src/ffi/cpp/select_gpu.sh $PY -u -m)
srun --overlap -N 4 --ntasks-per-node=1 -c 1 --mem=0 bash -c 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 15 > gpu_mem_prep_$(hostname).csv' & SMI=$!
if [ ! -s centroids.txt ]; then
  t kmeans "${L[@]}" centroid.kmeans_cli 60000 -i gnppm.in --seed 42 --fit-window 0:832,0:9010 || exit 1
  ln -sfn $(ls centroids_frac_*.txt | head -1) centroids.txt
fi
[ -s kin_ion.h5 ] || t kin_ion "${L[@]}" gw.kin_ion_io -i gnppm.in || exit 1
[ -s dipole.h5 ] || t dipole "${L[@]}" psp.get_dipole_mtxels -i gnppm.in || exit 1
kill $SMI 2>/dev/null
sacct -j $SLURM_JOB_ID -n -o jobid%18,jobname%14,elapsed,maxrss,maxrssnode > sacct_prep.txt
echo PREP60K_DONE; date
