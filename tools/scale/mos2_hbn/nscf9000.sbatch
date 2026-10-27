#!/bin/bash
#SBATCH -J mos2hbn_nscf9000
#SBATCH -A m2651_g
#SBATCH -q premium
#SBATCH -C gpu&hbm80g
#SBATCH -N 4
#SBATCH --gpus-per-node=4
#SBATCH -t 02:00:00
#SBATCH -o /global/cfs/cdirs/m4598/jackm/mos2_hbn/2c-wfn9000/slurm-nscf-%j.out
# 9000-band NSCF (nbnd 9200), GPU QE on 4 nodes: -nk 2 = one pool per IBZ k, 8 GPUs per pool.
# psi is ~66 GB per k at 9200 bands, ~8 GB per GPU here, so Davidson ndim 2 fits (as on the
# 3400-band run). disk_io low (GPU QE's direct-access wfc records fail at a few GB per rank).
# wfn.in: diago_full_acc = .false., conv_thr = 1e-6 (owner OK). Conversion: pp9000.sbatch.
set -uo pipefail
W=/global/cfs/cdirs/m4598/jackm/mos2_hbn; D=$W/2c-wfn9000; cd $D
mkdir -p hetero.save
cp $W/1-scf/hetero.save/charge-density.* $W/1-scf/hetero.save/data-file-schema.xml $W/1-scf/hetero.save/*.upf hetero.save/
module load espresso/7.5-libxc-7.0.0-gpu-cu13; export OMP_NUM_THREADS=8
srun -N4 --ntasks-per-node=1 bash -c 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 60 > gpu_mem_$(hostname).csv' & SMI=$!
date; /usr/bin/time -v -o wfn.time srun -N4 -n16 -c32 --cpu-bind=cores --gpus-per-node=4 \
    pw.x -npools 2 -ndiag 1 -in wfn.in > wfn.out 2>&1
kill $SMI 2>/dev/null
grep -q 'JOB DONE' wfn.out || { echo NSCF_FAILED; exit 1; }; echo NSCF_DONE; date
