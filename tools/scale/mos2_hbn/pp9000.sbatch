#!/bin/bash
#SBATCH -J mos2hbn_pp9000
#SBATCH -A m2651
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -o /global/cfs/cdirs/m4598/jackm/mos2_hbn/2c-wfn9000/slurm-pp-%j.out
# pw2bgw + wfn2hdf for the 9000-band NSCF, on one CPU node (submit with --dependency=afterok:<nscf>).
set -uo pipefail
D=/global/cfs/cdirs/m4598/jackm/mos2_hbn/2c-wfn9000; cd $D
grep -q 'JOB DONE' wfn.out || { echo NO_NSCF; exit 1; }
module load espresso/7.5-libxc-7.0.0-cpu; export OMP_NUM_THREADS=2; rm -f hetero.wfc*
/usr/bin/time -v -o pp.time srun -N1 -n64 -c4 --cpu-bind=cores pw2bgw.x -npools 2 -in wfn.pp.in > wfn.pp.out 2>&1
grep -q 'JOB DONE' wfn.pp.out || { echo PP_FAILED; exit 1; }
module load gpu; module load berkeleygw/4.0-nvhpc-23.9
/usr/bin/time -v -o hdf.time srun -N1 -n1 -c64 wfn2hdf.x BIN WFN WFN.h5 > wfn2hdf.out 2>&1
grep -q 'Conversion from binary to HDF5 file complete!' wfn2hdf.out || { echo HDF_FAILED; exit 1; }
echo MF9000_DONE; date
