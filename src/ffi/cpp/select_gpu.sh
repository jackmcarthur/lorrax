#!/usr/bin/env bash
# Per-rank GPU selector for SLATE's 1-device-per-process model.  Sets
# CUDA_VISIBLE_DEVICES=$SLURM_LOCALID (NERSC's Method 3 from
# https://docs.nersc.gov/jobs/affinity/) then execs its arguments in
# place, keeping PMI environment intact for mpich collective bootstrap.
#
# Used by run_shifter.sh between srun and the user command.  Unlike an
# inline `bash -c 'export ...; exec "$@"'`, writing this as a script
# file means srun invokes it as a single process image (so PMI_* env
# vars, which mpich's libmpi reads at MPI_Init_thread, aren't stripped
# by an intermediate subshell).
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
exec "$@"
