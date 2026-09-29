# shellcheck shell=bash
# ============================================================================
# Perlmutter machine settings for a GPU step launched from a clone with srun
# (docs/installation/perlmutter.md).  Source it in the shell that runs srun;
# srun passes the environment to every rank.  The CPU twin is cpu_mpi_env.sh.
#
#   source config/perlmutter/gpu_env.sh
#
# MPICH_GPU_SUPPORT_ENABLED=0.  The site's default craype-accel-nvidia80
# module exports 1, which makes Cray MPICH's MPI_Init demand the GTL library
# ("GPU_SUPPORT_ENABLED is requested, but GTL library is not linked") and
# abort.  Neither FFI leg links GTL: the host leg is CUDA-free and the CUDA
# leg's cuSOLVERMp/cuBLASMp communicate through NCCL.
# ============================================================================
export MPICH_GPU_SUPPORT_ENABLED=0
