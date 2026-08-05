#!/bin/bash
# Module-free srun+shifter runner for the zeta-ridge worktree
# (adapted from runs/MoS2/A_bse_w0_resolvent_2026-07-16/lxrun_free.sh;
# PYTHONPATH points at THIS worktree's src).  1 GPU, 1 rank.
#   usage: JID=<jobid> ./lxrun_ridge.sh <workdir> <python args...>
set -euo pipefail

JID="${JID:?set JID to the salloc job id}"
WD="$1"; shift

IMAGE="nvcr.io/nvidia/jax:25.04-py3"
SRC=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_ridge_wt/src
TESTS=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_ridge_wt
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
DEPS=/pscratch/sd/j/jackm/lorrax_sandbox/sources
NVHPC=/global/homes/j/jackm/software/lorrax_nvhpc
PHDF5=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage
SLATE=/global/homes/j/jackm/software/lorrax_slate_cray/stage
SLATE_INSTALL=/global/homes/j/jackm/software/slate/install
MPICH=/opt/udiImage/modules/mpich
DARSHAN=/global/common/software/nersc9/darshan/default/lib

PYPATH="$SRC:$SITE:$DEPS"
LDLIB="$SLATE_INSTALL/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:$NVHPC/0.7.2_cuda12.9/math_libs/12.9/lib64:$MPICH:$MPICH/dep:$DARSHAN"
JAXCACHE="${SCRATCH:-$HOME}/.jax_cache"

srun --jobid="$JID" --overlap -N1 -n1 --gres=gpu:1 --cpu-bind=cores \
  --chdir="$WD" \
  shifter --image="$IMAGE" --module=gpu,mpich \
    --volume="$NVHPC:/lorrax_nvhpc" \
    --volume="$PHDF5:/lorrax_phdf5" \
    --volume="$SLATE:/lorrax_slate" \
    --env=PYTHONPATH="$PYPATH" \
    --env=HDF5_USE_FILE_LOCKING=FALSE \
    --env=LD_LIBRARY_PATH="$LDLIB" \
    --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 \
    --env=MPICH_GPU_SUPPORT_ENABLED=1 \
    --env=JAX_COMPILATION_CACHE_DIR="$JAXCACHE" \
    --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    --env=TF_GPU_ALLOCATOR=cuda_malloc_async \
    "$@"
