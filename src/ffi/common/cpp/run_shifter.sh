#!/usr/bin/env bash
# run_shifter.sh — launch a command inside the LORRAX Shifter image with
# the staged NVIDIA HPC SDK (cuSOLVERMp + libcal) bind-mounted.
#
# Two modes:
#   (a) with SLURM_JOBID set: run via `srun ... shifter ... "$@"` on the
#       allocation's compute node(s) — use for test runs.
#   (b) without SLURM_JOBID: run `shifter ... "$@"` directly (login node) —
#       use only for compile steps that don't need a GPU.
#
# Environment:
#   LORRAX_FFI_NVHPC_DIR   host path to the staged nvhpc subset.  Default:
#                          /pscratch/sd/jackm/lorrax_nvhpc
#   LORRAX_FFI_IMAGE       shifter image tag.  Default: nvcr.io/nvidia/jax:25.04-py3
#   LORRAX_NGPU            for srun-mode, # GPUs to request (default 1)

set -euo pipefail

NVHPC_HOST="${LORRAX_FFI_NVHPC_DIR:-/pscratch/sd/j/jackm/lorrax_nvhpc}"
IMAGE="${LORRAX_FFI_IMAGE:-nvcr.io/nvidia/jax:25.04-py3}"
NGPU="${LORRAX_NGPU:-1}"

if [[ ! -d "${NVHPC_HOST}" ]]; then
    echo "run_shifter.sh: staged NVHPC dir ${NVHPC_HOST} does not exist."
    echo "  Create it with:"
    echo "    mkdir -p ${NVHPC_HOST}/25.5_cuda12.9/{math_libs/12.9/lib64,math_libs/12.9/targets/x86_64-linux/include,comm_libs/12.9/nccl/include}"
    echo "    cp -a /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/include/cusolverMp*.h \\"
    echo "          ${NVHPC_HOST}/25.5_cuda12.9/math_libs/12.9/targets/x86_64-linux/include/"
    echo "    cp -a /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64/{libcusolverMp*,libcal*} \\"
    echo "          ${NVHPC_HOST}/25.5_cuda12.9/math_libs/12.9/lib64/"
    echo "    cp -a /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/comm_libs/12.9/nccl/include/nccl.h \\"
    echo "          ${NVHPC_HOST}/25.5_cuda12.9/comm_libs/12.9/nccl/include/"
    exit 2
fi

# LORRAX site-packages + src, same as the lorrax modulefile.
: "${LORRAX_SRC:=/global/u2/j/jackm/software/lorrax/src}"
: "${LORRAX_SITE:=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site}"
PYPATH="${LORRAX_SRC}:${LORRAX_SITE}"

# NB: the staged NVHPC tree is bind-mounted at /lorrax_nvhpc inside the
# container; the .so's embedded RPATH points there.
SHIFTER_ARGS=(
    shifter --module=gpu --image="${IMAGE}"
    --volume="${NVHPC_HOST}:/lorrax_nvhpc"
    --env=PYTHONPATH="${PYPATH}"
    --env=HDF5_USE_FILE_LOCKING=FALSE
    --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
    --env=TF_GPU_ALLOCATOR=cuda_malloc_async
    --env=LD_LIBRARY_PATH=/lorrax_nvhpc/25.5_cuda12.9/math_libs/12.9/lib64
)

if [[ -n "${SLURM_JOBID:-}" && -z "${SLURM_STEP_ID:-}" ]]; then
    jobflag="--jobid=${SLURM_JOBID}"
    exec srun "${jobflag}" --gres=gpu:"${NGPU}" -N 1 -n "${NGPU}" \
        "${SHIFTER_ARGS[@]}" "$@"
else
    exec "${SHIFTER_ARGS[@]}" "$@"
fi
