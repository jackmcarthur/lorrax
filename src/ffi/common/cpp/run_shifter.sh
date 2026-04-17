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
PHDF5_HOST="${LORRAX_FFI_PHDF5_DIR:-/pscratch/sd/j/jackm/lorrax_phdf5/stage}"
IMAGE="${LORRAX_FFI_IMAGE:-nvcr.io/nvidia/jax:25.04-py3}"
NGPU="${LORRAX_NGPU:-1}"
# LORRAX_NTASKS controls the # of srun tasks.  For multi-process JAX
# (cusolverMp) it should equal NGPU (1 rank per GPU).  For single-process
# multi-GPU (cusolverMg), set LORRAX_NTASKS=1 LORRAX_NGPU=4.
NTASKS="${LORRAX_NTASKS:-${NGPU}}"

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

# NB: staged third-party trees are bind-mounted inside the container; the
# .so's embedded RPATH points at those bind-mount targets.  Layout:
#   /lorrax_nvhpc        ← NVHPC SDK subset (cuSOLVERMp + libcal + headers)
#   /lorrax_phdf5        ← parallel HDF5 stage (conda-forge openmpi variant)
VOL_FLAGS=(--volume="${NVHPC_HOST}:/lorrax_nvhpc")
if [[ -d "${PHDF5_HOST}" ]]; then
    VOL_FLAGS+=(--volume="${PHDF5_HOST}:/lorrax_phdf5")
fi

SHIFTER_ARGS=(
    shifter --module=gpu --image="${IMAGE}"
    "${VOL_FLAGS[@]}"
    --env=PYTHONPATH="${PYPATH}"
    --env=HDF5_USE_FILE_LOCKING=FALSE
    --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
    --env=TF_GPU_ALLOCATOR=cuda_malloc_async
    # Library search order: phdf5 stage first (so libhdf5 finds its
    # libsz/libaec bundled alongside), then NVHPC (cusolverMp), then
    # container's OpenMPI (satisfies libmpi.so.40 NEEDED in libhdf5.so).
    --env=LD_LIBRARY_PATH=/lorrax_phdf5/lib:/lorrax_nvhpc/25.5_cuda12.9/math_libs/12.9/lib64:/opt/hpcx/ompi/lib
)

if [[ -n "${SLURM_JOBID:-}" && -z "${SLURM_STEP_ID:-}" ]]; then
    jobflag="--jobid=${SLURM_JOBID}"
    # --mpi=pmix lets the container's OpenMPI bootstrap via PMIx (NERSC
    # Perlmutter's Slurm ships pmix_v4).  Without this, OpenMPI's PMI
    # client can't reach Slurm's PMI server and MPI_Init fails.
    exec srun "${jobflag}" --mpi=pmix \
        --gres=gpu:"${NGPU}" -N 1 -n "${NTASKS}" \
        "${SHIFTER_ARGS[@]}" "$@"
else
    exec "${SHIFTER_ARGS[@]}" "$@"
fi
