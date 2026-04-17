#!/usr/bin/env bash
# run_shifter.sh — launch a command inside the LORRAX Shifter image with
# the staged NVIDIA HPC SDK (cuSOLVERMp + libcal) and parallel HDF5
# bind-mounted.
#
# Two modes:
#   (a) with SLURM_JOBID set: run via `srun ... shifter ... "$@"` on the
#       allocation's compute node(s) — use for test runs.
#   (b) without SLURM_JOBID: run `shifter ... "$@"` directly (login node) —
#       use only for compile steps that don't need a GPU.
#
# MPI STACK — pick one of:
#
#   LORRAX_PHDF5_MPI_STACK=openmpi   (default, verified, ~4.5 GB/s)
#     - container's HPC-X OpenMPI at /opt/hpcx/ompi satisfies libmpi.so.40
#     - phdf5 stage: conda-forge HDF5 1.14 linked against openmpi
#     - default LORRAX_FFI_PHDF5_DIR:
#         /pscratch/sd/$USER/lorrax_phdf5_openmpi/stage
#     - srun --mpi=pmix
#
#   LORRAX_PHDF5_MPI_STACK=mpich
#     - shifter --module=mpich bind-mounts Cray MPICH (libmpi.so.12)
#     - phdf5 stage: copy of cray-hdf5-parallel (1.12, libmpi_gnu_*.so.12)
#     - default LORRAX_FFI_PHDF5_DIR:
#         /pscratch/sd/$USER/lorrax_phdf5_cray/stage
#     - srun --mpi=pmi2
#     - KNOWN ISSUE: intermittent OOMs in ad_cray_write_coll.c:669 for
#       large collective writes; see src/ffi/PORTING.md.
#
# Other env:
#   LORRAX_FFI_NVHPC_DIR   host path to the staged nvhpc subset.
#   LORRAX_FFI_IMAGE       shifter image tag.  Default: nvcr.io/nvidia/jax:25.04-py3
#   LORRAX_NGPU            for srun-mode, # GPUs to request (default 1)
#   LORRAX_NNODES          for srun-mode, # nodes
#   LORRAX_NTASKS          for srun-mode, # total ranks

set -euo pipefail

MPI_STACK="${LORRAX_PHDF5_MPI_STACK:-openmpi}"

NVHPC_HOST="${LORRAX_FFI_NVHPC_DIR:-/pscratch/sd/j/jackm/lorrax_nvhpc}"
IMAGE="${LORRAX_FFI_IMAGE:-nvcr.io/nvidia/jax:25.04-py3}"
NGPU="${LORRAX_NGPU:-1}"
NTASKS="${LORRAX_NTASKS:-${NGPU}}"
NNODES="${LORRAX_NNODES:-1}"

# Stack-specific defaults: phdf5 stage path, Shifter module list, inside-
# container MPI include / lib paths, srun --mpi flavor.
case "${MPI_STACK}" in
    openmpi)
        PHDF5_DEFAULT="/pscratch/sd/j/jackm/lorrax_phdf5_openmpi/stage"
        SHIFTER_MODULES="gpu"
        MPI_LIB_DIR_CT="/opt/hpcx/ompi/lib"
        MPI_INCLUDE_DIR_CT="/opt/hpcx/ompi/include"
        MPI_TYPE_DEFAULT="pmix"
        ;;
    mpich)
        PHDF5_DEFAULT="/pscratch/sd/j/jackm/lorrax_phdf5_cray/stage"
        SHIFTER_MODULES="gpu,mpich"
        MPI_LIB_DIR_CT="/opt/udiImage/modules/mpich"
        MPI_INCLUDE_DIR_CT="/lorrax_phdf5/include"  # staged MPICH headers
        MPI_TYPE_DEFAULT="pmi2"
        ;;
    *)
        echo "run_shifter.sh: LORRAX_PHDF5_MPI_STACK=${MPI_STACK} not recognised; use 'openmpi' or 'mpich'." >&2
        exit 2
        ;;
esac

PHDF5_HOST="${LORRAX_FFI_PHDF5_DIR:-${PHDF5_DEFAULT}}"

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

: "${LORRAX_SRC:=/global/u2/j/jackm/software/lorrax/src}"
: "${LORRAX_SITE:=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site}"
PYPATH="${LORRAX_SRC}:${LORRAX_SITE}"

# Staged third-party trees are bind-mounted inside the container at
# stable paths: /lorrax_nvhpc and /lorrax_phdf5.
VOL_FLAGS=(--volume="${NVHPC_HOST}:/lorrax_nvhpc")
if [[ -d "${PHDF5_HOST}" ]]; then
    VOL_FLAGS+=(--volume="${PHDF5_HOST}:/lorrax_phdf5")
fi

# LD_LIBRARY_PATH: phdf5 stage first (so libhdf5 + its SONAME-shim
# symlinks are found), then NVHPC (cusolverMp), then the stack's MPI
# runtime dir.
LDLIB="/lorrax_phdf5/lib:/lorrax_nvhpc/25.5_cuda12.9/math_libs/12.9/lib64:${MPI_LIB_DIR_CT}"
if [[ "${MPI_STACK}" == mpich ]]; then
    # mpich module ships its own PMI/libfabric deps under dep/
    LDLIB="${LDLIB}:/opt/udiImage/modules/mpich/dep"
fi

SHIFTER_ARGS=(
    shifter --module="${SHIFTER_MODULES}" --image="${IMAGE}"
    "${VOL_FLAGS[@]}"
    --env=PYTHONPATH="${PYPATH}"
    --env=HDF5_USE_FILE_LOCKING=FALSE
    --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
    --env=TF_GPU_ALLOCATOR=cuda_malloc_async
    --env=LD_LIBRARY_PATH="${LDLIB}"
    # Expose the chosen stack's MPI include + lib paths to CMake so the
    # FFI build picks the right headers / library without guessing.
    --env=LORRAX_MPI_INCLUDE_DIR="${MPI_INCLUDE_DIR_CT}"
    --env=LORRAX_MPICH_LIB_DIR="${MPI_LIB_DIR_CT}"
    --env=LORRAX_PHDF5_MPI_STACK="${MPI_STACK}"
)

if [[ -n "${SLURM_JOBID:-}" && -z "${SLURM_STEP_ID:-}" ]]; then
    jobflag="--jobid=${SLURM_JOBID}"
    : "${LORRAX_MPI_TYPE:=${MPI_TYPE_DEFAULT}}"
    exec srun "${jobflag}" --mpi="${LORRAX_MPI_TYPE}" \
        --gres=gpu:"${NGPU}" -N "${NNODES}" -n "${NTASKS}" \
        "${SHIFTER_ARGS[@]}" "$@"
else
    exec "${SHIFTER_ARGS[@]}" "$@"
fi
