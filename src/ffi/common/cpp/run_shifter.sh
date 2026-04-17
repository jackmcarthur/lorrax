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
# phdf5 stage: a copy of the cluster's cray-hdf5-parallel (or equivalent
# parallel-HDF5) module, kept on /pscratch so Shifter is willing to
# bind-mount it (NERSC's udiRoot.conf won't accept --volume sources
# under /global/homes).  Regenerate with
#   ~/software/lorrax_phdf5_cray/rebuild_stage.sh
# The stage only needs to contain {bin,include,lib} + a shim symlink
# libmpi_gnu_*.so.12 -> /opt/udiImage/modules/mpich/libmpi.so.12 so
# the host HDF5's NEEDED libmpi resolves to the shifter mpich module.
PHDF5_HOST="${LORRAX_FFI_PHDF5_DIR:-/pscratch/sd/j/jackm/lorrax_phdf5_cray/stage}"
IMAGE="${LORRAX_FFI_IMAGE:-nvcr.io/nvidia/jax:25.04-py3}"
NGPU="${LORRAX_NGPU:-1}"
# LORRAX_NTASKS = total ranks across the whole job (world_size).
# LORRAX_NNODES = # of nodes (4 GPUs per Perlmutter node).  For multi-
# process single-node JAX default NNODES=1 and NTASKS=NGPU.  For 16-GPU
# 4-node jobs use LORRAX_NNODES=4 LORRAX_NGPU=4 LORRAX_NTASKS=16.
NTASKS="${LORRAX_NTASKS:-${NGPU}}"
NNODES="${LORRAX_NNODES:-1}"

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
    # --module=mpich bind-mounts Cray MPICH (MPICH-ABI libmpi.so.12 + PMI
    # + libfabric + libcxi deps) at /opt/udiImage/modules/mpich.  This is
    # what lets the bind-mounted cray-hdf5-parallel load inside the JAX
    # container and also gives us Cray's Lustre-aware collective buffering
    # for H5Dwrite.  --module=gpu bind-mounts libcuda / NCCL user-space.
    shifter --module=gpu,mpich --image="${IMAGE}"
    "${VOL_FLAGS[@]}"
    --env=PYTHONPATH="${PYPATH}"
    --env=HDF5_USE_FILE_LOCKING=FALSE
    --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
    --env=TF_GPU_ALLOCATOR=cuda_malloc_async
    # Library search order: phdf5 stage first (so the staged cray
    # libhdf5 is found + its shim symlink for libmpi_gnu_*.so.12 resolves
    # back to the shifter mpich module's libmpi.so.12).  Then NVHPC
    # (cusolverMp).  Then the shifter mpich module's own paths, which
    # the module's siteEnvPrepend already adds, but we list them here
    # too in case the caller overrides LD_LIBRARY_PATH elsewhere.
    --env=LD_LIBRARY_PATH=/lorrax_phdf5/lib:/lorrax_nvhpc/25.5_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep
)

if [[ -n "${SLURM_JOBID:-}" && -z "${SLURM_STEP_ID:-}" ]]; then
    jobflag="--jobid=${SLURM_JOBID}"
    # Cray MPICH (from shifter --module=mpich) bootstraps via PMI2, not
    # PMIx.  The shifter mpich module bind-mounts libpmi / libpmi2 from
    # the Cray PE; NERSC's Slurm on Perlmutter supports pmi2 natively.
    : "${LORRAX_MPI_TYPE:=pmi2}"
    exec srun "${jobflag}" --mpi="${LORRAX_MPI_TYPE}" \
        --gres=gpu:"${NGPU}" -N "${NNODES}" -n "${NTASKS}" \
        "${SHIFTER_ARGS[@]}" "$@"
else
    exec "${SHIFTER_ARGS[@]}" "$@"
fi
