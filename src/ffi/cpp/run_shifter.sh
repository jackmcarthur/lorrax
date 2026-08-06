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
#   LORRAX_PHDF5_MPI_STACK=mpich   (default as of 2026-04-20)
#     - shifter --module=mpich bind-mounts Cray MPICH (libmpi.so.12)
#     - phdf5 stage: copy of cray-hdf5-parallel (1.12, libmpi_gnu_*.so.12)
#     - default LORRAX_FFI_PHDF5_DIR:
#         $HOME/software/lorrax_phdf5_cray/stage
#     - srun --mpi=pmi2
#     - Perf (1 node / 4 GPUs / 4.29 GB C128): 3.79 GB/s, +24% over
#       openmpi; at MoS2 3×3 scale within noise of openmpi.
#     - Requires post-2026-04-20 defaults (indep writes + non-coll meta)
#       to avoid ad_cray_write_coll.c:669 OOM at ≥ 1 GB/rank writes.
#
#   LORRAX_PHDF5_MPI_STACK=openmpi
#     - container's HPC-X OpenMPI at /opt/hpcx/ompi satisfies libmpi.so.40
#     - phdf5 stage: conda-forge HDF5 1.14 linked against openmpi
#     - default LORRAX_FFI_PHDF5_DIR:
#         $HOME/software/lorrax_phdf5_openmpi/stage
#     - srun --mpi=pmix
#     - Kept as fallback for non-Cray clusters.
#
# Other env:
#   LORRAX_FFI_NVHPC_DIR   host path to the staged nvhpc subset.
#   LORRAX_FFI_IMAGE       shifter image tag.  Default: nvcr.io/nvidia/jax:25.04-py3
#   LORRAX_NGPU            for srun-mode, # GPUs to request (default 1)
#   LORRAX_NNODES          for srun-mode, # nodes
#   LORRAX_NTASKS          for srun-mode, # total ranks

set -euo pipefail

MPI_STACK="${LORRAX_PHDF5_MPI_STACK:-mpich}"

NVHPC_HOST="${LORRAX_FFI_NVHPC_DIR:-$HOME/software/lorrax_nvhpc}"
IMAGE="${LORRAX_FFI_IMAGE:-nvcr.io/nvidia/jax:25.04-py3}"
NGPU="${LORRAX_NGPU:-1}"
NTASKS="${LORRAX_NTASKS:-${NGPU}}"
NNODES="${LORRAX_NNODES:-1}"

# Stack-specific defaults: phdf5 stage path, Shifter module list, inside-
# container MPI include / lib paths, srun --mpi flavor.
case "${MPI_STACK}" in
    openmpi)
        PHDF5_DEFAULT="$HOME/software/lorrax_phdf5_openmpi/stage"
        SHIFTER_MODULES="gpu"
        MPI_LIB_DIR_CT="/opt/hpcx/ompi/lib"
        MPI_INCLUDE_DIR_CT="/opt/hpcx/ompi/include"
        MPI_TYPE_DEFAULT="pmix"
        ;;
    mpich)
        PHDF5_DEFAULT="$HOME/software/lorrax_phdf5_cray/stage"
        SHIFTER_MODULES="gpu,mpich"
        MPI_LIB_DIR_CT="/opt/udiImage/modules/mpich"
        MPI_INCLUDE_DIR_CT="/lorrax_phdf5/include"  # staged MPICH headers
        # --mpi=cray_shasta is the PMI protocol that shifter-mpich's libmpi
        # speaks.  pmi2/pmix both produce singleton-MPI (each rank thinks
        # world_size==1) — observed while bringing up slate FFI.  phdf5
        # "worked" on pmi2 only because its default now uses independent
        # I/O (each rank writes its own shard with no collective handshake
        # — see src/ffi/cpp/phdf5/ctx.h), so it silently did the right
        # thing despite singleton MPI.
        MPI_TYPE_DEFAULT="cray_shasta"
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

# Derive the LORRAX src/ dir from this script's own location
# (src/ffi/cpp/run_shifter.sh -> ../.. = src) rather than a
# hardcoded per-user path.  Override LORRAX_SRC to point elsewhere.
_RUN_SHIFTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LORRAX_SRC:=$(cd "${_RUN_SHIFTER_DIR}/../.." && pwd)}"
# Supplemental site-packages bind-mounted into the container.  No default:
# this is site-specific (see config/perlmutter/site_config.sh
# LORRAX_SITE_PACKAGES).  Empty = none.
: "${LORRAX_SITE:=}"
if [[ -n "${LORRAX_SITE}" ]]; then
    PYPATH="${LORRAX_SRC}:${LORRAX_SITE}"
else
    PYPATH="${LORRAX_SRC}"
fi

# Staged third-party trees are bind-mounted inside the container at
# stable paths: /lorrax_nvhpc, /lorrax_phdf5, /lorrax_slate.
VOL_FLAGS=(--volume="${NVHPC_HOST}:/lorrax_nvhpc")
if [[ -d "${PHDF5_HOST}" ]]; then
    VOL_FLAGS+=(--volume="${PHDF5_HOST}:/lorrax_phdf5")
fi
# SLATE stage: Cray libsci + libmpi_gtl_cuda + libxpmem + liblustreapi.
# Populated by src/ffi/cpp/stage/slate_stage_cray.sh.  Skipped if absent.
: "${LORRAX_FFI_SLATE_DIR:=$HOME/software/lorrax_slate_cray/stage}"
SLATE_INSTALL_HOST="${LORRAX_SLATE_INSTALL_DIR:-$HOME/software/slate/install}"
if [[ -d "${LORRAX_FFI_SLATE_DIR}" ]]; then
    VOL_FLAGS+=(--volume="${LORRAX_FFI_SLATE_DIR}:/lorrax_slate")
fi

# Which cuSOLVERMp / cuBLASMp stage to RUN against, as a subpath under the
# /lorrax_nvhpc bind-mount.  SOURCE OF TRUTH: the site config's
# LORRAX_NVHPC_SUBPATH (config/perlmutter/site_config.sh).
#
# THIS MUST NOT BE HARDCODED, and was until 2026-08-05.  The staged tree
# carries SEVERAL cuSOLVERMp builds and they ALL export the same SONAME
# libcusolverMp.so.0, so whichever directory comes first in LD_LIBRARY_PATH
# silently decides which implementation runs — nothing fails, nothing warns.
# On Perlmutter 25.5_cuda12.9 is cuSolverMp 0.6.0, which returns WRONG
# getrf/getrs answers on any Px>1 AND Py>1 mesh (site_config.sh ~L80), while
# 0.7.2_cuda12.9 carries both the CAL->NCCL ABI fix and the race fix and is
# what site_config selects.  Hardcoding 25.5 here silently overrode that
# choice for every run launched through this script: a wrong-answer path,
# not a performance one.
: "${LORRAX_NVHPC_SUBPATH:=0.7.2_cuda12.9/math_libs/12.9/lib64}"
if [[ ! -e "${NVHPC_HOST}/${LORRAX_NVHPC_SUBPATH}/libcusolverMp.so" ]]; then
    echo "run_shifter.sh: LORRAX_NVHPC_SUBPATH='${LORRAX_NVHPC_SUBPATH}' does not" >&2
    echo "  name a cuSOLVERMp stage under ${NVHPC_HOST}." >&2
    echo "  Looked for: ${NVHPC_HOST}/${LORRAX_NVHPC_SUBPATH}/libcusolverMp.so" >&2
    echo "  Set LORRAX_NVHPC_SUBPATH (see config/perlmutter/site_config.sh)." >&2
    exit 2
fi
# LD_LIBRARY_PATH: slate install (libslate + bundled blaspp/lapackpp) + slate
# cray-stack stage (libsci etc.) come first; then phdf5 stage (libhdf5 +
# SONAME shims, including libmpi_gnu_123.so.12 reused by SLATE); the SELECTED
# NVHPC stage (cusolverMp/cublasmp); the 25.5 stage AFTER it purely as the
# fallback for libcal.so.0, which only that tree ships and which the .so
# carries in DT_NEEDED when built against it; stack's MPI runtime; darshan
# (libdarshan.so.0 via siteFs).
LDLIB="${SLATE_INSTALL_HOST}/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/${LORRAX_NVHPC_SUBPATH}:/lorrax_nvhpc/25.5_cuda12.9/math_libs/12.9/lib64:${MPI_LIB_DIR_CT}:/global/common/software/nersc9/darshan/default/lib"
if [[ "${MPI_STACK}" == mpich ]]; then
    # mpich module ships its own PMI/libfabric deps under dep/
    LDLIB="${LDLIB}:/opt/udiImage/modules/mpich/dep"
fi

# Shifter --module=mpich prepends /opt/udiImage/modules/mpich to
# LD_LIBRARY_PATH AFTER --env=LD_LIBRARY_PATH is applied, so that dir wins
# for any SONAME both stages provide.  Of those, only libmpi_gtl_cuda.so.0
# matters: shifter ships a CUDA-11-linked version (needs libcudart.so.11.0
# not in our CUDA-12 container) while our Cray-module stage has the
# CUDA-12-linked one.  LD_PRELOAD the staged version so it's already loaded
# when SLATE asks for it.
SLATE_PRELOAD=""
if [[ -f "${LORRAX_FFI_SLATE_DIR}/lib/libmpi_gtl_cuda.so.0" ]]; then
    SLATE_PRELOAD="/lorrax_slate/lib/libmpi_gtl_cuda.so.0"
fi

SHIFTER_ARGS=(
    shifter --module="${SHIFTER_MODULES}" --image="${IMAGE}"
    "${VOL_FLAGS[@]}"
    --env=PYTHONPATH="${PYPATH}"
    --env=HDF5_USE_FILE_LOCKING=FALSE
    --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
    --env=TF_GPU_ALLOCATOR=cuda_malloc_async
    --env=LD_LIBRARY_PATH="${LDLIB}"
    --env=LD_PRELOAD="${SLATE_PRELOAD}"
    # Activate Cray MPICH's GPU-Direct RDMA path so MPI_Send with a
    # device pointer goes GPU->GPU over Slingshot, not staged through
    # host RAM.  Required for any decent perf in SLATE (and other
    # MPI-based GPU libs); pairs with the libmpi_gtl_cuda LD_PRELOAD
    # above.  Shifter strips the host-set env, so we re-set it here.
    --env=MPICH_GPU_SUPPORT_ENABLED=1
    # Expose the chosen stack's MPI include + lib paths to CMake so the
    # FFI build picks the right headers / library without guessing.
    --env=LORRAX_MPI_INCLUDE_DIR="${MPI_INCLUDE_DIR_CT}"
    --env=LORRAX_MPICH_LIB_DIR="${MPI_LIB_DIR_CT}"
    --env=LORRAX_PHDF5_MPI_STACK="${MPI_STACK}"
)

if [[ -n "${SLURM_JOBID:-}" && -z "${SLURM_STEP_ID:-}" ]]; then
    jobflag="--jobid=${SLURM_JOBID}"
    : "${LORRAX_MPI_TYPE:=${MPI_TYPE_DEFAULT}}"
    # Per-rank GPU isolation via CUDA_VISIBLE_DEVICES=$SLURM_LOCALID —
    # NERSC-documented "Method 3" at https://docs.nersc.gov/jobs/affinity/.
    # Needed so SLATE's blas::get_device_count() returns 1 per rank
    # (matching JAX's 1-process-per-GPU model).  --gpus-per-task=1 would
    # also achieve this via cgroups but it breaks JAX's distributed
    # topology sync (JAX queries each process's device table assuming all
    # ranks expose the same local ordinals).
    # For SLATE, each rank needs CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    # (1-GPU-per-process), and JAX needs local_device_ids=[0].  Opt-in via
    # LORRAX_SELECT_GPU=1 so phdf5/cusolvermp (which use the default
    # all-GPUs-visible model) stay unchanged.
    SRUN_WRAPPER=()
    if [[ "${LORRAX_SELECT_GPU:-0}" == "1" ]]; then
        SRUN_WRAPPER=("$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/select_gpu.sh")
    fi
    # in_container.sh runs inside shifter and re-exports
    # MPICH_GPU_SUPPORT_ENABLED=1 (which shifter's --module=mpich
    # explicitly unsets per /etc/shifter/udiRoot.conf).  Required so
    # MPI calls with device pointers go GPU->GPU via Slingshot
    # GPU-Direct RDMA.  Path is the host path; resolves the same
    # inside the container via /global/u2 siteFs.
    IN_CONTAINER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/in_container.sh"
    exec srun "${jobflag}" --mpi="${LORRAX_MPI_TYPE}" \
        --gres=gpu:"${NGPU}" -N "${NNODES}" -n "${NTASKS}" \
        "${SRUN_WRAPPER[@]}" "${SHIFTER_ARGS[@]}" "${IN_CONTAINER}" "$@"
else
    exec "${SHIFTER_ARGS[@]}" "$@"
fi
