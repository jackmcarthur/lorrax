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
# MPI STACK — controlled by LORRAX_MPI_STACK (default: cray_mpich).
# Stack constants are read from config/mpi_stacks/${LORRAX_MPI_STACK}.sh,
# which is the single source of truth for LORRAX_MPI_TYPE_DEFAULT,
# LORRAX_SHIFTER_MODULES, LORRAX_MPI_LIB_DIR_CT, LORRAX_MPI_INCLUDE_DIR_CT,
# and LORRAX_GTL_PRELOAD.  To add a new stack, add a .sh + .cmake pair in
# config/mpi_stacks/ — do not add another case block here.
#
# Currently supported stacks:
#   cray_mpich (default) — Shifter --module=mpich, --mpi=cray_shasta.
#                          Used on Perlmutter (Cray Slingshot, A100).
#                          Perf at 1 node / 4 GPUs / 4.29 GB C128:
#                            3.79 GB/s (+24% over openmpi baseline).
#   openmpi              — stub; not yet wired (see Blitz #3 follow-up).
#                          Intended for Apptainer-based clusters.
#
# Other env:
#   LORRAX_REPO            root of the lorrax checkout (auto-detected)
#   LORRAX_FFI_NVHPC_DIR   host path to the staged nvhpc subset.
#   LORRAX_FFI_IMAGE       shifter image tag.  Default: nvcr.io/nvidia/jax:25.04-py3
#   LORRAX_NGPU            for srun-mode, # GPUs to request (default 1)
#   LORRAX_NNODES          for srun-mode, # nodes
#   LORRAX_NTASKS          for srun-mode, # total ranks

set -euo pipefail

# Locate the repo root so we can find config/mpi_stacks/.
# LORRAX_REPO may be set by the caller; if not, derive it from this script's
# own path (src/ffi/common/cpp/ → four levels up).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LORRAX_REPO:="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"}"

MPI_STACK="${LORRAX_MPI_STACK:-cray_mpich}"
STACK_FILE="${LORRAX_REPO}/config/mpi_stacks/${MPI_STACK}.sh"
if [[ ! -f "${STACK_FILE}" ]]; then
    echo "run_shifter.sh: no stack file for LORRAX_MPI_STACK=${MPI_STACK}" >&2
    echo "  Expected: ${STACK_FILE}" >&2
    echo "  Available stacks: $(ls "${LORRAX_REPO}/config/mpi_stacks/"*.sh 2>/dev/null | xargs -n1 basename | sed 's/\.sh//' | tr '\n' ' ')" >&2
    exit 2
fi
# Source the stack file; it exports LORRAX_MPI_STACK_NAME, LORRAX_MPI_TYPE_DEFAULT,
# LORRAX_SHIFTER_MODULES, LORRAX_MPI_LIB_DIR_CT, LORRAX_MPI_INCLUDE_DIR_CT,
# LORRAX_GTL_PRELOAD.
# shellcheck source=/dev/null
source "${STACK_FILE}"

# Validate that the stack file set the required variables (catches stub stacks
# like openmpi.sh that are not yet wired).
for _var in LORRAX_MPI_TYPE_DEFAULT LORRAX_SHIFTER_MODULES LORRAX_MPI_LIB_DIR_CT LORRAX_MPI_INCLUDE_DIR_CT; do
    if [[ -z "${!_var:-}" ]]; then
        echo "run_shifter.sh: stack '${MPI_STACK}' did not set ${_var}; is it a stub?" >&2
        exit 2
    fi
done
unset _var

# Stack-specific phdf5 stage path default (not in the .sh because it embeds
# $USER and is deployment-specific rather than stack-architecture-specific).
case "${MPI_STACK}" in
    cray_mpich)
        PHDF5_DEFAULT="/pscratch/sd/${USER}/lorrax_phdf5_cray/stage" ;;
    openmpi)
        PHDF5_DEFAULT="/pscratch/sd/${USER}/lorrax_phdf5_openmpi/stage" ;;
    *)
        PHDF5_DEFAULT="/pscratch/sd/${USER}/lorrax_phdf5_${MPI_STACK}/stage" ;;
esac

PHDF5_HOST="${LORRAX_FFI_PHDF5_DIR:-${PHDF5_DEFAULT}}"
SHIFTER_MODULES="${LORRAX_SHIFTER_MODULES}"
MPI_LIB_DIR_CT="${LORRAX_MPI_LIB_DIR_CT}"
MPI_INCLUDE_DIR_CT="${LORRAX_MPI_INCLUDE_DIR_CT}"
MPI_TYPE_DEFAULT="${LORRAX_MPI_TYPE_DEFAULT}"

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
# stable paths: /lorrax_nvhpc, /lorrax_phdf5, /lorrax_slate.
VOL_FLAGS=(--volume="${NVHPC_HOST}:/lorrax_nvhpc")
if [[ -d "${PHDF5_HOST}" ]]; then
    VOL_FLAGS+=(--volume="${PHDF5_HOST}:/lorrax_phdf5")
fi
# SLATE stage: Cray libsci + libmpi_gtl_cuda + libxpmem + liblustreapi.
# Populated by src/ffi/slate/scripts/stage_cray.sh.  Skipped if absent.
: "${LORRAX_FFI_SLATE_DIR:=/pscratch/sd/j/jackm/lorrax_slate_cray/stage}"
SLATE_INSTALL_HOST="${LORRAX_SLATE_INSTALL_DIR:-/global/homes/j/jackm/software/slate/install}"
if [[ -d "${LORRAX_FFI_SLATE_DIR}" ]]; then
    VOL_FLAGS+=(--volume="${LORRAX_FFI_SLATE_DIR}:/lorrax_slate")
fi

# LD_LIBRARY_PATH: slate install (libslate + bundled blaspp/lapackpp) + slate
# cray-stack stage (libsci etc.) come first; then phdf5 stage (libhdf5 +
# SONAME shims, including libmpi_gnu_123.so.12 reused by SLATE); NVHPC
# (cusolverMp); stack's MPI runtime; darshan (libdarshan.so.0 via siteFs).
LDLIB="${SLATE_INSTALL_HOST}/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/25.5_cuda12.9/math_libs/12.9/lib64:${MPI_LIB_DIR_CT}:/global/common/software/nersc9/darshan/default/lib"
if [[ "${MPI_STACK}" == cray_mpich ]]; then
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
# LORRAX_GTL_PRELOAD is the container-side path defined in the stack file;
# we only activate it if the host-side staging directory exists.
SLATE_PRELOAD=""
_gtl_ct="${LORRAX_GTL_PRELOAD:-}"
if [[ -n "${_gtl_ct}" && -d "${LORRAX_FFI_SLATE_DIR}" ]]; then
    SLATE_PRELOAD="${_gtl_ct}"
fi
unset _gtl_ct

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
    --env=LORRAX_MPI_STACK="${MPI_STACK}"
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
