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
#     - phdf5 stage: copy of cray-hdf5-parallel/1.14.3.7
#       (SOVERSION 310, libmpi_gnu_123.so.12 -> the mpich shim)
#     - default LORRAX_FFI_PHDF5_DIR:
#         $HOME/software/lorrax_phdf5_cray_1.14.3.7/stage
#       The version is in the path deliberately.  This tree is what the
#       DEVICE leg links (CMakeLists.txt defaults HDF5_ROOT to the mount),
#       and build_ffi_host.sh links the module of the same version on bare
#       metal.  They were 1.12 and 1.14 for months and nothing compared
#       them; GATE 7 now does.
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
#   LORRAX_PLATFORM        gpu (default) | cpu.  Sets MPICH_GPU_SUPPORT_ENABLED
#                          for the launch; inferred from JAX_PLATFORMS when
#                          unset, so `JAX_PLATFORMS=cpu run_shifter.sh ...`
#                          is enough for the CPU leg.
#   LORRAX_MPICH_GPU_SUPPORT  0|1 — override the above outright.
#   LORRAX_FFI_NVHPC_DIR   host path to the staged nvhpc subset.
#   LORRAX_FFI_IMAGE       shifter image tag.  Default:
#                          ghcr.io/nvidia/jax:jax-2025-07-21 (jax 0.7.0,
#                          CUDA 12.9) -- the tag site_config.sh runs.
#   LORRAX_NGPU            for srun-mode, # GPUs to request (default 1)
#   LORRAX_NNODES          for srun-mode, # nodes
#   LORRAX_NTASKS          for srun-mode, # total ranks

set -euo pipefail

MPI_STACK="${LORRAX_PHDF5_MPI_STACK:-mpich}"

# Which platform this launch is FOR.  Decides MPICH_GPU_SUPPORT_ENABLED
# both here and in in_container.sh (see the long note there): 1 on GPU for
# GPU-Direct RDMA, 0 on CPU because Cray MPICH aborts at MPI_Init_thread
# trying to dlopen the GTL against a CUDA runtime a JAX_PLATFORMS=cpu run
# does not have.  Derived from JAX_PLATFORMS when not stated, so the CPU
# leg needs no second variable; "gpu" when neither is set.
: "${LORRAX_PLATFORM:=}"
if [[ -z "${LORRAX_PLATFORM}" ]]; then
    _jp="${JAX_PLATFORMS:-}"
    case "${_jp%%,*}" in
        cpu|host) LORRAX_PLATFORM=cpu ;;
        *)        LORRAX_PLATFORM=gpu ;;
    esac
fi
case "${LORRAX_PLATFORM}" in
    cpu|host) MPICH_GPU_SUPPORT=0 ;;
    gpu|cuda) MPICH_GPU_SUPPORT=1 ;;
    *) echo "run_shifter.sh: LORRAX_PLATFORM='${LORRAX_PLATFORM}' not recognised; use 'gpu' or 'cpu'." >&2
       exit 2 ;;
esac
# An explicit LORRAX_MPICH_GPU_SUPPORT outranks the platform inference.
case "${LORRAX_MPICH_GPU_SUPPORT:-}" in
    0|1) MPICH_GPU_SUPPORT="${LORRAX_MPICH_GPU_SUPPORT}" ;;
    "")  ;;
    *) echo "run_shifter.sh: LORRAX_MPICH_GPU_SUPPORT='${LORRAX_MPICH_GPU_SUPPORT}' is not 0 or 1." >&2
       exit 2 ;;
esac

NVHPC_HOST="${LORRAX_FFI_NVHPC_DIR:-$HOME/software/lorrax_nvhpc}"
# config/perlmutter/site_config.sh owns this tag and the reason for it.
IMAGE="${LORRAX_FFI_IMAGE:-ghcr.io/nvidia/jax:jax-2025-07-21}"
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
        PHDF5_DEFAULT="$HOME/software/lorrax_phdf5_cray_1.14.3.7/stage"
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
# FFTW3 stage: the DOUBLE-PRECISION SERIAL cray-fftw engine, and nothing
# else.  Populated by src/ffi/cpp/stage/fftw_stage_cray.sh.  Skipped if
# absent (the image is then simply an MKL-less, FFTW-less site and the flat-k
# handler refuses at first use, naming every candidate it tried).
#
# WHY A MOUNT IS THE ONLY REPAIR.  The flat-k FFT handler resolves the FFTW3
# advanced interface at RUN time and deliberately keeps it out of DT_NEEDED
# (GATE 5).  Run-time resolution still needs a file to resolve TO, and
# measured 2026-08-06 in-container on a compute node, this image has none:
# `find /usr /usr/local /opt /lib /lib64 /lorrax_{slate,phdf5,nvhpc} -name
# 'libfftw3*'` is EMPTY, every ladder candidate returns "cannot open shared
# object file", and /opt/cray/pe does not exist at all — so the host .so's
# RPATH, which is what makes the BARE-HOST leg work, resolves nothing here.
# Shifter at NERSC will not --volume a /opt/ system path (udiRoot siteFs), so
# bind-mounting /opt/cray/pe/fftw directly is not available either.
#
# WHAT MUST NOT BE DONE INSTEAD: this image DOES ship libcufftw.so.11 in its
# ldconfig cache, and it exports fftw_plan_many_dft / fftw_execute_dft /
# fftw_destroy_plan — all three names the ladder binds.  Pointing
# LORRAX_FFTW3_SO at it makes every FFT cell go green while the HOST
# handler's transforms run on the GPU.  src/ffi/cpp/gate_one_fftw.sh
# (GATE 8) is what tells those two states apart.
: "${LORRAX_FFI_FFTW_DIR:=$HOME/software/lorrax_fftw_cray/stage}"
if [[ -d "${LORRAX_FFI_FFTW_DIR}" ]]; then
    VOL_FLAGS+=(--volume="${LORRAX_FFI_FFTW_DIR}:/lorrax_fftw")
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
#
# THE 25.5 FALLBACK IS NOW VESTIGIAL FOR THE SUPPORTED BUILD, and is kept
# only for older artifacts.  As of the 2026-08-06 corrective rebuild the
# deployed .so is built against 0.7.2 with LORRAX_FFI_HAVE_CAL=OFF, so it
# has NO libcal.so.0 in DT_NEEDED and no cal_* undefined symbols at all
# (`nm -D` count 0; the pre-rebuild .so had cal_comm_create/cal_comm_destroy).
# Nothing in the supported configuration reads that directory any more.
#
# What it still does is keep a SECOND libcusolverMp.so.0 on the search path
# permanently — the exact duplicate-SONAME hazard described 20 lines above.
# It is harmless ONLY because the SELECTED stage is listed before it; that
# ordering is the single thing standing between this launcher and silently
# running cuSolverMp 0.6.0, whose getrf/getrs is wrong on any Px>1 AND Py>1
# mesh.  Removing the entry is NOT done here because it would break any
# still-CAL-linked .so loudly at dlopen; that is the owner's call, not this
# script's.  Left as a comment rather than a silent edit.
#
# /lorrax_fftw/lib is listed LAST of the lorrax stages, deliberately.  It
# holds exactly one library and no other entry on this path ships a
# libfftw3.*, so its position cannot decide a version the way the two NVHPC
# entries above do — and putting it last means it can never shadow a vendor
# library either.  A directory whose ordering is load-bearing is a hazard;
# this one's is not, and that is a property worth keeping.
LDLIB="${SLATE_INSTALL_HOST}/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/${LORRAX_NVHPC_SUBPATH}:/lorrax_nvhpc/25.5_cuda12.9/math_libs/12.9/lib64:/lorrax_fftw/lib:${MPI_LIB_DIR_CT}:/global/common/software/nersc9/darshan/default/lib"
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
    # PER PLATFORM, not unconditionally 1: on the CPU leg it must be 0 or
    # Cray MPICH aborts in MPI_Init_thread (fixed 2026-08-06 here and in
    # in_container.sh, which shifter --module=mpich forces to have the
    # last word because it UNSETS this name on the way in).
    --env=MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT}"
    # Carried under a LORRAX_ name because shifter's mpich module unsets
    # MPICH_GPU_SUPPORT_ENABLED itself; in_container.sh re-derives from
    # these two on the far side of that boundary.
    --env=LORRAX_PLATFORM="${LORRAX_PLATFORM}"
    --env=LORRAX_MPICH_GPU_SUPPORT="${MPICH_GPU_SUPPORT}"
    # Expose the chosen stack's MPI include + lib paths to CMake so the
    # FFI build picks the right headers / library without guessing.
    --env=LORRAX_MPI_INCLUDE_DIR="${MPI_INCLUDE_DIR_CT}"
    --env=LORRAX_MPICH_LIB_DIR="${MPI_LIB_DIR_CT}"
    --env=LORRAX_PHDF5_MPI_STACK="${MPI_STACK}"
    # ONE source of truth for which cuSOLVERMp stage this container uses:
    # LORRAX_NVHPC_SUBPATH decides the RUNTIME library (LD_LIBRARY_PATH
    # above) and, through these, the one a build launched in here COMPILES
    # against (src/ffi/cpp/build.sh).  Passing both means the .so and the
    # library it will run against agree by construction instead of by two
    # people remembering the same string.  Before 2026-08-06 build.sh
    # hardcoded 25.5_cuda12.9 and silently disagreed with this line.
    --env=LORRAX_NVHPC_SUBPATH="${LORRAX_NVHPC_SUBPATH}"
    --env=LORRAX_NVHPC_ROOT="/lorrax_nvhpc/${LORRAX_NVHPC_SUBPATH%%/*}"
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
    # MPICH_GPU_SUPPORT_ENABLED (which shifter's --module=mpich
    # explicitly unsets per /etc/shifter/udiRoot.conf) at the value this
    # launch's platform needs: 1 on GPU so MPI calls with device pointers
    # go GPU->GPU via Slingshot GPU-Direct RDMA, 0 on CPU so
    # MPI_Init_thread does not abort looking for the GTL's CUDA runtime.
    # Path is the host path; resolves the same inside the container via
    # /global/u2 siteFs.
    IN_CONTAINER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/in_container.sh"
    exec srun "${jobflag}" --mpi="${LORRAX_MPI_TYPE}" \
        --gres=gpu:"${NGPU}" -N "${NNODES}" -n "${NTASKS}" \
        "${SRUN_WRAPPER[@]}" "${SHIFTER_ARGS[@]}" "${IN_CONTAINER}" "$@"
else
    exec "${SHIFTER_ARGS[@]}" "$@"
fi
