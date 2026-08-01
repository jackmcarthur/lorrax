#!/usr/bin/env bash
# ============================================================================
# build_ffi.sh — build liblorrax_ffi.so on Frontera.  RUN INSIDE the
# python:3.12 apptainer container, AFTER stage_ffi_deps.sh.
#
#   config/frontera/build_ffi.sh [--fresh]
#
# Two modes (env LORRAX_FFI_PHDF5):
#   0 (default) : eigh-only, NCCL-native.  No MPI/HDF5, no nvcc.
#                 -> $LORRAX_FFI_STAGE/build/liblorrax_ffi.so
#   1           : + phdf5 (host Intel-MPI parallel HDF5, bind-mounted).
#                 -> $LORRAX_FFI_STAGE/build_phdf5/liblorrax_ffi.so
# Both keep CAL off (NCCL-native cuSOLVERMp 0.9) and cuBLASMp fused kernels
# off (pip toolchain has ptxas but no nvcc driver).
# ============================================================================
set -euo pipefail

# No default: the old $HOME/software/lorrax guess pointed at a stale checkout
# on this machine and a silently-wrong source tree is worse than a refusal
# (same :?-contract as build_mpiwrapper.sh / build_ffi_host.sh).
: "${LORRAX_ROOT:?set LORRAX_ROOT to the worktree/repo root (contains config/frontera and src/ffi)}"
: "${LORRAX_VENV:=$WORK/lorrax_env/.venv}"
: "${LORRAX_FFI_STAGE:=$WORK/lorrax_ffi}"
: "${LORRAX_FFI_PHDF5:=0}"
: "${LORRAX_HDF5_ROOT:=/home1/apps/intel19/impi19_0/phdf5/1.14.6}"
: "${LORRAX_IMPI_ROOT:=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64}"

PY="$LORRAX_VENV/bin/python"
CMAKE="$LORRAX_VENV/bin/cmake"
CUDA_ROOT="$LORRAX_FFI_STAGE/cuda_root"
STAGE="$LORRAX_FFI_STAGE/stage"
SRC="$LORRAX_ROOT/src/ffi/cpp"

for p in "$PY" "$CMAKE" "$CUDA_ROOT/include/cuda_runtime.h" "$STAGE/include/cusolverMp.h"; do
    [ -e "$p" ] || { echo "[build] missing prerequisite: $p (run stage_ffi_deps.sh)" >&2; exit 2; }
done

export PATH="$CUDA_ROOT/bin:$LORRAX_VENV/bin:$PATH"
export CUDA_HOME="$CUDA_ROOT"

ARGS=(
    -DLORRAX_FFI_PLATFORM=cuda
    -DCMAKE_BUILD_TYPE=Release
    -DPython3_EXECUTABLE="$PY"
    -DLORRAX_FFI_HAVE_CAL=OFF
    -DLORRAX_FFI_HAVE_CUBLASMP=OFF
    -DCUDA_TOOLKIT_ROOT="$CUDA_ROOT"
    -DNCCL_INCLUDE="$CUDA_ROOT/include"
    -DNCCL_LIBRARY="$CUDA_ROOT/lib64/libnccl.so"
    -DCUSOLVERMP_INCLUDE_DIR="$STAGE/include"
    -DCUSOLVERMP_LIB_DIR="$STAGE/lib"
    -DLORRAX_SLATE_INSTALL_DIR="$LORRAX_FFI_STAGE/_no_slate"
)

if [ "$LORRAX_FFI_PHDF5" = "1" ]; then
    BUILD="$LORRAX_FFI_STAGE/build_phdf5"
    echo "[build] phdf5 ON — host Intel-MPI HDF5 at $LORRAX_HDF5_ROOT"
    [ -e "$LORRAX_HDF5_ROOT/include/H5pubconf.h" ] || { echo "[build] HDF5 not found (bind /home1)" >&2; exit 2; }
    [ -e "$LORRAX_IMPI_ROOT/include/mpi.h" ]       || { echo "[build] Intel MPI not found (bind /opt)" >&2; exit 2; }
    # HDF5's own hdf5-config.cmake calls find_package(MPI); satisfy CMake's
    # FindMPI with Intel MPI's gcc/g++ wrappers (the container has gcc/g++
    # but no icc).  The wrappers do $I_MPI_ROOT/intel64/... so I_MPI_ROOT is
    # the .../linux/mpi parent, NOT the intel64 subdir.
    export I_MPI_ROOT="${LORRAX_IMPI_ROOT%/intel64}"
    export I_MPI_CC=gcc      # mpigcc defaults to icc (absent) -> force gcc
    export I_MPI_CXX=g++
    # libmpi.so DT_NEEDs Intel's bundled libfabric (fi_strerror@FABRIC_1.0);
    # put it (+ the mpi libdirs) on gcc's link search so both the FindMPI
    # test compile and the final .so link resolve it.
    export LIBRARY_PATH="$LORRAX_IMPI_ROOT/libfabric/lib:$LORRAX_IMPI_ROOT/lib/release:$LORRAX_IMPI_ROOT/lib:${LIBRARY_PATH:-}"
    ARGS+=(
        -DLORRAX_FFI_HAVE_PHDF5=ON
        -DHDF5_ROOT="$LORRAX_HDF5_ROOT"
        -DHDF5_PREFER_PARALLEL=ON
        -DLORRAX_MPI_INCLUDE_DIR="$LORRAX_IMPI_ROOT/include"
        -DLORRAX_MPICH_LIB_DIR="$LORRAX_IMPI_ROOT/lib/release"
        # Shadow CMake's builtin FindMPI (which insists on a compile test)
        # with the Frontera stub — HDF5's config only needs a valid MPI target.
        -DCMAKE_MODULE_PATH="$LORRAX_ROOT/config/frontera/cmake"
        -DLORRAX_IMPI_ROOT="$LORRAX_IMPI_ROOT"
    )
else
    BUILD="$LORRAX_FFI_STAGE/build"
    ARGS+=( -DLORRAX_FFI_HAVE_PHDF5=OFF )
fi

if [ "${1:-}" == "--fresh" ]; then rm -rf "$BUILD"; fi
mkdir -p "$BUILD"

echo "[build] configuring (phdf5=$LORRAX_FFI_PHDF5) -> $BUILD ..."
"$CMAKE" -S "$SRC" -B "$BUILD" -G Ninja "${ARGS[@]}"
echo "[build] compiling..."
"$CMAKE" --build "$BUILD" --parallel

SO="$BUILD/liblorrax_ffi.so"
[ -f "$SO" ] || { echo "[build] FAILED: no $SO" >&2; exit 1; }
echo "[build] --- artifact: $SO ---"; ls -lh "$SO"
echo "[build] --- libraries this .so will load at run time ---"
readelf -d "$SO" | grep NEEDED
echo "[build] done."
