#!/usr/bin/env bash
# ============================================================================
# build_ffi_host.sh — build liblorrax_ffi_host.so (the CUDA-free host-platform
# FFI library) on Frontera, WITH the phdf5 read handlers.  RUN INSIDE the
# python:3.12 apptainer container (glibc 2.28+), same host Intel-MPI parallel
# HDF5 the CUDA build_ffi.sh uses for its phdf5 path — minus every
# CUDA/NCCL/cuSOLVERMp dependency.
#
#   config/frontera/build_ffi_host.sh [--fresh]
#
# Output: $LORRAX_FFI_HOST_STAGE/liblorrax_ffi_host.so
#   (default $LORRAX_FFI_STAGE_WTA/build_host; point LORRAX_FFI_HOST_SO there
#    at runtime).  A UNIQUE stage dir so it never clobbers the shared
#    $WORK/lorrax_ffi CUDA .so.
# ============================================================================
set -euo pipefail

: "${LORRAX_ROOT:?set LORRAX_ROOT to the worktree/repo root (contains src/ffi)}"
: "${LORRAX_VENV:=$WORK/lorrax_env/.venv}"
: "${LORRAX_FFI_STAGE_WTA:=$WORK/lorrax_ffi_wtA}"
: "${LORRAX_HDF5_ROOT:=/home1/apps/intel19/impi19_0/phdf5/1.14.6}"
: "${LORRAX_IMPI_ROOT:=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64}"

PY="$LORRAX_VENV/bin/python"
CMAKE="$LORRAX_VENV/bin/cmake"
NINJA="$LORRAX_VENV/bin/ninja"
SRC="$LORRAX_ROOT/src/ffi/common/cpp/host"
BUILD="${LORRAX_FFI_HOST_STAGE:-$LORRAX_FFI_STAGE_WTA/build_host}"

# cmake + ninja live in the venv (pip-installed by stage_ffi_deps.sh); put
# them on PATH so cmake's -G Ninja resolves the build program.
export PATH="$LORRAX_VENV/bin:$PATH"

for p in "$PY" "$CMAKE" "$LORRAX_HDF5_ROOT/include/H5pubconf.h" \
         "$LORRAX_IMPI_ROOT/include/mpi.h" "$SRC/CMakeLists.txt"; do
    [ -e "$p" ] || { echo "[build_host] missing prerequisite: $p" >&2; exit 2; }
done

# Intel MPI wrapper env (same as build_ffi.sh's phdf5 branch): HDF5's
# hdf5-config.cmake runs find_package(MPI); the config/frontera/cmake FindMPI
# stub (on CMAKE_MODULE_PATH) satisfies it with Intel MPI + libfabric.  The
# container has gcc/g++ but no icc, so force the wrappers to gcc/g++.
export I_MPI_ROOT="${LORRAX_IMPI_ROOT%/intel64}"
export I_MPI_CC=gcc
export I_MPI_CXX=g++
export LIBRARY_PATH="$LORRAX_IMPI_ROOT/libfabric/lib:$LORRAX_IMPI_ROOT/lib/release:$LORRAX_IMPI_ROOT/lib:${LIBRARY_PATH:-}"

ARGS=(
    -S "$SRC" -B "$BUILD" -G Ninja
    -DCMAKE_MAKE_PROGRAM="$NINJA"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_COMPILER=g++
    -DPython3_EXECUTABLE="$PY"            # jax.ffi.include_dir() probe uses this
    -DLORRAX_FFI_HAVE_PHDF5=ON
    -DLORRAX_HOST_HAVE_SLATE=OFF          # no SLATE host install on Frontera
    -DHDF5_ROOT="$LORRAX_HDF5_ROOT"
    -DHDF5_PREFER_PARALLEL=ON
    -DLORRAX_MPI_INCLUDE_DIR="$LORRAX_IMPI_ROOT/include"
    -DLORRAX_MPICH_LIB_DIR="$LORRAX_IMPI_ROOT/lib/release"
    -DCMAKE_MODULE_PATH="$LORRAX_ROOT/config/frontera/cmake"
    -DLORRAX_IMPI_ROOT="$LORRAX_IMPI_ROOT"
)

if [ "${1:-}" == "--fresh" ]; then rm -rf "$BUILD"; fi
mkdir -p "$BUILD"

echo "[build_host] configuring -> $BUILD ..."
"$CMAKE" "${ARGS[@]}"
echo "[build_host] compiling..."
"$CMAKE" --build "$BUILD" --parallel

SO="$BUILD/liblorrax_ffi_host.so"
[ -f "$SO" ] || { echo "[build_host] FAILED: no $SO" >&2; exit 1; }
echo "[build_host] --- artifact: $SO ---"; ls -lh "$SO"
echo "[build_host] --- DT_NEEDED ---"; readelf -d "$SO" | grep NEEDED || true
if readelf -d "$SO" | grep NEEDED | grep -qiE 'cuda|nccl|nvshmem|cusolver|cublas'; then
    echo "[build_host] FAILED: host lib links a CUDA-stack library." >&2
    exit 1
fi
echo "[build_host] CUDA-free OK.  done."
