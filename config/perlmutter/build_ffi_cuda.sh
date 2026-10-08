#!/usr/bin/env bash
# ============================================================================
# build_ffi_cuda.sh — build liblorrax_ffi.so (the CUDA leg) on NERSC
# Perlmutter, bare host, against NERSC cudatoolkit/13.2 and this checkout's
# uv venv.  The host-leg twin is build_ffi_host.sh in this directory.
#
#   config/perlmutter/build_ffi_cuda.sh [--fresh]
#
# Needs: `uv sync --extra cuda13` run in this checkout (the venv supplies the
# Python and jaxlib the leg is built for) and `uv` on PATH (it stages the
# cuSOLVERMp / cuBLASMp wheels once).
#
# Output: <checkout>/src/ffi/cpp/build/liblorrax_ffi.so, where the loader
# looks by default, so a run needs no LORRAX_FFI_SO.
# Stage:  $LORRAX_BUILD_PREFIX/cusolvermp-<ver>  (default <checkout>/.build)
#
# This is the recipe that built the lorrax_A module's CUDA leg
# (stack.sh + setup_env.sh + build_ffi_phdf5.sh of the runtime recipe),
# moved into the repo on 2026-09-29.  The pins below are that recipe's.
# The CUDA leg is compiled with plain GCC, not the Cray `cc` wrapper: with
# craype-accel-nvidia80 loaded the wrapper injects the CUDA-12
# libmpi_gtl_cuda.  cuSOLVERMp and cuBLASMp communicate through NCCL, and
# Cray MPICH GPU support stays off.  The MPI is named explicitly, pinned once
# for both legs in ffi_mpi.sh.
# ============================================================================
set -euo pipefail

LORRAX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$LORRAX_ROOT/src/ffi/cpp"
BUILD="$SRC/build"
PREFIX="${LORRAX_BUILD_PREFIX:-$LORRAX_ROOT/.build}"
PY="$LORRAX_ROOT/.venv/bin/python"

# --- site pins (the lorrax_A recipe's values) -------------------------------
CUDA_MODULE="cudatoolkit/13.2"
CUSOLVERMP_VERSION="0.9.1.9318.post1"
CUBLASMP_VERSION="0.10.0.3695"
PHDF5_VERSION="1.14.3.7"
PHDF5_ROOT="/opt/cray/pe/hdf5-parallel/$PHDF5_VERSION/gnu/12.3"
# shellcheck disable=SC1091
source "$LORRAX_ROOT/config/perlmutter/ffi_mpi.sh"
MPICH_ROOT="$LORRAX_PM_MPICH_ROOT"

die() { echo "[build_ffi_cuda] REFUSED: $*" >&2; exit 2; }

[[ -x "$PY" ]] || die "no venv at $LORRAX_ROOT/.venv; run 'uv sync --extra cuda13' in $LORRAX_ROOT first"
"$PY" -c 'import jax_plugins.xla_cuda13' 2>/dev/null \
    || die "the venv has no CUDA-13 jax plugin; run 'uv sync --extra cuda13' (the plain 'uv sync' installs CPU jax only)"
UV="$(command -v uv || true)"

# --- modules ----------------------------------------------------------------
if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /opt/cray/pe/lmod/lmod/init/bash
fi
module unload cudatoolkit >/dev/null 2>&1 || true
module load gcc-native/14 "$CUDA_MODULE" cmake
CUDA="${CUDA_HOME:?$CUDA_MODULE did not set CUDA_HOME}"
SDK="$(cd "$CUDA/../.." && pwd)"          # .../hpc_sdk/Linux_x86_64/<ver>
CUDA_VER="$(basename "$CUDA")"            # 13.2
MATH="$SDK/math_libs/$CUDA_VER"
COMM="$SDK/comm_libs/$CUDA_VER"
for p in "$CUDA/bin/nvcc" "$MATH/lib64/libcusolver.so" "$COMM/nccl/lib/libnccl.so" \
         "$PHDF5_ROOT/include/hdf5.h" "$MPICH_ROOT/include/mpi.h" "$LORRAX_PM_MPI_LIBRARY"; do
    [[ -e "$p" ]] || die "missing site file $p"
done

# --- cuSOLVERMp / cuBLASMp stage (vendor wheels, installed once) ------------
STAGE="$PREFIX/cusolvermp-$CUSOLVERMP_VERSION"
if [[ ! -e "$STAGE/lib/libcusolverMp.so" || ! -e "$STAGE/lib/libcublasmp.so" ]]; then
    [[ -n "$UV" ]] || die "uv is not on PATH (it stages the cuSOLVERMp wheels)"
    echo "[build_ffi_cuda] staging cuSOLVERMp $CUSOLVERMP_VERSION + cuBLASMp $CUBLASMP_VERSION in $STAGE"
    rm -rf "$STAGE"
    "$UV" pip install --quiet --no-deps --python "$PY" --target "$STAGE/wheels" \
        "nvidia-cusolvermp-cu13==$CUSOLVERMP_VERSION" \
        "nvidia-cublasmp-cu13==$CUBLASMP_VERSION"
    mkdir -p "$STAGE/include" "$STAGE/lib"
    W="$STAGE/wheels/nvidia"
    ln -sfn "$W/cu13/include/cusolverMp.h"          "$STAGE/include/cusolverMp.h"
    ln -sfn "$W/cublasmp/cu13/include/cublasmp.h"   "$STAGE/include/cublasmp.h"
    ln -sfn "$W/cu13/lib/libcusolverMp.so.0"        "$STAGE/lib/libcusolverMp.so.0"
    ln -sfn libcusolverMp.so.0                      "$STAGE/lib/libcusolverMp.so"
    ln -sfn "$W/cublasmp/cu13/lib/libcublasmp.so.0" "$STAGE/lib/libcublasmp.so.0"
    ln -sfn libcublasmp.so.0                        "$STAGE/lib/libcublasmp.so"
fi
for p in "$STAGE/include/cusolverMp.h" "$STAGE/include/cublasmp.h" \
         "$STAGE/lib/libcusolverMp.so.0" "$STAGE/lib/libcublasmp.so.0"; do
    [[ -e "$p" ]] || die "the vendor wheels did not supply $p (delete $STAGE and rerun)"
done

# --- configure + build ------------------------------------------------------
# The library search path is a DT_RPATH (--disable-new-dtags), which the loader
# reads BEFORE LD_LIBRARY_PATH.  Two things depend on that order:
#   - cudatoolkit/13.2 puts the HPC SDK's math_libs on LD_LIBRARY_PATH, which
#     carries cuSOLVERMp 0.8 / cuBLASMp 0.8 under the same SONAMEs as the
#     staged 0.9.1 / 0.10 this leg is compiled against.  Found first, they
#     fail inside the first distributed eigh (cusolverMpSyevd status=7).
#   - /opt/cray/pe/lib64 (the ld.so.cache fallback) points
#     libhdf5_parallel_gnu.so.310 at the site-default HDF5, which links the
#     site-default MPI, so the pinned HDF5 and MPI lib dirs are listed too.
[[ "${1:-}" == "--fresh" ]] && rm -rf "${BUILD:?}"
cmake --fresh -S "$SRC" -B "$BUILD" \
    -DLORRAX_FFI_PLATFORM=cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$(command -v gcc)" \
    -DCMAKE_CXX_COMPILER="$(command -v g++)" \
    -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
    -DPython3_EXECUTABLE="$PY" \
    -DCUDA_TOOLKIT_ROOT="$CUDA" \
    -DNVHPC_ROOT="$SDK" \
    -DNVHPC_CUDA_SUBDIR="$CUDA_VER" \
    -DCUFFT_LIBRARY="$MATH/lib64/libcufft.so" \
    -DCUFFT_INCLUDE_DIR="$MATH/include" \
    -DNVRTC_LIBRARY="$CUDA/lib64/libnvrtc.so" \
    -DNVRTC_INCLUDE_DIR="$CUDA/include" \
    -DCUSOLVERMP_INCLUDE_DIR="$STAGE/include" \
    -DCUSOLVERMP_LIB_DIR="$STAGE/lib" \
    -DCUSOLVER_LIBRARY="$MATH/lib64/libcusolver.so" \
    -DNCCL_INCLUDE="$COMM/nccl/include" \
    -DNCCL_LIBRARY="$COMM/nccl/lib/libnccl.so" \
    -DLORRAX_FFI_HAVE_CUBLASMP=ON \
    -DLORRAX_FFI_HAVE_CUFFT=ON \
    -DLORRAX_FFI_HAVE_CAL=OFF \
    -DLORRAX_FFI_HAVE_PHDF5=ON \
    -DHDF5_ROOT="$PHDF5_ROOT" \
    -DHDF5_PREFER_PARALLEL=ON \
    -DLORRAX_MPI_INCLUDE_DIR="$MPICH_ROOT/include" \
    -DLORRAX_MPICH_LIB_DIR="$MPICH_ROOT/lib" \
    -DLORRAX_MPI_LIBRARY="$LORRAX_PM_MPI_LIBRARY" \
    -DLORRAX_SLATE_INSTALL_DIR="$PREFIX/no-slate" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags -Wl,-rpath,$PHDF5_ROOT/lib:$MPICH_ROOT/lib"
cmake --build "$BUILD" --parallel "${LORRAX_BUILD_JOBS:-16}"
SO="$BUILD/liblorrax_ffi.so"
[[ -f "$SO" ]] || { echo "[build_ffi_cuda] FAILED: no .so produced" >&2; exit 1; }

"$SRC/stage/stamp_provenance.sh" "$SO" \
    leg=cuda cuda_module="$CUDA_MODULE" cuda_root="$CUDA" \
    cusolvermp="$CUSOLVERMP_VERSION" cublasmp="$CUBLASMP_VERSION" \
    phdf5="cray-hdf5-parallel/$PHDF5_VERSION" phdf5_root="$PHDF5_ROOT" \
    hdf5_soversion=310 mpi="$LORRAX_PM_MPICH_MODULE" mpi_root="$MPICH_ROOT" \
    || echo "[build_ffi_cuda] WARNING: provenance stamp failed" >&2

# --- post-link gates (scripts/verify_ffi_build.sh) --------------------------
LD_LIBRARY_PATH="$STAGE/lib:$CUDA/lib64:$MATH/lib64:$COMM/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LORRAX_FFI_EXPECT_BACKENDS=cusolvermp,cufft,phdf5 \
LORRAX_FFI_EXPECT_MPI="$LORRAX_PM_MPI_SONAME" \
LORRAX_FFI_EXPECT_HDF5_SOVERSION=310 \
LORRAX_PHDF5_STAGE="$PHDF5_ROOT" \
LORRAX_FFI_VERIFY_ENV=runtime \
LORRAX_GATE_FFTW_PY="$PY" \
GATE_TAG=build_ffi_cuda \
    bash "$LORRAX_ROOT/scripts/verify_ffi_build.sh" --leg cuda "$SO"

echo "[build_ffi_cuda] done: $SO"
