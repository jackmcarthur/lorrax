#!/usr/bin/env bash
# Build the CUDA FFI directly against Perlmutter's CUDA 13.2 module stack.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/stack.sh"
lorrax_cuda13_load_build_modules

if [[ ! -x "$LORRAX_CUDA13_ENV/bin/python" || \
      ! -e "$LORRAX_CUDA13_STAGE/lib/libcusolverMp.so" ]]; then
    echo "[cuda13 build] environment/stage missing; run:" >&2
    echo "  bash $HERE/setup_env.sh" >&2
    exit 2
fi

SRC="$LORRAX_CUDA13_ROOT/src/ffi/cpp"
SO="$LORRAX_CUDA13_BUILD/liblorrax_ffi.so"
CC_BIN="$(command -v gcc)"
CXX_BIN="$(command -v g++)"
NVCC_BIN="$LORRAX_CUDA13_CUDA/bin/nvcc"

# Do not use the Cray `cc` wrapper here.  With craype-accel-nvidia80 loaded it
# injects the CUDA-12 libmpi_gtl_cuda even into CMake's compiler probe.  The
# cuSOLVERMp route is NCCL-based and does not need MPI/GTL, so plain GCC gives
# this artifact a single, auditable CUDA-13 closure.
cmake --fresh -S "$SRC" -B "$LORRAX_CUDA13_BUILD" \
    -DLORRAX_FFI_PLATFORM=cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC_BIN" \
    -DCMAKE_CXX_COMPILER="$CXX_BIN" \
    -DCMAKE_CUDA_COMPILER="$NVCC_BIN" \
    -DPython3_EXECUTABLE="$LORRAX_CUDA13_ENV/bin/python" \
    -DCUDA_TOOLKIT_ROOT="$LORRAX_CUDA13_CUDA" \
    -DNVHPC_ROOT="$LORRAX_CUDA13_SDK" \
    -DNVHPC_CUDA_SUBDIR=13.2 \
    -DCUSOLVERMP_INCLUDE_DIR="$LORRAX_CUDA13_STAGE/include" \
    -DCUSOLVERMP_LIB_DIR="$LORRAX_CUDA13_STAGE/lib" \
    -DCUSOLVER_LIBRARY="$LORRAX_CUDA13_MATH/lib64/libcusolver.so" \
    -DNCCL_INCLUDE="$LORRAX_CUDA13_COMM/nccl/include" \
    -DNCCL_LIBRARY="$LORRAX_CUDA13_COMM/nccl/lib/libnccl.so" \
    -DLORRAX_FFI_HAVE_CUBLASMP=ON \
    -DLORRAX_FFI_HAVE_CUFFT=ON \
    -DLORRAX_FFI_HAVE_CAL=OFF \
    -DLORRAX_FFI_HAVE_PHDF5=OFF \
    -DLORRAX_SLATE_INSTALL_DIR="$LORRAX_CUDA13_NATIVE/no-slate"

cmake --build "$LORRAX_CUDA13_BUILD" --parallel "${LORRAX_BUILD_JOBS:-8}"

LORRAX_ROOT="$LORRAX_CUDA13_ROOT" \
    bash "$SRC/stage/stamp_provenance.sh" "$SO" \
    leg=cuda \
    cuda_module="$LORRAX_CUDA13_TOOLKIT_MODULE" \
    cuda_root="$LORRAX_CUDA13_CUDA" \
    jax="$LORRAX_CUDA13_JAX_VERSION" \
    cusolvermp="$LORRAX_CUDA13_CUSOLVERMP_VERSION" \
    cublasmp="$LORRAX_CUDA13_CUBLASMP_VERSION"

LD_LIBRARY_PATH="$(lorrax_cuda13_runtime_ld_library_path)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LORRAX_ROOT="$LORRAX_CUDA13_ROOT" \
LORRAX_FFI_EXPECT_BACKENDS=cusolvermp,cublasmp,cufft \
LORRAX_FFI_EXPECT_MPI=none \
LORRAX_FFI_VERIFY_ENV=runtime \
LORRAX_GATE_FFTW_PY="$LORRAX_CUDA13_ENV/bin/python" \
    bash "$LORRAX_CUDA13_ROOT/scripts/verify_ffi_build.sh" --leg cuda "$SO"

echo "[cuda13 build] accepted artifact: $SO"
