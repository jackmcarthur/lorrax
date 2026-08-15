#!/usr/bin/env bash
# Build the CUDA-13 FFI *with* the parallel-HDF5 handler, bare-host.
#
# WHY THIS FILE EXISTS.  The sibling build_ffi.sh hardcodes
# -DLORRAX_FFI_HAVE_PHDF5=OFF, so the 2026-08-14 18:20 artifact in
# build_cuda13_module/ exports ZERO phdf5 symbols (stamp `phdf5=0`,
# verifier "GATE 7 N/A: this artifact links no HDF5").  Every driver run
# therefore had to borrow a CUDA-12 .so built against cray-hdf5-parallel
# **1.12** (SONAME libhdf5_parallel_gnu_123.so.200) while this stack's
# h5py is 3.16.0 / HDF5 2.0.0.  This script produces the same artifact
# WITH phdf5, into a SIBLING build dir so build_cuda13_module/ is never
# overwritten.
#
# HDF5 CHOICE, STATED NOT INFERRED.  Perlmutter offers exactly three
# parallel HDF5 modules -- cray-hdf5-parallel/{1.12.2.9,1.14.3.1,1.14.3.7}
# -- and no 2.x in any form.  1.14.3.7 is the newest, is the machine
# default (`/opt/cray/pe/lib64/libhdf5_parallel_gnu.so.310` symlinks into
# it), and is the version the repo already designated as the single-version
# phdf5 stage (CLAIMS 110).  SOVERSION 310.
#
# MPI.  cray-mpich 9.0.1 gnu/12.3, SONAME libmpi_gnu_123.so.12 -- which is
# what cray-hdf5-parallel/1.14.3.7 itself carries as DT_NEEDED, so HDF5 and
# the FFI share ONE libmpi by construction.  Both SONAMEs are in the
# ldconfig cache via /opt/cray/pe/lib64, so the artifact resolves with NO
# LD_LIBRARY_PATH help -- which matters because the lorrax_A/B modulefiles
# REPLACE LD_LIBRARY_PATH rather than appending to it.
#
# LORRAX_MPI_LIBRARY is passed explicitly: CMakeLists' find_library(NAMES
# mpi ...) would otherwise pick up `libmpi.a` from the Cray lib dir (there
# is no libmpi.so there) and statically link MPI into the .so.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/stack.sh"
lorrax_cuda13_load_build_modules

LORRAX_CUDA13_BUILD="${LORRAX_CUDA13_PHDF5_BUILD:-$LORRAX_CUDA13_ROOT/src/ffi/cpp/build_cuda13_phdf5}"

# Site facts for the phdf5 leg.
PHDF5_VERSION="${LORRAX_CUDA13_PHDF5_VERSION:-1.14.3.7}"
PHDF5_ROOT="${LORRAX_CUDA13_PHDF5_ROOT:-/opt/cray/pe/hdf5-parallel/$PHDF5_VERSION/gnu/12.3}"
MPICH_VERSION="${LORRAX_CUDA13_MPICH_VERSION:-9.0.1}"
MPICH_ROOT="${LORRAX_CUDA13_MPICH_ROOT:-/opt/cray/pe/mpich/$MPICH_VERSION/ofi/gnu/12.3}"
MPI_LIB="${LORRAX_CUDA13_MPI_LIBRARY:-$MPICH_ROOT/lib/libmpi_gnu_123.so}"

for p in "$PHDF5_ROOT/include/hdf5.h" "$PHDF5_ROOT/lib/libhdf5.so" \
         "$MPICH_ROOT/include/mpi.h" "$MPI_LIB"; do
    [[ -e "$p" ]] || { echo "[cuda13 phdf5] missing: $p" >&2; exit 2; }
done

if [[ ! -x "$LORRAX_CUDA13_ENV/bin/python" || \
      ! -e "$LORRAX_CUDA13_STAGE/lib/libcusolverMp.so" ]]; then
    echo "[cuda13 phdf5] environment/stage missing; run:" >&2
    echo "  bash $HERE/setup_env.sh" >&2
    exit 2
fi

SRC="$LORRAX_CUDA13_ROOT/src/ffi/cpp"
SO="$LORRAX_CUDA13_BUILD/liblorrax_ffi.so"
CC_BIN="$(command -v gcc)"
CXX_BIN="$(command -v g++)"
NVCC_BIN="$LORRAX_CUDA13_CUDA/bin/nvcc"

# Same reasoning as build_ffi.sh: plain GCC, not the Cray `cc` wrapper, so
# the CUDA-12 libmpi_gtl_cuda is not injected.  We name the MPI library
# ourselves instead.
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
    -DLORRAX_FFI_HAVE_PHDF5=ON \
    -DHDF5_ROOT="$PHDF5_ROOT" \
    -DHDF5_PREFER_PARALLEL=ON \
    -DLORRAX_MPI_INCLUDE_DIR="$MPICH_ROOT/include" \
    -DLORRAX_MPICH_LIB_DIR="$MPICH_ROOT/lib" \
    -DLORRAX_MPI_LIBRARY="$MPI_LIB" \
    -DLORRAX_SLATE_INSTALL_DIR="$LORRAX_CUDA13_NATIVE/no-slate"

cmake --build "$LORRAX_CUDA13_BUILD" --parallel "${LORRAX_BUILD_JOBS:-8}"

LORRAX_ROOT="$LORRAX_CUDA13_ROOT" \
    bash "$SRC/stage/stamp_provenance.sh" "$SO" \
    leg=cuda \
    cuda_module="$LORRAX_CUDA13_TOOLKIT_MODULE" \
    cuda_root="$LORRAX_CUDA13_CUDA" \
    jax="$LORRAX_CUDA13_JAX_VERSION" \
    cusolvermp="$LORRAX_CUDA13_CUSOLVERMP_VERSION" \
    cublasmp="$LORRAX_CUDA13_CUBLASMP_VERSION" \
    phdf5="cray-hdf5-parallel/$PHDF5_VERSION" \
    phdf5_root="$PHDF5_ROOT" \
    hdf5_soversion=310 \
    mpi="cray-mpich/$MPICH_VERSION" \
    mpi_root="$MPICH_ROOT"

LD_LIBRARY_PATH="$(lorrax_cuda13_runtime_ld_library_path)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LORRAX_ROOT="$LORRAX_CUDA13_ROOT" \
LORRAX_FFI_EXPECT_BACKENDS=cusolvermp,cublasmp,cufft,phdf5 \
LORRAX_FFI_EXPECT_MPI=libmpi_gnu_123.so.12 \
LORRAX_FFI_EXPECT_HDF5_SOVERSION=310 \
LORRAX_PHDF5_STAGE="$PHDF5_ROOT" \
LORRAX_FFI_VERIFY_ENV=runtime \
LORRAX_GATE_FFTW_PY="$LORRAX_CUDA13_ENV/bin/python" \
    bash "$LORRAX_CUDA13_ROOT/scripts/verify_ffi_build.sh" --leg cuda "$SO"

echo "[cuda13 phdf5] accepted artifact: $SO"
