#!/usr/bin/env bash
# Build the CUDA-13 FFI (cusolvermp + cublasmp + cufft + phdf5) on a generic
# cloud box, entirely against the pip toolkit staged by setup_env.sh.
#
# Cloud twin of config/perlmutter/cuda13_module/build_ffi.sh +
# build_ffi_phdf5.sh, collapsed into one script: unlike Perlmutter there is
# no CUDA-12 Shifter artifact to protect, so this lane has exactly one build
# dir and it carries the phdf5 handlers (OpenMPI + own-built HDF5 1.14).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/stack.sh"

CU13="$(lorrax_cloud_cuda_root)"
if [[ ! -x "$LORRAX_CLOUD_ENV/bin/python" || \
      ! -e "$LORRAX_CLOUD_STAGE/lib/libcusolverMp.so" || \
      ! -e "$LORRAX_CLOUD_PHDF5/lib/libhdf5.so" ]]; then
    echo "[cloud build] environment/stage missing; run:" >&2
    echo "  bash $HERE/setup_env.sh" >&2
    exit 2
fi

SRC="$LORRAX_CLOUD_ROOT/src/ffi/cpp"
SO="$LORRAX_CLOUD_BUILD/liblorrax_ffi.so"
CMAKE="$LORRAX_CLOUD_ENV/bin/cmake"

# MPI facts from the distro OpenMPI (what HDF5 was compiled against).
MPI_INCLUDE="$(mpicc --showme:incdirs 2>/dev/null | awk '{print $1}')"
MPI_LIBDIR="$(mpicc --showme:libdirs 2>/dev/null | awk '{print $1}')"
MPI_LIB="$MPI_LIBDIR/libmpi.so"
for path in "$MPI_INCLUDE/mpi.h" "$MPI_LIB"; do
    [[ -e "$path" ]] || { echo "[cloud build] REFUSED: no $path (need libopenmpi-dev)" >&2; exit 2; }
done

# OMPI_SKIP_MPICXX: OpenMPI's mpi.h otherwise emits references to the
# deprecated MPI C++ bindings (libmpi_cxx), which nothing links —
# measured as `undefined symbol: _ZN3MPI8Datatype4FreeEv` at dlopen.
"$CMAKE" --fresh -S "$SRC" -B "$LORRAX_CLOUD_BUILD" -G Ninja \
    -DLORRAX_FFI_PLATFORM=cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-DOMPI_SKIP_MPICXX=1" \
    -DCMAKE_MAKE_PROGRAM="$LORRAX_CLOUD_ENV/bin/ninja" \
    -DCMAKE_C_COMPILER="$(command -v gcc)" \
    -DCMAKE_CXX_COMPILER="$(command -v g++)" \
    -DCMAKE_CUDA_COMPILER="$CU13/bin/nvcc" \
    -DPython3_EXECUTABLE="$LORRAX_CLOUD_ENV/bin/python" \
    -DCUDA_TOOLKIT_ROOT="$CU13" \
    -DCUSOLVERMP_INCLUDE_DIR="$LORRAX_CLOUD_STAGE/include" \
    -DCUSOLVERMP_LIB_DIR="$LORRAX_CLOUD_STAGE/lib" \
    -DNCCL_INCLUDE="$(lorrax_cloud_nccl_root)/include" \
    -DNCCL_LIBRARY="$(lorrax_cloud_nccl_root)/lib/libnccl.so.2" \
    -DLORRAX_FFI_HAVE_CUBLASMP=ON \
    -DLORRAX_FFI_HAVE_CUFFT=ON \
    -DLORRAX_FFI_HAVE_CAL=OFF \
    -DLORRAX_FFI_HAVE_PHDF5=ON \
    -DHDF5_ROOT="$LORRAX_CLOUD_PHDF5" \
    -DHDF5_PREFER_PARALLEL=ON \
    -DLORRAX_MPI_INCLUDE_DIR="$MPI_INCLUDE" \
    -DLORRAX_MPICH_LIB_DIR="$MPI_LIBDIR" \
    -DLORRAX_MPI_LIBRARY="$MPI_LIB" \
    -DLORRAX_SLATE_INSTALL_DIR="$LORRAX_CLOUD_NATIVE/no-slate"

"$CMAKE" --build "$LORRAX_CLOUD_BUILD" --parallel "${LORRAX_BUILD_JOBS:-$(nproc)}"

[[ -f "$SO" ]] || { echo "[cloud build] FAILED: no $SO" >&2; exit 1; }

LORRAX_ROOT="$LORRAX_CLOUD_ROOT" \
    bash "$SRC/stage/stamp_provenance.sh" "$SO" \
    leg=cuda \
    cuda_root="$CU13" \
    jax="$LORRAX_CLOUD_JAX_VERSION" \
    cusolvermp="$LORRAX_CLOUD_CUSOLVERMP_VERSION" \
    cublasmp="$LORRAX_CLOUD_CUBLASMP_VERSION" \
    phdf5="$LORRAX_CLOUD_PHDF5_VERSION"

# This box IS the runtime environment (no container boundary), so run the
# full acceptance contract, closure gates included.
LD_LIBRARY_PATH="$(lorrax_cloud_runtime_ld_library_path)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LORRAX_ROOT="$LORRAX_CLOUD_ROOT" \
LORRAX_FFI_EXPECT_BACKENDS=cusolvermp,cublasmp,cufft,phdf5 \
LORRAX_FFI_EXPECT_MPI="libmpi.so.40" \
LORRAX_FFI_VERIFY_ENV=runtime \
LORRAX_GATE_FFTW_PY="$LORRAX_CLOUD_ENV/bin/python" \
    bash "$LORRAX_CLOUD_ROOT/scripts/verify_ffi_build.sh" --leg cuda "$SO"

echo "[cloud build] --- accepted artifact: $SO ---"; ls -lh "$SO"
echo "[cloud build] --- libraries this .so will load at run time ---"
readelf -d "$SO" | grep NEEDED
echo "[cloud build] done.  Export before running:"
echo "  export LORRAX_FFI_SO=$SO"
echo "  export LD_LIBRARY_PATH=$(lorrax_cloud_runtime_ld_library_path):\$LD_LIBRARY_PATH"
echo "  export PYTHONPATH=$(lorrax_cloud_pythonpath)"
