#!/usr/bin/env bash
# Build the CUDA-13 FFI (cusolvermp + cublasmp + cufft + phdf5) on a local
# machine, entirely against the pip toolkit staged by setup_env.sh.
#
# Local twin of config/perlmutter/build_ffi_cuda.sh: one build dir, which
# carries the phdf5 handlers (OpenMPI + own-built HDF5 1.14).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/stack.sh"

CU13="$(lorrax_local_cuda_root)"
if [[ ! -x "$LORRAX_LOCAL_ENV/bin/python" || \
      ! -e "$LORRAX_LOCAL_STAGE/lib/libcusolverMp.so" || \
      ! -e "$LORRAX_LOCAL_PHDF5/lib/libhdf5.so" ]]; then
    echo "[local build] environment/stage missing; run:" >&2
    echo "  bash $HERE/setup_env.sh" >&2
    exit 2
fi

SRC="$LORRAX_LOCAL_ROOT/src/ffi/cpp"
SO="$LORRAX_LOCAL_BUILD/liblorrax_ffi.so"
CMAKE="$LORRAX_LOCAL_ENV/bin/cmake"

# MPI facts from the distro OpenMPI (what HDF5 was compiled against).
MPI_INCLUDE="$(mpicc --showme:incdirs 2>/dev/null | awk '{print $1}')"
MPI_LIBDIR="$(mpicc --showme:libdirs 2>/dev/null | awk '{print $1}')"
MPI_LIB="$MPI_LIBDIR/libmpi.so"
for path in "$MPI_INCLUDE/mpi.h" "$MPI_LIB"; do
    [[ -e "$path" ]] || { echo "[local build] REFUSED: no $path (need libopenmpi-dev)" >&2; exit 2; }
done

"$CMAKE" --fresh -S "$SRC" -B "$LORRAX_LOCAL_BUILD" -G Ninja \
    -DLORRAX_FFI_PLATFORM=cuda \
    -DLORRAX_FFI_POST_BUILD=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MAKE_PROGRAM="$LORRAX_LOCAL_ENV/bin/ninja" \
    -DCMAKE_C_COMPILER="$(command -v gcc)" \
    -DCMAKE_CXX_COMPILER="$(command -v g++)" \
    -DCMAKE_CUDA_COMPILER="$CU13/bin/nvcc" \
    -DPython3_EXECUTABLE="$LORRAX_LOCAL_ENV/bin/python" \
    -DCUDA_TOOLKIT_ROOT="$CU13" \
    -DCUSOLVERMP_INCLUDE_DIR="$LORRAX_LOCAL_STAGE/include" \
    -DCUSOLVERMP_LIB_DIR="$LORRAX_LOCAL_STAGE/lib" \
    -DNCCL_INCLUDE="$(lorrax_local_nccl_root)/include" \
    -DNCCL_LIBRARY="$(lorrax_local_nccl_root)/lib/libnccl.so.2" \
    -DLORRAX_FFI_HAVE_CUBLASMP=ON \
    -DLORRAX_FFI_HAVE_CUFFT=ON \
    -DLORRAX_FFI_HAVE_CAL=OFF \
    -DLORRAX_FFI_HAVE_PHDF5=ON \
    -DHDF5_ROOT="$LORRAX_LOCAL_PHDF5" \
    -DHDF5_PREFER_PARALLEL=ON \
    -DLORRAX_MPI_INCLUDE_DIR="$MPI_INCLUDE" \
    -DLORRAX_MPICH_LIB_DIR="$MPI_LIBDIR" \
    -DLORRAX_MPI_LIBRARY="$MPI_LIB" \

"$CMAKE" --build "$LORRAX_LOCAL_BUILD" --parallel "${LORRAX_BUILD_JOBS:-$(nproc)}"

[[ -f "$SO" ]] || { echo "[local build] FAILED: no $SO" >&2; exit 1; }

# Dependency closure (lxkit.native_provider): this box is the runtime, so the
# runtime search path is what the record pins; the loader enforces it.
"$LORRAX_LOCAL_ENV/bin/python" "$SRC/stage/write_checkout_closure.py" "$SO" \
    --python "$LORRAX_LOCAL_ENV/bin/python" --private-dir "$LORRAX_LOCAL_STAGE/lib" \
    --library-dirs "$(lorrax_local_runtime_ld_library_path)" --skip-missing-dirs

LORRAX_ROOT="$LORRAX_LOCAL_ROOT" \
    bash "$SRC/stage/stamp_provenance.sh" "$SO" \
    closure_sha256="$("$LORRAX_LOCAL_ENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["record_sha256"])' "$SO.closure.json")" \
    leg=cuda \
    cuda_root="$CU13" \
    jax="$LORRAX_LOCAL_JAX_VERSION" \
    cusolvermp="$LORRAX_LOCAL_CUSOLVERMP_VERSION" \
    cublasmp="$LORRAX_LOCAL_CUBLASMP_VERSION" \
    phdf5="$LORRAX_LOCAL_PHDF5_VERSION"

# This box IS the runtime environment (no container boundary), so run the
# full acceptance contract, closure gates included.
LD_LIBRARY_PATH="$(lorrax_local_runtime_ld_library_path)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LORRAX_ROOT="$LORRAX_LOCAL_ROOT" \
LORRAX_FFI_EXPECT_BACKENDS=cusolvermp,cufft,phdf5 \
LORRAX_FFI_EXPECT_MPI="libmpi.so.40" \
LORRAX_FFI_VERIFY_ENV=runtime \
LORRAX_GATE_FFTW_PY="$LORRAX_LOCAL_ENV/bin/python" \
    bash "$LORRAX_LOCAL_ROOT/scripts/verify_ffi_build.sh" --leg cuda "$SO"

echo "[local build] --- accepted artifact: $SO ---"; ls -lh "$SO"
echo "[local build] --- libraries this .so will load at run time ---"
readelf -d "$SO" | grep NEEDED
echo "[local build] done.  Export before running:"
echo "  export LORRAX_FFI_SO=$SO"
echo "  export LD_LIBRARY_PATH=$(lorrax_local_runtime_ld_library_path):\$LD_LIBRARY_PATH"
echo "  export PYTHONPATH=$(lorrax_local_pythonpath)"
