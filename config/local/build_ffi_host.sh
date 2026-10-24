#!/usr/bin/env bash
# Build liblorrax_ffi_host.so (the CUDA-free host-platform FFI library) on a
# local Ubuntu machine.  Local twin of config/frontera/build_ffi_host.sh.
#
# Backends on this lane: phdf5 (own-built HDF5 1.14 over distro OpenMPI),
# scalapack (distro netlib libscalapack-openmpi — exports all eleven required
# symbols including the C-BLACS interface, measured 2026-08-26), and the
# CBLAS GEMM handlers (distro OpenBLAS).  The FFTW3-ABI host FFT binds its engine at run time (docs/architecture/ffi_layout.md §3c) and the
# XLA FFT lowering stands.
#
# Extra apt prerequisites beyond setup_env.sh's list:
#   apt-get install -y libscalapack-openmpi-dev libopenblas-dev
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/stack.sh"

LORRAX_LOCAL_HOST_BUILD="${LORRAX_LOCAL_HOST_BUILD:-$LORRAX_LOCAL_ROOT/src/ffi/cpp/build_host_local}"

if [[ ! -x "$LORRAX_LOCAL_ENV/bin/python" || ! -e "$LORRAX_LOCAL_PHDF5/lib/libhdf5.so" ]]; then
    echo "[local host build] environment missing; run: bash $HERE/setup_env.sh" >&2
    exit 2
fi

SCALAPACK_SO="${LORRAX_LOCAL_SCALAPACK:-/usr/lib/x86_64-linux-gnu/libscalapack-openmpi.so}"
CBLAS_H="${LORRAX_LOCAL_CBLAS_H:-/usr/include/x86_64-linux-gnu/cblas.h}"
CBLAS_SO="${LORRAX_LOCAL_CBLAS_SO:-/usr/lib/x86_64-linux-gnu/libopenblas.so}"
for path in "$SCALAPACK_SO" "$CBLAS_H" "$CBLAS_SO"; do
    [[ -e "$path" ]] || { echo "[local host build] REFUSED: no $path (see apt line in header)" >&2; exit 2; }
done

# CMake's CBLAS probe wants one LORRAX_CBLAS_DIR prefix with include/ and
# lib/; Ubuntu splits them across /usr/include|lib/x86_64-linux-gnu.  Stage a
# symlink prefix (same idiom as the cusolvermp stage in setup_env.sh).
CBLAS_STAGE="$LORRAX_LOCAL_NATIVE/cblas"
mkdir -p "$CBLAS_STAGE/include" "$CBLAS_STAGE/lib"
ln -sfn "$CBLAS_H" "$CBLAS_STAGE/include/cblas.h"
ln -sfn "$CBLAS_SO" "$CBLAS_STAGE/lib/libopenblas.so"

MPI_INCLUDE="$(mpicc --showme:incdirs 2>/dev/null | awk '{print $1}')"
MPI_LIBDIR="$(mpicc --showme:libdirs 2>/dev/null | awk '{print $1}')"
MPI_LIB="$MPI_LIBDIR/libmpi.so"

SRC="$LORRAX_LOCAL_ROOT/src/ffi/cpp"
SO="$LORRAX_LOCAL_HOST_BUILD/liblorrax_ffi_host.so"
CMAKE="$LORRAX_LOCAL_ENV/bin/cmake"

LORRAX_CBLAS_DIR="$CBLAS_STAGE" \
"$CMAKE" --fresh -S "$SRC" -B "$LORRAX_LOCAL_HOST_BUILD" -G Ninja \
    -DLORRAX_FFI_PLATFORM=host \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$LORRAX_LOCAL_ENV/bin/python" \
    -DLORRAX_SCALAPACK_LIBRARIES="$SCALAPACK_SO" \
    -DLORRAX_FFI_HAVE_PHDF5=ON \
    -DHDF5_ROOT="$LORRAX_LOCAL_PHDF5" \
    -DHDF5_PREFER_PARALLEL=ON \
    -DLORRAX_MPI_INCLUDE_DIR="$MPI_INCLUDE" \
    -DLORRAX_MPICH_LIB_DIR="$MPI_LIBDIR" \
    -DLORRAX_MPI_LIBRARY="$MPI_LIB"

"$CMAKE" --build "$LORRAX_LOCAL_HOST_BUILD" --parallel "${LORRAX_BUILD_JOBS:-$(nproc)}"

[[ -f "$SO" ]] || { echo "[local host build] FAILED: no $SO" >&2; exit 1; }

# Dependency closure (lxkit.native_provider), as build_ffi.sh writes for the
# CUDA leg: this machine is the runtime, so the runtime search path is what the
# record pins. Without the record every host-platform run refuses at start-up.
"$LORRAX_LOCAL_ENV/bin/python" "$SRC/stage/write_checkout_closure.py" "$SO" \
    --library-dirs "$(lorrax_local_runtime_ld_library_path)" --skip-missing-dirs || {
        echo "[local host build] FAILED: dependency closure" >&2; exit 1; }

echo "[local host build] --- artifact: $SO ---"; ls -lh "$SO"
readelf -d "$SO" | grep NEEDED
echo "[local host build] export before running:  LORRAX_FFI_HOST_SO=$SO"
