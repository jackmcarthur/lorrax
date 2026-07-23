#!/usr/bin/env bash
# ============================================================================
# build_ffi.sh — build liblorrax_ffi.so on Frontera (eigh-only, NCCL-native).
#
# RUN INSIDE the python:3.12 apptainer container, AFTER stage_ffi_deps.sh:
#   apptainer exec --nv --bind /home1,/work2,/scratch1,/scratch2 \
#       $LORRAX_SIF bash config/frontera/build_ffi.sh [--fresh]
#
# Drives the STOCK src/ffi/common/cpp/CMakeLists.txt (no repo edits beyond the
# LORRAX_FFI_HAVE_{CAL,PHDF5} option guards) with explicit -D overrides:
#   * CAL off   — cuSOLVERMp >= 0.7 is NCCL-native; no libcal/cal.h.
#   * phdf5 off — deferred; add later via `module load phdf5` (host MPI).
#   * SLATE off — auto (not installed).
# Output: $LORRAX_FFI_STAGE/build/liblorrax_ffi.so
# ============================================================================
set -euo pipefail

: "${LORRAX_ROOT:=$HOME/software/lorrax}"
: "${LORRAX_VENV:=$WORK/lorrax_env/.venv}"
: "${LORRAX_FFI_STAGE:=$WORK/lorrax_ffi}"

PY="$LORRAX_VENV/bin/python"
CMAKE="$LORRAX_VENV/bin/cmake"
CUDA_ROOT="$LORRAX_FFI_STAGE/cuda_root"
STAGE="$LORRAX_FFI_STAGE/stage"
BUILD="$LORRAX_FFI_STAGE/build"
SRC="$LORRAX_ROOT/src/ffi/common/cpp"

for p in "$PY" "$CMAKE" "$CUDA_ROOT/include/cuda_runtime.h" "$STAGE/include/cusolverMp.h"; do
    [ -e "$p" ] || { echo "[build] missing prerequisite: $p (run stage_ffi_deps.sh)" >&2; exit 2; }
done

# Make ptxas + the venv's cmake/ninja discoverable.  No nvcc is needed:
# the eigh-only build (LORRAX_FFI_HAVE_CUBLASMP=OFF) has no .cu sources.
export PATH="$CUDA_ROOT/bin:$LORRAX_VENV/bin:$PATH"
export CUDA_HOME="$CUDA_ROOT"

if [ "${1:-}" == "--fresh" ]; then rm -rf "$BUILD"; fi
mkdir -p "$BUILD"

echo "[build] configuring (CAL=OFF phdf5=OFF cublasmp=OFF, eigh-only)..."
"$CMAKE" -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$PY" \
    -DLORRAX_FFI_HAVE_CAL=OFF \
    -DLORRAX_FFI_HAVE_PHDF5=OFF \
    -DLORRAX_FFI_HAVE_CUBLASMP=OFF \
    -DCUDA_TOOLKIT_ROOT="$CUDA_ROOT" \
    -DNCCL_INCLUDE="$CUDA_ROOT/include" \
    -DNCCL_LIBRARY="$CUDA_ROOT/lib64/libnccl.so" \
    -DCUSOLVERMP_INCLUDE_DIR="$STAGE/include" \
    -DCUSOLVERMP_LIB_DIR="$STAGE/lib" \
    -DLORRAX_SLATE_INSTALL_DIR="$LORRAX_FFI_STAGE/_no_slate"

echo "[build] compiling..."
"$CMAKE" --build "$BUILD" --parallel

SO="$BUILD/liblorrax_ffi.so"
[ -f "$SO" ] || { echo "[build] FAILED: no $SO" >&2; exit 1; }
echo "[build] --- artifact: $SO ---"
ls -lh "$SO"
echo "[build] --- DT_NEEDED ---"
readelf -d "$SO" | grep NEEDED
echo "[build] done."
