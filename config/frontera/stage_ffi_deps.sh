#!/usr/bin/env bash
# ============================================================================
# stage_ffi_deps.sh — one-time staging for the Frontera cuSOLVERMp FFI build.
#
# RUN INSIDE the python:3.12 apptainer container (glibc 2.28+), e.g.:
#   apptainer exec --nv --bind /home1,/work2,/scratch1,/scratch2 \
#       $LORRAX_SIF bash config/frontera/stage_ffi_deps.sh
#
# What it does (all under $LORRAX_FFI_STAGE on $WORK — nothing in $HOME):
#   1. pip-installs nvidia-cusolvermp-cu12 + nvidia-cublasmp-cu12 (NCCL-native
#      distributed linalg; no NVHPC/CAL needed) and cmake/ninja into the venv.
#   2. Assembles a unified CUDA "toolkit root" by symlinking the scattered
#      pip nvidia-*-cu12 packages (nvcc, headers, libs) into one tree, so the
#      stock CMakeLists' find_package/CUDA-language logic resolves a toolkit.
#   3. Stages cusolverMp.h/cublasmp.h + libcusolverMp.so/libcublasmp.so with
#      the bare-.so symlinks CMake links against (wheels ship only .so.0).
#
# Idempotent: safe to re-run.
# ============================================================================
set -euo pipefail

: "${LORRAX_VENV:=$WORK/lorrax_env/.venv}"
: "${LORRAX_FFI_STAGE:=$WORK/lorrax_ffi}"
: "${LORRAX_CUSOLVERMP_PIN:=nvidia-cusolvermp-cu12}"   # add ==0.9.* to pin
: "${LORRAX_CUBLASMP_PIN:=nvidia-cublasmp-cu12}"
UVBIN="${UV_BIN:-$WORK/bin/uv}"

PY="$LORRAX_VENV/bin/python"
[ -x "$PY" ] || { echo "[stage] venv python not found at $PY" >&2; exit 2; }

echo "[stage] === 1. pip deps into venv ==="
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/uv_cache}"
export UV_PYTHON_DOWNLOADS=never
# --python targets the existing venv; these are pure runtime + build tools.
"$UVBIN" pip install --python "$PY" \
    "$LORRAX_CUSOLVERMP_PIN" "$LORRAX_CUBLASMP_PIN" \
    nvidia-nvtx-cu12 cmake ninja

SP=$("$PY" -c "import site,sys; print([p for p in site.getsitepackages() if p.endswith('site-packages')][0])")
NV="$SP/nvidia"
echo "[stage] site-packages: $SP"
[ -d "$NV" ] || { echo "[stage] no nvidia/ dir under site-packages" >&2; exit 2; }

CUDA_ROOT="$LORRAX_FFI_STAGE/cuda_root"
STAGE="$LORRAX_FFI_STAGE/stage"
rm -rf "$CUDA_ROOT" "$STAGE"
mkdir -p "$CUDA_ROOT/bin" "$CUDA_ROOT/include" "$CUDA_ROOT/lib64" \
         "$STAGE/include" "$STAGE/lib"

echo "[stage] === 2. assemble unified CUDA root at $CUDA_ROOT ==="
# nvcc toolchain (nvcc, ptxas, cicc, cudafe++, nvlink, fatbinary, bin2c, ...).
# Symlink each entry by explicit basename (no trailing-slash ambiguity; dir
# entries like crt/ become directory symlinks fine with -n).
NVCC_PKG="$NV/cuda_nvcc"
for f in "$NVCC_PKG"/bin/*; do
    [ -e "$f" ] && ln -sfn "$f" "$CUDA_ROOT/bin/$(basename "$f")"
done
[ -d "$NVCC_PKG/nvvm" ] && ln -sfn "$NVCC_PKG/nvvm" "$CUDA_ROOT/nvvm"

# Merge headers from every nvidia package that ships include/.  cp -rsf makes
# a symlink mirror and merges into existing subdirs (e.g. crt/).  Force each
# so re-runs and cross-package overlaps don't abort.
for pkg in cuda_runtime cuda_nvcc cuda_cccl cuda_cupti nvtx cublas cusolver cusparse nccl nvjitlink; do
    inc="$NV/$pkg/include"
    [ -d "$inc" ] && cp -rsf "$inc"/. "$CUDA_ROOT/include/" 2>/dev/null || true
done

# Merge all shared libs into lib64 (symlinks preserve versioned SONAMEs),
# and add a bare libX.so -> libX.so.N alias so CMake's find_library(cudart,
# cusolver, ...) resolves (the wheels ship only versioned .so.N).
for pkg in cuda_runtime cuda_nvrtc cublas cusolver cusparse nccl nvjitlink; do
    libd="$NV/$pkg/lib"
    [ -d "$libd" ] || continue
    for so in "$libd"/*.so*; do
        [ -e "$so" ] || continue
        base=$(basename "$so")
        ln -sfn "$so" "$CUDA_ROOT/lib64/$base"
        bare="${base%%.so*}.so"           # libcudart.so.12 -> libcudart.so
        [ "$bare" != "$base" ] && ln -sfn "$so" "$CUDA_ROOT/lib64/$bare"
    done
done

echo "[stage] === 3. stage cuSOLVERMp / cuBLASMp headers + libs ==="
ln -sf "$NV/cu12/include/cusolverMp.h"            "$STAGE/include/cusolverMp.h"
ln -sf "$NV/cublasmp/cu12/include/cublasmp.h"     "$STAGE/include/cublasmp.h"
# CMake links the bare .so name; the wheels ship only .so.0 — add the symlink.
ln -sf "$NV/cu12/lib/libcusolverMp.so.0"          "$STAGE/lib/libcusolverMp.so.0"
ln -sf "$NV/cu12/lib/libcusolverMp.so.0"          "$STAGE/lib/libcusolverMp.so"
ln -sf "$NV/cublasmp/cu12/lib/libcublasmp.so.0"   "$STAGE/lib/libcublasmp.so.0"
ln -sf "$NV/cublasmp/cu12/lib/libcublasmp.so.0"   "$STAGE/lib/libcublasmp.so"

echo "[stage] === sanity ==="
echo "  cuda_runtime.h: $([ -e "$CUDA_ROOT/include/cuda_runtime.h" ] && echo yes || echo MISSING)"
echo "  libcudart.so:   $([ -e "$CUDA_ROOT/lib64/libcudart.so" ] && echo yes || echo MISSING)"
echo "  nccl.h:      $([ -e "$CUDA_ROOT/include/nccl.h" ] && echo yes || echo MISSING)"
echo "  libcudart:   $(ls "$CUDA_ROOT"/lib64/libcudart.so* 2>/dev/null | head -1 || echo MISSING)"
echo "  cusolverMp.h:$([ -e "$STAGE/include/cusolverMp.h" ] && echo yes || echo MISSING)"
echo "  libcusolverMp.so: $([ -e "$STAGE/lib/libcusolverMp.so" ] && echo yes || echo MISSING)"
echo "[stage] done. CUDA_ROOT=$CUDA_ROOT  STAGE=$STAGE"
