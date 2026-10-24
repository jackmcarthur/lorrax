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
# NCCL: the pip wheel nvidia-nccl-cu13, on every route (owner decision
# 2026-10-17): the uv.lock pin a clone's JAX maps, also installed in the module
# venv (module/stack.sh).  The leg compiles and links against the wheel and its
# closure records it; a run that maps any other libnccl refuses.  Never the HPC
# SDK's comm_libs NCCL.
NCCL_WHEEL="nvidia-nccl-cu13"
NCCL_VERSION="2.30.7"
# shellcheck disable=SC1091
source "$LORRAX_ROOT/config/perlmutter/ffi_mpi.sh"
PHDF5_ROOT="$LORRAX_PM_HDF5_ROOT"
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
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
NCCL="$SITE/nvidia/nccl"
_nccl_have="$(ls -d "$SITE"/nvidia_nccl_cu13-*.dist-info 2>/dev/null | sed 's/.*nvidia_nccl_cu13-\(.*\)\.dist-info/\1/')"
[[ "$_nccl_have" == "$NCCL_VERSION" ]] \
    || die "the venv's $NCCL_WHEEL is '${_nccl_have:-absent}', the pin is $NCCL_VERSION (uv.lock; module/stack.sh)"
for p in "$CUDA/bin/nvcc" "$MATH/lib64/libcusolver.so" "$NCCL/include/nccl.h" "$NCCL/lib/libnccl.so.2" \
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
# The leg's only search path is $ORIGIN (CMakeLists.txt).  cudatoolkit/13.2
# puts the HPC SDK's cuSOLVERMp 0.8 / cuBLASMp 0.8 on LD_LIBRARY_PATH under the
# same SONAMEs as the staged 0.9.1 / 0.10 this leg is compiled against (found
# first, they fail the first distributed eigh with cusolverMpSyevd status=7).
# The build records the leg's dependency closure (below) and the loader enforces
# it on every route: a clone, the module and a sealed bundle alike open exactly
# the recorded files and refuse a process that maps another copy.
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
    -DNCCL_INCLUDE="$NCCL/include" \
    -DNCCL_LIBRARY="$NCCL/lib/libnccl.so.2" \
    -DLORRAX_FFI_HAVE_CUBLASMP=ON \
    -DLORRAX_FFI_HAVE_CUFFT=ON \
    -DLORRAX_FFI_HAVE_CAL=OFF \
    -DLORRAX_FFI_HAVE_PHDF5=ON \
    -DHDF5_ROOT="$PHDF5_ROOT" \
    -DHDF5_PREFER_PARALLEL=ON \
    -DLORRAX_MPI_INCLUDE_DIR="$MPICH_ROOT/include" \
    -DLORRAX_MPICH_LIB_DIR="$MPICH_ROOT/lib" \
    -DLORRAX_MPI_LIBRARY="$LORRAX_PM_MPI_LIBRARY"
cmake --build "$BUILD" --parallel "${LORRAX_BUILD_JOBS:-16}"
SO="$BUILD/liblorrax_ffi.so"
[[ -f "$SO" ]] || { echo "[build_ffi_cuda] FAILED: no .so produced" >&2; exit 1; }

# --- dependency closure (lxkit.native_provider) ----------------------------
# The leg's $ORIGIN RUNPATH loses to LD_LIBRARY_PATH (the HPC SDK cuSOLVERMp 0.8
# above), so the build records every library of the leg's closure: the staged
# cuSOLVERMp/cuBLASMp pair, the CUDA math/NCCL files the venv's JAX plugin maps
# (or, in a venv without CUDA wheels such as the module's, the toolkit's), and
# the pinned MPI/HDF5.  The loader opens exactly those files and refuses a
# process that maps any other copy; a clone needs no LD_LIBRARY_PATH.
"$PY" "$SRC/stage/write_checkout_closure.py" "$SO" --python "$PY" \
    --private-dir "$STAGE/lib" --process-dir "$NCCL/lib" \
    --library-dirs "$LORRAX_PM_RUNTIME_LIB_DIRS:$MATH/lib64:$CUDA/lib64"
# The recorded NCCL must be the pinned wheel's file, whatever else the record holds.
"$PY" - "$SO.closure.json" "$NCCL/lib/libnccl.so.2" "$NCCL_VERSION" <<'PYEOF' || die "the closure's NCCL is not the pinned wheel"
import json, os, sys
doc = json.load(open(sys.argv[1]))
nccl = [l for l in doc["libraries"] if l["family"] == "libnccl"]
want = os.path.realpath(sys.argv[2])
ok = len(nccl) == 1 and nccl[0]["path"] == want and nccl[0].get("version") == f"nvidia_nccl_cu13=={sys.argv[3]}"
print(f"[build_ffi_cuda] closure NCCL: {[(l['path'], l.get('version')) for l in nccl]} (want {want}, {sys.argv[3]})")
sys.exit(0 if ok else 1)
PYEOF
CLOSURE_SHA="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["record_sha256"])' "$SO.closure.json")"

"$SRC/stage/stamp_provenance.sh" "$SO" \
    leg=cuda cuda_module="$CUDA_MODULE" cuda_root="$CUDA" \
    cusolvermp="$CUSOLVERMP_VERSION" cublasmp="$CUBLASMP_VERSION" \
    phdf5="$LORRAX_PM_HDF5_MODULE" phdf5_root="$PHDF5_ROOT" \
    hdf5_soversion=310 mpi="$LORRAX_PM_MPICH_MODULE" mpi_root="$MPICH_ROOT" \
    nccl="$NCCL_WHEEL==$NCCL_VERSION" nccl_root="$NCCL" \
    closure_sha256="$CLOSURE_SHA" \
    || echo "[build_ffi_cuda] WARNING: provenance stamp failed" >&2

# --- post-link gates (scripts/verify_ffi_build.sh) --------------------------
LD_LIBRARY_PATH="$STAGE/lib:$NCCL/lib:$CUDA/lib64:$MATH/lib64:$LORRAX_PM_RUNTIME_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LORRAX_FFI_EXPECT_BACKENDS=cusolvermp,cufft,phdf5 \
LORRAX_FFI_EXPECT_MPI="$LORRAX_PM_MPI_SONAME" \
LORRAX_FFI_EXPECT_HDF5_SOVERSION=310 \
LORRAX_PHDF5_STAGE="$PHDF5_ROOT" \
LORRAX_FFI_VERIFY_ENV=runtime \
LORRAX_GATE_FFTW_PY="$PY" \
GATE_TAG=build_ffi_cuda \
    bash "$LORRAX_ROOT/scripts/verify_ffi_build.sh" --leg cuda "$SO"

echo "[build_ffi_cuda] done: $SO"
