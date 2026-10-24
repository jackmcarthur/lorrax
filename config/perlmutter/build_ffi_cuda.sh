#!/usr/bin/env bash
# ============================================================================
# build_ffi_cuda.sh — build liblorrax_ffi.so (the CUDA leg) on NERSC
# Perlmutter, bare host, entirely against this checkout's uv venv.  The
# host-leg twin is build_ffi_host.sh in this directory.
#
#   config/perlmutter/build_ffi_cuda.sh [--fresh]
#
# Needs: `uv sync --extra cuda13` run in this checkout.  The cuda13 extra is
# the whole CUDA side of the build: nvcc and the CUDA math libraries (the
# wheels JAX's plugin maps at run time), NCCL, and cuSOLVERMp / cuBLASMp.
# The same toolchain config/local/build_ffi.sh uses; no cudatoolkit module.
#
# Output: <checkout>/src/ffi/cpp/build/liblorrax_ffi.so, where the loader
# looks by default, so a run needs no LORRAX_FFI_SO.
#
# The paths are config/perlmutter/ffi_site.cmake's, through the
# perlmutter-cuda preset (src/ffi/cpp/CMakePresets.json).  The leg is compiled
# with plain GCC, not the Cray `cc` wrapper: with craype-accel-nvidia80 loaded
# the wrapper injects the CUDA-12 libmpi_gtl_cuda.  cuSOLVERMp and cuBLASMp
# communicate through NCCL, and Cray MPICH GPU support stays off.  The MPI is
# named explicitly, pinned once for both legs in ffi_mpi.sh.
# ============================================================================
set -euo pipefail

LORRAX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$LORRAX_ROOT/src/ffi/cpp"
BUILD="$SRC/build"
PY="$LORRAX_ROOT/.venv/bin/python"

# --- site pins: asserted against the venv, which uv.lock fills --------------
CUSOLVERMP_VERSION="0.9.1.9318.post1"   # also module/stack.sh, pyproject.toml
CUBLASMP_VERSION="0.10.0.3695"          # also module/stack.sh, pyproject.toml
# NCCL: the pip wheel nvidia-nccl-cu13, on every route (owner decision
# 2026-10-17): the uv.lock pin a clone's JAX maps, also installed in the module
# venv (module/stack.sh).  The leg compiles and links against the wheel and its
# closure records it; a run that maps any other libnccl refuses.  Never the HPC
# SDK's comm_libs NCCL.
NCCL_WHEEL="nvidia-nccl-cu13"
NCCL_VERSION="2.30.7"
# shellcheck disable=SC1091
source "$LORRAX_ROOT/config/perlmutter/ffi_mpi.sh"

die() { echo "[build_ffi_cuda] REFUSED: $*" >&2; exit 2; }

[[ -x "$PY" ]] || die "no venv at $LORRAX_ROOT/.venv; run 'uv sync --extra cuda13' in $LORRAX_ROOT first"
"$PY" -c 'import jax_plugins.xla_cuda13' 2>/dev/null \
    || die "the venv has no CUDA-13 jax plugin; run 'uv sync --extra cuda13' (the plain 'uv sync' installs CPU jax only)"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
NCCL="$SITE/nvidia/nccl"
have() { ls -d "$SITE/$1"-*.dist-info 2>/dev/null | sed "s/.*$1-\(.*\)\.dist-info/\1/"; }
[[ "$(have nvidia_nccl_cu13)" == "$NCCL_VERSION" ]] \
    || die "the venv's $NCCL_WHEEL is '$(have nvidia_nccl_cu13)', the pin is $NCCL_VERSION (uv.lock; module/stack.sh)"
[[ "$(have nvidia_cusolvermp_cu13)" == "$CUSOLVERMP_VERSION" && "$(have nvidia_cublasmp_cu13)" == "$CUBLASMP_VERSION" ]] \
    || die "the venv's cuSOLVERMp/cuBLASMp are '$(have nvidia_cusolvermp_cu13)'/'$(have nvidia_cublasmp_cu13)', the pins are $CUSOLVERMP_VERSION/$CUBLASMP_VERSION; run 'uv sync --extra cuda13'"

# CMakePresets.json needs cmake >= 3.23 (the system cmake is older).
if ! printf '3.23\n%s\n' "$(cmake --version 2>/dev/null | awk 'NR==1{print $3}')" | sort -VC; then
    # shellcheck disable=SC1091
    type module >/dev/null 2>&1 || source /opt/cray/pe/lmod/lmod/init/bash
    module load cmake
fi

# --- configure + build ------------------------------------------------------
# The leg's only search path is $ORIGIN (CMakeLists.txt).  The post-build step
# (cmake/LorraxPostBuild.cmake) records the leg's dependency closure — the
# cuSOLVERMp/cuBLASMp pair, the CUDA math/NCCL files the venv's JAX plugin
# maps, and the pinned MPI/HDF5 — which the loader enforces on every route (a
# clone, the module and a sealed bundle open exactly the recorded files and
# refuse a process that maps another copy, such as the cuSOLVERMp 0.8 that
# the cudatoolkit module's HPC SDK puts on LD_LIBRARY_PATH under the same
# SONAME), then runs scripts/verify_ffi_build.sh, so the build fails if any
# gate does.
[[ "${1:-}" == "--fresh" ]] && rm -rf "${BUILD:?}"
(cd "$SRC" && cmake --preset perlmutter-cuda --fresh -B "$BUILD")
cmake --build "$BUILD" --parallel "${LORRAX_BUILD_JOBS:-16}"
SO="$BUILD/liblorrax_ffi.so"
[[ -f "$SO" ]] || { echo "[build_ffi_cuda] FAILED: no .so produced" >&2; exit 1; }

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
    leg=cuda nvcc="$("$SITE/nvidia/cu13/bin/nvcc" --version | tail -1)" cuda_root="$SITE/nvidia/cu13" \
    cusolvermp="$CUSOLVERMP_VERSION" cublasmp="$CUBLASMP_VERSION" \
    phdf5="$LORRAX_PM_HDF5_MODULE" phdf5_root="$LORRAX_PM_HDF5_ROOT" \
    hdf5_soversion=310 mpi="$LORRAX_PM_MPICH_MODULE" mpi_root="$LORRAX_PM_MPICH_ROOT" \
    nccl="$NCCL_WHEEL==$NCCL_VERSION" nccl_root="$NCCL" \
    closure_sha256="$CLOSURE_SHA" \
    || echo "[build_ffi_cuda] WARNING: provenance stamp failed" >&2

echo "[build_ffi_cuda] done: $SO"
