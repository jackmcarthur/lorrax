#!/usr/bin/env bash
# build_host.sh — build liblorrax_ffi_host.so, the CUDA-free host-platform
# FFI library (SLATE host handlers; see host/CMakeLists.txt).
#
# Runs HOST-SIDE under the Cray PE (no Shifter container) against the
# gpu_backend=none SLATE install.  cmake + the ~30 s compile are fine on a
# login node; to route through the allocation instead:
#
#   srun --jobid=$SLURM_JOBID --overlap -N1 -n1 -c 32 \
#       bash src/ffi/common/cpp/host/build_host.sh
#
# XLA FFI headers: they must match the RUNTIME jaxlib — the Shifter
# container's jax, NOT the host venv's (which may ship a newer XLA FFI API;
# newer-headers-on-older-runtime is the unsupported direction).  This
# script stages the container's jax.ffi.include_dir() tree to
# $HOME/software/lorrax_xla_ffi_headers/<image tag>/ once and reuses it.
#
# Output: src/ffi/common/cpp/host/build/liblorrax_ffi_host.so
#         (ffi_loader.py finds it there; override with LORRAX_FFI_HOST_SO)
#
# Overrides (env):
#   LORRAX_FFI_HOST_BUILD_DIR      alternate build dir
#   LORRAX_SLATE_HOST_INSTALL_DIR  SLATE none-backend install
#                                  (default $HOME/software/slate_builds/cpu/install)
#   LORRAX_XLA_FFI_HEADERS_DIR     pre-staged header dir (skips staging)
#   LORRAX_FFI_HOST_IMAGE          Shifter image to stage headers from
#                                  (default nvcr.io/nvidia/jax:25.04-py3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${LORRAX_FFI_HOST_BUILD_DIR:-${SCRIPT_DIR}/build}"
SLATE_DIR="${LORRAX_SLATE_HOST_INSTALL_DIR:-$HOME/software/slate_builds/cpu/install}"
IMAGE="${LORRAX_FFI_HOST_IMAGE:-nvcr.io/nvidia/jax:25.04-py3}"
IMAGE_TAG="${IMAGE##*:}"
HDR_DIR="${LORRAX_XLA_FFI_HEADERS_DIR:-$HOME/software/lorrax_xla_ffi_headers/${IMAGE_TAG}}"

if [[ ! -d "${SLATE_DIR}/lib64" ]]; then
    echo "[build_host] ERROR: SLATE gpu_backend=none install not found at" >&2
    echo "[build_host]   ${SLATE_DIR}" >&2
    echo "[build_host] Build it first:" >&2
    echo "[build_host]   bash src/ffi/slate/scripts/build_perlmutter.sh cpu" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Modules (same pattern as slate/scripts/build_perlmutter.sh cpu): Cray PE,
# no CUDA — with craype-accel-nvidia80 loaded the CC wrapper would link
# libmpi_gtl_cuda, whose libcuda.so.1 dependency defeats the purpose.
# ---------------------------------------------------------------------------
if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash
fi
module load PrgEnv-gnu
module load cray-libsci
module load cmake
module unload craype-accel-nvidia80 2>/dev/null || true
module unload cudatoolkit           2>/dev/null || true

# ---------------------------------------------------------------------------
# Stage the container's XLA FFI headers (once).
# ---------------------------------------------------------------------------
if [[ ! -f "${HDR_DIR}/xla/ffi/api/ffi.h" ]]; then
    echo "[build_host] staging XLA FFI headers from ${IMAGE} -> ${HDR_DIR}"
    mkdir -p "${HDR_DIR}"
    STAGE_CMD=(shifter "--image=${IMAGE}" bash -c
        "cp -r \$(python3 -c 'import jax.ffi,sys;sys.stdout.write(jax.ffi.include_dir())')/xla '${HDR_DIR}/'")
    if [[ -n "${SLURM_JOBID:-}" ]]; then
        srun --jobid="${SLURM_JOBID}" --overlap -N1 -n1 "${STAGE_CMD[@]}"
    else
        "${STAGE_CMD[@]}"
    fi
fi
if [[ ! -f "${HDR_DIR}/xla/ffi/api/ffi.h" ]]; then
    echo "[build_host] ERROR: header staging failed (no xla/ffi/api/ffi.h" >&2
    echo "[build_host] under ${HDR_DIR}).  If shifter is flaky on the login" >&2
    echo "[build_host] node, rerun with SLURM_JOBID set (see KNOWN_SANDBOX_ERRORS)." >&2
    exit 2
fi

echo "[build_host] SLATE (none) = ${SLATE_DIR}"
echo "[build_host] XLA headers  = ${HDR_DIR}"
echo "[build_host] build dir    = ${BUILD_DIR}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [[ "${1:-}" == "--fresh" ]]; then
    echo "[build_host] --fresh: wiping ${BUILD_DIR}"
    rm -rf "${BUILD_DIR:?}/"* "${BUILD_DIR:?}/".??* 2>/dev/null || true
fi

cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=CC \
    -DLORRAX_XLA_FFI_INCLUDE_DIR="${HDR_DIR}" \
    -DLORRAX_SLATE_HOST_INSTALL_DIR="${SLATE_DIR}"

cmake --build . --parallel

SO_FILE="${BUILD_DIR}/liblorrax_ffi_host.so"
if [[ ! -f "${SO_FILE}" ]]; then
    echo "[build_host] FAILED: no .so produced." >&2
    exit 1
fi

echo
echo "[build_host] --- CUDA-free proof (readelf NEEDED) ---"
readelf -d "${SO_FILE}" | grep NEEDED
if readelf -d "${SO_FILE}" | grep NEEDED | grep -qiE 'cuda|nccl|nvshmem|cal\.so'; then
    echo "[build_host] FAILED: ${SO_FILE} links a CUDA-stack library — the" >&2
    echo "[build_host] host lib must be CUDA-free.  Check that cudatoolkit /" >&2
    echo "[build_host] craype-accel-nvidia80 are unloaded and the SLATE" >&2
    echo "[build_host] install really is the gpu_backend=none variant." >&2
    exit 1
fi

echo
echo "[build_host] done.  .so at: ${SO_FILE}"
