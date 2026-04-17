#!/usr/bin/env bash
# Build liblorrax_ffi.so inside the Shifter container.
#
# Typical usage (from lorrax root):
#   lxalloc                                       # 1-node, 4-GPU alloc
#   src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh
#
# The sibling `run_shifter.sh` launches Shifter with the HPC SDK
# bind-mounted at /lorrax_nvhpc so the build can see cuSOLVERMp headers
# and libraries.  Alternatively, run this script by hand inside a
# shifter shell that already has /lorrax_nvhpc mounted.
#
# Output: src/ffi/common/cpp/build/liblorrax_ffi*.so

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Default HPC SDK install (a staged subset; see AGENTS.md).  The staged
# tree is bind-mounted into Shifter at this path.  Override by setting
# LORRAX_NVHPC_ROOT.
NVHPC_ROOT="${LORRAX_NVHPC_ROOT:-/lorrax_nvhpc/25.5_cuda12.9}"
NVHPC_CUDA="${LORRAX_NVHPC_CUDA:-12.9}"

echo "[build] NVHPC_ROOT = ${NVHPC_ROOT}"
echo "[build] CUDA sub   = ${NVHPC_CUDA}"
echo "[build] sources    = ${SCRIPT_DIR}"
echo "[build] build dir  = ${BUILD_DIR}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [[ "${1:-}" == "--fresh" ]]; then
    echo "[build] --fresh: wiping ${BUILD_DIR}"
    rm -rf "${BUILD_DIR:?}/"* "${BUILD_DIR:?}/".??* 2>/dev/null || true
fi

# Force the container's Python (/usr/bin/python3) so we pick up the
# jaxlib headers that match the runtime (/opt/jaxlibs), not some external
# virtualenv's jaxlib that happens to be on PATH first.  The XLA FFI API
# version is baked into the header — headers and runtime must match.
PYTHON_EXE="${LORRAX_FFI_PYTHON:-/usr/bin/python3}"

cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="${PYTHON_EXE}" \
    -DNVHPC_ROOT="${NVHPC_ROOT}" \
    -DNVHPC_CUDA_SUBDIR="${NVHPC_CUDA}"

cmake --build . --parallel

echo
echo "[build] --- artifacts ---"
ls -lh "${BUILD_DIR}"/liblorrax_ffi*.so 2>/dev/null || {
    echo "[build] FAILED: no .so produced."
    exit 1
}

SO_FILE=$(ls "${BUILD_DIR}"/liblorrax_ffi*.so 2>/dev/null | head -1)
echo
echo "[build] --- ldd check ---"
ldd "${SO_FILE}" | grep -Ei 'cusolver|nccl|cuda|cal' || true

echo
echo "[build] done.  .so at: ${SO_FILE}"
