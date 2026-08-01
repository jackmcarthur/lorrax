#!/usr/bin/env bash
# Build liblorrax_ffi.so inside the Shifter container.
#
# Typical usage (from lorrax root):
#   lxalloc                                       # 1-node, 4-GPU alloc
#   src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh
#
# The sibling `run_shifter.sh` launches Shifter with the HPC SDK
# bind-mounted at /lorrax_nvhpc so the build can see cuSOLVERMp headers
# and libraries.  Alternatively, run this script by hand inside a
# shifter shell that already has /lorrax_nvhpc mounted.
#
# Output: src/ffi/cpp/build/liblorrax_ffi*.so

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# LORRAX_FFI_BUILD_DIR: alternate build dir (default: in-tree build/).
# Useful for building against a different SLATE install without racing a
# concurrently-used in-tree .so; point LORRAX_FFI_SO at the result.
BUILD_DIR="${LORRAX_FFI_BUILD_DIR:-${SCRIPT_DIR}/build}"

# Default HPC SDK install (a staged subset; see AGENTS.md).  The staged
# tree is bind-mounted into Shifter at this path.  Override by setting
# LORRAX_NVHPC_ROOT.
NVHPC_ROOT="${LORRAX_NVHPC_ROOT:-/lorrax_nvhpc/25.5_cuda12.9}"
NVHPC_CUDA="${LORRAX_NVHPC_CUDA:-12.9}"

# MPI stack sanity check.  The CMake config (see CMakeLists.txt ~L283-305)
# reads LORRAX_MPI_INCLUDE_DIR + LORRAX_MPICH_LIB_DIR and silently falls
# back to /opt/hpcx/ompi/lib if either is unset.  That fallback links the
# FFI against HPC-X OpenMPI (DT_NEEDED libmpi.so.40), which then races
# the Cray MPICH bind-mount at runtime — see KNOWN_SANDBOX_ERRORS.md
# 2026-05-10.  This script is normally invoked via run_shifter.sh which
# exports both vars; fail loudly when they're missing instead of producing
# a silently-broken .so.  Set LORRAX_FFI_ALLOW_DEFAULT_MPI=1 to opt out
# (e.g. for the OpenMPI build path on non-Cray sites).
if [[ -z "${LORRAX_FFI_ALLOW_DEFAULT_MPI:-}" ]]; then
    if [[ -z "${LORRAX_MPI_INCLUDE_DIR:-}" || -z "${LORRAX_MPICH_LIB_DIR:-}" ]]; then
        echo "[build] ERROR: LORRAX_MPI_INCLUDE_DIR / LORRAX_MPICH_LIB_DIR not set." >&2
        echo "[build]   include='${LORRAX_MPI_INCLUDE_DIR:-<unset>}'" >&2
        echo "[build]   libdir ='${LORRAX_MPICH_LIB_DIR:-<unset>}'"   >&2
        echo "[build] Without these CMake falls back to HPC-X OpenMPI at /opt/hpcx/ompi" >&2
        echo "[build] and the produced .so will ask for libmpi.so.40 at run time —" >&2
        echo "[build] the wrong MPI library name for Cray MPICH, which segfaults" >&2
        echo "[build] the runtime path" >&2
        echo "[build] (see KNOWN_SANDBOX_ERRORS.md 2026-05-10)." >&2
        echo "[build] Invoke via src/ffi/cpp/run_shifter.sh, which exports both," >&2
        echo "[build] or set LORRAX_FFI_ALLOW_DEFAULT_MPI=1 to bypass this check." >&2
        exit 2
    fi
fi

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
    -DLORRAX_FFI_PLATFORM=cuda \
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
