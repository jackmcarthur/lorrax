#!/usr/bin/env bash
# stage_pypi.sh — stage a standalone CUDA-12 cuSolverMp build from PyPI.
#
# Sister script to stage_nvhpc.sh (which stages cuSolverMp from a full
# NVHPC SDK install).  PyPI ships standalone ``nvidia-cusolvermp-cu12``
# wheels that are smaller (~30–110 MB), don't drag in the rest of the
# HPC SDK, and decouple cuSolverMp upgrades from the NVHPC release
# cadence.  We use this to ship cuSolverMp 0.7.2 (the first version
# with both the CAL→NCCL ABI fix AND the 0.7.0/0.7.1 race-condition
# follow-up) on top of the JAX 25.04 container's CUDA 12.9 runtime.
#
# Run on a NERSC login node (no container, no GPU):
#
#   src/ffi/cpp/stage/cusolvermp_stage_pypi.sh                 # 0.7.2 default
#   CUSOLVERMP_VERSION=0.8.0.3126 src/ffi/cpp/stage/cusolvermp_stage_pypi.sh
#
# Override defaults with env vars:
#   CUSOLVERMP_VERSION   PyPI version, default 0.7.2.888
#   CUSOLVERMP_SHORT     short name for the staging dir, default 0.7.2
#   NVHPC_CUDA           CUDA toolkit subdir, default 12.9 (matches the
#                        JAX 25.04-py3 container's libcudart.so.12.9.37)
#   LORRAX_FFI_NVHPC_DIR root of staging tree, default
#                        /pscratch/sd/${USER:0:1}/${USER}/lorrax_nvhpc
#
# Note: cuSolverMp 0.8.0+ requires NCCL ≥ 2.27 (ncclCommWindowRegister
# symbol).  The JAX 25.04 container ships NCCL 2.26.3, so 0.8.0 won't
# dlopen there — stage NCCL 2.27+ separately or stay on 0.7.x.

set -euo pipefail

: "${CUSOLVERMP_VERSION:=0.7.2.888}"
# Strip the trailing build number for the short tag (e.g. 0.7.2.888 → 0.7.2).
: "${CUSOLVERMP_SHORT:=$(echo "${CUSOLVERMP_VERSION}" | cut -d. -f1-3)}"
: "${NVHPC_CUDA:=12.9}"
: "${LORRAX_FFI_NVHPC_DIR:=/pscratch/sd/${USER:0:1}/${USER}/lorrax_nvhpc}"

STAGE="${LORRAX_FFI_NVHPC_DIR}/${CUSOLVERMP_SHORT}_cuda${NVHPC_CUDA}"

echo "[stage] cusolvermp ${CUSOLVERMP_VERSION} → ${STAGE}"

LIB_DIR="${STAGE}/math_libs/${NVHPC_CUDA}/lib64"
INC_DIR="${STAGE}/math_libs/${NVHPC_CUDA}/targets/x86_64-linux/include"
mkdir -p "${LIB_DIR}" "${INC_DIR}"

# Resolve the wheel URL via PyPI's JSON API (so we don't have to hardcode
# the manylinux tag for each version).
WHEEL_URL=$(
    curl -sS "https://pypi.org/pypi/nvidia-cusolvermp-cu12/${CUSOLVERMP_VERSION}/json" \
    | python3 -c "
import json, sys
d = json.load(sys.stdin)
for f in d.get('urls', []):
    if 'manylinux' in f['filename'] and 'x86_64' in f['filename']:
        print(f['url']); break
"
)
if [[ -z "${WHEEL_URL}" ]]; then
    echo "[stage] FAILED: no x86_64 wheel for nvidia-cusolvermp-cu12==${CUSOLVERMP_VERSION}"
    exit 2
fi
echo "[stage] wheel: ${WHEEL_URL}"

TMPDIR=$(mktemp -d)
trap 'rm -rf "${TMPDIR}"' EXIT
curl -sLo "${TMPDIR}/cusolvermp.whl" "${WHEEL_URL}"
unzip -q -o "${TMPDIR}/cusolvermp.whl" -d "${TMPDIR}/extracted"

# The wheel ships nvidia/cu12/lib/libcusolverMp.so.0 (the SO_NAME), not
# the versioned filename.  We stage as the versioned name + soname/devel
# symlinks to match the layout stage_nvhpc.sh produces, so the rest of
# the toolchain (CMake search, runtime LD_LIBRARY_PATH) is path-symmetric.
cp "${TMPDIR}/extracted/nvidia/cu12/lib/libcusolverMp.so.0" \
   "${LIB_DIR}/libcusolverMp.so.${CUSOLVERMP_SHORT}.0"
ln -sf "libcusolverMp.so.${CUSOLVERMP_SHORT}.0" "${LIB_DIR}/libcusolverMp.so.0"
ln -sf "libcusolverMp.so.0"                     "${LIB_DIR}/libcusolverMp.so"

# Header: 0.7.x wheels ship one; 0.7.2 in particular doesn't (only the
# .so).  Reuse 0.6.0's header if absent — the C symbols our FFI calls
# (cusolverMpCreate, cusolverMpCreateDeviceGrid, cusolverMpGetVersion,
# cusolverMpGetrf, cusolverMpGetrs, cusolverMpPotrf, cusolverMpPotrs)
# are ABI-stable across 0.6 → 0.7+; only the cusolverMpCreateDeviceGrid
# COMM-PARAMETER TYPE changed (cal_comm_t → ncclComm_t), and our patched
# FFI casts at the call site so the C declaration's tag doesn't matter.
WHEEL_HEADER="${TMPDIR}/extracted/nvidia/cu12/include/cusolverMp.h"
if [[ -f "${WHEEL_HEADER}" ]]; then
    cp "${WHEEL_HEADER}" "${INC_DIR}/cusolverMp.h"
    echo "[stage] using wheel header"
else
    LEGACY_HEADER="${LORRAX_FFI_NVHPC_DIR}/25.5_cuda12.9/math_libs/${NVHPC_CUDA}/targets/x86_64-linux/include/cusolverMp.h"
    if [[ -f "${LEGACY_HEADER}" ]]; then
        cp "${LEGACY_HEADER}" "${INC_DIR}/cusolverMp.h"
        echo "[stage] wheel had no header; reused 0.6.0's at ${LEGACY_HEADER}"
    else
        echo "[stage] FAILED: no header in wheel and no 0.6.0 fallback at ${LEGACY_HEADER}"
        exit 2
    fi
fi

echo "[stage] done."
ls -la "${LIB_DIR}/libcusolverMp.so"* "${INC_DIR}/cusolverMp.h"
