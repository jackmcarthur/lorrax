#!/usr/bin/env bash
# stage_pypi.sh — stage a standalone CUDA-12 cuSolverMp build from PyPI.
#
# Sister script to stage_nvhpc.sh (which stages cuSolverMp from a full
# NVHPC SDK install).  PyPI ships standalone ``nvidia-cusolvermp-cu12``
# wheels that are smaller (~30–110 MB), don't drag in the rest of the
# HPC SDK, and decouple cuSolverMp upgrades from the NVHPC release
# cadence. The default is the full released 0.9.1 CUDA-12 provider, which
# fixes the large-N eigenvector failures of 0.4.2--0.9.0. Stage its matching
# cuBLASMp with cusolvermp_stage_cublasmp_redist.sh and NCCL >= 2.29.2.
#
# Run on a login node (no container, no GPU):
#   src/ffi/cpp/stage/cusolvermp_stage_pypi.sh
#
# Override defaults with env vars:
#   CUSOLVERMP_VERSION   PyPI version, default 0.9.1.9318.post1
#   CUSOLVERMP_SHORT     stage tag, default first three version components
#   NVHPC_CUDA           CUDA toolkit subdir, default 12.9
#   LORRAX_FFI_NVHPC_DIR root of the staging tree
#
# A pre-0.9.1 stage is not supported, even where its small-case tests pass.

set -euo pipefail

: "${CUSOLVERMP_VERSION:=0.9.1.9318.post1}"
# Strip the trailing build number for the short tag (e.g. 0.9.1.9318.post1 → 0.9.1).
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

# The released wheel supplies its own matching API header. Do not substitute
# a header from an old CAL provider when staging a supported NCCL release.
WHEEL_HEADER="${TMPDIR}/extracted/nvidia/cu12/include/cusolverMp.h"
if [[ ! -f "${WHEEL_HEADER}" ]]; then
    echo "[stage] FAILED: released cuSOLVERMp wheel lacks cusolverMp.h"
    exit 2
fi
cp "${WHEEL_HEADER}" "${INC_DIR}/cusolverMp.h"

echo "[stage] done."
ls -la "${LIB_DIR}/libcusolverMp.so"* "${INC_DIR}/cusolverMp.h"
