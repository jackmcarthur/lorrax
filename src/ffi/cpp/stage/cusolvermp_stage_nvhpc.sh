#!/usr/bin/env bash
# stage_nvhpc.sh — stage the minimum NVIDIA HPC SDK subset (cuSOLVERMp
# + libcal + headers) into a /pscratch dir Shifter can bind-mount.
#
# Shifter at NERSC blocks --volume from /opt/nvidia, so we have to copy
# the pieces the FFI actually uses into /pscratch first.  Less than
# 100 MB in total.  One-time setup.
#
# Run on a NERSC login node (no container, no GPU):
#
#   module load nvhpc              # (only needed to discover HPCSDK_ROOT)
#   src/ffi/cpp/stage/cusolvermp_stage_nvhpc.sh
#
# Override the source version / install prefix by setting env vars:
#   NVHPC_VERSION        default 25.5
#   NVHPC_CUDA           default 12.9
#   NVHPC_ROOT           default /opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_VERSION}
#   LORRAX_FFI_NVHPC_DIR default /pscratch/sd/${USER:0:1}/${USER}/lorrax_nvhpc

set -euo pipefail

: "${NVHPC_VERSION:=25.5}"
: "${NVHPC_CUDA:=12.9}"
: "${NVHPC_ROOT:=/opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_VERSION}}"
: "${LORRAX_FFI_NVHPC_DIR:=/pscratch/sd/${USER:0:1}/${USER}/lorrax_nvhpc}"

STAGE="${LORRAX_FFI_NVHPC_DIR}/${NVHPC_VERSION}_cuda${NVHPC_CUDA}"

if [[ ! -d "${NVHPC_ROOT}" ]]; then
    echo "stage_nvhpc.sh: NVHPC_ROOT=${NVHPC_ROOT} not found."
    echo "  Load the nvhpc module or set NVHPC_ROOT explicitly."
    exit 2
fi

echo "[stage] src: ${NVHPC_ROOT}"
echo "[stage] dst: ${STAGE}"

mkdir -p \
    "${STAGE}/math_libs/${NVHPC_CUDA}/lib64" \
    "${STAGE}/math_libs/${NVHPC_CUDA}/targets/x86_64-linux/include" \
    "${STAGE}/comm_libs/${NVHPC_CUDA}/nccl/include"

cp -a "${NVHPC_ROOT}/math_libs/${NVHPC_CUDA}/targets/x86_64-linux/include/"cusolverMp*.h \
      "${STAGE}/math_libs/${NVHPC_CUDA}/targets/x86_64-linux/include/"

cp -a "${NVHPC_ROOT}/math_libs/${NVHPC_CUDA}/lib64/"libcusolverMp* \
      "${NVHPC_ROOT}/math_libs/${NVHPC_CUDA}/lib64/"libcal* \
      "${STAGE}/math_libs/${NVHPC_CUDA}/lib64/"

cp -a "${NVHPC_ROOT}/comm_libs/${NVHPC_CUDA}/nccl/include/nccl.h" \
      "${STAGE}/comm_libs/${NVHPC_CUDA}/nccl/include/"

echo "[stage] done.  libcusolverMp:"
readelf -d "${STAGE}/math_libs/${NVHPC_CUDA}/lib64/libcusolverMp.so" 2>/dev/null | \
    grep -E "SONAME" | head -2
echo
echo "[stage] dst: ${LORRAX_FFI_NVHPC_DIR}"
echo "[stage] run_shifter.sh picks this up automatically."
