#!/usr/bin/env bash
# stage_cublasmp_redist.sh — download NVIDIA's standalone cuBLASMp redist
# and stage it next to a standalone cuSOLVERMp stage so the two comm ABIs
# match at runtime.
#
# Why this exists: the HPC-SDK stage (stage_nvhpc.sh, e.g. 25.5) ships
# cuBLASMp 0.4.x (CAL comm ABI).  A newer standalone cuSOLVERMp stage
# (e.g. 0.7.2, NCCL comm ABI) contains NO cuBLASMp, so at runtime the
# loader fell back to the old 0.4.0 via the .so RUNPATH while cuSOLVERMp
# 0.7.2 resolved via LD_LIBRARY_PATH — a mixed-generation stack in one
# process.  cublasMpGridCreate then received an ncclComm_t where the
# 0.4.x library expected a cal_comm_t and failed with status=6 on every
# mesh (found 2026-07-10; the cublasmp FFI backend had been dead since
# the 0.7.2 cuSOLVERMp stage on 2026-05-10).
#
# Pairing rule (comm ABI generations must match):
#   cuSOLVERMp <= 0.6.x  <->  cuBLASMp <= 0.4.x   (CAL)
#   cuSOLVERMp >= 0.7.0  <->  cuBLASMp >= 0.5.0   (NCCL)
#
# Run on a login node (needs outbound HTTPS):
#
#   src/ffi/cusolvermp/scripts/stage_cublasmp_redist.sh
#
# Overrides:
#   CUBLASMP_VERSION      default 0.5.1.65  (full redist version string)
#   CUBLASMP_CUDA         default cuda12
#   LORRAX_FFI_NVHPC_DIR  default $HOME/software/lorrax_nvhpc
#   LORRAX_CUSOLVERMP_STAGE  default 0.7.2_cuda12.9 (stage dir to add to)

set -euo pipefail

: "${CUBLASMP_VERSION:=0.5.1.65}"
: "${CUBLASMP_CUDA:=cuda12}"
: "${LORRAX_FFI_NVHPC_DIR:=$HOME/software/lorrax_nvhpc}"
: "${LORRAX_CUSOLVERMP_STAGE:=0.7.2_cuda12.9}"

STAGE="${LORRAX_FFI_NVHPC_DIR}/${LORRAX_CUSOLVERMP_STAGE}/math_libs/12.9"
ARCHIVE="libcublasmp-linux-x86_64-${CUBLASMP_VERSION}_${CUBLASMP_CUDA}-archive"
URL="https://developer.download.nvidia.com/compute/cublasmp/redist/libcublasmp/linux-x86_64/${ARCHIVE}.tar.xz"

if [[ ! -d "${STAGE}/lib64" ]]; then
    echo "stage_cublasmp_redist.sh: ${STAGE}/lib64 not found."
    echo "  Stage cuSOLVERMp first (or set LORRAX_CUSOLVERMP_STAGE)."
    exit 2
fi

DL="${LORRAX_FFI_NVHPC_DIR}/_dl"
mkdir -p "${DL}"
cd "${DL}"

if [[ ! -f "${ARCHIVE}.tar.xz" ]]; then
    echo "[stage] downloading ${URL}"
    curl -fsSO "${URL}"
fi
tar xf "${ARCHIVE}.tar.xz"

cp -a "${ARCHIVE}/lib/"libcublasmp.so* "${STAGE}/lib64/"
mkdir -p "${STAGE}/targets/x86_64-linux/include"
cp -a "${ARCHIVE}/include/cublasmp.h" "${STAGE}/targets/x86_64-linux/include/"

# libcublasmp NEEDs libnvshmem_host.so.3, and RUNPATH is NOT transitive:
# liblorrax_ffi.so's runpath doesn't apply when the loader resolves
# cublasmp's own deps, so nvshmem must sit in the same staged lib64
# (which is on LD_LIBRARY_PATH).  Copy it from the HPC-SDK stage.
NVSHMEM_SRC="${LORRAX_FFI_NVHPC_DIR}/25.5_cuda12.9/math_libs/12.9/lib64"
if [[ -e "${NVSHMEM_SRC}/libnvshmem_host.so.3" ]]; then
    cp -a "${NVSHMEM_SRC}/"libnvshmem_host.so* \
          "${NVSHMEM_SRC}/"nvshmem_bootstrap_uid.so* \
          "${STAGE}/lib64/"
else
    echo "[stage] WARNING: libnvshmem_host.so.3 not found at ${NVSHMEM_SRC};"
    echo "        stage it manually or libcublasmp will fail to load."
fi

echo "[stage] done.  ${STAGE}/lib64:"
ls -l "${STAGE}/lib64/" | grep cublasmp
echo
echo "[stage] runtime check: ensure_cublasmp() logs the loaded version +"
echo "        comm path on rank 0 at first cuBLASMp call."
