#!/usr/bin/env bash
# Shared site facts for the generic-cloud CUDA 13 lane (rented H100/A100
# boxes: Vast.ai, RunPod, DataCrunch, Lambda — any Ubuntu box with an NVIDIA
# datacenter driver).  Source this file; setup_env.sh and build_ffi.sh do so.
#
# This is the cloud twin of config/perlmutter/cuda13_module/stack.sh.  The
# two are deliberately STRUCTURALLY PARALLEL so `diff` shows only values.
# The load-bearing difference: Perlmutter takes the CUDA toolchain from
# NERSC's Lmod modules and NVHPC; here EVERYTHING comes from pip wheels.
# jax[cuda13] pulls the full CUDA 13 userspace INCLUDING nvcc, and the
# CUDA-13-generation wheels install into ONE shared prefix,
# site-packages/nvidia/cu13/{bin,include,lib} — which is exactly the
# CUDA_TOOLKIT_ROOT shape src/ffi/cpp/CMakeLists.txt expects.  The only
# things the host must provide are the kernel driver and /dev/nvidia*.

LORRAX_CLOUD_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LORRAX_CLOUD_ROOT="${LORRAX_CLOUD_ROOT:-$(cd "$LORRAX_CLOUD_CONFIG_DIR/../.." && pwd)}"
LORRAX_CLOUD_ENV="${LORRAX_CLOUD_ENV:-$LORRAX_CLOUD_ROOT/.venv}"
LORRAX_CLOUD_NATIVE="${LORRAX_CLOUD_NATIVE:-$LORRAX_CLOUD_ROOT/.native}"
LORRAX_CLOUD_STAGE="${LORRAX_CLOUD_STAGE:-$LORRAX_CLOUD_NATIVE/cusolvermp-0.9.1_cuda13}"
LORRAX_CLOUD_BUILD="${LORRAX_CLOUD_BUILD:-$LORRAX_CLOUD_ROOT/src/ffi/cpp/build_cloud}"

# Parallel HDF5 is built from source against the distro OpenMPI (Ubuntu's
# libhdf5-openmpi-dev is 1.10, below the 1.12 floor in src/ffi/PORTING.md).
LORRAX_CLOUD_PHDF5_VERSION="${LORRAX_CLOUD_PHDF5_VERSION:-1.14.6}"
LORRAX_CLOUD_PHDF5="${LORRAX_CLOUD_PHDF5:-$LORRAX_CLOUD_NATIVE/phdf5-$LORRAX_CLOUD_PHDF5_VERSION}"

LORRAX_CLOUD_JAX_VERSION="${LORRAX_CLOUD_JAX_VERSION:-0.9.1}"
# Perlmutter's lane pins cuBLASMp 0.10.0.3695; that wheel is on neither
# pypi.org nor pypi.nvidia.com as of 2026-08-26, so this lane pins the
# newest published cu13 wheel instead.  compat.h guards the API delta.
LORRAX_CLOUD_CUSOLVERMP_VERSION="${LORRAX_CLOUD_CUSOLVERMP_VERSION:-0.9.1.9318.post1}"
LORRAX_CLOUD_CUBLASMP_VERSION="${LORRAX_CLOUD_CUBLASMP_VERSION:-0.9.1.3056}"

lorrax_cloud_site_packages() {
    "$LORRAX_CLOUD_ENV/bin/python" -c \
        'import sysconfig; print(sysconfig.get_paths()["purelib"])'
}

# The unified pip CUDA 13 toolkit root (nvcc, headers, every lib*.so).
lorrax_cloud_cuda_root() {
    printf '%s/nvidia/cu13\n' "$(lorrax_cloud_site_packages)"
}

# NCCL, cuDNN and NVSHMEM version independently of the toolkit and keep
# their own wheel prefixes beside nvidia/cu13 (measured 2026-08-26,
# nvidia-nccl-cu13 2.28.9 / nvidia-cudnn-cu13 9.13 / jax 0.9.1 closure).
lorrax_cloud_nccl_root() {
    printf '%s/nvidia/nccl\n' "$(lorrax_cloud_site_packages)"
}

lorrax_cloud_runtime_ld_library_path() {
    local site cu13; site="$(lorrax_cloud_site_packages)"; cu13="$(lorrax_cloud_cuda_root)"
    printf '%s:%s:%s:%s:%s:%s:%s\n' \
        "$LORRAX_CLOUD_STAGE/lib" \
        "$cu13/lib" \
        "$site/nvidia/nccl/lib" \
        "$site/nvidia/cudnn/lib" \
        "$site/nvidia/nvshmem/lib" \
        "$site/nvidia/cublasmp/cu13/lib" \
        "$LORRAX_CLOUD_PHDF5/lib"
}

lorrax_cloud_pythonpath() {
    printf '%s:%s\n' \
        "$LORRAX_CLOUD_ROOT/src" \
        "$LORRAX_CLOUD_ROOT/services/distrib_la/src"
}
