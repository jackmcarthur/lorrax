#!/usr/bin/env bash
# Shared facts for the local CUDA 13 lane: one Ubuntu machine with an NVIDIA
# driver (a workstation, a laptop under WSL2, or a rented single-node GPU box).
# Source this file; setup_env.sh, build_ffi.sh, build_ffi_host.sh and launch.sh do so.
#
# This is the local twin of config/perlmutter/module/stack.sh.  The
# two are deliberately STRUCTURALLY PARALLEL so `diff` shows only values.
# The load-bearing difference: Perlmutter takes the CUDA toolchain from
# NERSC's Lmod modules and NVHPC; here everything comes from pip wheels.
# jax[cuda13] pulls the full CUDA 13 userspace INCLUDING nvcc, and the
# CUDA-13-generation wheels install into ONE shared prefix,
# site-packages/nvidia/cu13/{bin,include,lib} — which is exactly the
# CUDA_TOOLKIT_ROOT shape src/ffi/cpp/CMakeLists.txt expects.  The only
# things the host must provide are the kernel driver and /dev/nvidia*.

LORRAX_LOCAL_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LORRAX_LOCAL_ROOT="${LORRAX_LOCAL_ROOT:-$(cd "$LORRAX_LOCAL_CONFIG_DIR/../.." && pwd)}"
LORRAX_LOCAL_ENV="${LORRAX_LOCAL_ENV:-$LORRAX_LOCAL_ROOT/.venv}"
LORRAX_LOCAL_NATIVE="${LORRAX_LOCAL_NATIVE:-$LORRAX_LOCAL_ROOT/.native}"
LORRAX_LOCAL_STAGE="${LORRAX_LOCAL_STAGE:-$LORRAX_LOCAL_NATIVE/cusolvermp-0.9.1_cuda13}"
LORRAX_LOCAL_BUILD="${LORRAX_LOCAL_BUILD:-$LORRAX_LOCAL_ROOT/src/ffi/cpp/build_local}"

# Parallel HDF5 is built from source against the distro OpenMPI (Ubuntu's
# libhdf5-openmpi-dev is 1.10, below the 1.12 floor in docs/installation/ffi-build.md).
LORRAX_LOCAL_PHDF5_VERSION="${LORRAX_LOCAL_PHDF5_VERSION:-1.14.6}"
LORRAX_LOCAL_PHDF5="${LORRAX_LOCAL_PHDF5:-$LORRAX_LOCAL_NATIVE/phdf5-$LORRAX_LOCAL_PHDF5_VERSION}"

LORRAX_LOCAL_JAX_VERSION="${LORRAX_LOCAL_JAX_VERSION:-0.9.1}"
# The same cuSOLVERMp/cuBLASMp pair as the cuda13 extra in pyproject.toml
# (the Perlmutter lane); cuSOLVERMp 0.9.1.9318.post1 itself requires
# cuBLASMp >= 0.10.0.3695.
LORRAX_LOCAL_CUSOLVERMP_VERSION="${LORRAX_LOCAL_CUSOLVERMP_VERSION:-0.9.1.9318.post1}"
LORRAX_LOCAL_CUBLASMP_VERSION="${LORRAX_LOCAL_CUBLASMP_VERSION:-0.10.0.3695}"
# nvidia-mathdx: header-only cuFFTDx, REQUIRED on NVIDIA GPUs (the k-convolution
# router's only CUDA backend; docs/architecture/decisions.md 2026-09-24).
LORRAX_LOCAL_MATHDX_VERSION="${LORRAX_LOCAL_MATHDX_VERSION:-25.6.0}"

lorrax_local_site_packages() {
    "$LORRAX_LOCAL_ENV/bin/python" -c \
        'import sysconfig; print(sysconfig.get_paths()["purelib"])'
}

# The unified pip CUDA 13 toolkit root (nvcc, headers, every lib*.so).
lorrax_local_cuda_root() {
    printf '%s/nvidia/cu13\n' "$(lorrax_local_site_packages)"
}

# NCCL, cuDNN and NVSHMEM version independently of the toolkit and keep
# their own wheel prefixes beside nvidia/cu13 (measured 2026-08-26,
# nvidia-nccl-cu13 2.28.9 / nvidia-cudnn-cu13 9.13 / jax 0.9.1 closure).
lorrax_local_nccl_root() {
    printf '%s/nvidia/nccl\n' "$(lorrax_local_site_packages)"
}

# The directory the distro OpenMPI's libmpi resolves to (on Ubuntu the multiarch
# system directory, which also holds the distro ScaLAPACK and OpenBLAS the host
# leg links). The dependency closure (write_checkout_closure.py) refuses an MPI
# it finds in no pinned directory, so it is pinned with the rest of the path.
lorrax_local_mpi_libdir() {
    local d; d="$(mpicc --showme:libdirs 2>/dev/null | awk '{print $1}')"
    [[ -n "$d" && -e "$d/libmpi.so" ]] && dirname "$(readlink -f "$d/libmpi.so")"
}

lorrax_local_runtime_ld_library_path() {
    local site cu13; site="$(lorrax_local_site_packages)"; cu13="$(lorrax_local_cuda_root)"
    printf '%s:%s:%s:%s:%s:%s:%s:%s\n' \
        "$LORRAX_LOCAL_STAGE/lib" \
        "$cu13/lib" \
        "$site/nvidia/nccl/lib" \
        "$site/nvidia/cudnn/lib" \
        "$site/nvidia/nvshmem/lib" \
        "$site/nvidia/cublasmp/cu13/lib" \
        "$LORRAX_LOCAL_PHDF5/lib" \
        "$(lorrax_local_mpi_libdir)"
}

lorrax_local_pythonpath() {
    printf '%s:%s\n' \
        "$LORRAX_LOCAL_ROOT/src" \
        "$LORRAX_LOCAL_ROOT/services/distrib_la/src"
}
