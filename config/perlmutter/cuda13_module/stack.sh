#!/usr/bin/env bash
# Shared site facts for the experimental bare-host CUDA 13 module lane.
# Source this file; setup_env.sh, build_ffi.sh, and install_module.sh do so.

LORRAX_CUDA13_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LORRAX_CUDA13_ROOT="${LORRAX_CUDA13_ROOT:-$(cd "$LORRAX_CUDA13_CONFIG_DIR/../../.." && pwd)}"
LORRAX_CUDA13_ENV="${LORRAX_CUDA13_ENV:-$LORRAX_CUDA13_ROOT/.venv}"
LORRAX_CUDA13_NATIVE="${LORRAX_CUDA13_NATIVE:-$LORRAX_CUDA13_ROOT/.native}"
LORRAX_CUDA13_STAGE="${LORRAX_CUDA13_STAGE:-$LORRAX_CUDA13_NATIVE/cusolvermp-0.9.1_cuda13.2}"
LORRAX_CUDA13_BUILD="${LORRAX_CUDA13_BUILD:-$LORRAX_CUDA13_ROOT/src/ffi/cpp/build_cuda13_module}"

LORRAX_CUDA13_SDK="${LORRAX_CUDA13_SDK:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.5}"
LORRAX_CUDA13_CUDA="${LORRAX_CUDA13_CUDA:-$LORRAX_CUDA13_SDK/cuda/13.2}"
LORRAX_CUDA13_MATH="${LORRAX_CUDA13_MATH:-$LORRAX_CUDA13_SDK/math_libs/13.2}"
LORRAX_CUDA13_COMM="${LORRAX_CUDA13_COMM:-$LORRAX_CUDA13_SDK/comm_libs/13.2}"
LORRAX_CUDA13_COMPAT="${LORRAX_CUDA13_COMPAT:-/usr/local/cuda-13.2/compat}"

LORRAX_CUDA13_PYTHON_MODULE="${LORRAX_CUDA13_PYTHON_MODULE:-python/3.12-26.1.0}"
LORRAX_CUDA13_TOOLKIT_MODULE="${LORRAX_CUDA13_TOOLKIT_MODULE:-cudatoolkit/13.2}"
LORRAX_CUDA13_COMPILER_MODULE="${LORRAX_CUDA13_COMPILER_MODULE:-gcc-native/14}"
LORRAX_CUDA13_CMAKE_MODULE="${LORRAX_CUDA13_CMAKE_MODULE:-cmake}"

LORRAX_CUDA13_JAX_VERSION="${LORRAX_CUDA13_JAX_VERSION:-0.9.1}"
LORRAX_CUDA13_CUDNN_VERSION="${LORRAX_CUDA13_CUDNN_VERSION:-9.12.0.46}"
LORRAX_CUDA13_CUSOLVERMP_VERSION="${LORRAX_CUDA13_CUSOLVERMP_VERSION:-0.9.1.9318.post1}"
LORRAX_CUDA13_CUBLASMP_VERSION="${LORRAX_CUDA13_CUBLASMP_VERSION:-0.10.0.3695}"

lorrax_cuda13_load_runtime_modules() {
    if ! type module >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source /usr/share/lmod/lmod/init/bash
    fi
    # Both are Lmod families.  Unload first so a caller's default Python 3.13
    # or CUDA 12.9 cannot survive an apparently successful `module load`.
    module unload python >/dev/null 2>&1 || true
    module unload cudatoolkit >/dev/null 2>&1 || true
    module load "$LORRAX_CUDA13_PYTHON_MODULE" "$LORRAX_CUDA13_TOOLKIT_MODULE"
}

lorrax_cuda13_load_build_modules() {
    lorrax_cuda13_load_runtime_modules
    module load "$LORRAX_CUDA13_COMPILER_MODULE" "$LORRAX_CUDA13_CMAKE_MODULE"
}

lorrax_cuda13_site_packages() {
    "$LORRAX_CUDA13_ENV/bin/python" -c \
        'import sysconfig; print(sysconfig.get_paths()["purelib"])'
}

lorrax_cuda13_cudnn_lib() {
    printf '%s/nvidia/cudnn/lib\n' "$(lorrax_cuda13_site_packages)"
}

lorrax_cuda13_runtime_ld_library_path() {
    printf '%s:%s:%s:%s:%s:%s:%s:%s\n' \
        "$LORRAX_CUDA13_STAGE/lib" \
        "$(lorrax_cuda13_cudnn_lib)" \
        "$LORRAX_CUDA13_CUDA/lib64" \
        "$LORRAX_CUDA13_CUDA/nvvm/lib64" \
        "$LORRAX_CUDA13_CUDA/extras/CUPTI/lib64" \
        "$LORRAX_CUDA13_MATH/lib64" \
        "$LORRAX_CUDA13_COMM/nccl/lib" \
        "$LORRAX_CUDA13_COMPAT"
}

lorrax_cuda13_pythonpath() {
    printf '%s:%s\n' \
        "$LORRAX_CUDA13_ROOT/src" \
        "$LORRAX_CUDA13_ROOT/services/distrib_la/src"
}
