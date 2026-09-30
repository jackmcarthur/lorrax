# shellcheck shell=bash
# ============================================================================
# The Perlmutter module stack: pins and prefix layout, sourced by every script
# in this directory (docs/installation/perlmutter-module.md).
#
#   export LORRAX_MODULE_PREFIX=/abs/install/prefix     # required, no default
#
# Layout under the prefix:
#   python/                          uv-managed CPython the venv is built on
#   venv/                            the runtime venv (setup_env.sh)
#   build/                           SLATE + cuSOLVERMp stage, shared by builds
#   releases/source-<rev8>/          git-archive snapshots (publish.sh)
#   releases/<rev8>-bundle-<id12>/   sealed FFI bundles (publish.sh)
#   modulefiles/<name>/<version>.lua the published module (publish.sh)
#
# The CUDA toolkit comes from the site's cudatoolkit module; the JAX CUDA
# plugin is the `cuda13-local` build, which installs no CUDA runtime of its
# own.  The FFI legs are built by ../build_ffi_{host,cuda}.sh, whose MPI and
# HDF5 pins (../ffi_mpi.sh) this file does not repeat.
# ============================================================================

LORRAX_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

lorrax_module_die() { echo "[lorrax-module] REFUSED: $*" >&2; exit 2; }

[[ -n "${LORRAX_MODULE_PREFIX:-}" ]] \
    || lorrax_module_die "set LORRAX_MODULE_PREFIX to the absolute install prefix"
[[ "$LORRAX_MODULE_PREFIX" == /* ]] \
    || lorrax_module_die "LORRAX_MODULE_PREFIX must be absolute: $LORRAX_MODULE_PREFIX"
PREFIX="$(realpath -m "$LORRAX_MODULE_PREFIX")"
VENV="$PREFIX/venv"
RELEASES="$PREFIX/releases"
MODULE_NAME="${LORRAX_MODULE_NAME:-lorrax}"
MODULE_VERSION="${LORRAX_MODULE_VERSION:-0.1.0}"

# --- pins --------------------------------------------------------------------
PYTHON_VERSION="3.12"
TOOLKIT_MODULE="cudatoolkit/13.2"
JAX_VERSION="0.9.1"
CUDNN_VERSION="9.12.0.46"            # nvidia-cudnn-cu13 (NERSC cuDNN targets CUDA 12)
CUSOLVERMP_VERSION="0.9.1.9318.post1" # also pinned in ../build_ffi_cuda.sh
CUBLASMP_VERSION="0.10.0.3695"        # also pinned in ../build_ffi_cuda.sh
MATHDX_VERSION="25.6.0"              # pyproject.toml [cuda13]; required on NVIDIA GPUs
PHDF5_ROOT="/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.3"   # ../build_ffi_cuda.sh
# shellcheck disable=SC1091
source "$LORRAX_MODULE_DIR/../ffi_mpi.sh"

lorrax_module_load_toolkit() {
    if ! type module >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source /opt/cray/pe/lmod/lmod/init/bash
    fi
    module unload cudatoolkit >/dev/null 2>&1 || true
    module load "$TOOLKIT_MODULE"
    CUDA="${CUDA_HOME:?$TOOLKIT_MODULE did not set CUDA_HOME}"
    SDK="$(cd "$CUDA/../.." && pwd)"
    CUDA_VER="$(basename "$CUDA")"
}

lorrax_module_site_packages() {
    "$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
}

# A clean Git checkout root at a full revision; prints the revision.
lorrax_module_clean_rev() {
    local src="$1" top rev
    top="$(git -C "$src" rev-parse --show-toplevel 2>/dev/null)" \
        || lorrax_module_die "not a Git checkout: $src"
    [[ "$(realpath -e "$src")" == "$(realpath -e "$top")" ]] \
        || lorrax_module_die "name the checkout root ($top), not $src"
    rev="$(git -C "$src" rev-parse --verify HEAD)"
    [[ "$rev" =~ ^[0-9a-f]{40}$ ]] || lorrax_module_die "not a full revision: $rev"
    [[ -z "$(git -C "$src" status --porcelain)" ]] \
        || lorrax_module_die "checkout is not clean: $src ($(git -C "$src" status --porcelain | head -3 | tr '\n' ' '))"
    printf '%s\n' "$rev"
}
