# shellcheck shell=bash
# ============================================================================
# The Perlmutter module stack: pins and prefix layout, sourced by every script
# in this directory (docs/installation/perlmutter.md, section 2).
#
#   export LORRAX_MODULE_PREFIX=/abs/install/prefix     # required, no default
#
# Layout under the prefix:
#   python/                          uv-managed CPython the venv is built on
#   venv/                            the runtime venv (setup_env.sh); LORRAX_MODULE_VENV
#                                    names another absolute path
#   releases/source-<rev8>/          git-archive snapshots (publish.sh)
#   releases/<rev8>-bundle-<id12>/   sealed FFI bundles (publish.sh)
#   modulefiles/<name>/<version>.lua the published module (publish.sh)
#
# The CUDA toolchain, runtime and math libraries are a clone's: the wheels of
# the checkout's uv.lock `cuda13` extra (JAX's cuda13 plugin, nvidia-cublas,
# -cufft, -cusolver, -cusparse, -cuda-runtime, -cuda-nvrtc, -nvjitlink,
# -cuda-nvcc, cuDNN, NCCL, nvidia-mathdx, cuSOLVERMp/cuBLASMp), installed by
# setup_env.sh exactly as locked; the legs are compiled with that nvcc.  The
# modulefile loads the site's cudatoolkit module so PATH and CUDA_HOME are
# those of a clone shell; none
# of its CUDA libraries is on the module's LD_LIBRARY_PATH or in a leg's
# closure.  The FFI legs are built by ../build_ffi_{host,cuda}.sh, whose MPI
# and HDF5 pins (../ffi_mpi.sh) this file does not repeat.
# ============================================================================

LORRAX_MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

lorrax_module_die() { echo "[lorrax-module] REFUSED: $*" >&2; exit 2; }

[[ -n "${LORRAX_MODULE_PREFIX:-}" ]] \
    || lorrax_module_die "set LORRAX_MODULE_PREFIX to the absolute install prefix"
[[ "$LORRAX_MODULE_PREFIX" == /* ]] \
    || lorrax_module_die "LORRAX_MODULE_PREFIX must be absolute: $LORRAX_MODULE_PREFIX"
PREFIX="$(realpath -m "$LORRAX_MODULE_PREFIX")"
# LORRAX_MODULE_VENV: an absolute venv path for a prefix that predates this
# layout (lorrax_cuda13_runtime keeps its runtime venv at $PREFIX/.venv).
[[ "${LORRAX_MODULE_VENV:-/}" == /* ]] \
    || lorrax_module_die "LORRAX_MODULE_VENV must be absolute: $LORRAX_MODULE_VENV"
VENV="$(realpath -m "${LORRAX_MODULE_VENV:-$PREFIX/venv}")"
RELEASES="$PREFIX/releases"
MODULE_NAME="${LORRAX_MODULE_NAME:-lorrax}"
MODULE_VERSION="${LORRAX_MODULE_VERSION:-0.1.0}"

# --- pins --------------------------------------------------------------------
PYTHON_VERSION="3.12"
TOOLKIT_MODULE="cudatoolkit/13.2"
JAX_VERSION="0.9.1"                  # asserted against the lock by setup_env.sh
LOCK_EXTRA="cuda13"                  # the extra a clone syncs (uv sync --extra cuda13)
CUSOLVERMP_VERSION="0.9.1.9318.post1" # also pinned in ../build_ffi_cuda.sh
CUBLASMP_VERSION="0.10.0.3695"        # also pinned in ../build_ffi_cuda.sh
NCCL_VERSION="2.30.7"                # nvidia-nccl-cu13: the uv.lock pin, on every route
                                     # (also ../build_ffi_cuda.sh); never comm_libs NCCL;
                                     # asserted against the lock by setup_env.sh
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
