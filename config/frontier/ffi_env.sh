# shellcheck shell=bash
# ============================================================================
# ffi_env.sh — run environment for liblorrax_ffi_host.so on OLCF Frontier.
#
#   source config/frontier/ffi_env.sh
#
# Sets:
#   LORRAX_FFI_HOST_SO   the host leg (default: this checkout's
#                        src/ffi/cpp/build_host/liblorrax_ffi_host.so, as
#                        built by config/frontier/build_ffi_host.sh)
#   LD_LIBRARY_PATH      prepended with the pinned HDF5 / cray-mpich / LibSci directories from
#                        config/frontier/ffi_mpi.sh.  The leg's only search path
#                        is $ORIGIN, so these entries decide which machine
#                        libraries it loads; they must come ahead of
#                        /opt/cray/pe/lib64, which maps the site defaults.
#   (module)             darshan-runtime is unloaded when the module command
#                        exists (it is a link-time injector; unloading keeps
#                        any build in this shell free of libdarshan).
#
# It does NOT activate a venv, load ROCm, or set JAX_PLATFORMS: those belong to
# the GPU/run environment.  The FFTW engine is found at run time through the
# dlopen hint compiled into the leg (cray-fftw 3.3.10.11 x86_trento).
# Idempotent: re-sourcing does not duplicate LD_LIBRARY_PATH entries.
# ============================================================================
_lorrax_fr_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_lorrax_fr_root="$(cd "$_lorrax_fr_dir/../.." && pwd)"
# shellcheck disable=SC1091
. "$_lorrax_fr_dir/ffi_mpi.sh"

export LORRAX_FFI_HOST_SO="${LORRAX_FFI_HOST_SO:-$_lorrax_fr_root/src/ffi/cpp/build_host/liblorrax_ffi_host.so}"
if [ ! -f "$LORRAX_FFI_HOST_SO" ]; then
    echo "[ffi_env] WARNING: LORRAX_FFI_HOST_SO=$LORRAX_FFI_HOST_SO is not a file;" >&2
    echo "[ffi_env]   the loader will refuse.  Build: bash config/frontier/build_ffi_host.sh" >&2
fi

_lorrax_fr_prepend="$LORRAX_OLCF_RUNTIME_LIB_DIRS"
_lorrax_fr_rest=":${LD_LIBRARY_PATH:-}:"
for _p in $(echo "$_lorrax_fr_prepend" | tr ':' ' '); do
    _lorrax_fr_rest="${_lorrax_fr_rest//:$_p:/:}"
done
_lorrax_fr_rest="${_lorrax_fr_rest#:}"; _lorrax_fr_rest="${_lorrax_fr_rest%:}"
export LD_LIBRARY_PATH="$_lorrax_fr_prepend${_lorrax_fr_rest:+:$_lorrax_fr_rest}"

if type module >/dev/null 2>&1; then
    module unload darshan-runtime >/dev/null 2>&1 || true
fi
unset _lorrax_fr_dir _lorrax_fr_root _lorrax_fr_prepend _lorrax_fr_rest _p
