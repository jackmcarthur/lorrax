# shellcheck shell=bash
# ============================================================================
# rocm_env.sh — launch environment of a ROCm (MI250X) LORRAX step on Frontier.
#
#   source config/frontier/rocm_env.sh      (after the venv is active)
#
# Sets:
#   LD_LIBRARY_PATH  prepended with ~/.cache/lorrax/rocr-hmem/<id>, one
#                    directory per wheel (<id>: the md5 of its path, so two
#                    venvs never share a link), holding libhsa-runtime64.so
#                    -> the ROCm wheel's
#                    libhsa-runtime64.so.1.  libfabric's ROCr memory
#                    interface dlopens that unversioned name; the wheel ships
#                    only the .so.1, so without the link RCCL's OFI plugin
#                    cannot register device memory and a multi-node step dies
#                    in its first inter-node collective (site.toml [rccl]
#                    launch_libraries; runtime.network_env refuses the step).
#                    The link names the same file the PJRT plugin maps, so
#                    the process keeps one HSA runtime.
#
# It sets no NCCL_*/FI_* variable: runtime.network_env applies the site's
# [rccl.env] (OLCF rccl-net-plugin/1.0) to multi-node steps, a caller's
# export winning.  Idempotent.
# ============================================================================
_lorrax_hsa="$(python3 - <<'PY' 2>/dev/null
import glob, os, sys
import _rocm_sdk_core
hits = sorted(glob.glob(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "lib", "libhsa-runtime64.so.1")))
print(hits[0] if hits else "")
PY
)"
if [ -z "$_lorrax_hsa" ]; then
    echo "[rocm_env] WARNING: the active python has no ROCm wheel (_rocm_sdk_core); LD_LIBRARY_PATH unchanged." >&2
else
    _lorrax_shim="${XDG_CACHE_HOME:-$HOME/.cache}/lorrax/rocr-hmem/$(printf %s "$_lorrax_hsa" | md5sum | cut -c1-12)"
    mkdir -p "$_lorrax_shim"
    ln -sfn "$_lorrax_hsa" "$_lorrax_shim/libhsa-runtime64.so"
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$_lorrax_shim:"*) ;;
        *) export LD_LIBRARY_PATH="$_lorrax_shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi
unset _lorrax_hsa _lorrax_shim
