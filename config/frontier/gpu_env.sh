# shellcheck shell=bash
# Source once before srun. Use the build's explicit environment, an already
# active environment, or the same default as the Frontier build scripts.
# This discovers the ROCm plugin without importing JAX or opening a device.
_lorrax_frontier_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_lorrax_frontier_venv="${LORRAX_OLCF_VENV:-${VIRTUAL_ENV:-$HOME/envs/lx-jax091-rocm7141}}"
if [[ ! -x "$_lorrax_frontier_venv/bin/python" || ! -f "$_lorrax_frontier_venv/bin/activate" ]]; then
    printf '[frontier gpu_env] No Python environment at %s. Set LORRAX_OLCF_VENV to the environment used to build the native libraries.\n' "$_lorrax_frontier_venv" >&2
    unset _lorrax_frontier_root _lorrax_frontier_venv
    return 1
fi
export LORRAX_OLCF_VENV="$_lorrax_frontier_venv"
source "$_lorrax_frontier_venv/bin/activate"
export LORRAX_CHECKOUT="$_lorrax_frontier_root" LORRAX_ROOT="$_lorrax_frontier_root"
# runtime.source_closure owns the service source roots; a clone needs src only.
export PYTHONPATH="$_lorrax_frontier_root/src"
if ! "$VIRTUAL_ENV/bin/python" - <<'PY'
import importlib.util
import pkgutil
import sys

spec = importlib.util.find_spec("jax_plugins")
if spec is None or not any(m.name.startswith("xla_rocm") for m in
                           pkgutil.iter_modules(spec.submodule_search_locations or ())):
    sys.exit("[frontier gpu_env] The selected Python environment has no JAX ROCm plugin. "
             "Use the ROCm environment used for the Frontier build.")
PY
then
    unset _lorrax_frontier_root _lorrax_frontier_venv
    return 1
fi
source "$_lorrax_frontier_root/config/frontier/ffi_env.sh"
source "$_lorrax_frontier_root/config/frontier/rocm_env.sh"
unset _lorrax_frontier_root _lorrax_frontier_venv
