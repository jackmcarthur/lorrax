#!/usr/bin/env bash
# Source from the run scripts, after activating the environment used to build
# LORRAX and entering an allocation. Site selection belongs to runtime.
set -euo pipefail
: "${LORRAX:=${LORRAX_ROOT:-}}"
: "${LORRAX:?Set LORRAX to your built LORRAX clone}"
export LORRAX_CHECKOUT="$LORRAX" LORRAX_ROOT="$LORRAX"
export PYTHONPATH="$LORRAX/src"
# An activated external ROCm environment must win over a checkout's .venv.
# The latter remains the Perlmutter clone-install fallback.
if [[ -n ${VIRTUAL_ENV:-} ]]; then
    lorrax_python_bin="$VIRTUAL_ENV/bin/python"
elif [[ -x "$LORRAX/.venv/bin/python" ]]; then
    lorrax_python_bin="$LORRAX/.venv/bin/python"
else
    lorrax_python_bin="$(command -v python3)"
fi
[[ -x "$lorrax_python_bin" ]] || { printf 'Activate the Python environment used to build LORRAX.\n' >&2; return 1; }
export PATH="$(dirname "$lorrax_python_bin"):$PATH"
lorrax_site="$("$lorrax_python_bin" -c 'from runtime.network_env import site_profile; p = site_profile(); print(p["site"] if p else "")')"
case "$lorrax_site" in
    perlmutter)
        lorrax_launch=(srun -N 1 -n 4 -c 16 --threads-per-core=1 --cpu-bind=threads)
        if [[ ${JAX_PLATFORMS:-} == cpu ]]; then
            source "$LORRAX/config/perlmutter/cpu_mpi_env.sh"
        else
            source "$LORRAX/config/perlmutter/gpu_env.sh"
            lorrax_launch+=(--gpus-per-node=4 "$LORRAX/src/ffi/cpp/select_gpu.sh")
        fi
        ;;
    frontier)
        source "$LORRAX/config/frontier/gpu_env.sh"
        lorrax_python_bin="$(command -v python)"
        lorrax_launch=(srun -N 1 -n 4 -c 7 --threads-per-core=1 --cpu-bind=cores)
        if [[ ${JAX_PLATFORMS:-} != cpu ]]; then
            # Slurm assigns one MI250X GCD per rank; the CUDA selector is not
            # involved. The tutorial uses four GCDs (half a Frontier node).
            lorrax_launch+=(--gpus-per-task=1 --gpu-bind=closest)
        fi
        ;;
    *)
        printf 'This tutorial launcher supports Perlmutter and Frontier. Select the installed site profile as described in the machine guide.\n' >&2
        return 1
        ;;
esac
lorrax_run() { "${lorrax_launch[@]}" "$lorrax_python_bin" -u -m "$@"; }
lorrax_python() { "$lorrax_python_bin" "$@"; }
