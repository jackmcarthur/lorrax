#!/usr/bin/env bash
# Source from the run scripts, after entering a GPU allocation.
set -euo pipefail
: "${LORRAX:=${LORRAX_ROOT:-}}"
: "${LORRAX:?Set LORRAX to your built LORRAX clone}"
export LORRAX_CHECKOUT="$LORRAX" LORRAX_ROOT="$LORRAX"
export PYTHONPATH="$LORRAX/src:$LORRAX"
for lorrax_service in "$LORRAX"/services/*/src; do
    export PYTHONPATH="$PYTHONPATH:$lorrax_service"
done
if [[ ${JAX_PLATFORMS:-} == cpu ]]; then
    source "$LORRAX/config/perlmutter/cpu_mpi_env.sh"
    lorrax_run() {
        srun -N 1 -n 4 -c 16 --threads-per-core=1 --cpu-bind=threads \
            "$LORRAX/.venv/bin/python" -u -m "$@"
    }
else
    source "$LORRAX/config/perlmutter/gpu_env.sh"
    lorrax_run() {
        srun -N 1 -n 4 -c 16 --gpus-per-node=4 --threads-per-core=1 \
            --cpu-bind=threads "$LORRAX/src/ffi/cpp/select_gpu.sh" \
            "$LORRAX/.venv/bin/python" -u -m "$@"
    }
fi
