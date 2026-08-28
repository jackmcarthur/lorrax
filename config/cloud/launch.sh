#!/usr/bin/env bash
# Launch an N-process LORRAX GPU run on a single cloud box (no SLURM).
#
#   config/cloud/launch.sh -n 4 python -u config/perlmutter/cuda13_module/verify_runtime.py
#   config/cloud/launch.sh -n 4 python -u tests/bench/cusolvermp_eigh_test.py -n128 --grid 2 2
#   config/cloud/launch.sh -n 4 python -u -m gw.gw_jax -i cohsex.in
#
# mpirun provides process placement AND the MPI world the phdf5 handlers
# need; the per-rank preamble maps OpenMPI's rank vars onto the JAX_* names
# src/runtime/__init__.py resolves (JAX_PROCESS_COUNT / JAX_PROCESS_INDEX /
# JAX_COORDINATOR_ADDRESS), and pins one GPU per rank the same way
# Perlmutter's srun does (CUDA_VISIBLE_DEVICES=<local rank>).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/stack.sh"

NP=4
if [[ "${1:-}" == "-n" ]]; then NP="$2"; shift 2; fi
[[ $# -ge 1 ]] || { echo "usage: launch.sh [-n N] <cmd...>" >&2; exit 2; }

SO="${LORRAX_FFI_SO:-$LORRAX_CLOUD_BUILD/liblorrax_ffi.so}"
[[ -e "$SO" ]] || { echo "[cloud launch] REFUSED: no FFI .so at $SO (run build_ffi.sh)" >&2; exit 2; }
HOST_SO="${LORRAX_FFI_HOST_SO:-$LORRAX_CLOUD_ROOT/src/ffi/cpp/build_host_cloud/liblorrax_ffi_host.so}"
[[ -e "$HOST_SO" ]] || { echo "[cloud launch] REFUSED: no host .so at $HOST_SO (run build_ffi_host.sh)" >&2; exit 2; }

export LORRAX_FFI_SO="$SO"
export LORRAX_FFI_HOST_SO="$HOST_SO"
export LD_LIBRARY_PATH="$(lorrax_cloud_runtime_ld_library_path):${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(lorrax_cloud_pythonpath)${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$LORRAX_CLOUD_ENV/bin:$PATH"
# One coordinator per launch, unique port, so two launches on one box never
# join each other's incarnation (see init_jax_distributed's docstring).
export JAX_COORDINATOR_ADDRESS="${JAX_COORDINATOR_ADDRESS:-127.0.0.1:$((12000 + RANDOM % 20000))}"
export JAX_PROCESS_COUNT="$NP"
# Don't let XLA pre-grab the pool cuSOLVERMp/NCCL must share (config/README.md).
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

exec mpirun -n "$NP" --bind-to none \
    bash -c 'export JAX_PROCESS_INDEX=$OMPI_COMM_WORLD_RANK
             export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
             exec "$@"' _ "$@"
