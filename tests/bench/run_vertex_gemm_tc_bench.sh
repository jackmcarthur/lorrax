#!/usr/bin/env bash
# One compute-node leg: capability audit, standalone FFI build, and sweep.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 /absolute/evidence/root [size [comma-separated-ops]]" >&2
  exit 2
fi
evidence=$1
size=${2:-both}
ops=${3:-bse_encode,bse_decode_mu,bse_decode_nu,sigma_right,sigma_left}
case "$evidence" in
  /*) ;;
  *) echo "evidence path must be absolute: $evidence" >&2; exit 2 ;;
esac

repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"

mkdir -p "$evidence"/{build,hlo,logs,profiles}
export BENCH_ALLOC=bfc
unset XLA_PYTHON_CLIENT_ALLOCATOR

echo "HOST=$(hostname) JID=${SLURM_JOB_ID:-unset} STEP=${SLURM_STEP_ID:-unset}"
python3 -c 'import jax; print("JAX", jax.__version__, jax.devices())'
echo "CUBLAS_API"
grep -En 'Zgemm3m|ZgemmStridedBatched' \
  /usr/local/cuda/include/cublas_v2.h \
  /usr/local/cuda/targets/x86_64-linux/include/cublas_v2.h 2>/dev/null \
  | head -40 || true
echo "CUTENSOR_CAPABILITY"
ldconfig -p 2>/dev/null | grep -Ei cutensor || true
find /usr/local /opt -maxdepth 5 -iname '*cutensor*' -print 2>/dev/null \
  | head -40 || true
python3 -c 'import importlib.util; print({n: bool(importlib.util.find_spec(n)) for n in ("cutensor", "cuquantum", "cupy")})'
nsys --version

probe="$evidence/build/libvertex_gemm_tc_probe.so"
tests/bench/build_vertex_gemm_tc_ffi.sh "$probe"
python3 -u tests/bench/bench_vertex_gemm_tc.py \
  --size "$size" \
  --ops "$ops" \
  --ffi-so "$probe" \
  --warmup 10 \
  --samples 15 \
  --hlo-dir "$evidence/hlo" \
  --json "$evidence/results_${size}.json"
