#!/usr/bin/env bash
# Nsight Systems profiles for the dominant encode/decode/Sigma GEMM shapes.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/evidence/root" >&2
  exit 2
fi
evidence=$1
case "$evidence" in
  /*) ;;
  *) echo "evidence path must be absolute: $evidence" >&2; exit 2 ;;
esac

repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
mkdir -p "$evidence"/{build,profiles,logs}
export BENCH_ALLOC=bfc
unset XLA_PYTHON_CLIENT_ALLOCATOR
probe="$evidence/build/libvertex_gemm_tc_probe.so"
tests/bench/build_vertex_gemm_tc_ffi.sh "$probe" >/dev/null

echo "HOST=$(hostname) JID=${SLURM_JOB_ID:-unset} STEP=${SLURM_STEP_ID:-unset}"
nsys --version
for size in fixture production; do
  for op in bse_encode bse_decode_mu sigma_right; do
    for arm in baseline batched ffi; do
      stem="$evidence/profiles/${size}_${op}_${arm}"
      echo "PROFILE size=$size op=$op arm=$arm out=$stem"
      nsys profile \
        --output="$stem" \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --sample=none \
        --cpuctxsw=none \
        python3 -u tests/bench/bench_vertex_gemm_tc.py \
          --size "$size" --ops "$op" --arms "$arm" \
          --ffi-so "$probe" --warmup 10 --profile --profile-reps 20
    done
  done
done
