#!/bin/bash
# go_fit.sh <qlo> <qhi> <tag> [--max-blocks N]
# ONE fit farm leg on ONE GPU under BFC@0.85, placed with -G=1.
# scripts/perf_mpa16/go_fit.sh with this lane's env.sh; the leg itself is
# perf_mpa16's fit_leg.py, copied verbatim, so the walls compare.
set -u
source /pscratch/sd/j/jackm/mpa_fitperf_0810/scripts/env.sh
QLO=$1; QHI=$2; TAG=$3; shift 3
RUN=$BASE/runs/$TAG; mkdir -p "$RUN"; cd "$RUN"
OUT=$STORES/fit_${TAG}.h5
rm -f "$OUT"
echo "=== fit leg $TAG : q [$QLO,$QHI) ==="
echo "=== allocator regime: BFC@0.85 (allocator unset, prealloc=false, mem_fraction=0.85) ==="
echo "=== source sha: $(git -C $WT rev-parse HEAD) ==="
date +"=== launch %H:%M:%S ==="
~/bin/lx run -G=1 --wait 14400 -- $(bfc_env) WC_STORE=$WC_STORE WC_NAME=$WC_NAME \
  python -u $WT/scripts/perf_mpa_fit/fit_leg.py "$QLO" "$QHI" "$OUT" --tag "$TAG" "$@" \
  2>&1 | tee $REPORTS/fit_${TAG}.log
echo "=== fit leg $TAG rc=${PIPESTATUS[0]} ==="
date +"=== done %H:%M:%S ==="
