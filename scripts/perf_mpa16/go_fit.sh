#!/bin/bash
# go_fit.sh <qlo> <qhi> <tag> [--max-blocks N]
# ONE fit farm leg on ONE GPU under BFC@0.85, placed with -G=1.
set -u
source /pscratch/sd/j/jackm/mpa_farm16_0810/scripts/env.sh
QLO=$1; QHI=$2; TAG=$3; shift 3
RUN=$BASE/runs/$TAG; mkdir -p "$RUN"; cd "$RUN"
OUT=$STORES/fit_${TAG}.h5
rm -f "$OUT"
echo "=== fit leg $TAG : q [$QLO,$QHI) ==="
echo "=== allocator regime: BFC@0.85 (allocator unset, prealloc=false, mem_fraction=0.85) ==="
echo "=== source sha: $(git -C $WT rev-parse HEAD) ==="
date +"=== launch %H:%M:%S ==="
~/bin/lx run -G=1 --wait 14400 -- $(bfc_env) WC_STORE=$WC_STORE WC_NAME=$WC_NAME \
  python -u $WT/scripts/perf_mpa16/fit_leg.py "$QLO" "$QHI" "$OUT" --tag "$TAG" "$@" \
  2>&1 | tee $REPORTS/fit_${TAG}.log
echo "=== fit leg $TAG rc=${PIPESTATUS[0]} ==="
date +"=== done %H:%M:%S ==="
