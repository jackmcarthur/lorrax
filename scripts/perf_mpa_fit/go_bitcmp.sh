#!/bin/bash
# go_bitcmp.sh <tag> -- <A.h5> <B.h5> <qlo> <qhi> [<A.h5> <B.h5> <qlo> <qhi> ...]
# Byte comparisons of fit stores, all of them in ONE container step that
# reserves NO GPU (`-G=0`, charged zero by the pool) because h5py and
# numpy are the whole of it and a GPU would sit idle for minutes.
set -u
source /pscratch/sd/j/jackm/mpa_fitperf_0810/scripts/env.sh
TAG=$1; shift
[ "${1:-}" = "--" ] && shift
LOG=$REPORTS/bitcmp_${TAG}.log
: > "$LOG"
echo "=== bitcmp $TAG ===" | tee -a "$LOG"
date +"=== launch %H:%M:%S ===" | tee -a "$LOG"
while [ $# -ge 4 ]; do
  A=$1; B=$2; QLO=$3; QHI=$4; shift 4
  echo "" | tee -a "$LOG"
  echo "--- $A vs $B over q [$QLO,$QHI) ---" | tee -a "$LOG"
  ~/bin/lx run -G=0 --wait 7200 -- env JAX_PLATFORMS=cpu \
    python -u $WT/scripts/perf_mpa_fit/bitcmp_store.py \
    "$A" "$B" --q-lo "$QLO" --q-hi "$QHI" 2>&1 | tee -a "$LOG"
  echo "=== rc=${PIPESTATUS[0]} ===" | tee -a "$LOG"
done
date +"=== done %H:%M:%S ===" | tee -a "$LOG"
