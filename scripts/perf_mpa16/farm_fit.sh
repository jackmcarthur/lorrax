#!/bin/bash
# farm_fit.sh <n_legs> <tag_prefix> [--max-blocks N]
# The full-BZ (64 q) Pade fit, split into <n_legs> equal q windows, all
# launched at once as -G=1 legs.  With n_legs=16 this is the run that
# actually fills a 4-node pool's 16 GPUs -- the fit is the ONLY stage of
# the MPA path that can, because the pass splits over poles alone and
# n_p=8 caps it at eight.
#
# Run DETACHED (nohup) from the login node: a leg whose stdout is an ssh
# channel loses its log when the channel closes.
set -u
source /pscratch/sd/j/jackm/mpa_farm16_0810/scripts/env.sh
N=$1; PFX=$2; shift 2
NQ=64
date +"=== FARM %H:%M:%S : $N legs over $NQ q ==="
T0=$(date +%s)
for ((i = 0; i < N; i++)); do
  QLO=$((i * NQ / N)); QHI=$(((i + 1) * NQ / N))
  ( bash $BASE/scripts/go_fit.sh $QLO $QHI ${PFX}_$i "$@" \
      > $REPORTS/launch_${PFX}_$i.log 2>&1 ) &
  sleep 3
done
wait
T1=$(date +%s)
echo "=== FARM WALL: $((T1 - T0)) s for $N legs (allocator BFC@0.85) ==="
date +"=== done %H:%M:%S ==="
grep -h "LEG WALL\|THROUGHPUT\|BRING-UP TOTAL" $REPORTS/fit_${PFX}_*.log
