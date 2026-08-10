#!/bin/bash
# farm_fit.sh <n_legs> <tag_prefix> [n_q] [--max-blocks N]
# The Pade fit over <n_q> q, split into <n_legs> equal q windows, all
# launched at once as -G=1 legs.  scripts/perf_mpa16/farm_fit.sh with two
# differences: the q count is an argument, because the wedge world fits 8
# q where the full BZ fits 64 and the whole point of this lane's table is
# to state both walls; and the launch stagger is 2 s rather than 3.
#
# Run DETACHED (nohup) from the login node: a leg whose stdout is an ssh
# channel loses its log when the channel closes.
set -u
source /pscratch/sd/j/jackm/mpa_fitperf_0810/scripts/env.sh
N=$1; PFX=$2; shift 2
NQ=64
case "${1:-}" in
  ''|--*) ;;
  *) NQ=$1; shift ;;
esac
date +"=== FARM %H:%M:%S : $N legs over $NQ q ==="
T0=$(date +%s)
for ((i = 0; i < N; i++)); do
  QLO=$((i * NQ / N)); QHI=$(((i + 1) * NQ / N))
  ( bash $BASE/scripts/go_fit.sh $QLO $QHI ${PFX}_$i "$@" \
      > $REPORTS/launch_${PFX}_$i.log 2>&1 ) &
  sleep 2
done
wait
T1=$(date +%s)
echo "=== FARM WALL: $((T1 - T0)) s for $N legs over $NQ q (allocator BFC@0.85) ==="
date +"=== done %H:%M:%S ==="
grep -h "LEG WALL\|THROUGHPUT\|BRING-UP TOTAL\|steady block" $REPORTS/fit_${PFX}_*.log
