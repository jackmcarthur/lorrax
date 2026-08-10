#!/bin/bash
# farm_pass.sh <tag_prefix> [poles...]
# The n_p=8 Sigma pass, farmed one pole per -G=1 leg, ALL AT ONCE.
#
# EIGHT LEGS IS THE CEILING, and that is the finding this script exists to
# exhibit rather than a choice it makes.  `mpa_pole_subset` is the only
# axis the pass can be split on -- there is no q subset key -- so an n_p=8
# pass has exactly eight independently schedulable pieces and a 16-GPU
# pool runs half idle no matter how the legs are placed.
set -u
source /pscratch/sd/j/jackm/mpa_farm16_0810/scripts/env.sh
PFX=$1; shift
POLES="${*:-0 1 2 3 4 5 6 7}"
date +"=== PASS FARM %H:%M:%S : poles [$POLES] ==="
T0=$(date +%s)
for P in $POLES; do
  ( bash $BASE/scripts/go_pass.sh $P ${PFX}_p$P \
      > $REPORTS/launch_${PFX}_p$P.log 2>&1 ) &
  sleep 5
done
wait
T1=$(date +%s)
echo "=== PASS FARM WALL: $((T1 - T0)) s (allocator BFC@0.85) ==="
date +"=== done %H:%M:%S ==="
for P in $POLES; do
  echo "p$P: $(grep -oE 'exit [0-9]+ in [0-9]+ s' $REPORTS/pass_${PFX}_p$P.log | tail -1) | $(grep -oE 'tau dispatches: [0-9]+' $REPORTS/pass_${PFX}_p$P.log | tail -1)"
done
