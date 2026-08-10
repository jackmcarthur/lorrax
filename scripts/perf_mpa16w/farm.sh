#!/bin/bash
# farm.sh census <prefix> [poles...]        -- the census farm, one leg per pole
# farm.sh window <manifest>                 -- the window-group farm, one leg per manifest row
# farm.sh pole   <prefix> [poles...]        -- the pole farm, the comparison arm
#
# ONE LAUNCHER FOR ALL THREE ARMS, so the stagger, the log naming and the
# wall-clock stamp are the same object in each and the walls are
# comparable.  Run DETACHED (nohup) from the login node: a leg whose
# stdout is an ssh channel loses its log when the channel closes.
#
# THE STAGGER IS 5 s AND IS NOT FREE.  Sixteen legs each polling slurm's
# control plane inside a 48-second window is what lost leg 12 of the
# 2026-08-10 fit farm (`lx_pool: timeout running scontrol show step`).
# The stagger reduces the chance; the MANIFEST is what makes it visible
# when it happens anyway, which is the half that matters.
set -u
source $(dirname "$0")/env.sh
MODE=$1; shift
T0=$(date +%s)

case "$MODE" in
census)
  PFX=$1; shift; POLES="${*:-0 1 2 3 4 5 6 7}"
  date +"=== CENSUS FARM %H:%M:%S : poles [$POLES] ==="
  for P in $POLES; do
    ( bash $BASE/scripts/go_census.sh $P ${PFX}_p$P \
        > $REPORTS/launch_census_${PFX}_p$P.log 2>&1 ) &
    sleep 5
  done
  wait
  ;;
window)
  MAN=$1
  N=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['legs']))" "$MAN")
  date +"=== WINDOW FARM %H:%M:%S : $N legs from $MAN ==="
  for ((i = 0; i < N; i++)); do
    read -r LEG POLES GSUB OUT < <(python3 -c "
import json,sys
L=json.load(open(sys.argv[1]))['legs'][int(sys.argv[2])]
print(L['id'], ','.join(str(p) for p in L['poles']), L['group_subset'], L['output'])
" "$MAN" $i)
    ( bash $BASE/scripts/go_leg.sh "$LEG" "$POLES" "$GSUB" "$OUT" "$MAN" \
        > $REPORTS/launch_${LEG}.log 2>&1 ) &
    sleep 5
  done
  wait
  ;;
pole)
  PFX=$1; shift; POLES="${*:-0 1 2 3 4 5 6 7}"
  date +"=== POLE FARM %H:%M:%S : poles [$POLES] ==="
  for P in $POLES; do
    ( bash $BASE/scripts/go_pole.sh $P ${PFX}_p$P \
        > $REPORTS/launch_pole_${PFX}_p$P.log 2>&1 ) &
    sleep 5
  done
  wait
  ;;
*)
  echo "farm.sh: unknown mode '$MODE'"; exit 2 ;;
esac

T1=$(date +%s)
echo "=== $MODE FARM WALL: $((T1 - T0)) s (allocator BFC@0.85) ==="
date +"=== done %H:%M:%S ==="
