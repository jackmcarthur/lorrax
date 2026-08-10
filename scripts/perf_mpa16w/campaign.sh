#!/bin/bash
# campaign.sh -- the plan-once measurement, end to end and unattended.
#
# Runs detached from a login node.  Every stage is idempotent on its
# outputs, so a stage that loses a leg to co-tenancy is finished by the
# retry loop rather than by re-running the whole farm: the manifest is
# what makes "which cubes are missing" a question with an answer, and
# this script is that answer applied.
#
#   balance -> arm A (plan-once, 16 legs) -> arm B (re-plan, 16 legs)
#           -> the verify leg -> both merges and the gates
#
# THE TWO ARMS ARE THE MEASUREMENT.  Arm A is the lever; arm B is the same
# sixteen legs on the pre-plan-store route at the same anchor, and it is
# both the before-wall and the cubes arm A's bit-identity is judged
# against.  Neither is quoted from §9: the anchor's windowed_exp_iEt flip
# moved Σ bytes, so a pre-anchor cube is not admissible as either arm.
set -u
source $(dirname "$0")/env.sh
CAP=${CAP:-16}

launch_missing() {                    # launch_missing <manifest> <plan_once>
  local MAN=$1 PO=$2 ROUND MISSING N
  # A MISSING MANIFEST IS NOT AN EMPTY FARM.  The first run of this script
  # printed "all legs landed" twice against manifests the balance step had
  # refused to write, and reported two zero-second walls -- which is the
  # farm's own failure mode (an absence reading as a result) reproduced in
  # the launcher.  So the launcher refuses it too.
  if [ ! -s "$MAN" ]; then
    echo "  REFUSED: no manifest at $MAN -- the balance step did not write "
    echo "  one, so there are no legs to launch and NOTHING here landed."
    return 3
  fi
  for ROUND in 1 2 3; do
    MISSING=0
    N=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['legs']))" "$MAN")
    for ((i = 0; i < N; i++)); do
      read -r LEG POLES GSUB OUT < <(python3 -c "
import json,sys
L=json.load(open(sys.argv[1]))['legs'][int(sys.argv[2])]
print(L['id'], ','.join(str(p) for p in L['poles']), L['group_subset'], L['output'])
" "$MAN" $i)
      if [ ! -s "$OUT" ]; then
        MISSING=$((MISSING + 1))
        while [ $(jobs -rp | wc -l) -ge $CAP ]; do sleep 15; done
        ( PLAN_ONCE=$PO bash $BASE/scripts/go_leg.sh \
            "$LEG" "$POLES" "$GSUB" "$OUT" "$MAN" \
            > $REPORTS/launch_${LEG}_po${PO}.log 2>&1 ) &
        sleep 5
      fi
    done
    wait
    [ $MISSING -eq 0 ] && { echo "  all legs landed (round $ROUND)"; return 0; }
    date +"  round $ROUND left legs missing; retrying at %H:%M:%S"
  done
  echo "  STILL MISSING after 3 rounds -- the merge will refuse and name them"
}

date +"=== CAMPAIGN START %H:%M:%S ==="
echo "=== source sha: $(git -C $WT rev-parse HEAD) ==="
echo "=== census shas: $(python3 -c "
import glob, json
print(sorted({json.load(open(p))['sha'][:12] for p in glob.glob('$CENSUS/*.json')}))") ==="
if ! python3 $BASE/scripts/balance.py $CENSUS 16 $PARTIALS \
        $BASE/manifest16.json 2>&1 | tail -25; then
  echo "=== BALANCE REFUSED: no farm is launched.  The census and the tree "
  echo "=== that would run its legs have to be one tree; re-take the census."
  exit 3
fi
mkdir -p $BASE/partials_replan
python3 $BASE/scripts/balance.py $CENSUS 16 $BASE/partials_replan \
  $BASE/manifest16_replan.json > $REPORTS/BALANCE_REPLAN.log 2>&1

T0=$(date +%s)
date +"=== ARM A: plan-once, 16 legs, %H:%M:%S ==="
launch_missing $BASE/manifest16.json 1
T1=$(date +%s)
echo "=== ARM A WALL: $((T1 - T0)) s (allocator BFC@0.85) ==="

date +"=== ARM B: re-plan per leg, 16 legs, %H:%M:%S ==="
launch_missing $BASE/manifest16_replan.json 0
T2=$(date +%s)
echo "=== ARM B WALL: $((T2 - T1)) s (allocator BFC@0.85) ==="

date +"=== VERIFY LEG %H:%M:%S ==="
bash $BASE/scripts/planonce.sh verify leg00 > $REPORTS/VERIFY.log 2>&1
grep -E "PLAN-VERIFY|MPA-FIXED-TERM" $REPORTS/leg_leg00_verify.log | tail -8

date +"=== GATES %H:%M:%S ==="
bash $BASE/scripts/planonce.sh gate_replan > /dev/null 2>&1
REF_NPY=$BASE/gate_out_replan/sigma_c_window.npy \
REF_DIR=$BASE/partials_replan \
  bash $BASE/scripts/planonce.sh gate > /dev/null 2>&1
tail -40 $REPORTS/GATES.log
date +"=== CAMPAIGN DONE %H:%M:%S ==="
