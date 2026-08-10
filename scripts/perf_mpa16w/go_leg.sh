#!/bin/bash
# go_leg.sh <leg_id> <poles> <group_subset> <out.h5> <manifest>
# ONE WINDOW-GROUP FARM LEG on one GPU under BFC@0.85.
#
# The only difference from a pole leg is the mpa_group_subset key, which
# names the runs of window groups this leg owns.  The planner runs
# unchanged and its output is sliced, so the groups, their membership,
# their certified rules and their tau nodes are the ones the unsplit walk
# would have used.  The manifest is passed so the leg can check the
# planner's partition against the one the balance was struck from.
set -u
source $(dirname "$0")/env.sh
LEG=$1; POLES=$2; GSUB=$3; OUT=$4; MAN=$5
RUN=$BASE/runs/$LEG; mkrun "$RUN"; cd "$RUN"
mkdeck "$LEG" "$POLES" "$OUT"
{
  echo "mpa_group_subset = $GSUB"
  echo "mpa_farm_manifest = $MAN"
} >> deck.in
# PLAN_ONCE=1 (the default) loads this leg's window groups from the plan
# store the census farm wrote; PLAN_ONCE=0 is the pre-2026-08-10 route,
# kept as the A/B arm because the whole claim of this lane is a
# difference between the two.  PLAN_VERIFY=1 additionally plans every
# owned branch the old way and refuses unless the loaded plan is
# bit-identical -- it costs exactly the time the store saves, so it is a
# gate leg and not a production one.
if [ "${PLAN_ONCE:-1}" = "1" ]; then
  echo "mpa_plan_store = $PLANS" >> deck.in
  if [ "${PLAN_VERIFY:-0}" = "1" ]; then
    echo "mpa_plan_verify = true" >> deck.in
  fi
fi
echo "=== window-farm leg $LEG : poles [$POLES] ==="
echo "=== groups: $GSUB ==="
echo "=== allocator regime: BFC@0.85 ==="
echo "=== source sha: $(git -C $WT rev-parse HEAD) ==="
date +"=== launch %H:%M:%S ==="
rm -f "$OUT"
~/bin/lx run -G=1 --wait 14400 -- $(bfc_env) \
  python -u -m gw.gw_jax -i deck.in 2>&1 | tee $REPORTS/leg_${LEG}.log
echo "=== window-farm leg $LEG rc=${PIPESTATUS[0]} ==="
date +"=== done %H:%M:%S ==="
