#!/bin/bash
# go_census.sh <pole> <tag>
# ONE census leg: plan every branch of one pole and integrate NOTHING.
#
# The census is what the farm balance is struck from, and it cannot be
# skipped or guessed: a window group's cost is its tau-node count, and
# that number only exists once the group's certified rule has been built.
# A census leg exits 0 after writing its table and says so in a banner.
#
# IT IS ALSO THE PLANNING STEP.  With mpa_plan_store set, the plan this
# leg computes for each of its pole's four branches is written to the
# store, addressed by a digest over every input the planner read.  The
# integrating legs then LOAD their groups instead of re-deriving them --
# which is the ~65 s per pole-touch a sixteen-leg farm otherwise paid
# sixteen times over for eight poles of planning (§9.5).  Nothing about
# the census itself changes: the same call, the same table, the same
# digests.
set -u
source $(dirname "$0")/env.sh
P=$1; TAG=$2
RUN=$BASE/runs/$TAG; mkrun "$RUN"; cd "$RUN"
mkdeck "$TAG" "$P" ""
echo "mpa_pass_census_out = $CENSUS/${TAG}.json" >> deck.in
if [ "${PLAN_ONCE:-1}" = "1" ]; then
  echo "mpa_plan_store = $PLANS" >> deck.in
fi
echo "=== census leg $TAG : pole $P ==="
echo "=== allocator regime: BFC@0.85 ==="
echo "=== source sha: $(git -C $WT rev-parse HEAD) ==="
date +"=== launch %H:%M:%S ==="
rm -f $CENSUS/${TAG}.json
~/bin/lx run -G=1 --wait 7200 -- $(bfc_env) \
  python -u -m gw.gw_jax -i deck.in 2>&1 | tee $REPORTS/census_${TAG}.log
echo "=== census leg $TAG rc=${PIPESTATUS[0]} ==="
date +"=== done %H:%M:%S ==="
