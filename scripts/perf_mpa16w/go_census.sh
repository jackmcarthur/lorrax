#!/bin/bash
# go_census.sh <pole> <tag>
# ONE census leg: plan every branch of one pole and integrate NOTHING.
#
# The census is what the farm balance is struck from, and it cannot be
# skipped or guessed: a window group's cost is its tau-node count, and
# that number only exists once the group's certified rule has been built.
# A census leg exits 0 after writing its table and says so in a banner.
set -u
source $(dirname "$0")/env.sh
P=$1; TAG=$2
RUN=$BASE/runs/$TAG; mkrun "$RUN"; cd "$RUN"
mkdeck "$TAG" "$P" ""
echo "mpa_pass_census_out = $CENSUS/${TAG}.json" >> deck.in
echo "=== census leg $TAG : pole $P ==="
echo "=== allocator regime: BFC@0.85 ==="
echo "=== source sha: $(git -C $WT rev-parse HEAD) ==="
date +"=== launch %H:%M:%S ==="
rm -f $CENSUS/${TAG}.json
~/bin/lx run -G=1 --wait 7200 -- $(bfc_env) \
  python -u -m gw.gw_jax -i deck.in 2>&1 | tee $REPORTS/census_${TAG}.log
echo "=== census leg $TAG rc=${PIPESTATUS[0]} ==="
date +"=== done %H:%M:%S ==="
