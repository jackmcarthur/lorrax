#!/bin/bash
# go_pole.sh <pole> <tag>
# ONE POLE-FARM LEG -- the comparison arm, re-measured at THIS sha.
#
# This is the 2026-08-10 baseline leg unchanged (one pole, -G=1, BFC@0.85)
# and it is re-run rather than quoted because the baseline was taken at
# 66bc3fb1 and this branch is off 5b8bdbea.  It serves two purposes: the
# 8-leg wall to compare the 16-leg wall against, and the per-pole cube the
# window-farmed cubes are checked against.  A pole leg's cube IS the
# single-process computation of that pole -- the same groups, in the same
# order, folded by one accumulator -- so it is the single-leg arm of the
# bit-identity gate as well as the pole-farm arm.
set -u
source $(dirname "$0")/env.sh
P=$1; TAG=$2
RUN=$BASE/runs/$TAG; mkrun "$RUN"; cd "$RUN"
mkdeck "$TAG" "$P" "$POLEPARTS/${TAG}.h5"
echo "=== pole leg $TAG : pole $P ==="
echo "=== allocator regime: BFC@0.85 ==="
echo "=== source sha: $(git -C $WT rev-parse HEAD) ==="
date +"=== launch %H:%M:%S ==="
rm -f $POLEPARTS/${TAG}.h5
~/bin/lx run -G=1 --wait 14400 -- $(bfc_env) \
  python -u -m gw.gw_jax -i deck.in 2>&1 | tee $REPORTS/pole_${TAG}.log
echo "=== pole leg $TAG rc=${PIPESTATUS[0]} ==="
date +"=== done %H:%M:%S ==="
