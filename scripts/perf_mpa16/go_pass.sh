#!/bin/bash
# go_pass.sh <pole> <tag>
# ONE pole pass of the n_p=8 W_c field on ONE GPU under BFC@0.85.
#
# The difference from the 2026-08-09 production leg: -G=1 one-GPU
# PLACEMENT instead of a whole-node `-G=4 -n=1` reservation, so four legs
# share a node.  Everything the deck computes is unchanged.
set -u
source /pscratch/sd/j/jackm/mpa_farm16_0810/scripts/env.sh
P=$1; TAG=$2
RUN=$BASE/runs/$TAG; mkrun "$RUN"; cd "$RUN"
sed -e "s|MPA_FIT_FILE|$FIT_NP8|" \
    -e "s|MPA_HEAD_LABEL|as_shipped|" \
    -e "s|MPA_POLE_SUBSET|$P|" \
    -e "s|MPA_PASS_PARTIAL_OUT|$PARTIALS/${TAG}.h5|" \
    -e "s|MPA_PASS_PARTIAL_IN||" \
    -e "s|SIGMA_DIAG_FILE|PARTIAL_NOT_A_SELF_ENERGY_${TAG}.dat|" \
    $PROD/scripts/deck_np8.in > deck.in
echo "=== pass leg $TAG : pole $P ==="
echo "=== allocator regime: BFC@0.85 (allocator unset, prealloc=false, mem_fraction=0.85) ==="
echo "=== source sha: $(git -C $WT rev-parse HEAD) ==="
date +"=== launch %H:%M:%S ==="
rm -f $PARTIALS/${TAG}.h5
~/bin/lx run -G=1 --wait 14400 -- $(bfc_env) \
  python -u -m gw.gw_jax -i deck.in \
  2>&1 | tee $REPORTS/pass_${TAG}.log
echo "=== pass leg $TAG rc=${PIPESTATUS[0]} ==="
date +"=== done %H:%M:%S ==="
