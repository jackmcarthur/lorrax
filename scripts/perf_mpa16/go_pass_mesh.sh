#!/bin/bash
# go_pass_mesh.sh <pole> <tag> [gpus]
# ONE pole pass on a MULTI-GPU MESH -- the L4 measurement.
#
# WHY THIS IS WORTH MEASURING.  The pass sink is already declared
# `P(None, None, 'x', 'y')`, so the band axes shard the moment a leg is
# given a mesh, and ppm_sigma's mesh machinery is inherited whole.  MPA
# legs have only ever run -G=1, so the efficiency of a 2x2 mesh on this
# path is simply unknown -- and it is the one route to putting sixteen
# GPUs under a pass that needs no new deck key at all.
#
# -n = -G, one rank per GPU: the runtime builds its mesh across PROCESSES
# and refuses a mesh whose size is not jax.process_count().
set -u
source /pscratch/sd/j/jackm/mpa_farm16_0810/scripts/env.sh
P=$1; TAG=$2; G=${3:-4}
RUN=$BASE/runs/$TAG; mkrun "$RUN"; cd "$RUN"
sed -e "s|MPA_FIT_FILE|$FIT_NP8|" \
    -e "s|MPA_HEAD_LABEL|as_shipped|" \
    -e "s|MPA_POLE_SUBSET|$P|" \
    -e "s|MPA_PASS_PARTIAL_OUT|$PARTIALS/${TAG}.h5|" \
    -e "s|MPA_PASS_PARTIAL_IN||" \
    -e "s|SIGMA_DIAG_FILE|PARTIAL_NOT_A_SELF_ENERGY_${TAG}.dat|" \
    $PROD/scripts/deck_np8.in > deck.in
echo "=== mesh pass leg $TAG : pole $P on $G GPUs ==="
echo "=== allocator regime: BFC@0.85 (allocator unset, prealloc=false, mem_fraction=0.85) ==="
echo "=== source sha: $(git -C $WT rev-parse HEAD) ==="
date +"=== launch %H:%M:%S ==="
rm -f $PARTIALS/${TAG}.h5
~/bin/lx run -G=$G -n=$G --wait 14400 -- $(bfc_env) \
  python -u -m gw.gw_jax -i deck.in \
  2>&1 | tee $REPORTS/mesh_${TAG}.log
echo "=== mesh pass leg $TAG rc=${PIPESTATUS[0]} ==="
date +"=== done %H:%M:%S ==="
