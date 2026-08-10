#!/bin/bash
# planonce.sh <stage>  -- the 2026-08-10 plan-once campaign, one stage at a time.
#
# The window farm's own harness with one thing added: the census farm
# writes the window plans and the legs load them.  Every stage is a
# separate invocation because each one is a farm whose result the next one
# reads, and a script that ran them back to back would be a script that
# could not be resumed after a leg that lost its GPU -- which is the
# failure mode this deck has already produced twice (§9.7).
#
#   census   -- 8 legs, one per pole: the tau table AND the plan store
#   balance  -- fold the censuses, strike the 16-leg balance, DECLARE the
#               plans as manifest inputs
#   farm     -- 16 legs, loading their groups from the store
#   verify   -- ONE leg that plans both ways and refuses unless the loaded
#               plan is bit-identical: the gate, and the in-leg A/B of
#               plan seconds against load seconds
#   replan   -- ONE leg on the pre-plan-store route, for the fixed-term
#               "before" at THIS sha rather than a quoted one
#   gate     -- merge, red twins, and the comparison against a reference
#
# Allocator BFC@0.85 throughout (env.sh), -G=1 legs, four to a node.
set -u
source $(dirname "$0")/env.sh
STAGE=$1; shift
REF_NPY=${REF_NPY:-/pscratch/sd/j/jackm/mpa_winfarm_0810/gate_out/sigma_c_window.npy}
REF_DIR=${REF_DIR:-/pscratch/sd/j/jackm/mpa_winfarm_0810/partials}
MAN=${MAN:-$BASE/manifest16.json}

case "$STAGE" in
census)
  POLES="${*:-0 1 2 3 4 5 6 7}"
  PLAN_ONCE=1 bash $BASE/scripts/farm.sh census cen $POLES
  ;;
balance)
  python3 $BASE/scripts/balance.py $CENSUS 16 $PARTIALS $MAN
  # The second arm's manifest is struck from the SAME census, so its
  # partition is the same partition -- only the cubes' directory differs.
  # Two manifests rather than one because a leg's output path comes out of
  # the manifest, and the two arms must not write into each other.
  mkdir -p $BASE/partials_replan
  python3 $BASE/scripts/balance.py $CENSUS 16 $BASE/partials_replan \
    $BASE/manifest16_replan.json
  ;;
farm)
  PLAN_ONCE=1 bash $BASE/scripts/farm.sh window $MAN
  ;;
farm_replan)
  # The A/B arm: the same sixteen legs on the pre-plan-store route, at
  # THIS sha and this anchor.  It is what the plan-once arm's cubes are
  # compared against and where the "before" fixed term and wall come
  # from -- both re-cut here rather than quoted from §9, because the
  # anchor's windowed_exp_iEt flip moved Σ bytes and a pre-anchor cube is
  # not admissible as either arm.
  PLAN_ONCE=0 bash $BASE/scripts/farm.sh window $BASE/manifest16_replan.json
  ;;
verify)
  LEG=${1:-leg00}
  read -r POLES GSUB OUT < <(python3 -c "
import json,sys
for L in json.load(open(sys.argv[1]))['legs']:
    if L['id'] == sys.argv[2]:
        print(','.join(str(p) for p in L['poles']), L['group_subset'],
              L['output'].replace('.h5', '_verify.h5'))
        break
" "$MAN" "$LEG")
  PLAN_ONCE=1 PLAN_VERIFY=1 bash $BASE/scripts/go_leg.sh \
    "${LEG}_verify" "$POLES" "$GSUB" "$OUT" "$MAN"
  ;;
replan)
  LEG=${1:-leg00}
  read -r POLES GSUB OUT < <(python3 -c "
import json,sys
for L in json.load(open(sys.argv[1]))['legs']:
    if L['id'] == sys.argv[2]:
        print(','.join(str(p) for p in L['poles']), L['group_subset'],
              L['output'].replace('.h5', '_replan.h5'))
        break
" "$MAN" "$LEG")
  PLAN_ONCE=0 bash $BASE/scripts/go_leg.sh \
    "${LEG}_replan" "$POLES" "$GSUB" "$OUT" "$MAN"
  ;;
gate)
  # Host-side: a merge, three red twins and two comparisons, none of which
  # touches a device.  -G=0 is a container step that reserves no GPU, so
  # the gate does not take a card off the farm to do arithmetic on cubes.
  ~/bin/lx run -G=0 --wait 3600 -- env JAX_PLATFORMS=cpu \
    python -u $BASE/scripts/merge_and_gate.py \
      "$MAN" "$POLEPARTS" "$BASE/gate_out" "$REF_NPY" "$REF_DIR" \
    2>&1 | tee $REPORTS/GATES.log
  ;;
gate_replan)
  # Arm B merged on its own, to produce the reference Σ_c the plan-once
  # arm is then compared against.
  ~/bin/lx run -G=0 --wait 3600 -- env JAX_PLATFORMS=cpu \
    python -u $BASE/scripts/merge_and_gate.py \
      "$BASE/manifest16_replan.json" "$POLEPARTS" "$BASE/gate_out_replan" \
    2>&1 | tee $REPORTS/GATES_REPLAN.log
  ;;
*)
  echo "planonce.sh: unknown stage '$STAGE'"; exit 2 ;;
esac
