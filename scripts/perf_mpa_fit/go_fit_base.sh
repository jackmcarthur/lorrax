#!/bin/bash
# go_fit_base.sh <qlo> <qhi> <tag> [--max-blocks N]
# THE RED-TWIN ARM: the same leg, on the same harness, against the fit
# path as it stands at 5b8bdbea -- two fits and three Pade solves per
# element.  Everything else is go_fit.sh: same node placement, same
# allocator regime, same deck, same store.
#
# The checkout differs and nothing else, which is why LORRAX_CHECKOUT is
# the only line that changes: $BASE/wt_base is this branch with
# src/gw/mpa/fit_driver.py reverted to 5b8bdbea and fit_conditioning.py
# removed, so the harness is identical and the fit path is the shipped
# one.
set -u
source /pscratch/sd/j/jackm/mpa_fitperf_0810/scripts/env.sh
export LORRAX_CHECKOUT=$BASE/wt_base
QLO=$1; QHI=$2; TAG=$3; shift 3
RUN=$BASE/runs/$TAG; mkdir -p "$RUN"; cd "$RUN"
OUT=$STORES/fit_${TAG}.h5
rm -f "$OUT"
echo "=== BASELINE fit leg $TAG : q [$QLO,$QHI) ==="
echo "=== allocator regime: BFC@0.85 (allocator unset, prealloc=false, mem_fraction=0.85) ==="
echo "=== source tree: $LORRAX_CHECKOUT ==="
echo "=== fit_driver.py sha: $(git -C $BASE/wt_base log -1 --format=%h -- src/gw/mpa/fit_driver.py 2>/dev/null) (reverted to 5b8bdbea) ==="
date +"=== launch %H:%M:%S ==="
~/bin/lx run -G=1 --wait 14400 -- $(bfc_env) WC_STORE=$WC_STORE WC_NAME=$WC_NAME \
  python -u $BASE/wt_base/scripts/perf_mpa_fit/fit_leg.py "$QLO" "$QHI" "$OUT" --tag "$TAG" "$@" \
  2>&1 | tee $REPORTS/fit_${TAG}.log
echo "=== baseline fit leg $TAG rc=${PIPESTATUS[0]} ==="
date +"=== done %H:%M:%S ==="
