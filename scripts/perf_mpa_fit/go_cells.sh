#!/bin/bash
# go_cells.sh <tag> [--base]
# The branch's cell battery on a whole node's four GPUs.
#
#   default    the default fast gate, then the fit stage's own suites
#              (--census, because every fit cell is census-marked and the
#              default tier is the Si driver plus the services)
#   --base     the SAME fast gate against $BASE/wt_base, which is this
#              branch with fit_driver.py reverted to 5b8bdbea and
#              fit_conditioning.py removed.  Zero delta between the two
#              runs is the claim; a count quoted against a remembered
#              number would not be.
set -u
source /pscratch/sd/j/jackm/mpa_fitperf_0810/scripts/env.sh
TAG=$1
MODE=${2:-new}
if [ "$MODE" = "--base" ]; then
  export LORRAX_CHECKOUT=$BASE/wt_base
  cd $BASE/wt_base
  SUFFIX=base
else
  cd $WT
  SUFFIX=new
fi
echo "=== cells $TAG ($SUFFIX) ==="
echo "=== source tree: $LORRAX_CHECKOUT ==="
echo "=== sha: $(git -C $LORRAX_CHECKOUT rev-parse --short HEAD) ==="
date +"=== launch %H:%M:%S ==="
echo "--- (1) DEFAULT FAST GATE ---"
~/bin/lx test --wait 5400 2>&1 | tee $REPORTS/cells_fastgate_${TAG}_${SUFFIX}.log
echo "=== fast gate rc=${PIPESTATUS[0]} ==="
if [ "$MODE" != "--base" ]; then
  echo "--- (2) THE FIT'S OWN SUITES, CENSUS TIER ---"
  ~/bin/lx test --wait 5400 --census tests/test_mpa_fit_driver.py \
    tests/test_mpa_fit_kernel.py tests/test_mpa_fit_energy_unit.py \
    tests/test_mpa_store.py tests/test_mpa_screening_content.py \
    2>&1 | tee $REPORTS/cells_fit_${TAG}.log
  echo "=== fit suites rc=${PIPESTATUS[0]} ==="
fi
date +"=== done %H:%M:%S ==="
