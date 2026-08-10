#!/bin/bash
# go_cells.sh <tag>
# The branch's cell battery, in ONE leg on a whole node's four GPUs:
#   1. the default fast gate, untouched and unfiltered
#   2. the fit stage's own certification suites, which is where the
#      byte-identity cells live
# Combined because pool capacity is the binding constraint and both arms
# want the same node.
set -u
source /pscratch/sd/j/jackm/mpa_fitperf_0810/scripts/env.sh
TAG=$1
cd $WT
echo "=== cells $TAG ==="
echo "=== source sha: $(git -C $WT rev-parse HEAD) ==="
date +"=== launch %H:%M:%S ==="
echo "--- (1) DEFAULT FAST GATE ---"
~/bin/lx test 2>&1 | tee $REPORTS/cells_fastgate_${TAG}.log
echo "=== fast gate rc=${PIPESTATUS[0]} ==="
echo "--- (2) THE FIT'S OWN SUITES ---"
~/bin/lx test -m "census or not census" -p no:cacheprovider \
  tests/test_mpa_fit_driver.py tests/test_mpa_fit_kernel.py \
  tests/test_mpa_fit_energy_unit.py tests/test_mpa_store.py \
  tests/test_mpa_screening_content.py \
  2>&1 | tee $REPORTS/cells_fit_${TAG}.log
echo "=== fit suites rc=${PIPESTATUS[0]} ==="
date +"=== done %H:%M:%S ==="
