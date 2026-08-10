#!/bin/bash
# go_probe.sh <tag> [-- extra args to probe_fit_jit.py]
# The fit-stage probe on ONE GPU under BFC@0.85, placed with -G=1.
set -u
source /pscratch/sd/j/jackm/mpa_fitperf_0810/scripts/env.sh
TAG=$1; shift
RUN=$BASE/runs/$TAG; mkdir -p "$RUN"; cd "$RUN"
echo "=== probe $TAG ==="
echo "=== allocator regime: BFC@0.85 (allocator unset, prealloc=false, mem_fraction=0.85) ==="
echo "=== source sha: $(git -C $WT rev-parse HEAD) ==="
date +"=== launch %H:%M:%S ==="
~/bin/lx run -G=1 --wait 7200 -- $(bfc_env) WC_STORE=$WC_STORE WC_NAME=$WC_NAME \
  python -u $WT/scripts/perf_mpa_fit/probe_fit_jit.py "$@" \
  2>&1 | tee $REPORTS/probe_${TAG}.log
echo "=== probe $TAG rc=${PIPESTATUS[0]} ==="
date +"=== done %H:%M:%S ==="
