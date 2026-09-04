#!/usr/bin/env bash
# Tier-2 cross-P invariance gate driver (Perlmutter / lx).
#
# LORRAX runs one process per device, so the P=4 leg must be launched as 4
# tasks — a single python process cannot host it.  This driver prepares the
# run dirs, launches each leg with one task per GPU, and runs the comparison.
# `lx` allocates or attaches; set both Slurm job-id spellings only when
# attaching to a verified custom allocation.
#
#   cd <repo> && bash tests/multi_device/run_tier2.sh [gnppm|bispinor ...]
#
# Elsewhere (non-Perlmutter): run each fixture copy with your own launcher
# (1 task × 1 GPU, then 4 tasks × 4 GPUs) and call
#   python3 tests/multi_device/eqp_invariance_cross_p.py compare <case> <p1> <p4>
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${LORRAX_TIER2_WORKDIR:-$REPO/.tier2_cross_p}"
CASES=("${@:-gnppm bispinor}")
[ $# -eq 0 ] && CASES=(gnppm bispinor)

declare -A FIXTURE=( [gnppm]="gnppm_debug"     [bispinor]="bispinor_debug" )
declare -A INPUT=(   [gnppm]="gnppm_test.in"   [bispinor]="bispinor_test.in" )

if ! command -v lx >/dev/null 2>&1; then
    echo "run_tier2.sh: lx not found; use the supported Perlmutter launcher" >&2
    exit 2
fi
if [ -n "${LORRAX_CHECKOUT:-}" ] && \
   [ "$(readlink -f "$LORRAX_CHECKOUT")" != "$REPO" ]; then
    echo "run_tier2.sh: LORRAX_CHECKOUT does not name this source tree" >&2
    exit 2
fi
export LX_BASE_MODULE="${LX_BASE_MODULE:-lorrax_A}"
export LORRAX_CHECKOUT="$REPO"
SOURCE_PATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

rc=0
for case in "${CASES[@]}"; do
    fix="$REPO/tests/regression/${FIXTURE[$case]}"
    for leg in p1 p4; do
        dir="$WORK/${case}_${leg}"
        rm -rf "$dir"; mkdir -p "$WORK"
        cp -r "$fix" "$dir"; rm -rf "$dir/tmp"
        echo "== $case $leg =="
        if [ "$leg" = p1 ]; then ngpu=1; else ngpu=4; fi
        ( cd "$dir" && \
          lx run -N 1 -G "$ngpu" -n "$ngpu" -- \
          env LORRAX_RUN_DIR="$dir" PYTHONPATH="$SOURCE_PATH" \
          python3 -u -m gw.gw_jax -i "${INPUT[$case]}" > run.log 2>&1 ) \
          || { echo "$case $leg FAILED — see $dir/run.log"; rc=1; continue 2; }
    done
    lx run -N 1 -G 0 -n 1 -- \
      env JAX_PLATFORMS=cpu PYTHONPATH="$SOURCE_PATH" \
      python3 "$REPO/tests/multi_device/eqp_invariance_cross_p.py" \
        compare "$case" "$WORK/${case}_p1" "$WORK/${case}_p4" || rc=1
done
exit $rc
