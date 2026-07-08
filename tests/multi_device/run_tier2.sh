#!/usr/bin/env bash
# Tier-2 cross-P invariance gate driver (Perlmutter / lorrax module + lxrun).
#
# LORRAX runs one process per device, so the P=4 leg must be launched as 4
# tasks — a single python process cannot host it.  This driver prepares the
# run dirs, launches each leg with lxrun (LORRAX_NGPU picks the per-node GPU
# / task count), and runs the compare step.  Needs: a GPU allocation
# (lxalloc/lxattach), `module load lorrax_X lorrax_agent`.
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

if ! type lxrun >/dev/null 2>&1; then
    # lxrun is a shell function (not exported to child shells) — try to
    # self-load the modules on Perlmutter; honor LORRAX_MODULE for the
    # checkout letter (default: lorrax_D).
    if [ -r /etc/profile.d/zzz-lmod.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/zzz-lmod.sh
        module use /global/homes/j/jackm/modulefiles 2>/dev/null || true
        module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles 2>/dev/null || true
        module load "${LORRAX_MODULE:-lorrax_D}" lorrax_agent 2>/dev/null || true
    fi
fi
if ! type lxrun >/dev/null 2>&1; then
    echo "run_tier2.sh: lxrun not found — module load lorrax_X lorrax_agent first" >&2
    exit 2
fi

rc=0
for case in "${CASES[@]}"; do
    fix="$REPO/tests/regression/${FIXTURE[$case]}"
    for leg in p1 p4; do
        dir="$WORK/${case}_${leg}"
        rm -rf "$dir"; mkdir -p "$WORK"
        cp -r "$fix" "$dir"; rm -rf "$dir/tmp"
        echo "== $case $leg =="
        ( cd "$dir" && \
          LORRAX_NGPU=$([ "$leg" = p1 ] && echo 1 || echo 4) \
          lxrun python3 -u -m gw.gw_jax -i "${INPUT[$case]}" > run.log 2>&1 ) \
          || { echo "$case $leg FAILED — see $dir/run.log"; rc=1; continue 2; }
    done
    LORRAX_NGPU=1 lxrun python3 "$REPO/tests/multi_device/eqp_invariance_cross_p.py" \
        compare "$case" "$WORK/${case}_p1" "$WORK/${case}_p4" || rc=1
done
exit $rc
