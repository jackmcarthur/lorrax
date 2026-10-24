#!/usr/bin/env bash
# ============================================================================
# postinstall.sh: the install's one compute-node step, installed by
# install_deps.sh as PREFIX/bin/lorrax-postinstall (which sources
# PREFIX/lorrax-env.sh first).  Run it once per install, and again after
# rebuilding the FFI, inside a one-node GPU allocation:
#
#   salloc -N 1 -C gpu -G 4 -q interactive -t 0:30:00 -A <project>_g lorrax-postinstall
#
# 1. The NVRTC kernel store: every image of scripts/build_cubin_store.py's grid
#    table into the per-user image cache (kconv_mathdx/images), with
#    min(16, CPUs/8) workers on the node's GPUs.  Skipped when a store for this
#    GPU architecture, toolchain, FFI build and grid list is already complete.
#    389 images, 3.1 GB, 247 s with 16 workers on a 4xA100 node (2026-10-23).
#    Without it, each image a run meets first compiles for about 7 s on every
#    rank that misses it (module gnppm 259 s against 59 s with the images).
# 2. The performance self-check, scripts/perf_check.py, with the production
#    launch line.  A pass writes the stamp runtime.install_verification_stamp
#    names; until then every GPU run's startup block warns INSTALL NOT VERIFIED.
#
# Exit 0: verified.  1: the self-check failed.  2: the store or the check could
# not run.  3: not inside a GPU allocation (the line above is printed).
# ============================================================================
set -uo pipefail
ROOT="${LORRAX_ROOT:?LORRAX_ROOT is not set: run lorrax-postinstall, or source PREFIX/lorrax-env.sh first}"
PY="$ROOT/.venv/bin/python"
SELF="$(command -v lorrax-postinstall 2>/dev/null || echo "$0")"
say() { printf '[lorrax-postinstall] %s\n' "$*"; }

gpu_allocation() {
    [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -q 'gres/gpu'
}
if ! gpu_allocation; then
    acct="$(sacctmgr -nP show assoc user="${USER:-$(id -un)}" format=account 2>/dev/null | grep '_g$' | sort -u | head -1)"
    cat <<EOF

######################################################################################
  INSTALL NOT VERIFIED.  One compute-node step remains: it builds the NVRTC kernel
  store (about 4 min) and checks the install's speed.  From a login node run:

    salloc -N 1 -C gpu -G 4 -q interactive -t 0:30:00 -A ${acct:-<project>_g} $SELF

######################################################################################
EOF
    exit 3
fi

cpus="${SLURM_CPUS_ON_NODE:-$(nproc)}"
jobs=$((cpus / 8)); ((jobs < 1)) && jobs=1; ((jobs > 16)) && jobs=16
say "NVRTC kernel store: $jobs workers on job $SLURM_JOB_ID"
t0=$SECONDS
srun -N 1 -n 1 -c "$cpus" --gpus-per-node=4 "$PY" -u "$ROOT/scripts/build_cubin_store.py" --jobs "$jobs"
store_rc=$?
say "kernel store rc=$store_rc in $((SECONDS - t0)) s"

if [[ -f "$ROOT/scripts/perf_check.py" ]]; then
    say "performance self-check (4 ranks, the production launch line)"
    srun -N 1 -n 4 --gpus-per-node=4 "$ROOT/src/ffi/cpp/select_gpu.sh" "$PY" -u "$ROOT/scripts/perf_check.py"
    check_rc=$?
else
    say "this tree has no scripts/perf_check.py: the install cannot be verified"
    check_rc=2
fi

if ((check_rc == 0 && store_rc == 0)); then
    say "INSTALL VERIFIED"
    exit 0
fi
say "INSTALL NOT VERIFIED (kernel store rc=$store_rc, self-check rc=$check_rc); see the messages above"
((check_rc == 1)) && exit 1
exit 2
