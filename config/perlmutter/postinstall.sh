#!/usr/bin/env bash
# ============================================================================
# postinstall.sh: the install's one compute-node step, installed by
# install_deps.sh as PREFIX/bin/lorrax-postinstall (which sources
# PREFIX/lorrax-env.sh first; install_deps.sh copies this file to
# PREFIX/share/lorrax/).  Run it once per install, and again after a reinstall,
# inside a one-node GPU allocation:
#
#   salloc -N 1 -C gpu -G 4 -q interactive -t 0:30:00 -A <project>_g lorrax-postinstall
#
# Both steps run the way every job runs: through PREFIX/bin/lorrax-run (the
# runtime image restored node-local, then the image's venv), or, for an
# --in-place install, from the tree.
# 1. The NVRTC kernel store: every image of scripts/build_cubin_store.py's grid
#    table into the per-user image cache (lxkit.user_cache_root()/kconv_mathdx/
#    images, the directory every run and the self-check read), with
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
SELF="$(command -v lorrax-postinstall 2>/dev/null || echo "$0")"
say() { printf '[lorrax-postinstall] %s\n' "$*"; }

if [[ "${LORRAX_RUNTIME:-}" == image ]]; then
    RUN=("${LORRAX_PREFIX:?LORRAX_PREFIX is not set: run lorrax-postinstall}/bin/lorrax-run")
    STORE_RUN=("${RUN[@]}" --all-gpus)
    META="$(readlink -f "$LORRAX_PREFIX/current").meta"
    TREE="$(sed -n 's/^tree=//p' "$META" 2>/dev/null | head -n 1)"
    [[ -n "$TREE" ]] || { say "no runtime image at $LORRAX_PREFIX/current: rerun install_deps.sh"; exit 2; }
    SRC="$TREE/lorrax"      # the canonical path: valid on the compute node once lorrax-run restored it
    PY=python               # the image's venv python (lorrax-run puts it first on PATH)
else
    SRC="${LORRAX_ROOT:?LORRAX_ROOT is not set: run lorrax-postinstall, or source PREFIX/lorrax-env.sh first}"
    PY="$SRC/.venv/bin/python"
    RUN=("$SRC/src/ffi/cpp/select_gpu.sh")
    STORE_RUN=()
fi

job_info=""
[[ -n "${SLURM_JOB_ID:-}" ]] && job_info="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null)"
if [[ -z "$job_info" ]] || ! grep -q 'gres/gpu' <<< "$job_info"; then
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

# CPUs of one allocated node, from the allocation itself: SLURM_CPUS_ON_NODE
# describes the node a step runs on, and nproc here is the login node's (a
# login node's 256 made `srun -c 244`, which never placed).
nodes="$(sed -n 's/.*NumNodes=\([0-9]*\).*/\1/p' <<< "$job_info" | head -n 1)"
ncpu="$(sed -n 's/.*NumCPUs=\([0-9]*\).*/\1/p' <<< "$job_info" | head -n 1)"
if [[ -n "$nodes" && -n "$ncpu" && "$nodes" -gt 0 ]]; then
    cpus=$((ncpu / nodes))
elif [[ -n "${SLURM_STEP_ID:-}" && -n "${SLURM_CPUS_ON_NODE:-}" ]]; then
    cpus="$SLURM_CPUS_ON_NODE"
else
    say "cannot tell how many CPUs job $SLURM_JOB_ID has per node (scontrol show job gave no NumCPUs/NumNodes); not starting a step that might never place"
    exit 2
fi
jobs=$((cpus / 8)); ((jobs < 1)) && jobs=1; ((jobs > 16)) && jobs=16

say "NVRTC kernel store: $jobs workers on job $SLURM_JOB_ID ($cpus CPUs per node)"
t0=$SECONDS
srun -N 1 -n 1 -c "$cpus" --gpus-per-node=4 "${STORE_RUN[@]}" "$PY" -u "$SRC/scripts/build_cubin_store.py" --jobs "$jobs"
store_rc=$?
say "kernel store rc=$store_rc in $((SECONDS - t0)) s"

say "dependency check: 64x64 distributed cuSOLVERMp Cholesky (4 ranks)"
srun -N 1 -n 4 --gpus-per-node=4 "${RUN[@]}" "$PY" -u "$SRC/scripts/dep_check.py" cholesky
dep_rc=$?

say "performance self-check (4 ranks, the production launch line)"
t0=$SECONDS
srun -N 1 -n 4 --gpus-per-node=4 "${RUN[@]}" "$PY" -u "$SRC/scripts/perf_check.py"
check_rc=$?
say "self-check rc=$check_rc in $((SECONDS - t0)) s"

if ((dep_rc != 0)); then
    say "INSTALL NOT VERIFIED: the distributed Cholesky failed (its [dep-check] line above names the fix)"
    exit 1
fi
if ((check_rc == 0 && store_rc == 0)); then
    say "INSTALL VERIFIED"
    exit 0
fi
say "INSTALL NOT VERIFIED (kernel store rc=$store_rc, self-check rc=$check_rc); see the messages above"
((check_rc == 1)) && exit 1
exit 2
