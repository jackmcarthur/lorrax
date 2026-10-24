#!/usr/bin/env bash
# Per-rank GPU selector for the 1-device-per-process model.  Picks the
# ONE device this rank should use, exports it as CUDA_VISIBLE_DEVICES, then
# execs its arguments in place, keeping PMI environment intact for mpich
# collective bootstrap.
#
# Used by run_shifter.sh and by `lx run` between srun and the user command.
# Unlike an inline `bash -c 'export ...; exec "$@"'`, writing this as a
# script file means srun invokes it as a single process image (so PMI_* env
# vars, which mpich's libmpi reads at MPI_Init_thread, aren't stripped by an
# intermediate subshell).
#
# WHICH DEVICE, and why it is no longer the local rank id
# -------------------------------------------------------
# This used to be, in full, `CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}` —
# NERSC's "Method 3" (https://docs.nersc.gov/jobs/affinity/), the rank's
# index on the node used directly as a device ordinal.  That is the right
# answer only when the devices this step may touch happen to be numbered
# 0..n-1 in local-rank order.  It is a coincidence of the common case, not
# a rule; the rule is spelled out by Slurm itself, in the environment it
# hands the step.  So take the LOCALID-th entry of the list Slurm assigned,
# and fall back to the bare local rank only when Slurm said nothing at all.
#
# On a 4-rank, 4-GPU step Slurm assigns 0,1,2,3 and this returns the local
# rank for every rank — byte-for-byte the old behaviour, which is the point:
# the multi-rank path is what production runs on and it does not move.  What
# changes is the case the old line got wrong, where the assignment is not
# 0-based (a step given the node's GPUs 2,3): the old line pinned ranks 0
# and 1 to devices 0 and 1, which are devices the step was not given.
#
# The list is read from CUDA_VISIBLE_DEVICES FIRST, on purpose.  Under
# cgroup device containment Slurm exports the ordinals as the CUDA runtime
# inside this step will number them, while SLURM_STEP_GPUS / SLURM_JOB_GPUS
# are in the node's physical numbering.  Indexing the physical list would
# name a device this step cannot open the moment the two numberings differ,
# so those are consulted only when CUDA_VISIBLE_DEVICES is unset.  (Measured
# on Perlmutter 2026-08-09: a step given one GPU can open exactly one
# /dev/nvidia*, and Slurm hands it CUDA_VISIBLE_DEVICES=0 regardless of
# which physical device that is.)
#
# LORRAX_GPU_DEVICE overrides all of it: a comma-separated list of device
# ordinals indexed by local rank.  This is the hook a launcher needs in
# order to PLACE a step, and it exists because Slurm will not do it.  Two
# concurrent one-rank steps in one allocation are both assigned the same
# physical GPU — measured every way the step can be asked for, `--overlap`
# and not, `--gres=gpu:1`, `--gpus=1` and `--gpus-per-task=1`, 2026-08-09 —
# so a fleet that runs several one-GPU legs per node stacks them all on one
# device while the rest of the node idles.  Nothing readable from inside the
# step distinguishes those legs; only the scheduler that placed them knows,
# and this is how it says so.
#
# Production placement is silent: the driver's processor block reports the
# device count once.  LORRAX_DEBUG_PRINT=1 exposes the per-rank binding line,
# using the same and only debug switch as the driver it launches.  Refusals
# remain unconditional because a short explicit placement would put two ranks
# on one device.
set -u

_select_gpu_nth() {
    # $1 = comma-separated list, $2 = index.  Echoes the index-th entry and
    # returns 0; returns 1 when the list is empty or has no such entry.
    local list=$1 idx=$2
    [[ -n "$list" ]] || return 1
    local IFS=,
    # shellcheck disable=SC2206  # word splitting on IFS=, is the intent
    local -a entries=($list)
    [[ ${#entries[@]} -gt "$idx" ]] || return 1
    [[ -n "${entries[$idx]}" ]] || return 1
    printf '%s' "${entries[$idx]}"
}

_select_gpu_localid=${SLURM_LOCALID:-0}
_select_gpu_device=""
_select_gpu_from=""

if [[ -n "${LORRAX_GPU_DEVICE:-}" ]]; then
    if _select_gpu_device=$(_select_gpu_nth "${LORRAX_GPU_DEVICE}" \
                                            "${_select_gpu_localid}"); then
        _select_gpu_from="LORRAX_GPU_DEVICE"
    else
        # An explicit assignment that does not cover this rank is a launcher
        # bug, and the two ways to paper over it -- reuse the last entry, or
        # fall through to the derivation -- both end in two ranks quietly
        # sharing a device, which is the failure this whole file exists to
        # stop.  Refuse instead.
        echo "[select_gpu] LORRAX_GPU_DEVICE='${LORRAX_GPU_DEVICE}' has no" \
             "entry for local rank ${_select_gpu_localid}; it must list one" \
             "device per rank on this node." >&2
        exit 2
    fi
fi

if [[ -z "${_select_gpu_device}" ]]; then
    for _select_gpu_var in CUDA_VISIBLE_DEVICES SLURM_STEP_GPUS \
                           SLURM_JOB_GPUS; do
        if _select_gpu_device=$(_select_gpu_nth \
                "${!_select_gpu_var:-}" "${_select_gpu_localid}"); then
            _select_gpu_from="${_select_gpu_var}"
            break
        fi
    done
fi

if [[ -z "${_select_gpu_device}" ]]; then
    # Slurm named no devices at all (a launch outside a GPU step, or with
    # GRES unreported).  The historical behaviour is the best guess left.
    _select_gpu_device="${_select_gpu_localid}"
    _select_gpu_from="SLURM_LOCALID (no GPU assignment in the environment)"
fi

export CUDA_VISIBLE_DEVICES="${_select_gpu_device}"

# CPU THREADS, derived from the CPUs Slurm gave this rank
# --------------------------------------------------------
# The plain step `srun -N 1 -n 4 --gpus-per-node=4` gives each Perlmutter rank
# a quarter of the node WITH both hardware threads of each core (32 CPUs on
# 16 cores), and leaves OMP_NUM_THREADS unset, so every OpenMP team (LibSci,
# the host FFT) starts 32 threads per rank and XLA's CPU pool takes all 32.
# The reference clone runs used `-c 16 --threads-per-core=1
# --cpu-bind=threads` with OMP_NUM_THREADS=14, OPENBLAS_NUM_THREADS=1 and
# MKL_NUM_THREADS=1 (docs/environment/machines/perlmutter.md, Threads per
# rank).  Here the rank reaches that shape from whatever Slurm assigned it:
# its affinity mask is cut to one hardware thread per physical core, and
# OMP_NUM_THREADS is that core count less two (the MPICH progress thread and
# Python's I/O threads), at least 1; the BLAS teams default to 1.  Where the
# core topology cannot be read (no sysfs thread_siblings_list, no taskset)
# the mask is left alone and the count is taken from the logical CPUs.  A
# caller that exported OMP_NUM_THREADS owns the threading, and nothing here
# changes.  LORRAX_RANK_THREADS records which case ran; the runtime prints it
# in the startup CPU row.
_select_gpu_threads() {
    local list cpu c sib s lo hi lower keep="" n=0 total=0 topo=1 how
    list=$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status 2>/dev/null)
    if [[ -z "$list" ]]; then
        export LORRAX_RANK_THREADS="select_gpu: no affinity mask readable; threads unchanged"
        return 0
    fi
    local -A in_mask=()
    local -a cpus=()
    local IFS=,
    for s in $list; do
        lo=${s%-*}; hi=${s#*-}
        for ((cpu = lo; cpu <= hi; cpu++)); do in_mask[$cpu]=1; cpus+=("$cpu"); done
    done
    for cpu in "${cpus[@]}"; do
        total=$((total + 1))
        if [[ -r "/sys/devices/system/cpu/cpu$cpu/topology/thread_siblings_list" ]]; then
            sib=$(< "/sys/devices/system/cpu/cpu$cpu/topology/thread_siblings_list")
        else
            sib=$cpu; topo=0
        fi
        lower=0
        for s in $sib; do
            lo=${s%-*}; hi=${s#*-}
            for ((c = lo; c <= hi && c < cpu; c++)); do [[ -n "${in_mask[$c]:-}" ]] && lower=1; done
        done
        if ((lower == 0)); then keep+="${keep:+,}$cpu"; n=$((n + 1)); fi
    done
    unset IFS
    if ((topo == 0)); then
        n=$total; how="mask of $total CPUs left alone (no sysfs core topology)"
    elif ((n == total)); then
        how="$total CPUs, one per core already"
    elif ! command -v taskset >/dev/null 2>&1; then
        n=$total; how="mask of $total CPUs left alone (no taskset)"
    elif taskset -pc "$keep" $$ >/dev/null 2>&1; then
        how="mask cut from $total CPUs to $n, one per physical core"
    else
        n=$total; how="mask of $total CPUs left alone (taskset refused)"
    fi
    export OMP_NUM_THREADS=$((n > 3 ? n - 2 : 1))
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export LORRAX_RANK_THREADS="select_gpu: $how; OMP $OMP_NUM_THREADS, OpenBLAS $OPENBLAS_NUM_THREADS, MKL $MKL_NUM_THREADS"
}
if [[ -n "${OMP_NUM_THREADS:-}" ]]; then
    export LORRAX_RANK_THREADS="select_gpu: OMP_NUM_THREADS=${OMP_NUM_THREADS} exported by the caller; mask and threads unchanged"
else
    _select_gpu_threads
fi

case "${LORRAX_DEBUG_PRINT:-0}" in
    1|on|ON|true|TRUE|yes|YES)
        echo "[select_gpu] $(hostname) local rank ${_select_gpu_localid} ->" \
             "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
             "(from ${_select_gpu_from}); CPUs $(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status);" \
             "${LORRAX_RANK_THREADS:-}" >&2
        ;;
esac

unset -f _select_gpu_nth _select_gpu_threads
unset -v _select_gpu_localid _select_gpu_device _select_gpu_from \
         _select_gpu_var
exec "$@"
