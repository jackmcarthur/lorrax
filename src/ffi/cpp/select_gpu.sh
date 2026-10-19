#!/usr/bin/env bash
# Per-rank GPU selector for SLATE's 1-device-per-process model.  Picks the
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

case "${LORRAX_DEBUG_PRINT:-0}" in
    1|on|ON|true|TRUE|yes|YES)
        echo "[select_gpu] $(hostname) local rank ${_select_gpu_localid} ->" \
             "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
             "(from ${_select_gpu_from})" >&2
        ;;
esac

unset -f _select_gpu_nth
unset -v _select_gpu_localid _select_gpu_device _select_gpu_from \
         _select_gpu_var
exec "$@"
