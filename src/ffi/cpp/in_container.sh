#!/usr/bin/env bash
# Runs INSIDE the Shifter container.  Re-asserts env vars that
# shifter's --module=mpich explicitly unsets (notably
# MPICH_GPU_SUPPORT_ENABLED, see /etc/shifter/udiRoot.conf line
# `module_mpich_siteEnvUnset = MPICH_GPU_SUPPORT_ENABLED`), then exec's
# its arguments.
#
# Used as the final wrapper between the shifter invocation and the
# user command in run_shifter.sh.
#
# MPICH_GPU_SUPPORT_ENABLED IS PER-PLATFORM, NOT A CONSTANT.
# ----------------------------------------------------------
# GPU leg: it must be 1.  Cray MPICH only takes the GPU-Direct RDMA path
# (MPI_Send straight from a device pointer over Slingshot, no host bounce)
# when this is set, and it pairs with the libmpi_gtl_cuda LD_PRELOAD
# run_shifter.sh arranges.  Without it SLATE / cuSOLVERMp / phdf5 D2H
# staging all still WORK — just slowly — so nothing announces the loss.
#
# CPU leg: it must be 0.  With it set, Cray MPICH tries to dlopen the GTL
# (libmpi_gtl_cuda.so.0) at MPI_Init_thread and ABORTS the process when the
# CUDA runtime it wants is absent, which is exactly the case for a
# JAX_PLATFORMS=cpu run against the CUDA-free host FFI.  Until 2026-08-06
# this script exported 1 unconditionally, which made the CPU FFI leg
# unusable in-container without a manual
# `... in_container.sh env MPICH_GPU_SUPPORT_ENABLED=0 python3 ...`
# override — a hardcoded environment fact silently substituted for the
# real one (docs/architecture/slab_io.md, Failure modes).
#
# Resolution order, most explicit first:
#   1. LORRAX_MPICH_GPU_SUPPORT=0|1  — say it outright, wins over everything.
#      (This spelling, not MPICH_GPU_SUPPORT_ENABLED itself: shifter's
#      --module=mpich UNSETS that name on the way in, so it cannot carry a
#      caller's intent across the container boundary.  LORRAX_* survives.)
#   2. LORRAX_PLATFORM=cpu|gpu       — the run's platform, set by
#      run_shifter.sh (which derives it from JAX_PLATFORMS when unset).
#   3. JAX_PLATFORMS                 — if it was forwarded into the
#      container, its FIRST entry is the platform JAX will take.
#   4. gpu — the historical default, and what every GPU launch wants.
set -euo pipefail

_lrx_gpu_support() {
    case "${LORRAX_MPICH_GPU_SUPPORT:-}" in
        0|1) printf %s "${LORRAX_MPICH_GPU_SUPPORT}"; return ;;
        "")  ;;
        *)   echo "in_container.sh: LORRAX_MPICH_GPU_SUPPORT=" \
                  "'${LORRAX_MPICH_GPU_SUPPORT}' is not 0 or 1." >&2
             exit 2 ;;
    esac
    local plat="${LORRAX_PLATFORM:-}"
    if [[ -z "${plat}" ]]; then
        # First entry of JAX_PLATFORMS ("cpu", "cuda,cpu", ...).
        plat="${JAX_PLATFORMS:-}"
        plat="${plat%%,*}"
    fi
    case "${plat}" in
        cpu|host) printf 0 ;;
        *)        printf 1 ;;
    esac
}

export MPICH_GPU_SUPPORT_ENABLED="$(_lrx_gpu_support)"
exec "$@"
