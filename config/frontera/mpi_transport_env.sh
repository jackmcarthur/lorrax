#!/usr/bin/env bash
# ============================================================================
# mpi_transport_env.sh — Intel-MPI transport hygiene for Frontera.  Source it
# (inside or outside the container) before any run that touches MPI: impl=mpi
# CPU collectives, phdf5 MPI-IO, mpi4py, the FFI host .so, MPI microbenches.
#
# Split out of ffi_env.sh (2026-07-31, env audit P0.3) and applied
# UNCONDITIONALLY: this block used to hide behind LORRAX_FFI_PHDF5=1, so a
# plain CPU user sourcing ffi_env.sh got NO transport hygiene — no PMI2 glue,
# no provider case-block, no UCX setdefaults — and inherited whatever the
# login shell leaked (e.g. TACC's FI_PROVIDER=mlx, or the PMI-1
# I_MPI_PMI_LIBRARY that cannot bootstrap --mpi=pmi2).
#
# Everything here is either a setdefault (inherited values win) or a
# deterministic unset/set documented in docs/dev/env_vars.md §5.  Resolved
# transport choices are ANNOUNCED on rank 0 (announce-or-refuse doctrine);
# the I_MPI_DEBUG>=4 "libfabric provider:" banner remains the only
# trustworthy provider observable (fi_info false-negatives on mlx).
# ============================================================================

: "${LORRAX_IMPI_ROOT:=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64}"
export I_MPI_ROOT="${LORRAX_IMPI_ROOT%/intel64}"   # wrappers/runtime append intel64/

_lorrax_mpi_say () { [ "${SLURM_PROCID:-0}" = "0" ] && echo "[mpi_transport_env] $*"; return 0; }

# --- PMI glue (srun --mpi=pmi2 bootstrap) ----------------------------------
# MUST override unconditionally: TACC's login environment sets
# I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so, a host PMI-1 lib that is the wrong
# protocol for --mpi=pmi2 AND absent inside the container -> MPIR_pmi_init
# fails.  Force our staged PMI2 lib.  Nothing dlopens it unless MPI actually
# initializes under srun, so setting it in non-MPI shells is harmless.
export I_MPI_PMI_LIBRARY="${LORRAX_PMI2_LIB:-$WORK/host_pmi/libpmi2.so.0}"
if [ ! -f "$I_MPI_PMI_LIBRARY" ]; then
    _lorrax_mpi_say "WARNING: staged PMI2 lib not found: $I_MPI_PMI_LIBRARY"
    _lorrax_mpi_say "  srun --mpi=pmi2 + Intel MPI will fail at MPIR_pmi_init."
    _lorrax_mpi_say "  Remedy: run config/frontera/stage_host_pmi.sh once (login node)"
    _lorrax_mpi_say "  or point LORRAX_PMI2_LIB at an existing copy."
fi

# --- fabrics ---------------------------------------------------------------
# Default shm:ofi — Intel MPI 2019+'s own multi-node default, stated here as
# documentation of intent plus a guard against stray inherited values.  The
# old ffi_env.sh default was bare `shm`, an rtx single-node BRING-UP value
# (skips OFI/libfabric init entirely) that silently pinned every sourcing
# run to single-node transport; it stays reachable as the explicit rtx hatch:
#     LORRAX_MPI_FABRICS=shm
export I_MPI_FABRICS="${LORRAX_MPI_FABRICS:-shm:ofi}"
_lorrax_mpi_say "I_MPI_FABRICS=$I_MPI_FABRICS (LORRAX_MPI_FABRICS=${LORRAX_MPI_FABRICS:-<unset, default shm:ofi>})"

# Provider .so resolution — REQUIRED in-container (AU measured: with it
# unset, PMPI_Init aborts "addrinfo() failed ... No data available":
# libfabric finds no providers.  mpivars.sh, which normally sets it, is
# not sourced in-container).
export FI_PROVIDER_PATH="$LORRAX_IMPI_ROOT/libfabric/lib/prov"

# ---- libfabric provider (scorecard AP root cause, AS.5 spec) --------------
# The old `FI_PROVIDER="${FI_PROVIDER:-tcp}"` seed was an rtx/mlx4 bring-up
# workaround that leaked to CLX production: on Frontera CLX HDR leave
# FI_PROVIDER UNSET and Intel MPI auto-selects the native `mlx` (UCX)
# provider — 1.07 us / 11.4 GB/s vs tcp's 10.9 us / 2.15 GB/s; pzheevd
# n=2448 P=144 12 s/q -> 0.5-0.9 s/q.  NOTE the seed also never did what it
# claimed: TACC's default impi module exports FI_PROVIDER=mlx in every
# login/compute shell, so the `:-tcp` fallback usually saw an inherited
# value — the case-block below unsets/sets it deterministically.
#   LORRAX_MPI_PROVIDER=auto  (default) unset -> IMPI picks mlx on CLX
#   LORRAX_MPI_PROVIDER=tcp   the rtx/mlx4 escape hatch (IPoIB via ib0)
#   LORRAX_MPI_PROVIDER=<x>   force-request provider <x>
#                             (never `verbs` at P>=144 one-block: AP.4)
# In-container, mlx additionally needs the staged host RDMA/UCX userspace
# (AS.1 hostlibs pattern) and NO --bind under /dev.  Without it, auto
# degrades to tcp — announced by the I_MPI_DEBUG banner.
case "${LORRAX_MPI_PROVIDER:-auto}" in
  auto) unset FI_PROVIDER FI_TCP_IFACE 2>/dev/null
        _lorrax_mpi_say "provider auto: FI_PROVIDER unset -> IMPI native pick (mlx on CLX); trust only the I_MPI_DEBUG banner" ;;
  tcp)  IB_IF=$(ip -o -4 addr show 2>/dev/null | awk '/ib/{print $2; exit}')
        export FI_PROVIDER=tcp FI_TCP_IFACE=${IB_IF:-ib0}
        _lorrax_mpi_say "provider tcp (rtx escape hatch): FI_TCP_IFACE=$FI_TCP_IFACE" ;;
  *)    export FI_PROVIDER="$LORRAX_MPI_PROVIDER"
        _lorrax_mpi_say "provider forced: FI_PROVIDER=$FI_PROVIDER" ;;
esac

# ---- UCX tunings for the mlx provider (AU A/B, audit fix/zq 2026-07-28) ---
# TACC's default impi module exports these into every login/compute shell,
# so EVERY AP/AS mlx number (1.07 us / 11.4 GB/s pingpong, pzheevd
# 0.5-0.9 s/q) was measured UNDER them.  AU measured that stripping them
# leaves 8 B latency unchanged but DOUBLES the 1 MiB 32-rank Allreduce
# (419 -> 799 us) — load-bearing for large-message collectives.  Setdefault
# the six certified-harness values (mos2_4x4_test/gw800_merged.sbatch, now
# vendored as templates/gw_dev.sbatch) so a stripped launch environment
# (cron, clean ssh) does not silently lose 2x; inherited values always win.
# Skipped under the tcp escape hatch: rtx/mlx4 has no dc_x transport, and
# UCX is not in the tcp path anyway.  Do not add other UCX knobs
# (env_vars.md section 5).
if [ "${LORRAX_MPI_PROVIDER:-auto}" != "tcp" ]; then
  export UCX_TLS=${UCX_TLS:-knem,dc_x,rc}
  export UCX_RC_MLX5_RETRY_COUNT=${UCX_RC_MLX5_RETRY_COUNT:-40}
  export UCX_RC_MLX5_TIMEOUT=${UCX_RC_MLX5_TIMEOUT:-1200000.00us}
  export UCX_DC_MLX5_RETRY_COUNT=${UCX_DC_MLX5_RETRY_COUNT:-40}
  export UCX_DC_MLX5_TIMEOUT=${UCX_DC_MLX5_TIMEOUT:-1200000.00us}
  export UCX_UD_MLX5_TIMEOUT=${UCX_UD_MLX5_TIMEOUT:-600000000.00us}
fi

# Provider banner ("libfabric provider: ...") is mandatory telemetry —
# fi_info reports -61 for mlx even where it works; trust only the banner.
export I_MPI_DEBUG="${I_MPI_DEBUG:-4}"

unset -f _lorrax_mpi_say
