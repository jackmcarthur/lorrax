#!/usr/bin/env bash
# ============================================================================
# ffi_env.sh — source INSIDE the container before running an FFI test/driver.
# Sets LD_LIBRARY_PATH so liblorrax_ffi.so's deps (libcusolverMp/libcublasmp
# from the stage, and libnccl/libcudart/... from the venv's pip nvidia
# wheels) resolve, plus the cuSOLVERMp + JAX runtime knobs.
# ============================================================================
: "${LORRAX_VENV:=$WORK/lorrax_env/.venv}"
: "${LORRAX_FFI_STAGE:=$WORK/lorrax_ffi}"

# The built shared object (ffi_loader.py reads LORRAX_FFI_SO first).
export LORRAX_FFI_SO="${LORRAX_FFI_SO:-$LORRAX_FFI_STAGE/build/liblorrax_ffi.so}"

# Runtime library search path: staged cuSOLVERMp/cuBLASMp + all venv pip
# nvidia-*-cu12 lib dirs (libnccl, libcudart, libcublas, libcusolver, ...).
_NV_LIBS=$(find "$LORRAX_VENV"/lib/python*/site-packages/nvidia -maxdepth 2 \
             -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="$LORRAX_FFI_STAGE/stage/lib:${_NV_LIBS}${LD_LIBRARY_PATH:-}"

# cuSOLVERMp / cuBLASMp: route CAL/collectives through NCCL (no IB/UCC), and
# use the CUDA async mempool so JAX and the solver share VRAM (avoids the
# NCCL-starved-of-VRAM -> cusolverMpSyevd status=7 failure).
export CUSOLVERMP_FORCE_NCCL=1
# ---------------------------------------------------------------------------
# WHY THIS FILE MAY SET A DIFFERENT ALLOCATOR THAN runtime.set_default_env()
#
# PREALLOCATE=false is NOT a difference -- it is the same canonical value the
# runtime now sets for every driver, restated here because this file is also
# sourced by bare FFI tests that never enter Python's bootstrap.
#
# ALLOCATOR=cuda_async IS a deliberate deployment-only override, and this is
# the one file entitled to make it.  Measured on 8x Quadro RTX 5000 across 2
# nodes (job 7882447, 6 GiB of live XLA arrays on a 15.74 GB card):
#
#     allocator            XLA overhead   largest cuFFT plan still creatable
#     default (BFC)+false   2.13 GB        7.16 GB
#     cuda_async    +false  0.19 GB        9.20 GB
#
# so cuda_async is worth ~2 GB per card to the out-of-XLA cuFFT arena, and
# unlike `platform` it still reports peak_bytes_in_use.  It is not the
# in-code default only because on Turing (sm_75) cudaMallocAsync REQUIRES the
# command-buffer restriction set further down this same file (see the
# XLA_FLAGS block below) -- and this file is the only place that sets both
# halves together.  A run that sources this script gets the matched pair; a
# run that does not gets the runtime's safe BFC default.
#
# Do NOT move cuda_async into runtime.set_default_env() without moving that
# XLA_FLAGS mitigation with it.  `export` here (not setdefault) is correct:
# it must beat the runtime's setdefault, which is written to yield.
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# JAX numerics
export JAX_ENABLE_X64=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

# NOTE (scorecard AG, then AH): this file deliberately does NOT set
# ISDF_JAX_CACHE_DIR, and that is now the *right* default rather than a
# workaround.  History: the JAX persistent compile cache used to hang every
# multi-process GPU run — process 0 alone writes entries, the old per-rank
# cache layout therefore kept the peers' dirs empty forever, process 0 hit
# and skipped compilation, and its peers blocked forever in XLA:GPU's
# cross-process autotuner key-value exchange (the silent
# `load_centroid_wfns` hang).  common/jax_compile_cache.py used to refuse
# the cache at jax.process_count() > 1 for that reason; it now REPAIRS it
# (process-invariant cache key + a coordination-service agreement on the
# usable entry set + atomic writes), so multi-process runs sourcing this
# file get the warm-compile win too.  Do not "fix" anything by exporting
# ISDF_JAX_CACHE_DIR="" here — that throws the win away at every P.

# Turing (sm_75) + driver 535 + cudaMallocAsync: XLA CUDA-graph capture of
# FUSION/WHILE command buffers fails "Failed to add memset node to a CUDA
# graph (CUDA_ERROR_INVALID_VALUE)" (e.g. the ζ-fit r-chunk).  Keep the
# library-call graphs (cuBLAS / custom-call — the GEMMs and cuSolverMp) but
# exclude FUSION/WHILE where the failing memset lives.  Set
# LORRAX_XLA_CMDBUF="" to disable command buffers entirely.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer=${LORRAX_XLA_CMDBUF:-CUBLAS,CUBLASLT,CUSTOM_CALL}"

# NCCL on Frontera rtx (ConnectX-3/mlx4, no NVLink): single-node uses PCIe
# P2P intra-socket / host-staging cross-socket. These are safe single-node
# defaults; for multi-node, IB-over-mlx4 may need NCCL_IB_DISABLE=1.
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-PHB}"

# --- phdf5 (host Intel-MPI parallel HDF5, hybrid-mounted) -------------------
# Set LORRAX_FFI_PHDF5=1 before sourcing to run the phdf5-inclusive .so.
if [ "${LORRAX_FFI_PHDF5:-0}" = "1" ]; then
    export LORRAX_FFI_SO="${LORRAX_FFI_SO_PHDF5:-$LORRAX_FFI_STAGE/build_phdf5/liblorrax_ffi.so}"
    HDF5_ROOT_DIR="${LORRAX_HDF5_ROOT:-/home1/apps/intel19/impi19_0/phdf5/1.14.6}"
    IMPI="${LORRAX_IMPI_ROOT:-/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64}"
    export I_MPI_ROOT="${IMPI%/intel64}"   # wrappers/runtime append intel64/
    # Intel compiler runtime (libimf/libintlc/libsvml — the Intel-built HDF5
    # and libfabric NEED these).  2020.1 runtime is ABI-compatible with the
    # 2020.4 MPI.
    ICC_RT="${LORRAX_ICC_RUNTIME:-/opt/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin}"
    # Intel MPI runtime libs (libmpi.so.12 + bundled libfabric) + HDF5.
    export LD_LIBRARY_PATH="$HDF5_ROOT_DIR/lib:$IMPI/lib/release:$IMPI/lib:$IMPI/libfabric/lib:$ICC_RT:$LD_LIBRARY_PATH"
    # Bootstrap Intel MPI under srun --mpi=pmi2 via the host slurm PMI2 lib
    # (pre-copied to $WORK/host_pmi).  MUST override unconditionally: TACC's
    # login environment sets I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so, a host
    # PMI-1 lib that is the wrong protocol for --mpi=pmi2 AND absent inside
    # the container -> MPIR_pmi_init fails.  Force our staged PMI2 lib.
    export I_MPI_PMI_LIBRARY="${LORRAX_PMI2_LIB:-$WORK/host_pmi/libpmi2.so.0}"
    # Single node: shared-memory-only transport skips OFI/libfabric init
    # entirely (the tcp/mlx4 OFI provider fails "addrinfo() No data
    # available" in-container on rtx/ConnectX-3).  For multi-node, set
    # LORRAX_MPI_FABRICS=shm:ofi and pick the provider below.
    export I_MPI_FABRICS="${LORRAX_MPI_FABRICS:-shm}"
    # Provider .so resolution — REQUIRED in-container (AU measured: with it
    # unset, PMPI_Init aborts "addrinfo() failed ... No data available":
    # libfabric finds no providers.  mpivars.sh, which normally sets it, is
    # not sourced in-container).
    export FI_PROVIDER_PATH="$IMPI/libfabric/lib/prov"
    # ---- libfabric provider (scorecard AP root cause, AS.5 spec) ----------
    # The old `FI_PROVIDER="${FI_PROVIDER:-tcp}"` seed here was an rtx/mlx4
    # bring-up workaround that leaked to CLX production: on Frontera CLX HDR
    # leave FI_PROVIDER UNSET and Intel MPI auto-selects the native `mlx`
    # (UCX) provider — 1.07 us / 11.4 GB/s vs tcp's 10.9 us / 2.15 GB/s;
    # pzheevd n=2448 P=144 12 s/q -> 0.5-0.9 s/q.  NOTE the seed also never
    # did what it claimed: TACC's default impi module exports FI_PROVIDER=mlx
    # in every login/compute shell, so the `:-tcp` fallback usually saw an
    # inherited value — the case-block below unsets/sets it deterministically.
    #   LORRAX_MPI_PROVIDER=auto  (default) unset -> IMPI picks mlx on CLX
    #   LORRAX_MPI_PROVIDER=tcp   the rtx/mlx4 escape hatch (IPoIB via ib0)
    #   LORRAX_MPI_PROVIDER=<x>   force-request provider <x>
    #                             (never `verbs` at P>=144 one-block: AP.4)
    # In-container, mlx additionally needs the staged host RDMA/UCX userspace
    # (AS.1 hostlibs pattern) and NO --bind under /dev.  Without it, auto
    # degrades to tcp — announced by the I_MPI_DEBUG banner.
    case "${LORRAX_MPI_PROVIDER:-auto}" in
      auto) unset FI_PROVIDER FI_TCP_IFACE 2>/dev/null ;;
      tcp)  IB_IF=$(ip -o -4 addr show 2>/dev/null | awk '/ib/{print $2; exit}')
            export FI_PROVIDER=tcp FI_TCP_IFACE=${IB_IF:-ib0} ;;
      *)    export FI_PROVIDER="$LORRAX_MPI_PROVIDER" ;;
    esac
    # ---- UCX tunings for the mlx provider (AU A/B, audit fix/zq 2026-07-28)
    # TACC's default impi module exports these into every login/compute
    # shell, so EVERY AP/AS mlx number (1.07 us / 11.4 GB/s pingpong,
    # pzheevd 0.5-0.9 s/q) was measured UNDER them.  AU measured that
    # stripping them leaves 8 B latency unchanged but DOUBLES the 1 MiB
    # 32-rank Allreduce (419 -> 799 us) — load-bearing for large-message
    # collectives.  Setdefault the six certified-harness values
    # (mos2_4x4_test/gw800_merged.sbatch) so a stripped launch environment
    # (cron, clean ssh) does not silently lose 2x; inherited values always
    # win.  Skipped under the tcp escape hatch: rtx/mlx4 has no dc_x
    # transport, and UCX is not in the tcp path anyway.  Do not add other
    # UCX knobs (env_vars.md section 5).
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
    export HDF5_USE_FILE_LOCKING=FALSE
fi
