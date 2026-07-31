#!/usr/bin/env bash
# ============================================================================
# gpu_env.sh — Frontera rtx GPU/FFI environment.  Source INSIDE the container
# before running an FFI test/driver on a GPU node.
#
# Split out of ffi_env.sh (2026-07-31, env audit P0.3): this file owns the
# GPU-only half — the FFI .so path, the venv nvidia wheel LD_LIBRARY_PATH,
# cuSOLVERMp/NCCL knobs, JAX numerics, and the cuda_async + sm_75 XLA_FLAGS
# PAIRING (the two halves must travel together — see the allocator comment
# below).  MPI transport hygiene lives in mpi_transport_env.sh; the phdf5
# staging block lives in the back-compat ffi_env.sh shim.
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
