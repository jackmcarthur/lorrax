#!/usr/bin/env bash
# ============================================================================
# gpu_env.sh — Frontera rtx GPU/FFI environment.  Source INSIDE the container
# before running an FFI test/driver on a GPU node.
#
# Split out of ffi_env.sh (2026-07-31, env audit P0.3): this file owns the
# GPU-only half — the FFI .so path, the venv nvidia wheel LD_LIBRARY_PATH,
# cuSOLVERMp/NCCL knobs, JAX numerics, and the sm_75 command-buffer XLA_FLAGS
# that the runtime's cuda_async pool needs on Turing (see the allocator
# comment below).  MPI transport hygiene lives in mpi_transport_env.sh; the phdf5
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
# The GPU memory pool is runtime policy: runtime.set_default_gpu_pool()
# selects cuda_async with its pool reserved on every CUDA run (measured
# rationale in that docstring; this file used to export ALLOCATOR=cuda_async
# with PREALLOCATE=false, the unreserved pool that re-maps memory at every
# synchronize).  What stays HERE is the one piece that is specific to this
# machine: the sm_75 + driver-535 command-buffer XLA_FLAGS restriction below,
# which cudaMallocAsync needs on Turing.  Every Frontera GPU run sources this
# file (it is also where the FFI .so path comes from), so the pair still
# travels together.

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
