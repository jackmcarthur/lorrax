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
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# JAX numerics
export JAX_ENABLE_X64=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

# NCCL on Frontera rtx (ConnectX-3/mlx4, no NVLink): single-node uses PCIe
# P2P intra-socket / host-staging cross-socket. These are safe single-node
# defaults; for multi-node, IB-over-mlx4 may need NCCL_IB_DISABLE=1.
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-PHB}"
