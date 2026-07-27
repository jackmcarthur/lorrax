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

# NOTE (scorecard AG): this file deliberately does NOT set
# ISDF_JAX_CACHE_DIR.  The JAX persistent compile cache must be off on
# multi-process runs — process 0 alone writes it, so its peers diverge and
# block forever in XLA:GPU's cross-process autotuner key-value exchange
# (the silent `load_centroid_wfns` hang).  That is now enforced IN-TREE and
# process-count-aware by common/jax_compile_cache.py, which refuses the
# cache at jax.process_count() > 1 and prints why, so single-process GPU
# runs sourcing this file keep their warm-compile win.  Do not "fix" this
# by exporting ISDF_JAX_CACHE_DIR="" here — that would also disable the
# cache at P == 1, where it is safe and useful.

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
    # available" in-container).  For multi-node, set LORRAX_MPI_FABRICS=ofi
    # + FI_PROVIDER and sort out the ConnectX-3 fabric.
    export I_MPI_FABRICS="${LORRAX_MPI_FABRICS:-shm}"
    export FI_PROVIDER="${FI_PROVIDER:-tcp}"
    export HDF5_USE_FILE_LOCKING=FALSE
fi
