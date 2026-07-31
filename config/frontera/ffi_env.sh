#!/usr/bin/env bash
# ============================================================================
# ffi_env.sh — BACK-COMPAT SHIM (deprecated entry point, 2026-07-31 split).
#
# This file used to be four scripts in one: GPU/FFI env, MPI transport
# hygiene, the phdf5 staging block, and the provider/UCX dials.  It is now a
# thin shim kept only so existing harnesses and docs that
# `source config/frontera/ffi_env.sh` keep working.  New scripts and jobs
# should source the split pieces directly:
#
#   gpu_env.sh            — rtx CUDA env: FFI .so, nvidia wheel libs, the
#                           cuda_async + sm_75 XLA_FLAGS pairing
#   mpi_transport_env.sh  — Intel-MPI transport: PMI2 glue, fabrics,
#                           LORRAX_MPI_PROVIDER case-block, UCX setdefaults.
#                           Now applied UNCONDITIONALLY (it used to hide
#                           behind LORRAX_FFI_PHDF5=1, leaving plain CPU
#                           sourcing runs with no transport hygiene).
#
# BEHAVIOUR CHANGES vs the pre-split file (both announced at source time):
#   * transport hygiene no longer requires LORRAX_FFI_PHDF5=1;
#   * I_MPI_FABRICS defaults to shm:ofi (Intel MPI's own multi-node
#     default), not bare `shm` (an rtx single-node bring-up value that
#     silently pinned transport).  LORRAX_MPI_FABRICS=shm is the explicit
#     rtx hatch.
#
# The phdf5 STAGING block (the .so swap + host Intel-MPI HDF5 + Intel
# runtime LD_LIBRARY_PATH) still lives here, still gated on
# LORRAX_FFI_PHDF5=1.
# ============================================================================

_lorrax_ffi_env_dir=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

. "$_lorrax_ffi_env_dir/gpu_env.sh"
. "$_lorrax_ffi_env_dir/mpi_transport_env.sh"

# --- phdf5 (host Intel-MPI parallel HDF5, hybrid-mounted) -------------------
# Set LORRAX_FFI_PHDF5=1 before sourcing to run the phdf5-inclusive .so.
if [ "${LORRAX_FFI_PHDF5:-0}" = "1" ]; then
    export LORRAX_FFI_SO="${LORRAX_FFI_SO_PHDF5:-$LORRAX_FFI_STAGE/build_phdf5/liblorrax_ffi.so}"
    HDF5_ROOT_DIR="${LORRAX_HDF5_ROOT:-/home1/apps/intel19/impi19_0/phdf5/1.14.6}"
    # LORRAX_IMPI_ROOT / I_MPI_ROOT are already resolved by
    # mpi_transport_env.sh above.
    # Intel compiler runtime (libimf/libintlc/libsvml — the Intel-built HDF5
    # and libfabric NEED these).  2020.1 runtime is ABI-compatible with the
    # 2020.4 MPI.
    ICC_RT="${LORRAX_ICC_RUNTIME:-/opt/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin}"
    # Intel MPI runtime libs (libmpi.so.12 + bundled libfabric) + HDF5.
    export LD_LIBRARY_PATH="$HDF5_ROOT_DIR/lib:$LORRAX_IMPI_ROOT/lib/release:$LORRAX_IMPI_ROOT/lib:$LORRAX_IMPI_ROOT/libfabric/lib:$ICC_RT:$LD_LIBRARY_PATH"
    export HDF5_USE_FILE_LOCKING=FALSE
fi

unset _lorrax_ffi_env_dir
