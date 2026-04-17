// ctx.h — per-file context for the parallel-HDF5 FFI.
//
// Created at open_file time on a collective MPI + H5Fcreate path,
// reused across every write to that file, torn down on close_file.
// One PhdfCtx per open file.  openPMD-api pattern: all property lists
// are cached at open time and reused; no per-call churn.

#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>
#include <mpi.h>
#include <hdf5.h>

namespace lorrax_ffi::phdf5 {

struct PhdfCtx {
    // Identity / process grid
    int rank = -1;
    int world_size = 0;
    int p = 0, q = 0;                            // 2-D mesh shape
    std::string path;

    // Owned MPI + HDF5 resources (created at open, destroyed at close).
    MPI_Comm comm            = MPI_COMM_NULL;
    bool     owns_comm       = false;            // dup'd from MPI_COMM_WORLD? then free on close
    hid_t    file_id         = H5I_INVALID_HID;
    hid_t    fapl_id         = H5I_INVALID_HID;  // mpio + coll_meta + align
    hid_t    fcpl_id         = H5I_INVALID_HID;
    hid_t    dxpl_coll       = H5I_INVALID_HID;  // H5FD_MPIO_COLLECTIVE
    hid_t    dxpl_indep      = H5I_INVALID_HID;  // H5FD_MPIO_INDEPENDENT
    hid_t    dcpl_id         = H5I_INVALID_HID;  // fill=NEVER, alloc=EARLY, chunked layout

    // Cache of open datasets, keyed by HDF5 path ("ds_name" or
    // "/group/ds_name").  H5Dcreate on first write, H5Dopen on re-open.
    std::unordered_map<std::string, hid_t> open_datasets;
    std::mutex                             datasets_mu;

    // Staging — one reusable pinned host buffer.  Writes are synchronous
    // (block-per-call), so no pool is needed.  Grown on demand.
    cudaStream_t stream              = nullptr;
    void*        pinned_buf          = nullptr;
    size_t       pinned_capacity     = 0;

    // Tuning flags, read from env at open time.
    bool   use_collective    = true;
    bool   coll_metadata     = true;
    size_t align_threshold   = 1 << 20;          // 1 MiB
    size_t align_length      = 1 << 20;
};

// Grow ctx->pinned_buf to at least `need_bytes` (cudaMallocHost).
// Idempotent; returns false on cudaMalloc failure.
bool ensure_pinned(PhdfCtx* ctx, size_t need_bytes);

}  // namespace lorrax_ffi::phdf5
