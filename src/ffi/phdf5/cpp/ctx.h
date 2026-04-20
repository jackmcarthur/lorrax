// ctx.h — per-file context for the parallel-HDF5 FFI.
//
// Created at open_file time on a collective MPI + H5Fcreate path,
// reused across every write to that file, torn down on close_file.
// One PhdfCtx per open file.  openPMD-api pattern: all property lists
// are cached at open time and reused; no per-call churn.

#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
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

    // Staging — private CUDA stream for D2H copies.  The pinned host
    // buffers are allocated per-call (see write_ffi.cc) so consecutive
    // async dispatches don't race on a shared buffer; fields below
    // survive as a no-op fallback for any future sync code path.
    cudaStream_t stream              = nullptr;
    void*        pinned_buf          = nullptr;
    size_t       pinned_capacity     = 0;
    int          cuda_device         = -1;         // CUDA device id the main thread was on at open_ctx time; the writer_thread binds here before running any CUDA API (otherwise GPU pointers from main thread's context aren't recognised → cudaErrorMemoryAllocation on cudaMemcpyAsync).
    // Reused I/O-completion events.  Created in open_ctx, destroyed
    // in close_ctx.  Per-call cudaEventCreate/Destroy blocks for
    // ~800 ms on non-root ranks when xla_stream has a backlog
    // (measured 2026-04-18) — likely a cuda_async stream-ordered
    // allocator interaction.  Pooling: one event for write (D2H
    // completion signalled to writer thread), one for read (H2D
    // completion signalled to xla_stream so downstream ops wait).
    // Safe to reuse because reads and writes are serialised through
    // the Python worker / writer thread.
    cudaEvent_t  d2h_event           = nullptr;
    cudaEvent_t  h2d_event           = nullptr;

    // ─── Async-write worker ─────────────────────────────────────────
    // Single dedicated thread drains ``task_queue`` in FIFO order.
    // Every XLA-FFI-dispatched ``write_slab`` enqueues one task; the
    // worker runs the H5Dwrite MPI-IO collective.  One thread per ctx
    // (not per call) guarantees writes rendezvous in the same order
    // on every rank, which is the MPI-IO collective correctness
    // requirement.  (With detached per-call threads, OS scheduling
    // could reorder task execution between ranks and deadlock the
    // collective.)
    std::thread                       writer_thread;
    std::mutex                        queue_mu;
    std::condition_variable           queue_cv;
    std::deque<std::function<void()>> task_queue;
    bool                              shutdown_flag = false;

    // Tuning flags, read from env at open time.  Default: collective
    // reads (OpenMPI-ROMIO's two-phase I/O is optimal) + independent
    // writes (avoids Cray MPICH's ad_cray_write_coll.c:669 OOM at
    // >~1 GB/rank, neutral on OpenMPI at our measured sizes).  Metadata
    // defaults to non-collective so file-level ops (H5Dcreate/extend)
    // also bypass the Cray collective driver — and per ARCHITECTURE.md
    // non-collective meta is ~100 ms faster on OpenMPI small writes too.
    bool   use_collective_read  = true;
    bool   use_collective_write = false;
    bool   coll_metadata        = false;
    size_t align_threshold   = 1 << 20;          // 1 MiB
    size_t align_length      = 1 << 20;
};

// Grow ctx->pinned_buf to at least `need_bytes` (cudaMallocHost).
// Idempotent; returns false on cudaMalloc failure.
bool ensure_pinned(PhdfCtx* ctx, size_t need_bytes);

}  // namespace lorrax_ffi::phdf5
