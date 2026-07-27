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

// LORRAX_FFI_NO_CUDA is defined by the host-platform build (see
// common/cpp/host/CMakeLists.txt).  The collective MPI-IO read core in
// read_ffi.cc / context.cc is hardware-agnostic; only the device-staging
// tail (H2D vs a plain host memcpy) and the staging-buffer allocator
// (cudaMallocHost vs aligned host malloc) switch on this flag.  The CUDA
// build leaves it undefined and compiles the stream/event/pinned members
// below; the host build drops them.
#ifndef LORRAX_FFI_NO_CUDA
#include <cuda_runtime.h>
#endif
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

    // Staging — a single host buffer H5Dread lands into before the
    // device-staging tail.  On the CUDA build it is cudaMallocHost-pinned
    // and copied H2D on ``stream`` with completion signalled via
    // ``h2d_event``; on the host build it is a plain aligned malloc and the
    // tail is a std::memcpy into the (host-resident) XLA output buffer.
    // The buffer + capacity fields are shared by both; the CUDA-only
    // stream/event/device fields below are compiled out on the host build.
    void*        pinned_buf          = nullptr;
    size_t       pinned_capacity     = 0;
#ifndef LORRAX_FFI_NO_CUDA
    // private CUDA stream for D2H/H2D copies.
    cudaStream_t stream              = nullptr;
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
#endif

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
    // reads AND collective writes (two-phase aggregation; the strided
    // 2-D tile pattern of the production writes is ~3 orders faster
    // collective — scorecard AI — and the default now matches the
    // Python phdf5_host writer so LORRAX_PHDF5_COLLECTIVE_WRITES=0
    // means "independent" in every writer.  Historical Cray-MPICH
    // caution: ad_cray_write_coll.c:669 OOM at >~1 GB/rank on that
    // stack — use the env override there).  Metadata defaults to
    // non-collective so file-level ops (H5Dcreate/extend) bypass the
    // collective driver — per ARCHITECTURE.md non-collective meta is
    // ~100 ms faster on OpenMPI small writes too.  dedup_replicas
    // drops all-but-one writer of a replica group's identical
    // hyperslab (mesh axes not consumed by the array's sharding):
    // required for correctness under collective writes (overlapping
    // selections are undefined) and pure waste-removal under
    // independent writes.  LORRAX_PHDF5_DEDUP_REPLICAS=0 disables.
    bool   use_collective_read  = true;
    bool   use_collective_write = true;
    bool   coll_metadata        = false;
    bool   dedup_replicas       = true;
    size_t align_threshold   = 1 << 20;          // 1 MiB
    size_t align_length      = 1 << 20;
};

// Grow ctx->pinned_buf to at least `need_bytes` (cudaMallocHost).
// Idempotent; returns false on cudaMalloc failure.
bool ensure_pinned(PhdfCtx* ctx, size_t need_bytes);

}  // namespace lorrax_ffi::phdf5
