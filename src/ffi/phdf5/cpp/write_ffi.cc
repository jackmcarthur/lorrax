// write_ffi.cc — XLA FFI handler that writes a sharded JAX array to a
// hyperslab of an open parallel-HDF5 dataset.
//
// N-D hyperslab, block-partitioned sharding.  Attrs (same on every rank
// because shard_map compiles the body once):
//
//   offset_base[N]    — global origin of the hyperslab the caller wants
//                       to land in the file.
//   mesh_shape[M]     — the JAX Mesh's shape, row-major.
//   axis_for_dim[N]   — for each array dim, which mesh axis shards it
//                       (-1 = replicated).
//
// count[N] comes from A.dimensions() (this rank's local shard shape —
// same on every rank for block-partitioned layouts).
//
// Per-rank offset is derived in C++ by un-raveling ctx->rank through
// mesh_shape, then advancing offset_base along every sharded dim.  This
// matches jax.sharding.NamedSharding(P(...)) on a Mesh with the given
// `mesh_shape`, for any subset of dims being replicated vs sharded.
//
// ─── Async handler + Python worker thread ─────────────────────────────
// This handler returns an ``ffi::Future`` rather than ``ffi::Error``.
// XLA detects the async return type via ``ResultEncoding<stage, Future>``
// template specialization (xla/ffi/api/ffi.h:1239), creates an
// XLA_FFI_Future, and hands it back to the runtime.
//
// Measured 2026-04-18: ffi::Future on its own does NOT release the
// Python main thread.  ``jit(ffi_call)(A)`` blocks Python until the
// Future is marked Available, i.e. until ``H5Dwrite`` completes on the
// writer thread below (350ms observed with a 300ms artificial sleep
// after SetAvailable — see report.md in
// reports/session_2026-04-18_async_probe/).  ffi::Future only helps
// overlap downstream XLA ops; Python dispatch is still serialized.
//
// To actually free the Python main thread, ``_slab_io_ffi.py`` adds a
// second layer of async: a Python-level worker thread that owns the
// ``jit(ffi_call)(A).block_until_ready()`` call in FIFO order.  Main
// thread enqueues onto that Python queue and returns in ~0.2ms.
//
// What this C++ file buys us:
//  - Writer thread + task queue serialise ``H5Dwrite`` across ranks
//    in dispatch order, which is the MPI-IO collective rendezvous
//    requirement.  Per-call detached threads could let OS scheduling
//    reorder H5Dwrites between ranks.
//  - ``ensure_pinned`` grows ``ctx->pinned_buf`` once (first call)
//    and reuses it on every subsequent write.  The Python worker
//    thread serialises dispatches so no second write is in flight
//    when the next handler D2Hs into the buffer.
//  - ``ctx->d2h_event`` is a single ctx-owned cudaEvent, reused
//    across writes (re-recorded, re-synced).  Creating a fresh event
//    per call and destroying it in the writer thread caused
//    cudaEventDestroy to block ~800 ms on 3 of 4 ranks at MoS2 3x3
//    (measured 2026-04-18) — see the mystery-solved note below.
//
// Work split:
//  - XLA executor thread (fast path, ~0.1 ms): ensure_pinned, kick
//    cudaMemcpyAsync D2H on ctx->stream, cudaEventRecord the ctx's
//    reusable d2h_event, enqueue a task on the writer thread, return
//    the Future.
//  - Writer thread (slow path, one per ctx): cudaEventSynchronize on
//    ctx->d2h_event, H5Dwrite (MPI-IO collective, ~750 ms),
//    promise.SetAvailable().
//
// ─── The 800 ms "cudaEventDestroy" mystery (resolved 2026-04-18) ───
// Original design created a fresh event in the handler and destroyed
// it in the writer thread.  At MoS2 3x3 scale on 4 ranks, non-rank-0
// writer threads blocked ~800 ms inside ``cudaEventDestroy``.
// Instrumented per-stage timestamps pinned it to the destroy call
// specifically; skipping destroy (leaking events) or reusing a single
// event (this file) both eliminate the stall.  Root cause not fully
// understood, likely a cuda_async stream-ordered allocator
// interaction — destroying an event with outstanding xla_stream
// dependencies appears to wait on that stream to drain.  The fix
// here — one event per ctx — is clean and correct.

#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "xla/ffi/api/ffi.h"

#include "../../common/cpp/ffi_helpers.h"
#include "ctx.h"
#include "phdf5_interface.h"

namespace lorrax_ffi::phdf5 {

namespace ffi = ::xla::ffi;

// Un-ravel a linear rank id through a row-major mesh shape into
// per-axis coordinates.  rank = sum(coord[i] * prod(shape[i+1:])).
static std::vector<int64_t> unravel_rank(
    int64_t rank, ffi::Span<const int64_t> mesh_shape)
{
    std::vector<int64_t> coord(mesh_shape.size(), 0);
    int64_t r = rank;
    for (ssize_t i = (ssize_t)mesh_shape.size() - 1; i >= 0; --i) {
        coord[i] = r % mesh_shape[i];
        r /= mesh_shape[i];
    }
    return coord;
}

template <typename T>
static std::string vec_to_string(const std::vector<T>& v)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ",";
        os << v[i];
    }
    os << "]";
    return os.str();
}

static bool write_debug_enabled()
{
    const char* env = std::getenv("LORRAX_PHDF5_WRITE_DEBUG");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

static std::string h5_object_name(hid_t id)
{
    ssize_t n = H5Iget_name(id, nullptr, 0);
    if (n <= 0) return "<unknown>";
    std::string name((size_t)n + 1, '\0');
    H5Iget_name(id, name.data(), (size_t)n + 1);
    name.resize((size_t)n);
    return name;
}

// Run H5Dwrite on the ctx writer thread.  If ``wait_for_d2h`` is
// true, first cudaEventSynchronize on the ctx's reusable d2h_event
// so the pinned host buffer is valid.  We deliberately DO NOT
// destroy the event here: cudaEventDestroy on a recently-recorded
// event (per-call) blocks 700-800 ms on 3 of 4 ranks when xla_stream
// has a main-thread backlog (measured 2026-04-18, cause: likely
// an interaction with CUDA's stream-ordered allocator).  Using a
// single ctx-owned event reused across writes avoids the per-call
// destroy entirely; the event is destroyed once in close_ctx.
static void async_worker(
    PhdfCtx* ctx,
    hid_t ds_id,
    hid_t native_type,
    std::vector<hsize_t> offset,
    std::vector<hsize_t> count,
    void* pinned_buf,
    bool wait_for_d2h,
    ffi::Promise promise)
{
    if (wait_for_d2h) {
        cudaEventSynchronize(ctx->d2h_event);
    }
    const int rank = (int)offset.size();
    hid_t filespace = H5Dget_space(ds_id);
    if (filespace < 0) {
        promise.SetError(ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 async write: H5Dget_space failed"));
        return;
    }
    int file_rank = H5Sget_simple_extent_ndims(filespace);
    if (file_rank != rank) {
        std::ostringstream os;
        os << "phdf5 async write: dataset rank mismatch"
           << " ds=" << h5_object_name(ds_id)
           << " file_rank=" << file_rank
           << " write_rank=" << rank;
        H5Sclose(filespace);
        promise.SetError(ffi::Error(ffi::ErrorCode::kInternal, os.str()));
        return;
    }
    std::vector<hsize_t> extent((size_t)rank, 0);
    std::vector<hsize_t> max_extent((size_t)rank, 0);
    if (H5Sget_simple_extent_dims(filespace, extent.data(), max_extent.data()) < 0) {
        H5Sclose(filespace);
        promise.SetError(ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 async write: H5Sget_simple_extent_dims failed"));
        return;
    }
    bool out_of_bounds = false;
    for (int d = 0; d < rank; ++d) {
        if (offset[(size_t)d] + count[(size_t)d] > extent[(size_t)d]) {
            out_of_bounds = true;
        }
    }
    bool debug = write_debug_enabled();
    if (debug || out_of_bounds) {
        std::fprintf(stderr,
            "[phdf5-write rank=%d ds=%s id=%lld] extent=%s offset=%s count=%s "
            "collective=%d oob=%d\n",
            ctx->rank, h5_object_name(ds_id).c_str(), (long long)ds_id,
            vec_to_string(extent).c_str(), vec_to_string(offset).c_str(),
            vec_to_string(count).c_str(), ctx->use_collective_write ? 1 : 0,
            out_of_bounds ? 1 : 0);
        std::fflush(stderr);
    }
    if (out_of_bounds) {
        std::ostringstream os;
        os << "phdf5 async write: hyperslab out of bounds"
           << " ds=" << h5_object_name(ds_id)
           << " extent=" << vec_to_string(extent)
           << " offset=" << vec_to_string(offset)
           << " count=" << vec_to_string(count)
           << " rank=" << ctx->rank;
        H5Sclose(filespace);
        promise.SetError(ffi::Error(ffi::ErrorCode::kInternal, os.str()));
        return;
    }
    if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                             offset.data(), nullptr,
                             count.data(),  nullptr) < 0) {
        if (debug) H5Eprint2(H5E_DEFAULT, stderr);
        H5Sclose(filespace);
        promise.SetError(ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 async write: H5Sselect_hyperslab failed"));
        return;
    }
    hid_t memspace = H5Screate_simple(rank, count.data(), nullptr);
    if (memspace < 0) {
        H5Sclose(filespace);
        promise.SetError(ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 async write: H5Screate_simple(memspace) failed"));
        return;
    }

    hid_t dxpl = ctx->use_collective_write ? ctx->dxpl_coll : ctx->dxpl_indep;
    herr_t st = H5Dwrite(ds_id, native_type, memspace, filespace, dxpl,
                         pinned_buf);
    if (st < 0 && debug) H5Eprint2(H5E_DEFAULT, stderr);

    H5Sclose(memspace);
    H5Sclose(filespace);

    if (st < 0) {
        promise.SetError(ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 async write: H5Dwrite failed"));
        return;
    }
    promise.SetAvailable();
}

// ---- Dispatch ------------------------------------------------------------
// NOTE on ``offset_base`` being a Buffer rather than an Attr: offset
// changes per chunk (0, r, 2r, 3r for zeta) and FFI Attrs are baked
// into the XLA compile at dispatch time, so a fresh compile is needed
// for every distinct offset.  At MoS2 3x3 that meant 9 × 900 MiB HLO
// modules compiling just for FFI writes (measured via the profiling
// stack's HLO dump 2026-04-19).  Passing offset as a traced device
// buffer makes the trace signature shape-only (ndim), so shard_map
// closures compile ONCE per (ds_id, ndim, dtype, sharding).
static ffi::Future WriteDispatch(
    cudaStream_t xla_stream,
    ffi::AnyBuffer A,
    ffi::Buffer<ffi::DataType::S64> offset_buf,   // shape (ndim,)
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> token_out,
    int64_t ctx_handle,
    int64_t ds_id,
    ffi::Span<const int64_t> mesh_shape,
    ffi::Span<const int64_t> axis_count_per_dim,
    ffi::Span<const int64_t> axis_flat)
{
    auto fail = [](ffi::Error err) {
        ffi::Promise p;
        ffi::Future f(p);
        p.SetError(std::move(err));
        return f;
    };

    auto* ctx = reinterpret_cast<PhdfCtx*>(ctx_handle);
    if (!ctx) {
        return fail(ffi::Error(ffi::ErrorCode::kInvalidArgument,
                               "phdf5 write: ctx_handle is null"));
    }

    const auto dims = A.dimensions();
    const size_t N = dims.size();
    if (offset_buf.dimensions().size() != 1 ||
        (size_t)offset_buf.dimensions()[0] != N ||
        axis_count_per_dim.size() != N) {
        std::ostringstream os;
        os << "phdf5 write: rank mismatch  A.ndim=" << N
           << " offset_buf.ndim=" << offset_buf.dimensions().size()
           << " axis_count_per_dim.size=" << axis_count_per_dim.size();
        return fail(ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str()));
    }

    // D2H-copy the small offset buffer (N × 8 bytes = 24-40 bytes
    // typically).  Blocking cudaMemcpy; microseconds.
    std::vector<int64_t> offset_host(N);
    cudaError_t ce_off = cudaMemcpy(offset_host.data(), offset_buf.untyped_data(),
                                    N * sizeof(int64_t), cudaMemcpyDeviceToHost);
    if (ce_off != cudaSuccess) {
        std::ostringstream os;
        os << "phdf5 write: cudaMemcpy(offset) failed: "
           << cudaGetErrorString(ce_off);
        return fail(ffi::Error(ffi::ErrorCode::kInternal, os.str()));
    }

    std::vector<int64_t> coord = unravel_rank(ctx->rank, mesh_shape);
    std::vector<hsize_t> offset(N), count(N);
    size_t flat_idx = 0;
    for (size_t d = 0; d < N; ++d) {
        count[d] = (hsize_t)dims[d];
        offset[d] = (hsize_t)offset_host[d];
        int64_t na = axis_count_per_dim[d];
        // Dim d is sharded over `na` mesh axes; leftmost is slowest,
        // rightmost has stride=1.  rank_coord = sum_k coord[ax_k] * prod(mesh_shape[ax_{k+1:}]).
        int64_t rank_coord = 0;
        int64_t stride_acc = 1;
        for (int64_t k = na - 1; k >= 0; --k) {
            int64_t ax = axis_flat[flat_idx + k];
            if (ax < 0 || (size_t)ax >= mesh_shape.size()) {
                std::ostringstream os;
                os << "phdf5 write: bad axis " << ax
                   << " at dim " << d << " axis " << k;
                return fail(ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str()));
            }
            rank_coord += coord[ax] * stride_acc;
            stride_acc *= mesh_shape[ax];
        }
        offset[d] += rank_coord * count[d];
        flat_idx += (size_t)na;
    }

    // Element size + HDF5 native type per dtype (we don't dispatch by
    // template since the async body is dtype-agnostic — H5Dwrite takes
    // native_type and a void*).
    size_t elt_bytes = 0;
    hid_t native_type = H5I_INVALID_HID;
    const auto dtype = A.element_type();
    switch (dtype) {
        case ffi::DataType::F32:
            elt_bytes = sizeof(float);
            native_type = dt::h5_native_type<float>();
            break;
        case ffi::DataType::F64:
            elt_bytes = sizeof(double);
            native_type = dt::h5_native_type<double>();
            break;
        case ffi::DataType::S32:
            elt_bytes = sizeof(int32_t);
            native_type = dt::h5_native_type<int32_t>();
            break;
        case ffi::DataType::S64:
            elt_bytes = sizeof(int64_t);
            native_type = dt::h5_native_type<int64_t>();
            break;
        case ffi::DataType::C64:
            elt_bytes = sizeof(std::complex<float>);
            native_type = dt::h5_native_type<std::complex<float>>();
            break;
        case ffi::DataType::C128:
            elt_bytes = sizeof(std::complex<double>);
            native_type = dt::h5_native_type<std::complex<double>>();
            break;
        default: {
            std::ostringstream os;
            os << "phdf5 write: unsupported dtype " << static_cast<int>(dtype);
            return fail(ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str()));
        }
    }

    hid_t dset = (hid_t)ds_id;
    if (dset < 0 || H5Iis_valid(dset) <= 0) {
        return fail(ffi::Error(ffi::ErrorCode::kInvalidArgument,
                               "phdf5 write: ds_id is invalid"));
    }

    size_t n_local_elts = 1;
    for (auto c : count) n_local_elts *= (size_t)c;
    const size_t bytes = n_local_elts * elt_bytes;

    // Reuse ``ctx->pinned_buf``, growing on demand.  Safe because the
    // Python worker thread serializes write_slab calls: the prior
    // H5Dwrite has already completed (buffer released) by the time
    // the next dispatch reaches us.  ``cudaMallocHost`` at 270 MB is
    // ~50-100 ms — a per-call cost we can't afford.
    if (!ensure_pinned(ctx, bytes)) {
        std::ostringstream os;
        os << "phdf5 write: ensure_pinned(" << bytes << ") failed";
        return fail(ffi::Error(ffi::ErrorCode::kResourceExhausted, os.str()));
    }
    void* pinned_buf = ctx->pinned_buf;

    // D2H: cudaMemcpyAsync on ctx's private stream, then record the
    // ctx's REUSABLE d2h_event so the writer thread can sync on it.
    // Handler returns immediately after dispatching the copy.
    //
    // We deliberately do NOT cudaStreamWaitEvent(xla_stream, ev) to
    // guard A from XLA buffer donation.  Doing so, combined with
    // per-call event creation, caused cudaEventDestroy in the writer
    // thread to block ~800 ms on non-rank-0 processes (measured
    // 2026-04-18, likely an interaction with the cuda_async
    // stream-ordered allocator).  Using a single ctx-owned event
    // reused across writes avoids the per-call destroy entirely; the
    // event is destroyed once in close_ctx.  Correctness: Python's
    // ``A.block_until_ready()`` on the main thread before enqueue
    // ensures A's device storage is finalised before we read it; the
    // Python worker serialises writes so by the time the next
    // handler runs, the previous H5Dwrite has already consumed
    // pinned_buf.
    cudaError_t ce = cudaMemcpyAsync(pinned_buf, A.untyped_data(), bytes,
                                     cudaMemcpyDeviceToHost, ctx->stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "phdf5 write: cudaMemcpyAsync D2H failed: "
           << cudaGetErrorString(ce);
        return fail(ffi::Error(ffi::ErrorCode::kInternal, os.str()));
    }
    cudaEventRecord(ctx->d2h_event, ctx->stream);

    // Standard Promise/Future handshake: construct promise, construct
    // future from it (one-shot), then move promise into the task
    // closure.  The task runs on ctx->writer_thread in FIFO order.
    ffi::Promise promise;
    ffi::Future future(promise);

    auto task = [ctx, dset, native_type,
                 offset = std::move(offset),
                 count  = std::move(count),
                 pinned_buf,
                 promise = std::move(promise)]() mutable
    {
        async_worker(ctx, dset, native_type,
                     std::move(offset), std::move(count),
                     pinned_buf, /*wait_for_d2h=*/true,
                     std::move(promise));
    };

    {
        std::lock_guard<std::mutex> lk(ctx->queue_mu);
        ctx->task_queue.emplace_back(std::move(task));
    }
    ctx->queue_cv.notify_one();

    (void)token_out;
    return future;
}

}  // namespace lorrax_ffi::phdf5

// ---- FFI binding ---------------------------------------------------------
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PhdfWriteFfi, lorrax_ffi::phdf5::WriteDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()   // offset_base
        .Ret<xla::ffi::Buffer<xla::ffi::DataType::S32>>()
        .Attr<int64_t>("ctx_handle")
        .Attr<int64_t>("ds_id")
        .Attr<xla::ffi::Span<const int64_t>>("mesh_shape")
        .Attr<xla::ffi::Span<const int64_t>>("axis_count_per_dim")
        .Attr<xla::ffi::Span<const int64_t>>("axis_flat"));
