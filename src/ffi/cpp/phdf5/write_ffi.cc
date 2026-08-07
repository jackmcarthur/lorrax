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
//
// ─── LORRAX_FFI_NO_CUDA: the same TU builds the host lib ──────────────
// This file compiles UNCHANGED into liblorrax_ffi_host.so (the CUDA-free
// host-platform library) under -DLORRAX_FFI_NO_CUDA, exactly as
// read_ffi.cc does.  The collective MPI-IO write core — per-rank
// hyperslab derivation, valid_shape clipping, empty selection, the FIFO
// writer thread, H5Dwrite — is byte-identical on both platforms.  The
// three seams are:
//   1. handler binding: ``PhdfWriteHostFfi`` with no PlatformStream Ctx
//      (platform_seam.h),
//   2. index copy-in: the (ndim × int64) offset / valid_shape buffers are
//      a plain host read instead of cudaMemcpy D2H (platform_seam.h),
//   3. payload staging: the CUDA path cudaMemcpyAsync's the local shard
//      D2H into the ctx's pinned buffer and has the writer thread wait on
//      ``ctx->d2h_event``; on the host platform the XLA buffer IS host
//      memory, so H5Dwrite reads the shard IN PLACE — no staging copy, no
//      pinned allocation, no event.  That elision is the point: at MoS2
//      12×12 / 606 centroids the sharded ζ tile is 0.75 GB per rank, and
//      the allgather writer it replaces cost 12.05 GB of collective plus
//      a second host copy (scorecard AB.3).
// Lifetime of the in-place source pointer: ``_slab_io_ffi.write_slab``
// dispatches through a Python worker that holds a reference to ``A`` and
// blocks on the returned token, and XLA does not donate the input, so A's
// buffer stays live until the Future this handler returns is resolved —
// i.e. until H5Dwrite has finished reading it.  This is the same
// contract the CUDA path relies on for its cudaMemcpyAsync source.

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

#ifndef LORRAX_FFI_NO_CUDA
#include <cuda_runtime.h>
#endif
#include <hdf5.h>

#include "xla/ffi/api/ffi.h"

#include "../common/ffi_helpers.h"
#include "ctx.h"
#include "phdf5_interface.h"
#include "platform_seam.h"
#include "shard_index.h"

namespace lorrax_ffi::phdf5 {

namespace ffi = ::xla::ffi;

// ``unravel_rank`` / ``vec_to_string`` / ``validate_shard_encoding`` /
// ``checked_buffer_bytes`` / ``announce_error`` are in shard_index.h, shared
// verbatim with read_ffi.cc so the two TUs cannot drift.

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

// Run H5Dwrite on the ctx writer thread.  ``src_buf`` is the ctx's
// pinned staging buffer on the CUDA build and the XLA input buffer
// itself on the host build (seam 3).  If ``wait_for_d2h`` is
// true (CUDA only), first cudaEventSynchronize on the ctx's reusable
// d2h_event so the pinned host buffer is valid; on the host build
// there is no copy in flight and the flag is always false.
// We deliberately DO NOT
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
    std::vector<hsize_t> file_count,
    std::vector<hsize_t> mem_dims,
    std::vector<int64_t> offset_base,    // pre-shard origin, SAME on every rank
    std::vector<int64_t> valid_shape,    // logical extent, SAME on every rank
    void* src_buf,
    bool wait_for_d2h,
    size_t bytes,                        // per-rank payload, for the timing line
    ffi::Promise promise)
{
    // Diagnostic timing, symmetric with read_ffi.cc's two do_time sites.
    // LORRAX_PHDF5_TIME used to cover the READ path only, so setting it on
    // a full run produced no write lines at all while its name and
    // docs/dev/env_vars.md both promised phdf5 timing generally (measured
    // 2026-08-07: zero write lines on a run that wrote 1.9 GB).  Timing
    // lives HERE, on the ctx writer thread, because this is where the D2H
    // wait and the collective H5Dwrite actually happen — the dispatch half
    // of a write returns as soon as the task is queued, so a timer there
    // would report queueing latency and call it I/O.
    const bool do_time = env_flag("LORRAX_PHDF5_TIME", false);
    auto now = []() { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t0 = now();

#ifndef LORRAX_FFI_NO_CUDA
    if (wait_for_d2h) {
        cudaEventSynchronize(ctx->d2h_event);
    }
#else
    (void)wait_for_d2h;   // no D2H copy on the host build (seam 3)
#endif
    auto t_d2h = now();
    // Every error return below goes through here.  It announces first,
    // flushed, THEN sets the Promise: when a handler refuses, this rank does
    // not enter the collective its peers are inside, and the job dies minutes
    // later at a barrier deadline with the Promise's message still buffered
    // (audit 2026-08-04: two 32-node legs, no traceback anywhere).
    auto fail_task = [&](ffi::ErrorCode code, const std::string& msg) {
        announce_error(ctx, msg);
        promise.SetError(ffi::Error(code, msg));
    };

    const int rank = (int)offset.size();
    hid_t filespace = H5Dget_space(ds_id);
    if (filespace < 0) {
        fail_task(ffi::ErrorCode::kInternal,
                  "phdf5 async write: H5Dget_space failed");
        return;
    }
    {
        std::string rank_err;
        if (!check_dataset_rank(filespace, rank, "phdf5 async write",
                                nullptr, &rank_err)) {
            rank_err += " ds=" + h5_object_name(ds_id);
            H5Sclose(filespace);
            fail_task(ffi::ErrorCode::kInternal, rank_err);
            return;
        }
    }
    std::vector<hsize_t> extent((size_t)rank, 0);
    std::vector<hsize_t> max_extent((size_t)rank, 0);
    if (H5Sget_simple_extent_dims(filespace, extent.data(), max_extent.data()) < 0) {
        H5Sclose(filespace);
        fail_task(ffi::ErrorCode::kInternal,
                  "phdf5 async write: H5Sget_simple_extent_dims failed");
        return;
    }
    // ── Emptiness is PER-RANK; the bounds test must be RANK-INVARIANT. ──
    //
    // `empty_selection` decides which selection this rank makes.  An empty
    // selection writes nothing and `offset` never reaches HDF5: the empty
    // branch below replaces the hyperslab with H5Sselect_none, which takes
    // no offset.  Two callers arrive here that way BY DESIGN -- a rank whose
    // local block is entirely mu padding (the shard loop above zeroes
    // file_count when local_start >= valid_shape) and a non-canonical
    // replica writer (`replica_dup`, which zeroes every count precisely to
    // keep the collective H5Dwrite rendezvous).  Both still carry an offset
    // advanced by `rank_coord * mem_dims[d]`, measured against a file that
    // stores the LOGICAL extent -- so bounds-testing THAT rejects a write
    // that was always going to be a no-op (commit d935ce7; two 32-node legs).
    //
    // The bounds test itself is now taken on `offset_base + valid_shape`,
    // the LOGICAL slab, instead of on this rank's advanced offset.  Both
    // buffers are replicated control vectors -- identical on every rank by
    // construction -- and max over ranks of (offset[d] + file_count[d]) is
    // exactly offset_base[d] + valid_shape[d], so this accepts and refuses
    // exactly the same calls as the per-rank test.  What changes is WHO
    // refuses: every rank, or none.  The per-rank form reached its verdict
    // only on ranks with a non-empty selection, so a caller overrunning the
    // dataset while some ranks were wholly padding refused on a PROPER
    // SUBSET -- the empty ranks entered the collective H5Dwrite alone and
    // stranded the communicator, the same failure mode d935ce7 fixed, one
    // step further out.  A refusal that fires on a subset of ranks inside a
    // collective is a hang, not an error.
    //
    // A globally empty request (valid_shape[d] == 0 on any dim: nothing is
    // selected on ANY rank) is not bounds-tested at all -- same doctrine.
    bool empty_selection = false;
    for (int d = 0; d < rank; ++d) {
        if (file_count[(size_t)d] == 0) { empty_selection = true; break; }
    }
    bool globally_empty = false;
    for (int d = 0; d < rank; ++d) {
        if (valid_shape[(size_t)d] == 0) { globally_empty = true; break; }
    }
    bool out_of_bounds = false;
    if (!globally_empty) {
        for (int d = 0; d < rank; ++d) {
            if ((hsize_t)(offset_base[(size_t)d] + valid_shape[(size_t)d])
                    > extent[(size_t)d]) {
                out_of_bounds = true;
            }
        }
    }
    bool debug = write_debug_enabled();
    if (debug || out_of_bounds) {
        std::fprintf(stderr,
            "[phdf5-write rank=%d ds=%s id=%lld] extent=%s offset=%s count=%s "
            "mem_dims=%s base=%s valid=%s collective=%d oob=%d empty=%d\n",
            ctx->rank, h5_object_name(ds_id).c_str(), (long long)ds_id,
            vec_to_string(extent).c_str(), vec_to_string(offset).c_str(),
            vec_to_string(file_count).c_str(), vec_to_string(mem_dims).c_str(),
            vec_to_string(offset_base).c_str(),
            vec_to_string(valid_shape).c_str(),
            ctx->use_collective_write ? 1 : 0,
            out_of_bounds ? 1 : 0, empty_selection ? 1 : 0);
        std::fflush(stderr);
    }
    if (out_of_bounds) {
        std::ostringstream os;
        os << "phdf5 async write: logical slab out of bounds"
           << " ds=" << h5_object_name(ds_id)
           << " extent=" << vec_to_string(extent)
           << " offset_base=" << vec_to_string(offset_base)
           << " valid_shape=" << vec_to_string(valid_shape)
           << " (this rank: offset=" << vec_to_string(offset)
           << " count=" << vec_to_string(file_count) << ")"
           << " rank=" << ctx->rank
           << " -- refused identically on every rank";
        H5Sclose(filespace);
        // Already printed above by the debug||oob banner; do not double it.
        promise.SetError(ffi::Error(ffi::ErrorCode::kInternal, os.str()));
        return;
    }
    hid_t memspace = H5Screate_simple(rank, mem_dims.data(), nullptr);
    if (memspace < 0) {
        H5Sclose(filespace);
        fail_task(ffi::ErrorCode::kInternal,
                  "phdf5 async write: H5Screate_simple(memspace) failed");
        return;
    }
    auto t_spaces = now();

    if (empty_selection) {
        if (H5Sselect_none(filespace) < 0 || H5Sselect_none(memspace) < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            fail_task(ffi::ErrorCode::kInternal,
                      "phdf5 async write: H5Sselect_none failed");
            return;
        }
    } else if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                             offset.data(), nullptr,
                             file_count.data(),  nullptr) < 0) {
        if (debug) H5Eprint2(H5E_DEFAULT, stderr);
        H5Sclose(memspace);
        H5Sclose(filespace);
        fail_task(ffi::ErrorCode::kInternal,
                  "phdf5 async write: H5Sselect_hyperslab failed");
        return;
    } else {
        std::vector<hsize_t> mem_start((size_t)rank, 0);
        if (H5Sselect_hyperslab(memspace, H5S_SELECT_SET,
                                mem_start.data(), nullptr,
                                file_count.data(), nullptr) < 0) {
            if (debug) H5Eprint2(H5E_DEFAULT, stderr);
            H5Sclose(memspace);
            H5Sclose(filespace);
            fail_task(ffi::ErrorCode::kInternal,
                      "phdf5 async write: H5Sselect_hyperslab(mem) failed");
            return;
        }
    }

    auto t_select = now();

    hid_t dxpl = ctx->use_collective_write ? ctx->dxpl_coll : ctx->dxpl_indep;
    herr_t st = H5Dwrite(ds_id, native_type, memspace, filespace, dxpl,
                         src_buf);
    if (st < 0 && debug) H5Eprint2(H5E_DEFAULT, stderr);
    auto t_write = now();

    H5Sclose(memspace);
    H5Sclose(filespace);

    if (st < 0) {
        fail_task(ffi::ErrorCode::kInternal,
                  "phdf5 async write: H5Dwrite failed ds=" +
                  h5_object_name(ds_id));
        return;
    }

    // Rank 0 only, like the read side.  ``empty`` is reported because a
    // rank writing an empty selection still enters the collective, and a
    // line showing write=... with empty=1 is the signature of a shard that
    // contributed nothing but paid the barrier.
    if (do_time && ctx->rank == 0) {
        const double wr_ms = ms(t_select, t_write);
        std::fprintf(stderr,
            "[phdf5 write r0] ds=%s bytes/rank=%zu coll=%d empty=%d  "
            "d2h_wait=%.2f  spaces=%.2f  select=%.2f  write=%.2f  "
            "total=%.2f (ms)  %.0f MB/s\n",
            h5_object_name(ds_id).c_str(), bytes,
            ctx->use_collective_write ? 1 : 0, empty_selection ? 1 : 0,
            ms(t0, t_d2h), ms(t_d2h, t_spaces), ms(t_spaces, t_select),
            wr_ms, ms(t0, t_write),
            (wr_ms > 0.0 ? (double)bytes / 1e3 / wr_ms : 0.0));
        std::fflush(stderr);
    }
    promise.SetAvailable();
}

// ---- Dispatch ------------------------------------------------------------
// Padding contract: A.dimensions() is the equal-block physical shard.
// valid_shape is the logical file prefix; ranks past that prefix write
// an empty selection, and the last valid rank writes a clipped prefix.
// NOTE on ``offset_base`` being a Buffer rather than an Attr: offset
// changes per chunk (0, r, 2r, 3r for zeta) and FFI Attrs are baked
// into the XLA compile at dispatch time, so a fresh compile is needed
// for every distinct offset.  At MoS2 3x3 that meant 9 × 900 MiB HLO
// modules compiling just for FFI writes (measured via the profiling
// stack's HLO dump 2026-04-19).  Passing offset as a traced device
// buffer makes the trace signature shape-only (ndim), so shard_map
// closures compile ONCE per (ds_id, ndim, dtype, sharding).
// ``ctx_handle``/``ds_id`` follow offset_base for the SAME reason, one turn
// of the screw further out.  An Attr is baked into the compiled module, and
// ctx_handle is a heap address that differs every process, so the persistent
// compile cache could never hit a SlabIO module and rewrote it on every run.
// MEASURED 2026-08-07, byte-identical workload into a private cache dir:
// jit__per_rank entries 4 -> 8 -> 12 over three runs while a plain jit
// control stayed at 1; the shared np1 cache had accumulated 6813 such
// corpses out of 14443 entries.  Moving ds_id out too collapses the module
// count from O(files x datasets) to O(ndim, dtype, sharding).
static ffi::Future WriteDispatch(
    LRX_STREAM_PARAM
    ffi::AnyBuffer A,
    ffi::Buffer<ffi::DataType::S64> handle_buf,   // shape (2,) {ctx_handle, ds_id}
    ffi::Buffer<ffi::DataType::S64> offset_buf,   // shape (ndim,)
    ffi::Buffer<ffi::DataType::S64> valid_shape_buf, // shape (ndim,)
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> token_out,
    ffi::Span<const int64_t> mesh_shape,
    ffi::Span<const int64_t> axis_count_per_dim,
    ffi::Span<const int64_t> axis_flat)
{
    // The handle pair is replicated across the mesh (the shard_map passes it
    // with ``P()``), so it keeps the every-rank-or-none property the refusals
    // below depend on.
    if (handle_buf.dimensions().size() != 1 ||
        handle_buf.dimensions()[0] != 2) {
        ffi::Promise p;
        ffi::Future f(p);
        p.SetError(ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "phdf5 write: handle_buf must have shape (2,) == "
            "{ctx_handle, ds_id}"));
        return f;
    }
    int64_t handle_host[2] = {0, 0};
    {
        // No ctx yet, so there is nothing to announce through.
        std::string herr;
        if (!copy_index_to_host(handle_host, handle_buf.untyped_data(),
                                2 * sizeof(int64_t), &herr)) {
            ffi::Promise p;
            ffi::Future f(p);
            p.SetError(ffi::Error(
                ffi::ErrorCode::kInternal,
                "phdf5 write: copy(handle) failed: " + herr));
            return f;
        }
    }
    auto* ctx = reinterpret_cast<PhdfCtx*>(handle_host[0]);
    const hid_t ds_id = (hid_t)handle_host[1];

    // Every dispatch-level refusal below happens BEFORE the task is enqueued,
    // so the rank never reaches H5Dwrite.  That is only safe because every
    // input tested here is replicated by construction — the FFI Attrs are
    // baked into the compiled module, the handle buffer is replicated, A's
    // dimensions are the equal-block shard shape, and ds_id/ctx_handle come
    // from collective lifecycle calls — so these refuse on every rank or on
    // none.  See shard_index.h.
    auto fail = [ctx](ffi::ErrorCode code, const std::string& msg) {
        announce_error(ctx, msg);
        ffi::Promise p;
        ffi::Future f(p);
        p.SetError(ffi::Error(code, msg));
        return f;
    };

    if (!ctx) {
        return fail(ffi::ErrorCode::kInvalidArgument,
                    "phdf5 write: ctx_handle is null");
    }

    const auto dims = A.dimensions();
    const size_t N = dims.size();
    if (N == 0) {
        return fail(ffi::ErrorCode::kInvalidArgument,
                    "phdf5 write: operand is 0-D; there is no hyperslab to "
                    "write");
    }
    if (offset_buf.dimensions().size() != 1 ||
        (size_t)offset_buf.dimensions()[0] != N ||
        valid_shape_buf.dimensions().size() != 1 ||
        (size_t)valid_shape_buf.dimensions()[0] != N ||
        axis_count_per_dim.size() != N) {
        std::ostringstream os;
        os << "phdf5 write: rank mismatch  A.ndim=" << N
           << " offset_buf.ndim=" << offset_buf.dimensions().size()
           << " valid_shape_buf.ndim=" << valid_shape_buf.dimensions().size()
           << " axis_count_per_dim.size=" << axis_count_per_dim.size();
        return fail(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    {
        std::string enc_err;
        if (!validate_shard_encoding(ctx, "phdf5 write", mesh_shape,
                                     axis_count_per_dim, axis_flat,
                                     &enc_err)) {
            return fail(ffi::ErrorCode::kInvalidArgument, enc_err);
        }
    }

    // Seam 2 — fetch the small index buffers (N × 8 bytes = 24-40 bytes
    // typically).  CUDA: blocking cudaMemcpy D2H, microseconds.  Host:
    // the XLA buffer is already host-resident, so a plain memcpy.
    std::string idx_err;
    std::vector<int64_t> offset_host(N);
    if (!copy_index_to_host(offset_host.data(), offset_buf.untyped_data(),
                            N * sizeof(int64_t), &idx_err)) {
        return fail(ffi::ErrorCode::kInternal,
                    "phdf5 write: index copy(offset) failed: " + idx_err);
    }
    std::vector<int64_t> valid_shape_host(N);
    if (!copy_index_to_host(valid_shape_host.data(),
                            valid_shape_buf.untyped_data(),
                            N * sizeof(int64_t), &idx_err)) {
        return fail(ffi::ErrorCode::kInternal,
                    "phdf5 write: index copy(valid_shape) failed: " + idx_err);
    }

    std::vector<int64_t> coord = unravel_rank(ctx->rank, mesh_shape);

    // Replica dedup: a mesh axis that shards NO dim of this array is a
    // replica axis — every rank along it holds the same shard and would
    // compute the same hyperslab.  Under collective MPI-IO overlapping
    // selections are undefined behaviour (identical bytes or not);
    // under independent MPI-IO they are redundant writes.  One
    // canonical writer per replica group: the rank with coord 0 on
    // every unconsumed mesh axis.  Everyone else drops to a null
    // selection (they still ENTER the collective H5Dwrite below —
    // collective transfer requires every rank at the call).  Purely
    // rank-local: no communication, unlike the Python host backend's
    // allgather-keyed dedup, because coord + axis_flat determine the
    // duplicate set exactly.  Same env knob as the host backend:
    // LORRAX_PHDF5_DEDUP_REPLICAS=0 disables.
    bool replica_dup = false;
    if (ctx->dedup_replicas) {
        std::vector<char> axis_used(mesh_shape.size(), 0);
        for (size_t i = 0; i < axis_flat.size(); ++i) {
            int64_t ax = axis_flat[i];
            if (ax >= 0 && (size_t)ax < axis_used.size())
                axis_used[(size_t)ax] = 1;
        }
        for (size_t ax = 0; ax < axis_used.size(); ++ax) {
            if (!axis_used[ax] && coord[ax] != 0) { replica_dup = true; break; }
        }
    }

    std::vector<hsize_t> offset(N), file_count(N), mem_dims(N);
    size_t flat_idx = 0;
    for (size_t d = 0; d < N; ++d) {
        if (offset_host[d] < 0 || valid_shape_host[d] < 0) {
            std::ostringstream os;
            os << "phdf5 write: negative offset/valid_shape at dim " << d
               << " offset=" << offset_host[d]
               << " valid_shape=" << valid_shape_host[d];
            return fail(ffi::ErrorCode::kInvalidArgument, os.str());
        }
        if (offset_host[d] > INT64_MAX - valid_shape_host[d]) {
            std::ostringstream os;
            os << "phdf5 write: offset+valid_shape overflows int64 at dim "
               << d << " (" << offset_host[d] << "+"
               << valid_shape_host[d] << "); the bounds test would wrap "
               << "negative and accept an out-of-range slab";
            return fail(ffi::ErrorCode::kInvalidArgument, os.str());
        }
        mem_dims[d] = (hsize_t)dims[d];
        offset[d] = (hsize_t)offset_host[d];
        int64_t na = axis_count_per_dim[d];
        // Dim d is sharded over `na` mesh axes; leftmost is slowest,
        // rightmost has stride=1.  rank_coord = sum_k coord[ax_k] * prod(mesh_shape[ax_{k+1:}]).
        int64_t rank_coord = 0;
        int64_t stride_acc = 1;
        for (int64_t k = na - 1; k >= 0; --k) {
            // flat_idx + k is in range: validate_shard_encoding checked that
            // axis_count_per_dim sums to exactly axis_flat.size().
            int64_t ax = axis_flat[flat_idx + k];
            if (ax < 0 || (size_t)ax >= mesh_shape.size()) {
                std::ostringstream os;
                os << "phdf5 write: bad axis " << ax
                   << " at dim " << d << " axis " << k;
                return fail(ffi::ErrorCode::kInvalidArgument, os.str());
            }
            rank_coord += coord[ax] * stride_acc;
            stride_acc *= mesh_shape[ax];
        }
        int64_t local_start = rank_coord * (int64_t)mem_dims[d];
        if (local_start >= valid_shape_host[d]) {
            file_count[d] = 0;
        } else {
            int64_t remaining = valid_shape_host[d] - local_start;
            file_count[d] = (hsize_t)(
                remaining < (int64_t)mem_dims[d] ? remaining : (int64_t)mem_dims[d]);
        }
        offset[d] += (hsize_t)local_start;
        flat_idx += (size_t)na;
    }
    if (replica_dup) {
        // Not the canonical writer of this replica group: write nothing,
        // but keep the (possibly collective) H5Dwrite rendezvous via the
        // async body's empty-selection path (H5Sselect_none).
        for (size_t d = 0; d < N; ++d) file_count[d] = 0;
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
            return fail(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }

    hid_t dset = (hid_t)ds_id;
    if (dset < 0 || H5Iis_valid(dset) <= 0) {
        return fail(ffi::ErrorCode::kInvalidArgument,
                    "phdf5 write: ds_id is invalid");
    }

    size_t bytes = 0;
    {
        std::string ov_err;
        if (!checked_buffer_bytes(mem_dims, elt_bytes, "phdf5 write",
                                  &bytes, &ov_err)) {
            return fail(ffi::ErrorCode::kInvalidArgument, ov_err);
        }
    }

    // ---- Seam 3: get the local shard to a host pointer H5Dwrite can read.
#ifdef LORRAX_FFI_NO_CUDA
    // Host platform: the XLA buffer already IS host memory.  H5Dwrite reads
    // it IN PLACE — no staging allocation, no copy, nothing to wait on.
    // (Staging here would cost a second full copy of the per-rank ζ shard
    // — 0.75 GB/rank at 606 centroids / P=16 — for zero benefit.)  The
    // pointer stays valid until the Future resolves; see the header note.
    void* src_buf = A.untyped_data();
    const bool wait_for_d2h = false;
    (void)bytes;
#else
    // Reuse ``ctx->pinned_buf``, growing on demand.  Safe because the
    // Python worker thread serializes write_slab calls: the prior
    // H5Dwrite has already completed (buffer released) by the time
    // the next dispatch reaches us.  ``cudaMallocHost`` at 270 MB is
    // ~50-100 ms — a per-call cost we can't afford.
    // NOTE (rank-invariance): this is the one dispatch-level refusal in this
    // handler whose input is NOT replicated — a host-memory OOM is per-rank.
    // A rank that hits it never enters the collective H5Dwrite, so the job
    // hangs rather than reporting.  Closing that needs a cross-rank error
    // rendezvous on a communicator of our own; registered in
    // KNOWN_LORRAX_ISSUES.md, not attempted here.  The announce is what makes
    // it diagnosable in the meantime.  CUDA build only: the host build stages
    // nothing on the write path.
    if (!ensure_pinned(ctx, bytes)) {
        std::ostringstream os;
        os << "phdf5 write: ensure_pinned(" << bytes << ") failed";
        return fail(ffi::ErrorCode::kResourceExhausted, os.str());
    }
    void* src_buf = ctx->pinned_buf;
    const bool wait_for_d2h = true;

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
    cudaError_t ce = cudaMemcpyAsync(src_buf, A.untyped_data(), bytes,
                                     cudaMemcpyDeviceToHost, ctx->stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "phdf5 write: cudaMemcpyAsync D2H failed: "
           << cudaGetErrorString(ce);
        return fail(ffi::ErrorCode::kInternal, os.str());
    }
    cudaEventRecord(ctx->d2h_event, ctx->stream);
#endif  // LORRAX_FFI_NO_CUDA

    // Standard Promise/Future handshake: construct promise, construct
    // future from it (one-shot), then move promise into the task
    // closure.  The task runs on ctx->writer_thread in FIFO order.
    ffi::Promise promise;
    ffi::Future future(promise);

    auto task = [ctx, dset, native_type,
                 offset = std::move(offset),
                 file_count = std::move(file_count),
                 mem_dims = std::move(mem_dims),
                 offset_base = std::move(offset_host),
                 valid_shape = std::move(valid_shape_host),
                 src_buf, wait_for_d2h, bytes,
                 promise = std::move(promise)]() mutable
    {
        async_worker(ctx, dset, native_type,
                     std::move(offset), std::move(file_count),
                     std::move(mem_dims),
                     std::move(offset_base), std::move(valid_shape),
                     src_buf, wait_for_d2h, bytes,
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
// One dispatch body (above), two bindings: the CUDA build emits
// ``PhdfWriteFfi`` with the platform stream as a leading Ctx; the host build
// emits ``PhdfWriteHostFfi`` with none (XLA's CPU runtime calls the handler
// with host buffers on the calling thread).  Both register under the SAME
// jax.ffi target string ``lorrax_phdf5_write`` — see platform_seam.h.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LRX_PHDF_HANDLER(PhdfWrite), lorrax_ffi::phdf5::WriteDispatch,
    xla::ffi::Ffi::Bind()
        LRX_PHDF_STREAM_CTX
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()   // handle {ctx, ds}
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()   // offset_base
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()   // valid_shape
        .Ret<xla::ffi::Buffer<xla::ffi::DataType::S32>>()
        .Attr<xla::ffi::Span<const int64_t>>("mesh_shape")
        .Attr<xla::ffi::Span<const int64_t>>("axis_count_per_dim")
        .Attr<xla::ffi::Span<const int64_t>>("axis_flat"));
