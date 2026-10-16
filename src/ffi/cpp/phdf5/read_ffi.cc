// read_ffi.cc — mirror of write_ffi.cc for H5Dread.  Same N-D block-
// partitioned sharding semantics, same pooled-event + runtime-offset
// lessons from the write-path investigation.
//
// Work shape per call:
//  1. H5Dread (MPI-IO collective) into the ctx's SYNCHRONOUS-READER
//     staging buffer ``ctx->read_buf`` on the XLA executor thread —
//     blocks for the read duration.  (``ctx->pinned_buf`` is the writer
//     thread's; ctx.h OWNERSHIP says why they are two buffers.)
//  2. cudaMemcpyAsync H2D onto ``ctx->stream`` into the XLA-allocated
//     output buffer ``A_out``.
//  3. cudaEventRecord(ctx->h2d_event) so downstream ops on xla_stream
//     wait for the H2D to complete.
//  4. cudaStreamWaitEvent(xla_stream, h2d_event) — sets up the
//     dependency. Large read buffers are retired through the per-file worker
//     after DMA completes; the handler does not wait for that cleanup.
//
// No per-call cudaEventCreate/Destroy: that causes a ~800 ms stall
// on non-rank-0 processes under JAX's cuda_async allocator (measured
// 2026-04-18 on writes; same mechanism applies here).

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// LORRAX_FFI_NO_CUDA (host build): the collective MPI-IO read core below —
// hyperslab selection + H5Dread into the staging buffer — is byte-identical
// on both platforms.  Only three things switch on the flag:
//   1. the handler's leading PlatformStream<cudaStream_t> ctx (absent on the
//      CPU platform, which hands the handler host buffers directly),
//   2. the small offset/count buffer copy (cudaMemcpy D2H vs a plain read of
//      the already-host-resident XLA buffer), and
//   3. the device-staging tail (cudaMemcpyAsync H2D + event vs std::memcpy
//      into the host-resident XLA output buffer).
// The host handlers register under DISTINCT symbol names (PhdfRead*HostFfi)
// so the two platform .so's can co-exist in one process under RTLD_GLOBAL —
// the same split slate/scalapack use.
// Seams 1 (handler binding) and 2 (index copy-in) live in platform_seam.h,
// shared verbatim with write_ffi.cc; seam 3 (the staging tail) is below.
#include "../common/leg.h"
#if LRX_LEG_DEVICE
#include "gpu_runtime.h"
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

// Stage ``bytes`` from the staging buffer ``src`` (where H5Dread landed) into
// the XLA output buffer ``d_dst``.  CUDA: async H2D on ctx->stream, recorded
// on the pooled h2d_event, with xla_stream made to wait so downstream ops
// defer until the copy completes.  Host: d_dst is host memory, so a plain
// memcpy.
//
// ``src`` is a PARAMETER, not ``ctx->pinned_buf``: the synchronous readers
// stage through ``ctx->read_buf`` and only the writer-thread task stages
// through ``ctx->pinned_buf`` (ctx.h OWNERSHIP).  Reading the buffer off the
// ctx here would have quietly re-merged the two.
#if !LRX_LEG_DEVICE
static inline ffi::Error stage_host_to_output(
    PhdfCtx* ctx, const void* src, void* d_dst, size_t bytes)
{
    (void)ctx;
    std::memcpy(d_dst, src, bytes);
    return ffi::Error::Success();
}
#else
static inline ffi::Error stage_host_to_output(
    lrx::gpu::Stream xla_stream, PhdfCtx* ctx, const void* src,
    void* d_dst, size_t bytes)
{
    // ABI 7: the destination may be a donated buffer whose fill is still
    // queued on xla_stream; the copy on ctx->stream must follow it (ctx.h
    // order_event).  Device-side wait, no host sync.
    LORRAX_GPU_CHECK(lrx::gpu::event_record(ctx->order_event, xla_stream));
    LORRAX_GPU_CHECK(lrx::gpu::stream_wait_event(ctx->stream, ctx->order_event, 0));
    LORRAX_GPU_CHECK(lrx::gpu::memcpy_async(d_dst, src, bytes,
                                      lrx::gpu::kHostToDevice, ctx->stream));
    LORRAX_GPU_CHECK(lrx::gpu::event_record(ctx->h2d_event, ctx->stream));
    LORRAX_GPU_CHECK(lrx::gpu::stream_wait_event(xla_stream, ctx->h2d_event, 0));
    return ffi::Error::Success();
}
#endif

// Keep small pinned buffers for repeated reads, but never retain a large
// slab in each of several open file contexts. The threshold bounds idle
// read staging to 32 MiB per file; it does not cap the live H5Dread slab.
// This is the synchronous-read buffer only. The writer thread owns
// pinned_buf, including the async k-chunk union reader.
static constexpr size_t kMaxRetainedReadStaging = size_t{32} << 20;

static inline ffi::Error finish_read(
    LRX_STREAM_PARAM PhdfCtx* ctx, const void* src, void* d_dst, size_t bytes)
{
    FFI_RETURN_IF_ERROR(
        stage_host_to_output(LRX_STREAM_ARG ctx, src, d_dst, bytes));
    if (ctx->read_capacity.load(std::memory_order_acquire) >
        kMaxRetainedReadStaging) retire_read_buf(ctx);
    return ffi::Error::Success();
}

// ``copy_index_to_host`` (seam 2) is in platform_seam.h — shared with
// write_ffi.cc so the two TUs cannot drift.  ``unravel_rank`` /
// ``vec_to_string`` / ``validate_shard_encoding`` / ``checked_buffer_bytes``
// / ``check_dataset_rank`` / ``announce_error`` are in shard_index.h, for the
// same reason.

// The rank's hyperslab of an equal-block sharded output (the padding contract
// above ReadDispatch): ``mem_dims`` is the rank's block, ``offset`` its file
// origin, ``file_count`` the part of it inside ``valid_shape`` (0 on a dim the
// rank holds only padding).  Shared by ReadDispatch and ReadMultiDispatch, so
// the two cannot place a rank differently.  Every refusal is computed from
// replicated operands.
static bool rank_hyperslab(
    const PhdfCtx* ctx, const char* who,
    ffi::Span<const int64_t> mesh_shape,
    ffi::Span<const int64_t> axis_count_per_dim,
    ffi::Span<const int64_t> axis_flat,
    const int64_t* dims, size_t N,
    const int64_t* offset_host, const int64_t* valid_shape_host,
    std::vector<hsize_t>* offset_out, std::vector<hsize_t>* count_out,
    std::vector<hsize_t>* mem_dims_out, std::string* err)
{
    offset_out->assign(N, 0);
    count_out->assign(N, 0);
    mem_dims_out->assign(N, 0);
    std::vector<int64_t> coord = unravel_rank(ctx->rank, mesh_shape);
    auto& offset = *offset_out;
    auto& file_count = *count_out;
    auto& mem_dims = *mem_dims_out;
    size_t flat_idx = 0;
    for (size_t d = 0; d < N; ++d) {
        if (offset_host[d] < 0 || valid_shape_host[d] < 0) {
            std::ostringstream os;
            os << who << ": negative offset/valid_shape at dim " << d
               << " offset=" << offset_host[d]
               << " valid_shape=" << valid_shape_host[d] << "\n"
               << descriptor_forensics(
                      offset_host[d] < 0 ? offset_host[d]
                                         : valid_shape_host[d]);
            *err = os.str();
            return false;
        }
        if (offset_host[d] > INT64_MAX - valid_shape_host[d]) {
            std::ostringstream os;
            os << who << ": offset+valid_shape overflows int64 at dim "
               << d << " (" << offset_host[d] << "+" << valid_shape_host[d]
               << "); the bounds test would wrap negative and accept an "
               << "out-of-range slab";
            *err = os.str();
            return false;
        }
        mem_dims[d] = (hsize_t)dims[d];
        offset[d] = (hsize_t)offset_host[d];
        int64_t na = axis_count_per_dim[d];
        int64_t rank_coord = 0;
        int64_t stride_acc = 1;
        for (int64_t k = na - 1; k >= 0; --k) {
            // In range: validate_shard_encoding checked the length agreement.
            int64_t ax = axis_flat[flat_idx + k];
            if (ax < 0 || (size_t)ax >= mesh_shape.size()) {
                std::ostringstream os;
                os << who << ": bad axis " << ax
                   << " at dim " << d << " axis " << k;
                *err = os.str();
                return false;
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
    return true;
}

// The rank's file/memory selection of one equal-block read, with the
// rank-invariant bounds test first: ReadImpl's one request and every request of
// ReadMultiDispatch take the SAME test in the same order, so a refusal fires on
// every rank or none (the empty-rank rendezvous below is per rank and never
// bounds-tested).  On success the caller owns and closes both dataspaces.
static bool select_rank_slab(
    PhdfCtx* ctx, const char* who, hid_t dset, int rank,
    const std::vector<hsize_t>& offset,
    const std::vector<hsize_t>& file_count,
    const std::vector<hsize_t>& mem_dims,
    const std::vector<int64_t>& offset_base,   // SAME on every rank
    const std::vector<int64_t>& valid_shape,   // SAME on every rank
    hid_t* filespace_out, hid_t* memspace_out,
    ffi::ErrorCode* code, std::string* err)
{
    auto refuse = [&](ffi::ErrorCode c, const std::string& msg) {
        *code = c;
        *err = msg;
        return false;
    };
    const std::string w(who);
    if (dset < 0 || H5Iis_valid(dset) <= 0) {
        return refuse(ffi::ErrorCode::kInvalidArgument, w + ": ds_id is invalid");
    }
    hid_t filespace = H5Dget_space(dset);
    if (filespace < 0) {
        return refuse(ffi::ErrorCode::kInternal, w + ": H5Dget_space failed");
    }
    // Rank agreement, checked BEFORE H5Sget_simple_extent_dims below writes
    // file_rank entries into an rank-sized buffer.  write_ffi.cc has always
    // had this guard; the read path did not, and reading an N-D request
    // against an (N+k)-D dataset would have overrun the extent vector.
    // Rank-invariant (the dataset and the request are the same on every
    // rank), so it refuses everywhere or nowhere.
    {
        std::string rank_err;
        if (!check_dataset_rank(filespace, rank, who,
                                nullptr, &rank_err)) {
            H5Sclose(filespace);
            return refuse(ffi::ErrorCode::kInvalidArgument, rank_err);
        }
    }
    hid_t memspace = H5Screate_simple(rank, mem_dims.data(), nullptr);
    if (memspace < 0) {
        H5Sclose(filespace);
        return refuse(ffi::ErrorCode::kInternal, w + ": H5Screate_simple(memspace) failed");
    }
    // ── Emptiness is PER-RANK; the bounds test is RANK-INVARIANT. ──
    //
    // `empty_selection` picks this rank's selection.  A rank whose local
    // block lies wholly in the mu padding gets file_count == 0 from the
    // shard loop, selects nothing, and still enters the collective H5Dread
    // its peers are inside; its output stays at the memset zeros.  That is
    // the read-side twin of the write path's empty rendezvous and it must
    // never be bounds-tested (write_ffi.cc, commit d935ce7).
    //
    // Until now this handler took NO bounds test at all, which made
    // H5Sselect_hyperslab the de-facto one -- and that fires only on ranks
    // with a non-empty selection.  So a caller whose logical slab overran
    // the dataset while some ranks were wholly padding refused on a PROPER
    // SUBSET: the empty ranks entered the collective H5Dread alone and
    // stranded the communicator.  offset_base and valid_shape are
    // replicated control vectors, identical on every rank, and max over
    // ranks of (offset[d] + file_count[d]) is exactly offset_base[d] +
    // valid_shape[d] -- so testing THOSE reaches the same verdict on every
    // rank, before anyone enters the collective.  A globally empty request
    // (valid_shape[d] == 0 anywhere) is not bounds-tested, same doctrine.
    std::vector<hsize_t> extent((size_t)rank, 0);
    if (H5Sget_simple_extent_dims(filespace, extent.data(), nullptr) < 0) {
        H5Sclose(memspace);
        H5Sclose(filespace);
        return refuse(ffi::ErrorCode::kInternal, w + ": H5Sget_simple_extent_dims failed");
    }
    bool globally_empty = false;
    for (int d = 0; d < rank; ++d) {
        if (valid_shape[(size_t)d] == 0) { globally_empty = true; break; }
    }
    if (!globally_empty) {
        bool out_of_bounds = false;
        for (int d = 0; d < rank; ++d) {
            if ((hsize_t)(offset_base[(size_t)d] + valid_shape[(size_t)d])
                    > extent[(size_t)d]) {
                out_of_bounds = true;
            }
        }
        if (out_of_bounds) {
            std::ostringstream os;
            os << who << ": logical slab out of bounds"
               << " extent=" << vec_to_string(extent)
               << " offset_base=" << vec_to_string(offset_base)
               << " valid_shape=" << vec_to_string(valid_shape)
               << " rank=" << ctx->rank
               << " -- refused identically on every rank";
            // The FIRST out-of-range operand is the one worth decoding.  A
            // value that reads back as a small double is a stale allocation,
            // not arithmetic — see descriptor_forensics.
            for (int d2 = 0; d2 < rank; ++d2) {
                if ((hsize_t)(offset_base[(size_t)d2]
                              + valid_shape[(size_t)d2]) > extent[(size_t)d2]) {
                    os << "\n" << descriptor_forensics(
                        offset_base[(size_t)d2] != 0 ? offset_base[(size_t)d2]
                                                     : valid_shape[(size_t)d2]);
                    break;
                }
            }
            H5Sclose(memspace);
            H5Sclose(filespace);
            return refuse(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
    bool empty_selection = false;
    for (auto c : file_count) {
        if (c == 0) empty_selection = true;
    }
    if (empty_selection) {
        if (H5Sselect_none(filespace) < 0 || H5Sselect_none(memspace) < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            return refuse(ffi::ErrorCode::kInternal, w + ": H5Sselect_none failed");
        }
    } else if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                             offset.data(), nullptr,
                             file_count.data(),  nullptr) < 0) {
        H5Sclose(memspace);
        H5Sclose(filespace);
        return refuse(ffi::ErrorCode::kInternal, w + ": H5Sselect_hyperslab failed");
    } else {
        std::vector<hsize_t> mem_start((size_t)rank, 0);
        if (H5Sselect_hyperslab(memspace, H5S_SELECT_SET,
                                mem_start.data(), nullptr,
                                file_count.data(), nullptr) < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            return refuse(ffi::ErrorCode::kInternal, w + ": H5Sselect_hyperslab(mem) failed");
        }
    }
    *filespace_out = filespace;
    *memspace_out = memspace;
    return true;
}

template <typename T>
static ffi::Error ReadImpl(
    LRX_STREAM_PARAM
    PhdfCtx* ctx,
    T* d_dst,
    const std::vector<hsize_t>& offset,
    const std::vector<hsize_t>& file_count,
    const std::vector<hsize_t>& mem_dims,
    const std::vector<int64_t>& offset_base,   // SAME on every rank
    const std::vector<int64_t>& valid_shape,   // SAME on every rank
    hid_t ds_id)
{
    // Announce before returning: a rank that refuses here does not enter the
    // collective H5Dread its peers are inside, and the message otherwise dies
    // with the process (see shard_index.h announce_error).
    auto fail_read = [ctx](ffi::ErrorCode code, const std::string& msg) {
        announce_error(ctx, msg);
        return ffi::Error(code, msg);
    };

    const int rank = (int)offset.size();
    size_t bytes = 0;
    {
        std::string ov_err;
        if (!checked_buffer_bytes(mem_dims, sizeof(T), "phdf5 read",
                                  &bytes, &ov_err)) {
            return fail_read(ffi::ErrorCode::kInvalidArgument, ov_err);
        }
    }

    // ``read_buf``, NOT ``pinned_buf``: this handler is synchronous and runs
    // on the XLA thread, so it used to race the ctx writer thread on one
    // shared buffer (audit 2026-08-04 item 4).  ctx.h OWNERSHIP has the split.
    if (!ensure_read_buf(ctx, bytes)) {
        std::ostringstream os;
        os << "phdf5 read: staging-buffer alloc of " << bytes << " bytes failed";
        return fail_read(ffi::ErrorCode::kResourceExhausted, os.str());
    }
    void* stage = ctx->read_buf;
    std::memset(stage, 0, bytes);

    hid_t native_type = dt::h5_native_type<T>();
    hid_t filespace = H5I_INVALID_HID, memspace = H5I_INVALID_HID;
    {
        ffi::ErrorCode code = ffi::ErrorCode::kInternal;
        std::string sel_err;
        if (!select_rank_slab(ctx, "phdf5 read", ds_id, rank, offset, file_count,
                              mem_dims, offset_base, valid_shape,
                              &filespace, &memspace, &code, &sel_err)) {
            return fail_read(code, sel_err);
        }
    }

    hid_t dxpl = ctx->use_collective_read ? ctx->dxpl_coll : ctx->dxpl_indep;
    const auto native_t0 = ctx->timing_enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    const auto selected = ctx->timing_enabled
        ? H5Sget_select_npoints(filespace) : 0;
    herr_t st = H5Dread(ds_id, native_type, memspace, filespace, dxpl, stage);
    if (ctx->timing_enabled && st >= 0) {
        ctx->read_calls.fetch_add(1, std::memory_order_relaxed);
        ctx->read_bytes.fetch_add(static_cast<uint64_t>(selected) * sizeof(T),
                                  std::memory_order_relaxed);
        ctx->read_ns.fetch_add(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - native_t0).count()),
            std::memory_order_relaxed);
    }
    H5Sclose(memspace);
    H5Sclose(filespace);
    if (st < 0) {
        return fail_read(ffi::ErrorCode::kInternal,
                         "phdf5 read: H5Dread failed");
    }

    // Device-staging tail (the ONLY hardware-specific step): CUDA async H2D
    // to the XLA output buffer + pooled-event fence so xla_stream defers
    // until the copy completes; host = plain memcpy into the host-resident
    // output.  XLA's FFI contract guarantees the output buffer is available
    // for writing at handler entry, so no cross-stream wait is needed.
    return finish_read(LRX_STREAM_ARG ctx, stage, d_dst, bytes);
}

// Padding contract: A_out dimensions are the equal-block physical
// output shape. valid_shape is the logical file prefix to read; ranks
// outside it get zeros, and edge ranks read clipped hyperslabs.
static ffi::Error ReadDispatch(
    LRX_STREAM_PARAM
    ffi::AnyBuffer dst,                    // the caller's output, aliased to A_out
    ffi::Result<ffi::AnyBuffer> A_out,
    ffi::Span<const int64_t> mesh_shape,
    ffi::Span<const int64_t> axis_count_per_dim,
    ffi::Span<const int64_t> axis_flat)
{
    // Control by POSTED ADDRESS (ctx.h, handler ABI 7): SlabIO allocated
    // ``dst``, aliased it to ``A_out`` and posted {ctx, ds_id, offset[N],
    // valid_shape[N]} under its address.  No control operand, so no device
    // array, no D2H copy-in and no xla_stream sync; the ctx address and ds_id
    // still never enter the HLO (the reason ABI 2 made them runtime values:
    // as Attrs they forked a module per process and per dataset).
    const auto dims = A_out->dimensions();
    const size_t N = dims.size();
    if (N == 0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "phdf5 read: output is 0-D; there is no hyperslab to read");
    }
    if (dst.untyped_data() != A_out->untyped_data()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
            "GATE phdf5_posted_control: phdf5 read: output is not the aliased "
            "destination SlabIO posted for (input_output_aliases not honoured)");
    }
    std::vector<int64_t> control;
    {
        std::string perr;
        if (!take_control(reinterpret_cast<uint64_t>(A_out->untyped_data()),
                          kCtlRead, &control, &perr)) {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, "phdf5 read: " + perr);
        }
    }
    if (control.size() != 2 + 2 * N) {
        std::ostringstream os;
        os << "phdf5 read: posted control has " << control.size()
           << " values; want 2 + 2 ndim = " << 2 + 2 * N;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    auto* ctx = reinterpret_cast<PhdfCtx*>(control[0]);
    const hid_t ds_id = (hid_t)control[1];
    if (!ctx) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "phdf5 read: ctx_handle is null");
    }
    {
        std::string live_err;
        if (!check_live_ctx(ctx, control[0], "phdf5 read", &live_err)) {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, live_err);
        }
    }
    // Every refusal below is computed from replicated values (Attrs, the
    // posted control -- the same on every rank by construction -- and the
    // equal-block output shape), so it fires on every rank or none.
    auto fail = [ctx](ffi::ErrorCode code, const std::string& msg) {
        announce_error(ctx, msg);
        return ffi::Error(code, msg);
    };
    if (axis_count_per_dim.size() != N) {
        std::ostringstream os;
        os << "phdf5 read: rank mismatch  A.ndim=" << N
           << " axis_count_per_dim.size=" << axis_count_per_dim.size();
        return fail(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    {
        std::string enc_err;
        if (!validate_shard_encoding(ctx, "phdf5 read", mesh_shape,
                                     axis_count_per_dim, axis_flat,
                                     &enc_err)) {
            return fail(ffi::ErrorCode::kInvalidArgument, enc_err);
        }
    }
    std::vector<int64_t> offset_host(control.begin() + 2, control.begin() + 2 + N);
    std::vector<int64_t> valid_shape_host(control.begin() + 2 + N, control.end());

    std::vector<hsize_t> offset(N), file_count(N), mem_dims(N);
    {
        std::string hs_err;
        if (!rank_hyperslab(ctx, "phdf5 read", mesh_shape, axis_count_per_dim,
                            axis_flat, dims.begin(), N, offset_host.data(),
                            valid_shape_host.data(), &offset, &file_count,
                            &mem_dims, &hs_err)) {
            return fail(ffi::ErrorCode::kInvalidArgument, hs_err);
        }
    }

    const auto dtype = A_out->element_type();
    switch (dtype) {
        case ffi::DataType::F32:
            return ReadImpl<float>(LRX_STREAM_ARG ctx,
                static_cast<float*>(A_out->untyped_data()),
                offset, file_count, mem_dims,
                offset_host, valid_shape_host, (hid_t)ds_id);
        case ffi::DataType::F64:
            return ReadImpl<double>(LRX_STREAM_ARG ctx,
                static_cast<double*>(A_out->untyped_data()),
                offset, file_count, mem_dims,
                offset_host, valid_shape_host, (hid_t)ds_id);
        case ffi::DataType::S32:
            return ReadImpl<int32_t>(LRX_STREAM_ARG ctx,
                static_cast<int32_t*>(A_out->untyped_data()),
                offset, file_count, mem_dims,
                offset_host, valid_shape_host, (hid_t)ds_id);
        case ffi::DataType::S64:
            return ReadImpl<int64_t>(LRX_STREAM_ARG ctx,
                static_cast<int64_t*>(A_out->untyped_data()),
                offset, file_count, mem_dims,
                offset_host, valid_shape_host, (hid_t)ds_id);
        case ffi::DataType::C64:
            return ReadImpl<std::complex<float>>(LRX_STREAM_ARG ctx,
                static_cast<std::complex<float>*>(A_out->untyped_data()),
                offset, file_count, mem_dims,
                offset_host, valid_shape_host, (hid_t)ds_id);
        case ffi::DataType::C128:
            return ReadImpl<std::complex<double>>(LRX_STREAM_ARG ctx,
                static_cast<std::complex<double>*>(A_out->untyped_data()),
                offset, file_count, mem_dims,
                offset_host, valid_shape_host, (hid_t)ds_id);
        default: {
            std::ostringstream os;
            os << "phdf5 read: unsupported dtype " << static_cast<int>(dtype);
            return fail(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Read-MULTI handler — N datasets of one file, ONE handler invocation, ONE
//  collective ``H5Dread_multi`` (HDF5 >= 1.14; this handler is not compiled
//  against an older library, so ``probe_target`` names the absent symbol).
//
//  Every output is an equal-block shard under the SAME sharding encoding and
//  rank, with its own dataset, dtype, shape, origin and valid extent: exactly
//  N ``ReadDispatch`` requests, placed by the same ``rank_hyperslab`` and
//  selected by the same ``select_rank_slab``, so the bytes that land are the
//  ones N ``read_slab`` calls return.  What it removes is per-request: N
//  dispatches, N descriptor puts and their D2H copy-ins (each a stream sync on
//  a device leg; since ABI 7 every read posts its control instead),
//  N collective rendezvous and N Python ``block_until_ready``.
//
//  The handler is synchronous on the XLA thread, like ReadDispatch: the
//  requests are staged into ONE ``ctx->read_buf`` region (each 256 B aligned),
//  then each output is copied H2D on ``ctx->stream`` behind one pooled event
//  that ``xla_stream`` waits on.  No device memory beyond the N XLA outputs;
//  the host staging is the sum of the N rank blocks (ReadDispatch's was the
//  largest one), retired after the copy above 32 MiB exactly as before.
//
//  Posted control (ctx.h, ABI 7), keyed by output 0's address:
//  {ctx, N, ds_id[N], offset[N][ndim], valid[N][ndim], address[N]}.
// ────────────────────────────────────────────────────────────────────────────
#if H5_VERSION_GE(1, 14, 0)
static bool native_type_of(ffi::DataType t, hid_t* type, size_t* size) {
    switch (t) {
        case ffi::DataType::F32:  *type = dt::h5_native_type<float>();   *size = sizeof(float);   return true;
        case ffi::DataType::F64:  *type = dt::h5_native_type<double>();  *size = sizeof(double);  return true;
        case ffi::DataType::S32:  *type = dt::h5_native_type<int32_t>(); *size = sizeof(int32_t); return true;
        case ffi::DataType::S64:  *type = dt::h5_native_type<int64_t>(); *size = sizeof(int64_t); return true;
        case ffi::DataType::C64:  *type = dt::h5_native_type<std::complex<float>>();
                                  *size = sizeof(std::complex<float>);  return true;
        case ffi::DataType::C128: *type = dt::h5_native_type<std::complex<double>>();
                                  *size = sizeof(std::complex<double>); return true;
        default: return false;
    }
}

static ffi::Error ReadMultiDispatch(
    LRX_STREAM_PARAM
    ffi::RemainingArgs dsts,               // the caller's outputs, aliased to outs
    ffi::RemainingRets outs,
    ffi::Span<const int64_t> mesh_shape,
    ffi::Span<const int64_t> axis_count_per_dim,
    ffi::Span<const int64_t> axis_flat)
{
    static const char* kWho = "phdf5 read_multi";
    const size_t n_out = outs.size();
    const size_t ndim = axis_count_per_dim.size();
    const size_t want = 2 + n_out * (2 + 2 * ndim);
    if (n_out == 0 || ndim == 0 || dsts.size() != n_out) {
        std::ostringstream os;
        os << kWho << ": " << dsts.size() << " destinations for " << n_out
           << " outputs of ndim " << ndim;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    // Control by POSTED ADDRESS (ctx.h, handler ABI 7), keyed by output 0:
    // {ctx, N, ds_id[N], offset[N][ndim], valid[N][ndim], address[N]}.
    std::vector<int64_t> control;
    {
        auto r0 = outs.get<ffi::AnyBuffer>(0);
        if (!r0.has_value()) {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              std::string(kWho) + ": output decode failed");
        }
        std::string perr;
        if (!take_control(reinterpret_cast<uint64_t>(r0.value()->untyped_data()),
                          kCtlReadMulti, &control, &perr)) {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              std::string(kWho) + ": " + perr);
        }
    }
    if (control.size() != want) {
        std::ostringstream os;
        os << kWho << ": posted control has " << control.size()
           << " values; want 2 + N(2 + 2 ndim) = " << want;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    for (size_t i = 0; i < n_out; ++i) {
        auto ri = outs.get<ffi::AnyBuffer>(i);
        auto di = dsts.get<ffi::AnyBuffer>(i);
        const uint64_t posted = (uint64_t)control[2 + n_out * (1 + 2 * ndim) + i];
        if (!ri.has_value() || !di.has_value() ||
            ri.value()->untyped_data() != di.value().untyped_data() ||
            reinterpret_cast<uint64_t>(ri.value()->untyped_data()) != posted) {
            std::ostringstream os;
            os << "GATE phdf5_posted_control: " << kWho << ": output " << i
               << " is not the aliased destination SlabIO posted (0x" << std::hex
               << posted << std::dec << ")";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
    auto* ctx = reinterpret_cast<PhdfCtx*>(control[0]);
    if (!ctx) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          std::string(kWho) + ": ctx_handle is null");
    }
    {
        std::string live_err;
        if (!check_live_ctx(ctx, control[0], kWho, &live_err)) {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, live_err);
        }
    }
    auto fail = [ctx](ffi::ErrorCode code, const std::string& msg) {
        announce_error(ctx, msg);
        return ffi::Error(code, msg);
    };
    if ((size_t)control[1] != n_out) {
        std::ostringstream os;
        os << kWho << ": control names " << control[1] << " requests for "
           << n_out << " outputs";
        return fail(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    {
        std::string enc_err;
        if (!validate_shard_encoding(ctx, kWho, mesh_shape, axis_count_per_dim,
                                     axis_flat, &enc_err)) {
            return fail(ffi::ErrorCode::kInvalidArgument, enc_err);
        }
    }
    const int64_t* ds_ids = control.data() + 2;
    const int64_t* offsets = ds_ids + n_out;
    const int64_t* valids = offsets + n_out * ndim;

    struct Request {
        hid_t ds, type;
        void* dst;
        size_t bytes, stage_off;
        std::vector<hsize_t> offset, count, mem_dims;
    };
    std::vector<Request> req(n_out);
    size_t total = 0;
    for (size_t i = 0; i < n_out; ++i) {
        auto r = outs.get<ffi::AnyBuffer>(i);
        if (!r.has_value()) {
            return fail(ffi::ErrorCode::kInvalidArgument,
                        std::string(kWho) + ": output decode failed");
        }
        ffi::Result<ffi::AnyBuffer> out = r.value();
        const auto dims = out->dimensions();
        if (dims.size() != ndim) {
            std::ostringstream os;
            os << kWho << ": output " << i << " is " << dims.size()
               << "-D; the sharding encoding is " << ndim << "-D";
            return fail(ffi::ErrorCode::kInvalidArgument, os.str());
        }
        size_t elsize = 0;
        if (!native_type_of(out->element_type(), &req[i].type, &elsize)) {
            std::ostringstream os;
            os << kWho << ": output " << i << " unsupported dtype "
               << static_cast<int>(out->element_type());
            return fail(ffi::ErrorCode::kInvalidArgument, os.str());
        }
        std::string err;
        if (!rank_hyperslab(ctx, kWho, mesh_shape, axis_count_per_dim, axis_flat,
                            dims.begin(), ndim, offsets + i * ndim,
                            valids + i * ndim, &req[i].offset, &req[i].count,
                            &req[i].mem_dims, &err) ||
            !checked_buffer_bytes(req[i].mem_dims, elsize, kWho,
                                  &req[i].bytes, &err)) {
            return fail(ffi::ErrorCode::kInvalidArgument, err);
        }
        req[i].ds = (hid_t)ds_ids[i];
        req[i].dst = out->untyped_data();
        req[i].stage_off = total;
        total += (req[i].bytes + 255) & ~size_t{255};
    }
    if (!ensure_read_buf(ctx, total ? total : 1)) {
        std::ostringstream os;
        os << kWho << ": staging-buffer alloc of " << total << " bytes failed";
        return fail(ffi::ErrorCode::kResourceExhausted, os.str());
    }
    char* stage = static_cast<char*>(ctx->read_buf);
    std::memset(stage, 0, total);

    // Every rank lists every request in the same order (shapes are the
    // equal-block shards, identical on every rank); a rank's empty block is
    // an H5S_NONE selection inside the one collective, as in ReadImpl.
    std::vector<hid_t> dsets, types, mems, files;
    std::vector<void*> bufs;
    auto close_all = [&]() {
        for (hid_t m : mems) H5Sclose(m);
        for (hid_t f : files) H5Sclose(f);
    };
    for (size_t i = 0; i < n_out; ++i) {
        if (req[i].bytes == 0) continue;     // a zero-size output on every rank
        std::vector<int64_t> off_base(offsets + i * ndim, offsets + (i + 1) * ndim);
        std::vector<int64_t> valid(valids + i * ndim, valids + (i + 1) * ndim);
        hid_t fs = H5I_INVALID_HID, ms = H5I_INVALID_HID;
        ffi::ErrorCode code = ffi::ErrorCode::kInternal;
        std::string err;
        if (!select_rank_slab(ctx, kWho, req[i].ds, (int)ndim, req[i].offset,
                              req[i].count, req[i].mem_dims, off_base, valid,
                              &fs, &ms, &code, &err)) {
            close_all();
            std::ostringstream os;
            os << err << " (request " << i << " of " << n_out << ")";
            return fail(code, os.str());
        }
        dsets.push_back(req[i].ds);
        types.push_back(req[i].type);
        mems.push_back(ms);
        files.push_back(fs);
        bufs.push_back(stage + req[i].stage_off);
    }
    hid_t dxpl = ctx->use_collective_read ? ctx->dxpl_coll : ctx->dxpl_indep;
    const auto native_t0 = ctx->timing_enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    uint64_t selected = 0;
    if (ctx->timing_enabled) {
        for (size_t j = 0; j < files.size(); ++j) {
            size_t es = H5Tget_size(types[j]);
            selected += (uint64_t)H5Sget_select_npoints(files[j]) * es;
        }
    }
    herr_t st = dsets.empty() ? 0 :
        H5Dread_multi(dsets.size(), dsets.data(), types.data(), mems.data(),
                      files.data(), dxpl, bufs.data());
    if (ctx->timing_enabled && st >= 0) {
        ctx->read_calls.fetch_add(1, std::memory_order_relaxed);
        ctx->read_bytes.fetch_add(selected, std::memory_order_relaxed);
        ctx->read_ns.fetch_add(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - native_t0).count()),
            std::memory_order_relaxed);
    }
    close_all();
    if (st < 0) {
        return fail(ffi::ErrorCode::kInternal, std::string(kWho) + ": H5Dread_multi failed");
    }
    // Device-staging tail: N async copies, ONE event, ONE stream wait.  The
    // copies follow the destinations' fills on xla_stream (ctx.h order_event).
#if LRX_LEG_DEVICE
    LORRAX_GPU_CHECK(lrx::gpu::event_record(ctx->order_event, xla_stream));
    LORRAX_GPU_CHECK(lrx::gpu::stream_wait_event(ctx->stream, ctx->order_event, 0));
    for (size_t i = 0; i < n_out; ++i) {
        if (req[i].bytes == 0) continue;
        LORRAX_GPU_CHECK(lrx::gpu::memcpy_async(req[i].dst, stage + req[i].stage_off,
                                          req[i].bytes, lrx::gpu::kHostToDevice,
                                          ctx->stream));
    }
    LORRAX_GPU_CHECK(lrx::gpu::event_record(ctx->h2d_event, ctx->stream));
    LORRAX_GPU_CHECK(lrx::gpu::stream_wait_event(xla_stream, ctx->h2d_event, 0));
#else
    for (size_t i = 0; i < n_out; ++i) {
        if (req[i].bytes) std::memcpy(req[i].dst, stage + req[i].stage_off, req[i].bytes);
    }
#endif
    if (ctx->read_capacity.load(std::memory_order_acquire) >
        kMaxRetainedReadStaging) retire_read_buf(ctx);
    return ffi::Error::Success();
}
#endif  // H5_VERSION_GE(1, 14, 0)

// ────────────────────────────────────────────────────────────────────────────
//  Read-kchunk handler — read n_kchunk independently-located hyperslab
//  windows of a dataset into a single output with a prepended n_kchunk
//  axis, via ONE handler invocation.
//
//  Use case: a band chunk of a BGW-convention WFN.h5.  ``wfns/coeffs`` is
//  stored as a 4-D ``(nbands, nspinor, ngktot, 2)`` dataset where each
//  k-point's coefficients occupy a contiguous [kpt_starts[k],
//  kpt_starts[k]+ngk[k]) window along dim 2.  To pull a band chunk for a
//  list of k-points — potentially out of order (e.g. the irreducible
//  wedge under symmetry) — we want ONE FFI call that returns an
//  ``(n_kchunk, bands_per_shard, nspinor, ngkmax, 2)`` tensor with bands
//  sharded on a combined mesh axis and n_kchunk replicated.
//
//  Why not ``H5S_SELECT_OR`` (compound hyperslab) for a single H5Dread?
//  When ``ngk`` varies across k-points, an ngkmax-sized slab at kpt_starts[k]
//  overlaps the next k's coefficients by (ngkmax - ngk[k]) elements on
//  dim 2.  HDF5's hyperslab union deduplicates those elements, so the
//  filespace selection has fewer elements than the regular
//  ``(n_kchunk, …)`` memspace, and H5Dread rejects the mismatch.
//  A correct SELECT_OR path would either (a) restrict to uniform-ngk
//  inputs, or (b) use a variable-shape memspace that matches the union.
//  Both defeat the static-shape requirement.  Sequential H5Dreads from
//  inside one handler keep correctness general and still give us the
//  critical property — ONE XLA dispatch per band chunk.  n_kchunk MPI-IO
//  collectives happen inside the handler; all ranks enter them in the
//  same program order, so the collective rendezvous is trivially correct.
// ────────────────────────────────────────────────────────────────────────────

template <typename T>
static ffi::Error ReadKchunkImpl(
    LRX_STREAM_PARAM
    PhdfCtx* ctx,
    T* d_dst,
    const std::vector<std::vector<hsize_t>>& offsets,  // (n_kchunk, N_file)
    const std::vector<hsize_t>& count,                  // (N_file,)
    hid_t ds_id)
{
    const int n_kchunk = (int)offsets.size();
    const int N_file = (int)count.size();
    // Per-phase timing follows the one driver debug-print switch.
    const bool do_time = env_flag("LORRAX_DEBUG_PRINT", false);
    auto now = []() { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t0 = now();

    auto fail_kchunk = [ctx](ffi::ErrorCode code, const std::string& msg) {
        announce_error(ctx, msg);
        return ffi::Error(code, msg);
    };

    if (n_kchunk <= 0) {
        return fail_kchunk(ffi::ErrorCode::kInvalidArgument,
                           "phdf5 read_kchunk: n_kchunk must be >= 1");
    }
    size_t bytes = 0;
    size_t per_k_elts = 1;
    {
        std::string ov_err;
        std::vector<hsize_t> all = count;
        all.push_back((hsize_t)n_kchunk);
        if (!checked_buffer_bytes(count, 1, "phdf5 read_kchunk",
                                  &per_k_elts, &ov_err) ||
            !checked_buffer_bytes(all, sizeof(T), "phdf5 read_kchunk",
                                  &bytes, &ov_err)) {
            return fail_kchunk(ffi::ErrorCode::kInvalidArgument, ov_err);
        }
    }

    // ``read_buf``: this handler returns ffi::Error, i.e. it runs to
    // completion on the XLA thread, so it is a SYNCHRONOUS reader and must
    // not share the writer thread's staging buffer (ctx.h OWNERSHIP).  The
    // audit's item-4 note named only ReadImpl; this is its twin.
    if (!ensure_read_buf(ctx, bytes)) {
        std::ostringstream os;
        os << "phdf5 read_kchunk: staging-buffer alloc of " << bytes << " bytes failed";
        return fail_kchunk(ffi::ErrorCode::kResourceExhausted, os.str());
    }
    void* stage = ctx->read_buf;
    auto t_pin = now();

    hid_t native_type = dt::h5_native_type<T>();
    if (ds_id < 0 || H5Iis_valid(ds_id) <= 0) {
        return fail_kchunk(ffi::ErrorCode::kInvalidArgument,
                           "phdf5 read_kchunk: ds_id is invalid");
    }

    hid_t dxpl = ctx->use_collective_read ? ctx->dxpl_coll : ctx->dxpl_indep;

    hid_t filespace = H5Dget_space(ds_id);
    if (filespace < 0) {
        return fail_kchunk(ffi::ErrorCode::kInternal,
                           "phdf5 read_kchunk: H5Dget_space failed");
    }
    // The sibling of read_ffi's own dataset-rank check.  H5Sselect_hyperslab
    // reads `file_rank` entries from ``offsets[k]`` and ``count``, both of
    // which have exactly N_file; an (N_file+k)-D dataset therefore reads k
    // entries past the end of two vectors, with the values landing in a file
    // offset.  Rank-invariant: same dataset, same request, every rank.
    {
        std::string rank_err;
        if (!check_dataset_rank(filespace, N_file, "phdf5 read_kchunk",
                                nullptr, &rank_err)) {
            H5Sclose(filespace);
            return fail_kchunk(ffi::ErrorCode::kInvalidArgument, rank_err);
        }
    }
    hid_t memspace = H5Screate_simple(N_file, count.data(), nullptr);
    if (memspace < 0) {
        H5Sclose(filespace);
        return fail_kchunk(ffi::ErrorCode::kInternal,
                           "phdf5 read_kchunk: H5Screate_simple(memspace) failed");
    }
    auto t_spaces = now();

    double t_select_total = 0.0, t_read_total = 0.0;
    for (int k = 0; k < n_kchunk; ++k) {
        auto ta = now();
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                 offsets[k].data(), nullptr,
                                 count.data(), nullptr) < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            std::ostringstream os;
            os << "phdf5 read_kchunk: H5Sselect_hyperslab failed at k=" << k;
            return fail_kchunk(ffi::ErrorCode::kInternal, os.str());
        }
        auto tb = now();
        T* k_buf = static_cast<T*>(stage) + (size_t)k * per_k_elts;
        herr_t st = H5Dread(ds_id, native_type, memspace, filespace, dxpl, k_buf);
        auto tc = now();
        if (ctx->timing_enabled && st >= 0) {
            ctx->read_calls.fetch_add(1, std::memory_order_relaxed);
            ctx->read_bytes.fetch_add(per_k_elts * sizeof(T),
                                      std::memory_order_relaxed);
            ctx->read_ns.fetch_add(static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(tc - tb).count()),
                std::memory_order_relaxed);
        }
        t_select_total += ms(ta, tb);
        t_read_total += ms(tb, tc);
        if (st < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            std::ostringstream os;
            os << "phdf5 read_kchunk: H5Dread failed at k=" << k;
            return fail_kchunk(ffi::ErrorCode::kInternal, os.str());
        }
    }
    H5Sclose(memspace);
    H5Sclose(filespace);
    auto t_reads = now();

    FFI_RETURN_IF_ERROR(
        finish_read(LRX_STREAM_ARG ctx, stage, d_dst, bytes));
    auto t_h2d = now();

    if (do_time && ctx->rank == 0) {
        std::fprintf(stderr,
            "[phdf5 kchunk r0] n_k=%d bytes/rank=%zu  "
            "pin=%.2f  spaces=%.2f  select=%.2f  read=%.2f  "
            "(read/k=%.2f)  h2d_setup=%.2f  total=%.2f (ms)\n",
            n_kchunk, bytes,
            ms(t0, t_pin), ms(t_pin, t_spaces),
            t_select_total, t_read_total, t_read_total / n_kchunk,
            ms(t_reads, t_h2d), ms(t0, t_h2d));
    }
    return ffi::Error::Success();
}

static ffi::Error ReadKchunkDispatch(
    LRX_STREAM_PARAM
    ffi::Buffer<ffi::DataType::S64> handle_buf,   // shape (2,) {ctx_handle, ds_id}
    ffi::Buffer<ffi::DataType::S64> offset_buf,   // shape (n_kchunk, N_file)
    ffi::Result<ffi::AnyBuffer> A_out,             // shape (n_kchunk, per-rank file dims...)
    ffi::Span<const int64_t> mesh_shape,
    ffi::Span<const int64_t> axis_count_per_dim,   // length N_file
    ffi::Span<const int64_t> axis_flat,
    int64_t n_kchunk_attr)
{
    // ``ctx_handle``/``ds_id`` arrive as a RUNTIME buffer, not as FFI Attrs —
    // see the long note above ReadDispatch.  Same argument, same fix: as an
    // Attr the heap address forked a module per process, so the persistent
    // compile cache could never hit one.  MEASURED for this handler
    // 2026-08-07: +2 entries per distinct ctx address, attributed by
    // xla_dump to the two union modules.
    if (handle_buf.dimensions().size() != 1 ||
        handle_buf.dimensions()[0] != 2) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "phdf5 read_kchunk: handle_buf must have shape (2,) == "
            "{ctx_handle, ds_id}");
    }
    int64_t handle_host[2] = {0, 0};
    {
        // No ctx yet, so there is nothing to announce through: return
        // plainly.  A failure here hits every rank identically.
        std::string herr;
        if (!copy_index_to_host(LRX_STREAM_ARG handle_host,
                                handle_buf.untyped_data(),
                                2 * sizeof(int64_t), &herr)) {
            return ffi::Error(
                ffi::ErrorCode::kInternal,
                "phdf5 read_kchunk: copy(handle) failed: " + herr);
        }
    }
    auto* ctx = reinterpret_cast<PhdfCtx*>(handle_host[0]);
    const hid_t ds_id = (hid_t)handle_host[1];

    auto fail = [ctx](ffi::ErrorCode code, const std::string& msg) {
        announce_error(ctx, msg);
        return ffi::Error(code, msg);
    };

    if (!ctx) {
        return fail(ffi::ErrorCode::kInvalidArgument,
                    "phdf5 read_kchunk: ctx_handle is null");
    }
    // Audit B2 — see the note in ReadDispatch.
    {
        std::string live_err;
        if (!check_live_ctx(ctx, handle_host[0], "phdf5 read_kchunk",
                            &live_err)) {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, live_err);
        }
    }

    const auto dims = A_out->dimensions();
    if (dims.size() < 2) {
        return fail(ffi::ErrorCode::kInvalidArgument,
            "phdf5 read_kchunk: output must have at least 2 dims "
            "(n_kchunk + file dims)");
    }
    const int64_t n_kchunk = (int64_t)dims[0];
    const size_t N_file = dims.size() - 1;

    if (n_kchunk != n_kchunk_attr) {
        std::ostringstream os;
        os << "phdf5 read_kchunk: dims[0]=" << n_kchunk
           << " != n_kchunk attr=" << n_kchunk_attr;
        return fail(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    if (axis_count_per_dim.size() != N_file) {
        std::ostringstream os;
        os << "phdf5 read_kchunk: axis_count_per_dim.size="
           << axis_count_per_dim.size() << " != N_file=" << N_file;
        return fail(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    const auto obd = offset_buf.dimensions();
    if (obd.size() != 2 || (int64_t)obd[0] != n_kchunk
                       || (size_t)obd[1] != N_file) {
        std::ostringstream os;
        os << "phdf5 read_kchunk: offset_buf shape must be "
           << "(" << n_kchunk << "," << N_file << "); got "
           << "(" << (obd.size() >= 1 ? obd[0] : -1)
           << "," << (obd.size() >= 2 ? obd[1] : -1) << ")";
        return fail(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    {
        std::string enc_err;
        if (!validate_shard_encoding(ctx, "phdf5 read_kchunk", mesh_shape,
                                     axis_count_per_dim, axis_flat,
                                     &enc_err)) {
            return fail(ffi::ErrorCode::kInvalidArgument, enc_err);
        }
    }

    // Bring the small offset buffer (n_kchunk × N_file × 8 bytes) to host.
    std::vector<int64_t> offset_host((size_t)n_kchunk * N_file);
    {
        std::string cperr;
        if (!copy_index_to_host(LRX_STREAM_ARG offset_host.data(),
                                offset_buf.untyped_data(),
                                offset_host.size() * sizeof(int64_t), &cperr)) {
            return fail(ffi::ErrorCode::kInternal,
                        "phdf5 read_kchunk: copy(offset) failed: " + cperr);
        }
    }

    // Derive per-rank coord and per-dim rank_coord (same encoding as
    // the single-offset handler: axis_count_per_dim[d] = number of mesh
    // axes sharding file dim d, axis_flat is the flattened index list).
    std::vector<int64_t> coord = unravel_rank(ctx->rank, mesh_shape);
    std::vector<int64_t> rank_coord_per_dim(N_file, 0);
    {
        size_t flat_idx = 0;
        for (size_t d = 0; d < N_file; ++d) {
            int64_t na = axis_count_per_dim[d];
            int64_t rc = 0;
            int64_t stride = 1;
            for (int64_t k = na - 1; k >= 0; --k) {
                int64_t ax = axis_flat[flat_idx + k];
                if (ax < 0 || (size_t)ax >= mesh_shape.size()) {
                    std::ostringstream os;
                    os << "phdf5 read_kchunk: bad axis " << ax
                       << " at dim " << d << " axis " << k;
                    return fail(ffi::ErrorCode::kInvalidArgument, os.str());
                }
                rc += coord[ax] * stride;
                stride *= mesh_shape[ax];
            }
            rank_coord_per_dim[d] = rc;
            flat_idx += (size_t)na;
        }
    }

    // count = per-rank per-k slab shape = dims[1 .. 1+N_file).
    std::vector<hsize_t> count(N_file);
    for (size_t d = 0; d < N_file; ++d) {
        count[d] = (hsize_t)dims[d + 1];
    }

    // Build per-k file offsets: offset_host[k*N_file + d] is the user's
    // offset_base for this k on file dim d.  Add rank_coord_per_dim[d] *
    // count[d] to get THIS rank's file offset for this k.
    std::vector<std::vector<hsize_t>> offsets(
        (size_t)n_kchunk, std::vector<hsize_t>(N_file, 0));
    for (int64_t k = 0; k < n_kchunk; ++k) {
        for (size_t d = 0; d < N_file; ++d) {
            int64_t off_kd =
                offset_host[(size_t)k * N_file + d] +
                rank_coord_per_dim[d] * (int64_t)count[d];
            if (off_kd < 0) {
                std::ostringstream os;
                os << "phdf5 read_kchunk: negative offset at k=" << k
                   << " d=" << d << " off=" << off_kd;
                return fail(ffi::ErrorCode::kInvalidArgument, os.str());
            }
            offsets[(size_t)k][d] = (hsize_t)off_kd;
        }
    }

    const auto dtype = A_out->element_type();
    switch (dtype) {
        case ffi::DataType::F32:
            return ReadKchunkImpl<float>(LRX_STREAM_ARG ctx,
                static_cast<float*>(A_out->untyped_data()),
                offsets, count, (hid_t)ds_id);
        case ffi::DataType::F64:
            return ReadKchunkImpl<double>(LRX_STREAM_ARG ctx,
                static_cast<double*>(A_out->untyped_data()),
                offsets, count, (hid_t)ds_id);
        case ffi::DataType::S32:
            return ReadKchunkImpl<int32_t>(LRX_STREAM_ARG ctx,
                static_cast<int32_t*>(A_out->untyped_data()),
                offsets, count, (hid_t)ds_id);
        case ffi::DataType::S64:
            return ReadKchunkImpl<int64_t>(LRX_STREAM_ARG ctx,
                static_cast<int64_t*>(A_out->untyped_data()),
                offsets, count, (hid_t)ds_id);
        case ffi::DataType::C64:
            return ReadKchunkImpl<std::complex<float>>(LRX_STREAM_ARG ctx,
                static_cast<std::complex<float>*>(A_out->untyped_data()),
                offsets, count, (hid_t)ds_id);
        case ffi::DataType::C128:
            return ReadKchunkImpl<std::complex<double>>(LRX_STREAM_ARG ctx,
                static_cast<std::complex<double>*>(A_out->untyped_data()),
                offsets, count, (hid_t)ds_id);
        default: {
            std::ostringstream os;
            os << "phdf5 read_kchunk: unsupported dtype " << static_cast<int>(dtype);
            return fail(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
//  Read-kchunk-UNION handler — pulls n_kchunk non-overlapping hyperslabs
//  of a dataset via H5S_SELECT_OR compound selections, one INDEPENDENT
//  H5Dread per chunk of output rows (the worker below says why).  Output
//  buffer is (n_kchunk, *per_rank_max_shape); for each k, the FIRST
//  count[k][d] cells of dim d in that k's output slot get filled from file
//  offset offset[k][d], and the remaining padding cells are zero.
//
//  Caller must guarantee the per-k (offset, count) file slabs are
//  pairwise disjoint; otherwise HDF5's union-of-hyperslabs deduplicates
//  and the element counts mismatch the memspace.  For WFN.h5-style
//  data, passing count[k] = (bpr, ns, ngk[k], 2) (the *actual* ngk, not
//  ngkmax) gives disjoint per-k slabs even when ngk varies.
// ────────────────────────────────────────────────────────────────────────────

// Row-major element runs ``emit(offset, length)`` of a union of boxes in a
// C-order array of ``shape``.  Box b sits at index ``off[b][wax]`` of the
// window axis ``wax`` (ascending in b, one index each).  Row-major order over
// the union is: every outer index (dims before ``wax``) in order, then every
// box that contains it in order, then that box's inner block.  This is the
// order HDF5 walks the same memory selection in, so it is the order a packed
// H5Dread fills.  (HDF5's own H5Ssel_iter_get_seq_list does the same walk at
// ~1 ms per run on an OR'd selection; VI3 P4, 2850 runs: 2.75 s.)
template <typename Emit>
static void for_each_union_run(const std::vector<hsize_t>& shape, int wax,
                               const std::vector<std::vector<hsize_t>>& off,
                               const std::vector<std::vector<hsize_t>>& cnt,
                               Emit&& emit)
{
    const int N = (int)shape.size();
    if (off.empty()) return;
    std::vector<hsize_t> stride((size_t)N, 1);
    for (int d = N - 2; d >= 0; --d) stride[d] = stride[d + 1] * shape[d + 1];
    std::vector<hsize_t> lo((size_t)wax), hi((size_t)wax), o((size_t)wax);
    for (int d = 0; d < wax; ++d) {
        lo[d] = off[0][d];
        hi[d] = off[0][d] + cnt[0][d];
        for (size_t b = 1; b < off.size(); ++b) {
            lo[d] = std::min(lo[d], off[b][d]);
            hi[d] = std::max(hi[d], off[b][d] + cnt[b][d]);
        }
        o[d] = lo[d];
    }
    std::vector<hsize_t> j((size_t)N, 0);
    for (;;) {
        hsize_t base = 0;
        for (int d = 0; d < wax; ++d) base += o[d] * stride[d];
        for (size_t b = 0; b < off.size(); ++b) {
            bool in = true;
            for (int d = 0; d < wax && in; ++d)
                in = o[d] >= off[b][d] && o[d] < off[b][d] + cnt[b][d];
            if (!in) continue;
            const hsize_t kbase = base + off[b][wax] * stride[wax];
            if (wax == N - 1) { emit(kbase, (hsize_t)1); continue; }
            // Innermost partial dim t: dims past it are whole rows.
            int t = wax + 1;
            for (int d = N - 1; d > wax; --d)
                if (cnt[b][d] != shape[d] || off[b][d] != 0) { t = d; break; }
            const hsize_t run = cnt[b][t] * stride[t];
            for (int d = wax + 1; d < t; ++d) j[d] = off[b][d];
            for (;;) {
                hsize_t at = kbase + off[b][t] * stride[t];
                for (int d = wax + 1; d < t; ++d) at += j[d] * stride[d];
                emit(at, run);
                int d = t - 1;
                for (; d > wax; --d) {
                    if (++j[d] < off[b][d] + cnt[b][d]) break;
                    j[d] = off[b][d];
                }
                if (d == wax) break;
            }
        }
        int d = wax - 1;
        for (; d >= 0; --d) {
            if (++o[d] < hi[d]) break;
            o[d] = lo[d];
        }
        if (d < 0) break;
    }
}

// Mirrors write_ffi.cc's async_worker.  Runs on ``ctx->writer_thread``;
// picks up the next queued read task, does the H5Dread + async H2D, and
// fires the caller's Promise.  Shares the single ``ctx->writer_thread``
// and ``ctx->task_queue`` with the async-write path since tasks FIFO-
// drain through one C++ thread — either writes or reads queue up in
// dispatch order.
//
// Observed end-to-end impact on MoS2 3×3 and Si pseudobands at 4×A100:
// ~1% faster than the equivalent sync path.  This was smaller than
// projected because JAX's default async dispatch already hides most
// of the read latency on xla_stream — the handler invocation is
// microseconds, and downstream xla_stream compute implicitly waits
// only as long as the worker-thread H5Dread takes anyway.  The
// explicit ``ffi::Future`` still gives XLA permission to schedule
// unrelated work concurrently with the H5Dread and mirrors the
// write-path design for consistency, and the benefit is expected to
// grow at TB-scale WFN reads (100s of ms per band-chunk) — but at the
// scales we've measured so far it's a small perf delta.  See
// ``common/phdf5_async_overlap_bench.py`` for the measurement.
//
// This is the ONE user of ``ctx->pinned_buf`` on the host build (ctx.h
// OWNERSHIP): the synchronous readers stage through ``ctx->read_buf``, so
// nothing on another thread can be inside this buffer here.  The task uses
// it as a two-chunk ring.  Its reuse is safe because:
//   1. Tasks run strictly sequentially on the writer thread.
//   2. The task *starts* by ``cudaEventSynchronize(h2d_event)`` which
//      blocks until the PREVIOUS task's async H2D has drained.
//   3. Inside the task, chunk c's copies are issued only after the event
//      recorded behind chunk c-1's copies has completed, so the H5Dread of
//      chunk c+1 never fills the half that a copy is still reading.
//   4. At the end the task records ``h2d_event`` again and syncs
//      before firing the Promise, so when downstream XLA reads
//      ``d_dst`` the H2D is already complete.
static void async_read_kchunk_union_worker(
    PhdfCtx* ctx,
    hid_t ds_id,
    hid_t native_type,
    void* d_dst,
    size_t element_size,
    std::vector<std::vector<hsize_t>> file_offsets,
    std::vector<std::vector<hsize_t>> file_counts,
    std::vector<hsize_t> per_rank_max_shape,
    std::vector<char> sharded_dim,
    int kchunk_axis,
    ffi::Promise promise)
{
    const int n_kchunk = (int)file_offsets.size();
    const int N_file = (int)per_rank_max_shape.size();
    const int N_out = N_file + 1;
    const bool do_time = env_flag("LORRAX_DEBUG_PRINT", false);
    auto now = []() { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t0 = now();

    // Announce before setting the Promise: this task runs after the caller
    // has returned, so an error here reaches Python only if the process
    // survives to report it, and the failure mode under test is that it does
    // not (shard_index.h announce_error).
    auto fail_task = [&](ffi::ErrorCode code, const std::string& msg) {
        announce_error(ctx, msg);
        promise.SetError(ffi::Error(code, msg));
    };

    // Block until the previous task's async H2D has finished reading
    // from ``pinned_buf`` — guards the pinned-buffer reuse.  On the host
    // build the staging tail is a synchronous std::memcpy that fully
    // drains before the worker picks up the next FIFO task, so the buffer
    // is already free and there is nothing to wait on.
#if LRX_LEG_DEVICE
    (void)lrx::gpu::event_synchronize(ctx->h2d_event);
#endif
    auto t_prev_h2d_done = now();

    if (n_kchunk <= 0 || N_file <= 0) {
        fail_task(ffi::ErrorCode::kInvalidArgument,
                  "phdf5 read_kchunk_union: n_kchunk and N_file must be >= 1");
        return;
    }
    if (ds_id < 0 || H5Iis_valid(ds_id) <= 0) {
        // ``ReadImpl`` and ``ReadKchunkImpl`` have always checked this; the
        // union path did not, and H5Dget_space on a stale hid_t is where a
        // closed-then-reused handle turns into an HDF5 error stack instead of
        // a sentence.
        fail_task(ffi::ErrorCode::kInvalidArgument,
                  "phdf5 read_kchunk_union: ds_id is invalid");
        return;
    }

    size_t bytes = 0;
    {
        std::string ov_err;
        std::vector<hsize_t> all = per_rank_max_shape;
        all.push_back((hsize_t)n_kchunk);
        if (!checked_buffer_bytes(all, element_size,
                                  "phdf5 read_kchunk_union", &bytes,
                                  &ov_err)) {
            fail_task(ffi::ErrorCode::kInvalidArgument, ov_err);
            return;
        }
    }

    std::vector<hsize_t> mem_shape(N_out);
    {
        int fi = 0;
        for (int d = 0; d < N_out; ++d) {
            if (d == kchunk_axis) {
                mem_shape[d] = (hsize_t)n_kchunk;
            } else {
                mem_shape[d] = per_rank_max_shape[fi++];
            }
        }
    }

    // The read is cut into chunks of output rows (dim 0) and each chunk goes
    // file -> PACKED pinned staging -> its places in the output:
    //
    //  * The memory side of the H5Dread is one contiguous run (a 1-D space of
    //    the chunk's point count).  HDF5 maps the n-th selected file element
    //    to the n-th selected memory element, both in row-major order, so
    //    reading packed and then walking the memory selection's own
    //    (offset, length) runs in order puts every element exactly where the
    //    one-shot union H5Dread put it (the caller's windows are disjoint and
    //    ascending, and rows are the outermost dim of both orders).
    //  * A WFN band block (each window whole along the dims after the one
    //    the windows vary in) is read INDEPENDENTLY: its file runs are long,
    //    and two-phase aggregation only re-shipped every byte through 16 MiB
    //    collective-buffer rounds.  VI3 12x12 P16, bands 0:360, 120.2 GB
    //    (runs/runtime/p2d_io_20260924): collective union 56.8 s cold /
    //    44.4 s warm; independent 17.8 / 7.6 s; the contiguous-memory
    //    collective read 21.4 / 15.7 s.  Short strided windows (bank
    //    rectangles) stay collective; there every rank enters every chunk's
    //    H5Dread, since the chunk count is the same on every rank.
    //  * Staging is two packed chunks of pinned memory plus, on CUDA, one
    //    padded chunk (<= 3 x max(row, 256 MiB)), not the whole per-rank
    //    slab: the old path pinned and zeroed ~psi/P per call (4.1 s of the
    //    56.8 s above) and kept it pinned.  Chunk c+1 is read while chunk c's
    //    copies drain.  Long runs go to the device one DMA each; short runs
    //    are placed in the zeroed padded chunk on the host and sent in one
    //    DMA.  Pad cells are zeroed on the output itself.
    constexpr size_t kUnionChunkBytes = size_t{256} << 20;
    const hsize_t n_rows = mem_shape[0];
    size_t row_bytes = element_size;
    for (int d = 1; d < N_out; ++d) row_bytes *= (size_t)mem_shape[d];
    hsize_t rows_per_chunk = (hsize_t)std::max<size_t>(
        1, kUnionChunkBytes / std::max<size_t>(row_bytes, 1));
    if (rows_per_chunk > n_rows) rows_per_chunk = std::max<hsize_t>(n_rows, 1);
    const size_t chunk_cap = (size_t)rows_per_chunk * row_bytes;
    // Output dim 0 is the window axis when kchunk_axis == 0; otherwise it is
    // file dim 0 (the per-rank shape keeps file order around the window axis).
    const bool rows_are_windows = (kchunk_axis == 0);

#if !LRX_LEG_DEVICE
    constexpr int kStagingChunks = 2;
#else
    constexpr int kStagingChunks = 3;   // + one padded chunk for short runs
#endif
    if (!ensure_pinned(ctx, kStagingChunks * chunk_cap)) {
        std::ostringstream os;
        os << "phdf5 read_kchunk_union: staging-buffer alloc of "
           << kStagingChunks * chunk_cap << " bytes failed";
        fail_task(ffi::ErrorCode::kResourceExhausted, os.str());
        return;
    }
    char* ring[2] = {static_cast<char*>(ctx->pinned_buf),
                     static_cast<char*>(ctx->pinned_buf) + chunk_cap};
#if !LRX_LEG_DEVICE
    std::memset(d_dst, 0, bytes);
#else
    {
        lrx::gpu::Error ce = lrx::gpu::memset_async(d_dst, 0, bytes, ctx->stream);
        if (ce != lrx::gpu::kSuccess) {
            fail_task(ffi::ErrorCode::kInternal,
                std::string("phdf5 read_kchunk_union: " LRX_GPU_API "MemsetAsync: ") +
                lrx::gpu::error_string(ce));
            return;
        }
    }
#endif
    auto t_pin = now();

    hid_t memspace = H5Screate_simple(N_out, mem_shape.data(), nullptr);
    if (memspace < 0) {
        fail_task(ffi::ErrorCode::kInternal,
            "phdf5 read_kchunk_union: H5Screate_simple(memspace) failed");
        return;
    }
    hid_t filespace = H5Dget_space(ds_id);
    if (filespace < 0) {
        H5Sclose(memspace);
        fail_task(ffi::ErrorCode::kInternal,
            "phdf5 read_kchunk_union: H5Dget_space failed");
        return;
    }
    // Third sibling of read_ffi's dataset-rank check: the per-k
    // ``file_offsets[k]`` / ``file_counts[k]`` vectors have exactly N_file
    // entries and H5Sselect_hyperslab reads `file_rank` of them.
    {
        std::string rank_err;
        if (!check_dataset_rank(filespace, N_file,
                                "phdf5 read_kchunk_union", nullptr,
                                &rank_err)) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            fail_task(ffi::ErrorCode::kInvalidArgument, rank_err);
            return;
        }
    }
    auto fail_spaces = [&](ffi::ErrorCode code, const std::string& msg) {
        H5Sclose(memspace);
        H5Sclose(filespace);
        fail_task(code, msg);
    };
    std::vector<hsize_t> extent(N_file, 0);
    if (H5Sget_simple_extent_dims(filespace, extent.data(), nullptr) < 0) {
        fail_spaces(ffi::ErrorCode::kInternal,
            "phdf5 read_kchunk_union: H5Sget_simple_extent_dims failed");
        return;
    }
    // Transfer mode, the same verdict on every rank (it reads only the
    // extents and the counts of unsharded dims): INDEPENDENT when each
    // window is whole along every dim after the one the windows vary in, so
    // a window row is one long contiguous file run (a WFN band block);
    // otherwise collective as configured, because short strided pieces (a
    // bank rectangle) are what two-phase aggregation is for.
    bool independent = kchunk_axis < N_out - 1;
    for (int k = 0; k < n_kchunk && independent; ++k) {
        for (int d = kchunk_axis + 1; d < N_file; ++d) {
            const hsize_t c = sharded_dim[d] ? per_rank_max_shape[d]
                                             : file_counts[k][d];
            if (c != extent[d]) { independent = false; break; }
        }
    }
    const hid_t dxpl = independent ? ctx->dxpl_indep
        : (ctx->use_collective_read ? ctx->dxpl_coll : ctx->dxpl_indep);
    const bool collective = (dxpl == ctx->dxpl_coll);

    std::vector<hsize_t> f_off(N_file), f_cnt(N_file);
    std::vector<hsize_t> m_off(N_out), m_cnt(N_out);
    std::vector<std::vector<hsize_t>> box_off, box_cnt;
    std::vector<std::pair<size_t, size_t>> runs;
    size_t n_padded = 0;
    hsize_t npts_total = 0;
    size_t n_chunks = 0;
    double sel_ms = 0.0, read_ms = 0.0, h2d_ms = 0.0;
    size_t n_runs = 0;
    bool copies_in_flight = false;
    for (hsize_t r0 = 0; r0 < n_rows; r0 += rows_per_chunk, ++n_chunks) {
        const hsize_t r1 = std::min<hsize_t>(n_rows, r0 + rows_per_chunk);
        auto t_c0 = now();
        H5Sselect_none(filespace);
        H5Sselect_none(memspace);
        box_off.clear();
        box_cnt.clear();
        for (int k = 0; k < n_kchunk; ++k) {
            int fi = 0;
            for (int d = 0; d < N_out; ++d) {
                if (d == kchunk_axis) {
                    m_off[d] = (hsize_t)k;
                    m_cnt[d] = 1;
                } else {
                    m_off[d] = 0;
                    m_cnt[d] = file_counts[k][fi++];
                }
            }
            const hsize_t lo = std::max<hsize_t>(m_off[0], r0);
            const hsize_t hi = std::min<hsize_t>(m_off[0] + m_cnt[0], r1);
            if (hi <= lo) continue;
            f_off = file_offsets[k];
            f_cnt = file_counts[k];
            if (!rows_are_windows) {
                f_off[0] += lo - m_off[0];
                f_cnt[0] = hi - lo;
            }
            m_off[0] = lo;
            m_cnt[0] = hi - lo;
            bool empty = false;
            for (auto c : f_cnt) if (c == 0) empty = true;
            if (empty) continue;
            if (H5Sselect_hyperslab(filespace, H5S_SELECT_OR, f_off.data(),
                                    nullptr, f_cnt.data(), nullptr) < 0) {
                std::ostringstream os;
                os << "phdf5 read_kchunk_union: file H5Sselect_hyperslab(OR) "
                   << "failed at k=" << k;
                fail_spaces(ffi::ErrorCode::kInternal, os.str());
                return;
            }
            if (H5Sselect_hyperslab(memspace, H5S_SELECT_OR, m_off.data(),
                                    nullptr, m_cnt.data(), nullptr) < 0) {
                std::ostringstream os;
                os << "phdf5 read_kchunk_union: mem H5Sselect_hyperslab(OR) "
                   << "failed at k=" << k;
                fail_spaces(ffi::ErrorCode::kInternal, os.str());
                return;
            }
            box_off.push_back(m_off);
            box_cnt.push_back(m_cnt);
        }
        const hssize_t npts_file = H5Sget_select_npoints(filespace);
        const hssize_t npts_mem  = H5Sget_select_npoints(memspace);
        if (npts_file != npts_mem || npts_file < 0) {
            std::ostringstream os;
            os << "phdf5 read_kchunk_union: selection size mismatch — "
               << "filespace=" << npts_file << " memspace=" << npts_mem
               << " (overlapping per-k slabs?)";
            fail_spaces(ffi::ErrorCode::kInvalidArgument, os.str());
            return;
        }
        const size_t chunk_bytes = (size_t)npts_file * element_size;
        if (chunk_bytes > chunk_cap) {
            std::ostringstream os;
            os << "phdf5 read_kchunk_union: chunk of " << chunk_bytes
               << " bytes exceeds its staging (" << chunk_cap << ")";
            fail_spaces(ffi::ErrorCode::kInternal, os.str());
            return;
        }
        char* buf = ring[n_chunks & 1];
        auto t_c1 = now();
        // Collective: every rank enters every chunk's H5Dread (the chunk count
        // is the same everywhere), an empty rank with an empty selection.
        if (npts_file > 0 || collective) {
            const hsize_t n1 = std::max<hsize_t>((hsize_t)npts_file, 1);
            hid_t packed = H5Screate_simple(1, &n1, nullptr);
            if (packed < 0 || (npts_file == 0 && H5Sselect_none(packed) < 0)) {
                if (packed >= 0) H5Sclose(packed);
                fail_spaces(ffi::ErrorCode::kInternal,
                    "phdf5 read_kchunk_union: packed memory space failed");
                return;
            }
            herr_t st = H5Dread(ds_id, native_type, packed, filespace, dxpl,
                                buf);
            H5Sclose(packed);
            if (st < 0) {
                fail_spaces(ffi::ErrorCode::kInternal,
                            "phdf5 read_kchunk_union: H5Dread failed");
                return;
            }
        }
        auto t_c2 = now();
#if LRX_LEG_DEVICE
        // The previous chunk's copies read the OTHER ring half (or the padded
        // chunk), which this chunk is about to reuse.
        if (copies_in_flight) (void)lrx::gpu::event_synchronize(ctx->h2d_event);
#endif
        if (npts_file > 0) {
            // Adjacent runs merge.
            runs.clear();
            for_each_union_run(mem_shape, kchunk_axis, box_off, box_cnt,
                               [&](hsize_t at, hsize_t len) {
                const size_t a = (size_t)at * element_size;
                const size_t l = (size_t)len * element_size;
                if (!runs.empty() && runs.back().first + runs.back().second == a)
                    runs.back().second += l;
                else
                    runs.emplace_back(a, l);
            });
            size_t cursor = 0;
            for (const auto& r : runs) cursor += r.second;
            if (cursor != chunk_bytes) {
                std::ostringstream os;
                os << "phdf5 read_kchunk_union: runs cover " << cursor
                   << " bytes of a " << chunk_bytes << "-byte chunk";
                fail_spaces(ffi::ErrorCode::kInternal, os.str());
                return;
            }
            n_runs += runs.size();
            cursor = 0;
#if !LRX_LEG_DEVICE
            for (const auto& r : runs) {
                std::memcpy(static_cast<char*>(d_dst) + r.first, buf + cursor,
                            r.second);
                cursor += r.second;
            }
#else
            // Long runs: one DMA each, straight from the packed chunk.  Short
            // runs (mean < 64 KiB, e.g. bank rectangles): place them in a
            // zeroed padded copy of the chunk on the host, then one DMA.
            lrx::gpu::Error ce = lrx::gpu::kSuccess;
            if (chunk_bytes >= runs.size() * (size_t{64} << 10)) {
                for (const auto& r : runs) {
                    if (ce == lrx::gpu::kSuccess)
                        ce = lrx::gpu::memcpy_async(
                            static_cast<char*>(d_dst) + r.first, buf + cursor,
                            r.second, lrx::gpu::kHostToDevice, ctx->stream);
                    cursor += r.second;
                }
            } else {
                char* pad = static_cast<char*>(ctx->pinned_buf) + 2 * chunk_cap;
                const size_t base = (size_t)r0 * row_bytes;
                const size_t span = (size_t)(r1 - r0) * row_bytes;
                std::memset(pad, 0, span);
                for (const auto& r : runs) {
                    std::memcpy(pad + (r.first - base), buf + cursor, r.second);
                    cursor += r.second;
                }
                ++n_padded;
                ce = lrx::gpu::memcpy_async(static_cast<char*>(d_dst) + base, pad,
                                     span, lrx::gpu::kHostToDevice, ctx->stream);
            }
            if (ce != lrx::gpu::kSuccess) {
                fail_spaces(ffi::ErrorCode::kInternal,
                    std::string("phdf5 read_kchunk_union: " LRX_GPU_API "MemcpyAsync: ") +
                    lrx::gpu::error_string(ce));
                return;
            }
            (void)lrx::gpu::event_record(ctx->h2d_event, ctx->stream);
            copies_in_flight = true;
#endif
        }
        auto t_c3 = now();
        npts_total += (hsize_t)npts_file;
        sel_ms += ms(t_c0, t_c1);
        read_ms += ms(t_c1, t_c2);
        h2d_ms += ms(t_c2, t_c3);
    }
    H5Sclose(memspace);
    H5Sclose(filespace);
    if (ctx->timing_enabled) {
        ctx->read_calls.fetch_add(1, std::memory_order_relaxed);
        ctx->read_bytes.fetch_add(static_cast<uint64_t>(npts_total) * element_size,
                                  std::memory_order_relaxed);
        ctx->read_ns.fetch_add(static_cast<uint64_t>(read_ms * 1.0e6),
                               std::memory_order_relaxed);
    }

    // The Promise fires only once every copy (and the pad memset) is on the
    // device, so downstream XLA ops read a complete ``d_dst``.
#if LRX_LEG_DEVICE
    (void)lrx::gpu::event_record(ctx->h2d_event, ctx->stream);
    (void)lrx::gpu::event_synchronize(ctx->h2d_event);
#endif
    auto t_h2d = now();

    if (do_time && ctx->rank == 0) {
        std::fprintf(stderr,
            "[phdf5 kchunk-union r0] n_k=%d bytes/rank=%zu chunks=%zu "
            "rows/chunk=%llu staging=%zu  prev_h2d_wait=%.2f  zero=%.2f  "
            "select=%.2f  read=%.2f  scatter=%.2f (runs=%zu, padded chunks=%zu)"
            "  total=%.2f (ms)  [selected_elts=%lld, %s]\n",
            n_kchunk, bytes, n_chunks, (unsigned long long)rows_per_chunk,
            (size_t)kStagingChunks * chunk_cap,
            ms(t0, t_prev_h2d_done),
            ms(t_prev_h2d_done, t_pin),
            sel_ms, read_ms, h2d_ms, n_runs, n_padded,
            ms(t0, t_h2d),
            (long long)npts_total, collective ? "collective" : "independent");
    }
    promise.SetAvailable();
}

static ffi::Future ReadKchunkUnionDispatch(
    LRX_STREAM_PARAM
    ffi::Buffer<ffi::DataType::S64> handle_buf,   // (2,) {ctx_handle, ds_id}
    ffi::Buffer<ffi::DataType::S64> offset_buf,   // (n_kchunk, N_file)
    ffi::Buffer<ffi::DataType::S64> count_buf,    // (n_kchunk, N_file)
    ffi::Result<ffi::AnyBuffer> A_out,
    ffi::Span<const int64_t> mesh_shape,
    ffi::Span<const int64_t> axis_count_per_dim,
    ffi::Span<const int64_t> axis_flat,
    int64_t n_kchunk_attr,
    int64_t kchunk_axis_attr)  // position of the n_kchunk axis in A_out
{
    // ``ctx_handle``/``ds_id`` as a RUNTIME buffer — see ReadDispatch.  This
    // handler is the one xla_dump attributed the measured +2-entries-per-
    // ctx-address growth to.
    //
    // NOTE: this dispatch returns ffi::Future, so a failure BEFORE ``ctx`` is
    // known cannot return a bare ffi::Error — it must still hand back a
    // Future carrying the error.  ``fail`` below needs ctx (it announces), so
    // the two pre-ctx failures build their own Promise/Future pair.
    auto fail_early = [](ffi::ErrorCode code, const std::string& msg) {
        ffi::Promise p;
        ffi::Future f(p);
        p.SetError(ffi::Error(code, msg));
        return f;
    };
    if (handle_buf.dimensions().size() != 1 ||
        handle_buf.dimensions()[0] != 2) {
        return fail_early(
            ffi::ErrorCode::kInvalidArgument,
            "phdf5 read_kchunk_union: handle_buf must have shape (2,) == "
            "{ctx_handle, ds_id}");
    }
    int64_t handle_host[2] = {0, 0};
    {
        std::string herr;
        if (!copy_index_to_host(LRX_STREAM_ARG handle_host,
                                handle_buf.untyped_data(),
                                2 * sizeof(int64_t), &herr)) {
            return fail_early(
                ffi::ErrorCode::kInternal,
                "phdf5 read_kchunk_union: copy(handle) failed: " + herr);
        }
    }
    auto* ctx = reinterpret_cast<PhdfCtx*>(handle_host[0]);
    const hid_t ds_id = (hid_t)handle_host[1];

    auto fail = [ctx](ffi::ErrorCode code, const std::string& msg) {
        announce_error(ctx, msg);
        ffi::Promise p;
        ffi::Future f(p);
        p.SetError(ffi::Error(code, msg));
        return f;
    };

    if (!ctx) {
        return fail(ffi::ErrorCode::kInvalidArgument,
                    "phdf5 read_kchunk_union: ctx_handle is null");
    }
    // Audit B2 — see the note in ReadDispatch.  This dispatch returns a
    // Future, so the refusal goes back through ``fail_early`` (which does not
    // touch ctx), not through ``fail``.
    {
        std::string live_err;
        if (!check_live_ctx(ctx, handle_host[0], "phdf5 read_kchunk_union",
                            &live_err)) {
            return fail_early(ffi::ErrorCode::kInvalidArgument, live_err);
        }
    }

    const auto dims = A_out->dimensions();
    if (dims.size() < 2) {
        return fail(ffi::ErrorCode::kInvalidArgument,
            "phdf5 read_kchunk_union: output must have >= 2 dims");
    }
    const size_t N_out = dims.size();
    const size_t N_file = N_out - 1;
    if (kchunk_axis_attr < 0 || (size_t)kchunk_axis_attr >= N_out) {
        return fail(ffi::ErrorCode::kInvalidArgument,
            "phdf5 read_kchunk_union: kchunk_axis out of range");
    }
    const int kax = (int)kchunk_axis_attr;
    const int64_t n_kchunk = (int64_t)dims[kax];
    if (n_kchunk != n_kchunk_attr) {
        return fail(ffi::ErrorCode::kInvalidArgument,
            "phdf5 read_kchunk_union: dims[kchunk_axis] mismatch vs n_kchunk attr");
    }
    if (axis_count_per_dim.size() != N_file) {
        return fail(ffi::ErrorCode::kInvalidArgument,
            "phdf5 read_kchunk_union: axis_count_per_dim / N_file mismatch");
    }
    const auto obd = offset_buf.dimensions();
    const auto cbd = count_buf.dimensions();
    if (obd.size() != 2 || (int64_t)obd[0] != n_kchunk || (size_t)obd[1] != N_file
        || cbd.size() != 2 || (int64_t)cbd[0] != n_kchunk || (size_t)cbd[1] != N_file) {
        return fail(ffi::ErrorCode::kInvalidArgument,
            "phdf5 read_kchunk_union: offset_buf / count_buf shape mismatch");
    }
    {
        std::string enc_err;
        if (!validate_shard_encoding(ctx, "phdf5 read_kchunk_union",
                                     mesh_shape, axis_count_per_dim,
                                     axis_flat, &enc_err)) {
            return fail(ffi::ErrorCode::kInvalidArgument, enc_err);
        }
    }

    // Bring both small index buffers to host (D2H on CUDA; already host-
    // resident on the host build).
    std::vector<int64_t> off_host((size_t)n_kchunk * N_file);
    std::vector<int64_t> cnt_host((size_t)n_kchunk * N_file);
    {
        std::string cperr;
        if (!copy_index_to_host(LRX_STREAM_ARG off_host.data(),
                                offset_buf.untyped_data(),
                                off_host.size() * sizeof(int64_t), &cperr)) {
            return fail(ffi::ErrorCode::kInternal,
                "phdf5 read_kchunk_union: copy(offset): " + cperr);
        }
        if (!copy_index_to_host(LRX_STREAM_ARG cnt_host.data(),
                                count_buf.untyped_data(),
                                cnt_host.size() * sizeof(int64_t), &cperr)) {
            return fail(ffi::ErrorCode::kInternal,
                "phdf5 read_kchunk_union: copy(count): " + cperr);
        }
    }

    // Per-rank coord shift on sharded dims (same encoding as single-
    // offset handler).  Applied to both offset and count uniformly:
    // offset[d] += rank_coord[d] * per_rank_max_shape[d]; count is
    // per-k user value (bounded above by per_rank_max_shape[d]).
    std::vector<int64_t> coord = unravel_rank(ctx->rank, mesh_shape);
    std::vector<int64_t> rank_coord_per_dim(N_file, 0);
    {
        size_t flat_idx = 0;
        for (size_t d = 0; d < N_file; ++d) {
            int64_t na = axis_count_per_dim[d];
            int64_t rc = 0, stride = 1;
            for (int64_t k = na - 1; k >= 0; --k) {
                int64_t ax = axis_flat[flat_idx + k];
                if (ax < 0 || (size_t)ax >= mesh_shape.size()) {
                    return fail(ffi::ErrorCode::kInvalidArgument,
                        "phdf5 read_kchunk_union: bad axis");
                }
                rc += coord[ax] * stride;
                stride *= mesh_shape[ax];
            }
            rank_coord_per_dim[d] = rc;
            flat_idx += (size_t)na;
        }
    }

    // Which file dims are sharded (their counts differ across ranks).
    std::vector<char> sharded(N_file, 0);
    for (size_t d = 0; d < N_file; ++d) sharded[d] = axis_count_per_dim[d] > 0;

    // per_rank_max_shape: dims with kchunk axis removed.
    std::vector<hsize_t> per_rank_max(N_file);
    {
        size_t di = 0;
        for (size_t d = 0; d < N_out; ++d) {
            if ((int)d == kax) continue;
            per_rank_max[di++] = (hsize_t)dims[d];
        }
    }

    // Build the per-k file offset/count vectors, with per-rank shard shift.
    std::vector<std::vector<hsize_t>> offsets(
        (size_t)n_kchunk, std::vector<hsize_t>(N_file, 0));
    std::vector<std::vector<hsize_t>> counts(
        (size_t)n_kchunk, std::vector<hsize_t>(N_file, 0));
    for (int64_t k = 0; k < n_kchunk; ++k) {
        for (size_t d = 0; d < N_file; ++d) {
            int64_t off = off_host[(size_t)k * N_file + d]
                        + rank_coord_per_dim[d] * (int64_t)per_rank_max[d];
            int64_t cnt = cnt_host[(size_t)k * N_file + d];
            if (off < 0 || cnt < 0 || cnt > (int64_t)per_rank_max[d]) {
                std::ostringstream os;
                os << "phdf5 read_kchunk_union: bad (off,cnt) at k=" << k
                   << " d=" << d << " off=" << off << " cnt=" << cnt
                   << " max=" << per_rank_max[d];
                return fail(ffi::ErrorCode::kInvalidArgument, os.str());
            }
            offsets[(size_t)k][d] = (hsize_t)off;
            counts[(size_t)k][d] = (hsize_t)cnt;
        }
    }

    // Dtype → (native_type, element_size) — one lookup, dtype-erased
    // task body uses void* + memcpy bytes downstream.
    size_t element_size = 0;
    hid_t native_type = H5I_INVALID_HID;
    const auto dtype = A_out->element_type();
    switch (dtype) {
        case ffi::DataType::F32:  element_size = sizeof(float);
                                   native_type = dt::h5_native_type<float>();  break;
        case ffi::DataType::F64:  element_size = sizeof(double);
                                   native_type = dt::h5_native_type<double>(); break;
        case ffi::DataType::S32:  element_size = sizeof(int32_t);
                                   native_type = dt::h5_native_type<int32_t>(); break;
        case ffi::DataType::S64:  element_size = sizeof(int64_t);
                                   native_type = dt::h5_native_type<int64_t>(); break;
        case ffi::DataType::C64:  element_size = sizeof(std::complex<float>);
                                   native_type = dt::h5_native_type<std::complex<float>>();  break;
        case ffi::DataType::C128: element_size = sizeof(std::complex<double>);
                                   native_type = dt::h5_native_type<std::complex<double>>(); break;
        default:
            return fail(ffi::ErrorCode::kInvalidArgument,
                "phdf5 read_kchunk_union: unsupported dtype");
    }

    // Async handoff: Promise/Future pair; task lambda runs the H5Dread
    // + async H2D on the ctx's writer thread (shared with write path —
    // tasks FIFO-drain through one C++ thread).  Main thread returns
    // the Future immediately so XLA's scheduler can run downstream ops
    // concurrent with the in-flight H5Dread.
    ffi::Promise promise;
    ffi::Future future(promise);
    auto task = [ctx,
                 ds_id,
                 native_type,
                 d_dst = A_out->untyped_data(),
                 element_size,
                 offsets = std::move(offsets),
                 counts  = std::move(counts),
                 per_rank_max = std::move(per_rank_max),
                 sharded = std::move(sharded),
                 kax,
                 promise = std::move(promise)]() mutable
    {
        async_read_kchunk_union_worker(
            ctx, ds_id, native_type, d_dst, element_size,
            std::move(offsets), std::move(counts),
            std::move(per_rank_max), std::move(sharded), kax,
            std::move(promise));
    };
    // ``xla_stream`` is used ABOVE, by the three control-operand copies (seam
    // 2 is stream-ordered — SLAB_IO_ROOT_CAUSE_AUDIT.md B1).  The QUEUED TASK
    // below does not need it: it owns its own synchronization via h2d_event.
    {
        std::lock_guard<std::mutex> lk(ctx->queue_mu);
        ctx->task_queue.emplace_back(std::move(task));
    }
    ctx->queue_cv.notify_one();
    return future;
}

}  // namespace lorrax_ffi::phdf5

// Handler registration.  The three handlers share one dispatch body each
// (compiled from the SAME functions above); the platform split is only in
// the binding: the CUDA build names them Phdf*Ffi and prepends
// ``.Ctx<PlatformStream<cudaStream_t>>()``; the host build names them
// Phdf*HostFfi and omits the stream ctx (XLA's CPU runtime hands the
// handler host buffers on the calling thread).  ffi_loader.py registers the
// CUDA names under platform="CUDA" and the Host names under platform="cpu",
// both against the same jax.ffi target strings, so the ffi_call sites stay
// platform-agnostic — the split jaxlib uses for cpu-lapack vs cuda-cusolver.
// ``LRX_PHDF_HANDLER`` / ``LRX_PHDF_STREAM_CTX`` are defined in
// platform_seam.h (seam 1), shared with write_ffi.cc.

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LRX_PHDF_HANDLER(PhdfRead), lorrax_ffi::phdf5::ReadDispatch,
    xla::ffi::Ffi::Bind()
        LRX_PHDF_STREAM_CTX
        .Arg<xla::ffi::AnyBuffer>()   // destination, aliased to the result (ABI 7)
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<xla::ffi::Span<const int64_t>>("mesh_shape")
        .Attr<xla::ffi::Span<const int64_t>>("axis_count_per_dim")
        .Attr<xla::ffi::Span<const int64_t>>("axis_flat"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LRX_PHDF_HANDLER(PhdfReadKchunk), lorrax_ffi::phdf5::ReadKchunkDispatch,
    xla::ffi::Ffi::Bind()
        LRX_PHDF_STREAM_CTX
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()   // handle {ctx, ds}
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()   // offset_buf (n_kchunk, N_file)
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<xla::ffi::Span<const int64_t>>("mesh_shape")
        .Attr<xla::ffi::Span<const int64_t>>("axis_count_per_dim")
        .Attr<xla::ffi::Span<const int64_t>>("axis_flat")
        .Attr<int64_t>("n_kchunk"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LRX_PHDF_HANDLER(PhdfReadKchunkUnion),
    lorrax_ffi::phdf5::ReadKchunkUnionDispatch,
    xla::ffi::Ffi::Bind()
        LRX_PHDF_STREAM_CTX
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()   // handle {ctx, ds}
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()   // offset_buf (n_kchunk, N_file)
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()   // count_buf  (n_kchunk, N_file)
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<xla::ffi::Span<const int64_t>>("mesh_shape")
        .Attr<xla::ffi::Span<const int64_t>>("axis_count_per_dim")
        .Attr<xla::ffi::Span<const int64_t>>("axis_flat")
        .Attr<int64_t>("n_kchunk")
        .Attr<int64_t>("kchunk_axis"));

#if H5_VERSION_GE(1, 14, 0)
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LRX_PHDF_HANDLER(PhdfReadMulti), lorrax_ffi::phdf5::ReadMultiDispatch,
    xla::ffi::Ffi::Bind()
        LRX_PHDF_STREAM_CTX
        .RemainingArgs()   // the N destinations, aliased to the results (ABI 7)
        .RemainingRets()   // the N outputs
        .Attr<xla::ffi::Span<const int64_t>>("mesh_shape")
        .Attr<xla::ffi::Span<const int64_t>>("axis_count_per_dim")
        .Attr<xla::ffi::Span<const int64_t>>("axis_flat"));
#endif
