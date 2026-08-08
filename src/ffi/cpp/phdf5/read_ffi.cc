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
//     dependency; handler returns without a blocking
//     cudaEventSynchronize because xla_stream will wait on its own
//     before reading the output buffer.
//
// No per-call cudaEventCreate/Destroy: that causes a ~800 ms stall
// on non-rank-0 processes under JAX's cuda_async allocator (measured
// 2026-04-18 on writes; same mechanism applies here).

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
#ifdef LORRAX_FFI_NO_CUDA
static inline ffi::Error stage_host_to_output(
    PhdfCtx* ctx, const void* src, void* d_dst, size_t bytes)
{
    (void)ctx;
    std::memcpy(d_dst, src, bytes);
    return ffi::Error::Success();
}
#else
static inline ffi::Error stage_host_to_output(
    cudaStream_t xla_stream, PhdfCtx* ctx, const void* src,
    void* d_dst, size_t bytes)
{
    LORRAX_CUDA_CHECK(cudaMemcpyAsync(d_dst, src, bytes,
                                      cudaMemcpyHostToDevice, ctx->stream));
    LORRAX_CUDA_CHECK(cudaEventRecord(ctx->h2d_event, ctx->stream));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(xla_stream, ctx->h2d_event, 0));
    return ffi::Error::Success();
}
#endif

// ``copy_index_to_host`` (seam 2) is in platform_seam.h — shared with
// write_ffi.cc so the two TUs cannot drift.  ``unravel_rank`` /
// ``vec_to_string`` / ``validate_shard_encoding`` / ``checked_buffer_bytes``
// / ``check_dataset_rank`` / ``announce_error`` are in shard_index.h, for the
// same reason.

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

    hid_t dset = ds_id;
    hid_t native_type = dt::h5_native_type<T>();
    if (dset < 0 || H5Iis_valid(dset) <= 0) {
        return fail_read(ffi::ErrorCode::kInvalidArgument,
                         "phdf5 read: ds_id is invalid");
    }

    hid_t filespace = H5Dget_space(dset);
    if (filespace < 0) {
        return fail_read(ffi::ErrorCode::kInternal,
                         "phdf5 read: H5Dget_space failed");
    }
    // Rank agreement, checked BEFORE H5Sget_simple_extent_dims below writes
    // file_rank entries into an rank-sized buffer.  write_ffi.cc has always
    // had this guard; the read path did not, and reading an N-D request
    // against an (N+k)-D dataset would have overrun the extent vector.
    // Rank-invariant (the dataset and the request are the same on every
    // rank), so it refuses everywhere or nowhere.
    {
        std::string rank_err;
        if (!check_dataset_rank(filespace, rank, "phdf5 read",
                                nullptr, &rank_err)) {
            H5Sclose(filespace);
            return fail_read(ffi::ErrorCode::kInvalidArgument, rank_err);
        }
    }
    hid_t memspace = H5Screate_simple(rank, mem_dims.data(), nullptr);
    if (memspace < 0) {
        H5Sclose(filespace);
        return fail_read(ffi::ErrorCode::kInternal,
                         "phdf5 read: H5Screate_simple(memspace) failed");
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
        return fail_read(ffi::ErrorCode::kInternal,
                         "phdf5 read: H5Sget_simple_extent_dims failed");
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
            os << "phdf5 read: logical slab out of bounds"
               << " extent=" << vec_to_string(extent)
               << " offset_base=" << vec_to_string(offset_base)
               << " valid_shape=" << vec_to_string(valid_shape)
               << " rank=" << ctx->rank
               << " -- refused identically on every rank";
            H5Sclose(memspace);
            H5Sclose(filespace);
            return fail_read(ffi::ErrorCode::kInvalidArgument, os.str());
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
            return fail_read(ffi::ErrorCode::kInternal,
                             "phdf5 read: H5Sselect_none failed");
        }
    } else if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                             offset.data(), nullptr,
                             file_count.data(),  nullptr) < 0) {
        H5Sclose(memspace);
        H5Sclose(filespace);
        return fail_read(ffi::ErrorCode::kInternal,
                         "phdf5 read: H5Sselect_hyperslab failed");
    } else {
        std::vector<hsize_t> mem_start((size_t)rank, 0);
        if (H5Sselect_hyperslab(memspace, H5S_SELECT_SET,
                                mem_start.data(), nullptr,
                                file_count.data(), nullptr) < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            return fail_read(ffi::ErrorCode::kInternal,
                             "phdf5 read: H5Sselect_hyperslab(mem) failed");
        }
    }

    hid_t dxpl = ctx->use_collective_read ? ctx->dxpl_coll : ctx->dxpl_indep;
    herr_t st = H5Dread(dset, native_type, memspace, filespace, dxpl, stage);
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
    return stage_host_to_output(LRX_STREAM_ARG ctx, stage, d_dst, bytes);
}

// Padding contract: A_out dimensions are the equal-block physical
// output shape. valid_shape is the logical file prefix to read; ranks
// outside it get zeros, and edge ranks read clipped hyperslabs.
static ffi::Error ReadDispatch(
    LRX_STREAM_PARAM
    ffi::Buffer<ffi::DataType::S64> handle_buf,   // shape (2,) {ctx_handle, ds_id}
    ffi::Buffer<ffi::DataType::S64> offset_buf,   // shape (ndim,)
    ffi::Buffer<ffi::DataType::S64> valid_shape_buf, // shape (ndim,)
    ffi::Result<ffi::AnyBuffer> A_out,
    ffi::Span<const int64_t> mesh_shape,
    ffi::Span<const int64_t> axis_count_per_dim,
    ffi::Span<const int64_t> axis_flat)
{
    // ``ctx_handle`` and ``ds_id`` arrive as a RUNTIME buffer, not as FFI
    // Attrs.  Attrs are baked into the HLO, and ctx_handle is a heap
    // address, so as an Attr it made every module unique PER PROCESS: the
    // persistent compile cache could never hit one and rewrote it on every
    // run.  MEASURED 2026-08-07 with a byte-identical workload and a
    // private cache dir: jit__per_rank entries went 4 -> 8 -> 12 over three
    // runs while a plain jit control stayed at 1, and the shared np1 cache
    // held 6813 such corpses out of 14443 entries.  ds_id moves with it —
    // as an Attr it also forked a module per (file, dataset) pair, so the
    // module count scaled with the deck instead of with the geometry.
    // This is the same argument that already made ``offset_base`` a Buffer;
    // see the note above WriteDispatch in write_ffi.cc.
    //
    // Both values are replicated across the mesh (the shard_map passes them
    // with ``P()``), so every refusal below is still computed from inputs
    // that are identical on every rank.
    if (handle_buf.dimensions().size() != 1 ||
        handle_buf.dimensions()[0] != 2) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "phdf5 read: handle_buf must have shape (2,) == "
            "{ctx_handle, ds_id}");
    }
    int64_t handle_host[2] = {0, 0};
    {
        // No ctx yet, so there is nothing to announce through: return
        // plainly.  A failure here is a broken D2H copy, not a collective
        // divergence — it hits every rank identically.
        std::string herr;
        if (!copy_index_to_host(handle_host, handle_buf.untyped_data(),
                                2 * sizeof(int64_t), &herr)) {
            return ffi::Error(ffi::ErrorCode::kInternal,
                              "phdf5 read: copy(handle) failed: " + herr);
        }
    }
    auto* ctx = reinterpret_cast<PhdfCtx*>(handle_host[0]);
    const hid_t ds_id = (hid_t)handle_host[1];

    // Every refusal in this dispatch is computed from replicated inputs (FFI
    // Attrs, the replicated handle buffer, the equal-block output shape, a
    // collectively-opened ds_id), so it fires on every rank or none — the
    // only kind a collective tolerates.
    auto fail = [ctx](ffi::ErrorCode code, const std::string& msg) {
        announce_error(ctx, msg);
        return ffi::Error(code, msg);
    };

    if (!ctx) {
        return fail(ffi::ErrorCode::kInvalidArgument,
                    "phdf5 read: ctx_handle is null");
    }

    const auto dims = A_out->dimensions();
    const size_t N = dims.size();
    if (N == 0) {
        return fail(ffi::ErrorCode::kInvalidArgument,
                    "phdf5 read: output is 0-D; there is no hyperslab to read");
    }
    if (offset_buf.dimensions().size() != 1 ||
        (size_t)offset_buf.dimensions()[0] != N ||
        valid_shape_buf.dimensions().size() != 1 ||
        (size_t)valid_shape_buf.dimensions()[0] != N ||
        axis_count_per_dim.size() != N) {
        std::ostringstream os;
        os << "phdf5 read: rank mismatch  A.ndim=" << N
           << " offset_buf.ndim=" << offset_buf.dimensions().size()
           << " valid_shape_buf.ndim=" << valid_shape_buf.dimensions().size()
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

    // Bring the small offset + valid_shape buffers (N × 8 bytes each) into
    // host memory (D2H on CUDA; already host-resident on the host build).
    std::vector<int64_t> offset_host(N);
    std::string cperr;
    if (!copy_index_to_host(offset_host.data(), offset_buf.untyped_data(),
                            N * sizeof(int64_t), &cperr)) {
        return fail(ffi::ErrorCode::kInternal,
                    "phdf5 read: copy(offset) failed: " + cperr);
    }
    std::vector<int64_t> valid_shape_host(N);
    if (!copy_index_to_host(valid_shape_host.data(),
                            valid_shape_buf.untyped_data(),
                            N * sizeof(int64_t), &cperr)) {
        return fail(ffi::ErrorCode::kInternal,
                    "phdf5 read: copy(valid_shape) failed: " + cperr);
    }

    std::vector<int64_t> coord = unravel_rank(ctx->rank, mesh_shape);
    std::vector<hsize_t> offset(N), file_count(N), mem_dims(N);
    size_t flat_idx = 0;
    for (size_t d = 0; d < N; ++d) {
        if (offset_host[d] < 0 || valid_shape_host[d] < 0) {
            std::ostringstream os;
            os << "phdf5 read: negative offset/valid_shape at dim " << d
               << " offset=" << offset_host[d]
               << " valid_shape=" << valid_shape_host[d];
            return fail(ffi::ErrorCode::kInvalidArgument, os.str());
        }
        if (offset_host[d] > INT64_MAX - valid_shape_host[d]) {
            std::ostringstream os;
            os << "phdf5 read: offset+valid_shape overflows int64 at dim "
               << d << " (" << offset_host[d] << "+" << valid_shape_host[d]
               << "); the bounds test would wrap negative and accept an "
               << "out-of-range slab";
            return fail(ffi::ErrorCode::kInvalidArgument, os.str());
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
                os << "phdf5 read: bad axis " << ax
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
    // Diagnostic timing (LORRAX_PHDF5_TIME=1 to enable per-phase prints).
    // Parsed, not merely PRESENT: this was `getenv(...) != nullptr`, so
    // LORRAX_PHDF5_TIME=0 turned timing ON.  Grammar lives in ctx.h.
    const bool do_time = env_flag("LORRAX_PHDF5_TIME", false);
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
        stage_host_to_output(LRX_STREAM_ARG ctx, stage, d_dst, bytes));
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
        if (!copy_index_to_host(handle_host, handle_buf.untyped_data(),
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
        if (!copy_index_to_host(offset_host.data(), offset_buf.untyped_data(),
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
//  Read-kchunk-UNION handler — ONE H5Dread that pulls n_kchunk
//  non-overlapping hyperslabs of a dataset via H5S_SELECT_OR compound
//  selection.  Output buffer is (n_kchunk, *per_rank_max_shape); for each
//  k, the FIRST count[k][d] cells of dim d in that k's output slot get
//  filled from file offset offset[k][d], and the remaining padding cells
//  stay at their initial zero.
//
//  Caller must guarantee the per-k (offset, count) file slabs are
//  pairwise disjoint; otherwise HDF5's union-of-hyperslabs deduplicates
//  and the element counts mismatch the memspace.  For WFN.h5-style
//  data, passing count[k] = (bpr, ns, ngk[k], 2) (the *actual* ngk, not
//  ngkmax) gives disjoint per-k slabs even when ngk varies.
// ────────────────────────────────────────────────────────────────────────────

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
// nothing on another thread can be inside this buffer here.  Its reuse
// across tasks is safe because:
//   1. Tasks run strictly sequentially on the writer thread.
//   2. The task *starts* by ``cudaEventSynchronize(h2d_event)`` which
//      blocks until the PREVIOUS task's async H2D has drained — so the
//      ``memset`` + ``H5Dread`` that follow don't stomp a memcpy that's
//      still reading from ``pinned_buf``.
//   3. At the end the task records ``h2d_event`` again and syncs
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
    int kchunk_axis,
    ffi::Promise promise)
{
    const int n_kchunk = (int)file_offsets.size();
    const int N_file = (int)per_rank_max_shape.size();
    const int N_out = N_file + 1;
    const bool do_time = env_flag("LORRAX_PHDF5_TIME", false);
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
#ifndef LORRAX_FFI_NO_CUDA
    cudaEventSynchronize(ctx->h2d_event);
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

    if (!ensure_pinned(ctx, bytes)) {
        std::ostringstream os;
        os << "phdf5 read_kchunk_union: staging-buffer alloc of " << bytes << " bytes failed";
        fail_task(ffi::ErrorCode::kResourceExhausted, os.str());
        return;
    }
    // Zero the pinned buffer so padding cells stay at exact zero.
    std::memset(ctx->pinned_buf, 0, bytes);
    auto t_pin = now();

    hid_t dxpl = ctx->use_collective_read ? ctx->dxpl_coll : ctx->dxpl_indep;

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
    H5Sselect_none(filespace);
    H5Sselect_none(memspace);
    auto t_spaces = now();

    std::vector<hsize_t> mem_off(N_out, 0);
    std::vector<hsize_t> mem_count(N_out);
    for (int k = 0; k < n_kchunk; ++k) {
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_OR,
                                 file_offsets[k].data(), nullptr,
                                 file_counts[k].data(), nullptr) < 0) {
            H5Sclose(memspace); H5Sclose(filespace);
            std::ostringstream os;
            os << "phdf5 read_kchunk_union: file H5Sselect_hyperslab(OR) "
               << "failed at k=" << k;
            fail_task(ffi::ErrorCode::kInternal, os.str());
            return;
        }
        int fi = 0;
        for (int d = 0; d < N_out; ++d) {
            if (d == kchunk_axis) {
                mem_off[d] = (hsize_t)k;
                mem_count[d] = 1;
            } else {
                mem_off[d] = 0;
                mem_count[d] = file_counts[k][fi++];
            }
        }
        if (H5Sselect_hyperslab(memspace, H5S_SELECT_OR,
                                 mem_off.data(), nullptr,
                                 mem_count.data(), nullptr) < 0) {
            H5Sclose(memspace); H5Sclose(filespace);
            std::ostringstream os;
            os << "phdf5 read_kchunk_union: mem H5Sselect_hyperslab(OR) "
               << "failed at k=" << k;
            fail_task(ffi::ErrorCode::kInternal, os.str());
            return;
        }
    }
    auto t_select = now();

    hssize_t npts_file = H5Sget_select_npoints(filespace);
    hssize_t npts_mem  = H5Sget_select_npoints(memspace);
    if (npts_file != npts_mem) {
        H5Sclose(memspace); H5Sclose(filespace);
        std::ostringstream os;
        os << "phdf5 read_kchunk_union: selection size mismatch — "
           << "filespace=" << npts_file << " memspace=" << npts_mem
           << " (overlapping per-k slabs?)";
        fail_task(ffi::ErrorCode::kInvalidArgument, os.str());
        return;
    }

    herr_t st = H5Dread(ds_id, native_type, memspace, filespace, dxpl,
                        ctx->pinned_buf);
    auto t_read = now();
    H5Sclose(memspace);
    H5Sclose(filespace);
    if (st < 0) {
        fail_task(ffi::ErrorCode::kInternal,
                  "phdf5 read_kchunk_union: H5Dread failed");
        return;
    }

    // Queue H2D on ctx->stream; record h2d_event at completion; wait on
    // the event before firing the Promise so ``d_dst`` is guaranteed on
    // device by the time XLA schedules downstream consumers.  Overlap
    // with the NEXT task's H5Dread still works because both live on
    // ctx->writer_thread and pipeline on its FIFO queue — the next
    // task's cudaEventSynchronize(h2d_event) at start is a no-op (the
    // event is already in its recorded-complete state by then).
#ifdef LORRAX_FFI_NO_CUDA
    // Host: d_dst is host memory; the memcpy completes synchronously so
    // ``d_dst`` is fully populated before the Promise fires below.
    std::memcpy(d_dst, ctx->pinned_buf, bytes);
#else
    cudaError_t ce = cudaMemcpyAsync(
        d_dst, ctx->pinned_buf, bytes, cudaMemcpyHostToDevice, ctx->stream);
    if (ce != cudaSuccess) {
        fail_task(ffi::ErrorCode::kInternal,
            std::string("phdf5 read_kchunk_union: cudaMemcpyAsync: ") +
            cudaGetErrorString(ce));
        return;
    }
    cudaEventRecord(ctx->h2d_event, ctx->stream);
    cudaEventSynchronize(ctx->h2d_event);
#endif
    auto t_h2d = now();

    if (do_time && ctx->rank == 0) {
        std::fprintf(stderr,
            "[phdf5 kchunk-union r0] n_k=%d bytes/rank=%zu  "
            "prev_h2d_wait=%.2f  pin+zero=%.2f  spaces=%.2f  select=%.2f  "
            "read=%.2f  h2d_setup=%.2f  total=%.2f (ms)  [selected_elts=%lld]\n",
            n_kchunk, bytes,
            ms(t0, t_prev_h2d_done),
            ms(t_prev_h2d_done, t_pin),
            ms(t_pin, t_spaces),
            ms(t_spaces, t_select),
            ms(t_select, t_read),
            ms(t_read, t_h2d),
            ms(t0, t_h2d),
            (long long)npts_file);
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
        if (!copy_index_to_host(handle_host, handle_buf.untyped_data(),
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
        if (!copy_index_to_host(off_host.data(), offset_buf.untyped_data(),
                                off_host.size() * sizeof(int64_t), &cperr)) {
            return fail(ffi::ErrorCode::kInternal,
                "phdf5 read_kchunk_union: copy(offset): " + cperr);
        }
        if (!copy_index_to_host(cnt_host.data(), count_buf.untyped_data(),
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
                 kax,
                 promise = std::move(promise)]() mutable
    {
        async_read_kchunk_union_worker(
            ctx, ds_id, native_type, d_dst, element_size,
            std::move(offsets), std::move(counts),
            std::move(per_rank_max), kax,
            std::move(promise));
    };
#ifndef LORRAX_FFI_NO_CUDA
    (void)xla_stream;  // unused: task owns its own synchronization via h2d_event
#endif
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
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()  // handle {ctx, ds}
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()  // offset_base
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()  // valid_shape
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
