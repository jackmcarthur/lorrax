// read_ffi.cc — mirror of write_ffi.cc for H5Dread.  Same N-D block-
// partitioned sharding semantics, same pooled-event + runtime-offset
// lessons from the write-path investigation.
//
// Work shape per call:
//  1. H5Dread (MPI-IO collective) into ``ctx->pinned_buf`` on the
//     XLA executor thread — blocks for the read duration.
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

#include <cuda_runtime.h>
#include <hdf5.h>

#include "xla/ffi/api/ffi.h"

#include "../../common/cpp/ffi_helpers.h"
#include "ctx.h"
#include "phdf5_interface.h"

namespace lorrax_ffi::phdf5 {

namespace ffi = ::xla::ffi;

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
static ffi::Error ReadImpl(
    cudaStream_t xla_stream,
    PhdfCtx* ctx,
    T* d_dst,
    const std::vector<hsize_t>& offset,
    const std::vector<hsize_t>& count,
    hid_t ds_id)
{
    const int rank = (int)offset.size();
    size_t n_local_elts = 1;
    for (auto c : count) n_local_elts *= (size_t)c;
    const size_t bytes = n_local_elts * sizeof(T);

    if (!ensure_pinned(ctx, bytes)) {
        std::ostringstream os;
        os << "phdf5 read: cudaMallocHost(" << bytes << ") failed";
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, os.str());
    }

    hid_t dset = ds_id;
    hid_t native_type = dt::h5_native_type<T>();
    if (dset < 0 || H5Iis_valid(dset) <= 0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
            "phdf5 read: ds_id is invalid");
    }

    hid_t filespace = H5Dget_space(dset);
    if (filespace < 0) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 read: H5Dget_space failed");
    }
    if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                             offset.data(), nullptr,
                             count.data(),  nullptr) < 0) {
        H5Sclose(filespace);
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 read: H5Sselect_hyperslab failed");
    }
    hid_t memspace = H5Screate_simple(rank, count.data(), nullptr);
    if (memspace < 0) {
        H5Sclose(filespace);
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 read: H5Screate_simple(memspace) failed");
    }

    hid_t dxpl = ctx->use_collective ? ctx->dxpl_coll : ctx->dxpl_indep;
    herr_t st = H5Dread(dset, native_type, memspace, filespace, dxpl,
                        ctx->pinned_buf);
    H5Sclose(memspace);
    H5Sclose(filespace);
    if (st < 0) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 read: H5Dread failed");
    }

    // Async H2D to the XLA-allocated output buffer.  No cross-stream-
    // wait for xla_stream state: XLA's FFI contract guarantees the
    // output buffer is available for writing at handler entry.
    LORRAX_CUDA_CHECK(cudaMemcpyAsync(d_dst, ctx->pinned_buf, bytes,
                                      cudaMemcpyHostToDevice, ctx->stream));

    // Record completion on the pooled ctx event and make xla_stream
    // wait for it.  Downstream ops on xla_stream will defer until the
    // H2D is done.  No cudaEventSynchronize here: we don't need to
    // block the handler thread — xla_stream's own dependency handles
    // correctness.  No cudaEventDestroy: the event is owned by ctx.
    LORRAX_CUDA_CHECK(cudaEventRecord(ctx->h2d_event, ctx->stream));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(xla_stream, ctx->h2d_event, 0));

    return ffi::Error::Success();
}

static ffi::Error ReadDispatch(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::S64> offset_buf,   // shape (ndim,)
    ffi::Result<ffi::AnyBuffer> A_out,
    int64_t ctx_handle,
    int64_t ds_id,
    ffi::Span<const int64_t> mesh_shape,
    ffi::Span<const int64_t> axis_count_per_dim,
    ffi::Span<const int64_t> axis_flat)
{
    auto* ctx = reinterpret_cast<PhdfCtx*>(ctx_handle);
    if (!ctx) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "phdf5 read: ctx_handle is null");
    }

    const auto dims = A_out->dimensions();
    const size_t N = dims.size();
    if (offset_buf.dimensions().size() != 1 ||
        (size_t)offset_buf.dimensions()[0] != N ||
        axis_count_per_dim.size() != N) {
        std::ostringstream os;
        os << "phdf5 read: rank mismatch  A.ndim=" << N
           << " offset_buf.ndim=" << offset_buf.dimensions().size()
           << " axis_count_per_dim.size=" << axis_count_per_dim.size();
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }

    // D2H-copy the small offset buffer (N × 8 bytes).
    std::vector<int64_t> offset_host(N);
    cudaError_t ce_off = cudaMemcpy(offset_host.data(), offset_buf.untyped_data(),
                                    N * sizeof(int64_t), cudaMemcpyDeviceToHost);
    if (ce_off != cudaSuccess) {
        std::ostringstream os;
        os << "phdf5 read: cudaMemcpy(offset) failed: "
           << cudaGetErrorString(ce_off);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    std::vector<int64_t> coord = unravel_rank(ctx->rank, mesh_shape);
    std::vector<hsize_t> offset(N), count(N);
    size_t flat_idx = 0;
    for (size_t d = 0; d < N; ++d) {
        count[d] = (hsize_t)dims[d];
        offset[d] = (hsize_t)offset_host[d];
        int64_t na = axis_count_per_dim[d];
        int64_t rank_coord = 0;
        int64_t stride_acc = 1;
        for (int64_t k = na - 1; k >= 0; --k) {
            int64_t ax = axis_flat[flat_idx + k];
            if (ax < 0 || (size_t)ax >= mesh_shape.size()) {
                std::ostringstream os;
                os << "phdf5 read: bad axis " << ax
                   << " at dim " << d << " axis " << k;
                return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
            }
            rank_coord += coord[ax] * stride_acc;
            stride_acc *= mesh_shape[ax];
        }
        offset[d] += rank_coord * count[d];
        flat_idx += (size_t)na;
    }

    const auto dtype = A_out->element_type();
    switch (dtype) {
        case ffi::DataType::F32:
            return ReadImpl<float>(stream, ctx,
                static_cast<float*>(A_out->untyped_data()),
                offset, count, (hid_t)ds_id);
        case ffi::DataType::F64:
            return ReadImpl<double>(stream, ctx,
                static_cast<double*>(A_out->untyped_data()),
                offset, count, (hid_t)ds_id);
        case ffi::DataType::S32:
            return ReadImpl<int32_t>(stream, ctx,
                static_cast<int32_t*>(A_out->untyped_data()),
                offset, count, (hid_t)ds_id);
        case ffi::DataType::S64:
            return ReadImpl<int64_t>(stream, ctx,
                static_cast<int64_t*>(A_out->untyped_data()),
                offset, count, (hid_t)ds_id);
        case ffi::DataType::C64:
            return ReadImpl<std::complex<float>>(stream, ctx,
                static_cast<std::complex<float>*>(A_out->untyped_data()),
                offset, count, (hid_t)ds_id);
        case ffi::DataType::C128:
            return ReadImpl<std::complex<double>>(stream, ctx,
                static_cast<std::complex<double>*>(A_out->untyped_data()),
                offset, count, (hid_t)ds_id);
        default: {
            std::ostringstream os;
            os << "phdf5 read: unsupported dtype " << static_cast<int>(dtype);
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
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
    cudaStream_t xla_stream,
    PhdfCtx* ctx,
    T* d_dst,
    const std::vector<std::vector<hsize_t>>& offsets,  // (n_kchunk, N_file)
    const std::vector<hsize_t>& count,                  // (N_file,)
    hid_t ds_id)
{
    const int n_kchunk = (int)offsets.size();
    const int N_file = (int)count.size();
    // Diagnostic timing (LORRAX_PHDF5_TIME=1 to enable per-phase prints).
    const bool do_time = std::getenv("LORRAX_PHDF5_TIME") != nullptr;
    auto now = []() { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t0 = now();

    size_t per_k_elts = 1;
    for (auto c : count) per_k_elts *= (size_t)c;
    const size_t total_elts = per_k_elts * (size_t)n_kchunk;
    const size_t bytes = total_elts * sizeof(T);

    if (!ensure_pinned(ctx, bytes)) {
        std::ostringstream os;
        os << "phdf5 read_kchunk: cudaMallocHost(" << bytes << ") failed";
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, os.str());
    }
    auto t_pin = now();

    hid_t native_type = dt::h5_native_type<T>();
    if (ds_id < 0 || H5Iis_valid(ds_id) <= 0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
            "phdf5 read_kchunk: ds_id is invalid");
    }

    hid_t dxpl = ctx->use_collective ? ctx->dxpl_coll : ctx->dxpl_indep;

    hid_t filespace = H5Dget_space(ds_id);
    if (filespace < 0) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 read_kchunk: H5Dget_space failed");
    }
    hid_t memspace = H5Screate_simple(N_file, count.data(), nullptr);
    if (memspace < 0) {
        H5Sclose(filespace);
        return ffi::Error(ffi::ErrorCode::kInternal,
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
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
        auto tb = now();
        T* k_buf = static_cast<T*>(ctx->pinned_buf) + (size_t)k * per_k_elts;
        herr_t st = H5Dread(ds_id, native_type, memspace, filespace, dxpl, k_buf);
        auto tc = now();
        t_select_total += ms(ta, tb);
        t_read_total += ms(tb, tc);
        if (st < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            std::ostringstream os;
            os << "phdf5 read_kchunk: H5Dread failed at k=" << k;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
    }
    H5Sclose(memspace);
    H5Sclose(filespace);
    auto t_reads = now();

    LORRAX_CUDA_CHECK(cudaMemcpyAsync(d_dst, ctx->pinned_buf, bytes,
                                      cudaMemcpyHostToDevice, ctx->stream));
    LORRAX_CUDA_CHECK(cudaEventRecord(ctx->h2d_event, ctx->stream));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(xla_stream, ctx->h2d_event, 0));
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
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::S64> offset_buf,   // shape (n_kchunk, N_file)
    ffi::Result<ffi::AnyBuffer> A_out,             // shape (n_kchunk, per-rank file dims...)
    int64_t ctx_handle,
    int64_t ds_id,
    ffi::Span<const int64_t> mesh_shape,
    ffi::Span<const int64_t> axis_count_per_dim,   // length N_file
    ffi::Span<const int64_t> axis_flat,
    int64_t n_kchunk_attr)
{
    auto* ctx = reinterpret_cast<PhdfCtx*>(ctx_handle);
    if (!ctx) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "phdf5 read_kchunk: ctx_handle is null");
    }

    const auto dims = A_out->dimensions();
    if (dims.size() < 2) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
            "phdf5 read_kchunk: output must have at least 2 dims "
            "(n_kchunk + file dims)");
    }
    const int64_t n_kchunk = (int64_t)dims[0];
    const size_t N_file = dims.size() - 1;

    if (n_kchunk != n_kchunk_attr) {
        std::ostringstream os;
        os << "phdf5 read_kchunk: dims[0]=" << n_kchunk
           << " != n_kchunk attr=" << n_kchunk_attr;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    if (axis_count_per_dim.size() != N_file) {
        std::ostringstream os;
        os << "phdf5 read_kchunk: axis_count_per_dim.size="
           << axis_count_per_dim.size() << " != N_file=" << N_file;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    const auto obd = offset_buf.dimensions();
    if (obd.size() != 2 || (int64_t)obd[0] != n_kchunk
                       || (size_t)obd[1] != N_file) {
        std::ostringstream os;
        os << "phdf5 read_kchunk: offset_buf shape must be "
           << "(" << n_kchunk << "," << N_file << "); got "
           << "(" << (obd.size() >= 1 ? obd[0] : -1)
           << "," << (obd.size() >= 2 ? obd[1] : -1) << ")";
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }

    // D2H-copy the small offset buffer (n_kchunk × N_file × 8 bytes).
    std::vector<int64_t> offset_host((size_t)n_kchunk * N_file);
    cudaError_t ce_off = cudaMemcpy(
        offset_host.data(), offset_buf.untyped_data(),
        offset_host.size() * sizeof(int64_t), cudaMemcpyDeviceToHost);
    if (ce_off != cudaSuccess) {
        std::ostringstream os;
        os << "phdf5 read_kchunk: cudaMemcpy(offset) failed: "
           << cudaGetErrorString(ce_off);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
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
                    return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
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
                return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
            }
            offsets[(size_t)k][d] = (hsize_t)off_kd;
        }
    }

    const auto dtype = A_out->element_type();
    switch (dtype) {
        case ffi::DataType::F32:
            return ReadKchunkImpl<float>(stream, ctx,
                static_cast<float*>(A_out->untyped_data()),
                offsets, count, (hid_t)ds_id);
        case ffi::DataType::F64:
            return ReadKchunkImpl<double>(stream, ctx,
                static_cast<double*>(A_out->untyped_data()),
                offsets, count, (hid_t)ds_id);
        case ffi::DataType::S32:
            return ReadKchunkImpl<int32_t>(stream, ctx,
                static_cast<int32_t*>(A_out->untyped_data()),
                offsets, count, (hid_t)ds_id);
        case ffi::DataType::S64:
            return ReadKchunkImpl<int64_t>(stream, ctx,
                static_cast<int64_t*>(A_out->untyped_data()),
                offsets, count, (hid_t)ds_id);
        case ffi::DataType::C64:
            return ReadKchunkImpl<std::complex<float>>(stream, ctx,
                static_cast<std::complex<float>*>(A_out->untyped_data()),
                offsets, count, (hid_t)ds_id);
        case ffi::DataType::C128:
            return ReadKchunkImpl<std::complex<double>>(stream, ctx,
                static_cast<std::complex<double>*>(A_out->untyped_data()),
                offsets, count, (hid_t)ds_id);
        default: {
            std::ostringstream os;
            os << "phdf5 read_kchunk: unsupported dtype " << static_cast<int>(dtype);
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::phdf5

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PhdfReadFfi, lorrax_ffi::phdf5::ReadDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()   // offset_base
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("ctx_handle")
        .Attr<int64_t>("ds_id")
        .Attr<xla::ffi::Span<const int64_t>>("mesh_shape")
        .Attr<xla::ffi::Span<const int64_t>>("axis_count_per_dim")
        .Attr<xla::ffi::Span<const int64_t>>("axis_flat"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PhdfReadKchunkFfi, lorrax_ffi::phdf5::ReadKchunkDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()   // offset_buf (n_kchunk, N_file)
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("ctx_handle")
        .Attr<int64_t>("ds_id")
        .Attr<xla::ffi::Span<const int64_t>>("mesh_shape")
        .Attr<xla::ffi::Span<const int64_t>>("axis_count_per_dim")
        .Attr<xla::ffi::Span<const int64_t>>("axis_flat")
        .Attr<int64_t>("n_kchunk"));
