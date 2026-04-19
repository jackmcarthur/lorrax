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

#include <complex>
#include <cstdint>
#include <cstdio>
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
