// read_ffi.cc — mirror of write_ffi.cc for H5Dread.  Same N-D block-
// partitioned attrs (offset_base, mesh_shape, axis_for_dim); count[]
// comes from A_out.dimensions().

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

    cudaEvent_t ev_prod;
    LORRAX_CUDA_CHECK(cudaEventCreateWithFlags(&ev_prod, cudaEventDisableTiming));
    LORRAX_CUDA_CHECK(cudaEventRecord(ev_prod, xla_stream));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(ctx->stream, ev_prod, 0));
    LORRAX_CUDA_CHECK(cudaEventDestroy(ev_prod));

    LORRAX_CUDA_CHECK(cudaMemcpyAsync(d_dst, ctx->pinned_buf, bytes,
                                      cudaMemcpyHostToDevice, ctx->stream));

    cudaEvent_t ev_done;
    LORRAX_CUDA_CHECK(cudaEventCreateWithFlags(&ev_done, cudaEventDisableTiming));
    LORRAX_CUDA_CHECK(cudaEventRecord(ev_done, ctx->stream));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(xla_stream, ev_done, 0));
    LORRAX_CUDA_CHECK(cudaEventSynchronize(ev_done));
    LORRAX_CUDA_CHECK(cudaEventDestroy(ev_done));

    return ffi::Error::Success();
}

static ffi::Error ReadDispatch(
    cudaStream_t stream,
    ffi::Result<ffi::AnyBuffer> A_out,
    int64_t ctx_handle,
    int64_t ds_id,
    ffi::Span<const int64_t> offset_base,
    ffi::Span<const int64_t> mesh_shape,
    ffi::Span<const int64_t> axis_for_dim)
{
    auto* ctx = reinterpret_cast<PhdfCtx*>(ctx_handle);
    if (!ctx) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "phdf5 read: ctx_handle is null");
    }

    const auto dims = A_out->dimensions();
    const size_t N = dims.size();
    if (offset_base.size() != N || axis_for_dim.size() != N) {
        std::ostringstream os;
        os << "phdf5 read: rank mismatch  A.ndim=" << N
           << " offset_base.size=" << offset_base.size()
           << " axis_for_dim.size=" << axis_for_dim.size();
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }

    std::vector<int64_t> coord = unravel_rank(ctx->rank, mesh_shape);
    std::vector<hsize_t> offset(N), count(N);
    for (size_t d = 0; d < N; ++d) {
        count[d] = (hsize_t)dims[d];
        offset[d] = (hsize_t)offset_base[d];
        int64_t ax = axis_for_dim[d];
        if (ax >= 0) {
            if ((size_t)ax >= mesh_shape.size()) {
                std::ostringstream os;
                os << "phdf5 read: axis_for_dim[" << d << "]=" << ax
                   << " out of range for mesh_shape.size=" << mesh_shape.size();
                return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
            }
            offset[d] += coord[ax] * count[d];
        }
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
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("ctx_handle")
        .Attr<int64_t>("ds_id")
        .Attr<xla::ffi::Span<const int64_t>>("offset_base")
        .Attr<xla::ffi::Span<const int64_t>>("mesh_shape")
        .Attr<xla::ffi::Span<const int64_t>>("axis_for_dim"));
