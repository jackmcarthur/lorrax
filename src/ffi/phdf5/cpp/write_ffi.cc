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

// ---- Impl ----------------------------------------------------------------
template <typename T>
static ffi::Error WriteImpl(
    cudaStream_t xla_stream,
    PhdfCtx* ctx,
    const T* d_src,
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
        os << "phdf5 write: cudaMallocHost(" << bytes << ") failed";
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, os.str());
    }

    cudaEvent_t ev_prod;
    LORRAX_CUDA_CHECK(cudaEventCreateWithFlags(&ev_prod, cudaEventDisableTiming));
    LORRAX_CUDA_CHECK(cudaEventRecord(ev_prod, xla_stream));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(ctx->stream, ev_prod, 0));
    LORRAX_CUDA_CHECK(cudaEventDestroy(ev_prod));

    LORRAX_CUDA_CHECK(cudaMemcpyAsync(ctx->pinned_buf, d_src, bytes,
                                      cudaMemcpyDeviceToHost, ctx->stream));

    cudaEvent_t ev_done;
    LORRAX_CUDA_CHECK(cudaEventCreateWithFlags(&ev_done, cudaEventDisableTiming));
    LORRAX_CUDA_CHECK(cudaEventRecord(ev_done, ctx->stream));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(xla_stream, ev_done, 0));
    LORRAX_CUDA_CHECK(cudaEventSynchronize(ev_done));
    LORRAX_CUDA_CHECK(cudaEventDestroy(ev_done));

    hid_t dset = ds_id;
    hid_t native_type = dt::h5_native_type<T>();
    if (dset < 0 || H5Iis_valid(dset) <= 0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
            "phdf5 write: ds_id is invalid");
    }

    hid_t filespace = H5Dget_space(dset);
    if (filespace < 0) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 write: H5Dget_space failed");
    }
    if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                             offset.data(), nullptr,
                             count.data(),  nullptr) < 0) {
        H5Sclose(filespace);
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 write: H5Sselect_hyperslab failed");
    }
    hid_t memspace = H5Screate_simple(rank, count.data(), nullptr);
    if (memspace < 0) {
        H5Sclose(filespace);
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 write: H5Screate_simple(memspace) failed");
    }

    hid_t dxpl = ctx->use_collective ? ctx->dxpl_coll : ctx->dxpl_indep;
    herr_t st = H5Dwrite(dset, native_type, memspace, filespace, dxpl,
                         ctx->pinned_buf);

    H5Sclose(memspace);
    H5Sclose(filespace);

    if (st < 0) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 write: H5Dwrite failed");
    }
    return ffi::Error::Success();
}

// ---- Dispatch ------------------------------------------------------------
static ffi::Error WriteDispatch(
    cudaStream_t stream,
    ffi::AnyBuffer A,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> token_out,
    int64_t ctx_handle,
    int64_t ds_id,
    ffi::Span<const int64_t> offset_base,
    ffi::Span<const int64_t> mesh_shape,
    ffi::Span<const int64_t> axis_count_per_dim,
    ffi::Span<const int64_t> axis_flat)
{
    auto* ctx = reinterpret_cast<PhdfCtx*>(ctx_handle);
    if (!ctx) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "phdf5 write: ctx_handle is null");
    }

    const auto dims = A.dimensions();
    const size_t N = dims.size();
    if (offset_base.size() != N || axis_count_per_dim.size() != N) {
        std::ostringstream os;
        os << "phdf5 write: rank mismatch  A.ndim=" << N
           << " offset_base.size=" << offset_base.size()
           << " axis_count_per_dim.size=" << axis_count_per_dim.size();
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }

    std::vector<int64_t> coord = unravel_rank(ctx->rank, mesh_shape);
    std::vector<hsize_t> offset(N), count(N);
    size_t flat_idx = 0;
    for (size_t d = 0; d < N; ++d) {
        count[d] = (hsize_t)dims[d];
        offset[d] = (hsize_t)offset_base[d];
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
                return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
            }
            rank_coord += coord[ax] * stride_acc;
            stride_acc *= mesh_shape[ax];
        }
        offset[d] += rank_coord * count[d];
        flat_idx += (size_t)na;
    }

    const auto dtype = A.element_type();
    switch (dtype) {
        case ffi::DataType::F32:
            return WriteImpl<float>(stream, ctx,
                static_cast<const float*>(A.untyped_data()),
                offset, count, (hid_t)ds_id);
        case ffi::DataType::F64:
            return WriteImpl<double>(stream, ctx,
                static_cast<const double*>(A.untyped_data()),
                offset, count, (hid_t)ds_id);
        case ffi::DataType::S32:
            return WriteImpl<int32_t>(stream, ctx,
                static_cast<const int32_t*>(A.untyped_data()),
                offset, count, (hid_t)ds_id);
        case ffi::DataType::S64:
            return WriteImpl<int64_t>(stream, ctx,
                static_cast<const int64_t*>(A.untyped_data()),
                offset, count, (hid_t)ds_id);
        case ffi::DataType::C64:
            return WriteImpl<std::complex<float>>(stream, ctx,
                static_cast<const std::complex<float>*>(A.untyped_data()),
                offset, count, (hid_t)ds_id);
        case ffi::DataType::C128:
            return WriteImpl<std::complex<double>>(stream, ctx,
                static_cast<const std::complex<double>*>(A.untyped_data()),
                offset, count, (hid_t)ds_id);
        default: {
            std::ostringstream os;
            os << "phdf5 write: unsupported dtype " << static_cast<int>(dtype);
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
    (void)token_out;
}

}  // namespace lorrax_ffi::phdf5

// ---- FFI binding ---------------------------------------------------------
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PhdfWriteFfi, lorrax_ffi::phdf5::WriteDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::Buffer<xla::ffi::DataType::S32>>()
        .Attr<int64_t>("ctx_handle")
        .Attr<int64_t>("ds_id")
        .Attr<xla::ffi::Span<const int64_t>>("offset_base")
        .Attr<xla::ffi::Span<const int64_t>>("mesh_shape")
        .Attr<xla::ffi::Span<const int64_t>>("axis_count_per_dim")
        .Attr<xla::ffi::Span<const int64_t>>("axis_flat"));
