// write_ffi.cc — XLA FFI handler that writes a sharded JAX array to a
// hyperslab of an open parallel-HDF5 dataset.
//
// Structure mirrors cusolvermp/cpp/eigh_ffi.cc:
//   WriteImpl<T>    — dtype-specialized body
//   WriteDispatch   — shape+dtype validation, switch over element type
//   XLA_FFI_DEFINE_HANDLER_SYMBOL(PhdfWriteFfi, WriteDispatch, Bind())
//
// The write is **synchronous from the FFI's POV** — it blocks the
// calling host thread until H5Dwrite returns.  The CUDA stream is
// unaffected, so device compute queued before the FFI keeps running
// throughout the write.  See the plan for the "async-enough" argument.
//
// Future upgrade path (see README):
// - Async VOL: replace H5Dwrite with H5Dwrite_async + event set,
//   grow pinned_buf into a pool keyed by outstanding tokens.
// - GDS (cuFile): replace D2H + pinned staging with a direct
//   GPU→disk VFD once cuFile stabilizes in our environment.

#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "xla/ffi/api/ffi.h"

#include "../../common/cpp/ffi_helpers.h"
#include "ctx.h"
#include "phdf5_interface.h"

namespace lorrax_ffi::phdf5 {

namespace ffi = ::xla::ffi;

// Per-dtype helpers live in phdf5_interface.h — h5_native_type<T>(),
// complex64_type(), complex128_type().  Pulled into this TU via the
// include above; call as dt::h5_native_type<T>().

// ---- Impl ----------------------------------------------------------------
// ds_id is the hid_t of a dataset previously created via
// lrx_phdf5_ensure_dataset (Python-side collective).  We reuse it here
// for the write — no string attr plumbing needed.
template <typename T>
static ffi::Error WriteImpl(
    cudaStream_t xla_stream,
    PhdfCtx* ctx,
    const T* d_src,
    int64_t local_rows, int64_t local_cols,
    int64_t global_rows, int64_t global_cols,
    int64_t row_start,  int64_t col_start,
    hid_t ds_id)
{
    const size_t elem_bytes  = sizeof(T);
    const size_t n_local_elts = (size_t)local_rows * (size_t)local_cols;
    const size_t bytes       = n_local_elts * elem_bytes;

    // (1) pinned buffer sized to at least `bytes`.
    if (!ensure_pinned(ctx, bytes)) {
        std::ostringstream os;
        os << "phdf5 write: cudaMallocHost(" << bytes << ") failed";
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, os.str());
    }

    // (2) ctx stream waits for XLA's producers.
    cudaEvent_t ev_prod;
    LORRAX_CUDA_CHECK(cudaEventCreateWithFlags(&ev_prod, cudaEventDisableTiming));
    LORRAX_CUDA_CHECK(cudaEventRecord(ev_prod, xla_stream));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(ctx->stream, ev_prod, 0));
    LORRAX_CUDA_CHECK(cudaEventDestroy(ev_prod));

    // (3) async D2H memcpy on ctx stream.
    LORRAX_CUDA_CHECK(cudaMemcpyAsync(ctx->pinned_buf, d_src, bytes,
                                      cudaMemcpyDeviceToHost, ctx->stream));

    // (4) record a completion event so XLA can resume its stream safely
    //     (the input device buffer becomes free to reuse once the memcpy
    //     is queued — actually it's free as soon as XLA's stream waits
    //     for OUR event; the memcpy reads the source buffer async, but
    //     XLA's stream won't dispatch a conflicting write before the
    //     event resolves).
    cudaEvent_t ev_done;
    LORRAX_CUDA_CHECK(cudaEventCreateWithFlags(&ev_done, cudaEventDisableTiming));
    LORRAX_CUDA_CHECK(cudaEventRecord(ev_done, ctx->stream));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(xla_stream, ev_done, 0));

    // (5) block the host thread on the memcpy event — pinned_buf must
    //     be valid before H5Dwrite dereferences it.  Sub-ms for typical
    //     shard sizes.
    LORRAX_CUDA_CHECK(cudaEventSynchronize(ev_done));
    LORRAX_CUDA_CHECK(cudaEventDestroy(ev_done));

    // (6) dataset was created Python-side via lrx_phdf5_ensure_dataset;
    // we just use the hid_t.  Validate it's still live.
    hid_t dset = ds_id;
    hid_t native_type = dt::h5_native_type<T>();
    if (dset < 0 || H5Iis_valid(dset) <= 0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
            "phdf5 write: ds_id is invalid (register via "
            "phdf5_ensure_dataset before writing)");
    }

    // (7) select the rank's hyperslab.
    hsize_t offset[2] = { (hsize_t)row_start,  (hsize_t)col_start };
    hsize_t count[2]  = { (hsize_t)local_rows, (hsize_t)local_cols };
    hid_t filespace = H5Dget_space(dset);
    if (filespace < 0) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 write: H5Dget_space failed");
    }
    if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr,
                             count, nullptr) < 0) {
        H5Sclose(filespace);
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 write: H5Sselect_hyperslab failed");
    }
    hid_t memspace = H5Screate_simple(2, count, nullptr);
    if (memspace < 0) {
        H5Sclose(filespace);
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 write: H5Screate_simple(memspace) failed");
    }

    // (8) blocking collective-MPI-IO write.  Host thread is parked
    //     here; device stream keeps running.
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
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> token_out,  // (1,) int32
    int64_t ctx_handle,
    int64_t ds_id,                                           // hid_t
    int64_t n_rows, int64_t n_cols)
{
    auto* ctx = reinterpret_cast<PhdfCtx*>(ctx_handle);
    if (!ctx) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "phdf5 write: ctx_handle is null");
    }

    auto dims = A.dimensions();
    if (dims.size() != 2) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "phdf5 write: A must be rank-2 inside shard_map");
    }
    const int64_t local_rows = dims[0];
    const int64_t local_cols = dims[1];

    // Row-major mesh convention: rank r owns (row=r/q, col=r%q).  This
    // matches jax.sharding.NamedSharding(P('x','y')) default on a
    // (p,q) Mesh.
    const int64_t my_row    = ctx->rank / ctx->q;
    const int64_t my_col    = ctx->rank % ctx->q;
    const int64_t row_start = my_row * local_rows;
    const int64_t col_start = my_col * local_cols;

    const auto dtype = A.element_type();
    switch (dtype) {
        case ffi::DataType::F32:
            return WriteImpl<float>(stream, ctx,
                static_cast<const float*>(A.untyped_data()),
                local_rows, local_cols, n_rows, n_cols,
                row_start, col_start, (hid_t)ds_id);
        case ffi::DataType::F64:
            return WriteImpl<double>(stream, ctx,
                static_cast<const double*>(A.untyped_data()),
                local_rows, local_cols, n_rows, n_cols,
                row_start, col_start, (hid_t)ds_id);
        case ffi::DataType::S32:
            return WriteImpl<int32_t>(stream, ctx,
                static_cast<const int32_t*>(A.untyped_data()),
                local_rows, local_cols, n_rows, n_cols,
                row_start, col_start, (hid_t)ds_id);
        case ffi::DataType::S64:
            return WriteImpl<int64_t>(stream, ctx,
                static_cast<const int64_t*>(A.untyped_data()),
                local_rows, local_cols, n_rows, n_cols,
                row_start, col_start, (hid_t)ds_id);
        case ffi::DataType::C64:
            return WriteImpl<std::complex<float>>(stream, ctx,
                static_cast<const std::complex<float>*>(A.untyped_data()),
                local_rows, local_cols, n_rows, n_cols,
                row_start, col_start, (hid_t)ds_id);
        case ffi::DataType::C128:
            return WriteImpl<std::complex<double>>(stream, ctx,
                static_cast<const std::complex<double>*>(A.untyped_data()),
                local_rows, local_cols, n_rows, n_cols,
                row_start, col_start, (hid_t)ds_id);
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
        .Arg<xla::ffi::AnyBuffer>()                         // A (local shard)
        .Ret<xla::ffi::Buffer<xla::ffi::DataType::S32>>()   // token (1,) i32
        .Attr<int64_t>("ctx_handle")
        .Attr<int64_t>("ds_id")                             // hid_t, from ensure_dataset
        .Attr<int64_t>("n_rows")
        .Attr<int64_t>("n_cols"));
