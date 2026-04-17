// read_ffi.cc — XLA FFI handler that reads a hyperslab of an open
// parallel-HDF5 dataset into a sharded JAX array (mirror of write_ffi.cc).
//
// Same structure as the writer:
//   ReadImpl<T>    — dtype-specialized body
//   ReadDispatch   — shape+dtype validation, switch over element type
//   XLA_FFI_DEFINE_HANDLER_SYMBOL(PhdfReadFfi, ReadDispatch, Bind())
//
// The read is **synchronous from the FFI's POV** (blocking H5Dread on the
// host thread).  Inside a shard_map with ``out_specs=P('x','y')`` each rank
// returns its local shard; XLA stitches the shards into a globally-sharded
// output.  The CUDA stream is unaffected — device compute queued before
// the FFI keeps running.

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

// ---- Impl ----------------------------------------------------------------
// ds_id is the hid_t of an already-opened dataset (via
// lrx_phdf5_open_dataset_ro on the Python side).  We select the rank's
// hyperslab, H5Dread into pinned host memory, then H2D-memcpy to the
// XLA-provided output buffer.
template <typename T>
static ffi::Error ReadImpl(
    cudaStream_t xla_stream,
    PhdfCtx* ctx,
    T* d_dst,                                 // XLA-owned device output buffer
    int64_t local_rows, int64_t local_cols,
    int64_t /*global_rows*/, int64_t /*global_cols*/,
    int64_t row_start,  int64_t col_start,
    hid_t ds_id)
{
    const size_t elem_bytes   = sizeof(T);
    const size_t n_local_elts = (size_t)local_rows * (size_t)local_cols;
    const size_t bytes        = n_local_elts * elem_bytes;

    // (1) pinned buffer sized to at least `bytes`.
    if (!ensure_pinned(ctx, bytes)) {
        std::ostringstream os;
        os << "phdf5 read: cudaMallocHost(" << bytes << ") failed";
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, os.str());
    }

    // (2) validate cached dataset id.
    hid_t dset = ds_id;
    hid_t native_type = dt::h5_native_type<T>();
    if (dset < 0 || H5Iis_valid(dset) <= 0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
            "phdf5 read: ds_id is invalid (register via "
            "phdf5_open_dataset_ro before reading)");
    }

    // (3) select the rank's hyperslab in the file.
    hsize_t offset[2] = { (hsize_t)row_start,  (hsize_t)col_start };
    hsize_t count[2]  = { (hsize_t)local_rows, (hsize_t)local_cols };
    hid_t filespace = H5Dget_space(dset);
    if (filespace < 0) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 read: H5Dget_space failed");
    }
    if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr,
                             count, nullptr) < 0) {
        H5Sclose(filespace);
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 read: H5Sselect_hyperslab failed");
    }
    hid_t memspace = H5Screate_simple(2, count, nullptr);
    if (memspace < 0) {
        H5Sclose(filespace);
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 read: H5Screate_simple(memspace) failed");
    }

    // (4) blocking collective-MPI-IO read into pinned host buffer.  Host
    //     thread is parked here; device stream keeps running.
    hid_t dxpl = ctx->use_collective ? ctx->dxpl_coll : ctx->dxpl_indep;
    herr_t st = H5Dread(dset, native_type, memspace, filespace, dxpl,
                        ctx->pinned_buf);
    H5Sclose(memspace);
    H5Sclose(filespace);
    if (st < 0) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "phdf5 read: H5Dread failed");
    }

    // (5) H2D memcpy: pinned host → XLA's device output buffer on the ctx
    //     stream.  Before issuing, make ctx stream wait on xla_stream so
    //     any prior producer on the output buffer has finished.
    cudaEvent_t ev_prod;
    LORRAX_CUDA_CHECK(cudaEventCreateWithFlags(&ev_prod, cudaEventDisableTiming));
    LORRAX_CUDA_CHECK(cudaEventRecord(ev_prod, xla_stream));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(ctx->stream, ev_prod, 0));
    LORRAX_CUDA_CHECK(cudaEventDestroy(ev_prod));

    LORRAX_CUDA_CHECK(cudaMemcpyAsync(d_dst, ctx->pinned_buf, bytes,
                                      cudaMemcpyHostToDevice, ctx->stream));

    // (6) cross-stream ordering: XLA's stream must wait until the H2D
    //     memcpy has landed before any downstream op reads d_dst.
    cudaEvent_t ev_done;
    LORRAX_CUDA_CHECK(cudaEventCreateWithFlags(&ev_done, cudaEventDisableTiming));
    LORRAX_CUDA_CHECK(cudaEventRecord(ev_done, ctx->stream));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(xla_stream, ev_done, 0));

    // (7) block the host thread on H2D completion so the pinned buffer
    //     is safe to reuse on the next call.  Matches the write path's
    //     single-pinned-buffer invariant: only one read/write in flight.
    LORRAX_CUDA_CHECK(cudaEventSynchronize(ev_done));
    LORRAX_CUDA_CHECK(cudaEventDestroy(ev_done));

    return ffi::Error::Success();
}

// ---- Dispatch ------------------------------------------------------------
static ffi::Error ReadDispatch(
    cudaStream_t stream,
    ffi::Result<ffi::AnyBuffer> A_out,
    int64_t ctx_handle,
    int64_t ds_id,                                           // hid_t
    int64_t n_rows, int64_t n_cols)
{
    auto* ctx = reinterpret_cast<PhdfCtx*>(ctx_handle);
    if (!ctx) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "phdf5 read: ctx_handle is null");
    }

    auto dims = A_out->dimensions();
    if (dims.size() != 2) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "phdf5 read: A must be rank-2 inside shard_map");
    }
    const int64_t local_rows = dims[0];
    const int64_t local_cols = dims[1];

    // Row-major mesh convention matches write_ffi.cc.
    const int64_t my_row    = ctx->rank / ctx->q;
    const int64_t my_col    = ctx->rank % ctx->q;
    const int64_t row_start = my_row * local_rows;
    const int64_t col_start = my_col * local_cols;

    const auto dtype = A_out->element_type();
    switch (dtype) {
        case ffi::DataType::F32:
            return ReadImpl<float>(stream, ctx,
                static_cast<float*>(A_out->untyped_data()),
                local_rows, local_cols, n_rows, n_cols,
                row_start, col_start, (hid_t)ds_id);
        case ffi::DataType::F64:
            return ReadImpl<double>(stream, ctx,
                static_cast<double*>(A_out->untyped_data()),
                local_rows, local_cols, n_rows, n_cols,
                row_start, col_start, (hid_t)ds_id);
        case ffi::DataType::S32:
            return ReadImpl<int32_t>(stream, ctx,
                static_cast<int32_t*>(A_out->untyped_data()),
                local_rows, local_cols, n_rows, n_cols,
                row_start, col_start, (hid_t)ds_id);
        case ffi::DataType::S64:
            return ReadImpl<int64_t>(stream, ctx,
                static_cast<int64_t*>(A_out->untyped_data()),
                local_rows, local_cols, n_rows, n_cols,
                row_start, col_start, (hid_t)ds_id);
        case ffi::DataType::C64:
            return ReadImpl<std::complex<float>>(stream, ctx,
                static_cast<std::complex<float>*>(A_out->untyped_data()),
                local_rows, local_cols, n_rows, n_cols,
                row_start, col_start, (hid_t)ds_id);
        case ffi::DataType::C128:
            return ReadImpl<std::complex<double>>(stream, ctx,
                static_cast<std::complex<double>*>(A_out->untyped_data()),
                local_rows, local_cols, n_rows, n_cols,
                row_start, col_start, (hid_t)ds_id);
        default: {
            std::ostringstream os;
            os << "phdf5 read: unsupported dtype " << static_cast<int>(dtype);
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::phdf5

// ---- FFI binding ---------------------------------------------------------
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PhdfReadFfi, lorrax_ffi::phdf5::ReadDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ret<xla::ffi::AnyBuffer>()                         // A (local shard)
        .Attr<int64_t>("ctx_handle")
        .Attr<int64_t>("ds_id")                             // hid_t, from open_dataset_ro
        .Attr<int64_t>("n_rows")
        .Attr<int64_t>("n_cols"));
