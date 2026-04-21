// batched_potrf_ffi.cc — XLA FFI handler for a stack of cusolverMpPotrf
// calls sharing the full (Px, Py) process grid.
//
// No native batched potrf in cuSOLVERMp (confirmed through v0.8.0).
// The "batching" is a C++ for-loop over q; the handle, grid, matrix
// descriptor, and workspace are built once per FFI call and reused.
//
// Sharding contract (matches distributed_eigh):
//   A : shape (Nq, N, N), P(None, 'x', 'y') on (Px, Py) mesh.
//       Per-rank local: (Nq, N/Px, N/Py) row-major.
//       Python pre-transposes the inner two dims → (Nq, N/Py, N/Px)
//       row-major ≡ (N/Px, N/Py) col-major per slice, which is what
//       cuSOLVERMp's grid (Px, Py) expects with mb=N/Px, nb=N/Py,
//       lld=N/Px.
//
// Context (`ctx_handle`) is the world-wide LorraxCusolverMpCtx (the
// same one used by distributed_eigh).  Must be created with
// `grid_layout_col_major=false` so cuSOLVERMp's rank→tile mapping
// matches JAX's row-major mesh reshape (rank = x_idx*Py + y_idx).

#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#include <cuda_runtime.h>
#include <cusolverMp.h>

#include "xla/ffi/api/ffi.h"

#include "../../common/cpp/ffi_helpers.h"
#include "cusolvermp_interface.h"
#include "ctx.h"

namespace lorrax_ffi::cusolvermp_batched_potrf {

namespace ffi = ::xla::ffi;
using lorrax_ffi::cusolvermp::LorraxCusolverMpCtx;
using lorrax_ffi::cusolvermp::ensure_workspace;
namespace mp = lorrax_ffi::cusolvermp::mp;

static ffi::Error cross_stream_wait_pooled(cudaStream_t waiter,
                                           cudaStream_t signaller,
                                           cudaEvent_t  ev) {
    LORRAX_CUDA_CHECK(cudaEventRecord(ev, signaller));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(waiter, ev, 0));
    return ffi::Error::Success();
}

template <typename T>
static ffi::Error BatchedPotrfImpl(
    int64_t nq, int64_t n, int64_t mb, int64_t nb,
    cudaStream_t xla_stream,
    LorraxCusolverMpCtx* ctx,
    const T* d_A_in, T* d_L_out)
{
    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        ctx->stream, xla_stream, ctx->ev_xla_in));

    const int Px = ctx->p;
    const int Py = ctx->q;
    const int64_t lld_A      = (n + Px - 1) / Px;   // local rows per rank
    const int64_t local_cols = (n + Py - 1) / Py;   // local cols per rank
    const int64_t slice_elems = lld_A * local_cols;

    // Copy full batch into output, then factor each slice in place.
    LORRAX_CUDA_CHECK(cudaMemcpyAsync(
        d_L_out, d_A_in,
        nq * slice_elems * sizeof(T),
        cudaMemcpyDeviceToDevice, ctx->stream));

    cusolverMpMatrixDescriptor_t descA = nullptr;
    LORRAX_LIB_CHECK(
        cusolverMpCreateMatrixDesc(&descA, ctx->grid,
                                   mp::CudaDataTypeOf<T>::value,
                                   n, n, mb, nb, 0, 0, lld_A),
        CUSOLVER_STATUS_SUCCESS, "cusolverMpCreateMatrixDesc(A)");

    size_t d_ws = 0, h_ws = 0;
    cusolverStatus_t mp_st = mp::PotrfBufferSize<T>(
        ctx->handle, CUBLAS_FILL_MODE_LOWER, n,
        d_L_out, 1, 1, descA, &d_ws, &h_ws);
    if (mp_st != CUSOLVER_STATUS_SUCCESS) {
        cusolverMpDestroyMatrixDesc(descA);
        std::ostringstream os;
        os << "cusolverMpPotrf_bufferSize failed: status=" << (int)mp_st;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    try {
        ensure_workspace(ctx, d_ws, h_ws);
    } catch (const std::exception& ex) {
        cusolverMpDestroyMatrixDesc(descA);
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, ex.what());
    }

    for (int64_t q = 0; q < nq; ++q) {
        T* slice_ptr = d_L_out + q * slice_elems;
        mp_st = mp::Potrf<T>(
            ctx->handle, CUBLAS_FILL_MODE_LOWER, n,
            slice_ptr, 1, 1, descA,
            ctx->d_workspace, ctx->d_workspace_bytes,
            ctx->h_workspace, ctx->h_workspace_bytes,
            ctx->d_info);
        if (mp_st != CUSOLVER_STATUS_SUCCESS) {
            cusolverMpDestroyMatrixDesc(descA);
            std::ostringstream os;
            os << "cusolverMpPotrf (q=" << q << ") failed: status=" << (int)mp_st;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
    }

    cusolverMpDestroyMatrixDesc(descA);

    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        xla_stream, ctx->stream, ctx->ev_ctx_out));
    return ffi::Error::Success();
}

static ffi::Error BatchedPotrfDispatch(
    cudaStream_t stream,
    ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> L_out,
    int64_t nq, int64_t n, int64_t mb, int64_t nb,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "batched_potrf: ctx_handle is null");
    }
    const auto dtype = A.element_type();
    if (L_out->element_type() != dtype) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "batched_potrf: L output dtype must match A");
    }
    switch (dtype) {
        case ffi::DataType::F64:
            return BatchedPotrfImpl<double>(
                nq, n, mb, nb, stream, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<double*>(L_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return BatchedPotrfImpl<C128>(
                nq, n, mb, nb, stream, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<C128*>(L_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "batched_potrf: unsupported dtype " << (int)dtype
               << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::cusolvermp_batched_potrf

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CusolverMpBatchedPotrfFfi,
    lorrax_ffi::cusolvermp_batched_potrf::BatchedPotrfDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()      // A
        .Ret<xla::ffi::AnyBuffer>()      // L
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("mb")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_handle"));
