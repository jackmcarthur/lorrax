// batched_potrs_ffi.cc — XLA FFI handler for a batched cuSOLVERMp potrs
// (combined L^{-H} L^{-1} B, i.e. solve A X = B given factored A = L L^H).
//
// Same structure as batched_potrf_ffi.cc: C++ for-loop over local batch
// on a per-X-row sub-comm; grid + descriptors reused across iterations.
//
// Sharding contract:
//   L : (Nbatch, N, N)    P('x','y',None) — batched cholesky output,
//                         already in col-major (via Python inner-dim
//                         transpose at potrf time).
//   B : (Nbatch, N, Mrhs) P('x', None, 'y') → Python transposes inner
//                         dims → (Nbatch_local, Mrhs/Py, N) row-major
//                         ≡ (N, Mrhs/Py) col-major per slice.
//   X : same shape/layout as B; overwritten in place by potrs.
//
// One bufferSize query at Mrhs-wide shape; if the caller loops over
// RHS chunks with different Mrhs, each chunk gets its own FFI call.

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

namespace lorrax_ffi::cusolvermp_batched_potrs {

namespace ffi = ::xla::ffi;
using lorrax_ffi::cusolvermp::LorraxCusolverMpSubRowCtx;
using lorrax_ffi::cusolvermp::ensure_subrow_workspace;
namespace mp = lorrax_ffi::cusolvermp::mp;

// See batched_potrf_ffi.cc for the rationale for the pooled-event pattern.
static ffi::Error cross_stream_wait_pooled(cudaStream_t waiter,
                                           cudaStream_t signaller,
                                           cudaEvent_t  ev) {
    LORRAX_CUDA_CHECK(cudaEventRecord(ev, signaller));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(waiter, ev, 0));
    return ffi::Error::Success();
}

template <typename T>
static ffi::Error BatchedPotrsImpl(
    int64_t nbatch_local, int64_t n, int64_t mrhs,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    cudaStream_t xla_stream,
    LorraxCusolverMpSubRowCtx* ctx,
    const T* d_L, const T* d_B_in, T* d_X_out)
{
    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        ctx->stream, xla_stream, ctx->ev_xla_in));

    const int64_t Py   = ctx->Py;
    const int64_t lldA = n;
    const int64_t lldB = n;
    const int64_t local_colsA = (n    + Py - 1) / Py;
    const int64_t local_colsB = (mrhs + Py - 1) / Py;
    const int64_t A_slice = n * local_colsA;
    const int64_t B_slice = n * local_colsB;

    // Copy B into X; potrs overwrites in place.
    LORRAX_CUDA_CHECK(cudaMemcpyAsync(
        d_X_out, d_B_in,
        nbatch_local * B_slice * sizeof(T),
        cudaMemcpyDeviceToDevice, ctx->stream));

    cusolverMpMatrixDescriptor_t descA = nullptr, descB = nullptr;
    LORRAX_LIB_CHECK(
        cusolverMpCreateMatrixDesc(&descA, ctx->grid,
                                   mp::CudaDataTypeOf<T>::value,
                                   n, n, mb_a, nb_a, 0, 0, lldA),
        CUSOLVER_STATUS_SUCCESS, "cusolverMpCreateMatrixDesc(A)");
    LORRAX_LIB_CHECK(
        cusolverMpCreateMatrixDesc(&descB, ctx->grid,
                                   mp::CudaDataTypeOf<T>::value,
                                   n, mrhs, mb_b, nb_b, 0, 0, lldB),
        CUSOLVER_STATUS_SUCCESS, "cusolverMpCreateMatrixDesc(B)");

    size_t d_ws = 0, h_ws = 0;
    cusolverStatus_t mp_st = mp::PotrsBufferSize<T>(
        ctx->handle, CUBLAS_FILL_MODE_LOWER, n, mrhs,
        d_L,      1, 1, descA,
        d_X_out,  1, 1, descB,
        &d_ws, &h_ws);
    if (mp_st != CUSOLVER_STATUS_SUCCESS) {
        cusolverMpDestroyMatrixDesc(descA);
        cusolverMpDestroyMatrixDesc(descB);
        std::ostringstream os;
        os << "cusolverMpPotrs_bufferSize failed: status=" << (int)mp_st;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    try {
        ensure_subrow_workspace(ctx, d_ws, h_ws);
    } catch (const std::exception& ex) {
        cusolverMpDestroyMatrixDesc(descA);
        cusolverMpDestroyMatrixDesc(descB);
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, ex.what());
    }

    // d_info never read per iter (see batched_potrf_ffi.cc).
    for (int64_t b = 0; b < nbatch_local; ++b) {
        T* A_slice_ptr = const_cast<T*>(d_L) + b * A_slice;
        T* X_slice_ptr = d_X_out + b * B_slice;
        mp_st = mp::Potrs<T>(
            ctx->handle, CUBLAS_FILL_MODE_LOWER, n, mrhs,
            A_slice_ptr, 1, 1, descA,
            X_slice_ptr, 1, 1, descB,
            ctx->d_workspace, ctx->d_workspace_bytes,
            ctx->h_workspace, ctx->h_workspace_bytes,
            ctx->d_info);
        if (mp_st != CUSOLVER_STATUS_SUCCESS) {
            cusolverMpDestroyMatrixDesc(descA);
            cusolverMpDestroyMatrixDesc(descB);
            std::ostringstream os;
            os << "cusolverMpPotrs (batch " << b << ") failed: status=" << (int)mp_st;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
    }

    cusolverMpDestroyMatrixDesc(descA);
    cusolverMpDestroyMatrixDesc(descB);

    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        xla_stream, ctx->stream, ctx->ev_ctx_out));
    return ffi::Error::Success();
}

static ffi::Error BatchedPotrsDispatch(
    cudaStream_t stream,
    ffi::AnyBuffer L,
    ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> X_out,
    int64_t nbatch_local, int64_t n, int64_t mrhs,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<LorraxCusolverMpSubRowCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "batched_potrs: ctx_handle is null");
    }
    const auto dtype = L.element_type();
    if (B.element_type() != dtype || X_out->element_type() != dtype) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "batched_potrs: L, B, X must share dtype");
    }
    switch (dtype) {
        case ffi::DataType::F64:
            return BatchedPotrsImpl<double>(
                nbatch_local, n, mrhs, mb_a, nb_a, mb_b, nb_b, stream, ctx,
                static_cast<const double*>(L.untyped_data()),
                static_cast<const double*>(B.untyped_data()),
                static_cast<double*>(X_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return BatchedPotrsImpl<C128>(
                nbatch_local, n, mrhs, mb_a, nb_a, mb_b, nb_b, stream, ctx,
                static_cast<const C128*>(L.untyped_data()),
                static_cast<const C128*>(B.untyped_data()),
                static_cast<C128*>(X_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "batched_potrs: unsupported dtype " << (int)dtype
               << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::cusolvermp_batched_potrs

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CusolverMpBatchedPotrsFfi,
    lorrax_ffi::cusolvermp_batched_potrs::BatchedPotrsDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()      // L (factored)
        .Arg<xla::ffi::AnyBuffer>()      // B (RHS)
        .Ret<xla::ffi::AnyBuffer>()      // X (solution)
        .Attr<int64_t>("nbatch_local")
        .Attr<int64_t>("n")
        .Attr<int64_t>("mrhs")
        .Attr<int64_t>("mb_a")
        .Attr<int64_t>("nb_a")
        .Attr<int64_t>("mb_b")
        .Attr<int64_t>("nb_b")
        .Attr<int64_t>("ctx_handle"));
