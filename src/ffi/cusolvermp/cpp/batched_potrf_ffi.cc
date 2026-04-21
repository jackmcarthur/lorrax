// batched_potrf_ffi.cc — XLA FFI handler for a batched cuSOLVERMp potrf
// on a per-X-row sub-comm.
//
// cuSOLVERMp has no native batched potrf (confirmed against headers
// through v0.8.0, 2026-04).  The "batching" here is a C++ for-loop
// over the local batch dimension — the sub-comm, handle, grid, and
// matrix descriptor are all created once per FFI call and reused
// across iterations, so per-iteration overhead is just the kernel
// dispatch + collective traffic.
//
// Sharding contract (matches src/ffi/slate/batched.py):
//   A : shape (Nbatch, N, N), P('x', None, 'y') on (Px, Py) mesh.
//       Per-rank local:   (Nbatch/Px, N, N/Py) row-major.
//       Python transposes inner dims → (Nbatch/Px, N/Py, N) row-major
//       ≡ (N, N/Py) col-major per slice — what cuSOLVERMp expects with
//       grid (1, Py), lld = N.
//
// The context (ctx_handle) must be a SubRowCtx: one per-X-row sub-comm
// of size Py, grid (1, Py).  See context.cc / subrow_context.

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
using lorrax_ffi::cusolvermp::LorraxCusolverMpSubRowCtx;
using lorrax_ffi::cusolvermp::ensure_subrow_workspace;
namespace mp = lorrax_ffi::cusolvermp::mp;

// Record on signaller, wait on waiter — no host-level stream sync.
static ffi::Error cross_stream_wait(cudaStream_t waiter,
                                    cudaStream_t signaller) {
    cudaEvent_t ev;
    LORRAX_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    LORRAX_CUDA_CHECK(cudaEventRecord(ev, signaller));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(waiter, ev, 0));
    LORRAX_CUDA_CHECK(cudaEventDestroy(ev));
    return ffi::Error::Success();
}

template <typename T>
static ffi::Error BatchedPotrfImpl(
    int64_t nbatch_local, int64_t n, int64_t mb, int64_t nb,
    cudaStream_t xla_stream,
    LorraxCusolverMpSubRowCtx* ctx,
    const T* d_A_in, T* d_L_out)
{
    FFI_RETURN_IF_ERROR(cross_stream_wait(ctx->stream, xla_stream));

    const int64_t Py   = ctx->Py;
    const int64_t lld  = n;                    // grid (1, Py) → full rows per rank
    const int64_t local_cols = (n + Py - 1) / Py;
    const int64_t slice_elems = n * local_cols;   // col-major per slice

    // One D2D copy of the whole batch into the output, then factor each
    // slice in place.  Mirrors the SLATE batched potrf pattern.
    LORRAX_CUDA_CHECK(cudaMemcpyAsync(
        d_L_out, d_A_in,
        nbatch_local * slice_elems * sizeof(T),
        cudaMemcpyDeviceToDevice, ctx->stream));

    // One shared descriptor (all slices have identical (N, MB, NB)).
    cusolverMpMatrixDescriptor_t descA = nullptr;
    LORRAX_LIB_CHECK(
        cusolverMpCreateMatrixDesc(&descA, ctx->grid,
                                   mp::CudaDataTypeOf<T>::value,
                                   n, n, mb, nb, 0, 0, lld),
        CUSOLVER_STATUS_SUCCESS, "cusolverMpCreateMatrixDesc(A)");

    // bufferSize once; all slices have identical footprint.
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
        ensure_subrow_workspace(ctx, d_ws, h_ws);
    } catch (const std::exception& ex) {
        cusolverMpDestroyMatrixDesc(descA);
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, ex.what());
    }

    for (int64_t b = 0; b < nbatch_local; ++b) {
        T* slice_ptr = d_L_out + b * slice_elems;
        LORRAX_CUDA_CHECK(
            cudaMemsetAsync(ctx->d_info, 0, sizeof(int), ctx->stream));
        mp_st = mp::Potrf<T>(
            ctx->handle, CUBLAS_FILL_MODE_LOWER, n,
            slice_ptr, 1, 1, descA,
            ctx->d_workspace, ctx->d_workspace_bytes,
            ctx->h_workspace, ctx->h_workspace_bytes,
            ctx->d_info);
        if (mp_st != CUSOLVER_STATUS_SUCCESS) {
            cusolverMpDestroyMatrixDesc(descA);
            std::ostringstream os;
            os << "cusolverMpPotrf (batch " << b << ") failed: status=" << (int)mp_st;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
    }

    cusolverMpDestroyMatrixDesc(descA);

    FFI_RETURN_IF_ERROR(cross_stream_wait(xla_stream, ctx->stream));
    return ffi::Error::Success();
}

static ffi::Error BatchedPotrfDispatch(
    cudaStream_t stream,
    ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> L_out,
    int64_t nbatch_local, int64_t n, int64_t mb, int64_t nb,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<LorraxCusolverMpSubRowCtx*>(ctx_handle);
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
                nbatch_local, n, mb, nb, stream, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<double*>(L_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return BatchedPotrfImpl<C128>(
                nbatch_local, n, mb, nb, stream, ctx,
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
        .Attr<int64_t>("nbatch_local")
        .Attr<int64_t>("n")
        .Attr<int64_t>("mb")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_handle"));
