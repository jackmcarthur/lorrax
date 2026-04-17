// eigh_ffi.cc — XLA FFI handler for cusolverMpSyevd (Hermitian / real-symm
//                eigensolver across a 2-D process grid).
//
// Structure (mirrors jaxlib/gpu/solver_kernels_ffi.cc):
//
//   EighImpl<T>       — does the actual cuSOLVERMp dance for one scalar type
//   EighDispatch      — shape/dtype validation + dtype switch
//   XLA_FFI_DEFINE_HANDLER_SYMBOL(EighMpFfi, EighDispatch, Bind()...)
//
// Patterns adopted from jaxlib:
//   - ffi::ScratchAllocator for per-call device workspace (so we don't
//     fight XLA's MEM_FRACTION preallocation)
//   - FFI_RETURN_IF_ERROR / LORRAX_*_CHECK for one-line error paths
//   - per-dtype cusolvermp::mp::{SyevdBufferSize,Syevd}<T> shims
//
// LORRAX-specific pieces that stay on the Ctx (not scratch):
//   - cusolverMpHandle_t / cal_comm_t / cusolverMpGrid_t  (one-time setup)
//   - host-side workspace buffer (cuSOLVERMp needs a host malloc; XLA's
//     scratch allocator is device-only)
//   - d_info (tiny; allocated once on ctx)
//   - NCCL scratch in the CAL→NCCL shim (lives outside any FFI call)

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

namespace lorrax_ffi::cusolvermp {

namespace ffi = ::xla::ffi;

// ---------------------------------------------------------------------------
//  Profile switch
// ---------------------------------------------------------------------------
static inline bool profile_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("LORRAX_FFI_PROFILE");
        return e != nullptr && *e == '1';
    }();
    return v;
}

// ---------------------------------------------------------------------------
//  Cross-stream join.  XLA's stream produced the input; our ctx stream is
//  what cusolverMp uses.  Record an event on one side, wait on the other
//  — no host-level cudaStreamSynchronize needed in the hot path.
// ---------------------------------------------------------------------------
static ffi::Error cross_stream_wait(cudaStream_t waiter,
                                    cudaStream_t signaller) {
    cudaEvent_t ev;
    LORRAX_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    LORRAX_CUDA_CHECK(cudaEventRecord(ev, signaller));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(waiter, ev, 0));
    LORRAX_CUDA_CHECK(cudaEventDestroy(ev));
    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
//  Impl — the actual cuSOLVERMp solve for one scalar type.
// ---------------------------------------------------------------------------
template <typename T>
static ffi::Error EighImpl(
    int64_t n, int64_t mb, int64_t nb,
    bool compute_evecs,
    cudaStream_t xla_stream,
    ffi::ScratchAllocator& scratch,
    LorraxCusolverMpCtx* ctx,
    const T* d_A,
    mp::RealOf_t<T>* d_W,
    T* d_Q)
{
    const bool prof = profile_enabled();
    cudaEvent_t ev[6];
    if (prof) {
        for (int i = 0; i < 6; ++i)
            LORRAX_CUDA_CHECK(cudaEventCreate(&ev[i]));
        LORRAX_CUDA_CHECK(cudaEventRecord(ev[0], ctx->stream));
    }

    FFI_RETURN_IF_ERROR(cross_stream_wait(ctx->stream, xla_stream));
    if (prof) LORRAX_CUDA_CHECK(cudaEventRecord(ev[1], ctx->stream));

    const int64_t llda = (n + ctx->p - 1) / ctx->p;   // = n/p when exact
    cusolverMpMatrixDescriptor_t descA = nullptr, descQ = nullptr;
    LORRAX_LIB_CHECK(
        cusolverMpCreateMatrixDesc(&descA, ctx->grid,
                                   mp::CudaDataTypeOf<T>::value,
                                   n, n, mb, nb, 0, 0, llda),
        CUSOLVER_STATUS_SUCCESS, "cusolverMpCreateMatrixDesc(A)");
    LORRAX_LIB_CHECK(
        cusolverMpCreateMatrixDesc(&descQ, ctx->grid,
                                   mp::CudaDataTypeOf<T>::value,
                                   n, n, mb, nb, 0, 0, llda),
        CUSOLVER_STATUS_SUCCESS, "cusolverMpCreateMatrixDesc(Q)");
    if (prof) LORRAX_CUDA_CHECK(cudaEventRecord(ev[2], ctx->stream));

    const char jobz = compute_evecs ? 'V' : 'N';
    size_t d_ws_bytes = 0, h_ws_bytes = 0;

    cusolverStatus_t mp_st = mp::SyevdBufferSize<T>(
        ctx->handle, jobz, CUBLAS_FILL_MODE_LOWER, n,
        d_A, 1, 1, descA,
        d_W,
        d_Q, 1, 1, descQ,
        &d_ws_bytes, &h_ws_bytes);
    if (mp_st != CUSOLVER_STATUS_SUCCESS) {
        cusolverMpDestroyMatrixDesc(descA);
        cusolverMpDestroyMatrixDesc(descQ);
        std::ostringstream os;
        os << "cusolverMpSyevd_bufferSize failed: status=" << (int)mp_st;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    if (prof) LORRAX_CUDA_CHECK(cudaEventRecord(ev[3], ctx->stream));

    // Device workspace from XLA's scratch pool — plays nice with
    // MEM_FRACTION reservation (cusolverMp's own libcal/UCC internals
    // still do their own cudaMalloc, so MEM_FRACTION=0.5 remains needed
    // until/unless that's fixed upstream; this at least keeps OUR
    // workspace from double-dipping).
    auto ws_opt = scratch.Allocate(d_ws_bytes);
    if (!ws_opt.has_value()) {
        cusolverMpDestroyMatrixDesc(descA);
        cusolverMpDestroyMatrixDesc(descQ);
        std::ostringstream os;
        os << "eigh: XLA scratch allocator failed to provide "
           << d_ws_bytes << " bytes";
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, os.str());
    }
    void* d_workspace = *ws_opt;

    // Host workspace stays on Ctx (scratch is device-only).
    if (h_ws_bytes > ctx->h_workspace_bytes) {
        if (ctx->h_workspace) std::free(ctx->h_workspace);
        ctx->h_workspace = std::malloc(h_ws_bytes);
        if (!ctx->h_workspace) {
            cusolverMpDestroyMatrixDesc(descA);
            cusolverMpDestroyMatrixDesc(descQ);
            return ffi::Error(ffi::ErrorCode::kInternal,
                              "malloc(host_workspace) failed");
        }
        ctx->h_workspace_bytes = h_ws_bytes;
    }

    LORRAX_CUDA_CHECK(
        cudaMemsetAsync(ctx->d_info, 0, sizeof(int), ctx->stream));

    // cuSOLVERMp overwrites A's tile (Householder tridiagonalisation);
    // const-cast once at the call site.
    mp_st = mp::Syevd<T>(
        ctx->handle, jobz, CUBLAS_FILL_MODE_LOWER, n,
        const_cast<T*>(d_A), 1, 1, descA,
        d_W,
        d_Q, 1, 1, descQ,
        d_workspace, d_ws_bytes,
        ctx->h_workspace, h_ws_bytes,
        ctx->d_info);
    if (prof) LORRAX_CUDA_CHECK(cudaEventRecord(ev[4], ctx->stream));

    cusolverMpDestroyMatrixDesc(descA);
    cusolverMpDestroyMatrixDesc(descQ);

    if (mp_st != CUSOLVER_STATUS_SUCCESS) {
        std::ostringstream os;
        os << "cusolverMpSyevd failed: status=" << (int)mp_st;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    FFI_RETURN_IF_ERROR(cross_stream_wait(xla_stream, ctx->stream));
    if (prof) LORRAX_CUDA_CHECK(cudaEventRecord(ev[5], ctx->stream));

    if (prof) {
        LORRAX_CUDA_CHECK(cudaEventSynchronize(ev[5]));
        float ts[5];
        cudaEventElapsedTime(&ts[0], ev[0], ev[1]);
        cudaEventElapsedTime(&ts[1], ev[1], ev[2]);
        cudaEventElapsedTime(&ts[2], ev[2], ev[3]);
        cudaEventElapsedTime(&ts[3], ev[3], ev[4]);
        cudaEventElapsedTime(&ts[4], ev[4], ev[5]);
        if (ctx->rank == 0) {
            std::fprintf(stderr,
                "[lorrax.ffi.prof] eigh n=%ld mb=%ld nb=%ld  "
                "cross-wait=%.2f ms  desc=%.2f ms  "
                "bufferSize=%.2f ms  syevd=%.2f ms  tail=%.2f ms  "
                "TOTAL(events)=%.2f ms\n",
                (long)n, (long)mb, (long)nb,
                ts[0], ts[1], ts[2], ts[3], ts[4],
                ts[0] + ts[1] + ts[2] + ts[3] + ts[4]);
        }
        for (int i = 0; i < 6; ++i) cudaEventDestroy(ev[i]);
    }

    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
//  Dispatch — validates shapes, looks up Ctx, switches on dtype.
// ---------------------------------------------------------------------------
static ffi::Error EighDispatch(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> W_out,
    ffi::Result<ffi::AnyBuffer> Q_out,
    int64_t n, int64_t mb, int64_t nb,
    int64_t ctx_handle, bool compute_evecs)
{
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "eigh: ctx_handle is null (context not initialized "
                          "on this process?)");
    }

    const auto dtype = A.element_type();
    if (Q_out->element_type() != dtype) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "eigh: output Q and input A must have matching element type");
    }
    if (W_out->element_type() != ffi::DataType::F64) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "eigh: eigenvalue output W must be F64 (cuSOLVERMp convention)");
    }

    // Shape checks are light — Python side enforces the hard invariants
    // (n % p == 0, square matrix, correct mesh axes).  If a caller ever
    // violates this we'll get a cuSOLVERMp status error below, which is
    // still actionable.

    switch (dtype) {
        case ffi::DataType::F64:
            return EighImpl<double>(
                n, mb, nb, compute_evecs, stream, scratch, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<double*>(W_out->untyped_data()),
                static_cast<double*>(Q_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return EighImpl<C128>(
                n, mb, nb, compute_evecs, stream, scratch, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<double*>(W_out->untyped_data()),
                static_cast<C128*>(Q_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "eigh: unsupported dtype " << static_cast<int>(dtype)
               << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::cusolvermp

// ---------------------------------------------------------------------------
//  FFI binding.
// ---------------------------------------------------------------------------
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    EighMpFfi, lorrax_ffi::cusolvermp::EighDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()              // A
        .Ret<xla::ffi::AnyBuffer>()              // W (evals)
        .Ret<xla::ffi::AnyBuffer>()              // Q (evecs)
        .Attr<int64_t>("n")
        .Attr<int64_t>("mb")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_handle")
        .Attr<bool>("compute_evecs"));
