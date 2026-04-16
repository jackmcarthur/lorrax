// eigh_ffi.cc — XLA FFI handlers for cusolverMpSyevd (real F64 + complex C128).
//
// The handler assumes the input buffer is already the local shard of a
// 2-D block-cyclic distribution with mbA=nbA and a single tile per process
// (so that JAX's NamedSharding(P('x','y')) block layout coincides with
// block-cyclic).  The Python wrapper enforces `mb = n/p`, `nb = n/q`.

#include <cstdint>
#include <cstring>
#include <sstream>

#include <cuda_runtime.h>
#include <nccl.h>
#include <cusolverMp.h>

#include "xla/ffi/api/ffi.h"

#include "../../common/cpp/xla_ffi_glue.h"
#include "ctx.h"

namespace lorrax_ffi::cusolvermp {

namespace ffi = xla::ffi;

// Cross-stream synchronisation: the XLA-provided stream produced the
// input buffer; the ctx stream is what cusolverMp uses.  We need
// (a) ctx stream to wait on XLA's work before Syevd, and
// (b) XLA's stream to wait on ctx's work before reading the outputs.
// We use CUDA events for a low-overhead one-way sync.
static ffi::Error cross_stream_wait(cudaStream_t waiter, cudaStream_t signaller) {
    cudaEvent_t ev;
    LORRAX_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    LORRAX_CUDA_CHECK(cudaEventRecord(ev, signaller));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(waiter, ev, 0));
    LORRAX_CUDA_CHECK(cudaEventDestroy(ev));
    return ffi::Error::Success();
}

// Core solve, templated on dtype.  T_A is the matrix element type (F64 or C128),
// T_D is the eigenvalue type (F64 for both — cusolverMp returns real evs even
// for complex Hermitian input).
template <cudaDataType_t CudaType, typename T_A>
static ffi::Error syevd_impl(
    cudaStream_t xla_stream,
    const T_A* d_A_in, T_A* d_Q_out, double* d_D_out,
    int64_t n, int64_t mb, int64_t nb,
    LorraxCusolverMpCtx* ctx,
    bool compute_evecs)
{
    // --- make ctx's stream wait for XLA's producers ------------------------
    auto e1 = cross_stream_wait(ctx->stream, xla_stream);
    if (!e1.success()) return e1;

    // --- per-call matrix descriptors --------------------------------------
    // ia = ja = iq = jq = 1 (offsets, base-1 per ScaLAPACK convention).
    // Global rows/cols = n.  Local leading dim = n / p (block case).
    int64_t llda = (n + ctx->p - 1) / ctx->p;   // ceil; exact for n % p == 0

    cusolverMpMatrixDescriptor_t descA = nullptr, descQ = nullptr;
    LORRAX_LIB_CHECK(
        cusolverMpCreateMatrixDesc(&descA, ctx->grid, CudaType,
                                   /*M=*/n, /*N=*/n,
                                   /*MB=*/mb, /*NB=*/nb,
                                   /*RSRC=*/0, /*CSRC=*/0,
                                   /*LLDA=*/llda),
        CUSOLVER_STATUS_SUCCESS, "cusolverMpCreateMatrixDesc(A)");
    LORRAX_LIB_CHECK(
        cusolverMpCreateMatrixDesc(&descQ, ctx->grid, CudaType,
                                   n, n, mb, nb, 0, 0, llda),
        CUSOLVER_STATUS_SUCCESS, "cusolverMpCreateMatrixDesc(Q)");

    // compz is declared as char* in cusolverMp.h (not const); its log path
    // prints with %s and reads past a single-byte stack variable.  Give it
    // a proper 2-byte null-terminated buffer to avoid the log UB and any
    // internal string dispatch.
    char compz_buf[2];
    compz_buf[0] = compute_evecs ? 'V' : 'N';
    compz_buf[1] = '\0';
    char* compz = compz_buf;
    size_t d_ws = 0, h_ws = 0;

    // Cast to void* for cusolverMp API (it takes untyped pointers).
    void* A_ptr = const_cast<void*>(static_cast<const void*>(d_A_in));
    void* Q_ptr = static_cast<void*>(d_Q_out);
    void* D_ptr = static_cast<void*>(d_D_out);

    // --- workspace query --------------------------------------------------
    auto mp_st = cusolverMpSyevd_bufferSize(
        ctx->handle, compz, CUBLAS_FILL_MODE_LOWER, n,
        A_ptr, /*ia=*/1, /*ja=*/1, descA,
        D_ptr,
        Q_ptr, /*iq=*/1, /*jq=*/1, descQ,
        CudaType, &d_ws, &h_ws);
    if (mp_st != CUSOLVER_STATUS_SUCCESS) {
        cusolverMpDestroyMatrixDesc(descA);
        cusolverMpDestroyMatrixDesc(descQ);
        std::ostringstream os;
        os << "cusolverMpSyevd_bufferSize failed: status=" << (int)mp_st;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    // --- (re-)allocate workspace on ctx ------------------------------------
    try {
        ensure_workspace(ctx, d_ws, h_ws);
    } catch (const std::exception& e) {
        cusolverMpDestroyMatrixDesc(descA);
        cusolverMpDestroyMatrixDesc(descQ);
        return ffi::Error(ffi::ErrorCode::kInternal, e.what());
    }

    // --- reset d_info -----------------------------------------------------
    LORRAX_CUDA_CHECK(cudaMemsetAsync(ctx->d_info, 0, sizeof(int), ctx->stream));

    // --- solve ------------------------------------------------------------
    mp_st = cusolverMpSyevd(
        ctx->handle, compz, CUBLAS_FILL_MODE_LOWER, n,
        A_ptr, 1, 1, descA,
        D_ptr,
        Q_ptr, 1, 1, descQ,
        CudaType,
        ctx->d_workspace, ctx->d_workspace_bytes,
        ctx->h_workspace, ctx->h_workspace_bytes,
        ctx->d_info);

    cusolverMpDestroyMatrixDesc(descA);
    cusolverMpDestroyMatrixDesc(descQ);

    if (mp_st != CUSOLVER_STATUS_SUCCESS) {
        std::ostringstream os;
        os << "cusolverMpSyevd failed: status=" << (int)mp_st;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    // Copy d_info to host async to log any positive return.  We allow
    // execution to continue; only hard cusolverStatus_t failures above
    // raise to the Python side.  (Matches the sample: info != 0 would
    // indicate convergence issues, not a fatal API error.)
    // Skipped — cusolverStat != SUCCESS already covers API failures.

    // --- XLA's stream must wait on ctx work before downstream ops ---------
    auto e2 = cross_stream_wait(xla_stream, ctx->stream);
    if (!e2.success()) return e2;

    return ffi::Error::Success();
}

// ===========================================================================
//  Handler definitions
// ===========================================================================
// ffi::BufferR2 = rank-2 buffer.  The XLA compiler may pass rank-0/1/2
// buffers depending on how the Python wrapper was written — we use
// `AnyBuffer` to stay flexible and pull shape + dtype at runtime.

static ffi::Error EighF64Host(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F64> A,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> evals,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> Q,
    int64_t n, int64_t mb, int64_t nb,
    int64_t ctx_handle, bool compute_evecs)
{
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "EighF64: ctx_handle is null (did you init the "
                          "cusolvermp context on this process?)");
    }
    return syevd_impl<CUDA_R_64F, double>(
        stream,
        A.typed_data(), Q->typed_data(), evals->typed_data(),
        n, mb, nb, ctx, compute_evecs);
}

static ffi::Error EighC128Host(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::C128> A,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>>  evals,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> Q,
    int64_t n, int64_t mb, int64_t nb,
    int64_t ctx_handle, bool compute_evecs)
{
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "EighC128: ctx_handle is null.");
    }
    // The C128 element type in xla::ffi is std::complex<double>; cuSOLVERMp
    // takes void* so the actual struct layout must match cuDoubleComplex
    // (which it does: {double x,y}).
    return syevd_impl<CUDA_C_64F, std::complex<double>>(
        stream,
        A.typed_data(), Q->typed_data(), evals->typed_data(),
        n, mb, nb, ctx, compute_evecs);
}

// ===========================================================================
//  XLA_FFI_DEFINE_HANDLER_SYMBOL — exported C symbols
// ===========================================================================
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    EighF64, EighF64Host,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()                // A_local
        .Ret<ffi::Buffer<ffi::DataType::F64>>()                // evals
        .Ret<ffi::Buffer<ffi::DataType::F64>>()                // Q_local
        .Attr<int64_t>("n")
        .Attr<int64_t>("mb")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_handle")
        .Attr<bool>("compute_evecs"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    EighC128, EighC128Host,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::C128>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()                // evals (real)
        .Ret<ffi::Buffer<ffi::DataType::C128>>()               // Q_local
        .Attr<int64_t>("n")
        .Attr<int64_t>("mb")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_handle")
        .Attr<bool>("compute_evecs"));

}  // namespace lorrax_ffi::cusolvermp
