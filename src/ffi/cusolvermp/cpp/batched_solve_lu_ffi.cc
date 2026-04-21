// batched_solve_lu_ffi.cc — per-q cuSOLVERMp LU factor+solve on the
// world-wide (Px, Py) grid.  Does getrf followed immediately by getrs,
// per q-slice, so callers never see the pivot vector.
//
// *** KNOWN BUG — cuSOLVERMp 0.6.0 (NVHPC 25.5) 2D-grid LU incorrectness ***
//
// On a true 2D process grid (Px > 1 AND Py > 1), this handler returns
// wrong answers for most (N, NRHS) configurations: status=SUCCESS and
// info=0 from both Getrf and Getrs, but X buffer contains garbage
// (residual |AX-B|/|B| ≈ 0.3–0.5).  The bug is inside cuSOLVERMp —
// potrf/potrs on the same ctx/grid/descriptor-setup work correctly at
// all tested sizes.  Pass/fail sample on a 2×2 grid:
//    N=64, nrhs=2   PASS
//    N=128, nrhs=8  PASS     (occasional sweet spots)
//    N=128, nrhs=1..128 all other values: FAIL
// Works correctly on 1×N and N×1 meshes for all sizes — it is specifically
// the 2D distributed LU path that is broken.
//
// Debugging attempts that did NOT fix it: grid layout (ROW vs COL
// major), ipiv size (LOCr+MB vs LOCc+NB vs both), pivoting off
// (d_ipiv=NULL), shared vs separate getrf/getrs workspaces, zeroing
// ipiv beforehand, XLA DCE guards (returning A_factored as output),
// materialising jnp.transpose.  Reference: NVIDIA mp_getrf_getrs.c
// sample and documented APIs all match this handler's usage.
//
// Upstream status: cuSOLVERMp release notes show later versions (0.7+)
// list getrf correctness fixes.  Re-test when we upgrade NVHPC.
//
// Workaround: w_isdf's low_mem W-solve uses the symmetric Cholesky
// identity W = X (I − X† χ X)⁻¹ X† instead (X = Cholesky(V)), routing
// entirely through the working potrf/potrs kernels.
//
// Sharding contract:
//   A : (Nq, N, N)       P(None, 'x', 'y')  — Python pre-transposes the
//                        inner two dims to match cuSOLVERMp's col-major
//                        grid expectation (same trick as potrf/potrs).
//   B : (Nq, N, NRHS)    P(None, 'x', 'y')  — inner dims pre-transposed
//                        to (NRHS/Py, N/Px) row-major ≡ col-major
//                        (N/Px × NRHS/Py) per slice.
//   X : same layout as B; written in place.
//
// descA : (N, N) with mb=nb=N/Px=N/Py (caller passes mb_a, nb_a).
// descB : (N, NRHS) with mb_b=N/Px (row block matches A), nb_b=NRHS/Py.
//
// A is overwritten with its LU factors in place (we don't need the
// original A after the solve).  Callers should declare
// `input_output_aliases={0: <out-index>}` on a jit surround if they
// want XLA to donate A's buffer.  The pivot vector is allocated per
// call via cudaMallocAsync on ctx->stream and freed before the ctx→xla
// cross-stream event.

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

namespace lorrax_ffi::cusolvermp_batched_solve_lu {

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
static ffi::Error BatchedSolveLuImpl(
    int64_t nq, int64_t n, int64_t nrhs,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    cudaStream_t xla_stream,
    LorraxCusolverMpCtx* ctx,
    const T* d_A_in, const T* d_B_in,
    T* d_A_factored_out, T* d_X_out)
{
    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        ctx->stream, xla_stream, ctx->ev_xla_in));

    const int Px = ctx->p;
    const int Py = ctx->q;
    const int64_t lld_A = (n + Px - 1) / Px;
    const int64_t A_local_cols = (n    + Py - 1) / Py;
    const int64_t B_local_cols = (nrhs + Py - 1) / Py;
    const int64_t A_slice = lld_A * A_local_cols;
    const int64_t B_slice = lld_A * B_local_cols;
    // Per-slice local ipiv length.  cuSOLVERMp's ipiv is distributed
    // along the process-column dim — the NVIDIA mp_getrf_getrs sample
    // allocates exactly LOCc(N) per rank.  Match that allocation.
    const int64_t ipiv_slice = A_local_cols;

    // Copy A into A_factored_out (unless aliased); copy B into X_out
    // (unless aliased).  getrf and getrs are in-place on their output
    // buffers.
    if (d_A_factored_out != static_cast<const T*>(d_A_in)) {
        LORRAX_CUDA_CHECK(cudaMemcpyAsync(
            d_A_factored_out, d_A_in,
            nq * A_slice * sizeof(T),
            cudaMemcpyDeviceToDevice, ctx->stream));
    }
    if (d_X_out != static_cast<const T*>(d_B_in)) {
        LORRAX_CUDA_CHECK(cudaMemcpyAsync(
            d_X_out, d_B_in,
            nq * B_slice * sizeof(T),
            cudaMemcpyDeviceToDevice, ctx->stream));
    }

    // Allocate per-call ipiv (nq × ipiv_slice int64).  Use plain
    // cudaMalloc — the pool-based async path correlated with
    // non-deterministic wrong answers on 2D grids.
    int64_t* d_ipiv = nullptr;
    const size_t ipiv_bytes = static_cast<size_t>(nq) * ipiv_slice * sizeof(int64_t);
    LORRAX_CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_ipiv), ipiv_bytes));
    LORRAX_CUDA_CHECK(cudaMemsetAsync(d_ipiv, 0, ipiv_bytes, ctx->stream));

    cusolverMpMatrixDescriptor_t descA = nullptr, descB = nullptr;
    LORRAX_LIB_CHECK(
        cusolverMpCreateMatrixDesc(&descA, ctx->grid,
                                   mp::CudaDataTypeOf<T>::value,
                                   n, n, mb_a, nb_a, 0, 0, lld_A),
        CUSOLVER_STATUS_SUCCESS, "cusolverMpCreateMatrixDesc(A)");
    LORRAX_LIB_CHECK(
        cusolverMpCreateMatrixDesc(&descB, ctx->grid,
                                   mp::CudaDataTypeOf<T>::value,
                                   n, nrhs, mb_b, nb_b, 0, 0, lld_A),
        CUSOLVER_STATUS_SUCCESS, "cusolverMpCreateMatrixDesc(B)");

    auto cleanup = [&]() {
        cusolverMpDestroyMatrixDesc(descA);
        cusolverMpDestroyMatrixDesc(descB);
        cudaFree(d_ipiv);
    };

    // The NVIDIA mp_getrf_getrs sample allocates SEPARATE workspaces
    // for Getrf and Getrs — sharing one workspace buffer gave
    // non-deterministic wrong answers on 2D grids.  Allocate both per-
    // call via cudaMallocAsync so the CUDA pool can recycle them.
    size_t d_ws_f = 0, h_ws_f = 0, d_ws_s = 0, h_ws_s = 0;
    cusolverStatus_t mp_st = mp::GetrfBufferSize<T>(
        ctx->handle, n, n,
        d_A_factored_out, 1, 1, descA,
        d_ipiv, &d_ws_f, &h_ws_f);
    if (mp_st != CUSOLVER_STATUS_SUCCESS) {
        cleanup();
        std::ostringstream os;
        os << "cusolverMpGetrf_bufferSize failed: status=" << (int)mp_st;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    mp_st = mp::GetrsBufferSize<T>(
        ctx->handle, CUBLAS_OP_N, n, nrhs,
        d_A_factored_out, 1, 1, descA,
        d_ipiv,
        d_X_out, 1, 1, descB,
        &d_ws_s, &h_ws_s);
    if (mp_st != CUSOLVER_STATUS_SUCCESS) {
        cleanup();
        std::ostringstream os;
        os << "cusolverMpGetrs_bufferSize failed: status=" << (int)mp_st;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    void* d_ws_f_buf = nullptr;
    void* d_ws_s_buf = nullptr;
    void* h_ws_f_buf = nullptr;
    void* h_ws_s_buf = nullptr;
    if (d_ws_f > 0) LORRAX_CUDA_CHECK(cudaMallocAsync(
        &d_ws_f_buf, d_ws_f, ctx->stream));
    if (d_ws_s > 0) LORRAX_CUDA_CHECK(cudaMallocAsync(
        &d_ws_s_buf, d_ws_s, ctx->stream));
    if (h_ws_f > 0) {
        h_ws_f_buf = std::malloc(h_ws_f);
        if (!h_ws_f_buf) { cleanup();
            return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                              "malloc(h_ws_getrf) failed"); }
    }
    if (h_ws_s > 0) {
        h_ws_s_buf = std::malloc(h_ws_s);
        if (!h_ws_s_buf) { cleanup();
            if (h_ws_f_buf) std::free(h_ws_f_buf);
            return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                              "malloc(h_ws_getrs) failed"); }
    }

    auto cleanup_ws = [&]() {
        if (d_ws_f_buf) cudaFreeAsync(d_ws_f_buf, ctx->stream);
        if (d_ws_s_buf) cudaFreeAsync(d_ws_s_buf, ctx->stream);
        if (h_ws_f_buf) std::free(h_ws_f_buf);
        if (h_ws_s_buf) std::free(h_ws_s_buf);
    };

    const bool debug = std::getenv("LORRAX_LU_DEBUG") != nullptr;
    const bool no_pivot = std::getenv("LORRAX_LU_NO_PIVOT") != nullptr;

    for (int64_t q = 0; q < nq; ++q) {
        T* A_slice_ptr = d_A_factored_out + q * A_slice;
        T* X_slice_ptr = d_X_out + q * B_slice;
        int64_t* ipiv_slice_ptr = no_pivot ? nullptr : (d_ipiv + q * ipiv_slice);

        mp_st = mp::Getrf<T>(
            ctx->handle, n, n,
            A_slice_ptr, 1, 1, descA,
            ipiv_slice_ptr,
            d_ws_f_buf, d_ws_f,
            h_ws_f_buf, h_ws_f,
            ctx->d_info);
        if (mp_st != CUSOLVER_STATUS_SUCCESS) {
            cleanup_ws(); cleanup();
            std::ostringstream os;
            os << "cusolverMpGetrf (q=" << q << ") failed: status=" << (int)mp_st;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
        if (debug) {
            int info_h = 0;
            LORRAX_CUDA_CHECK(cudaMemcpyAsync(&info_h, ctx->d_info, sizeof(int),
                                               cudaMemcpyDeviceToHost, ctx->stream));
            LORRAX_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            std::fprintf(stderr, "[LU_DEBUG] q=%lld getrf info=%d\n",
                         (long long)q, info_h);
        }

        mp_st = mp::Getrs<T>(
            ctx->handle, CUBLAS_OP_N, n, nrhs,
            A_slice_ptr, 1, 1, descA,
            ipiv_slice_ptr,
            X_slice_ptr, 1, 1, descB,
            d_ws_s_buf, d_ws_s,
            h_ws_s_buf, h_ws_s,
            ctx->d_info);
        if (mp_st != CUSOLVER_STATUS_SUCCESS) {
            cleanup_ws(); cleanup();
            std::ostringstream os;
            os << "cusolverMpGetrs (q=" << q << ") failed: status=" << (int)mp_st;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
        if (debug) {
            int info_h = 0;
            LORRAX_CUDA_CHECK(cudaMemcpyAsync(&info_h, ctx->d_info, sizeof(int),
                                               cudaMemcpyDeviceToHost, ctx->stream));
            LORRAX_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            std::fprintf(stderr, "[LU_DEBUG] q=%lld getrs info=%d\n",
                         (long long)q, info_h);
        }
    }

    cleanup_ws();
    cleanup();

    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        xla_stream, ctx->stream, ctx->ev_ctx_out));
    return ffi::Error::Success();
}

static ffi::Error BatchedSolveLuDispatch(
    cudaStream_t stream,
    ffi::AnyBuffer A,
    ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> X_out,
    int64_t nq, int64_t n, int64_t nrhs,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "batched_solve_lu: ctx_handle is null");
    }
    const auto dtype = A.element_type();
    if (B.element_type() != dtype || X_out->element_type() != dtype) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "batched_solve_lu: A, B, X_out must share dtype");
    }
    // A's buffer is donated — getrf factors in place.  Cast away const
    // to let the FFI write to it.
    switch (dtype) {
        case ffi::DataType::F64:
            return BatchedSolveLuImpl<double>(
                nq, n, nrhs, mb_a, nb_a, mb_b, nb_b, stream, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<const double*>(B.untyped_data()),
                const_cast<double*>(static_cast<const double*>(A.untyped_data())),
                static_cast<double*>(X_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return BatchedSolveLuImpl<C128>(
                nq, n, nrhs, mb_a, nb_a, mb_b, nb_b, stream, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<const C128*>(B.untyped_data()),
                const_cast<C128*>(static_cast<const C128*>(A.untyped_data())),
                static_cast<C128*>(X_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "batched_solve_lu: unsupported dtype " << (int)dtype
               << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::cusolvermp_batched_solve_lu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CusolverMpBatchedSolveLuFfi,
    lorrax_ffi::cusolvermp_batched_solve_lu::BatchedSolveLuDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()      // A (donated, LU factored in place)
        .Arg<xla::ffi::AnyBuffer>()      // B
        .Ret<xla::ffi::AnyBuffer>()      // X (aliased to B — in-place solve)
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("mb_a")
        .Attr<int64_t>("nb_a")
        .Attr<int64_t>("mb_b")
        .Attr<int64_t>("nb_b")
        .Attr<int64_t>("ctx_handle"));
