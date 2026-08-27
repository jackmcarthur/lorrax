// batched_solve_lu_ffi.cc — per-q cuSOLVERMp LU factor+solve on the
// world-wide (Px, Py) grid.  Does getrf followed immediately by getrs,
// per q-slice, so callers never see the pivot vector.
//
// *** Historical note — 2D-grid LU was broken in 0.6.0, fixed in 0.7.2 ***
//
// On cuSOLVERMp 0.6.0 (NVHPC 25.5) this handler returned wrong answers
// for most (N, NRHS) configurations on true 2D process grids: status=
// SUCCESS and info=0 from both Getrf and Getrs, but X buffer contained
// garbage (residual |AX-B|/|B| ≈ 0.3–0.5).  The Cholesky path
// (potrf/potrs) on the same ctx/grid/descriptor-setup worked correctly
// at all tested sizes — the bug was specific to the 2D distributed LU
// kernel in cuSOLVERMp itself.
//
// **Resolved in cuSOLVERMp 0.7.2** (current default; see
// ``stage_pypi.sh``).  Validated end-to-end on the MoS2 3×3 D3h
// bispinor smoke (2×2 mesh, transverse γ̃^i channels) — eqp0
// matches the legacy per-q ``jnp.linalg.solve`` path to float ULP.
// See ``isdf_fitting._resolve_solver_kind_transverse`` for the
// dispatch (default ``auto`` → cuSolverMp on true 2D meshes; override
// via cohsex.in ``cusolvermp_lu``).
//
// (Earlier 0.6.0-era workaround in w_isdf's low_mem W-solve — the
// symmetric Cholesky identity ``W = X (I − X† χ X)⁻¹ X†`` — is no
// longer required on 0.7.2 but is retained because it has favourable
// memory characteristics for the W solve.)
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
#include <cstdlib>
#include <exception>
#include <sstream>

#include <cuda_runtime.h>
#include <cusolverMp.h>

#include "xla/ffi/api/ffi.h"

#include "../common/ffi_helpers.h"
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
    }

    cleanup_ws();
    cleanup();

    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        xla_stream, ctx->stream, ctx->ev_ctx_out));
    return ffi::Error::Success();
}

// Split sibling of BatchedSolveLuImpl.  LU and ipiv remain device-resident
// between calls so a physics stage can pay getrf once and issue getrs for
// every RHS chunk.  ipiv_len is the LOCAL cuSOLVERMp extent LOCc(N)=N/Py;
// Python represents the rank-private rows as P(None,('x','y')), hence each
// rank stores exactly nq*N/Py int64 values (never a replicated global pivot).
template <typename T>
static ffi::Error BatchedGetrfImpl(
    int64_t nq, int64_t n, int64_t mb, int64_t nb, int64_t ipiv_len,
    cudaStream_t xla_stream, LorraxCusolverMpCtx* ctx,
    const T* d_A_in, T* d_LU_out, int64_t* d_ipiv_out)
{
    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        ctx->stream, xla_stream, ctx->ev_xla_in));
    const int Px = ctx->p;
    const int Py = ctx->q;
    const int64_t lld_A = (n + Px - 1) / Px;
    const int64_t local_cols = (n + Py - 1) / Py;
    if (ipiv_len != local_cols) {
        std::ostringstream os;
        os << "cusolvermp.getrf: ipiv_len=" << ipiv_len
           << " != LOCc(N)=" << local_cols;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    const int64_t A_slice = lld_A * local_cols;
    if (d_LU_out != d_A_in) {
        LORRAX_CUDA_CHECK(cudaMemcpyAsync(
            d_LU_out, d_A_in, nq * A_slice * sizeof(T),
            cudaMemcpyDeviceToDevice, ctx->stream));
    }
    LORRAX_CUDA_CHECK(cudaMemsetAsync(
        d_ipiv_out, 0, nq * ipiv_len * sizeof(int64_t), ctx->stream));

    cusolverMpMatrixDescriptor_t descA = nullptr;
    LORRAX_LIB_CHECK(
        cusolverMpCreateMatrixDesc(&descA, ctx->grid,
                                   mp::CudaDataTypeOf<T>::value,
                                   n, n, mb, nb, 0, 0, lld_A),
        CUSOLVER_STATUS_SUCCESS, "cusolverMpCreateMatrixDesc(A)");
    const bool no_pivot = std::getenv("LORRAX_LU_NO_PIVOT") != nullptr;
    int64_t* piv0 = no_pivot ? nullptr : d_ipiv_out;
    size_t d_ws = 0, h_ws = 0;
    cusolverStatus_t mp_st = mp::GetrfBufferSize<T>(
        ctx->handle, n, n, d_LU_out, 1, 1, descA, piv0, &d_ws, &h_ws);
    if (mp_st != CUSOLVER_STATUS_SUCCESS) {
        cusolverMpDestroyMatrixDesc(descA);
        std::ostringstream os;
        os << "cusolverMpGetrf_bufferSize failed: status=" << (int)mp_st;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    try {
        ensure_workspace(ctx, d_ws, h_ws);
    } catch (const std::exception& ex) {
        cusolverMpDestroyMatrixDesc(descA);
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, ex.what());
    }
    for (int64_t q = 0; q < nq; ++q) {
        T* LU_q = d_LU_out + q * A_slice;
        int64_t* piv_q = no_pivot ? nullptr : d_ipiv_out + q * ipiv_len;
        mp_st = mp::Getrf<T>(
            ctx->handle, n, n, LU_q, 1, 1, descA, piv_q,
            ctx->d_workspace, ctx->d_workspace_bytes,
            ctx->h_workspace, ctx->h_workspace_bytes, ctx->d_info);
        if (mp_st != CUSOLVER_STATUS_SUCCESS) {
            cusolverMpDestroyMatrixDesc(descA);
            std::ostringstream os;
            os << "cusolverMpGetrf (q=" << q << ") failed: status="
               << (int)mp_st;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
    }
    cusolverMpDestroyMatrixDesc(descA);
    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        xla_stream, ctx->stream, ctx->ev_ctx_out));
    return ffi::Error::Success();
}

template <typename T>
static ffi::Error BatchedGetrsImpl(
    int64_t nq, int64_t n, int64_t nrhs,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    int64_t ipiv_len, cudaStream_t xla_stream, LorraxCusolverMpCtx* ctx,
    const T* d_LU, const int64_t* d_ipiv, const T* d_B_in, T* d_X_out)
{
    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        ctx->stream, xla_stream, ctx->ev_xla_in));
    const int Px = ctx->p;
    const int Py = ctx->q;
    const int64_t lld_A = (n + Px - 1) / Px;
    const int64_t A_local_cols = (n + Py - 1) / Py;
    const int64_t B_local_cols = (nrhs + Py - 1) / Py;
    if (ipiv_len != A_local_cols) {
        std::ostringstream os;
        os << "cusolvermp.getrs: ipiv_len=" << ipiv_len
           << " != LOCc(N)=" << A_local_cols;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    const int64_t A_slice = lld_A * A_local_cols;
    const int64_t B_slice = lld_A * B_local_cols;
    if (d_X_out != d_B_in) {
        LORRAX_CUDA_CHECK(cudaMemcpyAsync(
            d_X_out, d_B_in, nq * B_slice * sizeof(T),
            cudaMemcpyDeviceToDevice, ctx->stream));
    }
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
    const bool no_pivot = std::getenv("LORRAX_LU_NO_PIVOT") != nullptr;
    int64_t* piv0 = no_pivot ? nullptr : const_cast<int64_t*>(d_ipiv);
    size_t d_ws = 0, h_ws = 0;
    cusolverStatus_t mp_st = mp::GetrsBufferSize<T>(
        ctx->handle, CUBLAS_OP_N, n, nrhs,
        const_cast<T*>(d_LU), 1, 1, descA, piv0,
        d_X_out, 1, 1, descB, &d_ws, &h_ws);
    if (mp_st != CUSOLVER_STATUS_SUCCESS) {
        cusolverMpDestroyMatrixDesc(descA);
        cusolverMpDestroyMatrixDesc(descB);
        std::ostringstream os;
        os << "cusolverMpGetrs_bufferSize failed: status=" << (int)mp_st;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    // This workspace may be the same persistent allocation getrf used.
    // Reuse is safe across the SPLIT handlers because getrf records
    // ev_ctx_out and makes the XLA stream wait before returning its LU/ipiv;
    // this getrs cannot begin until those inputs are ready, then its opening
    // xla->ctx event orders the context stream after that completion.  The
    // fused handler cannot establish that FFI boundary and therefore keeps
    // two simultaneous workspaces (see its measured 2-D-grid warning above).
    try {
        ensure_workspace(ctx, d_ws, h_ws);
    } catch (const std::exception& ex) {
        cusolverMpDestroyMatrixDesc(descA);
        cusolverMpDestroyMatrixDesc(descB);
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, ex.what());
    }
    for (int64_t q = 0; q < nq; ++q) {
        T* LU_q = const_cast<T*>(d_LU) + q * A_slice;
        T* X_q = d_X_out + q * B_slice;
        int64_t* piv_q = no_pivot ? nullptr
                                  : const_cast<int64_t*>(d_ipiv)
                                        + q * ipiv_len;
        mp_st = mp::Getrs<T>(
            ctx->handle, CUBLAS_OP_N, n, nrhs, LU_q, 1, 1, descA, piv_q,
            X_q, 1, 1, descB,
            ctx->d_workspace, ctx->d_workspace_bytes,
            ctx->h_workspace, ctx->h_workspace_bytes, ctx->d_info);
        if (mp_st != CUSOLVER_STATUS_SUCCESS) {
            cusolverMpDestroyMatrixDesc(descA);
            cusolverMpDestroyMatrixDesc(descB);
            std::ostringstream os;
            os << "cusolverMpGetrs (q=" << q << ") failed: status="
               << (int)mp_st;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
    }
    cusolverMpDestroyMatrixDesc(descA);
    cusolverMpDestroyMatrixDesc(descB);
    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        xla_stream, ctx->stream, ctx->ev_ctx_out));
    return ffi::Error::Success();
}

static ffi::Error BatchedGetrfDispatch(
    cudaStream_t stream, ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> LU_out,
    ffi::Result<ffi::AnyBuffer> ipiv_out,
    int64_t nq, int64_t n, int64_t mb, int64_t nb, int64_t ipiv_len,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (ctx == nullptr)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cusolvermp.getrf: ctx_handle is null");
    const auto dtype = A.element_type();
    if (LU_out->element_type() != dtype ||
        ipiv_out->element_type() != ffi::DataType::S64)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cusolvermp.getrf: LU must match A and ipiv must be int64");
    switch (dtype) {
        case ffi::DataType::F64:
            return BatchedGetrfImpl<double>(
                nq, n, mb, nb, ipiv_len, stream, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<double*>(LU_out->untyped_data()),
                static_cast<int64_t*>(ipiv_out->untyped_data()));
        case ffi::DataType::C128: {
            using C128 = std::complex<double>;
            return BatchedGetrfImpl<C128>(
                nq, n, mb, nb, ipiv_len, stream, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<C128*>(LU_out->untyped_data()),
                static_cast<int64_t*>(ipiv_out->untyped_data()));
        }
        default:
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "cusolvermp.getrf: supported dtypes are F64, C128");
    }
}

static ffi::Error BatchedGetrsDispatch(
    cudaStream_t stream, ffi::AnyBuffer LU, ffi::AnyBuffer ipiv,
    ffi::AnyBuffer B, ffi::Result<ffi::AnyBuffer> X_out,
    int64_t nq, int64_t n, int64_t nrhs,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    int64_t ipiv_len, int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (ctx == nullptr)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cusolvermp.getrs: ctx_handle is null");
    const auto dtype = LU.element_type();
    if (B.element_type() != dtype || X_out->element_type() != dtype ||
        ipiv.element_type() != ffi::DataType::S64)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cusolvermp.getrs: LU/B/X must share dtype and ipiv must be int64");
    switch (dtype) {
        case ffi::DataType::F64:
            return BatchedGetrsImpl<double>(
                nq, n, nrhs, mb_a, nb_a, mb_b, nb_b, ipiv_len,
                stream, ctx,
                static_cast<const double*>(LU.untyped_data()),
                static_cast<const int64_t*>(ipiv.untyped_data()),
                static_cast<const double*>(B.untyped_data()),
                static_cast<double*>(X_out->untyped_data()));
        case ffi::DataType::C128: {
            using C128 = std::complex<double>;
            return BatchedGetrsImpl<C128>(
                nq, n, nrhs, mb_a, nb_a, mb_b, nb_b, ipiv_len,
                stream, ctx,
                static_cast<const C128*>(LU.untyped_data()),
                static_cast<const int64_t*>(ipiv.untyped_data()),
                static_cast<const C128*>(B.untyped_data()),
                static_cast<C128*>(X_out->untyped_data()));
        }
        default:
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "cusolvermp.getrs: supported dtypes are F64, C128");
    }
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CusolverMpBatchedGetrfFfi,
    lorrax_ffi::cusolvermp_batched_solve_lu::BatchedGetrfDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("mb")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ipiv_len")
        .Attr<int64_t>("ctx_handle"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CusolverMpBatchedGetrsFfi,
    lorrax_ffi::cusolvermp_batched_solve_lu::BatchedGetrsDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("mb_a")
        .Attr<int64_t>("nb_a")
        .Attr<int64_t>("mb_b")
        .Attr<int64_t>("nb_b")
        .Attr<int64_t>("ipiv_len")
        .Attr<int64_t>("ctx_handle"));
