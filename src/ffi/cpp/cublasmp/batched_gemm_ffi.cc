// batched_gemm_ffi.cc — packed-q SUMMA behind the existing distributed
// GEMM ABI. Broadcast all q panels together; keep the original face layout.
//
// Computes: C[q] = alpha * op(A[q]) * op(B[q]) + beta * C[q]
// where op = 'N' / 'T' / 'C' per the transa, transb attrs.
//
// Sharding contract:
//   A : (Nq, M, K)  P(None, 'x', 'y')  (or transposed — see Python wrapper)
//   B : (Nq, K, N)  P(None, 'x', 'y')
//   C : (Nq, M, N)  P(None, 'x', 'y') — both input (for beta!=0) and output
//
// Per-slice local bytes: mb*nb row-major.  Python pre-transposes inner
// dims (identical to potrf/potrs) so the buffer bytes are col-major with
// lld = local rows.  descA/B/C encode global shapes + block sizes.
//
// Note: cuBLASMp's D output is a DISTINCT buffer from C (the spec is
// D = alpha op(A) op(B) + beta C).  We pass the same buffer for both so
// the call is effectively in-place (D === C), which matches standard
// BLAS semantics.

#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#include <cuda_runtime.h>
#include <cublasmp.h>

#include "xla/ffi/api/ffi.h"

#include "../common/ffi_helpers.h"
#include "../cusolvermp/ctx.h"
#include "cublasmp_interface.h"
#include "compat.h"

namespace lorrax_ffi::cublasmp_batched_gemm {

namespace ffi = ::xla::ffi;
using lorrax_ffi::cusolvermp::LorraxCusolverMpCtx;
using lorrax_ffi::cusolvermp::ensure_cublasmp;
namespace mp = lorrax_ffi::cublasmp::mp;

#define LORRAX_CUBLASMP_CHECK(expr, what)                                  \
    do {                                                                   \
        cublasMpStatus_t _st = (expr);                                     \
        if (_st != CUBLASMP_STATUS_SUCCESS) {                              \
            std::ostringstream _os;                                        \
            _os << (what) << " failed: status=" << (int)_st;               \
            return ffi::Error(ffi::ErrorCode::kInternal, _os.str());       \
        }                                                                  \
    } while (0)

static ffi::Error cross_stream_wait_pooled(cudaStream_t waiter,
                                           cudaStream_t signaller,
                                           cudaEvent_t  ev) {
    LORRAX_CUDA_CHECK(cudaEventRecord(ev, signaller));
    LORRAX_CUDA_CHECK(cudaStreamWaitEvent(waiter, ev, 0));
    return ffi::Error::Success();
}

template <typename T>
static ffi::Error BatchedGemmImpl(
    int64_t nq, int64_t m, int64_t n, int64_t k,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    int64_t mb_c, int64_t nb_c,
    int64_t lld_A, int64_t lld_B, int64_t lld_C,
    cublasOperation_t opA, cublasOperation_t opB,
    T alpha, T beta,
    cudaStream_t xla_stream,
    LorraxCusolverMpCtx* ctx,
    const T* d_A, const T* d_B, const T* d_C_in, T* d_C_out)
{
    // C[q] = alpha sum_s A[q,:,s] B[q,s,:] + beta C[q].
    // One complete face tile per rank; panel s belongs to column s for A,
    // row s for B. Pack all q in each broadcast (already contiguous in the
    // FFI's col-major, q-stacked input), then perform one strided batch.
    if (opA != CUBLAS_OP_N || opB != CUBLAS_OP_N || ctx->p != ctx->q
        || ctx->grid_layout_col_major || m % ctx->p || n % ctx->q
        || k % ctx->p || mb_a != m / ctx->p || nb_a != k / ctx->q
        || mb_b != k / ctx->p || nb_b != n / ctx->q
        || mb_c != m / ctx->p || nb_c != n / ctx->q
        || lld_A != mb_a || lld_B != mb_b || lld_C != mb_c) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
            "packed-q SUMMA requires N,N square row-major mesh and exact faces");
    }
    if (!ctx->summa_blas) {
        LORRAX_LIB_CHECK(ncclCommSplit(ctx->nccl_comm, ctx->rank / ctx->q,
            ctx->rank % ctx->q, &ctx->summa_row, nullptr), ncclSuccess, "NCCL row split");
        LORRAX_LIB_CHECK(ncclCommSplit(ctx->nccl_comm, ctx->rank % ctx->q,
            ctx->rank / ctx->q, &ctx->summa_col, nullptr), ncclSuccess, "NCCL col split");
        LORRAX_LIB_CHECK(cublasLtCreate(&ctx->summa_blas), CUBLAS_STATUS_SUCCESS, "cuBLAS create");
        if (ctx->rank == 0) {
            std::fprintf(stderr, "[lorrax packed-q SUMMA] service prototype: "
                "all-q panel broadcasts, strided batched local GEMMs; existing face ABI\n");
            std::fflush(stderr);
        }
    }
    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        ctx->stream, xla_stream, ctx->ev_xla_in));
    const int64_t a_stride = mb_a * nb_a;
    const int64_t b_stride = mb_b * nb_b;
    const int64_t c_stride = mb_c * nb_c;
    const size_t a_bytes = nq * a_stride * sizeof(T);
    const size_t b_bytes = nq * b_stride * sizeof(T);
    try {
        ensure_workspace(ctx, a_bytes + b_bytes, 0);
    } catch (const std::exception& ex) {
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, ex.what());
    }
    T* a_panel = static_cast<T*>(ctx->d_workspace);
    T* b_panel = a_panel + nq * a_stride;
    if (d_C_out != d_C_in) {
        LORRAX_CUDA_CHECK(cudaMemcpyAsync(d_C_out, d_C_in,
            nq * c_stride * sizeof(T), cudaMemcpyDeviceToDevice, ctx->stream));
    }
    // Use the Lt kernel family underlying cuBLASMp; retain a batch stride
    // instead of asking the legacy BLAS API to choose a batched algorithm.
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t ad = nullptr, bd = nullptr, cd = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    LORRAX_LIB_CHECK(cublasLtMatmulDescCreate(&desc,
        mp::ComputeTypeOf<T>::value, mp::CudaDataTypeOf<T>::value), CUBLAS_STATUS_SUCCESS, "Lt desc");
    LORRAX_LIB_CHECK(cublasLtMatrixLayoutCreate(&ad, mp::CudaDataTypeOf<T>::value,
        mb_a, nb_a, lld_A), CUBLAS_STATUS_SUCCESS, "Lt A layout");
    LORRAX_LIB_CHECK(cublasLtMatrixLayoutCreate(&bd, mp::CudaDataTypeOf<T>::value,
        mb_b, nb_b, lld_B), CUBLAS_STATUS_SUCCESS, "Lt B layout");
    LORRAX_LIB_CHECK(cublasLtMatrixLayoutCreate(&cd, mp::CudaDataTypeOf<T>::value,
        mb_c, nb_c, lld_C), CUBLAS_STATUS_SUCCESS, "Lt C layout");
    const int batches = static_cast<int>(nq);
    cublasLtMatrixLayout_t layouts[] = {ad, bd, cd};
    const int64_t strides[] = {a_stride, b_stride, c_stride};
    for (int i = 0; i < 3; ++i) {
        LORRAX_LIB_CHECK(cublasLtMatrixLayoutSetAttribute(layouts[i],
            CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches, sizeof(batches)), CUBLAS_STATUS_SUCCESS, "Lt batch");
        LORRAX_LIB_CHECK(cublasLtMatrixLayoutSetAttribute(layouts[i],
            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strides[i], sizeof(strides[i])), CUBLAS_STATUS_SUCCESS, "Lt stride");
    }
    LORRAX_LIB_CHECK(cublasLtMatmulPreferenceCreate(&pref), CUBLAS_STATUS_SUCCESS, "Lt preference");
    cublasLtMatmulHeuristicResult_t choice{};
    int choices = 0;
    LORRAX_LIB_CHECK(cublasLtMatmulAlgoGetHeuristic(ctx->summa_blas, desc,
        ad, bd, cd, cd, pref, 1, &choice, &choices), CUBLAS_STATUS_SUCCESS, "Lt heuristic");
    if (choices != 1) return ffi::Error(ffi::ErrorCode::kUnimplemented,
        "packed-q SUMMA: no zero-workspace strided Lt algorithm");
    const T one = T(1);
    for (int s = 0; s < ctx->p; ++s) {
        // NCCL broadcasts copy bytes only; no floating-point reduction.
        LORRAX_LIB_CHECK(ncclGroupStart(), ncclSuccess, "NCCL group start");
        const auto a_status = ncclBroadcast(d_A, a_panel, a_bytes,
            ncclUint8, s, ctx->summa_row, ctx->stream);
        const auto b_status = ncclBroadcast(d_B, b_panel, b_bytes,
            ncclUint8, s, ctx->summa_col, ctx->stream);
        const auto end_status = ncclGroupEnd();
        LORRAX_LIB_CHECK(a_status, ncclSuccess, "NCCL A broadcast");
        LORRAX_LIB_CHECK(b_status, ncclSuccess, "NCCL B broadcast");
        LORRAX_LIB_CHECK(end_status, ncclSuccess, "NCCL group end");
        const T* panel_beta = s == 0 ? &beta : &one;
        LORRAX_LIB_CHECK(cublasLtMatmul(ctx->summa_blas, desc, &alpha,
            a_panel, ad, b_panel, bd, panel_beta, d_C_out, cd, d_C_out, cd,
            &choice.algo, nullptr, 0, ctx->stream), CUBLAS_STATUS_SUCCESS, "packed-q Lt GEMM");
    }
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(ad);
    cublasLtMatrixLayoutDestroy(bd);
    cublasLtMatrixLayoutDestroy(cd);
    cublasLtMatmulDescDestroy(desc);

    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        xla_stream, ctx->stream, ctx->ev_ctx_out));
    return ffi::Error::Success();
}

static ffi::Error BatchedGemmDispatch(
    cudaStream_t stream,
    ffi::AnyBuffer A,
    ffi::AnyBuffer B,
    ffi::AnyBuffer C_in,
    ffi::Result<ffi::AnyBuffer> C_out,
    int64_t nq, int64_t m, int64_t n, int64_t k,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    int64_t mb_c, int64_t nb_c,
    int64_t lld_A, int64_t lld_B, int64_t lld_C,
    int64_t transa_code, int64_t transb_code,
    double alpha_re, double alpha_im,
    double beta_re,  double beta_im,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "batched_gemm: ctx_handle is null");
    }
    const auto dtype = A.element_type();
    if (B.element_type() != dtype || C_in.element_type() != dtype
        || C_out->element_type() != dtype) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "batched_gemm: A, B, C must share dtype");
    }
    auto op_from_code = [](int64_t c) {
        switch (c) {
            case 0: return CUBLAS_OP_N;
            case 1: return CUBLAS_OP_T;
            case 2: return CUBLAS_OP_C;
            default: return CUBLAS_OP_N;   // caller validates; default harmless
        }
    };
    const cublasOperation_t opA = op_from_code(transa_code);
    const cublasOperation_t opB = op_from_code(transb_code);

    switch (dtype) {
        case ffi::DataType::F64: {
            const double a = alpha_re;
            const double b = beta_re;
            return BatchedGemmImpl<double>(
                nq, m, n, k,
                mb_a, nb_a, mb_b, nb_b, mb_c, nb_c,
                lld_A, lld_B, lld_C,
                opA, opB, a, b, stream, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<const double*>(B.untyped_data()),
                static_cast<const double*>(C_in.untyped_data()),
                static_cast<double*>(C_out->untyped_data()));
        }
        case ffi::DataType::C128: {
            using C128 = std::complex<double>;
            const C128 a(alpha_re, alpha_im);
            const C128 b(beta_re,  beta_im);
            return BatchedGemmImpl<C128>(
                nq, m, n, k,
                mb_a, nb_a, mb_b, nb_b, mb_c, nb_c,
                lld_A, lld_B, lld_C,
                opA, opB, a, b, stream, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<const C128*>(B.untyped_data()),
                static_cast<const C128*>(C_in.untyped_data()),
                static_cast<C128*>(C_out->untyped_data()));
        }
        default: {
            std::ostringstream os;
            os << "batched_gemm: unsupported dtype " << (int)dtype
               << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::cublasmp_batched_gemm

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CublasMpBatchedGemmFfi,
    lorrax_ffi::cublasmp_batched_gemm::BatchedGemmDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()      // A
        .Arg<xla::ffi::AnyBuffer>()      // B
        .Arg<xla::ffi::AnyBuffer>()      // C (for beta*C)
        .Ret<xla::ffi::AnyBuffer>()      // C_out
        .Attr<int64_t>("nq")
        .Attr<int64_t>("m")
        .Attr<int64_t>("n")
        .Attr<int64_t>("k")
        .Attr<int64_t>("mb_a")
        .Attr<int64_t>("nb_a")
        .Attr<int64_t>("mb_b")
        .Attr<int64_t>("nb_b")
        .Attr<int64_t>("mb_c")
        .Attr<int64_t>("nb_c")
        .Attr<int64_t>("lld_a")
        .Attr<int64_t>("lld_b")
        .Attr<int64_t>("lld_c")
        .Attr<int64_t>("transa")
        .Attr<int64_t>("transb")
        .Attr<double>("alpha_re")
        .Attr<double>("alpha_im")
        .Attr<double>("beta_re")
        .Attr<double>("beta_im")
        .Attr<int64_t>("ctx_handle"));
