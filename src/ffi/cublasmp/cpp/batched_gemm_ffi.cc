// batched_gemm_ffi.cc — per-q distributed GEMM via cuBLASMp on the
// world-wide (Px, Py) grid.  Mirrors the cuSOLVERMp batched_potrf/potrs
// pattern: descriptors built once per FFI call, inner loop over q.
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

#include "../../common/cpp/ffi_helpers.h"
#include "../../cusolvermp/cpp/ctx.h"
#include "cublasmp_interface.h"

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
    ensure_cublasmp(ctx);

    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        ctx->stream, xla_stream, ctx->ev_xla_in));

    // Local per-rank slice sizes (full tile, one block per rank under our
    // sharding convention).
    const int Px = ctx->p;
    const int Py = ctx->q;
    const int64_t A_local_cols = (opA == CUBLAS_OP_N ? k : m + Py - 1) / Py;
    const int64_t B_local_cols = (opB == CUBLAS_OP_N ? n : k + Py - 1) / Py;
    const int64_t C_local_cols = (n + Py - 1) / Py;
    // Slices are nq-stacked.  Size per slice = lld * local_cols.
    // These are ROW counts * COL counts in the col-major view.
    const int64_t A_slice = lld_A * A_local_cols;
    const int64_t B_slice = lld_B * B_local_cols;
    const int64_t C_slice = lld_C * C_local_cols;

    // If caller didn't alias C_out to C_in, memcpy so beta*C semantics
    // use the correct input.  With aliasing, FFI is fully in-place.
    if (d_C_out != static_cast<const T*>(d_C_in)) {
        LORRAX_CUDA_CHECK(cudaMemcpyAsync(
            d_C_out, d_C_in,
            nq * C_slice * sizeof(T),
            cudaMemcpyDeviceToDevice, ctx->stream));
    }

    cublasMpMatrixDescriptor_t descA = nullptr, descB = nullptr, descC = nullptr;
    const int64_t A_rows = (opA == CUBLAS_OP_N) ? m : k;
    const int64_t A_cols = (opA == CUBLAS_OP_N) ? k : m;
    const int64_t B_rows = (opB == CUBLAS_OP_N) ? k : n;
    const int64_t B_cols = (opB == CUBLAS_OP_N) ? n : k;
    (void)A_rows; (void)A_cols; (void)B_rows; (void)B_cols;

    LORRAX_CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(
            A_rows, A_cols, mb_a, nb_a, 0, 0, lld_A,
            mp::CudaDataTypeOf<T>::value, ctx->cublasmp_grid, &descA),
        "cublasMpMatrixDescriptorCreate(A)");
    LORRAX_CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(
            B_rows, B_cols, mb_b, nb_b, 0, 0, lld_B,
            mp::CudaDataTypeOf<T>::value, ctx->cublasmp_grid, &descB),
        "cublasMpMatrixDescriptorCreate(B)");
    LORRAX_CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(
            m, n, mb_c, nb_c, 0, 0, lld_C,
            mp::CudaDataTypeOf<T>::value, ctx->cublasmp_grid, &descC),
        "cublasMpMatrixDescriptorCreate(C)");

    // Matmul descriptor carries transA/transB + compute type.
    cublasMpMatmulDescriptor_t matmulDesc = nullptr;
    LORRAX_CUBLASMP_CHECK(
        cublasMpMatmulDescriptorCreate(&matmulDesc, mp::ComputeTypeOf<T>::value),
        "cublasMpMatmulDescriptorCreate");
    LORRAX_CUBLASMP_CHECK(
        cublasMpMatmulDescriptorAttributeSet(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA,
            &opA, sizeof(opA)),
        "cublasMpMatmulDescriptorAttributeSet(TRANSA)");
    LORRAX_CUBLASMP_CHECK(
        cublasMpMatmulDescriptorAttributeSet(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB,
            &opB, sizeof(opB)),
        "cublasMpMatmulDescriptorAttributeSet(TRANSB)");

    auto cleanup = [&]() {
        cublasMpMatmulDescriptorDestroy(matmulDesc);
        cublasMpMatrixDescriptorDestroy(descA);
        cublasMpMatrixDescriptorDestroy(descB);
        cublasMpMatrixDescriptorDestroy(descC);
    };

    // Size workspace using the first slice's pointers; reuse for all q.
    size_t d_ws = 0, h_ws = 0;
    cublasMpStatus_t mp_st = mp::MatmulBufferSize<T>(
        ctx->cublasmp_handle, matmulDesc, m, n, k,
        &alpha,
        d_A,     1, 1, descA,
        d_B,     1, 1, descB,
        &beta,
        d_C_out, 1, 1, descC,
        d_C_out, 1, 1, descC,
        &d_ws, &h_ws);
    if (mp_st != CUBLASMP_STATUS_SUCCESS) {
        cleanup();
        std::ostringstream os;
        os << "cublasMpMatmul_bufferSize failed: status=" << (int)mp_st;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    try {
        ensure_workspace(ctx, d_ws, h_ws);
    } catch (const std::exception& ex) {
        cleanup();
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, ex.what());
    }

    for (int64_t q = 0; q < nq; ++q) {
        const T* A_q = d_A     + q * A_slice;
        const T* B_q = d_B     + q * B_slice;
        T*       C_q = d_C_out + q * C_slice;
        mp_st = mp::Matmul<T>(
            ctx->cublasmp_handle, matmulDesc, m, n, k,
            &alpha,
            A_q, 1, 1, descA,
            B_q, 1, 1, descB,
            &beta,
            C_q, 1, 1, descC,
            C_q, 1, 1, descC,
            ctx->d_workspace, ctx->d_workspace_bytes,
            ctx->h_workspace, ctx->h_workspace_bytes);
        if (mp_st != CUBLASMP_STATUS_SUCCESS) {
            cleanup();
            std::ostringstream os;
            os << "cublasMpMatmul (q=" << q << ") failed: status=" << (int)mp_st;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
    }

    cleanup();

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
