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

#include <algorithm>
#include <climits>
#include <type_traits>
#include <vector>
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
    const T* d_A, const T* d_B, const T* d_C_in, T* d_C_out,
    int64_t contraction_owner = -1, int64_t contraction_offset = 0, int64_t active_k = -1)
{
    ensure_cublasmp(ctx);

    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        ctx->stream, xla_stream, ctx->ev_xla_in));

    // Local per-rank slice sizes (full tile, one block per rank under our
    // sharding convention).
    const int Px = ctx->p;
    const int Py = ctx->q;
    const bool view = contraction_owner >= 0;
    const int64_t contraction_k = view ? active_k : k;
    const int row = ctx->grid_layout_col_major ? ctx->rank % Px : ctx->rank / Py;
    const int col = ctx->grid_layout_col_major ? ctx->rank / Px : ctx->rank % Py;
    // A/B keep their original allocation strides. A view only changes the
    // descriptor's logical band extent, owner, and base pointer.
    if (view) {
        if (col == contraction_owner) d_A += contraction_offset * lld_A;
        if (row == contraction_owner) d_B += contraction_offset;
        nb_a = active_k;
        mb_b = active_k;
    }
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
    const int64_t A_cols = (opA == CUBLAS_OP_N) ? contraction_k : m;
    const int64_t B_rows = (opB == CUBLAS_OP_N) ? contraction_k : n;
    const int64_t B_cols = (opB == CUBLAS_OP_N) ? n : k;
    (void)A_rows; (void)A_cols; (void)B_rows; (void)B_cols;

    LORRAX_CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(
            A_rows, A_cols, mb_a, nb_a, 0, view ? contraction_owner : 0, lld_A,
            mp::CudaDataTypeOf<T>::value, ctx->cublasmp_grid, &descA),
        "cublasMpMatrixDescriptorCreate(A)");
    LORRAX_CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(
            B_rows, B_cols, mb_b, nb_b, view ? contraction_owner : 0, 0, lld_B,
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
        lorrax_ffi::cublasmp::set_matmul_descriptor_attribute(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA,
            &opA, sizeof(opA)),
        "cublasMpMatmulDescriptorAttributeSet(TRANSA)");
    LORRAX_CUBLASMP_CHECK(
        lorrax_ffi::cublasmp::set_matmul_descriptor_attribute(
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
        ctx->cublasmp_handle, matmulDesc, m, n, contraction_k,
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
            ctx->cublasmp_handle, matmulDesc, m, n, contraction_k,
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


// Empty contractions still obey beta*C without touching any A/B element.
// The existing context owns the local BLAS handle and destroys it on teardown.
template <typename T>
static ffi::Error ScaleEmpty(cudaStream_t stream, LorraxCusolverMpCtx* ctx,
                            const T* src, T* dst, int64_t count, T beta) {
    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(ctx->stream, stream, ctx->ev_xla_in));
    if (beta == T(0)) {
        LORRAX_CUDA_CHECK(cudaMemsetAsync(dst, 0, count*sizeof(T), ctx->stream));
    } else {
        if (src != dst)
            LORRAX_CUDA_CHECK(cudaMemcpyAsync(dst, src, count*sizeof(T),
                                             cudaMemcpyDeviceToDevice, ctx->stream));
        if (beta != T(1)) {
            if (!ctx->local_blas_handle) {
                cublasHandle_t handle = nullptr;
                const auto status = cublasCreate(&handle);
                if (status != CUBLAS_STATUS_SUCCESS) {
                    if (handle) cublasDestroy(handle);
                    return ffi::Error::Internal("active GEMM: local cuBLAS handle creation failed");
                }
                ctx->local_blas_handle = handle;
            }
            if (cublasSetStream(ctx->local_blas_handle, ctx->stream) != CUBLAS_STATUS_SUCCESS)
                return ffi::Error::Internal("active GEMM: local cuBLAS stream binding failed");
            for (int64_t offset=0; offset<count;) {
                const int length = static_cast<int>(std::min<int64_t>(count-offset, INT_MAX));
                cublasStatus_t status;
                if constexpr (std::is_same_v<T,double>)
                    status = cublasDscal(ctx->local_blas_handle,length,&beta,dst+offset,1);
                else
                    status = cublasZscal(ctx->local_blas_handle,length,
                        reinterpret_cast<const cuDoubleComplex*>(&beta),
                        reinterpret_cast<cuDoubleComplex*>(dst+offset),1);
                if (status != CUBLAS_STATUS_SUCCESS)
                    return ffi::Error::Internal("active GEMM: empty output scaling failed");
                offset += length;
            }
        }
    }
    return cross_stream_wait_pooled(stream,ctx->stream,ctx->ev_ctx_out);
}

template <typename T>
static ffi::Error ActiveRangeImpl(
    cudaStream_t stream, LorraxCusolverMpCtx* ctx,
    const T* a, const T* b, const T* cin, T* out,
    const std::vector<int32_t>& bounds, int64_t n_bounds,
    int64_t nq,int64_t m,int64_t n,int64_t k,
    int64_t mb_a,int64_t nb_a,int64_t mb_b,int64_t nb_b,
    int64_t mb_c,int64_t nb_c,int64_t lda,int64_t ldb,int64_t ldc,
    T alpha,T beta) {
    bool common = true;
    for (int64_t i=1;i<n_bounds;++i)
        common = common && bounds[2*i]==bounds[0] && bounds[2*i+1]==bounds[1];
    const int64_t groups = common ? 1 : nq;
    const int64_t group_nq = common ? nq : 1;
    const int64_t a_stride=lda*(k/ctx->q), b_stride=ldb*(n/ctx->q), c_stride=ldc*(n/ctx->q);
    const int64_t slab=k/ctx->p;
    for (int64_t group=0;group<groups;++group) {
        const int64_t lo=bounds[common ? 0 : 2*group];
        const int64_t hi=bounds[common ? 1 : 2*group+1];
        const T* ag=a+group*a_stride;
        const T* bg=b+group*b_stride;
        const T* cg=cin+group*c_stride;
        T* dg=out+group*c_stride;
        if (lo==hi) {
            FFI_RETURN_IF_ERROR(ScaleEmpty(stream,ctx,cg,dg,group_nq*c_stride,beta));
            continue;
        }
        if (lo==0 && hi==k) {
            // Preserve the original dense call and its floating-point order.
            FFI_RETURN_IF_ERROR(BatchedGemmImpl<T>(group_nq,m,n,k,mb_a,nb_a,mb_b,nb_b,
                mb_c,nb_c,lda,ldb,ldc,CUBLAS_OP_N,CUBLAS_OP_N,alpha,beta,
                stream,ctx,ag,bg,cg,dg));
            continue;
        }
        bool first=true;
        for (int owner=0;owner<ctx->p;++owner) {
            const int64_t begin=std::max<int64_t>(lo,owner*slab);
            const int64_t end=std::min<int64_t>(hi,(owner+1)*slab);
            if (begin>=end) continue;
            FFI_RETURN_IF_ERROR(BatchedGemmImpl<T>(group_nq,m,n,k,mb_a,nb_a,mb_b,nb_b,
                mb_c,nb_c,lda,ldb,ldc,CUBLAS_OP_N,CUBLAS_OP_N,alpha,first ? beta : T(1),
                stream,ctx,ag,bg,first ? cg : dg,dg,owner,begin-owner*slab,end-begin));
            first=false;
        }
    }
    return ffi::Error::Success();
}

// Full physical allocation shapes and strides, with scalar or per-batch
// replicated bounds. Each exact original-owner intersection has a descriptor
// view; there is no selected-operand allocation and no padded contraction K.
static ffi::Error ActiveRangeDispatchWithBounds(
    cudaStream_t stream, ffi::AnyBuffer A, ffi::AnyBuffer B,
    ffi::AnyBuffer C_in, ffi::Result<ffi::AnyBuffer> C_out,
    const std::vector<int32_t>& intervals, int64_t n_bounds,
    int64_t nq,int64_t m,int64_t n,int64_t k,
    int64_t mb_a,int64_t nb_a,int64_t mb_b,int64_t nb_b,
    int64_t mb_c,int64_t nb_c,int64_t lda,int64_t ldb,int64_t ldc,
    int64_t transa_code,int64_t transb_code,
    double alpha_re,double alpha_im,double beta_re,double beta_im,
    int64_t ctx_handle) {
    auto* ctx=reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (!ctx || ctx->p<=0 || ctx->p!=ctx->q || transa_code!=0 || transb_code!=0 ||
        (n_bounds!=1 && n_bounds!=nq) ||
        k<=0 || k%ctx->p!=0)
        return ffi::Error::InvalidArgument("active GEMM requires square N,N faces and bounds(1|nq,2)");
    if (A.element_type()!=B.element_type() || A.element_type()!=C_in.element_type() ||
        A.element_type()!=C_out->element_type())
        return ffi::Error::InvalidArgument("active GEMM operand dtype mismatch");
    for (int64_t i=0;i<n_bounds;++i)
        if (intervals[2*i]<0 || intervals[2*i]>intervals[2*i+1] || intervals[2*i+1]>k)
            return ffi::Error::InvalidArgument("active GEMM requires 0 <= lo <= hi <= storage K");
    if (A.element_type()==ffi::DataType::F64)
        return ActiveRangeImpl<double>(stream,ctx,
            static_cast<const double*>(A.untyped_data()),static_cast<const double*>(B.untyped_data()),
            static_cast<const double*>(C_in.untyped_data()),static_cast<double*>(C_out->untyped_data()),
            intervals,n_bounds,nq,m,n,k,mb_a,nb_a,mb_b,nb_b,mb_c,nb_c,lda,ldb,ldc,alpha_re,beta_re);
    if (A.element_type()==ffi::DataType::C128) {
        using T=std::complex<double>;
        return ActiveRangeImpl<T>(stream,ctx,
            static_cast<const T*>(A.untyped_data()),static_cast<const T*>(B.untyped_data()),
            static_cast<const T*>(C_in.untyped_data()),static_cast<T*>(C_out->untyped_data()),
            intervals,n_bounds,nq,m,n,k,mb_a,nb_a,mb_b,nb_b,mb_c,nb_c,lda,ldb,ldc,
            T(alpha_re,alpha_im),T(beta_re,beta_im));
    }
    return ffi::Error::InvalidArgument("active GEMM supports float64/complex128");
}

static ffi::Error ActiveRangeDispatch(
    cudaStream_t stream, ffi::AnyBuffer A, ffi::AnyBuffer B,
    ffi::AnyBuffer C_in, ffi::BufferR2<ffi::S32> bounds,
    ffi::Result<ffi::AnyBuffer> C_out,
    int64_t nq,int64_t m,int64_t n,int64_t k,
    int64_t mb_a,int64_t nb_a,int64_t mb_b,int64_t nb_b,
    int64_t mb_c,int64_t nb_c,int64_t lda,int64_t ldb,int64_t ldc,
    int64_t transa_code,int64_t transb_code,
    double alpha_re,double alpha_im,double beta_re,double beta_im,
    int64_t ctx_handle) {
    if (bounds.dimensions()[1]!=2 ||
        (bounds.dimensions()[0]!=1 && bounds.dimensions()[0]!=nq))
        return ffi::Error::InvalidArgument(
            "active GEMM requires square N,N faces and bounds(1|nq,2)");
    const int64_t n_bounds=bounds.dimensions()[0];
    std::vector<int32_t> intervals(2*n_bounds);
    LORRAX_CUDA_CHECK(cudaMemcpyAsync(intervals.data(),bounds.typed_data(),
        intervals.size()*sizeof(int32_t),cudaMemcpyDeviceToHost,stream));
    LORRAX_CUDA_CHECK(cudaStreamSynchronize(stream));
    return ActiveRangeDispatchWithBounds(stream,A,B,C_in,C_out,intervals,n_bounds,
        nq,m,n,k,mb_a,nb_a,mb_b,nb_b,mb_c,nb_c,lda,ldb,ldc,
        transa_code,transb_code,alpha_re,alpha_im,beta_re,beta_im,ctx_handle);
}

static ffi::Error PreparedActiveRangeDispatch(
    cudaStream_t stream, ffi::AnyBuffer A, ffi::AnyBuffer B,
    ffi::AnyBuffer C_in, ffi::Result<ffi::AnyBuffer> C_out,
    int64_t nq,int64_t m,int64_t n,int64_t k,
    int64_t mb_a,int64_t nb_a,int64_t mb_b,int64_t nb_b,
    int64_t mb_c,int64_t nb_c,int64_t lda,int64_t ldb,int64_t ldc,
    int64_t transa_code,int64_t transb_code,
    double alpha_re,double alpha_im,double beta_re,double beta_im,
    int64_t ctx_handle, ffi::Span<const int64_t> active_bounds) {
    const size_t pair_count=active_bounds.size()/2;
    if (nq<1 || active_bounds.size()%2!=0 ||
        (pair_count!=1 && pair_count!=static_cast<size_t>(nq)))
        return ffi::Error::InvalidArgument(
            "prepared active GEMM active_bounds must contain 1 or nq pairs");
    const int64_t n_bounds=static_cast<int64_t>(pair_count);
    std::vector<int32_t> intervals(active_bounds.size());
    for (size_t i=0;i<active_bounds.size();++i) {
        if (active_bounds[i]<0 || active_bounds[i]>INT_MAX)
            return ffi::Error::InvalidArgument(
                "prepared active GEMM bounds must fit nonnegative int32");
        intervals[i]=static_cast<int32_t>(active_bounds[i]);
    }
    return ActiveRangeDispatchWithBounds(stream,A,B,C_in,C_out,intervals,n_bounds,
        nq,m,n,k,mb_a,nb_a,mb_b,nb_b,mb_c,nb_c,lda,ldb,ldc,
        transa_code,transb_code,alpha_re,alpha_im,beta_re,beta_im,ctx_handle);
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CublasMpPreparedActiveRangeGemmFfi,
    lorrax_ffi::cublasmp_batched_gemm::PreparedActiveRangeDispatch,
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
        .Attr<int64_t>("ctx_handle")
        .Attr<xla::ffi::Span<const int64_t>>("active_bounds"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CublasMpActiveRangeGemmFfi,
    lorrax_ffi::cublasmp_batched_gemm::ActiveRangeDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()      // A
        .Arg<xla::ffi::AnyBuffer>()      // B
        .Arg<xla::ffi::AnyBuffer>()      // C (for beta*C)
        .Arg<xla::ffi::BufferR2<xla::ffi::S32>>() // (1|nq,2) bounds
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

// Query-only N,N planning door; all batches reuse one workspace in the
// execution handler. Sizing never reads the supplied device address token.
extern "C" int lrx_gemm_workspace_bytes(
    int64_t ctx_handle, int64_t m, int64_t n, int64_t k, int complex128,
    uint64_t* device_bytes, uint64_t* host_bytes) {
    if (!ctx_handle || !device_bytes || !host_bytes ||
        m < 1 || n < 1 || k < 1 || (complex128 != 0 && complex128 != 1)) return -1;
    *device_bytes = 0; *host_bytes = 0;
    using namespace lorrax_ffi::cublasmp_batched_gemm;
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (m%ctx->p || n%ctx->q || k%ctx->p || k%ctx->q) return -2;
    try { ensure_cublasmp(ctx); } catch (...) { return -3; }
    cublasMpMatrixDescriptor_t a=nullptr, b=nullptr, c=nullptr;
    cublasMpMatmulDescriptor_t mm=nullptr;
    const auto dtype = complex128 ? CUDA_C_64F : CUDA_R_64F;
    auto status = cublasMpMatrixDescriptorCreate(m,k,m/ctx->p,k/ctx->q,
        0,0,m/ctx->p,dtype,ctx->cublasmp_grid,&a);
    if (status == CUBLASMP_STATUS_SUCCESS)
        status = cublasMpMatrixDescriptorCreate(k,n,k/ctx->p,n/ctx->q,
            0,0,k/ctx->p,dtype,ctx->cublasmp_grid,&b);
    if (status == CUBLASMP_STATUS_SUCCESS)
        status = cublasMpMatrixDescriptorCreate(m,n,m/ctx->p,n/ctx->q,
            0,0,m/ctx->p,dtype,ctx->cublasmp_grid,&c);
    if (status == CUBLASMP_STATUS_SUCCESS)
        status = cublasMpMatmulDescriptorCreate(&mm,CUBLAS_COMPUTE_64F);
    cublasOperation_t op = CUBLAS_OP_N;
    if (status == CUBLASMP_STATUS_SUCCESS)
        status = lorrax_ffi::cublasmp::set_matmul_descriptor_attribute(mm,
            CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA,&op,sizeof(op));
    if (status == CUBLASMP_STATUS_SUCCESS)
        status = lorrax_ffi::cublasmp::set_matmul_descriptor_attribute(mm,
            CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB,&op,sizeof(op));
    // Valid device address token for descriptor-only sizing; not read.
    const auto* z = reinterpret_cast<const std::complex<double>*>(ctx->d_info);
    const auto* d = reinterpret_cast<const double*>(ctx->d_info);
    size_t dw=0, hw=0;
    if (status == CUBLASMP_STATUS_SUCCESS) {
        if (complex128) {
            const std::complex<double> alpha(1,0), beta(1,0);
            status = mp::MatmulBufferSize<std::complex<double>>(
                ctx->cublasmp_handle,mm,m,n,k,&alpha,z,1,1,a,
                z,1,1,b,&beta,z,1,1,c,const_cast<std::complex<double>*>(z),1,1,c,&dw,&hw);
        } else {
            const double alpha=1, beta=1;
            status = mp::MatmulBufferSize<double>(ctx->cublasmp_handle,mm,m,n,k,
                &alpha,d,1,1,a,d,1,1,b,&beta,
                d,1,1,c,const_cast<double*>(d),1,1,c,&dw,&hw);
        }
    }
    if (mm) cublasMpMatmulDescriptorDestroy(mm);
    if (c) cublasMpMatrixDescriptorDestroy(c);
    if (b) cublasMpMatrixDescriptorDestroy(b);
    if (a) cublasMpMatrixDescriptorDestroy(a);
    if (status == CUBLASMP_STATUS_SUCCESS) {
        *device_bytes=dw; *host_bytes=hw;
    }
    return int(status);
}
