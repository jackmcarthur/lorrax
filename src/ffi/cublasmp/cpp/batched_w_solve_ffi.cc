// batched_w_solve_ffi.cc — fused distributed screened-Coulomb solve.
//
// Computes  W[q] = X[q] H[q]^{-1} X[q]^†  where
//   X[q]   = Cholesky(V[q])                 (cusolverMp potrf)
//   H[q]   = I - X[q]^† (pref * χ[q]) X[q]  (cublasMp gemms + identity kernel)
//
// Avoids the cuSOLVERMp-0.6.0 potrs-on-square-NRHS bug entirely by
// replacing potrs with two cublasMp trsm calls:
//   Y1 = X L_H^{-1}   (trsm side=R, uplo=L, trans=N)
//   Y  = Y1 L_H^{-H}  (trsm side=R, uplo=L, trans=C)
//   W  = Y X^†        (gemm opB=C)  ← X^† never materialised explicitly
//
// Sharding contract: V, χ, W all (Nq, N, N) sharded P(None, 'x', 'y').
// Python pre-transposes inner dims (standard col-major-bytes trick) so the
// FFI reads each local tile as col-major lld = N/Px with one tile per rank.

#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusolverMp.h>
#include <cublasmp.h>

#include "xla/ffi/api/ffi.h"

#include "../../common/cpp/ffi_helpers.h"
#include "../../cusolvermp/cpp/ctx.h"
#include "../../cusolvermp/cpp/cusolvermp_interface.h"
#include "cublasmp_interface.h"

// CUDA kernel launchers defined in w_solve_kernels.cu
extern "C" {
void lorrax_zero_upper_tri_f64(double* buf, int64_t mb, int64_t nb, int64_t nq,
                                int64_t x_idx, int64_t y_idx, cudaStream_t stream);
void lorrax_zero_upper_tri_c128(cuDoubleComplex* buf, int64_t mb, int64_t nb, int64_t nq,
                                 int64_t x_idx, int64_t y_idx, cudaStream_t stream);
void lorrax_identity_minus_T_f64(double* H_out, const double* T_in,
                                  int64_t mb, int64_t nb, int64_t nq,
                                  int64_t x_idx, int64_t y_idx, cudaStream_t stream);
void lorrax_identity_minus_T_c128(cuDoubleComplex* H_out, const cuDoubleComplex* T_in,
                                   int64_t mb, int64_t nb, int64_t nq,
                                   int64_t x_idx, int64_t y_idx, cudaStream_t stream);
}

namespace lorrax_ffi::cublasmp_w_solve {

namespace ffi = ::xla::ffi;
using lorrax_ffi::cusolvermp::LorraxCusolverMpCtx;
using lorrax_ffi::cusolvermp::ensure_cublasmp;
using lorrax_ffi::cusolvermp::ensure_workspace;
namespace cus = lorrax_ffi::cusolvermp::mp;
namespace cub = lorrax_ffi::cublasmp::mp;

#define LORRAX_CUBLASMP_CHECK(expr, what)                                  \
    do {                                                                   \
        cublasMpStatus_t _st = (expr);                                     \
        if (_st != CUBLASMP_STATUS_SUCCESS) {                              \
            std::ostringstream _os;                                        \
            _os << (what) << " failed: status=" << (int)_st;               \
            return ffi::Error(ffi::ErrorCode::kInternal, _os.str());       \
        }                                                                  \
    } while (0)

#define LORRAX_CUSOLVERMP_CHECK(expr, what)                                \
    do {                                                                   \
        cusolverStatus_t _st = (expr);                                     \
        if (_st != CUSOLVER_STATUS_SUCCESS) {                              \
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

// Dispatcher for the tril-mask kernel.
template <typename T>
static void launch_zero_upper_tri(T* buf, int64_t mb, int64_t nb, int64_t nq,
                                   int64_t x_idx, int64_t y_idx, cudaStream_t stream);
template <>
void launch_zero_upper_tri<double>(double* buf, int64_t mb, int64_t nb, int64_t nq,
                                    int64_t x_idx, int64_t y_idx, cudaStream_t stream) {
    lorrax_zero_upper_tri_f64(buf, mb, nb, nq, x_idx, y_idx, stream);
}
template <>
void launch_zero_upper_tri<std::complex<double>>(
    std::complex<double>* buf, int64_t mb, int64_t nb, int64_t nq,
    int64_t x_idx, int64_t y_idx, cudaStream_t stream) {
    lorrax_zero_upper_tri_c128(reinterpret_cast<cuDoubleComplex*>(buf),
                                mb, nb, nq, x_idx, y_idx, stream);
}

template <typename T>
static void launch_identity_minus_T(T* H_out, const T* T_in,
                                     int64_t mb, int64_t nb, int64_t nq,
                                     int64_t x_idx, int64_t y_idx,
                                     cudaStream_t stream);
template <>
void launch_identity_minus_T<double>(
    double* H_out, const double* T_in,
    int64_t mb, int64_t nb, int64_t nq,
    int64_t x_idx, int64_t y_idx, cudaStream_t stream) {
    lorrax_identity_minus_T_f64(H_out, T_in, mb, nb, nq, x_idx, y_idx, stream);
}
template <>
void launch_identity_minus_T<std::complex<double>>(
    std::complex<double>* H_out, const std::complex<double>* T_in,
    int64_t mb, int64_t nb, int64_t nq,
    int64_t x_idx, int64_t y_idx, cudaStream_t stream) {
    lorrax_identity_minus_T_c128(
        reinterpret_cast<cuDoubleComplex*>(H_out),
        reinterpret_cast<const cuDoubleComplex*>(T_in),
        mb, nb, nq, x_idx, y_idx, stream);
}

template <typename T>
static ffi::Error BatchedWSolveImpl(
    int64_t nq, int64_t n,
    T pref, T one, T zero,
    cudaStream_t xla_stream,
    LorraxCusolverMpCtx* ctx,
    const T* d_V_in, const T* d_chi_in,
    T* d_V_work, T* d_W_out,
    int64_t stop_after_step)
{
    ensure_cublasmp(ctx);

    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        ctx->stream, xla_stream, ctx->ev_xla_in));

    const int Px = ctx->p;
    const int Py = ctx->q;
    const int64_t mb = (n + Px - 1) / Px;
    const int64_t nb = (n + Py - 1) / Py;
    const int64_t lld = mb;
    const int64_t slice = mb * nb;      // complex elements per q-slice per rank

    // Each rank's (x_idx, y_idx) in the ROW-major grid (matches our ctx
    // with col_major=false: rank = x*Py + y).
    const int64_t x_idx = ctx->rank / Py;
    const int64_t y_idx = ctx->rank % Py;

    // Scratch buffers (per-call).  Use cudaMallocAsync on the ctx stream
    // so the CUDA pool reuses them across chunk-loop iterations.
    // d_V_work holds our in-place Cholesky factor; we never modify the
    // caller's V buffer (downstream code — e.g. sigma_coh — consumes
    // the original V).
    const size_t scratch_bytes = static_cast<size_t>(nq) * slice * sizeof(T);
    T *d_T_buf = nullptr, *d_T2_buf = nullptr, *d_Z_buf = nullptr;
    T *d_V_work_local = nullptr;
    LORRAX_CUDA_CHECK(cudaMallocAsync(
        reinterpret_cast<void**>(&d_V_work_local), scratch_bytes, ctx->stream));
    LORRAX_CUDA_CHECK(cudaMallocAsync(
        reinterpret_cast<void**>(&d_T_buf),  scratch_bytes, ctx->stream));
    LORRAX_CUDA_CHECK(cudaMallocAsync(
        reinterpret_cast<void**>(&d_T2_buf), scratch_bytes, ctx->stream));
    LORRAX_CUDA_CHECK(cudaMallocAsync(
        reinterpret_cast<void**>(&d_Z_buf),  scratch_bytes, ctx->stream));

    // Copy V_in → V_work so we can factor in place without touching V.
    LORRAX_CUDA_CHECK(cudaMemcpyAsync(
        d_V_work_local, d_V_in, scratch_bytes,
        cudaMemcpyDeviceToDevice, ctx->stream));
    d_V_work = d_V_work_local;

    // Matrix descriptors.  All six in-scope matrices (X=V_work, chi, T,
    // T2/H/L_H, Z, W_out) share the same (N, N) shape and block layout.
    cusolverMpMatrixDescriptor_t desc_cus = nullptr;
    cublasMpMatrixDescriptor_t    desc_blas = nullptr;
    LORRAX_CUSOLVERMP_CHECK(
        cusolverMpCreateMatrixDesc(
            &desc_cus, ctx->grid, cus::CudaDataTypeOf<T>::value,
            n, n, mb, nb, 0, 0, lld),
        "cusolverMpCreateMatrixDesc");
    LORRAX_CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(
            n, n, mb, nb, 0, 0, lld,
            cub::CudaDataTypeOf<T>::value, ctx->cublasmp_grid, &desc_blas),
        "cublasMpMatrixDescriptorCreate");

    cublasMpMatmulDescriptor_t mm_desc = nullptr;
    LORRAX_CUBLASMP_CHECK(
        cublasMpMatmulDescriptorCreate(&mm_desc, cub::ComputeTypeOf<T>::value),
        "cublasMpMatmulDescriptorCreate");

    auto set_mm_trans = [&](cublasOperation_t opA, cublasOperation_t opB)
        -> ffi::Error {
        LORRAX_CUBLASMP_CHECK(
            cublasMpMatmulDescriptorSetAttribute(
                mm_desc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA,
                &opA, sizeof(opA)),
            "matmulDesc TRANSA set");
        LORRAX_CUBLASMP_CHECK(
            cublasMpMatmulDescriptorSetAttribute(
                mm_desc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB,
                &opB, sizeof(opB)),
            "matmulDesc TRANSB set");
        return ffi::Error::Success();
    };

    auto cleanup = [&]() {
        cublasMpMatmulDescriptorDestroy(mm_desc);
        cublasMpMatrixDescriptorDestroy(desc_blas);
        cusolverMpDestroyMatrixDesc(desc_cus);
        cudaFreeAsync(d_V_work_local, ctx->stream);
        cudaFreeAsync(d_T_buf,  ctx->stream);
        cudaFreeAsync(d_T2_buf, ctx->stream);
        cudaFreeAsync(d_Z_buf,  ctx->stream);
    };

    // Unified workspace sizing.  Query each operation's bufferSize once
    // (on slice 0) and take the max, then ensure ctx->d_workspace is
    // large enough.  We reuse the same device/host workspace across
    // cuSOLVERMp potrfs AND cuBLASMp gemms/trsms — this is technically
    // mixing libraries on the same scratch, but the workspace is just
    // raw bytes and neither library holds state between calls.
    size_t d_ws_need = 0, h_ws_need = 0;
    {
        size_t d_ws = 0, h_ws = 0;
        // potrf(V)
        LORRAX_CUSOLVERMP_CHECK(
            cus::PotrfBufferSize<T>(
                ctx->handle, CUBLAS_FILL_MODE_LOWER, n,
                d_V_work, 1, 1, desc_cus, &d_ws, &h_ws),
            "potrf bufferSize");
        d_ws_need = std::max(d_ws_need, d_ws);
        h_ws_need = std::max(h_ws_need, h_ws);

        // gemm (use opA=C for the X^H @ chi sizing — cuBLASMp usually
        // returns the same size regardless of trans combo)
        cublasOperation_t opA = CUBLAS_OP_C, opB = CUBLAS_OP_N;
        cublasMpMatmulDescriptorSetAttribute(
            mm_desc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA,
            &opA, sizeof(opA));
        cublasMpMatmulDescriptorSetAttribute(
            mm_desc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB,
            &opB, sizeof(opB));
        LORRAX_CUBLASMP_CHECK(
            cub::MatmulBufferSize<T>(
                ctx->cublasmp_handle, mm_desc, n, n, n,
                &pref, d_V_work, 1, 1, desc_blas,
                       d_chi_in, 1, 1, desc_blas,
                &zero, d_T_buf, 1, 1, desc_blas,
                       d_T_buf, 1, 1, desc_blas,
                &d_ws, &h_ws),
            "gemm bufferSize");
        d_ws_need = std::max(d_ws_need, d_ws);
        h_ws_need = std::max(h_ws_need, h_ws);

        // trsm
        LORRAX_CUBLASMP_CHECK(
            cub::TrsmBufferSize<T>(
                ctx->cublasmp_handle,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                n, n, &one,
                d_T_buf,  1, 1, desc_blas,
                d_Z_buf,  1, 1, desc_blas,
                &d_ws, &h_ws),
            "trsm bufferSize");
        d_ws_need = std::max(d_ws_need, d_ws);
        h_ws_need = std::max(h_ws_need, h_ws);
    }
    try {
        ensure_workspace(ctx, d_ws_need, h_ws_need);
    } catch (const std::exception& ex) {
        cleanup();
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, ex.what());
    }

    // ------------------------------------------------------------------
    // Main per-q loop: 10 steps, no JAX boundary, all on ctx->stream.
    // ------------------------------------------------------------------
    for (int64_t q = 0; q < nq; ++q) {
        T* X_q    = d_V_work  + q * slice;      // X (Cholesky of V), overwrites V
        const T* chi_q = d_chi_in + q * slice;
        T* T_q    = d_T_buf   + q * slice;      // X^H chi
        T* T2_q   = d_T2_buf  + q * slice;      // T = (X^H chi) X, then H, then L_H
        T* Z_q    = d_Z_buf   + q * slice;      // X H^{-1}
        T* W_q    = d_W_out   + q * slice;

        // 1.  potrf(V_q) → X_q  (lower-tri Cholesky, in place on V's buffer)
        LORRAX_CUSOLVERMP_CHECK(
            cus::Potrf<T>(
                ctx->handle, CUBLAS_FILL_MODE_LOWER, n,
                X_q, 1, 1, desc_cus,
                ctx->d_workspace, ctx->d_workspace_bytes,
                ctx->h_workspace, ctx->h_workspace_bytes,
                ctx->d_info),
            "potrf(V) q=" + std::to_string(q));
        if (stop_after_step == 1) {
            LORRAX_CUDA_CHECK(cudaMemcpyAsync(
                W_q, X_q, slice * sizeof(T),
                cudaMemcpyDeviceToDevice, ctx->stream));
            goto done;
        }

        // 2.  Zero upper triangle of X (cuSOLVERMp leaves it untouched).
        launch_zero_upper_tri<T>(X_q, mb, nb, 1, x_idx, y_idx, ctx->stream);
        if (stop_after_step == 2) {
            LORRAX_CUDA_CHECK(cudaMemcpyAsync(
                W_q, X_q, slice * sizeof(T),
                cudaMemcpyDeviceToDevice, ctx->stream));
            goto done;
        }

        // 3.  T_tmp = pref * X^H @ chi   (gemm, opA=C, opB=N)
        FFI_RETURN_IF_ERROR(set_mm_trans(CUBLAS_OP_C, CUBLAS_OP_N));
        LORRAX_CUBLASMP_CHECK(
            cub::Matmul<T>(
                ctx->cublasmp_handle, mm_desc, n, n, n,
                &pref,  X_q,    1, 1, desc_blas,
                        chi_q,  1, 1, desc_blas,
                &zero,  T_q,    1, 1, desc_blas,
                        T_q,    1, 1, desc_blas,
                ctx->d_workspace, ctx->d_workspace_bytes,
                ctx->h_workspace, ctx->h_workspace_bytes),
            "gemm T_tmp q=" + std::to_string(q));
        if (stop_after_step == 3) {
            LORRAX_CUDA_CHECK(cudaMemcpyAsync(
                W_q, T_q, slice * sizeof(T),
                cudaMemcpyDeviceToDevice, ctx->stream));
            goto done;
        }

        // 4.  T = T_tmp @ X   (gemm, opA=N, opB=N)
        FFI_RETURN_IF_ERROR(set_mm_trans(CUBLAS_OP_N, CUBLAS_OP_N));
        LORRAX_CUBLASMP_CHECK(
            cub::Matmul<T>(
                ctx->cublasmp_handle, mm_desc, n, n, n,
                &one,  T_q,   1, 1, desc_blas,
                       X_q,   1, 1, desc_blas,
                &zero, T2_q,  1, 1, desc_blas,
                       T2_q,  1, 1, desc_blas,
                ctx->d_workspace, ctx->d_workspace_bytes,
                ctx->h_workspace, ctx->h_workspace_bytes),
            "gemm T q=" + std::to_string(q));
        if (stop_after_step == 4) {
            LORRAX_CUDA_CHECK(cudaMemcpyAsync(
                W_q, T2_q, slice * sizeof(T),
                cudaMemcpyDeviceToDevice, ctx->stream));
            goto done;
        }

        // 5.  H = I - T   (CUDA kernel, writes into T_q buffer,
        //                  reading from T2_q).
        launch_identity_minus_T<T>(T_q, T2_q, mb, nb, 1, x_idx, y_idx, ctx->stream);
        if (stop_after_step == 5) {
            LORRAX_CUDA_CHECK(cudaMemcpyAsync(
                W_q, T_q, slice * sizeof(T),
                cudaMemcpyDeviceToDevice, ctx->stream));
            goto done;
        }

        // 6.  potrf(H) → L_H   (in place on T_q)
        LORRAX_CUSOLVERMP_CHECK(
            cus::Potrf<T>(
                ctx->handle, CUBLAS_FILL_MODE_LOWER, n,
                T_q, 1, 1, desc_cus,
                ctx->d_workspace, ctx->d_workspace_bytes,
                ctx->h_workspace, ctx->h_workspace_bytes,
                ctx->d_info),
            "potrf(H) q=" + std::to_string(q));
        if (stop_after_step == 6) {
            LORRAX_CUDA_CHECK(cudaMemcpyAsync(
                W_q, T_q, slice * sizeof(T),
                cudaMemcpyDeviceToDevice, ctx->stream));
            goto done;
        }

        // 7.  Copy X → Z_q   (trsm uses in-place RHS and we need X
        //                      preserved for the final gemm).
        LORRAX_CUDA_CHECK(cudaMemcpyAsync(
            Z_q, X_q, slice * sizeof(T),
            cudaMemcpyDeviceToDevice, ctx->stream));
        if (stop_after_step == 7) {
            LORRAX_CUDA_CHECK(cudaMemcpyAsync(
                W_q, Z_q, slice * sizeof(T),
                cudaMemcpyDeviceToDevice, ctx->stream));
            goto done;
        }

        // H^{-1} = L_H^{-H} L_H^{-1}  (since H = L_H L_H^H), so to compute
        // Z = X H^{-1} we apply L_H^{-H} FIRST, then L_H^{-1}.  Applying
        // them in the other order would compute X (L_H^H L_H)^{-1}, which
        // is wrong.
        //
        // 8.  Z = X L_H^{-H}    (trsm side=RIGHT, uplo=L, trans=C)
        LORRAX_CUBLASMP_CHECK(
            cub::Trsm<T>(
                ctx->cublasmp_handle,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_C, CUBLAS_DIAG_NON_UNIT,
                n, n, &one,
                T_q, 1, 1, desc_blas,
                Z_q, 1, 1, desc_blas,
                ctx->d_workspace, ctx->d_workspace_bytes,
                ctx->h_workspace, ctx->h_workspace_bytes),
            "trsm L^{-H} q=" + std::to_string(q));
        if (stop_after_step == 8) {
            LORRAX_CUDA_CHECK(cudaMemcpyAsync(
                W_q, Z_q, slice * sizeof(T),
                cudaMemcpyDeviceToDevice, ctx->stream));
            goto done;
        }

        // 9.  Z = Z L_H^{-1}    (trsm side=RIGHT, uplo=L, trans=N)
        LORRAX_CUBLASMP_CHECK(
            cub::Trsm<T>(
                ctx->cublasmp_handle,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                n, n, &one,
                T_q, 1, 1, desc_blas,
                Z_q, 1, 1, desc_blas,
                ctx->d_workspace, ctx->d_workspace_bytes,
                ctx->h_workspace, ctx->h_workspace_bytes),
            "trsm L^{-1} q=" + std::to_string(q));
        if (stop_after_step == 9) {
            LORRAX_CUDA_CHECK(cudaMemcpyAsync(
                W_q, Z_q, slice * sizeof(T),
                cudaMemcpyDeviceToDevice, ctx->stream));
            goto done;
        }

        // 10. W = Z @ X^H       (gemm opA=N, opB=C — X^H computed
        //                         internally; never materialised.)
        FFI_RETURN_IF_ERROR(set_mm_trans(CUBLAS_OP_N, CUBLAS_OP_C));
        LORRAX_CUBLASMP_CHECK(
            cub::Matmul<T>(
                ctx->cublasmp_handle, mm_desc, n, n, n,
                &one,  Z_q,   1, 1, desc_blas,
                       X_q,   1, 1, desc_blas,
                &zero, W_q,   1, 1, desc_blas,
                       W_q,   1, 1, desc_blas,
                ctx->d_workspace, ctx->d_workspace_bytes,
                ctx->h_workspace, ctx->h_workspace_bytes),
            "gemm W q=" + std::to_string(q));
    }
done:

    cleanup();

    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        xla_stream, ctx->stream, ctx->ev_ctx_out));
    return ffi::Error::Success();
}

static ffi::Error BatchedWSolveDispatch(
    cudaStream_t stream,
    ffi::AnyBuffer V,
    ffi::AnyBuffer chi,
    ffi::Result<ffi::AnyBuffer> W_out,
    int64_t nq, int64_t n,
    double pref_re, double pref_im,
    int64_t ctx_handle,
    int64_t stop_after_step)
{
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "w_solve: ctx_handle is null");
    }
    const auto dtype = V.element_type();
    if (chi.element_type() != dtype || W_out->element_type() != dtype) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "w_solve: V, chi, W must share dtype");
    }
    // V is donated: we write X (Cholesky factor) in place on V's buffer.
    // The `const` on V.untyped_data() is a lie we un-cast once, protected
    // by the XLA input_output_aliases contract on the Python side.
    switch (dtype) {
        case ffi::DataType::F64: {
            // pref is real for F64
            return BatchedWSolveImpl<double>(
                nq, n,
                pref_re, 1.0, 0.0, stream, ctx,
                static_cast<const double*>(V.untyped_data()),
                static_cast<const double*>(chi.untyped_data()),
                // In-place on V's buffer (V is donated).
                const_cast<double*>(static_cast<const double*>(V.untyped_data())),
                static_cast<double*>(W_out->untyped_data()),
                stop_after_step);
        }
        case ffi::DataType::C128: {
            using C128 = std::complex<double>;
            return BatchedWSolveImpl<C128>(
                nq, n,
                C128(pref_re, pref_im), C128(1.0, 0.0), C128(0.0, 0.0),
                stream, ctx,
                static_cast<const C128*>(V.untyped_data()),
                static_cast<const C128*>(chi.untyped_data()),
                const_cast<C128*>(static_cast<const C128*>(V.untyped_data())),
                static_cast<C128*>(W_out->untyped_data()),
                stop_after_step);
        }
        default: {
            std::ostringstream os;
            os << "w_solve: unsupported dtype " << (int)dtype
               << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::cublasmp_w_solve

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CublasMpBatchedWSolveFfi,
    lorrax_ffi::cublasmp_w_solve::BatchedWSolveDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()      // V (donated)
        .Arg<xla::ffi::AnyBuffer>()      // chi
        .Ret<xla::ffi::AnyBuffer>()      // W
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<double>("pref_re")
        .Attr<double>("pref_im")
        .Attr<int64_t>("ctx_handle")
        .Attr<int64_t>("stop_after_step"));
