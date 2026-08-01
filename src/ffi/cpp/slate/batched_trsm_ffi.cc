// batched_trsm_ffi.cc — XLA FFI handler for a batched slate::trsm call.
//
// Same pattern as batched_potrf_ffi.cc: plain C++ for-loop over the local
// batch dimension, each iteration runs slate::trsm on a per-X-row sub-
// communicator.  Batch is distributed across X, each triangular factor /
// RHS is distributed across Y.
//
// Expected sharding (from Python):
//   A : shape (Nbatch, N, N), P('x', None, 'y') row-major (or the
//       equivalent SlateBatchedLowerL opaque handle, which is already
//       transposed).  Per-rank local:  (Nbatch/Px, N, N/Py) row-major.
//       Python transposes the inner two dims → (Nbatch/Px, N/Py, N)
//       row-major ≡ col-major per-slice, which is what SLATE wants.
//
//   B : shape (Nbatch, N, Nrhs), P('x', None, 'y') row-major.
//       Per-rank local (Nbatch/Px, N, Nrhs/Py).  Python transposes inner
//       two dims identically.  On output X has the same shape/layout as B.
//
// Context is the sub-row SlateCtx (p=1, q=Py).  Only side='L', op∈{'N','C'}
// are exercised by the cholesky-solve callers; the handler still accepts
// the full (side, uplo, op, diag) matrix for uniformity.

#include <complex>
#include <cstdint>
#include <cstdio>
#include <sstream>

#include <cuda_runtime.h>
#include <blas.hh>
#include <slate/slate.hh>

#include "xla/ffi/api/ffi.h"

#include "../common/ffi_helpers.h"
#include "ctx.h"

namespace lorrax_ffi::slate_batched_trsm {

namespace ffi = ::xla::ffi;
namespace sl  = ::slate;

template <typename T>
static ffi::Error BatchedTrsmImpl(
    int64_t nbatch_local, int64_t n, int64_t m, int64_t nb,
    int side_int, int uplo_int, int op_int, int diag_int,
    double alpha_re, double alpha_im,
    cudaStream_t xla_stream,
    lorrax_ffi::slate::SlateCtx* ctx,
    const T* d_A,
    const T* d_B_in,
    T* d_X_out)
{
    cudaError_t ce = cudaStreamSynchronize(xla_stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "slate.batched_trsm: cudaStreamSynchronize(xla_stream) failed: "
           << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    const int num_devices_visible = blas::get_device_count();
    if (num_devices_visible != 1) {
        std::ostringstream os;
        os << "slate.batched_trsm: blas::get_device_count()="
           << num_devices_visible
           << " but JAX one-process-per-GPU model requires exactly 1.";
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition, os.str());
    }

    const int p = ctx->p;       // 1 by construction
    const int q = ctx->q;       // Py
    if (p != 1) {
        return ffi::Error(
            ffi::ErrorCode::kFailedPrecondition,
            "slate.batched_trsm: expected sub-row ctx with p=1");
    }

    // Per-slice local sizes.  A is n×n (col-major, lld=n).
    //   For side==Left:  B / X are n×m.  lld_B = n.
    //   For side==Right: B / X are m×n.  lld_B = m.
    const int64_t lld_A = n;
    const int64_t A_slice_elems = n * ((n + q - 1) / q);

    const sl::Side side = (side_int == 1) ? sl::Side::Right : sl::Side::Left;
    const sl::Uplo uplo = (uplo_int == 1) ? sl::Uplo::Upper : sl::Uplo::Lower;
    const sl::Diag diag = (diag_int == 1) ? sl::Diag::Unit  : sl::Diag::NonUnit;
    const sl::Op   op   = (op_int == 2)   ? sl::Op::ConjTrans
                        : (op_int == 1)   ? sl::Op::Trans
                        :                   sl::Op::NoTrans;

    const int64_t B_rows = (side == sl::Side::Left) ? n : m;
    const int64_t B_cols = (side == sl::Side::Left) ? m : n;
    const int64_t lld_B  = B_rows;
    const int64_t B_slice_elems = B_rows * ((B_cols + q - 1) / q);

    // Copy the entire B batch to X, then solve each slice in place.
    ce = cudaMemcpyAsync(d_X_out, d_B_in,
                         nbatch_local * B_slice_elems * sizeof(T),
                         cudaMemcpyDeviceToDevice, xla_stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "slate.batched_trsm: cudaMemcpyAsync(B->X) failed: "
           << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    ce = cudaStreamSynchronize(xla_stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "slate.batched_trsm: cudaStreamSynchronize after B->X failed: "
           << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    sl::Options opts = {
        {sl::Option::Target, sl::Target::Devices},
    };

    T alpha;
    if constexpr (std::is_same_v<T, double>) {
        alpha = alpha_re;
    } else {
        alpha = T(alpha_re, alpha_im);
    }

    for (int64_t b = 0; b < nbatch_local; ++b) {
        T* A_slice = const_cast<T*>(d_A) + b * A_slice_elems;
        T* X_slice = d_X_out + b * B_slice_elems;
        T* A_ptrs[1] = { A_slice };
        T* X_ptrs[1] = { X_slice };
        const int num_devices = 1;

        sl::TriangularMatrix<T> A = sl::TriangularMatrix<T>::fromDevices(
            uplo, diag, n,
            A_ptrs, num_devices, lld_A, nb, p, q, ctx->comm);

        // X tile sizes: rows conform with A's tiles (p=1 makes any row
        // tiling layout-consistent); the column tile must be one per
        // rank when q > 1 so SLATE's block-cyclic assignment matches
        // JAX's contiguous block shards.  A square-nb X here silently
        // corrupted (or crashed) any Nrhs != N solve — same defect as
        // the single-matrix trsm.
        const int64_t nb_X = (q == 1) ? nb : B_cols / q;
        sl::Matrix<T> X = sl::Matrix<T>::fromDevices(
            B_rows, B_cols,
            X_ptrs, num_devices, lld_B, nb, nb_X, p, q, ctx->comm);

        auto Aop = (op == sl::Op::Trans)     ? transpose(A)
                 : (op == sl::Op::ConjTrans) ? conj_transpose(A)
                 :                              A;

        try {
            sl::trsm(side, alpha, Aop, X, opts);
        } catch (const std::exception& ex) {
            std::ostringstream os;
            os << "slate::trsm (batch " << b << ") threw: " << ex.what();
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
    }

    return ffi::Error::Success();
}

static ffi::Error BatchedTrsmDispatch(
    cudaStream_t stream,
    ffi::AnyBuffer A,
    ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> X_out,
    int64_t nbatch_local, int64_t n, int64_t m, int64_t nb,
    int64_t side, int64_t uplo, int64_t op, int64_t diag,
    double alpha_re, double alpha_im,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<lorrax_ffi::slate::SlateCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "slate.batched_trsm: ctx_handle is null");
    }

    const auto dtype = A.element_type();
    if (B.element_type() != dtype || X_out->element_type() != dtype) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "slate.batched_trsm: A, B, X must all share the same dtype");
    }

    switch (dtype) {
        case ffi::DataType::F64:
            return BatchedTrsmImpl<double>(
                nbatch_local, n, m, nb,
                (int)side, (int)uplo, (int)op, (int)diag,
                alpha_re, alpha_im, stream, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<const double*>(B.untyped_data()),
                static_cast<double*>(X_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return BatchedTrsmImpl<C128>(
                nbatch_local, n, m, nb,
                (int)side, (int)uplo, (int)op, (int)diag,
                alpha_re, alpha_im, stream, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<const C128*>(B.untyped_data()),
                static_cast<C128*>(X_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "slate.batched_trsm: unsupported dtype "
               << static_cast<int>(dtype) << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::slate_batched_trsm

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SlateBatchedTrsmFfi, lorrax_ffi::slate_batched_trsm::BatchedTrsmDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()              // A (triangular)
        .Arg<xla::ffi::AnyBuffer>()              // B (RHS)
        .Ret<xla::ffi::AnyBuffer>()              // X
        .Attr<int64_t>("nbatch_local")
        .Attr<int64_t>("n")
        .Attr<int64_t>("m")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("side")
        .Attr<int64_t>("uplo")
        .Attr<int64_t>("op")
        .Attr<int64_t>("diag")
        .Attr<double>("alpha_re")
        .Attr<double>("alpha_im")
        .Attr<int64_t>("ctx_handle"));
