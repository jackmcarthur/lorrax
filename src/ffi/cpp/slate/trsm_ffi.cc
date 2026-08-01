// trsm_ffi.cc — XLA FFI handler for slate::trsm (distributed triangular
// solve):    op(A) * X = alpha * B   (side == Left)
//         or       X * op(A) = alpha * B   (side == Right),
// with A triangular (Upper or Lower), op in {NoTrans, Trans, ConjTrans}.
// In-place on B: the solution X overwrites B.
//
// FFI I/O: A and B are inputs; X is output (same shape + dtype as B).
// Handler copies B -> X then runs slate::trsm in-place on X.  A is not
// modified.
//
// Layout contract: identical to potrf_ffi.cc / eigh_ffi.cc.
// A and B use the same ("x", "y") mesh, 2-D block-cyclic, col-major tiles
// with lda = nb, one tile per rank for nb = n/p.

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

namespace lorrax_ffi::slate_trsm {

namespace ffi = ::xla::ffi;
namespace sl  = ::slate;

template <typename T>
static ffi::Error TrsmImpl(
    int64_t n, int64_t m, int64_t nb,
    int side_int,     // 0 = Left, 1 = Right
    int uplo_int,     // 0 = Lower, 1 = Upper
    int op_int,       // 0 = NoTrans, 1 = Trans, 2 = ConjTrans
    int diag_int,     // 0 = NonUnit, 1 = Unit
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
        os << "slate.trsm: cudaStreamSynchronize(xla_stream) failed: "
           << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    const int num_devices_visible = blas::get_device_count();
    if (num_devices_visible != 1) {
        std::ostringstream os;
        os << "slate.trsm: blas::get_device_count()=" << num_devices_visible
           << " but JAX one-process-per-GPU model requires exactly 1.";
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition, os.str());
    }

    const int p = ctx->p;
    const int q = ctx->q;

    sl::Side side = (side_int == 1) ? sl::Side::Right : sl::Side::Left;

    // A is n×n; its local leading dim:
    const int64_t lld_A = (n + p - 1) / p;
    // B/X shape: (n, m) for side==Left (op(A) X = alpha B), (m, n) for
    // side==Right (X op(A) = alpha B).
    const int64_t X_rows = (side == sl::Side::Left) ? n : m;
    const int64_t X_cols = (side == sl::Side::Left) ? m : n;
    const int64_t lld_B = (X_rows + p - 1) / p;
    // Count how many elements of B/X are local to this rank.
    const int64_t local_rows = (X_rows + p - 1) / p;
    const int64_t local_cols = (X_cols + q - 1) / q;
    ce = cudaMemcpyAsync(d_X_out, d_B_in,
                         local_rows * local_cols * sizeof(T),
                         cudaMemcpyDeviceToDevice, xla_stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "slate.trsm: cudaMemcpyAsync(B -> X) failed: "
           << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    ce = cudaStreamSynchronize(xla_stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "slate.trsm: cudaStreamSynchronize after B->X copy failed: "
           << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    // Wrap A as a TriangularMatrix pointing at the caller's device shard.
    T* A_ptrs[1] = { const_cast<T*>(d_A) };
    T* X_ptrs[1] = { d_X_out };
    const int num_devices = 1;

    sl::Uplo uplo = (uplo_int == 1) ? sl::Uplo::Upper : sl::Uplo::Lower;
    sl::Diag diag = (diag_int == 1) ? sl::Diag::Unit  : sl::Diag::NonUnit;
    sl::Op   op   = (op_int == 2)   ? sl::Op::ConjTrans
                  : (op_int == 1)   ? sl::Op::Trans
                  :                   sl::Op::NoTrans;

    sl::TriangularMatrix<T> A = sl::TriangularMatrix<T>::fromDevices(
        uplo, diag, n,
        A_ptrs, num_devices, lld_A, nb, p, q, ctx->comm);

    // X tile sizes: along the solve dimension X's tiles must conform
    // with A's (= nb); along the free dimension the tile size must make
    // SLATE's block-cyclic assignment coincide with JAX's contiguous
    // block shards — one tile per rank when that mesh axis is >1 (a
    // square-nb X here was the m != n crash/corruption: tile (i,j) of a
    // 64×32 X with nb=32 on a 2×2 grid lands on ranks that hold no X
    // data at all).  With one proc on the axis any tile size is
    // layout-consistent (all tiles local, contiguous col-major).
    int64_t mb_X, nb_X;
    if (side == sl::Side::Left) {           // X is n × m
        mb_X = nb;
        nb_X = (q == 1) ? nb : X_cols / q;
    } else {                                // X is m × n
        mb_X = (p == 1) ? nb : X_rows / p;
        nb_X = nb;
    }
    sl::Matrix<T> X = sl::Matrix<T>::fromDevices(
        X_rows, X_cols, X_ptrs, num_devices, lld_B, mb_X, nb_X,
        p, q, ctx->comm);

    // Apply op() to A via SLATE's in-matrix transform (no data movement).
    auto Aop = (op == sl::Op::Trans)     ? transpose(A)
             : (op == sl::Op::ConjTrans) ? conj_transpose(A)
             :                              A;

    sl::Options opts = {
        {sl::Option::Target, sl::Target::Devices},
    };

    T alpha;
    if constexpr (std::is_same_v<T, double>) {
        alpha = alpha_re;
    } else {
        alpha = T(alpha_re, alpha_im);
    }

    try {
        sl::trsm(side, alpha, Aop, X, opts);
    } catch (const std::exception& ex) {
        std::ostringstream os;
        os << "slate::trsm threw: " << ex.what();
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    return ffi::Error::Success();
}

static ffi::Error TrsmDispatch(
    cudaStream_t stream,
    ffi::AnyBuffer A,
    ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> X_out,
    int64_t n, int64_t m, int64_t nb,
    int64_t side, int64_t uplo, int64_t op, int64_t diag,
    double alpha_re, double alpha_im,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<lorrax_ffi::slate::SlateCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "slate.trsm: ctx_handle is null");
    }

    const auto dtype = A.element_type();
    if (B.element_type() != dtype || X_out->element_type() != dtype) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "slate.trsm: A, B, X must all share the same dtype");
    }

    switch (dtype) {
        case ffi::DataType::F64:
            return TrsmImpl<double>(
                n, m, nb, (int)side, (int)uplo, (int)op, (int)diag,
                alpha_re, alpha_im, stream, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<const double*>(B.untyped_data()),
                static_cast<double*>(X_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return TrsmImpl<C128>(
                n, m, nb, (int)side, (int)uplo, (int)op, (int)diag,
                alpha_re, alpha_im, stream, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<const C128*>(B.untyped_data()),
                static_cast<C128*>(X_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "slate.trsm: unsupported dtype "
               << static_cast<int>(dtype) << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::slate_trsm

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SlateTrsmFfi, lorrax_ffi::slate_trsm::TrsmDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()              // A (triangular)
        .Arg<xla::ffi::AnyBuffer>()              // B (right-hand side)
        .Ret<xla::ffi::AnyBuffer>()              // X (solution, same shape as B)
        .Attr<int64_t>("n")
        .Attr<int64_t>("m")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("side")    // 0 Left, 1 Right
        .Attr<int64_t>("uplo")    // 0 Lower, 1 Upper
        .Attr<int64_t>("op")      // 0 NoTrans, 1 Trans, 2 ConjTrans
        .Attr<int64_t>("diag")    // 0 NonUnit, 1 Unit
        .Attr<double>("alpha_re")
        .Attr<double>("alpha_im")
        .Attr<int64_t>("ctx_handle"));
