// potrf_ffi.cc — XLA FFI handler for slate::potrf (distributed Cholesky
// factorization of a Hermitian positive-definite matrix).
//
// In-place op: A -> L (Uplo::Lower) with A = L L^H.  We expose a
// (Arg, Ret) pair (not XLA donation) for clarity — the handler does a
// D2D copy from the input buffer into the output buffer and then runs
// SLATE in-place on the output.  JAX's aliasing pass may collapse this
// into a true in-place op when the caller donates the input.
//
// Layout contract (same as eigh_ffi.cc; see ctx.h):
//   - Tiles are col-major with lda = nb.
//   - SLATE's fromDevices uses GridOrder::Col (rank(ti, tj) = ti + tj*p).
//   - JAX's P("x", "y") shard layout is effectively GridOrder::Row for
//     (x, y) mesh axis names, so SLATE sees A^T globally.  For a
//     Hermitian A, A^T = conj(A), so L returned by potrf on A^T is
//     conj(L_original).  Returning conj(L).T gives the caller the
//     Cholesky factor of their JAX A in the conventional column-major
//     SLATE sense; as a shard, they'd typically want L.T (real) or
//     L.conj().T (complex) to recover standard "lower triangular in
//     JAX row-major" form.  Tests document the exact transform.

#include <complex>
#include <cstdint>
#include <cstdio>
#include <sstream>

#include <cuda_runtime.h>
#include <blas.hh>
#include <slate/slate.hh>

#include "xla/ffi/api/ffi.h"

#include "../../common/cpp/ffi_helpers.h"
#include "ctx.h"

namespace lorrax_ffi::slate_potrf {

namespace ffi = ::xla::ffi;
namespace sl  = ::slate;

template <typename T>
static ffi::Error PotrfImpl(
    int64_t n, int64_t nb,
    cudaStream_t xla_stream,
    lorrax_ffi::slate::SlateCtx* ctx,
    const T* d_A_in,
    T* d_L_out)
{
    cudaError_t ce = cudaStreamSynchronize(xla_stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "slate.potrf: cudaStreamSynchronize(xla_stream) failed: "
           << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    const int num_devices_visible = blas::get_device_count();
    if (num_devices_visible != 1) {
        std::ostringstream os;
        os << "slate.potrf: blas::get_device_count()=" << num_devices_visible
           << " but JAX one-process-per-GPU model requires exactly 1. "
           << "Set CUDA_VISIBLE_DEVICES=$SLURM_LOCALID (LORRAX_SELECT_GPU=1).";
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition, os.str());
    }

    const int p = ctx->p;
    const int q = ctx->q;
    const int64_t lld = (n + p - 1) / p;

    // Copy the input shard into the output buffer, then factor in-place.
    const int64_t local_rows = (n + p - 1) / p;
    const int64_t local_cols = (n + q - 1) / q;
    ce = cudaMemcpyAsync(d_L_out, d_A_in,
                         local_rows * local_cols * sizeof(T),
                         cudaMemcpyDeviceToDevice, xla_stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "slate.potrf: cudaMemcpyAsync(A -> L) failed: "
           << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    // Let the memcpy finish before SLATE touches the tiles.
    ce = cudaStreamSynchronize(xla_stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "slate.potrf: cudaStreamSynchronize after A->L copy failed: "
           << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    T* L_ptrs[1] = { d_L_out };
    const int num_devices = 1;

    sl::HermitianMatrix<T> L = sl::HermitianMatrix<T>::fromDevices(
        sl::Uplo::Lower, n, L_ptrs, num_devices, lld, nb, p, q, ctx->comm);

    sl::Options opts = {
        {sl::Option::Target, sl::Target::Devices},
    };

    try {
        int64_t info = sl::potrf(L, opts);
        if (info != 0) {
            std::ostringstream os;
            os << "slate::potrf returned info=" << info
               << " (matrix not positive-definite at that leading minor)";
            return ffi::Error(ffi::ErrorCode::kFailedPrecondition, os.str());
        }
    } catch (const std::exception& ex) {
        std::ostringstream os;
        os << "slate::potrf threw: " << ex.what();
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    // Zero out the strict-upper tiles (in SLATE's tile grid) so the
    // resulting buffer is "clean L" — anything not written by potrf is
    // explicitly zero, no leftover A-data.  This matters for the chained
    // cholesky->trsm pattern: if we leave A's strict-upper data in
    // place, trsm's Triangular(Lower) wrapper still col-majorises that
    // region and uses some of it (despite uplo='Lower'), giving a few-
    // percent residual.
    //
    // Default-tile case (nb == n/p): each rank owns exactly one tile
    // (ti, tj) = (rank%p, rank/p) under SLATE GridOrder::Col.  Strict
    // upper means ti < tj.
    if (nb == (n + p - 1) / p) {
        const int my_ti = ctx->rank % p;
        const int my_tj = ctx->rank / p;
        if (my_ti < my_tj) {
            cudaError_t cz = cudaMemsetAsync(
                d_L_out, 0,
                local_rows * local_cols * sizeof(T),
                xla_stream);
            if (cz != cudaSuccess) {
                std::ostringstream os;
                os << "slate.potrf: cudaMemsetAsync(strict-upper tile) failed: "
                   << cudaGetErrorString(cz);
                return ffi::Error(ffi::ErrorCode::kInternal, os.str());
            }
        }
    }
    // For non-default nb (multiple tiles per rank), the right zero
    // pattern is per-tile within the rank's buffer; not implemented yet
    // since user code uses default nb (= n/p).

    return ffi::Error::Success();
}

static ffi::Error PotrfDispatch(
    cudaStream_t stream,
    ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> L_out,
    int64_t n, int64_t nb,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<lorrax_ffi::slate::SlateCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "slate.potrf: ctx_handle is null");
    }

    const auto dtype = A.element_type();
    if (L_out->element_type() != dtype) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "slate.potrf: L output dtype must match A input dtype");
    }

    switch (dtype) {
        case ffi::DataType::F64:
            return PotrfImpl<double>(
                n, nb, stream, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<double*>(L_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return PotrfImpl<C128>(
                n, nb, stream, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<C128*>(L_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "slate.potrf: unsupported dtype "
               << static_cast<int>(dtype) << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::slate_potrf

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SlatePotrfFfi, lorrax_ffi::slate_potrf::PotrfDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()              // A (Hermitian input)
        .Ret<xla::ffi::AnyBuffer>()              // L (Cholesky factor, lower)
        .Attr<int64_t>("n")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_handle"));
