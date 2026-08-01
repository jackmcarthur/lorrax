// batched_potrf_ffi.cc — XLA FFI handler for a batched slate::potrf call.
//
// SLATE has no native batched potrf.  This handler is a plain C++ for-loop
// over the local batch dimension, calling slate::potrf on each slice.  The
// batch is distributed *across* one procgrid axis (X), and each (N, N)
// matrix is distributed across the other axis (Y).  All ranks with the
// same x-coordinate share a sub-communicator (size Py) and cooperate on
// the Nbatch/Px slices assigned to their row.
//
// Expected sharding (from the Python wrapper):
//   A : shape (Nbatch, N, N), P('x', None, 'y') row-major.
//     Per-rank local bytes: (Nbatch/Px, N, N/Py) row-major.
//     Python transposes the inner two dims before calling this handler,
//     producing (Nbatch/Px, N/Py, N) row-major ≡ (Nbatch/Px, N, N/Py)
//     col-major per-slice — which is the layout SLATE's fromDevices
//     expects with p=1, q=Py, lld=N.
//
// Context (ctx_handle) must be the *sub-row* SlateCtx:
//   p=1, q=Py, comm = split of MPI_COMM_WORLD by x_rank.

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

namespace lorrax_ffi::slate_batched_potrf {

namespace ffi = ::xla::ffi;
namespace sl  = ::slate;

template <typename T>
static ffi::Error BatchedPotrfImpl(
    int64_t nbatch_local, int64_t n, int64_t nb,
    cudaStream_t xla_stream,
    lorrax_ffi::slate::SlateCtx* ctx,
    const T* d_A_in,
    T* d_L_out)
{
    cudaError_t ce = cudaStreamSynchronize(xla_stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "slate.batched_potrf: cudaStreamSynchronize(xla_stream) failed: "
           << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    const int num_devices_visible = blas::get_device_count();
    if (num_devices_visible != 1) {
        std::ostringstream os;
        os << "slate.batched_potrf: blas::get_device_count()="
           << num_devices_visible
           << " but JAX one-process-per-GPU model requires exactly 1.";
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition, os.str());
    }

    const int p = ctx->p;           // 1 by construction
    const int q = ctx->q;           // Py
    if (p != 1) {
        return ffi::Error(
            ffi::ErrorCode::kFailedPrecondition,
            "slate.batched_potrf: expected sub-row ctx with p=1");
    }
    const int64_t lld = n;
    const int64_t local_rows = n;
    const int64_t local_cols = (n + q - 1) / q;
    const int64_t slice_elems = local_rows * local_cols;

    // D2D copy of the entire batch, then factor each slice in place.
    ce = cudaMemcpyAsync(d_L_out, d_A_in,
                         nbatch_local * slice_elems * sizeof(T),
                         cudaMemcpyDeviceToDevice, xla_stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "slate.batched_potrf: cudaMemcpyAsync(A->L) failed: "
           << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    ce = cudaStreamSynchronize(xla_stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "slate.batched_potrf: cudaStreamSynchronize after A->L copy "
              "failed: " << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    sl::Options opts = {
        {sl::Option::Target, sl::Target::Devices},
    };

    for (int64_t b = 0; b < nbatch_local; ++b) {
        T* slice_ptr = d_L_out + b * slice_elems;
        T* L_ptrs[1] = { slice_ptr };
        const int num_devices = 1;

        sl::HermitianMatrix<T> L = sl::HermitianMatrix<T>::fromDevices(
            sl::Uplo::Lower, n, L_ptrs, num_devices, lld, nb, p, q, ctx->comm);

        try {
            int64_t info = sl::potrf(L, opts);
            if (info != 0) {
                std::ostringstream os;
                os << "slate::potrf (batch " << b << ") returned info=" << info
                   << " (matrix not positive-definite at that leading minor)";
                return ffi::Error(ffi::ErrorCode::kFailedPrecondition, os.str());
            }
        } catch (const std::exception& ex) {
            std::ostringstream os;
            os << "slate::potrf (batch " << b << ") threw: " << ex.what();
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
    }

    return ffi::Error::Success();
}

static ffi::Error BatchedPotrfDispatch(
    cudaStream_t stream,
    ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> L_out,
    int64_t nbatch_local, int64_t n, int64_t nb,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<lorrax_ffi::slate::SlateCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "slate.batched_potrf: ctx_handle is null");
    }

    const auto dtype = A.element_type();
    if (L_out->element_type() != dtype) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "slate.batched_potrf: L output dtype must match A input dtype");
    }

    switch (dtype) {
        case ffi::DataType::F64:
            return BatchedPotrfImpl<double>(
                nbatch_local, n, nb, stream, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<double*>(L_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return BatchedPotrfImpl<C128>(
                nbatch_local, n, nb, stream, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<C128*>(L_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "slate.batched_potrf: unsupported dtype "
               << static_cast<int>(dtype) << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::slate_batched_potrf

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SlateBatchedPotrfFfi, lorrax_ffi::slate_batched_potrf::BatchedPotrfDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()                      // A
        .Ret<xla::ffi::AnyBuffer>()                      // L
        .Attr<int64_t>("nbatch_local")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_handle"));
