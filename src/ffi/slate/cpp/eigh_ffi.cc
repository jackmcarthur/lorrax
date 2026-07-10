// eigh_ffi.cc — XLA FFI handler for slate::heev (distributed Hermitian
// eigendecomposition on GPU).
//
// Structure mirrors the cuSOLVERMp eigh FFI so both can coexist behind a
// common Python surface:
//
//   EighImpl<T>      — wraps local JAX shards in slate::HermitianMatrix /
//                      Matrix via fromDevices(), runs slate::heev.
//   EighDispatch     — shape/dtype validation + dtype switch.
//   XLA_FFI_DEFINE_HANDLER_SYMBOL(SlateEighFfi, EighDispatch, Bind()...)
//
// JAX sharding layout that this handler expects
//   A (input):   local shard LOCALLY TRANSPOSED by the Python wrapper
//                (shard_map + jnp.transpose), so the bytes are the
//                rank's (n/p, n/q) block in col-major layout — same
//                convention as potrf_ffi.cc / trsm_ffi.cc.  Combined
//                with the comm rank-remap (context.cc) SLATE sees the
//                true global A on any square p × q mesh.
//   Q (output):  SLATE col-major tiles; the Python wrapper reads them
//                back with out_specs P('y','x') + a local transpose to
//                recover Q at P('x','y') (cholesky handle convention).
//   W (output):  replicated.  SLATE's heev returns eigenvalues as a
//                std::vector<real_t> on each rank; we cudaMemcpy them
//                to the replicated W buffer.
//
// Input preservation: slate::heev overwrites its input with Householder
// reduction intermediates, but A is an XLA *input* buffer and must not
// be mutated (XLA may share it with other consumers).  We copy the
// local shard into the Q output buffer and let SLATE destroy that copy;
// the eigenvector tiles are then written over it after heev returns.

#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <sstream>
#include <type_traits>

#include <cuda_runtime.h>
#include <blas.hh>
#include <slate/slate.hh>

#include "xla/ffi/api/ffi.h"

#include "../../common/cpp/ffi_helpers.h"
#include "ctx.h"

namespace lorrax_ffi::slate_ffi {

namespace ffi = ::xla::ffi;
namespace sl  = ::slate;

// Map a dtype to its real-part type (for eigenvalues).
template <typename T> struct RealOf         { using type = T; };
template <> struct RealOf<std::complex<float>>  { using type = float; };
template <> struct RealOf<std::complex<double>> { using type = double; };

// ---------------------------------------------------------------------------
//  Impl — slate::heev for one scalar type.
// ---------------------------------------------------------------------------
template <typename T>
static ffi::Error EighImpl(
    int64_t n, int64_t nb,
    bool compute_evecs,
    cudaStream_t xla_stream,
    lorrax_ffi::slate::SlateCtx* ctx,
    T* d_A_local,
    typename RealOf<T>::type* d_W,
    T* d_Q_local)
{
    using real_t = typename RealOf<T>::type;

    // Make sure any prior XLA ops on d_A_local finished before SLATE
    // touches the memory.  SLATE manages its own streams internally, so we
    // just sync XLA's stream on this thread.
    cudaError_t ce = cudaStreamSynchronize(xla_stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "slate.eigh: cudaStreamSynchronize(xla_stream) failed: "
           << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    const int p = ctx->p;
    const int q = ctx->q;
    const int64_t lld_A = (n + p - 1) / p;   // local leading dim of A tile
    const int64_t lld_Q = lld_A;

    // SLATE's fromDevices takes an array of device pointers, one per
    // local device on this node.  SLATE's internal num_devices() is
    // queried via blas::get_device_count() at matrix-construction time
    // and must match the array length we hand in.  JAX's one-process-
    // per-GPU model requires each process to see exactly 1 GPU, so we
    // enforce that here (and set CUDA_VISIBLE_DEVICES on the launch side
    // via run_shifter.sh so SLATE agrees).
    const int num_devices_visible = blas::get_device_count();
    if (num_devices_visible != 1) {
        std::ostringstream os;
        os << "slate.eigh: blas::get_device_count()=" << num_devices_visible
           << " but JAX one-process-per-GPU model requires exactly 1. "
           << "Set CUDA_VISIBLE_DEVICES=$SLURM_LOCALID on the launcher "
           << "(run_shifter.sh does this when LORRAX_SLATE_STACK is set).";
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition, os.str());
    }
    const int num_devices = 1;

    // Copy the input shard into the Q output buffer: heev destroys its
    // input during the Householder reduction, and XLA input buffers
    // must never be mutated.  The Q buffer doubles as the scratch — the
    // eigenvector tiles overwrite it after heev returns.
    const int64_t local_rows = (n + p - 1) / p;
    const int64_t local_cols = (n + q - 1) / q;
    ce = cudaMemcpyAsync(d_Q_local, d_A_local,
                         local_rows * local_cols * sizeof(T),
                         cudaMemcpyDeviceToDevice, xla_stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "slate.eigh: cudaMemcpyAsync(A -> scratch) failed: "
           << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    ce = cudaStreamSynchronize(xla_stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "slate.eigh: cudaStreamSynchronize after A copy failed: "
           << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    T* A_ptrs[1] = { d_Q_local };

    // Build the distributed matrices over ctx->comm.  Lower triangular
    // storage — SLATE only supports Uplo::Lower for heev.
    sl::HermitianMatrix<T> A = sl::HermitianMatrix<T>::fromDevices(
        sl::Uplo::Lower, n, A_ptrs, num_devices, lld_A, nb, p, q, ctx->comm);

    std::vector<real_t> Lambda(n);

    sl::Options opts = {
        {sl::Option::Target, sl::Target::Devices},
    };

    try {
        if (compute_evecs) {
            // Let SLATE manage Z's storage internally, then copy each
            // local tile back to the JAX output buffer after heev.
            // fromDevices (pointing at JAX's d_Q_local) was empirically
            // observed to leave some ranks' buffers un-written — SLATE's
            // tile layout / origin handling inside heev doesn't always
            // hit the user-provided pointer.  Copying explicitly post-
            // heev is the robust path.
            sl::Matrix<T> Z(n, n, nb, p, q, ctx->comm);
            Z.insertLocalTiles(sl::Target::Devices);
            sl::heev(A, Lambda, Z, opts);

            // Copy local tile data back to d_Q_local (col-major within
            // the tile; the Python wrapper local-transposes back).  At
            // most one local tile per rank with our nb=n/p, though we
            // loop to stay correct for smaller nb.
            //
            // tileGetForReading is REQUIRED before touching tile data:
            // heev's back-transformation (unmtr_hb2st / unmtr_he2hb via
            // redistribute) can leave the valid copy of a tile on the
            // HOST, with the device instance stale.  Reading Z(ti, tj,
            // dev) directly returned the pre-back-transform bytes —
            // the "eigvec layout artifact" was stale MOSI state, not a
            // layout transform.
            cudaStream_t stream = xla_stream;
            for (int64_t ti = 0; ti < Z.mt(); ++ti) {
                for (int64_t tj = 0; tj < Z.nt(); ++tj) {
                    if (!Z.tileIsLocal(ti, tj)) continue;
                    const int dev = Z.tileDevice(ti, tj);
                    Z.tileGetForReading(ti, tj, dev,
                                        sl::LayoutConvert::ColMajor);
                    auto tile = Z(ti, tj, dev);
                    const T* src = tile.data();
                    const int64_t mb = tile.mb();
                    const int64_t nb_t = tile.nb();
                    const int64_t stride = tile.stride();
                    // Destination offset within d_Q_local: same mapping
                    // BaseTrapezoidMatrix::fromDevices would have used.
                    const int64_t ii = ti * nb, jj = tj * nb;
                    const int64_t ii_local = (ii / (nb * p)) * nb + (ii % nb);
                    const int64_t jj_local = (jj / (nb * q)) * nb + (jj % nb);
                    T* dst = d_Q_local + ii_local + jj_local * lld_Q;
                    cudaError_t ce = cudaMemcpy2DAsync(
                        dst, lld_Q * sizeof(T),
                        src, stride * sizeof(T),
                        mb * sizeof(T), nb_t,
                        cudaMemcpyDeviceToDevice, stream);
                    if (ce != cudaSuccess) {
                        std::ostringstream os;
                        os << "slate.eigh: cudaMemcpy2DAsync(Z tile (" << ti
                           << "," << tj << ") -> d_Q_local) failed: "
                           << cudaGetErrorString(ce);
                        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
                    }
                }
            }
            // The 2-D memcpys read SLATE-owned tile memory that Z's
            // destructor frees on scope exit — sync before Z dies.
            ce = cudaStreamSynchronize(stream);
            if (ce != cudaSuccess) {
                std::ostringstream os;
                os << "slate.eigh: cudaStreamSynchronize after Z copy "
                   << "failed: " << cudaGetErrorString(ce);
                return ffi::Error(ffi::ErrorCode::kInternal, os.str());
            }
        } else {
            sl::heev(A, Lambda, opts);
        }
    } catch (const std::exception& ex) {
        std::ostringstream os;
        os << "slate::heev threw: " << ex.what();
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    // Copy eigenvalues (host std::vector) → replicated device W buffer.
    ce = cudaMemcpyAsync(d_W, Lambda.data(), n * sizeof(real_t),
                         cudaMemcpyHostToDevice, xla_stream);
    if (ce != cudaSuccess) {
        std::ostringstream os;
        os << "slate.eigh: cudaMemcpyAsync(Lambda -> d_W) failed: "
           << cudaGetErrorString(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
//  Dispatch — validates shapes, looks up Ctx, switches on dtype.
// ---------------------------------------------------------------------------
static ffi::Error EighDispatch(
    cudaStream_t stream,
    ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> W_out,
    ffi::Result<ffi::AnyBuffer> Q_out,
    int64_t n, int64_t nb,
    int64_t ctx_handle, bool compute_evecs)
{
    auto* ctx = reinterpret_cast<lorrax_ffi::slate::SlateCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "slate.eigh: ctx_handle is null "
                          "(context not initialized on this process?)");
    }

    const auto dtype = A.element_type();
    if (Q_out->element_type() != dtype) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "slate.eigh: Q output and A input must have matching dtype");
    }
    const auto wdtype = W_out->element_type();
    const bool is_cplx = (dtype == ffi::DataType::C64 ||
                          dtype == ffi::DataType::C128);
    const auto expected_wdtype = is_cplx
        ? (dtype == ffi::DataType::C64 ? ffi::DataType::F32 : ffi::DataType::F64)
        : dtype;
    if (wdtype != expected_wdtype) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "slate.eigh: W output dtype must be the real part of A's dtype");
    }

    switch (dtype) {
        case ffi::DataType::F64:
            return EighImpl<double>(
                n, nb, compute_evecs, stream, ctx,
                static_cast<double*>(const_cast<void*>(A.untyped_data())),
                static_cast<double*>(W_out->untyped_data()),
                static_cast<double*>(Q_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return EighImpl<C128>(
                n, nb, compute_evecs, stream, ctx,
                static_cast<C128*>(const_cast<void*>(A.untyped_data())),
                static_cast<double*>(W_out->untyped_data()),
                static_cast<C128*>(Q_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "slate.eigh: unsupported dtype " << static_cast<int>(dtype)
               << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::slate_ffi

// ---------------------------------------------------------------------------
//  FFI binding.
// ---------------------------------------------------------------------------
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SlateEighFfi, lorrax_ffi::slate_ffi::EighDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()              // A (local shard)
        .Ret<xla::ffi::AnyBuffer>()              // W (n,) real, replicated
        .Ret<xla::ffi::AnyBuffer>()              // Q (local shard, same dtype as A)
        .Attr<int64_t>("n")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_handle")
        .Attr<bool>("compute_evecs"));
