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
//
// Device memory: SLATE's potrf calls reserveDeviceWorkspace(), one pool block per
// local (lower) tile, which also receives the remote tiles each step needs.  Its
// pool is seeded from the u8 WORKSPACE output (pool_seed.h;
// distrib_la._slate.potrf_workspace_bytes sizes it), so at one tile per rank the
// n/p x n/p block SLATE needs lives in XLA's arena, not in a device_malloc beside
// it (MI250X P4: L outside the arena per rank, 5.2 GB at n=30720 c128).

#include "../common/ctx_registry.h"
#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <sstream>

#include "../common/gpu_runtime.h"
#include <blas.hh>
#include <slate/slate.hh>

#include "xla/ffi/api/ffi.h"

#include "../common/ffi_helpers.h"
#include "ctx.h"
#include "pool_seed.h"

namespace lorrax_ffi::slate_potrf {

namespace ffi = ::xla::ffi;
namespace sl  = ::slate;

template <typename T>
static ffi::Error PotrfImpl(
    int64_t n, int64_t nb,
    lrx::gpu::Stream xla_stream,
    lorrax_ffi::slate::SlateCtx* ctx,
    const T* d_A_in,
    T* d_L_out,
    void* d_ws, std::size_t ws_bytes)
{
    lrx::gpu::Error ce = lrx::gpu::stream_synchronize(xla_stream);
    if (ce != lrx::gpu::kSuccess) {
        std::ostringstream os;
        os << "slate.potrf: " LRX_GPU_API "StreamSynchronize(xla_stream) failed: "
           << lrx::gpu::error_string(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    const int num_devices_visible = blas::get_device_count();
    if (num_devices_visible != 1) {
        std::ostringstream os;
        os << "slate.potrf: blas::get_device_count()=" << num_devices_visible
           << " but JAX one-process-per-GPU model requires exactly 1. "
           << "Set " LRX_GPU_VENDOR "_VISIBLE_DEVICES=$SLURM_LOCALID (LORRAX_SELECT_GPU=1).";
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition, os.str());
    }

    const int p = ctx->p;
    const int q = ctx->q;
    const int64_t lld = (n + p - 1) / p;

    // Copy the input shard into the output buffer (unless XLA aliased them), then
    // factor in place.
    const int64_t local_rows = (n + p - 1) / p;
    const int64_t local_cols = (n + q - 1) / q;
    if (static_cast<const void*>(d_L_out) != static_cast<const void*>(d_A_in))
    ce = lrx::gpu::memcpy_async(d_L_out, d_A_in,
                         local_rows * local_cols * sizeof(T),
                         lrx::gpu::kDeviceToDevice, xla_stream);
    if (ce != lrx::gpu::kSuccess) {
        std::ostringstream os;
        os << "slate.potrf: " LRX_GPU_API "MemcpyAsync(A -> L) failed: "
           << lrx::gpu::error_string(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    // Let the memcpy finish before SLATE touches the tiles.
    ce = lrx::gpu::stream_synchronize(xla_stream);
    if (ce != lrx::gpu::kSuccess) {
        std::ostringstream os;
        os << "slate.potrf: " LRX_GPU_API "StreamSynchronize after A->L copy failed: "
           << lrx::gpu::error_string(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    T* L_ptrs[1] = { d_L_out };
    const int num_devices = 1;

    sl::HermitianMatrix<T> L = sl::HermitianMatrix<T>::fromDevices(
        sl::Uplo::Lower, n, L_ptrs, num_devices, lld, nb, p, q, ctx->comm);

    sl::Options opts = {
        {sl::Option::Target, sl::Target::Devices},
    };

    // Seed the pool with SLATE's own reservation, one block per local lower tile
    // (reserveDeviceWorkspace; the measured high-water never exceeds it, PoolWatch).
    // distrib_la._slate.potrf_workspace_bytes sizes the buffer to it only when it is
    // larger than the outside allowance main budgets for it, and to nothing otherwise
    // (the carve then seeds nothing and SLATE allocates exactly what main does).
    int64_t local_lower = 0;
    for (int64_t j = 0; j < L.nt(); ++j)
        for (int64_t i = j; i < L.mt(); ++i)
            if (L.tileIsLocal(i, j)) ++local_lower;
    const int64_t want = local_lower;
    lorrax_ffi::slate::PoolSeed ws(d_ws, ws_bytes);
    const int64_t seeded = ws.seed(L, want);
    lorrax_ffi::slate::PoolWatch pw(lorrax_ffi::slate::pool_report_enabled());
    lorrax_ffi::slate::PoolWatchScope wl(pw, L, "L", seeded, local_lower);

    try {
        int64_t info = sl::potrf(L, opts);
        pw.sample();
        if (lorrax_ffi::slate::pool_report_enabled()) {
            int r = 0;
            MPI_Comm_rank(ctx->comm, &r);
            pw.report("potrf", r, n, nb, (int) sizeof(T));
            std::fprintf(stderr, "[slate.potrf] rank=%d n=%lld nb=%lld pool seeded=%lld "
                         "fallback=%lld blocks ws=%.3fGB\n", r, (long long) n, (long long) nb,
                         (long long) seeded,
                         (long long) lorrax_ffi::slate::pool_fallback(L, seeded), ws.size / 1e9);
        }
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

    // No explicit strict-upper zeroing: SLATE's potrf(Hermitian Lower)
    // only reads lower-tri tiles, and SLATE's trsm(Lower) downstream
    // also only reads lower-tri.  Leftover A-data in unused tiles is
    // ignored.  to_jax_lower() applies tril() if the user wants a
    // clean L for use outside SLATE.

    return ffi::Error::Success();
}

static ffi::Error PotrfDispatch(
    lrx::gpu::Stream stream,
    ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> L_out,
    ffi::Result<ffi::AnyBuffer> WS,
    int64_t n, int64_t nb,
    int64_t ctx_key)
{
    const int64_t ctx_handle = ::lorrax_ffi::ctx_registry::resolve(ctx_key, "slate");
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
                static_cast<double*>(L_out->untyped_data()),
                WS->untyped_data(), WS->size_bytes());
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return PotrfImpl<C128>(
                n, nb, stream, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<C128*>(L_out->untyped_data()),
                WS->untyped_data(), WS->size_bytes());
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
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()              // A (Hermitian input)
        .Ret<xla::ffi::AnyBuffer>()              // L (Cholesky factor, lower)
        .Ret<xla::ffi::AnyBuffer>()              // workspace (u8): SLATE's pool seed
        .Attr<int64_t>("n")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_key"));
