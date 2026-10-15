// batched_solve_lu_ffi.cc — XLA FFI handlers for the distributed LU family on
// SLATE: slate::getrf (MethodLU::PartialPiv, Target::Devices) + slate::getrs,
// per matrix of a (Nq, N, N) stack, each matrix spread over the WHOLE mesh.
//
// The SLATE twin of cusolvermp/batched_solve_lu_ffi.cc and of
// scalapack/{solve_lu,getrf_getrs}_ffi.cc, with the same three entries and the
// same operand contract, so distrib_la's facade drives all three backends the
// same way:
//
//   SlateBatchedSolveLuFactorsFfi   (A, B)        -> (X aliased B, LU aliased A)
//   SlateBatchedGetrfFfi            (A)           -> (LU aliased A, piv)
//   SlateBatchedGetrsFfi            (LU, piv, B)  -> (X aliased B)
//
// The fused entry runs exactly the getrf + getrs pair the split entries run
// (one core, below), so getrf-once + getrs-per-chunk reproduces the fused
// solve to the bit on the same mesh.
//
// Layout contract (same as potrf_ffi.cc / eigh_ffi.cc; see ctx.h):
//   - The world SlateCtx (p, q) with its rank remap: JAX shard (mx, my) is
//     SLATE tile (mx, my).
//   - Each rank's buffer is its (n/p, n/q) shard transposed by the Python
//     wrapper, i.e. the true block in col-major with lld = n/p.
//   - A tiles: nb = n / max(p, q) — one tile per rank on a square mesh, all
//     local column tiles on a p x 1 mesh.  B / X: row tile nb (B.mt() must
//     equal A.mt() for getrs), column tile nb_b = nrhs / q.
//
// Pivots.  slate::Pivots is vector<vector<Pivot>>, one vector per block
// column k with min(tileMb(k), tileNb(k)) entries; getrf MPI_Bcasts every
// panel's pivots to all ranks, so each rank holds the full, replicated set.
// The split entries serialise it to an int64 buffer of 2*n per matrix —
// (tileIndex, elementOffset) pairs in panel order — each rank its own copy.
// It is opaque to Python and only ever fed back to getrs on the same mesh.
//
// SLATE runs on its own device queues and reads/writes the origin tiles
// directly, so each handler synchronises the XLA stream before handing SLATE
// the buffers (the potrf/trsm handlers' rule).

#include "../common/ctx_registry.h"
#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "../common/gpu_runtime.h"
#include <blas.hh>
#include <slate/slate.hh>

#include "xla/ffi/api/ffi.h"

#include "../common/ffi_helpers.h"
#include "ctx.h"

namespace lorrax_ffi::slate_lu {

namespace ffi = ::xla::ffi;
namespace sl  = ::slate;

static ffi::Error gpu_fail(const char* what, const char* step, lrx::gpu::Error ce) {
    std::ostringstream os;
    os << "slate." << what << ": " << step << " failed: " << lrx::gpu::error_string(ce);
    return ffi::Error(ffi::ErrorCode::kInternal, os.str());
}

static ffi::Error preflight(const char* what, lrx::gpu::Stream stream,
                            const lorrax_ffi::slate::SlateCtx* ctx,
                            int64_t n, int64_t nb) {
    lrx::gpu::Error ce = lrx::gpu::stream_synchronize(stream);
    if (ce != lrx::gpu::kSuccess)
        return gpu_fail(what, LRX_GPU_API "StreamSynchronize(xla_stream)", ce);
    const int ndev = blas::get_device_count();
    if (ndev != 1) {
        std::ostringstream os;
        os << "slate." << what << ": blas::get_device_count()=" << ndev
           << " but JAX one-process-per-GPU model requires exactly 1. "
           << "Set " LRX_GPU_VENDOR "_VISIBLE_DEVICES=$SLURM_LOCALID (LORRAX_SELECT_GPU=1).";
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition, os.str());
    }
    const int p = ctx->p, q = ctx->q;
    const int64_t g = (p > q) ? p : q;
    if (n % p != 0 || n % q != 0 || nb != n / g) {
        std::ostringstream os;
        os << "slate." << what << ": layout n=" << n << ", nb=" << nb << " on a "
           << p << "x" << q << " ctx is not one tile per rank (want nb = n/max(p,q))";
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    return ffi::Error::Success();
}

// Copy src -> dst (no-op when XLA aliased them), then let it land.
static ffi::Error stage(const char* what, void* dst, const void* src, std::size_t bytes,
                        lrx::gpu::Stream stream) {
    if (dst == src || bytes == 0) return ffi::Error::Success();
    lrx::gpu::Error ce = lrx::gpu::memcpy_async(dst, src, bytes, lrx::gpu::kDeviceToDevice, stream);
    if (ce != lrx::gpu::kSuccess) return gpu_fail(what, LRX_GPU_API "MemcpyAsync(D2D)", ce);
    ce = lrx::gpu::stream_synchronize(stream);
    if (ce != lrx::gpu::kSuccess) return gpu_fail(what, LRX_GPU_API "StreamSynchronize(stage)", ce);
    return ffi::Error::Success();
}

template <typename T>
static sl::Matrix<T> wrap(T* ptr, int64_t m, int64_t ncols, int64_t mb, int64_t nb_cols,
                          const lorrax_ffi::slate::SlateCtx* ctx) {
    T* ptrs[1] = { ptr };
    const int64_t lld = m / ctx->p;
    return sl::Matrix<T>::fromDevices(m, ncols, ptrs, 1, lld, mb, nb_cols,
                                      ctx->p, ctx->q, ctx->comm);
}

static const sl::Options& lu_opts() {
    static const sl::Options opts = {
        {sl::Option::Target,   sl::Target::Devices},
        {sl::Option::MethodLU, sl::MethodLU::PartialPiv},
    };
    return opts;
}

template <typename T>
static ffi::Error factor_one(const char* what, sl::Matrix<T>& A, sl::Pivots& piv, int64_t iq) {
    int64_t info = 0;
    try {
        info = sl::getrf(A, piv, lu_opts());
    } catch (const std::exception& ex) {
        std::ostringstream os;
        os << "slate::getrf (q=" << iq << ") threw: " << ex.what();
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    if (info != 0) {
        std::ostringstream os;
        os << "slate." << what << ": slate::getrf (q=" << iq << ") info=" << info
           << " (U(" << info << "," << info << ") is exactly zero: the matrix is singular)";
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition, os.str());
    }
    return ffi::Error::Success();
}

template <typename T>
static ffi::Error solve_one(sl::Matrix<T>& LU, sl::Pivots& piv, sl::Matrix<T>& X, int64_t iq) {
    try {
        sl::getrs(LU, piv, X, lu_opts());
    } catch (const std::exception& ex) {
        std::ostringstream os;
        os << "slate::getrs (q=" << iq << ") threw: " << ex.what();
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    return ffi::Error::Success();
}

// The pivot vector sizes getrf produces for this A geometry (min(mb, nb) per
// block column); the serialised form is their concatenation.
template <typename T>
static std::vector<int64_t> panel_lengths(const sl::Matrix<T>& A) {
    std::vector<int64_t> len;
    const int64_t kt = std::min(A.mt(), A.nt());
    for (int64_t k = 0; k < kt; ++k)
        len.push_back(std::min(A.tileMb(k), A.tileNb(k)));
    return len;
}

// ---- fused: getrf + getrs, pivots internal --------------------------------
template <typename T>
static ffi::Error SolveLuFactorsImpl(lrx::gpu::Stream stream, lorrax_ffi::slate::SlateCtx* ctx,
                                     int64_t nq, int64_t n, int64_t nrhs, int64_t nb, int64_t nb_b,
                                     const T* A, const T* B, T* X, T* LU) {
    const char* what = "batched_solve_lu";
    if (auto e = preflight(what, stream, ctx, n, nb); !e.success()) return e;
    const int64_t a_elems = (n / ctx->p) * (n / ctx->q);
    const int64_t b_elems = (n / ctx->p) * (nrhs / ctx->q);
    if (auto e = stage(what, LU, A, nq * a_elems * sizeof(T), stream); !e.success()) return e;
    if (auto e = stage(what, X, B, nq * b_elems * sizeof(T), stream); !e.success()) return e;
    for (int64_t iq = 0; iq < nq; ++iq) {
        auto Aq = wrap<T>(LU + iq * a_elems, n, n, nb, nb, ctx);
        auto Xq = wrap<T>(X + iq * b_elems, n, nrhs, nb, nb_b, ctx);
        sl::Pivots piv;
        if (auto e = factor_one<T>(what, Aq, piv, iq); !e.success()) return e;
        if (auto e = solve_one<T>(Aq, piv, Xq, iq); !e.success()) return e;
    }
    return ffi::Error::Success();
}

// ---- split: getrf, pivots serialised ---------------------------------------
template <typename T>
static ffi::Error GetrfImpl(lrx::gpu::Stream stream, lorrax_ffi::slate::SlateCtx* ctx,
                            int64_t nq, int64_t n, int64_t nb, int64_t piv_len,
                            const T* A, T* LU, int64_t* d_piv) {
    const char* what = "batched_getrf";
    if (auto e = preflight(what, stream, ctx, n, nb); !e.success()) return e;
    if (piv_len != 2 * n) {
        std::ostringstream os;
        os << "slate." << what << ": piv_len=" << piv_len << " != 2*n=" << 2 * n;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    const int64_t a_elems = (n / ctx->p) * (n / ctx->q);
    if (auto e = stage(what, LU, A, nq * a_elems * sizeof(T), stream); !e.success()) return e;
    std::vector<int64_t> h_piv(static_cast<std::size_t>(nq * piv_len), 0);
    for (int64_t iq = 0; iq < nq; ++iq) {
        auto Aq = wrap<T>(LU + iq * a_elems, n, n, nb, nb, ctx);
        sl::Pivots piv;
        if (auto e = factor_one<T>(what, Aq, piv, iq); !e.success()) return e;
        int64_t* out = h_piv.data() + iq * piv_len;
        int64_t at = 0;
        for (const auto& panel : piv)
            for (const auto& pv : panel) {
                if (at + 2 > piv_len)
                    return ffi::Error(ffi::ErrorCode::kInternal,
                                      "slate.batched_getrf: getrf returned more than n pivots");
                out[at++] = pv.tileIndex();
                out[at++] = pv.elementOffset();
            }
        if (at != piv_len)
            return ffi::Error(ffi::ErrorCode::kInternal,
                              "slate.batched_getrf: getrf returned fewer than n pivots");
    }
    lrx::gpu::Error ce = lrx::gpu::memcpy_async(d_piv, h_piv.data(), h_piv.size() * sizeof(int64_t),
                                                lrx::gpu::kHostToDevice, stream);
    if (ce != lrx::gpu::kSuccess) return gpu_fail(what, LRX_GPU_API "MemcpyAsync(piv H2D)", ce);
    ce = lrx::gpu::stream_synchronize(stream);    // h_piv dies at return
    if (ce != lrx::gpu::kSuccess) return gpu_fail(what, LRX_GPU_API "StreamSynchronize(piv)", ce);
    return ffi::Error::Success();
}

// ---- split: getrs from the serialised pivots ---------------------------------
template <typename T>
static ffi::Error GetrsImpl(lrx::gpu::Stream stream, lorrax_ffi::slate::SlateCtx* ctx,
                            int64_t nq, int64_t n, int64_t nrhs, int64_t nb, int64_t nb_b,
                            int64_t piv_len, const T* LU, const int64_t* d_piv,
                            const T* B, T* X) {
    const char* what = "batched_getrs";
    if (auto e = preflight(what, stream, ctx, n, nb); !e.success()) return e;
    if (piv_len != 2 * n) {
        std::ostringstream os;
        os << "slate." << what << ": piv_len=" << piv_len << " != 2*n=" << 2 * n;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    const int64_t a_elems = (n / ctx->p) * (n / ctx->q);
    const int64_t b_elems = (n / ctx->p) * (nrhs / ctx->q);
    if (auto e = stage(what, X, B, nq * b_elems * sizeof(T), stream); !e.success()) return e;
    std::vector<int64_t> h_piv(static_cast<std::size_t>(nq * piv_len), 0);
    lrx::gpu::Error ce = lrx::gpu::memcpy_async(h_piv.data(), d_piv, h_piv.size() * sizeof(int64_t),
                                                lrx::gpu::kDeviceToHost, stream);
    if (ce != lrx::gpu::kSuccess) return gpu_fail(what, LRX_GPU_API "MemcpyAsync(piv D2H)", ce);
    ce = lrx::gpu::stream_synchronize(stream);
    if (ce != lrx::gpu::kSuccess) return gpu_fail(what, LRX_GPU_API "StreamSynchronize(piv)", ce);
    for (int64_t iq = 0; iq < nq; ++iq) {
        // getrs only reads the factors (triangular views + row swaps on B);
        // the const_cast is the batched_trsm handler's, for the same reason.
        auto Aq = wrap<T>(const_cast<T*>(LU) + iq * a_elems, n, n, nb, nb, ctx);
        auto Xq = wrap<T>(X + iq * b_elems, n, nrhs, nb, nb_b, ctx);
        const int64_t* in = h_piv.data() + iq * piv_len;
        sl::Pivots piv;
        int64_t at = 0;
        for (int64_t len : panel_lengths(Aq)) {
            std::vector<sl::Pivot> panel;
            panel.reserve(static_cast<std::size_t>(len));
            for (int64_t i = 0; i < len; ++i, at += 2)
                panel.emplace_back(in[at], in[at + 1]);
            piv.push_back(std::move(panel));
        }
        if (at != piv_len)
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "slate.batched_getrs: pivot buffer does not match this A geometry "
                              "(feed back getrf's piv verbatim, same mesh)");
        if (auto e = solve_one<T>(Aq, piv, Xq, iq); !e.success()) return e;
    }
    return ffi::Error::Success();
}

static lorrax_ffi::slate::SlateCtx* ctx_of(int64_t ctx_key) {
    const int64_t h = ::lorrax_ffi::ctx_registry::resolve(ctx_key, "slate");
    return reinterpret_cast<lorrax_ffi::slate::SlateCtx*>(h);
}

static ffi::Error bad_dtype(const char* what, ffi::DataType dt) {
    std::ostringstream os;
    os << "slate." << what << ": unsupported dtype " << static_cast<int>(dt)
       << " (supported: F64, C128)";
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
}

static ffi::Error SolveLuFactorsDispatch(
    lrx::gpu::Stream stream, ffi::AnyBuffer A, ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> X, ffi::Result<ffi::AnyBuffer> LU,
    int64_t nq, int64_t n, int64_t nrhs, int64_t nb, int64_t nb_b, int64_t ctx_key)
{
    auto* ctx = ctx_of(ctx_key);
    if (ctx == nullptr)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "slate.batched_solve_lu: ctx is null");
    const auto dt = A.element_type();
    if (B.element_type() != dt || X->element_type() != dt || LU->element_type() != dt)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "slate.batched_solve_lu: A, B, X, LU must share one dtype");
    using C128 = std::complex<double>;
    switch (dt) {
        case ffi::DataType::F64:
            return SolveLuFactorsImpl<double>(stream, ctx, nq, n, nrhs, nb, nb_b,
                static_cast<const double*>(A.untyped_data()), static_cast<const double*>(B.untyped_data()),
                static_cast<double*>(X->untyped_data()), static_cast<double*>(LU->untyped_data()));
        case ffi::DataType::C128:
            return SolveLuFactorsImpl<C128>(stream, ctx, nq, n, nrhs, nb, nb_b,
                static_cast<const C128*>(A.untyped_data()), static_cast<const C128*>(B.untyped_data()),
                static_cast<C128*>(X->untyped_data()), static_cast<C128*>(LU->untyped_data()));
        default: return bad_dtype("batched_solve_lu", dt);
    }
}

static ffi::Error GetrfDispatch(
    lrx::gpu::Stream stream, ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> LU, ffi::Result<ffi::AnyBuffer> piv,
    int64_t nq, int64_t n, int64_t nb, int64_t piv_len, int64_t ctx_key)
{
    auto* ctx = ctx_of(ctx_key);
    if (ctx == nullptr)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "slate.batched_getrf: ctx is null");
    const auto dt = A.element_type();
    if (LU->element_type() != dt)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "slate.batched_getrf: LU dtype != A dtype");
    if (piv->element_type() != ffi::DataType::S64)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "slate.batched_getrf: piv must be int64");
    auto* d_piv = static_cast<int64_t*>(piv->untyped_data());
    using C128 = std::complex<double>;
    switch (dt) {
        case ffi::DataType::F64:
            return GetrfImpl<double>(stream, ctx, nq, n, nb, piv_len,
                static_cast<const double*>(A.untyped_data()), static_cast<double*>(LU->untyped_data()), d_piv);
        case ffi::DataType::C128:
            return GetrfImpl<C128>(stream, ctx, nq, n, nb, piv_len,
                static_cast<const C128*>(A.untyped_data()), static_cast<C128*>(LU->untyped_data()), d_piv);
        default: return bad_dtype("batched_getrf", dt);
    }
}

static ffi::Error GetrsDispatch(
    lrx::gpu::Stream stream, ffi::AnyBuffer LU, ffi::AnyBuffer piv, ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> X,
    int64_t nq, int64_t n, int64_t nrhs, int64_t nb, int64_t nb_b, int64_t piv_len,
    int64_t ctx_key)
{
    auto* ctx = ctx_of(ctx_key);
    if (ctx == nullptr)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "slate.batched_getrs: ctx is null");
    const auto dt = LU.element_type();
    if (B.element_type() != dt || X->element_type() != dt)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "slate.batched_getrs: LU, B, X must share one dtype");
    if (piv.element_type() != ffi::DataType::S64)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "slate.batched_getrs: piv must be int64");
    const auto* d_piv = static_cast<const int64_t*>(piv.untyped_data());
    using C128 = std::complex<double>;
    switch (dt) {
        case ffi::DataType::F64:
            return GetrsImpl<double>(stream, ctx, nq, n, nrhs, nb, nb_b, piv_len,
                static_cast<const double*>(LU.untyped_data()), d_piv,
                static_cast<const double*>(B.untyped_data()), static_cast<double*>(X->untyped_data()));
        case ffi::DataType::C128:
            return GetrsImpl<C128>(stream, ctx, nq, n, nrhs, nb, nb_b, piv_len,
                static_cast<const C128*>(LU.untyped_data()), d_piv,
                static_cast<const C128*>(B.untyped_data()), static_cast<C128*>(X->untyped_data()));
        default: return bad_dtype("batched_getrs", dt);
    }
}

}  // namespace lorrax_ffi::slate_lu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SlateBatchedSolveLuFactorsFfi, lorrax_ffi::slate_lu::SolveLuFactorsDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()              // A
        .Arg<xla::ffi::AnyBuffer>()              // B
        .Ret<xla::ffi::AnyBuffer>()              // X (aliased to B)
        .Ret<xla::ffi::AnyBuffer>()              // LU factors (aliased to A)
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("nb")                     // A tile: n / max(p, q)
        .Attr<int64_t>("nb_b")                   // B column tile: nrhs / q
        .Attr<int64_t>("ctx_key"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SlateBatchedGetrfFfi, lorrax_ffi::slate_lu::GetrfDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()              // A (donated; factored in place)
        .Ret<xla::ffi::AnyBuffer>()              // LU (aliased to A)
        .Ret<xla::ffi::AnyBuffer>()              // piv (int64, 2n per matrix, replicated)
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("piv_len")
        .Attr<int64_t>("ctx_key"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SlateBatchedGetrsFfi, lorrax_ffi::slate_lu::GetrsDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()              // LU (as getrf returned it)
        .Arg<xla::ffi::AnyBuffer>()              // piv (as getrf returned it)
        .Arg<xla::ffi::AnyBuffer>()              // B (donated; solved in place)
        .Ret<xla::ffi::AnyBuffer>()              // X (aliased to B)
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("nb_b")
        .Attr<int64_t>("piv_len")
        .Attr<int64_t>("ctx_key"));
