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
// Layout contract (same as eigh_ffi.cc; see ctx.h):
//   - The world SlateCtx (p, q) with its rank remap: JAX shard (mx, my) is
//     SLATE process (mx, my).
//   - Each rank's buffer is its (n/p, n/q) shard transposed by the Python
//     wrapper, i.e. the true block in col-major with lld = n/p.
//   - Block-cyclic BY RELABELING (distrib_la/_cyclic.py): A tiles are nb x nb
//     with nb dividing n/p and n/q.  The rank's local buffer is read at the
//     offsets fromDevices would use, so SLATE factors Pi A Pi'^T (Pi sends
//     local row block c of x-rank r to global block c*p + r; Pi' the same over
//     q).  B / X: row tile nb (B.mt() == A.mt()), column tile nb_b = nrhs / q,
//     read through the same Pi.  On a square mesh Pi' = Pi and X comes back in
//     A's row order with no movement; on a p x 1 mesh X's rows come back
//     block-cyclic and the Python wrapper puts them in order (one all_to_all).
//   - The operands are COPIED into SLATE-owned device tiles (stride = tile
//     edge) and the results copied back: a wrapped buffer at nb < n/p has tile
//     stride lld = n/p beside SLATE's own nb-stride workspace tiles, the
//     non-uniform-stride class of SLATE's device batching (internal_batch.hh:290),
//     and the owned tiles are uniform by construction (eigh_ffi.cc's rule).
//     At nb = n/max(p, q) this is the former one-tile-per-rank layout.
//   - Device memory: every handler takes one more output, a u8 WORKSPACE the
//     Python wrapper sizes (distrib_la._slate.lu_workspace_bytes) and never
//     reads.  A's and X's SLATE pools are seeded from it (pool_seed.h), so the
//     owned tiles and getrf/getrs's reserveDeviceWorkspace() are XLA buffers,
//     not device_malloc calls outside the arena.
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
#include "pool_seed.h"

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
    if (n % p != 0 || n % q != 0 || nb < 1 || (n / p) % nb != 0 || (n / q) % nb != 0) {
        std::ostringstream os;
        os << "slate." << what << ": layout n=" << n << ", nb=" << nb << " on a "
           << p << "x" << q << " ctx: nb must divide both local edges n/p and n/q";
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    return ffi::Error::Success();
}

// The rank's col-major local block (leading dim lld) <-> M's local tiles, at
// the offsets fromDevices would use: tile (ti, tj) of x-rank ti % p, y-rank
// tj % q sits at local row (ti / p) * mb, local column (tj / q) * nbc.
template <typename T>
static ffi::Error copy_in(const char* what, sl::Matrix<T>& M, const T* src, int64_t lld,
                          int p, int q, lrx::gpu::Stream stream) {
    const int64_t mb = M.tileMb(0), nbc = M.tileNb(0);
    for (int64_t tj = 0; tj < M.nt(); ++tj)
        for (int64_t ti = 0; ti < M.mt(); ++ti) {
            if (!M.tileIsLocal(ti, tj)) continue;
            auto t = M(ti, tj, M.tileDevice(ti, tj));
            const T* from = src + (ti / p) * mb + (tj / q) * nbc * lld;
            lrx::gpu::Error ce = lrx::gpu::memcpy_2d_async(
                t.data(), t.stride() * sizeof(T), from, lld * sizeof(T),
                t.mb() * sizeof(T), t.nb(), lrx::gpu::kDeviceToDevice, stream);
            if (ce != lrx::gpu::kSuccess)
                return gpu_fail(what, LRX_GPU_API "Memcpy2DAsync(local block -> tile)", ce);
        }
    lrx::gpu::Error ce = lrx::gpu::stream_synchronize(stream);
    if (ce != lrx::gpu::kSuccess) return gpu_fail(what, LRX_GPU_API "StreamSynchronize(copy in)", ce);
    return ffi::Error::Success();
}

// tileGetForReading first: a routine can leave a tile's valid copy on the host
// (eigh_ffi.cc's stale-MOSI finding).  Synchronised before M frees its tiles.
template <typename T>
static ffi::Error copy_out(const char* what, sl::Matrix<T>& M, T* dst, int64_t lld,
                           int p, int q, lrx::gpu::Stream stream) {
    const int64_t mb = M.tileMb(0), nbc = M.tileNb(0);
    for (int64_t tj = 0; tj < M.nt(); ++tj)
        for (int64_t ti = 0; ti < M.mt(); ++ti) {
            if (!M.tileIsLocal(ti, tj)) continue;
            const int dev = M.tileDevice(ti, tj);
            M.tileGetForReading(ti, tj, dev, sl::LayoutConvert::ColMajor);
            auto t = M(ti, tj, dev);
            T* to = dst + (ti / p) * mb + (tj / q) * nbc * lld;
            lrx::gpu::Error ce = lrx::gpu::memcpy_2d_async(
                to, lld * sizeof(T), t.data(), t.stride() * sizeof(T),
                t.mb() * sizeof(T), t.nb(), lrx::gpu::kDeviceToDevice, stream);
            if (ce != lrx::gpu::kSuccess)
                return gpu_fail(what, LRX_GPU_API "Memcpy2DAsync(tile -> local block)", ce);
        }
    lrx::gpu::Error ce = lrx::gpu::stream_synchronize(stream);
    if (ce != lrx::gpu::kSuccess) return gpu_fail(what, LRX_GPU_API "StreamSynchronize(copy out)", ce);
    return ffi::Error::Success();
}

// SLATE-owned device tiles: A (n x n, nb x nb) or B / X (n x nrhs, nb x nb_b),
// their pool seeded from the workspace with one block per local tile plus the
// remote tiles the driver holds at once, at the measured lower envelope of the
// pool's high-water (pool_seed.h PoolWatch, MI250X 2x2, n = 4096 ... 30720), so a
// seed never exceeds what unseeded SLATE holds (distrib_la._slate.lu_workspace_bytes,
// the same rules):
//   A (getrf):  (mt/p + nt/q) / 2 remote tiles (measured 10 ... 243 beside the local).
//   X (getrs):  none (measured 0 ... 1: trsmB releases B's broadcast row after each
//               step, work_trsm.cc B_panel.releaseRemoteWorkspace).  The panel rule
//               this replaced seeded 2 (mt + nt) + 8 blocks of nb x nrhs/q -- at
//               nrhs = n ~2 n^2 / q elements of never-used workspace (50 GiB at
//               n = 51200 c128 on 2x2), the OOM it caused.
template <typename T>
static sl::Matrix<T> owned(int64_t m, int64_t ncols, int64_t mb, int64_t nb_cols,
                           const lorrax_ffi::slate::SlateCtx* ctx,
                           lorrax_ffi::slate::PoolSeed& ws, int64_t* seeded,
                           bool rhs = false) {
    sl::Matrix<T> M(m, ncols, mb, nb_cols, ctx->p, ctx->q, ctx->comm);
    int64_t local = 0;
    for (int64_t j = 0; j < M.nt(); ++j)
        for (int64_t i = 0; i < M.mt(); ++i)
            if (M.tileIsLocal(i, j)) ++local;
    const int64_t extra = rhs ? 0 : (M.mt() / ctx->p + M.nt() / ctx->q) / 2;
    *seeded = ws.seed(M, local + extra);
    M.insertLocalTiles(sl::Target::Devices);
    return M;
}

template <typename T>
static void pool_report(const char* what, const lorrax_ffi::slate::SlateCtx* ctx, int64_t iq,
                        sl::Matrix<T>& A, int64_t sa, sl::Matrix<T>* X, int64_t sx,
                        const lorrax_ffi::slate::PoolSeed& ws) {
    if (!lorrax_ffi::slate::pool_report_enabled()) return;
    int r = 0;
    MPI_Comm_rank(ctx->comm, &r);
    std::fprintf(stderr, "[slate.%s] rank=%d q=%lld pool seeded A=%lld X=%lld fallback A=%lld "
                 "X=%lld blocks; ws_used=%.3fGB ws=%.3fGB\n", what, r, (long long) iq,
                 (long long) sa, (long long) sx,
                 (long long) lorrax_ffi::slate::pool_fallback(A, sa),
                 (long long) (X ? lorrax_ffi::slate::pool_fallback(*X, sx) : 0),
                 ws.used / 1e9, ws.size / 1e9);
}

template <typename T>
static int64_t local_of(sl::Matrix<T>& M) {
    int64_t k = 0;
    for (int64_t j = 0; j < M.nt(); ++j)
        for (int64_t i = 0; i < M.mt(); ++i)
            if (M.tileIsLocal(i, j)) ++k;
    return k;
}

// The pools' measured high-water (pool_seed.h PoolWatch), one line per matrix.
static void hw_report(lorrax_ffi::slate::PoolWatch& pw, const char* what,
                      const lorrax_ffi::slate::SlateCtx* ctx, int64_t n, int64_t nb, int itemsize) {
    if (!lorrax_ffi::slate::pool_report_enabled()) return;
    int r = 0;
    MPI_Comm_rank(ctx->comm, &r);
    pw.report(what, r, n, nb, itemsize);
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
                                     const T* A, const T* B, T* X, T* LU,
                                     void* d_ws, std::size_t ws_bytes) {
    const char* what = "batched_solve_lu";
    if (auto e = preflight(what, stream, ctx, n, nb); !e.success()) return e;
    const int p = ctx->p, q = ctx->q;
    const int64_t lld = n / p;
    const int64_t a_elems = lld * (n / q);
    const int64_t b_elems = lld * (nrhs / q);
    for (int64_t iq = 0; iq < nq; ++iq) {
        lorrax_ffi::slate::PoolSeed ws(d_ws, ws_bytes);     // matrix iq's SLATE objects die below
        int64_t sa = 0, sx = 0;
        auto Aq = owned<T>(n, n, nb, nb, ctx, ws, &sa);
        auto Xq = owned<T>(n, nrhs, nb, nb_b, ctx, ws, &sx, /*rhs=*/true);
        lorrax_ffi::slate::PoolWatch pw(lorrax_ffi::slate::pool_report_enabled());
        lorrax_ffi::slate::PoolWatchScope wa(pw, Aq, "A", sa, local_of(Aq));
        lorrax_ffi::slate::PoolWatchScope wx(pw, Xq, "X", sx, local_of(Xq));
        if (auto e = copy_in<T>(what, Aq, A + iq * a_elems, lld, p, q, stream); !e.success()) return e;
        if (auto e = copy_in<T>(what, Xq, B + iq * b_elems, lld, p, q, stream); !e.success()) return e;
        sl::Pivots piv;
        if (auto e = factor_one<T>(what, Aq, piv, iq); !e.success()) return e;
        if (auto e = solve_one<T>(Aq, piv, Xq, iq); !e.success()) return e;
        pool_report<T>(what, ctx, iq, Aq, sa, &Xq, sx, ws);
        hw_report(pw, what, ctx, n, nb, (int) sizeof(T));
        if (auto e = copy_out<T>(what, Xq, X + iq * b_elems, lld, p, q, stream); !e.success()) return e;
        if (auto e = copy_out<T>(what, Aq, LU + iq * a_elems, lld, p, q, stream); !e.success()) return e;
    }
    return ffi::Error::Success();
}

// ---- split: getrf, pivots serialised ---------------------------------------
template <typename T>
static ffi::Error GetrfImpl(lrx::gpu::Stream stream, lorrax_ffi::slate::SlateCtx* ctx,
                            int64_t nq, int64_t n, int64_t nb, int64_t piv_len,
                            const T* A, T* LU, int64_t* d_piv,
                            void* d_ws, std::size_t ws_bytes) {
    const char* what = "batched_getrf";
    if (auto e = preflight(what, stream, ctx, n, nb); !e.success()) return e;
    if (piv_len != 2 * n) {
        std::ostringstream os;
        os << "slate." << what << ": piv_len=" << piv_len << " != 2*n=" << 2 * n;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    const int p = ctx->p, q = ctx->q;
    const int64_t lld = n / p;
    const int64_t a_elems = lld * (n / q);
    std::vector<int64_t> h_piv(static_cast<std::size_t>(nq * piv_len), 0);
    for (int64_t iq = 0; iq < nq; ++iq) {
        lorrax_ffi::slate::PoolSeed ws(d_ws, ws_bytes);
        int64_t sa = 0;
        auto Aq = owned<T>(n, n, nb, nb, ctx, ws, &sa);
        lorrax_ffi::slate::PoolWatch pw(lorrax_ffi::slate::pool_report_enabled());
        lorrax_ffi::slate::PoolWatchScope wa(pw, Aq, "A", sa, local_of(Aq));
        if (auto e = copy_in<T>(what, Aq, A + iq * a_elems, lld, p, q, stream); !e.success()) return e;
        sl::Pivots piv;
        if (auto e = factor_one<T>(what, Aq, piv, iq); !e.success()) return e;
        pool_report<T>(what, ctx, iq, Aq, sa, nullptr, 0, ws);
        hw_report(pw, what, ctx, n, nb, (int) sizeof(T));
        if (auto e = copy_out<T>(what, Aq, LU + iq * a_elems, lld, p, q, stream); !e.success()) return e;
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
                            const T* B, T* X, void* d_ws, std::size_t ws_bytes) {
    const char* what = "batched_getrs";
    if (auto e = preflight(what, stream, ctx, n, nb); !e.success()) return e;
    if (piv_len != 2 * n) {
        std::ostringstream os;
        os << "slate." << what << ": piv_len=" << piv_len << " != 2*n=" << 2 * n;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    const int p = ctx->p, q = ctx->q;
    const int64_t lld = n / p;
    const int64_t a_elems = lld * (n / q);
    const int64_t b_elems = lld * (nrhs / q);
    std::vector<int64_t> h_piv(static_cast<std::size_t>(nq * piv_len), 0);
    lrx::gpu::Error ce = lrx::gpu::memcpy_async(h_piv.data(), d_piv, h_piv.size() * sizeof(int64_t),
                                                lrx::gpu::kDeviceToHost, stream);
    if (ce != lrx::gpu::kSuccess) return gpu_fail(what, LRX_GPU_API "MemcpyAsync(piv D2H)", ce);
    ce = lrx::gpu::stream_synchronize(stream);
    if (ce != lrx::gpu::kSuccess) return gpu_fail(what, LRX_GPU_API "StreamSynchronize(piv)", ce);
    for (int64_t iq = 0; iq < nq; ++iq) {
        lorrax_ffi::slate::PoolSeed ws(d_ws, ws_bytes);
        int64_t sa = 0, sx = 0;
        auto Aq = owned<T>(n, n, nb, nb, ctx, ws, &sa);
        auto Xq = owned<T>(n, nrhs, nb, nb_b, ctx, ws, &sx, /*rhs=*/true);
        lorrax_ffi::slate::PoolWatch pw(lorrax_ffi::slate::pool_report_enabled());
        lorrax_ffi::slate::PoolWatchScope wa(pw, Aq, "A", sa, local_of(Aq));
        lorrax_ffi::slate::PoolWatchScope wx(pw, Xq, "X", sx, local_of(Xq));
        if (auto e = copy_in<T>(what, Aq, LU + iq * a_elems, lld, p, q, stream); !e.success()) return e;
        if (auto e = copy_in<T>(what, Xq, B + iq * b_elems, lld, p, q, stream); !e.success()) return e;
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
        pool_report<T>(what, ctx, iq, Aq, sa, &Xq, sx, ws);
        hw_report(pw, what, ctx, n, nb, (int) sizeof(T));
        if (auto e = copy_out<T>(what, Xq, X + iq * b_elems, lld, p, q, stream); !e.success()) return e;
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
    ffi::Result<ffi::AnyBuffer> X, ffi::Result<ffi::AnyBuffer> LU, ffi::Result<ffi::AnyBuffer> WS,
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
                static_cast<double*>(X->untyped_data()), static_cast<double*>(LU->untyped_data()),
                WS->untyped_data(), WS->size_bytes());
        case ffi::DataType::C128:
            return SolveLuFactorsImpl<C128>(stream, ctx, nq, n, nrhs, nb, nb_b,
                static_cast<const C128*>(A.untyped_data()), static_cast<const C128*>(B.untyped_data()),
                static_cast<C128*>(X->untyped_data()), static_cast<C128*>(LU->untyped_data()),
                WS->untyped_data(), WS->size_bytes());
        default: return bad_dtype("batched_solve_lu", dt);
    }
}

static ffi::Error GetrfDispatch(
    lrx::gpu::Stream stream, ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> LU, ffi::Result<ffi::AnyBuffer> piv, ffi::Result<ffi::AnyBuffer> WS,
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
                static_cast<const double*>(A.untyped_data()), static_cast<double*>(LU->untyped_data()), d_piv,
                WS->untyped_data(), WS->size_bytes());
        case ffi::DataType::C128:
            return GetrfImpl<C128>(stream, ctx, nq, n, nb, piv_len,
                static_cast<const C128*>(A.untyped_data()), static_cast<C128*>(LU->untyped_data()), d_piv,
                WS->untyped_data(), WS->size_bytes());
        default: return bad_dtype("batched_getrf", dt);
    }
}

static ffi::Error GetrsDispatch(
    lrx::gpu::Stream stream, ffi::AnyBuffer LU, ffi::AnyBuffer piv, ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> X, ffi::Result<ffi::AnyBuffer> WS,
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
                static_cast<const double*>(B.untyped_data()), static_cast<double*>(X->untyped_data()),
                WS->untyped_data(), WS->size_bytes());
        case ffi::DataType::C128:
            return GetrsImpl<C128>(stream, ctx, nq, n, nrhs, nb, nb_b, piv_len,
                static_cast<const C128*>(LU.untyped_data()), d_piv,
                static_cast<const C128*>(B.untyped_data()), static_cast<C128*>(X->untyped_data()),
                WS->untyped_data(), WS->size_bytes());
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
        .Ret<xla::ffi::AnyBuffer>()              // workspace (u8): SLATE's pool seed
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("nb")                     // A tile: divides n/p and n/q
        .Attr<int64_t>("nb_b")                   // B column tile: nrhs / q
        .Attr<int64_t>("ctx_key"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SlateBatchedGetrfFfi, lorrax_ffi::slate_lu::GetrfDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()              // A (donated; factored in place)
        .Ret<xla::ffi::AnyBuffer>()              // LU (aliased to A)
        .Ret<xla::ffi::AnyBuffer>()              // piv (int64, 2n per matrix, replicated)
        .Ret<xla::ffi::AnyBuffer>()              // workspace (u8): SLATE's pool seed
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
        .Ret<xla::ffi::AnyBuffer>()              // workspace (u8): SLATE's pool seed
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("nb_b")
        .Attr<int64_t>("piv_len")
        .Attr<int64_t>("ctx_key"));
