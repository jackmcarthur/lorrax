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
//                std::vector<real_t> on each rank; we copy them H->D
//                to the replicated W buffer.
//
// Input preservation: slate::heev overwrites its input with Householder
// reduction intermediates, but A is an XLA *input* buffer and must not
// be mutated (XLA may share it with other consumers).  We copy the
// local shard into SLATE-owned tiles (see EighImpl) and let SLATE destroy
// those; the eigenvector tiles are copied into Q after heev returns.
//
// Device memory: A, Z and the 1-D eigenvector matrix Z1d are SLATE-owned device
// tiles whose pool is SEEDED with this rank's bytes from XLA's scratch allocator
// (pool_seed.h ScratchSeed), so they live in XLA's arena.  The driver
// below is slate::heev's own stage sequence (v2025.05.28-1, src/heev.cc), made
// explicit so Z1d can be seeded and each stage timed: LORRAX_SLATE_TIMERS=1 prints
// one line per call per rank, plus the device bytes in use per stage (a sampler
// thread; diagnostics only).
//
// Tile size: any nb dividing n/p.  The Python wrapper relabels (passes the
// P('x','y') shard as-is with nb < n/p, so SLATE solves Pi A Pi^T) and
// un-permutes Q's columns; see distrib_la._cyclic.

#include "../common/ctx_registry.h"
#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <tuple>
#include <limits>
#include <vector>
#include <sstream>
#include <type_traits>

#include "../common/gpu_runtime.h"
#include <blas.hh>
#include <slate/slate.hh>

#include "xla/ffi/api/ffi.h"

#include "../common/ffi_helpers.h"
#include "ctx.h"
#include "pool_seed.h"

namespace lorrax_ffi::slate_ffi {

namespace ffi = ::xla::ffi;
namespace sl  = ::slate;
using lorrax_ffi::slate::PoolSeed;

// Map a dtype to its real-part type (for eigenvalues).
template <typename T> struct RealOf         { using type = T; };
template <> struct RealOf<std::complex<float>>  { using type = float; };
template <> struct RealOf<std::complex<double>> { using type = double; };

// Local tiles of M (lower triangle only for a Hermitian / trapezoid matrix).
template <typename M>
static int64_t local_tiles(M& A, bool lower_only) {
    int64_t k = 0;
    for (int64_t j = 0; j < A.nt(); ++j)
        for (int64_t i = lower_only ? j : 0; i < A.mt(); ++i)
            if (A.tileIsLocal(i, j)) ++k;
    return k;
}

// A's seed: he2hb's measured high-water on A's pool (PoolWatch, MI250X 2x2, n = 4096 ...
// 30720): every local tile of the FULL matrix less half a tile column, never below the
// lower tiles.  Z and Z1d are seeded with their local tiles only (their measured
// high-water).  Lower envelopes, so a seed never exceeds what unseeded SLATE holds
// on any rank (min over ranks of A's high-water = full local tiles - mt/2 at every
// measured n; Z, Z1d = local exactly).
template <typename M>
static int64_t a_seed_tiles(M& A) {
    return std::max(local_tiles(A, true), local_tiles(A, false) - A.mt() / 2);
}

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Diagonal and off-diagonal of the band after hb2st (bandwidth 1), lower storage
// on the host of rank 0: slate::internal::copyhb2st's reads.
template <typename T, typename real_t>
static void copy_tridiag(sl::HermitianBandMatrix<T>& Aband,
                         std::vector<real_t>& D, std::vector<real_t>& E) {
    const int64_t nt = Aband.nt();
    int64_t di = 0, ei = 0;
    for (int64_t i = 0; i < nt; ++i) {
        if (i > 0) {
            auto Tb = Aband(i, i - 1);
            E[ei++] = std::real(Tb(0, Tb.nb() - 1));
        }
        auto Td = Aband(i, i);
        const int64_t len = Td.nb();
        for (int64_t j = 0; j < len; ++j) D[di + j] = std::real(Td(j, j));
        for (int64_t j = 0; j + 1 < len; ++j) E[ei + j] = std::real(Td(j + 1, j));
        di += len;
        ei += len - 1;
    }
}

struct EighStages {
    double scale = 0, he2hb = 0, gather = 0, hb2st = 0, tridiag = 0, redist1 = 0,
           unmtr_hb2st = 0, redist2 = 0, unmtr_he2hb = 0;
    int64_t seeded = 0, fallback = 0;
    lorrax_ffi::slate::StageMemSampler* mem = nullptr;
    lorrax_ffi::slate::PoolWatch* pw = nullptr;
    bool scratch_failed = false;
    void at(int s) { if (mem) mem->at(s); }
};
// Stage indices of StageMemSampler: 0 copy-in, 1 he2hb, 2 gather+hb2st, 3 tridiag,
// 4 redistribute 1, 5 unmtr_hb2st, 6 redistribute 2, 7 unmtr_he2hb, 8 copy-out.
static const char* kStageNames[] = {"copyin", "he2hb", "hb2st", "tridiag", "redist1",
                                    "unmtr_hb2st", "redist2", "unmtr_he2hb", "copyout"};

// slate::heev (src/heev.cc) with A / Z / Z1d seeded and every stage timed.
template <typename T>
static void heev_seeded(sl::HermitianMatrix<T>& A, std::vector<typename RealOf<T>::type>& Lambda,
                        sl::Matrix<T>* Z, PoolSeed& ws, const sl::Options& opts,
                        EighStages& st, lrx::gpu::Stream stream) {
    using real_t = typename RealOf<T>::type;
    const int64_t n = A.n();
    const real_t safe_min = std::numeric_limits<real_t>::min();
    const real_t eps      = std::numeric_limits<real_t>::epsilon();
    const real_t sml_num  = safe_min / eps;
    const real_t big_num  = 1 / sml_num;
    const real_t sqrt_sml = std::sqrt(sml_num);
    const real_t sqrt_big = std::sqrt(big_num);
    double t = now_s();
    real_t Anorm = sl::norm(sl::Norm::Max, A, opts);
    if (std::isnan(Anorm) || std::isinf(Anorm)) {
        Lambda.assign(Lambda.size(), Anorm);
        return;
    }
    real_t alpha = 1.0;
    if (Anorm > 0 && Anorm < sqrt_sml) alpha = sqrt_sml;
    else if (Anorm > sqrt_big)         alpha = sqrt_big;
    if (alpha != 1.0) sl::scale(alpha, Anorm, A, opts);
    st.scale = now_s() - t;

    t = now_s();
    st.at(1);
    sl::TriangularFactors<T> Tf;
    sl::he2hb(A, Tf, opts);
    st.he2hb = now_s() - t;

    t = now_s();
    st.at(2);
    const int64_t nb = A.tileNb(0);
    sl::HermitianBandMatrix<T> Aband(A.uplo(), n, nb, nb, 1, 1, A.mpiComm());
    Aband.insertLocalTiles();
    Aband.he2hbGather(A);
    st.gather = now_s() - t;

    t = now_s();
    Lambda.resize(n);
    std::vector<real_t> E(n - 1);
    const int64_t vm = 2 * nb, nt = A.nt(), vn = nt * (nt + 1) / 2 * nb;
    sl::Matrix<T> V(vm, vn, vm, nb, 1, 1, A.mpiComm());
    if (A.mpiRank() == 0) {
        V.insertLocalTiles();
        sl::hb2st(Aband, V, opts);
        copy_tridiag(Aband, Lambda, E);
        Aband.releaseRemoteWorkspace();
    }
    st.hb2st = now_s() - t;

    MPI_Comm comm = A.mpiComm();
    if (Z == nullptr) {
        t = now_s();
        if (A.mpiRank() == 0) sl::sterf<real_t>(Lambda, E, opts);
        MPI_Bcast(Lambda.data(), (int) n, MPI_DOUBLE, 0, comm);
        st.tridiag = now_s() - t;
    } else {
        MPI_Bcast(Lambda.data(), (int) n,     MPI_DOUBLE, 0, comm);
        MPI_Bcast(E.data(),      (int) n - 1, MPI_DOUBLE, 0, comm);
        int mpi_size = 1;
        MPI_Comm_size(comm, &mpi_size);
        // Z1d: heev's 1 x P column-cyclic grid (all rows of a column tile on one
        // rank, unmtr_hb2st's requirement), here with a seeded pool.
        {
            sl::Matrix<T> Z1d(Z->m(), Z->n(), Z->tileNb(0), 1, mpi_size, comm);
            st.at(3);
            const int64_t want1 = local_tiles(Z1d, false);
            const int64_t got1 = ws.seed(Z1d, want1);
            st.seeded += got1;
            Z1d.insertLocalTiles(sl::Target::Devices);
            lorrax_ffi::slate::PoolWatchScope wz1(*st.pw, Z1d, "Z1d", got1,
                                                  local_tiles(Z1d, false));
            {
                t = now_s();
                if constexpr (!std::is_same_v<T, real_t>) {
                    auto Zreal = Z->template emptyLike<real_t>();
                    Zreal.insertLocalTiles();
                    sl::stedc(Lambda, E, Zreal);
                    sl::copy(Zreal, *Z);
                } else {
                    sl::stedc(Lambda, E, *Z);
                }
                st.tridiag = now_s() - t;
                t = now_s();
                st.at(4);
                sl::redistribute(*Z, Z1d, opts);
                st.redist1 = now_s() - t;
            }
            t = now_s();
            st.at(5);
            sl::unmtr_hb2st(sl::Side::Left, sl::Op::NoTrans, V, Z1d, opts);
            st.unmtr_hb2st = now_s() - t;
            t = now_s();
            st.at(6);
            sl::redistribute(Z1d, *Z, opts);
            st.redist2 = now_s() - t;
            st.fallback += lorrax_ffi::slate::pool_fallback(Z1d, got1);
        }
        t = now_s();
        st.at(7);
        sl::unmtr_he2hb(sl::Side::Left, sl::Op::NoTrans, A, Tf, *Z, opts);
        st.unmtr_he2hb = now_s() - t;
    }
    if (alpha != 1.0)
        blas::scal(n, Anorm / alpha, Lambda.data(), 1);
}

// ---------------------------------------------------------------------------
//  Impl — one scalar type.
// ---------------------------------------------------------------------------
template <typename T>
static ffi::Error EighImpl(
    int64_t n, int64_t nb,
    bool compute_evecs,
    lrx::gpu::Stream xla_stream,
    lorrax_ffi::slate::SlateCtx* ctx,
    T* d_A_local,
    typename RealOf<T>::type* d_W,
    T* d_Q_local,
    ffi::ScratchAllocator& scratch)
{
    using real_t = typename RealOf<T>::type;

    lrx::gpu::Error ce = lrx::gpu::stream_synchronize(xla_stream);
    if (ce != lrx::gpu::kSuccess) {
        std::ostringstream os;
        os << "slate.eigh: " LRX_GPU_API "StreamSynchronize(xla_stream) failed: "
           << lrx::gpu::error_string(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    const int p = ctx->p;
    const int q = ctx->q;
    const int64_t lld_A = (n + p - 1) / p;   // local leading dim of A tile
    const int64_t lld_Q = lld_A;

    // One device per process (JAX's model): SLATE's num_devices() must be 1.
    const int num_devices_visible = blas::get_device_count();
    if (num_devices_visible != 1) {
        std::ostringstream os;
        os << "slate.eigh: blas::get_device_count()=" << num_devices_visible
           << " but JAX one-process-per-GPU model requires exactly 1. "
           << "Set " LRX_GPU_VENDOR "_VISIBLE_DEVICES=$SLURM_LOCALID on the launcher "
           << "(run_shifter.sh does this when LORRAX_SLATE_STACK is set).";
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition, os.str());
    }

    // This rank's seed bytes (pool_seed.h ScratchSeed), sized once A exists.
    lorrax_ffi::slate::ScratchSeed smem;
    PoolSeed ws(nullptr, 0);
    EighStages st;
    const char* tv = std::getenv("LORRAX_SLATE_TIMERS");
    const bool report = (tv != nullptr && tv[0] != '\0' && tv[0] != '0') ||
                        lorrax_ffi::slate::pool_report_enabled();
    lorrax_ffi::slate::StageMemSampler mem(9, report);
    st.mem = &mem;
    lorrax_ffi::slate::PoolWatch pw(lorrax_ffi::slate::pool_report_enabled());
    st.pw = &pw;
    const double t_all = now_s();
    std::vector<real_t> Lambda(n);
    sl::Options opts = {
        {sl::Option::Target, sl::Target::Devices},
    };
    try {
        // A is copied into SLATE-owned device tiles (stride nb) whose pool is the
        // workspace: heev destroys its input (XLA inputs are immutable), and at
        // nb < n/p a wrapped buffer (lld = n/p) would sit beside SLATE's nb-stride
        // workspace tiles, which device-region batching rejects (internal_batch.hh
        // stride assert).  The local buffer holds the rank's block-cyclic tiles at
        // fromDevices' offsets (lld = n/p); the copy is the inverse of Z's below.
        sl::HermitianMatrix<T> A(sl::Uplo::Lower, n, nb, p, q, ctx->comm);
        {
            // Per-rank seed: A at he2hb's high-water rule, Z at this rank's local
            // tiles, Z1d at this rank's columns of the 1 x P column-cyclic grid
            // (tile column j on rank j mod P) -- each from this rank's own tiles.
            int rank = 0, size = 1;
            MPI_Comm_rank(ctx->comm, &rank);
            MPI_Comm_size(ctx->comm, &size);
            const int64_t a_want = a_seed_tiles(A);
            const int64_t z_want = compute_evecs ? local_tiles(A, false) : 0;
            const int64_t z1d_cols = rank < A.nt() ? (A.nt() - 1 - rank) / size + 1 : 0;
            const int64_t z1d_want = compute_evecs ? A.mt() * z1d_cols : 0;
            const std::size_t bs = lorrax_ffi::slate::block_bytes(A);
            smem = lorrax_ffi::slate::ScratchSeed(
                scratch, lorrax_ffi::slate::seed_bytes({{a_want, bs}, {z_want, bs}, {z1d_want, bs}}));
            ws = PoolSeed(smem.base, smem.size);
            st.scratch_failed = smem.failed;
        }
        const int64_t gotA = ws.seed(A, a_seed_tiles(A));
        st.seeded += gotA;
        A.insertLocalTiles(sl::Target::Devices);
        lorrax_ffi::slate::PoolWatchScope wa(pw, A, "A", gotA, local_tiles(A, true));
        for (int64_t tj = 0; tj < A.nt(); ++tj) {
            for (int64_t ti = tj; ti < A.mt(); ++ti) {
                if (!A.tileIsLocal(ti, tj)) continue;
                const int dev = A.tileDevice(ti, tj);
                auto tile = A(ti, tj, dev);
                const int64_t ii = ti * nb, jj = tj * nb;
                const int64_t ii_local = (ii / (nb * p)) * nb + (ii % nb);
                const int64_t jj_local = (jj / (nb * q)) * nb + (jj % nb);
                const T* src = d_A_local + ii_local + jj_local * lld_A;
                ce = lrx::gpu::memcpy_2d_async(
                    tile.data(), tile.stride() * sizeof(T),
                    src, lld_A * sizeof(T),
                    tile.mb() * sizeof(T), tile.nb(),
                    lrx::gpu::kDeviceToDevice, xla_stream);
                if (ce != lrx::gpu::kSuccess) {
                    std::ostringstream os;
                    os << "slate.eigh: " LRX_GPU_API "Memcpy2DAsync(A -> tile (" << ti
                       << "," << tj << ")) failed: " << lrx::gpu::error_string(ce);
                    return ffi::Error(ffi::ErrorCode::kInternal, os.str());
                }
            }
        }
        ce = lrx::gpu::stream_synchronize(xla_stream);
        if (ce != lrx::gpu::kSuccess) {
            std::ostringstream os;
            os << "slate.eigh: " LRX_GPU_API "StreamSynchronize after A copy failed: "
               << lrx::gpu::error_string(ce);
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }

        if (compute_evecs) {
            sl::Matrix<T> Z(n, n, nb, p, q, ctx->comm);
            const int64_t gotZ = ws.seed(Z, local_tiles(Z, false));
            st.seeded += gotZ;
            Z.insertLocalTiles(sl::Target::Devices);
            lorrax_ffi::slate::PoolWatchScope wz(pw, Z, "Z", gotZ, local_tiles(Z, false));
            heev_seeded(A, Lambda, &Z, ws, opts, st, xla_stream);
            st.at(8);

            // Local tiles -> d_Q_local at the fromDevices offsets of the rank's
            // block-cyclic tiles.  tileGetForReading is REQUIRED: the
            // back-transformation can leave the valid copy on the HOST.
            for (int64_t ti = 0; ti < Z.mt(); ++ti) {
                for (int64_t tj = 0; tj < Z.nt(); ++tj) {
                    if (!Z.tileIsLocal(ti, tj)) continue;
                    const int dev = Z.tileDevice(ti, tj);
                    Z.tileGetForReading(ti, tj, dev, sl::LayoutConvert::ColMajor);
                    auto tile = Z(ti, tj, dev);
                    const int64_t ii = ti * nb, jj = tj * nb;
                    const int64_t ii_local = (ii / (nb * p)) * nb + (ii % nb);
                    const int64_t jj_local = (jj / (nb * q)) * nb + (jj % nb);
                    T* dst = d_Q_local + ii_local + jj_local * lld_Q;
                    ce = lrx::gpu::memcpy_2d_async(
                        dst, lld_Q * sizeof(T),
                        tile.data(), tile.stride() * sizeof(T),
                        tile.mb() * sizeof(T), tile.nb(),
                        lrx::gpu::kDeviceToDevice, xla_stream);
                    if (ce != lrx::gpu::kSuccess) {
                        std::ostringstream os;
                        os << "slate.eigh: " LRX_GPU_API "Memcpy2DAsync(Z tile (" << ti
                           << "," << tj << ") -> d_Q_local) failed: "
                           << lrx::gpu::error_string(ce);
                        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
                    }
                }
            }
            // The copies read Z's tiles: finish before Z (and the pool) dies.
            ce = lrx::gpu::stream_synchronize(xla_stream);
            if (ce != lrx::gpu::kSuccess) {
                std::ostringstream os;
                os << "slate.eigh: " LRX_GPU_API "StreamSynchronize after Z copy "
                   << "failed: " << lrx::gpu::error_string(ce);
                return ffi::Error(ffi::ErrorCode::kInternal, os.str());
            }
            st.fallback += lorrax_ffi::slate::pool_fallback(Z, gotZ);
        } else {
            heev_seeded<T>(A, Lambda, nullptr, ws, opts, st, xla_stream);
        }
        st.fallback += lorrax_ffi::slate::pool_fallback(A, gotA);
    } catch (const std::exception& ex) {
        std::ostringstream os;
        os << "slate::heev (seeded driver) threw: " << ex.what();
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    if (report) {
        mem.stop = true;
        if (mem.th.joinable()) mem.th.join();
        int r = 0;
        MPI_Comm_rank(ctx->comm, &r);
        std::ostringstream ms;
        for (int k = 0; k < 9; ++k)
            ms << " " << kStageNames[k] << "=" << (double) mem.max_used[k] / 1e9;
        std::fprintf(stderr, "[slate.eigh.mem] rank=%d device_used_GB_max_per_stage:%s\n",
                     r, ms.str().c_str());
        std::fprintf(stderr,
            "[slate.eigh] rank=%d n=%lld nb=%lld total=%.2f scale=%.2f he2hb=%.2f "
            "gather=%.2f hb2st=%.2f tridiag=%.2f redist1=%.2f unmtr_hb2st=%.2f redist2=%.2f "
            "unmtr_he2hb=%.2f pool_seeded_blocks=%lld pool_fallback_blocks=%lld seeded_GB=%.3f "
            "scratch=%.3fGB%s\n",
            r, (long long) n, (long long) nb,
            now_s() - t_all, st.scale, st.he2hb, st.gather, st.hb2st, st.tridiag, st.redist1,
            st.unmtr_hb2st, st.redist2, st.unmtr_he2hb, (long long) st.seeded,
            (long long) st.fallback, ws.used / 1e9, ws.size / 1e9,
            st.scratch_failed ? " (scratch allocation FAILED: unseeded)" : "");
        pw.report("eigh", r, n, nb, (int) sizeof(T));
    }

    ce = lrx::gpu::memcpy_async(d_W, Lambda.data(), n * sizeof(real_t),
                         lrx::gpu::kHostToDevice, xla_stream);
    if (ce != lrx::gpu::kSuccess) {
        std::ostringstream os;
        os << "slate.eigh: " LRX_GPU_API "MemcpyAsync(Lambda -> d_W) failed: "
           << lrx::gpu::error_string(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    // Lambda is a host vector that dies at return: the copy must complete first.
    ce = lrx::gpu::stream_synchronize(xla_stream);
    if (ce != lrx::gpu::kSuccess) {
        std::ostringstream os;
        os << "slate.eigh: " LRX_GPU_API "StreamSynchronize(W) failed: "
           << lrx::gpu::error_string(ce);
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
//  Dispatch — validates shapes, looks up Ctx, switches on dtype.
// ---------------------------------------------------------------------------
static ffi::Error EighDispatch(
    lrx::gpu::Stream stream,
    ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> W_out,
    ffi::Result<ffi::AnyBuffer> Q_out,
    ffi::ScratchAllocator scratch,
    int64_t n, int64_t nb,
    int64_t ctx_key, bool compute_evecs)
{
    const int64_t ctx_handle = ::lorrax_ffi::ctx_registry::resolve(ctx_key, "slate");
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
                static_cast<double*>(Q_out->untyped_data()), scratch);
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return EighImpl<C128>(
                n, nb, compute_evecs, stream, ctx,
                static_cast<C128*>(const_cast<void*>(A.untyped_data())),
                static_cast<double*>(W_out->untyped_data()),
                static_cast<C128*>(Q_out->untyped_data()), scratch);
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
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()              // A (local shard)
        .Ret<xla::ffi::AnyBuffer>()              // W (n,) real, replicated
        .Ret<xla::ffi::AnyBuffer>()              // Q (local shard, same dtype as A)
        .Ctx<xla::ffi::ScratchAllocator>()       // this rank's pool seed (pool_seed.h)
        .Attr<int64_t>("n")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_key")
        .Attr<bool>("compute_evecs"));
