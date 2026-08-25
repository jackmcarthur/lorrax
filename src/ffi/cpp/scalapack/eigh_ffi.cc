// eigh_ffi.cc — batched distributed Hermitian eigendecomposition on the
// world-wide (Px, Py) grid via ScaLAPACK's divide & conquer solver
// (pzheevd / pdsyevd), HOST platform (JAX CPU backend).
//
// This is the CPU-permanent distributed eigh for LORRAX.  It replaces
// nothing: SLATE's host ``heev`` SIGSEGVs deterministically at every
// configuration tried (bug L-2, scorecard L §4 — reproduced on a 1x1
// mesh, single rank, so it is not MPI, not the comm remap, not the
// layout contract), while ScaLAPACK's getrf/getrs on the SAME library
// and context are clean.  ScaLAPACK comes from MKL, already linked for
// solve_lu (see src/ffi/cpp/CMakeLists.txt) — this handler
// adds ZERO link dependencies.
//
// Why pzheevd and not pzheevx / pzheevr:
//   * we always want ALL eigenpairs, which is exactly D&C's regime;
//   * pzheevx needs ORFAC/cluster heuristics and can return
//     non-orthogonal vectors for tight clusters without extra workspace
//     (the ζ-fit charge CCT is precisely a clustered spectrum);
//   * pzheevr (MRRR) is the documented fallback if a pzheevd bug shows
//     up — same descriptor contract, JOBZ='V', RANGE='A'.
// JOBZ is hard-wired to 'V': netlib PxHEEVD does not implement 'N', and
// every consumer here needs the eigenvectors.
//
// Layout contract (same family as the other host handlers):
//   A : (Nq, N, N)  P(None,'x','y'), inner dims pre-transposed by the
//                   Python wrapper -> per-slice col-major local block.
//   W : (Nq, N)     REPLICATED float64, ascending.  ScaLAPACK computes W
//                   on every process of the grid.
//   Z : same layout as A; eigenvectors as COLUMNS of the global matrix.
// descA == descZ, square blocks g = N / max(Px, Py) (pXheevd requires
// MB_A == NB_A), which restricts meshes to square or 1-D — on those the
// block-cyclic distribution IS JAX's contiguous shard layout.  The
// wrapper validates the geometry.
//
// A is NOT donated: pXheevd destroys its input, and XLA input buffers
// must not be mutated, so each q is memcpy'd into a scratch block first
// (one local block, N^2/P elements — 6 MB at N=2448 on 16 ranks).
//
// Concurrency: XLA's CPU runtime executes independent ffi_calls on its
// intra-op thread pool, so every collective here serializes on
// host_collective_mutex() (cpp/slate/ctx.h) exactly like the slate host
// handlers and scalapack solve_lu — they share the same comms.

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include "xla/ffi/api/ffi.h"

#include "blacs_grid.h"

namespace lorrax_ffi::scalapack_eigh {

namespace ffi = ::xla::ffi;
using lorrax_ffi::slate::SlateCtx;

// Workspace sizes for one pXheevd call on this descriptor.  The three
// lengths are taken as max(vendor query, the documented netlib lower
// bound) — the query is authoritative when it is right and the formula
// covers implementations whose query returns a stale minimum.  All three
// are O(N^2/P) at worst, so over-allocating is cheap insurance against a
// silent workspace overrun.
struct WorkSizes {
    int lwork  = 1;   // WORK  (complex for pzheevd, real for pdsyevd)
    int lrwork = 1;   // RWORK (pzheevd only)
    int liwork = 1;   // IWORK
};

// One templated seam over the two ABIs.  pdsyevd has NO RWORK argument,
// so the real specialisation ignores rwork/lrwork entirely.
template <typename T>
static void call_evd(const int* n, T* a, const int* desca, double* w,
                     T* z, const int* descz,
                     T* work, const int* lwork,
                     double* rwork, const int* lrwork,
                     int* iwork, const int* liwork, int* info)
{
    const int ione = 1;
    if constexpr (std::is_same_v<T, double>) {
        (void)rwork; (void)lrwork;
        pdsyevd_("V", "L", n, a, &ione, &ione, desca, w,
                 z, &ione, &ione, descz,
                 work, lwork, iwork, liwork, info);
    } else {
        pzheevd_("V", "L", n, a, &ione, &ione, desca, w,
                 z, &ione, &ione, descz,
                 work, lwork, rwork, lrwork, iwork, liwork, info);
    }
}

// Real part of a queried WORK(1), for both element types.
template <typename T>
static inline double work_query_value(const T& v) {
    if constexpr (std::is_same_v<T, double>) return v;
    else return std::real(v);
}

template <typename T>
static ffi::Error EighImpl(
    int64_t nq, int64_t n, int64_t g,
    SlateCtx* ctx,
    const T* A_in, double* W_out, T* Z_out)
{
    const int p = ctx->p;
    const int q = ctx->q;
    const int ictxt = lorrax_ffi::scalapack::blacs_ctxt_for(ctx);

    int nprow = 0, npcol = 0, myrow = 0, mycol = 0;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
    if (nprow != p || npcol != q) {
        std::ostringstream os;
        os << "scalapack.eigh: BLACS grid " << nprow << "x" << npcol
           << " != ctx grid " << p << "x" << q;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    const int i_n = static_cast<int>(n);
    const int i_g = static_cast<int>(g);
    const int izero = 0;

    const int np    = numroc_(&i_n, &i_g, &myrow, &izero, &nprow);
    const int nq_lc = numroc_(&i_n, &i_g, &mycol, &izero, &npcol);
    const int lld   = std::max(np, 1);
    const int64_t slice = static_cast<int64_t>(lld) * nq_lc;

    int desc[9], info = 0;
    descinit_(desc, &i_n, &i_n, &i_g, &i_g, &izero, &izero, &ictxt,
              &lld, &info);
    if (info != 0) {
        std::ostringstream os;
        os << "scalapack.eigh: descinit info=" << info;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }

    // ---- workspace: vendor query, floored by the netlib formulas -----
    std::vector<T> A_scratch(static_cast<size_t>(slice ? slice : 1));
    WorkSizes ws;
    {
        // NP0/MQ0/NQ0 are the "worst rank" local extents the reference
        // formulas are written against (row/col origin 0).
        const int nmax = std::max<int>(i_n, std::max(i_g, 2));
        const int np0  = numroc_(&nmax, &i_g, &izero, &izero, &nprow);
        const int mq0  = numroc_(&nmax, &i_g, &izero, &izero, &npcol);
        if constexpr (std::is_same_v<T, double>) {
            const int64_t trilwmin =
                3LL * i_n + std::max<int64_t>(
                    static_cast<int64_t>(i_g) * (np + 1), 3LL * i_g);
            ws.lwork = static_cast<int>(
                std::max<int64_t>(1LL + 6 * i_n + 2LL * np * nq_lc, trilwmin)
                + 2LL * i_n);
            ws.lrwork = 1;
        } else {
            ws.lwork = static_cast<int>(
                i_n + (static_cast<int64_t>(np0) + mq0 + i_g) * i_g);
            ws.lrwork = static_cast<int>(
                1LL + 9LL * i_n + 3LL * np0 * mq0);
        }
        ws.liwork = 7 * i_n + 8 * npcol + 2;

        // The eigenvector BACK-TRANSFORM (pXunmtr/pXormtr) is a separate
        // requirement that pXheevd's published LWORK formula does NOT
        // always cover, and neither does MKL's query.  MEASURED: real
        // symmetric n=32 on a 1x1 grid -> PDSYEVD asks for 2305 doubles,
        // PDORMTR needs 3072, and the run comes back INFO=0 with correct
        // EIGENVALUES and a silently garbage Z (the only visible sign is a
        // `PDORMTR parameter number 16 had an illegal value` line on
        // stderr from PXERBLA).  Complex escaped it only because
        // pzheevd's formula happens to be larger.  Floor both with
        // pXunmtr's own SIDE='L' bound.
        // pXunmtr/pXormtr does not get all of WORK: pXheevd carves TAU, D,
        // E, D2, E2 (5 vectors of length N) off the front and hands it the
        // TAIL, so the floor has to cover the bound PLUS those offsets.
        // 8N is that, with slack for vendor variation — an O(N) cushion on
        // an O(N²/P) buffer.
        const int64_t unmtr =
            std::max<int64_t>(static_cast<int64_t>(i_g) * (i_g - 1) / 2,
                              (static_cast<int64_t>(np0) + mq0) * i_g)
            + static_cast<int64_t>(i_g) * i_g;
        ws.lwork = std::max<int>(ws.lwork,
                                 static_cast<int>(unmtr + 8LL * i_n));

        // Vendor query (LWORK = -1): every pXheevd writes the required
        // lengths into WORK(1)/RWORK(1)/IWORK(1) and returns.
        //
        // The query is MANDATORY on MKL, not a refinement: MKL's pzheevd
        // returns an LWORK far above the netlib LWMIN on multi-rank grids
        // (measured 368x at N=20000 on a 2x2 grid) and REJECTS the netlib
        // minimum with INFO = -16.  So a failed query is fatal here — the
        // formulas above are only a floor, never a substitute.
        T      wq{};
        double rq = 0.0;
        int    iq = 0;
        const int minus1 = -1;
        info = 0;
        call_evd<T>(&i_n, A_scratch.data(), desc, W_out,
                    Z_out, desc, &wq, &minus1, &rq, &minus1, &iq, &minus1,
                    &info);
        if (info != 0) {
            std::ostringstream os;
            os << "scalapack.eigh: pXheevd workspace query failed, info="
               << info << " (the query is mandatory — MKL rejects the "
                          "reference LWMIN with info=-16)";
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
        ws.lwork  = std::max<int>(ws.lwork,  static_cast<int>(
                                      work_query_value<T>(wq)));
        ws.lrwork = std::max<int>(ws.lrwork, static_cast<int>(rq));
        ws.liwork = std::max<int>(ws.liwork, iq);
        ws.lwork  = std::max(ws.lwork, 1);
        ws.lrwork = std::max(ws.lrwork, 1);
        ws.liwork = std::max(ws.liwork, 1);

        // Workspace can dwarf the local matrix block on a small grid, and
        // it is invisible to the JAX-side memory planner (it is malloc'd
        // here, not an XLA buffer).  Driver debug prints it once per call so
        // a planner surprise is diagnosable.
        static const bool log_ws = mklpin::debug_print_here();
        if (log_ws) {
            const double gib = 1024.0 * 1024.0 * 1024.0;
            std::fprintf(
                stderr,
                "[scalapack.eigh] n=%d g=%d grid=%dx%d loc=%dx%d "
                "lwork=%d (%.4f GiB) lrwork=%d (%.4f GiB) liwork=%d\n",
                i_n, i_g, nprow, npcol, np, nq_lc,
                ws.lwork,  ws.lwork * sizeof(T) / gib,
                ws.lrwork, ws.lrwork * sizeof(double) / gib,
                ws.liwork);
        }
    }

    std::vector<T>      work(static_cast<size_t>(ws.lwork));
    std::vector<double> rwork(static_cast<size_t>(ws.lrwork));
    std::vector<int>    iwork(static_cast<size_t>(ws.liwork));

    for (int64_t iq = 0; iq < nq; ++iq) {
        // pXheevd destroys A; stage the (read-only) XLA input.
        if (slice) {
            std::memcpy(A_scratch.data(), A_in + iq * slice,
                        static_cast<size_t>(slice) * sizeof(T));
        }
        info = 0;
        call_evd<T>(&i_n, A_scratch.data(), desc, W_out + iq * n,
                    Z_out + iq * slice, desc,
                    work.data(), &ws.lwork,
                    rwork.data(), &ws.lrwork,
                    iwork.data(), &ws.liwork, &info);
        if (info != 0) {
            std::ostringstream os;
            os << "scalapack pXheevd (q=" << iq << ") info=" << info
               << (info > 0
                   ? " (the algorithm failed to converge / the eigenvector"
                     " back-transform failed on that block)"
                   : " (bad argument)");
            return ffi::Error(info > 0 ? ffi::ErrorCode::kInternal
                                       : ffi::ErrorCode::kInvalidArgument,
                              os.str());
        }
    }

    return ffi::Error::Success();
}

static ffi::Error EighDispatch(
    ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> W_out,
    ffi::Result<ffi::AnyBuffer> Z_out,
    int64_t nq, int64_t n, int64_t g,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<SlateCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "scalapack.eigh: ctx_handle is null");
    }
    // Same-comm collective serialization vs the slate/scalapack host
    // handlers — see host_collective_mutex() in cpp/slate/ctx.h.
    std::lock_guard<std::mutex> lock(
        lorrax_ffi::slate::host_collective_mutex());

    // Pin the MKL team for the pXheevd calls.  At the production grid
    // (12x12, g=204) the harness-wide MKL_NUM_THREADS=28 is CATASTROPHIC
    // for this solver: 11.28 s/q at 14 threads vs 0.463 s/q at 4 (24x,
    // measured — see blacs_grid.h).  LORRAX_SCALAPACK_MKL_THREADS
    // overrides; default = min(current, 4).  Thread-local, restored on
    // scope exit — the global setting (right for the local zheevd_
    // plan-A route) is untouched.
    lorrax_ffi::scalapack::MklThreadScope mkl_scope(
        lorrax_ffi::scalapack::scalapack_mkl_threads());

    const auto dtype = A.element_type();
    if (Z_out->element_type() != dtype) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "scalapack.eigh: Z output dtype must match A");
    }
    if (W_out->element_type() != ffi::DataType::F64) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "scalapack.eigh: W output must be float64 (real eigenvalues)");
    }

    // PROVENANCE.  Refuse if the pXheevd this process will actually call is
    // SLATE's overlay rather than the linked ScaLAPACK: that body is
    // slate::heev, i.e. bug L-2 — the deterministic SIGSEGV this backend
    // exists to avoid, and the one failure mode that leaves no traceback.
    // See the block in blacs_grid.h for the measured symbol table and the
    // LORRAX_SCALAPACK_ALLOW_SLATE_API waiver.
    {
        const char* evd = (dtype == ffi::DataType::F64) ? "pdsyevd_"
                                                        : "pzheevd_";
        const std::string refusal =
            lorrax_ffi::scalapack::scalapack_slate_api_refusal("eigh", evd);
        if (!refusal.empty()) {
            return ffi::Error(ffi::ErrorCode::kUnimplemented, refusal);
        }
    }

    switch (dtype) {
        case ffi::DataType::F64:
            return EighImpl<double>(
                nq, n, g, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<double*>(W_out->untyped_data()),
                static_cast<double*>(Z_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return EighImpl<C128>(
                nq, n, g, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<double*>(W_out->untyped_data()),
                static_cast<C128*>(Z_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "scalapack.eigh: unsupported dtype " << (int)dtype
               << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::scalapack_eigh

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScalapackEighHostFfi,
    lorrax_ffi::scalapack_eigh::EighDispatch,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()      // A (Nq local blocks, col-major)
        .Ret<xla::ffi::AnyBuffer>()      // W (Nq, N) float64, replicated
        .Ret<xla::ffi::AnyBuffer>()      // Z (Nq local blocks, col-major)
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("g")              // square block: N / max(Px, Py)
        .Attr<int64_t>("ctx_handle"));   // SlateCtx (shared with ffi.slate)
