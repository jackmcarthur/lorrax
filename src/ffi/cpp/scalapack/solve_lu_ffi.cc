// solve_lu_ffi.cc — per-q ScaLAPACK LU factor+solve (pXgetrf + pXgetrs)
// on the world-wide (Px, Py) grid, HOST platform (JAX CPU backend).
//
// This is the host twin of cpp/cusolvermp/batched_solve_lu_ffi.cc — the
// portable backend for the ``distributed_lu`` axis.  ScaLAPACK comes
// from Cray LibSci (libsci_gnu_mpi), which liblorrax_ffi_host.so already
// links for SLATE's BLAS — the port adds ZERO link dependencies.  The
// pXgetrf/pXgetrs and C-BLACS symbols are declared below by hand; libsci
// ships no C header for them.
//
// Grid: reuses the slate SlateCtx (pure MPI, ctx.h).  Its comm is
// rank-remapped so that BLACS "C" (column-major) grid order puts comm
// rank (mx + my*Px) at grid coords (mx, my) = the JAX mesh coords — the
// same pairing the slate handlers rely on for GridOrder::Col.  The grid
// and the Fortran-ABI prototypes live in ``blacs_grid.h``, shared with
// the other ScaLAPACK handlers.
//
// Layout contract (same family as the slate host handlers):
//   A : (Nq, N, N)     P(None,'x','y'), inner dims pre-transposed by the
//                      Python wrapper → per-slice col-major local block.
//   B : (Nq, N, NRHS)  same treatment.
//   X : same layout as B; solved in place (wrapper aliases B → X).
// descA/descB use SQUARE blocks g = N / max(Px, Py) — pXgetrf requires
// MB_A == NB_A, which restricts meshes to square or 1-D (the wrapper
// validates; on those meshes block-cyclic == JAX's contiguous shards).
// A is factored in place (donated by the wrapper); ipiv stays internal.

#include <complex>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include "xla/ffi/api/ffi.h"

// blacs_grid.h happens to pull both of these in, but this TU uses
// std::string (the refusal) and SlateCtx/host_collective_mutex directly —
// name the dependencies instead of leaning on transitive includes
// (2026-08-01 seam-audit leftover).
#include "../slate/ctx.h"
#include "blacs_grid.h"

namespace lorrax_ffi::scalapack_solve_lu {

namespace ffi = ::xla::ffi;
using lorrax_ffi::slate::SlateCtx;

template <typename T> struct Getrf;
template <> struct Getrf<double> {
    static void call(const int* m, const int* n, double* a, const int* ia,
                     const int* ja, const int* desca, int* ipiv, int* info)
    { pdgetrf_(m, n, a, ia, ja, desca, ipiv, info); }
};
template <> struct Getrf<std::complex<double>> {
    static void call(const int* m, const int* n, std::complex<double>* a,
                     const int* ia, const int* ja, const int* desca,
                     int* ipiv, int* info)
    { pzgetrf_(m, n, a, ia, ja, desca, ipiv, info); }
};

template <typename T> struct Getrs;
template <> struct Getrs<double> {
    static void call(const char* tr, const int* n, const int* nrhs,
                     const double* a, const int* ia, const int* ja,
                     const int* desca, const int* ipiv, double* b,
                     const int* ib, const int* jb, const int* descb,
                     int* info)
    { pdgetrs_(tr, n, nrhs, a, ia, ja, desca, ipiv, b, ib, jb, descb, info); }
};
template <> struct Getrs<std::complex<double>> {
    static void call(const char* tr, const int* n, const int* nrhs,
                     const std::complex<double>* a, const int* ia,
                     const int* ja, const int* desca, const int* ipiv,
                     std::complex<double>* b, const int* ib, const int* jb,
                     const int* descb, int* info)
    { pzgetrs_(tr, n, nrhs, a, ia, ja, desca, ipiv, b, ib, jb, descb, info); }
};

template <typename T>
static ffi::Error SolveLuImpl(
    int64_t nq, int64_t n, int64_t nrhs,
    int64_t g, int64_t nb_b,
    SlateCtx* ctx,
    const T* A_in, const T* B_in,
    T* A_scratch, T* X_out)
{
    const int p = ctx->p;
    const int q = ctx->q;
    const int ictxt = lorrax_ffi::scalapack::blacs_ctxt_for(ctx);

    int nprow = 0, npcol = 0, myrow = 0, mycol = 0;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
    if (nprow != p || npcol != q) {
        std::ostringstream os;
        os << "scalapack.solve_lu: BLACS grid " << nprow << "x" << npcol
           << " != ctx grid " << p << "x" << q;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    const int i_n = static_cast<int>(n);
    const int i_nrhs = static_cast<int>(nrhs);
    const int i_g = static_cast<int>(g);
    const int i_nb_b = static_cast<int>(nb_b);
    const int izero = 0, ione = 1;

    // Local extents from the block-cyclic distribution (== the JAX shard
    // extents on the square / 1-D meshes the wrapper allows).
    const int lld_a = numroc_(&i_n, &i_g, &myrow, &izero, &nprow);
    const int a_loc_cols = numroc_(&i_n, &i_g, &mycol, &izero, &npcol);
    const int b_loc_cols = numroc_(&i_nrhs, &i_nb_b, &mycol, &izero, &npcol);
    const int64_t A_slice = static_cast<int64_t>(lld_a) * a_loc_cols;
    const int64_t B_slice = static_cast<int64_t>(lld_a) * b_loc_cols;

    int desca[9], descb[9], info = 0;
    descinit_(desca, &i_n, &i_n, &i_g, &i_g, &izero, &izero, &ictxt,
              &lld_a, &info);
    if (info != 0) {
        std::ostringstream os;
        os << "scalapack.solve_lu: descinit(A) info=" << info;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    descinit_(descb, &i_n, &i_nrhs, &i_g, &i_nb_b, &izero, &izero, &ictxt,
              &lld_a, &info);
    if (info != 0) {
        std::ostringstream os;
        os << "scalapack.solve_lu: descinit(B) info=" << info;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }

    // In-place staging, same convention as the cusolvermp handler: A is
    // factored in its (donated) input buffer; B copied to X unless the
    // wrapper aliased them.
    if (A_scratch != static_cast<const T*>(A_in)) {
        std::memcpy(A_scratch, A_in, nq * A_slice * sizeof(T));
    }
    if (X_out != static_cast<const T*>(B_in)) {
        std::memcpy(X_out, B_in, nq * B_slice * sizeof(T));
    }

    // ipiv: LOCr(M_A) + MB_A per the ScaLAPACK spec; reused across q.
    std::vector<int> ipiv(static_cast<size_t>(lld_a) + i_g);

    for (int64_t iq = 0; iq < nq; ++iq) {
        T* A_slice_ptr = A_scratch + iq * A_slice;
        T* X_slice_ptr = X_out + iq * B_slice;

        info = 0;
        Getrf<T>::call(&i_n, &i_n, A_slice_ptr, &ione, &ione, desca,
                       ipiv.data(), &info);
        if (info != 0) {
            std::ostringstream os;
            os << "scalapack pXgetrf (q=" << iq << ") info=" << info
               << (info > 0 ? " (U singular at that pivot)" : " (bad argument)");
            return ffi::Error(info > 0 ? ffi::ErrorCode::kFailedPrecondition
                                       : ffi::ErrorCode::kInvalidArgument,
                              os.str());
        }

        info = 0;
        Getrs<T>::call("N", &i_n, &i_nrhs, A_slice_ptr, &ione, &ione, desca,
                       ipiv.data(), X_slice_ptr, &ione, &ione, descb, &info);
        if (info != 0) {
            std::ostringstream os;
            os << "scalapack pXgetrs (q=" << iq << ") info=" << info;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
    }

    return ffi::Error::Success();
}

static ffi::Error SolveLuDispatch(
    ffi::AnyBuffer A,
    ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> X_out,
    int64_t nq, int64_t n, int64_t nrhs,
    int64_t g, int64_t nb_b,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<SlateCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "scalapack.solve_lu: ctx_handle is null");
    }
    // Same-comm collective serialization vs the slate host handlers —
    // see host_collective_mutex() in cpp/slate/ctx.h.
    std::lock_guard<std::mutex> lock(
        lorrax_ffi::slate::host_collective_mutex());
    // MKL team pin, same rationale + knob as the eigh handler
    // (LORRAX_SCALAPACK_MKL_THREADS; measured matrix in blacs_grid.h).
    lorrax_ffi::scalapack::MklThreadScope mkl_scope(
        lorrax_ffi::scalapack::scalapack_mkl_threads());
    const auto dtype = A.element_type();
    if (B.element_type() != dtype || X_out->element_type() != dtype) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "scalapack.solve_lu: A, B, X must share dtype");
    }
    // PROVENANCE — same guard as the eigh handler.  The LU shims are not
    // bug L-2, but they carry the other two defects (hard-coded
    // MPI_COMM_WORLD vs this ctx's rank-remapped comm, and info hard-wired
    // to 0), so an unannounced substitution is still a silent-wrong-answer
    // route.  Checked on getrf AND getrs: the two live in different SLATE
    // translation units and a partial interposition is possible.
    // See blacs_grid.h for the measured symbol table and the waiver.
    for (const char* nm : {(dtype == ffi::DataType::F64) ? "pdgetrf_"
                                                        : "pzgetrf_",
                           (dtype == ffi::DataType::F64) ? "pdgetrs_"
                                                        : "pzgetrs_"}) {
        const std::string refusal =
            lorrax_ffi::scalapack::scalapack_slate_api_refusal("solve_lu", nm);
        if (!refusal.empty()) {
            return ffi::Error(ffi::ErrorCode::kUnimplemented, refusal);
        }
    }
    // A's buffer is donated — getrf factors in place (cast away const,
    // same convention as the cusolvermp twin).
    switch (dtype) {
        case ffi::DataType::F64:
            return SolveLuImpl<double>(
                nq, n, nrhs, g, nb_b, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<const double*>(B.untyped_data()),
                const_cast<double*>(static_cast<const double*>(A.untyped_data())),
                static_cast<double*>(X_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return SolveLuImpl<C128>(
                nq, n, nrhs, g, nb_b, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<const C128*>(B.untyped_data()),
                const_cast<C128*>(static_cast<const C128*>(A.untyped_data())),
                static_cast<C128*>(X_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "scalapack.solve_lu: unsupported dtype " << (int)dtype
               << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::scalapack_solve_lu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScalapackBatchedSolveLuHostFfi,
    lorrax_ffi::scalapack_solve_lu::SolveLuDispatch,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()      // A (donated, LU factored in place)
        .Arg<xla::ffi::AnyBuffer>()      // B
        .Ret<xla::ffi::AnyBuffer>()      // X (aliased to B — in-place solve)
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("g")              // square block: N / max(Px, Py)
        .Attr<int64_t>("nb_b")           // B col block: NRHS/Py (or NRHS if Py==1)
        .Attr<int64_t>("ctx_handle"));   // SlateCtx (shared with ffi.slate)
