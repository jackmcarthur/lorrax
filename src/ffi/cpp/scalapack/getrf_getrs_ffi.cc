// getrf_getrs_ffi.cc — the ScaLAPACK LU pair SPLIT into its two halves,
// HOST platform (JAX CPU backend): per-q pXgetrf (factor ONCE per
// transverse ζ channel, factors + ipiv returned) and per-q pXgetrs
// (back-solve per r-chunk against the stored factors).
//
// Twin of solve_lu_ffi.cc (the FUSED factor+solve handler, kept for the
// legacy passthrough path); every convention — grid, descriptors,
// layout, provenance guard, thread pin — is identical, and pXgetrf on a
// given matrix is bit-identical whether or not the pXgetrs follows
// immediately (same descriptors, same grid), which is the whole
// correctness claim of the split.  See solve_lu_ffi.cc for the layout
// contract prose; deltas only:
//
//   getrf:  A (Nq, N, N) donated → LU in place; ipiv OUT as an int32
//           buffer (Nq, ipiv_len), ipiv_len = LOCr(M_A) + MB_A — each
//           rank writes ITS OWN local ipiv rows (ScaLAPACK distributes
//           ipiv over process rows, replicated across columns; we store
//           each rank's copy verbatim, opaque to Python).
//   getrs:  LU + ipiv arrive exactly as getrf left them; B donated → X.
//
// ipiv dtype: the lp64 ScaLAPACK ABI's int is 32-bit, so the buffer is
// int32 end to end (no conversion).

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
// (same seam-audit convention as solve_lu_ffi.cc, 2026-08-01).
#include "../slate/ctx.h"
#include "blacs_grid.h"

namespace lorrax_ffi::scalapack_getrf_getrs {

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

// Shared descriptor/grid prologue.  Fills desca (+ optionally descb) and
// the local extents; returns a non-empty error string on failure.
static std::string GridPrologue(
    SlateCtx* ctx, int64_t n, int64_t g,
    int* ictxt_out, int* lld_a_out, int* a_loc_cols_out, int desca[9])
{
    const int p = ctx->p;
    const int q = ctx->q;
    const int ictxt = lorrax_ffi::scalapack::blacs_ctxt_for(ctx);
    int nprow = 0, npcol = 0, myrow = 0, mycol = 0;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
    if (nprow != p || npcol != q) {
        std::ostringstream os;
        os << "scalapack.getrf_getrs: BLACS grid " << nprow << "x" << npcol
           << " != ctx grid " << p << "x" << q;
        return os.str();
    }
    const int i_n = static_cast<int>(n);
    const int i_g = static_cast<int>(g);
    const int izero = 0;
    const int lld_a = numroc_(&i_n, &i_g, &myrow, &izero, &nprow);
    const int a_loc_cols = numroc_(&i_n, &i_g, &mycol, &izero, &npcol);
    int info = 0;
    descinit_(desca, &i_n, &i_n, &i_g, &i_g, &izero, &izero, &ictxt,
              &lld_a, &info);
    if (info != 0) {
        std::ostringstream os;
        os << "scalapack.getrf_getrs: descinit(A) info=" << info;
        return os.str();
    }
    *ictxt_out = ictxt;
    *lld_a_out = lld_a;
    *a_loc_cols_out = a_loc_cols;
    return std::string();
}

// ---------------------------------------------------------------------------
// getrf: factor once, return LU (in place) + this rank's ipiv rows.
// ---------------------------------------------------------------------------

template <typename T>
static ffi::Error GetrfImpl(
    int64_t nq, int64_t n, int64_t g, int64_t ipiv_len,
    SlateCtx* ctx,
    const T* A_in, T* LU_out, int32_t* ipiv_out)
{
    int ictxt = 0, lld_a = 0, a_loc_cols = 0;
    int desca[9];
    const std::string err = GridPrologue(ctx, n, g, &ictxt, &lld_a,
                                         &a_loc_cols, desca);
    if (!err.empty())
        return ffi::Error(ffi::ErrorCode::kInternal, err);

    // The ScaLAPACK spec sizes ipiv as LOCr(M_A) + MB_A; the wrapper
    // computed the same number from (n, Px, g) — refuse on any mismatch
    // rather than write past a mis-sized buffer.
    const int64_t want = static_cast<int64_t>(lld_a) + static_cast<int64_t>(g);
    if (want > ipiv_len) {
        std::ostringstream os;
        os << "scalapack.getrf: ipiv_len=" << ipiv_len
           << " smaller than LOCr+MB=" << want;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }

    const int64_t A_slice = static_cast<int64_t>(lld_a) * a_loc_cols;
    if (LU_out != static_cast<const T*>(A_in)) {
        std::memcpy(LU_out, A_in, nq * A_slice * sizeof(T));
    }

    const int i_n = static_cast<int>(n);
    const int ione = 1;
    std::vector<int> ipiv(static_cast<size_t>(want));
    for (int64_t iq = 0; iq < nq; ++iq) {
        T* A_slice_ptr = LU_out + iq * A_slice;
        int info = 0;
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
        int32_t* out_row = ipiv_out + iq * ipiv_len;
        for (int64_t i = 0; i < want; ++i)
            out_row[i] = static_cast<int32_t>(ipiv[i]);
        for (int64_t i = want; i < ipiv_len; ++i)
            out_row[i] = 0;
    }
    return ffi::Error::Success();
}

static ffi::Error GetrfDispatch(
    ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> LU_out,
    ffi::Result<ffi::AnyBuffer> ipiv_out,
    int64_t nq, int64_t n, int64_t g, int64_t ipiv_len,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<SlateCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "scalapack.getrf: ctx_handle is null");
    }
    std::lock_guard<std::mutex> lock(
        lorrax_ffi::slate::host_collective_mutex());
    lorrax_ffi::scalapack::MklThreadScope mkl_scope(
        lorrax_ffi::scalapack::scalapack_mkl_threads());
    const auto dtype = A.element_type();
    if (LU_out->element_type() != dtype) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "scalapack.getrf: A and LU must share dtype");
    }
    if (ipiv_out->element_type() != ffi::DataType::S32) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "scalapack.getrf: ipiv output must be int32");
    }
    // PROVENANCE guard — same rationale as solve_lu_ffi.cc.
    {
        const char* nm = (dtype == ffi::DataType::F64) ? "pdgetrf_"
                                                       : "pzgetrf_";
        const std::string refusal =
            lorrax_ffi::scalapack::scalapack_slate_api_refusal("getrf", nm);
        if (!refusal.empty()) {
            return ffi::Error(ffi::ErrorCode::kUnimplemented, refusal);
        }
    }
    switch (dtype) {
        case ffi::DataType::F64:
            return GetrfImpl<double>(
                nq, n, g, ipiv_len, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<double*>(LU_out->untyped_data()),
                static_cast<int32_t*>(ipiv_out->untyped_data()));
        case ffi::DataType::C128: {
            using C128 = std::complex<double>;
            return GetrfImpl<C128>(
                nq, n, g, ipiv_len, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<C128*>(LU_out->untyped_data()),
                static_cast<int32_t*>(ipiv_out->untyped_data()));
        }
        default: {
            std::ostringstream os;
            os << "scalapack.getrf: unsupported dtype " << (int)dtype
               << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

// ---------------------------------------------------------------------------
// getrs: back-solve against stored factors, per r-chunk.
// ---------------------------------------------------------------------------

template <typename T>
static ffi::Error GetrsImpl(
    int64_t nq, int64_t n, int64_t nrhs, int64_t g, int64_t nb_b,
    int64_t ipiv_len,
    SlateCtx* ctx,
    const T* LU_in, const int32_t* ipiv_in, const T* B_in, T* X_out)
{
    int ictxt = 0, lld_a = 0, a_loc_cols = 0;
    int desca[9];
    const std::string err = GridPrologue(ctx, n, g, &ictxt, &lld_a,
                                         &a_loc_cols, desca);
    if (!err.empty())
        return ffi::Error(ffi::ErrorCode::kInternal, err);

    const int i_n = static_cast<int>(n);
    const int i_nrhs = static_cast<int>(nrhs);
    const int i_g = static_cast<int>(g);
    const int i_nb_b = static_cast<int>(nb_b);
    const int izero = 0, ione = 1;
    int nprow = 0, npcol = 0, myrow = 0, mycol = 0;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
    const int b_loc_cols = numroc_(&i_nrhs, &i_nb_b, &mycol, &izero, &npcol);

    int descb[9], info = 0;
    descinit_(descb, &i_n, &i_nrhs, &i_g, &i_nb_b, &izero, &izero, &ictxt,
              &lld_a, &info);
    if (info != 0) {
        std::ostringstream os;
        os << "scalapack.getrs: descinit(B) info=" << info;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }

    const int64_t A_slice = static_cast<int64_t>(lld_a) * a_loc_cols;
    const int64_t B_slice = static_cast<int64_t>(lld_a) * b_loc_cols;
    const int64_t want = static_cast<int64_t>(lld_a) + static_cast<int64_t>(g);
    if (want > ipiv_len) {
        std::ostringstream os;
        os << "scalapack.getrs: ipiv_len=" << ipiv_len
           << " smaller than LOCr+MB=" << want;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    if (X_out != static_cast<const T*>(B_in)) {
        std::memcpy(X_out, B_in, nq * B_slice * sizeof(T));
    }

    std::vector<int> ipiv(static_cast<size_t>(want));
    for (int64_t iq = 0; iq < nq; ++iq) {
        const T* LU_slice_ptr = LU_in + iq * A_slice;
        T* X_slice_ptr = X_out + iq * B_slice;
        const int32_t* piv_row = ipiv_in + iq * ipiv_len;
        for (int64_t i = 0; i < want; ++i)
            ipiv[i] = static_cast<int>(piv_row[i]);
        info = 0;
        Getrs<T>::call("N", &i_n, &i_nrhs, LU_slice_ptr, &ione, &ione, desca,
                       ipiv.data(), X_slice_ptr, &ione, &ione, descb, &info);
        if (info != 0) {
            std::ostringstream os;
            os << "scalapack pXgetrs (q=" << iq << ") info=" << info;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
    }
    return ffi::Error::Success();
}

static ffi::Error GetrsDispatch(
    ffi::AnyBuffer LU,
    ffi::AnyBuffer ipiv,
    ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> X_out,
    int64_t nq, int64_t n, int64_t nrhs,
    int64_t g, int64_t nb_b, int64_t ipiv_len,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<SlateCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "scalapack.getrs: ctx_handle is null");
    }
    std::lock_guard<std::mutex> lock(
        lorrax_ffi::slate::host_collective_mutex());
    lorrax_ffi::scalapack::MklThreadScope mkl_scope(
        lorrax_ffi::scalapack::scalapack_mkl_threads());
    const auto dtype = LU.element_type();
    if (B.element_type() != dtype || X_out->element_type() != dtype) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "scalapack.getrs: LU, B, X must share dtype");
    }
    if (ipiv.element_type() != ffi::DataType::S32) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "scalapack.getrs: ipiv must be int32");
    }
    {
        const char* nm = (dtype == ffi::DataType::F64) ? "pdgetrs_"
                                                       : "pzgetrs_";
        const std::string refusal =
            lorrax_ffi::scalapack::scalapack_slate_api_refusal("getrs", nm);
        if (!refusal.empty()) {
            return ffi::Error(ffi::ErrorCode::kUnimplemented, refusal);
        }
    }
    switch (dtype) {
        case ffi::DataType::F64:
            return GetrsImpl<double>(
                nq, n, nrhs, g, nb_b, ipiv_len, ctx,
                static_cast<const double*>(LU.untyped_data()),
                static_cast<const int32_t*>(ipiv.untyped_data()),
                static_cast<const double*>(B.untyped_data()),
                static_cast<double*>(X_out->untyped_data()));
        case ffi::DataType::C128: {
            using C128 = std::complex<double>;
            return GetrsImpl<C128>(
                nq, n, nrhs, g, nb_b, ipiv_len, ctx,
                static_cast<const C128*>(LU.untyped_data()),
                static_cast<const int32_t*>(ipiv.untyped_data()),
                static_cast<const C128*>(B.untyped_data()),
                static_cast<C128*>(X_out->untyped_data()));
        }
        default: {
            std::ostringstream os;
            os << "scalapack.getrs: unsupported dtype " << (int)dtype
               << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::scalapack_getrf_getrs

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScalapackBatchedGetrfHostFfi,
    lorrax_ffi::scalapack_getrf_getrs::GetrfDispatch,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()      // A (donated; LU factored in place)
        .Ret<xla::ffi::AnyBuffer>()      // LU (aliased to A)
        .Ret<xla::ffi::AnyBuffer>()      // ipiv (int32, per-rank rows)
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("g")              // square block: N / max(Px, Py)
        .Attr<int64_t>("ipiv_len")       // per-rank ipiv extent: LOCr + MB
        .Attr<int64_t>("ctx_handle"));   // SlateCtx (shared with ffi.slate)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScalapackBatchedGetrsHostFfi,
    lorrax_ffi::scalapack_getrf_getrs::GetrsDispatch,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()      // LU (as getrf returned it)
        .Arg<xla::ffi::AnyBuffer>()      // ipiv (int32, per-rank rows)
        .Arg<xla::ffi::AnyBuffer>()      // B (donated — solved in place)
        .Ret<xla::ffi::AnyBuffer>()      // X (aliased to B)
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("g")
        .Attr<int64_t>("nb_b")           // B col block: NRHS/Py (or NRHS)
        .Attr<int64_t>("ipiv_len")
        .Attr<int64_t>("ctx_handle"));
