// host_ffi.cc — host-platform (JAX CPU backend) variants of the five SLATE
// FFI handlers: potrf, trsm, eigh, batched_potrf, batched_trsm.
//
// These are registered by ffi_loader.py under platform="cpu" with the SAME
// target names as the CUDA handlers (platform="CUDA"), mirroring how jaxlib
// registers its lapack kernels for cpu and cusolver kernels for CUDA: the
// jax.ffi.ffi_call sites in ffi/slate/*.py resolve the handler by the
// lowering platform and need no changes.  Compiled ONLY into the CUDA-free
// liblorrax_ffi_host.so (see common/cpp/host/CMakeLists.txt) — this TU must
// not include any CUDA header (ffi_helpers.h pulls cuda_runtime.h; the
// handlers here use plain sstream error formatting instead).
//
// Differences from the device TUs ({potrf,trsm,eigh,batched_*}_ffi.cc),
// which stay the single source of truth for the CUDA path:
//   - Ffi::Bind() has no Ctx<PlatformStream<cudaStream_t>>: XLA's CPU
//     runtime hands the handler host buffers on the calling thread, so
//     there is no stream to synchronize and no D2D staging — input→output
//     staging is a std::memcpy.
//   - Matrices wrap the host buffers via fromScaLAPACK() instead of
//     fromDevices().  Both interpret the local buffer as 2-D block-cyclic
//     col-major tiles on a GridOrder::Col process grid, so the layout
//     contract — Python-side local transpose + the MPI rank remap in
//     context.cc — is IDENTICAL and the tile-layout validation in
//     context.py applies unchanged.
//   - {Option::Target, Target::HostTask} instead of Target::Devices.
//   - No blas::get_device_count() == 1 precondition (meaningless on CPU).
//
// The eigh writeback keeps the tileGetForReading + explicit copy-out of the
// 2026-07-10 device fix: heev's back-transform (redistribute → unmtr_hb2st
// → unmtr_he2hb) makes no guarantee about WHERE the valid Z tiles live, so
// we ask the MOSI layer for a host-valid tile and copy it out, rather than
// assuming heev wrote through a wrapped user buffer.
//
// Concurrency note (same property as the device handlers): each handler
// issues MPI collectives on the context's dup'd comm, so concurrent
// invocations on the same comm would be rank-order-unsafe.  Call sites are
// data-dependent chains (one collective op in flight at a time).

#include <complex>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <type_traits>
#include <vector>

#include <blas.hh>
#include <slate/slate.hh>

#include "xla/ffi/api/ffi.h"

#include "ctx.h"

namespace lorrax_ffi::slate_host {

namespace ffi = ::xla::ffi;
namespace sl  = ::slate;

// Map a dtype to its real-part type (for eigenvalues).
template <typename T> struct RealOf              { using type = T; };
template <> struct RealOf<std::complex<float>>   { using type = float; };
template <> struct RealOf<std::complex<double>>  { using type = double; };

static inline sl::Options host_opts() {
    return {{sl::Option::Target, sl::Target::HostTask}};
}

static inline ffi::Error null_ctx_error(const char* what) {
    std::ostringstream os;
    os << what << ": ctx_handle is null (context not initialized on this "
       << "process?)";
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
}

// ---------------------------------------------------------------------------
//  potrf — distributed Cholesky, host execution.  Contract identical to
//  potrf_ffi.cc: copy input shard into the output buffer, factor in place,
//  L returned as SLATE col-major tiles (Python untransposes).
// ---------------------------------------------------------------------------
template <typename T>
static ffi::Error PotrfImpl(
    int64_t n, int64_t nb,
    lorrax_ffi::slate::SlateCtx* ctx,
    const T* A_in,
    T* L_out)
{
    const int p = ctx->p;
    const int q = ctx->q;
    const int64_t lld = (n + p - 1) / p;
    const int64_t local_rows = (n + p - 1) / p;
    const int64_t local_cols = (n + q - 1) / q;

    std::memcpy(L_out, A_in, local_rows * local_cols * sizeof(T));

    sl::HermitianMatrix<T> L = sl::HermitianMatrix<T>::fromScaLAPACK(
        sl::Uplo::Lower, n, L_out, lld, nb, p, q, ctx->comm);

    try {
        int64_t info = sl::potrf(L, host_opts());
        if (info != 0) {
            std::ostringstream os;
            os << "slate::potrf (host) returned info=" << info
               << " (matrix not positive-definite at that leading minor)";
            return ffi::Error(ffi::ErrorCode::kFailedPrecondition, os.str());
        }
    } catch (const std::exception& ex) {
        std::ostringstream os;
        os << "slate::potrf (host) threw: " << ex.what();
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    return ffi::Error::Success();
}

static ffi::Error PotrfDispatch(
    ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> L_out,
    int64_t n, int64_t nb,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<lorrax_ffi::slate::SlateCtx*>(ctx_handle);
    if (ctx == nullptr) return null_ctx_error("slate.potrf(host)");

    const auto dtype = A.element_type();
    if (L_out->element_type() != dtype) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "slate.potrf(host): L output dtype must match A input dtype");
    }

    switch (dtype) {
        case ffi::DataType::F64:
            return PotrfImpl<double>(
                n, nb, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<double*>(L_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return PotrfImpl<C128>(
                n, nb, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<C128*>(L_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "slate.potrf(host): unsupported dtype "
               << static_cast<int>(dtype) << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

// ---------------------------------------------------------------------------
//  trsm — distributed triangular solve, host execution.  Contract identical
//  to trsm_ffi.cc, including the per-dimension X tile sizes (solve dim
//  conforms with A's nb; free dim = one tile per rank on a multi-proc axis).
// ---------------------------------------------------------------------------
template <typename T>
static ffi::Error TrsmImpl(
    int64_t n, int64_t m, int64_t nb,
    int side_int, int uplo_int, int op_int, int diag_int,
    double alpha_re, double alpha_im,
    lorrax_ffi::slate::SlateCtx* ctx,
    const T* A_in,
    const T* B_in,
    T* X_out)
{
    const int p = ctx->p;
    const int q = ctx->q;

    sl::Side side = (side_int == 1) ? sl::Side::Right : sl::Side::Left;
    sl::Uplo uplo = (uplo_int == 1) ? sl::Uplo::Upper : sl::Uplo::Lower;
    sl::Diag diag = (diag_int == 1) ? sl::Diag::Unit  : sl::Diag::NonUnit;
    sl::Op   op   = (op_int == 2)   ? sl::Op::ConjTrans
                  : (op_int == 1)   ? sl::Op::Trans
                  :                   sl::Op::NoTrans;

    const int64_t lld_A = (n + p - 1) / p;
    const int64_t X_rows = (side == sl::Side::Left) ? n : m;
    const int64_t X_cols = (side == sl::Side::Left) ? m : n;
    const int64_t lld_B = (X_rows + p - 1) / p;
    const int64_t local_rows = (X_rows + p - 1) / p;
    const int64_t local_cols = (X_cols + q - 1) / q;

    std::memcpy(X_out, B_in, local_rows * local_cols * sizeof(T));

    sl::TriangularMatrix<T> A = sl::TriangularMatrix<T>::fromScaLAPACK(
        uplo, diag, n, const_cast<T*>(A_in), lld_A, nb, p, q, ctx->comm);

    // X tile sizes: same invariant as the device handler — the solve
    // dimension conforms with A's nb; the free dimension needs one tile
    // per rank on a multi-proc axis so SLATE's block-cyclic assignment
    // coincides with JAX's contiguous block shards.
    int64_t mb_X, nb_X;
    if (side == sl::Side::Left) {           // X is n × m
        mb_X = nb;
        nb_X = (q == 1) ? nb : X_cols / q;
    } else {                                // X is m × n
        mb_X = (p == 1) ? nb : X_rows / p;
        nb_X = nb;
    }
    sl::Matrix<T> X = sl::Matrix<T>::fromScaLAPACK(
        X_rows, X_cols, X_out, lld_B, mb_X, nb_X, p, q, ctx->comm);

    auto Aop = (op == sl::Op::Trans)     ? transpose(A)
             : (op == sl::Op::ConjTrans) ? conj_transpose(A)
             :                              A;

    T alpha;
    if constexpr (std::is_same_v<T, double>) {
        alpha = alpha_re;
    } else {
        alpha = T(alpha_re, alpha_im);
    }

    try {
        sl::trsm(side, alpha, Aop, X, host_opts());
    } catch (const std::exception& ex) {
        std::ostringstream os;
        os << "slate::trsm (host) threw: " << ex.what();
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    return ffi::Error::Success();
}

static ffi::Error TrsmDispatch(
    ffi::AnyBuffer A,
    ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> X_out,
    int64_t n, int64_t m, int64_t nb,
    int64_t side, int64_t uplo, int64_t op, int64_t diag,
    double alpha_re, double alpha_im,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<lorrax_ffi::slate::SlateCtx*>(ctx_handle);
    if (ctx == nullptr) return null_ctx_error("slate.trsm(host)");

    const auto dtype = A.element_type();
    if (B.element_type() != dtype || X_out->element_type() != dtype) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "slate.trsm(host): A, B, X must all share the same dtype");
    }

    switch (dtype) {
        case ffi::DataType::F64:
            return TrsmImpl<double>(
                n, m, nb, (int)side, (int)uplo, (int)op, (int)diag,
                alpha_re, alpha_im, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<const double*>(B.untyped_data()),
                static_cast<double*>(X_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return TrsmImpl<C128>(
                n, m, nb, (int)side, (int)uplo, (int)op, (int)diag,
                alpha_re, alpha_im, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<const C128*>(B.untyped_data()),
                static_cast<C128*>(X_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "slate.trsm(host): unsupported dtype "
               << static_cast<int>(dtype) << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

// ---------------------------------------------------------------------------
//  eigh — distributed Hermitian eigendecomposition, host execution.
//  Contract identical to eigh_ffi.cc: A copied into the Q buffer (heev
//  destroys its input; XLA input buffers must not be mutated), Z kept in
//  SLATE-internal storage and copied out tile-by-tile after heev.
// ---------------------------------------------------------------------------
template <typename T>
static ffi::Error EighImpl(
    int64_t n, int64_t nb,
    bool compute_evecs,
    lorrax_ffi::slate::SlateCtx* ctx,
    const T* A_in,
    typename RealOf<T>::type* W_out,
    T* Q_out)
{
    using real_t = typename RealOf<T>::type;

    const int p = ctx->p;
    const int q = ctx->q;
    const int64_t lld_A = (n + p - 1) / p;
    const int64_t lld_Q = lld_A;
    const int64_t local_rows = (n + p - 1) / p;
    const int64_t local_cols = (n + q - 1) / q;

    std::memcpy(Q_out, A_in, local_rows * local_cols * sizeof(T));

    sl::HermitianMatrix<T> A = sl::HermitianMatrix<T>::fromScaLAPACK(
        sl::Uplo::Lower, n, Q_out, lld_A, nb, p, q, ctx->comm);

    std::vector<real_t> Lambda(n);

    try {
        if (compute_evecs) {
            // SLATE-managed Z storage + explicit copy-out, exactly like
            // the device handler: heev's back-transform gives no
            // guarantee about which tile instance is valid, so ask the
            // MOSI layer (tileGetForReading, host overload) before
            // reading each local tile.
            sl::Matrix<T> Z(n, n, nb, p, q, ctx->comm);
            Z.insertLocalTiles();
            sl::heev(A, Lambda, Z, host_opts());

            for (int64_t ti = 0; ti < Z.mt(); ++ti) {
                for (int64_t tj = 0; tj < Z.nt(); ++tj) {
                    if (!Z.tileIsLocal(ti, tj)) continue;
                    Z.tileGetForReading(ti, tj, sl::LayoutConvert::ColMajor);
                    auto tile = Z(ti, tj);
                    const T* src = tile.data();
                    const int64_t mb_t = tile.mb();
                    const int64_t nb_t = tile.nb();
                    const int64_t stride = tile.stride();
                    // Destination offset within Q_out: same mapping the
                    // block-cyclic wrap (fromScaLAPACK) uses.
                    const int64_t ii = ti * nb, jj = tj * nb;
                    const int64_t ii_local = (ii / (nb * p)) * nb + (ii % nb);
                    const int64_t jj_local = (jj / (nb * q)) * nb + (jj % nb);
                    T* dst = Q_out + ii_local + jj_local * lld_Q;
                    for (int64_t c = 0; c < nb_t; ++c) {
                        std::memcpy(dst + c * lld_Q, src + c * stride,
                                    mb_t * sizeof(T));
                    }
                }
            }
        } else {
            sl::heev(A, Lambda, host_opts());
        }
    } catch (const std::exception& ex) {
        std::ostringstream os;
        os << "slate::heev (host) threw: " << ex.what();
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    std::memcpy(W_out, Lambda.data(), n * sizeof(real_t));

    return ffi::Error::Success();
}

static ffi::Error EighDispatch(
    ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> W_out,
    ffi::Result<ffi::AnyBuffer> Q_out,
    int64_t n, int64_t nb,
    int64_t ctx_handle, bool compute_evecs)
{
    auto* ctx = reinterpret_cast<lorrax_ffi::slate::SlateCtx*>(ctx_handle);
    if (ctx == nullptr) return null_ctx_error("slate.eigh(host)");

    const auto dtype = A.element_type();
    if (Q_out->element_type() != dtype) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "slate.eigh(host): Q output and A input must have matching dtype");
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
            "slate.eigh(host): W output dtype must be the real part of A's dtype");
    }

    switch (dtype) {
        case ffi::DataType::F64:
            return EighImpl<double>(
                n, nb, compute_evecs, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<double*>(W_out->untyped_data()),
                static_cast<double*>(Q_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return EighImpl<C128>(
                n, nb, compute_evecs, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<double*>(W_out->untyped_data()),
                static_cast<C128*>(Q_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "slate.eigh(host): unsupported dtype "
               << static_cast<int>(dtype) << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

// ---------------------------------------------------------------------------
//  batched potrf — for-loop over the local batch on the sub-row comm
//  (p=1, q=Py), host execution.  Contract identical to batched_potrf_ffi.cc.
// ---------------------------------------------------------------------------
template <typename T>
static ffi::Error BatchedPotrfImpl(
    int64_t nbatch_local, int64_t n, int64_t nb,
    lorrax_ffi::slate::SlateCtx* ctx,
    const T* A_in,
    T* L_out)
{
    const int p = ctx->p;           // 1 by construction
    const int q = ctx->q;           // Py
    if (p != 1) {
        return ffi::Error(
            ffi::ErrorCode::kFailedPrecondition,
            "slate.batched_potrf(host): expected sub-row ctx with p=1");
    }
    const int64_t lld = n;
    const int64_t slice_elems = n * ((n + q - 1) / q);

    std::memcpy(L_out, A_in, nbatch_local * slice_elems * sizeof(T));

    for (int64_t b = 0; b < nbatch_local; ++b) {
        T* slice_ptr = L_out + b * slice_elems;

        sl::HermitianMatrix<T> L = sl::HermitianMatrix<T>::fromScaLAPACK(
            sl::Uplo::Lower, n, slice_ptr, lld, nb, p, q, ctx->comm);

        try {
            int64_t info = sl::potrf(L, host_opts());
            if (info != 0) {
                std::ostringstream os;
                os << "slate::potrf (host, batch " << b << ") returned info="
                   << info << " (matrix not positive-definite at that "
                   << "leading minor)";
                return ffi::Error(ffi::ErrorCode::kFailedPrecondition, os.str());
            }
        } catch (const std::exception& ex) {
            std::ostringstream os;
            os << "slate::potrf (host, batch " << b << ") threw: " << ex.what();
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
    }

    return ffi::Error::Success();
}

static ffi::Error BatchedPotrfDispatch(
    ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> L_out,
    int64_t nbatch_local, int64_t n, int64_t nb,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<lorrax_ffi::slate::SlateCtx*>(ctx_handle);
    if (ctx == nullptr) return null_ctx_error("slate.batched_potrf(host)");

    const auto dtype = A.element_type();
    if (L_out->element_type() != dtype) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "slate.batched_potrf(host): L output dtype must match A input dtype");
    }

    switch (dtype) {
        case ffi::DataType::F64:
            return BatchedPotrfImpl<double>(
                nbatch_local, n, nb, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<double*>(L_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return BatchedPotrfImpl<C128>(
                nbatch_local, n, nb, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<C128*>(L_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "slate.batched_potrf(host): unsupported dtype "
               << static_cast<int>(dtype) << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

// ---------------------------------------------------------------------------
//  batched trsm — for-loop over the local batch on the sub-row comm,
//  host execution.  Contract identical to batched_trsm_ffi.cc.
// ---------------------------------------------------------------------------
template <typename T>
static ffi::Error BatchedTrsmImpl(
    int64_t nbatch_local, int64_t n, int64_t m, int64_t nb,
    int side_int, int uplo_int, int op_int, int diag_int,
    double alpha_re, double alpha_im,
    lorrax_ffi::slate::SlateCtx* ctx,
    const T* A_in,
    const T* B_in,
    T* X_out)
{
    const int p = ctx->p;       // 1 by construction
    const int q = ctx->q;       // Py
    if (p != 1) {
        return ffi::Error(
            ffi::ErrorCode::kFailedPrecondition,
            "slate.batched_trsm(host): expected sub-row ctx with p=1");
    }

    const int64_t lld_A = n;
    const int64_t A_slice_elems = n * ((n + q - 1) / q);

    const sl::Side side = (side_int == 1) ? sl::Side::Right : sl::Side::Left;
    const sl::Uplo uplo = (uplo_int == 1) ? sl::Uplo::Upper : sl::Uplo::Lower;
    const sl::Diag diag = (diag_int == 1) ? sl::Diag::Unit  : sl::Diag::NonUnit;
    const sl::Op   op   = (op_int == 2)   ? sl::Op::ConjTrans
                        : (op_int == 1)   ? sl::Op::Trans
                        :                   sl::Op::NoTrans;

    const int64_t B_rows = (side == sl::Side::Left) ? n : m;
    const int64_t B_cols = (side == sl::Side::Left) ? m : n;
    const int64_t lld_B  = B_rows;
    const int64_t B_slice_elems = B_rows * ((B_cols + q - 1) / q);

    std::memcpy(X_out, B_in, nbatch_local * B_slice_elems * sizeof(T));

    T alpha;
    if constexpr (std::is_same_v<T, double>) {
        alpha = alpha_re;
    } else {
        alpha = T(alpha_re, alpha_im);
    }

    for (int64_t b = 0; b < nbatch_local; ++b) {
        T* A_slice = const_cast<T*>(A_in) + b * A_slice_elems;
        T* X_slice = X_out + b * B_slice_elems;

        sl::TriangularMatrix<T> A = sl::TriangularMatrix<T>::fromScaLAPACK(
            uplo, diag, n, A_slice, lld_A, nb, p, q, ctx->comm);

        // Column tile: one per rank when q > 1 (same invariant as the
        // device handler; a square-nb X silently corrupted Nrhs != N).
        const int64_t nb_X = (q == 1) ? nb : B_cols / q;
        sl::Matrix<T> X = sl::Matrix<T>::fromScaLAPACK(
            B_rows, B_cols, X_slice, lld_B, nb, nb_X, p, q, ctx->comm);

        auto Aop = (op == sl::Op::Trans)     ? transpose(A)
                 : (op == sl::Op::ConjTrans) ? conj_transpose(A)
                 :                              A;

        try {
            sl::trsm(side, alpha, Aop, X, host_opts());
        } catch (const std::exception& ex) {
            std::ostringstream os;
            os << "slate::trsm (host, batch " << b << ") threw: " << ex.what();
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
    }

    return ffi::Error::Success();
}

static ffi::Error BatchedTrsmDispatch(
    ffi::AnyBuffer A,
    ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> X_out,
    int64_t nbatch_local, int64_t n, int64_t m, int64_t nb,
    int64_t side, int64_t uplo, int64_t op, int64_t diag,
    double alpha_re, double alpha_im,
    int64_t ctx_handle)
{
    auto* ctx = reinterpret_cast<lorrax_ffi::slate::SlateCtx*>(ctx_handle);
    if (ctx == nullptr) return null_ctx_error("slate.batched_trsm(host)");

    const auto dtype = A.element_type();
    if (B.element_type() != dtype || X_out->element_type() != dtype) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "slate.batched_trsm(host): A, B, X must all share the same dtype");
    }

    switch (dtype) {
        case ffi::DataType::F64:
            return BatchedTrsmImpl<double>(
                nbatch_local, n, m, nb,
                (int)side, (int)uplo, (int)op, (int)diag,
                alpha_re, alpha_im, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<const double*>(B.untyped_data()),
                static_cast<double*>(X_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return BatchedTrsmImpl<C128>(
                nbatch_local, n, m, nb,
                (int)side, (int)uplo, (int)op, (int)diag,
                alpha_re, alpha_im, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<const C128*>(B.untyped_data()),
                static_cast<C128*>(X_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "slate.batched_trsm(host): unsupported dtype "
               << static_cast<int>(dtype) << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::slate_host

// ---------------------------------------------------------------------------
//  FFI bindings — same attrs as the device handlers, no PlatformStream ctx.
// ---------------------------------------------------------------------------
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SlatePotrfHostFfi, lorrax_ffi::slate_host::PotrfDispatch,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()              // A (Hermitian input)
        .Ret<xla::ffi::AnyBuffer>()              // L (Cholesky factor, lower)
        .Attr<int64_t>("n")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_handle"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SlateTrsmHostFfi, lorrax_ffi::slate_host::TrsmDispatch,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()              // A (triangular)
        .Arg<xla::ffi::AnyBuffer>()              // B (right-hand side)
        .Ret<xla::ffi::AnyBuffer>()              // X (solution, same shape as B)
        .Attr<int64_t>("n")
        .Attr<int64_t>("m")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("side")    // 0 Left, 1 Right
        .Attr<int64_t>("uplo")    // 0 Lower, 1 Upper
        .Attr<int64_t>("op")      // 0 NoTrans, 1 Trans, 2 ConjTrans
        .Attr<int64_t>("diag")    // 0 NonUnit, 1 Unit
        .Attr<double>("alpha_re")
        .Attr<double>("alpha_im")
        .Attr<int64_t>("ctx_handle"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SlateEighHostFfi, lorrax_ffi::slate_host::EighDispatch,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()              // A (local shard)
        .Ret<xla::ffi::AnyBuffer>()              // W (n,) real, replicated
        .Ret<xla::ffi::AnyBuffer>()              // Q (local shard, same dtype as A)
        .Attr<int64_t>("n")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_handle")
        .Attr<bool>("compute_evecs"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SlateBatchedPotrfHostFfi, lorrax_ffi::slate_host::BatchedPotrfDispatch,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()                      // A
        .Ret<xla::ffi::AnyBuffer>()                      // L
        .Attr<int64_t>("nbatch_local")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_handle"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SlateBatchedTrsmHostFfi, lorrax_ffi::slate_host::BatchedTrsmDispatch,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()              // A (triangular)
        .Arg<xla::ffi::AnyBuffer>()              // B (RHS)
        .Ret<xla::ffi::AnyBuffer>()              // X
        .Attr<int64_t>("nbatch_local")
        .Attr<int64_t>("n")
        .Attr<int64_t>("m")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("side")
        .Attr<int64_t>("uplo")
        .Attr<int64_t>("op")
        .Attr<int64_t>("diag")
        .Attr<double>("alpha_re")
        .Attr<double>("alpha_im")
        .Attr<int64_t>("ctx_handle"));
