// Batched PBLAS pXgemm on the host process grid. Python locally transposes
// each face tile into column-major storage; BLACS/PBLAS owns communication.

#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>

#include "xla/ffi/api/ffi.h"

#include "../slate/ctx.h"
#include "blacs_grid.h"

namespace lorrax_ffi::scalapack_gemm {

namespace ffi = ::xla::ffi;
using lorrax_ffi::slate::SlateCtx;

template <typename T> struct Gemm;
template <> struct Gemm<double> {
    static void call(
        const char* ta, const char* tb, const int* m, const int* n,
        const int* k, const double* alpha, const double* a, const int* ia,
        const int* ja, const int* desca, const double* b, const int* ib,
        const int* jb, const int* descb, const double* beta, double* c,
        const int* ic, const int* jc, const int* descc) {
        pdgemm_(ta, tb, m, n, k, alpha, a, ia, ja, desca, b, ib, jb,
                descb, beta, c, ic, jc, descc);
    }
};
template <> struct Gemm<std::complex<double>> {
    static void call(
        const char* ta, const char* tb, const int* m, const int* n,
        const int* k, const std::complex<double>* alpha,
        const std::complex<double>* a, const int* ia, const int* ja,
        const int* desca, const std::complex<double>* b, const int* ib,
        const int* jb, const int* descb,
        const std::complex<double>* beta, std::complex<double>* c,
        const int* ic, const int* jc, const int* descc) {
        pzgemm_(ta, tb, m, n, k, alpha, a, ia, ja, desca, b, ib, jb,
                descb, beta, c, ic, jc, descc);
    }
};

static ffi::Error invalid(const std::string& message) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "scalapack.gemm: " + message);
}

static bool valid_extent(int64_t value) {
    return value > 0 && value <= std::numeric_limits<int>::max();
}

static bool dimensions_equal(const ffi::AnyBuffer& buffer,
                             int64_t d0, int64_t d1, int64_t d2) {
    const auto dims = buffer.dimensions();
    return dims.size() == 3 && dims[0] == d0 && dims[1] == d1
           && dims[2] == d2;
}

template <typename T>
static ffi::Error GemmImpl(
    int64_t nq, int64_t m, int64_t n, int64_t k,
    int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    int64_t mb_c, int64_t nb_c, char transa, char transb,
    T alpha, T beta, SlateCtx* ctx,
    const T* a, const T* b, const T* c_in, T* c_out) {
    const int ictxt = lorrax_ffi::scalapack::blacs_ctxt_for(ctx);
    int nprow = 0, npcol = 0, myrow = 0, mycol = 0;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
    if (nprow != ctx->p || npcol != ctx->q) {
        std::ostringstream os;
        os << "BLACS grid " << nprow << "x" << npcol << " != ctx grid "
           << ctx->p << "x" << ctx->q;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    const int izero = 0, ione = 1;
    const int i_ar = static_cast<int>(a_rows);
    const int i_ac = static_cast<int>(a_cols);
    const int i_br = static_cast<int>(b_rows);
    const int i_bc = static_cast<int>(b_cols);
    const int i_m = static_cast<int>(m);
    const int i_n = static_cast<int>(n);
    const int i_k = static_cast<int>(k);
    const int i_mba = static_cast<int>(mb_a);
    const int i_nba = static_cast<int>(nb_a);
    const int i_mbb = static_cast<int>(mb_b);
    const int i_nbb = static_cast<int>(nb_b);
    const int i_mbc = static_cast<int>(mb_c);
    const int i_nbc = static_cast<int>(nb_c);

    const int a_lr = numroc_(&i_ar, &i_mba, &myrow, &izero, &nprow);
    const int a_lc = numroc_(&i_ac, &i_nba, &mycol, &izero, &npcol);
    const int b_lr = numroc_(&i_br, &i_mbb, &myrow, &izero, &nprow);
    const int b_lc = numroc_(&i_bc, &i_nbb, &mycol, &izero, &npcol);
    const int c_lr = numroc_(&i_m, &i_mbc, &myrow, &izero, &nprow);
    const int c_lc = numroc_(&i_n, &i_nbc, &mycol, &izero, &npcol);

    int desca[9], descb[9], descc[9], info = 0;
    descinit_(desca, &i_ar, &i_ac, &i_mba, &i_nba, &izero, &izero,
              &ictxt, &a_lr, &info);
    if (info != 0) return invalid("descinit(A) info=" + std::to_string(info));
    descinit_(descb, &i_br, &i_bc, &i_mbb, &i_nbb, &izero, &izero,
              &ictxt, &b_lr, &info);
    if (info != 0) return invalid("descinit(B) info=" + std::to_string(info));
    descinit_(descc, &i_m, &i_n, &i_mbc, &i_nbc, &izero, &izero,
              &ictxt, &c_lr, &info);
    if (info != 0) return invalid("descinit(C) info=" + std::to_string(info));

    const int64_t a_slice = static_cast<int64_t>(a_lr) * a_lc;
    const int64_t b_slice = static_cast<int64_t>(b_lr) * b_lc;
    const int64_t c_slice = static_cast<int64_t>(c_lr) * c_lc;
    if (c_out != c_in) {
        std::memcpy(c_out, c_in,
                    static_cast<size_t>(nq * c_slice) * sizeof(T));
    }
    for (int64_t iq = 0; iq < nq; ++iq) {
        Gemm<T>::call(
            &transa, &transb, &i_m, &i_n, &i_k, &alpha,
            a + iq * a_slice, &ione, &ione, desca,
            b + iq * b_slice, &ione, &ione, descb, &beta,
            c_out + iq * c_slice, &ione, &ione, descc);
    }
    return ffi::Error::Success();
}

static ffi::Error GemmDispatch(
    ffi::AnyBuffer A, ffi::AnyBuffer B, ffi::AnyBuffer C_in,
    ffi::Result<ffi::AnyBuffer> C_out,
    int64_t nq, int64_t m, int64_t n, int64_t k,
    int64_t a_rows, int64_t a_cols, int64_t b_rows, int64_t b_cols,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    int64_t mb_c, int64_t nb_c, int64_t transa_code, int64_t transb_code,
    double alpha_re, double alpha_im, double beta_re, double beta_im,
    int64_t ctx_handle) {
    auto* ctx = reinterpret_cast<SlateCtx*>(ctx_handle);
    if (ctx == nullptr) return invalid("ctx_handle is null");
    std::lock_guard<std::mutex> lock(
        lorrax_ffi::slate::host_collective_mutex());
    lorrax_ffi::scalapack::MklThreadScope mkl_scope(
        lorrax_ffi::scalapack::scalapack_mkl_threads());

    for (int64_t value : {nq, m, n, k, a_rows, a_cols, b_rows, b_cols,
                          mb_a, nb_a, mb_b, nb_b, mb_c, nb_c}) {
        if (!valid_extent(value)) return invalid("invalid or oversized extent");
    }
    if (ctx->p != ctx->q) return invalid("requires a square process grid");
    if (a_rows % ctx->p || a_cols % ctx->q || b_rows % ctx->p
        || b_cols % ctx->q || m % ctx->p || n % ctx->q
        || mb_a != a_rows / ctx->p || nb_a != a_cols / ctx->q
        || mb_b != b_rows / ctx->p || nb_b != b_cols / ctx->q
        || mb_c != m / ctx->p || nb_c != n / ctx->q) {
        return invalid("descriptor blocks do not match one face per rank");
    }
    const char ops[] = {'N', 'T', 'C'};
    if (transa_code < 0 || transa_code > 2 || transb_code < 0
        || transb_code > 2) {
        return invalid("transa/transb code must be 0, 1, or 2");
    }
    const char transa = ops[transa_code];
    const char transb = ops[transb_code];
    const int64_t want_ar = transa == 'N' ? m : k;
    const int64_t want_ac = transa == 'N' ? k : m;
    const int64_t want_br = transb == 'N' ? k : n;
    const int64_t want_bc = transb == 'N' ? n : k;
    if (a_rows != want_ar || a_cols != want_ac || b_rows != want_br
        || b_cols != want_bc) {
        return invalid("physical operand shapes disagree with op(A)@op(B)");
    }

    const int64_t a_lr = a_rows / ctx->p, a_lc = a_cols / ctx->q;
    const int64_t b_lr = b_rows / ctx->p, b_lc = b_cols / ctx->q;
    const int64_t c_lr = m / ctx->p, c_lc = n / ctx->q;
    if (!dimensions_equal(A, nq, a_lc, a_lr)
        || !dimensions_equal(B, nq, b_lc, b_lr)
        || !dimensions_equal(C_in, nq, c_lc, c_lr)
        || !dimensions_equal(*C_out, nq, c_lc, c_lr)) {
        return invalid("local buffer dimensions disagree with descriptors");
    }
    const auto dtype = A.element_type();
    if (B.element_type() != dtype || C_in.element_type() != dtype
        || C_out->element_type() != dtype) {
        return invalid("A, B, C must share dtype");
    }
    if (dtype != ffi::DataType::F64 && dtype != ffi::DataType::C128) {
        return invalid("unsupported dtype (expected float64/complex128)");
    }
    const char* symbol = dtype == ffi::DataType::F64 ? "pdgemm_" : "pzgemm_";
    const std::string refusal =
        lorrax_ffi::scalapack::scalapack_slate_api_refusal("gemm", symbol);
    if (!refusal.empty()) {
        return ffi::Error(ffi::ErrorCode::kUnimplemented, refusal);
    }

    switch (dtype) {
        case ffi::DataType::F64:
            if (alpha_im != 0 || beta_im != 0)
                return invalid("complex alpha/beta with float64 operands");
            return GemmImpl<double>(
                nq, m, n, k, a_rows, a_cols, b_rows, b_cols,
                mb_a, nb_a, mb_b, nb_b, mb_c, nb_c, transa, transb,
                alpha_re, beta_re, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<const double*>(B.untyped_data()),
                static_cast<const double*>(C_in.untyped_data()),
                static_cast<double*>(C_out->untyped_data()));
        case ffi::DataType::C128: {
            using C128 = std::complex<double>;
            return GemmImpl<C128>(
                nq, m, n, k, a_rows, a_cols, b_rows, b_cols,
                mb_a, nb_a, mb_b, nb_b, mb_c, nb_c, transa, transb,
                C128(alpha_re, alpha_im), C128(beta_re, beta_im), ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<const C128*>(B.untyped_data()),
                static_cast<const C128*>(C_in.untyped_data()),
                static_cast<C128*>(C_out->untyped_data()));
        }
        default:
            return invalid("unsupported dtype (expected float64/complex128)");
    }
}

}  // namespace lorrax_ffi::scalapack_gemm

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScalapackBatchedGemmHostFfi,
    lorrax_ffi::scalapack_gemm::GemmDispatch,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nq")
        .Attr<int64_t>("m")
        .Attr<int64_t>("n")
        .Attr<int64_t>("k")
        .Attr<int64_t>("a_rows")
        .Attr<int64_t>("a_cols")
        .Attr<int64_t>("b_rows")
        .Attr<int64_t>("b_cols")
        .Attr<int64_t>("mb_a")
        .Attr<int64_t>("nb_a")
        .Attr<int64_t>("mb_b")
        .Attr<int64_t>("nb_b")
        .Attr<int64_t>("mb_c")
        .Attr<int64_t>("nb_c")
        .Attr<int64_t>("transa")
        .Attr<int64_t>("transb")
        .Attr<double>("alpha_re")
        .Attr<double>("alpha_im")
        .Attr<double>("beta_re")
        .Attr<double>("beta_im")
        .Attr<int64_t>("ctx_handle"));
