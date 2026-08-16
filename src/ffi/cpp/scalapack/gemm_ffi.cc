// gemm_ffi.cc — batched PBLAS p?gemm on the host 2-D process grid.
#include <complex>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

#include "xla/ffi/api/ffi.h"

#include "../slate/ctx.h"
#include "blacs_grid.h"

namespace lorrax_ffi::scalapack_gemm {
namespace ffi = ::xla::ffi;
using lorrax_ffi::slate::SlateCtx;

template <typename T> struct Gemm;
template <> struct Gemm<double> {
    static void call(const char* ta, const char* tb, const int* m,
                     const int* n, const int* k, const double* alpha,
                     const double* a, const int* ia, const int* ja,
                     const int* da, const double* b, const int* ib,
                     const int* jb, const int* db, const double* beta,
                     double* c, const int* ic, const int* jc, const int* dc)
    { pdgemm_(ta, tb, m, n, k, alpha, a, ia, ja, da,
              b, ib, jb, db, beta, c, ic, jc, dc); }
};
template <> struct Gemm<std::complex<double>> {
    using Z = std::complex<double>;
    static void call(const char* ta, const char* tb, const int* m,
                     const int* n, const int* k, const Z* alpha,
                     const Z* a, const int* ia, const int* ja, const int* da,
                     const Z* b, const int* ib, const int* jb, const int* db,
                     const Z* beta, Z* c, const int* ic, const int* jc,
                     const int* dc)
    { pzgemm_(ta, tb, m, n, k, alpha, a, ia, ja, da,
              b, ib, jb, db, beta, c, ic, jc, dc); }
};

template <typename T>
static ffi::Error Impl(
    int64_t nq, int64_t m, int64_t n, int64_t k,
    int64_t ar, int64_t ac, int64_t br, int64_t bc,
    int64_t mba, int64_t nba, int64_t mbb, int64_t nbb,
    int64_t mbc, int64_t nbc, int ta, int tb,
    T alpha, T beta, SlateCtx* ctx,
    const T* A, const T* B, const T* C, T* D)
{
    const int ictxt = lorrax_ffi::scalapack::blacs_ctxt_for(ctx);
    int nprow=0, npcol=0, myrow=0, mycol=0;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
    const int zero=0, one=1;
    auto init_desc = [&](int rows, int cols, int mb, int nb, int* d) {
        int lld = numroc_(&rows, &mb, &myrow, &zero, &nprow), info=0;
        descinit_(d, &rows, &cols, &mb, &nb, &zero, &zero,
                  &ictxt, &lld, &info);
        return info;
    };
    int da[9], db[9], dc[9];
    if (init_desc((int)ar,(int)ac,(int)mba,(int)nba,da) ||
        init_desc((int)br,(int)bc,(int)mbb,(int)nbb,db) ||
        init_desc((int)m, (int)n, (int)mbc,(int)nbc,dc))
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "scalapack.matmul: descinit failed");
    const int64_t asz=(ar/ctx->p)*(ac/ctx->q);
    const int64_t bsz=(br/ctx->p)*(bc/ctx->q);
    const int64_t csz=(m /ctx->p)*(n /ctx->q);
    if (D != C) std::memcpy(D, C, nq*csz*sizeof(T));
    const char* tas = ta == 0 ? "N" : (ta == 1 ? "T" : "C");
    const char* tbs = tb == 0 ? "N" : (tb == 1 ? "T" : "C");
    int im=(int)m, in=(int)n, ik=(int)k;
    for (int64_t q=0; q<nq; ++q)
        Gemm<T>::call(tas,tbs,&im,&in,&ik,&alpha,
                      A+q*asz,&one,&one,da, B+q*bsz,&one,&one,db,
                      &beta,D+q*csz,&one,&one,dc);
    return ffi::Error::Success();
}

static ffi::Error Dispatch(
    ffi::AnyBuffer A, ffi::AnyBuffer B, ffi::AnyBuffer C,
    ffi::Result<ffi::AnyBuffer> D,
    int64_t nq,int64_t m,int64_t n,int64_t k,
    int64_t ar,int64_t ac,int64_t br,int64_t bc,
    int64_t mba,int64_t nba,int64_t mbb,int64_t nbb,
    int64_t mbc,int64_t nbc,int64_t ta,int64_t tb,
    double are,double aim,double bre,double bim,int64_t handle)
{
    auto* ctx=reinterpret_cast<SlateCtx*>(handle);
    if (!ctx) return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                                "scalapack.matmul: null context");
    std::lock_guard<std::mutex> lock(lorrax_ffi::slate::host_collective_mutex());
    lorrax_ffi::scalapack::MklThreadScope scope(
        lorrax_ffi::scalapack::scalapack_mkl_threads());
    auto dt=A.element_type();
    if (B.element_type()!=dt || C.element_type()!=dt || D->element_type()!=dt)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "scalapack.matmul: dtypes disagree");
    const char* symbol = dt == ffi::DataType::F64 ? "pdgemm_" : "pzgemm_";
    std::string refusal =
        lorrax_ffi::scalapack::scalapack_slate_api_refusal("matmul", symbol);
    if (!refusal.empty())
        return ffi::Error(ffi::ErrorCode::kUnimplemented, refusal);
    if (dt==ffi::DataType::F64)
        return Impl<double>(nq,m,n,k,ar,ac,br,bc,mba,nba,mbb,nbb,mbc,nbc,
                            ta,tb,are,bre,ctx,
                            static_cast<const double*>(A.untyped_data()),
                            static_cast<const double*>(B.untyped_data()),
                            static_cast<const double*>(C.untyped_data()),
                            static_cast<double*>(D->untyped_data()));
    if (dt==ffi::DataType::C128) {
        using Z=std::complex<double>;
        return Impl<Z>(nq,m,n,k,ar,ac,br,bc,mba,nba,mbb,nbb,mbc,nbc,ta,tb,
                       Z(are,aim),Z(bre,bim),ctx,
                       static_cast<const Z*>(A.untyped_data()),
                       static_cast<const Z*>(B.untyped_data()),
                       static_cast<const Z*>(C.untyped_data()),
                       static_cast<Z*>(D->untyped_data()));
    }
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "scalapack.matmul supports F64/C128");
}
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScalapackBatchedGemmHostFfi, lorrax_ffi::scalapack_gemm::Dispatch,
    xla::ffi::Ffi::Bind()
      .Arg<xla::ffi::AnyBuffer>().Arg<xla::ffi::AnyBuffer>()
      .Arg<xla::ffi::AnyBuffer>().Ret<xla::ffi::AnyBuffer>()
      .Attr<int64_t>("nq").Attr<int64_t>("m").Attr<int64_t>("n").Attr<int64_t>("k")
      .Attr<int64_t>("a_rows").Attr<int64_t>("a_cols")
      .Attr<int64_t>("b_rows").Attr<int64_t>("b_cols")
      .Attr<int64_t>("mb_a").Attr<int64_t>("nb_a")
      .Attr<int64_t>("mb_b").Attr<int64_t>("nb_b")
      .Attr<int64_t>("mb_c").Attr<int64_t>("nb_c")
      .Attr<int64_t>("transa").Attr<int64_t>("transb")
      .Attr<double>("alpha_re").Attr<double>("alpha_im")
      .Attr<double>("beta_re").Attr<double>("beta_im")
      .Attr<int64_t>("ctx_handle"));
