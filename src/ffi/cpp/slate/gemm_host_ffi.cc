// gemm_host_ffi.cc — batched slate::multiply on the host 2-D process grid.
#include <complex>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>

#include <slate/slate.hh>
#include "xla/ffi/api/ffi.h"
#include "ctx.h"

namespace lorrax_ffi::slate_gemm_host {
namespace ffi=::xla::ffi; namespace sl=::slate;
template <typename T>
static ffi::Error Impl(int64_t nq,int64_t m,int64_t n,int64_t k,
    int64_t ar,int64_t ac,int64_t br,int64_t bc,
    int64_t mba,int64_t nba,int64_t mbb,int64_t nbb,
    int64_t mbc,int64_t nbc,int ta,int tb,T alpha,T beta,
    lorrax_ffi::slate::SlateCtx* ctx,const T* A,const T* B,const T* C,T* D)
{
    if (ctx->p != ctx->q)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "slate.matmul(host) requires a square process grid");
    const int64_t asz=(ar/ctx->p)*(ac/ctx->q),
                  bsz=(br/ctx->p)*(bc/ctx->q),
                  csz=(m /ctx->p)*(n /ctx->q);
    if (D!=C) std::memcpy(D,C,nq*csz*sizeof(T));
    sl::Options opts={{sl::Option::Target,sl::Target::HostTask}};
    for (int64_t q=0;q<nq;++q) {
        try {
            auto Am=sl::Matrix<T>::fromScaLAPACK(
                ar,ac,const_cast<T*>(A+q*asz),ar/ctx->p,
                mba,nba,ctx->p,ctx->q,ctx->comm);
            auto Bm=sl::Matrix<T>::fromScaLAPACK(
                br,bc,const_cast<T*>(B+q*bsz),br/ctx->p,
                mbb,nbb,ctx->p,ctx->q,ctx->comm);
            auto Dm=sl::Matrix<T>::fromScaLAPACK(
                m,n,D+q*csz,m/ctx->p,mbc,nbc,ctx->p,ctx->q,ctx->comm);
            auto Aop=ta==0 ? Am : (ta==1 ? transpose(Am) : conj_transpose(Am));
            auto Bop=tb==0 ? Bm : (tb==1 ? transpose(Bm) : conj_transpose(Bm));
            sl::multiply(alpha,Aop,Bop,beta,Dm,opts);
            Dm.tileUpdateAllOrigin();
        }
        catch (const std::exception& ex) {
            std::ostringstream os; os<<"slate::multiply(host) threw: "<<ex.what();
            return ffi::Error(ffi::ErrorCode::kInternal,os.str());
        }
    }
    return ffi::Error::Success();
}
static ffi::Error Dispatch(ffi::AnyBuffer A,ffi::AnyBuffer B,ffi::AnyBuffer C,
    ffi::Result<ffi::AnyBuffer> D,int64_t nq,int64_t m,int64_t n,int64_t k,
    int64_t ar,int64_t ac,int64_t br,int64_t bc,
    int64_t mba,int64_t nba,int64_t mbb,int64_t nbb,int64_t mbc,int64_t nbc,
    int64_t ta,int64_t tb,double are,double aim,double bre,double bim,int64_t h)
{
    auto* ctx=reinterpret_cast<lorrax_ffi::slate::SlateCtx*>(h);
    if (!ctx) return ffi::Error(ffi::ErrorCode::kInvalidArgument,"slate.matmul(host): null context");
    std::lock_guard<std::mutex> lock(lorrax_ffi::slate::host_collective_mutex());
    auto dt=A.element_type();
    if (B.element_type()!=dt || C.element_type()!=dt || D->element_type()!=dt)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,"slate.matmul(host): dtypes disagree");
    if (dt==ffi::DataType::F64)
        return Impl<double>(nq,m,n,k,ar,ac,br,bc,mba,nba,mbb,nbb,mbc,nbc,
            ta,tb,are,bre,ctx,static_cast<const double*>(A.untyped_data()),
            static_cast<const double*>(B.untyped_data()),static_cast<const double*>(C.untyped_data()),
            static_cast<double*>(D->untyped_data()));
    if (dt==ffi::DataType::C128) { using Z=std::complex<double>;
        return Impl<Z>(nq,m,n,k,ar,ac,br,bc,mba,nba,mbb,nbb,mbc,nbc,
            ta,tb,Z(are,aim),Z(bre,bim),ctx,static_cast<const Z*>(A.untyped_data()),
            static_cast<const Z*>(B.untyped_data()),static_cast<const Z*>(C.untyped_data()),
            static_cast<Z*>(D->untyped_data())); }
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,"slate.matmul(host) supports F64/C128");
}
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(
 SlateBatchedGemmHostFfi,lorrax_ffi::slate_gemm_host::Dispatch,xla::ffi::Ffi::Bind()
 .Arg<xla::ffi::AnyBuffer>().Arg<xla::ffi::AnyBuffer>().Arg<xla::ffi::AnyBuffer>().Ret<xla::ffi::AnyBuffer>()
 .Attr<int64_t>("nq").Attr<int64_t>("m").Attr<int64_t>("n").Attr<int64_t>("k")
 .Attr<int64_t>("a_rows").Attr<int64_t>("a_cols").Attr<int64_t>("b_rows").Attr<int64_t>("b_cols")
 .Attr<int64_t>("mb_a").Attr<int64_t>("nb_a").Attr<int64_t>("mb_b").Attr<int64_t>("nb_b")
 .Attr<int64_t>("mb_c").Attr<int64_t>("nb_c").Attr<int64_t>("transa").Attr<int64_t>("transb")
 .Attr<double>("alpha_re").Attr<double>("alpha_im").Attr<double>("beta_re").Attr<double>("beta_im")
 .Attr<int64_t>("ctx_handle"));
