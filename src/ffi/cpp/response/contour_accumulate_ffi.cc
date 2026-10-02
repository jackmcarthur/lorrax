#include <cuda_runtime.h>
#include <cstdint>
#include <optional>
#include "xla/ffi/api/ffi.h"
#include "../common/ffi_helpers.h"
namespace lorrax_ffi::contour {
namespace ffi = ::xla::ffi;
void launch(const void*, const void*, const void*, void*, int64_t, int64_t,
            cudaStream_t);
static ffi::Error accumulate(cudaStream_t stream, ffi::AnyBuffer a,
    ffi::AnyBuffer c, ffi::AnyBuffer p, ffi::Result<ffi::AnyBuffer> out) {
    if (a.element_type() != ffi::DataType::C128 ||
        c.element_type() != ffi::DataType::C128 ||
        p.element_type() != ffi::DataType::C128 ||
        out->element_type() != ffi::DataType::C128 ||
        a.dimensions().size() != 4 || c.dimensions().size() != 3 ||
        p.dimensions().size() != 1 || out->dimensions().size() != 4) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
            "contour accumulator requires complex128 [output,q,m,n], [q,m,n], [output]");
    }
    const int64_t outputs = p.element_count();
    const int64_t elements = c.element_count();
    if (outputs < 1 || elements < 1 || a.dimensions()[0] != outputs ||
        a.element_count() != static_cast<size_t>(outputs * elements) ||
        out->element_count() != a.element_count()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "contour accumulator shape mismatch");
    }
    if (outputs > 4LL * 65535 || elements > 128LL * 2147483647)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "contour accumulator exceeds CUDA grid limits");
    for (int i = 0; i < 4; ++i) {
        if (out->dimensions()[i] != a.dimensions()[i])
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "contour accumulator output shape mismatch");
    }
    for (int i = 0; i < 3; ++i) {
        if (a.dimensions()[i+1] != c.dimensions()[i])
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "contour contribution tile mismatch");
    }
    launch(a.untyped_data(), c.untyped_data(), p.untyped_data(),
           out->untyped_data(), elements, outputs, stream);
    LORRAX_CUDA_CHECK(cudaGetLastError());
    return ffi::Error::Success();
}
}
namespace lorrax_ffi::contour {
void launch_block(const void*, const void*, const void*, const void*, void*, int64_t, int64_t, int64_t,
                  int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t);
// A[o, q, m0 + m, n0 + n] += sum_s p[s, o] c[s, q, m, n] on one device tile, in place (output
// aliased to a); valid (2,) s32 bounds (m, n), the block's rows and columns that are not padding;
// origin (optional, s32 (2,)): a runtime (m0, n0) added to the attributes (a scanned pass's row
// offset), entries it moves outside A not touched.
static ffi::Error accumulate_block(cudaStream_t stream, ffi::AnyBuffer a, ffi::AnyBuffer c,
    ffi::AnyBuffer p, ffi::AnyBuffer valid, std::optional<ffi::AnyBuffer> origin,
    ffi::Result<ffi::AnyBuffer> out, int64_t m0, int64_t n0) {
    const auto C = ffi::DataType::C128;
    const auto ad = a.dimensions(), cd = c.dimensions(), pd = p.dimensions();
    if (a.element_type() != C || c.element_type() != C || p.element_type() != C ||
        out->element_type() != C || valid.element_type() != ffi::DataType::S32 ||
        ad.size() != 4 || cd.size() != 4 || pd.size() != 2 || valid.element_count() != 2 ||
        out->element_count() != a.element_count() ||
        (origin && (origin->element_type() != ffi::DataType::S32 || origin->element_count() != 2)))
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
            "contour block accumulator requires c128 A [output,q,M,N], c [terms,q,bm,bn], "
            "p [terms,output], s32 valid [2], s32 origin [2]");
    const int64_t outputs = ad[0], nq = ad[1], M = ad[2], N = ad[3];
    const int64_t terms = cd[0], bm = cd[2], bn = cd[3];
    if (cd[1] != nq || pd[0] != terms || pd[1] != outputs || terms < 1 || outputs < 1 ||
        (!origin && (m0 < 0 || n0 < 0 || m0 + bm > M || n0 + bn > N)))
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "contour block accumulator: the block does not lie inside A");
    if (outputs > 4LL * 65535 || nq * bm * bn > 128LL * 2147483647)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "contour block accumulator exceeds CUDA grid limits");
    if (out->untyped_data() != a.untyped_data())
        LORRAX_CUDA_CHECK(cudaMemcpyAsync(out->untyped_data(), a.untyped_data(), a.size_bytes(),
                                          cudaMemcpyDeviceToDevice, stream));
    if (nq * bm * bn > 0)
        launch_block(c.untyped_data(), p.untyped_data(), valid.untyped_data(),
                     origin ? origin->untyped_data() : nullptr, out->untyped_data(), outputs, nq, M, N, bm,
                     bn, m0, n0, terms, stream);
    LORRAX_CUDA_CHECK(cudaGetLastError());
    return ffi::Error::Success();
}
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(ContourAccumulateBlockFfi,
    lorrax_ffi::contour::accumulate_block,
    xla::ffi::Ffi::Bind().Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>().Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>().Arg<xla::ffi::AnyBuffer>()
        .OptionalArg<xla::ffi::AnyBuffer>()   // origin (s32 [2]): a runtime (m0, n0) added to the attributes
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("m0").Attr<int64_t>("n0"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(ContourAccumulateFfi,
    lorrax_ffi::contour::accumulate,
    xla::ffi::Ffi::Bind().Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>().Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>().Ret<xla::ffi::AnyBuffer>());
