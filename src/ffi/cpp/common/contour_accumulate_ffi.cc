#include <cuda_runtime.h>
#include <cstdint>
#include "xla/ffi/api/ffi.h"
#include "ffi_helpers.h"
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
XLA_FFI_DEFINE_HANDLER_SYMBOL(ContourAccumulateFfi,
    lorrax_ffi::contour::accumulate,
    xla::ffi::Ffi::Bind().Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>().Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>().Ret<xla::ffi::AnyBuffer>());
