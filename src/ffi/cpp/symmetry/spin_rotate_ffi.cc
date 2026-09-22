#include <cuda_runtime.h>
#include <cstdint>
#include "xla/ffi/api/ffi.h"
#include "ffi_helpers.h"

namespace lorrax_ffi::symmetry {
namespace ffi = ::xla::ffi;
void launch_spin_rotate(const void*, const void*, void*, int64_t, int,
                        int64_t, int64_t, cudaStream_t);

static ffi::Error rotate(cudaStream_t stream, ffi::AnyBuffer g,
                         ffi::AnyBuffer u, ffi::Result<ffi::AnyBuffer> out) {
    if (g.element_type() != ffi::DataType::C128 ||
        u.element_type() != ffi::DataType::C128 ||
        out->element_type() != ffi::DataType::C128 ||
        g.dimensions().size() != 5 || u.dimensions().size() != 3 ||
        out->dimensions().size() != 5)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                         "spin rotation requires complex128 G[k,mu,s,nu,s], U[k,s,s]");
    const auto d = g.dimensions();
    const int64_t nk = d[0], s = d[2], mu = d[1], nu = d[3];
    if ((s != 2 && s != 4) || d[4] != s || nk < 1 || mu < 1 || nu < 1 ||
        u.dimensions()[0] != nk || u.dimensions()[1] != s || u.dimensions()[2] != s ||
        g.element_count() / (s * s) > 128ULL * 2147483647)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "invalid spin rotation dimensions");
    for (int i = 0; i < 5; ++i)
        if (out->dimensions()[i] != d[i])
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, "spin rotation output shape mismatch");
    launch_spin_rotate(g.untyped_data(), u.untyped_data(), out->untyped_data(),
                       nk, s, mu, nu, stream);
    LORRAX_CUDA_CHECK(cudaGetLastError());
    return ffi::Error::Success();
}
}  // namespace lorrax_ffi::symmetry

XLA_FFI_DEFINE_HANDLER_SYMBOL(SpinRotateCentroidCudaFfi, lorrax_ffi::symmetry::rotate,
    xla::ffi::Ffi::Bind().Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>().Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>());
