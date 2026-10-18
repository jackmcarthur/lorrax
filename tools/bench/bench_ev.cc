// CUDA-event stamps on the XLA stream for the kconv microbenchmark (bench-only, not in the bundle).
#include <cuda_runtime.h>
#include <cstdint>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;
static cudaEvent_t g_ev[64] = {};
static ffi::Error EvRec(cudaStream_t s, ffi::AnyBuffer x, ffi::Result<ffi::AnyBuffer> y, int64_t slot) {
    if (slot < 0 || slot >= 64) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "slot");
    if (!g_ev[slot]) cudaEventCreate(&g_ev[slot]);
    if (x.untyped_data() != y->untyped_data())
        cudaMemcpyAsync(y->untyped_data(), x.untyped_data(), x.size_bytes(), cudaMemcpyDeviceToDevice, s);
    cudaEventRecord(g_ev[slot], s);
    return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(LrxBenchEvRec, EvRec,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Attr<int64_t>("slot"));
extern "C" float lrx_bench_elapsed(int a, int b) {
    cudaEventSynchronize(g_ev[b]);
    float ms = -1.f;
    cudaEventElapsedTime(&ms, g_ev[a], g_ev[b]);
    return ms;
}
