// gpu_runtime.h — the GPU-runtime vocabulary of LORRAX's device legs.
//
// The ONLY file in the shared sources that names a vendor runtime API.  A
// family written against `lrx::gpu::` compiles unchanged into the CUDA leg
// (CUDA runtime) and the ROCM leg (HIP runtime); the leg is chosen by
// common/leg.h.  The HIP runtime API mirrors the CUDA one, so every wrapper
// below is ONE definition over LRX_GPU_SYM (cuda##x / hip##x); the three
// names that genuinely differ (pinned host allocation and its free, the
// out-of-memory error) are spelled per vendor, visibly.
//
// Thin by design: inline forwarding functions and type aliases, no state,
// no policy.  On the CUDA leg every wrapper is the CUDA call it replaced,
// so CUDA behaviour is unchanged.  Messages name the vendor through
// LRX_GPU_API ("cuda" / "hip") so a CUDA error string reads exactly as
// before, e.g. LRX_GPU_API "MemcpyAsync" == "cudaMemcpyAsync".
//
// Library handles (BLAS, FFT, solver, collectives, RTC) are NOT here: each
// gets its own header when its family is ported (gpu_blas.h, gpu_fft.h, ...),
// so a family pulls in only the libraries it links.
#pragma once

#include "leg.h"

#if !LRX_LEG_DEVICE
#  error "gpu_runtime.h is device-leg only: the host leg makes no GPU runtime call"
#endif

#include <cstddef>
#include <string>

#if defined(LRX_LEG_ROCM)
#  include <hip/hip_runtime_api.h>
#  define LRX_GPU_SYM(x) hip##x
#  define LRX_GPU_API "hip"
#  define LRX_GPU_VENDOR "HIP"
#else
#  include <cuda_runtime.h>
#  define LRX_GPU_SYM(x) cuda##x
#  define LRX_GPU_API "cuda"
#  define LRX_GPU_VENDOR "CUDA"
#endif

namespace lrx::gpu {

using Error      = LRX_GPU_SYM(Error_t);
using Stream     = LRX_GPU_SYM(Stream_t);
using Event      = LRX_GPU_SYM(Event_t);
using MemcpyKind = LRX_GPU_SYM(MemcpyKind);

inline constexpr Error      kSuccess            = LRX_GPU_SYM(Success);
inline constexpr MemcpyKind kHostToDevice       = LRX_GPU_SYM(MemcpyHostToDevice);
inline constexpr MemcpyKind kDeviceToHost       = LRX_GPU_SYM(MemcpyDeviceToHost);
inline constexpr MemcpyKind kDeviceToDevice     = LRX_GPU_SYM(MemcpyDeviceToDevice);
inline constexpr unsigned   kStreamNonBlocking  = LRX_GPU_SYM(StreamNonBlocking);
inline constexpr unsigned   kEventDisableTiming = LRX_GPU_SYM(EventDisableTiming);
#if defined(LRX_LEG_ROCM)
inline constexpr Error      kErrorOutOfMemory   = hipErrorOutOfMemory;
#else
inline constexpr Error      kErrorOutOfMemory   = cudaErrorMemoryAllocation;
#endif

inline const char* error_name(Error e)   { return LRX_GPU_SYM(GetErrorName)(e); }
inline const char* error_string(Error e) { return LRX_GPU_SYM(GetErrorString)(e); }

inline Error get_device(int* dev) { return LRX_GPU_SYM(GetDevice)(dev); }
inline Error set_device(int dev)  { return LRX_GPU_SYM(SetDevice)(dev); }

inline Error stream_create_with_flags(Stream* s, unsigned flags) {
    return LRX_GPU_SYM(StreamCreateWithFlags)(s, flags);
}
inline Error stream_destroy(Stream s)     { return LRX_GPU_SYM(StreamDestroy)(s); }
inline Error stream_synchronize(Stream s) { return LRX_GPU_SYM(StreamSynchronize)(s); }
inline Error stream_wait_event(Stream s, Event e, unsigned flags) {
    return LRX_GPU_SYM(StreamWaitEvent)(s, e, flags);
}

inline Error event_create_with_flags(Event* e, unsigned flags) {
    return LRX_GPU_SYM(EventCreateWithFlags)(e, flags);
}
inline Error event_destroy(Event e)          { return LRX_GPU_SYM(EventDestroy)(e); }
inline Error event_record(Event e, Stream s) { return LRX_GPU_SYM(EventRecord)(e, s); }
inline Error event_synchronize(Event e)      { return LRX_GPU_SYM(EventSynchronize)(e); }

inline Error memcpy_async(void* dst, const void* src, std::size_t n,
                          MemcpyKind kind, Stream s) {
    return LRX_GPU_SYM(MemcpyAsync)(dst, src, n, kind, s);
}
// Pitched 2-D copy (the SLATE eigh handler's tile scatter).
inline Error memcpy_2d_async(void* dst, std::size_t dpitch, const void* src,
                             std::size_t spitch, std::size_t width,
                             std::size_t height, MemcpyKind kind, Stream s) {
    return LRX_GPU_SYM(Memcpy2DAsync)(dst, dpitch, src, spitch, width, height, kind, s);
}
inline Error memset_async(void* dst, int value, std::size_t n, Stream s) {
    return LRX_GPU_SYM(MemsetAsync)(dst, value, n, s);
}

// The device's code-object target, the run-time compilers' architecture (rtc_build.h
// arch_option): "sm_<major><minor>" on CUDA, the full target id ("gfx90a:sramecc+:xnack-",
// feature flags included, so an image is built for the mode the device runs in) on ROCm.
inline Error device_arch(int dev, std::string* arch) {
#if defined(LRX_LEG_ROCM)
    hipDeviceProp_t prop;
    const Error e = hipGetDeviceProperties(&prop, dev);
    if (e == kSuccess) *arch = prop.gcnArchName;
    return e;
#else
    int major = 0, minor = 0;
    Error e = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (e == kSuccess) e = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    if (e == kSuccess) *arch = "sm_" + std::to_string(major) + std::to_string(minor);
    return e;
#endif
}

// Page-locked host memory: the two calls whose names differ per vendor.
inline Error malloc_host(void** p, std::size_t n) {
#if defined(LRX_LEG_ROCM)
    return hipHostMalloc(p, n, hipHostMallocDefault);
#else
    return cudaMallocHost(p, n);
#endif
}
inline Error free_host(void* p) {
#if defined(LRX_LEG_ROCM)
    return hipHostFree(p);
#else
    return cudaFreeHost(p);
#endif
}

}  // namespace lrx::gpu
