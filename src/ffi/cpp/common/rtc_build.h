// rtc_build.h -- the device legs' one run-time compilation service.
//
// Compiles an embedded kernel source to a device image with the leg's RTC library (NVRTC on
// the CUDA leg, hipRTC on the ROCM leg; common/leg.h picks the leg), keeps the image in a disk
// cache keyed by everything that decides it, and loads it through the leg's module API.  Every
// handler that builds kernels at run time goes through here, so there is one compile path, one
// cache format and one key rule on both vendors:
//
//   CUDA  the nvidia-mathdx k-convolution family, its BSE outer kernels, the Fourier plan's
//         fused pair.  NVRTC is linked; the driver API is resolved lazily (libcuda is already
//         mapped by JAX, and build/login nodes have none to link).
//   ROCM  the direct-DFT k-leading family (kdft/).  The module API is the HIP runtime the leg
//         links; hipRTC is resolved lazily, from the directory of that HIP runtime, so the
//         process maps ONE ROCm release (the one the JAX plugin loaded) and the leg still loads
//         -- its other families still serve -- where no hipRTC is installed.
//
// Key (FNV-1a, in this order): the source; each embedded header's text ("\x1d" + text); each
// deciding option ("\x1f" + option); each toolchain version file ("\x1e" + basename + content);
// "<rtc><major>.<minor>@<RTC library real path>" (rtc = nvrtc | hiprtc); the caller's extra key.
// Include paths are not in the key (two installs of one version compile one image).  The
// device architecture is a deciding option (--gpu-architecture / --offload-arch), so it is
// keyed.  A version file that reads empty is reported in `missing`; the caller must then
// disable its disk cache rather than drop the file from the key.
//
// Disk image: "LRXKCONV1\n" + 16 hex key + 16 hex payload hash + '\n' + image, written to a
// unique temporary and renamed (atomic on one filesystem, so concurrent ranks cannot tear it),
// re-hashed on read.  A torn, foreign or non-image file is recompiled and overwritten; one the
// module loader refuses is deleted, recompiled once and replaced.  The image is a cubin (ELF)
// on CUDA and an AMDGPU code object (ELF, or a clang offload bundle) on ROCm.
//
// The CUDA spelling is the historical nvrtc_build.{h,cc} one, unchanged: the same calls in the
// same order, the same key bytes, the same messages ("nvrtcCompileProgram", "cuModuleLoadData"),
// so a CUDA image cached before the rename is still a hit.
#pragma once

#include "leg.h"

#if !LRX_LEG_DEVICE
#  error "rtc_build.h is device-leg only: the host leg compiles no kernel at run time"
#endif

#if defined(LRX_LEG_ROCM)
#  include <hip/hip_runtime_api.h>
#  define LRX_RTC_API "hiprtc"        // the RTC library's call prefix, in messages and the key
#  define LRX_MODULE_API "hip"        // the module API's call prefix, in messages
#else
#  include <cuda.h>
#  define LRX_RTC_API "nvrtc"
#  define LRX_MODULE_API "cu"
#endif

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lorrax_ffi::rtc {

#if defined(LRX_LEG_ROCM)
using Result   = hipError_t;
using Module   = hipModule_t;
using Function = hipFunction_t;
using Stream   = hipStream_t;
inline constexpr Result kOk = hipSuccess;
inline constexpr const char* kImageExt = ".hsaco";               // cache file extension
#else
using Result   = CUresult;
using Module   = CUmodule;
using Function = CUfunction;
using Stream   = CUstream;
inline constexpr Result kOk = CUDA_SUCCESS;
inline constexpr const char* kImageExt = ".cubin";
#endif

// The module API: resolved lazily on CUDA, the linked HIP runtime on ROCm.  The CUDA leg also
// carries the context and function-attribute calls its callers use; the ROCM leg needs neither
// (HIP binds the device's primary context itself, and gfx90a grants a block its 64 KiB of LDS
// without an opt-in).
struct DriverApi {
    Result (*ModuleLoadData)(Module*, const void*) = nullptr;
    Result (*ModuleGetFunction)(Function*, Module, const char*) = nullptr;
    Result (*LaunchKernel)(Function, unsigned, unsigned, unsigned,
                           unsigned, unsigned, unsigned, unsigned,
                           Stream, void**, void**) = nullptr;
#if defined(LRX_LEG_CUDA)
    CUresult (*CtxGetCurrent)(CUcontext*) = nullptr;
    CUresult (*GetErrorString)(CUresult, const char**) = nullptr;
    CUresult (*FuncSetAttribute)(CUfunction, int, int) = nullptr;
#endif
    bool ok = false;
    std::string err;
};
const DriverApi& driver_api();
std::string err_string(Result r);

struct Program {
    const char* src = nullptr;                                   // the embedded source
#if defined(LRX_LEG_ROCM)
    const char* name = "lrx_hiprtc.hip";                         // RTC program name
#else
    const char* name = "lrx_nvrtc.cu";
#endif
    std::vector<std::pair<const char*, const char*>> headers;    // embedded (include name, text)
    std::vector<std::string> defs;                               // options that decide the image
    std::vector<std::string> includes;                           // -I directories (not keyed)
    std::vector<std::string> version_files;                      // toolchain version headers
    std::string extra_key;                                       // hashed last
    const char* kernel = nullptr;                                // entry point
    std::vector<const char*> entries;                            // further entry points
};

uint64_t fnv1a(std::string_view data, uint64_t h = 1469598103934665603ULL);
std::string hex16(uint64_t v);
bool exists(const std::string& path);
std::string read_file(const std::string& path);

// The RTC option that selects `arch` (lrx::gpu::device_arch): "--gpu-architecture=sm_80" on
// CUDA, "--offload-arch=gfx90a:sramecc+:xnack-" on ROCm; and the file-name tag of `arch`
// ("sm80", "gfx90a": the target id without its feature flags).
std::string arch_option(const std::string& arch);
std::string arch_tag(const std::string& arch);

#if defined(LRX_LEG_CUDA)
std::string toolkit_include(std::string* why);                   // CUDA include beside libnvrtc

// An nvidia-mathdx program: the wheel at `root` (its nvidia/mathdx directory) and the CUDA
// toolkit include `cuda_inc` as -I paths; as version files `dx`'s own header
// (include/<dx>/<dx>_version.hpp), commonDx's, CUTLASS's and CCCL's; the wheel's dist-info
// name as the extra key.  Every mathdx kernel (cuFFTDx, cuBLASDx) is keyed by this one rule.
void mathdx_toolchain(const std::string& root, const std::string& cuda_inc, const char* dx,
                      Program* p);
#endif

// The program's key; `missing` lists version files that read empty.  On ROCm a hipRTC that
// cannot be resolved keys as version 0.0 at an empty path; build() then refuses with the reason.
uint64_t key(const Program& p, std::string* missing);

struct Image {
    Function fn = nullptr;
    std::vector<Function> fns;                                   // p.entries, in order
    double ms = 0.0;
    bool from_disk = false, stored = false, rebuilt_bad = false;
};

// Load `cache_path` (framed with `key_hex`) when it is non-empty and valid, else compile (and
// store it under `cache_dir`); load the module and resolve `p.kernel` and `p.entries` (passes of
// one program that run at their own register budgets).  On failure *where names
// the step and *err the reason.
bool build(const Program& p, const std::string& cache_dir, const std::string& cache_path,
           const std::string& key_hex, Image* out, std::string* where, std::string* err);

}  // namespace lorrax_ffi::rtc
