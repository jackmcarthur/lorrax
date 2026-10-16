// active_gemm_device_ffi.cc -- the active-range local GEMM with DEVICE-side bounds.
//
//   lorrax_active_gemm_device      C = alpha A[:, :, lo:hi] diag(w) B[:, lo:hi] + beta C  (C aliased)
//   lorrax_active_gemm_device_out  the same with no C operand (beta 0; every row written)
//
// The same operation, operands and layouts as cublas/local_active_gemm_ffi.cc (the
// lorrax_cublas_local_* targets: row-major (nq,m,K) @ (nq,K,n), int32 bounds (1|nq, 2)), with
// two differences that are the point of this family:
//   * the bounds stay on the device: the kernel (active_gemm_kernel.h) reads them, so a call
//     makes NO device-to-host copy and NO stream synchronization (the cuBLAS handler's
//     per-call read + sync drains the stream inside Sigma's tau loop: on MI250X Fe Sigma tau
//     13.5 s against 9.1 s for the masked XLA route);
//   * the weights are an operand applied on load, so no weighted copy of A is formed, and no
//     workspace is needed (nothing from XLA scratch).
// Invalid bounds cannot be refused without reading them: they give an all-NaN result (the
// guarded Python arm's contract), never a clamped interval.
//
// Leg-neutral: written against lrx::gpu (gpu_runtime.h) and rtc (rtc_build.h) only; the
// kernel text has no vendor spelling.  CMake builds it into the ROCM leg; the CUDA leg adopts
// it by listing this file (and its generated header) in LORRAX_FFI_SOURCES, nothing else.
// Images: one program, two entry points (complex128, float64), compiled once per (device)
// and disk-cached as active_gemm_t<TM>x<TN>x<TK>_<arch>_<key><ext> with the common key rule.
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "../common/ffi_helpers.h"
#include "../common/gpu_runtime.h"
#include "../common/rtc_build.h"
#include "generated/active_gemm_src.h"

namespace lorrax_ffi::active_gemm {

constexpr int kTm = 64, kTn = 64, kTk = 16, kThreads = 256;   // = active_gemm_kernel.h defaults

static ffi::Error fail(const char* gate, const std::string& why,
                       ffi::ErrorCode code = ffi::ErrorCode::kInvalidArgument) {
    return ffi::Error(code, std::string("GATE ") + gate + ": " + why);
}

struct Kernels { rtc::Function z = nullptr, d = nullptr; };
static std::mutex g_mu;
static std::map<int, Kernels> g_cache;
static std::map<int, std::string> g_fail;

static ffi::Error kernels(std::string_view cache_dir, Kernels* out) {
    int dev = 0;
    LORRAX_GPU_CHECK(lrx::gpu::get_device(&dev));
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_cache.find(dev); it != g_cache.end()) { *out = it->second; return ffi::Error::Success(); }
    if (auto it = g_fail.find(dev); it != g_fail.end())
        return fail("active-gemm-build", "(cached failure) " + it->second, ffi::ErrorCode::kInternal);
    std::string arch;
    LORRAX_GPU_CHECK(lrx::gpu::device_arch(dev, &arch));
    rtc::Program prog;
    prog.src = kKernelSrc;
    prog.name = "lrx_active_gemm";
    prog.kernel = "lrx_active_gemm_z";
    prog.entries = {"lrx_active_gemm_d"};
    prog.defs = {rtc::kOptOption, "-std=c++17", rtc::arch_option(arch),
                 "-DLRX_TM=" + std::to_string(kTm), "-DLRX_TN=" + std::to_string(kTn),
                 "-DLRX_TK=" + std::to_string(kTk)};
    const std::string key_hex = rtc::hex16(rtc::key(prog, nullptr));
    const std::string dir(cache_dir);
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/active_gemm_t" << kTm << "x" << kTn << "x" << kTk << "_"
             << rtc::arch_tag(arch) << "_" << key_hex << rtc::kImageExt;
        path = name.str();
    }
    rtc::Image img;
    std::string where, err;
    if (!rtc::build(prog, dir, path, key_hex, &img, &where, &err)) {
        g_fail.emplace(dev, where + " -- " + err);
        return fail("active-gemm-build", where + " -- " + err, ffi::ErrorCode::kInternal);
    }
    if (!img.from_disk)
        std::fprintf(stderr, "[active_gemm] " LRX_RTC_API " %s tile %dx%dx%d %s in %.1f ms%s\n",
                     img.rebuilt_bad ? "rebuilt (cached image refused)" : "built", kTm, kTn, kTk,
                     arch.c_str(), img.ms, img.stored ? " (stored)" : "");
    Kernels k;
    k.z = img.fn;
    k.d = img.fns.empty() ? nullptr : img.fns[0];
    if (k.d == nullptr)
        return fail("active-gemm-build", "entry lrx_active_gemm_d missing", ffi::ErrorCode::kInternal);
    g_cache.emplace(dev, k);
    *out = k;
    return ffi::Error::Success();
}

// c_in == nullptr: the `_out` binding (no C operand, beta = 0).
static ffi::Error Launch(lrx::gpu::Stream stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
                         ffi::AnyBuffer w, ffi::BufferR2<ffi::S32> bounds,
                         const ffi::AnyBuffer* c_in, ffi::Result<ffi::AnyBuffer> c_out,
                         int64_t nq, int64_t m, int64_t k, int64_t n, int64_t weighted,
                         double alpha_re, double alpha_im, double beta_re, double beta_im,
                         std::string_view cache_dir) {
    const auto bad = [](const std::string& why) { return fail("active-gemm-operands", why); };
    if (nq < 1 || m < 1 || k < 1 || n < 1)
        return bad("want positive nq, m, K, n");
    if (nq > 65535 || m > 65535LL * kTm || n > 2147483647LL || k > 2147483647LL ||
        m * k > (1LL << 62) / nq || k * n > (1LL << 62) / nq || m * n > (1LL << 62) / nq)
        return fail("active-gemm-grid", "got nq=" + std::to_string(nq) + ", m=" + std::to_string(m) +
                    "; want nq <= 65535 and m <= " + std::to_string(65535LL * kTm) +
                    " (one grid z-plane per row, 64-row tiles in y)");
    const ffi::DataType dt = a.element_type();
    if (dt != ffi::DataType::C128 && dt != ffi::DataType::F64)
        return bad("want complex128 or float64 operands");
    if (b.element_type() != dt || c_out->element_type() != dt || (c_in && c_in->element_type() != dt) ||
        (weighted && w.element_type() != dt))
        return bad("want every operand in one dtype");
    const auto ad = a.dimensions(), bd = b.dimensions(), od = c_out->dimensions();
    if (ad.size() != 3 || bd.size() != 3 || od.size() != 3 || ad[0] != nq || ad[1] != m ||
        ad[2] != k || bd[0] != nq || bd[1] != k || bd[2] != n || od[0] != nq || od[1] != m || od[2] != n)
        return bad("want A (nq,m,K), B (nq,K,n), C (nq,m,n)");
    if (c_in && c_in->untyped_data() != c_out->untyped_data())
        return bad("want the declared C input/output alias");
    if (!c_in && (beta_re != 0.0 || beta_im != 0.0))
        return bad("the no-C binding requires beta = 0");
    if (dt == ffi::DataType::F64 && (alpha_im != 0.0 || beta_im != 0.0))
        return bad("real operands require real alpha/beta");
    const auto wd = w.dimensions();
    if (weighted && (wd.size() != 2 || wd[0] != nq || wd[1] != k))
        return bad("want weights (nq,K) when weighted = 1");
    const auto bnd = bounds.dimensions();
    if (bnd[1] != 2 || (bnd[0] != 1 && bnd[0] != nq))
        return bad("want bounds (1|nq, 2)");

    Kernels ks;
    FFI_RETURN_IF_ERROR(kernels(cache_dir, &ks));
    rtc::Function fn = dt == ffi::DataType::C128 ? ks.z : ks.d;
    const void* A = a.untyped_data();
    const void* B = b.untyped_data();
    const void* W = weighted ? w.untyped_data() : nullptr;
    const int* BD = bounds.typed_data();
    void* C = c_out->untyped_data();
    int bstride = bnd[0] == 1 ? 0 : 2, nb = static_cast<int>(bnd[0]);
    int mi = static_cast<int>(m), ki = static_cast<int>(k), ni = static_cast<int>(n);
    int read_c = (c_in && (beta_re != 0.0 || beta_im != 0.0)) ? 1 : 0;
    void* args[] = {&A, &B, &W, &BD, &bstride, &nb, &C, &mi, &ki, &ni,
                    &alpha_re, &alpha_im, &beta_re, &beta_im, &read_c};
    const unsigned gx = static_cast<unsigned>((n + kTn - 1) / kTn);
    const unsigned gy = static_cast<unsigned>((m + kTm - 1) / kTm);
    const rtc::Result r = rtc::driver_api().LaunchKernel(
        fn, gx, gy, static_cast<unsigned>(nq), kThreads, 1, 1, 0,
        reinterpret_cast<rtc::Stream>(stream), args, nullptr);
    if (r != rtc::kOk)
        return fail("active-gemm-launch", LRX_MODULE_API "ModuleLaunchKernel: " + rtc::err_string(r),
                    ffi::ErrorCode::kInternal);
    return ffi::Error::Success();
}

static ffi::Error Dispatch(lrx::gpu::Stream s, ffi::AnyBuffer a, ffi::AnyBuffer b, ffi::AnyBuffer w,
                           ffi::BufferR2<ffi::S32> bounds, ffi::AnyBuffer c_in,
                           ffi::Result<ffi::AnyBuffer> c_out, int64_t nq, int64_t m, int64_t k,
                           int64_t n, int64_t weighted, double alpha_re, double alpha_im,
                           double beta_re, double beta_im, std::string_view cubin_dir) {
    return Launch(s, a, b, w, bounds, &c_in, c_out, nq, m, k, n, weighted, alpha_re, alpha_im,
                  beta_re, beta_im, cubin_dir);
}

static ffi::Error OutDispatch(lrx::gpu::Stream s, ffi::AnyBuffer a, ffi::AnyBuffer b,
                              ffi::AnyBuffer w, ffi::BufferR2<ffi::S32> bounds,
                              ffi::Result<ffi::AnyBuffer> c_out, int64_t nq, int64_t m, int64_t k,
                              int64_t n, int64_t weighted, double alpha_re, double alpha_im,
                              std::string_view cubin_dir) {
    return Launch(s, a, b, w, bounds, nullptr, c_out, nq, m, k, n, weighted, alpha_re, alpha_im,
                  0.0, 0.0, cubin_dir);
}

}  // namespace lorrax_ffi::active_gemm

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ActiveGemmDeviceFfi, lorrax_ffi::active_gemm::Dispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()                    // A (nq, m, K)
        .Arg<xla::ffi::AnyBuffer>()                    // B (nq, K, n)
        .Arg<xla::ffi::AnyBuffer>()                    // w (nq, K), or any 1-element buffer
        .Arg<xla::ffi::BufferR2<xla::ffi::S32>>()      // bounds (1|nq, 2), read on the device
        .Arg<xla::ffi::AnyBuffer>()                    // C (aliased to the result)
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nq")
        .Attr<int64_t>("m")
        .Attr<int64_t>("k")
        .Attr<int64_t>("n")
        .Attr<int64_t>("weighted")
        .Attr<double>("alpha_re")
        .Attr<double>("alpha_im")
        .Attr<double>("beta_re")
        .Attr<double>("beta_im")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ActiveGemmDeviceOutFfi, lorrax_ffi::active_gemm::OutDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::BufferR2<xla::ffi::S32>>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nq")
        .Attr<int64_t>("m")
        .Attr<int64_t>("k")
        .Attr<int64_t>("n")
        .Attr<int64_t>("weighted")
        .Attr<double>("alpha_re")
        .Attr<double>("alpha_im")
        .Attr<std::string_view>("cubin_dir"));
