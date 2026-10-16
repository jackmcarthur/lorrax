// kdft_klead_ffi.cc -- the k-leading kdft family: modes 2 and 3 of docs/architecture/kconv.md
// as run-time-compiled kernels (kdft_klead_kernel.h over the DFT core kdft_core.h).
//
//   lorrax_kdft_kconv_klead   U = fftn(scale * ifftn(T) * V_R[:, None, :, None, :])   (mode 2)
//   lorrax_kdft_kfft_klead    Y = scale * fftn|ifftn(X) over the leading flat-k axis   (mode 3)
//
// The same operations and operand layouts as the nvidia-mathdx targets
// lorrax_mathdx_kconv_klead / lorrax_mathdx_kfft_klead, without cuFFTDx.  A call is one or more
// PASSES (kdft_klead_plan.h): a pass transforms a contiguous group of k axes in LDS, a W-column
// tile per block, one HBM read and write each.  A grid whose k tile fits a block is one pass
// (the whole convolution); a larger grid (up to kNkMax points, axes up to kAxisMax of any
// length) takes 2*(groups-1)+1 passes for mode 2 and one per group for mode 3.  It is the ROCM
// leg's k-leading route (ffi.fft._ROCM_KLEAD_TARGETS); every call is gated against the XLA route
// on the same device (ffi.fft.xla_reference, tests/test_kconv_xla_gate.py).
//
// Leg-neutral: written against lrx::gpu (gpu_runtime.h) and rtc (rtc_build.h) only, so the
// one source would build into the CUDA leg as well; CMake lists it in the ROCM leg, where
// there is no cuFFTDx.
//
// Images: one per (pass mode, kernel, group, W, threads, precision, device architecture),
// compiled once per process and disk-cached in the router's cubin_dir as
// kdft_<conv|fft|mconv>_<gx>x<gy>x<gz>_w<W>t<NT>[_r|_rh][_c64]_<arch>_<key><ext> with the common
// key rule (_r: the register kernel, _rh: with the half exchange).  A refusal names its GATE.
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "../common/ffi_helpers.h"
#include "../common/gpu_runtime.h"
#include "../common/rtc_build.h"
#include "generated/kdft_klead_src.h"
#include "kdft_klead_ffi.h"

namespace lorrax_ffi::kdft {

using ::kdft::Pass;

static const char* mode_tag(int mode) { return mode == 2 ? "conv" : mode == 4 ? "mconv" : "fft"; }

static ffi::Error fail(const char* gate, const std::string& why,
                       ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
    return ffi::Error(code, std::string("GATE ") + gate + ": " + why);
}

static std::string grid_str(int64_t a, int64_t b, int64_t c) {
    return std::to_string(a) + "x" + std::to_string(b) + "x" + std::to_string(c);
}

// dev, mode, gx, gy, gz, w, nt, f32, reg*2 + xh
using Key = std::tuple<int, int, int, int, int, int, int, int, int>;
static std::mutex g_mu;
static std::map<Key, rtc::Function> g_cache;
static std::map<Key, std::string> g_fail;

ffi::Error build_pass(const Pass& p, bool f32, std::string_view cache_dir, rtc::Function* out) {
    int dev = 0;
    LORRAX_GPU_CHECK(lrx::gpu::get_device(&dev));
    const Key key{dev, p.mode, p.g[0], p.g[1], p.g[2], p.w, p.nt, f32 ? 1 : 0, p.reg * 2 + p.xh};
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_cache.find(key); it != g_cache.end()) { *out = it->second; return ffi::Error::Success(); }
    if (auto it = g_fail.find(key); it != g_fail.end()) return fail("kdft-build", "(cached failure) " + it->second);

    std::string arch;
    LORRAX_GPU_CHECK(lrx::gpu::device_arch(dev, &arch));
    rtc::Program prog;
    prog.src = kKernelSrc;
    prog.name = "lrx_kdft_klead";
    prog.kernel = "lrx_kdft_pass";
    prog.headers = {{"kdft_core.h", kCoreSrc}, {"kdft_plan.h", kPlanSrc}};
    prog.defs = {rtc::kOptOption, "-std=c++17", rtc::arch_option(arch),
                 "-DLRX_GX=" + std::to_string(p.g[0]), "-DLRX_GY=" + std::to_string(p.g[1]),
                 "-DLRX_GZ=" + std::to_string(p.g[2]), "-DLRX_W=" + std::to_string(p.w),
                 "-DLRX_NT=" + std::to_string(p.nt), "-DLRX_MODE=" + std::to_string(p.mode),
                 "-DLRX_REG=" + std::to_string(p.reg), "-DLRX_XH=" + std::to_string(p.xh),
                 std::string("-DLRX_F32=") + (f32 ? "1" : "0")};
    const std::string key_hex = rtc::hex16(rtc::key(prog, nullptr));
    const std::string dir(cache_dir);
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kdft_" << mode_tag(p.mode) << "_" << grid_str(p.g[0], p.g[1], p.g[2])
             << "_w" << p.w << "t" << p.nt << (p.reg ? (p.xh ? "_rh" : "_r") : "") << (f32 ? "_c64" : "")
             << "_" << rtc::arch_tag(arch) << "_"
             << key_hex << rtc::kImageExt;
        path = name.str();
    }
    rtc::Image img;
    std::string where, err;
    if (!rtc::build(prog, dir, path, key_hex, &img, &where, &err)) {
        g_fail.emplace(key, where + " -- " + err);
        return fail("kdft-build", where + " -- " + err);
    }
    if (!img.from_disk)
        std::fprintf(stderr, "[kconv_kdft] " LRX_RTC_API " %s %s%s%s group=(%d,%d,%d) %s W=%d NT=%d in %.1f ms%s\n",
                     img.rebuilt_bad ? "rebuilt (cached image refused)" : "built", mode_tag(p.mode),
                     p.reg ? (p.xh ? " reg/half" : " reg") : "", f32 ? " c64" : "", p.g[0], p.g[1], p.g[2],
                     arch.c_str(), p.w, p.nt, img.ms, img.stored ? " (stored)" : "");
    g_cache.emplace(key, img.fn);
    *out = img.fn;
    return ffi::Error::Success();
}

// The shared launch: X (nk, ncols) complex, in place on the result buffer.  For mode 2 the
// operand V_R (nk, mx, my) and T's (a, mx, b, my) extents decide the kernel's column map.
static ffi::Error launch(lrx::gpu::Stream stream, int mode, ffi::AnyBuffer X, const ffi::AnyBuffer* V,
                         ffi::Result<ffi::AnyBuffer> Y, int64_t nkx, int64_t nky, int64_t nkz,
                         double scale, int64_t forward, std::string_view cache_dir) {
    auto bad = [](const std::string& why) {
        return fail("kdft-operands", why, ffi::ErrorCode::kInvalidArgument);
    };
    const ffi::DataType dt = X.element_type();
    if (dt != ffi::DataType::C128 && dt != ffi::DataType::C64)
        return bad("want a complex128 or complex64 operand");
    if (Y->element_type() != dt || (V && V->element_type() != dt))
        return bad("want every operand in one precision (the router never casts)");
    if (nkx < 1 || nky < 1 || nkz < 1) return bad("want three positive k axes");
    if (nkx > ::kdft::kAxisMax || nky > ::kdft::kAxisMax || nkz > ::kdft::kAxisMax)
        return fail("kdft-axis", "got k-grid " + grid_str(nkx, nky, nkz) + "; want every k axis <= " +
                    std::to_string(::kdft::kAxisMax) + " (the verified range); the router keeps longer "
                    "axes on XLA", ffi::ErrorCode::kInvalidArgument);
    const int64_t nk = nkx * nky * nkz;
    if (nk > ::kdft::kNkMax)
        return fail("kdft-nk", "got k-grid " + grid_str(nkx, nky, nkz) + "; want nk <= " +
                    std::to_string(::kdft::kNkMax) + " (the verified range); the router keeps larger "
                    "grids on XLA", ffi::ErrorCode::kInvalidArgument);
    const auto xd = X.dimensions();
    if (xd.size() == 0 || xd[0] != nk) return bad("want the leading axis of length nk = " + std::to_string(nk));
    const int64_t ncols = static_cast<int64_t>(X.element_count()) / nk;
    if (static_cast<int64_t>(Y->element_count()) != nk * ncols) return bad("want the result shaped as the operand");
    int mx = 1, nb = 1, my = 1;
    if (mode == 2) {
        const auto vd = V->dimensions();
        if (xd.size() != 5 || vd.size() != 3 || vd[0] != nk || vd[1] != xd[2] || vd[2] != xd[4])
            return bad("want T (nk, a, mx, b, my) and V_R (nk, mx, my)");
        mx = static_cast<int>(xd[2]); nb = static_cast<int>(xd[3]); my = static_cast<int>(xd[4]);
    }
    if (ncols == 0) return ffi::Error::Success();

    const bool f32 = dt == ffi::DataType::C64;
    const std::vector<Pass> passes = ::kdft::plan_passes(static_cast<int>(nkx), static_cast<int>(nky),
                                                         static_cast<int>(nkz), mode, forward ? 1 : 0,
                                                         f32 ? 8 : 16);
    if (passes.empty())
        return fail("kdft-tile", "got k-grid " + grid_str(nkx, nky, nkz) + "; want a grouping of its axes "
                    "whose one-column tiles fit a block's " + std::to_string(::kdft::kLdsMax / 1024) +
                    " KiB of LDS", ffi::ErrorCode::kInvalidArgument);
    std::vector<rtc::Function> fns(passes.size());
    for (size_t i = 0; i < passes.size(); ++i)
        FFI_RETURN_IF_ERROR(build_pass(passes[i], f32, cache_dir, &fns[i]));

    void* t = Y->untyped_data();
    if (t != X.untyped_data())                                  // not aliased: work on the result
        LORRAX_GPU_CHECK(lrx::gpu::memcpy_async(t, X.untyped_data(), X.size_bytes(),
                                                lrx::gpu::kDeviceToDevice, stream));
    const void* v = V ? V->untyped_data() : nullptr;
    for (size_t i = 0; i < passes.size(); ++i) {
        const Pass& p = passes[i];
        // The scale rides on the convolution pass (mode 2) or the last transform pass (mode 3).
        const bool scaled = p.mode == 2 || (mode == 3 && i + 1 == passes.size());
        long long inner = p.inner_k * ncols;
        long long ntile = (inner + p.w - 1) / p.w;
        const long long blocks = p.outer * ntile;
        if (blocks > 0x7fffffffLL) return bad("want at most 2^31-1 column tiles per pass");
        int fwd = p.forward;
        double scale64 = scaled ? scale : 1.0;
        float scale32 = static_cast<float>(scale64);
        long long kob = 1;
        int m0 = 1, m1 = 1;
        void* args[] = {&t, &v, &inner, &ntile, &mx, &nb, &my,
                        f32 ? static_cast<void*>(&scale32) : static_cast<void*>(&scale64), &fwd, &kob, &m0, &m1};
        const rtc::Result r = rtc::driver_api().LaunchKernel(
            fns[i], static_cast<unsigned>(blocks), 1, 1, static_cast<unsigned>(p.nt), 1, 1, 0,
            reinterpret_cast<rtc::Stream>(stream), args, nullptr);
        if (r != rtc::kOk) return fail("kdft-launch", LRX_MODULE_API "ModuleLaunchKernel: " + rtc::err_string(r));
    }
    return ffi::Error::Success();
}

static ffi::Error KleadConv(lrx::gpu::Stream s, ffi::AnyBuffer T, ffi::AnyBuffer V,
                            ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz,
                            double scale, std::string_view cubin_dir) {
    return launch(s, 2, T, &V, U, nkx, nky, nkz, scale, 0, cubin_dir);
}

static ffi::Error KleadFft(lrx::gpu::Stream s, ffi::AnyBuffer X, ffi::Result<ffi::AnyBuffer> Y,
                           int64_t nkx, int64_t nky, int64_t nkz, double scale, int64_t forward,
                           std::string_view cubin_dir) {
    return launch(s, 3, X, nullptr, Y, nkx, nky, nkz, scale, forward, cubin_dir);
}

}  // namespace lorrax_ffi::kdft

#define LRX_KDFT_GRID_ATTRS      \
    .Attr<int64_t>("nkx")         \
    .Attr<int64_t>("nky")         \
    .Attr<int64_t>("nkz")         \
    .Attr<double>("scale")

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KDftKconvKleadFfi, lorrax_ffi::kdft::KleadConv,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        LRX_KDFT_GRID_ATTRS
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KDftKfftKleadFfi, lorrax_ffi::kdft::KleadFft,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        LRX_KDFT_GRID_ATTRS
        .Attr<int64_t>("forward")
        .Attr<std::string_view>("cubin_dir"));
