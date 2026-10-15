// kdft_klead_ffi.cc -- the direct-DFT k-leading family: modes 2 and 3 of
// docs/architecture/kconv.md as run-time-compiled kernels (kdft_klead_kernel.h).
//
//   lorrax_kdft_kconv_klead   U = fftn(scale * ifftn(T) * V_R[:, None, :, None, :])   (mode 2)
//   lorrax_kdft_kfft_klead    Y = scale * fftn|ifftn(X) over the leading flat-k axis   (mode 3)
//
// The same operations and operand layouts as the nvidia-mathdx targets
// lorrax_mathdx_kconv_klead / lorrax_mathdx_kfft_klead, without cuFFTDx: a block owns a column
// tile, transforms its k axes in LDS by direct DFT, and touches HBM once each way.  It is the
// ROCM leg's k-leading route (ffi.fft._ROCM_KLEAD_TARGETS); every call is gated against the
// XLA route on the same device (ffi.fft.xla_reference, tests/test_kconv_xla_gate.py).
//
// Leg-neutral: written against lrx::gpu (gpu_runtime.h) and rtc (rtc_build.h) only, so the
// one source would build into the CUDA leg as well; CMake lists it in the ROCM leg, where
// there is no cuFFTDx.
//
// Images: one per (mode, k-grid, precision, device architecture), compiled once per process
// and disk-cached in the router's cubin_dir as kdft_m<mode>_<nkx>x<nky>x<nkz>[_c64]_<arch>_<key>
// <ext> with the common key rule.  A refusal names its GATE.
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>

#include "../common/ffi_helpers.h"
#include "../common/gpu_runtime.h"
#include "../common/rtc_build.h"
#include "generated/kdft_klead_src.h"

namespace lorrax_ffi::kdft {

// Launch shape: a 16 KiB tile and 512 threads, so four blocks (2048 threads, gfx90a's full
// occupancy) share a CU's 64 KiB of LDS and the loads of one block overlap the DFTs of the
// others; the measured record is at the Python owner (ffi.fft, above _ROCM_KLEAD_TARGETS).
// A k-grid whose one-column tile exceeds the whole 64 KiB is refused (GATE kdft-tile).
static constexpr int kThreads = 512;
static constexpr long long kTileBudget = 16 * 1024;
static constexpr long long kLdsMax = 64 * 1024;
static constexpr int kColumnsMax = 64;
// The longest k axis: the direct DFT unrolls N*N FMAs per line and holds the line in registers
// (the cuFFTDx route's cap, ffi.fft.KCONV_AXIS_MAX == KDFT_AXIS_MAX).
static constexpr int kAxisMax = 40;

static ffi::Error fail(const char* gate, const std::string& why,
                       ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
    return ffi::Error(code, std::string("GATE ") + gate + ": " + why);
}

// Columns per block: the largest power of two <= kColumnsMax whose tile fits the budget, but
// at least kColumnsMin while the tile fits the LDS (a row segment of 8 complex128 is 128 B, one
// cache line); 0 when even one column does not fit the LDS.
static constexpr int kColumnsMin = 8;
static int columns_per_block(long long nk, long long cbytes) {
    const long long twiddles = 3LL * 64 * cbytes;               // generous: NX+NY+NZ <= 192
    auto fits = [&](long long c) { return nk * c * cbytes + twiddles <= kLdsMax; };
    if (!fits(1)) return 0;
    int c = kColumnsMax;
    while (c > kColumnsMin && nk * c * cbytes > kTileBudget) c /= 2;
    while (c > 1 && !fits(c)) c /= 2;
    return c;
}

struct Built {
    rtc::Function fn = nullptr;
    int columns = 0;
};
using Key = std::tuple<int, int, int, int, int, int>;           // dev, mode, nkx, nky, nkz, f32
static std::mutex g_mu;
static std::map<Key, Built> g_cache;
static std::map<Key, std::string> g_fail;

static ffi::Error build(int mode, int nkx, int nky, int nkz, bool f32, std::string_view cache_dir,
                        const Built** out) {
    int dev = 0;
    LORRAX_GPU_CHECK(lrx::gpu::get_device(&dev));
    const Key key{dev, mode, nkx, nky, nkz, f32 ? 1 : 0};
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_cache.find(key); it != g_cache.end()) { *out = &it->second; return ffi::Error::Success(); }
    if (auto it = g_fail.find(key); it != g_fail.end()) return fail("kdft-build", "(cached failure) " + it->second);

    std::string arch;
    LORRAX_GPU_CHECK(lrx::gpu::device_arch(dev, &arch));
    const long long nk = 1LL * nkx * nky * nkz;
    const int columns = columns_per_block(nk, f32 ? 8 : 16);
    if (columns == 0)
        return fail("kdft-tile", "got k-grid " + std::to_string(nkx) + "x" + std::to_string(nky) + "x" +
                    std::to_string(nkz) + "; want one column's k tile (nk complex) in a block's " +
                    std::to_string(kLdsMax / 1024) + " KiB of LDS; the router keeps such grids on XLA",
                    ffi::ErrorCode::kInvalidArgument);

    rtc::Program prog;
    prog.src = kKernelSrc;
    prog.name = "lrx_kdft_klead";
    prog.kernel = "lrx_kdft";
    prog.defs = {"-O3", "-std=c++17", rtc::arch_option(arch),
                 "-DLRX_NX=" + std::to_string(nkx), "-DLRX_NY=" + std::to_string(nky),
                 "-DLRX_NZ=" + std::to_string(nkz), "-DLRX_C=" + std::to_string(columns),
                 "-DLRX_NT=" + std::to_string(kThreads), "-DLRX_MODE=" + std::to_string(mode),
                 std::string("-DLRX_F32=") + (f32 ? "1" : "0")};
    const std::string key_hex = rtc::hex16(rtc::key(prog, nullptr));
    const std::string dir(cache_dir);
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kdft_m" << mode << "_" << nkx << "x" << nky << "x" << nkz << (f32 ? "_c64" : "")
             << "_" << rtc::arch_tag(arch) << "_" << key_hex << rtc::kImageExt;
        path = name.str();
    }
    rtc::Image img;
    std::string where, err;
    if (!rtc::build(prog, dir, path, key_hex, &img, &where, &err)) {
        g_fail.emplace(key, where + " -- " + err);
        return fail("kdft-build", where + " -- " + err);
    }
    if (!img.from_disk)
        std::fprintf(stderr, "[kconv_kdft] " LRX_RTC_API " %s mode=%d%s kgrid=(%d,%d,%d) %s C=%d in %.1f ms%s\n",
                     img.rebuilt_bad ? "rebuilt (cached image refused)" : "built", mode, f32 ? " c64" : "",
                     nkx, nky, nkz, arch.c_str(), columns, img.ms, img.stored ? " (stored)" : "");
    Built b;
    b.fn = img.fn;
    b.columns = columns;
    *out = &g_cache.emplace(key, b).first->second;
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
    if (nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax)
        return fail("kdft-axis", "want every k axis <= " + std::to_string(kAxisMax) + " (direct DFT); "
                    "the router keeps longer axes on XLA", ffi::ErrorCode::kInvalidArgument);
    const int64_t nk = nkx * nky * nkz;
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
    const Built* b = nullptr;
    FFI_RETURN_IF_ERROR(build(mode, static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                              f32, cache_dir, &b));
    void* t = Y->untyped_data();
    if (t != X.untyped_data())                                  // not aliased: work on the result
        LORRAX_GPU_CHECK(lrx::gpu::memcpy_async(t, X.untyped_data(), X.size_bytes(),
                                                lrx::gpu::kDeviceToDevice, stream));
    const void* v = V ? V->untyped_data() : nullptr;
    long long ncols_ll = ncols;
    int fwd = forward ? 1 : 0;
    double scale64 = scale;
    float scale32 = static_cast<float>(scale);
    void* args[] = {&t, &v, &ncols_ll, &mx, &nb, &my,
                    f32 ? static_cast<void*>(&scale32) : static_cast<void*>(&scale64), &fwd};
    const long long blocks = (ncols + b->columns - 1) / b->columns;
    if (blocks > 0x7fffffffLL) return bad("want at most 2^31-1 column tiles");
    const rtc::Result r = rtc::driver_api().LaunchKernel(
        b->fn, static_cast<unsigned>(blocks), 1, 1, kThreads, 1, 1, 0,
        reinterpret_cast<rtc::Stream>(stream), args, nullptr);
    if (r != rtc::kOk) return fail("kdft-launch", LRX_MODULE_API "ModuleLaunchKernel: " + rtc::err_string(r));
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
