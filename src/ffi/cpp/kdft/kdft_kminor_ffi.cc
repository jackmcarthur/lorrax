// kdft_kminor_ffi.cc -- the kdft k-minor family: modes 4 and 5 of
// docs/architecture/kconv.md as run-time-compiled kernels (kdft_kminor_kernel.h).
//
//   lorrax_kdft_kconv_kminor  U = fftn(scale * ifftn(X) * K[None, :, :, None, None, :])  (mode 4)
//                             X (d0,d1,d2,d3,d4,nk), K (d1,d2,nk); U in X's layout
//                             (out_layout 0) or (d0,nk,d3,d1,d4,d2) (out_layout 1)
//   lorrax_kdft_kfft_kminor   Y = scale * fftn|ifftn(X) over the trailing flat-k axis      (mode 5)
//
// The operations and operand layouts of the nvidia-mathdx targets lorrax_mathdx_kconv_kminor /
// lorrax_mathdx_kfft_kminor, without cuFFTDx: a block owns C consecutive rows (one contiguous
// HBM run), transforms their k axes in LDS (kdft_core.h Stockham DFT) and touches HBM once each way.  It is
// the ROCM leg's k-minor route (ffi.fft._ROCM_KLEAD_TARGETS, rows kconv_kminor / kfft_kminor);
// every call is gated against the XLA route on the same device (ffi.fft.xla_reference).
//
// Leg-neutral, as kdft_klead_ffi.cc: written against lrx::gpu and rtc only; CMake lists it in
// the ROCM leg.  Images: one per (mode, out_layout, k-grid, precision, device architecture),
// disk-cached in cubin_dir as kdftm_m<mode>[o1]_<nkx>x<nky>x<nkz>[_c64]_<arch>_<key><ext>.
// Like the CUDA modes 4/5 there is no split arm: one row's k tile must fit a block's LDS
// (GATE kdft-kminor-tile).
#include <climits>
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
#include "generated/kdft_klead_src.h"   // kCoreSrc, kPlanSrc: the shared DFT core
#include "generated/kdft_kminor_src.h"

namespace lorrax_ffi::kdft {
namespace {

// Launch shape, from a sweep of tile budget {8, 16, 32, 60} KiB x {256, 512, 1024} threads on
// one MI250X GCD (13 BSE tiles, modes 4 (both layouts) and 5, k-grids 8x8x1 .. 8x8x8): a 32 KiB
// budget and 512 threads is within ~20% of each tile's best and beats the k-leading family's
// 16 KiB / 512 on all 13 (mode 4 out_layout 1 at 8x8x8 6.22 -> 3.83 ms, whose permuted store
// wants long row runs).  The record is at the Python owner (ffi.fft, above _ROCM_KLEAD_TARGETS).
// Rows are contiguous in HBM, so unlike the k-leading tile there is no minimum row count for
// coalescing.  C is odd (the transposing load/store's bank rule): P|1 for the largest power of
// two P <= kRowsMax with nk*(P|1) complex in the budget -- the swept footprint -- else the
// largest odd C that fits the LDS.
constexpr int kThreads = 512;
constexpr long long kTileBudget = 32 * 1024;
constexpr long long kLdsMax = 64 * 1024;
constexpr int kRowsMax = 64;
constexpr int kAxisMax = 128;                // ffi.fft.KDFT_KMINOR_AXIS_MAX: the gated range

ffi::Error fail(const char* gate, const std::string& why,
                ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
    return ffi::Error(code, std::string("GATE ") + gate + ": " + why);
}

// The LDS one block takes: the padded tile, the twiddles, mode 4's per-row tables.
long long lds_bytes(long long nk, int c, int ntw, long long cbytes) {
    return nk * c * cbytes + ntw * cbytes + c * 12LL;
}
// Rows C and the odd LDS width W (C or C + 1) of one block: P|1 for the largest power of two
// P <= kRowsMax with nk*(P|1) complex in the budget -- the swept footprint -- or less to fit the
// LDS.  Every row real (C = W = P|1) for mode 5 and out_layout 0; out_layout 1 takes C = P,
// W = P|1 (a zero pad column), so its permuted store writes aligned runs of P rows: measured
// (one GCD) 8x8x1 3.98 -> 3.11, 6x6x6 4.87 -> 4.09, 8x8x8 4.45 -> 3.99 ms, while odd rows win for
// mode 5 and layout 0 (8x8x8: 1.73 -> 1.34 and 3.36 -> 2.53 ms).
struct Rows { int c = 0, w = 0; };
Rows rows_per_block(long long nk, int ntw, long long cbytes, bool pow2) {
    auto fits = [&](int w) { return lds_bytes(nk, w, ntw, cbytes) <= kLdsMax; };
    if (!fits(1)) return {};
    int p = kRowsMax;
    while (p > 1 && nk * (p | 1) * cbytes > kTileBudget) p /= 2;
    while (p > 1 && !fits(p | 1)) p /= 2;
    return pow2 ? Rows{p, p | 1} : Rows{p | 1, p | 1};
}

struct Built {
    rtc::Function fn = nullptr;
    int rows = 0;
};
using Key = std::tuple<int, int, int, int, int, int, int>;     // dev, mode, out, nkx, nky, nkz, f32
std::mutex g_mu;
std::map<Key, Built> g_cache;
std::map<Key, std::string> g_fail;

ffi::Error build(int mode, int out, int nkx, int nky, int nkz, bool f32, std::string_view cache_dir,
                 const Built** res) {
    int dev = 0;
    LORRAX_GPU_CHECK(lrx::gpu::get_device(&dev));
    const Key key{dev, mode, out, nkx, nky, nkz, f32 ? 1 : 0};
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_cache.find(key); it != g_cache.end()) { *res = &it->second; return ffi::Error::Success(); }
    if (auto it = g_fail.find(key); it != g_fail.end()) return fail("kdft-build", "(cached failure) " + it->second);

    std::string arch;
    LORRAX_GPU_CHECK(lrx::gpu::device_arch(dev, &arch));
    const long long nk = 1LL * nkx * nky * nkz;
    const bool pow2 = out == 1;                                 // aligned row runs for the permuted store
    const Rows rw = rows_per_block(nk, nkx + nky + nkz, f32 ? 8 : 16, pow2);
    const int rows = rw.c;
    if (rows == 0)
        return fail("kdft-kminor-tile", "got k-grid " + std::to_string(nkx) + "x" + std::to_string(nky) + "x" +
                    std::to_string(nkz) + "; want one row's k tile (nk complex) in a block's " +
                    std::to_string(kLdsMax / 1024) + " KiB of LDS (modes 4/5 have no split arm); "
                    "the router keeps such grids on XLA", ffi::ErrorCode::kInvalidArgument);

    rtc::Program prog;
    prog.src = kKminorSrc;
    prog.name = "lrx_kdft_kminor";
    prog.kernel = "lrx_kdft_kminor";
    prog.headers = {{"kdft_core.h", kCoreSrc}, {"kdft_plan.h", kPlanSrc}};
    prog.defs = {"-O3", "-std=c++17", rtc::arch_option(arch),
                 "-DLRX_NX=" + std::to_string(nkx), "-DLRX_NY=" + std::to_string(nky),
                 "-DLRX_NZ=" + std::to_string(nkz), "-DLRX_C=" + std::to_string(rows),
                 "-DLRX_W=" + std::to_string(rw.w),
                 "-DLRX_NT=" + std::to_string(kThreads), "-DLRX_MODE=" + std::to_string(mode),
                 "-DLRX_OUT=" + std::to_string(out), std::string("-DLRX_F32=") + (f32 ? "1" : "0")};
    const std::string key_hex = rtc::hex16(rtc::key(prog, nullptr));
    const std::string dir(cache_dir);
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kdftm_m" << mode << (out ? "o1" : "") << "_" << nkx << "x" << nky << "x" << nkz
             << (f32 ? "_c64" : "") << "_" << rtc::arch_tag(arch) << "_" << key_hex << rtc::kImageExt;
        path = name.str();
    }
    rtc::Image img;
    std::string where, err;
    if (!rtc::build(prog, dir, path, key_hex, &img, &where, &err)) {
        g_fail.emplace(key, where + " -- " + err);
        return fail("kdft-build", where + " -- " + err);
    }
    if (!img.from_disk)
        std::fprintf(stderr, "[kconv_kdft] " LRX_RTC_API " %s mode=%d%s%s kgrid=(%d,%d,%d) %s C=%d in %.1f ms%s\n",
                     img.rebuilt_bad ? "rebuilt (cached image refused)" : "built", mode, out ? " out1" : "",
                     f32 ? " c64" : "", nkx, nky, nkz, arch.c_str(), rows, img.ms, img.stored ? " (stored)" : "");
    Built b;
    b.fn = img.fn;
    b.rows = rows;
    *res = &g_cache.emplace(key, b).first->second;
    return ffi::Error::Success();
}

// The shared launch.  X (rows, nk) flat (mode 5) or (d0..d4, nk) (mode 4, with K (d1, d2, nk));
// the kernel reads X and writes Y, which may be the same buffer (aliased) or not.
ffi::Error launch(lrx::gpu::Stream stream, int mode, ffi::AnyBuffer X, const ffi::AnyBuffer* K,
                  ffi::Result<ffi::AnyBuffer> Y, int64_t nkx, int64_t nky, int64_t nkz, double scale,
                  int64_t forward, int64_t out_layout, std::string_view cache_dir) {
    auto bad = [](const std::string& why) {
        return fail("kdft-operands", why, ffi::ErrorCode::kInvalidArgument);
    };
    const ffi::DataType dt = X.element_type();
    if (dt != ffi::DataType::C128 && dt != ffi::DataType::C64)
        return bad("want a complex128 or complex64 operand");
    if (Y->element_type() != dt || (K && K->element_type() != dt))
        return bad("want every operand in one precision (the router never casts)");
    if (nkx < 1 || nky < 1 || nkz < 1) return bad("want three positive k axes");
    if (nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax)
        return fail("kdft-axis", "want every k axis <= " + std::to_string(kAxisMax) + " (the gated range); "
                    "the router keeps longer axes on XLA", ffi::ErrorCode::kInvalidArgument);
    const int64_t nk = nkx * nky * nkz;
    const auto xd = X.dimensions(), yd = Y->dimensions();
    if (xd.size() == 0 || xd[xd.size() - 1] != nk)
        return bad("want the trailing axis of length nk = " + std::to_string(nk));
    const int64_t rows = static_cast<int64_t>(X.element_count()) / nk;
    int d[5] = {1, 1, 1, 1, 1};
    if (mode == 4) {
        const auto kd = K->dimensions();
        if (xd.size() != 6 || kd.size() != 3 || kd[0] != xd[1] || kd[1] != xd[2] || kd[2] != nk ||
            (out_layout != 0 && out_layout != 1))
            return bad("want X (d0,d1,d2,d3,d4,nk), K (d1,d2,nk), out_layout 0|1");
        const int64_t want1[6] = {xd[0], nk, xd[3], xd[1], xd[4], xd[2]};
        bool ok = yd.size() == 6;
        for (int i = 0; ok && i < 6; ++i) ok = yd[i] == (out_layout == 0 ? xd[i] : want1[i]);
        if (!ok) return bad("want the result shaped by out_layout: X's shape (0) or (d0,nk,d3,d1,d4,d2) (1)");
        for (int i = 0; i < 5; ++i) d[i] = static_cast<int>(xd[i]);
    } else if (static_cast<int64_t>(Y->element_count()) != rows * nk) {
        return bad("want the result shaped as the operand");
    }
    if (rows == 0) return ffi::Error::Success();
    if (rows > INT_MAX) return bad("want at most 2^31-1 rows of nk");

    const bool f32 = dt == ffi::DataType::C64;
    const int out = mode == 4 ? static_cast<int>(out_layout) : 0;
    const Built* b = nullptr;
    FFI_RETURN_IF_ERROR(build(mode, out, static_cast<int>(nkx), static_cast<int>(nky),
                              static_cast<int>(nkz), f32, cache_dir, &b));
    const void* x = X.untyped_data();
    const void* kp = K ? K->untyped_data() : nullptr;
    void* y = Y->untyped_data();
    int rows_i = static_cast<int>(rows), fwd = forward ? 1 : 0;
    double scale64 = scale;
    float scale32 = static_cast<float>(scale);
    void* args[] = {&x, &kp, &y, &rows_i, &d[1], &d[2], &d[3], &d[4],
                    f32 ? static_cast<void*>(&scale32) : static_cast<void*>(&scale64), &fwd};
    const long long blocks = (rows + b->rows - 1) / b->rows;
    const rtc::Result r = rtc::driver_api().LaunchKernel(
        b->fn, static_cast<unsigned>(blocks), 1, 1, kThreads, 1, 1, 0,
        reinterpret_cast<rtc::Stream>(stream), args, nullptr);
    if (r != rtc::kOk) return fail("kdft-launch", LRX_MODULE_API "ModuleLaunchKernel: " + rtc::err_string(r));
    return ffi::Error::Success();
}

}  // namespace

static ffi::Error KminorConv(lrx::gpu::Stream s, ffi::AnyBuffer X, ffi::AnyBuffer K,
                             ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz,
                             double scale, int64_t out_layout, std::string_view cubin_dir) {
    return launch(s, 4, X, &K, U, nkx, nky, nkz, scale, 0, out_layout, cubin_dir);
}

static ffi::Error KminorFft(lrx::gpu::Stream s, ffi::AnyBuffer X, ffi::Result<ffi::AnyBuffer> Y,
                            int64_t nkx, int64_t nky, int64_t nkz, double scale, int64_t forward,
                            std::string_view cubin_dir) {
    return launch(s, 5, X, nullptr, Y, nkx, nky, nkz, scale, forward, 0, cubin_dir);
}

}  // namespace lorrax_ffi::kdft

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KDftKconvKminorFfi, lorrax_ffi::kdft::KminorConv,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("out_layout")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KDftKfftKminorFfi, lorrax_ffi::kdft::KminorFft,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("forward")
        .Attr<std::string_view>("cubin_dir"));
