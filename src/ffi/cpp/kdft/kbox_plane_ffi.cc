// kbox_plane_ffi.cc -- the ROCM leg's k-box mode 6: the ISDF route-G pair convolution read from the
// D-plane FFT output (ffi.fft.make_fused_conv_kplane), the port of kconv_mathdx_cuda_ffi.cc's plane
// mode (KConvMathdxPlaneCudaFfi) on the k-box stage (kdft/kbox_plane_kernel.h on cufft/kbox_stage.cuh,
// its line transform the kdft direct DFT under hipRTC).
//
//   lorrax_kdft_kconv_plane   D (nk, G, ns, 2C, ns, P), F (nk, G, P) -> U (nk, C, G*P)   (mode 6)
//
// The operands and attributes of lorrax_mathdx_kconv_plane less mathdx_root, so ffi.fft passes the
// same tables to either leg.  Single arm: a tile of NO output columns (2 ns^2 operand columns each)
// in half a block's 64 KiB of LDS (two blocks per CU), else in all of it; a grid whose single output
// column does not fit is refused (GATE kdft-kconv-plane-tile) and ffi.fft.kdft_plane_refusal keeps
// it on XLA by the same rule.  Gated against the XLA route on the same device
// (ffi.fft.xla_reference; tests/test_kdft_plane_gate.py).
//
// Leg-neutral host code (lrx::gpu, rtc); CMake lists it in the ROCM leg.  Images: one per (k-grid,
// ns, NO, device architecture), disk-cached as kbox_m6_<nkx>x<nky>x<nkz>_ns<ns>_<arch>_<key>.hsaco.
#include <algorithm>
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
#include "../common/leg.h"
#include "../common/rtc_build.h"
#include "../cufft/kbox_stage.cuh"          // the host half: Geometry (the padded bank row)
#include "generated/kbox_stage_src.h"
#include "generated/kbox_plane_src.h"

namespace lorrax_ffi::kbox_plane {

static constexpr long long kLds = LRX_LEG_BLOCK_LDS;   // 64 KiB on ROCM (common/leg.h)
static constexpr int kAxisMax = 40;                     // the stage's direct-DFT line
static constexpr int kThreads = 256;
static constexpr int kNoMax = 16;
static constexpr int kGridCap = 110 * 16;               // grid-stride: 110 CUs (kbox_unfold_ffi.cc)

static ffi::Error fail(const char* gate, const std::string& why,
                       ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
    return ffi::Error(code, std::string("GATE ") + gate + ": " + why);
}

struct PlaneGeo {                                       // kbox_plane_kernel.h, field for field
    long long G, C, P, ncols;
    double scale;
    int pl[4], pr[4], cl[4], cr[4];
};

// Output columns per tile (ffi.fft.kdft_plane_no mirrors it): the most (<= kNoMax) whose 2 ns^2
// padded columns each fit half the LDS, else all of it; 0 when one does not fit.
static int plan_no(int nkx, int nky, int nkz, int ns) {
    const long long per = 2LL * ns * ns * 16 * lrx_kbox::Geometry{nkx, nky, nkz}.rs();
    for (long long budget : {kLds / 2, kLds}) {
        const long long no = std::min<long long>(kNoMax, budget / per);
        if (no >= 1) return static_cast<int>(no);
    }
    return 0;
}

struct Built { rtc::Function fn = nullptr; int no = 0; long long smem = 0; };
using Key = std::tuple<int, int, int, int, int>;        // dev, nkx, nky, nkz, ns
static std::mutex g_mu;
static std::map<Key, Built> g_cache;
static std::map<Key, std::string> g_fail;

static ffi::Error build(int nkx, int nky, int nkz, int ns, std::string_view cache_dir, const Built** out) {
    int dev = 0;
    LORRAX_GPU_CHECK(lrx::gpu::get_device(&dev));
    const Key key{dev, nkx, nky, nkz, ns};
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_cache.find(key); it != g_cache.end()) { *out = &it->second; return ffi::Error::Success(); }
    if (auto it = g_fail.find(key); it != g_fail.end()) return fail("kdft-kconv-plane-build", "(cached failure) " + it->second);
    const int no = plan_no(nkx, nky, nkz, ns);
    if (no == 0) {
        std::ostringstream os;
        os << "got k-grid (" << nkx << "," << nky << "," << nkz << ") ns=" << ns << ", whose output column needs "
           << 2LL * ns * ns * 16 * lrx_kbox::Geometry{nkx, nky, nkz}.rs() << " B (2 ns^2 padded columns); want it in a "
           << "block's " << kLds / 1024 << " KiB of LDS; the router keeps such grids on XLA (ffi.fft.kdft_plane_refusal)";
        return fail("kdft-kconv-plane-tile", os.str(), ffi::ErrorCode::kInvalidArgument);
    }
    const int nk = nkx * nky * nkz;
    const int zr = (no * nk + kThreads - 1) / kThreads;
    std::string arch;
    LORRAX_GPU_CHECK(lrx::gpu::device_arch(dev, &arch));
    rtc::Program prog;
    prog.src = kKernelSrc;
    prog.name = "lrx_kbox_plane";
    prog.kernel = "lrx_kbox_m6";
    prog.headers = {{lorrax_ffi::kbox::kHeaderName, lorrax_ffi::kbox::kHeaderSrc}};
    prog.defs = {rtc::kOptOption, "-std=c++17", rtc::arch_option(arch),
                 "-DLRX_NX=" + std::to_string(nkx), "-DLRX_NY=" + std::to_string(nky), "-DLRX_NZ=" + std::to_string(nkz),
                 "-DLRX_NS=" + std::to_string(ns), "-DLRX_NO=" + std::to_string(no),
                 "-DLRX_NT=" + std::to_string(kThreads), "-DLRX_ZR=" + std::to_string(zr)};
    const std::string key_hex = rtc::hex16(rtc::key(prog, nullptr));
    const std::string dir(cache_dir);
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kbox_m6_" << nkx << "x" << nky << "x" << nkz << "_ns" << ns << "_" << rtc::arch_tag(arch)
             << "_" << key_hex << rtc::kImageExt;
        path = name.str();
    }
    rtc::Image img;
    std::string where, err;
    if (!rtc::build(prog, dir, path, key_hex, &img, &where, &err)) {
        g_fail.emplace(key, where + " -- " + err);
        return fail("kdft-kconv-plane-build", where + " -- " + err);
    }
    Built b;
    b.fn = img.fn;
    b.no = no;
    b.smem = 2LL * ns * ns * no * 16 * lrx_kbox::Geometry{nkx, nky, nkz}.rs();
    if (!img.from_disk)
        std::fprintf(stderr, "[kconv_kbox] " LRX_RTC_API " %s mode=6 kgrid=(%d,%d,%d) ns=%d %s no=%d threads=%d lds=%lld B"
                     " in %.1f ms%s\n", img.rebuilt_bad ? "rebuilt (cached image refused)" : "built", nkx, nky, nkz, ns,
                     arch.c_str(), no, kThreads, b.smem, img.ms, img.stored ? " (stored)" : "");
    *out = &g_cache.emplace(key, b).first->second;
    return ffi::Error::Success();
}

static ffi::Error Plane(lrx::gpu::Stream stream, ffi::AnyBuffer D, ffi::AnyBuffer F, ffi::Result<ffi::AnyBuffer> U,
                        int64_t nkx, int64_t nky, int64_t nkz, double scale, ffi::Span<const int64_t> perm_l,
                        ffi::Span<const int64_t> phase_l, ffi::Span<const int64_t> perm_r,
                        ffi::Span<const int64_t> phase_r, std::string_view cubin_dir) {
    auto bad = [](const std::string& why) { return fail("kdft-kconv-plane-operands", why, ffi::ErrorCode::kInvalidArgument); };
    if (nkx < 1 || nky < 1 || nkz < 1 || nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax)
        return fail("kdft-axis", "want every k axis in [1, " + std::to_string(kAxisMax) + "] (direct DFT); the router "
                    "keeps longer axes on XLA", ffi::ErrorCode::kInvalidArgument);
    const int64_t nk = nkx * nky * nkz;
    const auto C128 = ffi::DataType::C128;
    auto dd = D.dimensions(), fd = F.dimensions(), ud = U->dimensions();
    if (D.element_type() != C128 || F.element_type() != C128 || U->element_type() != C128 || dd.size() != 6 ||
        fd.size() != 3 || ud.size() != 3)
        return bad("want c128 D (nk,G,ns,2C,ns,P), F (nk,G,P), U (nk,C,G*P)");
    const int64_t G = dd[1], ns = dd[2], C2 = dd[3], P = dd[5];
    if (dd[0] != nk || dd[4] != ns || C2 % 2 || fd[0] != nk || fd[1] != G || fd[2] != P || ud[0] != nk ||
        ud[1] != C2 / 2 || ud[2] != G * P || (ns != 1 && ns != 2 && ns != 4))
        return bad("want D (nk,G,ns,2C,ns,P), F (nk,G,P), U (nk,C,G*P) with nk = nkx*nky*nkz, ns in {1,2,4}");
    if (static_cast<int64_t>(perm_l.size()) != ns || static_cast<int64_t>(perm_r.size()) != ns ||
        static_cast<int64_t>(phase_l.size()) != ns || static_cast<int64_t>(phase_r.size()) != ns)
        return bad("want perm_l, perm_r, phase_l, phase_r of ns entries");
    PlaneGeo g{};
    g.G = G; g.C = C2 / 2; g.P = P; g.ncols = g.C * G * P; g.scale = scale;
    for (int64_t i = 0; i < ns; ++i) {
        if (perm_l[i] < 0 || perm_l[i] >= ns || perm_r[i] < 0 || perm_r[i] >= ns || phase_l[i] < 0 || phase_l[i] > 3 ||
            phase_r[i] < 0 || phase_r[i] > 3)
            return bad("perm entries in [0, ns), phase codes in [0, 3]");
        g.pl[i] = static_cast<int>(perm_l[i]); g.pr[i] = static_cast<int>(perm_r[i]);
        g.cl[i] = static_cast<int>(phase_l[i]); g.cr[i] = static_cast<int>(phase_r[i]);
    }
    if (g.ncols == 0) return ffi::Error::Success();
    const Built* b = nullptr;
    FFI_RETURN_IF_ERROR(build(static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz), static_cast<int>(ns),
                              cubin_dir, &b));
    const void* dp = D.untyped_data();
    const void* fp = F.untyped_data();
    void* up = U->untyped_data();
    void* args[] = {(void*)&dp, (void*)&fp, (void*)&up, (void*)&g};
    const long long blocks = std::min<long long>((g.ncols + b->no - 1) / b->no, kGridCap);
    const rtc::Result r = rtc::driver_api().LaunchKernel(b->fn, static_cast<unsigned>(blocks), 1, 1, kThreads, 1, 1,
                                                         static_cast<unsigned>(b->smem), reinterpret_cast<rtc::Stream>(stream),
                                                         args, nullptr);
    if (r != rtc::kOk) return fail("kdft-kconv-plane-launch", LRX_MODULE_API "ModuleLaunchKernel: " + rtc::err_string(r));
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::kbox_plane

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KDftKconvPlaneFfi, lorrax_ffi::kbox_plane::Plane,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()     // D (nk, G, ns, 2C, ns, P)
        .Arg<xla::ffi::AnyBuffer>()     // F (nk, G, P)
        .Ret<xla::ffi::AnyBuffer>()     // U (nk, C, G*P)
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<xla::ffi::Span<const int64_t>>("perm_l")
        .Attr<xla::ffi::Span<const int64_t>>("phase_l")
        .Attr<xla::ffi::Span<const int64_t>>("perm_r")
        .Attr<xla::ffi::Span<const int64_t>>("phase_r")
        .Attr<std::string_view>("cubin_dir"));
