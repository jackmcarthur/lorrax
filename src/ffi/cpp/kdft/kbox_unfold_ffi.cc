// kbox_unfold_ffi.cc -- the ROCM leg's k-box symmetry unfold: mode 7 of docs/architecture/kconv.md
// (the Sigma k-leading convolution read from the raw-parent Green) as a run-time-compiled kernel on
// the k-box stage (kdft/kbox_unfold_kernel.h on cufft/kbox_stage.cuh, its line transform the kdft
// direct DFT under hipRTC).
//
//   lorrax_kdft_kconv_klead_unfold   U = scale * fftn(ifftn(U G~ U^dagger) * V)[kout]   (mode 7)
//
// The operands, attributes and layouts of lorrax_mathdx_kconv_klead_unfold_xblock (whole spin
// group, x block, conj(G) partner, optional live rows), so ffi.fft passes the same tables to
// either leg.  Single arm only: a tile of whole spin groups (two or more columns) in a block's
// 64 KiB of LDS, sized by the stage's own launch rule (lrx_kbox::kbox_plan); a grid that needs
// the split arm is refused (GATE kdft-kbox-residency) and ffi.fft.kdft_kbox_refusal keeps it on
// XLA by the same rule.  Every call is gated against the XLA route on the same device
// (ffi.fft.xla_reference; tests/test_kdft_kbox_gate.py).
//
// Leg-neutral host code (lrx::gpu, rtc); CMake lists it in the ROCM leg.  Images: one per (k-grid,
// ns, live, device architecture), disk-cached as kbox_m7_<nkx>x<nky>x<nkz>_ns<ns>[_live]_<arch>_<key>.
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
#include "../cufft/kbox_stage.cuh"          // the host half: the k-box launch rule (kbox_plan)
#include "generated/kbox_stage_src.h"       // the same file as text: the kernel's named header
#include "generated/kbox_unfold_src.h"

namespace lorrax_ffi::kbox_unfold {

// A block's LDS on gfx90a (64 KiB per CU, all of it addressable by one workgroup): the stage's
// opt-in budget.  The longest axis: the direct DFT holds a line in registers (KDFT_AXIS_MAX).
static constexpr long long kLds = LRX_LEG_BLOCK_LDS;   // the leg's block LDS (common/leg.h; 64 KiB on ROCM)
static constexpr int kAxisMax = 40;
static constexpr int kGridCap = 110 * 16;   // grid-stride: enough tiles in flight for 110 CUs

static ffi::Error fail(const char* gate, const std::string& why,
                       ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
    return ffi::Error(code, std::string("GATE ") + gate + ": " + why);
}

// The device struct of kbox_unfold_kernel.h, field for field.
struct UnfoldTab {
    const int *row, *trs, *lsrc, *rsrc;
    const double *mph, *nph, *spin;
    long long ml, nl;
    const int* kout;
    int conj_trs;
    long long x0, bx, xs, xn;
};

struct Built {
    rtc::Function fn = nullptr;
    int tr = 0, threads = 0;
    long long smem = 0;
};
using Key = std::tuple<int, int, int, int, int, int>;      // dev, nkx, nky, nkz, ns, live
static std::mutex g_mu;
static std::map<Key, Built> g_cache;
static std::map<Key, std::string> g_fail;

// The single arm's plan (kbox_plan with the mathdx mode-7 arguments: a group of ns^2 columns, one
// operand, c128, a convolution, min_tr 1) or "" when a block cannot hold a group of >= 2 columns.
static std::string plan(int nkx, int nky, int nkz, int ns, lrx_kbox::Plan* kp) {
    const int ss = ns * ns;
    *kp = lrx_kbox::kbox_plan(nkx, nky, nkz, ss, 1, 16, kLds, 2, 1);
    if (kp->arm == 0 && kp->tr * ss >= 2) return "";
    std::ostringstream os;
    os << "got k-grid (" << nkx << "," << nky << "," << nkz << ") ns=" << ns << ", whose spin group of " << ss
       << " padded columns needs " << 16LL * ss * lrx_kbox::Geometry{nkx, nky, nkz}.rs() << " B (two columns at"
       << " ns=1); want it in a block's " << kLds / 1024 << " KiB of LDS (the single arm; the split arm is not"
       << " ported to ROCm); the router keeps such grids on XLA (ffi.fft.kdft_kbox_refusal)";
    return os.str();
}

static ffi::Error build(int nkx, int nky, int nkz, int ns, bool live, std::string_view cache_dir,
                        const Built** out) {
    int dev = 0;
    LORRAX_GPU_CHECK(lrx::gpu::get_device(&dev));
    const Key key{dev, nkx, nky, nkz, ns, live ? 1 : 0};
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_cache.find(key); it != g_cache.end()) { *out = &it->second; return ffi::Error::Success(); }
    if (auto it = g_fail.find(key); it != g_fail.end()) return fail("kdft-kbox-build", "(cached failure) " + it->second);
    lrx_kbox::Plan kp{};
    if (std::string why = plan(nkx, nky, nkz, ns, &kp); !why.empty())
        return fail("kdft-kbox-residency", why, ffi::ErrorCode::kInvalidArgument);
    std::string arch;
    LORRAX_GPU_CHECK(lrx::gpu::device_arch(dev, &arch));
    const int tr = kp.tr * ns * ns;

    rtc::Program prog;
    prog.src = kKernelSrc;
    prog.name = "lrx_kbox_unfold";
    prog.kernel = "lrx_kbox_m7";
    prog.headers = {{lorrax_ffi::kbox::kHeaderName, lorrax_ffi::kbox::kHeaderSrc}};
    prog.defs = {rtc::kOptOption, "-std=c++17", rtc::arch_option(arch),
                 "-DLRX_NX=" + std::to_string(nkx), "-DLRX_NY=" + std::to_string(nky),
                 "-DLRX_NZ=" + std::to_string(nkz), "-DLRX_NS=" + std::to_string(ns),
                 "-DLRX_TR=" + std::to_string(tr), "-DLRX_NT=" + std::to_string(kp.threads),
                 std::string("-DLRX_LIVE=") + (live ? "1" : "0")};
    const std::string key_hex = rtc::hex16(rtc::key(prog, nullptr));
    const std::string dir(cache_dir);
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kbox_m7_" << nkx << "x" << nky << "x" << nkz << "_ns" << ns << (live ? "_live" : "")
             << "_" << rtc::arch_tag(arch) << "_" << key_hex << rtc::kImageExt;
        path = name.str();
    }
    rtc::Image img;
    std::string where, err;
    if (!rtc::build(prog, dir, path, key_hex, &img, &where, &err)) {
        g_fail.emplace(key, where + " -- " + err);
        return fail("kdft-kbox-build", where + " -- " + err);
    }
    if (!img.from_disk)
        std::fprintf(stderr, "[kconv_kbox] " LRX_RTC_API " %s mode=7 kgrid=(%d,%d,%d) ns=%d%s %s tr=%d threads=%d"
                     " lds=%lld B in %.1f ms%s\n", img.rebuilt_bad ? "rebuilt (cached image refused)" : "built",
                     nkx, nky, nkz, ns, live ? " live" : "", arch.c_str(), tr, kp.threads, kp.smem, img.ms,
                     img.stored ? " (stored)" : "");
    Built b;
    b.fn = img.fn;
    b.tr = tr;
    b.threads = kp.threads;
    b.smem = kp.smem;
    *out = &g_cache.emplace(key, b).first->second;
    return ffi::Error::Success();
}

static ffi::Error KleadUnfold(lrx::gpu::Stream stream, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row,
                              ffi::AnyBuffer trs, ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph,
                              ffi::AnyBuffer nph, ffi::AnyBuffer spin, ffi::AnyBuffer kout, ffi::AnyBuffer V,
                              ffi::RemainingArgs rest, ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky,
                              int64_t nkz, double scale, int64_t conj_src, int64_t x0, int64_t bx, int64_t xs,
                              int64_t xn, std::string_view cubin_dir) {
    auto bad = [](const std::string& why) { return fail("kdft-kbox-operands", why, ffi::ErrorCode::kInvalidArgument); };
    const ffi::AnyBuffer* live = nullptr;
    ffi::AnyBuffer live_buf = Gp;
    if (rest.size() > 1) return bad("want at most one operand after V: live s32 [2]");
    if (rest.size() == 1) {
        auto l = rest.get<ffi::AnyBuffer>(0);
        if (l.has_error()) return bad("live: not a buffer");
        live_buf = *l;
        live = &live_buf;
        if (live->element_type() != ffi::DataType::S32 || live->element_count() != 2)
            return bad("want live s32 [2]: the window rows [lo, hi) of a padded pass");
    }
    if (nkx < 1 || nky < 1 || nkz < 1 || nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax)
        return fail("kdft-axis", "want every k axis in [1, " + std::to_string(kAxisMax) + "] (direct DFT); the "
                    "router keeps longer axes on XLA", ffi::ErrorCode::kInvalidArgument);
    const int64_t nk = nkx * nky * nkz;
    auto is = [](ffi::AnyBuffer x, ffi::DataType t, std::initializer_list<int64_t> dims) {
        auto d = x.dimensions();
        return x.element_type() == t && d.size() == dims.size() && std::equal(d.begin(), d.end(), dims.begin());
    };
    const auto gd = Gp.dimensions(), sd = spin.dimensions();
    if (gd.size() != 3 || sd.size() != 3) return bad("want Gp (n_parent, ml, nl) and spin (nk, ns, ns)");
    const int64_t np = gd[0], ml = gd[1], nl = gd[2], ns = sd[1];
    const auto C = ffi::DataType::C128, I = ffi::DataType::S32;
    if ((ns != 1 && ns != 2 && ns != 4) || ml % ns || nl % ns || np < 1 || !is(Gp, C, {np, ml, nl}) ||
        !is(Gt, C, {np, ml, nl}) || !is(row, I, {nk}) || !is(trs, I, {nk}) || !is(lsrc, I, {nk, ml}) ||
        !is(rsrc, I, {nk, nl}) || !is(mph, C, {nk, ml}) || !is(nph, C, {nk, nl}) || !is(spin, C, {nk, ns, ns}) ||
        !is(V, C, {nk, ml / ns, nl / ns}) || !is(kout, I, {nk}))
        return bad("want c128 Gp=Gt (np,ml,nl); s32 row,trs,kout (nk), lsrc (nk,ml), rsrc (nk,nl); c128 mph "
                   "(nk,ml), nph (nk,nl), spin (nk,ns,ns), V (nk,ml/ns,nl/ns)");
    if (bx < 0 || (bx > 0 && (xn < 1 || x0 < 0 || xs < x0 + bx || x0 >= ml / ns)))
        return bad("want the x block 0 <= x0 < ml/ns, bx >= 1, x0 + bx <= xs, xn >= 1");
    const int64_t nx = bx > 0 ? xn * bx : ml / ns, my = nl / ns;
    const auto ud = U->dimensions();
    if (!(U->element_type() == C && ud.size() == 5 && ud[0] >= 1 && ud[1] == ns && ud[2] == nx && ud[3] == ns &&
          ud[4] == my))
        return bad("want U (n_out, ns, ml/ns or xn*bx, ns, nl/ns) c128");
    const int64_t pairs = nx * my;
    if (pairs == 0) return ffi::Error::Success();
    const Built* b = nullptr;
    FFI_RETURN_IF_ERROR(build(static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                              static_cast<int>(ns), live != nullptr, cubin_dir, &b));
    UnfoldTab t{static_cast<const int*>(row.untyped_data()), static_cast<const int*>(trs.untyped_data()),
                static_cast<const int*>(lsrc.untyped_data()), static_cast<const int*>(rsrc.untyped_data()),
                static_cast<const double*>(mph.untyped_data()), static_cast<const double*>(nph.untyped_data()),
                static_cast<const double*>(spin.untyped_data()), ml, nl,
                static_cast<const int*>(kout.untyped_data()), conj_src ? 2 : 0, x0, bx, xs, xn};
    const void* gpp = Gp.untyped_data();
    const void* gtp = Gt.untyped_data();
    const void* vp = V.untyped_data();
    void* up = U->untyped_data();
    double sc = scale;
    const void* livep = live ? live->untyped_data() : nullptr;
    void* args[] = {(void*)&gpp, (void*)&gtp, (void*)&vp, (void*)&up, (void*)&t, (void*)&sc, (void*)&livep};
    const long long ncols = pairs * ns * ns;
    const long long blocks = std::min<long long>((ncols + b->tr - 1) / b->tr, kGridCap);
    const rtc::Result r = rtc::driver_api().LaunchKernel(
        b->fn, static_cast<unsigned>(blocks), 1, 1, b->threads, 1, 1, static_cast<unsigned>(b->smem),
        reinterpret_cast<rtc::Stream>(stream), args, nullptr);
    if (r != rtc::kOk) return fail("kdft-kbox-launch", LRX_MODULE_API "ModuleLaunchKernel: " + rtc::err_string(r));
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::kbox_unfold

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KDftKconvKleadUnfoldFfi, lorrax_ffi::kbox_unfold::KleadUnfold,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()     // Gp
        .Arg<xla::ffi::AnyBuffer>()     // Gt
        .Arg<xla::ffi::AnyBuffer>()     // row
        .Arg<xla::ffi::AnyBuffer>()     // trs
        .Arg<xla::ffi::AnyBuffer>()     // lsrc
        .Arg<xla::ffi::AnyBuffer>()     // rsrc
        .Arg<xla::ffi::AnyBuffer>()     // mph
        .Arg<xla::ffi::AnyBuffer>()     // nph
        .Arg<xla::ffi::AnyBuffer>()     // spin
        .Arg<xla::ffi::AnyBuffer>()     // kout
        .Arg<xla::ffi::AnyBuffer>()     // V (W_R)
        .RemainingArgs()                // live (optional)
        .Ret<xla::ffi::AnyBuffer>()     // U
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("conj_src")
        .Attr<int64_t>("x0")
        .Attr<int64_t>("bx")
        .Attr<int64_t>("xs")
        .Attr<int64_t>("xn")
        .Attr<std::string_view>("cubin_dir"));
