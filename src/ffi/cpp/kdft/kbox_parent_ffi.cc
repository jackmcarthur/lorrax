// kbox_parent_ffi.cc -- the ROCM leg's k-box mode 1: the ISDF parent-load pair convolution
// (ffi.fft.make_fused_conv_kparent; isdf.core's C_q, the zeta_fit.CCT stage), the port of
// kconv_mathdx_cuda_ffi.cc's parent mode (KConvMathdxParentCudaFfi) on the k-box stage
// (kdft/kbox_parent_kernel.h on cufft/kbox_stage.cuh, its line transform the kdft direct DFT under hipRTC).
//
//   lorrax_kdft_kconv_parent   D_l, D_r (n_parent, ns, mu, ns, nu) + the ten typed tables -> U (nk, mu, nu)
//
// The operands and attributes of lorrax_mathdx_kconv_parent less mathdx_root, so ffi.fft passes the
// same tables (isdf.core._parent_conv_tables_local) and the same centroid_major input layout to either
// leg.  Mode 6's single-arm tile (kbox_plane_ffi.cc plan_no, the same 2 ns^2 operand columns per output
// column): NO output columns in half a block's 64 KiB of LDS, else in all of it; a grid whose single
// output column does not fit is refused (GATE kdft-kconv-parent-tile) and ffi.fft.kdft_parent_refusal
// keeps it on XLA by the same rule.  Gated against the XLA route on the same device
// (ffi.fft.xla_reference; tests/test_kdft_parent_gate.py).
//
// Leg-neutral host code (lrx::gpu, rtc); CMake lists it in the ROCM leg.  Images: one per (k-grid, ns,
// centroid_major, NO, device architecture), disk-cached as kbox_m1_<nkx>x<nky>x<nkz>_ns<ns>_cm<0|1>_<arch>_<key>.hsaco.
#include <algorithm>
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
#include "../common/leg.h"
#include "../common/rtc_build.h"
#include "../cufft/kbox_stage.cuh"          // the host half: Geometry (the padded bank row)
#include "generated/kbox_stage_src.h"
#include "generated/kbox_parent_src.h"

namespace lorrax_ffi::kbox_parent {

static constexpr long long kLds = LRX_LEG_BLOCK_LDS;   // 64 KiB on ROCM (common/leg.h)
static constexpr int kAxisMax = 40;                     // the stage's direct-DFT line
static constexpr int kThreads = 256;
static constexpr int kNoMax = 16;
static constexpr int kGridCap = 110 * 16;               // grid-stride: 110 CUs (kbox_unfold_ffi.cc)

static ffi::Error fail(const char* gate, const std::string& why,
                       ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
    return ffi::Error(code, std::string("GATE ") + gate + ": " + why);
}

struct ParentGeo {                                      // kbox_parent_kernel.h, field for field
    const int *irr, *sym, *left, *right, *trs;
    const double *L, *R, *q;
    const void *coef_l, *coef_r;
    long long mu, nu, ncols;
    double scale;
    int pl[4], pr[4], cl[4], cr[4];
};

// Output columns per tile (ffi.fft.kdft_parent_no mirrors it; kbox_plane_ffi.cc plan_no): the most
// (<= kNoMax) whose 2 ns^2 padded columns each fit half the LDS, else all of it; 0 when one does not fit.
static int plan_no(int nkx, int nky, int nkz, int ns) {
    const long long per = 2LL * ns * ns * 16 * lrx_kbox::Geometry{nkx, nky, nkz}.rs();
    for (long long budget : {kLds / 2, kLds}) {
        const long long no = std::min<long long>(kNoMax, budget / per);
        if (no >= 1) return static_cast<int>(no);
    }
    return 0;
}

struct Built { rtc::Function fn = nullptr; int no = 0; long long smem = 0; };
using Key = std::tuple<int, int, int, int, int, int>;   // dev, nkx, nky, nkz, ns, centroid_major
static std::mutex g_mu;
static std::map<Key, Built> g_cache;
static std::map<Key, std::string> g_fail;

static ffi::Error build(int nkx, int nky, int nkz, int ns, int cm, std::string_view cache_dir, const Built** out) {
    int dev = 0;
    LORRAX_GPU_CHECK(lrx::gpu::get_device(&dev));
    const Key key{dev, nkx, nky, nkz, ns, cm};
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_cache.find(key); it != g_cache.end()) { *out = &it->second; return ffi::Error::Success(); }
    if (auto it = g_fail.find(key); it != g_fail.end()) return fail("kdft-kconv-parent-build", "(cached failure) " + it->second);
    const int no = plan_no(nkx, nky, nkz, ns);
    if (no == 0) {
        std::ostringstream os;
        os << "got k-grid (" << nkx << "," << nky << "," << nkz << ") ns=" << ns << ", whose output column needs "
           << 2LL * ns * ns * 16 * lrx_kbox::Geometry{nkx, nky, nkz}.rs() << " B (2 ns^2 padded columns); want it in a "
           << "block's " << kLds / 1024 << " KiB of LDS; the router keeps such grids on XLA (ffi.fft.kdft_parent_refusal)";
        return fail("kdft-kconv-parent-tile", os.str(), ffi::ErrorCode::kInvalidArgument);
    }
    const int nk = nkx * nky * nkz;
    const int zr = (no * nk + kThreads - 1) / kThreads;
    std::string arch;
    LORRAX_GPU_CHECK(lrx::gpu::device_arch(dev, &arch));
    rtc::Program prog;
    prog.src = kKernelSrc;
    prog.name = "lrx_kbox_parent";
    prog.kernel = "lrx_kbox_m1";
    prog.headers = {{lorrax_ffi::kbox::kHeaderName, lorrax_ffi::kbox::kHeaderSrc}};
    prog.defs = {rtc::kOptOption, "-std=c++17", rtc::arch_option(arch),
                 "-DLRX_NX=" + std::to_string(nkx), "-DLRX_NY=" + std::to_string(nky), "-DLRX_NZ=" + std::to_string(nkz),
                 "-DLRX_NS=" + std::to_string(ns), "-DLRX_NO=" + std::to_string(no),
                 "-DLRX_NT=" + std::to_string(kThreads), "-DLRX_ZR=" + std::to_string(zr),
                 "-DLRX_CM=" + std::to_string(cm)};
    const std::string key_hex = rtc::hex16(rtc::key(prog, nullptr));
    const std::string dir(cache_dir);
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kbox_m1_" << nkx << "x" << nky << "x" << nkz << "_ns" << ns << "_cm" << cm << "_"
             << rtc::arch_tag(arch) << "_" << key_hex << rtc::kImageExt;
        path = name.str();
    }
    rtc::Image img;
    std::string where, err;
    if (!rtc::build(prog, dir, path, key_hex, &img, &where, &err)) {
        g_fail.emplace(key, where + " -- " + err);
        return fail("kdft-kconv-parent-build", where + " -- " + err);
    }
    Built b;
    b.fn = img.fn;
    b.no = no;
    b.smem = 2LL * ns * ns * no * 16 * lrx_kbox::Geometry{nkx, nky, nkz}.rs();
    if (!img.from_disk)
        std::fprintf(stderr, "[kconv_kbox] " LRX_RTC_API " %s mode=1 kgrid=(%d,%d,%d) ns=%d cm=%d %s no=%d threads=%d "
                     "lds=%lld B in %.1f ms%s\n", img.rebuilt_bad ? "rebuilt (cached image refused)" : "built", nkx, nky,
                     nkz, ns, cm, arch.c_str(), no, kThreads, b.smem, img.ms, img.stored ? " (stored)" : "");
    *out = &g_cache.emplace(key, b).first->second;
    return ffi::Error::Success();
}

static ffi::Error Parent(lrx::gpu::Stream stream, ffi::AnyBuffer Dl, ffi::AnyBuffer Dr, ffi::AnyBuffer irr,
                         ffi::AnyBuffer sym, ffi::AnyBuffer left, ffi::AnyBuffer right, ffi::AnyBuffer L,
                         ffi::AnyBuffer R, ffi::AnyBuffer q, ffi::AnyBuffer trs, ffi::AnyBuffer coef_l,
                         ffi::AnyBuffer coef_r, ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz,
                         double scale, int64_t centroid_major, ffi::Span<const int64_t> perm_l,
                         ffi::Span<const int64_t> phase_l, ffi::Span<const int64_t> perm_r,
                         ffi::Span<const int64_t> phase_r, std::string_view cubin_dir) {
    auto bad = [](const std::string& why) { return fail("kdft-kconv-parent-operands", why, ffi::ErrorCode::kInvalidArgument); };
    if (nkx < 1 || nky < 1 || nkz < 1 || nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax)
        return fail("kdft-axis", "want every k axis in [1, " + std::to_string(kAxisMax) + "] (direct DFT); the router "
                    "keeps longer axes on XLA", ffi::ErrorCode::kInvalidArgument);
    const int64_t nk = nkx * nky * nkz;
    const auto C128 = ffi::DataType::C128;
    auto ad = Dl.dimensions(), bd = Dr.dimensions(), ud = U->dimensions();
    if (Dl.element_type() != C128 || Dr.element_type() != C128 || U->element_type() != C128 || ad.size() != 5 ||
        bd.size() != 5 || ud.size() != 3 || !std::equal(ad.begin(), ad.end(), bd.begin()))
        return bad("want equal c128 D_l, D_r (n_parent, ns, mu, ns, nu) and c128 U (nk, mu, nu)");
    const int64_t np_ = ad[0], ns = ad[1], mu = ad[2], nu = ad[4];
    if (ad[3] != ns || (ns != 1 && ns != 2 && ns != 4) || ud[0] != nk || ud[1] != mu || ud[2] != nu || mu < 1 || nu < 1)
        return bad("want D (n_parent, ns, mu, ns, nu) with ns in {1,2,4} and U (nk, mu, nu), nk = nkx*nky*nkz");
    auto shape = [](ffi::AnyBuffer x, ffi::DataType t, std::vector<int64_t> dims) {
        auto d = x.dimensions();
        return x.element_type() == t && d.size() == dims.size() && std::equal(d.begin(), d.end(), dims.begin());
    };
    auto ld = left.dimensions();
    const int64_t ops = ld.size() == 2 ? ld[0] : 0;
    const auto S32 = ffi::DataType::S32, F64 = ffi::DataType::F64;
    if ((centroid_major != 0 && centroid_major != 1) || ops < 1 || !shape(irr, S32, {nk}) || !shape(sym, S32, {nk}) ||
        !shape(trs, S32, {nk}) || !shape(left, S32, {ops, mu}) || !shape(right, S32, {ops, nu}) ||
        !shape(L, F64, {ops, mu, 3}) || !shape(R, F64, {ops, nu, 3}) || !shape(q, F64, {np_, 3}) ||
        !shape(coef_l, C128, {nk, ns * ns, ns * ns}) || !shape(coef_r, C128, {nk, ns * ns, ns * ns}))
        return bad("typed parent tables: want irr, sym, trs (nk,) s32, left (ops, mu), right (ops, nu) s32, "
                   "L (ops, mu, 3), R (ops, nu, 3), q (n_parent, 3) f64, coef_l, coef_r (nk, ns^2, ns^2) c128, "
                   "centroid_major in {0, 1}");
    if (static_cast<int64_t>(perm_l.size()) != ns || static_cast<int64_t>(perm_r.size()) != ns ||
        static_cast<int64_t>(phase_l.size()) != ns || static_cast<int64_t>(phase_r.size()) != ns)
        return bad("want perm_l, perm_r, phase_l, phase_r of ns entries");
    ParentGeo g{};
    g.irr = static_cast<const int*>(irr.untyped_data());
    g.sym = static_cast<const int*>(sym.untyped_data());
    g.left = static_cast<const int*>(left.untyped_data());
    g.right = static_cast<const int*>(right.untyped_data());
    g.trs = static_cast<const int*>(trs.untyped_data());
    g.L = static_cast<const double*>(L.untyped_data());
    g.R = static_cast<const double*>(R.untyped_data());
    g.q = static_cast<const double*>(q.untyped_data());
    g.coef_l = coef_l.untyped_data();
    g.coef_r = coef_r.untyped_data();
    g.mu = mu; g.nu = nu; g.ncols = mu * nu; g.scale = scale;
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
                              static_cast<int>(centroid_major), cubin_dir, &b));
    const void* ap = Dl.untyped_data();
    const void* bp = Dr.untyped_data();
    void* up = U->untyped_data();
    void* args[] = {(void*)&ap, (void*)&bp, (void*)&up, (void*)&g};
    const long long blocks = std::min<long long>((g.ncols + b->no - 1) / b->no, kGridCap);
    const rtc::Result r = rtc::driver_api().LaunchKernel(b->fn, static_cast<unsigned>(blocks), 1, 1, kThreads, 1, 1,
                                                         static_cast<unsigned>(b->smem), reinterpret_cast<rtc::Stream>(stream),
                                                         args, nullptr);
    if (r != rtc::kOk) return fail("kdft-kconv-parent-launch", LRX_MODULE_API "ModuleLaunchKernel: " + rtc::err_string(r));
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::kbox_parent

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KDftKconvParentFfi, lorrax_ffi::kbox_parent::Parent,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()     // D_l (n_parent, ns, mu, ns, nu), physical layout per centroid_major
        .Arg<xla::ffi::AnyBuffer>()     // D_r
        .Arg<xla::ffi::AnyBuffer>()     // irr (nk,) s32
        .Arg<xla::ffi::AnyBuffer>()     // sym (nk,) s32
        .Arg<xla::ffi::AnyBuffer>()     // left (ops, mu) s32
        .Arg<xla::ffi::AnyBuffer>()     // right (ops, nu) s32
        .Arg<xla::ffi::AnyBuffer>()     // L (ops, mu, 3) f64
        .Arg<xla::ffi::AnyBuffer>()     // R (ops, nu, 3) f64
        .Arg<xla::ffi::AnyBuffer>()     // q (n_parent, 3) f64
        .Arg<xla::ffi::AnyBuffer>()     // trs (nk,) s32
        .Arg<xla::ffi::AnyBuffer>()     // coef_l (nk, ns^2, ns^2) c128
        .Arg<xla::ffi::AnyBuffer>()     // coef_r
        .Ret<xla::ffi::AnyBuffer>()     // U (nk, mu, nu)
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("centroid_major")
        .Attr<xla::ffi::Span<const int64_t>>("perm_l")
        .Attr<xla::ffi::Span<const int64_t>>("phase_l")
        .Attr<xla::ffi::Span<const int64_t>>("perm_r")
        .Attr<xla::ffi::Span<const int64_t>>("phase_r")
        .Attr<std::string_view>("cubin_dir"));
