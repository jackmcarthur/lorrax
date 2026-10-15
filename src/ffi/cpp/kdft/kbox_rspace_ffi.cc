// kbox_rspace_ffi.cc -- the ROCM leg's k-box R-space modes of docs/architecture/kconv.md as
// run-time-compiled kernels on the k-box stage (kdft/kbox_rspace_kernel.h on cufft/kbox_stage.cuh,
// its line transform the kdft direct DFT under hipRTC):
//
//   lorrax_kdft_kfft_klead_unfold    Y = scale * ifftn(U W~ Ur^dagger)               (mode 9)
//   lorrax_kdft_kconv_chi_unfold     acc += alpha * sum_ab conj(Gc') Gv' (+ c.c.)    (mode 11)
//   lorrax_kdft_kconv_chi_vertex     acc[ch] += the four-current channel traces      (mode 11, LRX_VTX)
//
// The operands, attributes and layouts of the mathdx targets lorrax_mathdx_kfft_klead_unfold /
// _kconv_chi_unfold / _kconv_chi_vertex less the CUDA-only attributes (mathdx_root, and
// scratch_bytes: no split arm here), so ffi.fft passes the same tables to either leg.  Single arm
// only: a tile of whole spin groups (mode 9) or whole pairs (mode 11, 2*ns^2 columns) in a block's
// 64 KiB of LDS, sized by the stage's own launch rule (lrx_kbox::kbox_plan with the mathdx
// family's arguments for the mode).  A grid that needs the split arm is refused by name (GATE
// kdft-kbox-residency) and ffi.fft.kdft_kbox_chi_refusal / kdft_kbox_wedge_refusal keep it on XLA
// by the same rule.  Every call is gated against the XLA route on the same device
// (ffi.fft.xla_reference; tests/test_kdft_kbox_gate.py).
//
// Leg-neutral host code (lrx::gpu, rtc); CMake lists it in the ROCM leg.  Images: one per (mode,
// k-grid, widths, variant, live, device architecture), disk-cached as
// kbox_m<mode>_<nkx>x<nky>x<nkz>_ns<ns>[x<nsr>][_c][_v][_live]_<arch>_<key>.
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
#include "../common/rtc_build.h"
#include "../cufft/kbox_stage.cuh"          // the host half: the k-box launch rule (kbox_plan)
#include "generated/kbox_stage_src.h"       // the same file as text: the kernel's named header
#include "generated/kbox_rspace_src.h"

namespace lorrax_ffi::kbox_rspace {

// A block's LDS on gfx90a (64 KiB per CU, all of it addressable by one workgroup).
static constexpr long long kLds = 64 * 1024;
// Each mode's own axis bound: the stage's line transform is the unrolled direct DFT (N^2 FMAs
// against compile-time twiddles), so neither widens with the k-leading pair's KDFT_AXIS_MAX.
// ffi.fft.KDFT_KBOX_WEDGE_AXIS_MAX / KDFT_KBOX_CHI_AXIS_MAX mirror them; the gate covers axis 40.
static constexpr int kAxisMaxWedge = 40;     // mode 9
static constexpr int kAxisMaxChi = 40;       // mode 11
static constexpr int kGridCap = 110 * 16;    // grid-stride: enough tiles in flight for 110 CUs

static ffi::Error fail(const char* gate, const std::string& why,
                       ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
    return ffi::Error(code, std::string("GATE ") + gate + ": " + why);
}

// The device structs of kbox_rspace_kernel.h, field for field.
struct UnfoldTab {
    const int *row, *trs, *lsrc, *rsrc;
    const double *mph, *nph, *spin, *spin_r;
    long long ml, nl;
    int conj_trs;
};
struct ChiArgs {
    const void *gv, *gvt, *gc, *gct;
    void* acc;
    const void* alpha;
    long long pairs, my;
    int n_out;
    double si;
    unsigned long long vperm_l, vphase_l, vperm_r, vphase_r;
    int vna, vnb;
    const void* sgn_c;
    const void* live;
};

struct Built {
    rtc::Function fn = nullptr;
    int tr = 0, threads = 0;
    long long smem = 0;
};
using Key = std::tuple<int, int, int, int, int, int, int, int, int>;   // dev, mode, grid, ns, nsr, variant, live
static std::mutex g_mu;
static std::map<Key, Built> g_cache;
static std::map<Key, std::string> g_fail;

// The single arm's plan, as the mathdx family's build() asks kbox_plan for the mode:
//   mode 9   a group of ns*nsr columns, one operand, one transform, min_tr 1 (tile = tr groups)
//   mode 11  a group of ns^2 columns, TWO operands (Gv, Gc), one transform, min_tr 1 (tile = tr pairs)
// Returns "" and the tile's columns, or why a block cannot hold one group / pair.
static std::string plan(int mode, int nkx, int nky, int nkz, int ns, int nsr, lrx_kbox::Plan* kp, int* trc) {
    const int ss = ns * nsr, operands = mode == 11 ? 2 : 1;
    *kp = lrx_kbox::kbox_plan(nkx, nky, nkz, ss, operands, 16, kLds, 1, 1);
    if (kp->arm == 0 && kp->tr >= 1) {
        *trc = kp->tr * ss * operands;
        return "";
    }
    std::ostringstream os;
    os << "got k-grid (" << nkx << "," << nky << "," << nkz << ") ns=" << ns;
    if (mode == 9) os << " nsr=" << nsr;
    os << ", whose " << (mode == 11 ? "Green pair of " : "spin group of ") << ss * operands
       << " padded columns needs " << 16LL * ss * operands * lrx_kbox::Geometry{nkx, nky, nkz}.rs()
       << " B; want it in a block's " << kLds / 1024 << " KiB of LDS (the single arm; the split arm is not"
       << " ported to ROCM); the router keeps such grids on XLA (ffi.fft."
       << (mode == 11 ? "kdft_kbox_chi_refusal" : "kdft_kbox_wedge_refusal") << ")";
    return os.str();
}

// variant: mode 11 bit 0 the static completion, bit 1 the four-current channels (LRX_VTX).
static ffi::Error build(int mode, int nkx, int nky, int nkz, int ns, int nsr, int variant, bool live,
                        std::string_view cache_dir, const Built** out) {
    int dev = 0;
    LORRAX_GPU_CHECK(lrx::gpu::get_device(&dev));
    const Key key{dev, mode, nkx, nky, nkz, ns, nsr, variant, live ? 1 : 0};
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_cache.find(key); it != g_cache.end()) { *out = &it->second; return ffi::Error::Success(); }
    if (auto it = g_fail.find(key); it != g_fail.end()) return fail("kdft-kbox-build", "(cached failure) " + it->second);
    lrx_kbox::Plan kp{};
    int trc = 0;
    if (std::string why = plan(mode, nkx, nky, nkz, ns, nsr, &kp, &trc); !why.empty())
        return fail("kdft-kbox-residency", why, ffi::ErrorCode::kInvalidArgument);
    std::string arch;
    LORRAX_GPU_CHECK(lrx::gpu::device_arch(dev, &arch));

    rtc::Program prog;
    prog.src = kKernelSrc;
    prog.name = "lrx_kbox_rspace";
    prog.kernel = mode == 11 ? "lrx_kbox_m11" : "lrx_kbox_m9";
    prog.headers = {{lorrax_ffi::kbox::kHeaderName, lorrax_ffi::kbox::kHeaderSrc}};
    prog.defs = {"-O3", "-std=c++17", rtc::arch_option(arch), "-DLRX_MODE=" + std::to_string(mode),
                 "-DLRX_NX=" + std::to_string(nkx), "-DLRX_NY=" + std::to_string(nky),
                 "-DLRX_NZ=" + std::to_string(nkz), "-DLRX_NS=" + std::to_string(ns),
                 "-DLRX_NSR=" + std::to_string(nsr), "-DLRX_TR=" + std::to_string(trc),
                 "-DLRX_NT=" + std::to_string(kp.threads), std::string("-DLRX_LIVE=") + (live ? "1" : "0")};
    if (mode == 11) {
        prog.defs.push_back("-DLRX_COMPLETE=" + std::to_string(variant & 1));
        prog.defs.push_back("-DLRX_VTX=" + std::to_string((variant >> 1) & 1));
    }
    const std::string key_hex = rtc::hex16(rtc::key(prog, nullptr));
    const std::string dir(cache_dir);
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kbox_m" << mode << "_" << nkx << "x" << nky << "x" << nkz << "_ns" << ns;
        if (mode == 9) name << "x" << nsr;
        if (mode == 11 && (variant & 1)) name << "_c";
        if (mode == 11 && (variant & 2)) name << "_v";
        name << (live ? "_live" : "") << "_" << rtc::arch_tag(arch) << "_" << key_hex << rtc::kImageExt;
        path = name.str();
    }
    rtc::Image img;
    std::string where, err;
    if (!rtc::build(prog, dir, path, key_hex, &img, &where, &err)) {
        g_fail.emplace(key, where + " -- " + err);
        return fail("kdft-kbox-build", where + " -- " + err);
    }
    if (!img.from_disk)
        std::fprintf(stderr, "[kconv_kbox] " LRX_RTC_API " %s mode=%d kgrid=(%d,%d,%d) ns=%d nsr=%d variant=%d%s %s"
                     " tr=%d threads=%d lds=%lld B in %.1f ms%s\n", img.rebuilt_bad ? "rebuilt (cached image refused)" : "built",
                     mode, nkx, nky, nkz, ns, nsr, variant, live ? " live" : "", arch.c_str(), trc, kp.threads, kp.smem,
                     img.ms, img.stored ? " (stored)" : "");
    Built b;
    b.fn = img.fn;
    b.tr = trc;
    b.threads = kp.threads;
    b.smem = kp.smem;
    *out = &g_cache.emplace(key, b).first->second;
    return ffi::Error::Success();
}

static ffi::Error launch(const Built* b, long long ncols, lrx::gpu::Stream stream, void** args) {
    const long long blocks = std::max(1LL, std::min<long long>((ncols + b->tr - 1) / b->tr, kGridCap));
    const rtc::Result r = rtc::driver_api().LaunchKernel(
        b->fn, static_cast<unsigned>(blocks), 1, 1, b->threads, 1, 1, static_cast<unsigned>(b->smem),
        reinterpret_cast<rtc::Stream>(stream), args, nullptr);
    if (r != rtc::kOk) return fail("kdft-kbox-launch", LRX_MODULE_API "ModuleLaunchKernel: " + rtc::err_string(r));
    return ffi::Error::Success();
}

static bool is(ffi::AnyBuffer x, ffi::DataType t, std::initializer_list<int64_t> dims) {
    auto d = x.dimensions();
    return x.element_type() == t && d.size() == dims.size() && std::equal(d.begin(), d.end(), dims.begin());
}

static ffi::Error axis_ok(int64_t nkx, int64_t nky, int64_t nkz, int max_axis) {
    if (nkx < 1 || nky < 1 || nkz < 1 || nkx > max_axis || nky > max_axis || nkz > max_axis)
        return fail("kdft-axis", "want every k axis in [1, " + std::to_string(max_axis) + "] (this k-box mode's "
                    "direct DFT); the router keeps longer axes on XLA", ffi::ErrorCode::kInvalidArgument);
    return ffi::Error::Success();
}

// The optional live operand (s32 [2]): at most one RemainingArgs buffer.
static ffi::Error live_of(ffi::RemainingArgs rest, ffi::AnyBuffer* buf, const ffi::AnyBuffer** live) {
    auto bad = [](const std::string& why) { return fail("kdft-kbox-operands", why, ffi::ErrorCode::kInvalidArgument); };
    *live = nullptr;
    if (rest.size() > 1) return bad("want at most one trailing operand: live s32 [2]");
    if (rest.size() == 1) {
        auto l = rest.get<ffi::AnyBuffer>(0);
        if (l.has_error()) return bad("live: not a buffer");
        *buf = *l;
        if (buf->element_type() != ffi::DataType::S32 || buf->element_count() != 2)
            return bad("want live s32 [2]: the window rows [lo, hi) of a padded pass");
        *live = buf;
    }
    return ffi::Error::Success();
}

// ---- mode 9 ---------------------------------------------------------------------------------
static ffi::Error KleadUnfoldFft(lrx::gpu::Stream stream, ffi::AnyBuffer Wp, ffi::AnyBuffer Wt, ffi::AnyBuffer row,
                                 ffi::AnyBuffer trs, ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph,
                                 ffi::AnyBuffer nph, ffi::AnyBuffer spin_l, ffi::AnyBuffer spin_r,
                                 ffi::RemainingArgs rest, ffi::Result<ffi::AnyBuffer> Y, int64_t nkx, int64_t nky,
                                 int64_t nkz, double scale, int64_t conj_trs, std::string_view cubin_dir) {
    auto bad = [](const std::string& why) { return fail("kdft-kbox-operands", why, ffi::ErrorCode::kInvalidArgument); };
    ffi::AnyBuffer live_buf = Wp;
    const ffi::AnyBuffer* live = nullptr;
    FFI_RETURN_IF_ERROR(live_of(rest, &live_buf, &live));
    FFI_RETURN_IF_ERROR(axis_ok(nkx, nky, nkz, kAxisMaxWedge));
    const int64_t nk = nkx * nky * nkz;
    const auto wd = Wp.dimensions(), ld = spin_l.dimensions(), rd = spin_r.dimensions();
    if (wd.size() != 3 || ld.size() != 3 || rd.size() != 3)
        return bad("want Wp (n_parent, ml, nl), spin_l (nk, nl_s, nl_s) and spin_r (nk, nr_s, nr_s)");
    const int64_t np = wd[0], ml = wd[1], nl = wd[2], nsl = ld[1], nsr = rd[1];
    const auto C = ffi::DataType::C128, I = ffi::DataType::S32;
    if (nsl < 1 || nsl > 4 || nsr < 1 || nsr > 4 || ml % nsl || nl % nsr || np < 1 ||
        (conj_trs != 0 && conj_trs != 1) || !is(Wp, C, {np, ml, nl}) || !is(Wt, C, {np, ml, nl}) ||
        !is(row, I, {nk}) || !is(trs, I, {nk}) || !is(lsrc, I, {nk, ml}) || !is(rsrc, I, {nk, nl}) ||
        !is(mph, C, {nk, ml}) || !is(nph, C, {nk, nl}) || !is(spin_l, C, {nk, nsl, nsl}) ||
        !is(spin_r, C, {nk, nsr, nsr}) || !is(*Y, C, {nk, ml, nl}))
        return bad("want c128 Wp=Wt (np,ml,nl); s32 row,trs (nk), lsrc (nk,ml), rsrc (nk,nl); c128 mph (nk,ml), "
                   "nph (nk,nl), spin_l (nk,nl_s,nl_s), spin_r (nk,nr_s,nr_s) with nl_s, nr_s in [1,4]; "
                   "Y (nk,ml,nl); conj_trs 0|1");
    const int64_t pairs = (ml / nsl) * (nl / nsr);
    if (pairs == 0) return ffi::Error::Success();
    const Built* b = nullptr;
    FFI_RETURN_IF_ERROR(build(9, static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                              static_cast<int>(nsl), static_cast<int>(nsr), 0, live != nullptr, cubin_dir, &b));
    UnfoldTab t{static_cast<const int*>(row.untyped_data()), static_cast<const int*>(trs.untyped_data()),
                static_cast<const int*>(lsrc.untyped_data()), static_cast<const int*>(rsrc.untyped_data()),
                static_cast<const double*>(mph.untyped_data()), static_cast<const double*>(nph.untyped_data()),
                static_cast<const double*>(spin_l.untyped_data()), static_cast<const double*>(spin_r.untyped_data()),
                ml, nl, static_cast<int>(conj_trs)};
    const void* wpp = Wp.untyped_data();
    const void* wtp = Wt.untyped_data();
    void* yp = Y->untyped_data();
    double sc = scale;
    const void* livep = live ? live->untyped_data() : nullptr;
    void* args[] = {(void*)&wpp, (void*)&wtp, (void*)&yp, (void*)&t, (void*)&sc, (void*)&livep};
    return launch(b, pairs * nsl * nsr, stream, args);
}

// ---- mode 11 --------------------------------------------------------------------------------
// The family's vertex packing (kconv_mathdx_cuda_ffi.cc pack_attrs): perm entry i at bits 4*i,
// phase code i at bits 2*i of one vertex.
static bool pack_vertex(const int64_t* perm, const int64_t* phase, int64_t ns, const char* side,
                        unsigned long long* pp, unsigned long long* hp, std::string* why) {
    unsigned seen = 0;
    *pp = 0;
    *hp = 0;
    for (int64_t i = 0; i < ns; ++i) {
        if (perm[i] < 0 || perm[i] >= ns || phase[i] < 0 || phase[i] > 3 || (seen & (1u << perm[i]))) {
            *why = std::string(side) + " perm must be a permutation of [0,ns) and phase codes in [0,3]";
            return false;
        }
        seen |= 1u << perm[i];
        *pp |= static_cast<unsigned long long>(perm[i]) << (4 * i);
        *hp |= static_cast<unsigned long long>(phase[i]) << (2 * i);
    }
    return true;
}

static ffi::Error ChiImpl(lrx::gpu::Stream stream, ffi::AnyBuffer Gv, ffi::AnyBuffer Gvt, ffi::AnyBuffer Gc,
                          ffi::AnyBuffer Gct, ffi::AnyBuffer row, ffi::AnyBuffer trs, ffi::AnyBuffer lsrc,
                          ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
                          ffi::AnyBuffer alpha, ffi::AnyBuffer acc_in, const ffi::AnyBuffer* sgn_c,
                          const ffi::AnyBuffer* live, ffi::Result<ffi::AnyBuffer> acc, int64_t nkx, int64_t nky,
                          int64_t nkz, double si, int64_t conj_trs, int64_t complete,
                          ffi::Span<const int64_t> perm_l, ffi::Span<const int64_t> phase_l,
                          ffi::Span<const int64_t> perm_r, ffi::Span<const int64_t> phase_r, int64_t na, int64_t nb,
                          std::string_view cubin_dir) {
    auto bad = [](const std::string& why) { return fail("kdft-kbox-operands", why, ffi::ErrorCode::kInvalidArgument); };
    FFI_RETURN_IF_ERROR(axis_ok(nkx, nky, nkz, kAxisMaxChi));
    const int64_t nk = nkx * nky * nkz;
    const auto gd = Gv.dimensions(), sd = spin.dimensions(), ad = acc_in.dimensions();
    if (gd.size() != 3 || sd.size() != 3 || ad.size() != 4)
        return bad("want Gv (n_parent, ml, nl), spin (nk, ns, ns) and acc (n_out, nk, mx, my)");
    const bool vtx = na > 0 || nb > 0;
    if (vtx && (na < 1 || nb < 1 || na > 3 || nb > 3 || complete != 0))
        return bad("want vertex counts na, nb in [1, 3] (or both 0) and no static completion with vertices");
    const int64_t n_ch = vtx ? na * nb : 1;
    if (ad[0] % n_ch) return bad("want acc planes = na * nb * n_out");
    const int64_t np = gd[0], ml = gd[1], nl = gd[2], ns = sd[1], n_out = ad[0] / n_ch;
    const auto C = ffi::DataType::C128, I = ffi::DataType::S32;
    if ((ns != 1 && ns != 2 && ns != 4) || ml % ns || nl % ns || np < 1 || n_out < 1 ||
        (conj_trs != 0 && conj_trs != 2) || (complete != 0 && complete != 1) || !is(Gv, C, {np, ml, nl}) ||
        !is(Gvt, C, {np, ml, nl}) || !is(Gc, C, {np, ml, nl}) || !is(Gct, C, {np, ml, nl}) || !is(row, I, {nk}) ||
        !is(trs, I, {nk}) || !is(lsrc, I, {nk, ml}) || !is(rsrc, I, {nk, nl}) || !is(mph, C, {nk, ml}) ||
        !is(nph, C, {nk, nl}) || !is(spin, C, {nk, ns, ns}) || !is(alpha, C, {n_out}) ||
        !is(acc_in, C, {n_ch * n_out, nk, ml / ns, nl / ns}) || !is(*acc, C, {n_ch * n_out, nk, ml / ns, nl / ns}))
        return bad("want c128 Gv=Gvt=Gc=Gct (np,ml,nl); s32 row,trs (nk), lsrc (nk,ml), rsrc (nk,nl); c128 mph "
                   "(nk,ml), nph (nk,nl), spin (nk,ns,ns), alpha (n_out,), acc (n_ch*n_out,nk,ml/ns,nl/ns); "
                   "conj_trs 0 (partner tiles) | 2 (the partner is conj(G)); complete 0|1");
    if (sgn_c && !is(*sgn_c, ffi::DataType::F64, {nk})) return bad("want the Gc sign (nk,) f64");
    unsigned long long vpl = 0, vhl = 0, vpr = 0, vhr = 0;
    if (vtx) {
        std::string why;
        auto pack = [&](ffi::Span<const int64_t> perm, ffi::Span<const int64_t> phase, int64_t count, const char* side,
                        unsigned long long* pp, unsigned long long* hp) {
            if (perm.size() != static_cast<size_t>(count * ns) || phase.size() != static_cast<size_t>(count * ns)) {
                why = std::string(side) + " perm/phase lengths != (vertices * ns)";
                return false;
            }
            const std::vector<int64_t> pv(perm.begin(), perm.end()), hv(phase.begin(), phase.end());
            for (int64_t i = 0; i < count; ++i) {
                unsigned long long p1 = 0, h1 = 0;
                if (!pack_vertex(pv.data() + i * ns, hv.data() + i * ns, ns, side, &p1, &h1, &why)) return false;
                *pp |= p1 << (16 * i);
                *hp |= h1 << (8 * i);
            }
            return true;
        };
        if (!pack(perm_l, phase_l, na, "left", &vpl, &vhl) || !pack(perm_r, phase_r, nb, "right", &vpr, &vhr))
            return bad(why);
    }
    const int64_t my = nl / ns, pairs = (ml / ns) * my;
    // acc is aliased to acc_in (input_output_aliases); a copy only if XLA did not alias.
    if (acc->untyped_data() != acc_in.untyped_data())
        LORRAX_GPU_CHECK(lrx::gpu::memcpy_async(acc->untyped_data(), acc_in.untyped_data(), acc_in.size_bytes(),
                                                lrx::gpu::kDeviceToDevice, stream));
    if (pairs == 0) return ffi::Error::Success();
    const Built* b = nullptr;
    FFI_RETURN_IF_ERROR(build(11, static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                              static_cast<int>(ns), static_cast<int>(ns),
                              static_cast<int>(complete) + (vtx ? 2 : 0), live != nullptr, cubin_dir, &b));
    UnfoldTab t{static_cast<const int*>(row.untyped_data()), static_cast<const int*>(trs.untyped_data()),
                static_cast<const int*>(lsrc.untyped_data()), static_cast<const int*>(rsrc.untyped_data()),
                static_cast<const double*>(mph.untyped_data()), static_cast<const double*>(nph.untyped_data()),
                static_cast<const double*>(spin.untyped_data()), nullptr, ml, nl, static_cast<int>(conj_trs)};
    ChiArgs a{Gv.untyped_data(), Gvt.untyped_data(), Gc.untyped_data(), Gct.untyped_data(), acc->untyped_data(),
              alpha.untyped_data(), pairs, my, static_cast<int>(n_out), si, vpl, vhl, vpr, vhr,
              static_cast<int>(vtx ? na : 0), static_cast<int>(vtx ? nb : 0),
              sgn_c ? sgn_c->untyped_data() : nullptr, live ? live->untyped_data() : nullptr};
    void* args[] = {(void*)&a, (void*)&t};
    return launch(b, pairs * 2 * ns * ns, stream, args);
}

static ffi::Error ChiUnfold(lrx::gpu::Stream stream, ffi::AnyBuffer Gv, ffi::AnyBuffer Gvt, ffi::AnyBuffer Gc,
                            ffi::AnyBuffer Gct, ffi::AnyBuffer row, ffi::AnyBuffer trs, ffi::AnyBuffer lsrc,
                            ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
                            ffi::AnyBuffer alpha, ffi::AnyBuffer acc_in, ffi::RemainingArgs rest,
                            ffi::Result<ffi::AnyBuffer> acc, int64_t nkx, int64_t nky, int64_t nkz, double si,
                            int64_t conj_trs, int64_t complete, std::string_view cubin_dir) {
    ffi::AnyBuffer live_buf = Gv;
    const ffi::AnyBuffer* live = nullptr;
    FFI_RETURN_IF_ERROR(live_of(rest, &live_buf, &live));
    return ChiImpl(stream, Gv, Gvt, Gc, Gct, row, trs, lsrc, rsrc, mph, nph, spin, alpha, acc_in, nullptr, live, acc,
                   nkx, nky, nkz, si, conj_trs, complete, {}, {}, {}, {}, 0, 0, cubin_dir);
}

static ffi::Error ChiVertex(lrx::gpu::Stream stream, ffi::AnyBuffer Gv, ffi::AnyBuffer Gvt, ffi::AnyBuffer Gc,
                            ffi::AnyBuffer Gct, ffi::AnyBuffer row, ffi::AnyBuffer trs, ffi::AnyBuffer lsrc,
                            ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
                            ffi::AnyBuffer alpha, ffi::AnyBuffer sgn_c, ffi::AnyBuffer acc_in, ffi::RemainingArgs rest,
                            ffi::Result<ffi::AnyBuffer> acc, int64_t nkx, int64_t nky, int64_t nkz, double si,
                            int64_t conj_trs, int64_t signed_c, ffi::Span<const int64_t> perm_l,
                            ffi::Span<const int64_t> phase_l, ffi::Span<const int64_t> perm_r,
                            ffi::Span<const int64_t> phase_r, int64_t na, int64_t nb, std::string_view cubin_dir) {
    if (na < 1 || nb < 1)
        return fail("kdft-kbox-operands", "want na, nb >= 1 (the plain pass is lorrax_kdft_kconv_chi_unfold)",
                    ffi::ErrorCode::kInvalidArgument);
    ffi::AnyBuffer live_buf = Gv;
    const ffi::AnyBuffer* live = nullptr;
    FFI_RETURN_IF_ERROR(live_of(rest, &live_buf, &live));
    return ChiImpl(stream, Gv, Gvt, Gc, Gct, row, trs, lsrc, rsrc, mph, nph, spin, alpha, acc_in,
                   signed_c ? &sgn_c : nullptr, live, acc, nkx, nky, nkz, si, conj_trs, 0, perm_l, phase_l, perm_r,
                   phase_r, na, nb, cubin_dir);
}

}  // namespace lorrax_ffi::kbox_rspace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KDftKfftKleadUnfoldFfi, lorrax_ffi::kbox_rspace::KleadUnfoldFft,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()     // Wp (the wedge rows)
        .Arg<xla::ffi::AnyBuffer>()     // Wt (its partner; unread when conj_trs = 1)
        .Arg<xla::ffi::AnyBuffer>()     // row
        .Arg<xla::ffi::AnyBuffer>()     // trs
        .Arg<xla::ffi::AnyBuffer>()     // lsrc
        .Arg<xla::ffi::AnyBuffer>()     // rsrc
        .Arg<xla::ffi::AnyBuffer>()     // mph
        .Arg<xla::ffi::AnyBuffer>()     // nph
        .Arg<xla::ffi::AnyBuffer>()     // spin_l
        .Arg<xla::ffi::AnyBuffer>()     // spin_r
        .RemainingArgs()                // live (optional)
        .Ret<xla::ffi::AnyBuffer>()     // Y (nk, ml, nl)
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("conj_trs")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KDftKconvChiUnfoldFfi, lorrax_ffi::kbox_rspace::ChiUnfold,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()     // Gv
        .Arg<xla::ffi::AnyBuffer>()     // Gvt
        .Arg<xla::ffi::AnyBuffer>()     // Gc
        .Arg<xla::ffi::AnyBuffer>()     // Gct
        .Arg<xla::ffi::AnyBuffer>()     // row
        .Arg<xla::ffi::AnyBuffer>()     // trs
        .Arg<xla::ffi::AnyBuffer>()     // lsrc
        .Arg<xla::ffi::AnyBuffer>()     // rsrc
        .Arg<xla::ffi::AnyBuffer>()     // mph
        .Arg<xla::ffi::AnyBuffer>()     // nph
        .Arg<xla::ffi::AnyBuffer>()     // spin
        .Arg<xla::ffi::AnyBuffer>()     // alpha (n_out,)
        .Arg<xla::ffi::AnyBuffer>()     // acc (n_out, nk, mx, my), aliased to the result
        .RemainingArgs()                // live (optional)
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("si")
        .Attr<int64_t>("conj_trs")
        .Attr<int64_t>("complete")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KDftKconvChiVertexFfi, lorrax_ffi::kbox_rspace::ChiVertex,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()     // Gv (lower Green quadrant)
        .Arg<xla::ffi::AnyBuffer>()     // Gvt
        .Arg<xla::ffi::AnyBuffer>()     // Gc (upper Green quadrant; the vertices act on it)
        .Arg<xla::ffi::AnyBuffer>()     // Gct
        .Arg<xla::ffi::AnyBuffer>()     // row
        .Arg<xla::ffi::AnyBuffer>()     // trs
        .Arg<xla::ffi::AnyBuffer>()     // lsrc
        .Arg<xla::ffi::AnyBuffer>()     // rsrc
        .Arg<xla::ffi::AnyBuffer>()     // mph
        .Arg<xla::ffi::AnyBuffer>()     // nph
        .Arg<xla::ffi::AnyBuffer>()     // spin
        .Arg<xla::ffi::AnyBuffer>()     // alpha (n_out,)
        .Arg<xla::ffi::AnyBuffer>()     // sgn_c (nk,) f64 (read when signed_c)
        .Arg<xla::ffi::AnyBuffer>()     // acc (na*nb*n_out, nk, mx, my), aliased to the result
        .RemainingArgs()                // live (optional)
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("si")
        .Attr<int64_t>("conj_trs")
        .Attr<int64_t>("signed_c")
        .Attr<xla::ffi::Span<const int64_t>>("perm_l")
        .Attr<xla::ffi::Span<const int64_t>>("phase_l")
        .Attr<xla::ffi::Span<const int64_t>>("perm_r")
        .Attr<xla::ffi::Span<const int64_t>>("phase_r")
        .Attr<int64_t>("na")
        .Attr<int64_t>("nb")
        .Attr<std::string_view>("cubin_dir"));
