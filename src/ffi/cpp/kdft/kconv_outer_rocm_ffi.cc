// kconv_outer_rocm_ffi.cc -- the ROCM leg's outer-product k-convolution (the BSE W term): the port of
// cufft/kconv_outer_cuda_ffi.cc on the k-box stage (kdft/kconv_outer_kernel.h on cufft/kbox_stage.cuh,
// its line transform the kdft direct DFT under hipRTC).
//
//   lorrax_kdft_kconv_klead_outer          U = s * fftn(ifftn(T) * V),  T = sum_K L R'  (never stored)
//   lorrax_kdft_kconv_klead_outer_decode   A = sum_{a,x} conj(Pc) U                     (U never stored)
//
// The operation, layouts and attributes of lorrax_mathdx_kconv_klead_outer: L (nk, na, mx, K),
// R (nk, K, nb, my), V (mx, my, nk) k-MINOR, U (nk, na, mx, nb, my), conj_r; less mathdx_root and
// the K padding (the FMA K sum takes any K).  The decode takes Pc (nk, nc, na, mx) as the BSE holds
// psi^X_c (no fragment tiling: the CUDA prep's Lr/Rr/Pcr serve the m8n8k4 fragments) and returns
// A (nk, nc, nb, my), the CUDA decode's output.  Work items, the V-reuse order, the (phase, item)
// combos, the slots and the counter are the CUDA kernels' (plan_phases is the CUDA rule).
//
// Tile (the one decision that differs): the CUDA bank is 8 x 8 columns in up to 227 KB of opt-in
// shared memory; a gfx90a block has 64 KiB of LDS, so the host takes the largest XB x YB box (YB = 8
// first, then XB = 8, 4, 2, 1, then YB = 4, 2, 1) whose bank fits half the LDS (two blocks per CU),
// else the whole LDS.  A grid whose single padded column does not fit (nk above ~4000) is refused
// (GATE kdft-kconv-outer-tile) and ffi.fft.klead_outer_refusal keeps it on the unfused chain
// (XLA encode + kdft mode 2) by the same rule.  Routing is measured, not just fit: ffi.fft sends a
// grid here only where the 8 x 8 box fits half the LDS and keeps the XLA decode (GATE
// kdft-kconv-outer-box / -decode-slower; the record beside ffi.fft.KDFT_OUTER_BOX_COLS).  Swept
// on one GCD (pool log kcv_sweep1): 512 threads beat 128 and 256; a 4 x 8 or 8 x 4 box at the full
// LDS (one block per CU) and filling every block slot with decode phases did not pay.  The decode also refuses an accumulator above
// kAccMax complex per thread (GATE kdft-kconv-outer-decode-tile).  Every call is gated against the
// XLA route on the same GCD (ffi.fft.xla_reference; tests/test_kdft_outer_gate.py).
//
// Leg-neutral host code (lrx::gpu, rtc); CMake lists it in the ROCM leg.  Images: one per (k-grid,
// K, tile, conj, decode nc, device architecture), disk-cached as kconv_outer_*_<arch>_<key>.hsaco.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>

#include "../common/ffi_helpers.h"
#include "../common/gpu_runtime.h"
#include "../common/leg.h"
#include "../common/rtc_build.h"
#include "../cufft/kbox_stage.cuh"          // the host half: Geometry (the padded bank row)
#include "generated/kbox_stage_src.h"       // the same file as text: the kernel's named header
#include "generated/kconv_outer_src.h"

namespace lorrax_ffi::kconv_outer_rocm {

static constexpr long long kLds = LRX_LEG_BLOCK_LDS;   // the leg's block LDS (common/leg.h; 64 KiB on ROCM)
static constexpr int kAxisMax = 40;                     // the stage's direct-DFT line (KDFT_KBOX_AXIS_MAX)
static constexpr int kCUs = 110;                        // MI250X GCD compute units (kbox_unfold_ffi.cc kGridCap)
static constexpr int kThreads = 512;                    // U kernel; the decode takes 512 or 1024 (kAccMax)
                                                        // (swept 128/256/512: 512 fastest, pool log kcv_sweep1)
static constexpr int kAccMax = 32;                      // decode accumulators per thread (complex): 64 VGPR pairs

static ffi::Error fail(const char* gate, const std::string& why,
                       ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
    return ffi::Error(code, std::string("GATE ") + gate + ": " + why);
}

// The device struct of kconv_outer_kernel.h, field for field.
struct OuterGeo {
    long long na, mx, nb, my;
    long long nxb, nyb;
    long long ngrp, per;
    long long items, combos;
    long long nc;
    double scale;
};

struct Tile { int xb = 0, yb = 0, bpc = 0; long long smem = 0; };

// The tile rule (docstring above; ffi.fft.kdft_outer_tile mirrors it).  xb = 0: no tile fits.
static Tile plan_tile(int nkx, int nky, int nkz) {
    const long long col = 16LL * lrx_kbox::Geometry{nkx, nky, nkz}.rs();
    for (long long budget : {kLds / 2, kLds})
        for (int yb : {8, 4, 2, 1})
            for (int xb : {8, 4, 2, 1})
                if ((yb == 8 || xb == 1) && xb * yb * col <= budget)
                    return Tile{xb, yb, static_cast<int>(std::max(1LL, kLds / (xb * yb * col))), xb * yb * col};
    return Tile{};
}

// The CUDA decode's phase count (kconv_outer_cuda_ffi.cc plan_phases), unchanged.
static long long plan_phases(long long items, long long pairs, long long slots_resident, long long slot_bytes,
                             long long u_bytes) {
    long long best = 1, best_cost = -1;
    const long long cap = std::max(items * slot_bytes, u_bytes / 16);
    for (long long ngrp = 1; ngrp <= pairs; ++ngrp) {
        const long long per = (pairs + ngrp - 1) / ngrp, n = (pairs + per - 1) / per;
        if (n != ngrp) continue;
        if (items * n * slot_bytes > cap) break;
        const long long waves = (items * n + slots_resident - 1) / slots_resident;
        const long long cost = waves * (per + 1);
        if (best_cost < 0 || cost < best_cost) { best = n; best_cost = cost; }
    }
    return best;
}

struct Built {
    rtc::Function fn = nullptr;
    Tile tile;
    int threads = 0;
};
using Key = std::tuple<int, int, int, int, int, int, int>;   // dev, nkx, nky, nkz, K, conj, decode nc (0: U)
static std::mutex g_mu;
static std::map<Key, Built> g_cache;
static std::map<Key, std::string> g_fail;

static ffi::Error build(int nkx, int nky, int nkz, int K, int conj, int nc, std::string_view cache_dir,
                        const Built** out) {
    int dev = 0;
    LORRAX_GPU_CHECK(lrx::gpu::get_device(&dev));
    const Key key{dev, nkx, nky, nkz, K, conj, nc};
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_cache.find(key); it != g_cache.end()) { *out = &it->second; return ffi::Error::Success(); }
    if (auto it = g_fail.find(key); it != g_fail.end())
        return fail("kdft-kconv-outer-build", "(cached failure) " + it->second);
    const Tile t = plan_tile(nkx, nky, nkz);
    if (t.xb == 0) {
        std::ostringstream os;
        os << "got k-grid (" << nkx << "," << nky << "," << nkz << ") whose padded column needs "
           << 16LL * lrx_kbox::Geometry{nkx, nky, nkz}.rs() << " B; want it in a block's " << kLds / 1024
           << " KiB of LDS (the outer load's resident k-box tile); the router keeps such grids on the XLA "
              "encode + kdft mode 2 chain (ffi.fft.klead_outer_refusal)";
        return fail("kdft-kconv-outer-tile", os.str(), ffi::ErrorCode::kInvalidArgument);
    }
    const int nk = nkx * nky * nkz;
    int threads = kThreads, acc = 0;
    if (nc > 0) {
        const long long cells = static_cast<long long>(nk) * nc * t.yb;
        threads = std::max(threads, (cells + threads - 1) / threads <= kAccMax / 2 ? threads : 2 * threads);
        acc = static_cast<int>((cells + threads - 1) / threads);
        if (acc > kAccMax) {
            std::ostringstream os;
            os << "got k-grid (" << nkx << "," << nky << "," << nkz << ") nc=" << nc << ", " << acc
               << " decode accumulators per thread at " << threads << " threads; want <= " << kAccMax
               << " (registers); the router keeps such shapes on the outer conv + XLA decode "
                  "(ffi.fft.klead_outer_decode_refusal)";
            return fail("kdft-kconv-outer-decode-tile", os.str(), ffi::ErrorCode::kInvalidArgument);
        }
    }
    std::string arch;
    LORRAX_GPU_CHECK(lrx::gpu::device_arch(dev, &arch));
    rtc::Program prog;
    prog.src = kKernelSrc;
    prog.name = "lrx_kconv_outer";
    prog.kernel = nc > 0 ? "lrx_kconv_outer_decode" : "lrx_kconv_outer";
    prog.headers = {{lorrax_ffi::kbox::kHeaderName, lorrax_ffi::kbox::kHeaderSrc}};
    prog.defs = {rtc::kOptOption, "-std=c++17", rtc::arch_option(arch),
                 "-DLRX_NX=" + std::to_string(nkx), "-DLRX_NY=" + std::to_string(nky),
                 "-DLRX_NZ=" + std::to_string(nkz), "-DLRX_K=" + std::to_string(K),
                 "-DLRX_XB=" + std::to_string(t.xb), "-DLRX_YB=" + std::to_string(t.yb),
                 "-DLRX_NT=" + std::to_string(threads), "-DLRX_CONJ=" + std::to_string(conj),
                 "-DLRX_DECODE=" + std::to_string(nc > 0 ? 1 : 0), "-DLRX_NC=" + std::to_string(std::max(nc, 1)),
                 "-DLRX_E=" + std::to_string(std::max(acc, 1))};
    const std::string key_hex = rtc::hex16(rtc::key(prog, nullptr));
    const std::string dir(cache_dir);
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kconv_outer_" << (nc > 0 ? "decode_" : "") << nkx << "x" << nky << "x" << nkz << "_K" << K
             << (nc > 0 ? "_nc" + std::to_string(nc) : std::string()) << (conj ? "_conj" : "") << "_"
             << rtc::arch_tag(arch) << "_" << key_hex << rtc::kImageExt;
        path = name.str();
    }
    rtc::Image img;
    std::string where, err;
    if (!rtc::build(prog, dir, path, key_hex, &img, &where, &err)) {
        g_fail.emplace(key, where + " -- " + err);
        return fail("kdft-kconv-outer-build", where + " -- " + err);
    }
    if (!img.from_disk)
        std::fprintf(stderr, "[kconv_outer] " LRX_RTC_API " %s %s kgrid=(%d,%d,%d) K=%d%s%s %s tile=%dx%d threads=%d"
                     " lds=%lld B in %.1f ms%s\n", img.rebuilt_bad ? "rebuilt (cached image refused)" : "built",
                     nc > 0 ? "decode" : "U", nkx, nky, nkz, K, conj ? " conj" : "",
                     nc > 0 ? (" nc=" + std::to_string(nc)).c_str() : "", arch.c_str(), t.xb, t.yb, threads,
                     t.smem, img.ms, img.stored ? " (stored)" : "");
    Built b;
    b.fn = img.fn;
    b.tile = t;
    b.threads = threads;
    *out = &g_cache.emplace(key, b).first->second;
    return ffi::Error::Success();
}

// Shape checks shared by both targets; fills the geometry's (na, mx, nb, my, K).
static ffi::Error check(ffi::AnyBuffer L, ffi::AnyBuffer R, ffi::AnyBuffer V, int64_t nkx, int64_t nky,
                        int64_t nkz, OuterGeo* g, int64_t* K) {
    auto bad = [](const std::string& why) { return fail("kdft-kconv-outer-operands", why, ffi::ErrorCode::kInvalidArgument); };
    if (nkx < 1 || nky < 1 || nkz < 1 || nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax)
        return fail("kdft-axis", "want every k axis in [1, " + std::to_string(kAxisMax) + "] (the stage's direct-DFT "
                    "line); the router keeps longer axes on the unfused chain", ffi::ErrorCode::kInvalidArgument);
    const auto C = ffi::DataType::C128;
    if (L.element_type() != C || R.element_type() != C || V.element_type() != C)
        return bad("operands must all be complex128");
    const int64_t nk = nkx * nky * nkz;
    auto ld = L.dimensions(), rd = R.dimensions(), vd = V.dimensions();
    if (ld.size() != 4 || rd.size() != 4 || vd.size() != 3)
        return bad("want L (nk,na,mx,K), R (nk,K,nb,my), V (mx,my,nk)");
    const int64_t na = ld[1], mx = ld[2], k = ld[3], nb = rd[2], my = rd[3];
    if (ld[0] != nk || rd[0] != nk || rd[1] != k || vd[0] != mx || vd[1] != my || vd[2] != nk || k < 1)
        return bad("want L (nk,na,mx,K), R (nk,K,nb,my), V (mx,my,nk) with nk = nkx*nky*nkz, K >= 1");
    g->na = na; g->mx = mx; g->nb = nb; g->my = my;
    *K = k;
    return ffi::Error::Success();
}

static ffi::Error launch(const Built* b, long long blocks, lrx::gpu::Stream stream, void** args) {
    const rtc::Result r = rtc::driver_api().LaunchKernel(
        b->fn, static_cast<unsigned>(blocks), 1, 1, b->threads, 1, 1, static_cast<unsigned>(b->tile.smem),
        reinterpret_cast<rtc::Stream>(stream), args, nullptr);
    if (r != rtc::kOk) return fail("kdft-kconv-outer-launch", LRX_MODULE_API "ModuleLaunchKernel: " + rtc::err_string(r));
    return ffi::Error::Success();
}

// L (nk, na, mx, K), R (nk, K, nb, my), V (mx, my, nk) -> U (nk, na, mx, nb, my), complex128.
static ffi::Error KleadOuter(lrx::gpu::Stream stream, ffi::AnyBuffer L, ffi::AnyBuffer R, ffi::AnyBuffer V,
                             ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz, double scale,
                             int64_t conj_r, std::string_view cubin_dir) {
    OuterGeo g{};
    int64_t K = 0;
    FFI_RETURN_IF_ERROR(check(L, R, V, nkx, nky, nkz, &g, &K));
    const int64_t nk = nkx * nky * nkz;
    auto ud = U->dimensions();
    if (U->element_type() != ffi::DataType::C128 || ud.size() != 5 || ud[0] != nk || ud[1] != g.na ||
        ud[2] != g.mx || ud[3] != g.nb || ud[4] != g.my)
        return fail("kdft-kconv-outer-operands", "want U (nk,na,mx,nb,my) c128", ffi::ErrorCode::kInvalidArgument);
    if (g.na * g.mx * g.nb * g.my == 0) return ffi::Error::Success();
    const Built* b = nullptr;
    FFI_RETURN_IF_ERROR(build(static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz), static_cast<int>(K),
                              conj_r ? 1 : 0, 0, cubin_dir, &b));
    g.nxb = (g.mx + b->tile.xb - 1) / b->tile.xb;
    g.nyb = (g.my + b->tile.yb - 1) / b->tile.yb;
    // Groups of (x block, a) pairs per (y block, b): about eight resident waves of items (the CUDA rule).
    const long long slots = static_cast<long long>(kCUs) * b->tile.bpc;
    const long long base = g.nb * g.nyb, pairs = g.na * g.nxb;
    g.ngrp = std::max(1LL, std::min(pairs, (8 * slots + base - 1) / base));
    g.per = (pairs + g.ngrp - 1) / g.ngrp;
    g.items = base * g.ngrp;
    g.combos = g.items;
    g.nc = 0;
    g.scale = scale;
    const void* lp = L.untyped_data();
    const void* rp = R.untyped_data();
    const void* vp = V.untyped_data();
    void* up = U->untyped_data();
    void* args[] = {(void*)&lp, (void*)&rp, (void*)&vp, (void*)&up, (void*)&g};
    return launch(b, std::min<long long>(g.items, 2147483647LL), stream, args);
}

// L, R, V as above, Pc (nk, nc, na, mx) -> A (nk, nc, nb, my), complex128.  Scratch: the combo slots
// and one counter per item (only when an item is split into phases).
static ffi::Error KleadOuterDecode(lrx::gpu::Stream stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer L,
                                   ffi::AnyBuffer R, ffi::AnyBuffer V, ffi::AnyBuffer Pc,
                                   ffi::Result<ffi::AnyBuffer> A, int64_t nkx, int64_t nky, int64_t nkz,
                                   double scale, int64_t conj_r, std::string_view cubin_dir) {
    OuterGeo g{};
    int64_t K = 0;
    FFI_RETURN_IF_ERROR(check(L, R, V, nkx, nky, nkz, &g, &K));
    const int64_t nk = nkx * nky * nkz;
    auto pd = Pc.dimensions(), ad = A->dimensions();
    const auto C = ffi::DataType::C128;
    if (Pc.element_type() != C || A->element_type() != C || pd.size() != 4 || ad.size() != 4 || pd[0] != nk ||
        pd[2] != g.na || pd[3] != g.mx || ad[0] != nk || ad[1] != pd[1] || ad[2] != g.nb || ad[3] != g.my || pd[1] < 1)
        return fail("kdft-kconv-outer-operands", "want Pc (nk,nc,na,mx) and A (nk,nc,nb,my) c128",
                    ffi::ErrorCode::kInvalidArgument);
    g.nc = pd[1];
    if (g.na * g.mx * g.nb * g.my == 0) return ffi::Error::Success();
    const Built* b = nullptr;
    FFI_RETURN_IF_ERROR(build(static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz), static_cast<int>(K),
                              conj_r ? 1 : 0, static_cast<int>(g.nc), cubin_dir, &b));
    g.nxb = (g.mx + b->tile.xb - 1) / b->tile.xb;
    g.nyb = (g.my + b->tile.yb - 1) / b->tile.yb;
    const long long items = g.nb * g.nyb, pairs = g.na * g.nxb;
    const long long slot_bytes = nk * g.nc * b->tile.yb * 16;
    const long long resident = static_cast<long long>(kCUs) * b->tile.bpc;
    g.ngrp = plan_phases(items, pairs, resident, slot_bytes, nk * g.na * g.mx * g.nb * g.my * 16);
    g.per = (pairs + g.ngrp - 1) / g.ngrp;
    g.items = items;
    g.combos = items * g.ngrp;
    g.scale = scale;
    if (g.combos >= (int64_t(1) << 31)) return fail("kdft-kconv-outer-operands", "too many (phase, item) combos");
    void* sp = nullptr;
    void* np = nullptr;
    if (g.ngrp > 1) {
        std::optional<void*> sl = scratch.Allocate(static_cast<size_t>(g.combos * slot_bytes));
        std::optional<void*> cn = scratch.Allocate(static_cast<size_t>(items * sizeof(int)));
        if (!sl.has_value() || !cn.has_value())
            return fail("kdft-kconv-outer-scratch", "scratch allocation (slots, counters)");
        sp = *sl;
        np = *cn;
        LORRAX_GPU_CHECK(lrx::gpu::memset_async(np, 0, static_cast<size_t>(items * sizeof(int)), stream));
    }
    const void* lp = L.untyped_data();
    const void* rp = R.untyped_data();
    const void* vp = V.untyped_data();
    const void* cp = Pc.untyped_data();
    void* ap = A->untyped_data();
    void* args[] = {(void*)&lp, (void*)&rp, (void*)&vp, (void*)&cp, (void*)&ap, (void*)&sp, (void*)&np, (void*)&g};
    return launch(b, std::min<long long>(g.combos, resident), stream, args);
}

}  // namespace lorrax_ffi::kconv_outer_rocm

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KDftKconvKleadOuterFfi, lorrax_ffi::kconv_outer_rocm::KleadOuter,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()     // L (nk, na, mx, K)
        .Arg<xla::ffi::AnyBuffer>()     // R (nk, K, nb, my)
        .Arg<xla::ffi::AnyBuffer>()     // V (mx, my, nk), R space, k-minor
        .Ret<xla::ffi::AnyBuffer>()     // U (nk, na, mx, nb, my)
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("conj_r")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KDftKconvKleadOuterDecodeFfi, lorrax_ffi::kconv_outer_rocm::KleadOuterDecode,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Ctx<xla::ffi::ScratchAllocator>()   // the combo slots and the item counters
        .Arg<xla::ffi::AnyBuffer>()     // L (nk, na, mx, K)
        .Arg<xla::ffi::AnyBuffer>()     // R (nk, K, nb, my)
        .Arg<xla::ffi::AnyBuffer>()     // V (mx, my, nk), R space, k-minor
        .Arg<xla::ffi::AnyBuffer>()     // Pc (nk, nc, na, mx)
        .Ret<xla::ffi::AnyBuffer>()     // A (nk, nc, nb, my)
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("conj_r")
        .Attr<std::string_view>("cubin_dir"));
