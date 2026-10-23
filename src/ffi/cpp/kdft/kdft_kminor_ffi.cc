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
// the ROCM leg.  The plan (kdft_kminor_plan.h) is the SINGLE arm (one pass, a block owns whole
// rows) or, past one block's tile or where it is cheaper, the SPLIT arm: row passes over the
// trailing axis group and k-leading column passes (kdft_klead_ffi.h build_pass) over the others,
// 2*(groups-1)+1 HBM round trips for mode 4 and one per group for mode 5.  The split mode 4 with
// out_layout 1 takes a scratch buffer of X's size from XLA (its permuted store cannot run in
// place).  Images: one per (mode, out_layout, row-pass grid, O_T, precision, device
// architecture), disk-cached in cubin_dir as
// kdftm_m<mode>[o1]_<nkx>x<nky>x<nkz>[_s<O_T>][_c64]_<arch>_<key><ext>; column passes as
// kdft_klead_ffi.cc names them.  The verified range is kNkMax / kAxisMax of kdft_klead_plan.h.
#include <array>
#include <climits>
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
#include "../common/c_abi.h"
#include "../common/gpu_runtime.h"
#include "../common/rtc_build.h"
#include "generated/kdft_klead_src.h"   // kCoreSrc, kPlanSrc: the shared DFT core
#include "generated/kdft_kminor_src.h"
#include "kdft_klead_ffi.h"
#include "kdft_kminor_plan.h"
#include "kdft_kminor_workspace.h"
#include <vector>

extern "C" std::int64_t LRX_C_ENTRY(lrx_kminor_workspace_bytes)(
        int nx, int ny, int nz, int mode, int out, std::int64_t rows,
        std::int64_t m, int cbytes) {
    return ::kdft::kminor_workspace_bytes(nx, ny, nz, mode, out, rows, m, cbytes);
}

namespace lorrax_ffi::kdft {
namespace {

using ::kdft::MinorPlan;
using ::kdft::Pass;
using ::kdft::Rows;
constexpr int kThreads = ::kdft::kMinorThreads;
constexpr int kAxisMax = ::kdft::kAxisMax;      // ffi.fft.KDFT_KMINOR_AXIS_MAX: the gated range

ffi::Error fail(const char* gate, const std::string& why,
                ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
    return ffi::Error(code, std::string("GATE ") + gate + ": " + why);
}

struct Built {
    rtc::Function fn = nullptr;
    int rows = 0;
    int order = -1;   // the split arm's out_layout-1 store's block order (row_blocks; -1: other passes)
};
// dev, mode, out, nkx, nky, nkz, f32, O_T, C, W, order
using Key = std::tuple<int, int, int, int, int, int, int, long long, int, int, int>;
std::mutex g_mu;
std::map<Key, Built> g_cache;
std::map<Key, std::string> g_fail;

// The image of one row pass (or the single arm): k-grid g, O_T (1 = single), rows rw; order: the
// split arm's out_layout-1 store's block order (kdft_kminor_kernel.h LRX_ORDER), -1 otherwise.
ffi::Error build(int mode, int out, const std::array<int, 3>& g, long long ot, Rows rw, bool f32,
                 std::string_view cache_dir, const Built** res, int order = -1) {
    if (!(out == 1 && ot > 1)) order = -1;
    int dev = 0;
    LORRAX_GPU_CHECK(lrx::gpu::get_device(&dev));
    const int nkx = g[0], nky = g[1], nkz = g[2];
    const Key key{dev, mode, out, nkx, nky, nkz, f32 ? 1 : 0, ot, rw.c, rw.w, order};
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_cache.find(key); it != g_cache.end()) { *res = &it->second; return ffi::Error::Success(); }
    if (auto it = g_fail.find(key); it != g_fail.end()) return fail("kdft-build", "(cached failure) " + it->second);

    std::string arch;
    LORRAX_GPU_CHECK(lrx::gpu::device_arch(dev, &arch));
    const int rows = rw.c;
    rtc::Program prog;
    prog.src = kKminorSrc;
    prog.name = "lrx_kdft_kminor";
    prog.kernel = "lrx_kdft_kminor";
    prog.headers = {{"kdft_core.h", kCoreSrc}, {"kdft_plan.h", kPlanSrc}};
    prog.defs = {rtc::kOptOption, "-std=c++17", rtc::arch_option(arch),
                 "-DLRX_NX=" + std::to_string(nkx), "-DLRX_NY=" + std::to_string(nky),
                 "-DLRX_NZ=" + std::to_string(nkz), "-DLRX_C=" + std::to_string(rows),
                 "-DLRX_W=" + std::to_string(rw.w), "-DLRX_SPLIT=" + std::to_string(ot),
                 "-DLRX_NT=" + std::to_string(kThreads), "-DLRX_MODE=" + std::to_string(mode),
                 "-DLRX_OUT=" + std::to_string(out), std::string("-DLRX_F32=") + (f32 ? "1" : "0")};
    if (order >= 0) prog.defs.push_back("-DLRX_ORDER=" + std::to_string(order));   // the l1 store only
    const std::string key_hex = rtc::hex16(rtc::key(prog, nullptr));
    const std::string dir(cache_dir);
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kdftm_m" << mode << (out ? "o1" : "") << "_" << nkx << "x" << nky << "x" << nkz
             << (ot > 1 ? "_s" + std::to_string(ot) : "") << (order >= 0 ? "_q" + std::to_string(order) : "")
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
        std::fprintf(stderr, "[kconv_kdft] " LRX_RTC_API " %s mode=%d%s%s kgrid=(%d,%d,%d) O_T=%lld %s C=%d in %.1f ms%s\n",
                     img.rebuilt_bad ? "rebuilt (cached image refused)" : "built", mode, out ? " out1" : "",
                     f32 ? " c64" : "", nkx, nky, nkz, ot, arch.c_str(), rows, img.ms, img.stored ? " (stored)" : "");
    Built b;
    b.fn = img.fn;
    b.rows = rows;
    b.order = order;
    *res = &g_cache.emplace(key, b).first->second;
    return ffi::Error::Success();
}

// The shared launch.  X (rows, nk) flat (mode 5) or (d0..d4, nk) (mode 4, with K (d1, d2, nk));
// the kernel reads X and writes Y, which may be the same buffer (aliased) or not.
ffi::Error launch(lrx::gpu::Stream stream, ffi::ScratchAllocator* scratch, int mode, ffi::AnyBuffer X,
                  const ffi::AnyBuffer* K, ffi::Result<ffi::AnyBuffer> Y, int64_t nkx, int64_t nky, int64_t nkz,
                  double scale, int64_t forward, int64_t out_layout, std::string_view cache_dir) {
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
    if (nk > ::kdft::kNkMax)
        return fail("kdft-nk", "got nk=" + std::to_string(nk) + "; want nk <= " + std::to_string(::kdft::kNkMax) +
                    " (the verified range); the router keeps larger grids on XLA", ffi::ErrorCode::kInvalidArgument);
    const MinorPlan pl = ::kdft::plan_kminor(static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                                             mode, forward ? 1 : 0, out, f32 ? 8 : 16,
                                             1LL * d[1] * d[2] * d[3] * d[4]);
    if (pl.rows.c == 0)
        return fail("kdft-kminor-tile", "got k-grid " + std::to_string(nkx) + "x" + std::to_string(nky) + "x" +
                    std::to_string(nkz) + "; want a row pass whose k tile fits a block's " +
                    std::to_string(::kdft::kLdsMax / 1024) + " KiB of LDS", ffi::ErrorCode::kInvalidArgument);
    const void* x = X.untyped_data();
    const void* kp = K ? K->untyped_data() : nullptr;
    void* y = Y->untyped_data();
    int rows_i = static_cast<int>(rows), fwd = forward ? 1 : 0;
    auto row_pass = [&](const Built* b, const void* src, void* dst, int dir, double sc) -> ffi::Error {
        double scale64 = sc;
        float scale32 = static_cast<float>(sc);
        int f = dir;
        void* args[] = {&src, &kp, &dst, &rows_i, &d[1], &d[2], &d[3], &d[4],
                        f32 ? static_cast<void*>(&scale32) : static_cast<void*>(&scale64), &f};
        const long long blocks = ::kdft::row_blocks(rows, pl.ot, b->rows, b->order);
        if (blocks > 0x7fffffffLL) return bad("want at most 2^31-1 row-pass blocks");
        const rtc::Result r = rtc::driver_api().LaunchKernel(
            b->fn, static_cast<unsigned>(blocks), 1, 1, kThreads, 1, 1, 0,
            reinterpret_cast<rtc::Stream>(stream), args, nullptr);
        if (r != rtc::kOk) return fail("kdft-launch", LRX_MODULE_API "ModuleLaunchKernel: " + rtc::err_string(r));
        return ffi::Error::Success();
    };

    if (!pl.split) {
        const Built* b = nullptr;
        FFI_RETURN_IF_ERROR(build(mode, out, pl.grid, 1, pl.rows, f32, cache_dir, &b));
        return row_pass(b, x, y, fwd, scale);
    }

    // The split arm.  Row-pass images: transform-only (mode 5); the last pass of a mode-4
    // out_layout-1 call stores the permuted layout.
    if (rows * pl.ot > INT_MAX) return bad("want at most 2^31-1 row-pass rows (rows * O_T)");
    const Built* rfirst = nullptr;
    const Built* rlast = nullptr;
    FFI_RETURN_IF_ERROR(build(5, 0, pl.grid, pl.ot, pl.rows, f32, cache_dir, &rfirst));
    if (mode == 4) FFI_RETURN_IF_ERROR(build(5, out, pl.grid, pl.ot, pl.last, f32, cache_dir, &rlast, pl.order));
    std::vector<rtc::Function> fns(pl.cols.size());
    for (size_t i = 0; i < pl.cols.size(); ++i)
        FFI_RETURN_IF_ERROR(build_pass(pl.cols[i], f32, cache_dir, &fns[i]));
    void* work = y;
    const auto work_bytes = ::kdft::kminor_workspace_bytes(pl, mode, out, X.size_bytes());
    if (work_bytes) {
        std::optional<void*> buf = scratch->Allocate(work_bytes);   // XLA's own alignment
        if (!buf.has_value())
            return fail("kdft-scratch", "got no " + std::to_string(X.size_bytes()) + "-byte scratch buffer from XLA "
                        "for the split arm's out_layout 1 (the permuted store cannot run in place)",
                        ffi::ErrorCode::kResourceExhausted);
        work = *buf;
    }
    const bool last_is_col = mode == 5;                          // mode 5 ends on a column pass
    FFI_RETURN_IF_ERROR(row_pass(rfirst, x, work, mode == 5 ? fwd : 0,
                                 mode == 5 && pl.cols.empty() ? scale : 1.0));
    int m0 = d[1] * d[2], m1 = d[3] * d[4];
    for (size_t i = 0; i < pl.cols.size(); ++i) {
        const Pass& p = pl.cols[i];
        const bool scaled = p.mode == 4 || (last_is_col && i + 1 == pl.cols.size());
        void* t = work;
        const void* v = kp;
        long long inner = p.inner_k;
        long long ntile = (inner + p.w - 1) / p.w;
        long long kob = p.outer;
        const long long blocks = rows * kob * ntile;
        if (blocks > 0x7fffffffLL) return bad("want at most 2^31-1 column tiles per pass");
        int one = 1, f = p.forward;
        double scale64 = scaled ? scale : 1.0;
        float scale32 = static_cast<float>(scale64);
        void* args[] = {&t, &v, &inner, &ntile, &one, &one, &one,
                        f32 ? static_cast<void*>(&scale32) : static_cast<void*>(&scale64), &f, &kob, &m0, &m1};
        const rtc::Result r = rtc::driver_api().LaunchKernel(
            fns[i], static_cast<unsigned>(blocks), 1, 1, static_cast<unsigned>(p.nt), 1, 1, 0,
            reinterpret_cast<rtc::Stream>(stream), args, nullptr);
        if (r != rtc::kOk) return fail("kdft-launch", LRX_MODULE_API "ModuleLaunchKernel: " + rtc::err_string(r));
    }
    if (mode == 4) FFI_RETURN_IF_ERROR(row_pass(rlast, work, y, 1, 1.0));
    return ffi::Error::Success();
}

}  // namespace

static ffi::Error KminorConv(lrx::gpu::Stream s, ffi::ScratchAllocator scratch, ffi::AnyBuffer X,
                             ffi::AnyBuffer K, ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky,
                             int64_t nkz, double scale, int64_t out_layout, std::string_view cubin_dir) {
    return launch(s, &scratch, 4, X, &K, U, nkx, nky, nkz, scale, 0, out_layout, cubin_dir);
}

static ffi::Error KminorFft(lrx::gpu::Stream s, ffi::AnyBuffer X, ffi::Result<ffi::AnyBuffer> Y,
                            int64_t nkx, int64_t nky, int64_t nkz, double scale, int64_t forward,
                            std::string_view cubin_dir) {
    return launch(s, nullptr, 5, X, nullptr, Y, nkx, nky, nkz, scale, forward, 0, cubin_dir);
}

}  // namespace lorrax_ffi::kdft

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KDftKconvKminorFfi, lorrax_ffi::kdft::KminorConv,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Ctx<xla::ffi::ScratchAllocator>()
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
