// plane_gather_ffi.cc -- the ROCM leg's mode 10: the route-G plane FFT read straight from its cylinder
// (ffi.fft.make_plane_fft_gather, LocalFourierPlan(in_gather=...)), the port of kconv_mathdx_cuda_ffi.cc's
// PlaneFftGather (kPlaneSrc) with the kdft direct DFT as the factor-line transform
// (kdft/plane_gather_kernel.h; the k-box stage header supplies the compile-time twiddles).
//
//   lorrax_kdft_plane_fft_gather   F (A, S, *R, n_col), gidx (rows, n_c), row_of (rows,), start ()
//                                  -> Y (A, n_pg, *R, n_b, n_c)
//
// The operands and attributes of lorrax_mathdx_plane_fft_gather less mathdx_root.  Residency: PB whole
// (n_b, n_c|1) planes per block, PB the most (<= 8) whose planes and static tables fit half a block's
// 64 KiB of LDS (two blocks per CU), else one plane in all of it; a plane that does not fit is refused
// (GATE kdft-plane-residency) and ffi.fft.kdft_plane_fft_refusal keeps it on XLA by the same rule (as
// it does an axis with no Good-Thomas split into factors <= 40, GATE kdft-plane-split).  Gated against
// the XLA route on the same device (ffi.fft.xla_reference; tests/test_kdft_parent_gate.py).
//
// Leg-neutral host code (lrx::gpu, rtc); CMake lists it in the ROCM leg.  Images: one per (plane,
// splits, occupied rows, device architecture), disk-cached as plane_m10_<nb>x<nc>_r<rows>_<arch>_<key>.hsaco.
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
#include "generated/kbox_stage_src.h"
#include "generated/plane_gather_src.h"

namespace lorrax_ffi::plane_gather {

static constexpr long long kLds = LRX_LEG_BLOCK_LDS;   // 64 KiB on ROCM (common/leg.h)
static constexpr int kFactorMax = 40;                   // the direct-DFT factor line (ffi.fft.KCONV_AXIS_MAX)
static constexpr int kThreads = 256;
static constexpr int kPbMax = 8;
static constexpr int kGridCap = 110 * 16;               // grid-stride: 110 CUs

static ffi::Error fail(const char* gate, const std::string& why,
                       ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
    return ffi::Error(code, std::string("GATE ") + gate + ": " + why);
}

struct PlaneGather {                                    // plane_gather_kernel.h, field for field
    const int* gidx;
    const int* row_of;
    const int* start;
    long long rows, n_col, planes, s_len, n_pg, inner;
};

// LDS of PB planes plus the kernel's static tables live[n_b] (1 B), rowb[rows] (4 B), foff[PB] (8 B),
// 16 B slack (ffi.fft.kdft_plane_fft_pb mirrors it).
static long long lds_bytes(int nb, int nc, int rows, int pb) {
    return 16LL * nb * (nc | 1) * pb + nb + 4LL * std::max(1, rows) + 8LL * pb + 16;
}
static int plan_pb(int nb, int nc, int rows) {
    int pb = 0;
    for (int p = 1; p <= kPbMax; ++p)
        if (lds_bytes(nb, nc, rows, p) <= kLds / 2) pb = p;
    if (pb == 0 && lds_bytes(nb, nc, rows, 1) <= kLds) pb = 1;
    return pb;
}

static bool split_ok(int64_t n, int64_t n1) {
    if (n < 2 || n1 < 2 || n % n1 != 0 || n1 > kFactorMax || n / n1 > kFactorMax) return false;
    int64_t a = n1, b = n / n1;
    while (b) { const int64_t t = a % b; a = b; b = t; }
    return a == 1;
}

struct Built { rtc::Function fn = nullptr; int pb = 0; long long smem = 0; };
using Key = std::tuple<int, int, int, int, int, int>;   // dev, nb, nc, b1, c1, rows
static std::mutex g_mu;
static std::map<Key, Built> g_cache;
static std::map<Key, std::string> g_fail;

static ffi::Error build(int nb, int nc, int b1, int c1, int rows, std::string_view cache_dir, const Built** out) {
    int dev = 0;
    LORRAX_GPU_CHECK(lrx::gpu::get_device(&dev));
    const Key key{dev, nb, nc, b1, c1, rows};
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_cache.find(key); it != g_cache.end()) { *out = &it->second; return ffi::Error::Success(); }
    if (auto it = g_fail.find(key); it != g_fail.end()) return fail("kdft-plane-build", "(cached failure) " + it->second);
    const int pb = plan_pb(nb, nc, rows);
    if (pb == 0) {
        std::ostringstream os;
        os << "got plane (" << nb << "," << nc << ") whose resident plane needs " << lds_bytes(nb, nc, rows, 1)
           << " B (16*n_b*(n_c|1) + static tables); want it in a block's " << kLds / 1024 << " KiB of LDS; the router "
              "keeps such planes on XLA (ffi.fft.kdft_plane_fft_refusal)";
        return fail("kdft-plane-residency", os.str(), ffi::ErrorCode::kInvalidArgument);
    }
    std::string arch;
    LORRAX_GPU_CHECK(lrx::gpu::device_arch(dev, &arch));
    rtc::Program prog;
    prog.src = kKernelSrc;
    prog.name = "lrx_plane_gather";
    prog.kernel = "lrx_plane_fft_gather";
    prog.headers = {{lorrax_ffi::kbox::kHeaderName, lorrax_ffi::kbox::kHeaderSrc}};
    prog.defs = {rtc::kOptOption, "-std=c++17", rtc::arch_option(arch),
                 "-DLRX_NB=" + std::to_string(nb), "-DLRX_NC=" + std::to_string(nc),
                 "-DLRX_B1=" + std::to_string(b1), "-DLRX_C1=" + std::to_string(c1),
                 "-DLRX_ROWS=" + std::to_string(std::max(1, rows)), "-DLRX_PB=" + std::to_string(pb),
                 "-DLRX_NT=" + std::to_string(kThreads)};
    const std::string key_hex = rtc::hex16(rtc::key(prog, nullptr));
    const std::string dir(cache_dir);
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/plane_m10_" << nb << "x" << nc << "_r" << rows << "_" << rtc::arch_tag(arch) << "_" << key_hex
             << rtc::kImageExt;
        path = name.str();
    }
    rtc::Image img;
    std::string where, err;
    if (!rtc::build(prog, dir, path, key_hex, &img, &where, &err)) {
        g_fail.emplace(key, where + " -- " + err);
        return fail("kdft-plane-build", where + " -- " + err);
    }
    Built b;
    b.fn = img.fn;
    b.pb = pb;
    b.smem = 16LL * nb * (nc | 1) * pb;                 // dynamic LDS; the static tables are the kernel's own
    if (!img.from_disk)
        std::fprintf(stderr, "[kconv_kbox] " LRX_RTC_API " %s mode=10 plane=(%d,%d) splits=(%d,%d) rows=%d %s pb=%d "
                     "threads=%d lds=%lld B in %.1f ms%s\n", img.rebuilt_bad ? "rebuilt (cached image refused)" : "built",
                     nb, nc, b1, c1, rows, arch.c_str(), pb, kThreads, b.smem, img.ms, img.stored ? " (stored)" : "");
    *out = &g_cache.emplace(key, b).first->second;
    return ffi::Error::Success();
}

static ffi::Error Gather(lrx::gpu::Stream stream, ffi::AnyBuffer F, ffi::AnyBuffer gidx, ffi::AnyBuffer row_of,
                         ffi::AnyBuffer start, ffi::Result<ffi::AnyBuffer> Y, int64_t nb, int64_t nc, int64_t b1,
                         int64_t c1, std::string_view cubin_dir) {
    auto bad = [](const std::string& why) { return fail("kdft-plane-operands", why, ffi::ErrorCode::kInvalidArgument); };
    // A prime power <= 40 runs as one factor line: (n, 1).
    auto ok = [](int64_t n, int64_t n1) { return (n1 == n && n >= 1 && n <= kFactorMax) || split_ok(n, n1); };
    if (!ok(nb, b1) || !ok(nc, c1)) {
        std::ostringstream os;
        os << "got plane (" << nb << "," << nc << ") with splits (" << b1 << "," << c1 << "); want n = n1*n2 with "
              "gcd(n1, n2) = 1 and 2 <= n1, n2 <= " << kFactorMax << " (n2 = 1 when n <= " << kFactorMax << "); the "
              "router (ffi.fft.make_plane_fft_gather) keeps such planes on XLA";
        return fail("kdft-plane-split", os.str(), ffi::ErrorCode::kInvalidArgument);
    }
    const auto C = ffi::DataType::C128, I = ffi::DataType::S32;
    auto fd = F.dimensions(), yd = Y->dimensions(), gd = gidx.dimensions(), rd = row_of.dimensions();
    if (F.element_type() != C || Y->element_type() != C || gidx.element_type() != I || row_of.element_type() != I ||
        start.element_type() != I || start.dimensions().size() != 0 || fd.size() < 1 || yd.size() != fd.size() + 1 ||
        yd[yd.size() - 2] != nb || yd[yd.size() - 1] != nc || gd.size() != 2 || gd[1] != nc || rd.size() != 1 ||
        rd[0] != gd[0] || gd[0] > nb)
        return bad("want F (A, S, *R, n_col) c128, Y (A, n_pg, *R, n_b, n_c) c128, gidx (rows, n_c) s32, "
                   "row_of (rows,) s32 with rows <= n_b and start () s32");
    const size_t L = fd.size() - 1;
    const long long A = L >= 1 ? fd[0] : 1, S = L >= 2 ? fd[1] : 1, n_pg = L >= 2 ? yd[1] : 1;
    long long inner = 1;
    for (size_t i = 2; i < L; ++i) {
        if (fd[i] != yd[i]) return bad("F and Y trailing batch dimensions differ");
        inner *= fd[i];
    }
    if ((L >= 1 && yd[0] != A) || n_pg > S) return bad("want Y (A, n_pg <= S, *R, n_b, n_c)");
    const long long planes = A * n_pg * inner;
    if (planes == 0) return ffi::Error::Success();
    const Built* k = nullptr;
    FFI_RETURN_IF_ERROR(build(static_cast<int>(nb), static_cast<int>(nc), static_cast<int>(b1), static_cast<int>(c1),
                              static_cast<int>(gd[0]), cubin_dir, &k));
    PlaneGather g{static_cast<const int*>(gidx.untyped_data()), static_cast<const int*>(row_of.untyped_data()),
                  static_cast<const int*>(start.untyped_data()), gd[0], fd[fd.size() - 1], planes, S, n_pg, inner};
    const void* fp = F.untyped_data();
    void* yp = Y->untyped_data();
    void* args[] = {(void*)&fp, (void*)&yp, (void*)&g};
    const long long blocks = std::min<long long>((planes + k->pb - 1) / k->pb, kGridCap);
    const rtc::Result r = rtc::driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, kThreads, 1, 1,
                                                         static_cast<unsigned>(k->smem), reinterpret_cast<rtc::Stream>(stream),
                                                         args, nullptr);
    if (r != rtc::kOk) return fail("kdft-plane-launch", LRX_MODULE_API "ModuleLaunchKernel: " + rtc::err_string(r));
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::plane_gather

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KDftPlaneFftGatherFfi, lorrax_ffi::plane_gather::Gather,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()     // F (A, S, *R, n_col)
        .Arg<xla::ffi::AnyBuffer>()     // gidx (rows, n_c) s32
        .Arg<xla::ffi::AnyBuffer>()     // row_of (rows,) s32
        .Arg<xla::ffi::AnyBuffer>()     // start () s32
        .Ret<xla::ffi::AnyBuffer>()     // Y (A, n_pg, *R, n_b, n_c)
        .Attr<int64_t>("nb")
        .Attr<int64_t>("nc")
        .Attr<int64_t>("b1")
        .Attr<int64_t>("c1")
        .Attr<std::string_view>("cubin_dir"));
