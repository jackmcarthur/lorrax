// active_devsize.cc -- the active-subspace handlers with DEVICE-side sizes (no host sync).
//
// The same targets, operands, layouts and aliases as active_ops.cc (the CUDA leg's cuBLAS
// handlers), but no handler reads its descriptor on the host: the kernels of
// active_dev_kernel.h (rtc-compiled, generated/active_dev_src.h) read [start, count] (and m,
// columns, compute_image) on the device, so a call is a few stream-ordered launches and never
// drains the stream (owner rule: no per-call D2H in hot paths).  Scratch for the k-split
// partials is the call's declared `blas` result (XLA-owned, 4 MiB): no private allocation.
// Invalid descriptors give NaN outputs (a kernel cannot raise without a host read).
//
// Leg-neutral (gpu_runtime.h + rtc_build.h only); CMake builds it into the ROCM leg, where it
// replaces active_ops.cc.  The CUDA leg keeps active_ops.cc until a CUDA device can gate this
// source (a one-line CMake change).  The projected eigensolve is not here: the ROCM plan solves
// the shifted, reflected capacity matrix through lorrax_rocsolver_batched_eigh (no size read).
#include <climits>
#include <cstdint>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>

#include "../common/ffi_helpers.h"
#include "../common/gpu_runtime.h"
#include "../common/rtc_build.h"
#include "generated/active_dev_src.h"

namespace lorrax_ffi::active_dev {

using B = ffi::BufferR2<ffi::C128>;
using I = ffi::BufferR1<ffi::S32>;
using R = ffi::Result<B>;
using Scratch = ffi::BufferR1<ffi::U8>;
using ScratchResult = ffi::Result<Scratch>;

enum Kernel { kPgemm, kReduce, kFillGram, kMirror, kCombine, kStore, kCount };
static const char* const kNames[kCount] = {"lrx_as_pgemm", "lrx_as_pgemm_reduce",
                                           "lrx_as_fill_gram", "lrx_as_project_mirror",
                                           "lrx_as_combine", "lrx_as_store"};
constexpr int kNt = 64;          // LRX_NT of the product kernels
constexpr int kEw = 256;         // threads of the elementwise kernels
constexpr int kTarget = 1024;    // product work items aimed for (persistent grid)
constexpr int kTileBytes = 4 * 8 * 16;

static ffi::Error fail(const char* gate, const std::string& why,
                       ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
    return ffi::Error(code, std::string("GATE ") + gate + ": " + why);
}

struct Fns { rtc::Function f[kCount]; };

static ffi::Error kernels(std::string_view cache_dir, const Fns** out) {
    static std::mutex mu;
    static std::map<int, Fns> cache;
    static std::map<int, std::string> failed;
    int dev = 0;
    LORRAX_GPU_CHECK(lrx::gpu::get_device(&dev));
    std::lock_guard<std::mutex> lock(mu);
    if (auto it = cache.find(dev); it != cache.end()) { *out = &it->second; return ffi::Error::Success(); }
    if (auto it = failed.find(dev); it != failed.end()) return fail("active-dev-build", "(cached failure) " + it->second);
    std::string arch;
    LORRAX_GPU_CHECK(lrx::gpu::device_arch(dev, &arch));
    rtc::Program prog;
    prog.src = kActiveDevSrc;
    prog.name = "lrx_active_dev";
    prog.kernel = kNames[0];
    for (int k = 1; k < kCount; ++k) prog.entries.push_back(kNames[k]);
    prog.defs = {rtc::kOptOption, "-std=c++17", rtc::arch_option(arch)};
    const std::string key_hex = rtc::hex16(rtc::key(prog, nullptr));
    const std::string dir(cache_dir);
    std::string path;
    if (!dir.empty()) path = dir + "/active_dev_" + rtc::arch_tag(arch) + "_" + key_hex + rtc::kImageExt;
    rtc::Image img;
    std::string where, err;
    if (!rtc::build(prog, dir, path, key_hex, &img, &where, &err)) {
        failed.emplace(dev, where + " -- " + err);
        return fail("active-dev-build", where + " -- " + err);
    }
    Fns f;
    f.f[0] = img.fn;
    for (int k = 1; k < kCount; ++k) f.f[k] = img.fns[k - 1];
    *out = &cache.emplace(dev, f).first->second;
    return ffi::Error::Success();
}

static ffi::Error launch(const Fns* f, Kernel k, long long work, int nt, lrx::gpu::Stream s,
                         void** args) {
    long long grid = (work + nt - 1) / nt;
    if (k == kPgemm) grid = work;                          // one block per work item, capped
    if (grid < 1) grid = 1;
    if (grid > 4096) grid = 4096;
    const rtc::Result r = rtc::driver_api().LaunchKernel(
        f->f[k], static_cast<unsigned>(grid), 1, 1, static_cast<unsigned>(nt), 1, 1, 0,
        reinterpret_cast<rtc::Stream>(s), args, nullptr);
    if (r != rtc::kOk)
        return fail("active-dev-launch", std::string(kNames[k]) + ": " LRX_MODULE_API "ModuleLaunchKernel: " + rtc::err_string(r));
    return ffi::Error::Success();
}

// out[ra + cap*rb] = <A_ra, B_rb> over the window of `mode` (0 Gram, 1 project).
static ffi::Error product(const Fns* f, lrx::gpu::Stream s, const void* a, const void* b,
                          void* out, ScratchResult& scratch, const int32_t* q, int mode,
                          int cap, int nbs, long long d) {
    void* part = scratch->typed_data();
    long long budget = static_cast<long long>(scratch->dimensions()[0]) / kTileBytes;
    int mode_ = mode, cap_ = cap, nbs_ = nbs, target = kTarget;
    long long d_ = d;
    const void* q_ = q;
    void* args[] = {&a, &b, &out, &part, &q_, &mode_, &cap_, &nbs_, &d_, &budget, &target};
    FFI_RETURN_IF_ERROR(launch(f, kPgemm, kTarget, kNt, s, args));
    void* rargs[] = {&out, &part, &q_, &mode_, &cap_, &nbs_, &d_, &budget, &target};
    const long long tiles = static_cast<long long>((cap + 3) / 4) * ((mode ? cap : nbs) + 7) / 8 * 32;
    return launch(f, kReduce, tiles, kEw, s, rargs);
}

static ffi::Error fill_gram(const Fns* f, lrx::gpu::Stream s, void* out, const int32_t* q,
                            int cap, int nbs) {
    const void* q_ = q;
    int cap_ = cap, nbs_ = nbs;
    void* args[] = {&out, &q_, &cap_, &nbs_};
    return launch(f, kFillGram, static_cast<long long>(cap) * nbs, kEw, s, args);
}

static ffi::Error combine(const Fns* f, lrx::gpu::Stream s, void* x, const void* v, const void* c,
                          const int32_t* q, int dk, int mode, int cap, int nb, long long d) {
    const void* q_ = q;
    int dk_ = dk, mode_ = mode, cap_ = cap, nb_ = nb;
    long long d_ = d;
    void* args[] = {&x, &v, &c, &q_, &dk_, &mode_, &cap_, &nb_, &d_};
    const long long items = ((d + kNt - 1) / kNt) * ((nb + 7) / 8);
    return launch(f, kCombine, items * kNt, kNt, s, args);
}

static bool fits(int64_t n) { return n >= 1 && n <= INT_MAX; }

static ffi::Error project(lrx::gpu::Stream s, B v, B hv, B h, I active, R out,
                          ScratchResult scratch, std::string_view cubin_dir) {
    if (active.dimensions()[0] != 3 || !fits(v.dimensions()[0]) || !fits(v.dimensions()[1]))
        return ffi::Error::InvalidArgument("project descriptor/dimension overflow");
    const int cap = v.dimensions()[0];
    const long long d = v.dimensions()[1];
    if (hv.dimensions()[0] != cap || hv.dimensions()[1] != d || h.dimensions()[0] != cap ||
        h.dimensions()[1] != cap || out->dimensions()[0] != cap || out->dimensions()[1] != cap)
        return ffi::Error::InvalidArgument("project buffer geometry");
    if (out->typed_data() != h.typed_data())
        return ffi::Error::InvalidArgument("active projection requires declared input/output alias");
    const Fns* f = nullptr;
    FFI_RETURN_IF_ERROR(kernels(cubin_dir, &f));
    const int32_t* q = active.typed_data();
    FFI_RETURN_IF_ERROR(product(f, s, v.typed_data(), hv.typed_data(), out->typed_data(), scratch,
                                q, 1, cap, cap, d));
    void* o = out->typed_data();
    const void* q_ = q;
    int cap_ = cap;
    void* args[] = {&o, &q_, &cap_};
    return launch(f, kMirror, static_cast<long long>(cap) * cap, kEw, s, args);
}

static ffi::Error reconstruct(lrx::gpu::Stream s, B v, B hv, B c, I active, R x, R hx,
                              ScratchResult scratch, std::string_view cubin_dir) {
    (void)scratch;
    if (active.dimensions()[0] != 4 || !fits(v.dimensions()[0]) || !fits(v.dimensions()[1]) ||
        !fits(x->dimensions()[0]))
        return ffi::Error::InvalidArgument("reconstruct descriptor/dimension overflow");
    const int cap = v.dimensions()[0], b = x->dimensions()[0];
    const long long d = v.dimensions()[1];
    if (hv.dimensions()[0] != cap || hv.dimensions()[1] != d || c.dimensions()[0] != cap ||
        c.dimensions()[1] != b || x->dimensions()[1] != d || hx->dimensions()[0] != b ||
        hx->dimensions()[1] != d)
        return ffi::Error::InvalidArgument("reconstruct buffer geometry");
    const Fns* f = nullptr;
    FFI_RETURN_IF_ERROR(kernels(cubin_dir, &f));
    const int32_t* q = active.typed_data();
    FFI_RETURN_IF_ERROR(combine(f, s, x->typed_data(), v.typed_data(), c.typed_data(), q, 1, 0, cap, b, d));
    return combine(f, s, hx->typed_data(), hv.typed_data(), c.typed_data(), q, 1, 2, cap, b, d);
}

static ffi::Error orthogonalize(lrx::gpu::Stream s, B v, B p, I active, R out, R work,
                                ScratchResult scratch, std::string_view cubin_dir) {
    if (active.dimensions()[0] != 2 || !fits(v.dimensions()[0]) || !fits(v.dimensions()[1]) ||
        !fits(p.dimensions()[0]))
        return ffi::Error::InvalidArgument("orthogonalize descriptor/dimension overflow");
    const int cap = v.dimensions()[0], b = p.dimensions()[0];
    const long long d = v.dimensions()[1];
    if (p.dimensions()[1] != d || out->dimensions()[0] != b || out->dimensions()[1] != d ||
        work->dimensions()[0] != cap || work->dimensions()[1] != b)
        return ffi::Error::InvalidArgument("orthogonalize buffer geometry");
    const Fns* f = nullptr;
    FFI_RETURN_IF_ERROR(kernels(cubin_dir, &f));
    if (out->typed_data() != p.typed_data())
        LORRAX_GPU_CHECK(lrx::gpu::memcpy_async(out->typed_data(), p.typed_data(),
                                                static_cast<size_t>(b) * d * 16,
                                                lrx::gpu::kDeviceToDevice, s));
    const int32_t* q = active.typed_data();
    for (int pass = 0; pass < 2; ++pass) {
        FFI_RETURN_IF_ERROR(product(f, s, v.typed_data(), out->typed_data(), work->typed_data(),
                                    scratch, q, 0, cap, b, d));
        FFI_RETURN_IF_ERROR(combine(f, s, out->typed_data(), v.typed_data(), work->typed_data(), q,
                                    0, 1, cap, b, d));
    }
    return ffi::Error::Success();
}

static ffi::Error check_subtract(B v, B p, B coefficients, I range, R out) {
    if (range.dimensions()[0] != 2 || !fits(v.dimensions()[0]) || !fits(v.dimensions()[1]) ||
        !fits(p.dimensions()[0]))
        return ffi::Error::InvalidArgument("active subspace subtract descriptor/dimension overflow");
    if (p.dimensions()[1] != v.dimensions()[1] || coefficients.dimensions()[0] != v.dimensions()[0] ||
        coefficients.dimensions()[1] != p.dimensions()[0] || out->dimensions()[0] != p.dimensions()[0] ||
        out->dimensions()[1] != p.dimensions()[1])
        return ffi::Error::InvalidArgument("active subspace subtract buffer geometry");
    if (out->typed_data() != p.typed_data())
        return ffi::Error::InvalidArgument("active subspace subtract requires declared P output alias");
    return ffi::Error::Success();
}

static ffi::Error subtract(lrx::gpu::Stream s, B v, B p, B coefficients, I range, R out,
                           ScratchResult scratch, std::string_view cubin_dir) {
    (void)scratch;
    FFI_RETURN_IF_ERROR(check_subtract(v, p, coefficients, range, out));
    const Fns* f = nullptr;
    FFI_RETURN_IF_ERROR(kernels(cubin_dir, &f));
    return combine(f, s, out->typed_data(), v.typed_data(), coefficients.typed_data(),
                   range.typed_data(), 0, 1, v.dimensions()[0], p.dimensions()[0], v.dimensions()[1]);
}

static ffi::Error subtract_gram(lrx::gpu::Stream s, B v, B p, B coefficients, I range, R out,
                                R next, ScratchResult scratch, std::string_view cubin_dir) {
    FFI_RETURN_IF_ERROR(check_subtract(v, p, coefficients, range, out));
    if (next->dimensions()[0] != coefficients.dimensions()[0] ||
        next->dimensions()[1] != coefficients.dimensions()[1] ||
        next->typed_data() != coefficients.typed_data())
        return ffi::Error::InvalidArgument("active subspace subtract/Gram requires declared C output alias");
    const Fns* f = nullptr;
    FFI_RETURN_IF_ERROR(kernels(cubin_dir, &f));
    const int cap = v.dimensions()[0], b = p.dimensions()[0];
    const long long d = v.dimensions()[1];
    const int32_t* q = range.typed_data();
    FFI_RETURN_IF_ERROR(combine(f, s, out->typed_data(), v.typed_data(), coefficients.typed_data(),
                                q, 0, 1, cap, b, d));
    FFI_RETURN_IF_ERROR(fill_gram(f, s, next->typed_data(), q, cap, b));
    return product(f, s, v.typed_data(), out->typed_data(), next->typed_data(), scratch, q, 0, cap, b, d);
}

static ffi::Error gram(lrx::gpu::Stream s, B v, B p, I range, R out, ScratchResult scratch,
                       std::string_view cubin_dir) {
    if (range.dimensions()[0] != 2 || !fits(v.dimensions()[0]) || !fits(v.dimensions()[1]) ||
        !fits(p.dimensions()[0]))
        return ffi::Error::InvalidArgument("gram descriptor/dimension overflow");
    const int cap = v.dimensions()[0], b = p.dimensions()[0];
    const long long d = v.dimensions()[1];
    if (p.dimensions()[1] != d || out->dimensions()[0] != cap || out->dimensions()[1] != b)
        return ffi::Error::InvalidArgument("gram buffer/range geometry");
    const Fns* f = nullptr;
    FFI_RETURN_IF_ERROR(kernels(cubin_dir, &f));
    FFI_RETURN_IF_ERROR(fill_gram(f, s, out->typed_data(), range.typed_data(), cap, b));
    return product(f, s, v.typed_data(), p.typed_data(), out->typed_data(), scratch,
                   range.typed_data(), 0, cap, b, d);
}

static ffi::Error store(lrx::gpu::Stream s, B v, B hv, B p, B hp, I range, R out, R hout,
                        std::string_view cubin_dir) {
    if (range.dimensions()[0] != 2 || v.dimensions()[0] != hv.dimensions()[0] ||
        v.dimensions()[1] != hv.dimensions()[1] || p.dimensions()[0] != hp.dimensions()[0] ||
        p.dimensions()[1] != hp.dimensions()[1] || v.dimensions()[1] != p.dimensions()[1] ||
        !fits(v.dimensions()[0]) || !fits(p.dimensions()[0]))
        return ffi::Error::InvalidArgument("active store buffer geometry");
    if (out->typed_data() != v.typed_data() || hout->typed_data() != hv.typed_data())
        return ffi::Error::InvalidArgument("active store requires declared input/output aliases");
    const Fns* f = nullptr;
    FFI_RETURN_IF_ERROR(kernels(cubin_dir, &f));
    void* o = out->typed_data();
    void* ho = hout->typed_data();
    const void* pp = p.typed_data();
    const void* hpp = hp.typed_data();
    const void* q = range.typed_data();
    int cap = v.dimensions()[0], pb = p.dimensions()[0];
    long long d = v.dimensions()[1];
    void* args[] = {&o, &ho, &pp, &hpp, &q, &cap, &pb, &d};
    return launch(f, kStore, static_cast<long long>(pb) * d, kEw, s, args);
}

}  // namespace lorrax_ffi::active_dev

using namespace lorrax_ffi::active_dev;
namespace xffi = ::xla::ffi;
#define LRX_CTX .Ctx<xffi::PlatformStream<lrx::gpu::Stream>>()
#define LRX_DIR .Attr<std::string_view>("cubin_dir")

XLA_FFI_DEFINE_HANDLER_SYMBOL(ActiveSubspaceProjectFfi, project,
    xffi::Ffi::Bind() LRX_CTX .Arg<B>().Arg<B>().Arg<B>().Arg<I>().Ret<B>().Ret<Scratch>() LRX_DIR);
XLA_FFI_DEFINE_HANDLER_SYMBOL(ActiveSubspaceReconstructFfi, reconstruct,
    xffi::Ffi::Bind() LRX_CTX .Arg<B>().Arg<B>().Arg<B>().Arg<I>().Ret<B>().Ret<B>().Ret<Scratch>() LRX_DIR);
XLA_FFI_DEFINE_HANDLER_SYMBOL(ActiveSubspaceOrthoFfi, orthogonalize,
    xffi::Ffi::Bind() LRX_CTX .Arg<B>().Arg<B>().Arg<I>().Ret<B>().Ret<B>().Ret<Scratch>() LRX_DIR);
XLA_FFI_DEFINE_HANDLER_SYMBOL(ActiveSubspaceSubtractFfi, subtract,
    xffi::Ffi::Bind() LRX_CTX .Arg<B>().Arg<B>().Arg<B>().Arg<I>().Ret<B>().Ret<Scratch>() LRX_DIR);
XLA_FFI_DEFINE_HANDLER_SYMBOL(ActiveSubspaceSubtractGramFfi, subtract_gram,
    xffi::Ffi::Bind() LRX_CTX .Arg<B>().Arg<B>().Arg<B>().Arg<I>().Ret<B>().Ret<B>().Ret<Scratch>() LRX_DIR);
XLA_FFI_DEFINE_HANDLER_SYMBOL(ActiveSubspaceGramFfi, gram,
    xffi::Ffi::Bind() LRX_CTX .Arg<B>().Arg<B>().Arg<I>().Ret<B>().Ret<Scratch>() LRX_DIR);
XLA_FFI_DEFINE_HANDLER_SYMBOL(ActiveSubspaceStoreFfi, store,
    xffi::Ffi::Bind() LRX_CTX .Arg<B>().Arg<B>().Arg<B>().Arg<B>().Arg<I>().Ret<B>().Ret<B>() LRX_DIR);
