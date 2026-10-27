// hipfft/fourier_plan_rocm_ffi.cc -- lorrax_fourier_plan_rocm: one local separable DFT with
// per-axis supports as ONE custom call, the ROCM leg of common.fourier_plan.LocalFourierPlan.
//
// A port of the CUDA leg (cufft/fourier_plan_cuda_ffi.cc + fourier_plan.cu), same contract,
// same attributes as its pair-less target `lorrax_fourier_plan` plus `cubin_dir` (the disk
// cache of the remap kernel's hipRTC image), same plan geometry and step order:
//
//   y = R_out · F_{sign} · E_in · x   over the d <= 3 trailing axes of x,
//
// row-major in and out: x (B…, K_0, …, K_{d-1}) -> y (B…, K'_0, …, K'_{d-1}).  The Python entry
// point chooses the backend per axis and the stage order; this handler executes it:
//
//   * GEMM axis a, current shape (L, K, R): R = 1 is one rocblas_zgemm (op T on the matrix);
//     otherwise one rocblas_zgemm_strided_batched over L with the matrix's stride 0.  No
//     transposes; the output stays row-major (the CUDA leg's two cuBLAS calls, argument for
//     argument: rocBLAS is column-major with the same operation enums' meaning).
//   * FFT group: one remap pass that zero-fills and gathers the embedded axes, one hipFFT Z2Z
//     plan per contiguous run of FFT axes (cuFFT's plan-many geometry, unchanged), one remap
//     pass for the restricted axes.  The FFT axes' scale rides the first GEMM's alpha, else a
//     remap pass, else a final scale remap.
//   * No fused pair: the CUDA pair is a cuBLASDx kernel (nvidia-mathdx); rocBLAS has no
//     device-side GEMM API, so a GEMM pair runs as two rocBLAS GEMMs -- the path the CUDA leg
//     takes for a pair that does not fit shared memory or a wheel it has not validated.
//
// Libraries: none linked.  rocBLAS and hipFFT are resolved from the copies this process
// already mapped (the JAX ROCm plugin maps the wheel's librocblas / libhipfft), else from the
// directory of the wheel library that is mapped, so the process keeps ONE ROCm (the rocsolver
// family's rule).  The remap kernel is hipRTC-built through common/rtc_build (the ROCM leg
// compiles nothing for the device at build time) and disk-cached under `cubin_dir`.
//
// Plans (device matrices, remap tables, hipFFT plans) are cached per (device, attributes,
// shape) for the process lifetime.  hipFFT's work areas are NOT owned by the plans
// (auto-allocation off); rocBLAS's workspace is sized once per plan in rocBLAS size-query mode;
// both occupy an explicit uint8 result on every call, as do the ping-pong intermediates, each
// sized by the largest intermediate it holds (the one deliberate departure from the CUDA leg,
// which allocates two buffers of the chain's largest extent: Fe's memory law leaves no room
// for that on MI250X).
// Every hipFFT and rocBLAS size is checked against INT_MAX and refused by name.
#include "../common/leg.h"
#if !defined(LRX_LEG_ROCM)
#  error "hipfft/fourier_plan_rocm_ffi.cc is ROCM-leg only (the CUDA leg is cufft/fourier_plan_cuda_ffi.cc)"
#endif

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "../common/ffi_helpers.h"
#include "../common/c_abi.h"
#include "../common/gpu_runtime.h"
#include "../common/mkl_thread_pin.h"
#include "../common/rtc_build.h"

namespace lorrax_ffi::fourier_plan_rocm {

namespace ffi = ::xla::ffi;

static ffi::Error err(const std::string& what) {
    return ffi::Error(ffi::ErrorCode::kInternal, "lorrax_fourier_plan_rocm: " + what);
}
static ffi::Error bad(const std::string& what) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "lorrax_fourier_plan_rocm: " + what);
}

// ---- the rocBLAS / hipFFT C ABI this file uses, spelled here (no headers needed) ----------
// Handles are opaque pointers, enums C ints, rocblas_int int32 (LP64), rocblas_stride int64,
// complex double two doubles.
struct Z { double x, y; };
using RbHandle = void*;
using RbStatus = int;                         // rocblas_status: 0 = success
constexpr int kRbOpNone = 111, kRbOpTrans = 112;  // rocblas_operation_{none,transpose}
using RbZgemm = RbStatus (*)(RbHandle, int, int, int32_t, int32_t, int32_t, const Z*, const Z*,
                             int32_t, const Z*, int32_t, const Z*, Z*, int32_t);
using RbZgemmSb = RbStatus (*)(RbHandle, int, int, int32_t, int32_t, int32_t, const Z*, const Z*,
                               int32_t, int64_t, const Z*, int32_t, int64_t, const Z*, Z*,
                               int32_t, int64_t, int32_t);
using FftHandle = void*;                      // hipfftHandle (struct hipfftHandle_t*)
using FftResult = int;                        // hipfftResult: 0 = HIPFFT_SUCCESS
constexpr int kFftZ2Z = 0x69;                 // HIPFFT_Z2Z
constexpr int kFftForward = -1, kFftBackward = 1;

struct Api {
    RbStatus (*rb_create)(RbHandle*) = nullptr;
    RbStatus (*rb_set_stream)(RbHandle, lrx::gpu::Stream) = nullptr;
    RbStatus (*rb_set_workspace)(RbHandle, void*, size_t) = nullptr;
    RbStatus (*rb_start_query)(RbHandle) = nullptr;
    RbStatus (*rb_stop_query)(RbHandle, size_t*) = nullptr;
    RbZgemm zgemm = nullptr;
    RbZgemmSb zgemm_sb = nullptr;
    FftResult (*fft_create)(FftHandle*) = nullptr;
    FftResult (*fft_set_auto)(FftHandle, int) = nullptr;
    FftResult (*fft_make_many)(FftHandle, int, int*, int*, int, int, int*, int, int, int, int,
                               size_t*) = nullptr;
    FftResult (*fft_set_stream)(FftHandle, lrx::gpu::Stream) = nullptr;
    FftResult (*fft_set_work)(FftHandle, void*) = nullptr;
    FftResult (*fft_exec_z2z)(FftHandle, Z*, Z*, int) = nullptr;
    FftResult (*fft_destroy)(FftHandle) = nullptr;
    std::string rocblas_path, hipfft_path, err;
};

static std::string loaded_path(void* sym) {
    Dl_info info{};
    return (sym && dladdr(sym, &info) && info.dli_fname) ? std::string(info.dli_fname) : std::string("?");
}

// The wheel library `soname`: the mapped copy, else the one beside a mapped wheel library.
static void* open_mapped(const char* soname, std::string* where) {
    if (void* h = dlopen(soname, RTLD_NOW | RTLD_NOLOAD)) return h;
    static const char* kSiblings[] = {"librocblas.so.5", "libhipfft.so.0", "librocfft.so.0",
                                      "librocsolver.so.0", "libhipblaslt.so.1", "libhipsolver.so.1",
                                      "libhipblas.so.3"};
    for (const char* s : kSiblings) {
        void* h = dlopen(s, RTLD_NOW | RTLD_NOLOAD);
        if (h == nullptr) continue;
        Dl_info info{};
        void* any = dlsym(h, "rocblas_create_handle");
        if (any == nullptr) any = dlsym(h, "hipfftCreate");
        if (any == nullptr) any = dlsym(h, "rocfft_setup");
        if (any == nullptr) any = dlsym(h, "rocsolver_get_version_string");
        if (any == nullptr) any = dlsym(h, "hipblasLtCreate");
        if (any == nullptr) any = dlsym(h, "hipsolverCreate");
        if (any == nullptr) any = dlsym(h, "hipblasCreate");
        if (any == nullptr || !dladdr(any, &info) || info.dli_fname == nullptr) continue;
        std::string dir(info.dli_fname);
        dir = dir.substr(0, dir.find_last_of('/'));
        const std::string path = dir + "/" + soname;
        if (void* h2 = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)) { *where = path; return h2; }
    }
    return nullptr;
}

static const Api& api() {
    static const Api a = [] {
        Api r;
        std::string w1, w2;
        void* hb = open_mapped("librocblas.so.5", &w1);
        void* hf = open_mapped("libhipfft.so.0", &w2);
        if (hb == nullptr || hf == nullptr) {
            r.err = std::string("GATE fourier-plan-rocm-mapped: ") +
                    (hb == nullptr ? "librocblas.so.5 " : "") + (hf == nullptr ? "libhipfft.so.0 " : "") +
                    "is neither mapped in this process nor beside a mapped ROCm wheel library; want "
                    "the copies the JAX ROCm plugin maps (one ROCm per process); fix: import jax with "
                    "the rocm plugin before the first call";
            return r;
        }
        auto need = [&](void* h, const char* name) {
            void* p = dlsym(h, name);
            if (p == nullptr) r.err += std::string(r.err.empty() ? "" : ", ") + name;
            return p;
        };
#define LRX_BIND(lib, field, name) r.field = reinterpret_cast<decltype(r.field)>(need(lib, name))
        LRX_BIND(hb, rb_create, "rocblas_create_handle");
        LRX_BIND(hb, rb_set_stream, "rocblas_set_stream");
        LRX_BIND(hb, rb_set_workspace, "rocblas_set_workspace");
        LRX_BIND(hb, rb_start_query, "rocblas_start_device_memory_size_query");
        LRX_BIND(hb, rb_stop_query, "rocblas_stop_device_memory_size_query");
        LRX_BIND(hb, zgemm, "rocblas_zgemm");
        LRX_BIND(hb, zgemm_sb, "rocblas_zgemm_strided_batched");
        LRX_BIND(hf, fft_create, "hipfftCreate");
        LRX_BIND(hf, fft_set_auto, "hipfftSetAutoAllocation");
        LRX_BIND(hf, fft_make_many, "hipfftMakePlanMany");
        LRX_BIND(hf, fft_set_stream, "hipfftSetStream");
        LRX_BIND(hf, fft_set_work, "hipfftSetWorkArea");
        LRX_BIND(hf, fft_exec_z2z, "hipfftExecZ2Z");
        LRX_BIND(hf, fft_destroy, "hipfftDestroy");
#undef LRX_BIND
        if (!r.err.empty()) {
            r.err = "GATE fourier-plan-rocm-symbols: dlsym found no " + r.err;
            return r;
        }
        r.rocblas_path = loaded_path(reinterpret_cast<void*>(r.zgemm));
        r.hipfft_path = loaded_path(reinterpret_cast<void*>(r.fft_exec_z2z));
        return r;
    }();
    return a;
}

// ---- the remap kernel (fourier_plan.cu's `remap`, hipRTC-built) ---------------------------
// out[b, o_0, …] = s · in[b, map_0[o_0], …], and 0 where any map entry is < 0.
struct RemapShape {
    int d;
    int64_t batch;
    int64_t out_ext[3];
    int64_t in_ext[3];
    const int32_t* map[3];
};

static const char* kRemapSrc = R"__lrx__(
struct Z { double x, y; };
struct RemapShape { int d; long long batch; long long out_ext[3]; long long in_ext[3]; const int* map[3]; };
extern "C" __global__ void __launch_bounds__(256)
lrx_plan_remap(const Z* __restrict__ in, Z* __restrict__ out, RemapShape sh, long long total,
               double sr, double si) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        long long rem = i, src = 0, stride = 1;
        bool zero = false;
        for (int a = sh.d - 1; a >= 0; --a) {
            const long long o = rem % sh.out_ext[a];
            rem /= sh.out_ext[a];
            const long long j = sh.map[a] ? (long long)sh.map[a][o] : o;
            zero |= (j < 0);
            src += j * stride;
            stride *= sh.in_ext[a];
        }
        src += rem * stride;           // rem is the batch index
        Z v = zero ? Z{0.0, 0.0} : in[src];
        out[i] = Z{sr * v.x - si * v.y, sr * v.y + si * v.x};
    }
}
)__lrx__";

static ffi::Error remap_kernel(int dev, const std::string& cache_dir, rtc::Function* out) {
    static std::mutex mu;
    static std::map<int, rtc::Function> built;
    std::lock_guard<std::mutex> lk(mu);
    if (auto it = built.find(dev); it != built.end()) { *out = it->second; return ffi::Error::Success(); }
    const rtc::DriverApi& dapi = rtc::driver_api();
    if (!dapi.ok) return err("remap: module API: " + dapi.err);
    std::string arch;
    LORRAX_GPU_CHECK(lrx::gpu::device_arch(dev, &arch));
    rtc::Program prog;
    prog.src = kRemapSrc;
    prog.name = "lrx_plan_remap.hip";
    prog.kernel = "lrx_plan_remap";
    prog.defs = {rtc::kOptOption, "-std=c++17", rtc::arch_option(arch)};
    std::string missing;
    const std::string key_hex = rtc::hex16(rtc::key(prog, &missing));
    const std::string dir = missing.empty() ? cache_dir : std::string();
    std::string path;
    if (!dir.empty())
        path = dir + "/plan_remap_" + rtc::arch_tag(arch) + "_" + key_hex + rtc::kImageExt;
    rtc::Image img;
    std::string where, e;
    if (!rtc::build(prog, dir, path, key_hex, &img, &where, &e)) return err("remap: " + where + ": " + e);
    if (mklpin::announce_here())
        std::fprintf(stderr, "[fourier_plan] ROCM remap kernel %s: %s in %.1f ms (%s)\n", arch.c_str(),
                     img.from_disk ? "disk-cache hit" : "hipRTC built", img.ms,
                     path.empty() ? "not cached" : (img.from_disk || img.stored ? path.c_str() : "store FAILED"));
    *out = built[dev] = img.fn;
    return ffi::Error::Success();
}

static ffi::Error launch_remap(rtc::Function fn, const void* in, void* out, const RemapShape& sh,
                               double sr, double si, lrx::gpu::Stream stream) {
    long long total = sh.batch;
    for (int a = 0; a < sh.d; ++a) total *= sh.out_ext[a];
    if (total == 0) return ffi::Error::Success();
    const unsigned threads = 256;
    long long blocks = (total + threads - 1) / threads;
    if (blocks > 65535LL * 64) blocks = 65535LL * 64;
    RemapShape s = sh;
    const void* X = in;
    void* Y = out;
    void* args[] = {&X, &Y, &s, &total, &sr, &si};
    const rtc::Result r = rtc::driver_api().LaunchKernel(fn, unsigned(blocks), 1, 1, threads, 1, 1, 0,
                                                         reinterpret_cast<rtc::Stream>(stream), args, nullptr);
    if (r != rtc::kOk) return err(std::string(LRX_MODULE_API "ModuleLaunchKernel (remap): ") + rtc::err_string(r));
    return ffi::Error::Success();
}

// ---- plans --------------------------------------------------------------------------------
struct Step {
    enum Kind { kGemm, kRemap, kFft } kind;
    int axis = -1;                         // GEMM axis
    Z* mat = nullptr;                      // device (N' × K) row-major
    RemapShape remap{};                    // remap geometry (maps are device pointers)
    double sr = 1.0, si = 0.0;             // remap scale, or GEMM alpha
    FftHandle fft = nullptr;               // one contiguous FFT run, looped `loops` times
    int64_t loops = 1, loop_stride = 0;
    size_t work = 0;                       // its hipFFT work area, bytes (explicit arena per call)
    int64_t in_ext[3]{}, out_ext[3]{};     // current extents before / after the step
};

struct Plan {
    int d = 0;
    int64_t batch = 1;
    int sign = -1;
    std::vector<Step> steps;
    int64_t max_elems = 0;                 // largest extent along the chain (x and y included)
    int64_t buf_elems[2] = {0, 0};         // ping-pong sizes: the largest intermediate each holds
    size_t max_work = 0;                   // largest hipFFT work area over the steps
    bool rb_ws_known = false;              // rocBLAS workspace sized (first call, query mode)
    size_t rb_ws = 0;
    std::vector<void*> owned;              // device allocations
};

constexpr int64_t kIntMax = 2147483647LL;
static bool fits_int(std::initializer_list<int64_t> v) {
    for (int64_t x : v) if (x < 0 || x > kIntMax) return false;
    return true;
}
constexpr double kPi = 3.14159265358979323846;

static std::mutex g_mu;
static std::map<std::string, std::unique_ptr<Plan>> g_plans;

// A[j', j] = s·exp(sign·2πi·(o[j']·i[j] mod n)/n), row-major (n_out × n_in).
static std::vector<Z> fourier_matrix(int64_t n, const int64_t* o, int64_t n_out, const int64_t* in,
                                     int64_t n_in, int sign, double s) {
    std::vector<Z> a(size_t(n_out) * size_t(n_in));
    for (int64_t r = 0; r < n_out; ++r)
        for (int64_t c = 0; c < n_in; ++c) {
            int64_t m = ((o[r] % n) * (in[c] % n)) % n;
            if (2 * m > n) m -= n;
            const double ang = sign * 2.0 * kPi * double(m) / double(n);
            a[size_t(r) * n_in + c] = Z{s * std::cos(ang), s * std::sin(ang)};
        }
    return a;
}

template <class T>
static ffi::Error upload(Plan& p, const std::vector<T>& h, T** dptr) {
    void* ptr = nullptr;
    LORRAX_GPU_CHECK(lrx::gpu::device_malloc(&ptr, h.size() * sizeof(T)));
    p.owned.push_back(ptr);
    LORRAX_GPU_CHECK(lrx::gpu::memcpy_sync(ptr, h.data(), h.size() * sizeof(T), lrx::gpu::kHostToDevice));
    *dptr = static_cast<T*>(ptr);
    return ffi::Error::Success();
}

static int64_t prod(const int64_t* e, int d) {
    int64_t r = 1;
    for (int a = 0; a < d; ++a) r *= e[a];
    return r;
}

// build_plan: the CUDA leg's build_plan without the fused pair (attributes identical).
static ffi::Error build_plan(Plan& p, int64_t batch, ffi::Span<const int64_t> n,
                             ffi::Span<const int64_t> kin, ffi::Span<const int64_t> kout,
                             ffi::Span<const int64_t> in_idx, ffi::Span<const int64_t> out_idx,
                             ffi::Span<const int64_t> sup_in, ffi::Span<const int64_t> sup_out,
                             ffi::Span<const int64_t> gemm, ffi::Span<const double> scale,
                             ffi::Span<const int64_t> order, int sign) {
    const Api& A = api();
    const int d = int(n.size());
    p.d = d;
    p.batch = batch;
    p.sign = sign;
    std::vector<int64_t> ioff(d, 0), ooff(d, 0);
    for (int a = 1; a < d; ++a) {
        ioff[a] = ioff[a - 1] + kin[a - 1];
        ooff[a] = ooff[a - 1] + kout[a - 1];
    }
    int64_t cur[3] = {1, 1, 1};
    for (int a = 0; a < d; ++a) cur[a] = kin[a];
    p.max_elems = batch * prod(cur, d);

    double fft_scale = 1.0;
    bool any_fft = false, any_gemm = false;
    for (int a = 0; a < d; ++a) {
        if (gemm[a]) any_gemm = true;
        else { any_fft = true; fft_scale *= scale[a]; }
    }
    bool fft_scale_pending = any_fft && fft_scale != 1.0 && !any_gemm;
    bool alpha_pending = any_fft && fft_scale != 1.0 && any_gemm;

    auto push_remap = [&](const int64_t* out_ext, std::vector<std::vector<int32_t>>& maps,
                          double sr) -> ffi::Error {
        Step st;
        st.kind = Step::kRemap;
        st.remap.d = d;
        st.remap.batch = batch;
        for (int a = 0; a < d; ++a) {
            st.remap.in_ext[a] = cur[a];
            st.remap.out_ext[a] = out_ext[a];
            st.remap.map[a] = nullptr;
            if (!maps[a].empty()) {
                int32_t* dm = nullptr;
                if (auto e = upload(p, maps[a], &dm); e.failure()) return e;
                st.remap.map[a] = dm;
            }
            st.in_ext[a] = cur[a];
            st.out_ext[a] = out_ext[a];
            cur[a] = out_ext[a];
        }
        st.sr = sr;
        p.steps.push_back(st);
        p.max_elems = std::max(p.max_elems, batch * prod(cur, d));
        return ffi::Error::Success();
    };

    for (size_t oi = 0; oi < order.size(); ++oi) {
        const int64_t ax = order[oi];
        if (ax >= 0) {                                    // one GEMM axis
            const int a = int(ax);
            Step st;
            st.kind = Step::kGemm;
            st.axis = a;
            auto h = fourier_matrix(n[a], out_idx.begin() + ooff[a], kout[a], in_idx.begin() + ioff[a],
                                    kin[a], sign, scale[a]);
            if (auto e = upload(p, h, &st.mat); e.failure()) return e;
            if (alpha_pending) {
                st.sr = fft_scale;
                alpha_pending = false;
            }
            for (int b = 0; b < d; ++b) st.in_ext[b] = cur[b];
            cur[a] = kout[a];
            for (int b = 0; b < d; ++b) st.out_ext[b] = cur[b];
            p.steps.push_back(st);
            p.max_elems = std::max(p.max_elems, batch * prod(cur, d));
            continue;
        }
        // The FFT group: embed (zero-fill + gather), transform, restrict.
        std::vector<std::vector<int32_t>> maps(d);
        int64_t ext[3] = {1, 1, 1};
        bool embed = false;
        for (int a = 0; a < d; ++a) {
            ext[a] = cur[a];
            if (!gemm[a] && sup_in[a]) {
                embed = true;
                ext[a] = n[a];
                maps[a].assign(size_t(n[a]), -1);
                for (int64_t j = 0; j < kin[a]; ++j) maps[a][size_t(in_idx[ioff[a] + j] % n[a])] = int32_t(j);
            }
        }
        if (embed) {
            if (auto e = push_remap(ext, maps, fft_scale_pending ? fft_scale : 1.0); e.failure()) return e;
            fft_scale_pending = false;
        }
        for (int f0 = 0; f0 < d;) {                       // hipFFT over each contiguous FFT run
            if (gemm[f0]) { ++f0; continue; }
            int f1 = f0;
            while (f1 + 1 < d && !gemm[f1 + 1]) ++f1;
            Step st;
            st.kind = Step::kFft;
            int rank = f1 - f0 + 1;
            int nn[3];
            for (int a = f0; a <= f1; ++a) nn[a - f0] = int(n[a]);
            const int64_t left = batch * prod(cur, f0);
            int64_t right = 1;
            for (int a = f1 + 1; a < d; ++a) right *= cur[a];
            int64_t box = 1;
            for (int q = 0; q < rank; ++q) box *= nn[q];
            if (!fits_int({box, left, right, box * right})) {
                std::ostringstream os;
                os << "GATE fourier-plan-int32: the FFT run over axes [" << f0 << "," << f1 << "] has box "
                   << box << ", " << left << " lines before and " << right << " after (box*after "
                   << box * right << "); want each <= 2^31-1 (hipFFT's plan sizes are int); fix: split "
                      "the batch";
                return bad(os.str());
            }
            FftHandle h = nullptr;
            if (A.fft_create(&h) != 0) return err("hipfftCreate");
            if (A.fft_set_auto(h, 0) != 0) return err("hipfftSetAutoAllocation");
            FftResult r;
            size_t work = 0;
            if (right == 1) {
                r = A.fft_make_many(h, rank, nn, nn, 1, int(box), nn, 1, int(box), kFftZ2Z, int(left), &work);
                st.loops = 1;
            } else if (left <= right) {
                r = A.fft_make_many(h, rank, nn, nn, int(right), 1, nn, int(right), 1, kFftZ2Z, int(right),
                                    &work);
                st.loops = left;
                st.loop_stride = box * right;
            } else {
                r = A.fft_make_many(h, rank, nn, nn, int(right), int(box * right), nn, int(right),
                                    int(box * right), kFftZ2Z, int(left), &work);
                st.loops = right;
                st.loop_stride = 1;
            }
            if (r != 0) {
                A.fft_destroy(h);
                std::ostringstream os;
                os << "hipfftMakePlanMany failed (" << r << ")";
                return err(os.str());
            }
            st.fft = h;
            st.work = work;
            p.max_work = std::max(p.max_work, work);
            for (int b = 0; b < d; ++b) st.in_ext[b] = st.out_ext[b] = cur[b];
            p.steps.push_back(st);
            f0 = f1 + 1;
        }
        std::vector<std::vector<int32_t>> tmaps(d);       // restriction of supported FFT axes
        int64_t text[3] = {1, 1, 1};
        bool take = false;
        for (int a = 0; a < d; ++a) {
            text[a] = cur[a];
            if (!gemm[a] && sup_out[a]) {
                take = true;
                text[a] = kout[a];
                tmaps[a].resize(size_t(kout[a]));
                for (int64_t j = 0; j < kout[a]; ++j) tmaps[a][size_t(j)] = int32_t(out_idx[ooff[a] + j] % n[a]);
            }
        }
        if (take || fft_scale_pending) {
            if (auto e = push_remap(text, tmaps, fft_scale_pending ? fft_scale : 1.0); e.failure()) return e;
            fft_scale_pending = false;
        }
    }
    // Scratch: step i (i < last) writes buffer i % 2, the last step writes y; each buffer is sized
    // by the intermediates it holds, not by the chain's largest extent (the CUDA leg allocates
    // two of max_elems).  Fe 25^3 sphere->box (13^3 -> 25.13^2 -> 25^2.13 -> y): 0.79 box, not 2.
    for (size_t i = 0; i + 1 < p.steps.size(); ++i) {
        int64_t e = batch;
        for (int a = 0; a < d; ++a) e *= p.steps[i].out_ext[a];
        p.buf_elems[i % 2] = std::max(p.buf_elems[i % 2], e);
    }
    return ffi::Error::Success();
}

static std::string plan_key(int dev, int64_t batch, ffi::Span<const int64_t> n, ffi::Span<const int64_t> kin,
                            ffi::Span<const int64_t> kout, ffi::Span<const int64_t> in_idx,
                            ffi::Span<const int64_t> out_idx, ffi::Span<const int64_t> sup_in,
                            ffi::Span<const int64_t> sup_out, ffi::Span<const int64_t> gemm,
                            ffi::Span<const double> scale, ffi::Span<const int64_t> order, int64_t sign) {
    std::string k;
    auto add = [&](const void* p, size_t bytes) { k.append(static_cast<const char*>(p), bytes); };
    add(&dev, sizeof dev);
    add(&batch, sizeof batch);
    add(&sign, sizeof sign);
    for (auto s : {n, kin, kout, in_idx, out_idx, sup_in, sup_out, gemm, order}) {
        const int64_t len = int64_t(s.size());
        add(&len, sizeof len);
        add(s.begin(), s.size() * sizeof(int64_t));
    }
    add(scale.begin(), scale.size() * sizeof(double));
    return k;
}

// One rocBLAS handle per device for the process; its mutex covers set_stream + workspace + calls.
struct DeviceHandle {
    RbHandle h = nullptr;
    std::mutex m;
};
static ffi::Error device_handle(int dev, DeviceHandle** out) {
    static std::mutex mu;
    static std::map<int, std::unique_ptr<DeviceHandle>> handles;
    std::lock_guard<std::mutex> lk(mu);
    auto& slot = handles[dev];
    if (!slot) {
        auto dh = std::make_unique<DeviceHandle>();
        LORRAX_LIB_CHECK(api().rb_create(&dh->h), 0, "rocblas_create_handle");
        // No default managed reserve beside the explicitly sized call arena.
        LORRAX_LIB_CHECK(api().rb_set_workspace(dh->h, nullptr, 0), 0, "rocblas_set_workspace (initialize)");
        slot = std::move(dh);
    }
    *out = slot.get();
    return ffi::Error::Success();
}

// One GEMM step (the CUDA leg's two cuBLAS calls, argument for argument).
static RbStatus gemm_step(const Api& A, RbHandle h, const Step& st, int d, int64_t batch, const void* src,
                          void* dst, std::string* refusal) {
    const int a = st.axis;
    int64_t L = batch, R = 1;
    for (int b = 0; b < a; ++b) L *= st.in_ext[b];
    for (int b = a + 1; b < d; ++b) R *= st.in_ext[b];
    const int64_t K = st.in_ext[a], Np = st.out_ext[a];
    if (!fits_int({L, R, K, Np, K * R, Np * R})) {
        std::ostringstream os;
        os << "GATE fourier-plan-int32: the GEMM on axis " << a << " has (L, K, N', R) = (" << L << ", " << K
           << ", " << Np << ", " << R << "); want each <= 2^31-1 (rocBLAS sizes are int); fix: split the batch";
        *refusal = os.str();
        return -1;
    }
    const Z alpha{st.sr, st.si}, beta{0.0, 0.0};
    const Z* X = static_cast<const Z*>(src);
    Z* Y = static_cast<Z*>(dst);
    if (R == 1)   // Y (L × N') = X (L × K) · Aᵀ, row-major
        return A.zgemm(h, kRbOpTrans, kRbOpNone, int32_t(Np), int32_t(L), int32_t(K), &alpha, st.mat,
                       int32_t(K), X, int32_t(K), &beta, Y, int32_t(Np));
    // per l: Y_l (N' × R) = A (N' × K) · X_l (K × R), A shared (stride 0)
    return A.zgemm_sb(h, kRbOpNone, kRbOpNone, int32_t(R), int32_t(Np), int32_t(K), &alpha, X, int32_t(R),
                      K * R, st.mat, int32_t(K), 0, &beta, Y, int32_t(R), Np * R, int32_t(L));
}

// Query and execution share the same cached plan, including vendor plan sizes.
static ffi::Error get_plan(int dev, int64_t batch, ffi::Span<const int64_t> n,
                           ffi::Span<const int64_t> kin, ffi::Span<const int64_t> kout,
                           ffi::Span<const int64_t> in_idx, ffi::Span<const int64_t> out_idx,
                           ffi::Span<const int64_t> sup_in, ffi::Span<const int64_t> sup_out,
                           ffi::Span<const int64_t> gemm, ffi::Span<const double> scale,
                           ffi::Span<const int64_t> order, int64_t sign, Plan** p) {
    const std::string key = plan_key(dev, batch, n, kin, kout, in_idx, out_idx, sup_in, sup_out, gemm, scale,
                                     order, sign);
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_plans.find(key);
        if (it == g_plans.end()) {
            auto np = std::make_unique<Plan>();
            if (auto e = build_plan(*np, batch, n, kin, kout, in_idx, out_idx, sup_in, sup_out, gemm, scale,
                                    order, int(sign));
                e.failure())
                return e;
            it = g_plans.emplace(key, std::move(np)).first;
        }
        *p = it->second.get();
    }
    return ffi::Error::Success();
}

// Size-query mode executes no GEMM and never dereferences these null data
// pointers. Fourier matrices/remap tables remain the existing plan's metadata.
static ffi::Error size_workspace(int dev, Plan* p) {
    bool any_gemm = false;
    for (const Step& st : p->steps) any_gemm |= st.kind == Step::kGemm;
    if (!any_gemm) return ffi::Error::Success();
    DeviceHandle* dh = nullptr;
    FFI_RETURN_IF_ERROR(device_handle(dev, &dh));
    std::lock_guard<std::mutex> lock(dh->m);
    if (p->rb_ws_known) return ffi::Error::Success();
    const Api& A = api();
    size_t ws = 0;
    LORRAX_LIB_CHECK(A.rb_start_query(dh->h), 0, "rocblas_start_device_memory_size_query");
    for (const Step& st : p->steps) {
        if (st.kind != Step::kGemm) continue;
        std::string refusal;
        const RbStatus status = gemm_step(A, dh->h, st, p->d, p->batch, nullptr, nullptr, &refusal);
        if (!refusal.empty()) { A.rb_stop_query(dh->h, &ws); return bad(refusal); }
        // rocblas_status_{success,size_increased,size_unchanged}.
        if (status != 0 && status != 9 && status != 10) {
            A.rb_stop_query(dh->h, &ws);
            return err("rocBLAS workspace query failed (" + std::to_string(status) + ")");
        }
    }
    LORRAX_LIB_CHECK(A.rb_stop_query(dh->h, &ws), 0, "rocblas_stop_device_memory_size_query");
    p->rb_ws = ws;
    p->rb_ws_known = true;
    return ffi::Error::Success();
}

// Serial hipFFT and rocBLAS steps reuse one vendor work region. Both
// ping-pong banks remain live beside it; every region starts at 256 B.
static ffi::Error workspace_layout(const Plan& p, size_t* total, size_t offsets[3]) {
    size_t sizes[3] = {std::max(p.max_work, p.rb_ws), 0, 0};
    const size_t limit = std::numeric_limits<size_t>::max();
    for (int i = 0; i < 2; ++i) {
        if (p.buf_elems[i] < 0 || static_cast<uint64_t>(p.buf_elems[i]) > limit / sizeof(Z))
            return bad("workspace intermediate byte count overflows size_t");
        sizes[i + 1] = size_t(p.buf_elems[i]) * sizeof(Z);
    }
    *total = 0;
    for (int i = 0; i < 3; ++i) {
        if (*total > limit - 255) return bad("workspace alignment overflows size_t");
        offsets[i] = (*total + 255) & ~size_t(255);
        if (sizes[i] > limit - offsets[i]) return bad("workspace total overflows size_t");
        *total = offsets[i] + sizes[i];
    }
    return ffi::Error::Success();
}

static ffi::Error run(lrx::gpu::Stream stream, ffi::AnyBuffer x,
                      ffi::Result<ffi::AnyBuffer> y, ffi::Result<ffi::Buffer<ffi::U8>> workspace,
                      ffi::Span<const int64_t> n, ffi::Span<const int64_t> kin,
                      ffi::Span<const int64_t> kout, ffi::Span<const int64_t> in_idx,
                      ffi::Span<const int64_t> out_idx, ffi::Span<const int64_t> sup_in,
                      ffi::Span<const int64_t> sup_out, ffi::Span<const int64_t> gemm,
                      ffi::Span<const double> scale, ffi::Span<const int64_t> order, int64_t sign,
                      std::string_view cubin_dir) {
    const Api& A = api();
    if (!A.err.empty()) return ffi::Error(ffi::ErrorCode::kFailedPrecondition, A.err);
    if (x.element_type() != ffi::DataType::C128 || y->element_type() != ffi::DataType::C128)
        return bad("complex128 only");
    const int d = int(n.size());
    auto xd = x.dimensions();
    if (d < 1 || d > 3 || int(xd.size()) < d) return bad("1 to 3 trailing transform axes");
    int64_t batch = 1;
    for (size_t i = 0; i + d < xd.size(); ++i) batch *= xd[i];
    for (int a = 0; a < d; ++a)
        if (xd[xd.size() - d + a] != kin[a]) return bad("x extent does not match kin");
    if (batch == 0) return ffi::Error::Success();

    int dev = 0;
    LORRAX_GPU_CHECK(lrx::gpu::get_device(&dev));
    Plan* p = nullptr;
    FFI_RETURN_IF_ERROR(get_plan(dev, batch, n, kin, kout, in_idx, out_idx,
                                sup_in, sup_out, gemm, scale, order, sign, &p));
    FFI_RETURN_IF_ERROR(size_workspace(dev, p));
    size_t total = 0, offsets[3];
    FFI_RETURN_IF_ERROR(workspace_layout(*p, &total, offsets));
    if (workspace->element_count() != total)
        return bad("explicit workspace result has " + std::to_string(workspace->element_count()) +
                   " bytes; exact plan requires " + std::to_string(total));
    auto* arena = workspace->typed_data();
    rtc::Function remap_fn = nullptr;
    bool any_gemm = false;
    for (const Step& st : p->steps) {
        if (st.kind == Step::kRemap && remap_fn == nullptr)
            FFI_RETURN_IF_ERROR(remap_kernel(dev, std::string(cubin_dir), &remap_fn));
        any_gemm |= (st.kind == Step::kGemm);
    }

    void* work = std::max(p->max_work, p->rb_ws) ? arena + offsets[0] : nullptr;
    void* buf[2] = {p->buf_elems[0] ? arena + offsets[1] : nullptr,
                    p->buf_elems[1] ? arena + offsets[2] : nullptr};
    DeviceHandle* dh = nullptr;
    std::unique_lock<std::mutex> lk;
    if (any_gemm) {
        FFI_RETURN_IF_ERROR(device_handle(dev, &dh));
        lk = std::unique_lock<std::mutex>(dh->m);
        LORRAX_LIB_CHECK(A.rb_set_stream(dh->h, stream), 0, "rocblas_set_stream");
        LORRAX_LIB_CHECK(A.rb_set_workspace(dh->h, p->rb_ws ? work : nullptr, p->rb_ws), 0, "rocblas_set_workspace");
    }
    auto execute = [&]() -> ffi::Error {
        const void* src = x.untyped_data();
        int which = 0;
        for (size_t i = 0; i < p->steps.size(); ++i) {
            Step& st = p->steps[i];
            const bool last = (i + 1 == p->steps.size());
            void* dst = last ? y->untyped_data() : buf[which];
            if (st.kind == Step::kGemm) {
                std::string refusal;
                const RbStatus s = gemm_step(A, dh->h, st, d, batch, src, dst, &refusal);
                if (!refusal.empty()) return bad(refusal);
                if (s != 0) {
                    std::ostringstream os;
                    os << "rocBLAS failed (" << s << ") on axis " << st.axis << " (" << A.rocblas_path << ")";
                    return err(os.str());
                }
            } else if (st.kind == Step::kRemap) {
                FFI_RETURN_IF_ERROR(launch_remap(remap_fn, src, dst, st.remap, st.sr, st.si, stream));
            } else {
                const int dir = p->sign < 0 ? kFftForward : kFftBackward;
                if (A.fft_set_stream(st.fft, stream) != 0) return err("hipfftSetStream");
                if (st.work > 0 && A.fft_set_work(st.fft, work) != 0) return err("hipfftSetWorkArea");
                auto* in = const_cast<Z*>(static_cast<const Z*>(src));
                auto* out = static_cast<Z*>(dst);
                for (int64_t l = 0; l < st.loops; ++l)
                    if (A.fft_exec_z2z(st.fft, in + l * st.loop_stride, out + l * st.loop_stride, dir) != 0)
                        return err("hipfftExecZ2Z (" + A.hipfft_path + ")");
            }
            LORRAX_GPU_CHECK(lrx::gpu::peek_at_last_error());
            src = dst;
            which ^= 1;
        }
        if (p->steps.empty())
            LORRAX_GPU_CHECK(lrx::gpu::memcpy_async(y->untyped_data(), x.untyped_data(),
                                                    size_t(batch) * prod(kin.begin(), d) * sizeof(Z),
                                                    lrx::gpu::kDeviceToDevice, stream));
        return ffi::Error::Success();
    };
    ffi::Error result = execute();
    // The handle must never retain an arena borrowed from an FFI result,
    // including on a numerical/remap/launch failure. Preserve that error.
    if (any_gemm) {
        const RbStatus status = A.rb_set_workspace(dh->h, nullptr, 0);
        if (result.failure()) return result;
        LORRAX_LIB_CHECK(status, 0, "rocblas_set_workspace (detach)");
    }
    return result;
}

}  // namespace lorrax_ffi::fourier_plan_rocm

// The descriptor is exactly the execution attributes, plus the local batch.
// First use builds the existing cached metadata/vendor plans, never field
// buffers or queried workspace. All subsequent queries reuse that owner.
extern "C" int LRX_C_ENTRY(lrx_fourier_plan_workspace_bytes)(
    int d, int64_t batch, const int64_t* n, const int64_t* kin, const int64_t* kout,
    const int64_t* in_idx, const int64_t* out_idx, const int64_t* sup_in,
    const int64_t* sup_out, const int64_t* gemm, const double* scale,
    const int64_t* order, int64_t order_size, int64_t sign,
    uint64_t* out, char* error, size_t capacity) {
    using namespace lorrax_ffi::fourier_plan_rocm;
    auto report = [&](const std::string& message) {
        if (error != nullptr && capacity > 0) {
            const size_t length = std::min(message.size(), capacity - 1);
            std::memcpy(error, message.data(), length);
            error[length] = '\0';
        }
        return 1;
    };
    if (out == nullptr) return report("workspace query needs an output pointer");
    *out = 0;
    if (d < 1 || d > 3 || batch < 0 || order_size < 1 || order_size > d + 1 ||
        (sign != -1 && sign != 1) || !n || !kin || !kout || !in_idx || !out_idx ||
        !sup_in || !sup_out || !gemm || !scale || !order)
        return report("workspace query needs a complete 1..3-axis descriptor");
    int64_t ni = 0, no = 0;
    uint64_t full = 1, input = 1, output = 1;
    const uint64_t limit = std::numeric_limits<int64_t>::max() / sizeof(Z);
    for (int a = 0; a < d; ++a) {
        if (n[a] < 1 || kin[a] < 1 || kout[a] < 1 ||
            n[a] > kIntMax || kin[a] > kIntMax || kout[a] > kIntMax ||
            full > limit / uint64_t(n[a]) || input > limit / uint64_t(kin[a]) ||
            output > limit / uint64_t(kout[a]))
            return report("workspace query axis/product exceeds the native range");
        full *= n[a]; input *= kin[a]; output *= kout[a];
        ni += kin[a]; no += kout[a];
    }
    if (uint64_t(batch) > limit / std::max({full, input, output}))
        return report("workspace query batch byte span overflows int64");
    if (batch == 0) return 0;
    try {
        if (!api().err.empty()) return report(api().err);
        int dev = 0;
        if (auto status = lrx::gpu::get_device(&dev); status != lrx::gpu::kSuccess)
            return report(lrx::gpu::error_string(status));
        Plan* p = nullptr;
        if (auto e = get_plan(dev, batch, {n, size_t(d)}, {kin, size_t(d)}, {kout, size_t(d)},
                              {in_idx, size_t(ni)}, {out_idx, size_t(no)},
                              {sup_in, size_t(d)}, {sup_out, size_t(d)}, {gemm, size_t(d)},
                              {scale, size_t(d)}, {order, size_t(order_size)}, sign, &p); e.failure())
            return report(e.message());
        if (auto e = size_workspace(dev, p); e.failure()) return report(e.message());
        size_t total = 0, offsets[3];
        if (auto e = workspace_layout(*p, &total, offsets); e.failure()) return report(e.message());
        *out = total;
        return 0;
    } catch (const std::exception& e) {
        return report(e.what());
    }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LorraxFourierPlanRocmFfi, lorrax_ffi::fourier_plan_rocm::run,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::Buffer<xla::ffi::U8>>()  // exact native arena, visible to AOT
        .Attr<xla::ffi::Span<const int64_t>>("n")
        .Attr<xla::ffi::Span<const int64_t>>("kin")
        .Attr<xla::ffi::Span<const int64_t>>("kout")
        .Attr<xla::ffi::Span<const int64_t>>("in_idx")
        .Attr<xla::ffi::Span<const int64_t>>("out_idx")
        .Attr<xla::ffi::Span<const int64_t>>("sup_in")
        .Attr<xla::ffi::Span<const int64_t>>("sup_out")
        .Attr<xla::ffi::Span<const int64_t>>("gemm")
        .Attr<xla::ffi::Span<const double>>("scale")
        .Attr<xla::ffi::Span<const int64_t>>("order")
        .Attr<int64_t>("sign")
        .Attr<std::string_view>("cubin_dir"));
