// conv_klead_cuda_ffi.cc — half-absorbed Sigma fused convolution, CUDA leg.
//
//   U[k,a,mx,b,my] = scale * FFT_k(IFFT_k(T) * IFFT_k(W)[k,mx,my])
//
// The public T/U seam stays in Sigma's native k-leading layout.  The Python
// wrapper pays the conditionally-authorized ONE transpose on entry and hands
// this handler Tm=(a,mx,b,my,nk), so consecutive x threads load a contiguous
// k-row exactly like the certified k-minor sibling.  W remains k-leading and
// is transformed inside the call; U is emitted k-leading directly from the
// store, absorbing the would-be output transpose.  Once resident, the direct
// twiddle-ring passes are unchanged: runtime axis extents, no per-size
// compilation, odd shared row stride, device-derived residency, and a named
// refusal.
//
// This remains one kernel implementation, selected only after the original
// native-copy prototype measured >20% below the k-minor per-byte rate.  The
// plan-based k-leading member (lorrax_mklfft_gw_conv) remains the caller's
// unsupported-shape fallback.

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <dlfcn.h>

#include "../common/mkl_thread_pin.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "xla/ffi/api/ffi.h"

namespace lorrax_ffi::conv_klead {

namespace ffi = ::xla::ffi;

static constexpr int kAxisMax = 24;
static constexpr int kSmemPreferred = 32768;
static constexpr int kEptMax = 8;
static constexpr int kEptPref = 4;
static constexpr int kMaxBlock = 512;
static constexpr int kBlockTarget = 256;
static constexpr int kRowsMax = 64;

static bool log_enabled() {
    static const bool on = [] {
        return mklpin::log_here("LORRAX_CONV_KLEAD_LOG") ||
               mklpin::log_here("LORRAX_FFT_FFI_LOG");
    }();
    return on;
}

static ffi::Error fail(const char* where, const std::string& detail) {
    std::ostringstream os;
    os << "conv_klead (k-leading direct fused conv CUDA FFI): " << where
       << " failed — " << detail;
    return ffi::Error(ffi::ErrorCode::kInternal, os.str());
}

#define LRX_CUDA_CHECK(expr, where)                                      \
    do {                                                                 \
        cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) {                                         \
            return fail((where), cudaGetErrorString(_e));                \
        }                                                                \
    } while (0)

struct DriverApi {
    CUresult (*ModuleLoadData)(CUmodule*, const void*) = nullptr;
    CUresult (*ModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*ModuleUnload)(CUmodule) = nullptr;
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                             unsigned, unsigned, unsigned, unsigned,
                             CUstream, void**, void**) = nullptr;
    CUresult (*CtxGetCurrent)(CUcontext*) = nullptr;
    CUresult (*GetErrorString)(CUresult, const char**) = nullptr;
    CUresult (*FuncSetAttribute)(CUfunction, int, int) = nullptr;
    bool ok = false;
    std::string err;
};

static const DriverApi& driver_api() {
    static DriverApi api = [] {
        DriverApi a;
        void* h = RTLD_DEFAULT;
        dlerror();
        if (dlsym(h, "cuLaunchKernel") == nullptr) {
            h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
            if (h == nullptr) {
                const char* e = dlerror();
                a.err = std::string("dlopen(libcuda.so.1): ") +
                        (e ? e : "(dlerror returned no detail)");
                return a;
            }
        }
        auto need = [&](const char* name) -> void* {
            dlerror();
            void* p = dlsym(h, name);
            if (p == nullptr) {
                const char* e = dlerror();
                if (!a.err.empty()) a.err += "; ";
                a.err += std::string("dlsym(") + name + "): " +
                         (e ? e : "symbol not found (no dlerror detail)");
            }
            return p;
        };
        a.ModuleLoadData = reinterpret_cast<decltype(a.ModuleLoadData)>(
            need("cuModuleLoadData"));
        a.ModuleGetFunction = reinterpret_cast<decltype(a.ModuleGetFunction)>(
            need("cuModuleGetFunction"));
        a.ModuleUnload = reinterpret_cast<decltype(a.ModuleUnload)>(
            need("cuModuleUnload"));
        a.LaunchKernel = reinterpret_cast<decltype(a.LaunchKernel)>(
            need("cuLaunchKernel"));
        a.CtxGetCurrent = reinterpret_cast<decltype(a.CtxGetCurrent)>(
            need("cuCtxGetCurrent"));
        a.GetErrorString = reinterpret_cast<decltype(a.GetErrorString)>(
            need("cuGetErrorString"));
        a.FuncSetAttribute = reinterpret_cast<decltype(a.FuncSetAttribute)>(
            need("cuFuncSetAttribute"));
        a.ok = a.ModuleLoadData && a.ModuleGetFunction && a.ModuleUnload &&
               a.LaunchKernel && a.CtxGetCurrent && a.GetErrorString &&
               a.FuncSetAttribute;
        return a;
    }();
    return api;
}

static std::string cu_err(CUresult r) {
    const DriverApi& api = driver_api();
    const char* s = nullptr;
    if (api.GetErrorString && api.GetErrorString(r, &s) == CUDA_SUCCESS && s) {
        return s;
    }
    std::ostringstream os;
    os << "CUresult=" << static_cast<int>(r);
    return os.str();
}

static const char* kKernelSrc = R"__lrx__(
struct __align__(16) lrx_c2 { double x, y; };

template <int EPT, int AXIS>
__device__ __forceinline__ void lrx_pass(
    const lrx_c2* rowp, const lrx_c2* twv,
    int nk, int lane, int tpr, int n0, int n1, int n2,
    const int (&kx)[EPT], const int (&ky)[EPT], const int (&kz)[EPT],
    double sgn, lrx_c2 (&acc)[EPT])
{
    const int len = (AXIS == 0) ? n0 : ((AXIS == 1) ? n1 : n2);
    const int stride = (AXIS == 0) ? (n1 * n2) : ((AXIS == 1) ? n2 : 1);
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int k = lane + e * tpr;
        if (k < nk) {
            int base, mid;
            if (AXIS == 0) {
                base = ky[e] * n2 + kz[e]; mid = kx[e];
            } else if (AXIS == 1) {
                base = kx[e] * n1 * n2 + kz[e]; mid = ky[e];
            } else {
                base = (kx[e] * n1 + ky[e]) * n2; mid = kz[e];
            }
            double ar = 0.0, ai = 0.0;
            int m = 0;
            for (int j = 0; j < len; ++j) {
                const lrx_c2 v = rowp[base + j * stride];
                const lrx_c2 w = twv[m];
                const double wi = sgn * w.y;
                ar += v.x * w.x - v.y * wi;
                ai += v.x * wi + v.y * w.x;
                m += mid;
                if (m >= len) m -= len;
            }
            acc[e].x = ar;
            acc[e].y = ai;
        }
    }
}

template <int EPT>
__device__ __forceinline__ void lrx_writeback(
    lrx_c2* rowp, int nk, int lane, int tpr, const lrx_c2 (&acc)[EPT])
{
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int k = lane + e * tpr;
        if (k < nk) rowp[k] = acc[e];
    }
}

template <int EPT>
__device__ __forceinline__ void lrx_conv_body(
    const lrx_c2* tin, const lrx_c2* __restrict__ wgt, lrx_c2* uout,
    long long Tg, long long Tv, long long mx, long long b, long long my,
    int nk, int n0, int n1, int n2, int SP, double scale, lrx_c2* sm)
{
    const int tpr = blockDim.x;
    const int RB = blockDim.y;
    const int tid = threadIdx.y * tpr + threadIdx.x;
    const int nthr = tpr * RB;
    const int lane = threadIdx.x;
    const int ntw = n0 + n1 + n2;
    const int tile = RB * nk;
    const long long r0 = (long long)blockIdx.x * RB;

    lrx_c2* ts = sm;
    lrx_c2* ws = ts + (long long)RB * SP;
    lrx_c2* tw = ws + (long long)RB * SP;
    long long* wrow = (long long*)(tw + ntw);

    for (int i = tid; i < ntw; i += nthr) {
        int m, len;
        if (i < n0) { m = i; len = n0; }
        else if (i < n0 + n1) { m = i - n0; len = n1; }
        else { m = i - n0 - n1; len = n2; }
        double s, c;
        sincospi(-2.0 * (double)m / (double)len, &s, &c);
        tw[i].x = c;
        tw[i].y = s;
    }
    const lrx_c2* twx = tw;
    const lrx_c2* twy = tw + n0;
    const lrx_c2* twz = tw + n0 + n1;

    // One 64-bit decomposition per ROW, not per k element.  T's trailing
    // order is (a,mx,b,my); W's is (mx,my).
    for (int j = tid; j < RB; j += nthr) {
        long long q = r0 + j;
        const long long iy = q % my; q /= my;
        q /= b;
        const long long ix = q % mx;
        wrow[j] = ix * my + iy;
    }
    __syncthreads();

    // Half-absorbed input: Tm is k-minor, so x threads read its contiguous
    // k-row.  W stays k-leading; its separate load keeps j fast so adjacent
    // flattened threads read adjacent mu-nu elements at fixed k.
    {
        const int j = threadIdx.y;
        const long long r = r0 + j;
        for (int k = lane; k < nk; k += tpr) {
            lrx_c2 tv = {0.0, 0.0};
            if (r < Tg) tv = tin[r * nk + k];
            ts[(long long)j * SP + k] = tv;
        }
    }
    {
        int i = tid;
        int k = i / RB, j = i - k * RB;
        const int dk = nthr / RB, dj = nthr - dk * RB;
        for (; i < tile; i += nthr) {
            const long long r = r0 + j;
            lrx_c2 wv = {0.0, 0.0};
            if (r < Tg) {
                wv = wgt[(long long)k * Tv + wrow[j]];
            }
            ws[(long long)j * SP + k] = wv;
            j += dj;
            if (j >= RB) { j -= RB; k += 1; }
            k += dk;
        }
    }
    __syncthreads();

    lrx_c2* trow = ts + (long long)threadIdx.y * SP;
    lrx_c2* wline = ws + (long long)threadIdx.y * SP;
    lrx_c2 ta[EPT], wa[EPT];
    int kx[EPT], ky[EPT], kz[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        int k = lane + e * tpr;
        if (k >= nk) k = 0;
        kz[e] = k % n2; k /= n2;
        ky[e] = k % n1;
        kx[e] = k / n1;
    }

#define LRX_MULT                                                    \
    {                                                               \
        _Pragma("unroll")                                          \
        for (int e = 0; e < EPT; ++e) {                             \
            const int k = lane + e * tpr;                           \
            if (k < nk) {                                           \
                const double ar = ta[e].x, ai = ta[e].y;            \
                ta[e].x = ar * wa[e].x - ai * wa[e].y;              \
                ta[e].y = ar * wa[e].y + ai * wa[e].x;              \
            }                                                       \
        }                                                           \
    }
#define LRX_SCALE                                                   \
    {                                                               \
        _Pragma("unroll")                                          \
        for (int e = 0; e < EPT; ++e) {                             \
            ta[e].x *= scale;                                       \
            ta[e].y *= scale;                                       \
        }                                                           \
    }
#define LRX_INVERSE_PASS(AX, TWV, LEN)                              \
    if ((LEN) > 1) {                                                \
        lrx_pass<EPT, AX>(trow, TWV, nk, lane, tpr, n0, n1, n2,    \
                          kx, ky, kz, -1.0, ta);                    \
        lrx_pass<EPT, AX>(wline, TWV, nk, lane, tpr, n0, n1, n2,  \
                          kx, ky, kz, -1.0, wa);                    \
        __syncthreads();                                            \
        if (--rem == 0) { LRX_MULT }                               \
        lrx_writeback<EPT>(trow, nk, lane, tpr, ta);               \
        if (rem != 0) lrx_writeback<EPT>(wline, nk, lane, tpr, wa);\
        __syncthreads();                                            \
    }
#define LRX_FORWARD_PASS(AX, TWV, LEN)                              \
    if ((LEN) > 1) {                                                \
        lrx_pass<EPT, AX>(trow, TWV, nk, lane, tpr, n0, n1, n2,    \
                          kx, ky, kz, 1.0, ta);                     \
        __syncthreads();                                            \
        if (--rem == 0) { LRX_SCALE }                              \
        lrx_writeback<EPT>(trow, nk, lane, tpr, ta);               \
        __syncthreads();                                            \
    }

    const int nact = (n0 > 1) + (n1 > 1) + (n2 > 1);
    if (nact == 0) {
#pragma unroll
        for (int e = 0; e < EPT; ++e) {
            const int k = lane + e * tpr;
            if (k < nk) {
                const lrx_c2 t = trow[k], w = wline[k];
                trow[k].x = (t.x * w.x - t.y * w.y) * scale;
                trow[k].y = (t.x * w.y + t.y * w.x) * scale;
            }
        }
        __syncthreads();
    } else {
        int rem = nact;
        LRX_INVERSE_PASS(2, twz, n2)
        LRX_INVERSE_PASS(1, twy, n1)
        LRX_INVERSE_PASS(0, twx, n0)
        rem = nact;
        LRX_FORWARD_PASS(2, twz, n2)
        LRX_FORWARD_PASS(1, twy, n1)
        LRX_FORWARD_PASS(0, twx, n0)
    }
#undef LRX_FORWARD_PASS
#undef LRX_INVERSE_PASS
#undef LRX_SCALE
#undef LRX_MULT

    // row-major shared -> k-leading global, the mirror of the load.  The
    // output stays in the exact layout Sigma's projection consumes.
    {
        int i = tid;
        int k = i / RB, j = i - k * RB;
        const int dk = nthr / RB, dj = nthr - dk * RB;
        for (; i < tile; i += nthr) {
            const long long r = r0 + j;
            if (r < Tg) {
                uout[(long long)k * Tg + r] =
                    ts[(long long)j * SP + k];
            }
            j += dj;
            if (j >= RB) { j -= RB; k += 1; }
            k += dk;
        }
    }
}

#define LRX_MAX_BLOCK 512
#define LRX_ENTRY(N)                                                     \
extern "C" __global__ __launch_bounds__(LRX_MAX_BLOCK)                  \
void lrx_conv_klead_c128_e##N(                                          \
    const lrx_c2* tin, const lrx_c2* __restrict__ wgt, lrx_c2* uout,    \
    long long Tg, long long Tv, long long mx, long long b, long long my,\
    int nk, int n0, int n1, int n2, int SP, double scale)               \
{                                                                        \
    extern __shared__ lrx_c2 sm[];                                      \
    lrx_conv_body<N>(tin, wgt, uout, Tg, Tv, mx, b, my, nk,             \
                     n0, n1, n2, SP, scale, sm);                         \
}

LRX_ENTRY(1) LRX_ENTRY(2) LRX_ENTRY(3) LRX_ENTRY(4)
LRX_ENTRY(5) LRX_ENTRY(6) LRX_ENTRY(7) LRX_ENTRY(8)
#undef LRX_ENTRY
)__lrx__";

static std::mutex g_mu;

struct KernelArms {
    CUfunction fn[kEptMax] = {nullptr};
    int smem_max = 0;
};

static ffi::Error ensure_kernels(const KernelArms** out) {
    static std::map<CUcontext, KernelArms> cache;
    static std::map<CUcontext, std::string> fail_cache;
    const DriverApi& api = driver_api();
    if (!api.ok) {
        return fail("driver-api resolve", api.err.empty()
            ? "required CUDA driver entry points are unavailable"
            : api.err);
    }
    CUcontext ctx = nullptr;
    api.CtxGetCurrent(&ctx);
    if (ctx == nullptr) {
        LRX_CUDA_CHECK(cudaFree(nullptr), "context bind (cudaFree(0))");
        api.CtxGetCurrent(&ctx);
        if (ctx == nullptr) return fail("context bind", "no current CUDA context");
    }
    auto it = cache.find(ctx);
    if (it != cache.end()) { *out = &it->second; return ffi::Error::Success(); }
    auto fit = fail_cache.find(ctx);
    if (fit != fail_cache.end()) {
        return fail("kernel build (cached failure, NVRTC not re-run)", fit->second);
    }
    auto fail_sticky = [&](const char* where,
                           const std::string& detail) -> ffi::Error {
        fail_cache.emplace(ctx, std::string(where) + " — " + detail);
        return fail(where, detail);
    };

    int dev = 0, cc_major = 0, cc_minor = 0, smem_optin = 0;
    LRX_CUDA_CHECK(cudaGetDevice(&dev), "cudaGetDevice");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(
        &cc_major, cudaDevAttrComputeCapabilityMajor, dev),
        "query compute capability (major)");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(
        &cc_minor, cudaDevAttrComputeCapabilityMinor, dev),
        "query compute capability (minor)");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(
        &smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev),
        "query max dynamic shared memory per block (opt-in)");

    nvrtcProgram prog = nullptr;
    nvrtcResult nr = nvrtcCreateProgram(
        &prog, kKernelSrc, "lrx_conv_klead.cu", 0, nullptr, nullptr);
    if (nr != NVRTC_SUCCESS) {
        return fail_sticky("nvrtcCreateProgram", nvrtcGetErrorString(nr));
    }
    char arch[64];
    std::snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d",
                  cc_major, cc_minor);
    const char* opts[] = {arch};
    nr = nvrtcCompileProgram(prog, 1, opts);
    if (nr != NVRTC_SUCCESS) {
        size_t log_sz = 0;
        std::string log;
        if (nvrtcGetProgramLogSize(prog, &log_sz) == NVRTC_SUCCESS &&
            log_sz > 1) {
            log.resize(log_sz);
            nvrtcGetProgramLog(prog, &log[0]);
        }
        nvrtcDestroyProgram(&prog);
        return fail_sticky("nvrtcCompileProgram",
                           std::string(nvrtcGetErrorString(nr)) + " — " + log);
    }
    std::vector<char> image;
    size_t sz = 0;
    bool used_cubin = false;
    if (nvrtcGetCUBINSize(prog, &sz) == NVRTC_SUCCESS && sz > 0) {
        image.resize(sz);
        nr = nvrtcGetCUBIN(prog, image.data());
        used_cubin = true;
    } else if (nvrtcGetPTXSize(prog, &sz) == NVRTC_SUCCESS && sz > 0) {
        image.resize(sz);
        nr = nvrtcGetPTX(prog, image.data());
    } else {
        nr = NVRTC_ERROR_INTERNAL_ERROR;
    }
    nvrtcDestroyProgram(&prog);
    if (nr != NVRTC_SUCCESS || image.empty()) {
        return fail_sticky("nvrtc get cubin/ptx", nvrtcGetErrorString(nr));
    }

    CUmodule mod = nullptr;
    CUresult cr = api.ModuleLoadData(&mod, image.data());
    if (cr != CUDA_SUCCESS) return fail_sticky("cuModuleLoadData", cu_err(cr));
    KernelArms arms;
    for (int e = 1; e <= kEptMax; ++e) {
        char name[64];
        std::snprintf(name, sizeof(name), "lrx_conv_klead_c128_e%d", e);
        cr = api.ModuleGetFunction(&arms.fn[e - 1], mod, name);
        if (cr != CUDA_SUCCESS) {
            api.ModuleUnload(mod);
            return fail_sticky("cuModuleGetFunction",
                               std::string(name) + ": " + cu_err(cr));
        }
        if (smem_optin > 49152) {
            // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES == 8.
            (void)api.FuncSetAttribute(arms.fn[e - 1], 8, smem_optin);
        }
    }
    arms.smem_max = smem_optin > 49152 ? smem_optin : 49152;
    if (log_enabled()) {
        std::fprintf(stderr,
            "[conv_klead] NVRTC kernels compiled for sm_%d%d (%zu B %s, "
            "device %d, %d arms); dynamic shared max %d B\n",
            cc_major, cc_minor, image.size(), used_cubin ? "cubin" : "ptx",
            dev, kEptMax, arms.smem_max);
    }
    auto res = cache.emplace(ctx, arms);
    *out = &res.first->second;
    return ffi::Error::Success();
}

struct LaunchCfg {
    int rb = 1;
    int tpr = 1;
    int ept = 1;
    int sp = 1;
    int smem = 0;
};

static bool plan_launch(int nk, int n0, int n1, int n2, int smem_max,
                        LaunchCfg* cfg, std::string* why) {
    const int ntw = n0 + n1 + n2;
    const int sp = nk | 1;
    // Two resident c128 rows (T and W), twiddle rings, and one int64 W-row
    // offset per T row.  This exact expression owns the residency bound.
    auto smem_for = [&](long long rows) {
        return 16LL * (2LL * rows * sp + ntw) + 8LL * rows;
    };
    long long budget = kSmemPreferred;
    if (smem_for(1) > budget) budget = smem_max;
    long long rb = (budget - 16LL * ntw) / (32LL * sp + 8LL);
    if (rb < 1) {
        std::ostringstream os;
        os << "k-grid product nk=" << nk << " needs " << smem_for(1)
           << " B of shared memory for ONE resident T/W row pair, but this "
              "device permits " << smem_max << " B.  Residency is derived "
              "from the loaded device, never guessed.  This direct handler "
              "cannot serve the shape; use lorrax_mklfft_gw_conv, the "
              "plan-based k-leading family member with no row-residency "
              "requirement.";
        *why = os.str();
        return false;
    }
    rb = std::min<long long>(rb, kRowsMax);

    long long tpr_want = (nk + kEptPref - 1) / kEptPref;
    if (tpr_want * rb < kBlockTarget) {
        tpr_want = std::max<long long>(tpr_want, kBlockTarget / rb);
    }
    tpr_want = std::min<long long>(std::max<long long>(tpr_want, 1), nk);
    long long tpr = 0;
    for (long long t = tpr_want; t <= nk; ++t) {
        if (nk % t == 0) { tpr = t; break; }
    }
    if (tpr == 0) tpr = tpr_want;
    while (tpr * rb > kMaxBlock && rb > 1) --rb;
    if (tpr * rb > kMaxBlock) {
        std::ostringstream os;
        os << "nk=" << nk << " requires " << tpr
           << " threads per row beyond the " << kMaxBlock
           << "-thread launch bound; use lorrax_mklfft_gw_conv.";
        *why = os.str();
        return false;
    }
    const long long ept = (nk + tpr - 1) / tpr;
    if (ept > kEptMax) {
        std::ostringstream os;
        os << "internal launch plan ept=" << ept << " exceeds " << kEptMax;
        *why = os.str();
        return false;
    }
    cfg->rb = static_cast<int>(rb);
    cfg->tpr = static_cast<int>(tpr);
    cfg->ept = static_cast<int>(ept);
    cfg->sp = sp;
    cfg->smem = static_cast<int>(smem_for(rb));
    if (cfg->smem > smem_max) {
        std::ostringstream os;
        os << "internal launch plan requests " << cfg->smem
           << " B over the device maximum " << smem_max << " B";
        *why = os.str();
        return false;
    }
    return true;
}

static ffi::Error ConvKLeadDispatch(
    cudaStream_t stream,
    ffi::AnyBuffer T, ffi::AnyBuffer W, ffi::Result<ffi::AnyBuffer> U,
    int64_t nkx, int64_t nky, int64_t nkz, double scale) {
    if (T.element_type() != ffi::DataType::C128 ||
        W.element_type() != ffi::DataType::C128 ||
        U->element_type() != ffi::DataType::C128) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "conv_klead: complex128 only — complex64 is unsupported and is "
            "never up-cast.  Use lorrax_mklfft_gw_conv or the caller's "
            "reference path for another dtype.");
    }
    auto td = T.dimensions();
    auto wd = W.dimensions();
    auto ud = U->dimensions();
    if (td.size() != 5 || wd.size() != 3 || ud.size() != 5) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "conv_klead: expected Tm (a,mx,b,my,nk), U (nk,a,mx,b,my), "
            "and W (nk,mx,my).");
    }
    const int64_t a = td[0], mx = td[1], b = td[2], my = td[3], nk = td[4];
    if (nkx < 1 || nky < 1 || nkz < 1 || nk != nkx * nky * nkz ||
        wd[0] != nk || wd[1] != mx || wd[2] != my ||
        ud[0] != nk || ud[1] != a || ud[2] != mx || ud[3] != b ||
        ud[4] != my) {
        std::ostringstream os;
        os << "conv_klead: shape mismatch — Tm(a=" << a << ",mx=" << mx
           << ",b=" << b << ",my=" << my << ",nk=" << nk << ") vs W("
           << wd[0] << "," << wd[1] << "," << wd[2] << ") vs kgrid ("
           << nkx << "," << nky << "," << nkz << ").";
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    if (nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax) {
        std::ostringstream os;
        os << "conv_klead: runtime axis extents must each be in [1,"
           << kAxisMax << "]; got (" << nkx << "," << nky << "," << nkz
           << ").  Use lorrax_mklfft_gw_conv for a larger axis.";
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    const int64_t Tg = a * mx * b * my;
    const int64_t Tv = mx * my;
    if (Tg == 0 || Tv == 0) return ffi::Error::Success();

    const KernelArms* arms = nullptr;
    ffi::Error err = ffi::Error::Success();
    {
        std::lock_guard<std::mutex> lock(g_mu);
        err = ensure_kernels(&arms);
    }
    if (!err.success()) return err;

    LaunchCfg cfg;
    std::string why;
    if (!plan_launch(static_cast<int>(nk), static_cast<int>(nkx),
                     static_cast<int>(nky), static_cast<int>(nkz),
                     arms->smem_max, &cfg, &why)) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "conv_klead: " + why);
    }

    const auto* t_in = static_cast<const double*>(T.untyped_data());
    const auto* w_in = static_cast<const double*>(W.untyped_data());
    auto* u_out = static_cast<double*>(U->untyped_data());
    if (log_enabled()) {
        static std::atomic<bool> once{false};
        if (!once.exchange(true)) {
            std::fprintf(stderr,
                "[conv_klead] first call: kgrid=(%lld,%lld,%lld) nk=%lld "
                "Tm=(%lld,%lld,%lld,%lld,nk) trail=%lld scale=%.9e rb=%d "
                "tpr=%d ept=%d sp=%d smem=%d B grid=%lld output=k-leading\n",
                (long long)nkx, (long long)nky, (long long)nkz,
                (long long)nk, (long long)a, (long long)mx, (long long)b,
                (long long)my, (long long)Tg, scale, cfg.rb, cfg.tpr,
                cfg.ept, cfg.sp, cfg.smem,
                (long long)((Tg + cfg.rb - 1) / cfg.rb));
        }
    }

    long long a_Tg = Tg, a_Tv = Tv, a_mx = mx, a_b = b, a_my = my;
    int a_nk = static_cast<int>(nk), a_n0 = static_cast<int>(nkx);
    int a_n1 = static_cast<int>(nky), a_n2 = static_cast<int>(nkz);
    int a_sp = cfg.sp;
    double a_scale = scale;
    void* args[] = {(void*)&t_in, (void*)&w_in, (void*)&u_out,
                    &a_Tg, &a_Tv, &a_mx, &a_b, &a_my,
                    &a_nk, &a_n0, &a_n1, &a_n2, &a_sp, &a_scale};
    const int64_t nblocks = (Tg + cfg.rb - 1) / cfg.rb;
    if (nblocks > 2147483647LL) {
        std::ostringstream os;
        os << "conv_klead: " << nblocks
           << " blocks exceeds grid.x; split a trailing axis.";
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    CUresult cr = driver_api().LaunchKernel(
        arms->fn[cfg.ept - 1], static_cast<unsigned>(nblocks), 1, 1,
        static_cast<unsigned>(cfg.tpr), static_cast<unsigned>(cfg.rb), 1,
        static_cast<unsigned>(cfg.smem), reinterpret_cast<CUstream>(stream),
        args, nullptr);
    if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::conv_klead

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CufftConvKLeadCudaFfi,
    lorrax_ffi::conv_klead::ConvKLeadDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale"));
