// kconv_mathdx_cuda_ffi.cc -- the NVIDIA k-convolution family on nvidia-mathdx.
//
// One fused, one-memory-pass kernel family whose FFTs are the LIBRARY's:
// cuFFTDx thread FFTs (nvidia-mathdx), compiled by NVRTC at run time once per
// (mode, nkx, nky, nkz, ns, CUDA context) and cached in process.  LORRAX
// writes no DFT, no twiddle table and no radix code here; the per-size
// specialisation is the library's, made at JIT time.  Ruling:
// docs/architecture/decisions.md 2026-09-24; layering:
// docs/architecture/ffi_layout.md "k-convolution router and the mathdx family".
//
// Modes (LRX_MODE of the embedded source; one handler per mode):
//   0 pair    U[kx,ky,kz,col,mu] = s * FFT_k( sum_ab phase_ab
//                 conj(IFFT_k A[k,a,col,mu,b]) * IFFT_k B[k,perm_l a,col,mu,perm_r b] )
//   1 parent  the same contraction with the typed parent load of
//             docs/architecture/ffi_layout.md "Parent-load ISDF pair
//             convolution": A/B are raw-parent (p,ns,mu,ns,nu) projectors and
//             the load applies the umklapp phases, the antiunitary conjugation
//             and the open-spin coefficients; U is (nk, mu, nu).
// A new mode adds (1) an entry under its LRX_MODE value in kSrc, (2) a mode
// code and a handler below, (3) a router factory in ffi/fft.py.
//
// Residency: every row (one (col,mu) pair) keeps three nk-long banks in shared
// memory; a k-grid whose row does not fit the device's opt-in shared memory,
// or an axis above the fp64 thread-FFT limit (40), is refused by name.
//
// Headers: the Python router passes the installed wheel's nvidia/mathdx
// directory as the string attribute `mathdx_root`; the CUDA toolkit include
// (for libcu++, include/cccl) is derived from the loaded libnvrtc.  No
// environment variable is read.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <dlfcn.h>
#include <sys/stat.h>

#include "../common/mkl_thread_pin.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "xla/ffi/api/ffi.h"

namespace lorrax_ffi::kconv_mathdx {

namespace ffi = ::xla::ffi;

static constexpr int kAxisMax = 40;        // cuFFTDx fp64 thread-FFT limit
static constexpr int kRowsMax = 16;
static constexpr int kThreads = 256;
static constexpr long long kSmemBudget = 100 * 1024;

static bool log_enabled() {
    static const bool on = [] { return mklpin::debug_print_here(); }();
    return on;
}

static ffi::Error fail(const char* where, const std::string& detail,
                       ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
    std::ostringstream os;
    os << "kconv_mathdx (fused cuFFTDx k-convolution): " << where
       << " failed -- " << detail;
    return ffi::Error(code, os.str());
}

#define LRX_CUDA_CHECK(expr, where)                                      \
    do {                                                                 \
        cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) return fail((where), cudaGetErrorString(_e)); \
    } while (0)

// Driver entry points resolved lazily (libcuda is already mapped by JAX).
struct DriverApi {
    CUresult (*ModuleLoadData)(CUmodule*, const void*) = nullptr;
    CUresult (*ModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
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
        if (dlsym(h, "cuLaunchKernel") == nullptr) {
            h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
            if (h == nullptr) { a.err = "dlopen(libcuda.so.1) failed"; return a; }
        }
        auto need = [&](const char* name) {
            void* p = dlsym(h, name);
            if (p == nullptr) a.err += std::string(a.err.empty() ? "" : "; ") + "dlsym(" + name + ")";
            return p;
        };
        a.ModuleLoadData = reinterpret_cast<decltype(a.ModuleLoadData)>(need("cuModuleLoadData"));
        a.ModuleGetFunction = reinterpret_cast<decltype(a.ModuleGetFunction)>(need("cuModuleGetFunction"));
        a.LaunchKernel = reinterpret_cast<decltype(a.LaunchKernel)>(need("cuLaunchKernel"));
        a.CtxGetCurrent = reinterpret_cast<decltype(a.CtxGetCurrent)>(need("cuCtxGetCurrent"));
        a.GetErrorString = reinterpret_cast<decltype(a.GetErrorString)>(need("cuGetErrorString"));
        a.FuncSetAttribute = reinterpret_cast<decltype(a.FuncSetAttribute)>(need("cuFuncSetAttribute"));
        a.ok = a.ModuleLoadData && a.ModuleGetFunction && a.LaunchKernel &&
               a.CtxGetCurrent && a.GetErrorString && a.FuncSetAttribute;
        return a;
    }();
    return api;
}

static std::string cu_err(CUresult r) {
    const char* text = nullptr;
    if (driver_api().GetErrorString && driver_api().GetErrorString(r, &text) == CUDA_SUCCESS && text)
        return text;
    return "CUresult=" + std::to_string(static_cast<int>(r));
}

struct ParentTables {
    const int *irr, *sym, *left, *right, *trs;
    const double *L, *R, *q, *coef_l, *coef_r;
    int mu, nu, centroid_major;
};

// ---------------------------------------------------------------------------
//  The embedded source.  Compile-time: LRX_MODE, LRX_NX/NY/NZ, LRX_NS, LRX_RB,
//  LRX_SM.  The transforms are cuFFTDx thread FFTs; everything else (loads,
//  the typed parent action, the spin contraction, the store) is the
//  2026-09-06 conv_kparent contract.
// ---------------------------------------------------------------------------
static const char* kSrc = R"__lrx__(
#include <cufftdx.hpp>

struct __align__(16) lrx_c2 { double x, y; };
struct ParentTables {
    const int *irr, *sym, *left, *right, *trs;
    const double *L, *R, *q, *coef_l, *coef_r;
    int mu, nu, centroid_major;
};

__device__ __forceinline__ lrx_c2 lrx_mul(lrx_c2 a, lrx_c2 b) {
    lrx_c2 z = {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
    return z;
}
__device__ __forceinline__ lrx_c2 lrx_phase(lrx_c2 z, int code) {
    if (code == 1) { lrx_c2 q = {-z.y, z.x}; return q; }
    if (code == 2) { lrx_c2 q = {-z.x, -z.y}; return q; }
    if (code == 3) { lrx_c2 q = {z.y, -z.x}; return q; }
    return z;
}

// P = conj(sum_cd U_ac conj(U_bd) T[phase_L D_cd conj(phase_R)]) from owner-local maps.
__device__ __forceinline__ lrx_c2 lrx_parent_load(
    const lrx_c2* d, ParentTables t, int k, int a, int b, long long row, int ns, bool right) {
    const int m = row / t.nu, n = row % t.nu;
    const int p = t.irr[k], op = t.sym[k];
    const int lm = t.left[op*t.mu+m], rn = t.right[op*t.nu+n];
    double dl = 0.0, dr = 0.0;
    for (int i = 0; i < 3; ++i) {
        dl += t.q[3*p+i]*t.L[3*(op*t.mu+m)+i];
        dr += t.q[3*p+i]*t.R[3*(op*t.nu+n)+i];
    }
    lrx_c2 pl, pr;
    sincospi(2.0*dl, &pl.y, &pl.x);
    sincospi(-2.0*dr, &pr.y, &pr.x);
    const lrx_c2* coef = reinterpret_cast<const lrx_c2*>(right ? t.coef_r : t.coef_l)
        + ((long long)k*ns*ns + a*ns + b)*ns*ns;
    lrx_c2 sum = {0.0, 0.0};
    for (int c = 0; c < ns; ++c) for (int e = 0; e < ns; ++e) {
        const lrx_c2 weight = coef[c*ns+e];
        if (weight.x == 0.0 && weight.y == 0.0) continue;
        const long long index = t.centroid_major
            ? ((((long long)p*t.nu + rn)*ns + e)*t.mu + lm)*ns + c
            : (((long long)p*ns + c)*t.mu + lm)*ns*t.nu + e*t.nu + rn;
        lrx_c2 v = lrx_mul(lrx_mul(pl, d[index]), pr);
        if (t.trs[k]) v.y = -v.y;
        v = lrx_mul(weight, v);
        sum.x += v.x; sum.y += v.y;
    }
    sum.y = -sum.y;
    return sum;
}

constexpr int NX = LRX_NX, NY = LRX_NY, NZ = LRX_NZ, NS = LRX_NS, RB = LRX_RB;
constexpr int NK = NX * NY * NZ, SP = NK | 1;

template <int N, cufftdx::fft_direction Dir>
using TFFT = decltype(cufftdx::Size<N>() + cufftdx::Precision<double>() +
                      cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<Dir>() +
                      cufftdx::Thread() + cufftdx::SM<LRX_SM>());

// One axis of the 3-D transform on RB resident rows: every block thread runs
// whole library line-FFTs (flat k is C-order, kz fastest).
template <int N, int STRIDE, cufftdx::fft_direction Dir>
__device__ __forceinline__ void axis_pass(lrx_c2* bank) {
    if constexpr (N > 1) {
        using F = TFFT<N, Dir>;
        using V = typename F::value_type;
        constexpr int lines = NK / N;
        for (int l = threadIdx.x; l < RB * lines; l += blockDim.x) {
            const int j = l / lines, li = l % lines;
            lrx_c2* p = bank + j * SP + (li / STRIDE) * N * STRIDE + (li % STRIDE);
            V v[F::storage_size];
#pragma unroll
            for (int e = 0; e < N; ++e) { v[e].x = p[e * STRIDE].x; v[e].y = p[e * STRIDE].y; }
            F().execute(v);
#pragma unroll
            for (int e = 0; e < N; ++e) { p[e * STRIDE].x = v[e].x; p[e * STRIDE].y = v[e].y; }
        }
    }
    __syncthreads();
}

template <cufftdx::fft_direction Dir>
__device__ __forceinline__ void transform3(lrx_c2* bank) {
    axis_pass<NZ, 1, Dir>(bank);
    axis_pass<NY, NZ, Dir>(bank);
    axis_pass<NX, NY * NZ, Dir>(bank);
}

// Modes 0 (pair) and 1 (parent): the post-pair spin-contracted k-convolution.
extern "C" __global__ void __launch_bounds__(256) lrx_kconv(
    const lrx_c2* __restrict__ ain, const lrx_c2* __restrict__ bin, lrx_c2* __restrict__ uout,
    long long rows, double scale, unsigned long long perm_l, unsigned long long phase_l,
    unsigned long long perm_r, unsigned long long phase_r, ParentTables tables) {
    extern __shared__ lrx_c2 sm[];
    lrx_c2* abank = sm;
    lrx_c2* bbank = sm + RB * SP;
    lrx_c2* accum = sm + 2 * RB * SP;
    const long long r0 = (long long)blockIdx.x * RB;
    using namespace cufftdx;
    for (int a = 0; a < NS; ++a) {
        const int ap = (perm_l >> (4*a)) & 15;
        const int pc_l = (phase_l >> (2*a)) & 3;
        for (int b = 0; b < NS; ++b) {
            const int bp = (perm_r >> (4*b)) & 15;
            const int pc = (pc_l + ((phase_r >> (2*b)) & 3)) & 3;
            for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
                const int k = i / RB, j = i % RB;
                const long long row = r0 + j;
                lrx_c2 av = {0.0, 0.0}, bv = {0.0, 0.0};
                if (row < rows) {
#if LRX_MODE == 1
                    av = lrx_parent_load(ain, tables, k, a, b, row, NS, false);
                    bv = lrx_parent_load(bin, tables, k, ap, bp, row, NS, true);
#else
                    const long long base = (long long)k * NS * rows * NS;
                    av = ain[base + (long long)a * rows * NS + row * NS + b];
                    bv = bin[base + (long long)ap * rows * NS + row * NS + bp];
#endif
                }
                abank[j * SP + k] = av;
                bbank[j * SP + k] = bv;
            }
            __syncthreads();
            transform3<fft_direction::inverse>(abank);
            transform3<fft_direction::inverse>(bbank);
            for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
                const int q = (i / NK) * SP + (i % NK);
                const lrx_c2 ac = {abank[q].x, -abank[q].y};
                const lrx_c2 term = lrx_phase(lrx_mul(ac, bbank[q]), pc);
                if (a == 0 && b == 0) accum[q] = term;
                else { accum[q].x += term.x; accum[q].y += term.y; }
            }
            __syncthreads();
        }
    }
    transform3<fft_direction::forward>(accum);
    for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
        const int k = i / RB, j = i % RB;
        const long long row = r0 + j;
        if (row < rows) {
            const lrx_c2 v = accum[j * SP + k];
            uout[(long long)k * rows + row] = {v.x * scale, v.y * scale};
        }
    }
}
)__lrx__";

// ---------------------------------------------------------------------------
//  Build and cache
// ---------------------------------------------------------------------------
struct Built { CUfunction fn = nullptr; int rb = 1; int smem = 0; double compile_ms = 0.0; };
using Key = std::tuple<CUcontext, int, int, int, int, int>;       // ctx, mode, nkx, nky, nkz, ns
static std::mutex g_mu;
static std::map<Key, Built> g_cache;
static std::map<Key, std::string> g_fail;

static bool exists(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0; }

// The CUDA toolkit include next to the loaded libnvrtc (lib64 or targets/<arch>/lib).
static std::string toolkit_include(std::string* why) {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&nvrtcVersion), &info) || !info.dli_fname) {
        *why = "dladdr(nvrtcVersion) found no library path"; return "";
    }
    std::string lib(info.dli_fname);
    lib = lib.substr(0, lib.find_last_of('/'));
    for (const char* rel : {"/../include", "/../../include"}) {
        const std::string inc = lib + rel;
        if (exists(inc + "/cccl/cuda/std/type_traits") || exists(inc + "/cuda/std/type_traits")) return inc;
    }
    *why = "no include/cccl/cuda/std/type_traits beside the loaded libnvrtc (" + lib + ")";
    return "";
}

static ffi::Error build(int mode, int nkx, int nky, int nkz, int ns, std::string_view mathdx_root,
                        const Built** out) {
    const DriverApi& api = driver_api();
    if (!api.ok) return fail("driver-api resolve", api.err);
    CUcontext ctx = nullptr;
    CUresult cr = api.CtxGetCurrent(&ctx);
    if (cr != CUDA_SUCCESS || ctx == nullptr) {
        LRX_CUDA_CHECK(cudaFree(nullptr), "context bind (cudaFree(0))");
        cr = api.CtxGetCurrent(&ctx);
        if (cr != CUDA_SUCCESS || ctx == nullptr) return fail("cuCtxGetCurrent", cu_err(cr));
    }
    const Key key{ctx, mode, nkx, nky, nkz, ns};
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_cache.find(key); it != g_cache.end()) { *out = &it->second; return ffi::Error::Success(); }
    if (auto it = g_fail.find(key); it != g_fail.end()) return fail("kernel build (cached failure)", it->second);
    auto sticky = [&](const char* where, const std::string& why,
                      ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
        g_fail.emplace(key, std::string(where) + " -- " + why);
        return fail(where, why, code);
    };

    int dev = 0, cc_major = 0, cc_minor = 0, smem_optin = 0;
    LRX_CUDA_CHECK(cudaGetDevice(&dev), "cudaGetDevice");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev), "cc major");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev), "cc minor");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev),
                   "max opt-in shared memory");
    const int nk = nkx * nky * nkz, sp = nk | 1;
    const long long row_bytes = 3LL * 16 * sp;
    long long rb = std::min<long long>(kRowsMax, kSmemBudget / row_bytes);
    if (rb < 1) rb = std::min<long long>(kRowsMax, smem_optin / row_bytes);
    if (rb < 1) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-residency: got k-grid (" << nkx << "," << nky << "," << nkz
           << ") whose resident row needs 3*16*(nk|1)=" << row_bytes << " B; want <= "
           << smem_optin << " B of opt-in shared memory on this device; why: the fused "
              "one-pass family keeps a k-row in shared memory; fix: a smaller k-grid (the "
              "family has no out-of-core arm)";
        return sticky("residency", os.str(), ffi::ErrorCode::kInvalidArgument);
    }
    std::string why;
    const std::string cuda_inc = toolkit_include(&why);
    if (cuda_inc.empty()) return sticky("CUDA toolkit headers for NVRTC", why);
    const std::string root(mathdx_root);
    const std::string inc = root + "/include", cutlass = root + "/external/cutlass/include";
    if (!exists(inc + "/cufftdx.hpp")) {
        return sticky("GATE mathdx-headers",
                      "got no cufftdx.hpp under " + inc + "; want the nvidia-mathdx wheel; fix: "
                      "pip install nvidia-mathdx", ffi::ErrorCode::kFailedPrecondition);
    }
    std::vector<std::string> o = {
        "--std=c++17", "--device-as-default-execution-space",
        "--gpu-architecture=sm_" + std::to_string(cc_major) + std::to_string(cc_minor),
        "-I" + inc, "-I" + cutlass, "-I" + cuda_inc, "-I" + cuda_inc + "/cccl",
        "-DLRX_MODE=" + std::to_string(mode), "-DLRX_NX=" + std::to_string(nkx),
        "-DLRX_NY=" + std::to_string(nky), "-DLRX_NZ=" + std::to_string(nkz),
        "-DLRX_NS=" + std::to_string(ns), "-DLRX_RB=" + std::to_string(rb),
        "-DLRX_SM=" + std::to_string(cc_major * 100 + cc_minor * 10)};
    std::vector<const char*> opts;
    for (auto& s : o) opts.push_back(s.c_str());
    const auto t0 = std::chrono::steady_clock::now();
    nvrtcProgram prog = nullptr;
    nvrtcResult nr = nvrtcCreateProgram(&prog, kSrc, "lrx_kconv_mathdx.cu", 0, nullptr, nullptr);
    if (nr != NVRTC_SUCCESS) return sticky("nvrtcCreateProgram", nvrtcGetErrorString(nr));
    nr = nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
    if (nr != NVRTC_SUCCESS) {
        size_t n = 0; std::string log;
        if (nvrtcGetProgramLogSize(prog, &n) == NVRTC_SUCCESS && n > 1) { log.resize(n); nvrtcGetProgramLog(prog, &log[0]); }
        nvrtcDestroyProgram(&prog);
        return sticky("nvrtcCompileProgram", std::string(nvrtcGetErrorString(nr)) + " -- " + log.substr(0, 4000));
    }
    size_t n = 0; std::vector<char> cubin;
    if (nvrtcGetCUBINSize(prog, &n) != NVRTC_SUCCESS || n == 0) {
        nvrtcDestroyProgram(&prog); return sticky("nvrtcGetCUBINSize", "empty cubin");
    }
    cubin.resize(n); nvrtcGetCUBIN(prog, cubin.data()); nvrtcDestroyProgram(&prog);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CUmodule module = nullptr;
    cr = api.ModuleLoadData(&module, cubin.data());
    if (cr != CUDA_SUCCESS) return sticky("cuModuleLoadData", cu_err(cr));
    Built b;
    cr = api.ModuleGetFunction(&b.fn, module, "lrx_kconv");
    if (cr != CUDA_SUCCESS) return sticky("cuModuleGetFunction", cu_err(cr));
    b.rb = static_cast<int>(rb);
    b.smem = static_cast<int>(rb * row_bytes);
    b.compile_ms = ms;
    if (b.smem > 49152) {
        cr = api.FuncSetAttribute(b.fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, b.smem);
        if (cr != CUDA_SUCCESS) return sticky("cuFuncSetAttribute", cu_err(cr));
    }
    if (mklpin::announce_here() || log_enabled()) {
        std::fprintf(stderr, "[kconv_mathdx] NVRTC built mode=%d kgrid=(%d,%d,%d) ns=%d sm_%d%d in %.1f ms "
                     "(rows/block=%d, smem=%d B)\n", mode, nkx, nky, nkz, ns, cc_major, cc_minor, ms,
                     b.rb, b.smem);
    }
    *out = &(g_cache[key] = b);
    return ffi::Error::Success();
}

static bool pack_attrs(ffi::Span<const int64_t> perm, ffi::Span<const int64_t> phase, int64_t ns,
                       const char* side, unsigned long long* pp, unsigned long long* hp, std::string* why) {
    if (perm.size() != static_cast<size_t>(ns) || phase.size() != static_cast<size_t>(ns)) {
        *why = std::string(side) + " perm/phase lengths != ns"; return false;
    }
    unsigned seen = 0; *pp = 0; *hp = 0;
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

static ffi::Error Launch(cudaStream_t stream, int mode, ffi::AnyBuffer A, ffi::AnyBuffer B,
                         ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz,
                         double scale, ffi::Span<const int64_t> perm_l, ffi::Span<const int64_t> phase_l,
                         ffi::Span<const int64_t> perm_r, ffi::Span<const int64_t> phase_r,
                         std::string_view mathdx_root, const ParentTables* parent) {
    if (A.element_type() != ffi::DataType::C128 || B.element_type() != ffi::DataType::C128 ||
        U->element_type() != ffi::DataType::C128)
        return fail("contract", "complex128 only", ffi::ErrorCode::kInvalidArgument);
    auto ad = A.dimensions(), bd = B.dimensions(), ud = U->dimensions();
    const size_t rank = parent ? 5 : 7;
    if (ad.size() != rank || bd.size() != rank || ud.size() != (parent ? 3 : 5))
        return fail("contract", "operand ranks", ffi::ErrorCode::kInvalidArgument);
    for (size_t i = 0; i < rank; ++i)
        if (ad[i] != bd[i]) return fail("contract", "A/B shapes differ", ffi::ErrorCode::kInvalidArgument);
    const int64_t ns = ad[parent ? 1 : 3], d0 = ad[parent ? 2 : 4], d1 = ad[parent ? 4 : 5];
    const int64_t nk = nkx * nky * nkz, rows = d0 * d1;
    const bool shape_ok = nkx >= 1 && nky >= 1 && nkz >= 1 && (parent
        ? (ad[3] == ns && ud[0] == nk && ud[1] == d0 && ud[2] == d1)
        : (ad[0] == nkx && ad[1] == nky && ad[2] == nkz && ad[6] == ns &&
           ud[0] == nkx && ud[1] == nky && ud[2] == nkz && ud[3] == d0 && ud[4] == d1));
    if (!shape_ok || ns < 1 || ns > 4)
        return fail("contract", "shape/attribute mismatch", ffi::ErrorCode::kInvalidArgument);
    if (nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-axis: got k-grid (" << nkx << "," << nky << "," << nkz
           << "); want every axis <= " << kAxisMax << " (the fp64 cuFFTDx thread-FFT limit)";
        return fail("axis cap", os.str(), ffi::ErrorCode::kInvalidArgument);
    }
    unsigned long long pl = 0, hl = 0, pr = 0, hr = 0; std::string why;
    if (!pack_attrs(perm_l, phase_l, ns, "left", &pl, &hl, &why) ||
        !pack_attrs(perm_r, phase_r, ns, "right", &pr, &hr, &why))
        return fail("vertex attributes", why, ffi::ErrorCode::kInvalidArgument);
    if (rows == 0) return ffi::Error::Success();
    const Built* k = nullptr;
    ffi::Error e = build(mode, static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                         static_cast<int>(ns), mathdx_root, &k);
    if (!e.success()) return e;
    const auto* ap = static_cast<const double*>(A.untyped_data());
    const auto* bp = static_cast<const double*>(B.untyped_data());
    auto* up = static_cast<double*>(U->untyped_data());
    long long rr = rows; double sc = scale;
    ParentTables none{};
    ParentTables tab = parent ? *parent : none;
    void* args[] = {(void*)&ap, (void*)&bp, (void*)&up, &rr, &sc, &pl, &hl, &pr, &hr, &tab};
    const long long blocks = (rows + k->rb - 1) / k->rb;
    if (blocks > 2147483647LL) return fail("launch", "grid.x overflow", ffi::ErrorCode::kInvalidArgument);
    CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, kThreads, 1, 1,
                                            static_cast<unsigned>(k->smem),
                                            reinterpret_cast<CUstream>(stream), args, nullptr);
    if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
    return ffi::Error::Success();
}

static ffi::Error PairDispatch(cudaStream_t stream, ffi::AnyBuffer A, ffi::AnyBuffer B,
                               ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz,
                               double scale, ffi::Span<const int64_t> perm_l, ffi::Span<const int64_t> phase_l,
                               ffi::Span<const int64_t> perm_r, ffi::Span<const int64_t> phase_r,
                               std::string_view mathdx_root) {
    return Launch(stream, 0, A, B, U, nkx, nky, nkz, scale, perm_l, phase_l, perm_r, phase_r,
                  mathdx_root, nullptr);
}

static ffi::Error ParentDispatch(
    cudaStream_t stream, ffi::AnyBuffer A, ffi::AnyBuffer B, ffi::AnyBuffer irr, ffi::AnyBuffer sym,
    ffi::AnyBuffer left, ffi::AnyBuffer right, ffi::AnyBuffer L, ffi::AnyBuffer R, ffi::AnyBuffer q,
    ffi::AnyBuffer trs, ffi::AnyBuffer coef_l, ffi::AnyBuffer coef_r, ffi::Result<ffi::AnyBuffer> U,
    int64_t nkx, int64_t nky, int64_t nkz, double scale, int64_t centroid_major,
    ffi::Span<const int64_t> perm_l, ffi::Span<const int64_t> phase_l,
    ffi::Span<const int64_t> perm_r, ffi::Span<const int64_t> phase_r, std::string_view mathdx_root) {
    const auto ad = A.dimensions();
    if (ad.size() != 5) return fail("parent contract", "D must have rank 5", ffi::ErrorCode::kInvalidArgument);
    const int64_t nk = nkx * nky * nkz, ns = ad[1], mu = ad[2], nu = ad[4];
    auto shape = [](ffi::AnyBuffer x, ffi::DataType t, std::vector<int64_t> dims) {
        auto d = x.dimensions();
        return x.element_type() == t && d.size() == dims.size() && std::equal(d.begin(), d.end(), dims.begin());
    };
    auto ld = left.dimensions();
    const int64_t ops = ld.size() == 2 ? ld[0] : 0;
    if ((centroid_major != 0 && centroid_major != 1) || mu < 1 || nu < 1 || ops < 1 ||
        !shape(irr, ffi::DataType::S32, {nk}) || !shape(sym, ffi::DataType::S32, {nk}) ||
        !shape(trs, ffi::DataType::S32, {nk}) || !shape(left, ffi::DataType::S32, {ops, mu}) ||
        !shape(right, ffi::DataType::S32, {ops, nu}) || !shape(L, ffi::DataType::F64, {ops, mu, 3}) ||
        !shape(R, ffi::DataType::F64, {ops, nu, 3}) || !shape(q, ffi::DataType::F64, {ad[0], 3}) ||
        !shape(coef_l, ffi::DataType::C128, {nk, ns * ns, ns * ns}) ||
        !shape(coef_r, ffi::DataType::C128, {nk, ns * ns, ns * ns}))
        return fail("parent tables", "typed table shape/dtype mismatch", ffi::ErrorCode::kInvalidArgument);
    ParentTables t{static_cast<const int*>(irr.untyped_data()), static_cast<const int*>(sym.untyped_data()),
                   static_cast<const int*>(left.untyped_data()), static_cast<const int*>(right.untyped_data()),
                   static_cast<const int*>(trs.untyped_data()), static_cast<const double*>(L.untyped_data()),
                   static_cast<const double*>(R.untyped_data()), static_cast<const double*>(q.untyped_data()),
                   static_cast<const double*>(coef_l.untyped_data()),
                   static_cast<const double*>(coef_r.untyped_data()),
                   static_cast<int>(mu), static_cast<int>(nu), static_cast<int>(centroid_major)};
    return Launch(stream, 1, A, B, U, nkx, nky, nkz, scale, perm_l, phase_l, perm_r, phase_r,
                  mathdx_root, &t);
}

}  // namespace lorrax_ffi::kconv_mathdx

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxPairCudaFfi, lorrax_ffi::kconv_mathdx::PairDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<xla::ffi::Span<const int64_t>>("perm_l")
        .Attr<xla::ffi::Span<const int64_t>>("phase_l")
        .Attr<xla::ffi::Span<const int64_t>>("perm_r")
        .Attr<xla::ffi::Span<const int64_t>>("phase_r")
        .Attr<std::string_view>("mathdx_root"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxParentCudaFfi, lorrax_ffi::kconv_mathdx::ParentDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("centroid_major")
        .Attr<xla::ffi::Span<const int64_t>>("perm_l")
        .Attr<xla::ffi::Span<const int64_t>>("phase_l")
        .Attr<xla::ffi::Span<const int64_t>>("perm_r")
        .Attr<xla::ffi::Span<const int64_t>>("phase_r")
        .Attr<std::string_view>("mathdx_root"));
