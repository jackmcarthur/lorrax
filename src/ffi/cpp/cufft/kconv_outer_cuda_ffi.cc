// kconv_outer_cuda_ffi.cc -- the k-convolution family's outer-product load (the BSE W term).
//
//   U[k,a,x,b,y] = s * FFT_k( IFFT_k T[.,a,x,b,y] * V[k,x,y] ),
//   T[k,a,x,b,y] = sum_K L[k,a,x,K] * R[k,K,b,y]          (formed on the load, never stored)
//
// Mode 2 of kconv_mathdx_cuda_ffi.cc (the k-leading stored-kernel convolution on the k-box
// stage) with its T operand replaced by the rank-K sum the BSE encode builds.  The BSE T is
// 2.15 GB per rank per trial on CrI3 8x8 (mu 1448 at P4) and its encode has K = min(n_c, n_v)
// (8 to 14): a ZGEMM that writes T to HBM at ~7 flop/B, then the convolution reads it back.
// Here T exists only in shared memory: the load forms it from the two ISDF legs and the
// transforms, the kernel multiply and the scaled store are mode 2's (kbox_stage.cuh transform3,
// the family's lrx_mul, the store-side scale), so U differs from the XLA encode + mode 2 chain
// only by the order of the K sum (round-off class).
//
// Tile.  A block holds TR = XB * YB columns: an XB x YB box of (x, y) at one (a, b).  Its
// threads are NK * YB, one (k, y) each: a thread keeps R[k, :, b, y] (K complex values) in
// registers for the whole work item and forms its XB columns at k from L[k, a, x, :] (L1-served:
// the YB threads of one k read the same K-run).  A work item is (y block, b, group of (x block,
// a) pairs): R is read once per item.  The na x nb visits of one V tile are adjacent in time (a
// fastest inside an item, b fastest across items), so V leaves HBM about once and U is written
// once; L (~12 MB) and R stay L2-resident, L is re-read from L2 once per (b, y block).
//
// Limits (named refusals; the router routes such a shape to the unfused chain):
// K <= kKMax (register-resident leg), NK * YB <= 512 threads, the tile in opt-in shared memory,
// every axis <= 40 (the fp64 cuFFTDx thread FFT).  complex128 only.
//
// The NVRTC build, the disk cache and the version keying are common/nvrtc_build.h's, exactly
// as the family's (the same mathdx toolchain headers enter the key).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>

#include "../common/mkl_thread_pin.h"
#include "../common/nvrtc_build.h"
#include "kbox_stage.cuh"          // host half: Geometry
#include "kbox_stage_src.h"        // the same file as text, embedded into the NVRTC program

#include <cuda.h>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace lorrax_ffi::kconv_outer {

namespace ffi = ::xla::ffi;

static constexpr int kAxisMax = 40;        // cuFFTDx fp64 thread-FFT limit
static constexpr int kKMax = 16;           // register-resident right leg
static constexpr int kThreadsMax = 512;

struct OuterGeo {                          // the embedded source declares the same struct
    long long na, mx, nb, my;              // U (nk, na, mx, nb, my)
    long long nxb, nyb;                    // x blocks, y blocks
    long long ngrp, per;                   // (a, x block) groups per (b, y block), pairs per group
    long long items;                       // nb * nyb * ngrp
    double scale;
};

static ffi::Error fail(const char* where, const std::string& detail,
                       ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
    std::ostringstream os;
    os << "kconv_outer (fused outer-product k-convolution): " << where << " failed -- " << detail;
    return ffi::Error(code, os.str());
}

static const char* kOuterSrc = R"__lrx__(
#include <cufftdx.hpp>
typedef double lrx_real;
struct __align__(16) lrx_c2 { double x, y; };
#include "kbox_stage.cuh"

struct OuterGeo {
    long long na, mx, nb, my;
    long long nxb, nyb;
    long long ngrp, per;
    long long items;
    double scale;
};

constexpr int NX = LRX_NX, NY = LRX_NY, NZ = LRX_NZ, NK = NX * NY * NZ;
constexpr int KK = LRX_K, YB = LRX_YB, XB = LRX_XB, TR = XB * YB;
using GK = lrx_kbox::Geo<NX, NY, NZ>;

// The family's kernel multiply (kconv_mathdx_cuda_ffi.cc lrx_mul), spelled identically.
__device__ __forceinline__ lrx_c2 lrx_mul(lrx_c2 a, lrx_c2 b) {
    lrx_c2 z = {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
    return z;
}

extern "C" __global__ void __launch_bounds__(LRX_THREADS, LRX_MINB) lrx_kconv_outer(
    const lrx_c2* __restrict__ L, const lrx_c2* __restrict__ R, const lrx_c2* __restrict__ V,
    lrx_c2* __restrict__ U, OuterGeo g) {
    extern __shared__ lrx_c2 bank[];
    using namespace cufftdx;
    const int ek = threadIdx.x / YB, ey = threadIdx.x % YB;       // the encode's (k, y)
    const long long ucols = g.na * g.mx * g.nb * g.my;            // U columns per k
    const long long pairs = g.na * g.nxb;
    for (long long it = blockIdx.x; it < g.items; it += gridDim.x) {
        const long long grp = it % g.ngrp, yb_ = it / g.ngrp;
        const long long b = yb_ % g.nb, y0 = (yb_ / g.nb) * YB;
        const long long y = y0 + ey;
        const bool yok = y < g.my;
        lrx_c2 r[KK];
#pragma unroll
        for (int q = 0; q < KK; ++q) {
            if (yok) r[q] = R[(((long long)ek * KK + q) * g.nb + b) * g.my + y];
            else { r[q].x = 0.0; r[q].y = 0.0; }
        }
        const long long p1 = min(pairs, (grp + 1) * g.per);
        for (long long p = grp * g.per; p < p1; ++p) {
            const long long a = p % g.na, x0 = (p / g.na) * XB;
            // Load: T[k, a, x, b, y] = sum_K L[k, a, x, K] R[k, K, b, y], one K-ordered fma chain.
#pragma unroll
            for (int xi = 0; xi < XB; ++xi) {
                const long long x = x0 + xi;
                lrx_c2 acc = {0.0, 0.0};
                if (yok && x < g.mx) {
                    const lrx_c2* l = L + (((long long)ek * g.na + a) * g.mx + x) * KK;
#pragma unroll
                    for (int q = 0; q < KK; ++q) {
                        const double2 lv = __ldg(reinterpret_cast<const double2*>(l) + q);
                        acc.x = fma(lv.x, r[q].x, acc.x);
                        acc.x = fma(-lv.y, r[q].y, acc.x);
                        acc.y = fma(lv.x, r[q].y, acc.y);
                        acc.y = fma(lv.y, r[q].x, acc.y);
                    }
                }
                bank[(xi * YB + ey) * GK::RS + GK::at(ek)] = acc;
            }
            __syncthreads();
            lrx_kbox::transform3<NX, NY, NZ, TR, LRX_SM, fft_direction::inverse>(bank);
            // Mid: the stored kernel V[k, x, y] (R space), mode 2's KernMid.
            for (int i = threadIdx.x; i < TR * NK; i += blockDim.x) {
                const int j = i % TR, k = i / TR;
                const long long x = x0 + j / YB, yy = y0 + j % YB;
                if (x < g.mx && yy < g.my) {
                    lrx_c2* e = bank + j * GK::RS + GK::at(k);
                    *e = lrx_mul(*e, V[((long long)k * g.mx + x) * g.my + yy]);
                }
            }
            __syncthreads();
            lrx_kbox::transform3<NX, NY, NZ, TR, LRX_SM, fft_direction::forward>(bank);
            // Store: mode 2's RowStore (v * scale), U k-leading (nk, na, mx, nb, my).
            for (int i = threadIdx.x; i < TR * NK; i += blockDim.x) {
                const int j = i % TR, k = i / TR;
                const long long x = x0 + j / YB, yy = y0 + j % YB;
                if (x < g.mx && yy < g.my) {
                    const lrx_c2 v = bank[j * GK::RS + GK::at(k)];
                    lrx_c2 w;
                    w.x = v.x * g.scale;
                    w.y = v.y * g.scale;
                    U[(long long)k * ucols + ((a * g.mx + x) * g.nb + b) * g.my + yy] = w;
                }
            }
            __syncthreads();
        }
    }
}
)__lrx__";

using nvrtc::DriverApi;
using nvrtc::driver_api;
using nvrtc::cu_err;

struct Built { CUfunction fn = nullptr; int threads = 0, yb = 0, xb = 0, smem = 0, minb = 1, sms = 0; };
using Key = std::tuple<CUcontext, int, int, int, int, int>;   // ctx, nkx, nky, nkz, K, yb
static std::mutex g_mu;
static std::map<Key, Built> g_cache;
static std::map<Key, std::string> g_fail;

// yb_req: 0 = the rule below, else the y width (a power of two) to measure.
// Rule: the widest y run (<= 8, 128-byte runs of the U store and the V read) whose NK * YB
// threads stay <= 256, so two blocks share an SM; XB = 32 / YB (a 32-column tile, mode 2's).
static int pick_yb(int nk, int yb_req) {
    if (yb_req > 0) return yb_req;
    int yb = 1;
    while (yb < 8 && nk * yb * 2 <= 256) yb *= 2;
    return yb;
}

static ffi::Error build(int nkx, int nky, int nkz, int K, int yb_req, std::string_view mathdx_root,
                        std::string_view cubin_dir, const Built** out) {
    const DriverApi& api = driver_api();
    if (!api.ok) return fail("driver-api resolve", api.err);
    CUcontext ctx = nullptr;
    CUresult cr = api.CtxGetCurrent(&ctx);
    if (cr != CUDA_SUCCESS || ctx == nullptr) {
        if (cudaFree(nullptr) != cudaSuccess) return fail("context bind", "cudaFree(0)");
        cr = api.CtxGetCurrent(&ctx);
        if (cr != CUDA_SUCCESS || ctx == nullptr) return fail("cuCtxGetCurrent", cu_err(cr));
    }
    const int nk = nkx * nky * nkz;
    const int yb = pick_yb(nk, yb_req);
    const Key key{ctx, nkx, nky, nkz, K, yb};
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_cache.find(key); it != g_cache.end()) { *out = &it->second; return ffi::Error::Success(); }
    if (auto it = g_fail.find(key); it != g_fail.end()) return fail("kernel build (cached failure)", it->second);
    auto sticky = [&](const char* where, const std::string& why,
                      ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
        g_fail.emplace(key, std::string(where) + " -- " + why);
        return fail(where, why, code);
    };
    int dev = 0, cc_major = 0, cc_minor = 0, smem_optin = 0, smem_sm = 0, sms = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
        return sticky("device attributes", "cudaDeviceGetAttribute");
    const int threads = nk * yb;
    const int xb = std::max(1, 32 / yb);
    const lrx_kbox::Geometry geo{nkx, nky, nkz};
    const long long smem = static_cast<long long>(xb) * yb * geo.rs() * 16;
    if (yb < 1 || (yb & (yb - 1)) || threads > kThreadsMax || smem > smem_optin) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-outer-tile: got k-grid (" << nkx << "," << nky << "," << nkz << ") at y width "
           << yb << ": " << threads << " threads and " << smem << " B of shared memory; want a power-of-two y "
           << "width, <= " << kThreadsMax << " threads (one per (k, y)) and <= " << smem_optin << " B; why: the "
              "outer load keeps one (k, y) leg per thread and a 32-column k-box tile resident; fix: none here -- "
              "ffi.fft.klead_outer_refusal routes such a grid to the unfused encode + mode 2 chain";
        return sticky("tile", os.str(), ffi::ErrorCode::kInvalidArgument);
    }
    const int minb = std::max(1, std::min<int>(2, static_cast<int>(smem_sm / (smem + 1024))));
    std::string why;
    const std::string cuda_inc = nvrtc::toolkit_include(&why);
    if (cuda_inc.empty()) return sticky("CUDA toolkit headers for NVRTC", why);
    const std::string root(mathdx_root);
    if (!nvrtc::exists(root + "/include/cufftdx.hpp"))
        return sticky("GATE mathdx-headers", "got no cufftdx.hpp under " + root + "/include; want the "
                      "nvidia-mathdx wheel; fix: pip install nvidia-mathdx", ffi::ErrorCode::kFailedPrecondition);
    nvrtc::Program prog;
    prog.src = kOuterSrc;
    prog.name = "lrx_kconv_outer.cu";
    prog.headers = {{kbox::kHeaderName, kbox::kHeaderSrc}};
    prog.defs = {
        "--std=c++17", "--device-as-default-execution-space", "--generate-line-info",
        "--gpu-architecture=sm_" + std::to_string(cc_major) + std::to_string(cc_minor),
        "-DLRX_NX=" + std::to_string(nkx), "-DLRX_NY=" + std::to_string(nky), "-DLRX_NZ=" + std::to_string(nkz),
        "-DLRX_K=" + std::to_string(K), "-DLRX_YB=" + std::to_string(yb), "-DLRX_XB=" + std::to_string(xb),
        "-DLRX_THREADS=" + std::to_string(threads), "-DLRX_MINB=" + std::to_string(minb),
        "-DLRX_SM=" + std::to_string(cc_major * 100 + cc_minor * 10)};
    nvrtc::mathdx_toolchain(root, cuda_inc, "cufftdx", &prog);
    prog.kernel = "lrx_kconv_outer";
    std::string missing;
    const std::string key_hex = nvrtc::hex16(nvrtc::key(prog, &missing));
    const std::string dir(missing.empty() ? std::string(cubin_dir) : std::string());
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kconv_outer_" << nkx << "x" << nky << "x" << nkz << "_K" << K << "_yb" << yb << "_sm"
             << cc_major << cc_minor << "_" << key_hex << ".cubin";
        path = name.str();
    }
    nvrtc::Image img;
    std::string where, err;
    if (!nvrtc::build(prog, dir, path, key_hex, &img, &where, &err)) return sticky(where.c_str(), err);
    Built b;
    b.fn = img.fn;
    b.threads = threads;
    b.yb = yb;
    b.xb = xb;
    b.smem = static_cast<int>(smem);
    b.minb = minb;
    b.sms = sms;
    cr = api.FuncSetAttribute(b.fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, b.smem);
    if (cr != CUDA_SUCCESS) return sticky("cuFuncSetAttribute", cu_err(cr));
    if (mklpin::announce_here() || mklpin::debug_print_here()) {
        std::fprintf(stderr, "[kconv_outer] %s kgrid=(%d,%d,%d) K=%d sm_%d%d in %.1f ms (tile %dx%d, %d threads, "
                     "%d blocks/SM, smem=%d B, cubin %s)\n",
                     img.from_disk ? "disk-cache hit" : "NVRTC built", nkx, nky, nkz, K, cc_major, cc_minor, img.ms,
                     xb, yb, threads, minb, b.smem,
                     path.empty() ? "not cached (no cubin_dir)" : (img.from_disk ? path.c_str() : "stored"));
    }
    *out = &(g_cache[key] = b);
    return ffi::Error::Success();
}

// L (nk, na, mx, K), R (nk, K, nb, my), V (nk, mx, my) -> U (nk, na, mx, nb, my), complex128.
static ffi::Error KleadOuterConv(cudaStream_t stream, ffi::AnyBuffer L, ffi::AnyBuffer R, ffi::AnyBuffer V,
                                 ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz,
                                 double scale, int64_t yb, std::string_view mathdx_root,
                                 std::string_view cubin_dir) {
    auto bad = [](const std::string& why) { return fail("klead outer conv", why, ffi::ErrorCode::kInvalidArgument); };
    if (nkx < 1 || nky < 1 || nkz < 1 || nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-axis: got k-grid (" << nkx << "," << nky << "," << nkz
           << "); want every axis in [1, " << kAxisMax << "] (the fp64 cuFFTDx thread-FFT limit)";
        return bad(os.str());
    }
    const auto C = ffi::DataType::C128;
    if (L.element_type() != C || R.element_type() != C || V.element_type() != C || U->element_type() != C)
        return bad("operands must all be complex128");
    const int64_t nk = nkx * nky * nkz;
    auto ld = L.dimensions(), rd = R.dimensions(), vd = V.dimensions(), ud = U->dimensions();
    if (ld.size() != 4 || rd.size() != 4 || vd.size() != 3 || ud.size() != 5)
        return bad("want L (nk,na,mx,K), R (nk,K,nb,my), V (nk,mx,my), U (nk,na,mx,nb,my)");
    const int64_t na = ld[1], mx = ld[2], K = ld[3], nb = rd[2], my = rd[3];
    if (ld[0] != nk || rd[0] != nk || rd[1] != K || vd[0] != nk || vd[1] != mx || vd[2] != my ||
        ud[0] != nk || ud[1] != na || ud[2] != mx || ud[3] != nb || ud[4] != my)
        return bad("want L (nk,na,mx,K), R (nk,K,nb,my), V (nk,mx,my), U (nk,na,mx,nb,my) with nk = nkx*nky*nkz");
    if (K < 1 || K > kKMax) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-outer-rank: got K=" << K << "; want 1 <= K <= " << kKMax
           << "; why: the right leg is register-resident; fix: none here -- ffi.fft.klead_outer_refusal "
              "routes such a rank to the unfused encode + mode 2 chain";
        return bad(os.str());
    }
    if (na * mx * nb * my == 0) return ffi::Error::Success();
    const Built* k = nullptr;
    if (ffi::Error e = build(static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                             static_cast<int>(K), static_cast<int>(yb), mathdx_root, cubin_dir, &k);
        !e.success())
        return e;
    OuterGeo g{};
    g.na = na; g.mx = mx; g.nb = nb; g.my = my;
    g.nxb = (mx + k->xb - 1) / k->xb;
    g.nyb = (my + k->yb - 1) / k->yb;
    // Groups of (a, x block) pairs per (b, y block): about eight resident waves of items, so
    // the last wave is short; each item reads its R leg once.
    const long long slots = static_cast<long long>(k->sms) * k->minb;
    const long long base = nb * g.nyb, pairs = na * g.nxb;
    g.ngrp = std::max(1LL, std::min(pairs, (8 * slots + base - 1) / base));
    g.per = (pairs + g.ngrp - 1) / g.ngrp;
    g.items = base * g.ngrp;
    g.scale = scale;
    const void* lp = L.untyped_data();
    const void* rp = R.untyped_data();
    const void* vp = V.untyped_data();
    void* up = U->untyped_data();
    void* args[] = {(void*)&lp, (void*)&rp, (void*)&vp, (void*)&up, (void*)&g};
    const long long blocks = std::min(g.items, 2147483647LL);
    CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, k->threads, 1, 1,
                                            static_cast<unsigned>(k->smem), reinterpret_cast<CUstream>(stream),
                                            args, nullptr);
    if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::kconv_outer

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxKleadOuterCudaFfi, lorrax_ffi::kconv_outer::KleadOuterConv,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()   // L (nk, na, mx, K)
        .Arg<xla::ffi::AnyBuffer>()   // R (nk, K, nb, my)
        .Arg<xla::ffi::AnyBuffer>()   // V (nk, mx, my), R space
        .Ret<xla::ffi::AnyBuffer>()   // U (nk, na, mx, nb, my)
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("yb")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));
