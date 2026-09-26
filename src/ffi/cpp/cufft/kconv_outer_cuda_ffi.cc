// kconv_outer_cuda_ffi.cc -- the k-convolution family's outer-product load (the BSE W term).
//
//   U[k,a,x,b,y] = s * FFT_k( IFFT_k T[.,a,x,b,y] * V[x,y,k] ),   V k-MINOR (the W_R tile as built)
//   T[k,a,x,b,y] = sum_K L[k,a,x,K] * R[k,K,b,y]          (formed on the load, never stored)
//                  (conj(R) with the attribute conj_r = 1: the BSE right leg is conj(psi_v),
//                   read from psi_v itself, so no conjugated copy is made)
//
// Mode 2 of kconv_mathdx_cuda_ffi.cc (the k-leading stored-kernel convolution on the k-box
// stage) with its T operand replaced by the rank-K sum the BSE encode builds.  The BSE T is
// 2.15 GB per rank per trial on CrI3 8x8 (mu 1448 at P4) and its encode has K = min(n_c, n_v)
// (8 to 14): a ZGEMM that writes T to HBM at ~7 flop/B, which the convolution then reads back.
// Here T exists only in shared memory.  The transforms, the kernel multiply and the scaled
// store are mode 2's (kbox_stage.cuh transform3, the family's lrx_mul, the store-side scale).
// The K sum runs on the fp64 tensor cores (mma.m8n8k4.f64) as Re += Lr Rr - Li Ri,
// Im += Lr Ri + Li Rr per 4-wide K chunk, in K order: measured bit for bit equal to XLA's
// batched ZGEMM of the same contraction (A100, cuBLAS DMMA), so U equals the XLA encode +
// mode 2 chain exactly (BSEMAX, 2026-09-26).
//
// Tile.  A block holds 64 columns: an 8 x 8 box of (x, y) at one (a, b), one m8n8 block per k.
// Warp w forms T at k = w, w + 16, ...: each lane reads one complex L[k, a, x0 + lane/4, K]
// and one R[k, K, b, y0 + lane/4] per chunk straight from global memory (the legs, ~12 MB
// each, are L2-resident; no shared staging), so shared memory is the k-box bank alone and two
// 512-thread blocks share an A100 SM.  A work item is (y block, b, group of (x block, a)
// pairs); the na x nb visits of one V tile are adjacent in time (a fastest inside an item, b
// fastest across items), so V leaves HBM about once and U is written once.
//
// Measured alternatives (CrI3 per-rank shape, warm, A100; runs/CrI3/500_bsemax_20260926):
// an fp64-FMA load with the R leg register-resident (one thread per (k, y)) ran 5.44-6.4 ms
// against this arm's 4.70 ms (25% occupancy); a 4 x 8 tile at four blocks/SM 5.32 ms.
//
// Limits (named refusals; ffi.fft.klead_outer_refusal mirrors them and routes such a shape to
// the unfused chain): the 64-column bank in opt-in shared memory, every axis <= 40 (the fp64
// cuFFTDx thread FFT), K a multiple of 4 (the door zero-pads), complex128 only.
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
static constexpr int kThreads = 512;       // 16 warps; one line per thread in each axis pass
static constexpr int kTile = 8;            // the m8n8 block: 8 x rows by 8 y columns

struct OuterGeo {                          // the embedded source declares the same struct
    long long na, mx, nb, my;              // U (nk, na, mx, nb, my)
    long long nxb, nyb;                    // x blocks, y blocks
    long long ngrp, per;                   // (x block, a) groups per (y block, b), pairs per group
    long long items;                       // nyb * nb * ngrp
    double scale;
    int conj_r;                            // 1: the load reads conj(R) (the BSE right leg is conj(psi))
    long long nc, nv;                      // decode arm: Pc (nk, nc, na, mx), Pv (nk, nv, nb, my)
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
    int conj_r;
    long long nc, nv;
};

constexpr int NX = LRX_NX, NY = LRX_NY, NZ = LRX_NZ, NK = NX * NY * NZ;
constexpr int KK = LRX_K, XB = 8, YB = 8, TR = XB * YB, NWARP = LRX_THREADS / 32;
static_assert(KK % 4 == 0, "K is a multiple of the m8n8k4 chunk (the door zero-pads)");
using GK = lrx_kbox::Geo<NX, NY, NZ>;

// The family's kernel multiply (kconv_mathdx_cuda_ffi.cc lrx_mul), spelled identically.
__device__ __forceinline__ lrx_c2 lrx_mul(lrx_c2 a, lrx_c2 b) {
    lrx_c2 z = {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
    return z;
}

// D += A B on the fp64 tensor cores: A 8x4 row (lane: row lane/4, col lane%4), B 4x8 col
// (lane: row lane%4, col lane/4), D 8x8 (lane: row lane/4, cols 2(lane%4) + {0, 1}).
__device__ __forceinline__ void lrx_dmma(double& d0, double& d1, double a, double b) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+d"(d0), "+d"(d1) : "d"(a), "d"(b));
}

#if LRX_DEC
// Decode arm (piece C prototype): no U store.  After the forward transform each warp folds its
// k columns into A[k, c, nu] += sum_mu conj(Pc[k, c, a, mu]) U[k, mu, nu] (DMMA, registers, the
// (t, mu)-first order of bse_stack_matvec._decode); at the item's end A meets Pv over nu and the
// (c, v, k) partial is added atomically into Y.
constexpr int MB = LRX_MB;                // 8-row c blocks
extern "C" __global__ void __launch_bounds__(LRX_THREADS, LRX_MINB) lrx_kconv_outer(
    const lrx_c2* __restrict__ L, const lrx_c2* __restrict__ R, const lrx_c2* __restrict__ V,
    const lrx_c2* __restrict__ Pc, const lrx_c2* __restrict__ Pv, double* __restrict__ Y, OuterGeo g) {
    lrx_c2* U = nullptr;
#else
extern "C" __global__ void __launch_bounds__(LRX_THREADS, LRX_MINB) lrx_kconv_outer(
    const lrx_c2* __restrict__ L, const lrx_c2* __restrict__ R, const lrx_c2* __restrict__ V,
    lrx_c2* __restrict__ U, OuterGeo g) {
#endif
    extern __shared__ lrx_c2 bank[];
    using namespace cufftdx;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int gr = lane >> 2, tg = lane & 3;
    const long long ucols = g.na * g.mx * g.nb * g.my;           // U columns per k
    const long long rk = g.nb * g.my;                             // R stride between K rows
    const long long pairs = g.na * g.nxb;
    // 32-bit offsets from per-tile 64-bit bases: every operand holds < 2^31 elements (the handler
    // checks), so only the bases need 64 bits -- fewer live registers at the 64-register cap.
    const int kl = (int)(g.na * g.mx * KK), ku = (int)ucols;
    const int nbmy = (int)(g.nb * g.my), rkk = (int)rk;
    for (long long it = blockIdx.x; it < g.items; it += gridDim.x) {
        const long long grp = it % g.ngrp, yb_ = it / g.ngrp;
        const int b = (int)(yb_ % g.nb), y0 = (int)(yb_ / g.nb) * YB;
        const int ylim = min(YB, (int)g.my - y0);
        const bool yok = gr < ylim;
        const double2* rb = reinterpret_cast<const double2*>(R + (long long)b * g.my + y0 + gr);
        const long long p0 = grp * g.per, p1 = min(pairs, p0 + g.per);
#if LRX_DEC
        double acc[(NK + NWARP - 1) / NWARP][MB][4];
#pragma unroll
        for (int kk = 0; kk < (NK + NWARP - 1) / NWARP; ++kk)
#pragma unroll
            for (int mb = 0; mb < MB; ++mb) { acc[kk][mb][0] = acc[kk][mb][1] = acc[kk][mb][2] = acc[kk][mb][3] = 0.0; }
#endif
        for (long long p = p0; p < p1; ++p) {
            const int a = (int)(p % g.na), x0 = (int)(p / g.na) * XB;
            const int xlim = min(XB, (int)g.mx - x0);
            const bool xok = gr < xlim;
            const double2* lb = reinterpret_cast<const double2*>(L + ((long long)a * g.mx + x0 + gr) * KK);
            // Load: T[k, a, x, b, y] = sum_K L[k, a, x, K] R[k, K, b, y] on the tensor cores.  With
            // conj_r the pair (-Li)(-Ri) is spelled Li Ri: IEEE products ignore a shared sign flip.
            for (int k = warp; k < NK; k += NWARP) {
                const double2* lp = lb + k * kl;
                const double2* rp = rb + k * KK * rkk;
                double re0 = 0.0, re1 = 0.0, im0 = 0.0, im1 = 0.0;
#pragma unroll
                for (int h = 0; h < KK / 4; ++h) {
                    const int q = 4 * h + tg;
                    const double2 lv = xok ? __ldg(lp + q) : make_double2(0.0, 0.0);
                    const double2 rv = yok ? __ldg(rp + q * rkk) : make_double2(0.0, 0.0);
                    lrx_dmma(re0, re1, lv.x, rv.x);
                    if (g.conj_r) {
                        lrx_dmma(re0, re1, lv.y, rv.y);
                        lrx_dmma(im0, im1, lv.x, -rv.y);
                    } else {
                        lrx_dmma(re0, re1, -lv.y, rv.y);
                        lrx_dmma(im0, im1, lv.x, rv.y);
                    }
                    lrx_dmma(im0, im1, lv.y, rv.x);
                }
                lrx_c2 v0, v1;
                v0.x = re0; v0.y = im0; v1.x = re1; v1.y = im1;
                bank[(gr * YB + 2 * tg) * GK::RS + GK::at(k)] = v0;
                bank[(gr * YB + 2 * tg + 1) * GK::RS + GK::at(k)] = v1;
            }
            __syncthreads();
            lrx_kbox::transform3<NX, NY, NZ, TR, LRX_SM, fft_direction::inverse>(bank);
            // Mid: the stored kernel V[x, y, k] (R space, k-MINOR: the caller's W_R tile as it is
            // built, no transpose), mode 2's KernMid product.  k runs fastest across threads, so a
            // warp reads 32 consecutive k of one (x, y) and walks one bank column.
            const lrx_c2* vb = V + ((long long)x0 * g.my + y0) * NK;
            for (int i = threadIdx.x; i < TR * NK; i += blockDim.x) {
                const int k = i % NK, j = i / NK, xi = j / YB, yi = j % YB;
                if (xi < xlim && yi < ylim) {
                    lrx_c2* e = bank + j * GK::RS + GK::at(k);
                    *e = lrx_mul(*e, vb[(xi * (int)g.my + yi) * NK + k]);
                }
            }
            __syncthreads();
            lrx_kbox::transform3<NX, NY, NZ, TR, LRX_SM, fft_direction::forward>(bank);
#if LRX_DEC
            // Decode: A[k, c, nu] += conj(Pc[k, c, a, x0 + mu]) U[k, mu, nu] (the scale is applied to Y).
#pragma unroll
            for (int kk = 0; kk < (NK + NWARP - 1) / NWARP; ++kk) {
                const int k = warp + kk * NWARP;
                if (k < NK) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const lrx_c2 ub = bank[((4 * h + tg) * YB + gr) * GK::RS + GK::at(k)];
                        const long long mu = x0 + 4 * h + tg;
#pragma unroll
                        for (int mb = 0; mb < MB; ++mb) {
                            const long long c = mb * 8 + gr;
                            const double2 pc = (c < g.nc && mu < g.mx)
                                ? __ldg(reinterpret_cast<const double2*>(Pc + (((long long)k * g.nc + c) * g.na + a) * g.mx + mu))
                                : make_double2(0.0, 0.0);
                            lrx_dmma(acc[kk][mb][0], acc[kk][mb][1], pc.x, ub.x);
                            lrx_dmma(acc[kk][mb][0], acc[kk][mb][1], pc.y, ub.y);
                            lrx_dmma(acc[kk][mb][2], acc[kk][mb][3], pc.x, ub.y);
                            lrx_dmma(acc[kk][mb][2], acc[kk][mb][3], -pc.y, ub.x);
                        }
                    }
                }
            }
            (void)U;
#else
            // Store: mode 2's RowStore (v * scale), U k-leading (nk, na, mx, nb, my).
            lrx_c2* ub = U + (((long long)a * g.mx + x0) * g.nb + b) * g.my + y0;
            for (int i = threadIdx.x; i < TR * NK; i += blockDim.x) {
                const int j = i % TR, k = i / TR, xi = j / YB, yi = j % YB;
                if (xi < xlim && yi < ylim) {
                    const lrx_c2 v = bank[j * GK::RS + GK::at(k)];
                    lrx_c2 w;
                    w.x = v.x * g.scale;
                    w.y = v.y * g.scale;
                    ub[k * ku + xi * nbmy + yi] = w;
                }
            }
#endif
            __syncthreads();
        }
#if LRX_DEC
        // Item end: Y[c, v, k] += scale * sum_nu Pv[k, v, b, y0 + nu] A[k, c, nu], one 8-row c block at a
        // time through the (now free) bank: A_s[k][c][nu] at k*64 + c*8 + nu.
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) {
#pragma unroll
            for (int kk = 0; kk < (NK + NWARP - 1) / NWARP; ++kk) {
                const int k = warp + kk * NWARP;
                if (k < NK) {
                    lrx_c2 a0, a1;
                    a0.x = acc[kk][mb][0]; a0.y = acc[kk][mb][2];
                    a1.x = acc[kk][mb][1]; a1.y = acc[kk][mb][3];
                    bank[k * 64 + gr * 8 + 2 * tg] = a0;
                    bank[k * 64 + gr * 8 + 2 * tg + 1] = a1;
                }
            }
            __syncthreads();
            for (long long o = threadIdx.x; o < (long long)NK * 8 * g.nv; o += blockDim.x) {
                const int k = (int)(o / (8 * g.nv)), cl = (int)((o / g.nv) % 8);
                const long long v = o % g.nv, c = mb * 8 + cl;
                if (c >= g.nc) continue;
                double sx = 0.0, sy = 0.0;
                for (int nu = 0; nu < YB; ++nu) {
                    if (y0 + nu >= g.my) break;
                    const lrx_c2 pv = Pv[(((long long)k * g.nv + v) * g.nb + b) * g.my + y0 + nu];
                    const lrx_c2 av = bank[k * 64 + cl * 8 + nu];
                    sx = fma(pv.x, av.x, sx); sx = fma(-pv.y, av.y, sx);
                    sy = fma(pv.x, av.y, sy); sy = fma(pv.y, av.x, sy);
                }
                double* yp = Y + 2 * ((c * g.nv + v) * NK + k);
                atomicAdd(yp, sx * g.scale);
                atomicAdd(yp + 1, sy * g.scale);
            }
            __syncthreads();
        }
#endif
    }
}
)__lrx__";

using nvrtc::DriverApi;
using nvrtc::driver_api;
using nvrtc::cu_err;

struct Built { CUfunction fn = nullptr; int smem = 0, minb = 1, sms = 0; };
using Key = std::tuple<CUcontext, int, int, int, int, int>;   // ctx, nkx, nky, nkz, K, decode c blocks (0: U arm)
static std::mutex g_mu;
static std::map<Key, Built> g_cache;
static std::map<Key, std::string> g_fail;

static ffi::Error build(int nkx, int nky, int nkz, int K, std::string_view mathdx_root,
                        std::string_view cubin_dir, const Built** out, int mb = 0) {
    const DriverApi& api = driver_api();
    if (!api.ok) return fail("driver-api resolve", api.err);
    CUcontext ctx = nullptr;
    CUresult cr = api.CtxGetCurrent(&ctx);
    if (cr != CUDA_SUCCESS || ctx == nullptr) {
        if (cudaFree(nullptr) != cudaSuccess) return fail("context bind", "cudaFree(0)");
        cr = api.CtxGetCurrent(&ctx);
        if (cr != CUDA_SUCCESS || ctx == nullptr) return fail("cuCtxGetCurrent", cu_err(cr));
    }
    const Key key{ctx, nkx, nky, nkz, K, mb};
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
    if (cc_major < 8)
        return sticky("GATE mathdx-kconv-outer-arch", "got sm_" + std::to_string(cc_major * 10 + cc_minor) +
                      "; want sm_80+ (fp64 mma.m8n8k4); fix: none here -- the router keeps the unfused chain",
                      ffi::ErrorCode::kFailedPrecondition);
    const lrx_kbox::Geometry geo{nkx, nky, nkz};
    const long long smem = static_cast<long long>(kTile) * kTile * geo.rs() * 16;
    if (smem > smem_optin) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-outer-tile: got k-grid (" << nkx << "," << nky << "," << nkz << ") whose "
           << kTile * kTile << "-column bank needs " << smem << " B; want <= " << smem_optin << " B of opt-in "
              "shared memory; why: the load forms one 8 x 8 block per k in a resident k-box tile; fix: none "
              "here -- ffi.fft.klead_outer_refusal routes such a grid to the unfused encode + mode 2 chain";
        return sticky("tile", os.str(), ffi::ErrorCode::kInvalidArgument);
    }
    const int minb = mb ? 1 : std::max(1, std::min<int>(2, static_cast<int>(smem_sm / (smem + 1024))));
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
        "-DLRX_K=" + std::to_string(K), "-DLRX_THREADS=" + std::to_string(kThreads),
        "-DLRX_MINB=" + std::to_string(minb), "-DLRX_SM=" + std::to_string(cc_major * 100 + cc_minor * 10),
        "-DLRX_DEC=" + std::string(mb ? "1" : "0"), "-DLRX_MB=" + std::to_string(mb ? mb : 1)};
    nvrtc::mathdx_toolchain(root, cuda_inc, "cufftdx", &prog);
    prog.kernel = "lrx_kconv_outer";
    std::string missing;
    const std::string key_hex = nvrtc::hex16(nvrtc::key(prog, &missing));
    const std::string dir(missing.empty() ? std::string(cubin_dir) : std::string());
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kconv_outer_" << nkx << "x" << nky << "x" << nkz << "_K" << K << (mb ? "_dec" + std::to_string(mb) : std::string()) << "_sm" << cc_major
             << cc_minor << "_" << key_hex << ".cubin";
        path = name.str();
    }
    nvrtc::Image img;
    std::string where, err;
    if (!nvrtc::build(prog, dir, path, key_hex, &img, &where, &err)) return sticky(where.c_str(), err);
    Built b;
    b.fn = img.fn;
    b.smem = static_cast<int>(smem);
    b.minb = minb;
    b.sms = sms;
    cr = api.FuncSetAttribute(b.fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, b.smem);
    if (cr != CUDA_SUCCESS) return sticky("cuFuncSetAttribute", cu_err(cr));
    if (mklpin::announce_here() || mklpin::debug_print_here()) {
        std::fprintf(stderr, "[kconv_outer] %s kgrid=(%d,%d,%d) K=%d sm_%d%d in %.1f ms (8x8 tile, %d threads, "
                     "%d blocks/SM, smem=%d B, cubin %s)\n",
                     img.from_disk ? "disk-cache hit" : "NVRTC built", nkx, nky, nkz, K, cc_major, cc_minor, img.ms,
                     kThreads, minb, b.smem,
                     path.empty() ? "not cached (no cubin_dir)" : (img.from_disk ? path.c_str() : "stored"));
    }
    *out = &(g_cache[key] = b);
    return ffi::Error::Success();
}

// L (nk, na, mx, K), R (nk, K, nb, my), V (mx, my, nk) -> U (nk, na, mx, nb, my), complex128.
static ffi::Error KleadOuterConv(cudaStream_t stream, ffi::AnyBuffer L, ffi::AnyBuffer R, ffi::AnyBuffer V,
                                 ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz,
                                 double scale, int64_t conj_r, std::string_view mathdx_root,
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
        return bad("want L (nk,na,mx,K), R (nk,K,nb,my), V (mx,my,nk), U (nk,na,mx,nb,my)");
    const int64_t na = ld[1], mx = ld[2], K = ld[3], nb = rd[2], my = rd[3];
    if (ld[0] != nk || rd[0] != nk || rd[1] != K || vd[0] != mx || vd[1] != my || vd[2] != nk ||
        ud[0] != nk || ud[1] != na || ud[2] != mx || ud[3] != nb || ud[4] != my)
        return bad("want L (nk,na,mx,K), R (nk,K,nb,my), V (mx,my,nk), U (nk,na,mx,nb,my) with nk = nkx*nky*nkz");
    if (K < 4 || K % 4) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-outer-rank: got K=" << K << "; want a positive multiple of 4 (the m8n8k4 "
              "chunk); fix: ffi.fft.make_local_kconv_klead_outer zero-pads the legs";
        return bad(os.str());
    }
    if (na * mx * nb * my == 0) return ffi::Error::Success();
    if (nk * na * mx * nb * my >= (int64_t(1) << 31) || nk * na * mx * K >= (int64_t(1) << 31) ||
        nk * K * nb * my >= (int64_t(1) << 31))
        return bad("an operand holds >= 2^31 elements (the load's 32-bit offsets); split the call");
    const Built* k = nullptr;
    if (ffi::Error e = build(static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                             static_cast<int>(K), mathdx_root, cubin_dir, &k);
        !e.success())
        return e;
    OuterGeo g{};
    g.na = na; g.mx = mx; g.nb = nb; g.my = my;
    g.nxb = (mx + kTile - 1) / kTile;
    g.nyb = (my + kTile - 1) / kTile;
    // Groups of (x block, a) pairs per (y block, b): about eight resident waves of items, so
    // the last wave is short.
    const long long slots = static_cast<long long>(k->sms) * k->minb;
    const long long base = nb * g.nyb, pairs = na * g.nxb;
    g.ngrp = std::max(1LL, std::min(pairs, (8 * slots + base - 1) / base));
    g.per = (pairs + g.ngrp - 1) / g.ngrp;
    g.items = base * g.ngrp;
    g.scale = scale;
    g.conj_r = conj_r ? 1 : 0;
    const void* lp = L.untyped_data();
    const void* rp = R.untyped_data();
    const void* vp = V.untyped_data();
    void* up = U->untyped_data();
    void* args[] = {(void*)&lp, (void*)&rp, (void*)&vp, (void*)&up, (void*)&g};
    const long long blocks = std::min(g.items, 2147483647LL);
    CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, kThreads, 1, 1,
                                            static_cast<unsigned>(k->smem), reinterpret_cast<CUstream>(stream),
                                            args, nullptr);
    if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
    return ffi::Error::Success();
}

// Piece C prototype: the decode fused into the store.  L, R, V as KleadOuterConv; Pc (nk, nc, na, mx)
// = psi_c_X, Pv (nk, nv, nb, my) = psi_v_Y; Y (nc, nv, nk) = scale * sum conj(Pc) Pv U, this rank's
// (mu_loc, nu_loc) partial in bse_stack_matvec._decode's order.  U is never stored.
static ffi::Error KleadOuterDecode(cudaStream_t stream, ffi::AnyBuffer L, ffi::AnyBuffer R, ffi::AnyBuffer V,
                                   ffi::AnyBuffer Pc, ffi::AnyBuffer Pv, ffi::Result<ffi::AnyBuffer> Y,
                                   int64_t nkx, int64_t nky, int64_t nkz, double scale, int64_t conj_r,
                                   std::string_view mathdx_root, std::string_view cubin_dir) {
    auto bad = [](const std::string& why) { return fail("klead outer decode", why, ffi::ErrorCode::kInvalidArgument); };
    const int64_t nk = nkx * nky * nkz;
    auto ld = L.dimensions(), rd = R.dimensions(), cd = Pc.dimensions(), pd = Pv.dimensions(), yd = Y->dimensions();
    if (ld.size() != 4 || rd.size() != 4 || cd.size() != 4 || pd.size() != 4 || yd.size() != 3)
        return bad("want L (nk,na,mx,K), R (nk,K,nb,my), Pc (nk,nc,na,mx), Pv (nk,nv,nb,my), Y (nc,nv,nk)");
    const int64_t na = ld[1], mx = ld[2], K = ld[3], nb = rd[2], my = rd[3], nc = cd[1], nv = pd[1];
    if (ld[0] != nk || rd[0] != nk || rd[1] != K || cd[0] != nk || cd[2] != na || cd[3] != mx || pd[0] != nk ||
        pd[2] != nb || pd[3] != my || yd[0] != nc || yd[1] != nv || yd[2] != nk || K % 4 || nc < 1 || nv < 1)
        return bad("operand shapes disagree");
    if (cudaMemsetAsync(Y->untyped_data(), 0, static_cast<size_t>(nc * nv * nk) * 16, stream) != cudaSuccess)
        return fail("klead outer decode", "cudaMemsetAsync(Y)");
    const Built* k = nullptr;
    const int mb = static_cast<int>((nc + 7) / 8);
    if (ffi::Error e = build(static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                             static_cast<int>(K), mathdx_root, cubin_dir, &k, mb);
        !e.success())
        return e;
    OuterGeo g{};
    g.na = na; g.mx = mx; g.nb = nb; g.my = my;
    g.nxb = (mx + kTile - 1) / kTile;
    g.nyb = (my + kTile - 1) / kTile;
    const long long slots = static_cast<long long>(k->sms) * k->minb;
    const long long base = nb * g.nyb, pairs = na * g.nxb;
    g.ngrp = std::max(1LL, std::min(pairs, (4 * slots + base - 1) / base));
    g.per = (pairs + g.ngrp - 1) / g.ngrp;
    g.items = base * g.ngrp;
    g.scale = scale;
    g.conj_r = conj_r ? 1 : 0;
    g.nc = nc; g.nv = nv;
    const void* lp = L.untyped_data();
    const void* rp = R.untyped_data();
    const void* vp = V.untyped_data();
    const void* cp = Pc.untyped_data();
    const void* pp = Pv.untyped_data();
    void* yp = Y->untyped_data();
    void* args[] = {(void*)&lp, (void*)&rp, (void*)&vp, (void*)&cp, (void*)&pp, (void*)&yp, (void*)&g};
    CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(std::min(g.items, 2147483647LL)), 1, 1,
                                            kThreads, 1, 1, static_cast<unsigned>(k->smem),
                                            reinterpret_cast<CUstream>(stream), args, nullptr);
    if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::kconv_outer

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxKleadOuterDecodeCudaFfi, lorrax_ffi::kconv_outer::KleadOuterDecode,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()   // L
        .Arg<xla::ffi::AnyBuffer>()   // R
        .Arg<xla::ffi::AnyBuffer>()   // V
        .Arg<xla::ffi::AnyBuffer>()   // Pc
        .Arg<xla::ffi::AnyBuffer>()   // Pv
        .Ret<xla::ffi::AnyBuffer>()   // Y (nc, nv, nk)
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("conj_r")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxKleadOuterCudaFfi, lorrax_ffi::kconv_outer::KleadOuterConv,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()   // L (nk, na, mx, K)
        .Arg<xla::ffi::AnyBuffer>()   // R (nk, K, nb, my)
        .Arg<xla::ffi::AnyBuffer>()   // V (mx, my, nk), R space, k-minor
        .Ret<xla::ffi::AnyBuffer>()   // U (nk, na, mx, nb, my)
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("conj_r")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));
