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
#include <cstdlib>
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

extern "C" __global__ void __launch_bounds__(LRX_THREADS, LRX_MINB) lrx_kconv_outer(
    const lrx_c2* __restrict__ L, const lrx_c2* __restrict__ R, const lrx_c2* __restrict__ V,
    lrx_c2* __restrict__ U, OuterGeo g) {
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
            __syncthreads();
        }
    }
}
)__lrx__";

using nvrtc::DriverApi;
using nvrtc::driver_api;
using nvrtc::cu_err;

struct Built { CUfunction fn = nullptr; int smem = 0, minb = 1, sms = 0; };
using Key = std::tuple<CUcontext, int, int, int, int>;   // ctx, nkx, nky, nkz, K
static std::mutex g_mu;
static std::map<Key, Built> g_cache;
static std::map<Key, std::string> g_fail;

static ffi::Error build(int nkx, int nky, int nkz, int K, std::string_view mathdx_root,
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
    const Key key{ctx, nkx, nky, nkz, K};
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
        "-DLRX_K=" + std::to_string(K), "-DLRX_THREADS=" + std::to_string(kThreads),
        "-DLRX_MINB=" + std::to_string(minb), "-DLRX_SM=" + std::to_string(cc_major * 100 + cc_minor * 10)};
    nvrtc::mathdx_toolchain(root, cuda_inc, "cufftdx", &prog);
    prog.kernel = "lrx_kconv_outer";
    std::string missing;
    const std::string key_hex = nvrtc::hex16(nvrtc::key(prog, &missing));
    const std::string dir(missing.empty() ? std::string(cubin_dir) : std::string());
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kconv_outer_" << nkx << "x" << nky << "x" << nkz << "_K" << K << "_sm" << cc_major
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


// ---------------------------------------------------------------------------------------------
// The decode fused into the store (BSEC, 2026-09-26): U never reaches HBM.
//
//   A[k,c,b,y] = s * sum_{a,x} conj(Pc[k,c,a,x]) FFT_k( IFFT_k T[.,a,x,b,y] * V[x,y,k] )
//
// the first contraction of bse_stack_matvec._decode ((t, mu) first), on the outer load's tile:
// after the forward transform each warp folds its k columns into registers, A[k, c, nu] +=
// conj(Pc)[c, mu] U[mu, nu] on the fp64 tensor cores, over the (a, x block) pairs of one
// (b, y block) item.  The (s, nu) contraction with Pv stays main's einsum (the caller's).
//
// Work: an item's pairs are cut into ngrp phases of `per` pairs; the combos (phase, item) are
// dealt round robin in phase-major order to one resident block per SM, so the blocks of a wave
// sweep the same x window together and psi_c, L and the V tiles stay in L2.  A combo writes its
// scaled partial to its own slot; the combo that completes an item (a counter) sums the item's
// slots in phase order into A (nk, n_c, b, my): deterministic, no atomics on data, no waiting.
// The handler plans ngrp from the wave count, with the slots capped at 1/16 of U's bytes.
// Operands arrive in fragment order from the door, so every warp load is 512 contiguous bytes
// and needs no predicate:
//   Lr  (na, nxb, NK, H, 8, 4)      lane (g, t) of chunk h at k:  L[k, a, 8 xb + g, 4h + t]
//   Rr  (nb, nyb, NK, H, 8, 4)      lane (g, t) of chunk h at k:  R[k, 4h + t, b, 8 yb + g]
//   Pcr (na, nxb, NK, MB, 2, 8, 4)  lane (g, t) of (mb, h) at k:  Pc[k, 8 mb + g, a, 8 xb + 4h + t]
// zero-padded past mx, my, n_c and K, so a padded cell of the tile holds T = 0 and adds 0.
// Per pair: load (T on DMMA, K chunks in order, as the U arm) -> bank, inverse transform3, Mid
// (V[x, y, k] staged in shared memory by cp.async one pair ahead; a tile shared by consecutive
// pairs is staged once), forward transform3, decode.  The bank column of tile cell (x, y) is
// x*8 + (y ^ f(x)), f(x) = 3(x&1) + 4((x>>1)&1): the load's D-fragment stores and the decode's
// B-fragment reads then each hit 8 distinct 16-byte bank groups per phase (RS is 1 mod 8).
// The transforms, Mid and the K sum are the U arm's, so A is main's contraction of the same U
// in a different summation order (round-off class).  One block per SM: the accumulator is
// NK * MB * 8 * 8 complex per block (128 KB at 8x8, n_c 14), half the register file.
static const char* kDecodeSrc = R"__lrx__(
#include <cufftdx.hpp>
typedef double lrx_real;
struct __align__(16) lrx_c2 { double x, y; };
#include "kbox_stage.cuh"

struct DecodeGeo {
    long long na, nxb, nb, nyb, mx, my, nc;
    long long pairs, items, per, combos;
    double scale;
};

constexpr int NX = LRX_NX, NY = LRX_NY, NZ = LRX_NZ, NK = NX * NY * NZ;
constexpr int KK = LRX_K, H = KK / 4, MB = LRX_MB, TR = 64, NWARP = LRX_THREADS / 32;
constexpr int KW = (NK + NWARP - 1) / NWARP;
constexpr unsigned SLOT = NK * MB * 64;            // one combo's partial, complex elements
constexpr int VS = NK | 1;                         // staged V row stride (odd: the x pass reads it by rows)
static_assert(KK % 4 == 0, "K is a multiple of the m8n8k4 chunk (the door zero-pads)");
using GK = lrx_kbox::Geo<NX, NY, NZ>;

__device__ __forceinline__ lrx_c2 lrx_mul(lrx_c2 a, lrx_c2 b) {
    lrx_c2 z = {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
    return z;
}
__device__ __forceinline__ void lrx_dmma(double& d0, double& d1, double a, double b) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+d"(d0), "+d"(d1) : "d"(a), "d"(b));
}
__device__ __forceinline__ int fperm(int x) { return 3 * (x & 1) + 4 * ((x >> 1) & 1); }
__device__ __forceinline__ int pcol(int x, int y) { return x * 8 + (y ^ fperm(x)); }

// cp.async of the (xb, yb) V tile into vs[j * VS + k], bank column order; padded cells skipped.
__device__ __forceinline__ void stage_v(lrx_c2* vs, const lrx_c2* __restrict__ V, int xb, int yb,
                                        const DecodeGeo& g) {
    const int x0 = xb * 8, y0 = yb * 8;
    const int xlim = min(8, (int)g.mx - x0), ylim = min(8, (int)g.my - y0);
    const lrx_c2* vb = V + ((long long)x0 * g.my + y0) * NK;
    for (int o = threadIdx.x; o < TR * NK; o += blockDim.x) {
        const int k = o % NK, j = o / NK, x = j >> 3, y = (j & 7) ^ fperm(x);
        if (x < xlim && y < ylim) lrx_kbox::cp_async<16>(vs + j * VS + k, vb + (unsigned)((x * (int)g.my + y) * NK + k));
    }
    lrx_kbox::cp_async_commit();
}

// transform3's inverse (kbox_stage.cuh, same lines, same order) with mode 2's Mid folded into its
// x pass: each x line is multiplied by the staged V as it leaves the thread FFT, so the Mid's bank
// pass and barrier are gone and every product is the one the separate Mid would form.  The staged
// tile is awaited before the x pass.  A grid without an x axis keeps transform3 and the loop Mid.
__device__ __forceinline__ void inverse_mid(lrx_c2* bank, const lrx_c2* vs, int xlim, int ylim) {
    using namespace cufftdx;
    constexpr int tr = TR;
    using G = GK;
    int j, li;
    if constexpr (NX > 1) {
        if constexpr (NZ > 1) {
            for (int l = threadIdx.x; l < tr * NX * NY; l += blockDim.x) {
                lrx_kbox::line_of<TR>(l, NX * NY, j, li);
                lrx_kbox::line_fft<NZ, LRX_SM, fft_direction::inverse>(bank + j * G::RS + li * G::ZP, 1);
            }
            __syncthreads();
        }
        if constexpr (NY > 1) {
            for (int l = threadIdx.x; l < tr * NX * NZ; l += blockDim.x) {
                lrx_kbox::line_of<TR>(l, NX * NZ, j, li);
                lrx_kbox::line_fft<NY, LRX_SM, fft_direction::inverse>(bank + j * G::RS + (li / NZ) * NY * G::ZP + li % NZ, G::ZP);
            }
        }
        lrx_kbox::cp_async_wait_all();
        __syncthreads();
        using F = lrx_kbox::ThreadFFT<NX, LRX_SM, fft_direction::inverse, lrx_c2>;
        for (int l = threadIdx.x; l < tr * NY * NZ; l += blockDim.x) {
            lrx_kbox::line_of<TR>(l, NY * NZ, j, li);
            lrx_c2* p = bank + j * G::RS + G::plane_at(li);
            typename F::value_type v[F::storage_size];
#pragma unroll
            for (int e = 0; e < NX; ++e) { v[e].x = p[e * NY * G::ZP].x; v[e].y = p[e * NY * G::ZP].y; }
            F().execute(v);
            const int x = j >> 3, y = (j & 7) ^ fperm(x);
            const bool ok = x < xlim && y < ylim;
#pragma unroll
            for (int e = 0; e < NX; ++e) {
                lrx_c2 w; w.x = v[e].x; w.y = v[e].y;
                if (ok) w = lrx_mul(w, vs[j * VS + e * NY * NZ + li]);
                p[e * NY * G::ZP].x = w.x; p[e * NY * G::ZP].y = w.y;
            }
        }
        __syncthreads();
    } else {
        lrx_kbox::transform3<NX, NY, NZ, TR, LRX_SM, fft_direction::inverse>(bank);
        lrx_kbox::cp_async_wait_all();
        __syncthreads();
        for (int i = threadIdx.x; i < TR * NK; i += blockDim.x) {
            const int k = i % NK, jj = i / NK, x = jj >> 3, y = (jj & 7) ^ fperm(x);
            if (x < xlim && y < ylim) {
                lrx_c2* e = bank + jj * G::RS + G::at(k);
                *e = lrx_mul(*e, vs[jj * VS + k]);
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(LRX_THREADS, 1) lrx_kconv_outer_decode(
    const lrx_c2* __restrict__ Lr, const lrx_c2* __restrict__ Rr, const lrx_c2* __restrict__ V,
    const lrx_c2* __restrict__ Pcr, lrx_c2* __restrict__ A, lrx_c2* __restrict__ slots,
    int* __restrict__ count, DecodeGeo g) {
    extern __shared__ lrx_c2 smem[];
    __shared__ int last;
    lrx_c2* bank = smem;
    lrx_c2* vs = smem + TR * GK::RS;
    using namespace cufftdx;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int gr = lane >> 2, tg = lane & 3;
    const double2* L2 = reinterpret_cast<const double2*>(Lr) + lane;
    const double2* R2 = reinterpret_cast<const double2*>(Rr) + lane;
    const double2* P2 = reinterpret_cast<const double2*>(Pcr) + lane;
    const int nb = (int)g.nb, na = (int)g.na, items = (int)g.items;
    double acc[KW][MB][4];
#pragma unroll
    for (int kk = 0; kk < KW; ++kk)
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) acc[kk][mb][0] = acc[kk][mb][1] = acc[kk][mb][2] = acc[kk][mb][3] = 0.0;
    int vx = -1, vy = -1;
    if (blockIdx.x < g.combos) {
        const int c = blockIdx.x, it = c % items, p = (c / items) * (int)g.per;
        vx = p / na; vy = it / nb;
        stage_v(vs, V, vx, vy, g);
    }
    for (int c = blockIdx.x; c < g.combos; c += gridDim.x) {
        const int ph = c / items, it = c % items;
        const int b = it % nb, yb = it / nb;
        const int p0 = ph * (int)g.per, p1 = min((int)g.pairs, p0 + (int)g.per);
        const double2* rb = R2 + (unsigned)(b * (int)g.nyb + yb) * (unsigned)(NK * H * 32);
        const int ylim = min(8, (int)g.my - yb * 8);
        for (int p = p0; p < p1; ++p) {
            const int a = p % na, xb = p / na;
            const int xlim = min(8, (int)g.mx - xb * 8);
            // Load: T[k, a, x, b, y] = sum_K L R on the tensor cores, K chunks in order (the U arm's
            // sums; re and im interleaved, each accumulator's order unchanged).
            const double2* lb = L2 + (unsigned)(a * (int)g.nxb + xb) * (unsigned)(NK * H * 32);
            for (int k = warp; k < NK; k += NWARP) {
                double re0 = 0.0, re1 = 0.0, im0 = 0.0, im1 = 0.0;
#pragma unroll
                for (int h = 0; h < H; ++h) {
                    const double2 lv = __ldg(lb + (unsigned)((k * H + h) * 32));
                    const double2 rv = __ldg(rb + (unsigned)((k * H + h) * 32));
                    lrx_dmma(re0, re1, lv.x, rv.x);
#if LRX_CONJ
                    lrx_dmma(im0, im1, lv.x, -rv.y);
                    lrx_dmma(re0, re1, lv.y, rv.y);
#else
                    lrx_dmma(im0, im1, lv.x, rv.y);
                    lrx_dmma(re0, re1, -lv.y, rv.y);
#endif
                    lrx_dmma(im0, im1, lv.y, rv.x);
                }
                lrx_c2 v0, v1;
                v0.x = re0; v0.y = im0; v1.x = re1; v1.y = im1;
                bank[pcol(gr, 2 * tg) * GK::RS + GK::at(k)] = v0;
                bank[pcol(gr, 2 * tg + 1) * GK::RS + GK::at(k)] = v1;
            }
            __syncthreads();
#if LRX_XMID
            inverse_mid(bank, vs, xlim, ylim);        // inverse transform, Mid on its x pass
#else
            lrx_kbox::transform3<NX, NY, NZ, TR, LRX_SM, fft_direction::inverse>(bank);
            lrx_kbox::cp_async_wait_all();
            __syncthreads();
            for (int i = threadIdx.x; i < TR * NK; i += blockDim.x) {
                const int k = i % NK, jj = i / NK, x = jj >> 3, y = (jj & 7) ^ fperm(x);
                if (x < xlim && y < ylim) {
                    lrx_c2* e = bank + jj * GK::RS + GK::at(k);
                    *e = lrx_mul(*e, vs[jj * VS + k]);
                }
            }
            __syncthreads();
#endif
            {   // the next pair's V tile (this combo's, or the next combo's first), unless vs holds it
                int xn = -1, yn = -1;
                if (p + 1 < p1) { xn = (p + 1) / na; yn = yb; }
                else if (c + (int)gridDim.x < g.combos) {
                    const int c2 = c + gridDim.x;
                    xn = ((c2 / items) * (int)g.per) / na; yn = (c2 % items) / nb;
                }
                if (xn >= 0 && (xn != vx || yn != vy)) { stage_v(vs, V, xn, yn, g); vx = xn; vy = yn; }
            }
            lrx_kbox::transform3<NX, NY, NZ, TR, LRX_SM, fft_direction::forward>(bank);
            // Decode: A[k, c, nu] += conj(Pc[k, c, a, 8 xb + mu]) U[k, mu, nu]; per accumulator the
            // sums run mu-chunk (h) in order, re: Pr Ur then Pi Ui, im: Pr Ui then -Pi Ur.
            const double2* pb = P2 + (unsigned)(a * (int)g.nxb + xb) * (unsigned)(NK * MB * 64);
            // Loads for two k columns are issued before their DMMAs (h outer: each accumulator
            // still takes h = 0 before h = 1).
            constexpr int KP = (KW > 1 && LRX_KP > 1) ? 2 : 1;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int k0 = 0; k0 < KW; k0 += KP) {
                    lrx_c2 ub[KP];
                    double2 pc[KP][MB];
#pragma unroll
                    for (int q = 0; q < KP; ++q) {
                        const int k = warp + (k0 + q) * NWARP;
                        const int kc = k < NK ? k : 0;
                        ub[q] = bank[pcol(4 * h + tg, gr) * GK::RS + GK::at(kc)];
#pragma unroll
                        for (int mb = 0; mb < MB; ++mb) pc[q][mb] = __ldg(pb + (unsigned)(((kc * MB + mb) * 2 + h) * 32));
                    }
#pragma unroll
                    for (int q = 0; q < KP; ++q) {
                        if (warp + (k0 + q) * NWARP < NK) {
                            const int kk = k0 + q;
#pragma unroll
                            for (int mb = 0; mb < MB; ++mb) lrx_dmma(acc[kk][mb][0], acc[kk][mb][1], pc[q][mb].x, ub[q].x);
#pragma unroll
                            for (int mb = 0; mb < MB; ++mb) lrx_dmma(acc[kk][mb][2], acc[kk][mb][3], pc[q][mb].x, ub[q].y);
#pragma unroll
                            for (int mb = 0; mb < MB; ++mb) lrx_dmma(acc[kk][mb][0], acc[kk][mb][1], pc[q][mb].y, ub[q].y);
#pragma unroll
                            for (int mb = 0; mb < MB; ++mb) lrx_dmma(acc[kk][mb][2], acc[kk][mb][3], -pc[q][mb].y, ub[q].x);
                        }
                    }
                }
            }
            __syncthreads();
        }
        // The combo's scaled partial to its slot c; the combo completing the item sums the item's
        // slots in phase order into A[k, c, b, y].
        lrx_c2* sb = slots + (unsigned long long)c * SLOT;
#pragma unroll
        for (int kk = 0; kk < KW; ++kk) {
            const int k = warp + kk * NWARP;
#pragma unroll
            for (int mb = 0; mb < MB; ++mb) {
                if (k < NK) {
                    lrx_c2 a0, a1;
                    a0.x = acc[kk][mb][0] * g.scale; a0.y = acc[kk][mb][2] * g.scale;
                    a1.x = acc[kk][mb][1] * g.scale; a1.y = acc[kk][mb][3] * g.scale;
                    lrx_c2* q = sb + ((k * MB + mb) * 8 + gr) * 8 + 2 * tg;
                    q[0] = a0;
                    q[1] = a1;
                }
                acc[kk][mb][0] = acc[kk][mb][1] = acc[kk][mb][2] = acc[kk][mb][3] = 0.0;
            }
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) last = atomicAdd(count + it, 1) == (int)(g.combos / items) - 1;
        __syncthreads();
        if (last) {
            __threadfence();
            const int nc = (int)g.nc, y0 = yb * 8, ngrp = (int)(g.combos / items);
            for (int e = threadIdx.x; e < NK * nc * 8; e += blockDim.x) {
                const int nu = e % 8, cc = (e / 8) % nc, k = e / (8 * nc);
                if (y0 + nu < (int)g.my) {
                    const unsigned o = (unsigned)((k * MB * 8 + cc) * 8 + nu);
                    double2 s = __ldcg(reinterpret_cast<const double2*>(slots + (unsigned long long)it * SLOT + o));
                    for (int q = 1; q < ngrp; ++q) {
                        const double2 t = __ldcg(reinterpret_cast<const double2*>(
                            slots + (unsigned long long)(q * items + it) * SLOT + o));
                        s.x += t.x; s.y += t.y;
                    }
                    lrx_c2 w; w.x = s.x; w.y = s.y;
                    A[(((long long)k * nc + cc) * nb + b) * g.my + y0 + nu] = w;
                }
            }
        }
    }
}
)__lrx__";

struct DecodeGeo {                          // the embedded source declares the same struct
    long long na, nxb, nb, nyb, mx, my, nc;
    long long pairs, items, per, combos;
    double scale;
};

using DKey = std::tuple<CUcontext, int, int, int, int, int, int>;   // ctx, nkx, nky, nkz, K, MB, conj_r (the process's LRX_DEC_EXP is fixed)
static std::map<DKey, Built> g_dcache;
static std::map<DKey, std::string> g_dfail;

static ffi::Error build_decode(int nkx, int nky, int nkz, int K, int mb, int conj_r, std::string_view mathdx_root,
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
    const DKey key{ctx, nkx, nky, nkz, K, mb, conj_r};
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_dcache.find(key); it != g_dcache.end()) { *out = &it->second; return ffi::Error::Success(); }
    if (auto it = g_dfail.find(key); it != g_dfail.end()) return fail("kernel build (cached failure)", it->second);
    auto sticky = [&](const char* where, const std::string& why,
                      ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
        g_dfail.emplace(key, std::string(where) + " -- " + why);
        return fail(where, why, code);
    };
    int dev = 0, cc_major = 0, cc_minor = 0, smem_optin = 0, sms = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
        return sticky("device attributes", "cudaDeviceGetAttribute");
    if (cc_major < 8)
        return sticky("GATE mathdx-kconv-outer-arch", "got sm_" + std::to_string(cc_major * 10 + cc_minor) +
                      "; want sm_80+ (fp64 mma.m8n8k4); fix: none here -- the router keeps the unfused chain",
                      ffi::ErrorCode::kFailedPrecondition);
    const lrx_kbox::Geometry geo{nkx, nky, nkz};
    const long long nk = static_cast<long long>(nkx) * nky * nkz;
    const long long smem = static_cast<long long>(kTile) * kTile * (geo.rs() + (nk | 1)) * 16;
    const long long kw = (nk + kThreads / 32 - 1) / (kThreads / 32);
    if (smem + 16 > smem_optin || kw * mb > 8) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-outer-decode-tile: got k-grid (" << nkx << "," << nky << "," << nkz << "), "
           << mb << " c blocks: bank + staged V " << smem << " B (want <= " << smem_optin << " B of opt-in shared "
              "memory) and " << kw * mb << " accumulator blocks per lane (want <= 8, the register file); fix: none "
              "here -- ffi.fft.klead_outer_decode_refusal routes such a shape to the outer conv + XLA decode";
        return sticky("tile", os.str(), ffi::ErrorCode::kInvalidArgument);
    }
    std::string why;
    const std::string cuda_inc = nvrtc::toolkit_include(&why);
    if (cuda_inc.empty()) return sticky("CUDA toolkit headers for NVRTC", why);
    const std::string root(mathdx_root);
    if (!nvrtc::exists(root + "/include/cufftdx.hpp"))
        return sticky("GATE mathdx-headers", "got no cufftdx.hpp under " + root + "/include; want the "
                      "nvidia-mathdx wheel; fix: pip install nvidia-mathdx", ffi::ErrorCode::kFailedPrecondition);
    // EXPERIMENT (BSEC): LRX_DEC_EXP="<kp><xmid>" picks the decode's load hoisting and the Mid fusion.
    const char* ev = std::getenv("LRX_DEC_EXP");
    const int exp_kp = (ev && ev[0] == '1') ? 1 : 2, exp_xmid = (ev && ev[0] && ev[1] == '0') ? 0 : 1;
    nvrtc::Program prog;
    prog.src = kDecodeSrc;
    prog.name = "lrx_kconv_outer_decode.cu";
    prog.headers = {{kbox::kHeaderName, kbox::kHeaderSrc}};
    prog.defs = {
        "--std=c++17", "--device-as-default-execution-space", "--generate-line-info",
        "--gpu-architecture=sm_" + std::to_string(cc_major) + std::to_string(cc_minor),
        "-DLRX_NX=" + std::to_string(nkx), "-DLRX_NY=" + std::to_string(nky), "-DLRX_NZ=" + std::to_string(nkz),
        "-DLRX_K=" + std::to_string(K), "-DLRX_THREADS=" + std::to_string(kThreads),
        "-DLRX_MB=" + std::to_string(mb), "-DLRX_CONJ=" + std::to_string(conj_r),
        "-DLRX_KP=" + std::to_string(exp_kp), "-DLRX_XMID=" + std::to_string(exp_xmid),
        "-DLRX_SM=" + std::to_string(cc_major * 100 + cc_minor * 10)};
    nvrtc::mathdx_toolchain(root, cuda_inc, "cufftdx", &prog);
    prog.kernel = "lrx_kconv_outer_decode";
    std::string missing;
    const std::string key_hex = nvrtc::hex16(nvrtc::key(prog, &missing));
    const std::string dir(missing.empty() ? std::string(cubin_dir) : std::string());
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kconv_outer_dec_" << nkx << "x" << nky << "x" << nkz << "_K" << K << "_mb" << mb << "_c"
             << conj_r << "_x" << exp_kp << exp_xmid << "_sm" << cc_major << cc_minor << "_" << key_hex << ".cubin";
        path = name.str();
    }
    nvrtc::Image img;
    std::string where, err;
    if (!nvrtc::build(prog, dir, path, key_hex, &img, &where, &err)) return sticky(where.c_str(), err);
    Built b;
    b.fn = img.fn;
    b.smem = static_cast<int>(smem);
    b.minb = 1;
    b.sms = sms;
    cr = api.FuncSetAttribute(b.fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, b.smem);
    if (cr != CUDA_SUCCESS) return sticky("cuFuncSetAttribute", cu_err(cr));
    if (mklpin::announce_here() || mklpin::debug_print_here()) {
        std::fprintf(stderr, "[kconv_outer] decode %s kgrid=(%d,%d,%d) K=%d c-blocks=%d conj_r=%d sm_%d%d in %.1f ms "
                     "(8x8 tile, %d threads, 1 block/SM, smem=%d B, cubin %s)\n",
                     img.from_disk ? "disk-cache hit" : "NVRTC built", nkx, nky, nkz, K, mb, conj_r, cc_major,
                     cc_minor, img.ms, kThreads, b.smem,
                     path.empty() ? "not cached (no cubin_dir)" : (img.from_disk ? path.c_str() : "stored"));
    }
    *out = &(g_dcache[key] = b);
    return ffi::Error::Success();
}

// The phase count: the fewest (waves x (pairs per combo + 1)), a flush weighed as one pair, with
// the slots (one combo partial each) capped at 1/16 of U's bytes (one phase always allowed).
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

// Lr (na, nxb, nk, H, 8, 4), Rr (nb, nyb, nk, H, 8, 4), V (mx, my, nk), Pcr (na, nxb, nk, MB, 2, 8, 4)
// -> A (nk, nc, nb, my), complex128.  Scratch: the combo slots and one counter per item.
static ffi::Error KleadOuterDecode(cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer Lr,
                                   ffi::AnyBuffer Rr, ffi::AnyBuffer V, ffi::AnyBuffer Pcr,
                                   ffi::Result<ffi::AnyBuffer> A, int64_t nkx, int64_t nky, int64_t nkz,
                                   double scale, int64_t conj_r, std::string_view mathdx_root,
                                   std::string_view cubin_dir) {
    auto bad = [](const std::string& why) { return fail("klead outer decode", why, ffi::ErrorCode::kInvalidArgument); };
    if (nkx < 1 || nky < 1 || nkz < 1 || nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-axis: got k-grid (" << nkx << "," << nky << "," << nkz
           << "); want every axis in [1, " << kAxisMax << "] (the fp64 cuFFTDx thread-FFT limit)";
        return bad(os.str());
    }
    const auto C = ffi::DataType::C128;
    if (Lr.element_type() != C || Rr.element_type() != C || V.element_type() != C || Pcr.element_type() != C ||
        A->element_type() != C)
        return bad("operands must all be complex128");
    const int64_t nk = nkx * nky * nkz;
    auto ld = Lr.dimensions(), rd = Rr.dimensions(), vd = V.dimensions(), pd = Pcr.dimensions(), ad = A->dimensions();
    if (ld.size() != 6 || rd.size() != 6 || vd.size() != 3 || pd.size() != 7 || ad.size() != 4)
        return bad("want Lr (na,nxb,nk,H,8,4), Rr (nb,nyb,nk,H,8,4), V (mx,my,nk), Pcr (na,nxb,nk,MB,2,8,4), "
                   "A (nk,nc,nb,my)");
    const int64_t na = ld[0], nxb = ld[1], H = ld[3], nb = rd[0], nyb = rd[1], mx = vd[0], my = vd[1], mb = pd[3];
    const int64_t nc = ad[1], items = nb * nyb, pairs = na * nxb;
    if (ld[2] != nk || ld[4] != 8 || ld[5] != 4 || rd[2] != nk || rd[3] != H || rd[4] != 8 || rd[5] != 4 ||
        vd[2] != nk || pd[0] != na || pd[1] != nxb || pd[2] != nk || pd[4] != 2 || pd[5] != 8 || pd[6] != 4 ||
        nxb != (mx + 7) / 8 || nyb != (my + 7) / 8 || H < 1 || mb < 1 || nc < 1 || nc > 8 * mb ||
        ad[0] != nk || ad[2] != nb || ad[3] != my)
        return bad("operand shapes disagree (ffi.fft.make_local_kconv_klead_outer_decode builds them)");
    if (mx * my * nk >= (int64_t(1) << 31) || na * nxb * nk * mb * 64 >= (int64_t(1) << 31) ||
        nb * nyb * nk * H * 32 >= (int64_t(1) << 31) || nk * nc * nb * my >= (int64_t(1) << 31))
        return bad("an operand holds >= 2^31 elements (the load's 32-bit offsets); split the call");
    if (items * pairs == 0) return ffi::Error::Success();
    const Built* k = nullptr;
    if (ffi::Error e = build_decode(static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                                    static_cast<int>(4 * H), static_cast<int>(mb), conj_r ? 1 : 0, mathdx_root,
                                    cubin_dir, &k);
        !e.success())
        return e;
    const long long slot_bytes = nk * mb * 64 * 16;
    const long long ngrp = plan_phases(items, pairs, k->sms, slot_bytes, nk * na * mx * nb * my * 16);
    DecodeGeo g{};
    g.na = na; g.nxb = nxb; g.nb = nb; g.nyb = nyb; g.mx = mx; g.my = my; g.nc = nc;
    g.pairs = pairs; g.items = items;
    g.per = (pairs + ngrp - 1) / ngrp;
    g.combos = items * ngrp;
    g.scale = scale;
    if (g.combos >= (int64_t(1) << 31)) return bad("too many (phase, item) combos for 32-bit indices");
    auto sl = scratch.Allocate(static_cast<size_t>(g.combos * slot_bytes));
    auto cn = scratch.Allocate(static_cast<size_t>(items * sizeof(int)));
    if (!sl.has_value() || !cn.has_value()) return fail("klead outer decode", "scratch allocation (slots, counters)");
    if (cudaMemsetAsync(*cn, 0, static_cast<size_t>(items * sizeof(int)), stream) != cudaSuccess)
        return fail("klead outer decode", "cudaMemsetAsync(counters)");
    const void* lp = Lr.untyped_data();
    const void* rp = Rr.untyped_data();
    const void* vp = V.untyped_data();
    const void* cp = Pcr.untyped_data();
    void* ap = A->untyped_data();
    void* sp = *sl;
    void* np = *cn;
    void* args[] = {(void*)&lp, (void*)&rp, (void*)&vp, (void*)&cp, (void*)&ap, (void*)&sp, (void*)&np, (void*)&g};
    const long long grid = std::min<long long>(k->sms, g.combos);
    CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(grid), 1, 1, kThreads, 1, 1,
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
        .Arg<xla::ffi::AnyBuffer>()   // V (mx, my, nk), R space, k-minor
        .Ret<xla::ffi::AnyBuffer>()   // U (nk, na, mx, nb, my)
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("conj_r")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxKleadOuterDecodeCudaFfi, lorrax_ffi::kconv_outer::KleadOuterDecode,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()   // Lr (na, nxb, nk, H, 8, 4)
        .Arg<xla::ffi::AnyBuffer>()   // Rr (nb, nyb, nk, H, 8, 4)
        .Arg<xla::ffi::AnyBuffer>()   // V (mx, my, nk), R space, k-minor
        .Arg<xla::ffi::AnyBuffer>()   // Pcr (na, nxb, nk, MB, 2, 8, 4)
        .Ret<xla::ffi::AnyBuffer>()   // A (nk, nc, nb, my)
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("conj_r")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));
