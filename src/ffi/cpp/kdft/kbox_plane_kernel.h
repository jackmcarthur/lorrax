// kbox_plane_kernel.h -- DEVICE SOURCE, run-time compiled by the leg's RTC (common/rtc_build.h); CMake
// embeds it as text (kbox_plane_src.h.in) beside the k-box stage's text (cufft/kbox_stage_src.h.in),
// which hipRTC reads as the named header "kbox_stage.cuh".  CUDA/HIP common subset.
//
// Mode 6 of docs/architecture/kconv.md (ffi.fft.make_fused_conv_kplane, the ISDF route-G pair
// convolution read from the D-plane FFT output), the ROCM port of kconv_mathdx_cuda_ffi.cc's plane
// mode on the k-box stage's single arm:
//
//   P^X_ab[k, (m, g, p)] = conj(F[k,g,p] * D[k,g,a,X+m,b,p])        X = 0 (L) | C (R)
//   U[k, q] = s * FFT_k sum_ab i^(cl[a]+cr[b]) conj(IFFT_k P^L_ab) IFFT_k P^R_{pl[a],pr[b]},   q = (m, g, p)
//
// The phase and the L/R split are applied on the load, so the transposed, phased, split copy of D
// that the XLA route builds is never written.  A tile holds NO consecutive output columns q, each
// with its 2*NS*NS operand columns (bank row (o*2 + side)*SS + a*NS + b): one load, the stage's
// inverse transform3 on all rows, the phased pair product into rows [0, NO) (registers across one
// barrier), the forward transform3 on those NO rows, and the scaled store.  One HBM pass.
//
// Compile-time constants (keyed): LRX_NX, LRX_NY, LRX_NZ, LRX_NS, LRX_NO, LRX_NT, LRX_ZR
// (= ceil(NO * NK / NT), the product's registers per thread).
#include "kbox_stage.cuh"

struct alignas(16) lrx_c2 { double x, y; };
constexpr int NX = LRX_NX, NY = LRX_NY, NZ = LRX_NZ, NK = NX * NY * NZ, NS = LRX_NS, SS = NS * NS;
constexpr int NO = LRX_NO, TR = NO * 2 * SS, ZR = LRX_ZR;
using GK = lrx_kbox::Geo<NX, NY, NZ>;
using lrx_kbox::fft_direction;

// The host's PlaneGeo (kbox_plane_ffi.cc), field for field.
struct PlaneGeo {
    long long G, C, P, ncols;                  // D (nk, G, NS, 2C, NS, P), F (nk, G, P), U (nk, C, G*P)
    double scale;
    int pl[4], pr[4], cl[4], cr[4];            // perms and phase codes (0:+1, 1:+i, 2:-1, 3:-i)
};

__device__ __forceinline__ lrx_c2 lrx_mul(lrx_c2 a, lrx_c2 b) {
    return lrx_c2{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
__device__ __forceinline__ lrx_c2 lrx_iq(lrx_c2 v, int q) {      // v * i^q, exact
    q &= 3;
    return q == 0 ? v : q == 1 ? lrx_c2{-v.y, v.x} : q == 2 ? lrx_c2{-v.x, -v.y} : lrx_c2{v.y, -v.x};
}

extern "C" __global__ void __launch_bounds__(LRX_NT)
lrx_kbox_m6(const lrx_c2* __restrict__ D, const lrx_c2* __restrict__ F, lrx_c2* __restrict__ U, PlaneGeo g) {
    extern __shared__ lrx_c2 bank[];
    const long long GP = g.G * g.P;
    for (long long col0 = (long long)blockIdx.x * NO; col0 < g.ncols; col0 += (long long)gridDim.x * NO) {
        // Load: consecutive threads on consecutive output columns (consecutive p in D and F).
        for (int i = threadIdx.x; i < TR * NK; i += blockDim.x) {
            const int o = i % NO, rest = i / NO, r = rest % (2 * SS), k = rest / (2 * SS);
            const int side = r / SS, a = (r % SS) / NS, b = r % NS;
            const long long q = col0 + o;
            lrx_c2 v{0.0, 0.0};
            if (q < g.ncols) {
                const long long m = q / GP, gp = q - m * GP, gg = gp / g.P, p = gp - gg * g.P;
                const long long kg = (long long)k * g.G + gg;
                const lrx_c2 d = D[(((kg * NS + a) * 2 * g.C + side * g.C + m) * NS + b) * g.P + p];
                const lrx_c2 f = F[kg * g.P + p];
                const lrx_c2 x = lrx_mul(d, f);
                v = lrx_c2{x.x, -x.y};
            }
            bank[(o * 2 * SS + r) * GK::RS + GK::at(k)] = v;
        }
        __syncthreads();
        lrx_kbox::transform3<NX, NY, NZ, TR, 0, fft_direction::inverse>(bank);
        __syncthreads();
        // The phased pair product, a then b in order (the XLA route's sum order), into registers.
        lrx_c2 z[ZR];
#pragma unroll
        for (int e = 0; e < ZR; ++e) {
            const int i = threadIdx.x + e * LRX_NT;
            z[e] = lrx_c2{0.0, 0.0};
            if (i < NO * NK) {
                const int o = i % NO, k = i / NO;
                const lrx_c2* L = bank + (o * 2 * SS) * GK::RS + GK::at(k);
                const lrx_c2* R = L + SS * GK::RS;
#pragma unroll
                for (int a = 0; a < NS; ++a)
#pragma unroll
                    for (int b = 0; b < NS; ++b) {
                        const lrx_c2 l = L[(a * NS + b) * GK::RS];
                        const lrx_c2 wl = lrx_iq(lrx_c2{l.x, -l.y}, g.cl[a] + g.cr[b]);
                        const lrx_c2 t = lrx_mul(wl, R[(g.pl[a] * NS + g.pr[b]) * GK::RS]);
                        z[e].x += t.x;
                        z[e].y += t.y;
                    }
            }
        }
        __syncthreads();
#pragma unroll
        for (int e = 0; e < ZR; ++e) {
            const int i = threadIdx.x + e * LRX_NT;
            if (i < NO * NK) bank[(i % NO) * GK::RS + GK::at(i / NO)] = z[e];
        }
        __syncthreads();
        lrx_kbox::transform3<NX, NY, NZ, NO, 0, fft_direction::forward>(bank);
        __syncthreads();
        for (int i = threadIdx.x; i < NO * NK; i += blockDim.x) {
            const int o = i % NO, k = i / NO;
            const long long q = col0 + o;
            if (q < g.ncols) {
                const lrx_c2 v = bank[o * GK::RS + GK::at(k)];
                U[(long long)k * g.ncols + q] = lrx_c2{v.x * g.scale, v.y * g.scale};
            }
        }
        __syncthreads();
    }
}
