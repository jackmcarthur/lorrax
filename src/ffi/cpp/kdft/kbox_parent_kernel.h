// kbox_parent_kernel.h -- DEVICE SOURCE, run-time compiled by the leg's RTC (common/rtc_build.h); CMake
// embeds it as text (kbox_parent_src.h.in) beside the k-box stage's text (cufft/kbox_stage_src.h.in),
// which hipRTC reads as the named header "kbox_stage.cuh".  CUDA/HIP common subset.
//
// Mode 1 of docs/architecture/kconv.md (ffi.fft.make_fused_conv_kparent, the zeta-fit pair Gram C_q of
// isdf.core: zeta_fit.CCT), the ROCM port of kconv_mathdx_cuda_ffi.cc's parent mode on the k-box
// stage's single arm, in mode 6's tile form (kbox_plane_kernel.h):
//
//   V^X_ce[k, m, n] = T_k( e^{2 pi i q_p.L[o,m]} D^X[p, c, left[o,m], e, right[o,n]] e^{-2 pi i q_p.R[o,n]} )
//   P^X_ab[k, (m, n)] = conj( sum_ce coef_X[k, a ns + b, c ns + e] V^X_ce[k, m, n] )     X = L | R
//   U[k, q] = s * FFT_k sum_ab i^(cl[a]+cr[b]) conj(IFFT_k P^L_ab) IFFT_k P^R_{pl[a],pr[b]},  q = m nu + n
//
// with p = irr[k], o = sym[k] and T_k complex conjugation on an antiunitary row (trs[k] != 0).  The
// typed parent unfold is applied on the load, so the two full-k open-spin copies (nk, ns, mu, nu, ns)
// that the XLA route writes are never formed.  One thread loads one (output column, k): the two
// phases once, the ns^2 sources of each side once, and all ns^2 projector entries of each side from
// them (the CUDA kernel reads every source once per entry; same products, same c, e order).  A tile
// holds NO consecutive output columns q, each with its 2*NS*NS operand columns (bank row
// (o*2 + side)*SS + a*NS + b): one load, the stage's inverse transform3 on all rows, the phased pair
// product into rows [0, NO), the forward transform3 on those NO rows, the scaled store.  One HBM pass.
//
// Compile-time constants (keyed): LRX_NX, LRX_NY, LRX_NZ, LRX_NS, LRX_NO, LRX_NT, LRX_ZR
// (= ceil(NO * NK / NT)), LRX_CM (1: D is centroid-major, physical (p, nu, e, mu, c)).
#include "kbox_stage.cuh"

struct alignas(16) lrx_c2 { double x, y; };
constexpr int NX = LRX_NX, NY = LRX_NY, NZ = LRX_NZ, NK = NX * NY * NZ, NS = LRX_NS, SS = NS * NS;
constexpr int NO = LRX_NO, TR = NO * 2 * SS, ZR = LRX_ZR;
using GK = lrx_kbox::Geo<NX, NY, NZ>;
using lrx_kbox::fft_direction;

// The host's ParentGeo (kbox_parent_ffi.cc), field for field.
struct ParentGeo {
    const int *irr, *sym, *left, *right, *trs;           // (nk) (nk) (ops, mu) (ops, nu) (nk)
    const double *L, *R, *q;                              // (ops, mu, 3) (ops, nu, 3) (n_parent, 3)
    const lrx_c2 *coef_l, *coef_r;                        // (nk, ns^2, ns^2)
    long long mu, nu, ncols;                              // D (n_parent, ns, mu, ns, nu); U (nk, mu, nu)
    double scale;
    int pl[4], pr[4], cl[4], cr[4];                       // perms and phase codes (0:+1, 1:+i, 2:-1, 3:-i)
};

__device__ __forceinline__ lrx_c2 lrx_mul(lrx_c2 a, lrx_c2 b) {
    return lrx_c2{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
__device__ __forceinline__ lrx_c2 lrx_iq(lrx_c2 v, int q) {      // v * i^q, exact
    q &= 3;
    return q == 0 ? v : q == 1 ? lrx_c2{-v.y, v.x} : q == 2 ? lrx_c2{-v.x, -v.y} : lrx_c2{v.y, -v.x};
}

// Element (p, c, lm, e, rn) of a parent projector in its physical layout.
__device__ __forceinline__ long long lrx_d_at(const ParentGeo& g, long long p, int c, long long lm, int e,
                                              long long rn) {
#if LRX_CM
    return ((((p * g.nu + rn) * NS + e) * g.mu + lm) * NS) + c;
#else
    return (((p * NS + c) * g.mu + lm) * NS + e) * g.nu + rn;
#endif
}

extern "C" __global__ void __launch_bounds__(LRX_NT)
lrx_kbox_m1(const lrx_c2* __restrict__ Dl, const lrx_c2* __restrict__ Dr, lrx_c2* __restrict__ U, ParentGeo g) {
    extern __shared__ lrx_c2 bank[];
    for (long long col0 = (long long)blockIdx.x * NO; col0 < g.ncols; col0 += (long long)gridDim.x * NO) {
        // Load: consecutive threads on consecutive output columns (consecutive n), then k.
        for (int i = threadIdx.x; i < NO * NK; i += blockDim.x) {
            const int o = i % NO, k = i / NO;
            const long long q = col0 + o;
            lrx_c2* row0 = bank + (o * 2 * SS) * GK::RS + GK::at(k);
            if (q >= g.ncols) {
#pragma unroll
                for (int r = 0; r < 2 * SS; ++r) row0[r * GK::RS] = lrx_c2{0.0, 0.0};
                continue;
            }
            const long long m = q / g.nu, n = q - m * g.nu;
            const int p = g.irr[k], op = g.sym[k];
            const long long lm = g.left[op * g.mu + m], rn = g.right[op * g.nu + n];
            double dl = 0.0, dr = 0.0;
#pragma unroll
            for (int a = 0; a < 3; ++a) {
                dl += g.q[3 * p + a] * g.L[3 * (op * g.mu + m) + a];
                dr += g.q[3 * p + a] * g.R[3 * (op * g.nu + n) + a];
            }
            lrx_c2 phl, phr;
            sincospi(2.0 * dl, &phl.y, &phl.x);
            sincospi(-2.0 * dr, &phr.y, &phr.x);
            const bool anti = g.trs[k] != 0;
#pragma unroll
            for (int side = 0; side < 2; ++side) {
                const lrx_c2* __restrict__ D = side ? Dr : Dl;
                const lrx_c2* __restrict__ cf = (side ? g.coef_r : g.coef_l) + (long long)k * SS * SS;
                lrx_c2 v[SS];
#pragma unroll
                for (int c = 0; c < NS; ++c)
#pragma unroll
                    for (int e = 0; e < NS; ++e) {
                        lrx_c2 t = lrx_mul(lrx_mul(phl, D[lrx_d_at(g, p, c, lm, e, rn)]), phr);
                        if (anti) t.y = -t.y;
                        v[c * NS + e] = t;
                    }
#pragma unroll
                for (int ab = 0; ab < SS; ++ab) {
                    lrx_c2 s{0.0, 0.0};
#pragma unroll
                    for (int ce = 0; ce < SS; ++ce) {
                        const lrx_c2 t = lrx_mul(cf[ab * SS + ce], v[ce]);
                        s.x += t.x;
                        s.y += t.y;
                    }
                    row0[(side * SS + ab) * GK::RS] = lrx_c2{s.x, -s.y};
                }
            }
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
                const lrx_c2* Lb = bank + (o * 2 * SS) * GK::RS + GK::at(k);
                const lrx_c2* Rb = Lb + SS * GK::RS;
#pragma unroll
                for (int a = 0; a < NS; ++a)
#pragma unroll
                    for (int b = 0; b < NS; ++b) {
                        const lrx_c2 l = Lb[(a * NS + b) * GK::RS];
                        const lrx_c2 wl = lrx_iq(lrx_c2{l.x, -l.y}, g.cl[a] + g.cr[b]);
                        const lrx_c2 t = lrx_mul(wl, Rb[(g.pl[a] * NS + g.pr[b]) * GK::RS]);
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
