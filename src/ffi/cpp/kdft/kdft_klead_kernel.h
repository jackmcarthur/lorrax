// kdft_klead_kernel.h -- DEVICE SOURCE, run-time compiled (common/rtc_build.h); the host
// compiler never sees it.  CMake embeds it, with the DFT core (kdft_core.h, kdft_plan.h), as
// text (kdft_klead_src.h.in), as the CUDA leg embeds cufft/kbox_stage.cuh.  CUDA/HIP common
// subset, so the same text compiles under hipRTC and NVRTC.
//
// One PASS of the k-leading pair, modes 2 and 3 of docs/architecture/kconv.md, on the flat
// (nk, ncols) operand, nk = NX*NY*NZ, k = (ix*NY + iy)*NZ + iz.  The host (kdft_klead_ffi.cc)
// splits the k axes into contiguous GROUPS; a pass transforms one group, viewing the operand as
// (O, G, I): O = the product of the axes before the group, G = the group's points, I = the
// product of the axes after it times ncols (contiguous).  A block owns W consecutive inner
// indices of one o: it loads the (G, W) tile into LDS once (rows of W contiguous elements),
// transforms the group's axes there (kdft::dft3), and stores it once.
//
//   LRX_MODE 3   one transform pass: Y = scale * (forward ? DFT : IDFT) over the group
//   LRX_MODE 2   the convolution pass of the TRAILING group (I = ncols):
//                IDFT, times scale * V_R[k, v(c)], DFT;  v(c) = (imx, imy) of column c
//   LRX_MODE 4   a k-MINOR convolution column pass (kdft_kminor_ffi.cc split arm): the operand is
//                (rows, nk) with k trailing, viewed as (O = rows*kob, G, I = the k axes after the
//                group); IDFT, times scale * K[krow(r), k], DFT, with r = o / kob the row,
//                k = ((o % kob)*G + g)*I + i, krow(r) = (r / m1) % m0
//
// A grid whose whole k tile fits is one mode-2 pass (one HBM read and write of T); otherwise
// mode 2 is IDFT passes over the leading groups, the trailing group's mode-2 pass, and DFT
// passes back.  Compile-time constants (all in the image key):
//   LRX_GX, LRX_GY, LRX_GZ   the group's axes (1 for absent)
//   LRX_W                    inner indices per block (the LDS tile is G*W complex)
//   LRX_NT                   threads per block
//   LRX_MODE                 2 | 3 | 4
//   LRX_REG                  0: the LDS pass (dft3: Stockham stages in LDS)
//                            1: the REGISTER pass: a thread holds whole lines of one group axis
//                               in VGPRs (kdft::line_reg), loaded from / stored to HBM directly;
//                               LDS only for the exchange between axes, full complex
//                               (LRX_XH 0) or real then imaginary halves (LRX_XH 1, half the
//                               footprint, so more blocks per CU).  Every group axis reg_ok().
//   LRX_F32                  0 (complex128) | 1 (complex64)
#include "kdft_core.h"

#ifndef LRX_REG
#define LRX_REG 0
#endif
#ifndef LRX_XH
#define LRX_XH 0
#endif
#define LRX_G (LRX_GX * LRX_GY * LRX_GZ)
#define LRX_TILE (LRX_G * LRX_W)

// The convolution factor of element (o, g) of column col: scale * V_R (mode 2) or scale * K
// (mode 4) -- the one place the two convolution passes differ.
__device__ __forceinline__ cplx conv_factor(const cplx* __restrict__ v, long long o, int g, long long col,
                                            long long inner, int mx, int nb, int my, long long kob,
                                            int m0, int m1, real scale) {
#if LRX_MODE == 4
    (void)mx; (void)nb; (void)my;
    const long long r = o / kob;
    const long long nk = kob * LRX_G * inner;
    const long long k = ((o - r * kob) * LRX_G + g) * inner + col;
    const cplx w = v[((r / m1) % m0) * nk + k];
#else
    (void)inner; (void)kob; (void)m0; (void)m1;
    const long long k = o * LRX_G + g;
    const long long q = col / my;
    const int imy = static_cast<int>(col - q * my);
    const int imx = static_cast<int>((q / nb) % mx);
    const cplx w = v[(k * mx + imx) * my + imy];
#endif
    return cplx{scale * w.x, scale * w.y};
}
__device__ __forceinline__ cplx cmul_(cplx a, cplx w) {
    return cplx{a.x * w.x - a.y * w.y, a.x * w.y + a.y * w.x};
}

#if LRX_REG
// ---- the register pass ---------------------------------------------------------------------
// Axis (N, S) of the group: S = the product of the group axes after it (its stride in g).  Its
// lines are l in [0, L), L = (G/N)*W: column w = l % W, lo = l / W, first point
// base = (lo/S)*S*N + lo%S, point j at g = base + j*S.  Thread t holds lines t + u*NT, u < LPT.
template <int N, int S>
struct RAx {
    static constexpr int L = (LRX_G / N) * LRX_W;
    static constexpr int LPT = (L + LRX_NT - 1) / LRX_NT;
    __device__ __forceinline__ static int line(int u) { return threadIdx.x + u * LRX_NT; }
    __device__ __forceinline__ static int gbase(int l) { const int lo = l / LRX_W; return (lo / S) * (S * N) + lo % S; }
    __device__ __forceinline__ static void load(cplx (*d)[N], const cplx* base, long long inner, long long i0) {
#pragma unroll
        for (int u = 0; u < LPT; ++u) {
            const int l = line(u);
            if (L % LRX_NT == 0 || l < L) {
                const int w = l % LRX_W, g0 = gbase(l);
                const bool live = i0 + w < inner;
#pragma unroll
                for (int j = 0; j < N; ++j)
                    d[u][j] = live ? base[static_cast<long long>(g0 + j * S) * inner + w] : cplx{0, 0};
            }
        }
    }
    __device__ __forceinline__ static void store(cplx (*d)[N], cplx* base, long long inner, long long i0, real sc) {
#pragma unroll
        for (int u = 0; u < LPT; ++u) {
            const int l = line(u);
            if (L % LRX_NT == 0 || l < L) {
                const int w = l % LRX_W, g0 = gbase(l);
                if (i0 + w < inner) {
#pragma unroll
                    for (int j = 0; j < N; ++j)
                        base[static_cast<long long>(g0 + j * S) * inner + w] = cplx{sc * d[u][j].x, sc * d[u][j].y};
                }
            }
        }
    }
    __device__ __forceinline__ static void dft(cplx (*d)[N], real sgn) {
#pragma unroll
        for (int u = 0; u < LPT; ++u)
            if (L % LRX_NT == 0 || line(u) < L) kdft::line_reg<N>(d[u], sgn);
    }
    // part: 0 = .x, 1 = .y, 2 = both (ex is cplx* when 2, real* otherwise)
    template <int PART>
    __device__ __forceinline__ static void put(cplx (*d)[N], void* ex) {
#pragma unroll
        for (int u = 0; u < LPT; ++u) {
            const int l = line(u);
            if (L % LRX_NT == 0 || l < L) {
                const int w = l % LRX_W, g0 = gbase(l);
#pragma unroll
                for (int j = 0; j < N; ++j) {
                    const int e = (g0 + j * S) * LRX_W + w;
                    if constexpr (PART == 2) static_cast<cplx*>(ex)[e] = d[u][j];
                    else static_cast<real*>(ex)[e] = PART == 0 ? d[u][j].x : d[u][j].y;
                }
            }
        }
    }
    template <int PART>
    __device__ __forceinline__ static void get(cplx (*d)[N], const void* ex) {
#pragma unroll
        for (int u = 0; u < LPT; ++u) {
            const int l = line(u);
            if (L % LRX_NT == 0 || l < L) {
                const int w = l % LRX_W, g0 = gbase(l);
#pragma unroll
                for (int j = 0; j < N; ++j) {
                    const int e = (g0 + j * S) * LRX_W + w;
                    if constexpr (PART == 2) d[u][j] = static_cast<const cplx*>(ex)[e];
                    else if constexpr (PART == 0) d[u][j].x = static_cast<const real*>(ex)[e];
                    else d[u][j].y = static_cast<const real*>(ex)[e];
                }
            }
        }
    }
    // times the convolution factor (modes 2/4), in registers: per line one base pointer into V
    // (or K) and one stride, since point j of a line is g0 + j*S (conv_factor's map, unrolled)
    __device__ __forceinline__ static void conv(cplx (*d)[N], const cplx* v, long long o, long long i0,
                                                long long inner, int mx, int nb, int my, long long kob,
                                                int m0, int m1, real scale) {
#pragma unroll
        for (int u = 0; u < LPT; ++u) {
            const int l = line(u);
            if (L % LRX_NT == 0 || l < L) {
                const int w = l % LRX_W, g0 = gbase(l);
                const long long col = i0 + w;
                if (col < inner) {
#if LRX_MODE == 4
                    (void)mx; (void)nb; (void)my;
                    const long long r = o / kob;
                    const long long nk = kob * LRX_G * inner;
                    const cplx* p = v + ((r / m1) % m0) * nk + ((o - r * kob) * LRX_G + g0) * inner + col;
                    const long long js = static_cast<long long>(S) * inner;
#else
                    (void)kob; (void)m0; (void)m1;
                    const long long q = col / my;
                    const int imy = static_cast<int>(col - q * my);
                    const int imx = static_cast<int>((q / nb) % mx);
                    const cplx* p = v + ((o * LRX_G + g0) * mx + imx) * my + imy;
                    const long long js = static_cast<long long>(S) * mx * my;
#endif
#pragma unroll
                    for (int j = 0; j < N; ++j) {
                        const cplx f = p[j * js];
                        d[u][j] = cmul_(d[u][j], cplx{scale * f.x, scale * f.y});
                    }
                }
            }
        }
    }
};
// Lines of axis A (registers) -> lines of axis B, through the LDS exchange buffer; the trailing
// barrier frees the buffer for the next exchange.
template <class A, class B, int NA_, int NB_>
__device__ __forceinline__ void exchange(cplx (*da)[NA_], cplx (*db)[NB_], void* ex) {
#if LRX_XH
    A::template put<0>(da, ex); __syncthreads();
    B::template get<0>(db, ex); __syncthreads();
    A::template put<1>(da, ex); __syncthreads();
    B::template get<1>(db, ex); __syncthreads();
#else
    A::template put<2>(da, ex); __syncthreads();
    B::template get<2>(db, ex); __syncthreads();
#endif
}

// The group's non-unit axes in order, outer to inner (unit axes dropped; G = 1 is one axis of 1).
#define LRX_NA ((LRX_GX > 1) + (LRX_GY > 1) + (LRX_GZ > 1))
#if LRX_NA == 0
#define AX0_N 1
#define AX0_S 1
#else
#define AX0_N (LRX_GX > 1 ? LRX_GX : LRX_GY > 1 ? LRX_GY : LRX_GZ)
#define AX0_S (LRX_GX > 1 ? LRX_GY * LRX_GZ : LRX_GY > 1 ? LRX_GZ : 1)
#endif
#if LRX_NA >= 2
#define AX1_N (LRX_GX > 1 && LRX_GY > 1 ? LRX_GY : LRX_GZ)
#define AX1_S (LRX_GX > 1 && LRX_GY > 1 ? LRX_GZ : 1)
#endif
#if LRX_NA == 3
#define AX2_N LRX_GZ
#define AX2_S 1
#endif
#endif  // LRX_REG

// t: the operand, in place.  v: V_R (nk, mx, my) (mode 2) or K (m0, nk) (mode 4).  ntile:
// ceil(I / W) blocks per o.  nb: T's b extent (columns are ((ia*mx + imx)*nb + ib)*my + imy).
// forward: mode 3's direction.  kob, m0, m1: mode 4's row map (above).
extern "C" __global__ void __launch_bounds__(LRX_NT)
lrx_kdft_pass(cplx* __restrict__ t, const cplx* __restrict__ v, long long inner, long long ntile,
              int mx, int nb, int my, real scale, int forward, long long kob, int m0, int m1) {
    const long long o = static_cast<long long>(blockIdx.x) / ntile;
    const long long i0 = (static_cast<long long>(blockIdx.x) - o * ntile) * LRX_W;
    cplx* const base = t + o * LRX_G * inner + i0;            // element (o, g, i0 + w) at g*inner + w

#if LRX_REG
#if LRX_NA >= 2
#if LRX_XH
    __shared__ real ex[LRX_TILE];
#else
    __shared__ cplx ex[LRX_TILE];
#endif
#endif
    typedef RAx<AX0_N, AX0_S> A0;
    cplx d0[A0::LPT][AX0_N];
#if LRX_MODE == 3
    const real sg = forward ? real(-1) : real(1);
#else
    const real sg = real(1);                                   // the inverse first
#endif
#if LRX_NA == 3
    typedef RAx<AX1_N, AX1_S> A1;
    typedef RAx<AX2_N, AX2_S> A2;
    cplx d1[A1::LPT][AX1_N], d2[A2::LPT][AX2_N];
    A2::load(d2, base, inner, i0);
    A2::dft(d2, sg);
    exchange<A2, A1>(d2, d1, ex);
    A1::dft(d1, sg);
    exchange<A1, A0>(d1, d0, ex);
    A0::dft(d0, sg);
#if LRX_MODE == 3
    (void)v; (void)mx; (void)nb; (void)my; (void)kob; (void)m0; (void)m1;
    A0::store(d0, base, inner, i0, scale);
#else
    (void)forward;
    A0::conv(d0, v, o, i0, inner, mx, nb, my, kob, m0, m1, scale);
    A0::dft(d0, real(-1));
    exchange<A0, A1>(d0, d1, ex);
    A1::dft(d1, real(-1));
    exchange<A1, A2>(d1, d2, ex);
    A2::dft(d2, real(-1));
    A2::store(d2, base, inner, i0, real(1));
#endif
#elif LRX_NA == 2
    typedef RAx<AX1_N, AX1_S> A1;
    cplx d1[A1::LPT][AX1_N];
    A1::load(d1, base, inner, i0);
    A1::dft(d1, sg);
    exchange<A1, A0>(d1, d0, ex);
    A0::dft(d0, sg);
#if LRX_MODE == 3
    (void)v; (void)mx; (void)nb; (void)my; (void)kob; (void)m0; (void)m1;
    A0::store(d0, base, inner, i0, scale);
#else
    (void)forward;
    A0::conv(d0, v, o, i0, inner, mx, nb, my, kob, m0, m1, scale);
    A0::dft(d0, real(-1));
    exchange<A0, A1>(d0, d1, ex);
    A1::dft(d1, real(-1));
    A1::store(d1, base, inner, i0, real(1));
#endif
#else   // one axis (or none): no exchange
    A0::load(d0, base, inner, i0);
    A0::dft(d0, sg);
#if LRX_MODE == 3
    (void)v; (void)mx; (void)nb; (void)my; (void)kob; (void)m0; (void)m1;
    A0::store(d0, base, inner, i0, scale);
#else
    (void)forward;
    A0::conv(d0, v, o, i0, inner, mx, nb, my, kob, m0, m1, scale);
    A0::dft(d0, real(-1));
    A0::store(d0, base, inner, i0, real(1));
#endif
#endif

#else   // ---- the LDS pass ----------------------------------------------------------------------
    __shared__ cplx tw[LRX_GX + LRX_GY + LRX_GZ];
    kdft::fill_twiddles<LRX_GX, LRX_GY, LRX_GZ, LRX_NT>(tw);
    __shared__ cplx s[LRX_TILE];
    for (int e = threadIdx.x; e < LRX_TILE; e += LRX_NT) {
        const int g = e / LRX_W, w = e % LRX_W;
        s[e] = i0 + w < inner ? base[static_cast<long long>(g) * inner + w] : cplx{0, 0};
    }
    __syncthreads();

#if LRX_MODE == 2 || LRX_MODE == 4
    (void)forward;
    kdft::dft3<LRX_GX, LRX_GY, LRX_GZ, LRX_W, LRX_NT>(s, tw, real(1));
    for (int e = threadIdx.x; e < LRX_TILE; e += LRX_NT) {
        const int g = e / LRX_W, w = e % LRX_W;
        const long long col = i0 + w;
        if (col < inner) s[e] = cmul_(s[e], conv_factor(v, o, g, col, inner, mx, nb, my, kob, m0, m1, scale));
    }
    __syncthreads();
    kdft::dft3<LRX_GX, LRX_GY, LRX_GZ, LRX_W, LRX_NT>(s, tw, real(-1));
    const real out_scale = real(1);
#else
    (void)v; (void)mx; (void)nb; (void)my; (void)kob; (void)m0; (void)m1;
    kdft::dft3<LRX_GX, LRX_GY, LRX_GZ, LRX_W, LRX_NT>(s, tw, forward ? real(-1) : real(1));
    const real out_scale = scale;
#endif

    for (int e = threadIdx.x; e < LRX_TILE; e += LRX_NT) {
        const int g = e / LRX_W, w = e % LRX_W;
        if (i0 + w < inner)
            base[static_cast<long long>(g) * inner + w] = cplx{out_scale * s[e].x, out_scale * s[e].y};
    }
#endif
}
