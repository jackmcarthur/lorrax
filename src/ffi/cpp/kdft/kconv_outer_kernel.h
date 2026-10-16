// kconv_outer_kernel.h -- DEVICE SOURCE, run-time compiled by the leg's RTC (common/rtc_build.h);
// the host compiler never sees it.  CMake embeds it as text (kconv_outer_src.h.in) beside the k-box
// stage's text (cufft/kbox_stage_src.h.in), which hipRTC reads as the named header "kbox_stage.cuh".
// The CUDA/HIP common subset: no vendor header or intrinsic here; the stage spells the leg's line
// transform (LRX_KBOX_DIRECT_DFT on hipRTC: the kdft direct DFT).
//
// The ROCM port of cufft/kconv_outer_cuda_ffi.cc (the BSE W term's outer-product load), its two
// kernels on the k-box stage's transform3:
//
//   lrx_kconv_outer          U[k,a,x,b,y] = s * FFT_k( IFFT_k T[.,a,x,b,y] * V[x,y,k] )
//   lrx_kconv_outer_decode   A[k,c,b,y]   = s * sum_{a,x} conj(Pc[k,c,a,x]) U[k,a,x,b,y]   (U never stored)
//
//   T[k,a,x,b,y] = sum_K L[k,a,x,K] R'[k,K,b,y],   R' = conj(R) when LRX_CONJ   (formed on the load)
//
// The CUDA kernel's design, item for item: a tile is an XB x YB box of (x, y) at one (a, b); a work
// item is (y block, b, group of (x block, a) pairs), a fastest inside an item and b fastest across
// items, so the na x nb visits of one V tile are adjacent in time; each pair is load (T formed in
// LDS from the two L2-resident legs, never stored), inverse transform3, mode 2's Mid with V k-MINOR
// (the W_R tile as the BSE builds it), forward transform3, and the scaled store (U) or the decode's
// (t, mu) contraction into per-thread accumulators (A).  The decode's (phase, item) combos, slots
// and the item counter are the CUDA decode's: the combo completing an item sums the item's slots in
// phase order (deterministic, no atomics on data).
// What differs, and why (gfx90a): the K sum and the decode run on the FP64 FMA pipe (the CUDA
// LRX_FMA arm's per-accumulator order: re += Lr Rr - Li Ri, im += Lr Ri + Li Rr, K in order), not
// on fp64 mma.m8n8k4, which has no MI250X spelling of the same fragment contract; the tile is
// XB x YB <= 8 x 8, sized by the host to a block's 64 KiB of LDS (an 8 x 8 bank at 8x8x1 is 66.5 KB);
// one block-wide group (no named barriers on gfx90a), so no ping-pong.
//
// Compile-time constants (keyed options): LRX_NX, LRX_NY, LRX_NZ, LRX_K, LRX_XB, LRX_YB, LRX_NT,
// LRX_CONJ, LRX_DECODE (0: U, 1: A), and for the decode LRX_NC (bands c) and LRX_E (accumulators
// per thread, ceil(NK * NC * YB / NT)).

#include "kbox_stage.cuh"

struct alignas(16) lrx_c2 { double x, y; };

constexpr int NX = LRX_NX, NY = LRX_NY, NZ = LRX_NZ, NK = NX * NY * NZ;
constexpr int KK = LRX_K, XB = LRX_XB, YB = LRX_YB, TR = XB * YB;
using GK = lrx_kbox::Geo<NX, NY, NZ>;
using lrx_kbox::fft_direction;

// The host's OuterGeo (kconv_outer_rocm_ffi.cc), field for field.
struct OuterGeo {
    long long na, mx, nb, my;              // U (nk, na, mx, nb, my)
    long long nxb, nyb;                    // x blocks, y blocks (of XB, YB)
    long long ngrp, per;                   // phases per item, pairs per phase
    long long items, combos;               // nyb * nb items; items * ngrp combos
    long long nc;                          // the decode's band count (A (nk, nc, nb, my))
    double scale;
};

// The family's kernel multiply (kconv_mathdx_cuda_ffi.cc lrx_mul), spelled identically.
__device__ __forceinline__ lrx_c2 lrx_mul(lrx_c2 a, lrx_c2 b) {
    return lrx_c2{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

// Load: bank[j, k] = T[k, a, x0 + xi, b, y0 + yi] (j = xi YB + yi), zero past mx, my.  Consecutive
// threads take consecutive y (R reads contiguous, the YB threads of one x share the L read).
__device__ __forceinline__ void load_t(lrx_c2* bank, const lrx_c2* __restrict__ L, const lrx_c2* __restrict__ R,
                                       const OuterGeo& g, int a, int b, int x0, int y0, int xlim, int ylim) {
    const long long rk = g.nb * g.my;                              // R stride between K rows
    for (int i = threadIdx.x; i < TR * NK; i += blockDim.x) {
        const int j = i % TR, k = i / TR, xi = j / YB, yi = j % YB;
        double re = 0.0, im = 0.0;
        if (xi < xlim && yi < ylim) {
            const lrx_c2* lp = L + (((long long)k * g.na + a) * g.mx + x0 + xi) * KK;
            const lrx_c2* rp = R + (long long)k * KK * rk + (long long)b * g.my + y0 + yi;
#pragma unroll 4
            for (int q = 0; q < KK; ++q) {
                const lrx_c2 lv = lp[q], rv = rp[q * rk];
#if LRX_CONJ
                re = fma(lv.x, rv.x, re); re = fma(lv.y, rv.y, re);
                im = fma(lv.y, rv.x, im); im = fma(-lv.x, rv.y, im);
#else
                re = fma(lv.x, rv.x, re); re = fma(-lv.y, rv.y, re);
                im = fma(lv.x, rv.y, im); im = fma(lv.y, rv.x, im);
#endif
            }
        }
        bank[j * GK::RS + GK::at(k)] = lrx_c2{re, im};
    }
    __syncthreads();
}

// Mid: mode 2's KernMid product with V[x, y, k] (R space, k-MINOR), k fastest across threads.
__device__ __forceinline__ void mid_v(lrx_c2* bank, const lrx_c2* __restrict__ V, const OuterGeo& g,
                                      int x0, int y0, int xlim, int ylim) {
    const lrx_c2* vb = V + ((long long)x0 * g.my + y0) * NK;
    for (int i = threadIdx.x; i < TR * NK; i += blockDim.x) {
        const int k = i % NK, j = i / NK, xi = j / YB, yi = j % YB;
        if (xi < xlim && yi < ylim) {
            lrx_c2* e = bank + j * GK::RS + GK::at(k);
            *e = lrx_mul(*e, vb[((long long)xi * g.my + yi) * NK + k]);
        }
    }
    __syncthreads();
}

// One pair's convolution in the bank: load, inverse transform3, Mid, forward transform3.
__device__ __forceinline__ void conv_pair(lrx_c2* bank, const lrx_c2* __restrict__ L, const lrx_c2* __restrict__ R,
                                          const lrx_c2* __restrict__ V, const OuterGeo& g, int a, int b, int x0,
                                          int y0, int xlim, int ylim) {
    load_t(bank, L, R, g, a, b, x0, y0, xlim, ylim);
    lrx_kbox::transform3<NX, NY, NZ, TR, 0, fft_direction::inverse>(bank);
    __syncthreads();                                               // a grid of one point runs no pass
    mid_v(bank, V, g, x0, y0, xlim, ylim);
    lrx_kbox::transform3<NX, NY, NZ, TR, 0, fft_direction::forward>(bank);
    __syncthreads();
}

#if !LRX_DECODE
extern "C" __global__ void __launch_bounds__(LRX_NT) lrx_kconv_outer(
    const lrx_c2* __restrict__ L, const lrx_c2* __restrict__ R, const lrx_c2* __restrict__ V,
    lrx_c2* __restrict__ U, OuterGeo g) {
    extern __shared__ lrx_c2 bank[];
    const long long pairs = g.na * g.nxb, ucols = g.na * g.mx * g.nb * g.my;
    for (long long it = blockIdx.x; it < g.items; it += gridDim.x) {
        const long long grp = it % g.ngrp, yb_ = it / g.ngrp;
        const int b = (int)(yb_ % g.nb), y0 = (int)(yb_ / g.nb) * YB;
        const int ylim = min(YB, (int)g.my - y0);
        const long long p0 = grp * g.per, p1 = min(pairs, p0 + g.per);
        for (long long p = p0; p < p1; ++p) {
            const int a = (int)(p % g.na), x0 = (int)(p / g.na) * XB;
            const int xlim = min(XB, (int)g.mx - x0);
            conv_pair(bank, L, R, V, g, a, b, x0, y0, xlim, ylim);
            // Store: mode 2's RowStore (v * scale), U k-leading (nk, na, mx, nb, my), y fastest.
            lrx_c2* ub = U + (((long long)a * g.mx + x0) * g.nb + b) * g.my + y0;
            for (int i = threadIdx.x; i < TR * NK; i += blockDim.x) {
                const int j = i % TR, k = i / TR, xi = j / YB, yi = j % YB;
                if (xi < xlim && yi < ylim) {
                    const lrx_c2 v = bank[j * GK::RS + GK::at(k)];
                    ub[k * ucols + (long long)xi * g.nb * g.my + yi] = lrx_c2{v.x * g.scale, v.y * g.scale};
                }
            }
            __syncthreads();
        }
    }
}
#else
constexpr int NC = LRX_NC, E = LRX_E;
constexpr long long SLOT = (long long)NK * NC * YB;               // one combo's partial, complex elements

// Pc (nk, nc, na, mx) as the BSE holds psi^X_c; slots (combos, NK, NC, YB); count (items) zeroed.
extern "C" __global__ void __launch_bounds__(LRX_NT) lrx_kconv_outer_decode(
    const lrx_c2* __restrict__ L, const lrx_c2* __restrict__ R, const lrx_c2* __restrict__ V,
    const lrx_c2* __restrict__ Pc, lrx_c2* __restrict__ A, lrx_c2* __restrict__ slots, int* __restrict__ count,
    OuterGeo g) {
    extern __shared__ lrx_c2 bank[];
    __shared__ int last;
    const int items = (int)g.items, nb = (int)g.nb, na = (int)g.na;
    const long long pairs = g.na * g.nxb;
    for (long long cb = blockIdx.x; cb < g.combos; cb += gridDim.x) {
        const int ph = (int)(cb / items), it = (int)(cb % items);
        const int b = it % nb, y0 = (it / nb) * YB;
        const int ylim = min(YB, (int)g.my - y0);
        const long long p0 = ph * g.per, p1 = min(pairs, p0 + g.per);
        lrx_c2 acc[E];
#pragma unroll
        for (int e = 0; e < E; ++e) acc[e] = lrx_c2{0.0, 0.0};
        for (long long p = p0; p < p1; ++p) {
            const int a = (int)(p % na), x0 = (int)(p / na) * XB;
            const int xlim = min(XB, (int)g.mx - x0);
            conv_pair(bank, L, R, V, g, a, b, x0, y0, xlim, ylim);
            // Decode: acc(k, c, yi) += sum_xi conj(Pc[k, c, a, x0 + xi]) U[k, xi, yi], xi in order;
            // re: Pr Ur then Pi Ui, im: Pr Ui then -Pi Ur (the CUDA decode's per-accumulator order).
#pragma unroll
            for (int e = 0; e < E; ++e) {
                const int i = threadIdx.x + e * LRX_NT;
                if (i < NK * NC * YB) {
                    const int yi = i % YB, c = (i / YB) % NC, k = i / (YB * NC);
                    const lrx_c2* pp = Pc + (((long long)k * NC + c) * na + a) * g.mx + x0;
                    double re = acc[e].x, im = acc[e].y;
                    for (int xi = 0; xi < xlim; ++xi) {
                        const lrx_c2 pv = pp[xi], uv = bank[(xi * YB + yi) * GK::RS + GK::at(k)];
                        re = fma(pv.x, uv.x, re); re = fma(pv.y, uv.y, re);
                        im = fma(pv.x, uv.y, im); im = fma(-pv.y, uv.x, im);
                    }
                    acc[e] = lrx_c2{re, im};
                }
            }
            __syncthreads();                                       // the bank is reloaded next pair
        }
        // The combo's scaled partial: straight into A when one phase covers the item, else to its slot,
        // and the combo completing the item sums the item's slots in phase order into A.
        const int ngrp = (int)g.ngrp;
        lrx_c2* sb = slots + (unsigned long long)cb * SLOT;
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const int i = threadIdx.x + e * LRX_NT;
            if (i < NK * NC * YB) {
                const int yi = i % YB, c = (i / YB) % NC, k = i / (YB * NC);
                const lrx_c2 w{acc[e].x * g.scale, acc[e].y * g.scale};
                if (ngrp == 1) {
                    if (yi < ylim) A[(((long long)k * NC + c) * nb + b) * g.my + y0 + yi] = w;
                } else {
                    sb[i] = w;
                }
            }
        }
        if (ngrp == 1) continue;
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) last = atomicAdd(count + it, 1) == ngrp - 1;
        __syncthreads();
        if (last) {
            __threadfence();
            for (int i = threadIdx.x; i < NK * NC * YB; i += blockDim.x) {
                const int yi = i % YB, c = (i / YB) % NC, k = i / (YB * NC);
                if (yi < ylim) {
                    const volatile lrx_c2* s0 = slots + (unsigned long long)it * SLOT + i;
                    double sx = s0->x, sy = s0->y;
                    for (int q = 1; q < ngrp; ++q) {
                        const volatile lrx_c2* sq = slots + (unsigned long long)(q * items + it) * SLOT + i;
                        sx += sq->x; sy += sq->y;
                    }
                    A[(((long long)k * NC + c) * nb + b) * g.my + y0 + yi] = lrx_c2{sx, sy};
                }
            }
        }
        __syncthreads();
    }
}
#endif
