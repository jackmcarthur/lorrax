// kdft_core.h -- DEVICE SOURCE, the kdft family's DFT core (run-time compiled through
// common/rtc_build.h, embedded as text next to the kernels that include it).  CUDA/HIP common
// subset: no vendor header, no vendor intrinsic.
//
// An LDS tile s[g*W + w] holds W columns of a group of up to three k axes (GX, GY, GZ),
// g = (ix*GY + iy)*GZ + iz.  dft3() transforms the group's axes in place.  Each axis of length N
// is a mixed-radix Stockham sequence (self-sorting, so the tile stays in natural order):
//
//   stage (radix R, P = product of the earlier radices, M = N/R), for i in [0, M), k = i mod P:
//     u_r = x[i + r*M] * w^(r*k/(P*R))        r in [0, R)        (w = exp(sgn 2 pi i))
//     y[(i - k)*R + k + q*P] = sum_r u_r w^(r*q/R)                q in [0, R)
//
// A work item is one butterfly (R <= kRmax: its R outputs in registers; bfly<R> below, a
// split-radix FFT tree with constant twiddles for R = 2, 4, 8, 16, the direct R-term sum for any
// other radix, its R twiddles held in registers) or a
// kChunk-pair slice of one (a prime R > kRmax, e.g. 97: outputs q and R-q for kChunk values of q,
// from the sums and differences of inputs r and R-r against one cos/sin, a quarter of the direct
// DFT's FMAs and twiddle reads).  Lanes
// run along w, so a wavefront's LDS accesses are consecutive and its twiddle indices uniform.
// A stage computes every item into registers, then a barrier, then writes: in place, any order
// (a single-stage axis, M = 1, writes each line straight back: its outputs are its own slots).
// Twiddles: one table w_N^m, m in [0, N), per axis, from sincospi(2m/N) (phase reduced mod N,
// correctly rounded); every stage indexes it (no recurrence, so no error growth with N).
//
// The radix plan (kdft_plan.h) depends on N only; the host reads the same header to size a
// pass's thread count.
//
// line_reg<N>() is the same plan on one line held in a thread's registers (fully unrolled, so
// every index and twiddle is a compile-time constant): the register pass (kdft_klead_kernel.h,
// LRX_REG) moves a whole line per LDS round trip instead of one per stage.
#pragma once

#if LRX_F32
typedef float real;
#define LRX_SINCOSPI sincospif
#else
typedef double real;
#define LRX_SINCOSPI sincospi
#endif

struct alignas(2 * sizeof(real)) cplx { real x, y; };

#include "kdft_plan.h"

namespace kdft {

template <int N> struct PlanOf { static constexpr Plan v = make_plan(N); };

__device__ __forceinline__ cplx cfma(cplx a, cplx b, cplx c) {          // c + a*b
    return cplx{fma(a.x, b.x, fma(-a.y, b.y, c.x)), fma(a.x, b.y, fma(a.y, b.x, c.y))};
}
__device__ __forceinline__ cplx twiddle(const cplx* tw, int m, real sgn) {
    const cplx w = tw[m];
    return cplx{w.x, sgn * w.y};
}
__device__ __forceinline__ cplx cmul(cplx a, cplx b) {
    return cplx{fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x)};
}

// Compile-time loops: sfor<B, E>(f) calls f(IC<i>{}) for i in [B, E), so an index used inside is
// a constant expression (twiddle constants, register-array indices).
template <int V> struct IC { static constexpr int value = V; };
template <int B, int E, class F>
__device__ __forceinline__ void sfor(F&& f) {
    if constexpr (B < E) {
        f(IC<B>{});
        sfor<B + 1, E>(f);
    }
}
// The constant twiddle w_N^M = exp(sgn * 2 pi i M / N): its cos and sin are compile-time values
// (kdft_plan.h trig2pi), so they are literal or scalar operands, never a register per lane.
template <int N, int M> struct TwC {
    static constexpr real c = real(trig2pi(M, N, 0));
    static constexpr real s = real(trig2pi(M, N, 1));
};
template <int N, int M>
__device__ __forceinline__ cplx twc(real sgn) { return cplx{TwC<N, M>::c, sgn * TwC<N, M>::s}; }

// u <- DFT_R(u) in natural order, R a power of two <= 16: radix-2 decimation in time, the
// trivial twiddles (m = 0, R/4) as swaps.
template <int R>
__device__ __forceinline__ void fft_pow2(cplx* u, real sgn) {
    if constexpr (R == 2) {
        const cplx a = u[0], b = u[1];
        u[0] = cplx{a.x + b.x, a.y + b.y};
        u[1] = cplx{a.x - b.x, a.y - b.y};
    } else if constexpr (R > 2) {
        constexpr int H = R / 2;
        cplx e[H], o[H];
#pragma unroll
        for (int j = 0; j < H; ++j) { e[j] = u[2 * j]; o[j] = u[2 * j + 1]; }
        fft_pow2<H>(e, sgn);
        fft_pow2<H>(o, sgn);
        sfor<0, H>([&](auto qi) {
            constexpr int q = decltype(qi)::value;
            cplx t = o[q];
            if constexpr (4 * q == R) t = cplx{-sgn * o[q].y, sgn * o[q].x};
            else if constexpr (q != 0) t = cmul(o[q], twc<R, q>(sgn));
            u[q] = cplx{e[q].x + t.x, e[q].y + t.y};
            u[q + H] = cplx{e[q].x - t.x, e[q].y - t.y};
        });
    }
}
// u <- DFT_R(u): y_q = sum_r u_r w_R^{rq}, constant twiddles.  A power of two <= 16: fft_pow2.
// An odd radix: the conjugate-pair form (X_q and X_{R-q} from the sums u_r + u_{R-r} and
// differences u_r - u_{R-r} against one cos / sin each: half the direct sum's FMAs).  Any other
// radix: the direct sum.
template <int R>
__device__ __forceinline__ void bfly(cplx* u, real sgn) {
    if constexpr (R == 2 || R == 4 || R == 8 || R == 16) {
        fft_pow2<R>(u, sgn);
    } else if constexpr (R % 2 == 1) {
        constexpr int H = (R - 1) / 2;
        cplx sp[H + 1], dm[H + 1];
        cplx y0 = u[0];
#pragma unroll
        for (int r = 1; r <= H; ++r) {
            sp[r] = cplx{u[r].x + u[R - r].x, u[r].y + u[R - r].y};
            dm[r] = cplx{u[r].x - u[R - r].x, u[r].y - u[R - r].y};
            y0 = cplx{y0.x + sp[r].x, y0.y + sp[r].y};
        }
        const cplx x0 = u[0];
        sfor<1, H + 1>([&](auto qi) {
            constexpr int q = decltype(qi)::value;
            cplx A = x0, D{0, 0};
            sfor<1, H + 1>([&](auto ri) {
                constexpr int r = decltype(ri)::value;
                constexpr int m = (r * q) % R;
                A = cplx{fma(sp[r].x, TwC<R, m>::c, A.x), fma(sp[r].y, TwC<R, m>::c, A.y)};
                D = cplx{fma(dm[r].x, TwC<R, m>::s, D.x), fma(dm[r].y, TwC<R, m>::s, D.y)};
            });
            u[q] = cplx{A.x - sgn * D.y, A.y + sgn * D.x};
            u[R - q] = cplx{A.x + sgn * D.y, A.y - sgn * D.x};
        });
        u[0] = y0;
    } else {
        cplx y[R];
        sfor<0, R>([&](auto qi) {
            constexpr int q = decltype(qi)::value;
            cplx a = u[0];
            sfor<1, R>([&](auto ri) {
                constexpr int r = decltype(ri)::value;
                a = cfma(u[r], twc<R, (r * q) % R>(sgn), a);
            });
            y[q] = a;
        });
#pragma unroll
        for (int q = 0; q < R; ++q) u[q] = y[q];
    }
}
template <int R> struct IsPow2Bfly { static constexpr bool v = R == 2 || R == 4 || R == 8 || R == 16; };

// The table w_N^m = exp(2 pi i m / N) of each group axis, laid out [GX | GY | GZ].
template <int GX, int GY, int GZ, int NT>
__device__ __forceinline__ void fill_twiddles(cplx* tw) {
    for (int m = threadIdx.x; m < GX + GY + GZ; m += NT) {
        const int n = m < GX ? GX : (m < GX + GY ? GY : GZ);
        const int q = m < GX ? m : (m < GX + GY ? m - GX : m - GX - GY);
        real sn, cs;
        LRX_SINCOSPI(real(2 * q) / real(n), &sn, &cs);
        tw[m] = cplx{cs, sn};
    }
}

// One Stockham stage of every line along an axis (length N, k-stride S in g units).
template <int N, int S, int R, int P, int G, int W, int NT>
__device__ __forceinline__ void stage(cplx* s, const cplx* tw, real sgn) {
    constexpr int M = N / R;
    constexpr bool WHOLE = R <= kRmax;
    // a radix above kRmax is odd (a prime): conjugate pairs q, R-q share one cos/sin, so a
    // work item forms kChunk pairs from the input sums x_r + x_{R-r} and differences x_r - x_{R-r}
    constexpr int HALF = (R - 1) / 2;
    constexpr int CH = WHOLE ? R : kChunk;
    constexpr int NC = WHOLE ? 1 : (HALF + CH - 1) / CH;
    constexpr int OUT = WHOLE ? R : 2 * CH + 1;                  // outputs per item (pairs + X_0)
    constexpr int ITEMS = (G / N) * W * M * NC;
    constexpr int IPT = (ITEMS + NT - 1) / NT;
    // M == 1 with whole butterflies: the item's outputs are its own inputs' slots -> in place
    constexpr bool INPLACE = WHOLE && M == 1;
    cplx acc[INPLACE ? 1 : IPT][OUT];
#pragma unroll
    for (int it = 0; it < IPT; ++it) {
        const int item = threadIdx.x + it * NT;
        if (ITEMS % NT == 0 || item < ITEMS) {
            const int w = item % W;
            int rest = item / W;
            const int ch = rest % NC; rest /= NC;
            const int i = rest % M;
            const int lo = rest / M;
            const int base = (lo / S) * (S * N) + (lo % S);
            const int k = i % P;
            cplx* a = acc[INPLACE ? 0 : it];
            if constexpr (WHOLE) {
                (void)ch;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    cplx x = s[(base + (i + r * M) * S) * W + w];
                    if constexpr (P > 1) {
                        if (r > 0) x = cmul(x, twiddle(tw, (N / (P * R)) * r * k, sgn));
                    }
                    a[r] = x;
                }
                bfly<R>(a, sgn);
                if constexpr (INPLACE) {
#pragma unroll
                    for (int q = 0; q < R; ++q) s[(base + q * S) * W + w] = a[q];
                }
            } else {
                // a[0..CH): sum_r s_r cos(2 pi r q / R); a[CH..2CH): sum_r d_r sin(...); a[2CH]: X_0
#pragma unroll
                for (int q = 0; q < OUT; ++q) a[q] = cplx{0, 0};
                const int q0 = 1 + ch * CH;
                auto load = [&](int r) {
                    cplx x = s[(base + (i + r * M) * S) * W + w];
                    if constexpr (P > 1) x = cmul(x, twiddle(tw, (N / (P * R)) * r * k, sgn));
                    return x;
                };
                const cplx x0 = load(0);
                int idx[CH], inc[CH];
#pragma unroll
                for (int q = 0; q < CH; ++q) { inc[q] = (q0 + q) % R; idx[q] = inc[q]; }
                real sx = x0.x, sy = x0.y;
                for (int r = 1; r <= HALF; ++r) {
                    const cplx xp = load(r), xm = load(R - r);
                    const cplx sr{xp.x + xm.x, xp.y + xm.y}, dr{xp.x - xm.x, xp.y - xm.y};
                    sx += sr.x; sy += sr.y;
#pragma unroll
                    for (int q = 0; q < CH; ++q) {
                        const cplx c = tw[(N / R) * idx[q]];             // (cos, sin) of 2 pi r q / R
                        a[q].x = fma(sr.x, c.x, a[q].x);
                        a[q].y = fma(sr.y, c.x, a[q].y);
                        a[CH + q].x = fma(dr.x, c.y, a[CH + q].x);
                        a[CH + q].y = fma(dr.y, c.y, a[CH + q].y);
                        idx[q] += inc[q];
                        if (idx[q] >= R) idx[q] -= R;
                    }
                }
                // X_q = x0 + A + i sgn D,  X_{R-q} = x0 + A - i sgn D
#pragma unroll
                for (int q = 0; q < CH; ++q) {
                    const cplx A = a[q], D = a[CH + q];
                    a[q] = cplx{x0.x + A.x - sgn * D.y, x0.y + A.y + sgn * D.x};
                    a[CH + q] = cplx{x0.x + A.x + sgn * D.y, x0.y + A.y - sgn * D.x};
                }
                a[2 * CH] = cplx{sx, sy};
            }
        }
    }
    if constexpr (INPLACE) {
        __syncthreads();
        return;
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < IPT; ++it) {
        const int item = threadIdx.x + it * NT;
        if (ITEMS % NT == 0 || item < ITEMS) {
            const int w = item % W;
            int rest = item / W;
            const int ch = rest % NC; rest /= NC;
            const int i = rest % M;
            const int lo = rest / M;
            const int base = (lo / S) * (S * N) + (lo % S);
            const int k = i % P;
            auto put = [&](int q, cplx val) {
                s[(base + ((i - k) * R + k + q * P) * S) * W + w] = val;
            };
            if constexpr (WHOLE) {
#pragma unroll
                for (int q = 0; q < R; ++q) put(q, acc[it][q]);
            } else {
                const int q0 = 1 + ch * CH;
#pragma unroll
                for (int q = 0; q < CH; ++q)
                    if (q0 + q <= HALF) { put(q0 + q, acc[it][q]); put(R - q0 - q, acc[it][CH + q]); }
                if (ch == 0) put(0, acc[it][2 * CH]);
            }
        }
    }
    __syncthreads();
}

template <int N, int S, int G, int W, int NT, int ST>
__device__ __forceinline__ void line_dft(cplx* s, const cplx* tw, real sgn) {
    if constexpr (ST < PlanOf<N>::v.n) {
        stage<N, S, PlanOf<N>::v.r[ST], PlanOf<N>::v.p[ST], G, W, NT>(s, tw, sgn);
        line_dft<N, S, G, W, NT, ST + 1>(s, tw, sgn);
    }
}

// The plan's stages on one line v[0..N) held in registers (natural order in and out): stage ST
// with radix R, P = the earlier radices' product, M = N/R, the Stockham map of the LDS stage.
// Every index is a compile-time constant (sfor), so v stays in VGPRs and every twiddle is a
// constant (TwC).  Radices above kRmax are not taken here (the register pass requires reg_ok).
template <int N, int ST>
__device__ __forceinline__ void reg_stages(cplx* v, real sgn) {
    if constexpr (ST < PlanOf<N>::v.n) {
        constexpr int R = PlanOf<N>::v.r[ST], P = PlanOf<N>::v.p[ST], M = N / R;
        cplx y[N];
        sfor<0, M>([&](auto ii) {
            constexpr int i = decltype(ii)::value;
            constexpr int k = i % P;
            cplx u[R];
            sfor<0, R>([&](auto ri) {
                constexpr int r = decltype(ri)::value;
                if constexpr (r > 0 && k > 0) u[r] = cmul(v[i + r * M], twc<N, (N / (P * R)) * r * k>(sgn));
                else u[r] = v[i + r * M];
            });
            bfly<R>(u, sgn);
#pragma unroll
            for (int q = 0; q < R; ++q) y[(i - k) * R + k + q * P] = u[q];
        });
#pragma unroll
        for (int j = 0; j < N; ++j) v[j] = y[j];
        reg_stages<N, ST + 1>(v, sgn);
    }
}
template <int N>
__device__ __forceinline__ void line_reg(cplx* v, real sgn) {
    reg_stages<N, 0>(v, sgn);
}

// The group's unnormalised DFT, in place: sgn = +1 inverse (exp(+2 pi i)), -1 forward.
template <int GX, int GY, int GZ, int W, int NT>
__device__ __forceinline__ void dft3(cplx* s, const cplx* tw, real sgn) {
    constexpr int G = GX * GY * GZ;
    line_dft<GX, GY * GZ, G, W, NT, 0>(s, tw, sgn);
    line_dft<GY, GZ, G, W, NT, 0>(s, tw + GX, sgn);
    line_dft<GZ, 1, G, W, NT, 0>(s, tw + GX + GY, sgn);
}

}  // namespace kdft
