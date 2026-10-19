// kbox_unfold_kernel.h -- DEVICE SOURCE, run-time compiled by the leg's RTC (common/rtc_build.h);
// the host compiler never sees it.  CMake embeds it as text (kbox_unfold_src.h.in) beside the
// k-box stage's text (cufft/kbox_stage_src.h.in), which hipRTC reads as the named header
// "kbox_stage.cuh".  The CUDA/HIP common subset: no vendor header or intrinsic here; the stage
// spells the leg's line transform (LRX_KBOX_DIRECT_DFT on hipRTC: the kdft direct DFT).
//
// Mode 7 of docs/architecture/kconv.md, the Sigma k-leading convolution read from the RAW-PARENT
// Green, on the k-box stage's single arm (kbox_stage.cuh):
//
//   U[ko, a, rx, b, y] = scale * fftn_k( ifftn_k( (U_k G~[row(k)] U_k^dagger)[x a, y b] ) * V[k, x, y] )
//
// for full k with ko = kout[k] >= 0, x the stored x-block row rx (lrx_x_of), and G~ the typed
// unfold of the parent tile (mph * G[lsrc, rsrc] * nph, a -1 source an exact zero; the partner Gt
// or conj(G) on an antiunitary row).  The tile holds TRC columns = whole spin groups (NS*NS columns
// per (x, y) pair): one thread per (k, pair) reads the pair's NS*NS sources once and forms the
// group's U g U^dagger in registers, as the mathdx family's register load (kconv_mathdx_cuda_ffi.cc
// lrx_unfold_pair / lrx_spin_row / lrx_unfold_direct, the same products in the same order).
// Then the stage's inverse transform3, the group Mid V[k, x, y], transform3 forward and the store
// through kout.  One HBM pass: the full-k Green is never written.
//
// Compile-time constants (keyed options): LRX_NX, LRX_NY, LRX_NZ, LRX_NS, LRX_TR (tile columns,
// whole groups), LRX_NT (threads), LRX_LIVE (1: the program reads the live rows operand).

#include "kbox_stage.cuh"

struct alignas(16) lrx_c2 { double x, y; };
constexpr int NX = LRX_NX, NY = LRX_NY, NZ = LRX_NZ, NS = LRX_NS, SS = NS * NS, TRC = LRX_TR;
static_assert(TRC % SS == 0, "tiles hold whole spin groups");
using KG = lrx_kbox::Geo<NX, NY, NZ>;

// The host's UnfoldTab (kbox_unfold_ffi.cc; field for field the mathdx family's): merged endpoints
// ml = mx*NS, nl = my*NS; kout the output row of full k (-1: transformed, not stored).
struct UnfoldTab {
    const int *row, *trs, *lsrc, *rsrc;
    const double *mph, *nph, *spin;
    long long ml, nl;
    const int* kout;
    int conj_trs;                              // 0 read the partner tile, 2 read conj(G)
    long long x0, bx, xs, xn;                  // the stored x block (bx = 0: every x)
};

__device__ __forceinline__ lrx_c2 lrx_mul(lrx_c2 a, lrx_c2 b) {
    return lrx_c2{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
__device__ __forceinline__ lrx_c2 lrx_mulf(lrx_c2 a, lrx_c2 b) {
    return lrx_c2{fma(a.x, b.x, -(a.y * b.y)), fma(a.x, b.y, a.y * b.x)};
}
__device__ __forceinline__ void lrx_cmac(lrx_c2& v, lrx_c2 a, lrx_c2 b) {
    v.x = fma(a.x, b.x, v.x); v.x = fma(-a.y, b.y, v.x);
    v.y = fma(a.x, b.y, v.y); v.y = fma(a.y, b.x, v.y);
}
__device__ __forceinline__ void lrx_cmac_conj(lrx_c2& v, lrx_c2 a, lrx_c2 b) {
    v.x = fma(a.x, b.x, v.x); v.x = fma(a.y, b.y, v.x);
    v.y = fma(a.y, b.x, v.y); v.y = fma(-a.x, b.y, v.y);
}

struct XBlock { long long x0, bx, xs, rows, mx; };
__device__ __forceinline__ XBlock lrx_x_block(const UnfoldTab& t) {
    const long long mx = t.ml / NS;
    return t.bx > 0 ? XBlock{t.x0, t.bx, t.xs, t.xn * t.bx, mx} : XBlock{0, mx, mx, mx, mx};
}
__device__ __forceinline__ long long lrx_x_of(const XBlock& b, long long r) {
    return (r / b.bx) * b.xs + b.x0 + r % b.bx;
}

// One (k, x, y) pair's typed unfold g[c][d] and the spin action u (lrx_unfold_pair).
__device__ __forceinline__ void lrx_unfold_pair(const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt,
                                                const UnfoldTab& t, int k, long long xx, long long yy,
                                                lrx_c2 (&g)[NS][NS], lrx_c2 (&u)[NS][NS]) {
    const lrx_c2* __restrict__ mph = reinterpret_cast<const lrx_c2*>(t.mph);
    const lrx_c2* __restrict__ nph = reinterpret_cast<const lrx_c2*>(t.nph);
    const lrx_c2* __restrict__ spin = reinterpret_cast<const lrx_c2*>(t.spin);
    const bool anti = t.trs[k] != 0, conj_src = anti && t.conj_trs == 2;
    const lrx_c2* src = ((anti && t.conj_trs == 0) ? gt : gp) + (long long)t.row[k] * t.ml * t.nl;
#pragma unroll
    for (int c = 0; c < NS; ++c) {
        const long long li = (long long)k * t.ml + xx * NS + c;
        const int ls = t.lsrc[li];
        const lrx_c2 mp = mph[li];
#pragma unroll
        for (int d = 0; d < NS; ++d) {
            const long long rj = (long long)k * t.nl + yy * NS + d;
            const int rs = t.rsrc[rj];
            lrx_c2 v{0.0, 0.0};
            if (ls >= 0 && rs >= 0) {
                lrx_c2 sv = src[(long long)ls * t.nl + rs];
                if (conj_src) sv.y = -sv.y;
                v = lrx_mulf(lrx_mulf(mp, sv), nph[rj]);
            }
            g[c][d] = v;
        }
    }
#pragma unroll
    for (int a = 0; a < NS; ++a)
#pragma unroll
        for (int b = 0; b < NS; ++b) u[a][b] = spin[((long long)k * NS + a) * NS + b];
}

// Row a of U g U^dagger (lrx_spin_row with Ur = U).
__device__ __forceinline__ void lrx_spin_row(const lrx_c2 (&u)[NS][NS], const lrx_c2 (&g)[NS][NS], int a,
                                             lrx_c2 (&out)[NS]) {
    lrx_c2 left[NS];
#pragma unroll
    for (int d = 0; d < NS; ++d) {
        lrx_c2 v{0.0, 0.0};
#pragma unroll
        for (int c = 0; c < NS; ++c) lrx_cmac(v, u[a][c], g[c][d]);
        left[d] = v;
    }
#pragma unroll
    for (int b = 0; b < NS; ++b) {
        lrx_c2 v{0.0, 0.0};
#pragma unroll
        for (int d = 0; d < NS; ++d) lrx_cmac_conj(v, left[d], u[b][d]);
        out[b] = v;
    }
}

struct M7Load {                                // the stage's direct Load: whole spin groups
    static constexpr bool kDirect = true, kFinish = false;
    const lrx_c2 *gp, *gt;
    const UnfoldTab* t;
    long long my, pairs;
    template <class View>
    __device__ void direct(const View& view, int k0, int k1, long long col0, int width, long long) const {
        const XBlock xb = lrx_x_block(*t);
        const int groups = width / SS;
        for (int i = threadIdx.x; i < (k1 - k0) * groups; i += blockDim.x) {
            const int k = k0 + i / groups, j = i % groups;
            const long long pr = col0 / SS + j;
            const long long rx = pr / my, yy = pr - rx * my, xx = lrx_x_of(xb, rx);
            if (pr < pairs && xx < xb.mx) {
                lrx_c2 g[NS][NS], u[NS][NS];
                lrx_unfold_pair(gp, gt, *t, k, xx, yy, g, u);
#pragma unroll
                for (int a = 0; a < NS; ++a) {
                    lrx_c2 out[NS];
                    lrx_spin_row(u, g, a, out);
#pragma unroll
                    for (int b = 0; b < NS; ++b) view(k, j * SS + a * NS + b) = out[b];
                }
            } else {
#pragma unroll
                for (int ab = 0; ab < SS; ++ab) view(k, j * SS + ab) = lrx_c2{0.0, 0.0};
            }
        }
    }
};

struct M7Mid {                                 // the pair's kernel V[k, x, y] on its whole group
    const lrx_c2* __restrict__ kern;
    XBlock xb;
    long long my;
    template <class Get>
    __device__ void group(int k, long long pr, const Get& get) const {
        const long long rx = pr / my, yy = pr - rx * my;
        const long long xx = lrx_x_of(xb, rx) < xb.mx ? lrx_x_of(xb, rx) : xb.mx - 1;
        const lrx_c2 wv = kern[((long long)k * xb.mx + xx) * my + yy];
#pragma unroll
        for (int q = 0; q < SS; ++q) get(q) = lrx_mul(get(q), wv);
    }
};

struct M7Store {                               // U[(ko, a, rx, b, y)], full-k row k at kout[k]
    lrx_c2* y;
    const UnfoldTab* t;
    long long rows, my;
    double scale;
    __device__ void put(int k, long long col, lrx_c2 v) const {
        const long long ko = t->kout[k];
        if (ko < 0) return;
        const long long pr = col / SS, rx = pr / my, yy = pr - rx * my;
        const int m = int(col % SS), a = m / NS, b = m % NS;
        y[((ko * NS + a) * rows + rx) * (my * NS) + b * my + yy] = lrx_c2{v.x * scale, v.y * scale};
    }
};

// live: the window rows [live[0], live[1]) of a padded pass (null: all).  Grid-stride over tiles.
extern "C" __global__ void __launch_bounds__(LRX_NT)
lrx_kbox_m7(const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt, const lrx_c2* __restrict__ kern,
            lrx_c2* __restrict__ y, UnfoldTab t, double scale, const int* __restrict__ live) {
    extern __shared__ lrx_c2 sm[];
    const XBlock xb = lrx_x_block(t);
    const long long my = t.nl / NS, pairs = xb.rows * my, ncols = pairs * SS;
    const M7Load ld{gp, gt, &t, my, pairs};
    const M7Mid mid{kern, xb, my};
    const M7Store st{y, &t, xb.rows, my, scale};
    const lrx_kbox::Live lv = lrx_kbox::Live::rows(live, my * SS, 0, true);
    const long long cb = lv.b(), ce = lv.e(ncols);
    using lrx_kbox::fft_direction;
    for (long long col0 = cb + (long long)blockIdx.x * TRC; col0 < ce; col0 += (long long)gridDim.x * TRC) {
        lrx_kbox::stage_tile<NX, NY, NZ, TRC>(sm, col0, ce, ld);
        lrx_kbox::transform3<NX, NY, NZ, TRC, 0, fft_direction::inverse>(sm);
        lrx_kbox::mid_group_tile<NX, NY, NZ, TRC, SS>(sm, col0, ce, mid);
        lrx_kbox::transform3<NX, NY, NZ, TRC, 0, fft_direction::forward>(sm);
        lrx_kbox::store_tile<NX, NY, NZ, TRC>(sm, col0, ce, st);
    }
    lrx_kbox::zero_dead<KG::NK, lrx_c2>(ncols, lv, st);
}
