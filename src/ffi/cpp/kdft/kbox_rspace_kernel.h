// kbox_rspace_kernel.h -- DEVICE SOURCE, run-time compiled by the leg's RTC (common/rtc_build.h);
// the host compiler never sees it.  CMake embeds it as text (kbox_rspace_src.h.in) beside the
// k-box stage's text (cufft/kbox_stage_src.h.in), which hipRTC reads as the named header
// "kbox_stage.cuh".  The CUDA/HIP common subset: no vendor header or intrinsic here; the stage
// spells the leg's line transform (LRX_KBOX_DIRECT_DFT on hipRTC: the kdft direct DFT).
//
// The k-box modes of docs/architecture/kconv.md whose result stays in R space (one inverse
// transform, no forward one), on the stage's single arm (kbox_stage.cuh), as the mathdx family
// spells them (cufft/kconv_mathdx_cuda_ffi.cc LRX_MODE 9 / 11: the same loads, the same products
// in the same order, the same Mid):
//
//   LRX_MODE 9   Y[k, x A, y B] = scale * ifftn_k( (U_k W~[row(k)] Ur_k^dagger)[x A, y B] )
//                an interaction read from its q wedge (WedgeLoad / WedgeStore).  Endpoint widths
//                NS (left) and NR (right); conj_trs 0 reads the partner tile on an antiunitary
//                row, 1 conjugates the phased product (a Hermitian interaction).
//   LRX_MODE 11  acc[o, k, x, y] += alpha[o] * sum_ab conj(si Gc'_ab) (si Gv'_ab)  (+ c.c.)
//                one chi0 tau node from the raw-parent Green pair, G' the unfolded Green after
//                the unnormalised inverse transform (ChiLoad / ChiMid); LRX_VTX: the na x nb
//                four-current channels with the Gc operand's per-k sign sgn_c (lrx_chi_vertex).
//
// One HBM pass: the full-k operand is never written.  Tile column c of mode 11's pair p = c / GRP
// is operand (c % GRP) / SS (0 Gv, 1 Gc) and spin element (a, b) = (c % SS) / NS, c % NS.
//
// Compile-time constants (keyed options): LRX_MODE, LRX_NX, LRX_NY, LRX_NZ, LRX_NS, LRX_NSR
// (mode 9's right width), LRX_TR (tile columns, whole groups), LRX_NT (threads), LRX_LIVE,
// and for mode 11 LRX_COMPLETE, LRX_VTX.

#include "kbox_stage.cuh"

struct alignas(16) lrx_c2 { double x, y; };
#ifndef LRX_NSR
#define LRX_NSR LRX_NS
#endif
constexpr int NX = LRX_NX, NY = LRX_NY, NZ = LRX_NZ, NS = LRX_NS, NR = LRX_NSR, SS = NS * NR, TRC = LRX_TR;
using KG = lrx_kbox::Geo<NX, NY, NZ>;
constexpr int NK = KG::NK;

// The host's UnfoldTab (kbox_rspace_ffi.cc), merged endpoints ml = mx*NS, nl = my*NR.
struct UnfoldTab {
    const int *row, *trs, *lsrc, *rsrc;
    const double *mph, *nph, *spin, *spin_r;   // spin_r: the right action (mode 11: spin itself)
    long long ml, nl;
    int conj_trs;                              // 0 partner tile, 1 conjugate the product, 2 conj(G)
};

// The family's fused product forms (kconv_mathdx_cuda_ffi.cc lrx_mulf / lrx_cmac / lrx_cmac_conj).
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
__device__ __forceinline__ lrx_c2 lrx_phase(lrx_c2 z, int code) {
    if (code == 1) return lrx_c2{-z.y, z.x};
    if (code == 2) return lrx_c2{-z.x, -z.y};
    if (code == 3) return lrx_c2{z.y, -z.x};
    return z;
}

// The typed unfold of one (k, x, y) pair and both spin actions (lrx_unfold_pair<SL, SR>).
template <int SL, int SR>
__device__ __forceinline__ void lrx_unfold_pair(const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt,
                                                const UnfoldTab& t, int k, long long xx, long long yy,
                                                lrx_c2 (&g)[SL][SR], lrx_c2 (&u)[SL][SL], lrx_c2 (&ur)[SR][SR]) {
    const lrx_c2* __restrict__ mph = reinterpret_cast<const lrx_c2*>(t.mph);
    const lrx_c2* __restrict__ nph = reinterpret_cast<const lrx_c2*>(t.nph);
    const lrx_c2* __restrict__ spin = reinterpret_cast<const lrx_c2*>(t.spin);
    const lrx_c2* __restrict__ spin_r = reinterpret_cast<const lrx_c2*>(t.spin_r ? t.spin_r : t.spin);
    const bool anti = t.trs[k] != 0, conj_row = anti && t.conj_trs == 1, conj_src = anti && t.conj_trs == 2;
    const lrx_c2* src = ((anti && t.conj_trs == 0) ? gt : gp) + (long long)t.row[k] * t.ml * t.nl;
#pragma unroll
    for (int c = 0; c < SL; ++c) {
        const long long li = (long long)k * t.ml + xx * SL + c;
        const int ls = t.lsrc[li];
        const lrx_c2 mp = mph[li];
#pragma unroll
        for (int d = 0; d < SR; ++d) {
            const long long rj = (long long)k * t.nl + yy * SR + d;
            const int rs = t.rsrc[rj];
            lrx_c2 v{0.0, 0.0};
            if (ls >= 0 && rs >= 0) {
                lrx_c2 sv = src[(long long)ls * t.nl + rs];
                if (conj_src) sv.y = -sv.y;
                v = lrx_mulf(lrx_mulf(mp, sv), nph[rj]);
                if (conj_row) v.y = -v.y;
            }
            g[c][d] = v;
        }
    }
#pragma unroll
    for (int a = 0; a < SL; ++a)
#pragma unroll
        for (int b = 0; b < SL; ++b) u[a][b] = spin[((long long)k * SL + a) * SL + b];
#pragma unroll
    for (int a = 0; a < SR; ++a)
#pragma unroll
        for (int b = 0; b < SR; ++b) ur[a][b] = spin_r[((long long)k * SR + a) * SR + b];
}

// Row a of U g Ur^dagger (lrx_spin_row<SL, SR>).
template <int SL, int SR>
__device__ __forceinline__ void lrx_spin_row(const lrx_c2 (&u)[SL][SL], const lrx_c2 (&ur)[SR][SR],
                                             const lrx_c2 (&g)[SL][SR], int a, lrx_c2 (&out)[SR]) {
    lrx_c2 left[SR];
#pragma unroll
    for (int d = 0; d < SR; ++d) {
        lrx_c2 v{0.0, 0.0};
#pragma unroll
        for (int c = 0; c < SL; ++c) lrx_cmac(v, u[a][c], g[c][d]);
        left[d] = v;
    }
#pragma unroll
    for (int b = 0; b < SR; ++b) {
        lrx_c2 v{0.0, 0.0};
#pragma unroll
        for (int d = 0; d < SR; ++d) lrx_cmac_conj(v, left[d], ur[b][d]);
        out[b] = v;
    }
}

#if LRX_MODE == 9
// ---- mode 9: the wedge load, the inverse transform, the k-leading store -----------------------
struct WedgeLoad {                             // whole spin groups (SS columns per pair)
    static constexpr bool kDirect = true, kFinish = false;
    const lrx_c2 *gp, *gt;
    const UnfoldTab* t;
    long long my;
    template <class View>
    __device__ void direct(const View& view, int k0, int k1, long long col0, int width, long long ncols) const {
        const long long pairs = ncols / SS;
        const int groups = width / SS;
        for (int i = threadIdx.x; i < (k1 - k0) * groups; i += blockDim.x) {
            const int k = k0 + i / groups, j = i % groups;
            const long long pr = col0 / SS + j;
            if (pr < pairs) {
                const long long xx = pr / my, yy = pr - xx * my;
                lrx_c2 g[NS][NR], u[NS][NS], ur[NR][NR];
                lrx_unfold_pair<NS, NR>(gp, gt, *t, k, xx, yy, g, u, ur);
#pragma unroll
                for (int a = 0; a < NS; ++a) {
                    lrx_c2 out[NR];
                    lrx_spin_row<NS, NR>(u, ur, g, a, out);
#pragma unroll
                    for (int b = 0; b < NR; ++b) view(k, j * SS + a * NR + b) = out[b];
                }
            } else {
#pragma unroll
                for (int e = 0; e < SS; ++e) view(k, j * SS + e) = lrx_c2{0.0, 0.0};
            }
        }
    }
};

struct WedgeStore {                            // Y[k, x*NS + a, y*NR + b], scaled as mode 3
    lrx_c2* y;
    const UnfoldTab* t;
    long long my;
    double scale;
    __device__ void put(int k, long long c, lrx_c2 v) const {
        const long long pr = c / SS, xx = pr / my, yy = pr - xx * my;
        const int a = int((c % SS) / NR), b = int(c % NR);
        y[((long long)k * t->ml + xx * NS + a) * t->nl + yy * NR + b] = lrx_c2{v.x * scale, v.y * scale};
    }
};

// live: the window rows [live[0], live[1]) of a padded pass (null: all); a row is my*SS columns.
extern "C" __global__ void __launch_bounds__(LRX_NT)
lrx_kbox_m9(const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt, lrx_c2* __restrict__ y, UnfoldTab t,
            double scale, const int* __restrict__ live) {
    extern __shared__ lrx_c2 sm[];
    const long long my = t.nl / NR, ncols = (t.ml / NS) * my * SS;
    const WedgeLoad ld{gp, gt, &t, my};
    const WedgeStore st{y, &t, my, scale};
    const lrx_kbox::Live lv = lrx_kbox::Live::rows(live, my * SS, 0, true);
    const long long ce = lv.e(ncols);
    using lrx_kbox::fft_direction;
    for (long long c0 = lv.b() + (long long)blockIdx.x * TRC; c0 < ce; c0 += (long long)gridDim.x * TRC) {
        lrx_kbox::stage_tile<NX, NY, NZ, TRC>(sm, c0, ce, ld);
        lrx_kbox::transform3<NX, NY, NZ, TRC, 0, fft_direction::inverse>(sm);
        lrx_kbox::store_tile<NX, NY, NZ, TRC>(sm, c0, ce, st);
    }
    lrx_kbox::zero_dead<NK, lrx_c2>(ncols, lv, st);
}

#elif LRX_MODE == 11
// ---- mode 11: the Green pair's load, the inverse transform, the trace into acc ----------------
static_assert(NR == NS, "mode 11 loads Greens: one spin width");
constexpr int GRP = 2 * SS;                    // columns per pair
static_assert(TRC % GRP == 0, "tiles hold whole pairs");

struct ChiArgs {                               // the host's ChiArgs (kbox_rspace_ffi.cc), field for field
    const lrx_c2 *gv, *gvt, *gc, *gct;         // parent Greens and partners (unread for conj_trs 2)
    lrx_c2* acc;                               // (n_ch * n_out, nk, mx, my), accumulated in place
    const lrx_c2* alpha;                       // (n_out,)
    long long pairs, my;
    int n_out;
    double si;                                 // the inverse transform's scale
    // LRX_VTX: vertex i of a side packed as mode 8's LorentzTab (perm entry (i, a) at bits
    // 16*i + 4*a, phase code (i, a) at bits 8*i + 2*a); channel ch = i*vnb + j writes acc planes
    // [ch*n_out, (ch+1)*n_out).
    unsigned long long vperm_l, vphase_l, vperm_r, vphase_r;
    int vna, vnb;
    const double* sgn_c;                       // LRX_VTX: (nk) real +-1 on the Gc operand's load; null: none
    const int* live;                           // the window rows [live[0], live[1]); null: all
};

struct ChiLoad {
    static constexpr bool kDirect = true, kFinish = false;
    const ChiArgs* a;
    const UnfoldTab* t;
    // Every (k, operand group) of the tile: the SS spin elements of one operand of one pair.
    template <class View>
    __device__ void direct(const View& view, int k0, int k1, long long col0, int width, long long ncols) const {
        const int groups = width / SS;
        for (int i = threadIdx.x; i < (k1 - k0) * groups; i += blockDim.x) {
            const int k = k0 + i / groups, j = i % groups;
            const long long c0 = col0 + (long long)j * SS;
            const long long pr = c0 / GRP;
            const int op = int((c0 % GRP) / SS);
            if (c0 < ncols) {
                const long long xx = pr / a->my, yy = pr - xx * a->my;
                lrx_c2 g[NS][NS], u[NS][NS], ur[NS][NS];
                lrx_unfold_pair<NS, NS>(op ? a->gc : a->gv, op ? a->gct : a->gvt, *t, k, xx, yy, g, u, ur);
#if LRX_VTX
                const double sg = (op && a->sgn_c) ? a->sgn_c[k] : 1.0;   // exact
#endif
#pragma unroll
                for (int r = 0; r < NS; ++r) {
                    lrx_c2 out[NS];
                    lrx_spin_row<NS, NS>(u, ur, g, r, out);
#pragma unroll
                    for (int b = 0; b < NS; ++b) {
#if LRX_VTX
                        out[b].x *= sg; out[b].y *= sg;
#endif
                        view(k, j * SS + r * NS + b) = out[b];
                    }
                }
            } else {
#pragma unroll
                for (int e = 0; e < SS; ++e) view(k, j * SS + e) = lrx_c2{0.0, 0.0};
            }
        }
    }
};

// The pair's R-space value from its GRP transformed values g(q) (q < SS: Gv, else Gc).
template <class Get>
__device__ __forceinline__ lrx_c2 lrx_chi_value(const Get& g, double si) {
    lrx_c2 v{0.0, 0.0};
#pragma unroll
    for (int q = 0; q < SS; ++q) {
        const lrx_c2 gv = g(q), gc = g(SS + q);
        const lrx_c2 a{gc.x * si, -(gc.y * si)};
        const lrx_c2 b{gv.x * si, gv.y * si};
        lrx_cmac(v, a, b);
    }
#if LRX_COMPLETE
    v = lrx_c2{v.x + v.x, v.y - v.y};          // v + conj(v): a real contour
#endif
    return v;
}

// acc[o, k, pair] += alpha[o] * v; acc is (n_out, nk, mx, my), a pair x*my + y.
__device__ __forceinline__ void lrx_chi_acc(const ChiArgs& a, int k, long long pr, lrx_c2 v) {
    for (int o = 0; o < a.n_out; ++o) {
        lrx_c2* e = a.acc + ((long long)o * NK + k) * a.pairs + pr;
        lrx_c2 w = *e;
        lrx_cmac(w, a.alpha[o], v);
        *e = w;
    }
}

#if LRX_VTX
// Channel ch = (i, j): sum_ab conj(ph_l[i,a]) ph_r[j,b] conj(si Gc'[pl a, pr b]) (si Gv'_ab) (a-major).
template <class Get>
__device__ __forceinline__ lrx_c2 lrx_chi_vertex(const Get& g, const ChiArgs& a, int ch) {
    const int i = ch / a.vnb, j = ch - (ch / a.vnb) * a.vnb;
    lrx_c2 v{0.0, 0.0};
#pragma unroll
    for (int p = 0; p < NS; ++p) {
        const int pa = (int)((a.vperm_l >> (16 * i + 4 * p)) & 15);
        const int ca = (int)((a.vphase_l >> (8 * i + 2 * p)) & 3);
#pragma unroll
        for (int r = 0; r < NS; ++r) {
            const int pb = (int)((a.vperm_r >> (16 * j + 4 * r)) & 15);
            const int cb = (int)((a.vphase_r >> (8 * j + 2 * r)) & 3);
            const lrx_c2 gv = g(p * NS + r), gc = g(SS + pa * NS + pb);
            const lrx_c2 w = lrx_phase(lrx_c2{gc.x * a.si, gc.y * a.si}, (ca + 4 - cb) & 3);
            const lrx_c2 x{w.x, -w.y};
            const lrx_c2 y{gv.x * a.si, gv.y * a.si};
            lrx_cmac(v, x, y);
        }
    }
    return v;
}

__device__ __forceinline__ void lrx_chi_acc_ch(const ChiArgs& a, int k, long long pr, int ch, lrx_c2 v) {
    for (int o = 0; o < a.n_out; ++o) {
        lrx_c2* e = a.acc + ((long long)(ch * a.n_out + o) * NK + k) * a.pairs + pr;
        lrx_c2 w = *e;
        lrx_cmac(w, a.alpha[o], v);
        *e = w;
    }
}
#endif

// The group Mid reduces the pair's GRP transformed values and accumulates them: one thread per
// (k, pair) of the tile, so every element of acc has one writer.
struct ChiMid {
    const ChiArgs* a;
    template <class Get>
    __device__ void group(int k, long long g, const Get& get) const {
#if LRX_VTX
        for (int ch = 0; ch < a->vna * a->vnb; ++ch) lrx_chi_acc_ch(*a, k, g, ch, lrx_chi_vertex(get, *a, ch));
#else
        lrx_chi_acc(*a, k, g, lrx_chi_value(get, a->si));
#endif
    }
};

// Grid-stride over tiles; the live columns [cb, ce) are whole pairs (a row is my*GRP columns);
// dead pairs add nothing to the accumulator.
extern "C" __global__ void __launch_bounds__(LRX_NT) lrx_kbox_m11(ChiArgs a, UnfoldTab t) {
    extern __shared__ lrx_c2 sm[];
    const long long ncols = a.pairs * GRP;
    const ChiLoad ld{&a, &t};
    const ChiMid mid{&a};
    const lrx_kbox::Live lv = lrx_kbox::Live::rows(a.live, a.my * GRP, 0, false);
    const long long cb = lv.b(), ce = lv.e(ncols);
    using lrx_kbox::fft_direction;
    for (long long col0 = cb + (long long)blockIdx.x * TRC; col0 < ce; col0 += (long long)gridDim.x * TRC) {
        lrx_kbox::stage_tile<NX, NY, NZ, TRC>(sm, col0, ce, ld);
        lrx_kbox::transform3<NX, NY, NZ, TRC, 0, fft_direction::inverse>(sm);
        lrx_kbox::mid_group_tile<NX, NY, NZ, TRC, GRP>(sm, col0, ce, mid);
    }
}
#else
#error "kbox_rspace_kernel.h: LRX_MODE must be 9 or 11"
#endif
