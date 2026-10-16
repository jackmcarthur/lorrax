// active_dev_kernel.h -- the active-subspace kernels with DEVICE-side sizes (rtc-compiled).
//
// Embedded as text (generated/active_dev_src.h) and compiled at run time by the leg's RTC
// (common/rtc_build.h: hipRTC on ROCM, NVRTC on CUDA).  Leg-neutral: no include, only the
// language features both RTCs share (extern "C" __global__, __shared__, __syncthreads,
// __longlong_as_double).  Every kernel reads its runtime window -- the int32 descriptor the
// Python plan stages on the device -- inside the kernel, so the host never reads a size and
// never synchronizes the stream (owner rule: no per-call D2H in hot paths).  The host passes
// only static extents (capacity, vector length, block width) and a fixed persistent grid;
// blocks walk a work list whose length is computed from the descriptor on the device.
//
// Layouts are the CUDA handlers' (active_ops.cc): vectors row-major (rows x d), coefficient
// panels column-major with leading dimension `cap` (element (r, j) at r + cap * j).
// An invalid descriptor writes NaN to the operation's outputs (the XLA route's contract; a
// device kernel cannot raise without a host read).
//
// Determinism: every sum has one fixed order (k-split partials reduced in split order), so a
// result is bitwise reproducible run to run.

struct lz { double x, y; };

#define LRX_NT 64          // threads of the product kernels (one wavefront on gfx90a)
#define LRX_RA 4           // A rows per product tile
#define LRX_RB 8           // B rows per product tile
#define LRX_TP (LRX_RA * LRX_RB)
#define LRX_JB 8           // coefficient columns per combine item

__device__ inline double lrx_nan() { return __longlong_as_double(0x7ff8000000000000LL); }
__device__ inline lz lrx_z(double x, double y) { lz r; r.x = x; r.y = y; return r; }
__device__ inline long long lrx_min(long long a, long long b) { return a < b ? a : b; }

// The product window.  mode 0 (Gram): A rows [q0, q0+q1), B rows [0, nbs).
// mode 1 (project, q = [m, start, r]): A rows [0, m), B rows [start, start+r), start+r == m.
struct LrxWin { int a0, na, b0, nb, ok; };
__device__ inline LrxWin lrx_window(int mode, const int* q, int cap, int nbs) {
    LrxWin w;
    if (mode == 0) {
        const int s = q[0], c = q[1];
        w.ok = s >= 0 && c >= 0 && (long long)s + c <= cap;
        w.a0 = s; w.na = c; w.b0 = 0; w.nb = nbs;
    } else {
        const int m = q[0], s = q[1], r = q[2];
        w.ok = m >= 1 && m <= cap && s >= 0 && r >= 0 && (long long)s + r == m;
        w.a0 = 0; w.na = m; w.b0 = s; w.nb = r;
    }
    if (!w.ok) { w.a0 = w.na = w.b0 = w.nb = 0; }
    return w;
}

// k-splits of one product, from device-known sizes only: enough items to fill the grid,
// at least 8 k per thread per split, and partials within the scratch budget.  1 = no
// partials: the tile result is written straight to its destination.
__device__ inline int lrx_splits(int na, int nb, long long d, long long budget_tiles, int target) {
    const long long tiles = (long long)((na + LRX_RA - 1) / LRX_RA) * ((nb + LRX_RB - 1) / LRX_RB);
    if (tiles == 0) return 0;
    long long s = (target + tiles - 1) / tiles;
    s = lrx_min(s, (d + LRX_NT * 8 - 1) / (LRX_NT * 8));
    s = lrx_min(s, budget_tiles / tiles);
    return (int)(s < 1 ? 1 : s);
}

// out[ra + cap*rb] = sum_k conj(A[ra, k]) * B[rb, k] over the window (one k-split per item).
extern "C" __global__ void lrx_as_pgemm(const lz* __restrict__ A, const lz* __restrict__ B,
                                        lz* out, lz* part, const int* q, int mode, int cap,
                                        int nbs, long long d, long long budget_tiles, int target) {
    const LrxWin w = lrx_window(mode, q, cap, nbs);
    const int nta = (w.na + LRX_RA - 1) / LRX_RA, ntb = (w.nb + LRX_RB - 1) / LRX_RB;
    const int ns = lrx_splits(w.na, w.nb, d, budget_tiles, target);
    const long long items = (long long)ns * nta * ntb;
    if (items == 0) return;
    const long long chunk = (d + ns - 1) / ns;
    __shared__ lz red[LRX_TP][LRX_NT];
    const int t = threadIdx.x;
    for (long long it = blockIdx.x; it < items; it += gridDim.x) {
        const int tb = (int)(it % ntb);
        const long long rest = it / ntb;
        const int ta = (int)(rest % nta);
        const long long sp = rest / nta;
        const long long k0 = sp * chunk, k1 = lrx_min(d, k0 + chunk);
        const int ia = w.a0 + ta * LRX_RA, ib = w.b0 + tb * LRX_RB;
        const int nra = (int)lrx_min(LRX_RA, w.na - ta * LRX_RA);
        const int nrb = (int)lrx_min(LRX_RB, w.nb - tb * LRX_RB);
        double ar[LRX_TP], ai[LRX_TP];
        for (int p = 0; p < LRX_TP; ++p) { ar[p] = 0.0; ai[p] = 0.0; }
        for (long long k = k0 + t; k < k1; k += LRX_NT) {
            lz a[LRX_RA], b[LRX_RB];
            for (int i = 0; i < LRX_RA; ++i) a[i] = i < nra ? A[(long long)(ia + i) * d + k] : lrx_z(0, 0);
            for (int j = 0; j < LRX_RB; ++j) b[j] = j < nrb ? B[(long long)(ib + j) * d + k] : lrx_z(0, 0);
            for (int i = 0; i < LRX_RA; ++i)
                for (int j = 0; j < LRX_RB; ++j) {
                    ar[i * LRX_RB + j] += a[i].x * b[j].x + a[i].y * b[j].y;
                    ai[i * LRX_RB + j] += a[i].x * b[j].y - a[i].y * b[j].x;
                }
        }
        for (int p = 0; p < LRX_TP; ++p) red[p][t] = lrx_z(ar[p], ai[p]);
        __syncthreads();
        if (t < LRX_TP) {
            double sr = 0.0, si = 0.0;
            for (int u = 0; u < LRX_NT; ++u) { sr += red[t][u].x; si += red[t][u].y; }
            const int i = t / LRX_RB, j = t % LRX_RB;
            if (ns == 1) {
                if (i < nra && j < nrb) out[(ia + i) + (long long)cap * (ib + j)] = lrx_z(sr, si);
            } else {
                part[it * LRX_TP + t] = lrx_z(sr, si);
            }
        }
        __syncthreads();
    }
}

// Sum the k-split partials in split order and write each tile entry to out[ra + cap*rb].
extern "C" __global__ void lrx_as_pgemm_reduce(lz* out, const lz* part, const int* q, int mode,
                                               int cap, int nbs, long long d,
                                               long long budget_tiles, int target) {
    const LrxWin w = lrx_window(mode, q, cap, nbs);
    const int nta = (w.na + LRX_RA - 1) / LRX_RA, ntb = (w.nb + LRX_RB - 1) / LRX_RB;
    const int ns = lrx_splits(w.na, w.nb, d, budget_tiles, target);
    if (ns <= 1) return;
    const long long n = (long long)nta * ntb * LRX_TP;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < n;
         e += (long long)gridDim.x * blockDim.x) {
        const long long tile = e / LRX_TP;
        const int p = (int)(e % LRX_TP);
        const int ta = (int)(tile / ntb), tb = (int)(tile % ntb);
        const int ra = w.a0 + ta * LRX_RA + p / LRX_RB, rb = w.b0 + tb * LRX_RB + p % LRX_RB;
        if (ra >= w.a0 + w.na || rb >= w.b0 + w.nb) continue;
        double sr = 0.0, si = 0.0;
        for (int sp = 0; sp < ns; ++sp) {
            const lz v = part[(((long long)sp * nta + ta) * ntb + tb) * LRX_TP + p];
            sr += v.x; si += v.y;
        }
        out[ra + (long long)cap * rb] = lrx_z(sr, si);
    }
}

// A Gram panel (cap x nbs, column-major): zero outside [q0, q0+q1); all NaN when invalid.
// Disjoint from the rows lrx_as_pgemm writes, so the two may run in either order.
extern "C" __global__ void lrx_as_fill_gram(lz* out, const int* q, int cap, int nbs) {
    const LrxWin w = lrx_window(0, q, cap, nbs);
    const long long n = (long long)cap * nbs;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < n;
         e += (long long)gridDim.x * blockDim.x) {
        const int r = (int)(e % cap);
        if (!w.ok) out[e] = lrx_z(lrx_nan(), lrx_nan());
        else if (r < w.a0 || r >= w.a0 + w.na) out[e] = lrx_z(0, 0);
    }
}

// Project's second half, after lrx_as_pgemm (mode 1) wrote G(i, j) = <V_i, HV_{start+j}> at
// out(i, start+j) for i < m.  The CUDA handler's two GEMMs leave
//   out(i, start+j)       = G(i, j)              for i < start,
//   out(start+j, i)       = conj(G(i, j))        for i < m (the second GEMM wins the overlap),
// so rows [start, m) x cols [0, start) are the conjugate transpose of the panel above them and
// the overlap block [start, m)^2 becomes the conjugate transpose of itself.  Each output entry
// has one writer.  Invalid descriptor: the whole matrix is NaN.
extern "C" __global__ void lrx_as_project_mirror(lz* out, const int* q, int cap) {
    const int m = q[0], s = q[1], r = q[2];
    const bool ok = m >= 1 && m <= cap && s >= 0 && r >= 0 && (long long)s + r == m;
    const long long stride = (long long)gridDim.x * blockDim.x;
    const long long e0 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (!ok) {
        for (long long e = e0; e < (long long)cap * cap; e += stride) out[e] = lrx_z(lrx_nan(), lrx_nan());
        return;
    }
    const long long n1 = (long long)r * s, n2 = (long long)r * (r + 1) / 2;
    for (long long e = e0; e < n1 + n2; e += stride) {
        if (e < n1) {
            const int i = (int)(e % s), j = (int)(e / s);
            const lz g = out[i + (long long)cap * (s + j)];
            out[(s + j) + (long long)cap * i] = lrx_z(g.x, -g.y);
        } else {
            const long long f = e - n1;                  // (a, c), a <= c: c(c+1)/2 + a
            int c = (int)((sqrt(8.0 * (double)f + 1.0) - 1.0) * 0.5);
            while ((long long)c * (c + 1) / 2 > f) --c;
            while ((long long)(c + 1) * (c + 2) / 2 <= f) ++c;
            const int a = (int)(f - (long long)c * (c + 1) / 2);
            const long long x = (s + a) + (long long)cap * (s + c), y = (s + c) + (long long)cap * (s + a);
            const lz vx = out[x], vy = out[y];
            out[x] = lrx_z(vy.x, -vy.y);
            out[y] = lrx_z(vx.x, -vx.y);
        }
    }
}

// X[j, k] (=|-=) sum_{r in window} C[r + cap*j] * V[r, k], j < nb (row-major X, nb x d).
//   dk 0: q = [start, count]                     (CGS2, subtraction)
//   dk 1: q = [m, columns, compute_image, start] (reconstruction)
//   mode 0: overwrite, columns >= `columns` zero; mode 1: subtract; mode 2: as 0, and all zero
//   when q[2] == 0 (the image of a reconstruction that does not want it).
extern "C" __global__ void lrx_as_combine(lz* X, const lz* __restrict__ V, const lz* __restrict__ C,
                                          const int* q, int dk, int mode, int cap, int nb, long long d) {
    int s, n, cols;
    bool ok, zero = false;
    if (dk == 0) {
        s = q[0]; n = q[1]; cols = nb;
        ok = s >= 0 && n >= 0 && (long long)s + n <= cap;
    } else {
        n = q[0]; cols = q[1]; s = q[3];
        ok = n >= 0 && s >= 0 && (long long)s + n <= cap && cols >= 0 && cols <= nb;
        zero = mode == 2 && q[2] == 0;
    }
    const int ntj = (nb + LRX_JB - 1) / LRX_JB;
    const long long nkb = (d + LRX_NT - 1) / LRX_NT, items = nkb * ntj;
    for (long long it = blockIdx.x; it < items; it += gridDim.x) {
        const int tj = (int)(it % ntj);
        const long long k = (it / ntj) * LRX_NT + threadIdx.x;
        if (k >= d) continue;
        double ar[LRX_JB], ai[LRX_JB];
        for (int j = 0; j < LRX_JB; ++j) { ar[j] = 0.0; ai[j] = 0.0; }
        const int jn = (int)lrx_min(LRX_JB, nb - tj * LRX_JB);
        if (ok && !zero) {
            for (int r = s; r < s + n; ++r) {
                const lz v = V[(long long)r * d + k];
                for (int j = 0; j < LRX_JB; ++j) {
                    if (j >= jn) break;
                    const lz c = C[r + (long long)cap * (tj * LRX_JB + j)];
                    ar[j] += c.x * v.x - c.y * v.y;
                    ai[j] += c.x * v.y + c.y * v.x;
                }
            }
        }
        for (int j = 0; j < jn; ++j) {
            const int jj = tj * LRX_JB + j;
            lz* x = X + (long long)jj * d + k;
            if (!ok) *x = lrx_z(lrx_nan(), lrx_nan());
            else if (mode == 1) *x = lrx_z(x->x - ar[j], x->y - ai[j]);
            else *x = (jj < cols && !zero) ? lrx_z(ar[j], ai[j]) : lrx_z(0, 0);
        }
    }
}

// V[start + i] = P[i], HV[start + i] = HP[i] for i < count; invalid: V and HV all NaN.
extern "C" __global__ void lrx_as_store(lz* V, lz* HV, const lz* __restrict__ P,
                                        const lz* __restrict__ HP, const int* q, int cap, int pb,
                                        long long d) {
    const int s = q[0], c = q[1];
    const bool ok = s >= 0 && c >= 0 && c <= pb && (long long)s + c <= cap;
    const long long stride = (long long)gridDim.x * blockDim.x;
    const long long e0 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (!ok) {
        for (long long e = e0; e < (long long)cap * d; e += stride) {
            V[e] = lrx_z(lrx_nan(), lrx_nan());
            HV[e] = lrx_z(lrx_nan(), lrx_nan());
        }
        return;
    }
    const long long n = (long long)c * d, o = (long long)s * d;
    for (long long e = e0; e < n; e += stride) { V[o + e] = P[e]; HV[o + e] = HP[e]; }
}
