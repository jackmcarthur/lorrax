// active_gemm_kernel.h -- the device-bounds active-range local GEMM, compiled at run time by
// the leg's RTC (common/rtc_build.h: hipRTC on ROCM, NVRTC on CUDA).  No include, no vendor
// spelling: the same text builds on both legs.
//
//   C[q] = alpha * A[q][:, lo:hi] diag(w[q][lo:hi]) B[q][lo:hi, :] + beta * C[q]
//
// A (nq, m, K), B (nq, K, n), C (nq, m, n) and w (nq, K) are row-major; bounds is int32
// (nb, 2), nb = 1 (one pair for every row, bstride 0) or nq (bstride 2).  The bounds are read
// HERE, on the device, by every block: no host read, no stream sync and no device control
// flow in the caller.  A block's K loop runs over [lo, hi) only, so the FLOPs are those of the
// active width and no inactive column of A, row of B or entry of w is ever loaded (NaN or
// garbage there never enters arithmetic: the masked route's contract without its operand
// copies).  W == nullptr: unweighted.  Invalid bounds (any pair outside 0 <= lo <= hi <= K)
// give an all-NaN C (the guarded arm's contract).  read_c == 0: C is written, never read
// (beta 0, or the no-C binding: every row is written, empty intervals as zeros).
//
// Tiling: a TM x TN tile of C per block of LRX_NT = 256 threads (16 x 16, 4 x 4 outputs each:
// rows ty + 16 r, columns tx + 16 c); the interval in TK-deep LDS slabs zero-filled past hi.
// LDS: 2 * TK * (TM + 1) elements = 16.6 KiB at TK 8 (complex128), so three blocks share a
// gfx90a CU's 64 KiB (TK 16, 33.3 KiB, allowed one: Fe W shape 2.7 ms vs rocBLAS 0.8 ms at
// full K); under the CUDA leg's 48 KiB static limit.  Grid (ceil(n/TN), ceil(m/TM), nq).
#ifndef LRX_TM
#define LRX_TM 64
#endif
#ifndef LRX_TN
#define LRX_TN 64
#endif
#ifndef LRX_TK
#define LRX_TK 8
#endif
#define LRX_NT 256
#ifndef LRX_MFMA
#define LRX_MFMA 0
#endif

struct alignas(16) lrx_z { double re, im; };

__device__ __forceinline__ double lrx_zero(double) { return 0.0; }
__device__ __forceinline__ lrx_z lrx_zero(lrx_z) { lrx_z z; z.re = 0.0; z.im = 0.0; return z; }
__device__ __forceinline__ double lrx_scalar(double re, double, double) { return re; }
__device__ __forceinline__ lrx_z lrx_scalar(double re, double im, lrx_z) {
    lrx_z z; z.re = re; z.im = im; return z;
}
__device__ __forceinline__ double lrx_mul(double a, double b) { return a * b; }
__device__ __forceinline__ lrx_z lrx_mul(lrx_z a, lrx_z b) {
    lrx_z z; z.re = a.re * b.re - a.im * b.im; z.im = a.re * b.im + a.im * b.re; return z;
}
__device__ __forceinline__ double lrx_add(double a, double b) { return a + b; }
__device__ __forceinline__ lrx_z lrx_add(lrx_z a, lrx_z b) {
    lrx_z z; z.re = a.re + b.re; z.im = a.im + b.im; return z;
}
__device__ __forceinline__ void lrx_mac(double& acc, double a, double b) { acc += a * b; }
__device__ __forceinline__ void lrx_mac(lrx_z& acc, lrx_z a, lrx_z b) {
    acc.re += a.re * b.re; acc.re -= a.im * b.im; acc.im += a.re * b.im; acc.im += a.im * b.re;
}
__device__ __forceinline__ double lrx_qnan() {
    union { long long i; double d; } u; u.i = 0x7ff8000000000000LL; return u.d;
}
__device__ __forceinline__ double lrx_nan(double) { return lrx_qnan(); }
__device__ __forceinline__ lrx_z lrx_nan(lrx_z) { lrx_z z; z.re = lrx_qnan(); z.im = lrx_qnan(); return z; }

template <typename T>
__device__ __forceinline__ void lrx_active_gemm(
    const T* __restrict__ A, const T* __restrict__ B, const T* __restrict__ W,
    const int* __restrict__ bounds, int bstride, int nb, T* C, int m, int k, int n,
    double alpha_re, double alpha_im, double beta_re, double beta_im, int read_c) {
    __shared__ T As[LRX_TK][LRX_TM + 1];
    __shared__ T Bs[LRX_TK][LRX_TN + 1];
    const T zero = lrx_zero(T());
    const int t = threadIdx.x, tx = t & 15, ty = t >> 4;
    const long long q = blockIdx.z;
    const int row0 = blockIdx.y * LRX_TM, col0 = blockIdx.x * LRX_TN;

    // The all-NaN contract: every pair is checked (nb <= nq; uniform per block).
    bool valid = true;
    for (int p = 0; p < nb; ++p) {
        const int l = bounds[2 * p], h = bounds[2 * p + 1];
        valid = valid && l >= 0 && l <= h && h <= k;
    }
    const int lo = valid ? bounds[q * bstride] : 0;
    const int hi = valid ? bounds[q * bstride + 1] : 0;

    T acc[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = zero;

    const T* Aq = A + q * (long long)m * k;
    const T* Bq = B + q * (long long)k * n;
    const T* Wq = W ? W + q * (long long)k : nullptr;
    for (int k0 = lo; k0 < hi; k0 += LRX_TK) {             // uniform per block
        // A slab (TM rows x TK columns, weighted on load) and B slab (TK rows x TN columns):
        // consecutive threads take consecutive columns (coalesced); past hi or the edge: zero.
        for (int e = t; e < LRX_TM * LRX_TK; e += LRX_NT) {
            const int r = e / LRX_TK, kc = e - r * LRX_TK, i = row0 + r, kk = k0 + kc;
            T v = zero;
            if (i < m && kk < hi) {
                v = Aq[(long long)i * k + kk];
                if (Wq) v = lrx_mul(v, Wq[kk]);
            }
            As[kc][r] = v;
        }
        for (int e = t; e < LRX_TK * LRX_TN; e += LRX_NT) {
            const int kr = e / LRX_TN, jc = e - kr * LRX_TN, jj = col0 + jc, kk = k0 + kr;
            T v = zero;
            if (kk < hi && jj < n) v = Bq[(long long)kk * n + jj];
            Bs[kr][jc] = v;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < LRX_TK; ++kk) {
            T a[4], b[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) a[r] = As[kk][ty + 16 * r];
#pragma unroll
            for (int c = 0; c < 4; ++c) b[c] = Bs[kk][tx + 16 * c];
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < 4; ++c) lrx_mac(acc[r][c], a[r], b[c]);
        }
        __syncthreads();
    }

    const T alpha = lrx_scalar(alpha_re, alpha_im, T());
    const T beta = lrx_scalar(beta_re, beta_im, T());
    T* Cq = C + q * (long long)m * n;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int i = row0 + ty + 16 * r;
        if (i >= m) continue;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int j = col0 + tx + 16 * c;
            if (j >= n) continue;
            const long long o = (long long)i * n + j;
            T v;
            if (!valid) {
                v = lrx_nan(T());
            } else {
                v = lrx_mul(alpha, acc[r][c]);
                if (read_c) v = lrx_add(v, lrx_mul(beta, Cq[o]));
            }
            Cq[o] = v;
        }
    }
}


// ---------------------------------------------------------------------------------------------
// LRX_MFMA 1 (ROCM leg, gfx90a / gfx94x; the host sets it from the device's target id): the same
// slabs, bounds, validity and epilogue, with the inner product on the FP64 matrix cores.  One
// v_mfma_f64_16x16x4f64 is D(16x16) += A(16x4) B(4x16) for a wavefront: lane l holds
// D[4(l/16)+r][l%16] (r = 0..3), supplies B[l/16][l%16], and supplies A[rho(l%16)][l/16] with
// rho(x) = 4 (x%4) + x/4 in D's row numbering -- MEASURED on MI250X with one-hot operands
// (mfma_probe, wave 8), not the naive A[l%16][l/16].  The block's four wavefronts tile C 2 x 2:
// 32 x 32 each (2 x 2 MFMA tiles, block 64 x 64) or 16 x 16 each (one tile, block 32 x 32).
// complex128 is the exact 4M form, four real MFMAs per tile:
//   Dre += Are Bre + (-Aim) Bim,  Dim += Are Bim + Aim Bre
// (negation is exact, so the rounding is that of the FMA path's four products; no 3M form).
// CUDA (sm_80+): the analogue is DMMA, mma.sync.aligned.m8n8k4.row.col.f64 -- a warp's lane l
// supplies A[l/4][l%4], B[l%4][l/4] and holds C[l/4][2(l%4)+{0,1}]; a 64 x 64 block tile is
// 8 warps x (8 x 8 DMMA tiles 4 x 2).  It is not written here because it cannot be tested
// (no CUDA GPU): the CUDA leg runs the FMA path above until a DMMA arm is gated on an A100.
#if LRX_MFMA
#if !defined(__AMDGCN__)
#error "LRX_MFMA is the AMDGPU matrix-core path (hipRTC on gfx90a / gfx94x only)"
#endif
#if (LRX_TK % 8) != 0 || LRX_NT != 256
#error "the MFMA path is laid out for 256 threads and TK a multiple of 8"
#endif
typedef double lrx_d4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ lrx_d4 lrx_mfma(double a, double b, lrx_d4 c) {
    return __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, 0, 0);
}
// One k4 step for a wave's NT x NT MFMA tiles into accumulator set kp.  Real: one MFMA per
// tile; complex: four (exact 4M form), issued in four passes over the tiles.
template <int NT, int KP>
__device__ __forceinline__ void lrx_mfma_step(lrx_d4 (&cr)[KP][NT][NT], lrx_d4 (&)[KP][NT][NT], int kp,
                                              const double (&a)[NT], const double (&b)[NT]) {
#pragma unroll
    for (int rt = 0; rt < NT; ++rt)
#pragma unroll
        for (int ct = 0; ct < NT; ++ct) cr[kp][rt][ct] = lrx_mfma(a[rt], b[ct], cr[kp][rt][ct]);
}
template <int NT, int KP>
__device__ __forceinline__ void lrx_mfma_step(lrx_d4 (&cr)[KP][NT][NT], lrx_d4 (&ci)[KP][NT][NT], int kp,
                                              const lrx_z (&a)[NT], const lrx_z (&b)[NT]) {
#pragma unroll
    for (int rt = 0; rt < NT; ++rt)
#pragma unroll
        for (int ct = 0; ct < NT; ++ct) cr[kp][rt][ct] = lrx_mfma(a[rt].re, b[ct].re, cr[kp][rt][ct]);
#pragma unroll
    for (int rt = 0; rt < NT; ++rt)
#pragma unroll
        for (int ct = 0; ct < NT; ++ct) ci[kp][rt][ct] = lrx_mfma(a[rt].re, b[ct].im, ci[kp][rt][ct]);
#pragma unroll
    for (int rt = 0; rt < NT; ++rt)
#pragma unroll
        for (int ct = 0; ct < NT; ++ct) cr[kp][rt][ct] = lrx_mfma(-a[rt].im, b[ct].im, cr[kp][rt][ct]);
#pragma unroll
    for (int rt = 0; rt < NT; ++rt)
#pragma unroll
        for (int ct = 0; ct < NT; ++ct) ci[kp][rt][ct] = lrx_mfma(a[rt].im, b[ct].re, ci[kp][rt][ct]);
}
__device__ __forceinline__ double lrx_acc(double re, double, double) { return re; }
__device__ __forceinline__ lrx_z lrx_acc(double re, double im, lrx_z) { lrx_z z; z.re = re; z.im = im; return z; }

// WT: the wave tile (32: 2 x 2 MFMA tiles, block 64 x 64; 16: one MFMA tile, block 32 x 32).
// The host picks per call from (m, n): Fe's W synthesis has m = n = 156, which 64-row tiles
// cover as 192 (34 % of the MFMA work is padding) and 32-row tiles as 160 (2.5 %).  WT 16 keeps
// two accumulator sets that alternate over k4 steps, so a wave still issues four independent
// MFMA chains; they are summed in the epilogue (a different, still exact-product, summation
// order: within the 1e-13 gate).
template <typename T, int WT>
__device__ __forceinline__ void lrx_active_gemm_mfma(
    const T* __restrict__ A, const T* __restrict__ B, const T* __restrict__ W,
    const int* __restrict__ bounds, int bstride, int nb, T* C, int m, int k, int n,
    double alpha_re, double alpha_im, double beta_re, double beta_im, int read_c) {
    constexpr int NT = WT / 16, KP = NT == 1 ? 2 : 1, BM = 2 * WT, BN = 2 * WT;
    __shared__ T As[LRX_TK][BM + 1];
    __shared__ T Bs[LRX_TK][BN + 1];
    const T zero = lrx_zero(T());
    const int t = threadIdx.x, lane = t & 63, wave = t >> 6;
    const int wr = wave >> 1, wc = wave & 1, li = lane & 15, lk = lane >> 4;
    // The A operand's row for this lane, in D's row numbering (measured on gfx90a, see above).
    const int arow = ((li & 3) << 2) | (li >> 2);
    const long long q = blockIdx.z;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    bool valid = true;
    for (int p = 0; p < nb; ++p) {
        const int l = bounds[2 * p], h = bounds[2 * p + 1];
        valid = valid && l >= 0 && l <= h && h <= k;
    }
    const int lo = valid ? bounds[q * bstride] : 0;
    const int hi = valid ? bounds[q * bstride + 1] : 0;
    lrx_d4 cr[KP][NT][NT], ci[KP][NT][NT];
#pragma unroll
    for (int kp = 0; kp < KP; ++kp)
#pragma unroll
        for (int rt = 0; rt < NT; ++rt)
#pragma unroll
            for (int ct = 0; ct < NT; ++ct)
#pragma unroll
                for (int r = 0; r < 4; ++r) { cr[kp][rt][ct][r] = 0.0; ci[kp][rt][ct][r] = 0.0; }
    const T* Aq = A + q * (long long)m * k;
    const T* Bq = B + q * (long long)k * n;
    const T* Wq = W ? W + q * (long long)k : nullptr;
    // Register prefetch: the next K slab's guarded global loads are issued before this slab's
    // MFMAs, so their latency overlaps the matrix work (no second LDS buffer, no workspace).
    constexpr int EA = BM * LRX_TK / LRX_NT, EB = LRX_TK * BN / LRX_NT;
    static_assert(EA * LRX_NT == BM * LRX_TK && EB * LRX_NT == LRX_TK * BN, "slab / thread split");
    T ra[EA], rb[EB];
    auto fetch = [&](int k0) {
#pragma unroll
        for (int x = 0; x < EA; ++x) {
            const int e = t + x * LRX_NT;
            const int r = e / LRX_TK, kc = e - r * LRX_TK, i = row0 + r, kk = k0 + kc;
            T v = zero;
            if (i < m && kk < hi) {
                v = Aq[(long long)i * k + kk];
                if (Wq) v = lrx_mul(v, Wq[kk]);
            }
            ra[x] = v;
        }
#pragma unroll
        for (int x = 0; x < EB; ++x) {
            const int e = t + x * LRX_NT;
            const int kr = e / BN, jc = e - kr * BN, jj = col0 + jc, kk = k0 + kr;
            T v = zero;
            if (kk < hi && jj < n) v = Bq[(long long)kk * n + jj];
            rb[x] = v;
        }
    };
    if (lo < hi) fetch(lo);
    for (int k0 = lo; k0 < hi; k0 += LRX_TK) {             // uniform per block
#pragma unroll
        for (int x = 0; x < EA; ++x) {
            const int e = t + x * LRX_NT, r = e / LRX_TK;
            As[e - r * LRX_TK][r] = ra[x];
        }
#pragma unroll
        for (int x = 0; x < EB; ++x) {
            const int e = t + x * LRX_NT, kr = e / BN;
            Bs[kr][e - kr * BN] = rb[x];
        }
        __syncthreads();
        if (k0 + LRX_TK < hi) fetch(k0 + LRX_TK);
#pragma unroll
        for (int ks = 0; ks < LRX_TK; ks += 4) {
            T a[NT], b[NT];
#pragma unroll
            for (int rt = 0; rt < NT; ++rt) a[rt] = As[ks + lk][wr * WT + rt * 16 + arow];
#pragma unroll
            for (int ct = 0; ct < NT; ++ct) b[ct] = Bs[ks + lk][wc * WT + ct * 16 + li];
            lrx_mfma_step<NT, KP>(cr, ci, (ks / 4) % KP, a, b);
        }
        __syncthreads();
    }
    const T alpha = lrx_scalar(alpha_re, alpha_im, T());
    const T beta = lrx_scalar(beta_re, beta_im, T());
    T* Cq = C + q * (long long)m * n;
#pragma unroll
    for (int rt = 0; rt < NT; ++rt)
#pragma unroll
        for (int ct = 0; ct < NT; ++ct) {
            const int j = col0 + wc * WT + ct * 16 + li;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int i = row0 + wr * WT + rt * 16 + 4 * lk + r;
                if (i >= m || j >= n) continue;
                double re = 0.0, im = 0.0;
#pragma unroll
                for (int kp = 0; kp < KP; ++kp) { re += cr[kp][rt][ct][r]; im += ci[kp][rt][ct][r]; }
                const long long o = (long long)i * n + j;
                T v;
                if (!valid) {
                    v = lrx_nan(T());
                } else {
                    v = lrx_mul(alpha, lrx_acc(re, im, T()));
                    if (read_c) v = lrx_add(v, lrx_mul(beta, Cq[o]));
                }
                Cq[o] = v;
            }
        }
}
#define LRX_GEMM_BODY(T) lrx_active_gemm_mfma<T, 32>
#else
#define LRX_GEMM_BODY(T) lrx_active_gemm<T>
#endif

#define LRX_ACTIVE_GEMM_ENTRY(name, T, BODY)                                                  \
    extern "C" __global__ void __launch_bounds__(LRX_NT) name(                                \
        const T* A, const T* B, const T* W, const int* bounds, int bstride, int nb, T* C,     \
        int m, int k, int n, double alpha_re, double alpha_im, double beta_re,                \
        double beta_im, int read_c) {                                                         \
        BODY(A, B, W, bounds, bstride, nb, C, m, k, n, alpha_re, alpha_im, beta_re, beta_im,  \
             read_c);                                                                         \
    }

// 64 x 64 block tile (both paths).
LRX_ACTIVE_GEMM_ENTRY(lrx_active_gemm_z, lrx_z, LRX_GEMM_BODY(lrx_z))
LRX_ACTIVE_GEMM_ENTRY(lrx_active_gemm_d, double, LRX_GEMM_BODY(double))
#if LRX_MFMA
// 32 x 32 block tile (MFMA path only; the host requests it where it pads less).
LRX_ACTIVE_GEMM_ENTRY(lrx_active_gemm_z32, lrx_z, (lrx_active_gemm_mfma<lrx_z, 16>))
LRX_ACTIVE_GEMM_ENTRY(lrx_active_gemm_d32, double, (lrx_active_gemm_mfma<double, 16>))
#endif
