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

#define LRX_ACTIVE_GEMM_ENTRY(name, T)                                                        \
    extern "C" __global__ void __launch_bounds__(LRX_NT) name(                                \
        const T* A, const T* B, const T* W, const int* bounds, int bstride, int nb, T* C,     \
        int m, int k, int n, double alpha_re, double alpha_im, double beta_re,                \
        double beta_im, int read_c) {                                                         \
        lrx_active_gemm<T>(A, B, W, bounds, bstride, nb, C, m, k, n, alpha_re, alpha_im,      \
                           beta_re, beta_im, read_c);                                         \
    }

LRX_ACTIVE_GEMM_ENTRY(lrx_active_gemm_z, lrx_z)
LRX_ACTIVE_GEMM_ENTRY(lrx_active_gemm_d, double)
