// plane_gather_kernel.h -- DEVICE SOURCE, run-time compiled by the leg's RTC (common/rtc_build.h); CMake
// embeds it as text (plane_gather_src.h.in) beside the k-box stage's text (cufft/kbox_stage_src.h.in),
// which hipRTC reads as the named header "kbox_stage.cuh" (for its direct-DFT twiddles only).
//
// Mode 10 of docs/architecture/kconv.md (ffi.fft.make_plane_fft_gather, LocalFourierPlan(in_gather=...):
// the route-G plane transform of the zeta fit), the ROCM port of kconv_mathdx_cuda_ffi.cc's kPlaneSrc:
//
//   Y[..., kb, kc] = sum_{b,c} plane[..., b, c] e^{-2 pi i (b kb / n_b + c kc / n_c)},
//   plane = the cylinder F (..., n_col) scattered to its static cells, zero elsewhere,
//
// without writing the zero plane.  The CUDA kernel's design: a block keeps PB whole (n_b, n_c|1) planes
// in LDS; it gathers the occupied rows' cells (gidx, row_of), runs the row transforms on those rows
// only, the column transforms on every column with dead rows read as zero, and stores each plane once,
// coalesced.  An axis n = n1 n2 with gcd(n1, n2) = 1 runs as the Good-Thomas n1 x n2 DFT (index maps,
// no twiddles; ffi.fft.plane_fft_split picks the split), each factor line (<= 40) one register
// Cooley-Tukey step of direct DFTs against the k-box stage's compile-time twiddles (line_fft below).  Two differences from CUDA:
// gfx90a has no asynchronous global -> LDS copy in this subset, so the gather is synchronous (the
// CUDA kernel's cp.async staging is not ported), and the grid is not persistent (grid-stride).
//
// Compile-time constants (keyed): LRX_NB, LRX_NC, LRX_B1, LRX_C1 (splits), LRX_ROWS (occupied rows),
// LRX_PB (planes per block), LRX_NT (threads).
#include "kbox_stage.cuh"

struct alignas(16) lrx_c2 { double x, y; };
struct PlaneGather {                    // the host's PlaneGather (plane_gather_ffi.cc), field for field
    const int* gidx;                    // (rows, n_c): cylinder column of each cell of an occupied row, -1 = empty
    const int* row_of;                  // (rows,): the plane row b of occupied row r
    const int* start;                   // () slab start on F's axis 1
    long long rows, n_col, planes, s_len, n_pg, inner;
};

constexpr int NB = LRX_NB, NC = LRX_NC;
constexpr int B1 = LRX_B1, B2 = NB / B1, C1 = LRX_C1, C2 = NC / C1;
constexpr int LD = NC | 1;              // odd row pitch: the column lines read conflict-free
constexpr int ROWS = LRX_ROWS;
constexpr int PB = LRX_PB;
constexpr int PS = NB * LD, GS = ROWS * NC;

// Good-Thomas output slot of frequency k (kconv_mathdx_cuda_ffi.cc slot()).
template <int N, int N1, int N2>
__device__ __forceinline__ int slot(int k) { return (N2 * (k % N1) + N1 * (k % N2)) % N; }

// The factor-line transform, forward, in registers: for M = R Q with R the largest divisor <= sqrt(M)
// (R = 1: M prime, the direct DFT) one Cooley-Tukey step, R direct Q-point DFTs of x[R q + r], the
// twiddles w_M^(r k2), then Q direct R-point DFTs: X[k2 + Q k1] = sum_r w_R^(r k1) w_M^(r k2) Y_r[k2].
// M (R + Q) complex products instead of M^2 (25: 250 for 625, 27: 324 for 729, 32: 384 for 1024); every
// twiddle a compile-time constant of the k-box stage (lrx_kbox::dft::Tw), as cuFFTDx's thread FFT
// specialises its line at JIT time on CUDA.
template <int M>
struct LineSplit {
    static constexpr int pick() { int r = 1; for (int d = 2; d * d <= M; ++d) if (M % d == 0) r = d; return r; }
    static constexpr int R = pick(), Q = M / R;
};
template <int M>
__device__ __forceinline__ void line_fft(double* ix, double* iy) {
    namespace dft = lrx_kbox::dft;
    constexpr int R = LineSplit<M>::R, Q = LineSplit<M>::Q;
    if constexpr (R == 1) {
        double ox[M], oy[M];
        dft::For<0, M>::run([&](auto m) {
            constexpr int Mo = decltype(m)::value;
            double ax = 0, ay = 0;
            dft::For<0, M>::run([&](auto j) {
                constexpr int J = decltype(j)::value;
                constexpr double wx = dft::Tw<M, (Mo * J) % M>::c, wy = -dft::Tw<M, (Mo * J) % M>::s;
                ax = fma(ix[J], wx, fma(-iy[J], wy, ax));
                ay = fma(ix[J], wy, fma(iy[J], wx, ay));
            });
            ox[Mo] = ax;
            oy[Mo] = ay;
        });
#pragma unroll
        for (int e = 0; e < M; ++e) { ix[e] = ox[e]; iy[e] = oy[e]; }
    } else {
        double yx[M], yy[M];                     // Y_r[k2] at r Q + k2, twiddled
        dft::For<0, R>::run([&](auto r_) {
            constexpr int Rr = decltype(r_)::value;
            dft::For<0, Q>::run([&](auto k_) {
                constexpr int K2 = decltype(k_)::value;
                double ax = 0, ay = 0;
                dft::For<0, Q>::run([&](auto q_) {
                    constexpr int Qq = decltype(q_)::value;
                    constexpr double wx = dft::Tw<Q, (Qq * K2) % Q>::c, wy = -dft::Tw<Q, (Qq * K2) % Q>::s;
                    ax = fma(ix[R * Qq + Rr], wx, fma(-iy[R * Qq + Rr], wy, ax));
                    ay = fma(ix[R * Qq + Rr], wy, fma(iy[R * Qq + Rr], wx, ay));
                });
                if constexpr ((Rr * K2) % M != 0) {
                    constexpr double tx = dft::Tw<M, (Rr * K2) % M>::c, ty = -dft::Tw<M, (Rr * K2) % M>::s;
                    const double bx = ax * tx - ay * ty, by = ax * ty + ay * tx;
                    ax = bx;
                    ay = by;
                }
                yx[Rr * Q + K2] = ax;
                yy[Rr * Q + K2] = ay;
            });
        });
        dft::For<0, Q>::run([&](auto k_) {
            constexpr int K2 = decltype(k_)::value;
            dft::For<0, R>::run([&](auto m_) {
                constexpr int K1 = decltype(m_)::value;
                double ax = 0, ay = 0;
                dft::For<0, R>::run([&](auto r_) {
                    constexpr int Rr = decltype(r_)::value;
                    constexpr double wx = dft::Tw<R, (Rr * K1) % R>::c, wy = -dft::Tw<R, (Rr * K1) % R>::s;
                    ax = fma(yx[Rr * Q + K2], wx, fma(-yy[Rr * Q + K2], wy, ax));
                    ay = fma(yx[Rr * Q + K2], wy, fma(yy[Rr * Q + K2], wx, ay));
                });
                ix[K2 + Q * K1] = ax;
                iy[K2 + Q * K1] = ay;
            });
        });
    }
}

// One factor pass on one line: the M elements at positions (step e + off) mod N (element stride es) of
// `base`, transformed forward in place by line_fft.  FIRST: M = N1, step = N2, off = N1 i; else
// M = N2, step = N1, off = N2 i.  `live` (the first column pass only) reads a dead plane row as zero.
template <int N, int N1, int N2, bool FIRST>
__device__ __forceinline__ void pfa_line(lrx_c2* base, int es, int i, const unsigned char* live) {
    constexpr int M = FIRST ? N1 : N2;
    if constexpr (M > 1) {
        constexpr int step = FIRST ? N2 : N1;
        const int off = (FIRST ? N1 : N2) * i;
        double ix[M], iy[M];
#pragma unroll
        for (int e = 0; e < M; ++e) {
            const int p = (step * e + off) % N;
            lrx_c2 z{0.0, 0.0};
            if (live == nullptr || live[p]) z = base[p * es];
            ix[e] = z.x;
            iy[e] = z.y;
        }
        line_fft<M>(ix, iy);
#pragma unroll
        for (int e = 0; e < M; ++e) base[((step * e + off) % N) * es] = lrx_c2{ix[e], iy[e]};
    }
}

extern "C" __global__ void __launch_bounds__(LRX_NT)
lrx_plane_fft_gather(const lrx_c2* __restrict__ fin, lrx_c2* __restrict__ yout, PlaneGather g) {
    extern __shared__ lrx_c2 buf[];                  // PB planes of (NB, LD)
    __shared__ unsigned char live[NB];
    __shared__ int rowb[ROWS];
    __shared__ long long foff[PB];
    const int rows = static_cast<int>(g.rows);       // <= ROWS
    for (int b = threadIdx.x; b < NB; b += blockDim.x) live[b] = 0;
    __syncthreads();
    for (int r = threadIdx.x; r < rows; r += blockDim.x) { rowb[r] = g.row_of[r]; live[g.row_of[r]] = 1; }
    // Output plane (a, j, r) of Y (A, n_pg, inner) reads F plane (a, start + j, r) of F (A, s_len, inner):
    // the slab F[:, start:start+n_pg] in place (start clamped as lax.dynamic_slice clamps).
    long long st = *g.start;
    st = st < 0 ? 0 : (st > g.s_len - g.n_pg ? g.s_len - g.n_pg : st);
    const long long groups = (g.planes + PB - 1) / PB;
    for (long long gi = blockIdx.x; gi < groups; gi += gridDim.x) {
        const long long p0 = gi * PB;
        const int np = g.planes - p0 < PB ? static_cast<int>(g.planes - p0) : PB;
        if (threadIdx.x < np) {
            const long long plane = p0 + threadIdx.x;
            const long long a = plane / (g.n_pg * g.inner), rem = plane - a * g.n_pg * g.inner;
            foff[threadIdx.x] = ((a * g.s_len + st) * g.inner + rem) * g.n_col;
        }
        __syncthreads();
        // The gather: the occupied rows' cells (an empty cell as zero); dead rows are never read.
        for (int t = threadIdx.x; t < np * GS; t += blockDim.x) {
            const int q = t / GS, u = t - q * GS, r = u / NC;
            if (r >= rows) continue;
            const int col = g.gidx[u];
            buf[q * PS + rowb[r] * LD + (u - r * NC)] = col >= 0 ? fin[foff[q] + col] : lrx_c2{0.0, 0.0};
        }
        __syncthreads();
        // Row transforms (along c) on the occupied rows only.
        for (int l = threadIdx.x; l < np * ROWS * C2; l += blockDim.x) {
            const int q = l / (ROWS * C2), u = l - q * ROWS * C2, r = u / C2;
            if (r >= rows) continue;
            pfa_line<NC, C1, C2, true>(buf + q * PS + rowb[r] * LD, 1, u % C2, nullptr);
        }
        __syncthreads();
        if constexpr (C2 > 1) {
            for (int l = threadIdx.x; l < np * ROWS * C1; l += blockDim.x) {
                const int q = l / (ROWS * C1), u = l - q * ROWS * C1, r = u / C1;
                if (r >= rows) continue;
                pfa_line<NC, C1, C2, false>(buf + q * PS + rowb[r] * LD, 1, u % C1, nullptr);
            }
            __syncthreads();
        }
        // Column transforms (along b) on every column; a dead row loads as zero.
        for (int l = threadIdx.x; l < np * NC * B2; l += blockDim.x) {
            const int q = l / (NC * B2), u = l - q * NC * B2;
            pfa_line<NB, B1, B2, true>(buf + q * PS + u % NC, LD, u / NC, live);
        }
        __syncthreads();
        if constexpr (B2 > 1) {
            for (int l = threadIdx.x; l < np * NC * B1; l += blockDim.x) {
                const int q = l / (NC * B1), u = l - q * NC * B1;
                pfa_line<NB, B1, B2, false>(buf + q * PS + u % NC, LD, u / NC, nullptr);
            }
            __syncthreads();
        }
        // The planes in natural (kb, kc) order, one coalesced store.
        lrx_c2* __restrict__ y = yout + p0 * static_cast<long long>(NB * NC);
        for (int t = threadIdx.x; t < np * NB * NC; t += blockDim.x) {
            const int q = t / (NB * NC), u = t - q * NB * NC, kb = u / NC, kc = u - kb * NC;
            y[t] = buf[q * PS + slot<NB, B1, B2>(kb) * LD + slot<NC, C1, C2>(kc)];
        }
        __syncthreads();
    }
}
