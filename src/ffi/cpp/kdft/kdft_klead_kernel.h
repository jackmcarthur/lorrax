// kdft_klead_kernel.h -- DEVICE SOURCE, run-time compiled (common/rtc_build.h); the host
// compiler never sees it.  CMake embeds it as text (kdft_klead_src.h.in), as the CUDA leg
// embeds cufft/kbox_stage.cuh.  Written in the CUDA/HIP common subset (no vendor header, no
// vendor intrinsic), so the same text compiles under hipRTC and NVRTC.
//
// The k-leading direct-DFT pass, modes 2 and 3 of docs/architecture/kconv.md on a flat
// (nk, ncols) tile, nk = NX*NY*NZ with k = (ix*NY + iy)*NZ + iz (the C order of
// reshape(kgrid + trail)):
//
//   mode 2   U[:, c] = fftn(scale * ifftn(T[:, c]) * V_R[:, v(c)])   T (nk, a, mx, b, my),
//            v(c) = (imx, imy) of column c,  V_R (nk, mx, my)         in place on T
//   mode 3   Y[:, c] = scale * (forward ? fftn : ifftn)(X[:, c])       unnormalised transforms
//
// One block owns C columns.  It loads the (nk, C) tile into LDS once (row k is C consecutive
// columns, so a wavefront's global access is contiguous), runs the three axis DFTs in LDS, and
// stores the tile once: one HBM read and one HBM write of T per pass, where the XLA route
// transposes the k axes minor, transforms, transposes back, multiplies and repeats.
//
// Each axis pass is a direct DFT: one thread holds one line (N values of one column along one
// axis) in registers, forms its N outputs with N*N complex FMAs against an LDS twiddle table
// (exp(2 pi i m / N), from sincospi, so a twiddle is correctly rounded), and writes the line
// back in place.  Lines of one pass are disjoint, so a pass needs one barrier.  The cost is
// N*(NX+NY+NZ) complex FMAs per point and transform; it is the right algorithm for the k axes'
// small N (the cuFFTDx route caps an axis at 40).
//
// Compile-time constants (the deciding options, so they are in the image key):
//   LRX_NX, LRX_NY, LRX_NZ   the k-grid
//   LRX_C                    columns per block (the LDS tile is NK*C complex)
//   LRX_NT                   threads per block
//   LRX_MODE                 2 (convolution) | 3 (one transform)
//   LRX_F32                  0 (complex128) | 1 (complex64)

#if LRX_F32
typedef float real;
#define LRX_SINCOSPI sincospif
#else
typedef double real;
#define LRX_SINCOSPI sincospi
#endif

struct alignas(2 * sizeof(real)) cplx { real x, y; };

#define NK (LRX_NX * LRX_NY * LRX_NZ)
#define TILE (NK * LRX_C)
#define NTW (LRX_NX + LRX_NY + LRX_NZ)

// One axis (length N, k-stride S) of the 3-D DFT, in place on the tile s[k*C + c];
// sgn = +1 for the inverse (exp(+2 pi i)), -1 for the forward transform.
template <int N, int S>
__device__ __forceinline__ void axis_pass(cplx* s, const cplx* tw, real sgn) {
    if (N == 1) return;
    constexpr int LINES = (NK / N) * LRX_C;
    for (int l = threadIdx.x; l < LINES; l += LRX_NT) {
        const int c = l % LRX_C, o = l / LRX_C;          // o: the line's index over the other axes
        const int base = (o / S) * (S * N) + (o % S);    // k of the line's first point
        cplx in[N];
#pragma unroll
        for (int j = 0; j < N; ++j) in[j] = s[(base + j * S) * LRX_C + c];
#pragma unroll
        for (int kd = 0; kd < N; ++kd) {
            real ax = 0, ay = 0;
#pragma unroll
            for (int j = 0; j < N; ++j) {
                const cplx w = tw[(kd * j) % N];
                const real wy = sgn * w.y;
                ax = fma(in[j].x, w.x, fma(-in[j].y, wy, ax));
                ay = fma(in[j].x, wy, fma(in[j].y, w.x, ay));
            }
            s[(base + kd * S) * LRX_C + c] = cplx{ax, ay};
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void dft3(cplx* s, const cplx* tw, real sgn) {
    axis_pass<LRX_NX, LRX_NY * LRX_NZ>(s, tw, sgn);
    axis_pass<LRX_NY, LRX_NZ>(s, tw + LRX_NX, sgn);
    axis_pass<LRX_NZ, 1>(s, tw + LRX_NX + LRX_NY, sgn);
}

// t: the (nk, ncols) tile, in place.  v: V_R (nk, mx, my) (mode 2; unused in mode 3).
// nb: T's b extent (columns are ((ia*mx + imx)*nb + ib)*my + imy).  forward: mode 3's direction.
extern "C" __global__ void __launch_bounds__(LRX_NT)
lrx_kdft(cplx* __restrict__ t, const cplx* __restrict__ v, long long ncols,
         int mx, int nb, int my, real scale, int forward) {
    __shared__ cplx s[TILE];
    __shared__ cplx tw[NTW];
    const long long col0 = static_cast<long long>(blockIdx.x) * LRX_C;

    for (int m = threadIdx.x; m < NTW; m += LRX_NT) {
        const int n = m < LRX_NX ? LRX_NX : (m < LRX_NX + LRX_NY ? LRX_NY : LRX_NZ);
        const int q = m < LRX_NX ? m : (m < LRX_NX + LRX_NY ? m - LRX_NX : m - LRX_NX - LRX_NY);
        real sn, cs;
        LRX_SINCOSPI(real(2 * q) / real(n), &sn, &cs);
        tw[m] = cplx{cs, sn};
    }
    for (int e = threadIdx.x; e < TILE; e += LRX_NT) {
        const int k = e / LRX_C, c = e % LRX_C;
        const long long col = col0 + c;
        s[e] = col < ncols ? t[k * ncols + col] : cplx{0, 0};
    }
    __syncthreads();

#if LRX_MODE == 2
    (void)forward;
    dft3(s, tw, real(1));
    for (int e = threadIdx.x; e < TILE; e += LRX_NT) {
        const int k = e / LRX_C, c = e % LRX_C;
        const long long col = col0 + c;
        if (col < ncols) {
            const long long q = col / my;
            const int imy = static_cast<int>(col - q * my);
            const int imx = static_cast<int>((q / nb) % mx);
            const cplx w = v[(static_cast<long long>(k) * mx + imx) * my + imy];
            const cplx a = s[e];
            s[e] = cplx{scale * (a.x * w.x - a.y * w.y), scale * (a.x * w.y + a.y * w.x)};
        }
    }
    __syncthreads();
    dft3(s, tw, real(-1));
    const real out_scale = real(1);
#else
    (void)v; (void)mx; (void)nb; (void)my;
    dft3(s, tw, forward ? real(-1) : real(1));
    const real out_scale = scale;
#endif

    for (int e = threadIdx.x; e < TILE; e += LRX_NT) {
        const int k = e / LRX_C, c = e % LRX_C;
        const long long col = col0 + c;
        if (col < ncols) t[k * ncols + col] = cplx{out_scale * s[e].x, out_scale * s[e].y};
    }
}
