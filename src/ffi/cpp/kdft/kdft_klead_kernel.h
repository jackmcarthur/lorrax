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
//
// A grid whose whole k tile fits is one mode-2 pass (one HBM read and write of T); otherwise
// mode 2 is IDFT passes over the leading groups, the trailing group's mode-2 pass, and DFT
// passes back.  Compile-time constants (all in the image key):
//   LRX_GX, LRX_GY, LRX_GZ   the group's axes (1 for absent)
//   LRX_W                    inner indices per block (the LDS tile is G*W complex)
//   LRX_NT                   threads per block
//   LRX_MODE                 2 | 3
//   LRX_F32                  0 (complex128) | 1 (complex64)
#include "kdft_core.h"

#define LRX_G (LRX_GX * LRX_GY * LRX_GZ)
#define LRX_TILE (LRX_G * LRX_W)

// t: the operand, in place.  v: V_R (nk, mx, my) (mode 2).  ntile: ceil(I / W) blocks per o.
// nb: T's b extent (columns are ((ia*mx + imx)*nb + ib)*my + imy).  forward: mode 3's direction.
extern "C" __global__ void __launch_bounds__(LRX_NT)
lrx_kdft_pass(cplx* __restrict__ t, const cplx* __restrict__ v, long long inner, long long ntile,
              int mx, int nb, int my, real scale, int forward) {
    __shared__ cplx s[LRX_TILE];
    __shared__ cplx tw[LRX_GX + LRX_GY + LRX_GZ];
    const long long o = static_cast<long long>(blockIdx.x) / ntile;
    const long long i0 = (static_cast<long long>(blockIdx.x) - o * ntile) * LRX_W;
    cplx* const base = t + o * LRX_G * inner + i0;            // element (o, g, i0 + w) at g*inner + w

    kdft::fill_twiddles<LRX_GX, LRX_GY, LRX_GZ, LRX_NT>(tw);
    for (int e = threadIdx.x; e < LRX_TILE; e += LRX_NT) {
        const int g = e / LRX_W, w = e % LRX_W;
        s[e] = i0 + w < inner ? base[static_cast<long long>(g) * inner + w] : cplx{0, 0};
    }
    __syncthreads();

#if LRX_MODE == 2
    (void)forward;
    kdft::dft3<LRX_GX, LRX_GY, LRX_GZ, LRX_W, LRX_NT>(s, tw, real(1));
    for (int e = threadIdx.x; e < LRX_TILE; e += LRX_NT) {
        const int g = e / LRX_W, w = e % LRX_W;
        const long long col = i0 + w;                          // I = ncols on the trailing group
        if (col < inner) {
            const long long k = o * LRX_G + g;
            const long long q = col / my;
            const int imy = static_cast<int>(col - q * my);
            const int imx = static_cast<int>((q / nb) % mx);
            const cplx wv = v[(k * mx + imx) * my + imy];
            const cplx a = s[e];
            s[e] = cplx{scale * (a.x * wv.x - a.y * wv.y), scale * (a.x * wv.y + a.y * wv.x)};
        }
    }
    __syncthreads();
    kdft::dft3<LRX_GX, LRX_GY, LRX_GZ, LRX_W, LRX_NT>(s, tw, real(-1));
    const real out_scale = real(1);
#else
    (void)v; (void)mx; (void)nb; (void)my;
    kdft::dft3<LRX_GX, LRX_GY, LRX_GZ, LRX_W, LRX_NT>(s, tw, forward ? real(-1) : real(1));
    const real out_scale = scale;
#endif

    for (int e = threadIdx.x; e < LRX_TILE; e += LRX_NT) {
        const int g = e / LRX_W, w = e % LRX_W;
        if (i0 + w < inner)
            base[static_cast<long long>(g) * inner + w] = cplx{out_scale * s[e].x, out_scale * s[e].y};
    }
}
