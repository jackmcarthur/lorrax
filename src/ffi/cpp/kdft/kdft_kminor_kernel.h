// kdft_kminor_kernel.h -- DEVICE SOURCE, run-time compiled (common/rtc_build.h); the host
// compiler never sees it.  CMake embeds it as text (kdft_kminor_src.h.in).  Written in the
// CUDA/HIP common subset, as kdft_klead_kernel.h.  NVRTC 13.4 compiles the same text for sm_80/sm_90
// while the tile stays within CUDA's 48 KiB static shared memory; tiles up to the 64 KiB the ROCm
// handler grants are ROCm-only (a CUDA build would need dynamic shared memory with the opt-in).
//
// The k-MINOR kdft pass, modes 4 and 5 of docs/architecture/kconv.md, on a (rows, nk)
// tile with k trailing, nk = NX*NY*NZ, k = (ix*NY + iy)*NZ + iz:
//
//   mode 4   U[r, :] = fftn(scale * ifftn(X[r, :]) * K[(r / m1) % m0, :])   X (d0,d1,d2,d3,d4,nk),
//            m0 = d1*d2, m1 = d3*d4, K (d1, d2, nk) in R space; stored in X's layout
//            (LRX_OUT 0, may alias X) or (d0, nk, d3, d1, d4, d2) (LRX_OUT 1)
//   mode 5   Y[r, :] = scale * (forward ? fftn : ifftn)(X[r, :])   (rows, nk), may alias X
//
// The SPLIT arm (kdft_kminor_ffi.cc, nk past one block's tile) uses this kernel as its ROW pass:
// LRX_NX..NZ are then the trailing axis group (G = NK points, contiguous within a row) and
// LRX_SPLIT = O_T = nkf / NK the points of the axes before it, so the pass transforms rows*O_T
// row-pass rows of NK.  Row-pass row r' is the run of (operand row r, leading index ot) at
// r*nkf + ot*NK: r' = r*O_T + ot (ot fastest, so a block reads one contiguous C*NK run), except
// for the LRX_OUT 1 store (mode 5 with LRX_OUT 1: the split arm's last pass), whose block owns C
// consecutive operand rows of one ot, so the permuted store writes runs of C rows; LRX_ORDER
// says which index the blocks walk fastest: 0 the row chunk (every block in flight stores one
// ot), 1 the ot.  The host picks it from the store's k stride (kdft_kminor_plan.h l1_order).
// (r fastest everywhere ran 3x slower on power-of-two nk, whose row stride nk*16 B lands every
// block's runs on the same HBM channels: 16^3 mode 5 10.3 ms, pool log kcore_after2.)
// One block owns C consecutive rows: a contiguous C*nk run of HBM, read once and written once.
// The tile is staged TRANSPOSED in LDS, s[k*W + c], W odd and >= C (a zeroed pad column when
// W > C): the DFT core (kdft_core.h, a mixed-radix Stockham line DFT on any (G, W) tile, lanes
// along w) transforms it as a (G = nk, W) tile, and the transposing load and store, whose
// consecutive threads walk k at an odd stride, hit distinct LDS banks.  The core and its radix plan (kdft_plan.h) are included
// unchanged, as embedded rtc headers shared with the k-leading kernel.
//
// Compile-time constants (deciding options, so they are in the image key):
//   LRX_NX, LRX_NY, LRX_NZ   the k-grid (the split arm: the trailing axis group)
//   LRX_SPLIT                1, or the split arm's O_T (row-pass rows per operand row)
//   LRX_ORDER                the layout-1 store's block order (LRX_OUT 1 with LRX_SPLIT > 1):
//                            0 row chunk fastest, 1 ot fastest (kdft_kminor_plan.h l1_order)
//   LRX_C                    rows per block
//   LRX_W                    the LDS tile's width, odd, C <= W <= C + 1 (the tile is NK*W complex)
//   LRX_NT                   threads per block
//   LRX_MODE                 4 (convolution) | 5 (one transform)
//   LRX_OUT                  the store: 0 (X's layout) | 1 ((d0, nk, d3, d1, d4, d2): mode 4, or a
//                            split arm's last row pass)
//   LRX_F32                  0 (complex128) | 1 (complex64)

#include "kdft_core.h"

#ifndef LRX_SPLIT
#define LRX_SPLIT 1
#endif
#ifndef LRX_ORDER
#define LRX_ORDER 0
#endif
#define LRX_OTMAP (LRX_OUT == 1 && LRX_SPLIT > 1)
#define NK (LRX_NX * LRX_NY * LRX_NZ)
#define NKF (NK * LRX_SPLIT)
#define NTW (LRX_NX + LRX_NY + LRX_NZ)
#if LRX_W % 2 == 0 || LRX_W < LRX_C || LRX_W > LRX_C + 1
#error "LRX_W must be odd and C or C + 1: the transposing load/store relies on an odd row stride"
#endif

// x: the (rows, nkf) operand; y: the result (== x when aliased: a block reads its rows whole
// before it stores them, and stores only its own rows in layout 0).  kern: K (m0, nk) (mode 4).
// d1..d4: X's extents (mode 4 and LRX_OUT 1); rows*LRX_SPLIT < 2^31 (the host checks).
extern "C" __global__ void __launch_bounds__(LRX_NT)
lrx_kdft_kminor(const cplx* x, const cplx* __restrict__ kern, cplx* y, int rows,
                int d1, int d2, int d3, int d4, real scale, int forward) {
    __shared__ cplx s[NK * LRX_W];
    __shared__ cplx tw[NTW];
    __shared__ long long roff[LRX_C];                           // a tile row's first element
#if LRX_MODE == 4
    __shared__ int krow[LRX_C];                                 // K's row of each tile row
#endif
#if LRX_OUT == 1
    __shared__ long long obase[LRX_C];                          // layout 1: the row's offset at k = 0
#endif
#if LRX_OTMAP
    // block -> (row chunk, ot), one block each (row_blocks); LRX_ORDER: which walks fastest
    const int nch = (rows + LRX_C - 1) / LRX_C;
    const int bot = LRX_ORDER == 0 ? static_cast<int>(blockIdx.x) / nch : static_cast<int>(blockIdx.x) % LRX_SPLIT;
    const int ch = LRX_ORDER == 0 ? static_cast<int>(blockIdx.x) - bot * nch : static_cast<int>(blockIdx.x) / LRX_SPLIT;
    const int r0 = ch * LRX_C;
    const int nr = rows - r0 < LRX_C ? rows - r0 : LRX_C;
#else
    const int nrp = rows * LRX_SPLIT;                           // row-pass rows
    const int r0 = blockIdx.x * LRX_C;
    const int nr = nrp - r0 < LRX_C ? nrp - r0 : LRX_C;        // live rows of this tile
#endif

    kdft::fill_twiddles<LRX_NX, LRX_NY, LRX_NZ, LRX_NT>(tw);
    for (int c = threadIdx.x; c < nr; c += LRX_NT) {
        const int rp = r0 + c;
#if LRX_OTMAP
        const int r = rp, ot = bot;
#elif LRX_OUT == 1
        const int r = rp, ot = 0;
#else
        const int r = rp / LRX_SPLIT, ot = rp % LRX_SPLIT;
#endif
        roff[c] = static_cast<long long>(r) * NKF + static_cast<long long>(ot) * NK;
#if LRX_MODE == 4 || LRX_OUT == 1
        int t = r;
        const int i4 = t % d4; t /= d4;
        const int i3 = t % d3; t /= d3;
        const int i2 = t % d2; t /= d2;
        const int i1 = t % d1; const int i0 = t / d1;
#if LRX_MODE == 4
        krow[c] = i1 * d2 + i2;
#endif
#if LRX_OUT == 1
        obase[c] = static_cast<long long>(i0) * NKF * d1 * d2 * d3 * d4 +
                   ((static_cast<long long>(i3) * d1 + i1) * d4 + i4) * d2 + i2 +
                   static_cast<long long>(ot) * NK * d1 * d2 * d3 * d4;
#endif
        (void)i0; (void)i3; (void)i4;
#endif
    }
    __syncthreads();
    for (int e = threadIdx.x; e < NK * LRX_C; e += LRX_NT) {    // runs of NK (one HBM run when SPLIT = 1)
        const int c = e / NK, k = e - c * NK;
        s[k * LRX_W + c] = c < nr ? x[roff[c] + k] : cplx{0, 0};
    }
    if (LRX_W > LRX_C)                                          // the pad column: zeros
        for (int k = threadIdx.x; k < NK; k += LRX_NT) s[k * LRX_W + LRX_C] = cplx{0, 0};
    __syncthreads();

#if LRX_MODE == 4
    (void)forward;
    kdft::dft3<LRX_NX, LRX_NY, LRX_NZ, LRX_W, LRX_NT>(s, tw, real(1));
    for (int e = threadIdx.x; e < nr * NK; e += LRX_NT) {       // K read k-contiguous
        const int c = e / NK, k = e - c * NK;
        const cplx w = kern[static_cast<long long>(krow[c]) * NKF + k];
        const cplx a = s[k * LRX_W + c];
        s[k * LRX_W + c] = cplx{scale * (a.x * w.x - a.y * w.y), scale * (a.x * w.y + a.y * w.x)};
    }
    __syncthreads();
    kdft::dft3<LRX_NX, LRX_NY, LRX_NZ, LRX_W, LRX_NT>(s, tw, real(-1));
    const real out_scale = real(1);
#else
    (void)kern;
#if LRX_OUT == 0
    (void)d1; (void)d2; (void)d3; (void)d4;
#endif
    kdft::dft3<LRX_NX, LRX_NY, LRX_NZ, LRX_W, LRX_NT>(s, tw, forward ? real(-1) : real(1));
    const real out_scale = scale;
#endif
    if (NK == 1) __syncthreads();                               // no stage ran, so no stage barrier

#if LRX_OUT == 1
    // (d0, nk, d3, d1, d4, d2): element (row, k) at obase[row] + k*d1*d2*d3*d4; rows fastest
    // across threads, so runs of consecutive i2 (rows with d3*d4 == 1) are contiguous.
    const long long kstride = static_cast<long long>(d1) * d2 * d3 * d4;
    for (int e = threadIdx.x; e < NK * nr; e += LRX_NT) {
        const int k = e / nr, c = e - k * nr;
        const cplx v = s[k * LRX_W + c];
        y[obase[c] + k * kstride] = cplx{out_scale * v.x, out_scale * v.y};
    }
#else
    for (int e = threadIdx.x; e < nr * NK; e += LRX_NT) {
        const int c = e / NK, k = e - c * NK;
        const cplx v = s[k * LRX_W + c];
        y[roff[c] + k] = cplx{out_scale * v.x, out_scale * v.y};
    }
#endif
}
