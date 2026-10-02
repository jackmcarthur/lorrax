// kconv_mathdx_cuda_ffi.cc -- the NVIDIA k-convolution family on nvidia-mathdx.
//
// One fused, one-memory-pass kernel family whose FFTs are the LIBRARY's:
// cuFFTDx thread FFTs (nvidia-mathdx), compiled by NVRTC at run time once per
// (mode, nkx, nky, nkz, ns, CUDA context) and cached in process.  LORRAX
// writes no DFT, no twiddle table and no radix code here; the per-size
// specialisation is the library's, made at JIT time.  Ruling:
// docs/architecture/decisions.md 2026-09-24; layering:
// docs/architecture/ffi_layout.md "k-convolution router and the mathdx family".
//
// Modes (LRX_MODE of the embedded source; one handler per mode):
//   0 pair    U[kx,ky,kz,col,mu] = s * FFT_k( sum_ab phase_ab
//                 conj(IFFT_k A[k,a,col,mu,b]) * IFFT_k B[k,perm_l a,col,mu,perm_r b] )
//   1 parent  the same contraction with the typed parent load of
//             docs/architecture/ffi_layout.md "Parent-load ISDF pair
//             convolution": A/B are raw-parent (p,ns,mu,ns,nu) projectors and
//             the load applies the umklapp phases, the antiunitary conjugation
//             and the open-spin coefficients; U is (nk, mu, nu).
//   2 klead conv   U[k,a,x,b,y] = s * FFT_k( IFFT_k T[.,a,x,b,y] * V[k,x,y] ),
//                  T/U k-LEADING (nk, a, mx, b, my), V (nk, mx, my) ALREADY in
//                  R space (mode 3 made it): the Sigma tau and COHSEX kernel.
//   3 klead fft    Y[k,r] = s * FFT^{+-}_k X[.,r] on a k-LEADING (nk, rows) tile.
//             Modes 2 and 3 run on the k-box stage (kbox_stage.cuh): tiles of
//             whole columns staged by cp.async into padded shared memory, or,
//             where two columns do not fit a block, plane and pencil passes
//             through the output in place; z,y then x each way, as before.
//   4 kminor conv  U = s * FFT_k( IFFT_k X[r,.] * K[(r/(d3 d4)) % (d1 d2), k] ),
//                  X (d0,d1,d2,d3,d4,nk) k-MINOR, K (d1,d2,nk) R space; the
//                  store emits X's layout (out_layout 0) or (d0,nk,d3,d1,d4,d2)
//                  (out_layout 1): the BSE ladder-W rung.
//   5 kminor fft   Y[r,k] = s * FFT^{+-}_k X[r,.] on a k-MINOR (rows, nk) tile.
//             Modes 4 and 5 run on the k-box stage's single arm: the Load reads
//             each column's k run, the Store writes k fastest (or mode 4's
//             out_layout 1 rows fastest).
//   6 plane   the pair contraction of mode 0 read straight from the route-G
//             D-plane transform output D (nk, g, ns, 2c, ns, p): the load
//             applies the Bloch phase F[k,g,p] and takes L = slots [0,c) and
//             R = slots [c,2c) of the 2c axis, so no transposed, phased or
//             split copy is made; U is (nk, c, g*p).  The element product
//             D*F is the one XLA formed before this mode existed (see
//             lrx_mul_xla), so mode 6 is meant to equal the old moveaxis +
//             mode 1 chain bit for bit.
//   7 klead unfold conv   mode 2 read from the RAW-PARENT Green tiles, on the k-box stage: per
//             full k the load gathers G[row(k), lsrc(k,i), rsrc(k,j)] (the
//             transposed-pair tile on an antiunitary row), applies the
//             umklapp phases mph(k,i), nph(k,j) and the ns x ns spin action
//             U_k in registers, and the store writes U spin-major
//             (n_out, a, mx, b, my), full-k row k at output row kout(k)
//             (-1: not stored; the Sigma consumers keep only the parent
//             rows), or one (d x d) output spin block (a0, b0) of it
//             (LRX_NA = d): every source is read, only the block stored.  The tables are symmetry_maps's (unfold_load_tables);
//             the products are fused (lrx_mulf, lrx_cmac): round-off equal to
//             the XLA unfold and the spin-rotate FFI it replaces.
//   8 klead lorentz conv   mode 7's load, then the four-current vertex sum in
//             R space: U = mult * sf * FFT_k( sum_{A,B} gamma_A (si * IFFT_k
//             G_unfolded) gamma_B^dagger * V[k,x,A,y,B] ) with V (nk, mx, nA,
//             my, nB) ALREADY in R space (mode 3 made it, scale si), the
//             signed-permutation vertices gamma_A (left) and gamma_B (right)
//             as attributes, and U spin-major (n_out, a, mx, b, my) through
//             mode 7's kout row map.  One
//             transform of G serves every Lorentz block; the scales and the
//             product/sum order of the vertex sum are those of the XLA chain it
//             replaces (mode-3 transforms and a scan over the blocks); mode 7's
//             fused load makes it round-off equal to that chain.  On the k-box
//             stage: the single arm when a spin group fits (the vertex sum a
//             group Mid, one thread per (k, pair)); else the split
//             arm, whose group pencil forms each (member, pair) on its own
//             thread with V staged by cp.async, chunked over pairs through an
//             intermediate no larger than U.
//   9 klead unfold fft   mode 3's inverse transform read from the WEDGE tiles
//             of an interaction (W, V, a pole field) through mode 7's load:
//             Y[k, i, j] = s * IFFT_k( L_k O_k R_k^dagger )[i, j] with O_k
//             the gathered, phased wedge tile, L_k (nk, nl_s, nl_s) and R_k
//             (nk, nr_s, nr_s) the endpoint actions (1 for a scalar
//             interaction, the Lorentz rotation for current blocks), and Y
//             k-LEADING (nk, ml, nl), merged endpoints i = x*nl_s + A,
//             j = y*nr_s + B: the R-space operand modes 2/7/8 take.  An
//             antiunitary row either reads the partner tile (pair_transpose)
//             or conjugates the phased product (conj_trs, a Hermitian
//             interaction: no partner tile).  The products are the fused
//             forms of mode 7's load: round-off equal to the XLA unfold
//             (symmetry_maps unfold_isdf_operator) and mode 3 they replace.
//             On the k-box stage's single arm (tiles of whole spin groups
//             when one fits, else single columns).
//  10 plane fft gather   Y[..., kb, kc] = FFT2_{b,c}(plane[..., b, c]) (forward,
//             unscaled: jnp.fft.fftn(norm='backward') over the last two axes)
//             where the plane is the route-G cylinder F (..., n_col) scattered
//             to its static cells and zero elsewhere.  The zero plane is never
//             written: the row FFTs run on the occupied rows only, gathering
//             their cells on load, then the column FFTs read dead rows as zero.
//             Its own embedded source, kPlaneSrc (see there).
//  11 klead chi unfold   one tau node of chi0 from the RAW-PARENT Green pair on
//             the k-box stage (kbox_stage.cuh, embedded as a named header): the
//             load gathers mode 7's typed unfold of Gv and Gc (every spin element
//             of a pair; conj_trs = 2 reads the antiunitary partner as conj(G)),
//             one inverse transform, then per (k, pair) the spin trace
//             v = sum_ab conj(si Gc'_ab) (si Gv'_ab) (+ conj(v) on a real contour)
//             and acc[o, k, x, y] += alpha[o] v in place.  The forward transform
//             follows the tau sum.  Single pass when a pair's 2*ns^2 columns fit,
//             else the plane pass and an R-space warp pencil (a pair's columns on one warp,
//             reduced by shuffles), chunked over pairs.
// A new mode adds (1) an entry under its LRX_MODE value in kSrc, (2) a mode
// code and a handler below, (3) a router factory in ffi/fft.py.
//
// Residency: modes 0/1/6 keep three nk-long banks per row (one (col,mu) pair) in
// shared memory; a k-grid whose row does not fit the device's
// opt-in shared memory, or an axis above the fp64 thread-FFT limit (40), is
// refused by name.  The k-box modes (2, 3, 8, 11) need one (ky, kz) plane of a
// tile in shared memory; modes 4, 5, 7 and 9 (single arm only) one whole padded k-box column.
// Modes 2, 3 and 5, and mode 4 with out_layout 0, may run in place: every block reads the
// elements it stores before it stores them.
//
// Headers: the Python router passes the installed wheel's nvidia/mathdx
// directory as the string attribute `mathdx_root`; the CUDA toolkit include
// (for libcu++, include/cccl) is derived from the loaded libnvrtc.
//
// Disk cache (common/nvrtc_build.h owns the key rule and the image format): the router passes `cubin_dir` (ffi.fft.cubin_cache_dir:
// $SCRATCH/.cache/lorrax/kconv_mathdx, else ~/.cache/lorrax/kconv_mathdx;
// "" = no disk cache).  A cubin is keyed by FNV-1a over the embedded source,
// the NVRTC options (mode, grid, ns, rows per block, sm) and the whole
// toolchain that can change the image: the cuFFTDx, commonDx, CUTLASS and
// CCCL version headers, the nvidia-mathdx wheel's dist-info name, and the
// NVRTC version with the loaded libnvrtc's real path (its patch level).  A
// version header that reads empty disables the disk cache for that build
// rather than dropping out of the key.  The image is written to a unique
// temporary and renamed (atomic on one filesystem, so concurrent ranks cannot
// tear it) and re-hashed on read; a torn, foreign or non-ELF file, or one the
// driver refuses to load, is deleted, recompiled once and replaced.  No
// environment variable is read here.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "../common/mkl_thread_pin.h"
#include "../common/lrx_async_gather.h"
#include "../common/nvrtc_build.h"
#include "kbox_stage.cuh"          // the host half: the k-box launch rule (kbox_plan)
#include "kbox_stage_src.h"       // the same file as text, embedded into mode 11's NVRTC program

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "xla/ffi/api/ffi.h"

namespace lorrax_ffi::kconv_mathdx {

namespace ffi = ::xla::ffi;

static constexpr int kAxisMax = 40;        // cuFFTDx fp64 thread-FFT limit
static constexpr int kRowsMax = 16;        // modes 0/1/6 (three banks per row)
static constexpr int kThreads = 256;
static constexpr long long kSmemBudget = 100 * 1024;
// Mode 10 packs small planes into one block up to this much shared memory.
static constexpr long long kPlaneGroupBytes = 64 * 1024;

static bool log_enabled() {
    static const bool on = [] { return mklpin::debug_print_here(); }();
    return on;
}

static ffi::Error fail(const char* where, const std::string& detail,
                       ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
    std::ostringstream os;
    os << "kconv_mathdx (fused cuFFTDx k-convolution): " << where
       << " failed -- " << detail;
    return ffi::Error(code, os.str());
}

#define LRX_CUDA_CHECK(expr, where)                                      \
    do {                                                                 \
        cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) return fail((where), cudaGetErrorString(_e)); \
    } while (0)

// The driver API, the NVRTC build and the cubin disk cache: common/nvrtc_build.h.
using nvrtc::DriverApi;
using nvrtc::driver_api;
using nvrtc::cu_err;

struct ParentTables {
    const int *irr, *sym, *left, *right, *trs;
    const double *L, *R, *q, *coef_l, *coef_r;
    int mu, nu, centroid_major;
};

// Mode 6 operand geometry (the embedded source declares the same struct).
struct PlaneTab {
    const double* phase;      // F (nk, g, p) complex128
    long long g, p, c;        // planes per group, points per plane, L/R slots
};

// Mode 7 typed-unfold tables (the embedded source declares the same struct).
struct UnfoldTab {
    const int *row, *trs, *lsrc, *rsrc;          // (nk) (nk) (nk,ml) (nk,nl)
    const double *mph, *nph, *spin;              // (nk,ml) (nk,nl) (nk,ns,ns) c128
    long long ml, nl;                            // merged local endpoints mx*ns, my*ns
    const int* kout;                             // (nk) output row of full k, -1 = none;
                                                 // null = every k at its own row
    int conj_trs;                                // antiunitary rows: 0 read the partner tile,
                                                 // 1 conjugate the phased product, 2 read conj(G)
    const double* spin_r;                        // (nk,nr_s,nr_s) right action; null = spin
    int a0, b0;                                  // mode 7: first rows of the output spin block
    long long x0, bx, xs, xn;                    // mode 7: the stored x block, rows r in [0, xn*bx):
                                                 // left centroid (r / bx)*xs + x0 + r % bx (bx = 0: all)
};

// Mode 8 vertex tables and scales (the embedded source declares the same struct).
// Vertex i of a side is packed like modes 0/1/6's single vertex, shifted by i:
// perm entry (i, a) at bits 16*i + 4*a, phase code (i, a) at bits 8*i + 2*a.
struct LorentzTab {
    unsigned long long perm_l, phase_l, perm_r, phase_r;
    int na, nb;
    double s_g, s_f, mult;
};

// Modes 2-5 row geometry (the embedded source declares the same struct).
struct RowGeo {
    long long rows;           // independent k-rows in the tile
    long long m0, m1, m2;     // mode 2: my, b*my, mx     mode 4: d1*d2, d3*d4, -
    long long d1, d2, d3, d4; // mode 4 out_layout 1 store permutation
    double scale;
    int forward;              // modes 3/5: 1 forward (exp -i), 0 inverse
    int out_layout;           // mode 4
};

// ---------------------------------------------------------------------------
//  The embedded source.  Compile-time: LRX_MODE, LRX_NX/NY/NZ, LRX_NS, LRX_RB,
//  LRX_SM.  The transforms are cuFFTDx thread FFTs; everything else (loads,
//  the typed parent action, the spin contraction, the store) is the
//  2026-09-06 conv_kparent contract.
// ---------------------------------------------------------------------------
static const char* kSrc = R"__lrx__(
#include <cufftdx.hpp>
#include "lrx_async_gather.cuh"

// LRX_F32: modes 2-5 also serve complex64 tiles (the fp32-GMRES BSE arm).
#if LRX_F32
typedef float lrx_real;
struct __align__(8) lrx_c2 { float x, y; };
#else
typedef double lrx_real;
struct __align__(16) lrx_c2 { double x, y; };
#endif
struct ParentTables {
    const int *irr, *sym, *left, *right, *trs;
    const double *L, *R, *q, *coef_l, *coef_r;
    int mu, nu, centroid_major;
};
struct PlaneTab {
    const double* phase;
    long long g, p, c;
};
struct UnfoldTab {
    const int *row, *trs, *lsrc, *rsrc;
    const double *mph, *nph, *spin;
    long long ml, nl;
    const int* kout;
    int conj_trs;
    const double* spin_r;
    int a0, b0;
    long long x0, bx, xs, xn;
};
// The output row of full k: kout[k] (-1 = not stored), or k itself.
__device__ __forceinline__ long long lrx_out_row(const UnfoldTab& t, int k) {
    return t.kout ? (long long)t.kout[k] : (long long)k;
}
struct LorentzTab {
    unsigned long long perm_l, phase_l, perm_r, phase_r;
    int na, nb;
    double s_g, s_f, mult;
};
struct RowGeo {
    long long rows;
    long long m0, m1, m2;
    long long d1, d2, d3, d4;
    double scale;
    int forward;
    int out_layout;
};

__device__ __forceinline__ lrx_c2 lrx_mul(lrx_c2 a, lrx_c2 b) {
    lrx_c2 z = {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
    return z;
}
#if !LRX_F32
// Mode 6 and mode 8's vertex round as the chains they replace.  Each form is
// spelled with round-to-nearest intrinsics or explicit fma, so NVRTC's own
// contraction cannot change it (runs/runtime/kconv_fused_load_20260924/fma_forms).
//
// (a+bi)(c+di) as XLA:GPU forms an HLO complex multiply: (ac - bd, ad + bc)
// with NO fused multiply-add (measured 4096/4096 against exact emulation,
// tests/multi_device/kconv_router_p4.py xla_cmul_form).
__device__ __forceinline__ lrx_c2 lrx_mul_xla(lrx_c2 a, lrx_c2 b) {
    lrx_c2 z = {__dsub_rn(__dmul_rn(a.x, b.x), __dmul_rn(a.y, b.y)),
                __dadd_rn(__dmul_rn(a.x, b.y), __dmul_rn(a.y, b.x))};
    return z;
}
// The unfold load (modes 7, 8, 9, 11), its tile finish and the chi trace use fused forms (owner
// 2026-09-25: round-off equal is fine; the gate is the router's ulp bounds and the reference decks
// at the print quantum or |dSigma| < 2 meV, TASTE 77): a product in two FMAs and two multiplies,
// and a product-sum v += a b in four FMAs.  They cut the FP64 pipe's busy time per chi0 tau node
// from 62 to 45 ms at base clock (6x6 bispinor, A100; U2c).
__device__ __forceinline__ lrx_c2 lrx_mulf(lrx_c2 a, lrx_c2 b) {
    lrx_c2 z = {fma(a.x, b.x, -__dmul_rn(a.y, b.y)), fma(a.x, b.y, __dmul_rn(a.y, b.x))};
    return z;
}
// v += a b and v += a conj(b), four fused multiply-adds each.
__device__ __forceinline__ void lrx_cmac(lrx_c2& v, lrx_c2 a, lrx_c2 b) {
    v.x = fma(a.x, b.x, v.x); v.x = fma(-a.y, b.y, v.x);
    v.y = fma(a.x, b.y, v.y); v.y = fma(a.y, b.x, v.y);
}
__device__ __forceinline__ void lrx_cmac_conj(lrx_c2& v, lrx_c2 a, lrx_c2 b) {
    v.x = fma(a.x, b.x, v.x); v.x = fma(a.y, b.y, v.x);
    v.y = fma(a.y, b.x, v.y); v.y = fma(-a.x, b.y, v.y);
}
#endif
__device__ __forceinline__ lrx_c2 lrx_phase(lrx_c2 z, int code) {
    if (code == 1) { lrx_c2 q = {-z.y, z.x}; return q; }
    if (code == 2) { lrx_c2 q = {-z.x, -z.y}; return q; }
    if (code == 3) { lrx_c2 q = {z.y, -z.x}; return q; }
    return z;
}

#if LRX_MODE == 1
// P = conj(sum_cd U_ac conj(U_bd) T[phase_L D_cd conj(phase_R)]) from owner-local maps.
__device__ __forceinline__ lrx_c2 lrx_parent_load(
    const lrx_c2* d, ParentTables t, int k, int a, int b, long long row, int ns, bool right) {
    const int m = row / t.nu, n = row % t.nu;
    const int p = t.irr[k], op = t.sym[k];
    const int lm = t.left[op*t.mu+m], rn = t.right[op*t.nu+n];
    double dl = 0.0, dr = 0.0;
    for (int i = 0; i < 3; ++i) {
        dl += t.q[3*p+i]*t.L[3*(op*t.mu+m)+i];
        dr += t.q[3*p+i]*t.R[3*(op*t.nu+n)+i];
    }
    lrx_c2 pl, pr;
    sincospi(2.0*dl, &pl.y, &pl.x);
    sincospi(-2.0*dr, &pr.y, &pr.x);
    const lrx_c2* coef = reinterpret_cast<const lrx_c2*>(right ? t.coef_r : t.coef_l)
        + ((long long)k*ns*ns + a*ns + b)*ns*ns;
    lrx_c2 sum = {0.0, 0.0};
    for (int c = 0; c < ns; ++c) for (int e = 0; e < ns; ++e) {
        const lrx_c2 weight = coef[c*ns+e];
        if (weight.x == 0.0 && weight.y == 0.0) continue;
        const long long index = t.centroid_major
            ? ((((long long)p*t.nu + rn)*ns + e)*t.mu + lm)*ns + c
            : (((long long)p*ns + c)*t.mu + lm)*ns*t.nu + e*t.nu + rn;
        lrx_c2 v = lrx_mul(lrx_mul(pl, d[index]), pr);
        if (t.trs[k]) v.y = -v.y;
        v = lrx_mul(weight, v);
        sum.x += v.x; sum.y += v.y;
    }
    sum.y = -sum.y;
    return sum;
}
#endif

constexpr int NX = LRX_NX, NY = LRX_NY, NZ = LRX_NZ, NS = LRX_NS, RB = LRX_RB;
constexpr int NK = NX * NY * NZ, SP = NK | 1;

template <int N, cufftdx::fft_direction Dir>
using TFFT = decltype(cufftdx::Size<N>() + cufftdx::Precision<lrx_real>() +
                      cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<Dir>() +
                      cufftdx::Thread() + cufftdx::SM<LRX_SM>());

// One axis of the 3-D transform on RB resident rows: every block thread runs
// whole library line-FFTs (flat k is C-order, kz fastest).  From 2 rows up the row
// runs fastest across threads: the 8 threads of a 16-byte shared phase take one line
// offset in min(RB, 8) rows SP apart, SP is odd, so they hit 8 distinct bank groups
// (RB >= 8) or at most ceil(8 / RB) threads share one (RB 4 at 8^3: 2-way).  Line
// fastest put them on lines N*STRIDE apart (8x8x1's y pass, 8^3's z pass: one group,
// 8-way).  Which thread runs a line does not change its arithmetic (bitwise).
template <int N, int STRIDE, cufftdx::fft_direction Dir>
__device__ __forceinline__ void axis_pass(lrx_c2* bank) {
    if constexpr (N > 1) {
        using F = TFFT<N, Dir>;
        using V = typename F::value_type;
        constexpr int lines = NK / N;
        for (int l = threadIdx.x; l < RB * lines; l += blockDim.x) {
            const int j = RB >= 2 ? l % RB : l / lines, li = RB >= 2 ? l / RB : l % lines;
            lrx_c2* p = bank + j * SP + (li / STRIDE) * N * STRIDE + (li % STRIDE);
            V v[F::storage_size];
#pragma unroll
            for (int e = 0; e < N; ++e) { v[e].x = p[e * STRIDE].x; v[e].y = p[e * STRIDE].y; }
            F().execute(v);
#pragma unroll
            for (int e = 0; e < N; ++e) { p[e * STRIDE].x = v[e].x; p[e * STRIDE].y = v[e].y; }
        }
        __syncthreads();                               // a pass that did not run writes nothing to publish
    }
}

template <cufftdx::fft_direction Dir>
__device__ __forceinline__ void transform3(lrx_c2* bank) {
    axis_pass<NZ, 1, Dir>(bank);
    axis_pass<NY, NZ, Dir>(bank);
    axis_pass<NX, NY * NZ, Dir>(bank);
}

#if LRX_MODE < 2 || LRX_MODE == 6
// Modes 0 (pair), 1 (parent) and 6 (plane): the post-pair spin-contracted
// k-convolution; they differ only in the load.
extern "C" __global__ void __launch_bounds__(256) lrx_kconv(
    const lrx_c2* __restrict__ ain, const lrx_c2* __restrict__ bin, lrx_c2* __restrict__ uout,
    long long rows, double scale, unsigned long long perm_l, unsigned long long phase_l,
    unsigned long long perm_r, unsigned long long phase_r, ParentTables tables, PlaneTab plane) {
    extern __shared__ lrx_c2 sm[];
    lrx_c2* abank = sm;
    lrx_c2* bbank = sm + RB * SP;
    lrx_c2* accum = sm + 2 * RB * SP;
    const long long r0 = (long long)blockIdx.x * RB;
    using namespace cufftdx;
#if LRX_MODE == 6
    // Row j of the block is U row (m, g, p) = r0 + j: its offset in D (minus
    // the k, a and b terms) and in F (minus k), computed once per block.
    __shared__ long long d_off[RB];
    __shared__ long long f_off[RB];
    const long long sb_ = plane.p, sm_ = NS * sb_, sa_ = 2 * plane.c * sm_;
    const long long sg_ = NS * sa_, sk_ = plane.g * sg_, fk_ = plane.g * plane.p;
    const lrx_c2* __restrict__ fph = reinterpret_cast<const lrx_c2*>(plane.phase);
    if (threadIdx.x < RB) {
        const long long row = r0 + threadIdx.x, nu = plane.g * plane.p;
        const long long m = row / nu, n = row - m * nu, g = n / plane.p, p = n - g * plane.p;
        d_off[threadIdx.x] = g * sg_ + m * sm_ + p;
        f_off[threadIdx.x] = g * plane.p + p;
    }
    __syncthreads();
#endif
    for (int a = 0; a < NS; ++a) {
        const int ap = (perm_l >> (4*a)) & 15;
        const int pc_l = (phase_l >> (2*a)) & 3;
        for (int b = 0; b < NS; ++b) {
            const int bp = (perm_r >> (4*b)) & 15;
            const int pc = (pc_l + ((phase_r >> (2*b)) & 3)) & 3;
            for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
                const int k = i / RB, j = i % RB;
                const long long row = r0 + j;
                lrx_c2 av = {0.0, 0.0}, bv = {0.0, 0.0};
                if (row < rows) {
#if LRX_MODE == 1
                    av = lrx_parent_load(ain, tables, k, a, b, row, NS, false);
                    bv = lrx_parent_load(bin, tables, k, ap, bp, row, NS, true);
#elif LRX_MODE == 6
                    // P^X = conj(F * D^X): the typed parent load of the
                    // identity plan (mode 1 with every table trivial).
                    const lrx_c2 f = fph[(long long)k * fk_ + f_off[j]];
                    const long long o = (long long)k * sk_ + d_off[j];
                    av = lrx_mul_xla(ain[o + a * sa_ + b * sb_], f);
                    bv = lrx_mul_xla(ain[o + ap * sa_ + plane.c * sm_ + bp * sb_], f);
                    av.y = -av.y;
                    bv.y = -bv.y;
#else
                    const long long base = (long long)k * NS * rows * NS;
                    av = ain[base + (long long)a * rows * NS + row * NS + b];
                    bv = bin[base + (long long)ap * rows * NS + row * NS + bp];
#endif
                }
                abank[j * SP + k] = av;
                bbank[j * SP + k] = bv;
            }
            __syncthreads();
            transform3<fft_direction::inverse>(abank);
            transform3<fft_direction::inverse>(bbank);
            for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
                const int q = (i / NK) * SP + (i % NK);
                const lrx_c2 ac = {abank[q].x, -abank[q].y};
                const lrx_c2 term = lrx_phase(lrx_mul(ac, bbank[q]), pc);
                if (a == 0 && b == 0) accum[q] = term;
                else { accum[q].x += term.x; accum[q].y += term.y; }
            }
            __syncthreads();
        }
    }
    transform3<fft_direction::forward>(accum);
    for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
        const int k = i / RB, j = i % RB;
        const long long row = r0 + j;
        if (row < rows) {
            const lrx_c2 v = accum[j * SP + k];
            uout[(long long)k * rows + row] = {v.x * scale, v.y * scale};
        }
    }
}
#elif LRX_MODE == 2 || LRX_MODE == 3
// Modes 2 (convolution) and 3 (transform) on the k-box stage (kbox_stage.cuh):
// k-LEADING tiles, element (k, r) at k*rows + r.  Single arm (LRX_ARM 0): TR
// columns per tile staged by cp.async into padded shared memory, the 3-D
// transform, [the kernel multiply, the forward transform], the scaled store.
// Split arm (LRX_ARM 1, a k-box whose columns do not fit two per block): the
// passes run through y in place, z,y then x each way as the single arm does,
// so both arms round as the resident kernel they replace:
//   phase 0  plane (z, y) of x -> y        phase 1  pencil x, then [the kernel | the scale]
//   phase 2  plane (z, y) forward, y -> y  phase 3  pencil x forward, then the scale   (mode 2)
// x and y may alias: every element a block stores is one it staged.
#include "kbox_stage.cuh"
constexpr bool CONV = LRX_MODE == 2;
constexpr int TRC = LRX_TR;

struct RowLoad {
    static constexpr bool kDirect = false, kFinish = false;
    const lrx_c2* x;
    long long rows;
    __device__ const lrx_c2* stage(int k, long long c) const { return x + (long long)k * rows + c; }
};
struct RowStore {
    lrx_c2* y;
    long long rows;
    double scale;
    __device__ void put(int k, long long c, lrx_c2 v) const {
        lrx_c2 w;
        w.x = (lrx_real)(v.x * scale);
        w.y = (lrx_real)(v.y * scale);
        y[(long long)k * rows + c] = w;
    }
};
struct KernMid {                                   // mode 2: the stored kernel V[k, x, y]
    const lrx_c2* __restrict__ kern;
    RowGeo g;
    __device__ lrx_c2 operator()(int k, long long row, lrx_c2 v) const {
        return lrx_mul(v, kern[(long long)k * (g.m2 * g.m0) + ((row / g.m1) % g.m2) * g.m0 + row % g.m0]);
    }
};
struct ScaleMid {
    double scale;
    __device__ lrx_c2 operator()(int, long long, lrx_c2 v) const {
        lrx_c2 w;
        w.x = (lrx_real)(v.x * scale);
        w.y = (lrx_real)(v.y * scale);
        return w;
    }
};

template <cufftdx::fft_direction Dir>
__device__ __forceinline__ void lrx_tile3(lrx_c2* sm) {
    lrx_kbox::transform3<NX, NY, NZ, TRC, LRX_SM, Dir>(sm);
}

extern "C" __global__ void __launch_bounds__(LRX_THREADS) lrx_kconv(
    const lrx_c2* x, const lrx_c2* __restrict__ kern, lrx_c2* y, RowGeo g, int phase) {
    extern __shared__ lrx_c2 sm[];
    using namespace cufftdx;
    using lrx_kbox::Plain;
    const bool inv = CONV || !g.forward;               // the first direction
#if LRX_ARM == 0
    (void)phase;
    const RowLoad ld{x, g.rows};
    // Mode 2: the a slabs (rows (a, x, b, y), mx*b*my rows each) read one kernel V[k, x, y], so
    // their tiles are interleaved, a fastest: the a tiles of one (x, y) run together and all but
    // the first find V in L2 (a slab walked alone re-reads V from DRAM a times).  A tile never
    // crosses a slab (its end bounds the tile); mode 3 is one slab.
    const long long seg = CONV ? g.m2 * g.m1 : g.rows;
    const long long na = g.rows / seg, tps = (seg + TRC - 1) / TRC;
    for (long long t = blockIdx.x; t < na * tps; t += gridDim.x) {
        const long long a = t % na, c0 = a * seg + (t / na) * TRC, end = (a + 1) * seg;
        lrx_kbox::stage_tile<NX, NY, NZ, TRC>(sm, c0, end, ld);
        if (inv) lrx_tile3<fft_direction::inverse>(sm);
        else lrx_tile3<fft_direction::forward>(sm);
        if constexpr (CONV) {
            lrx_kbox::mid_tile<NX, NY, NZ, TRC>(sm, c0, end, KernMid{kern, g});
            lrx_tile3<fft_direction::forward>(sm);
        }
        lrx_kbox::store_tile<NX, NY, NZ, TRC>(sm, c0, end, RowStore{y, g.rows, g.scale});
    }
#else
    const Plain<lrx_c2> yy{y, g.rows};
    if (phase == 0) {
        if (inv) lrx_kbox::plane_pass<NX, NY, NZ, LRX_SM, fft_direction::inverse, TRC>(sm, g.rows, RowLoad{x, g.rows}, yy);
        else lrx_kbox::plane_pass<NX, NY, NZ, LRX_SM, fft_direction::forward, TRC>(sm, g.rows, RowLoad{x, g.rows}, yy);
    } else if (phase == 1) {
        if constexpr (CONV)
            lrx_kbox::pencil_pass<NX, NY, NZ, LRX_SM, false, fft_direction::inverse>(y, g.rows, KernMid{kern, g});
        else if (inv)
            lrx_kbox::pencil_pass<NX, NY, NZ, LRX_SM, false, fft_direction::inverse>(y, g.rows, ScaleMid{g.scale});
        else
            lrx_kbox::pencil_pass<NX, NY, NZ, LRX_SM, false, fft_direction::forward>(y, g.rows, ScaleMid{g.scale});
    } else if (phase == 2) {
        lrx_kbox::plane_pass<NX, NY, NZ, LRX_SM, fft_direction::forward, TRC>(sm, g.rows, RowLoad{y, g.rows}, yy);
    } else {
        lrx_kbox::pencil_pass<NX, NY, NZ, LRX_SM, false, fft_direction::forward>(y, g.rows, ScaleMid{g.scale});
    }
#endif
}
#elif LRX_MODE < 7
// Modes 4 (convolution) and 5 (transform) on the k-box stage's single arm: k-MINOR tiles hold
// (r, k) at r*NK + k.  The Load reads each column's k run (k fastest across threads); the kernel
// multiply is the resident family's K[(r / m1) % m0, k]; the Store writes X's layout k fastest
// (out_layout 0) or mode 4's (d0, nk, d3, d1, d4, d2) rows fastest (out_layout 1).  Each line sees
// the family's thread FFT on the same inputs in the same axis order (z, y, x each way), so the
// result is the resident kernel's bit for bit.  x and y may alias (in place): a block stages its
// tile's columns whole before it stores them.  One column always fits (the host refuses a k-box
// row above the opt-in memory), so there is no split arm.
#include "kbox_stage.cuh"
constexpr bool CONV = (LRX_MODE == 4);
constexpr int TRC = LRX_TR;

struct MinorLoad {
    static constexpr bool kDirect = true, kFinish = false;
    const lrx_c2* x;
    template <class View>
    __device__ void direct(const View& view, int k0, int k1, long long col0, int width, long long ncols) const {
        const int nkr = k1 - k0;
        for (int i = threadIdx.x; i < nkr * width; i += blockDim.x) {
            const int k = k0 + i % nkr, j = i / nkr;
            lrx_c2 v = {0.0, 0.0};
            if (col0 + j < ncols) v = x[(col0 + j) * NK + k];
            view(k, j) = v;
        }
    }
};
struct MinorKernMid {                              // mode 4: K[(r / m1) % m0, k], R space
    const lrx_c2* __restrict__ kern;
    RowGeo g;
    __device__ lrx_c2 operator()(int k, long long row, lrx_c2 v) const {
        return lrx_mul(v, kern[((row / g.m1) % g.m0) * NK + k]);
    }
};
struct MinorStore {
    lrx_c2* y;
    RowGeo g;
    __device__ void put(int k, long long row, lrx_c2 v) const {
        lrx_c2 w;
        w.x = (lrx_real)(v.x * g.scale);
        w.y = (lrx_real)(v.y * g.scale);
        long long o = row * NK + k;
        if (LRX_MODE == 4 && g.out_layout == 1) {       // (d0,nk,d3,d1,d4,d2)
            long long t = row;
            const long long i4 = t % g.d4; t /= g.d4;
            const long long i3 = t % g.d3; t /= g.d3;
            const long long i2 = t % g.d2; t /= g.d2;
            const long long i1 = t % g.d1; const long long i0 = t / g.d1;
            o = ((((i0 * NK + k) * g.d3 + i3) * g.d1 + i1) * g.d4 + i4) * g.d2 + i2;
        }
        y[o] = w;
    }
};

extern "C" __global__ void __launch_bounds__(LRX_THREADS) lrx_kconv(
    const lrx_c2* x, const lrx_c2* __restrict__ kern, lrx_c2* y, RowGeo g, int phase) {
    extern __shared__ lrx_c2 sm[];
    using namespace cufftdx;
    (void)phase;
    const bool inv = CONV || !g.forward;               // the first direction
    const MinorLoad ld{x};
    const MinorStore st{y, g};
    const long long tiles = (g.rows + TRC - 1) / TRC;
    for (long long t = blockIdx.x; t < tiles; t += gridDim.x) {
        const long long c0 = t * TRC;
        lrx_kbox::stage_tile<NX, NY, NZ, TRC>(sm, c0, g.rows, ld);
        if (inv) lrx_kbox::transform3<NX, NY, NZ, TRC, LRX_SM, fft_direction::inverse>(sm);
        else lrx_kbox::transform3<NX, NY, NZ, TRC, LRX_SM, fft_direction::forward>(sm);
        if constexpr (CONV) {
            lrx_kbox::mid_tile<NX, NY, NZ, TRC>(sm, c0, g.rows, MinorKernMid{kern, g});
            lrx_kbox::transform3<NX, NY, NZ, TRC, LRX_SM, fft_direction::forward>(sm);
        }
        if (LRX_MODE == 4 && g.out_layout == 1) lrx_kbox::store_tile<NX, NY, NZ, TRC>(sm, c0, g.rows, st);
        else lrx_kbox::store_tile_kfast<NX, NY, NZ, TRC>(sm, c0, g.rows, st);
    }
}
#elif LRX_MODE <= 9 || LRX_MODE == 11
// Modes 7, 8, 9 and 11 share the unfold load below.
// Row r of a convolution is (pair, a, b) = (r / NS^2, (r % NS^2) / NS, r % NS)
// with pair = x*my + y; U[k, a, x, b, y] is stored spin-major.  When a block
// holds whole spin groups (the usual case) its load reads the NS*NS sources of a
// pair once and runs the spin action in registers for all NS*NS rows; when fewer
// rows fit (large k-grids) each row loads its own element, with the same arithmetic.
// NR is the right endpoint's width: NS for a Green (modes 7, 8); a Lorentz
// block's own width for mode 9, whose left width is NS.
#ifndef LRX_NSR
#define LRX_NSR LRX_NS
#endif
constexpr int NR = LRX_NSR;
constexpr int SS = NS * NR;
// Mode 7's output spin block: rows (a, b) in [a0, a0 + NA) x [b0, b0 + NA) of U G U^dagger
// (NA = NS: the whole spin group, every other mode).  A pass reads every source of a pair and
// transforms and stores only its block, so a caller bounds the output tile by passes.
#ifndef LRX_NA
#define LRX_NA LRX_NS
#endif
constexpr int NA = LRX_NA;
constexpr int SSO = (NA == NS) ? SS : NA * NA;   // rows per pair

// The typed unfold of one (k, x, y) pair (symmetry_maps unfold_isdf_operator,
// axis-local pair_transpose arm): source row, both endpoint gathers, then
// (mph * G) * nph, a -1 source being an exact zero; and U_k.
// (mph * G) * nph, a -1 source being an exact zero.  On an antiunitary row
// conj_trs selects the rule: 0 reads the partner tile; 1 conjugates the phased
// product (a Hermitian interaction); 2 reads conj(G) from G itself (a Green of
// real weights, whose partner IS conj(G): no partner tile exists).
// SL, SR: the left and right endpoint widths of the operand (template parameters, so one
// program loads a Green (NS, NS) and an interaction's Lorentz block (nA, nB) side by side).
template <int SL, int SR>
__device__ __forceinline__ void lrx_unfold_pair(
    const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt, const UnfoldTab& t,
    int k, long long xx, long long yy, lrx_c2 (&g)[SL][SR], lrx_c2 (&u)[SL][SL],
    lrx_c2 (&ur)[SR][SR]) {
    const lrx_c2* __restrict__ mph = reinterpret_cast<const lrx_c2*>(t.mph);
    const lrx_c2* __restrict__ nph = reinterpret_cast<const lrx_c2*>(t.nph);
    const lrx_c2* __restrict__ spin = reinterpret_cast<const lrx_c2*>(t.spin);
    const lrx_c2* __restrict__ spin_r =
        reinterpret_cast<const lrx_c2*>(t.spin_r ? t.spin_r : t.spin);
    const bool anti = t.trs[k] != 0, conj_row = anti && t.conj_trs == 1;
    const bool conj_src = anti && t.conj_trs == 2;
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
            lrx_c2 v = {0.0, 0.0};
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

// Row a of U G Ur^dagger, accumulated exactly as the spin-rotate FFI does
// (Ur = U for a Green).
template <int SL, int SR>
__device__ __forceinline__ void lrx_spin_row(const lrx_c2 (&u)[SL][SL], const lrx_c2 (&ur)[SR][SR],
                                             const lrx_c2 (&g)[SL][SR], int a, lrx_c2 (&out)[SR]) {
    lrx_c2 left[SR];
#pragma unroll
    for (int d = 0; d < SR; ++d) {
        lrx_c2 v = {0.0, 0.0};
#pragma unroll
        for (int c = 0; c < SL; ++c) lrx_cmac(v, u[a][c], g[c][d]);
        left[d] = v;
    }
#pragma unroll
    for (int b = 0; b < SR; ++b) {
        lrx_c2 v = {0.0, 0.0};
#pragma unroll
        for (int d = 0; d < SR; ++d) lrx_cmac_conj(v, left[d], ur[b][d]);
        out[b] = v;
    }
}

// Mode 7's stored x block (every other mode: every x): block row r is the left centroid
// x = (r / bx)*xs + x0 + r % bx, xn pieces of bx rows at stride xs, so a pass reads only its own
// pairs' sources; x >= mx is a padding row, loaded and stored as zero.
struct XBlock { long long x0, bx, xs, rows, mx; };
template <int SL = NS>
__device__ __forceinline__ XBlock lrx_x_block(const UnfoldTab& t) {
    const long long mx = t.ml / SL;
    return t.bx > 0 ? XBlock{t.x0, t.bx, t.xs, t.xn * t.bx, mx} : XBlock{0, mx, mx, mx, mx};
}
__device__ __forceinline__ long long lrx_x_of(const XBlock& b, long long r) {
    return (r / b.bx) * b.xs + b.x0 + r % b.bx;
}

// The typed unfold of a tile of columns into a direct Load's view (kbox_stage.cuh stage_tile):
// column c is (pair, a, b), pair = p0 + c / SSO = (block row, y) of the stored x block
// (lrx_x_block), and (a, b) its element of U O Ur^dagger (mode 7's NA < NS: of the stored spin
// block).  The tile holds whole spin groups: one thread per (k, pair) reads the pair's sources once
// and forms its elements in registers.  A pair past `pairs` or a padding row loads as zero.  Modes 7,
// 8 and 9 load through it.
// SL, SR: the operand's endpoint widths and SA its stored spin block (SA == SL: the whole group);
// the defaults are the program's Green (NS, NR) and mode 7's block (NA).  Mode 8 also loads an
// interaction's Lorentz block through <nA, nB, nA> (WedgeLoad beside LorLoad).
template <int SL = NS, int SR = NR, int SA = NA, class View>
__device__ __forceinline__ void lrx_unfold_direct(
    const View& view, const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt, const UnfoldTab& t,
    long long p0, long long my, long long pairs, int k0, int k1, long long col0, int width, long long ncols) {
    constexpr int SSO_ = (SA == SL) ? SL * SR : SA * SA;   // columns per pair
    const XBlock xb = lrx_x_block<SL>(t);
    const int groups = width / SSO_;
    for (int i = threadIdx.x; i < (k1 - k0) * groups; i += blockDim.x) {
        const int k = k0 + i / groups, j = i % groups;
        const long long pr = p0 + col0 / SSO_ + j;
        const long long rx = pr / my, yy = pr - rx * my, xx = lrx_x_of(xb, rx);
        if (pr < pairs && xx < xb.mx) {
            lrx_c2 g[SL][SR], u[SL][SL], ur[SR][SR];
            lrx_unfold_pair(gp, gt, t, k, xx, yy, g, u, ur);
#pragma unroll
            for (int a = 0; a < SL; ++a) {
                if constexpr (SA != SL) { if (a < t.a0 || a >= t.a0 + SA) continue; }
                lrx_c2 out[SR];
                lrx_spin_row(u, ur, g, a, out);
#pragma unroll
                for (int b = 0; b < SR; ++b) {
                    if constexpr (SA != SL) {
                        if (b < t.b0 || b >= t.b0 + SA) continue;
                        view(k, j * SSO_ + (a - t.a0) * SA + (b - t.b0)) = out[b];
                    } else {
                        view(k, j * SSO_ + a * SR + b) = out[b];
                    }
                }
            }
        } else {
#pragma unroll
            for (int ab = 0; ab < SSO_; ++ab) view(k, j * SSO_ + ab) = {0.0, 0.0};
        }
    }
}

#if LRX_TT
// The tile-table load of modes 7 and 11 (the host sets LRX_TT when two blocks of bank + tables
// fit an SM; kbox_stage.cuh UnfoldTiles prices them).  The block stages U_k and the per-k source
// rows once; per tile of TP pairs the endpoint tables (lsrc/rsrc slices, the phases mph/nph and,
// for mode 7, the pairs' kernel W_R[k, x, y]) go by cp.async one tile ahead (two buffers).  The
// raw sources go by cp.async straight into the bank cells that finish them, one per cell with
// indices from shared memory, then two shared-memory passes finish the typed unfold: per (k,
// operand group, column d) the phases (mph * G) * nph and left = U g; per (k, operand group, row
// a) left U^dagger.  The products and their order are lrx_unfold_pair's and lrx_spin_row's, so
// the tile is the register load's bit for bit.  The right action is U itself (neither door passes
// spin_r) and the whole spin group is loaded (NA == NS).
#include "kbox_stage.cuh"
static_assert(NR == NS && NA == NS, "the tile tables load whole Green spin groups");
#if LRX_MODE == 11
constexpr int TT_OPS = 2, TT_NW = 0;           // Gv and Gc per pair, no staged kernel
#else
constexpr int TT_OPS = 1, TT_NW = 1;           // one Green per pair, W_R[k, x, y] staged
#endif
constexpr int TP = LRX_TP;                     // pairs per tile
constexpr int TT_GRP = TT_OPS * SS;            // bank rows per pair
constexpr int TT_ROWS = TP * TT_GRP;           // bank rows per tile
constexpr int TT_GT = TP * TT_OPS;             // operand groups per tile
constexpr lrx_kbox::UnfoldTiles kTT{NK, NS, NR, TP, TT_NW};
constexpr long long TT_U = kTT.u(), TT_MP = kTT.mp(0), TT_NP = kTT.np(0), TT_W = kTT.w(0),
                    TT_OFF = kTT.off(), TT_LS = kTT.ls(0), TT_RS = kTT.rs(0), TT_FLAG = kTT.flag();
// Bank cell of row j at flat k: the k-box stage's padded row (odd z-line and row strides).
constexpr int TT_RSTRIDE = lrx_kbox::Geo<NX, NY, NZ>::RS;
__device__ __forceinline__ int tt_cell(int j, int k) { return j * TT_RSTRIDE + lrx_kbox::Geo<NX, NY, NZ>::at(k); }
struct TileTabs {
    char* s;
    __device__ lrx_c2* u() const { return reinterpret_cast<lrx_c2*>(s + TT_U); }
    __device__ lrx_c2* mp(int b) const { return reinterpret_cast<lrx_c2*>(s + TT_MP) + b * (TP * NS * NK); }
    __device__ lrx_c2* np(int b) const { return reinterpret_cast<lrx_c2*>(s + TT_NP) + b * (TP * NR * NK); }
    __device__ lrx_c2* w(int b) const { return reinterpret_cast<lrx_c2*>(s + TT_W) + b * (TP * TT_NW * NK); }
    __device__ long long* off() const { return reinterpret_cast<long long*>(s + TT_OFF); }
    __device__ int* ls(int b) const { return reinterpret_cast<int*>(s + TT_LS) + b * (TP * NK * NS); }
    __device__ int* rs(int b) const { return reinterpret_cast<int*>(s + TT_RS) + b * (TP * NK * NR); }
    __device__ int* flag() const { return reinterpret_cast<int*>(s + TT_FLAG); }
};

// Once per block: U_k (k innermost) by cp.async; the per-k source row offsets and the flags
// (bit 0 the partner tile, bit 1 conj(G), bit 2 conj the phased product) by plain stores.
__device__ __forceinline__ void tt_fixed(const UnfoldTab& t, const TileTabs& s) {
    const lrx_c2* spin = reinterpret_cast<const lrx_c2*>(t.spin);
    for (int i = threadIdx.x; i < NK * NS * NS; i += blockDim.x) {
        const int k = i % NK, e = i / NK;
        lrx_async::copy<16>(s.u() + i, spin + (long long)k * NS * NS + e);
    }
    for (int k = threadIdx.x; k < NK; k += blockDim.x) {
        const bool anti = t.trs[k] != 0;
        s.off()[k] = (long long)t.row[k] * t.ml * t.nl;
        s.flag()[k] = (anti && t.conj_trs == 0 ? 1 : 0) | (anti && t.conj_trs == 2 ? 2 : 0) |
                      (anti && t.conj_trs == 1 ? 4 : 0);
    }
}

// The tables of pairs [pr0, pr0 + npr) (npr <= TP; pair = x*my + y) into buffer b by cp.async
// (the caller commits): lsrc/rsrc slices [jp][k][c] (one copy of NS or NR indices), mph/nph
// [jp][c][k], and with TT_NW the kernel kern[(k*mx + x)*my + y] as w [jp][k].
__device__ __forceinline__ void tt_tile(const UnfoldTab& t, const TileTabs& s, int b, long long pr0, int npr,
                                        long long my, const lrx_c2* kern, long long mx) {
    const lrx_c2* mph = reinterpret_cast<const lrx_c2*>(t.mph);
    const lrx_c2* nph = reinterpret_cast<const lrx_c2*>(t.nph);
    const XBlock xb = lrx_x_block(t);                  // pair = (block row)*my + y; row -> x
    const long long x0 = pr0 / my, y0 = pr0 - x0 * my;
    for (int i = threadIdx.x; i < TP * NK; i += blockDim.x) {
        const int jp = i / NK, k = i % NK;
        if (jp >= npr) continue;
        long long xx = x0, yy = y0 + jp;
        while (yy >= my) { yy -= my; ++xx; }
        xx = lrx_x_of(xb, xx);
        if (xx >= xb.mx) {                             // a padding row: every source an exact zero
#pragma unroll
            for (int c = 0; c < NS; ++c) s.ls(b)[i * NS + c] = -1;
            xx = xb.mx - 1;                            // the tables below read a real row (unused)
        } else {
            lrx_async::copy<4 * NS>(s.ls(b) + i * NS, t.lsrc + (long long)k * t.ml + xx * NS);
        }
        lrx_async::copy<4 * NR>(s.rs(b) + i * NR, t.rsrc + (long long)k * t.nl + yy * NR);
        if constexpr (TT_NW > 0) lrx_async::copy<16>(s.w(b) + i, kern + ((long long)k * mx + xx) * my + yy);
    }
    for (int i = threadIdx.x; i < TP * (NS + NR) * NK; i += blockDim.x) {
        const int k = i % NK, q = i / NK, jp = q / (NS + NR), e = q % (NS + NR);
        if (jp >= npr) continue;
        long long xx = x0, yy = y0 + jp;
        while (yy >= my) { yy -= my; ++xx; }
        xx = min(lrx_x_of(xb, xx), xb.mx - 1);
        if (e < NS)
            lrx_async::copy<16>(s.mp(b) + (jp * NS + e) * NK + k, mph + (long long)k * t.ml + xx * NS + e);
        else
            lrx_async::copy<16>(s.np(b) + (jp * NR + e - NS) * NK + k, nph + (long long)k * t.nl + yy * NR + (e - NS));
    }
}

// One cp.async per cell of the tile, a -1 source or a pair past npr an exact zero; bank row j is
// (pair j / TT_GRP, operand (j / SS) % TT_OPS: g0 or g1, element j % SS).  Consecutive threads
// take consecutive rows of one k (the right sources of a centroid are contiguous); k fastest,
// which makes a warp's shared destinations contiguous, measured 6% slower (mode 11, A100).
__device__ __forceinline__ void tt_gather(const lrx_c2* g0, const lrx_c2* g0t, const lrx_c2* g1,
                                          const lrx_c2* g1t, const UnfoldTab& t, const TileTabs& s, int b,
                                          int npr, lrx_c2* bank) {
    const int* ls = s.ls(b);
    const int* rs = s.rs(b);
    for (int i = threadIdx.x; i < NK * TT_ROWS; i += blockDim.x) {
        const int j = i % TT_ROWS, k = i / TT_ROWS;
        const int jp = j / TT_GRP, op = (j / SS) % TT_OPS, e = j % SS;
        bool valid = false;
        const lrx_c2* src = g0;
        if (jp < npr) {
            const int l = ls[(jp * NK + k) * NS + e / NR], r = rs[(jp * NK + k) * NR + e % NR];
            const int f = s.flag()[k];
            valid = l >= 0 && r >= 0;
            if (valid)
                src = (op ? ((f & 1) ? g1t : g1) : ((f & 1) ? g0t : g0)) + s.off()[k] + (long long)l * t.nl + r;
        }
        lrx_async::cell16(bank + tt_cell(j, k), src, valid);
    }
}

// The typed unfold of the staged cells in place, from shared tables only; ends with a barrier.
// NS consecutive lanes of one warp own one (k, operand group): lane d forms column d's phases and
// left = U g (pass 1), then, after a warp barrier, lane a forms row a of left U^dagger (pass 2),
// so the column-to-row exchange never leaves the warp.  Every lane of a warp runs the same number
// of rounds (the warp barrier needs the whole warp); k is fastest across the groups of a warp.
static_assert(lrx_kbox::kWarp % NS == 0, "a warp holds whole spin groups");
__device__ __forceinline__ void tt_finish(const TileTabs& s, int b, int npr, lrx_c2* bank) {
    const lrx_c2* u = s.u();
    const lrx_c2* mp = s.mp(b);
    const lrx_c2* np = s.np(b);
    const int* ls = s.ls(b);
    const int* rs = s.rs(b);
    constexpr int CS = NR * TT_RSTRIDE;            // cell (c, d) of a group at col[c * CS]
    constexpr int ITEMS = NK * TT_GT * NS;
    for (int i0 = 0; i0 < ITEMS; i0 += blockDim.x) {
        const int i = i0 + threadIdx.x, g = i / NS, e = i % NS;
        const int k = g % NK, gi = g / NK, jp = gi / TT_OPS;
        const bool live = i < ITEMS && jp < npr;
        if (live) {                                    // pass 1: column d = e
            const int d = e;
            lrx_c2* col = bank + tt_cell(gi * SS + d, k);
            const int f = s.flag()[k];
            const bool conj_src = f & 2, conj_row = f & 4;
            const int r = rs[(jp * NK + k) * NR + d];
            const lrx_c2 nq = np[(jp * NR + d) * NK + k];
            lrx_c2 gcol[NS];
#pragma unroll
            for (int c = 0; c < NS; ++c) {
                lrx_c2 v = {0.0, 0.0};
                if (ls[(jp * NK + k) * NS + c] >= 0 && r >= 0) {
                    lrx_c2 sv = col[c * CS];
                    if (conj_src) sv.y = -sv.y;
                    v = lrx_mulf(lrx_mulf(mp[(jp * NS + c) * NK + k], sv), nq);
                    if (conj_row) v.y = -v.y;
                }
                gcol[c] = v;
            }
#pragma unroll
            for (int aa = 0; aa < NS; ++aa) {
                lrx_c2 v = {0.0, 0.0};
#pragma unroll
                for (int c = 0; c < NS; ++c) lrx_cmac(v, u[(aa * NS + c) * NK + k], gcol[c]);
                col[aa * CS] = v;
            }
        }
        __syncwarp();
        if (live) {                                    // pass 2: row a = e
            lrx_c2* row = bank + tt_cell(gi * SS + e * NR, k);
            lrx_c2 left[NR];
#pragma unroll
            for (int d = 0; d < NR; ++d) left[d] = row[d * TT_RSTRIDE];
#pragma unroll
            for (int bb = 0; bb < NR; ++bb) {
                lrx_c2 v = {0.0, 0.0};
#pragma unroll
                for (int d = 0; d < NR; ++d) lrx_cmac_conj(v, left[d], u[(bb * NR + d) * NK + k]);
                row[bb * TT_RSTRIDE] = v;
            }
        }
        __syncwarp();
    }
    __syncthreads();
}
#endif

#if LRX_MODE == 7
// Mode 7 on the k-box stage (kbox_stage.cuh), as mode 11: tiles of TRC columns, column c = pair
// c / SSO, spin element (a, b) = ((c % SSO) / NA, c % NA) of the stored block; pair = (stored x
// block row, y) (lrx_x_block).  Load, inverse transform, the Mid V[k, x, y] (W_R), forward
// transform, the scaled store through kout.  The load forms lrx_unfold_pair and lrx_spin_row, the
// Mid is lrx_mul, the store scales as the family's resident kernel did, and each line sees the same
// cuFFTDx thread FFT on the same inputs in the same axis order: bitwise with the resident arm this
// replaced.  Two loads, chosen at build (LRX_TT): the register load (a thread per (k, pair) forms a
// whole spin group; a tile narrower than one group forms one element per column), or the tile
// tables (n_s 4; tt_* above): a persistent grid, tile n + 1's tables (with its W_R) beside tile n's
// gather.
#include "kbox_stage.cuh"
constexpr int TRC = LRX_TR;                    // tile columns
using KG = lrx_kbox::Geo<NX, NY, NZ>;

struct M7Store {                               // U[(ko, a, rx, b, y)], full-k row k at kout[k]
    lrx_c2* y;
    const UnfoldTab* t;
    long long rows, my;
    double scale;
    long long c0;                              // the split arm's chunk: its first column
    __device__ void put(int k, long long col, lrx_c2 v) const {
        const long long ko = lrx_out_row(*t, k);
        if (ko < 0) return;
        const long long pr = (c0 + col) / SSO, rx = pr / my, yy = pr - rx * my;
        const int m = int((c0 + col) % SSO), a = m / NA, b = m % NA;
        y[((ko * NA + a) * rows + rx) * (my * NA) + b * my + yy] = {v.x * scale, v.y * scale};
    }
};

#if LRX_TT
static_assert(TRC == TT_ROWS, "the host passes the tile's pairs and columns together");
#else
static_assert(TRC % SSO == 0, "tiles hold whole spin groups");
struct M7Load {                                // the register load of the typed unfold, whole groups
    static constexpr bool kDirect = true, kFinish = false;
    const lrx_c2 *gp, *gt;
    const UnfoldTab* t;
    long long my, pairs, p0;                   // p0: the split arm's chunk, its first pair
    template <class View>
    __device__ void direct(const View& view, int k0, int k1, long long col0, int width, long long ncols) const {
        lrx_unfold_direct(view, gp, gt, *t, p0, my, pairs, k0, k1, col0, width, ncols);
    }
};

struct M7Mid {                                 // the kernel V[k, x, y] of a pair, times its columns
    const lrx_c2* __restrict__ kern;
    XBlock xb;
    long long my, p0;
    __device__ lrx_c2 w(int k, long long pr) const {
        const long long rx = pr / my, yy = pr - rx * my;
        return kern[((long long)k * xb.mx + min(lrx_x_of(xb, rx), xb.mx - 1)) * my + yy];
    }
    __device__ lrx_c2 operator()(int k, long long col, lrx_c2 v) const { return lrx_mul(v, w(k, p0 + col / SSO)); }
    template <class Get>                       // whole groups: one thread per (k, pair) reads V once
    __device__ void group(int k, long long pr, const Get& get) const {
        const lrx_c2 wv = w(k, pr);
#pragma unroll
        for (int q = 0; q < SSO; ++q) get(q) = lrx_mul(get(q), wv);
    }
};
#endif

#ifndef LRX_MINB
#define LRX_MINB 1
#endif
// yb, pb0, npairs: the split arms (a k-box that cannot hold a spin group of two or more columns a
// block) through the chunk yb of npairs * SSO columns, one entry point per pass so each runs at its
// own register budget (one kernel holds every pass at the largest pass's registers).  lrx_kconv:
// the plane (z, y) pass from the load.  LRX_ARM 1, as modes 2/3 (yb k-leading): lrx_kconv_mid the
// x pencil and the Mid, lrx_kconv_pass the plane (z, y) forward (phase 2) and the x pencil forward
// and the store (phase 3).  LRX_ARM 2, one column's box fits a block (yb k-minor): lrx_kconv_col
// the rest on the resident column (kbox_stage.cuh column_pass).  The same lines in the same axis
// order as the single arm: bitwise.
#define M7_ARGS const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt, \
    const lrx_c2* __restrict__ kern, lrx_c2* __restrict__ y, UnfoldTab t, double scale, \
    lrx_c2* yb, long long pb0, long long npairs, int phase, const int* __restrict__ live
// live: the window rows [live[0], live[1]) of a padded pass (kbox_stage.cuh Live; null: all).
#define M7_LIVE(zero) lrx_kbox::Live::rows(live, (t.nl / NR) * SSO, pb0 * SSO, zero)
#if LRX_ARM >= 1
extern "C" __global__ void __launch_bounds__(LRX_THREADS) lrx_kconv(M7_ARGS) {
    extern __shared__ lrx_c2 sm[];
    (void)kern; (void)y; (void)scale; (void)phase;
    const XBlock xb = lrx_x_block(t);
    const long long my = t.nl / NR, nc = npairs * SSO;
    const M7Load ld{gp, gt, &t, my, xb.rows * my, pb0};
#if LRX_ARM == 2
    lrx_kbox::plane_pass<NX, NY, NZ, LRX_SM, cufftdx::fft_direction::inverse, TRC, true, true>(
        sm, nc, ld, lrx_kbox::PlainK<lrx_c2>{yb, (long long)KG::NK}, M7_LIVE(false));
#else
    lrx_kbox::plane_pass<NX, NY, NZ, LRX_SM, cufftdx::fft_direction::inverse, TRC, false, true>(
        sm, nc, ld, lrx_kbox::Plain<lrx_c2>{yb, nc}, M7_LIVE(false));
#endif
}
#if LRX_ARM == 2
extern "C" __global__ void __launch_bounds__(LRX_THREADS2) lrx_kconv_col(M7_ARGS) {
    extern __shared__ lrx_c2 sm[];
    (void)gp; (void)gt; (void)phase;
    const XBlock xb = lrx_x_block(t);
    const long long my = t.nl / NR;
    lrx_kbox::column_pass<NX, NY, NZ, LRX_SM>(sm, yb, npairs * SSO, M7Mid{kern, xb, my, pb0},
                                              M7Store{y, &t, xb.rows, my, scale, pb0 * SSO}, M7_LIVE(true));
}
#else
struct M7Id {
    __device__ lrx_c2 operator()(int, long long, lrx_c2 v) const { return v; }
};
extern "C" __global__ void __launch_bounds__(LRX_THREADS2) lrx_kconv_mid(M7_ARGS) {
    (void)gp; (void)gt; (void)y; (void)scale; (void)phase;
    const XBlock xb = lrx_x_block(t);
    lrx_kbox::pencil_pass<NX, NY, NZ, LRX_SM, false, cufftdx::fft_direction::inverse>(
        yb, npairs * SSO, M7Mid{kern, xb, t.nl / NR, pb0}, M7_LIVE(false));
}
extern "C" __global__ void __launch_bounds__(LRX_THREADS3) lrx_kconv_pass(M7_ARGS) {
    extern __shared__ lrx_c2 sm[];
    using namespace cufftdx;
    (void)gp; (void)gt; (void)kern;
    const XBlock xb = lrx_x_block(t);
    const long long my = t.nl / NR, nc = npairs * SSO;
    const lrx_kbox::Plain<lrx_c2> yy{yb, nc};
    if (phase == 2)
        lrx_kbox::plane_pass<NX, NY, NZ, LRX_SM, fft_direction::forward, TRC>(sm, nc, yy, yy, M7_LIVE(false));
    else
        lrx_kbox::pencil_pass<NX, NY, NZ, LRX_SM, false, fft_direction::forward>(
            yb, nc, M7Id{}, M7Store{y, &t, xb.rows, my, scale, pb0 * SSO}, M7_LIVE(true));
}
#endif
#else
extern "C" __global__ void __launch_bounds__(LRX_THREADS, LRX_MINB) lrx_kconv(M7_ARGS) {
    extern __shared__ lrx_c2 sm[];
    using namespace cufftdx;
    (void)yb; (void)pb0; (void)npairs; (void)phase;
    const XBlock xb = lrx_x_block(t);
    const long long my = t.nl / NR, pairs = xb.rows * my, ncols = pairs * SSO;
    const M7Store st{y, &t, xb.rows, my, scale, 0};
    const lrx_kbox::Live lv = M7_LIVE(true);           // the live columns [cb, ce), whole pairs
    const long long cb = lv.b(), ce = lv.e(ncols);
#if LRX_TT
    const TileTabs s{reinterpret_cast<char*>(sm + TRC * KG::RS)};
    const long long stride = (long long)gridDim.x * TP, pe = ce / SS;
    auto npr_of = [&](long long q) { return (int)min((long long)TP, pe - q); };
    long long p0 = cb / SS + (long long)blockIdx.x * TP;
    tt_fixed(t, s);
    if (p0 < pe) tt_tile(t, s, 0, p0, npr_of(p0), my, kern, xb.mx);
    lrx_async::commit();
    for (int b = 0; p0 < pe; p0 += stride, b ^= 1) {
        lrx_async::wait_all();
        __syncthreads();                               // tables b in; the previous tile's bank reads done
        const int npr = npr_of(p0);
        tt_gather(gp, gt, gp, gt, t, s, b, npr, sm);
        lrx_async::commit();
        if (p0 + stride < pe) tt_tile(t, s, b ^ 1, p0 + stride, npr_of(p0 + stride), my, kern, xb.mx);
        lrx_async::commit();
        lrx_async::wait_prior<1>();                     // this tile's cells (not the next tables)
        __syncthreads();
        tt_finish(s, b, npr, sm);
        lrx_kbox::transform3<NX, NY, NZ, TRC, LRX_SM, fft_direction::inverse>(sm);
        const lrx_c2* w = s.w(b);
        for (int i = threadIdx.x; i < TRC * NK; i += blockDim.x) {
            const int k = i / TRC, j = i % TRC;
            if (j / SS < npr) sm[tt_cell(j, k)] = lrx_mul(sm[tt_cell(j, k)], w[(j / SS) * NK + k]);
        }
        __syncthreads();
        lrx_kbox::transform3<NX, NY, NZ, TRC, LRX_SM, fft_direction::forward>(sm);
        lrx_kbox::store_tile<NX, NY, NZ, TRC>(sm, p0 * SS, ce, st);   // the loop top syncs
    }
#else
    const M7Load ld{gp, gt, &t, my, pairs, 0};
    const M7Mid mid{kern, xb, my, 0};
    for (long long col0 = cb + (long long)blockIdx.x * TRC; col0 < ce; col0 += (long long)gridDim.x * TRC) {
        lrx_kbox::stage_tile<NX, NY, NZ, TRC>(sm, col0, ce, ld);
        lrx_kbox::transform3<NX, NY, NZ, TRC, LRX_SM, fft_direction::inverse>(sm);
        lrx_kbox::mid_group_tile<NX, NY, NZ, TRC, SSO>(sm, col0, ce, mid);
        lrx_kbox::transform3<NX, NY, NZ, TRC, LRX_SM, fft_direction::forward>(sm);
        lrx_kbox::store_tile<NX, NY, NZ, TRC>(sm, col0, ce, st);
    }
#endif
    lrx_kbox::zero_dead<KG::NK, lrx_c2>(ncols, lv, st);
}
#endif
#elif LRX_MODE == 8
// Mode 8 runs on the k-box stage (kbox_stage.cuh) in two arms, chosen at build by kbox_plan from the
// grid and the opt-in shared memory (a tile of whole spin groups, ns^2 columns each):
//   LRX_ARM 0, single pass, when a spin group fits: stage_tile with the unfolded Green as a direct
//            Load (every spin element of a pair, formed as the family's unfold load forms it), the
//            inverse transform, the vertex sum as a group Mid (one thread per (k, pair)), the
//            forward transform, the scaled kout Store: the resident kernel it replaced, on the
//            stage's padded bank.
//   LRX_ARM 1, the split arm, where it does not, chunked over pairs through the (nk, npairs * SS)
//            intermediate y:
//     phase 0  plane (z, y) inverse of the unfolded Green -> y
//     phase 1  group pencil: the x inverse of every member; each (member, t) thread forms its
//              member's vertex sum from the group's values and V[k, x, A, y, B] (staged by cp.async)
//     phase 2  plane (z, y) forward, y -> y
//     phase 3  pencil x forward, the scales, the store into U (kout rows)
// Per (k, pair, a, b), with g = s_g * (the transformed Green), the sum runs block by block in (A, B)
// order: acc += (i^code g[perm_A[a]][perm_B[b]]) * V[A, B], XLA's product (no FMA) and __dadd_rn
// from zero, as the scan over blocks did; the transforms run z,y then x each way.  Both arms round
// alike, and as the resident kernel they replaced (one thread per (k, pair), 182 registers: its
// Mid ran on PB*NK threads, 144 of 256 on CrI3 6x6 ns 4).
#include "kbox_stage.cuh"
static_assert(NR == NS, "mode 8 loads Greens: one spin width");
constexpr int TRC = LRX_TR;                    // tile columns (whole spin groups)

struct LorArgs {
    const lrx_c2 *gp, *gt, *kern;              // parent Green, partner, V (nk, mx, na, my, nb)
    lrx_c2* u;                                 // (n_out, NS, mx, NS, my)
    lrx_c2* y;                                 // split arm: the (NK, npairs * SS) intermediate
    long long p0, npairs, mx, my;              // this launch's pairs [p0, p0 + npairs)
    const lrx_c2 *wp, *wt;                     // LRX_WA: the interaction's parent tile and partner
    lrx_c2* yw;                                // LRX_WA split arm: the (NK, npairs * WS) W_R chunk
    double sw;                                 // LRX_WA: the interaction's inverse-transform scale
    const int* live;                           // the window rows [live[0], live[1]) of a padded pass; null: all
};

// LRX_WA > 0: the interaction is read from its irreducible-q parent tile (nA = LRX_WA, nB = LRX_WB
// Lorentz components per endpoint) through its own unfold tables tw, on the load of its inverse
// transform (mode 9's WedgeLoad): no full-q W and no full-grid W_R.  A pair's nA*nB W columns sit
// beside its NS*NS Green columns in the tile; the vertex Mid reads W_R = sw * IFFT(unfolded W)
// from the bank, the value mode 9 stores, so the result is the two-door chain's bit for bit.
#ifndef LRX_WA
#define LRX_WA 0
#define LRX_WB 0
#endif
#if LRX_WA > 0
constexpr int WA = LRX_WA, WB = LRX_WB, WS = WA * WB;
struct LorWLoad {                              // W column c: pair p0 + c / WS, block (c % WS) / WB, c % WB
    static constexpr bool kDirect = true, kFinish = false;
    const LorArgs* a;
    const UnfoldTab* t;
    template <class View>
    __device__ void direct(const View& view, int k0, int k1, long long col0, int width,
                           long long ncols) const {
        lrx_unfold_direct<WA, WB, WA>(view, a->wp, a->wt, *t, a->p0, a->my, a->p0 + ncols / WS,
                                                        k0, k1, col0, width, ncols);
    }
};
#endif

struct LorLoad {                               // tile column c: pair p0 + c / SS, member c % SS
    static constexpr bool kDirect = true, kFinish = false;
    const LorArgs* a;
    const UnfoldTab* t;
    template <class View>
    __device__ void direct(const View& view, int k0, int k1, long long col0, int width,
                           long long ncols) const {
        lrx_unfold_direct(view, a->gp, a->gt, *t, a->p0, a->my, a->p0 + ncols / SS, k0, k1, col0, width,
                                ncols);
    }
};

// A member's vertex tables: block e = (A, B) = (e / nb, e % nb) reads group element src[e] =
// perm_A[a] * NS + perm_B[b] with the quarter turn code[e] = phase_A[a] - phase_B[b] (member = a*NS + b).
struct LorMember {
    int src[16], code[16], ne;
};
__device__ __forceinline__ LorMember lor_member(const LorentzTab& v, int member) {
    LorMember m;
    const int ma = member / NS, mb = member % NS;
    m.ne = v.na * v.nb;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int ia = e / v.nb, ib = e % v.nb;
        if (e < m.ne) {
            const int pa = (int)((v.perm_l >> (16 * ia + 4 * ma)) & 15);
            const int ca = (int)((v.phase_l >> (8 * ia + 2 * ma)) & 3);
            const int pb = (int)((v.perm_r >> (16 * ib + 4 * mb)) & 15);
            const int cb = (int)((v.phase_r >> (8 * ib + 2 * mb)) & 3);
            m.src[e] = pa * NS + pb;
            m.code[e] = (ca + 4 - cb) & 3;
        } else {
            m.src[e] = 0;
            m.code[e] = 0;
        }
    }
    return m;
}

struct LorFinal {                              // (z s_f) mult
    double s_f, mult;
    __device__ lrx_c2 operator()(int, long long, lrx_c2 z) const {
        return {__dmul_rn(__dmul_rn(z.x, s_f), mult), __dmul_rn(__dmul_rn(z.y, s_f), mult)};
    }
};

// The functors hold the kernel arguments they use by value: a store through a->u could alias the
// argument struct in the compiler's view and reload it after every store.
struct LorStore {                              // U[(ko, a, x, b, y)], full-k row k at kout[k]
    lrx_c2* u;
    const int* kout;
    long long p0, mx, my, nl;
    __device__ LorStore(const LorArgs& a, const UnfoldTab& t)
        : u(a.u), kout(t.kout), p0(a.p0), mx(a.mx), my(a.my), nl(t.nl) {}
    __device__ void put(int k, long long col, lrx_c2 v) const {
        const long long ko = kout ? (long long)kout[k] : (long long)k;
        if (ko < 0) return;
        const long long pr = p0 + col / SS, xx = pr / my, yy = pr - xx * my;
        const int m = int(col % SS), ma = m / NS, mb = m % NS;
        u[((ko * NS + ma) * mx + xx) * nl + mb * my + yy] = v;
    }
};

#if LRX_ARM == 0
// The vertex sum as a group Mid: one thread per (k, pair) holds the pair's NS*NS transformed
// values and accumulates every member, block by block in (A, B) order, reading V through L1 (the
// resident kernel's Mid; a member Mid, one thread per (k, pair, member), measured 0.76-0.94x).
struct LorGroupMid {
    const lrx_c2* kern;
    long long p0, mx, my;
    LorentzTab v;
#if LRX_WA > 0
    const lrx_c2* wbank;                       // the tile's W columns, transformed, unscaled
    long long g0;                              // the tile's first pair (group) index
    double sw;
#endif
    template <class Get>
    __device__ void group(int k, long long grp, const Get& get) const {
        lrx_c2 acc[SS];
#pragma unroll
        for (int ab = 0; ab < SS; ++ab) {
            acc[ab].x = 0.0;
            acc[ab].y = 0.0;
        }
#if LRX_WA > 0
        constexpr int RSW = lrx_kbox::Geo<NX, NY, NZ>::RS;
        const lrx_c2* __restrict__ w = wbank + (grp - g0) * WS * RSW + lrx_kbox::Geo<NX, NY, NZ>::at(k);
#else
        const long long pr = p0 + grp, xx = pr / my, yy = pr - xx * my;
        const long long wy = my * v.nb, wx = (long long)v.na * wy;
        const lrx_c2* __restrict__ w = kern + ((long long)k * mx + xx) * wx + yy * v.nb;
#endif
        for (int ia = 0; ia < v.na; ++ia) {
            for (int ib = 0; ib < v.nb; ++ib) {
#if LRX_WA > 0
                const lrx_c2 wz = w[(ia * WB + ib) * RSW];
                const lrx_c2 wv = {__dmul_rn(wz.x, sw), __dmul_rn(wz.y, sw)};
#else
                const lrx_c2 wv = w[ia * wy + ib];
#endif
#pragma unroll
                for (int aa = 0; aa < NS; ++aa) {
                    const int pa = (int)((v.perm_l >> (16 * ia + 4 * aa)) & 15);
                    const int ca = (int)((v.phase_l >> (8 * ia + 2 * aa)) & 3);
#pragma unroll
                    for (int bb = 0; bb < NS; ++bb) {
                        const int pb = (int)((v.perm_r >> (16 * ib + 4 * bb)) & 15);
                        const int cb = (int)((v.phase_r >> (8 * ib + 2 * bb)) & 3);
                        // g = s_g * (the transformed value), read from the bank where it is used (a
                        // runtime-indexed register array would live in local memory)
                        const lrx_c2 z = get(pa * NS + pb);
                        const lrx_c2 g = {__dmul_rn(z.x, v.s_g), __dmul_rn(z.y, v.s_g)};
                        const lrx_c2 p = lrx_mul_xla(lrx_phase(g, (ca + 4 - cb) & 3), wv);
                        acc[aa * NS + bb].x = __dadd_rn(acc[aa * NS + bb].x, p.x);
                        acc[aa * NS + bb].y = __dadd_rn(acc[aa * NS + bb].y, p.y);
                    }
                }
            }
        }
#pragma unroll
        for (int ab = 0; ab < SS; ++ab) get(ab) = acc[ab];
    }
};

struct LorScaledStore {
    LorStore st;
    LorFinal fin;
    __device__ void put(int k, long long col, lrx_c2 z) const { st.put(k, col, fin(k, col, z)); }
};

extern "C" __global__ void __launch_bounds__(LRX_THREADS) lrx_kconv(LorArgs a, UnfoldTab t, LorentzTab v, int phase,
                                                                  UnfoldTab tw) {
    extern __shared__ lrx_c2 sm[];
    using namespace cufftdx;
    (void)phase;
    (void)tw;
    const long long ncols = a.npairs * SS;
    const LorLoad ld{&a, &t};
    const LorScaledStore st{LorStore(a, t), LorFinal{v.s_f, v.mult}};
#if LRX_WA > 0
    // The W columns of the tile's pairs follow its Green columns in the bank (same padded row
    // stride), so one inverse transform of TRC + TRW rows serves both.
    constexpr int TRW = (TRC / SS) * WS;
    lrx_c2* smw = sm + (long long)TRC * lrx_kbox::Geo<NX, NY, NZ>::RS;
    const long long nw = a.npairs * WS;
    const LorWLoad wld{&a, &tw};
#endif
    const lrx_kbox::Live lv = lrx_kbox::Live::rows(a.live, a.my * SS, a.p0 * SS, true);
    const long long ce = lv.e(ncols);
    for (long long c0 = lv.b() + (long long)blockIdx.x * TRC; c0 < ce; c0 += (long long)gridDim.x * TRC) {
        lrx_kbox::stage_tile<NX, NY, NZ, TRC>(sm, c0, ce, ld);
#if LRX_WA > 0
        lrx_kbox::stage_tile<NX, NY, NZ, TRW>(smw, (c0 / SS) * WS, (ce / SS) * WS, wld);
        lrx_kbox::transform3<NX, NY, NZ, TRC + TRW, LRX_SM, fft_direction::inverse>(sm);
        const LorGroupMid mid{a.kern, a.p0, a.mx, a.my, v, smw, c0 / SS, a.sw};
#else
        lrx_kbox::transform3<NX, NY, NZ, TRC, LRX_SM, fft_direction::inverse>(sm);
        const LorGroupMid mid{a.kern, a.p0, a.mx, a.my, v};
#endif
        // k fastest across threads: consecutive threads read one group's column at consecutive k,
        // distinct banks (pairs fastest put a phase's 8 threads 16 padded columns apart, one bank).
        lrx_kbox::mid_group_tile<NX, NY, NZ, TRC, SS, true, true>(sm, c0, ce, mid);
        lrx_kbox::transform3<NX, NY, NZ, TRC, LRX_SM, fft_direction::forward>(sm);
        lrx_kbox::store_tile<NX, NY, NZ, TRC>(sm, c0, ce, st);
    }
    lrx_kbox::zero_dead<lrx_kbox::Geo<NX, NY, NZ>::NK, lrx_c2>(ncols, lv, st);
}
#else
// Split arm, phase 1 (the vertex pencil, kbox_stage.cuh pencil_group_warp_pass with a pair's SS
// columns on one warp): a member's vertex sum from the group's values and V, in the staged
// pencil's order (bitwise), V read through L1; the R-space line goes back to y.
struct LorWarpMid {                            // held by value: a store through y must not reload them
    lrx_c2* y;
    const lrx_c2 *kern, *yw;
    long long ncols, npairs, p0, mx, my;
    LorentzTab v;
    LorMember mem;
    template <class Get>
    __device__ lrx_c2 value(int k, long long inst, int, const Get& get) const {
#if LRX_WA > 0
        const lrx_c2* __restrict__ aux = yw + ((long long)k * npairs + inst) * WS;
#else
        const long long pr = p0 + inst, xx = pr / my, yy = pr - xx * my, sa = my * v.nb;   // sa: V's A stride
        const lrx_c2* __restrict__ aux = kern + (((long long)k * mx + xx) * v.na * my + yy) * v.nb;
#endif
        lrx_c2 acc = {0.0, 0.0};
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            if (e < mem.ne) {                          // the same on every lane: get stays warp-wide
                const lrx_c2 z = get(mem.src[e]);
                const lrx_c2 g = {__dmul_rn(z.x, v.s_g), __dmul_rn(z.y, v.s_g)};
#if LRX_WA > 0
                const lrx_c2 ve = aux[e];
#else
                const lrx_c2 ve = aux[(e / v.nb) * sa + e % v.nb];
#endif
                const lrx_c2 q = lrx_mul_xla(lrx_phase(g, mem.code[e]), ve);
                acc.x = __dadd_rn(acc.x, q.x);
                acc.y = __dadd_rn(acc.y, q.y);
            }
        }
        return acc;
    }
    __device__ void finish(long long p, long long inst, int member, bool live, const lrx_c2 (&g)[NX]) const {
        if (!live) return;
#pragma unroll
        for (int kx = 0; kx < NX; ++kx) y[((long long)kx * NY * NZ + p) * ncols + inst * SS + member] = g[kx];
    }
};

struct LorYLoad {                              // the intermediate, staged by cp.async
    static constexpr bool kDirect = false, kFinish = false;
    const lrx_c2* y;
    long long n;
    __device__ const lrx_c2* stage(int k, long long c) const { return y + (long long)k * n + c; }
};

#if LRX_WA > 0
struct LorWScale {                             // the chunk's W_R scale, as mode 9's store applies it
    double sw;
    __device__ lrx_c2 operator()(int, long long, lrx_c2 v) const { return {v.x * sw, v.y * sw}; }
};
#endif

extern "C" __global__ void __launch_bounds__(LRX_THREADS) lrx_kconv(LorArgs a, UnfoldTab t, LorentzTab v, int phase,
                                                                  UnfoldTab tw) {
    extern __shared__ lrx_c2 sm[];
    using namespace cufftdx;
    (void)tw;
    const long long ncols = a.npairs * SS;
    const lrx_kbox::Plain<lrx_c2> yy{a.y, ncols};
#if LRX_WA > 0
    // phases 4 and 5 (before phase 1 of each chunk): the chunk's W_R into yw (NK, npairs*WS), as phase
    // 0 does the Green's: a plane (z, y) inverse of the unfolded W on plane tiles of LRX_TRW whole
    // Lorentz groups (the grouped load reads a pair's tables once per (k, pair)), then the x inverse
    // with mode 9's scale in place.  Same lines, same axis order: mode 9's single arm bit for bit.
    if (phase == 4 || phase == 5) {
        const long long nw = a.npairs * WS;
        const lrx_kbox::Live lw = lrx_kbox::Live::rows(a.live, a.my * WS, a.p0 * WS, false);
        if (phase == 4)
            lrx_kbox::plane_pass<NX, NY, NZ, LRX_SM, fft_direction::inverse, LRX_TRW>(
                sm, nw, LorWLoad{&a, &tw}, lrx_kbox::Plain<lrx_c2>{a.yw, nw}, lw);
        else
            lrx_kbox::pencil_pass<NX, NY, NZ, LRX_SM, false, fft_direction::inverse>(a.yw, nw, LorWScale{a.sw}, lw);
        return;
    }
#endif
    const lrx_kbox::Live lv = lrx_kbox::Live::rows(a.live, a.my * SS, a.p0 * SS, phase == 3);
    if (phase == 0) {
        lrx_kbox::plane_pass<NX, NY, NZ, LRX_SM, fft_direction::inverse, TRC>(sm, ncols, LorLoad{&a, &t}, yy, lv);
    } else if (phase == 1) {
        const LorWarpMid mid{a.y, a.kern, a.yw, ncols, a.npairs, a.p0, a.mx, a.my, v,
                             lor_member(v, (int)(threadIdx.x % SS))};
        lrx_kbox::pencil_group_warp_pass<NX, NY, NZ, LRX_SM, SS, LRX_THREADS>(
            a.y, ncols, a.npairs, mid, lrx_kbox::Live::rows(a.live, a.my, a.p0, false));
    } else if (phase == 2) {
        lrx_kbox::plane_pass<NX, NY, NZ, LRX_SM, fft_direction::forward, TRC>(sm, ncols, LorYLoad{a.y, ncols}, yy, lv);
    } else {
        lrx_kbox::pencil_pass<NX, NY, NZ, LRX_SM, false, fft_direction::forward>(
            a.y, ncols, LorFinal{v.s_f, v.mult}, LorStore(a, t), lv);
    }
}
#endif
#elif LRX_MODE == 9
// Mode 9: mode 3's inverse transform of the interaction unfolded on its load from the wedge tiles.
// Tile column c is row (pair, A, B) = (c / SS, (c % SS) / NR, c % NR); Y k-LEADING (nk, ml, nl) at
// merged endpoints (x*NS + A, y*NR + B), scaled as mode 3 scales.  The tiles hold whole spin groups:
// one thread per (k, pair) reads the pair's sources once and forms every (A, B) of U O Ur^dagger.
// Single arm (two groups fit a block): one HBM pass.  Split arm (as modes 2/3): phase 0 the plane
// (z, y) pass from the load into Y unscaled, phase 1 the x pencil and the scale in place on Y (a
// pure transform: Y's own columns serve).  The values, the thread FFTs and their axis order are
// the resident kernel's either way, so the result is too, bit for bit.
#include "kbox_stage.cuh"
constexpr int TRC = LRX_TR;                    // tile columns (whole spin groups)

struct WedgeLoad {
    static constexpr bool kDirect = true, kFinish = false;
    const lrx_c2 *gp, *gt;
    const UnfoldTab* t;
    long long my;
    template <class View>
    __device__ void direct(const View& view, int k0, int k1, long long col0, int width,
                           long long ncols) const {
        lrx_unfold_direct(view, gp, gt, *t, 0, my, ncols / SS, k0, k1, col0, width, ncols);
    }
};

struct WedgeStore {
    lrx_c2* y;
    const UnfoldTab* t;
    long long my;
    double scale;
    __device__ void put(int k, long long c, lrx_c2 v) const {
        const long long pr = c / SS, xx = pr / my, yy = pr - xx * my;
        const int a = (int)((c % SS) / NR), b = (int)(c % NR);
        y[((long long)k * t->ml + xx * NS + a) * t->nl + yy * NR + b] = {v.x * scale, v.y * scale};
    }
};

struct WedgeScale {                            // the split arm's last axis: the store's scale
    double scale;
    __device__ lrx_c2 operator()(int, long long, lrx_c2 v) const { return {v.x * scale, v.y * scale}; }
};

// live: the window rows [live[0], live[1]) of a padded pass (null: all); a row is my * SS tile
// columns, and as many of Y's own columns (the split arm's pencil).
extern "C" __global__ void __launch_bounds__(LRX_THREADS) lrx_kconv(
    const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt, lrx_c2* __restrict__ y,
    UnfoldTab t, double scale, int phase, const int* __restrict__ live) {
    extern __shared__ lrx_c2 sm[];
    using namespace cufftdx;
    const long long my = t.nl / NR, ncols = (t.ml / NS) * my * SS;
    const WedgeLoad ld{gp, gt, &t, my};
#if LRX_ARM == 0
    (void)phase;
    const WedgeStore st{y, &t, my, scale};
    const lrx_kbox::Live lv = lrx_kbox::Live::rows(live, my * SS, 0, true);
    const long long ce = lv.e(ncols);
    for (long long c0 = lv.b() + (long long)blockIdx.x * TRC; c0 < ce; c0 += (long long)gridDim.x * TRC) {
        lrx_kbox::stage_tile<NX, NY, NZ, TRC>(sm, c0, ce, ld);
        lrx_kbox::transform3<NX, NY, NZ, TRC, LRX_SM, fft_direction::inverse>(sm);
        lrx_kbox::store_tile<NX, NY, NZ, TRC>(sm, c0, ce, st);
    }
    lrx_kbox::zero_dead<lrx_kbox::Geo<NX, NY, NZ>::NK, lrx_c2>(ncols, lv, st);
#else
    if (phase == 0)
        lrx_kbox::plane_pass<NX, NY, NZ, LRX_SM, fft_direction::inverse, TRC>(
            sm, ncols, ld, WedgeStore{y, &t, my, 1.0}, lrx_kbox::Live::rows(live, my * SS, 0, false));
    else
        lrx_kbox::pencil_pass<NX, NY, NZ, LRX_SM, false, fft_direction::inverse>(
            y, ncols, WedgeScale{scale}, lrx_kbox::Live::rows(live, my * SS, 0, true));
#endif
}
#else
// Mode 11: the chi0 two-operand pass on the k-box stage (kbox_stage.cuh).
// Tile column c of pair p = c / GRP: operand c % GRP / SS (0 = Gv, 1 = Gc) and
// spin element (a, b) = (c % SS) / NS, c % NS, the unfolded parent Green
// U_k G[row(k)] U_k^dagger of that operand (mode 7's load; conj_trs = 2 reads
// the antiunitary partner as conj(G)).  Inverse k-transform, then per (k, pair)
//     v = sum_ab conj(si Gc'_ab) * (si Gv'_ab)          (a-major, as the spin-pair stream)
//     v = v + conj(v)                                   (LRX_COMPLETE: a real contour)
//     acc[o, k, x, y] += alpha[o] * v                   (o < n_out)
// chi_R accumulates in R space; one forward transform follows the tau sum.
#include "kbox_stage.cuh"
static_assert(NR == NS, "mode 11 loads Greens: one spin width");
constexpr int GRP = 2 * SS;                    // columns per pair
constexpr int TRC = LRX_TR;                    // tile columns (the plan's tr groups * GRP)

struct ChiArgs {
    const lrx_c2 *gv, *gvt, *gc, *gct;         // parent Greens and partners (unread for conj_trs 1, 2)
    lrx_c2* acc;                               // (n_out, nk, mx, my), accumulated in place
    const lrx_c2* alpha;                       // (n_out,)
    lrx_c2* y;                                 // split arm: the (nk, ncols) intermediate
    long long p0, npairs, pairs, my;           // this launch's pairs [p0, p0 + npairs) of pairs
    int n_out;
    double si;                                 // the inverse transform's scale
    // LRX_VTX: the four-current channels.  Vertex i of a side is packed as mode 8's LorentzTab
    // (perm entry (i, a) at bits 16*i + 4*a, phase code (i, a) at bits 8*i + 2*a); channel
    // ch = i*vnb + j writes acc planes [ch*n_out, (ch+1)*n_out).  vch0: the split arm's first
    // channel of this pencil pass (members vch0 .. vch0 + GRP - 1).
    unsigned long long vperm_l, vphase_l, vperm_r, vphase_r;
    int vna, vnb, vch0;
    const double* sgn_c;                       // LRX_VTX: (nk) real +-1 on the Gc operand's load; null: none
    const int* live;                           // the window rows [live[0], live[1]) of a padded pass; null: all
};

struct ChiLoad {
    static constexpr bool kDirect = true, kFinish = false;
    const ChiArgs* a;
    const UnfoldTab* t;
    // Every (k, operand group) of the tile: the NS*NS spin elements of one operand of one pair.
    template <class View>
    __device__ void direct(const View& view, int k0, int k1, long long col0, int width,
                           long long ncols) const {
        constexpr int og = SS;
        const int groups = width / og;
        for (int i = threadIdx.x; i < (k1 - k0) * groups; i += blockDim.x) {
            const int k = k0 + i / groups, j = i % groups;
            const long long c0 = col0 + (long long)j * og;
            const long long pr = a->p0 + c0 / GRP;
            const int op = int((c0 % GRP) / og);
            if (c0 < ncols) {
                const long long xx = pr / a->my, yy = pr - xx * a->my;
                lrx_c2 g[NS][NR], u[NS][NS], ur[NR][NR];
                lrx_unfold_pair(op ? a->gc : a->gv, op ? a->gct : a->gvt, *t, k, xx, yy, g, u, ur);
#if LRX_VTX
                // A Dirac-half quadrant's own sign p_k relative to the shared tables (exact).
                const double sg = (op && a->sgn_c) ? a->sgn_c[k] : 1.0;
#endif
#pragma unroll
                for (int r = 0; r < NS; ++r) {
                    lrx_c2 out[NR];
                    lrx_spin_row(u, ur, g, r, out);
#pragma unroll
                    for (int b = 0; b < NR; ++b) {
#if LRX_VTX
                        out[b].x *= sg; out[b].y *= sg;
#endif
                        view(k, j * og + r * NR + b) = out[b];
                    }
                }
            } else {
#pragma unroll
                for (int e = 0; e < og; ++e) view(k, j * og + e) = {0.0, 0.0};
            }
        }
    }
};

// The pair's R-space value from its GRP transformed values g(q) (q < SS: Gv, else Gc).
template <class Get>
__device__ __forceinline__ lrx_c2 lrx_chi_value(const Get& g, double si) {
    lrx_c2 v = {0.0, 0.0};
#pragma unroll
    for (int q = 0; q < SS; ++q) {
        const lrx_c2 gv = g(q), gc = g(SS + q);
        const lrx_c2 a = {gc.x * si, -(gc.y * si)};
        const lrx_c2 b = {gv.x * si, gv.y * si};
        lrx_cmac(v, a, b);
    }
#if LRX_COMPLETE
    v = {__dadd_rn(v.x, v.x), __dsub_rn(v.y, v.y)};
#endif
    return v;
}

// acc[o, k, pair] += alpha[o] * v for every output o.  acc is (n_out, nk, mx, my) and a pair is
// x*my + y, so the element sits at (o*nk + k)*pairs + pair.
__device__ __forceinline__ void lrx_chi_acc(const ChiArgs& a, int k, long long pr, lrx_c2 v) {
    for (int o = 0; o < a.n_out; ++o) {
        lrx_c2* e = a.acc + ((long long)o * NK + k) * a.pairs + pr;
        lrx_c2 w = *e;
        lrx_cmac(w, a.alpha[o], v);
        *e = w;
    }
}

#if LRX_VTX
// Channel ch = (i, j): the four-current trace of the pair's two transformed operands,
//     v = sum_ab conj(ph_l[i,a]) ph_r[j,b] conj(si Gc'[pl a, pr b]) (si Gv'_ab)   (a-major)
// the product conj(ph_l conj(ph_r) si Gc') being an exact rotation; the identity vertex is
// lrx_chi_value's product bit for bit.  The vertices act on the Gc operand's spin indices.
template <class Get>
__device__ __forceinline__ lrx_c2 lrx_chi_vertex(const Get& g, const ChiArgs& a, int ch) {
    const int i = ch / a.vnb, j = ch - (ch / a.vnb) * a.vnb;
    lrx_c2 v = {0.0, 0.0};
#pragma unroll
    for (int p = 0; p < NS; ++p) {
        const int pa = (int)((a.vperm_l >> (16 * i + 4 * p)) & 15);
        const int ca = (int)((a.vphase_l >> (8 * i + 2 * p)) & 3);
#pragma unroll
        for (int r = 0; r < NS; ++r) {
            const int pb = (int)((a.vperm_r >> (16 * j + 4 * r)) & 15);
            const int cb = (int)((a.vphase_r >> (8 * j + 2 * r)) & 3);
            const lrx_c2 gv = g(p * NS + r), gc = g(SS + pa * NS + pb);
            const lrx_c2 w = lrx_phase({gc.x * a.si, gc.y * a.si}, (ca + 4 - cb) & 3);
            const lrx_c2 x = {w.x, -w.y};
            const lrx_c2 y = {gv.x * a.si, gv.y * a.si};
            lrx_cmac(v, x, y);
        }
    }
    return v;
}

// acc plane ch*n_out + o += alpha[o] * v.
__device__ __forceinline__ void lrx_chi_acc_ch(const ChiArgs& a, int k, long long pr, int ch, lrx_c2 v) {
    for (int o = 0; o < a.n_out; ++o) {
        lrx_c2* e = a.acc + ((long long)(ch * a.n_out + o) * NK + k) * a.pairs + pr;
        lrx_c2 w = *e;
        lrx_cmac(w, a.alpha[o], v);
        *e = w;
    }
}
#endif

// Single arm: the group Mid reduces the pair's GRP transformed values and accumulates them.
// mid_group_tile runs one thread per (k, pair) of the tile, so every thread of the block takes
// part in the accumulation (a column-wise Store would leave it to the pairs' leading columns).
struct ChiMid {
    const ChiArgs* a;
    template <class Get>
    __device__ void group(int k, long long g, const Get& get) const {
#if LRX_VTX
        for (int ch = 0; ch < a->vna * a->vnb; ++ch)
            lrx_chi_acc_ch(*a, k, a->p0 + g, ch, lrx_chi_vertex(get, *a, ch));
#else
        lrx_chi_acc(*a, k, a->p0 + g, lrx_chi_value(get, a->si));
#endif
    }
};

#if LRX_ARM == 1
// Split arm, pencil pass (phase 1; kbox_stage.cuh pencil_group_warp_pass with a pair's GRP columns
// on one warp).  Per kx every lane of the pair forms the value lrx_chi_value / lrx_chi_vertex forms on
// the single arm, from the same operands in the same order.  The plain pass spreads the accumulation
// over the pair's lanes (lane m takes kx = m, m + GRP, ...); a vertex pass accumulates channel
// vch0 + m on lane m (all kx).  Each lane issues its acc loads before its stores: one lane updating
// 20 kx serially (the stores may alias the next load) was the pass's latency.
struct ChiWarpMid {
    const ChiArgs* a;
    int ch, chv;                                  // vertex pass: this lane's channel; chv < channels
    template <class Get>
    __device__ lrx_c2 value(int, long long, int, const Get& get) const {
#if LRX_VTX
        return lrx_chi_vertex(get, *a, chv);      // a lane past the channels computes chv and discards
#else
        return lrx_chi_value(get, a->si);
#endif
    }
    __device__ void finish(long long p, long long inst, int member, bool live, const lrx_c2 (&v)[NX]) const {
        constexpr long long PL = (long long)NY * NZ;
        const long long pr = a->p0 + inst;
#if LRX_VTX
        if (!live || ch >= a->vna * a->vnb) return;
        constexpr int CH = 4;                         // acc loads in flight per lane
        for (int o = 0; o < a->n_out; ++o) {
            lrx_c2* e = a->acc + ((long long)(ch * a->n_out + o) * NK + p) * a->pairs + pr;
#pragma unroll
            for (int k0 = 0; k0 < NX; k0 += CH) {
                lrx_c2 w4[CH];
#pragma unroll
                for (int i = 0; i < CH; ++i)
                    if (k0 + i < NX) w4[i] = e[(long long)(k0 + i) * PL * a->pairs];
#pragma unroll
                for (int i = 0; i < CH; ++i)
                    if (k0 + i < NX) { lrx_cmac(w4[i], a->alpha[o], v[k0 + i]); e[(long long)(k0 + i) * PL * a->pairs] = w4[i]; }
            }
        }
#else
        constexpr int J = (NX + GRP - 1) / GRP;       // kx rows per lane
        lrx_c2 mine[J];
#pragma unroll
        for (int kx = 0; kx < NX; ++kx)
            if (kx % GRP == member) mine[kx / GRP] = v[kx];
        if (!live) return;
        for (int o = 0; o < a->n_out; ++o) {
            lrx_c2* e = a->acc + ((long long)o * NK + p) * a->pairs + pr;
            lrx_c2 w3[J];
#pragma unroll
            for (int j = 0; j < J; ++j)
                if (member + j * GRP < NX) w3[j] = e[(long long)(member + j * GRP) * PL * a->pairs];
#pragma unroll
            for (int j = 0; j < J; ++j)
                if (member + j * GRP < NX) {
                    lrx_cmac(w3[j], a->alpha[o], mine[j]);
                    e[(long long)(member + j * GRP) * PL * a->pairs] = w3[j];
                }
        }
#endif
    }
};
#endif

// LRX_MINB: blocks per SM the plan's shared memory admits (2 with the tile tables); the register
// budget follows it.
#ifndef LRX_MINB
#define LRX_MINB 1
#endif
extern "C" __global__ void __launch_bounds__(LRX_THREADS, LRX_MINB) lrx_kconv(ChiArgs a, UnfoldTab t, int phase) {
    extern __shared__ lrx_c2 sm[];
    using namespace cufftdx;
    const long long ncols = a.npairs * GRP;
    const ChiLoad ld{&a, &t};
    // The live columns [cb, ce) (whole pairs); dead pairs add nothing to the accumulator.
    const lrx_kbox::Live lv = lrx_kbox::Live::rows(a.live, a.my * GRP, a.p0 * GRP, false);
    const long long cb = lv.b(), ce = lv.e(ncols);
#if LRX_ARM == 0 && LRX_TT
    (void)phase;
    (void)ld;
    // A persistent grid (the host launches the resident blocks): tile n + 1's tables load beside
    // tile n's gather, then the finish, the inverse transform and the accumulating Mid.
    const ChiMid mid{&a};
    static_assert(TRC == TT_ROWS, "the host passes the tile's pairs and columns together");
    const TileTabs s{reinterpret_cast<char*>(sm + TRC * lrx_kbox::Geo<NX, NY, NZ>::RS)};
    const long long stride = (long long)gridDim.x * TRC;
    auto npr_of = [&](long long c0) { return (int)min((long long)TP, (ce - c0) / GRP); };
    long long col0 = cb + (long long)blockIdx.x * TRC;
    tt_fixed(t, s);
    if (col0 < ce) tt_tile(t, s, 0, a.p0 + col0 / GRP, npr_of(col0), a.my, nullptr, 0);
    lrx_async::commit();
    for (int b = 0; col0 < ce; col0 += stride, b ^= 1) {
        lrx_async::wait_all();
        __syncthreads();                               // tables b in; the previous tile's bank reads done
        const int npr = npr_of(col0);
        tt_gather(a.gv, a.gvt, a.gc, a.gct, t, s, b, npr, sm);
        lrx_async::commit();
        if (col0 + stride < ce)
            tt_tile(t, s, b ^ 1, a.p0 + (col0 + stride) / GRP, npr_of(col0 + stride), a.my, nullptr, 0);
        lrx_async::commit();
        lrx_async::wait_prior<1>();                     // this tile's cells (not the next tables)
        __syncthreads();
        tt_finish(s, b, npr, sm);
        lrx_kbox::transform3<NX, NY, NZ, TRC, LRX_SM, fft_direction::inverse>(sm);
        lrx_kbox::mid_group_tile<NX, NY, NZ, TRC, GRP, false>(sm, col0, ce, mid);   // the loop top syncs
    }
#elif LRX_ARM == 0
    (void)phase;
    const ChiMid mid{&a};
    for (long long col0 = cb + (long long)blockIdx.x * TRC; col0 < ce; col0 += (long long)gridDim.x * TRC) {
        lrx_kbox::stage_tile<NX, NY, NZ, TRC>(sm, col0, ce, ld);
        lrx_kbox::transform3<NX, NY, NZ, TRC, LRX_SM, fft_direction::inverse>(sm);
        lrx_kbox::mid_group_tile<NX, NY, NZ, TRC, GRP>(sm, col0, ce, mid);
    }
#else
    if (phase == 0) {
        lrx_kbox::plane_pass<NX, NY, NZ, LRX_SM, fft_direction::inverse, TRC>(
            sm, ncols, ld, lrx_kbox::Plain<lrx_c2>{a.y, ncols}, lv);
    } else {
        const int ch = a.vch0 + (int)(threadIdx.x % GRP), nch = a.vna * a.vnb;
        lrx_kbox::pencil_group_warp_pass<NX, NY, NZ, LRX_SM, GRP, LRX_THREADS>(
            a.y, ncols, a.npairs, ChiWarpMid{&a, ch, ch < nch ? ch : nch - 1},
            lrx_kbox::Live::rows(a.live, a.my, a.p0, false));
    }
#endif
}
#endif
#endif
)__lrx__";

// ---------------------------------------------------------------------------
//  Mode 10: the plane FFT with gather-on-load.  Its own embedded source: it
//  shares the build, the NVRTC options and the cubin cache with the family,
//  not its kernels.  Compile-time: LRX_NX = n_b, LRX_NY = n_c (the plane),
//  LRX_NZ = b1, LRX_NS = c1: the Good-Thomas splits n_b = b1*b2 and
//  n_c = c1*c2 (gcd 1, every factor <= 40; b2 = 1 only for a prime power <= 40), so
//  every line FFT is a cuFFTDx thread FFT and the splits need index maps only,
//  no twiddle.  A block keeps LRX_PB whole (n_b, n_c|1) planes in shared
//  memory: row passes on the occupied rows, column passes on all columns, one
//  coalesced store.  HBM traffic is one read of the cylinder and one write of
//  the plane.
// ---------------------------------------------------------------------------
static const char* kPlaneSrc = R"__lrx__(
#include <cufftdx.hpp>
#include "lrx_async_gather.cuh"

struct __align__(16) lrx_c2 { double x, y; };
struct PlaneGather {
    const int* gidx;       // (rows, n_c): cylinder column of each cell of an occupied row, -1 = empty
    const int* row_of;     // (rows,): the plane row b of occupied row r
    const int* start;      // () slab start on F's axis 1
    long long rows, n_col, planes, s_len, n_pg, inner;
};

constexpr int NB = LRX_NX, NC = LRX_NY;
constexpr int B1 = LRX_NZ, B2 = NB / B1, C1 = LRX_NS, C2 = NC / C1;
constexpr int LD = NC | 1;             // odd row pitch: the column lines read conflict-free
constexpr int ROWS = LRX_ROWS;         // the table's occupied rows (the loops also guard the run-time count)
constexpr bool STAGE = LRX_STAGE;      // a (ROWS, NC) staging block per plane fits beside the planes
constexpr int PB = LRX_PB;             // planes per block

template <int M>
using TFFT = decltype(cufftdx::Size<M>() + cufftdx::Precision<double>() +
                      cufftdx::Type<cufftdx::fft_type::c2c>() +
                      cufftdx::Direction<cufftdx::fft_direction::forward>() +
                      cufftdx::Thread() + cufftdx::SM<LRX_SM>());

// Good-Thomas: for N = N1 N2 with gcd(N1, N2) = 1 the length-N DFT is the
// N1 x N2 DFT of x[(N2 i1 + N1 i2) mod N], and its output (k1, k2) is X[k] for
// k = k1 (mod N1), k = k2 (mod N2).  Run in place, frequency k ends at slot(k).
template <int N, int N1, int N2>
__device__ __forceinline__ int slot(int k) { return (N2 * (k % N1) + N1 * (k % N2)) % N; }

// One factor pass on one line: the M elements at positions (step e + off) mod N
// (element stride es) of `base`.  FIRST: M = N1, step = N2, off = N1 i; else
// M = N2, step = N1, off = N2 i.  `live` (the first column pass only) reads a
// dead plane row as zero; `src` (the first row pass with staging) reads the
// line from the staging block and writes it to `base`.
template <int N, int N1, int N2, bool FIRST, bool SRC = false>
__device__ __forceinline__ void pfa_line(lrx_c2* base, int es, int i, const unsigned char* live,
                                         const lrx_c2* src = nullptr) {
    constexpr int M = FIRST ? N1 : N2;
    if constexpr (M > 1) {
        using F = TFFT<M>;
        using V = typename F::value_type;
        constexpr int step = FIRST ? N2 : N1;
        const int off = (FIRST ? N1 : N2) * i;
        V v[F::storage_size];
#pragma unroll
        for (int e = 0; e < M; ++e) {
            const int p = (step * e + off) % N;
            lrx_c2 z = {0.0, 0.0};
            if constexpr (SRC) z = src[p * es];               // the staged row (es = 1)
            else if (live == nullptr || live[p]) z = base[p * es];
            v[e].x = z.x; v[e].y = z.y;
        }
        F().execute(v);
#pragma unroll
        for (int e = 0; e < M; ++e) {
            const int p = (step * e + off) % N;
            base[p * es].x = v[e].x; base[p * es].y = v[e].y;
        }
    }
}

// PB planes per block (small planes share a block, so a pass has enough lines);
// LRX_RB blocks per SM the shared memory allows (capped at 2), so the register
// cap lets them all in.  Blocks are persistent (the host caps the grid at the
// resident count).  With STAGE the next group's occupied cells are gathered
// asynchronously (lrx_async, cp.async) into a (ROWS, NC) staging block per
// plane while this group's passes run; the first row pass reads them there.
// The table is a run-time argument; its row count ROWS is compiled in, so the
// index math divides by constants (a run-time count cost 2-15% at 25^2-54^2).
extern "C" __global__ void __launch_bounds__(LRX_THREADS, LRX_RB) lrx_kconv(
    const lrx_c2* __restrict__ fin, lrx_c2* __restrict__ yout, PlaneGather g) {
    extern __shared__ lrx_c2 buf[];                  // PB planes of (NB, LD), then PB staging (ROWS, NC)
    constexpr int PS = NB * LD, SS = ROWS * NC;
    lrx_c2* stg = buf + PB * PS;
    __shared__ unsigned char live[NB];
    __shared__ int rowb[ROWS];
    __shared__ long long foff[PB];
    const int rows = static_cast<int>(g.rows);       // <= ROWS
    for (int b = threadIdx.x; b < NB; b += blockDim.x) live[b] = 0;
    __syncthreads();
    for (int r = threadIdx.x; r < rows; r += blockDim.x) { rowb[r] = g.row_of[r]; live[g.row_of[r]] = 1; }
    // Output plane (a, j, r) of Y (A, n_pg, inner) reads F plane (a, start + j, r)
    // of F (A, s_len, inner): the slab F[:, start:start+n_pg] in place.
    long long st = *g.start;
    st = st < 0 ? 0 : (st > g.s_len - g.n_pg ? g.s_len - g.n_pg : st);
    const long long groups = (g.planes + PB - 1) / PB;
    // The cylinder offsets of group gi's planes, then one async gather of their
    // occupied cells (zeros implicit) into the staging blocks or the planes.
    auto issue = [&](long long gi) {
        const long long p0 = gi * PB;
        const int np = g.planes - p0 < PB ? static_cast<int>(g.planes - p0) : PB;
        if (threadIdx.x < np) {
            const long long plane = p0 + threadIdx.x;
            const long long a = plane / (g.n_pg * g.inner), rem = plane - a * g.n_pg * g.inner;
            foff[threadIdx.x] = ((a * g.s_len + st) * g.inner + rem) * g.n_col;
        }
        __syncthreads();
        for (int t = threadIdx.x; t < np * SS; t += blockDim.x) {
            const int q = t / SS, u = t - q * SS, r = u / NC;
            if (r >= rows) continue;
            const int col = g.gidx[u];
            lrx_c2* dst = STAGE ? stg + t : buf + q * PS + rowb[r] * LD + (u - r * NC);
            lrx_async::cell16(dst, fin + foff[q] + (col >= 0 ? col : 0), col >= 0);
        }
        lrx_async::commit();
    };
    if (static_cast<long long>(blockIdx.x) < groups) issue(blockIdx.x);
    for (long long gi = blockIdx.x; gi < groups; gi += gridDim.x) {
        const long long p0 = gi * PB;
        const int np = g.planes - p0 < PB ? static_cast<int>(g.planes - p0) : PB;
        lrx_async::wait_all();
        __syncthreads();
        // Row FFTs (along c) on the occupied rows only; with STAGE the first
        // factor pass reads the staged rows and writes the planes.
        for (int l = threadIdx.x; l < np * ROWS * C2; l += blockDim.x) {
            const int q = l / (ROWS * C2), u = l - q * ROWS * C2, r = u / C2;
            if (r >= rows) continue;
            pfa_line<NC, C1, C2, true, STAGE>(buf + q * PS + rowb[r] * LD, 1, u % C2, nullptr,
                                              stg + q * SS + r * NC);
        }
        __syncthreads();
        const long long gn = gi + gridDim.x;
        if (STAGE && gn < groups) issue(gn);         // the staging is free: prefetch
        if constexpr (C2 > 1) {
            for (int l = threadIdx.x; l < np * ROWS * C1; l += blockDim.x) {
                const int q = l / (ROWS * C1), u = l - q * ROWS * C1, r = u / C1;
                if (r >= rows) continue;
                pfa_line<NC, C1, C2, false>(buf + q * PS + rowb[r] * LD, 1, u % C1, nullptr);
            }
            __syncthreads();
        }
        // Column FFTs (along b) on every column; a dead row loads as zero.
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
        if (!STAGE && gn < groups) issue(gn);        // the planes are free
    }
}
)__lrx__";

// ---------------------------------------------------------------------------
//  Build and cache
// ---------------------------------------------------------------------------
struct Built { CUfunction fn = nullptr; int rb = 1; int smem = 0; double compile_ms = 0.0; int threads = kThreads;
               long long grid_cap = 0;     // grid_cap: mode 10's resident blocks (0 = none)
               // Mode 11 (k-box stage): arm 0 single pass (tr tile columns, smem), arm 1 split:
               // the plane pass (tr = its tile columns, smem) and the group pencil (threads2, smem2).
               int arm = 0, tr = 0, ty = 0, threads2 = 0, smem2 = 0, sms = 0;
               // Mode 8 with the interaction read from its parents (variant wa*8 + wb), split arm: the
               // W_R chunk pass (phase 4), tiles of trw columns at threads3 / smem3.
               int trw = 0, threads3 = 0, smem3 = 0;
               // Mode 7, split arm: the mid (fn2 at threads2) and pass (fn3 at threads3 / smem3) entries.
               CUfunction fn2 = nullptr, fn3 = nullptr; };
// A split arm's plane tile of whole groups of ss columns.  Mode 8: enough groups for a unit of the
// gathered load per thread (256 per plane) within two blocks per SM.  Modes 7 and 9 (wide): the
// most whole groups the opt-in memory holds, one block per SM; the tile's pairs share their per-k
// unfold tables (20^3 ns 2: 24 columns, the mode-7 gather 3.88 -> 3.65 ms against 16; one column per
// plane wrote 16-byte runs and ran mode 9 no faster than its single arm, KCOLRES).
static int split_plane_tile(int nky, int nkz, long long pr, int ss, long long smem_optin, bool wide = false) {
    if (wide) return static_cast<int>(std::max<long long>(1, smem_optin / (pr * 16) / ss)) * ss;
    int grp = 1;
    while (static_cast<long long>(nky) * nkz * grp < kThreads && 2LL * ss * (2 * grp) * pr * 16 <= smem_optin)
        grp *= 2;
    return ss * grp;
}
// GATE mathdx-kconv-kbox-residency for mode 7's split arm when its plane tile (whole groups of ss
// columns) exceeds the opt-in memory even at one group: "" when it fits (e.g. ns 4 at >= 26^3 on an
// A100).  Mode 9's split tile is checked with the k-box family's other tiles.
static std::string split_tile_refusal(int mode, int nkx, int nky, int nkz, int ns, long long pr, int tile,
                                      long long smem_optin) {
    const long long need = static_cast<long long>(tile) * pr * 16;
    if (need <= smem_optin) return "";
    std::ostringstream os;
    os << "GATE mathdx-kconv-kbox-residency: got k-grid (" << nkx << "," << nky << "," << nkz << ") with ns=" << ns
       << " (mode " << mode << "), whose split plane tile of one spin group needs " << need << " B; want <= "
       << smem_optin << " B of opt-in shared memory on this device; why: the plane pass keeps a (ky, kz) plane "
          "of whole spin groups resident; fix: a smaller k-grid";
    return os.str();
}
// ctx, mode, nkx, nky, nkz, ns, nsr, f32, variant (mode 11: the static completion; mode 7: its output spin
// block; mode 8: wa*8 + wb, the interaction's Lorentz widths when it is read from its parents, 0 = V_R)
using Key = std::tuple<CUcontext, int, int, int, int, int, int, int, int>;
constexpr int kLiveBit = 1 << 16;                  // build(variant | kLiveBit): the door's live program
static std::mutex g_mu;
static std::map<Key, Built> g_cache;
static std::map<Key, std::string> g_fail;

using nvrtc::exists;
using nvrtc::toolkit_include;

// The tile-table plan of modes 7 and 11 (kbox_stage.cuh UnfoldTiles).  A block holds tp pairs'
// bank (bank_pair bytes each) and the tables; the load needs <= 64 registers at 256 threads, so
// shared memory sets the residency, up to four blocks per SM, and never fewer than two (one
// block per SM serialises its phases: U2b, ncu).  Among the tiles of at most tp_max pairs (a
// power of two): most_blocks takes the one that keeps the most blocks resident (mode 11: one
// transform and a reduction per tile), else the largest (mode 7: two transforms and a store per
// tile, whose axis passes idle a block below ~256 lines); tp = 0: none fits.  Measured on the
// 6x6 bispinor harnesses (A100, U2c): mode 11 96.6 ms at 1 pair x 4 blocks vs 100.7 at 2 x 2;
// mode 7 65.6 ms at 2 pairs x 3 blocks vs 80.5 at 1 x 4.
//
// The tables pay only where the register load cannot keep two blocks resident: it holds a
// pair's g, U and Ur live per thread, 3 ns^2 complex values = 12 ns^2 registers, against 128 per
// thread at two blocks of 256.  At ns = 4 that is 192 (172-254 measured, one block per SM; the
// tables won 1.7-1.9x); at ns = 2 it is 48 and the register load already runs three blocks per
// SM, while the tables, O(ns) per pair against an O(ns^2) bank, shrink the tile to two pairs (CrI3
// 8x8 ns 2: Sigma tau 5.58 -> 6.24 s on the tables).  So: the tables when 12 ns^2 > 64.
static bool tile_tables_pay(int ns) { return 12 * ns * ns > 64; }
struct TilePlan { int tp = 0, blocks = 0; long long smem = 0; std::string err; };
static TilePlan tile_table_plan(int dev, int tp_max, long long bank_pair, int nk, int ns, int nw,
                                bool most_blocks) {
    TilePlan p;
    int smem_sm = 0, smem_rsv = 0, optin = 0;
    if (cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&smem_rsv, cudaDevAttrReservedSharedMemoryPerBlock, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
        p.err = "shared memory per SM / reserved per block / opt-in per block";
        return p;
    }
    for (int tp = std::max(1, tp_max); tp >= 1; tp /= 2) {
        const long long blk = tp * bank_pair + lrx_kbox::UnfoldTiles{nk, ns, ns, tp, nw}.bytes();
        const int nb = blk <= optin ? static_cast<int>(std::min<long long>(4, smem_sm / (blk + smem_rsv))) : 0;
        if (nb >= 2 && nb > p.blocks) { p.tp = tp; p.blocks = nb; p.smem = blk; }
        if (!most_blocks && p.tp) break;
    }
    return p;
}

// nsr: the right endpoint width of mode 9 (0 = ns, every other mode).
static ffi::Error build(int mode, int nkx, int nky, int nkz, int ns, bool f32,
                        std::string_view mathdx_root, std::string_view cubin_dir, const Built** out,
                        int nsr = 0, int variant = 0) {
    if (nsr == 0) nsr = ns;
    const DriverApi& api = driver_api();
    if (!api.ok) return fail("driver-api resolve", api.err);
    CUcontext ctx = nullptr;
    CUresult cr = api.CtxGetCurrent(&ctx);
    if (cr != CUDA_SUCCESS || ctx == nullptr) {
        LRX_CUDA_CHECK(cudaFree(nullptr), "context bind (cudaFree(0))");
        cr = api.CtxGetCurrent(&ctx);
        if (cr != CUDA_SUCCESS || ctx == nullptr) return fail("cuCtxGetCurrent", cu_err(cr));
    }
    const Key key{ctx, mode, nkx, nky, nkz, ns, nsr, f32 ? 1 : 0, variant};
    // kLiveBit: a padded pass's program (the door's live operand present), built apart so the plain
    // program keeps no live bounds in registers (kbox_stage.cuh Live, LRX_LIVE).
    const int live_prog = (variant & kLiveBit) ? 1 : 0;
    variant &= ~kLiveBit;
    std::lock_guard<std::mutex> lock(g_mu);
    if (auto it = g_cache.find(key); it != g_cache.end()) { *out = &it->second; return ffi::Error::Success(); }
    if (auto it = g_fail.find(key); it != g_fail.end()) return fail("kernel build (cached failure)", it->second);
    auto sticky = [&](const char* where, const std::string& why,
                      ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
        g_fail.emplace(key, std::string(where) + " -- " + why);
        return fail(where, why, code);
    };

    int dev = 0, cc_major = 0, cc_minor = 0, smem_optin = 0;
    LRX_CUDA_CHECK(cudaGetDevice(&dev), "cudaGetDevice");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev), "cc major");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev), "cc minor");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev),
                   "max opt-in shared memory");
    const int nk = nkx * nky * nkz, sp = nk | 1;
    // Modes 0/1/6 keep three nk-long banks per row (complex128); every other mode sets its own
    // tile below.  The budget never exceeds this device's opt-in maximum (99 KiB on sm_86/89/120,
    // below 100 KiB): an unclamped budget asked for more than the device grants at launch.
    const bool pair = mode < 2 || mode == 6;
    long long row_bytes = 3LL * 16 * sp;
    long long rb = 1;
    if (pair) {
        rb = std::min<long long>(kRowsMax, std::min<long long>(kSmemBudget, smem_optin) / row_bytes);
        if (rb < 1) rb = std::min<long long>(kRowsMax, smem_optin / row_bytes);
    }
    // Mode 8 runs on the k-box stage: the single arm when a tile of whole spin groups (ns^2 padded
    // columns each, plus the pair's wa*wb W columns when W is read from its parents) fits the
    // opt-in memory, else the split arm.
    const int lor_wa = mode == 8 ? variant / 8 : 0, lor_wb = mode == 8 ? variant % 8 : 0;
    const int lor_ws = lor_wa * lor_wb;
    const lrx_kbox::Plan lor_plan = mode == 8
        ? lrx_kbox::kbox_plan(nkx, nky, nkz, ns * ns + lor_ws, 1, 16, smem_optin, 2, 1) : lrx_kbox::Plan{};
    const bool lor_split = mode == 8 && lor_plan.arm == 1;
    // Mode 11 runs on the k-box stage: its launch rule (kbox_plan) decides the arm, the tile and
    // the shared memory from the grid and this device's opt-in budget; RB is unused.
    const int chi_grp = 2 * ns * ns;
    lrx_kbox::Plan kplan{};
    int chi_trc = 0, chi_ty = 0, chi_threads = 0, chi_tt = 0, chi_minb = 1, chi_carve = 100;
    long long chi_smem = 0, chi_smem2 = 0;
    if (mode == 11) {
        kplan = lrx_kbox::kbox_plan(nkx, nky, nkz, ns * ns, 2, 16, smem_optin, 1, 1);  // min_tr 1: a gathered group load
        const lrx_kbox::Geometry g{nkx, nky, nkz};
        rb = 1;
        chi_ty = std::max(1, kThreads / chi_grp);
        if (kplan.arm == 0) {
            chi_trc = kplan.tr * chi_grp;                  // the plan counts whole pairs
            chi_threads = kplan.threads;
            chi_smem = kplan.smem;
            // The tile tables (sm_80+ cp.async; tile_table_plan): bank + UnfoldTiles per block at
            // two or more blocks per SM; none fits: the register load at the plan's tile.
            // Not with vertices (variant bit 1): their Gc sign is applied on the register load only.
            if (cc_major >= 8 && kplan.threads == kThreads && tile_tables_pay(ns) && !(variant & 2)) {
                const TilePlan tt = tile_table_plan(dev, kplan.tr, static_cast<long long>(chi_grp) * g.rs() * 16,
                                                    nk, ns, 0, true);
                if (!tt.err.empty()) return fail("device attributes", tt.err);
                if (tt.tp > 0) {
                    chi_tt = tt.tp;
                    chi_minb = tt.blocks;
                    chi_trc = tt.tp * chi_grp;
                    chi_smem = tt.smem;
                }
            }
        } else {
            // Split arm: plane tiles of kplan.tr (16) columns, whole spin groups; the group pencil
            // reduces by warp shuffles and stages nothing.  Both passes run 16 warps per SM under a
            // 128-register bound when the pencil's x-line (4*nkx registers) leaves room: nkx <= 20,
            // or <= 12 with vertices (ptxas sm_80 at the bound: 20^3 ns 2 and the vertex pencil at
            // 12^3 no spill; 30^3 464 B, the vertex pencil 60 B at 14^3 and 444 B at 20^3; past them
            // one 256-thread block per SM).  Two 256-thread blocks where two tiles fit the SM's
            // shared memory, with the carveout that holds them and the rest left to L1; else one
            // 512-thread block (one 16-column tile per SM: two 8-column tiles ran the plane pass
            // 1.17x slower, and a carveout of 100 cost 1.10x, ncu at 20^3: a tile's pairs share
            // their tables in L1).  At 256 threads and one block the 108 KB tile (20^3) and the
            // 92 KB pencil stage held 8 warps per SM, both passes latency-bound at ~0.5 TB/s.
            int smem_sm = 0, smem_rsv = 0;
            LRX_CUDA_CHECK(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev),
                           "shared memory per SM");
            LRX_CUDA_CHECK(cudaDeviceGetAttribute(&smem_rsv, cudaDevAttrReservedSharedMemoryPerBlock, dev),
                           "reserved shared memory per block");
            const bool wide = nkx <= ((variant & 2) ? 12 : 20);
            chi_trc = kplan.tr;
            chi_smem = static_cast<long long>(chi_trc) * g.pr() * 16;
            chi_smem2 = 0;
            const bool two = wide && 2 * (chi_smem + smem_rsv) <= smem_sm;
            chi_threads = wide && !two ? 2 * kThreads : kThreads;
            chi_ty = chi_threads / chi_grp;
            chi_minb = two ? 2 : 1;
            chi_carve = static_cast<int>(std::min<long long>(
                100, (100LL * chi_minb * (chi_smem + smem_rsv) + smem_sm - 1) / smem_sm));
        }
    }
    // Modes 2-5, 8 and 9 run on the k-box stage: kbox_plan decides the arm, the tile and the shared
    // memory from the grid and this device's opt-in budget (RB is unused).  Modes 2/3 stage
    // k-leading tiles (128-byte runs per k) and have a split arm; modes 4/5 (k-minor) and 9 (the
    // gathered wedge unfold) run the single arm only, with one-column tiles as the floor (min_tr
    // 1), which is what the resident rows they replaced could hold.  Mode 9 takes tiles of whole
    // spin groups when one fits (its grouped load reads a pair's sources once), else single
    // columns.  Mode 8 takes the single arm when a spin group fits and the split arm otherwise: the
    // group pencil forms each member on its own thread (3766 vs 2156 us at 8^3 in the standalone
    // bench for a per-(k, pair) Mid).  The single arm keeps the resident kernel's per-(k, pair)
    // group Mid: a member Mid measured 0.76-0.94x at the CrI3 6x6 and Fe 4^3 ns 4 doors (V staged
    // or not), since the register load fixes one block per SM either way.  Its load holds a spin
    // group per thread (48 complex values at ns 4), so it runs 256 threads, not the plan's 512 (a
    // 128-register cap would spill it).  Kept 2026-09-25 (FP): mode 3 is 90% of the
    // FFT-family time of an Fe 8^3 shared-pole SC map, and the stage is 1.14-1.23x at its door
    // (claim 2779); mode 8's split arm serves 10^3-16^3 at ns 4.
    const bool kbox_rows = (mode >= 2 && mode <= 5) || mode == 9;
    const bool kbox_single_only = mode == 4 || mode == 5;
    int kb_arm = 0, kb_tr = 0, kb_ty = 0, kb_threads = 0, kb_threads2 = 0;
    long long kb_smem = 0, kb_smem2 = 0;
    int lor_trw = 0, kb_threads3 = 0;                  // mode 8 from W parents, split arm
    long long kb_smem3 = 0;
    if (kbox_rows || mode == 8) {
        const lrx_kbox::Geometry g{nkx, nky, nkz};
        rb = 1;
        if (mode == 9) {
            // Whole spin groups per tile: the single arm when a block holds a group of two or more
            // columns (the plan counts groups), else the split arm on plane tiles of whole groups.
            // One column per block (the single arm's floor before) wrote 20^3 W_R at ~85 GB/s, 8
            // warps per SM; split, 64.9 -> 32.0 ms (20^3), 1.48x at 16^3 (KCOLRES).
            const int ss = ns * nsr;
            const lrx_kbox::Plan kp = lrx_kbox::kbox_plan(nkx, nky, nkz, ss, 1, 16, smem_optin, 1, 1);
            kb_arm = kp.arm != 0 || kp.tr * ss < 2 ? 1 : 0;
            kb_tr = kb_arm == 0 ? kp.tr * ss : split_plane_tile(nky, nkz, g.pr(), ss, smem_optin, true);
            // Split: 512 threads (16 warps a block, one block a SM by the tile) where the x-line
            // leaves the 128-register bound room (nkx <= 20), as mode 11's split arm.
            kb_threads = kb_arm == 0 ? kp.threads : (nkx <= 20 ? 2 * kThreads : kThreads);
            kb_smem = kb_arm == 0 ? kp.smem : static_cast<long long>(kb_tr) * g.pr() * 16;
        } else if (kbox_rows) {
            const lrx_kbox::Plan kp = lrx_kbox::kbox_plan(nkx, nky, nkz, 1, 1, f32 ? 8 : 16, smem_optin,
                                                          mode == 2 || mode == 4 ? 2 : 1,
                                                          kbox_single_only ? 1 : 2);
            kb_arm = kp.arm;
            kb_tr = kp.tr;
            kb_threads = kp.arm == 0 ? kp.threads : kThreads;
            kb_smem = kp.smem;                         // single: the tile; split: the plane pass
        } else if (!lor_split) {
            kb_arm = 0;
            kb_tr = lor_plan.tr * ns * ns;             // the plan counts whole spin groups
            kb_threads = kThreads;
            kb_smem = lor_plan.smem;
        } else {
            const int ss = ns * ns;
            kb_arm = 1;
            // Plane tiles of whole spin groups: the gathered load runs one thread per (k, group)
            // of a (ky, kz) plane, so a tile holds enough groups for a unit per thread (256),
            // within two blocks per SM.
            kb_tr = split_plane_tile(nky, nkz, g.pr(), ss, smem_optin);
            // The vertex pencil holds a pair's ss columns on one warp and stages nothing (its block
            // stage of nkx * ty * (ss + 17) elements refused the two-spinor Dirac-quarter doors at
            // 16^3-20^3 and held 2-4 warps per SM where it fit).
            kb_ty = kThreads / ss;                     // pairs per pencil block
            kb_threads = kThreads;
            kb_threads2 = kThreads;
            kb_smem = static_cast<long long>(kb_tr) * g.pr() * 16;
            kb_smem2 = 0;
            if (lor_ws > 0) {                          // the W_R chunk: plane tiles of whole Lorentz groups
                lor_trw = split_plane_tile(nky, nkz, g.pr(), lor_ws, smem_optin);   // as the Green's tiles
                kb_smem3 = static_cast<long long>(lor_trw) * g.pr() * 16;
                kb_threads3 = kThreads;
            }
        }
        if (kb_smem > smem_optin || kb_smem2 > smem_optin || kb_smem3 > smem_optin ||
            (kbox_single_only && kb_arm != 0)) {
            std::ostringstream os;
            os << "GATE mathdx-kconv-kbox-residency: got k-grid (" << nkx << "," << nky << "," << nkz << ") with ns="
               << ns << " (mode " << mode << "), whose k-box " << (kb_arm ? "plane/pencil" : "tile") << " needs "
               << kb_smem << " / " << kb_smem2 << " / " << kb_smem3 << " B; want <= " << smem_optin
               << " B of opt-in shared memory on "
                  "this device" << (kbox_single_only ? " for one whole column (modes 4 and 5 have no split arm)" : "")
               << "; why: the stage keeps a (ky, kz) plane of its tile columns resident; fix: a smaller k-grid";
            return sticky("residency", os.str(), ffi::ErrorCode::kInvalidArgument);
        }
    }
    // Mode 7 runs on the k-box stage's single arm: kbox_plan sizes the tile in whole pairs (one pair
    // = the blk^2 columns of its stored spin block; a convolution: 512 threads at >= 384 lines), or,
    // where one pair does not fit the opt-in memory, in columns (one element per column; the register
    // load of the resident arm it replaces at such grids).  Neither fits: refused by name.  The tile
    // tables (whole 4-spinor groups, sm_80+ cp.async) take the largest tile of at most the plan's
    // pairs whose bank and tables (UnfoldTiles, W_R staged) fit two blocks on an SM; otherwise the
    // register load.  A 4-spinor register load keeps 256 threads and the whole register file (12 ns^2
    // = 192 live registers); an n_s <= 2 register load is built for the blocks the tile's shared memory
    // admits, at most two (Fe 8^3: 2 x 512 threads at <= 64 registers; unbounded, NVCC took 95 and
    // held one block per SM).
    const int blk = (mode == 7 && variant > 0) ? variant : ns;    // mode 7's output spin block
    int m7_tp = 0, m7_blocks = 0, m7_tr = 0, m7_threads = 0, m7_arm = 0, m7_mid = 0, m7_pass = 0;
    long long m7_smem = 0, m7_col_smem = 0;
    if (mode == 7) {
        // Whole spin groups per tile: the single arm when a block holds a group of two or more
        // columns, else a split arm on plane tiles of whole groups: arm 2 when one column's padded
        // box fits a block (the gather plane pass, then the column pass), else arm 1 (four passes,
        // as modes 2/3).  One column per block (the single arm's floor before) re-gathered a pair's
        // group for each column: 20^3 ns 2, 1.3 TB of L2 per P64-tile call at 25% warps; arm 1,
        // 555 -> 237 ms; arm 2 with separate entries, 237 -> 180 ms (KCOLRES).  One group of 4
        // columns per block (12^3 ns 2) stays single: the split ran it 0.84x.
        const lrx_kbox::Geometry g{nkx, nky, nkz};
        const int sso = blk * blk;
        const lrx_kbox::Plan kp = lrx_kbox::kbox_plan(nkx, nky, nkz, sso, 1, 16, smem_optin, 2, 1);
        m7_arm = kp.arm != 0 || kp.tr * sso < 2 ? 1 : 0;
        if (m7_arm != 0) {
            m7_tr = split_plane_tile(nky, nkz, g.pr(), sso, smem_optin, true);
            const std::string why = split_tile_refusal(7, nkx, nky, nkz, ns, g.pr(), m7_tr, smem_optin);
            if (!why.empty()) return sticky("residency", why, ffi::ErrorCode::kInvalidArgument);
            // Entry threads: 512 for the plane passes and the column pass (<= 128 registers, no
            // spill); the x pencil with the Mid holds 212 registers and the ns-4 gather (g, U, U_r:
            // 48 complex) 184-226, so 256.
            m7_threads = ns <= 2 ? 2 * kThreads : kThreads;
            m7_mid = kThreads;
            m7_pass = 2 * kThreads;
            m7_smem = static_cast<long long>(m7_tr) * g.pr() * 16;
            m7_blocks = 1;
            if (g.rs() * 16 <= smem_optin) {           // one column's box fits a block: arm 2
                m7_arm = 2;
                m7_mid = 2 * kThreads;
                m7_col_smem = g.rs() * 16;
            }
        } else {
            m7_tr = kp.tr * sso;                       // the plan counts whole spin groups
            m7_threads = tile_tables_pay(ns) ? kThreads : kp.threads;
            m7_smem = kp.smem;
            if (!tile_tables_pay(ns)) {
                int smem_sm = 0, smem_rsv = 0;
                LRX_CUDA_CHECK(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev),
                               "shared memory per SM");
                LRX_CUDA_CHECK(cudaDeviceGetAttribute(&smem_rsv, cudaDevAttrReservedSharedMemoryPerBlock, dev),
                               "reserved shared memory per block");
                m7_blocks = static_cast<int>(std::max(1LL, std::min(2LL, smem_sm / (m7_smem + smem_rsv))));
            }
            if (blk == ns && nsr == ns && cc_major >= 8 && tile_tables_pay(ns)) {
                const TilePlan tt = tile_table_plan(dev, kp.tr, static_cast<long long>(sso) * g.rs() * 16, nk, ns, 1,
                                                    false);
                if (!tt.err.empty()) return fail("device attributes", tt.err);
                if (tt.tp > 0) {
                    m7_tp = tt.tp;
                    m7_blocks = tt.blocks;
                    m7_smem = tt.smem;
                    m7_tr = tt.tp * sso;
                }
            }
        }
        rb = m7_tr;
    }
    long long plane_minb = 1;                          // mode 10: blocks per SM (LRX_RB)
    long long plane_static = 0;                        // mode 10: its static tables, bytes
    long long plane_stage = 0;                         // mode 10: one plane's staging block, bytes (0 = off)
    // Mode 10 packs (c1, occupied rows) into ns.
    const int plane_c1 = ns & 255, plane_rows = ns >> 8;
    if (mode == 10) {                                  // whole (n_b, n_c|1) planes per block
        row_bytes = 16LL * nkx * (nky | 1);
        // The kernel's static tables live[n_b] + rowb[rows] (int) + foff[PB] (long long)
        // share the block's opt-in budget with the dynamic planes; +16 B alignment
        // slack.  ffi.fft.plane_resident_bytes is the bound for PB = 1, no staging.
        auto stat = [&](long long pb) { return 5LL * nkx + 8 * pb + 16; };
        rb = row_bytes + stat(1) <= smem_optin
            ? std::max(1LL, std::min(8LL, kPlaneGroupBytes / row_bytes)) : 0;
        // The asynchronous gather stages the next group's occupied rows beside the
        // planes when that fits the opt-in budget; otherwise it gathers in place.
        // Kept 2026-09-25 (FP): 1.11-1.37x on mode 10 at 25^2-80^2 (A100), and mode 10 is
        // ~20-40% of the FFT-family time of the CrI3 one-shots.
        const long long stage = 16LL * plane_rows * nky;
        if (rb >= 1 && stage > 0) {
            long long pb = std::max(1LL, std::min(8LL, kPlaneGroupBytes / (row_bytes + stage)));
            if (pb * (row_bytes + stage) + stat(pb) <= smem_optin) { rb = pb; plane_stage = stage; }
        }
        plane_static = stat(std::max(1LL, rb));
        int smem_sm = 0;
        LRX_CUDA_CHECK(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev),
                       "shared memory per SM");
        if (rb < 1) {
            std::ostringstream os;
            os << "GATE mathdx-plane-residency: got plane (" << nkx << "," << nky << ") whose resident "
                  "plane needs 16*n_b*(n_c|1) + static tables = " << row_bytes << " + " << plane_static
               << " B; want <= " << smem_optin << " B of opt-in shared memory on this device; why: the plane "
                  "FFT keeps one plane and its row tables in shared memory; fix: "
                  "ffi.fft.make_plane_fft_gather routes such a plane to the XLA route at plan build";
            return sticky("residency", os.str(), ffi::ErrorCode::kInvalidArgument);
        }
        plane_minb = std::max<long long>(
            1, std::min<long long>(2, smem_sm / (rb * (row_bytes + plane_stage) + plane_static + 1024)));
    }
    if (mode == 11 && (kplan.arm == 1 ? (chi_trc % (ns * ns) != 0 || chi_smem > smem_optin || chi_smem2 > smem_optin)
                                      : chi_smem > smem_optin)) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-chi-residency: got k-grid (" << nkx << "," << nky << "," << nkz << ") with ns="
           << ns << ", whose k-box " << (kplan.arm ? "split" : "single") << " arm needs " << chi_smem << " / "
           << chi_smem2 << " B; want <= " << smem_optin << " B of opt-in shared memory and plane tiles of whole "
              "spin groups; why: mode 11 forms each pair from its 2*ns^2 transformed columns; fix: none here -- "
              "the chi0 route asks ffi.fft.chi_unfold_refusal first and keeps its face kernel for such a grid, "
              "so reaching this gate means that predicate and this rule disagree";
        return sticky("residency", os.str(), ffi::ErrorCode::kInvalidArgument);
    }
    if (rb < 1) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-residency: got k-grid (" << nkx << "," << nky << "," << nkz
           << ") whose resident row needs 3*16*(nk|1)=" << row_bytes << " B"
           << "; want <= " << smem_optin << " B of opt-in shared memory on this device; why: the fused "
              "one-pass family keeps a k-row in shared memory; fix: a smaller k-grid (the "
              "family has no out-of-core arm)";
        return sticky("residency", os.str(), ffi::ErrorCode::kInvalidArgument);
    }
    std::string why;
    const std::string cuda_inc = toolkit_include(&why);
    if (cuda_inc.empty()) return sticky("CUDA toolkit headers for NVRTC", why);
    const std::string root(mathdx_root);
    const std::string inc = root + "/include";
    if (!exists(inc + "/cufftdx.hpp")) {
        return sticky("GATE mathdx-headers",
                      "got no cufftdx.hpp under " + inc + "; want the nvidia-mathdx wheel; fix: "
                      "pip install nvidia-mathdx", ffi::ErrorCode::kFailedPrecondition);
    }
    // Options that decide the cubin (include paths do not: two installs of one
    // wheel version compile the same image).  Line info adds source tables for
    // ncu's source counters and leaves the SASS unchanged.
    std::vector<std::string> defs = {
        "--std=c++17", "--device-as-default-execution-space", "--generate-line-info",
        "--gpu-architecture=sm_" + std::to_string(cc_major) + std::to_string(cc_minor),
        "-DLRX_MODE=" + std::to_string(mode), "-DLRX_NX=" + std::to_string(nkx),
        "-DLRX_NY=" + std::to_string(nky), "-DLRX_NZ=" + std::to_string(nkz),
        "-DLRX_NS=" + std::to_string(mode == 10 ? plane_c1 : ns), "-DLRX_NSR=" + std::to_string(nsr),
        "-DLRX_RB=" + std::to_string(mode == 10 ? plane_minb : rb),
        "-DLRX_F32=" + std::string(f32 ? "1" : "0"),
        "-DLRX_SM=" + std::to_string(cc_major * 100 + cc_minor * 10)};
    if (mode == 7 || mode == 8 || mode == 9 || mode == 11) defs.push_back("-DLRX_LIVE=" + std::to_string(live_prog));
    // Mode 10: planes per block; a block that has its SM alone runs 512 threads (A100 80^2:
    // 3.26 -> 2.60 ms; kept 2026-09-25, FP: the 80^2 production planes).
    const int plane_threads = mode == 10 && plane_minb == 1 ? 512 : kThreads;
    if (mode == 7 && blk != ns) defs.push_back("-DLRX_NA=" + std::to_string(blk));
    if (mode == 7) {
        defs.push_back("-DLRX_ARM=" + std::to_string(m7_arm));
        defs.push_back("-DLRX_TT=" + std::string(m7_tp ? "1" : "0"));
        defs.push_back("-DLRX_TP=" + std::to_string(m7_tp));
        defs.push_back("-DLRX_MINB=" + std::to_string(std::max(1, m7_blocks)));
        defs.push_back("-DLRX_TR=" + std::to_string(m7_tr));
        defs.push_back("-DLRX_THREADS=" + std::to_string(m7_threads));
        if (m7_arm) defs.push_back("-DLRX_THREADS2=" + std::to_string(m7_mid));
        if (m7_arm == 1) defs.push_back("-DLRX_THREADS3=" + std::to_string(m7_pass));
    }
    if (kbox_rows || mode == 8) {
        defs.push_back("-DLRX_ARM=" + std::to_string(kb_arm));
        defs.push_back("-DLRX_TR=" + std::to_string(kb_tr));
        defs.push_back("-DLRX_TY=" + std::to_string(kb_ty));
        defs.push_back("-DLRX_THREADS=" + std::to_string(kb_threads));
        if (mode == 8 && lor_ws > 0) {
            defs.push_back("-DLRX_WA=" + std::to_string(lor_wa));
            defs.push_back("-DLRX_WB=" + std::to_string(lor_wb));
            defs.push_back("-DLRX_TRW=" + std::to_string(lor_trw));
        }
    }
    if (mode == 11) {
        defs.push_back("-DLRX_ARM=" + std::to_string(kplan.arm));
        defs.push_back("-DLRX_TR=" + std::to_string(chi_trc));
        defs.push_back("-DLRX_TY=" + std::to_string(chi_ty));
        defs.push_back("-DLRX_THREADS=" + std::to_string(chi_threads));
        defs.push_back("-DLRX_COMPLETE=" + std::to_string(variant & 1));
        defs.push_back("-DLRX_VTX=" + std::to_string((variant >> 1) & 1));
        defs.push_back("-DLRX_TT=" + std::to_string(chi_tt ? 1 : 0));
        defs.push_back("-DLRX_TP=" + std::to_string(chi_tt));
        defs.push_back("-DLRX_MINB=" + std::to_string(chi_minb));
    }
    if (mode == 10) {
        defs.push_back("-DLRX_PB=" + std::to_string(rb));
        defs.push_back("-DLRX_ROWS=" + std::to_string(std::max(1, plane_rows)));
        defs.push_back("-DLRX_STAGE=" + std::string(plane_stage > 0 ? "1" : "0"));
        defs.push_back("-DLRX_THREADS=" + std::to_string(plane_threads));
    }
    // Disk cache key: source, the embedded async-gather header when the source
    // includes it, deciding options and the toolchain (nvrtc_build.h).
    namespace ag = lorrax_ffi::async_gather;
    nvrtc::Program prog;
    prog.src = mode == 10 ? kPlaneSrc : kSrc;
    prog.name = "lrx_kconv_mathdx.cu";
    if (std::string_view(prog.src).find(ag::kHeaderName) != std::string_view::npos)
        prog.headers = {{ag::kHeaderName, ag::kHeaderSrc}};
    if (mode == 7 || mode == 8 || mode == 11 || kbox_rows)
        prog.headers.push_back({kbox::kHeaderName, kbox::kHeaderSrc});
    prog.defs = defs;
    nvrtc::mathdx_toolchain(root, cuda_inc, "cufftdx", &prog);
    prog.kernel = "lrx_kconv";
    if (mode == 7 && m7_arm == 1) prog.entries = {"lrx_kconv_mid", "lrx_kconv_pass"};
    if (mode == 7 && m7_arm == 2) prog.entries = {"lrx_kconv_col"};
    std::string missing;
    const std::string key_hex = nvrtc::hex16(nvrtc::key(prog, &missing));
    const std::string dir(missing.empty() ? std::string(cubin_dir) : std::string());
    if (!missing.empty() && !std::string(cubin_dir).empty() && (mklpin::announce_here() || log_enabled()))
        std::fprintf(stderr, "[kconv_mathdx] disk cubin cache OFF for this build: empty version header(s) %s "
                     "would drop out of the key\n", missing.c_str());
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kconv_m" << mode << (mode == 8 && variant ? "w" + std::to_string(variant) : std::string())
             << "_" << nkx << "x" << nky << "x" << nkz << "_ns" << ns
             << (nsr != ns ? "x" + std::to_string(nsr) : std::string()) << (f32 ? "_c64" : "") << "_sm" << cc_major << cc_minor << "_" << key_hex << ".cubin";
        path = name.str();
    }
    nvrtc::Image img;
    std::string where, err;
    if (!nvrtc::build(prog, dir, path, key_hex, &img, &where, &err)) return sticky(where.c_str(), err);
    const bool from_disk = img.from_disk, stored = img.stored, rebuilt_bad = img.rebuilt_bad;
    const double ms = img.ms;
    Built b;
    b.fn = img.fn;
    b.rb = static_cast<int>(rb);
    b.threads = plane_threads;
    b.smem = static_cast<int>(rb * (row_bytes + plane_stage));
    if (kbox_rows || mode == 8) {
        int sms = 0;
        LRX_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SM count");
        b.arm = kb_arm;
        b.tr = kb_tr;
        b.rb = kb_tr;                                  // logged as the tile's columns
        b.ty = kb_ty;
        b.threads = kb_threads;
        b.smem = static_cast<int>(kb_smem);
        b.threads2 = kb_threads2;
        b.smem2 = static_cast<int>(kb_smem2);
        b.sms = sms;
        b.trw = lor_trw;
        b.threads3 = kb_threads3;
        b.smem3 = static_cast<int>(kb_smem3);
    }
    if (mode == 11) {
        int sms = 0;
        LRX_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SM count");
        b.arm = kplan.arm;
        b.tr = chi_trc;
        b.rb = chi_trc;                                // logged as the tile's columns
        b.ty = chi_ty;
        b.threads = chi_threads;
        b.smem = static_cast<int>(chi_smem);
        b.threads2 = chi_grp * chi_ty;
        b.smem2 = static_cast<int>(chi_smem2);
        b.sms = sms;
        if (chi_tt) b.grid_cap = static_cast<long long>(sms) * chi_minb;   // a persistent grid
    }
    if (mode == 7) {                                   // tiles of tr columns; the tile tables: a persistent grid
        int sms = 0;
        LRX_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SM count");
        b.tr = m7_tr;
        b.rb = m7_tr;                                  // logged as the tile's columns
        b.threads = m7_threads;
        b.smem = static_cast<int>(m7_smem);
        b.arm = m7_arm;
        b.sms = sms;
        if (m7_arm) {                                  // the split arms' later entries
            b.fn2 = img.fns[0];
            b.threads2 = m7_mid;
        }
        if (m7_arm == 1) {
            b.fn3 = img.fns[1];
            b.threads3 = m7_pass;
            b.smem3 = b.smem;
        }
        if (m7_arm == 2) b.smem2 = static_cast<int>(m7_col_smem);
        if (m7_tp) b.grid_cap = static_cast<long long>(sms) * m7_blocks;
    }
    if (mode == 10) {                                  // persistent blocks: the resident count
        int sms = 0;
        LRX_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SM count");
        b.grid_cap = static_cast<long long>(sms) * plane_minb;
    }
    b.compile_ms = ms;
    // Mode 10 always sets the dynamic limit: its static tables count against the
    // 48 KiB default too, so a plane just under 48 KiB would fail at launch.
    if (b.smem > 49152 || b.smem2 > 49152 || b.smem3 > 49152 || mode == 10) {
        cr = api.FuncSetAttribute(b.fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  std::max(b.smem, std::max(b.smem2, b.smem3)));
        if (cr == CUDA_SUCCESS && b.fn3)
            cr = api.FuncSetAttribute(b.fn3, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, b.smem3);
        if (cr == CUDA_SUCCESS && mode == 7 && b.fn2 && b.smem2 > 0)
            cr = api.FuncSetAttribute(b.fn2, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, b.smem2);
        if (cr != CUDA_SUCCESS && mode == 10) {
            std::ostringstream os;
            os << "GATE mathdx-plane-residency: got plane (" << nkx << "," << nky << "), " << b.smem
               << " B of dynamic shared memory (+ ~" << plane_static << " B static) that this device refused ("
               << cu_err(cr) << "); want <= " << smem_optin << " B in all; fix: "
                  "ffi.fft.make_plane_fft_gather routes such a plane to the XLA route at plan build";
            return sticky("cuFuncSetAttribute", os.str(), ffi::ErrorCode::kInvalidArgument);
        }
        if (cr != CUDA_SUCCESS) return sticky("cuFuncSetAttribute", cu_err(cr));
    }
    // Modes 11 (two blocks per SM) and 7: a shared-memory carveout that holds every planned block,
    // so the driver does not pick a split that holds one block fewer (a hint; residency is
    // unchanged if it declines): the largest for the tile tables and mode 7, the smallest that
    // holds them for mode 11's split arm (chi_carve; the rest is L1).
    if ((mode == 11 && chi_minb > 1) || mode == 7) {
        cr = api.FuncSetAttribute(b.fn, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                  mode == 11 && kplan.arm == 1 ? chi_carve : 100);
        for (CUfunction f : {b.smem2 > 0 ? b.fn2 : nullptr, b.fn3})   // mode 7's shared-memory entries
            if (cr == CUDA_SUCCESS && f && mode == 7) cr = api.FuncSetAttribute(f, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100);
        if (cr != CUDA_SUCCESS) return sticky("cuFuncSetAttribute(carveout)", cu_err(cr));
    }
    if (mklpin::announce_here() || log_enabled()) {
        std::fprintf(stderr, "[kconv_mathdx] %s mode=%d%s kgrid=(%d,%d,%d) ns=%d sm_%d%d in %.1f ms "
                     "(rows/block=%d, smem=%d B, cubin %s)\n",
                     from_disk ? "disk-cache hit" : (rebuilt_bad ? "NVRTC rebuilt (cached image refused)" : "NVRTC built"), mode, f32 ? " c64" : "", nkx, nky, nkz, ns,
                     cc_major, cc_minor, ms, b.rb, b.smem,
                     path.empty() ? "not cached (no cubin_dir)"
                                  : (from_disk ? path.c_str() : (stored ? "stored" : "store FAILED")));
    }
    *out = &(g_cache[key] = b);
    return ffi::Error::Success();
}

static bool pack_attrs(ffi::Span<const int64_t> perm, ffi::Span<const int64_t> phase, int64_t ns,
                       const char* side, unsigned long long* pp, unsigned long long* hp, std::string* why) {
    if (perm.size() != static_cast<size_t>(ns) || phase.size() != static_cast<size_t>(ns)) {
        *why = std::string(side) + " perm/phase lengths != ns"; return false;
    }
    unsigned seen = 0; *pp = 0; *hp = 0;
    for (int64_t i = 0; i < ns; ++i) {
        if (perm[i] < 0 || perm[i] >= ns || phase[i] < 0 || phase[i] > 3 || (seen & (1u << perm[i]))) {
            *why = std::string(side) + " perm must be a permutation of [0,ns) and phase codes in [0,3]";
            return false;
        }
        seen |= 1u << perm[i];
        *pp |= static_cast<unsigned long long>(perm[i]) << (4 * i);
        *hp |= static_cast<unsigned long long>(phase[i]) << (2 * i);
    }
    return true;
}

static ffi::Error Launch(cudaStream_t stream, int mode, ffi::AnyBuffer A, ffi::AnyBuffer B,
                         ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz,
                         double scale, ffi::Span<const int64_t> perm_l, ffi::Span<const int64_t> phase_l,
                         ffi::Span<const int64_t> perm_r, ffi::Span<const int64_t> phase_r,
                         std::string_view mathdx_root, std::string_view cubin_dir,
                         const ParentTables* parent, PlaneTab* plane = nullptr) {
    if (A.element_type() != ffi::DataType::C128 || B.element_type() != ffi::DataType::C128 ||
        U->element_type() != ffi::DataType::C128)
        return fail("contract", "complex128 only", ffi::ErrorCode::kInvalidArgument);
    auto ad = A.dimensions(), bd = B.dimensions(), ud = U->dimensions();
    int64_t ns = 0, rows = 0;
    const int64_t nk = nkx * nky * nkz;
    bool shape_ok = nkx >= 1 && nky >= 1 && nkz >= 1;
    if (plane) {                                      // D (nk,g,ns,2c,ns,p), F (nk,g,p), U (nk,c,g*p)
        shape_ok = shape_ok && ad.size() == 6 && bd.size() == 3 && ud.size() == 3 &&
                   ad[0] == nk && ad[3] % 2 == 0 && ad[4] == ad[2] && bd[0] == nk &&
                   bd[1] == ad[1] && bd[2] == ad[5] && ud[0] == nk && ud[1] == ad[3] / 2 &&
                   ud[2] == ad[1] * ad[5];
        if (shape_ok) {
            ns = ad[2];
            plane->g = ad[1]; plane->c = ad[3] / 2; plane->p = ad[5];
            plane->phase = static_cast<const double*>(B.untyped_data());
            rows = plane->c * plane->g * plane->p;
        }
    } else {
        const size_t rank = parent ? 5 : 7;
        if (ad.size() != rank || bd.size() != rank || ud.size() != (parent ? 3 : 5))
            return fail("contract", "operand ranks", ffi::ErrorCode::kInvalidArgument);
        for (size_t i = 0; i < rank; ++i)
            if (ad[i] != bd[i]) return fail("contract", "A/B shapes differ", ffi::ErrorCode::kInvalidArgument);
        ns = ad[parent ? 1 : 3];
        const int64_t d0 = ad[parent ? 2 : 4], d1 = ad[parent ? 4 : 5];
        rows = d0 * d1;
        shape_ok = shape_ok && (parent
            ? (ad[3] == ns && ud[0] == nk && ud[1] == d0 && ud[2] == d1)
            : (ad[0] == nkx && ad[1] == nky && ad[2] == nkz && ad[6] == ns &&
               ud[0] == nkx && ud[1] == nky && ud[2] == nkz && ud[3] == d0 && ud[4] == d1));
    }
    if (!shape_ok || ns < 1 || ns > 4)
        return fail("contract", "shape/attribute mismatch", ffi::ErrorCode::kInvalidArgument);
    if (nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-axis: got k-grid (" << nkx << "," << nky << "," << nkz
           << "); want every axis <= " << kAxisMax << " (the fp64 cuFFTDx thread-FFT limit)";
        return fail("axis cap", os.str(), ffi::ErrorCode::kInvalidArgument);
    }
    unsigned long long pl = 0, hl = 0, pr = 0, hr = 0; std::string why;
    if (!pack_attrs(perm_l, phase_l, ns, "left", &pl, &hl, &why) ||
        !pack_attrs(perm_r, phase_r, ns, "right", &pr, &hr, &why))
        return fail("vertex attributes", why, ffi::ErrorCode::kInvalidArgument);
    if (rows == 0) return ffi::Error::Success();
    const Built* k = nullptr;
    ffi::Error e = build(mode, static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                         static_cast<int>(ns), false, mathdx_root, cubin_dir, &k);
    if (!e.success()) return e;
    const auto* ap = static_cast<const double*>(A.untyped_data());
    const auto* bp = static_cast<const double*>(B.untyped_data());
    auto* up = static_cast<double*>(U->untyped_data());
    long long rr = rows; double sc = scale;
    ParentTables none{};
    ParentTables tab = parent ? *parent : none;
    PlaneTab flat{};
    PlaneTab pt = plane ? *plane : flat;
    void* args[] = {(void*)&ap, (void*)&bp, (void*)&up, &rr, &sc, &pl, &hl, &pr, &hr, &tab, &pt};
    const long long blocks = (rows + k->rb - 1) / k->rb;
    if (blocks > 2147483647LL) return fail("launch", "grid.x overflow", ffi::ErrorCode::kInvalidArgument);
    CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, kThreads, 1, 1,
                                            static_cast<unsigned>(k->smem),
                                            reinterpret_cast<CUstream>(stream), args, nullptr);
    if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
    return ffi::Error::Success();
}

static ffi::Error PairDispatch(cudaStream_t stream, ffi::AnyBuffer A, ffi::AnyBuffer B,
                               ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz,
                               double scale, ffi::Span<const int64_t> perm_l, ffi::Span<const int64_t> phase_l,
                               ffi::Span<const int64_t> perm_r, ffi::Span<const int64_t> phase_r,
                               std::string_view mathdx_root, std::string_view cubin_dir) {
    return Launch(stream, 0, A, B, U, nkx, nky, nkz, scale, perm_l, phase_l, perm_r, phase_r,
                  mathdx_root, cubin_dir, nullptr);
}

static ffi::Error ParentDispatch(
    cudaStream_t stream, ffi::AnyBuffer A, ffi::AnyBuffer B, ffi::AnyBuffer irr, ffi::AnyBuffer sym,
    ffi::AnyBuffer left, ffi::AnyBuffer right, ffi::AnyBuffer L, ffi::AnyBuffer R, ffi::AnyBuffer q,
    ffi::AnyBuffer trs, ffi::AnyBuffer coef_l, ffi::AnyBuffer coef_r, ffi::Result<ffi::AnyBuffer> U,
    int64_t nkx, int64_t nky, int64_t nkz, double scale, int64_t centroid_major,
    ffi::Span<const int64_t> perm_l, ffi::Span<const int64_t> phase_l,
    ffi::Span<const int64_t> perm_r, ffi::Span<const int64_t> phase_r, std::string_view mathdx_root,
    std::string_view cubin_dir) {
    const auto ad = A.dimensions();
    if (ad.size() != 5) return fail("parent contract", "D must have rank 5", ffi::ErrorCode::kInvalidArgument);
    const int64_t nk = nkx * nky * nkz, ns = ad[1], mu = ad[2], nu = ad[4];
    auto shape = [](ffi::AnyBuffer x, ffi::DataType t, std::vector<int64_t> dims) {
        auto d = x.dimensions();
        return x.element_type() == t && d.size() == dims.size() && std::equal(d.begin(), d.end(), dims.begin());
    };
    auto ld = left.dimensions();
    const int64_t ops = ld.size() == 2 ? ld[0] : 0;
    if ((centroid_major != 0 && centroid_major != 1) || mu < 1 || nu < 1 || ops < 1 ||
        !shape(irr, ffi::DataType::S32, {nk}) || !shape(sym, ffi::DataType::S32, {nk}) ||
        !shape(trs, ffi::DataType::S32, {nk}) || !shape(left, ffi::DataType::S32, {ops, mu}) ||
        !shape(right, ffi::DataType::S32, {ops, nu}) || !shape(L, ffi::DataType::F64, {ops, mu, 3}) ||
        !shape(R, ffi::DataType::F64, {ops, nu, 3}) || !shape(q, ffi::DataType::F64, {ad[0], 3}) ||
        !shape(coef_l, ffi::DataType::C128, {nk, ns * ns, ns * ns}) ||
        !shape(coef_r, ffi::DataType::C128, {nk, ns * ns, ns * ns}))
        return fail("parent tables", "typed table shape/dtype mismatch", ffi::ErrorCode::kInvalidArgument);
    ParentTables t{static_cast<const int*>(irr.untyped_data()), static_cast<const int*>(sym.untyped_data()),
                   static_cast<const int*>(left.untyped_data()), static_cast<const int*>(right.untyped_data()),
                   static_cast<const int*>(trs.untyped_data()), static_cast<const double*>(L.untyped_data()),
                   static_cast<const double*>(R.untyped_data()), static_cast<const double*>(q.untyped_data()),
                   static_cast<const double*>(coef_l.untyped_data()),
                   static_cast<const double*>(coef_r.untyped_data()),
                   static_cast<int>(mu), static_cast<int>(nu), static_cast<int>(centroid_major)};
    return Launch(stream, 1, A, B, U, nkx, nky, nkz, scale, perm_l, phase_l, perm_r, phase_r,
                  mathdx_root, cubin_dir, &t);
}

// Mode 6: D (nk, g, ns, 2c, ns, p) as the route-G D-plane transform left it,
// F (nk, g, p) its Bloch phase; U (nk, c, g*p).
static ffi::Error PlaneDispatch(cudaStream_t stream, ffi::AnyBuffer D, ffi::AnyBuffer F,
                                ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz,
                                double scale, ffi::Span<const int64_t> perm_l,
                                ffi::Span<const int64_t> phase_l, ffi::Span<const int64_t> perm_r,
                                ffi::Span<const int64_t> phase_r, std::string_view mathdx_root,
                                std::string_view cubin_dir) {
    PlaneTab plane{};
    return Launch(stream, 6, D, F, U, nkx, nky, nkz, scale, perm_l, phase_l, perm_r, phase_r,
                  mathdx_root, cubin_dir, nullptr, &plane);
}

// Modes 2-5 (the k-box stage).  `mode` fixes the layout; shapes are checked
// against it here, geometry derived from the operand dimensions.
static ffi::Error LaunchRows(cudaStream_t stream, int mode, ffi::AnyBuffer X, const ffi::AnyBuffer* K,
                             ffi::Result<ffi::AnyBuffer> Y, int64_t nkx, int64_t nky, int64_t nkz,
                             double scale, int64_t forward, int64_t out_layout,
                             std::string_view mathdx_root, std::string_view cubin_dir) {
    const char* what[] = {"", "", "klead conv", "klead fft", "kminor conv", "kminor fft"};
    auto bad = [&](const std::string& why) {
        return fail(what[mode], why, ffi::ErrorCode::kInvalidArgument);
    };
    const ffi::DataType dt = X.element_type();
    if ((dt != ffi::DataType::C128 && dt != ffi::DataType::C64) || Y->element_type() != dt ||
        (K && K->element_type() != dt))
        return bad("operands must all be complex128 or all complex64");
    const bool f32 = dt == ffi::DataType::C64;
    if (nkx < 1 || nky < 1 || nkz < 1) return bad("k-grid axes must be >= 1");
    if (nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-axis: got k-grid (" << nkx << "," << nky << "," << nkz
           << "); want every axis <= " << kAxisMax << " (the fp64 cuFFTDx thread-FFT limit)";
        return bad(os.str());
    }
    const int64_t nk = nkx * nky * nkz;
    auto xd = X.dimensions(), yd = Y->dimensions();
    RowGeo g{};
    g.scale = scale; g.forward = forward ? 1 : 0; g.out_layout = static_cast<int>(out_layout);
    auto same = [](auto a, auto b) { return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin()); };
    if (mode == 2) {                                   // T (nk,a,mx,b,my), V (nk,mx,my)
        auto kd = K->dimensions();
        if (xd.size() != 5 || kd.size() != 3 || !same(xd, yd) || xd[0] != nk || kd[0] != nk ||
            kd[1] != xd[2] || kd[2] != xd[4])
            return bad("want T=U (nk,a,mx,b,my) and V (nk,mx,my) with nk = nkx*nky*nkz");
        g.rows = xd[1] * xd[2] * xd[3] * xd[4];
        g.m0 = xd[4]; g.m1 = xd[3] * xd[4]; g.m2 = xd[2];
    } else if (mode == 4) {                            // X (d0..d4,nk), K (d1,d2,nk)
        auto kd = K->dimensions();
        if (xd.size() != 6 || kd.size() != 3 || xd[5] != nk || kd[0] != xd[1] || kd[1] != xd[2] ||
            kd[2] != nk || (out_layout != 0 && out_layout != 1))
            return bad("want X (d0,d1,d2,d3,d4,nk), K (d1,d2,nk), out_layout 0|1");
        const int64_t want1[6] = {xd[0], nk, xd[3], xd[1], xd[4], xd[2]};
        if (yd.size() != 6 || !(out_layout == 0 ? same(xd, yd) : std::equal(yd.begin(), yd.end(), want1)))
            return bad("output shape does not match out_layout");
        g.rows = xd[0] * xd[1] * xd[2] * xd[3] * xd[4];
        g.m0 = xd[1] * xd[2]; g.m1 = xd[3] * xd[4];
        g.d1 = xd[1]; g.d2 = xd[2]; g.d3 = xd[3]; g.d4 = xd[4];
    } else {                                           // 3: (nk, rows)   5: (rows, nk)
        if (xd.size() != 2 || !same(xd, yd) || xd[mode == 3 ? 0 : 1] != nk)
            return bad(mode == 3 ? "want X=Y (nk, rows)" : "want X=Y (rows, nk)");
        g.rows = xd[mode == 3 ? 1 : 0];
    }
    if (g.rows == 0) return ffi::Error::Success();
    const Built* k = nullptr;
    ffi::Error e = build(mode, static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz), 1,
                         f32, mathdx_root, cubin_dir, &k);
    if (!e.success()) return e;
    const void* xp = X.untyped_data();
    const void* kp = K ? K->untyped_data() : nullptr;
    void* yp = Y->untyped_data();
    // The k-box stage: the single arm, or modes 2/3's split passes.
    auto launch = [&](int phase, long long blocks, int threads, int smem) -> ffi::Error {
        blocks = std::max(1LL, std::min(blocks, 2147483647LL));
        void* args[] = {(void*)&xp, (void*)&kp, (void*)&yp, (void*)&g, (void*)&phase};
        CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, threads, 1, 1,
                                                static_cast<unsigned>(smem),
                                                reinterpret_cast<CUstream>(stream), args, nullptr);
        if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
        return ffi::Error::Success();
    };
    if (k->arm == 0) {                             // one block per tile; mode 2 tiles each a slab
        const long long seg = mode == 2 ? g.m2 * g.m1 : g.rows;
        return launch(0, (g.rows / seg) * ((seg + k->tr - 1) / k->tr), k->threads, k->smem);
    }
    const long long cap = static_cast<long long>(k->sms) * 8;
    const long long plane = nkx * ((g.rows + k->tr - 1) / k->tr);
    const long long pencil = (nky * nkz * g.rows + kThreads - 1) / kThreads;
    const int phases = mode == 2 ? 4 : 2;          // conv: z,y | x.V | z,y | x.s ; fft: z,y | x.s
    for (int ph = 0; ph < phases; ++ph) {
        const bool is_plane = ph % 2 == 0;
        if (auto e = launch(ph, std::min(is_plane ? plane : pencil, cap), kThreads, is_plane ? k->smem : 0);
            !e.success())
            return e;
    }
    return ffi::Error::Success();
}

// Mode 7: the Sigma k-leading convolution read from the raw-parent Green tiles
// through the typed-unfold tables; U (n_out, ns, mx, ns, my) spin-major, full-k
// row k stored at kout[k] (-1 = not stored).  kout == nullptr is the previous
// target's contract (every k at its own row, n_out = nk), kept so an older
// source tree still runs on this library.  bx > 0 stores the x block of rows
// r in [0, xn*bx), left centroid (r / bx)*xs + x0 + r % bx (x >= mx a zero
// padding row), U (n_out, ns, xn*bx, ns, my): the pass reads only those pairs'
// sources, so a caller that bounds the output tile by x blocks reads the Green
// and W once in all (output spin blocks re-read them per block).
static ffi::Error KleadUnfoldImpl(
    cudaStream_t stream, ffi::ScratchAllocator& scratch, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    const ffi::AnyBuffer* kout, ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx,
    int64_t nky, int64_t nkz, double scale, std::string_view mathdx_root, std::string_view cubin_dir,
    int64_t conj_src = 0, int64_t spin_block = 0, int64_t a0 = 0, int64_t b0 = 0, int64_t x0 = 0,
    int64_t bx = 0, int64_t xs = 0, int64_t xn = 0, const ffi::AnyBuffer* live = nullptr) {
    auto bad = [](const std::string& why) {
        return fail("klead unfold conv", why, ffi::ErrorCode::kInvalidArgument);
    };
    if (live && (live->element_type() != ffi::DataType::S32 || live->element_count() != 2))
        return bad("want live s32 [2]: the window rows [lo, hi) of a padded pass");
    if (nkx < 1 || nky < 1 || nkz < 1 || nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-axis: got k-grid (" << nkx << "," << nky << "," << nkz
           << "); want every axis in [1, " << kAxisMax << "] (the fp64 cuFFTDx thread-FFT limit)";
        return bad(os.str());
    }
    const int64_t nk = nkx * nky * nkz;
    auto is = [](ffi::AnyBuffer x, ffi::DataType t, std::vector<int64_t> dims) {
        auto d = x.dimensions();
        return x.element_type() == t && d.size() == dims.size() && std::equal(d.begin(), d.end(), dims.begin());
    };
    const auto gd = Gp.dimensions(), sd = spin.dimensions();
    if (gd.size() != 3 || sd.size() != 3) return bad("want Gp (n_parent, ml, nl) and spin (nk, ns, ns)");
    const int64_t np = gd[0], ml = gd[1], nl = gd[2], ns = sd[1];
    const int64_t nx = (ns > 0 && bx > 0) ? xn * bx : (ns > 0 ? ml / ns : 0);   // stored x rows
    const auto C = ffi::DataType::C128, I = ffi::DataType::S32;
    if ((ns != 1 && ns != 2 && ns != 4) || ml % ns || nl % ns || np < 1 ||
        !is(Gp, C, {np, ml, nl}) || !is(Gt, C, {np, ml, nl}) || !is(row, I, {nk}) || !is(trs, I, {nk}) ||
        !is(lsrc, I, {nk, ml}) || !is(rsrc, I, {nk, nl}) || !is(mph, C, {nk, ml}) || !is(nph, C, {nk, nl}) ||
        !is(spin, C, {nk, ns, ns}) || !is(V, C, {nk, ml / ns, nl / ns}) ||
        (kout != nullptr && !is(*kout, I, {nk})) ||
        !(U->element_type() == C && U->dimensions().size() == 5 && U->dimensions()[0] >= 1 &&
          (kout != nullptr || U->dimensions()[0] == nk) &&
          U->dimensions()[1] == (spin_block ? spin_block : ns) && U->dimensions()[2] == nx &&
          U->dimensions()[3] == (spin_block ? spin_block : ns) && U->dimensions()[4] == nl / ns))
        return bad("want c128 Gp=Gt (np,ml,nl); s32 row,trs,kout (nk), lsrc (nk,ml), rsrc (nk,nl); c128 "
                   "mph (nk,ml), nph (nk,nl), spin (nk,ns,ns), V (nk,ml/ns,nl/ns); U "
                   "(n_out,ns,ml/ns or xn*bx,ns,nl/ns), n_out = nk without kout");
    // The output spin block: rows [a0, a0 + d) x [b0, b0 + d) of the spin group (d = ns: all).
    const int64_t d = spin_block ? spin_block : ns;
    if (d < 1 || ns % d || a0 < 0 || b0 < 0 || a0 % d || b0 % d || a0 >= ns || b0 >= ns)
        return bad("want spin_block d dividing ns and block origins a0, b0 in [0, ns), multiples of d");
    // The x block stores the whole spin group (a spin block and an x block do not combine); its
    // pieces [x0, x0 + bx) at stride xs do not overlap and start inside the tile.
    if (bx < 0 || (bx > 0 && (d != ns || xn < 1 || x0 < 0 || xs < x0 + bx || x0 >= ml / ns)))
        return bad("want the x block 0 <= x0 < ml/ns, bx >= 1, x0 + bx <= xs, xn >= 1, whole spin group");
    const int64_t pairs = nx * (nl / ns);
    if (pairs == 0) return ffi::Error::Success();
    const Built* k = nullptr;
    ffi::Error e = build(7, static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                         static_cast<int>(ns), false, mathdx_root, cubin_dir, &k, 0,
                         (d == ns ? 0 : static_cast<int>(d)) | (live ? kLiveBit : 0));
    if (!e.success()) return e;
    UnfoldTab t{static_cast<const int*>(row.untyped_data()), static_cast<const int*>(trs.untyped_data()),
                static_cast<const int*>(lsrc.untyped_data()), static_cast<const int*>(rsrc.untyped_data()),
                static_cast<const double*>(mph.untyped_data()), static_cast<const double*>(nph.untyped_data()),
                static_cast<const double*>(spin.untyped_data()), ml, nl,
                kout ? static_cast<const int*>(kout->untyped_data()) : nullptr, conj_src ? 2 : 0, nullptr,
                static_cast<int>(a0), static_cast<int>(b0), static_cast<long long>(x0),
                static_cast<long long>(bx), static_cast<long long>(xs), static_cast<long long>(xn)};
    const void* gpp = Gp.untyped_data();
    const void* gtp = Gt.untyped_data();
    const void* vp = V.untyped_data();
    void* up = U->untyped_data();
    double sc = scale;
    void* yb = nullptr;
    long long p0 = 0, npairs = pairs;
    int phase = 0;
    const void* livep = live ? live->untyped_data() : nullptr;
    void* args[] = {(void*)&gpp, (void*)&gtp, (void*)&vp, (void*)&up, (void*)&t, (void*)&sc,
                    (void*)&yb, (void*)&p0, (void*)&npairs, (void*)&phase, (void*)&livep};
    const long long ncols = pairs * d * d;
    auto launch = [&](long long blocks, int threads, int smem, CUfunction fn) -> ffi::Error {
        if (blocks > 2147483647LL) return bad("grid.x overflow");
        CUresult cr = driver_api().LaunchKernel(fn, static_cast<unsigned>(blocks), 1, 1, threads, 1, 1,
                                                static_cast<unsigned>(smem), reinterpret_cast<CUstream>(stream),
                                                args, nullptr);
        return cr == CUDA_SUCCESS ? ffi::Error::Success() : fail("cuLaunchKernel", cu_err(cr));
    };
    if (k->arm == 0) {
        long long blocks = (ncols + k->tr - 1) / k->tr;                 // one block per tile
        if (k->grid_cap > 0) blocks = std::min(blocks, k->grid_cap);   // the tile tables' persistent grid
        return launch(blocks, k->threads, k->smem, k->fn);
    }
    // Split arm: chunks of pairs through a (nk, chunk * d^2) intermediate of at most 1 GiB (the
    // mode-11 and mode-8 bound); a chunk only groups pairs into launches.
    const long long per_pair = nk * d * d * 16, cap = static_cast<long long>(k->sms) * 8;
    const long long chunk = std::max(1LL, std::min<long long>(pairs, (1LL << 30) / per_pair));
    auto ybuf = scratch.Allocate(static_cast<size_t>(chunk * per_pair));
    if (!ybuf.has_value()) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-unfold-scratch: got a refused " << chunk * per_pair << " B intermediate ("
           << chunk << " pairs); want the XLA scratch allocator to grant it";
        return fail("scratch", os.str(), ffi::ErrorCode::kResourceExhausted);
    }
    yb = *ybuf;
    for (p0 = 0; p0 < pairs; p0 += chunk) {
        npairs = std::min(chunk, pairs - p0);
        const long long nc = npairs * d * d;
        const long long plane = std::min(nkx * ((nc + k->tr - 1) / k->tr), cap);
        auto pencil = [&](int threads) { return std::min((nky * nkz * nc + threads - 1) / threads, cap); };
        if (k->arm == 2) {                             // the plane pass, then the resident columns
            if (auto e0 = launch(plane, k->threads, k->smem, k->fn); !e0.success()) return e0;
            phase = 1;
            if (auto e1 = launch(std::min(nc, cap), k->threads2, k->smem2, k->fn2); !e1.success()) return e1;
            continue;
        }
        for (phase = 0; phase < 4; ++phase) {
            const ffi::Error e1 = phase == 0 ? launch(plane, k->threads, k->smem, k->fn)
                                : phase == 1 ? launch(pencil(k->threads2), k->threads2, 0, k->fn2)
                                : phase == 2 ? launch(plane, k->threads3, k->smem3, k->fn3)
                                             : launch(pencil(k->threads3), k->threads3, 0, k->fn3);
            if (!e1.success()) return e1;
        }
    }
    return ffi::Error::Success();
}

// The parent-row targets.  `_rows` keeps its historical signature (every older source tree calls
// it); `_block` adds the conj-on-load partner and the output spin block (U2).
static ffi::Error KleadUnfoldRowsConv(
    cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    ffi::AnyBuffer kout, ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky,
    int64_t nkz, double scale, std::string_view mathdx_root, std::string_view cubin_dir) {
    return KleadUnfoldImpl(stream, scratch, Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, &kout, V, U, nkx, nky,
                           nkz, scale, mathdx_root, cubin_dir);
}
static ffi::Error KleadUnfoldBlockConv(
    cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    ffi::AnyBuffer kout, ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky,
    int64_t nkz, double scale, int64_t conj_src, int64_t spin_block, int64_t a0, int64_t b0,
    std::string_view mathdx_root, std::string_view cubin_dir) {
    return KleadUnfoldImpl(stream, scratch, Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, &kout, V, U, nkx, nky,
                           nkz, scale, mathdx_root, cubin_dir, conj_src, spin_block, a0, b0);
}
// `_xblock`: the conj-on-load partner and the stored x block (whole spin group); live, when
// given (a pass padded to a scan's largest pass): its live window rows [lo, hi), the rest no
// gather or transform and stored as zeros.
static ffi::Error KleadUnfoldXBlockConv(
    cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    ffi::AnyBuffer kout, ffi::AnyBuffer V, std::optional<ffi::AnyBuffer> live, ffi::Result<ffi::AnyBuffer> U,
    int64_t nkx, int64_t nky, int64_t nkz, double scale, int64_t conj_src, int64_t x0, int64_t bx, int64_t xs,
    int64_t xn, std::string_view mathdx_root, std::string_view cubin_dir) {
    return KleadUnfoldImpl(stream, scratch, Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, &kout, V, U, nkx, nky,
                           nkz, scale, mathdx_root, cubin_dir, conj_src, 0, 0, 0, x0, bx, xs, xn,
                           live ? &*live : nullptr);
}
static ffi::Error KleadUnfoldConv(
    cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz, double scale,
    std::string_view mathdx_root, std::string_view cubin_dir) {
    return KleadUnfoldImpl(stream, scratch, Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, nullptr, V, U, nkx, nky,
                           nkz, scale, mathdx_root, cubin_dir);
}

// Mode 8: mode 7's load plus the Lorentz vertex sum; V (nk, mx, nA, my, nB) in R
// space, perm/phase (nA*ns) left and (nB*ns) right, U (n_out, ns, mx, ns, my)
// through mode 7's kout row map (nullptr: the previous target, n_out = nk).
// Mode 8's split-arm arguments, as the embedded source declares them (LorArgs).
struct LorentzArgs {
    const void *gp, *gt, *kern;
    void* u;
    void* y;
    long long p0, npairs, mx, my;
    const void *wp, *wt;
    void* yw;
    double sw;
    const void* live;
};

// The interaction read from its irreducible-q parent tiles (the `_wparent` target): the tile, its
// partner and its unfold tables, as mode 9 takes them (conj_trs 0: the partner tile).
struct WParent {
    ffi::AnyBuffer Wp, Wt, row, trs, lsrc, rsrc, mph, nph, spin_l, spin_r;
    double scale;
};

static ffi::Error KleadLorentzImpl(
    cudaStream_t stream, ffi::ScratchAllocator& scratch, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    const ffi::AnyBuffer* kout, ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx,
    int64_t nky, int64_t nkz, double scale_g, double scale_f, double mult,
    ffi::Span<const int64_t> perm_l, ffi::Span<const int64_t> phase_l, ffi::Span<const int64_t> perm_r,
    ffi::Span<const int64_t> phase_r, std::string_view mathdx_root, std::string_view cubin_dir,
    int64_t conj_src = 0, const WParent* W = nullptr, const ffi::AnyBuffer* live = nullptr) {
    auto bad = [](const std::string& why) {
        return fail("klead lorentz conv", why, ffi::ErrorCode::kInvalidArgument);
    };
    if (live && (live->element_type() != ffi::DataType::S32 || live->element_count() != 2))
        return bad("want live s32 [2]: the window rows [lo, hi) of a padded pass");
    if (nkx < 1 || nky < 1 || nkz < 1 || nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-axis: got k-grid (" << nkx << "," << nky << "," << nkz
           << "); want every axis in [1, " << kAxisMax << "] (the fp64 cuFFTDx thread-FFT limit)";
        return bad(os.str());
    }
    const int64_t nk = nkx * nky * nkz;
    auto is = [](ffi::AnyBuffer x, ffi::DataType t, std::vector<int64_t> dims) {
        auto d = x.dimensions();
        return x.element_type() == t && d.size() == dims.size() && std::equal(d.begin(), d.end(), dims.begin());
    };
    const auto gd = Gp.dimensions(), sd = spin.dimensions();
    if (gd.size() != 3 || sd.size() != 3)
        return bad("want Gp (n_parent, ml, nl) and spin (nk, ns, ns)");
    const int64_t np = gd[0], ml = gd[1], nl = gd[2], ns = sd[1];
    // The interaction: V (nk, mx, nA, my, nB) in R space, or (W != nullptr) its parent tile
    // Wp (nq_irr, mx*nA, my*nB) with spin_l (nk, nA, nA) and spin_r (nk, nB, nB).
    int64_t na = 0, nb = 0, nwq = 0;
    if (W == nullptr) {
        const auto vd = V.dimensions();
        if (vd.size() != 5) return bad("want V (nk, mx, nA, my, nB)");
        na = vd[2];
        nb = vd[4];
    } else {
        const auto wd = W->Wp.dimensions(), ld = W->spin_l.dimensions(), rd = W->spin_r.dimensions();
        if (wd.size() != 3 || ld.size() != 3 || rd.size() != 3)
            return bad("want Wp (nq_irr, mx*nA, my*nB), spin_l (nk, nA, nA), spin_r (nk, nB, nB)");
        na = ld[1];
        nb = rd[1];
        nwq = wd[0];
    }
    const auto C = ffi::DataType::C128, I = ffi::DataType::S32;
    if (W != nullptr) {
        const int64_t wl = (ml / std::max<int64_t>(ns, 1)) * na, wr = (nl / std::max<int64_t>(ns, 1)) * nb;
        if (na < 1 || na > 4 || nb < 1 || nb > 4 || nwq < 1 || !is(W->Wp, C, {nwq, wl, wr}) ||
            !is(W->Wt, C, {nwq, wl, wr}) || !is(W->row, I, {nk}) || !is(W->trs, I, {nk}) ||
            !is(W->lsrc, I, {nk, wl}) || !is(W->rsrc, I, {nk, wr}) || !is(W->mph, C, {nk, wl}) ||
            !is(W->nph, C, {nk, wr}) || !is(W->spin_l, C, {nk, na, na}) || !is(W->spin_r, C, {nk, nb, nb}))
            return bad("want c128 Wp=Wt (nq_irr, mx*nA, my*nB) on the Green's pair grid; s32 wrow, wtrs (nk), "
                       "wlsrc (nk, mx*nA), wrsrc (nk, my*nB); c128 wmph, wnph, spin_l (nk,nA,nA), "
                       "spin_r (nk,nB,nB) with nA, nB in [1,4]");
    }
    if ((ns != 1 && ns != 2 && ns != 4) || ml % ns || nl % ns || np < 1 || na < 1 || na > 4 || nb < 1 ||
        nb > 4 || !is(Gp, C, {np, ml, nl}) || !is(Gt, C, {np, ml, nl}) || !is(row, I, {nk}) ||
        !is(trs, I, {nk}) || !is(lsrc, I, {nk, ml}) || !is(rsrc, I, {nk, nl}) || !is(mph, C, {nk, ml}) ||
        !is(nph, C, {nk, nl}) || !is(spin, C, {nk, ns, ns}) ||
        (W == nullptr && !is(V, C, {nk, ml / ns, na, nl / ns, nb})) ||
        (kout != nullptr && !is(*kout, I, {nk})) ||
        !(U->element_type() == C && U->dimensions().size() == 5 && U->dimensions()[0] >= 1 &&
          (kout != nullptr || U->dimensions()[0] == nk) &&
          U->dimensions()[1] == ns && U->dimensions()[2] == ml / ns && U->dimensions()[3] == ns &&
          U->dimensions()[4] == nl / ns))
        return bad("want c128 Gp=Gt (np,ml,nl); s32 row,trs,kout (nk), lsrc (nk,ml), rsrc (nk,nl); c128 "
                   "mph (nk,ml), nph (nk,nl), spin (nk,ns,ns), V (nk,ml/ns,nA,nl/ns,nB) with nA,nB in "
                   "[1,4]; U (n_out,ns,ml/ns,ns,nl/ns), n_out = nk without kout");
    LorentzTab v{};
    std::string why;
    auto pack = [&](ffi::Span<const int64_t> perm, ffi::Span<const int64_t> phase, int64_t count,
                    const char* side, unsigned long long* pp, unsigned long long* hp) {
        if (perm.size() != static_cast<size_t>(count * ns) || phase.size() != static_cast<size_t>(count * ns)) {
            why = std::string(side) + " perm/phase lengths != (vertices * ns)"; return false;
        }
        for (int64_t i = 0; i < count; ++i) {
            unsigned long long p1 = 0, h1 = 0;
            if (!pack_attrs(ffi::Span<const int64_t>(perm.begin() + i * ns, ns),
                            ffi::Span<const int64_t>(phase.begin() + i * ns, ns), ns, side, &p1, &h1, &why))
                return false;
            *pp |= p1 << (16 * i);
            *hp |= h1 << (8 * i);
        }
        return true;
    };
    if (!pack(perm_l, phase_l, na, "left", &v.perm_l, &v.phase_l) ||
        !pack(perm_r, phase_r, nb, "right", &v.perm_r, &v.phase_r))
        return fail("vertex attributes", why, ffi::ErrorCode::kInvalidArgument);
    v.na = static_cast<int>(na); v.nb = static_cast<int>(nb);
    v.s_g = scale_g; v.s_f = scale_f; v.mult = mult;
    const int64_t pairs = (ml / ns) * (nl / ns);
    if (pairs == 0) return ffi::Error::Success();
    const Built* k = nullptr;
    ffi::Error e = build(8, static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                         static_cast<int>(ns), false, mathdx_root, cubin_dir, &k, 0,
                         (W ? static_cast<int>(na * 8 + nb) : 0) | (live ? kLiveBit : 0));
    if (!e.success()) return e;
    UnfoldTab t{static_cast<const int*>(row.untyped_data()), static_cast<const int*>(trs.untyped_data()),
                static_cast<const int*>(lsrc.untyped_data()), static_cast<const int*>(rsrc.untyped_data()),
                static_cast<const double*>(mph.untyped_data()), static_cast<const double*>(nph.untyped_data()),
                static_cast<const double*>(spin.untyped_data()), ml, nl,
                kout ? static_cast<const int*>(kout->untyped_data()) : nullptr, conj_src ? 2 : 0, nullptr, 0, 0};
    UnfoldTab tw{};
    if (W != nullptr)
        tw = UnfoldTab{static_cast<const int*>(W->row.untyped_data()), static_cast<const int*>(W->trs.untyped_data()),
                       static_cast<const int*>(W->lsrc.untyped_data()), static_cast<const int*>(W->rsrc.untyped_data()),
                       static_cast<const double*>(W->mph.untyped_data()), static_cast<const double*>(W->nph.untyped_data()),
                       static_cast<const double*>(W->spin_l.untyped_data()), (ml / ns) * na, (nl / ns) * nb, nullptr, 0,
                       static_cast<const double*>(W->spin_r.untyped_data()), 0, 0};
    const long long ss = ns * ns, ws = W ? na * nb : 0;
    LorentzArgs a{Gp.untyped_data(), Gt.untyped_data(), W ? nullptr : V.untyped_data(), U->untyped_data(), nullptr,
                  0, pairs, ml / ns, nl / ns, W ? W->Wp.untyped_data() : nullptr,
                  W ? W->Wt.untyped_data() : nullptr, nullptr, W ? W->scale : 0.0,
                  live ? live->untyped_data() : nullptr};
    auto launch = [&](int phase, long long blocks, int threads, int smem) -> ffi::Error {
        blocks = std::max(1LL, std::min(blocks, 2147483647LL));
        void* args[] = {(void*)&a, (void*)&t, (void*)&v, (void*)&phase, (void*)&tw};
        CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, threads, 1, 1,
                                                static_cast<unsigned>(smem),
                                                reinterpret_cast<CUstream>(stream), args, nullptr);
        if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
        return ffi::Error::Success();
    };
    if (k->arm == 0)                                   // the single arm: one block per tile of whole spin groups
        return launch(0, (pairs * ss + k->tr - 1) / k->tr, k->threads, k->smem);
    // The k-box split arm, chunked over pairs through a (nk, chunk * ns^2) intermediate no
    // larger than U itself (the budget: the output this call writes, n_out >= 1 rows of nk).
    // From W parents the chunk's W_R rides beside it: (nk, chunk * nA*nB), the same bound over ns^2.
    // The intermediate is also capped at 1 GiB (the mode-11 tile bound): a chunk only groups pairs
    // into launches, so the values do not depend on it (Fe 20^3 P36 TT: 38 GB of scratch otherwise).
    const long long per_pair = nk * (ss + ws) * 16;
    const long long n_out = U->dimensions()[0];
    const long long chunk = std::max(1LL, std::min<long long>(std::min<long long>(pairs, pairs * n_out / nk),
                                                             (1LL << 30) / per_pair));
    auto y = scratch.Allocate(static_cast<size_t>(chunk * per_pair));
    if (!y.has_value()) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-lorentz-scratch: got a refused " << chunk * per_pair << " B intermediate ("
           << chunk << " pairs); want the XLA scratch allocator to grant it";
        return fail("scratch", os.str(), ffi::ErrorCode::kResourceExhausted);
    }
    a.y = *y;
    if (W != nullptr) a.yw = reinterpret_cast<char*>(*y) + chunk * nk * ss * 16;
    const long long cap = static_cast<long long>(k->sms) * 8;
    for (long long p0 = 0; p0 < pairs; p0 += chunk) {
        a.p0 = p0;
        a.npairs = std::min(chunk, pairs - p0);
        const long long ncols = a.npairs * ss;
        const long long plane = nkx * ((ncols + k->tr - 1) / k->tr);
        const long long group = nky * nkz * ((a.npairs + k->ty - 1) / k->ty);
        const long long pencil = (nky * nkz * ncols + kThreads - 1) / kThreads;
        if (auto e = launch(0, std::min(plane, cap), k->threads, k->smem); !e.success()) return e;
        if (W != nullptr) {
            const long long nw = a.npairs * ws;
            const long long wplane = nkx * ((nw + k->trw - 1) / k->trw);
            const long long wpencil = (nky * nkz * nw + kThreads - 1) / kThreads;
            if (auto e = launch(4, std::min(wplane, cap), k->threads3, k->smem3); !e.success()) return e;
            if (auto e = launch(5, std::min(wpencil, cap), kThreads, 0); !e.success()) return e;
        }
        if (auto e = launch(1, std::min(group, cap), k->threads2, k->smem2); !e.success()) return e;
        if (auto e = launch(2, std::min(plane, cap), k->threads, k->smem); !e.success()) return e;
        if (auto e = launch(3, std::min(pencil, cap), kThreads, 0); !e.success()) return e;
    }
    return ffi::Error::Success();
}

static ffi::Error KleadLorentzRowsConv(
    cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    ffi::AnyBuffer kout, ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky,
    int64_t nkz, double scale_g, double scale_f, double mult, ffi::Span<const int64_t> perm_l,
    ffi::Span<const int64_t> phase_l, ffi::Span<const int64_t> perm_r, ffi::Span<const int64_t> phase_r,
    std::string_view mathdx_root, std::string_view cubin_dir) {
    return KleadLorentzImpl(stream, scratch, Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, &kout, V, U, nkx, nky, nkz,
                            scale_g, scale_f, mult, perm_l, phase_l, perm_r, phase_r, mathdx_root, cubin_dir);
}
static ffi::Error KleadLorentzConjConv(
    cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    ffi::AnyBuffer kout, ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky,
    int64_t nkz, double scale_g, double scale_f, double mult, ffi::Span<const int64_t> perm_l,
    ffi::Span<const int64_t> phase_l, ffi::Span<const int64_t> perm_r, ffi::Span<const int64_t> phase_r,
    int64_t conj_src, std::string_view mathdx_root, std::string_view cubin_dir) {
    return KleadLorentzImpl(stream, scratch, Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, &kout, V, U, nkx, nky, nkz,
                            scale_g, scale_f, mult, perm_l, phase_l, perm_r, phase_r, mathdx_root, cubin_dir,
                            conj_src);
}
// `_wparent`: the interaction read from its irreducible-q parent tiles (W never full-q, W_R never
// full-grid); the V operand of the other targets is absent.
static ffi::Error KleadLorentzWParentConv(
    cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row,
    ffi::AnyBuffer trs, ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph,
    ffi::AnyBuffer spin, ffi::AnyBuffer kout, ffi::AnyBuffer Wp, ffi::AnyBuffer Wt, ffi::AnyBuffer wrow,
    ffi::AnyBuffer wtrs, ffi::AnyBuffer wlsrc, ffi::AnyBuffer wrsrc, ffi::AnyBuffer wmph, ffi::AnyBuffer wnph,
    ffi::AnyBuffer wspin_l, ffi::AnyBuffer wspin_r, std::optional<ffi::AnyBuffer> live,
    ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz, double scale_g, double scale_f,
    double mult, double scale_w, ffi::Span<const int64_t> perm_l, ffi::Span<const int64_t> phase_l,
    ffi::Span<const int64_t> perm_r, ffi::Span<const int64_t> phase_r, int64_t conj_src,
    std::string_view mathdx_root, std::string_view cubin_dir) {
    const WParent w{Wp, Wt, wrow, wtrs, wlsrc, wrsrc, wmph, wnph, wspin_l, wspin_r, scale_w};
    return KleadLorentzImpl(stream, scratch, Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, &kout, Wp, U, nkx, nky,
                            nkz, scale_g, scale_f, mult, perm_l, phase_l, perm_r, phase_r, mathdx_root, cubin_dir,
                            conj_src, &w, live ? &*live : nullptr);
}
static ffi::Error KleadLorentzConv(
    cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz,
    double scale_g, double scale_f, double mult, ffi::Span<const int64_t> perm_l,
    ffi::Span<const int64_t> phase_l, ffi::Span<const int64_t> perm_r, ffi::Span<const int64_t> phase_r,
    std::string_view mathdx_root, std::string_view cubin_dir) {
    return KleadLorentzImpl(stream, scratch, Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, nullptr, V, U, nkx, nky,
                            nkz, scale_g, scale_f, mult, perm_l, phase_l, perm_r, phase_r, mathdx_root,
                            cubin_dir);
}

// Mode 9: an interaction's inverse k-transform read from its wedge tiles
// through the unfold tables; Y (nk, ml, nl) k-leading R space.  Wt is the
// partner tile (pair_transpose) and is not read when conj_trs = 1.
static ffi::Error KleadUnfoldFftImpl(
    cudaStream_t stream, ffi::AnyBuffer Wp, ffi::AnyBuffer Wt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin_l,
    ffi::AnyBuffer spin_r, const ffi::AnyBuffer* live, ffi::Result<ffi::AnyBuffer> Y, int64_t nkx, int64_t nky,
    int64_t nkz, double scale, int64_t conj_trs, std::string_view mathdx_root, std::string_view cubin_dir) {
    auto bad = [](const std::string& why) {
        return fail("klead unfold fft", why, ffi::ErrorCode::kInvalidArgument);
    };
    if (live && (live->element_type() != ffi::DataType::S32 || live->element_count() != 2))
        return bad("want live s32 [2]: the window rows [lo, hi) of a padded pass");
    if (nkx < 1 || nky < 1 || nkz < 1 || nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-axis: got k-grid (" << nkx << "," << nky << "," << nkz
           << "); want every axis in [1, " << kAxisMax << "] (the fp64 cuFFTDx thread-FFT limit)";
        return bad(os.str());
    }
    const int64_t nk = nkx * nky * nkz;
    auto is = [](ffi::AnyBuffer x, ffi::DataType t, std::vector<int64_t> dims) {
        auto d = x.dimensions();
        return x.element_type() == t && d.size() == dims.size() && std::equal(d.begin(), d.end(), dims.begin());
    };
    const auto wd = Wp.dimensions(), ld = spin_l.dimensions(), rd = spin_r.dimensions();
    if (wd.size() != 3 || ld.size() != 3 || rd.size() != 3)
        return bad("want Wp (n_parent, ml, nl), spin_l (nk, nl_s, nl_s) and spin_r (nk, nr_s, nr_s)");
    const int64_t np = wd[0], ml = wd[1], nl = wd[2], nsl = ld[1], nsr = rd[1];
    const auto C = ffi::DataType::C128, I = ffi::DataType::S32;
    if (nsl < 1 || nsl > 4 || nsr < 1 || nsr > 4 || ml % nsl || nl % nsr || np < 1 ||
        (conj_trs != 0 && conj_trs != 1) ||
        !is(Wp, C, {np, ml, nl}) || !is(Wt, C, {np, ml, nl}) || !is(row, I, {nk}) || !is(trs, I, {nk}) ||
        !is(lsrc, I, {nk, ml}) || !is(rsrc, I, {nk, nl}) || !is(mph, C, {nk, ml}) || !is(nph, C, {nk, nl}) ||
        !is(spin_l, C, {nk, nsl, nsl}) || !is(spin_r, C, {nk, nsr, nsr}) || !is(*Y, C, {nk, ml, nl}))
        return bad("want c128 Wp=Wt (np,ml,nl); s32 row,trs (nk), lsrc (nk,ml), rsrc (nk,nl); c128 "
                   "mph (nk,ml), nph (nk,nl), spin_l (nk,nl_s,nl_s), spin_r (nk,nr_s,nr_s) with "
                   "nl_s, nr_s in [1,4]; Y (nk,ml,nl); conj_trs 0|1");
    const int64_t pairs = (ml / nsl) * (nl / nsr);
    if (pairs == 0) return ffi::Error::Success();
    const Built* k = nullptr;
    ffi::Error e = build(9, static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                         static_cast<int>(nsl), false, mathdx_root, cubin_dir, &k, static_cast<int>(nsr),
                         live ? kLiveBit : 0);
    if (!e.success()) return e;
    UnfoldTab t{static_cast<const int*>(row.untyped_data()), static_cast<const int*>(trs.untyped_data()),
                static_cast<const int*>(lsrc.untyped_data()), static_cast<const int*>(rsrc.untyped_data()),
                static_cast<const double*>(mph.untyped_data()), static_cast<const double*>(nph.untyped_data()),
                static_cast<const double*>(spin_l.untyped_data()), ml, nl, nullptr,
                static_cast<int>(conj_trs), static_cast<const double*>(spin_r.untyped_data()), 0, 0};
    const void* wpp = Wp.untyped_data();
    const void* wtp = Wt.untyped_data();
    void* yp = Y->untyped_data();
    double sc = scale;
    int phase = 0;
    const void* livep = live ? live->untyped_data() : nullptr;
    void* args[] = {(void*)&wpp, (void*)&wtp, (void*)&yp, (void*)&t, (void*)&sc, (void*)&phase, (void*)&livep};
    const long long rows = pairs * nsl * nsr, cap = static_cast<long long>(k->sms) * 8;
    auto launch = [&](long long blocks, int threads, int smem) -> ffi::Error {
        CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(std::min(blocks, 2147483647LL)), 1, 1,
                                                threads, 1, 1, static_cast<unsigned>(smem),
                                                reinterpret_cast<CUstream>(stream), args, nullptr);
        return cr == CUDA_SUCCESS ? ffi::Error::Success() : fail("cuLaunchKernel", cu_err(cr));
    };
    if (k->arm == 0) return launch((rows + k->tr - 1) / k->tr, k->threads, k->smem);   // one block per tile
    if (auto e0 = launch(std::min(nkx * ((rows + k->tr - 1) / k->tr), cap), k->threads, k->smem); !e0.success())
        return e0;
    phase = 1;
    return launch(std::min((nky * nkz * rows + k->threads - 1) / k->threads, cap), k->threads, 0);
}
static ffi::Error KleadUnfoldFft(
    cudaStream_t stream, ffi::AnyBuffer Wp, ffi::AnyBuffer Wt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin_l,
    ffi::AnyBuffer spin_r, std::optional<ffi::AnyBuffer> live, ffi::Result<ffi::AnyBuffer> Y, int64_t nkx,
    int64_t nky, int64_t nkz, double scale, int64_t conj_trs, std::string_view mathdx_root,
    std::string_view cubin_dir) {
    return KleadUnfoldFftImpl(stream, Wp, Wt, row, trs, lsrc, rsrc, mph, nph, spin_l, spin_r, live ? &*live : nullptr,
                              Y, nkx, nky, nkz, scale, conj_trs, mathdx_root, cubin_dir);
}

// Mode 11 geometry, as the embedded source declares it (ChiArgs).
struct ChiArgs {
    const void *gv, *gvt, *gc, *gct;
    void* acc;
    const void* alpha;
    void* y;
    long long p0, npairs, pairs, my;
    int n_out;
    double si;
    unsigned long long vperm_l, vphase_l, vperm_r, vphase_r;
    int vna, vnb, vch0;
    const void* sgn_c;
    const void* live;
};

// Mode 11: chi_R accumulation from the raw-parent Green pair.  acc (n_out, nk, mx, my) in place;
// with vertices (na, nb > 0) acc (na*nb*n_out, nk, mx, my), channel-major.
static ffi::Error KleadChiUnfoldImpl(
    cudaStream_t stream, ffi::ScratchAllocator& scratch, ffi::AnyBuffer Gv, ffi::AnyBuffer Gvt,
    ffi::AnyBuffer Gc, ffi::AnyBuffer Gct, ffi::AnyBuffer row, ffi::AnyBuffer trs, ffi::AnyBuffer lsrc,
    ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin, ffi::AnyBuffer alpha,
    ffi::AnyBuffer acc_in, ffi::Result<ffi::AnyBuffer> acc, int64_t nkx, int64_t nky, int64_t nkz, double si,
    int64_t conj_trs, int64_t complete, int64_t scratch_bytes, std::string_view mathdx_root,
    std::string_view cubin_dir, ffi::Span<const int64_t> perm_l, ffi::Span<const int64_t> phase_l,
    ffi::Span<const int64_t> perm_r, ffi::Span<const int64_t> phase_r, int64_t na, int64_t nb,
    const ffi::AnyBuffer* sgn_c = nullptr, const ffi::AnyBuffer* live = nullptr) {
    auto bad = [](const std::string& why) {
        return fail("klead chi unfold", why, ffi::ErrorCode::kInvalidArgument);
    };
    if (live && (live->element_type() != ffi::DataType::S32 || live->element_count() != 2))
        return bad("want live s32 [2]: the window rows [lo, hi) of a padded pass");
    if (nkx < 1 || nky < 1 || nkz < 1 || nkx > kAxisMax || nky > kAxisMax || nkz > kAxisMax) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-axis: got k-grid (" << nkx << "," << nky << "," << nkz
           << "); want every axis in [1, " << kAxisMax << "] (the fp64 cuFFTDx thread-FFT limit)";
        return bad(os.str());
    }
    const int64_t nk = nkx * nky * nkz;
    auto is = [](ffi::AnyBuffer x, ffi::DataType t, std::vector<int64_t> dims) {
        auto d = x.dimensions();
        return x.element_type() == t && d.size() == dims.size() && std::equal(d.begin(), d.end(), dims.begin());
    };
    const auto gd = Gv.dimensions(), sd = spin.dimensions(), ad = acc_in.dimensions();
    if (gd.size() != 3 || sd.size() != 3 || ad.size() != 4)
        return bad("want Gv (n_parent, ml, nl), spin (nk, ns, ns) and acc (n_out, nk, mx, my)");
    // Vertices: na x nb channels, each n_out planes (channel-major); none: one channel.
    const bool vtx = na > 0 || nb > 0;
    if (vtx && (na < 1 || nb < 1 || na > 3 || nb > 3 || complete != 0))
        return bad("want vertex counts na, nb in [1, 3] (or both 0) and no static completion with vertices");
    const int64_t n_ch = vtx ? na * nb : 1;
    if (ad.size() == 4 && ad[0] % n_ch)
        return bad("want acc planes = na * nb * n_out");
    const int64_t np = gd[0], ml = gd[1], nl = gd[2], ns = sd[1], n_out = ad.size() == 4 ? ad[0] / n_ch : 0;
    const auto C = ffi::DataType::C128, I = ffi::DataType::S32;
    if ((ns != 1 && ns != 2 && ns != 4) || ml % ns || nl % ns || np < 1 || n_out < 1 ||
        (conj_trs != 0 && conj_trs != 2) || (complete != 0 && complete != 1) ||
        !is(Gv, C, {np, ml, nl}) || !is(Gvt, C, {np, ml, nl}) || !is(Gc, C, {np, ml, nl}) ||
        !is(Gct, C, {np, ml, nl}) || !is(row, I, {nk}) || !is(trs, I, {nk}) || !is(lsrc, I, {nk, ml}) ||
        !is(rsrc, I, {nk, nl}) || !is(mph, C, {nk, ml}) || !is(nph, C, {nk, nl}) || !is(spin, C, {nk, ns, ns}) ||
        !is(alpha, C, {n_out}) || !is(acc_in, C, {n_ch * n_out, nk, ml / ns, nl / ns}) ||
        !is(*acc, C, {n_ch * n_out, nk, ml / ns, nl / ns}))
        return bad("want c128 Gv=Gvt=Gc=Gct (np,ml,nl); s32 row,trs (nk), lsrc (nk,ml), rsrc (nk,nl); c128 "
                   "mph (nk,ml), nph (nk,nl), spin (nk,ns,ns), alpha (n_out,), acc (n_out,nk,ml/ns,nl/ns); "
                   "conj_trs 0 (partner tiles) | 2 (the partner is conj(G)); complete 0|1");
    if (sgn_c && !is(*sgn_c, ffi::DataType::F64, {nk}))
        return bad("want the Gc sign (nk,) f64");
    unsigned long long vpl = 0, vhl = 0, vpr = 0, vhr = 0;
    if (vtx) {
        std::string why;
        auto pack = [&](ffi::Span<const int64_t> perm, ffi::Span<const int64_t> phase, int64_t count,
                        const char* side, unsigned long long* pp, unsigned long long* hp) {
            if (perm.size() != static_cast<size_t>(count * ns) || phase.size() != static_cast<size_t>(count * ns)) {
                why = std::string(side) + " perm/phase lengths != (vertices * ns)"; return false;
            }
            for (int64_t i = 0; i < count; ++i) {
                unsigned long long p1 = 0, h1 = 0;
                if (!pack_attrs(ffi::Span<const int64_t>(perm.begin() + i * ns, ns),
                                ffi::Span<const int64_t>(phase.begin() + i * ns, ns), ns, side, &p1, &h1, &why))
                    return false;
                *pp |= p1 << (16 * i);
                *hp |= h1 << (8 * i);
            }
            return true;
        };
        if (!pack(perm_l, phase_l, na, "left", &vpl, &vhl) || !pack(perm_r, phase_r, nb, "right", &vpr, &vhr))
            return bad(why);
    }
    const int64_t my = nl / ns, pairs = (ml / ns) * my;
    if (pairs == 0) return ffi::Error::Success();
    const Built* k = nullptr;
    // variant: bit 0 the static completion, bit 1 the four-current channels (LRX_VTX).
    ffi::Error e = build(11, static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                         static_cast<int>(ns), false, mathdx_root, cubin_dir, &k, 0,
                         (static_cast<int>(complete) + (vtx ? 2 : 0)) | (live ? kLiveBit : 0));
    if (!e.success()) return e;
    UnfoldTab t{static_cast<const int*>(row.untyped_data()), static_cast<const int*>(trs.untyped_data()),
                static_cast<const int*>(lsrc.untyped_data()), static_cast<const int*>(rsrc.untyped_data()),
                static_cast<const double*>(mph.untyped_data()), static_cast<const double*>(nph.untyped_data()),
                static_cast<const double*>(spin.untyped_data()), ml, nl, nullptr, static_cast<int>(conj_trs),
                nullptr, 0, 0};
    // acc is aliased to acc_in (input_output_aliases); a copy only if XLA did not alias.
    if (acc->untyped_data() != acc_in.untyped_data())
        LRX_CUDA_CHECK(cudaMemcpyAsync(acc->untyped_data(), acc_in.untyped_data(), acc_in.size_bytes(),
                                       cudaMemcpyDeviceToDevice, stream), "acc copy");
    const long long grp = 2 * ns * ns;
    ChiArgs a{Gv.untyped_data(), Gvt.untyped_data(), Gc.untyped_data(), Gct.untyped_data(),
              acc->untyped_data(), alpha.untyped_data(), nullptr, 0, pairs, pairs, my,
              static_cast<int>(n_out), si, vpl, vhl, vpr, vhr, static_cast<int>(vtx ? na : 0),
              static_cast<int>(vtx ? nb : 0), 0, sgn_c ? sgn_c->untyped_data() : nullptr,
              live ? live->untyped_data() : nullptr};
    auto launch = [&](int phase, long long blocks, int threads, int smem) -> ffi::Error {
        blocks = std::max(1LL, std::min(blocks, 2147483647LL));
        void* args[] = {(void*)&a, (void*)&t, (void*)&phase};
        CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, threads, 1, 1,
                                                static_cast<unsigned>(smem),
                                                reinterpret_cast<CUstream>(stream), args, nullptr);
        if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
        return ffi::Error::Success();
    };
    if (k->arm == 0) {
        const long long tiles = (pairs * grp + k->tr - 1) / k->tr;
        return launch(2, k->grid_cap > 0 ? std::min(tiles, k->grid_cap) : tiles, k->threads, k->smem);
    }
    // Split arm: chunks of pairs through a (nk, chunk * GRP) intermediate of at most scratch_bytes.
    const long long per_pair = nk * grp * 16;
    const long long chunk = std::max(1LL, std::min<long long>(pairs, std::max<int64_t>(scratch_bytes, 0) / per_pair));
    auto y = scratch.Allocate(static_cast<size_t>(chunk * per_pair));
    if (!y.has_value()) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-chi-scratch: got a refused " << chunk * per_pair << " B intermediate ("
           << chunk << " pairs); want the XLA scratch allocator to grant it; fix: a smaller scratch_bytes";
        return fail("scratch", os.str(), ffi::ErrorCode::kResourceExhausted);
    }
    a.y = *y;
    const long long cap = static_cast<long long>(k->sms) * 8;
    for (long long p0 = 0; p0 < pairs; p0 += chunk) {
        a.p0 = p0;
        a.npairs = std::min(chunk, pairs - p0);
        const long long ncols = a.npairs * grp;
        const long long plane_items = nkx * ((ncols + k->tr - 1) / k->tr);
        const long long pencil_items = nky * nkz * ((a.npairs + k->ty - 1) / k->ty);
        if (auto e0 = launch(0, std::min(plane_items, cap), k->threads, k->smem); !e0.success()) return e0;
        // One pencil pass per GRP channels (one pass without vertices): member m carries channel
        // vch0 + m, re-reading the same intermediate.
        for (int64_t c0 = 0; c0 < n_ch; c0 += grp) {
            a.vch0 = static_cast<int>(c0);
            if (auto e1 = launch(1, std::min(pencil_items, cap), k->threads2, k->smem2); !e1.success()) return e1;
        }
    }
    return ffi::Error::Success();
}

static ffi::Error KleadChiUnfold(
    cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer Gv, ffi::AnyBuffer Gvt,
    ffi::AnyBuffer Gc, ffi::AnyBuffer Gct, ffi::AnyBuffer row, ffi::AnyBuffer trs, ffi::AnyBuffer lsrc,
    ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin, ffi::AnyBuffer alpha,
    ffi::AnyBuffer acc_in, std::optional<ffi::AnyBuffer> live, ffi::Result<ffi::AnyBuffer> acc, int64_t nkx,
    int64_t nky, int64_t nkz, double si, int64_t conj_trs, int64_t complete, int64_t scratch_bytes,
    std::string_view mathdx_root, std::string_view cubin_dir) {
    return KleadChiUnfoldImpl(stream, scratch, Gv, Gvt, Gc, Gct, row, trs, lsrc, rsrc, mph, nph, spin, alpha,
                              acc_in, acc, nkx, nky, nkz, si, conj_trs, complete, scratch_bytes, mathdx_root,
                              cubin_dir, {}, {}, {}, {}, 0, 0, nullptr, live ? &*live : nullptr);
}

// Mode 11 with the four-current vertices: na x nb channels of (perm, phase) monomials on the Gc
// operand's spin indices (mode 8's packing), acc (na*nb*n_out, nk, mx, my), channel-major.
static ffi::Error KleadChiVertex(
    cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer Gv, ffi::AnyBuffer Gvt,
    ffi::AnyBuffer Gc, ffi::AnyBuffer Gct, ffi::AnyBuffer row, ffi::AnyBuffer trs, ffi::AnyBuffer lsrc,
    ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin, ffi::AnyBuffer alpha,
    ffi::AnyBuffer sgn_c, ffi::AnyBuffer acc_in, std::optional<ffi::AnyBuffer> live, ffi::Result<ffi::AnyBuffer> acc,
    int64_t nkx, int64_t nky, int64_t nkz, double si, int64_t conj_trs, int64_t scratch_bytes, int64_t signed_c,
    ffi::Span<const int64_t> perm_l, ffi::Span<const int64_t> phase_l,
    ffi::Span<const int64_t> perm_r, ffi::Span<const int64_t> phase_r, int64_t na, int64_t nb,
    std::string_view mathdx_root, std::string_view cubin_dir) {
    if (na < 1 || nb < 1)
        return fail("klead chi vertex", "want na, nb >= 1 (the plain pass is lorrax_mathdx_kconv_chi_unfold)",
                    ffi::ErrorCode::kInvalidArgument);
    return KleadChiUnfoldImpl(stream, scratch, Gv, Gvt, Gc, Gct, row, trs, lsrc, rsrc, mph, nph, spin, alpha,
                              acc_in, acc, nkx, nky, nkz, si, conj_trs, 0, scratch_bytes, mathdx_root,
                              cubin_dir, perm_l, phase_l, perm_r, phase_r, na, nb, signed_c ? &sgn_c : nullptr,
                              live ? &*live : nullptr);
}

static ffi::Error KleadConv(cudaStream_t s, ffi::AnyBuffer T, ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U,
                            int64_t nkx, int64_t nky, int64_t nkz, double scale,
                            std::string_view mathdx_root, std::string_view cubin_dir) {
    return LaunchRows(s, 2, T, &V, U, nkx, nky, nkz, scale, 0, 0, mathdx_root, cubin_dir);
}
static ffi::Error KleadFft(cudaStream_t s, ffi::AnyBuffer X, ffi::Result<ffi::AnyBuffer> Y,
                           int64_t nkx, int64_t nky, int64_t nkz, double scale, int64_t forward,
                           std::string_view mathdx_root, std::string_view cubin_dir) {
    return LaunchRows(s, 3, X, nullptr, Y, nkx, nky, nkz, scale, forward, 0, mathdx_root, cubin_dir);
}
static ffi::Error KminorConv(cudaStream_t s, ffi::AnyBuffer X, ffi::AnyBuffer K, ffi::Result<ffi::AnyBuffer> U,
                             int64_t nkx, int64_t nky, int64_t nkz, double scale, int64_t out_layout,
                             std::string_view mathdx_root, std::string_view cubin_dir) {
    return LaunchRows(s, 4, X, &K, U, nkx, nky, nkz, scale, 0, out_layout, mathdx_root, cubin_dir);
}
static ffi::Error KminorFft(cudaStream_t s, ffi::AnyBuffer X, ffi::Result<ffi::AnyBuffer> Y,
                            int64_t nkx, int64_t nky, int64_t nkz, double scale, int64_t forward,
                            std::string_view mathdx_root, std::string_view cubin_dir) {
    return LaunchRows(s, 5, X, nullptr, Y, nkx, nky, nkz, scale, forward, 0, mathdx_root, cubin_dir);
}

// Mode 10 operand tables (the embedded plane source declares the same struct).
struct PlaneGather {
    const int* gidx;       // (rows, n_c) cylinder column of each cell of an occupied row, -1 = empty
    const int* row_of;     // (rows,) plane row of each occupied row
    const int* start;      // () slab start on F's axis 1 (clamped as lax.dynamic_slice clamps)
    long long rows, n_col, planes, s_len, n_pg, inner;
};

// Mode 10: Y (A, n_pg, *R, n_b, n_c) = the forward unscaled 2-D FFT of the
// planes that the slab F[:, start:start+n_pg] of F (A, S, *R, n_col) fills
// through the static tables (the plain form is n_pg = S, start = 0); b1 | n_b
// and c1 | n_c are the Good-Thomas splits ffi.fft.plane_fft_split chose.
static ffi::Error PlaneFftGather(cudaStream_t stream, ffi::AnyBuffer F, ffi::AnyBuffer gidx,
                                 ffi::AnyBuffer row_of, ffi::AnyBuffer start,
                                 ffi::Result<ffi::AnyBuffer> Y, int64_t nb,
                                 int64_t nc, int64_t b1, int64_t c1, std::string_view mathdx_root,
                                 std::string_view cubin_dir) {
    auto bad = [](const std::string& why) {
        return fail("plane fft gather", why, ffi::ErrorCode::kInvalidArgument);
    };
    auto split_ok = [](int64_t n, int64_t n1) {
        if (n < 2 || n1 < 2 || n % n1 != 0 || n1 > kAxisMax || n / n1 > kAxisMax) return false;
        int64_t a = n1, b = n / n1;
        while (b) { const int64_t t = a % b; a = b; b = t; }
        return a == 1;
    };
    if (!split_ok(nb, b1) || !split_ok(nc, c1)) {
        std::ostringstream os;
        os << "GATE mathdx-plane-split: got plane (" << nb << "," << nc << ") with splits (" << b1 << ","
           << c1 << "); want n = n1*n2 with gcd(n1, n2) = 1 and 2 <= n1, n2 <= " << kAxisMax
           << " (n2 = 1 when n <= " << kAxisMax << "); why: every line FFT is a cuFFTDx thread FFT; "
              "fix: ffi.fft.make_plane_fft_gather routes such a plane to the XLA route at plan build";
        return bad(os.str());
    }
    const auto C = ffi::DataType::C128, I = ffi::DataType::S32;
    auto fd = F.dimensions(), yd = Y->dimensions(), gd = gidx.dimensions(), rd = row_of.dimensions();
    if (F.element_type() != C || Y->element_type() != C || gidx.element_type() != I ||
        row_of.element_type() != I || start.element_type() != I || start.dimensions().size() != 0 ||
        fd.size() < 1 || yd.size() != fd.size() + 1 || yd[yd.size() - 2] != nb ||
        yd[yd.size() - 1] != nc || gd.size() != 2 || gd[1] != nc || rd.size() != 1 || rd[0] != gd[0] ||
        gd[0] > nb)
        return bad("want F (A, S, *R, n_col) c128, Y (A, n_pg, *R, n_b, n_c) c128, gidx (rows, n_c) s32, "
                   "row_of (rows,) s32 with rows <= n_b and start () s32");
    // F's L batch axes read as (A, S, *R): A, S (F) / n_pg (Y) and inner = prod(R).
    const size_t L = fd.size() - 1;
    const long long A = L >= 1 ? fd[0] : 1, S = L >= 2 ? fd[1] : 1, n_pg = L >= 2 ? yd[1] : 1;
    long long inner = 1;
    for (size_t i = 2; i < L; ++i) {
        if (fd[i] != yd[i]) return bad("F and Y trailing batch dimensions differ");
        inner *= fd[i];
    }
    if ((L >= 1 && yd[0] != A) || n_pg > S) return bad("want Y (A, n_pg <= S, *R, n_b, n_c)");
    const long long planes = A * n_pg * inner;
    if (planes == 0) return ffi::Error::Success();
    const Built* k = nullptr;
    // ns carries (c1, occupied rows): the passes' trip counts and the staging are
    // compile-time, so every index divisor is a constant.  A door's table is fixed
    // for its run, so this is one image per run, as keying on the shape alone was.
    const int64_t row_cap = std::max<int64_t>(1, gd[0]);
    ffi::Error e = build(10, static_cast<int>(nb), static_cast<int>(nc), static_cast<int>(b1),
                         static_cast<int>(c1 | (row_cap << 8)), false, mathdx_root, cubin_dir, &k);
    if (!e.success()) return e;
    PlaneGather g{static_cast<const int*>(gidx.untyped_data()), static_cast<const int*>(row_of.untyped_data()),
                  static_cast<const int*>(start.untyped_data()), gd[0], fd[fd.size() - 1], planes, S, n_pg,
                  inner};
    const void* fp = F.untyped_data();
    void* yp = Y->untyped_data();
    void* args[] = {(void*)&fp, (void*)&yp, (void*)&g};
    long long groups = (planes + k->rb - 1) / k->rb;
    if (k->grid_cap > 0) groups = std::min(groups, k->grid_cap);
    const unsigned blocks = static_cast<unsigned>(std::min<long long>(groups, 2147483647LL));
    CUresult cr = driver_api().LaunchKernel(k->fn, blocks, 1, 1, k->threads, 1, 1, static_cast<unsigned>(k->smem),
                                            reinterpret_cast<CUstream>(stream), args, nullptr);
    if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::kconv_mathdx

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxPairCudaFfi, lorrax_ffi::kconv_mathdx::PairDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<xla::ffi::Span<const int64_t>>("perm_l")
        .Attr<xla::ffi::Span<const int64_t>>("phase_l")
        .Attr<xla::ffi::Span<const int64_t>>("perm_r")
        .Attr<xla::ffi::Span<const int64_t>>("phase_r")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxPlaneCudaFfi, lorrax_ffi::kconv_mathdx::PlaneDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<xla::ffi::Span<const int64_t>>("perm_l")
        .Attr<xla::ffi::Span<const int64_t>>("phase_l")
        .Attr<xla::ffi::Span<const int64_t>>("perm_r")
        .Attr<xla::ffi::Span<const int64_t>>("phase_r")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxParentCudaFfi, lorrax_ffi::kconv_mathdx::ParentDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("centroid_major")
        .Attr<xla::ffi::Span<const int64_t>>("perm_l")
        .Attr<xla::ffi::Span<const int64_t>>("phase_l")
        .Attr<xla::ffi::Span<const int64_t>>("perm_r")
        .Attr<xla::ffi::Span<const int64_t>>("phase_r")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

#define LRX_KCONV_GRID_ATTRS                     \
    .Attr<int64_t>("nkx")                         \
    .Attr<int64_t>("nky")                         \
    .Attr<int64_t>("nkz")                         \
    .Attr<double>("scale")

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxKleadCudaFfi, lorrax_ffi::kconv_mathdx::KleadConv,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        LRX_KCONV_GRID_ATTRS
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxKleadUnfoldCudaFfi, lorrax_ffi::kconv_mathdx::KleadUnfoldConv,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()   // Gp
        .Arg<xla::ffi::AnyBuffer>()   // Gt
        .Arg<xla::ffi::AnyBuffer>()   // row
        .Arg<xla::ffi::AnyBuffer>()   // trs
        .Arg<xla::ffi::AnyBuffer>()   // lsrc
        .Arg<xla::ffi::AnyBuffer>()   // rsrc
        .Arg<xla::ffi::AnyBuffer>()   // mph
        .Arg<xla::ffi::AnyBuffer>()   // nph
        .Arg<xla::ffi::AnyBuffer>()   // spin
        .Arg<xla::ffi::AnyBuffer>()   // V (R space)
        .Ret<xla::ffi::AnyBuffer>()
        LRX_KCONV_GRID_ATTRS
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxKleadLorentzCudaFfi, lorrax_ffi::kconv_mathdx::KleadLorentzConv,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()   // Gp
        .Arg<xla::ffi::AnyBuffer>()   // Gt
        .Arg<xla::ffi::AnyBuffer>()   // row
        .Arg<xla::ffi::AnyBuffer>()   // trs
        .Arg<xla::ffi::AnyBuffer>()   // lsrc
        .Arg<xla::ffi::AnyBuffer>()   // rsrc
        .Arg<xla::ffi::AnyBuffer>()   // mph
        .Arg<xla::ffi::AnyBuffer>()   // nph
        .Arg<xla::ffi::AnyBuffer>()   // spin
        .Arg<xla::ffi::AnyBuffer>()   // V (nk, mx, nA, my, nB), R space
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale_g")
        .Attr<double>("scale_f")
        .Attr<double>("mult")
        .Attr<xla::ffi::Span<const int64_t>>("perm_l")
        .Attr<xla::ffi::Span<const int64_t>>("phase_l")
        .Attr<xla::ffi::Span<const int64_t>>("perm_r")
        .Attr<xla::ffi::Span<const int64_t>>("phase_r")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

// The row-map targets: the same kernels with the kout operand (full-k row k
// stored at kout[k], -1 = not stored).  The two targets above are the previous
// contract (every k stored) for older source trees on this library.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxKleadUnfoldRowsCudaFfi, lorrax_ffi::kconv_mathdx::KleadUnfoldRowsConv,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()   // Gp
        .Arg<xla::ffi::AnyBuffer>()   // Gt
        .Arg<xla::ffi::AnyBuffer>()   // row
        .Arg<xla::ffi::AnyBuffer>()   // trs
        .Arg<xla::ffi::AnyBuffer>()   // lsrc
        .Arg<xla::ffi::AnyBuffer>()   // rsrc
        .Arg<xla::ffi::AnyBuffer>()   // mph
        .Arg<xla::ffi::AnyBuffer>()   // nph
        .Arg<xla::ffi::AnyBuffer>()   // spin
        .Arg<xla::ffi::AnyBuffer>()   // kout
        .Arg<xla::ffi::AnyBuffer>()   // V (R space)
        .Ret<xla::ffi::AnyBuffer>()
        LRX_KCONV_GRID_ATTRS
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxKleadUnfoldBlockCudaFfi, lorrax_ffi::kconv_mathdx::KleadUnfoldBlockConv,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()   // Gp
        .Arg<xla::ffi::AnyBuffer>()   // Gt (unread when conj_src)
        .Arg<xla::ffi::AnyBuffer>()   // row
        .Arg<xla::ffi::AnyBuffer>()   // trs
        .Arg<xla::ffi::AnyBuffer>()   // lsrc
        .Arg<xla::ffi::AnyBuffer>()   // rsrc
        .Arg<xla::ffi::AnyBuffer>()   // mph
        .Arg<xla::ffi::AnyBuffer>()   // nph
        .Arg<xla::ffi::AnyBuffer>()   // spin
        .Arg<xla::ffi::AnyBuffer>()   // kout
        .Arg<xla::ffi::AnyBuffer>()   // V (R space)
        .Ret<xla::ffi::AnyBuffer>()
        LRX_KCONV_GRID_ATTRS
        .Attr<int64_t>("conj_src")    // 1: the antiunitary partner is conj(Gp); Gt unread
        .Attr<int64_t>("spin_block")  // d: store the (d x d) output spin block at (a0, b0); 0 = all
        .Attr<int64_t>("a0")
        .Attr<int64_t>("b0")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxKleadUnfoldXBlockCudaFfi, lorrax_ffi::kconv_mathdx::KleadUnfoldXBlockConv,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()   // Gp
        .Arg<xla::ffi::AnyBuffer>()   // Gt (unread when conj_src)
        .Arg<xla::ffi::AnyBuffer>()   // row
        .Arg<xla::ffi::AnyBuffer>()   // trs
        .Arg<xla::ffi::AnyBuffer>()   // lsrc
        .Arg<xla::ffi::AnyBuffer>()   // rsrc
        .Arg<xla::ffi::AnyBuffer>()   // mph
        .Arg<xla::ffi::AnyBuffer>()   // nph
        .Arg<xla::ffi::AnyBuffer>()   // spin
        .Arg<xla::ffi::AnyBuffer>()   // kout
        .Arg<xla::ffi::AnyBuffer>()   // V (R space)
        .OptionalArg<xla::ffi::AnyBuffer>()   // live (s32 [2]): a padded pass's live window rows [lo, hi)
        .Ret<xla::ffi::AnyBuffer>()
        LRX_KCONV_GRID_ATTRS
        .Attr<int64_t>("conj_src")    // 1: the antiunitary partner is conj(Gp); Gt unread
        .Attr<int64_t>("x0")          // the stored x block: rows r in [0, xn*bx) are the left
        .Attr<int64_t>("bx")          //   centroids (r / bx)*xs + x0 + r % bx; bx = 0: every x
        .Attr<int64_t>("xs")
        .Attr<int64_t>("xn")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxKleadLorentzRowsCudaFfi, lorrax_ffi::kconv_mathdx::KleadLorentzRowsConv,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()   // Gp
        .Arg<xla::ffi::AnyBuffer>()   // Gt
        .Arg<xla::ffi::AnyBuffer>()   // row
        .Arg<xla::ffi::AnyBuffer>()   // trs
        .Arg<xla::ffi::AnyBuffer>()   // lsrc
        .Arg<xla::ffi::AnyBuffer>()   // rsrc
        .Arg<xla::ffi::AnyBuffer>()   // mph
        .Arg<xla::ffi::AnyBuffer>()   // nph
        .Arg<xla::ffi::AnyBuffer>()   // spin
        .Arg<xla::ffi::AnyBuffer>()   // kout
        .Arg<xla::ffi::AnyBuffer>()   // V (nk, mx, nA, my, nB), R space
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale_g")
        .Attr<double>("scale_f")
        .Attr<double>("mult")
        .Attr<xla::ffi::Span<const int64_t>>("perm_l")
        .Attr<xla::ffi::Span<const int64_t>>("phase_l")
        .Attr<xla::ffi::Span<const int64_t>>("perm_r")
        .Attr<xla::ffi::Span<const int64_t>>("phase_r")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxKleadLorentzConjCudaFfi, lorrax_ffi::kconv_mathdx::KleadLorentzConjConv,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()   // Gp
        .Arg<xla::ffi::AnyBuffer>()   // Gt
        .Arg<xla::ffi::AnyBuffer>()   // row
        .Arg<xla::ffi::AnyBuffer>()   // trs
        .Arg<xla::ffi::AnyBuffer>()   // lsrc
        .Arg<xla::ffi::AnyBuffer>()   // rsrc
        .Arg<xla::ffi::AnyBuffer>()   // mph
        .Arg<xla::ffi::AnyBuffer>()   // nph
        .Arg<xla::ffi::AnyBuffer>()   // spin
        .Arg<xla::ffi::AnyBuffer>()   // kout
        .Arg<xla::ffi::AnyBuffer>()   // V (nk, mx, nA, my, nB), R space
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale_g")
        .Attr<double>("scale_f")
        .Attr<double>("mult")
        .Attr<xla::ffi::Span<const int64_t>>("perm_l")
        .Attr<xla::ffi::Span<const int64_t>>("phase_l")
        .Attr<xla::ffi::Span<const int64_t>>("perm_r")
        .Attr<xla::ffi::Span<const int64_t>>("phase_r")
        .Attr<int64_t>("conj_src")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxKleadLorentzWParentCudaFfi, lorrax_ffi::kconv_mathdx::KleadLorentzWParentConv,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()   // Gp
        .Arg<xla::ffi::AnyBuffer>()   // Gt
        .Arg<xla::ffi::AnyBuffer>()   // row
        .Arg<xla::ffi::AnyBuffer>()   // trs
        .Arg<xla::ffi::AnyBuffer>()   // lsrc
        .Arg<xla::ffi::AnyBuffer>()   // rsrc
        .Arg<xla::ffi::AnyBuffer>()   // mph
        .Arg<xla::ffi::AnyBuffer>()   // nph
        .Arg<xla::ffi::AnyBuffer>()   // spin
        .Arg<xla::ffi::AnyBuffer>()   // kout
        .Arg<xla::ffi::AnyBuffer>()   // Wp (nq_irr, mx*nA, my*nB): the interaction's parent tile
        .Arg<xla::ffi::AnyBuffer>()   // Wt (its partner tile, pair_transpose)
        .Arg<xla::ffi::AnyBuffer>()   // wrow
        .Arg<xla::ffi::AnyBuffer>()   // wtrs
        .Arg<xla::ffi::AnyBuffer>()   // wlsrc
        .Arg<xla::ffi::AnyBuffer>()   // wrsrc
        .Arg<xla::ffi::AnyBuffer>()   // wmph
        .Arg<xla::ffi::AnyBuffer>()   // wnph
        .Arg<xla::ffi::AnyBuffer>()   // wspin_l (nk, nA, nA)
        .Arg<xla::ffi::AnyBuffer>()   // wspin_r (nk, nB, nB)
        .OptionalArg<xla::ffi::AnyBuffer>()   // live (s32 [2]): a padded pass's live window rows [lo, hi)
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale_g")
        .Attr<double>("scale_f")
        .Attr<double>("mult")
        .Attr<double>("scale_w")
        .Attr<xla::ffi::Span<const int64_t>>("perm_l")
        .Attr<xla::ffi::Span<const int64_t>>("phase_l")
        .Attr<xla::ffi::Span<const int64_t>>("perm_r")
        .Attr<xla::ffi::Span<const int64_t>>("phase_r")
        .Attr<int64_t>("conj_src")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KFftMathdxKleadUnfoldCudaFfi, lorrax_ffi::kconv_mathdx::KleadUnfoldFft,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()   // Wp (wedge tiles)
        .Arg<xla::ffi::AnyBuffer>()   // Wt (partner tiles; unread when conj_trs)
        .Arg<xla::ffi::AnyBuffer>()   // row
        .Arg<xla::ffi::AnyBuffer>()   // trs
        .Arg<xla::ffi::AnyBuffer>()   // lsrc
        .Arg<xla::ffi::AnyBuffer>()   // rsrc
        .Arg<xla::ffi::AnyBuffer>()   // mph
        .Arg<xla::ffi::AnyBuffer>()   // nph
        .Arg<xla::ffi::AnyBuffer>()   // spin_l
        .Arg<xla::ffi::AnyBuffer>()   // spin_r
        .OptionalArg<xla::ffi::AnyBuffer>()   // live (s32 [2]): a padded pass's live window rows [lo, hi)
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("conj_trs")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxChiUnfoldCudaFfi, lorrax_ffi::kconv_mathdx::KleadChiUnfold,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()   // Gv (raw-parent valence Green)
        .Arg<xla::ffi::AnyBuffer>()   // Gvt (its partner tiles; unread when conj_trs = 2)
        .Arg<xla::ffi::AnyBuffer>()   // Gc (raw-parent conduction Green)
        .Arg<xla::ffi::AnyBuffer>()   // Gct
        .Arg<xla::ffi::AnyBuffer>()   // row
        .Arg<xla::ffi::AnyBuffer>()   // trs
        .Arg<xla::ffi::AnyBuffer>()   // lsrc
        .Arg<xla::ffi::AnyBuffer>()   // rsrc
        .Arg<xla::ffi::AnyBuffer>()   // mph
        .Arg<xla::ffi::AnyBuffer>()   // nph
        .Arg<xla::ffi::AnyBuffer>()   // spin
        .Arg<xla::ffi::AnyBuffer>()   // alpha (n_out,)
        .Arg<xla::ffi::AnyBuffer>()   // acc (n_out, nk, mx, my), aliased to the result
        .OptionalArg<xla::ffi::AnyBuffer>()   // live (s32 [2]): a padded pass's live window rows [lo, hi)
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("si")
        .Attr<int64_t>("conj_trs")
        .Attr<int64_t>("complete")
        .Attr<int64_t>("scratch_bytes")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxChiVertexCudaFfi, lorrax_ffi::kconv_mathdx::KleadChiVertex,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()   // Gv (raw-parent lower Green quadrant)
        .Arg<xla::ffi::AnyBuffer>()   // Gvt
        .Arg<xla::ffi::AnyBuffer>()   // Gc (raw-parent upper Green quadrant; the vertices act on it)
        .Arg<xla::ffi::AnyBuffer>()   // Gct
        .Arg<xla::ffi::AnyBuffer>()   // row
        .Arg<xla::ffi::AnyBuffer>()   // trs
        .Arg<xla::ffi::AnyBuffer>()   // lsrc
        .Arg<xla::ffi::AnyBuffer>()   // rsrc
        .Arg<xla::ffi::AnyBuffer>()   // mph
        .Arg<xla::ffi::AnyBuffer>()   // nph
        .Arg<xla::ffi::AnyBuffer>()   // spin
        .Arg<xla::ffi::AnyBuffer>()   // alpha (n_out,)
        .Arg<xla::ffi::AnyBuffer>()   // sgn_c (nk,) f64: the Gc operand's per-k sign (read when signed_c)
        .Arg<xla::ffi::AnyBuffer>()   // acc (na*nb*n_out, nk, mx, my), aliased to the result
        .OptionalArg<xla::ffi::AnyBuffer>()   // live (s32 [2]): a padded pass's live window rows [lo, hi)
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("si")
        .Attr<int64_t>("conj_trs")
        .Attr<int64_t>("scratch_bytes")
        .Attr<int64_t>("signed_c")
        .Attr<xla::ffi::Span<const int64_t>>("perm_l")
        .Attr<xla::ffi::Span<const int64_t>>("phase_l")
        .Attr<xla::ffi::Span<const int64_t>>("perm_r")
        .Attr<xla::ffi::Span<const int64_t>>("phase_r")
        .Attr<int64_t>("na")
        .Attr<int64_t>("nb")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KFftMathdxKleadCudaFfi, lorrax_ffi::kconv_mathdx::KleadFft,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        LRX_KCONV_GRID_ATTRS
        .Attr<int64_t>("forward")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KConvMathdxKminorCudaFfi, lorrax_ffi::kconv_mathdx::KminorConv,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        LRX_KCONV_GRID_ATTRS
        .Attr<int64_t>("out_layout")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    KFftMathdxKminorCudaFfi, lorrax_ffi::kconv_mathdx::KminorFft,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        LRX_KCONV_GRID_ATTRS
        .Attr<int64_t>("forward")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PlaneFftGatherMathdxCudaFfi, lorrax_ffi::kconv_mathdx::PlaneFftGather,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nb")
        .Attr<int64_t>("nc")
        .Attr<int64_t>("b1")
        .Attr<int64_t>("c1")
        .Attr<std::string_view>("mathdx_root")
        .Attr<std::string_view>("cubin_dir"));
