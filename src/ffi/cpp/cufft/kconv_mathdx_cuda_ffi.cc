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
//   6 plane   the pair contraction of mode 0 read straight from the route-G
//             D-plane transform output D (nk, g, ns, 2c, ns, p): the load
//             applies the Bloch phase F[k,g,p] and takes L = slots [0,c) and
//             R = slots [c,2c) of the 2c axis, so no transposed, phased or
//             split copy is made; U is (nk, c, g*p).  The element product
//             D*F is the one XLA formed before this mode existed (see
//             lrx_mul_xla), so mode 6 is meant to equal the old moveaxis +
//             mode 1 chain bit for bit.
//   7 klead unfold conv   mode 2 read from the RAW-PARENT Green tiles: per
//             full k the load gathers G[row(k), lsrc(k,i), rsrc(k,j)] (the
//             transposed-pair tile on an antiunitary row), applies the
//             umklapp phases mph(k,i), nph(k,j) and the ns x ns spin action
//             U_k in registers, and the store writes U spin-major
//             (n_out, a, mx, b, my), full-k row k at output row kout(k)
//             (-1: not stored; the Sigma consumers keep only the parent
//             rows), or one (d x d) output spin block (a0, b0) of it
//             (LRX_NA = d): every source is read, only the block stored.  The tables are symmetry_maps's (unfold_load_tables);
//             every product rounds as the XLA unfold and the spin-rotate FFI
//             it replaces.
//   8 klead lorentz conv   mode 7's load, then the four-current vertex sum in
//             R space: U = mult * sf * FFT_k( sum_{A,B} gamma_A (si * IFFT_k
//             G_unfolded) gamma_B^dagger * V[k,x,A,y,B] ) with V (nk, mx, nA,
//             my, nB) ALREADY in R space (mode 3 made it, scale si), the
//             signed-permutation vertices gamma_A (left) and gamma_B (right)
//             as attributes, and U spin-major (n_out, a, mx, b, my) through
//             mode 7's kout row map.  One
//             transform of G serves every Lorentz block; the scales and the
//             product/sum order are those of the XLA chain it replaces (mode-3
//             transforms and a scan over the blocks), so it is meant to equal
//             that chain bit for bit.  On the k-box stage's split arm: the
//             group pencil forms each (member, pair) on its own thread with V
//             staged by cp.async; chunked over pairs through an intermediate no
//             larger than U.
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
//             interaction: no partner tile).  The products round as the XLA
//             unfold (symmetry_maps unfold_isdf_operator) and mode 3 they
//             replace.
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
//             else the plane pass and an R-space group pencil, chunked over pairs.
// A new mode adds (1) an entry under its LRX_MODE value in kSrc, (2) a mode
// code and a handler below, (3) a router factory in ffi/fft.py.
//
// Residency: modes 0/1/6 keep three nk-long banks per row (one (col,mu) pair) in
// shared memory, modes 4/5 one; a k-grid whose row does not fit the device's
// opt-in shared memory, or an axis above the fp64 thread-FFT limit (40), is
// refused by name.  The k-box modes (2, 3, 8, 11) need one (ky, kz) plane of a
// tile in shared memory.  Modes 2/3/5 may run in place: every block reads the
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
// Modes 2-5 keep ONE bank per row; ~48 KiB per block lets three blocks share
// an A100 SM.  ponytail: rows-per-block is a fixed heuristic, not tuned per grid.
static constexpr int kRowsMax1 = 64;
static constexpr long long kSmemBudget1 = 48 * 1024;
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
// Products that must round exactly as the chains modes 6 and 7 replace.  Each
// is spelled with round-to-nearest intrinsics or explicit fma, so NVRTC's own
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
// cuCmul(a, b) as nvcc compiles it in cpp/symmetry/spin_rotate.cu (SASS):
// re = fma(a.x, b.x, -(a.y b.y)), im = fma(a.x, b.y, a.y b.x).
__device__ __forceinline__ lrx_c2 lrx_mul_cu(lrx_c2 a, lrx_c2 b) {
    lrx_c2 z = {fma(a.x, b.x, -__dmul_rn(a.y, b.y)), fma(a.x, b.y, __dmul_rn(a.y, b.x))};
    return z;
}
// cuCmul(a, cuConj(u)) as nvcc compiles it there: the negation folds, so
// re = fma(a.x, u.x, a.y u.y) and im = fma(a.y, u.x, -(a.x u.y)).
__device__ __forceinline__ lrx_c2 lrx_mul_cu_conj(lrx_c2 a, lrx_c2 u) {
    lrx_c2 z = {fma(a.x, u.x, __dmul_rn(a.y, u.y)), fma(a.y, u.x, -__dmul_rn(a.x, u.y))};
    return z;
}
// The spin action U G U^dagger of the chain mode 7 replaces: the
// spin-rotate FFI (nvcc) for ns = 2, 4; for ns = 1 the unfold rotates in XLA
// (symmetry_maps._rotate_open_spin_centroid_operator), which does not fuse.
// Mode 9's endpoint actions replace an XLA rotation, so they round as XLA.
__device__ __forceinline__ lrx_c2 lrx_rot_mul(lrx_c2 u, lrx_c2 g) {
#if LRX_NS == 1 || LRX_MODE == 9
    return lrx_mul_xla(u, g);
#else
    return lrx_mul_cu(u, g);
#endif
}
__device__ __forceinline__ lrx_c2 lrx_rot_mul_conj(lrx_c2 l, lrx_c2 u) {
#if LRX_NS == 1 || LRX_MODE == 9
    const lrx_c2 uc = {u.x, -u.y};
    return lrx_mul_xla(l, uc);
#else
    return lrx_mul_cu_conj(l, u);
#endif
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
// whole library line-FFTs (flat k is C-order, kz fastest).
template <int N, int STRIDE, cufftdx::fft_direction Dir>
__device__ __forceinline__ void axis_pass(lrx_c2* bank) {
    if constexpr (N > 1) {
        using F = TFFT<N, Dir>;
        using V = typename F::value_type;
        constexpr int lines = NK / N;
        for (int l = threadIdx.x; l < RB * lines; l += blockDim.x) {
            const int j = l / lines, li = l % lines;
            lrx_c2* p = bank + j * SP + (li / STRIDE) * N * STRIDE + (li % STRIDE);
            V v[F::storage_size];
#pragma unroll
            for (int e = 0; e < N; ++e) { v[e].x = p[e * STRIDE].x; v[e].y = p[e * STRIDE].y; }
            F().execute(v);
#pragma unroll
            for (int e = 0; e < N; ++e) { p[e * STRIDE].x = v[e].x; p[e * STRIDE].y = v[e].y; }
        }
    }
    __syncthreads();
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
    for (long long c0 = (long long)blockIdx.x * TRC; c0 < g.rows; c0 += (long long)gridDim.x * TRC) {
        lrx_kbox::stage_tile<NX, NY, NZ, TRC>(sm, c0, g.rows, ld);
        if (inv) lrx_tile3<fft_direction::inverse>(sm);
        else lrx_tile3<fft_direction::forward>(sm);
        if constexpr (CONV) {
            lrx_kbox::mid_tile<NX, NY, NZ, TRC>(sm, c0, g.rows, KernMid{kern, g});
            lrx_tile3<fft_direction::forward>(sm);
        }
        lrx_kbox::store_tile<NX, NY, NZ, TRC>(sm, c0, g.rows, RowStore{y, g.rows, g.scale});
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
// Modes 4-5: one resident bank per row.  k-MINOR tiles hold (r, k) at r*NK + k
// and walk k fastest.  x and y may alias (in place): every element a block
// stores is one it loaded.
constexpr bool CONV = (LRX_MODE == 4);

__device__ __forceinline__ long long lrx_elem(long long row, int k, long long rows) {
    return row * NK + k;
}

extern "C" __global__ void __launch_bounds__(256) lrx_kconv(
    const lrx_c2* x, const lrx_c2* __restrict__ kern, lrx_c2* y, RowGeo g) {
    extern __shared__ lrx_c2 sm[];
    const long long r0 = (long long)blockIdx.x * RB;
    using namespace cufftdx;
    for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
        const int k = i % NK, j = i / NK;
        const long long row = r0 + j;
        lrx_c2 v = {0.0, 0.0};
        if (row < g.rows) v = x[lrx_elem(row, k, g.rows)];
        sm[j * SP + k] = v;
    }
    __syncthreads();
    if (CONV || !g.forward) transform3<fft_direction::inverse>(sm);
    else transform3<fft_direction::forward>(sm);
    if (CONV) {
        for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
            const int k = i % NK, j = i / NK;
            const long long row = r0 + j;
            if (row < g.rows) {
                const long long kidx = ((row / g.m1) % g.m0) * NK + k;
                sm[j * SP + k] = lrx_mul(sm[j * SP + k], kern[kidx]);
            }
        }
        __syncthreads();
        transform3<fft_direction::forward>(sm);
    }
    for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
        const int k = i % NK, j = i / NK;
        const long long row = r0 + j;
        if (row < g.rows) {
            const lrx_c2 v = sm[j * SP + k];
            lrx_c2 w;
            w.x = (lrx_real)(v.x * g.scale);
            w.y = (lrx_real)(v.y * g.scale);
            long long o = lrx_elem(row, k, g.rows);
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
    }
}
#elif LRX_MODE <= 9 || LRX_MODE == 11
// Modes 7, 8, 9 and 11 share the unfold load below.
// Mode 7: mode 2 on the full-k Green read from its raw parents.  Row r of the
// convolution is (pair, a, b) = (r / NS^2, (r % NS^2) / NS, r % NS) with pair =
// x*my + y; U[k, a, x, b, y] is stored spin-major.  When RB holds whole spin
// groups (the usual case) a block's load reads the NS*NS sources of a pair once
// and runs the spin action in registers for all NS*NS rows; when fewer rows fit
// (large k-grids) each bank loads its own row, with the same arithmetic.
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
constexpr bool GROUPED = (RB % SSO) == 0;

// The typed unfold of one (k, x, y) pair (symmetry_maps unfold_isdf_operator,
// axis-local pair_transpose arm): source row, both endpoint gathers, then
// (mph * G) * nph, a -1 source being an exact zero; and U_k.
// (mph * G) * nph, a -1 source being an exact zero.  On an antiunitary row
// conj_trs selects the rule: 0 reads the partner tile; 1 conjugates the phased
// product (a Hermitian interaction); 2 reads conj(G) from G itself (a Green of
// real weights, whose partner IS conj(G): no partner tile exists).
__device__ __forceinline__ void lrx_unfold_pair(
    const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt, const UnfoldTab& t,
    int k, long long xx, long long yy, lrx_c2 (&g)[NS][NR], lrx_c2 (&u)[NS][NS],
    lrx_c2 (&ur)[NR][NR]) {
    const lrx_c2* __restrict__ mph = reinterpret_cast<const lrx_c2*>(t.mph);
    const lrx_c2* __restrict__ nph = reinterpret_cast<const lrx_c2*>(t.nph);
    const lrx_c2* __restrict__ spin = reinterpret_cast<const lrx_c2*>(t.spin);
    const lrx_c2* __restrict__ spin_r =
        reinterpret_cast<const lrx_c2*>(t.spin_r ? t.spin_r : t.spin);
    const bool anti = t.trs[k] != 0, conj_row = anti && t.conj_trs == 1;
    const bool conj_src = anti && t.conj_trs == 2;
    const lrx_c2* src = ((anti && t.conj_trs == 0) ? gt : gp) + (long long)t.row[k] * t.ml * t.nl;
#pragma unroll
    for (int c = 0; c < NS; ++c) {
        const long long li = (long long)k * t.ml + xx * NS + c;
        const int ls = t.lsrc[li];
        const lrx_c2 mp = mph[li];
#pragma unroll
        for (int d = 0; d < NR; ++d) {
            const long long rj = (long long)k * t.nl + yy * NR + d;
            const int rs = t.rsrc[rj];
            lrx_c2 v = {0.0, 0.0};
            if (ls >= 0 && rs >= 0) {
                lrx_c2 sv = src[(long long)ls * t.nl + rs];
                if (conj_src) sv.y = -sv.y;
                v = lrx_mul_xla(lrx_mul_xla(mp, sv), nph[rj]);
                if (conj_row) v.y = -v.y;
            }
            g[c][d] = v;
        }
    }
#pragma unroll
    for (int a = 0; a < NS; ++a)
#pragma unroll
        for (int b = 0; b < NS; ++b) u[a][b] = spin[((long long)k * NS + a) * NS + b];
#pragma unroll
    for (int a = 0; a < NR; ++a)
#pragma unroll
        for (int b = 0; b < NR; ++b) ur[a][b] = spin_r[((long long)k * NR + a) * NR + b];
}

// Row a of U G Ur^dagger, accumulated exactly as the spin-rotate FFI does
// (Ur = U for a Green).
__device__ __forceinline__ void lrx_spin_row(const lrx_c2 (&u)[NS][NS], const lrx_c2 (&ur)[NR][NR],
                                             const lrx_c2 (&g)[NS][NR], int a, lrx_c2 (&out)[NR]) {
    lrx_c2 left[NR];
#pragma unroll
    for (int d = 0; d < NR; ++d) {
        lrx_c2 v = {0.0, 0.0};
#pragma unroll
        for (int c = 0; c < NS; ++c) {
            const lrx_c2 p = lrx_rot_mul(u[a][c], g[c][d]);
            v.x = __dadd_rn(v.x, p.x); v.y = __dadd_rn(v.y, p.y);
        }
        left[d] = v;
    }
#pragma unroll
    for (int b = 0; b < NR; ++b) {
        lrx_c2 v = {0.0, 0.0};
#pragma unroll
        for (int d = 0; d < NR; ++d) {
            const lrx_c2 p = lrx_rot_mul_conj(left[d], ur[b][d]);
            v.x = __dadd_rn(v.x, p.x); v.y = __dadd_rn(v.y, p.y);
        }
        out[b] = v;
    }
}

// The grouped load (RB holds whole spin groups): PB pairs per block, the NS*NS
// sources of a pair read once and U G U^dagger formed in registers; bank
// (jp, a, b) = row jp*SS + a*NS + b holds all NK values of that element.
template <int PB>
__device__ __forceinline__ void lrx_group_load(
    const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt, const UnfoldTab& t,
    long long p0, long long pairs, long long my, lrx_c2* sm) {
    for (int i = threadIdx.x; i < PB * NK; i += blockDim.x) {
        const int k = i / PB, jp = i % PB;
        const long long pr = p0 + jp;
        if (pr < pairs) {
            const long long xx = pr / my, yy = pr - xx * my;
            lrx_c2 g[NS][NR], u[NS][NS], ur[NR][NR];
            lrx_unfold_pair(gp, gt, t, k, xx, yy, g, u, ur);
#pragma unroll
            for (int a = 0; a < NS; ++a) {
                if constexpr (NA != NS) { if (a < t.a0 || a >= t.a0 + NA) continue; }
                lrx_c2 out[NR];
                lrx_spin_row(u, ur, g, a, out);
#pragma unroll
                for (int b = 0; b < NR; ++b) {
                    if constexpr (NA != NS) {
                        if (b < t.b0 || b >= t.b0 + NA) continue;
                        sm[(jp * SSO + (a - t.a0) * NA + (b - t.b0)) * SP + k] = out[b];
                    } else {
                        sm[(jp * SS + a * NR + b) * SP + k] = out[b];
                    }
                }
            }
        } else {
#pragma unroll
            for (int ab = 0; ab < SSO; ++ab) sm[(jp * SSO + ab) * SP + k] = {0.0, 0.0};
        }
    }
}

// The load of modes 7 and 9: the block's RB rows r = (pair, a, b) of the
// unfolded operand into the banks, grouped when RB holds whole spin groups.
__device__ __forceinline__ void lrx_unfold_load(
    const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt, const UnfoldTab& t,
    long long r0, lrx_c2* sm) {
    const long long my = t.nl / NR, pairs = (t.ml / NS) * my;
    if constexpr (GROUPED) {
        lrx_group_load<RB / SSO>(gp, gt, t, r0 / SSO, pairs, my, sm);
    } else {
        for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
            const int k = i / RB, j = i % RB;
            const long long r = r0 + j, pr = r / SSO;
            lrx_c2 v = {0.0, 0.0};
            if (pr < pairs) {
                const long long xx = pr / my, yy = pr - xx * my;
                const int a = (NA == NS) ? (int)((r % SS) / NR) : t.a0 + (int)((r % SSO) / NA);
                const int b = (NA == NS) ? (int)(r % NR) : t.b0 + (int)((r % SSO) % NA);
                lrx_c2 g[NS][NR], u[NS][NS], ur[NR][NR], out[NR];
                lrx_unfold_pair(gp, gt, t, k, xx, yy, g, u, ur);
                lrx_spin_row(u, ur, g, a, out);
                v = out[b];
            }
            sm[j * SP + k] = v;
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
// Bank cell of row j at flat k: mode 11 the k-box stage's padded row, mode 7 the family's.
#if LRX_MODE == 11
constexpr int TT_RSTRIDE = lrx_kbox::Geo<NX, NY, NZ>::RS;
__device__ __forceinline__ int tt_cell(int j, int k) { return j * TT_RSTRIDE + lrx_kbox::Geo<NX, NY, NZ>::at(k); }
#else
constexpr int TT_RSTRIDE = SP;
__device__ __forceinline__ int tt_cell(int j, int k) { return j * TT_RSTRIDE + k; }
#endif
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
    const long long x0 = pr0 / my, y0 = pr0 - x0 * my;
    for (int i = threadIdx.x; i < TP * NK; i += blockDim.x) {
        const int jp = i / NK, k = i % NK;
        if (jp >= npr) continue;
        long long xx = x0, yy = y0 + jp;
        while (yy >= my) { yy -= my; ++xx; }
        lrx_async::copy<4 * NS>(s.ls(b) + i * NS, t.lsrc + (long long)k * t.ml + xx * NS);
        lrx_async::copy<4 * NR>(s.rs(b) + i * NR, t.rsrc + (long long)k * t.nl + yy * NR);
        if constexpr (TT_NW > 0) lrx_async::copy<16>(s.w(b) + i, kern + ((long long)k * mx + xx) * my + yy);
    }
    for (int i = threadIdx.x; i < TP * (NS + NR) * NK; i += blockDim.x) {
        const int k = i % NK, q = i / NK, jp = q / (NS + NR), e = q % (NS + NR);
        if (jp >= npr) continue;
        long long xx = x0, yy = y0 + jp;
        while (yy >= my) { yy -= my; ++xx; }
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
__device__ __forceinline__ void tt_finish(const TileTabs& s, int b, int npr, lrx_c2* bank) {
    const lrx_c2* u = s.u();
    const lrx_c2* mp = s.mp(b);
    const lrx_c2* np = s.np(b);
    const int* ls = s.ls(b);
    const int* rs = s.rs(b);
    for (int i = threadIdx.x; i < NK * TT_GT * NR; i += blockDim.x) {
        const int k = i % NK, q = i / NK, gi = q / NR, d = q % NR, jp = gi / TT_OPS;
        if (jp >= npr) continue;
        lrx_c2* col = bank + tt_cell(gi * SS + d, k);                   // cell (c, d) at col[c * CS]
        constexpr int CS = NR * TT_RSTRIDE;
        const int f = s.flag()[k];
        const bool conj_src = f & 2, conj_row = f & 4;
        const int r = rs[(jp * NK + k) * NR + d];
        const lrx_c2 nq = np[(jp * NR + d) * NK + k];
        lrx_c2 g[NS];
#pragma unroll
        for (int c = 0; c < NS; ++c) {
            lrx_c2 v = {0.0, 0.0};
            if (ls[(jp * NK + k) * NS + c] >= 0 && r >= 0) {
                lrx_c2 sv = col[c * CS];
                if (conj_src) sv.y = -sv.y;
                v = lrx_mul_xla(lrx_mul_xla(mp[(jp * NS + c) * NK + k], sv), nq);
                if (conj_row) v.y = -v.y;
            }
            g[c] = v;
        }
#pragma unroll
        for (int aa = 0; aa < NS; ++aa) {
            lrx_c2 v = {0.0, 0.0};
#pragma unroll
            for (int c = 0; c < NS; ++c) {
                const lrx_c2 p = lrx_rot_mul(u[(aa * NS + c) * NK + k], g[c]);
                v.x = __dadd_rn(v.x, p.x); v.y = __dadd_rn(v.y, p.y);
            }
            col[aa * CS] = v;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NK * TT_GT * NS; i += blockDim.x) {
        const int k = i % NK, q = i / NK, gi = q / NS, ra = q % NS, jp = gi / TT_OPS;
        if (jp >= npr) continue;
        lrx_c2* row = bank + tt_cell(gi * SS + ra * NR, k);             // cell (ra, d) at row[d * RS1]
        constexpr int RS1 = TT_RSTRIDE;
        lrx_c2 left[NR];
#pragma unroll
        for (int d = 0; d < NR; ++d) left[d] = row[d * RS1];
#pragma unroll
        for (int bb = 0; bb < NR; ++bb) {
            lrx_c2 v = {0.0, 0.0};
#pragma unroll
            for (int d = 0; d < NR; ++d) {
                const lrx_c2 p = lrx_rot_mul_conj(left[d], u[(bb * NR + d) * NK + k]);
                v.x = __dadd_rn(v.x, p.x); v.y = __dadd_rn(v.y, p.y);
            }
            row[bb * RS1] = v;
        }
    }
    __syncthreads();
}
#endif

#if LRX_MODE == 7 && LRX_TT
// Mode 7 on the tile tables: a persistent grid over tiles of TP pairs (RB = TP * SS rows); tile
// n + 1's tables (with its W_R values) load beside tile n's gather, then the finish, the inverse
// transform, the Mid from shared W_R, the forward transform and the store, as below.
#ifndef LRX_MINB
#define LRX_MINB 1
#endif
extern "C" __global__ void __launch_bounds__(256, LRX_MINB) lrx_kconv(
    const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt,
    const lrx_c2* __restrict__ kern, lrx_c2* __restrict__ y, UnfoldTab t, double scale) {
    static_assert(RB == TT_ROWS, "the host passes the tile's pairs and rows together");
    extern __shared__ lrx_c2 sm[];
    using namespace cufftdx;
    const long long mx = t.ml / NS, my = t.nl / NS, pairs = mx * my;
    const TileTabs s{reinterpret_cast<char*>(sm + RB * SP)};
    const long long stride = (long long)gridDim.x * TP;
    auto npr_of = [&](long long q) { return (int)min((long long)TP, pairs - q); };
    long long p0 = (long long)blockIdx.x * TP;
    tt_fixed(t, s);
    if (p0 < pairs) tt_tile(t, s, 0, p0, npr_of(p0), my, kern, mx);
    lrx_async::commit();
    for (int b = 0; p0 < pairs; p0 += stride, b ^= 1) {
        lrx_async::wait_all();
        __syncthreads();                               // tables b in; the previous tile's bank reads done
        const int npr = npr_of(p0);
        tt_gather(gp, gt, gp, gt, t, s, b, npr, sm);
        lrx_async::commit();
        if (p0 + stride < pairs) tt_tile(t, s, b ^ 1, p0 + stride, npr_of(p0 + stride), my, kern, mx);
        lrx_async::commit();
        lrx_async::wait_prior<1>();                     // this tile's cells (not the next tables)
        __syncthreads();
        tt_finish(s, b, npr, sm);
        transform3<fft_direction::inverse>(sm);
        const lrx_c2* w = s.w(b);
        for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
            const int k = i / RB, j = i % RB;
            if (j / SS < npr) sm[j * SP + k] = lrx_mul(sm[j * SP + k], w[(j / SS) * NK + k]);
        }
        __syncthreads();
        transform3<fft_direction::forward>(sm);
        const long long x0 = p0 / my, y0 = p0 - x0 * my;
        for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
            const int k = i / RB, j = i % RB, jp = j / SS;
            const long long ko = lrx_out_row(t, k);
            if (jp < npr && ko >= 0) {
                long long xx = x0, yy = y0 + jp;
                while (yy >= my) { yy -= my; ++xx; }
                const int a = (j % SS) / NS, bb = j % NS;
                const lrx_c2 v = sm[j * SP + k];
                y[((ko * NS + a) * mx + xx) * (my * NS) + bb * my + yy] = {v.x * scale, v.y * scale};
            }
        }
        __syncthreads();                               // the bank is read before the next gather
    }
}
#elif LRX_MODE == 7
extern "C" __global__ void __launch_bounds__(256) lrx_kconv(
    const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt,
    const lrx_c2* __restrict__ kern, lrx_c2* __restrict__ y, UnfoldTab t, double scale) {
    extern __shared__ lrx_c2 sm[];
    using namespace cufftdx;
    const long long mx = t.ml / NS, my = t.nl / NS, pairs = mx * my;
    const long long r0 = (long long)blockIdx.x * RB;
    lrx_unfold_load(gp, gt, t, r0, sm);
    __syncthreads();
    transform3<fft_direction::inverse>(sm);
    for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
        const int k = i / RB, j = i % RB;
        const long long pr = (r0 + j) / SSO;
        if (pr < pairs) {
            const long long xx = pr / my, yy = pr - xx * my;
            sm[j * SP + k] = lrx_mul(sm[j * SP + k], kern[((long long)k * mx + xx) * my + yy]);
        }
    }
    __syncthreads();
    transform3<fft_direction::forward>(sm);
    for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
        const int k = i / RB, j = i % RB;
        const long long r = r0 + j, pr = r / SSO;
        const long long ko = lrx_out_row(t, k);
        if (pr < pairs && ko >= 0) {
            const long long xx = pr / my, yy = pr - xx * my;
            // (a, b) within the stored block: U is (n_out, NA, mx, NA, my).
            const int a = (int)((r % SSO) / NA), b = (int)(r % NA);
            const lrx_c2 v = sm[j * SP + k];
            y[((ko * NA + a) * mx + xx) * (my * NA) + b * my + yy] = {v.x * scale, v.y * scale};
        }
    }
}
#elif LRX_MODE == 8
// Mode 8 has two arms, chosen at build from the grid and the opt-in shared memory:
// LRX_ARM 0, the resident kernel, when a spin group's ns^2 rows fit one block (the unfolded
// load and the vertex sum in one pass: A100 door, ns 4, 1.04-1.44x the split at 4^3-8^3 and
// 6x6x1/12x12x1); LRX_ARM 1, the k-box split arm, where they do not (it has no residency limit).
#if LRX_ARM == 0
// Mode 8: mode 7's load, then the Lorentz vertex sum in R space.  Per (k, pair)
// one thread holds the NS*NS transformed Green values g (scaled by si as the
// mode-3 transform stores them) and accumulates, block by block in (A, B)
// order, acc[a][b] += (i^code g[perm_A[a]][perm_B[b]]) * V[k, x, A, y, B]: the
// signed permutation is exact, the product is XLA's (no FMA) and the sum starts
// at zero, as the scan over blocks did.  The result goes back to the same banks.
static_assert(GROUPED && RB >= SS, "mode 8 keeps whole spin groups resident (the host picks RB)");
extern "C" __global__ void __launch_bounds__(256) lrx_kconv(
    const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt,
    const lrx_c2* __restrict__ kern, lrx_c2* __restrict__ y, UnfoldTab t, LorentzTab v) {
    extern __shared__ lrx_c2 sm[];
    using namespace cufftdx;
    constexpr int PB = RB / SS;
    const long long mx = t.ml / NS, my = t.nl / NS, pairs = mx * my;
    const long long p0 = (long long)blockIdx.x * PB;
    lrx_group_load<PB>(gp, gt, t, p0, pairs, my, sm);
    __syncthreads();
    transform3<fft_direction::inverse>(sm);
    const long long wy = (long long)my * v.nb, wx = (long long)v.na * wy;
    for (int i = threadIdx.x; i < PB * NK; i += blockDim.x) {
        const int k = i / PB, jp = i % PB;
        const long long pr = p0 + jp;
        if (pr >= pairs) continue;
        const long long xx = pr / my, yy = pr - xx * my;
        lrx_c2 g[SS], acc[SS];
#pragma unroll
        for (int ab = 0; ab < SS; ++ab) {
            const lrx_c2 z = sm[(jp * SS + ab) * SP + k];
            g[ab].x = __dmul_rn(z.x, v.s_g);
            g[ab].y = __dmul_rn(z.y, v.s_g);
            acc[ab].x = 0.0;
            acc[ab].y = 0.0;
        }
        const lrx_c2* __restrict__ w = kern + ((long long)k * mx + xx) * wx + yy * v.nb;
        for (int ia = 0; ia < v.na; ++ia) {
            for (int ib = 0; ib < v.nb; ++ib) {
                const lrx_c2 wv = w[ia * wy + ib];
#pragma unroll
                for (int a = 0; a < NS; ++a) {
                    const int pa = (int)((v.perm_l >> (16 * ia + 4 * a)) & 15);
                    const int ca = (int)((v.phase_l >> (8 * ia + 2 * a)) & 3);
#pragma unroll
                    for (int b = 0; b < NS; ++b) {
                        const int pb = (int)((v.perm_r >> (16 * ib + 4 * b)) & 15);
                        const int cb = (int)((v.phase_r >> (8 * ib + 2 * b)) & 3);
                        // phase_A[a] * conj(phase_B[b]) as one quarter turn.
                        const lrx_c2 p = lrx_mul_xla(lrx_phase(g[pa * NS + pb], (ca + 4 - cb) & 3), wv);
                        acc[a * NS + b].x = __dadd_rn(acc[a * NS + b].x, p.x);
                        acc[a * NS + b].y = __dadd_rn(acc[a * NS + b].y, p.y);
                    }
                }
            }
        }
#pragma unroll
        for (int ab = 0; ab < SS; ++ab) sm[(jp * SS + ab) * SP + k] = acc[ab];
    }
    __syncthreads();
    transform3<fft_direction::forward>(sm);
    for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
        const int k = i / RB, j = i % RB;
        const long long r = (long long)blockIdx.x * RB + j, pr = r / SS;
        const long long ko = lrx_out_row(t, k);
        if (pr < pairs && ko >= 0) {
            const long long xx = pr / my, yy = pr - xx * my;
            const int a = (int)((r % SS) / NS), b = (int)(r % NS);
            const lrx_c2 z = sm[j * SP + k];
            y[((ko * NS + a) * mx + xx) * t.nl + b * my + yy] =
                {__dmul_rn(__dmul_rn(z.x, v.s_f), v.mult), __dmul_rn(__dmul_rn(z.y, v.s_f), v.mult)};
        }
    }
}
#else
// Arm 1: the k-box stage's split arm (kbox_stage.cuh), chunked over pairs through the
// (nk, npairs * SS) intermediate y:
//   phase 0  plane (z, y) inverse of the unfolded Green (every spin element of a pair, formed as
//            lrx_group_load forms it) -> y
//   phase 1  group pencil: the x inverse of every member; each (member, t) thread forms its
//            member's vertex sum from the group's values and V[k, x, A, y, B] (staged by cp.async)
//   phase 2  plane (z, y) forward, y -> y
//   phase 3  pencil x forward, the scales, the store into U (kout rows)
// Per (k, pair, a, b), with g = s_g * (the transformed Green), the sum runs block by block in
// (A, B) order: acc += (i^code g[perm_A[a]][perm_B[b]]) * V[A, B], XLA's product (no FMA) and
// __dadd_rn from zero, and the transforms run z,y then x each way: it rounds as arm 0.
#include "kbox_stage.cuh"
static_assert(NR == NS, "mode 8 loads Greens: one spin width");
constexpr int TRC = LRX_TR;                    // plane tile columns (whole spin groups)
constexpr int TY = LRX_TY;                     // pair instances per group-pencil block

struct LorArgs {
    const lrx_c2 *gp, *gt, *kern;              // parent Green, partner, V (nk, mx, na, my, nb)
    lrx_c2* u;                                 // (n_out, NS, mx, NS, my)
    lrx_c2* y;                                 // (NK, npairs * SS) intermediate
    long long p0, npairs, mx, my;              // this chunk's pairs [p0, p0 + npairs)
};

struct LorLoad {                               // tile column c: pair p0 + c / SS, member c % SS
    static constexpr bool kDirect = true, kFinish = false;
    const LorArgs* a;
    const UnfoldTab* t;
    template <class View>
    __device__ void direct(const View& view, int k0, int k1, long long col0, int width,
                           long long ncols) const {
        const int groups = width / SS;
        for (int i = threadIdx.x; i < (k1 - k0) * groups; i += blockDim.x) {
            const int k = k0 + i / groups, j = i % groups;
            const long long c0 = col0 + (long long)j * SS;
            if (c0 < ncols) {
                const long long pr = a->p0 + c0 / SS, xx = pr / a->my, yy = pr - xx * a->my;
                lrx_c2 g[NS][NR], u[NS][NS], ur[NR][NR];
                lrx_unfold_pair(a->gp, a->gt, *t, k, xx, yy, g, u, ur);
#pragma unroll
                for (int r = 0; r < NS; ++r) {
                    lrx_c2 out[NR];
                    lrx_spin_row(u, ur, g, r, out);
#pragma unroll
                    for (int b = 0; b < NR; ++b) view(k, j * SS + r * NR + b) = out[b];
                }
            } else {
#pragma unroll
                for (int e = 0; e < SS; ++e) view(k, j * SS + e) = {0.0, 0.0};
            }
        }
    }
};

struct LorCols {
    __device__ long long col(long long inst, int member) const { return inst * SS + member; }
};

struct LorMid {                                // the vertex sum; V staged per (kx, instance)
    static constexpr int kAux = 16;            // na * nb <= 16
    const LorArgs* a;
    LorentzTab v;
    __device__ void stage_aux(lrx_c2* saux, long long p, long long inst0, int ld) const {
        const int ne = v.na * v.nb;
        for (int i = threadIdx.x; i < NX * 16 * TY; i += blockDim.x) {
            const int e = i % 16, t = (i / 16) % TY, kx = i / (16 * TY);
            if (e >= ne) continue;
            long long inst = inst0 + t;
            if (inst >= a->npairs) inst = inst0;
            const long long pr = a->p0 + inst, xx = pr / a->my, yy = pr - xx * a->my;
            const long long k = (long long)kx * NY * NZ + p;
            const int A = e / v.nb, B = e % v.nb;
            lrx_kbox::cp_async<16>(saux + (kx * TY + t) * ld + e,
                                   a->kern + (((k * a->mx + xx) * v.na + A) * a->my + yy) * v.nb + B);
        }
    }
    struct F {
        int src[16], code[16], ne;
        double s_g;
        __device__ lrx_c2 operator()(const lrx_c2* grp, const lrx_c2* aux) const {
            lrx_c2 acc = {0.0, 0.0};
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                if (e < ne) {
                    const lrx_c2 z = grp[src[e] * TY];
                    const lrx_c2 g = {__dmul_rn(z.x, s_g), __dmul_rn(z.y, s_g)};
                    const lrx_c2 q = lrx_mul_xla(lrx_phase(g, code[e]), aux[e]);
                    acc.x = __dadd_rn(acc.x, q.x);
                    acc.y = __dadd_rn(acc.y, q.y);
                }
            }
            return acc;
        }
    };
    __device__ F bind(int member) const {       // member = a * NS + b
        F f;
        const int ma = member / NS, mb = member % NS;
        f.ne = v.na * v.nb;
        f.s_g = v.s_g;
#pragma unroll
        for (int e = 0; e < 16; ++e) {
            const int ia = e / v.nb, ib = e % v.nb;
            if (e < f.ne) {
                const int pa = (int)((v.perm_l >> (16 * ia + 4 * ma)) & 15);
                const int ca = (int)((v.phase_l >> (8 * ia + 2 * ma)) & 3);
                const int pb = (int)((v.perm_r >> (16 * ib + 4 * mb)) & 15);
                const int cb = (int)((v.phase_r >> (8 * ib + 2 * mb)) & 3);
                f.src[e] = pa * NS + pb;
                f.code[e] = (ca + 4 - cb) & 3;
            } else {
                f.src[e] = 0;
                f.code[e] = 0;
            }
        }
        return f;
    }
};

struct LorYLoad {                              // the intermediate, staged by cp.async
    static constexpr bool kDirect = false, kFinish = false;
    const lrx_c2* y;
    long long n;
    __device__ const lrx_c2* stage(int k, long long c) const { return y + (long long)k * n + c; }
};

struct LorFinal {                              // (z s_f) mult, as the resident store scaled
    double s_f, mult;
    __device__ lrx_c2 operator()(int, long long, lrx_c2 z) const {
        return {__dmul_rn(__dmul_rn(z.x, s_f), mult), __dmul_rn(__dmul_rn(z.y, s_f), mult)};
    }
};

struct LorStore {                              // U[(ko, a, x, b, y)], full-k row k at kout[k]
    const LorArgs* a;
    const UnfoldTab* t;
    __device__ void put(int k, long long col, lrx_c2 v) const {
        const long long ko = lrx_out_row(*t, k);
        if (ko < 0) return;
        const long long pr = a->p0 + col / SS, xx = pr / a->my, yy = pr - xx * a->my;
        const int m = int(col % SS), ma = m / NS, mb = m % NS;
        a->u[((ko * NS + ma) * a->mx + xx) * t->nl + mb * a->my + yy] = v;
    }
};

extern "C" __global__ void __launch_bounds__(256) lrx_kconv(LorArgs a, UnfoldTab t, LorentzTab v, int phase) {
    extern __shared__ lrx_c2 sm[];
    using namespace cufftdx;
    const long long ncols = a.npairs * SS;
    const lrx_kbox::Plain<lrx_c2> yy{a.y, ncols};
    if (phase == 0) {
        lrx_kbox::plane_pass<NX, NY, NZ, LRX_SM, fft_direction::inverse, TRC>(sm, ncols, LorLoad{&a, &t}, yy);
    } else if (phase == 1) {
        lrx_kbox::pencil_group_pass<NX, NY, NZ, LRX_SM, SS, TY, false>(
            a.y, sm, ncols, a.npairs, LorCols{}, LorMid{&a, v}, yy);
    } else if (phase == 2) {
        lrx_kbox::plane_pass<NX, NY, NZ, LRX_SM, fft_direction::forward, TRC>(sm, ncols, LorYLoad{a.y, ncols}, yy);
    } else {
        lrx_kbox::pencil_pass<NX, NY, NZ, LRX_SM, false, fft_direction::forward>(
            a.y, ncols, LorFinal{v.s_f, v.mult}, LorStore{&a, &t});
    }
}
#endif
#elif LRX_MODE == 9
// Mode 9: mode 3's inverse transform of the interaction unfolded on its load
// from the wedge tiles; Y k-LEADING (nk, ml, nl), the row (pair, A, B) stored
// at merged endpoints (x*NS + A, y*NR + B), scaled as mode 3 scales.
extern "C" __global__ void __launch_bounds__(256) lrx_kconv(
    const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt, lrx_c2* __restrict__ y,
    UnfoldTab t, double scale) {
    extern __shared__ lrx_c2 sm[];
    using namespace cufftdx;
    const long long my = t.nl / NR, pairs = (t.ml / NS) * my;
    const long long r0 = (long long)blockIdx.x * RB;
    lrx_unfold_load(gp, gt, t, r0, sm);
    __syncthreads();
    transform3<fft_direction::inverse>(sm);
    for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
        const int k = i / RB, j = i % RB;
        const long long r = r0 + j, pr = r / SS;
        if (pr < pairs) {
            const long long xx = pr / my, yy = pr - xx * my;
            const int a = (int)((r % SS) / NR), b = (int)(r % NR);
            const lrx_c2 v = sm[j * SP + k];
            y[((long long)k * t.ml + xx * NS + a) * t.nl + yy * NR + b] = {v.x * scale, v.y * scale};
        }
    }
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
#pragma unroll
                for (int r = 0; r < NS; ++r) {
                    lrx_c2 out[NR];
                    lrx_spin_row(u, ur, g, r, out);
#pragma unroll
                    for (int b = 0; b < NR; ++b) view(k, j * og + r * NR + b) = out[b];
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
        const lrx_c2 p = lrx_mul_xla(a, b);
        v.x = __dadd_rn(v.x, p.x); v.y = __dadd_rn(v.y, p.y);
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
        const lrx_c2 p = lrx_mul_xla(a.alpha[o], v);
        *e = {__dadd_rn(e->x, p.x), __dadd_rn(e->y, p.y)};
    }
}

// Single arm: the group Mid reduces the pair's GRP transformed values and accumulates them.
// mid_group_tile runs one thread per (k, pair) of the tile, so every thread of the block takes
// part in the accumulation (a column-wise Store would leave it to the pairs' leading columns).
struct ChiMid {
    const ChiArgs* a;
    template <class Get>
    __device__ void group(int k, long long g, const Get& get) const {
        lrx_chi_acc(*a, k, a->p0 + g, lrx_chi_value(get, a->si));
    }
};

struct ChiStore {                              // split arm: the pair's leading column accumulates
    const ChiArgs* a;
    __device__ void put(int k, long long col, lrx_c2 v) const {
        if (col % GRP) return;
        lrx_chi_acc(*a, k, a->p0 + col / GRP, v);
    }
};

#if LRX_ARM == 1
// Split arm, pencil pass: the x-line inverse of every member, then member 0 forms the pair.
struct ChiCols {
    __device__ long long col(long long inst, int member) const { return inst * GRP + member; }
};
struct ChiPencilMid {
    static constexpr int kAux = 0;
    double si;
    __device__ void stage_aux(lrx_c2*, long long, long long, int) const {}
    struct F {
        int member;
        double si;
        __device__ lrx_c2 operator()(const lrx_c2* grp, const lrx_c2*) const {
            if (member != 0) return grp[member * LRX_TY];
            return lrx_chi_value([&](int q) { return grp[q * LRX_TY]; }, si);
        }
    };
    __device__ F bind(int member) const { return F{member, si}; }
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
#if LRX_ARM == 0 && LRX_TT
    (void)phase;
    (void)ld;
    // A persistent grid (the host launches the resident blocks): tile n + 1's tables load beside
    // tile n's gather, then the finish, the inverse transform and the accumulating Mid.
    const ChiMid mid{&a};
    static_assert(TRC == TT_ROWS, "the host passes the tile's pairs and columns together");
    const TileTabs s{reinterpret_cast<char*>(sm + TRC * lrx_kbox::Geo<NX, NY, NZ>::RS)};
    const long long stride = (long long)gridDim.x * TRC;
    auto npr_of = [&](long long c0) { return (int)min((long long)TP, a.npairs - c0 / GRP); };
    long long col0 = (long long)blockIdx.x * TRC;
    tt_fixed(t, s);
    if (col0 < ncols) tt_tile(t, s, 0, a.p0 + col0 / GRP, npr_of(col0), a.my, nullptr, 0);
    lrx_async::commit();
    for (int b = 0; col0 < ncols; col0 += stride, b ^= 1) {
        lrx_async::wait_all();
        __syncthreads();                               // tables b in; the previous tile's bank reads done
        const int npr = npr_of(col0);
        tt_gather(a.gv, a.gvt, a.gc, a.gct, t, s, b, npr, sm);
        lrx_async::commit();
        if (col0 + stride < ncols)
            tt_tile(t, s, b ^ 1, a.p0 + (col0 + stride) / GRP, npr_of(col0 + stride), a.my, nullptr, 0);
        lrx_async::commit();
        lrx_async::wait_prior<1>();                     // this tile's cells (not the next tables)
        __syncthreads();
        tt_finish(s, b, npr, sm);
        lrx_kbox::transform3<NX, NY, NZ, TRC, LRX_SM, fft_direction::inverse>(sm);
        lrx_kbox::mid_group_tile<NX, NY, NZ, TRC, GRP>(sm, col0, ncols, mid);
    }
#elif LRX_ARM == 0
    (void)phase;
    const ChiMid mid{&a};
    for (long long col0 = (long long)blockIdx.x * TRC; col0 < ncols; col0 += (long long)gridDim.x * TRC) {
        lrx_kbox::stage_tile<NX, NY, NZ, TRC>(sm, col0, ncols, ld);
        lrx_kbox::transform3<NX, NY, NZ, TRC, LRX_SM, fft_direction::inverse>(sm);
        lrx_kbox::mid_group_tile<NX, NY, NZ, TRC, GRP>(sm, col0, ncols, mid);
    }
#else
    const ChiStore st{&a};
    if (phase == 0) {
        lrx_kbox::plane_pass<NX, NY, NZ, LRX_SM, fft_direction::inverse, TRC>(
            sm, ncols, ld, lrx_kbox::Plain<lrx_c2>{a.y, ncols});
    } else {
        lrx_kbox::pencil_group_pass<NX, NY, NZ, LRX_SM, GRP, LRX_TY, false>(
            a.y, sm, ncols, a.npairs, ChiCols{}, ChiPencilMid{a.si}, st);
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
               int arm = 0, tr = 0, ty = 0, threads2 = 0, smem2 = 0, sms = 0; };
// ctx, mode, nkx, nky, nkz, ns, nsr, f32, variant (mode 11: the static completion; mode 7: its output spin block)
using Key = std::tuple<CUcontext, int, int, int, int, int, int, int, int>;
static std::mutex g_mu;
static std::map<Key, Built> g_cache;
static std::map<Key, std::string> g_fail;

using nvrtc::exists;
using nvrtc::toolkit_include;

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
    const bool pair = mode < 2 || mode == 6;           // three banks per row
    const int banks = pair ? 3 : 1;
    const long long rows_max = pair ? kRowsMax : kRowsMax1;
    long long row_bytes = static_cast<long long>(banks) * (f32 ? 8 : 16) * sp;
    // The budget never exceeds this device's opt-in maximum (99 KiB on sm_86/89/120, below the
    // pair modes' 100 KiB): an unclamped budget asked for more than the device grants at launch.
    const long long budget = std::min<long long>(pair ? kSmemBudget : kSmemBudget1, smem_optin);
    long long rb = std::min<long long>(rows_max, budget / row_bytes);
    // Mode 8's resident arm sums the Lorentz blocks across a pair's spin rows, so it needs a
    // whole spin group per block: reach for the opt-in shared memory first.
    if (rb < 1 || (mode == 8 && rb < ns * ns)) rb = std::min<long long>(rows_max, smem_optin / row_bytes);
    const bool lor_split = mode == 8 && rb < ns * ns;   // no resident spin group: the split arm
    // Mode 11 runs on the k-box stage: its launch rule (kbox_plan) decides the arm, the tile and
    // the shared memory from the grid and this device's opt-in budget; RB is unused.
    const int chi_grp = 2 * ns * ns;
    lrx_kbox::Plan kplan{};
    int chi_trc = 0, chi_ty = 0, chi_threads = 0, chi_tt = 0, chi_minb = 1;
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
            // The tile tables (sm_80+ cp.async): the largest tile, at most the plan's, whose bank
            // and tables (UnfoldTiles) fit two blocks on an SM with the device's per-block
            // reservation; none fits: the register load at the plan's tile.
            int smem_sm = 0, smem_rsv = 0;
            LRX_CUDA_CHECK(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev),
                           "max shared memory per SM");
            LRX_CUDA_CHECK(cudaDeviceGetAttribute(&smem_rsv, cudaDevAttrReservedSharedMemoryPerBlock, dev),
                           "reserved shared memory per block");
            for (int tp = kplan.tr; cc_major >= 8 && tp >= 1 && kplan.threads == kThreads; tp /= 2) {
                const long long blk = static_cast<long long>(tp) * chi_grp * g.rs() * 16 +
                                      lrx_kbox::UnfoldTiles{nk, ns, ns, tp, 0}.bytes();
                if (blk <= smem_optin && 2 * (blk + smem_rsv) <= smem_sm) {
                    chi_tt = tp;
                    chi_minb = 2;
                    chi_trc = tp * chi_grp;
                    chi_smem = blk;
                    break;
                }
            }
        } else {
            chi_trc = kplan.tr;                            // plane tiles of tr columns (whole spin groups)
            chi_threads = kThreads;
            chi_smem = static_cast<long long>(chi_trc) * g.pr() * 16;
            chi_smem2 = (static_cast<long long>(nkx) * chi_grp * chi_ty + static_cast<long long>(nkx) * chi_ty) * 16;
        }
    }
    // Modes 2 and 3 run on the k-box stage: kbox_plan decides the arm, the tile and the shared
    // memory from the grid and this device's opt-in budget (RB is unused).  Mode 8 takes the
    // stage's split arm where its resident arm has no room for a spin group: the group pencil
    // forms each member on its own thread (a single arm's group Mid runs one thread per (k, pair):
    // 3766 vs 2156 us at 8^3 in the standalone bench).
    const bool kbox_rows = mode == 2 || mode == 3;
    int kb_arm = 0, kb_tr = 0, kb_ty = 0, kb_threads = 0, kb_threads2 = 0;
    long long kb_smem = 0, kb_smem2 = 0;
    if (kbox_rows || lor_split) {
        const lrx_kbox::Geometry g{nkx, nky, nkz};
        rb = 1;
        if (kbox_rows) {
            const lrx_kbox::Plan kp = lrx_kbox::kbox_plan(nkx, nky, nkz, 1, 1, f32 ? 8 : 16, smem_optin,
                                                          mode == 2 ? 2 : 1);
            kb_arm = kp.arm;
            kb_tr = kp.tr;
            kb_threads = kp.arm == 0 ? kp.threads : kThreads;
            kb_smem = kp.smem;                         // single: the tile; split: the plane pass
        } else {
            const int ss = ns * ns;
            kb_arm = 1;
            // Plane tiles of whole spin groups: the gathered load runs one thread per (k, group)
            // of a (ky, kz) plane, so a tile holds enough groups for a unit per thread (256),
            // within two blocks per SM.
            int grp = 1;
            while (static_cast<long long>(nky) * nkz * grp < kThreads &&
                   2LL * ss * (2 * grp) * g.pr() * 16 <= smem_optin)
                grp *= 2;
            kb_tr = ss * grp;
            kb_ty = std::max(1, 128 / ss);
            kb_threads = kThreads;
            kb_threads2 = ss * kb_ty;
            kb_smem = static_cast<long long>(kb_tr) * g.pr() * 16;
            kb_smem2 = (static_cast<long long>(nkx) * ss * kb_ty + static_cast<long long>(nkx) * kb_ty * 17) * 16;
        }
        if (kb_smem > smem_optin || kb_smem2 > smem_optin) {
            std::ostringstream os;
            os << "GATE mathdx-kconv-kbox-residency: got k-grid (" << nkx << "," << nky << "," << nkz << ") with ns="
               << ns << ", whose k-box " << (kb_arm ? "plane/pencil" : "tile") << " needs " << kb_smem << " / "
               << kb_smem2 << " B; want <= " << smem_optin << " B of opt-in shared memory on this device; why: the "
                  "stage keeps a (ky, kz) plane of its tile columns resident; fix: a smaller k-grid";
            return sticky("residency", os.str(), ffi::ErrorCode::kInvalidArgument);
        }
    }
    const int blk = (mode == 7 && variant > 0) ? variant : ns;    // mode 7's output spin block
    const int grp_rows = (mode == 7 && blk != ns) ? blk * blk : ns * nsr;
    if ((mode == 7 || mode == 8 || mode == 9) && rb >= grp_rows)
        rb -= rb % grp_rows;                           // whole spin groups: the grouped load
    // Mode 7 on the tile tables (whole spin groups, sm_80+ cp.async): the largest tile of whole
    // pairs, at most the grouped load's, whose bank and tables (UnfoldTiles, W_R staged) fit two
    // blocks on an SM with the device's per-block reservation; none fits: the register load.
    int m7_tp = 0;
    long long m7_smem = 0;
    if (mode == 7 && blk == ns && nsr == ns && cc_major >= 8 && rb >= grp_rows) {
        int smem_sm = 0, smem_rsv = 0;
        LRX_CUDA_CHECK(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev),
                       "max shared memory per SM");
        LRX_CUDA_CHECK(cudaDeviceGetAttribute(&smem_rsv, cudaDevAttrReservedSharedMemoryPerBlock, dev),
                       "reserved shared memory per block");
        for (int tp = static_cast<int>(rb / grp_rows); tp >= 1; tp /= 2) {
            const long long bytes = static_cast<long long>(tp) * grp_rows * row_bytes +
                                    lrx_kbox::UnfoldTiles{nk, ns, ns, tp, 1}.bytes();
            if (bytes <= smem_optin && 2 * (bytes + smem_rsv) <= smem_sm) {
                m7_tp = tp;
                m7_smem = bytes;
                rb = static_cast<long long>(tp) * grp_rows;
                break;
            }
        }
    }
    // (fewer rows than one spin group: mode 7 loads per bank, as mode 2 would fit)
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
           << ") whose resident row needs " << banks << "*" << (f32 ? 8 : 16) << "*(nk|1)=" << row_bytes << " B"
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
    // Mode 10: planes per block; a block that has its SM alone runs 512 threads.
    const int plane_threads = mode == 10 && plane_minb == 1 ? 512 : kThreads;
    if (mode == 7 && blk != ns) defs.push_back("-DLRX_NA=" + std::to_string(blk));
    if (mode == 7) {
        defs.push_back("-DLRX_TT=" + std::string(m7_tp ? "1" : "0"));
        defs.push_back("-DLRX_TP=" + std::to_string(m7_tp));
        defs.push_back("-DLRX_MINB=" + std::string(m7_tp ? "2" : "1"));
    }
    if (mode == 8 && !lor_split) defs.push_back("-DLRX_ARM=0");
    if (kbox_rows || lor_split) {
        defs.push_back("-DLRX_ARM=" + std::to_string(kb_arm));
        defs.push_back("-DLRX_TR=" + std::to_string(kb_tr));
        defs.push_back("-DLRX_TY=" + std::to_string(kb_ty));
        defs.push_back("-DLRX_THREADS=" + std::to_string(kb_threads));
    }
    if (mode == 11) {
        defs.push_back("-DLRX_ARM=" + std::to_string(kplan.arm));
        defs.push_back("-DLRX_TR=" + std::to_string(chi_trc));
        defs.push_back("-DLRX_TY=" + std::to_string(chi_ty));
        defs.push_back("-DLRX_THREADS=" + std::to_string(chi_threads));
        defs.push_back("-DLRX_COMPLETE=" + std::to_string(variant));
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
    if (mode == 11 || kbox_rows || lor_split || m7_tp)
        prog.headers.push_back({kbox::kHeaderName, kbox::kHeaderSrc});
    prog.defs = defs;
    nvrtc::mathdx_toolchain(root, cuda_inc, "cufftdx", &prog);
    prog.kernel = "lrx_kconv";
    std::string missing;
    const std::string key_hex = nvrtc::hex16(nvrtc::key(prog, &missing));
    const std::string dir(missing.empty() ? std::string(cubin_dir) : std::string());
    if (!missing.empty() && !std::string(cubin_dir).empty() && (mklpin::announce_here() || log_enabled()))
        std::fprintf(stderr, "[kconv_mathdx] disk cubin cache OFF for this build: empty version header(s) %s "
                     "would drop out of the key\n", missing.c_str());
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kconv_m" << mode << "_" << nkx << "x" << nky << "x" << nkz << "_ns" << ns
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
    if (kbox_rows || lor_split) {
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
    if (m7_tp) {                                       // a persistent grid of two blocks per SM
        int sms = 0;
        LRX_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SM count");
        b.smem = static_cast<int>(m7_smem);
        b.grid_cap = static_cast<long long>(sms) * 2;
    }
    if (mode == 10) {                                  // persistent blocks: the resident count
        int sms = 0;
        LRX_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SM count");
        b.grid_cap = static_cast<long long>(sms) * plane_minb;
    }
    b.compile_ms = ms;
    // Mode 10 always sets the dynamic limit: its static tables count against the
    // 48 KiB default too, so a plane just under 48 KiB would fail at launch.
    if (b.smem > 49152 || b.smem2 > 49152 || mode == 10) {
        cr = api.FuncSetAttribute(b.fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, std::max(b.smem, b.smem2));
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
    // Mode 11 at two blocks per SM: the largest shared-memory carveout, so the driver does not
    // pick a split that holds one block (a hint; residency is unchanged if it declines).
    if ((mode == 11 && chi_minb > 1) || m7_tp) {
        cr = api.FuncSetAttribute(b.fn, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100);
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

// Modes 2-5: one bank per row.  `mode` fixes the layout; shapes are checked
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
    if (mode == 2 || mode == 3) {                      // the k-box stage (single arm, or the split passes)
        auto launch = [&](int phase, long long blocks, int threads, int smem) -> ffi::Error {
            blocks = std::max(1LL, std::min(blocks, 2147483647LL));
            void* args[] = {(void*)&xp, (void*)&kp, (void*)&yp, (void*)&g, (void*)&phase};
            CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, threads, 1, 1,
                                                    static_cast<unsigned>(smem),
                                                    reinterpret_cast<CUstream>(stream), args, nullptr);
            if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
            return ffi::Error::Success();
        };
        if (k->arm == 0) return launch(0, (g.rows + k->tr - 1) / k->tr, k->threads, k->smem);
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
    void* args[] = {(void*)&xp, (void*)&kp, (void*)&yp, (void*)&g};
    const long long blocks = (g.rows + k->rb - 1) / k->rb;
    if (blocks > 2147483647LL) return bad("grid.x overflow");
    CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, kThreads, 1, 1,
                                            static_cast<unsigned>(k->smem),
                                            reinterpret_cast<CUstream>(stream), args, nullptr);
    if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
    return ffi::Error::Success();
}

// Mode 7: the Sigma k-leading convolution read from the raw-parent Green tiles
// through the typed-unfold tables; U (n_out, ns, mx, ns, my) spin-major, full-k
// row k stored at kout[k] (-1 = not stored).  kout == nullptr is the previous
// target's contract (every k at its own row, n_out = nk), kept so an older
// source tree still runs on this library.
static ffi::Error KleadUnfoldImpl(
    cudaStream_t stream, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    const ffi::AnyBuffer* kout, ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx,
    int64_t nky, int64_t nkz, double scale, std::string_view mathdx_root, std::string_view cubin_dir,
    int64_t conj_src = 0, int64_t spin_block = 0, int64_t a0 = 0, int64_t b0 = 0) {
    auto bad = [](const std::string& why) {
        return fail("klead unfold conv", why, ffi::ErrorCode::kInvalidArgument);
    };
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
    const auto C = ffi::DataType::C128, I = ffi::DataType::S32;
    if ((ns != 1 && ns != 2 && ns != 4) || ml % ns || nl % ns || np < 1 ||
        !is(Gp, C, {np, ml, nl}) || !is(Gt, C, {np, ml, nl}) || !is(row, I, {nk}) || !is(trs, I, {nk}) ||
        !is(lsrc, I, {nk, ml}) || !is(rsrc, I, {nk, nl}) || !is(mph, C, {nk, ml}) || !is(nph, C, {nk, nl}) ||
        !is(spin, C, {nk, ns, ns}) || !is(V, C, {nk, ml / ns, nl / ns}) ||
        (kout != nullptr && !is(*kout, I, {nk})) ||
        !(U->element_type() == C && U->dimensions().size() == 5 && U->dimensions()[0] >= 1 &&
          (kout != nullptr || U->dimensions()[0] == nk) &&
          U->dimensions()[1] == (spin_block ? spin_block : ns) && U->dimensions()[2] == ml / ns &&
          U->dimensions()[3] == (spin_block ? spin_block : ns) && U->dimensions()[4] == nl / ns))
        return bad("want c128 Gp=Gt (np,ml,nl); s32 row,trs,kout (nk), lsrc (nk,ml), rsrc (nk,nl); c128 "
                   "mph (nk,ml), nph (nk,nl), spin (nk,ns,ns), V (nk,ml/ns,nl/ns); U "
                   "(n_out,ns,ml/ns,ns,nl/ns), n_out = nk without kout");
    // The output spin block: rows [a0, a0 + d) x [b0, b0 + d) of the spin group (d = ns: all).
    const int64_t d = spin_block ? spin_block : ns;
    if (d < 1 || ns % d || a0 < 0 || b0 < 0 || a0 % d || b0 % d || a0 >= ns || b0 >= ns)
        return bad("want spin_block d dividing ns and block origins a0, b0 in [0, ns), multiples of d");
    const int64_t pairs = (ml / ns) * (nl / ns);
    if (pairs == 0) return ffi::Error::Success();
    const Built* k = nullptr;
    ffi::Error e = build(7, static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                         static_cast<int>(ns), false, mathdx_root, cubin_dir, &k, 0,
                         d == ns ? 0 : static_cast<int>(d));
    if (!e.success()) return e;
    UnfoldTab t{static_cast<const int*>(row.untyped_data()), static_cast<const int*>(trs.untyped_data()),
                static_cast<const int*>(lsrc.untyped_data()), static_cast<const int*>(rsrc.untyped_data()),
                static_cast<const double*>(mph.untyped_data()), static_cast<const double*>(nph.untyped_data()),
                static_cast<const double*>(spin.untyped_data()), ml, nl,
                kout ? static_cast<const int*>(kout->untyped_data()) : nullptr, conj_src ? 2 : 0, nullptr,
                static_cast<int>(a0), static_cast<int>(b0)};
    const void* gpp = Gp.untyped_data();
    const void* gtp = Gt.untyped_data();
    const void* vp = V.untyped_data();
    void* up = U->untyped_data();
    double sc = scale;
    void* args[] = {(void*)&gpp, (void*)&gtp, (void*)&vp, (void*)&up, (void*)&t, (void*)&sc};
    const long long rows = pairs * d * d;
    long long blocks = (rows + k->rb - 1) / k->rb;
    if (k->grid_cap > 0) blocks = std::min(blocks, k->grid_cap);   // the tile tables' persistent grid
    if (blocks > 2147483647LL) return bad("grid.x overflow");
    CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, kThreads, 1, 1,
                                            static_cast<unsigned>(k->smem),
                                            reinterpret_cast<CUstream>(stream), args, nullptr);
    if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
    return ffi::Error::Success();
}

// The parent-row targets.  `_rows` keeps its historical signature (every older source tree calls
// it); `_block` adds the conj-on-load partner and the output spin block (U2).
static ffi::Error KleadUnfoldRowsConv(
    cudaStream_t stream, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    ffi::AnyBuffer kout, ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky,
    int64_t nkz, double scale, std::string_view mathdx_root, std::string_view cubin_dir) {
    return KleadUnfoldImpl(stream, Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, &kout, V, U, nkx, nky,
                           nkz, scale, mathdx_root, cubin_dir);
}
static ffi::Error KleadUnfoldBlockConv(
    cudaStream_t stream, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    ffi::AnyBuffer kout, ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky,
    int64_t nkz, double scale, int64_t conj_src, int64_t spin_block, int64_t a0, int64_t b0,
    std::string_view mathdx_root, std::string_view cubin_dir) {
    return KleadUnfoldImpl(stream, Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, &kout, V, U, nkx, nky,
                           nkz, scale, mathdx_root, cubin_dir, conj_src, spin_block, a0, b0);
}
static ffi::Error KleadUnfoldConv(
    cudaStream_t stream, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz, double scale,
    std::string_view mathdx_root, std::string_view cubin_dir) {
    return KleadUnfoldImpl(stream, Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, nullptr, V, U, nkx, nky,
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
};

static ffi::Error KleadLorentzImpl(
    cudaStream_t stream, ffi::ScratchAllocator& scratch, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    const ffi::AnyBuffer* kout, ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx,
    int64_t nky, int64_t nkz, double scale_g, double scale_f, double mult,
    ffi::Span<const int64_t> perm_l, ffi::Span<const int64_t> phase_l, ffi::Span<const int64_t> perm_r,
    ffi::Span<const int64_t> phase_r, std::string_view mathdx_root, std::string_view cubin_dir,
    int64_t conj_src = 0) {
    auto bad = [](const std::string& why) {
        return fail("klead lorentz conv", why, ffi::ErrorCode::kInvalidArgument);
    };
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
    const auto gd = Gp.dimensions(), sd = spin.dimensions(), vd = V.dimensions();
    if (gd.size() != 3 || sd.size() != 3 || vd.size() != 5)
        return bad("want Gp (n_parent, ml, nl), spin (nk, ns, ns) and V (nk, mx, nA, my, nB)");
    const int64_t np = gd[0], ml = gd[1], nl = gd[2], ns = sd[1], na = vd[2], nb = vd[4];
    const auto C = ffi::DataType::C128, I = ffi::DataType::S32;
    if ((ns != 1 && ns != 2 && ns != 4) || ml % ns || nl % ns || np < 1 || na < 1 || na > 4 || nb < 1 ||
        nb > 4 || !is(Gp, C, {np, ml, nl}) || !is(Gt, C, {np, ml, nl}) || !is(row, I, {nk}) ||
        !is(trs, I, {nk}) || !is(lsrc, I, {nk, ml}) || !is(rsrc, I, {nk, nl}) || !is(mph, C, {nk, ml}) ||
        !is(nph, C, {nk, nl}) || !is(spin, C, {nk, ns, ns}) || !is(V, C, {nk, ml / ns, na, nl / ns, nb}) ||
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
                         static_cast<int>(ns), false, mathdx_root, cubin_dir, &k);
    if (!e.success()) return e;
    UnfoldTab t{static_cast<const int*>(row.untyped_data()), static_cast<const int*>(trs.untyped_data()),
                static_cast<const int*>(lsrc.untyped_data()), static_cast<const int*>(rsrc.untyped_data()),
                static_cast<const double*>(mph.untyped_data()), static_cast<const double*>(nph.untyped_data()),
                static_cast<const double*>(spin.untyped_data()), ml, nl,
                kout ? static_cast<const int*>(kout->untyped_data()) : nullptr, conj_src ? 2 : 0, nullptr, 0, 0};
    if (k->arm == 0) {                                 // the resident arm: whole spin groups per block
        const void* gpp = Gp.untyped_data();
        const void* gtp = Gt.untyped_data();
        const void* vp = V.untyped_data();
        void* up = U->untyped_data();
        void* args[] = {(void*)&gpp, (void*)&gtp, (void*)&vp, (void*)&up, (void*)&t, (void*)&v};
        const long long blocks = (pairs * ns * ns + k->rb - 1) / k->rb;
        if (blocks > 2147483647LL) return bad("grid.x overflow");
        CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, kThreads, 1, 1,
                                                static_cast<unsigned>(k->smem),
                                                reinterpret_cast<CUstream>(stream), args, nullptr);
        if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
        return ffi::Error::Success();
    }
    // The k-box split arm, chunked over pairs through a (nk, chunk * ns^2) intermediate no
    // larger than U itself (the budget: the output this call writes, n_out >= 1 rows of nk).
    const long long ss = ns * ns, per_pair = nk * ss * 16;
    const long long n_out = U->dimensions()[0];
    const long long chunk = std::max(1LL, std::min<long long>(pairs, pairs * n_out / nk));
    auto y = scratch.Allocate(static_cast<size_t>(chunk * per_pair));
    if (!y.has_value()) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-lorentz-scratch: got a refused " << chunk * per_pair << " B intermediate ("
           << chunk << " pairs); want the XLA scratch allocator to grant it";
        return fail("scratch", os.str(), ffi::ErrorCode::kResourceExhausted);
    }
    LorentzArgs a{Gp.untyped_data(), Gt.untyped_data(), V.untyped_data(), U->untyped_data(), *y,
                  0, 0, ml / ns, nl / ns};
    auto launch = [&](int phase, long long blocks, int threads, int smem) -> ffi::Error {
        blocks = std::max(1LL, std::min(blocks, 2147483647LL));
        void* args[] = {(void*)&a, (void*)&t, (void*)&v, (void*)&phase};
        CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, threads, 1, 1,
                                                static_cast<unsigned>(smem),
                                                reinterpret_cast<CUstream>(stream), args, nullptr);
        if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
        return ffi::Error::Success();
    };
    const long long cap = static_cast<long long>(k->sms) * 8;
    for (long long p0 = 0; p0 < pairs; p0 += chunk) {
        a.p0 = p0;
        a.npairs = std::min(chunk, pairs - p0);
        const long long ncols = a.npairs * ss;
        const long long plane = nkx * ((ncols + k->tr - 1) / k->tr);
        const long long group = nky * nkz * ((a.npairs + k->ty - 1) / k->ty);
        const long long pencil = (nky * nkz * ncols + kThreads - 1) / kThreads;
        if (auto e = launch(0, std::min(plane, cap), k->threads, k->smem); !e.success()) return e;
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
static ffi::Error KleadUnfoldFft(
    cudaStream_t stream, ffi::AnyBuffer Wp, ffi::AnyBuffer Wt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin_l,
    ffi::AnyBuffer spin_r, ffi::Result<ffi::AnyBuffer> Y, int64_t nkx, int64_t nky, int64_t nkz,
    double scale, int64_t conj_trs, std::string_view mathdx_root, std::string_view cubin_dir) {
    auto bad = [](const std::string& why) {
        return fail("klead unfold fft", why, ffi::ErrorCode::kInvalidArgument);
    };
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
                         static_cast<int>(nsl), false, mathdx_root, cubin_dir, &k, static_cast<int>(nsr));
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
    void* args[] = {(void*)&wpp, (void*)&wtp, (void*)&yp, (void*)&t, (void*)&sc};
    const long long rows = pairs * nsl * nsr;
    const long long blocks = (rows + k->rb - 1) / k->rb;
    if (blocks > 2147483647LL) return bad("grid.x overflow");
    CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, kThreads, 1, 1,
                                            static_cast<unsigned>(k->smem),
                                            reinterpret_cast<CUstream>(stream), args, nullptr);
    if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
    return ffi::Error::Success();
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
};

// Mode 11: chi_R accumulation from the raw-parent Green pair.  acc (n_out, nk, mx, my) in place.
static ffi::Error KleadChiUnfold(
    cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer Gv, ffi::AnyBuffer Gvt,
    ffi::AnyBuffer Gc, ffi::AnyBuffer Gct, ffi::AnyBuffer row, ffi::AnyBuffer trs, ffi::AnyBuffer lsrc,
    ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin, ffi::AnyBuffer alpha,
    ffi::AnyBuffer acc_in, ffi::Result<ffi::AnyBuffer> acc, int64_t nkx, int64_t nky, int64_t nkz, double si,
    int64_t conj_trs, int64_t complete, int64_t scratch_bytes, std::string_view mathdx_root,
    std::string_view cubin_dir) {
    auto bad = [](const std::string& why) {
        return fail("klead chi unfold", why, ffi::ErrorCode::kInvalidArgument);
    };
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
    const int64_t np = gd[0], ml = gd[1], nl = gd[2], ns = sd[1], n_out = ad[0];
    const auto C = ffi::DataType::C128, I = ffi::DataType::S32;
    if ((ns != 1 && ns != 2 && ns != 4) || ml % ns || nl % ns || np < 1 || n_out < 1 ||
        (conj_trs != 0 && conj_trs != 2) || (complete != 0 && complete != 1) ||
        !is(Gv, C, {np, ml, nl}) || !is(Gvt, C, {np, ml, nl}) || !is(Gc, C, {np, ml, nl}) ||
        !is(Gct, C, {np, ml, nl}) || !is(row, I, {nk}) || !is(trs, I, {nk}) || !is(lsrc, I, {nk, ml}) ||
        !is(rsrc, I, {nk, nl}) || !is(mph, C, {nk, ml}) || !is(nph, C, {nk, nl}) || !is(spin, C, {nk, ns, ns}) ||
        !is(alpha, C, {n_out}) || !is(acc_in, C, {n_out, nk, ml / ns, nl / ns}) ||
        !is(*acc, C, {n_out, nk, ml / ns, nl / ns}))
        return bad("want c128 Gv=Gvt=Gc=Gct (np,ml,nl); s32 row,trs (nk), lsrc (nk,ml), rsrc (nk,nl); c128 "
                   "mph (nk,ml), nph (nk,nl), spin (nk,ns,ns), alpha (n_out,), acc (n_out,nk,ml/ns,nl/ns); "
                   "conj_trs 0 (partner tiles) | 2 (the partner is conj(G)); complete 0|1");
    const int64_t my = nl / ns, pairs = (ml / ns) * my;
    if (pairs == 0) return ffi::Error::Success();
    const Built* k = nullptr;
    ffi::Error e = build(11, static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                         static_cast<int>(ns), false, mathdx_root, cubin_dir, &k, 0, static_cast<int>(complete));
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
              static_cast<int>(n_out), si};
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
        if (auto e1 = launch(1, std::min(pencil_items, cap), k->threads2, k->smem2); !e1.success()) return e1;
    }
    return ffi::Error::Success();
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
