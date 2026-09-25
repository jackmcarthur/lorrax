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
//             rows).  The tables are symmetry_maps's (unfold_load_tables);
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
//             that chain bit for bit.  It needs a whole spin group per block.
//  10 plane fft gather   Y[..., kb, kc] = FFT2_{b,c}(plane[..., b, c]) (forward,
//             unscaled: jnp.fft.fftn(norm='backward') over the last two axes)
//             where the plane is the route-G cylinder F (..., n_col) scattered
//             to its static cells and zero elsewhere.  The zero plane is never
//             written: the row FFTs run on the occupied rows only, gathering
//             their cells on load, then the column FFTs read dead rows as zero.
//             Its own embedded source, kPlaneSrc (see there).
// A new mode adds (1) an entry under its LRX_MODE value in kSrc, (2) a mode
// code and a handler below, (3) a router factory in ffi/fft.py.
//
// Residency: modes 0/1/6 keep three nk-long banks per row (one (col,mu) pair) in
// shared memory, modes 2-5 one; a k-grid whose row does not fit the device's
// opt-in shared memory, or an axis above the fp64 thread-FFT limit (40), is
// refused by name.  Modes 2/3/5 may run in place: every block reads all nk
// values of its own rows before it stores any of them.
//
// Headers: the Python router passes the installed wheel's nvidia/mathdx
// directory as the string attribute `mathdx_root`; the CUDA toolkit include
// (for libcu++, include/cccl) is derived from the loaded libnvrtc.
//
// Disk cache: the router passes `cubin_dir` (ffi.fft.cubin_cache_dir:
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
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <fstream>
#include <random>

#include "../common/mkl_thread_pin.h"
#include "../common/lrx_async_gather.h"

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

// Driver entry points resolved lazily (libcuda is already mapped by JAX).
struct DriverApi {
    CUresult (*ModuleLoadData)(CUmodule*, const void*) = nullptr;
    CUresult (*ModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                             unsigned, unsigned, unsigned, unsigned,
                             CUstream, void**, void**) = nullptr;
    CUresult (*CtxGetCurrent)(CUcontext*) = nullptr;
    CUresult (*GetErrorString)(CUresult, const char**) = nullptr;
    CUresult (*FuncSetAttribute)(CUfunction, int, int) = nullptr;
    bool ok = false;
    std::string err;
};

static const DriverApi& driver_api() {
    static DriverApi api = [] {
        DriverApi a;
        void* h = RTLD_DEFAULT;
        if (dlsym(h, "cuLaunchKernel") == nullptr) {
            h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
            if (h == nullptr) { a.err = "dlopen(libcuda.so.1) failed"; return a; }
        }
        auto need = [&](const char* name) {
            void* p = dlsym(h, name);
            if (p == nullptr) a.err += std::string(a.err.empty() ? "" : "; ") + "dlsym(" + name + ")";
            return p;
        };
        a.ModuleLoadData = reinterpret_cast<decltype(a.ModuleLoadData)>(need("cuModuleLoadData"));
        a.ModuleGetFunction = reinterpret_cast<decltype(a.ModuleGetFunction)>(need("cuModuleGetFunction"));
        a.LaunchKernel = reinterpret_cast<decltype(a.LaunchKernel)>(need("cuLaunchKernel"));
        a.CtxGetCurrent = reinterpret_cast<decltype(a.CtxGetCurrent)>(need("cuCtxGetCurrent"));
        a.GetErrorString = reinterpret_cast<decltype(a.GetErrorString)>(need("cuGetErrorString"));
        a.FuncSetAttribute = reinterpret_cast<decltype(a.FuncSetAttribute)>(need("cuFuncSetAttribute"));
        a.ok = a.ModuleLoadData && a.ModuleGetFunction && a.LaunchKernel &&
               a.CtxGetCurrent && a.GetErrorString && a.FuncSetAttribute;
        return a;
    }();
    return api;
}

static std::string cu_err(CUresult r) {
    const char* text = nullptr;
    if (driver_api().GetErrorString && driver_api().GetErrorString(r, &text) == CUDA_SUCCESS && text)
        return text;
    return "CUresult=" + std::to_string(static_cast<int>(r));
}

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
__device__ __forceinline__ lrx_c2 lrx_rot_mul(lrx_c2 u, lrx_c2 g) {
#if LRX_NS == 1
    return lrx_mul_xla(u, g);
#else
    return lrx_mul_cu(u, g);
#endif
}
__device__ __forceinline__ lrx_c2 lrx_rot_mul_conj(lrx_c2 l, lrx_c2 u) {
#if LRX_NS == 1
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
#elif LRX_MODE < 7
// Modes 2-5: one resident bank per row.  k-LEADING tiles (2, 3) hold element
// (k, r) at k*rows + r, so a block's load walks its RB rows fastest (coalesced
// over rows); k-MINOR tiles (4, 5) hold (r, k) at r*NK + k and walk k fastest.
// x and y may alias (in place): every element a block stores is one it loaded.
constexpr bool KLEAD = (LRX_MODE == 2 || LRX_MODE == 3);
constexpr bool CONV = (LRX_MODE == 2 || LRX_MODE == 4);

__device__ __forceinline__ long long lrx_elem(long long row, int k, long long rows) {
    return KLEAD ? (long long)k * rows + row : row * NK + k;
}

extern "C" __global__ void __launch_bounds__(256) lrx_kconv(
    const lrx_c2* x, const lrx_c2* __restrict__ kern, lrx_c2* y, RowGeo g) {
    extern __shared__ lrx_c2 sm[];
    const long long r0 = (long long)blockIdx.x * RB;
    using namespace cufftdx;
    for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
        const int k = KLEAD ? i / RB : i % NK, j = KLEAD ? i % RB : i / NK;
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
            const int k = KLEAD ? i / RB : i % NK, j = KLEAD ? i % RB : i / NK;
            const long long row = r0 + j;
            if (row < g.rows) {
                const long long kidx = (LRX_MODE == 2)
                    ? (long long)k * (g.m2 * g.m0) + ((row / g.m1) % g.m2) * g.m0 + row % g.m0
                    : ((row / g.m1) % g.m0) * NK + k;
                sm[j * SP + k] = lrx_mul(sm[j * SP + k], kern[kidx]);
            }
        }
        __syncthreads();
        transform3<fft_direction::forward>(sm);
    }
    for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
        const int k = KLEAD ? i / RB : i % NK, j = KLEAD ? i % RB : i / NK;
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
#else
// Modes 7 and 8 share the unfold load below.
// Mode 7: mode 2 on the full-k Green read from its raw parents.  Row r of the
// convolution is (pair, a, b) = (r / NS^2, (r % NS^2) / NS, r % NS) with pair =
// x*my + y; U[k, a, x, b, y] is stored spin-major.  When RB holds whole spin
// groups (the usual case) a block's load reads the NS*NS sources of a pair once
// and runs the spin action in registers for all NS*NS rows; when fewer rows fit
// (large k-grids) each bank loads its own row, with the same arithmetic.
constexpr int SS = NS * NS;
constexpr bool GROUPED = (RB % SS) == 0;

// The typed unfold of one (k, x, y) pair (symmetry_maps unfold_isdf_operator,
// axis-local pair_transpose arm): source row, both endpoint gathers, then
// (mph * G) * nph, a -1 source being an exact zero; and U_k.
__device__ __forceinline__ void lrx_unfold_pair(
    const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt, const UnfoldTab& t,
    int k, long long xx, long long yy, lrx_c2 (&g)[NS][NS], lrx_c2 (&u)[NS][NS]) {
    const lrx_c2* __restrict__ mph = reinterpret_cast<const lrx_c2*>(t.mph);
    const lrx_c2* __restrict__ nph = reinterpret_cast<const lrx_c2*>(t.nph);
    const lrx_c2* __restrict__ spin = reinterpret_cast<const lrx_c2*>(t.spin);
    const lrx_c2* src = (t.trs[k] ? gt : gp) + (long long)t.row[k] * t.ml * t.nl;
#pragma unroll
    for (int c = 0; c < NS; ++c) {
        const long long li = (long long)k * t.ml + xx * NS + c;
        const int ls = t.lsrc[li];
        const lrx_c2 mp = mph[li];
#pragma unroll
        for (int d = 0; d < NS; ++d) {
            const long long rj = (long long)k * t.nl + yy * NS + d;
            const int rs = t.rsrc[rj];
            lrx_c2 v = {0.0, 0.0};
            if (ls >= 0 && rs >= 0)
                v = lrx_mul_xla(lrx_mul_xla(mp, src[(long long)ls * t.nl + rs]), nph[rj]);
            g[c][d] = v;
        }
    }
#pragma unroll
    for (int a = 0; a < NS; ++a)
#pragma unroll
        for (int b = 0; b < NS; ++b) u[a][b] = spin[((long long)k * NS + a) * NS + b];
}

// Row a of U G U^dagger, accumulated exactly as the spin-rotate FFI does.
__device__ __forceinline__ void lrx_spin_row(const lrx_c2 (&u)[NS][NS], const lrx_c2 (&g)[NS][NS],
                                             int a, lrx_c2 (&out)[NS]) {
    lrx_c2 left[NS];
#pragma unroll
    for (int d = 0; d < NS; ++d) {
        lrx_c2 v = {0.0, 0.0};
#pragma unroll
        for (int c = 0; c < NS; ++c) {
            const lrx_c2 p = lrx_rot_mul(u[a][c], g[c][d]);
            v.x = __dadd_rn(v.x, p.x); v.y = __dadd_rn(v.y, p.y);
        }
        left[d] = v;
    }
#pragma unroll
    for (int b = 0; b < NS; ++b) {
        lrx_c2 v = {0.0, 0.0};
#pragma unroll
        for (int d = 0; d < NS; ++d) {
            const lrx_c2 p = lrx_rot_mul_conj(left[d], u[b][d]);
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
            lrx_c2 g[NS][NS], u[NS][NS];
            lrx_unfold_pair(gp, gt, t, k, xx, yy, g, u);
#pragma unroll
            for (int a = 0; a < NS; ++a) {
                lrx_c2 out[NS];
                lrx_spin_row(u, g, a, out);
#pragma unroll
                for (int b = 0; b < NS; ++b) sm[(jp * SS + a * NS + b) * SP + k] = out[b];
            }
        } else {
#pragma unroll
            for (int ab = 0; ab < SS; ++ab) sm[(jp * SS + ab) * SP + k] = {0.0, 0.0};
        }
    }
}

#if LRX_MODE == 7
extern "C" __global__ void __launch_bounds__(256) lrx_kconv(
    const lrx_c2* __restrict__ gp, const lrx_c2* __restrict__ gt,
    const lrx_c2* __restrict__ kern, lrx_c2* __restrict__ y, UnfoldTab t, double scale) {
    extern __shared__ lrx_c2 sm[];
    using namespace cufftdx;
    const long long mx = t.ml / NS, my = t.nl / NS, pairs = mx * my;
    const long long r0 = (long long)blockIdx.x * RB;
    if constexpr (GROUPED) {
        lrx_group_load<RB / SS>(gp, gt, t, r0 / SS, pairs, my, sm);
    } else {
        for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
            const int k = i / RB, j = i % RB;
            const long long r = r0 + j, pr = r / SS;
            lrx_c2 v = {0.0, 0.0};
            if (pr < pairs) {
                const long long xx = pr / my, yy = pr - xx * my;
                const int a = (int)((r % SS) / NS), b = (int)(r % NS);
                lrx_c2 g[NS][NS], u[NS][NS], out[NS];
                lrx_unfold_pair(gp, gt, t, k, xx, yy, g, u);
                lrx_spin_row(u, g, a, out);
                v = out[b];
            }
            sm[j * SP + k] = v;
        }
    }
    __syncthreads();
    transform3<fft_direction::inverse>(sm);
    for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
        const int k = i / RB, j = i % RB;
        const long long pr = (r0 + j) / SS;
        if (pr < pairs) {
            const long long xx = pr / my, yy = pr - xx * my;
            sm[j * SP + k] = lrx_mul(sm[j * SP + k], kern[((long long)k * mx + xx) * my + yy]);
        }
    }
    __syncthreads();
    transform3<fft_direction::forward>(sm);
    for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
        const int k = i / RB, j = i % RB;
        const long long r = r0 + j, pr = r / SS;
        const long long ko = lrx_out_row(t, k);
        if (pr < pairs && ko >= 0) {
            const long long xx = pr / my, yy = pr - xx * my;
            const int a = (int)((r % SS) / NS), b = (int)(r % NS);
            const lrx_c2 v = sm[j * SP + k];
            y[((ko * NS + a) * mx + xx) * t.nl + b * my + yy] = {v.x * scale, v.y * scale};
        }
    }
}
#else
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
constexpr int ROWS = LRX_ROWS;         // staging capacity: the table's occupied rows, rounded up to 8
constexpr bool STAGE = LRX_STAGE;      // a (ROWS, NC) staging block per plane fits beside the planes

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
template <int N, int N1, int N2, bool FIRST>
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
            if (src != nullptr) z = src[p * es];              // the staged row (es = 1)
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

// LRX_PB planes per block (small planes share a block, so a pass has enough
// lines); LRX_RB blocks per SM the shared planes allow (capped at 2), so the
// register cap lets them all in.  Blocks are persistent (the host caps the grid
// at the resident count).  With STAGE the next group's occupied cells are
// gathered asynchronously (lrx_async, cp.async) into the staging blocks while
// this group's passes run; the first row pass reads them from there.
extern "C" __global__ void __launch_bounds__(LRX_THREADS, LRX_RB) lrx_kconv(
    const lrx_c2* __restrict__ fin, lrx_c2* __restrict__ yout, PlaneGather g) {
    extern __shared__ lrx_c2 buf[];                  // PB planes of (NB, LD), then PB staging (ROWS, NC)
    constexpr int PB = LRX_PB, PS = NB * LD, SS = ROWS * NC;
    lrx_c2* stg = buf + PB * PS;
    __shared__ unsigned char live[NB];
    __shared__ int rowb[NB];
    __shared__ long long foff[2][PB];
    const int rows = static_cast<int>(g.rows), rn = rows * NC;   // rows <= ROWS
    for (int b = threadIdx.x; b < NB; b += blockDim.x) live[b] = 0;
    __syncthreads();
    for (int r = threadIdx.x; r < rows; r += blockDim.x) { rowb[r] = g.row_of[r]; live[g.row_of[r]] = 1; }
    // Output plane (a, j, r) of Y (A, n_pg, inner) reads F plane (a, start + j, r)
    // of F (A, s_len, inner): the slab F[:, start:start+n_pg] in place.
    long long st = *g.start;
    st = st < 0 ? 0 : (st > g.s_len - g.n_pg ? g.s_len - g.n_pg : st);
    const long long groups = (g.planes + PB - 1) / PB;
    // The cylinder offsets of group gi's planes into foff[fs], then one async
    // gather of their occupied cells (zeros implicit) into staging or the planes.
    auto issue = [&](long long gi, int fs) {
        const long long p0 = gi * PB;
        const int np = g.planes - p0 < PB ? static_cast<int>(g.planes - p0) : PB;
        if (threadIdx.x < np) {
            const long long plane = p0 + threadIdx.x;
            const long long a = plane / (g.n_pg * g.inner), rem = plane - a * g.n_pg * g.inner;
            foff[fs][threadIdx.x] = ((a * g.s_len + st) * g.inner + rem) * g.n_col;
        }
        __syncthreads();
        for (int t = threadIdx.x; t < np * rn; t += blockDim.x) {
            const int q = t / rn, u = t - q * rn, col = g.gidx[u];
            lrx_c2* dst = STAGE ? stg + q * SS + u : buf + q * PS + rowb[u / NC] * LD + (u - (u / NC) * NC);
            lrx_async::cell16(dst, fin + foff[fs][q] + (col >= 0 ? col : 0), col >= 0);
        }
        lrx_async::commit();
    };
    int fs = 0;
    if (static_cast<long long>(blockIdx.x) < groups) issue(blockIdx.x, fs);
    for (long long gi = blockIdx.x; gi < groups; gi += gridDim.x) {
        const long long p0 = gi * PB;
        const int np = g.planes - p0 < PB ? static_cast<int>(g.planes - p0) : PB;
        lrx_async::wait_all();
        __syncthreads();
        // Row FFTs (along c) on the occupied rows only; with STAGE the first
        // factor pass reads the staged rows and writes the planes.
        for (int l = threadIdx.x; l < np * rows * C2; l += blockDim.x) {
            const int q = l / (rows * C2), u = l - q * rows * C2, r = u / C2;
            pfa_line<NC, C1, C2, true>(buf + q * PS + rowb[r] * LD, 1, u % C2, nullptr,
                                       STAGE ? stg + q * SS + r * NC : nullptr);
        }
        __syncthreads();
        const long long gn = gi + gridDim.x;
        if (STAGE && gn < groups) { fs ^= 1; issue(gn, fs); }   // staging is free: prefetch
        if constexpr (C2 > 1) {
            for (int l = threadIdx.x; l < np * rows * C1; l += blockDim.x) {
                const int q = l / (rows * C1), u = l - q * rows * C1;
                pfa_line<NC, C1, C2, false>(buf + q * PS + rowb[u / C1] * LD, 1, u % C1, nullptr);
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
        if (!STAGE && gn < groups) { fs ^= 1; issue(gn, fs); }  // the planes are free
    }
}
)__lrx__";

// ---------------------------------------------------------------------------
//  Build and cache
// ---------------------------------------------------------------------------
struct Built { CUfunction fn = nullptr; int rb = 1; int smem = 0; double compile_ms = 0.0; int threads = kThreads;
               long long grid_cap = 0; };   // grid_cap: mode 10's resident blocks (0 = none)
using Key = std::tuple<CUcontext, int, int, int, int, int, int>;  // ctx, mode, nkx, nky, nkz, ns, f32
static std::mutex g_mu;
static std::map<Key, Built> g_cache;
static std::map<Key, std::string> g_fail;

static bool exists(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0; }

// The CUDA toolkit include next to the loaded libnvrtc (lib64 or targets/<arch>/lib).
static std::string toolkit_include(std::string* why) {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&nvrtcVersion), &info) || !info.dli_fname) {
        *why = "dladdr(nvrtcVersion) found no library path"; return "";
    }
    std::string lib(info.dli_fname);
    lib = lib.substr(0, lib.find_last_of('/'));
    for (const char* rel : {"/../include", "/../../include"}) {
        const std::string inc = lib + rel;
        if (exists(inc + "/cccl/cuda/std/type_traits") || exists(inc + "/cuda/std/type_traits")) return inc;
    }
    *why = "no include/cccl/cuda/std/type_traits beside the loaded libnvrtc (" + lib + ")";
    return "";
}

// The loaded libnvrtc's real path: its file name carries the patch level
// (libnvrtc.so.13.2.78), which nvrtcVersion's major.minor does not.
static std::string nvrtc_library_realpath() {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&nvrtcVersion), &info) || !info.dli_fname) return "";
    char buf[4096];
    return realpath(info.dli_fname, buf) ? std::string(buf) : std::string(info.dli_fname);
}

// The nvidia-mathdx wheel's dist-info directory name(s) beside `root`
// (<site>/nvidia/mathdx -> <site>/nvidia_mathdx-<version>.dist-info): one
// listing of one directory; "" when the headers are not a wheel install.
static std::string mathdx_dist_info(const std::string& root) {
    const std::string site = root + "/../..";
    DIR* d = opendir(site.c_str());
    if (!d) return "";
    std::vector<std::string> names;
    while (dirent* e = readdir(d)) {
        const std::string n(e->d_name);
        if (n.rfind("nvidia_mathdx-", 0) == 0 && n.size() > 10 && n.substr(n.size() - 10) == ".dist-info")
            names.push_back(n);
    }
    closedir(d);
    std::sort(names.begin(), names.end());
    std::string out;
    for (const auto& n : names) out += n + ";";
    return out;
}

// A cubin is an ELF image; anything else on disk is not one of ours.
static bool is_elf(const std::vector<char>& b) {
    return b.size() > 4 && b[0] == 0x7f && b[1] == 'E' && b[2] == 'L' && b[3] == 'F';
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::ostringstream os; os << f.rdbuf();
    return os.str();
}

static uint64_t fnv1a(std::string_view data, uint64_t h = 1469598103934665603ULL) {
    for (unsigned char c : data) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

static std::string hex16(uint64_t v) {
    char b[17]; std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v)); return b;
}

// mkdir -p; true when the directory exists afterwards.
static bool make_dirs(const std::string& dir) {
    if (dir.empty()) return false;
    std::string cur;
    std::stringstream ss(dir);
    std::string part;
    if (dir[0] == '/') cur = "/";
    while (std::getline(ss, part, '/')) {
        if (part.empty()) continue;
        cur += part + "/";
        if (mkdir(cur.c_str(), 0775) != 0 && errno != EEXIST) return false;
    }
    return exists(dir);
}

// On-disk image: "LRXKCONV1\n" + 16 hex key + 16 hex payload hash + '\n' + cubin.
static constexpr std::string_view kMagic = "LRXKCONV1\n";

static bool disk_load(const std::string& path, const std::string& key_hex, std::vector<char>* cubin) {
    const std::string blob = read_file(path);
    const size_t head = kMagic.size() + 33;
    if (blob.size() <= head || blob.compare(0, kMagic.size(), kMagic) != 0) return false;
    if (blob.compare(kMagic.size(), 16, key_hex) != 0) return false;
    const std::string_view payload(blob.data() + head, blob.size() - head);
    if (blob.compare(kMagic.size() + 16, 16, hex16(fnv1a(payload))) != 0) return false;
    cubin->assign(payload.begin(), payload.end());
    return true;
}

// Unique temporary + rename: concurrent ranks each publish a whole file.
static bool disk_store(const std::string& dir, const std::string& path, const std::string& key_hex,
                       const std::vector<char>& cubin) {
    if (!make_dirs(dir)) return false;
    std::random_device rd;
    const std::string tmp = path + ".tmp." + std::to_string(getpid()) + "." + hex16(rd() ^ (uint64_t(rd()) << 32));
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        const std::string_view payload(cubin.data(), cubin.size());
        f.write(kMagic.data(), kMagic.size());
        f << key_hex << hex16(fnv1a(payload)) << '\n';
        f.write(cubin.data(), static_cast<std::streamsize>(cubin.size()));
        if (!f.good()) { f.close(); unlink(tmp.c_str()); return false; }
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) { unlink(tmp.c_str()); return false; }
    return true;
}

static ffi::Error build(int mode, int nkx, int nky, int nkz, int ns, bool f32,
                        std::string_view mathdx_root, std::string_view cubin_dir, const Built** out) {
    const DriverApi& api = driver_api();
    if (!api.ok) return fail("driver-api resolve", api.err);
    CUcontext ctx = nullptr;
    CUresult cr = api.CtxGetCurrent(&ctx);
    if (cr != CUDA_SUCCESS || ctx == nullptr) {
        LRX_CUDA_CHECK(cudaFree(nullptr), "context bind (cudaFree(0))");
        cr = api.CtxGetCurrent(&ctx);
        if (cr != CUDA_SUCCESS || ctx == nullptr) return fail("cuCtxGetCurrent", cu_err(cr));
    }
    const Key key{ctx, mode, nkx, nky, nkz, ns, f32 ? 1 : 0};
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
    long long rb = std::min<long long>(rows_max, (pair ? kSmemBudget : kSmemBudget1) / row_bytes);
    // Mode 8 sums the Lorentz blocks across a pair's spin rows, so it needs a
    // whole spin group resident: reach for the opt-in shared memory first.
    if (rb < 1 || (mode == 8 && rb < ns * ns)) rb = std::min<long long>(rows_max, smem_optin / row_bytes);
    if ((mode == 7 || mode == 8) && rb >= ns * ns) rb -= rb % (ns * ns);  // whole spin groups: the grouped load
    // (fewer rows than one spin group: mode 7 loads per bank, as mode 2 would fit)
    long long plane_minb = 1;                          // mode 10: blocks per SM (LRX_RB)
    long long plane_static = 0;                        // mode 10: its static tables, bytes
    long long plane_stage = 0;                         // mode 10: one plane's staging block, bytes (0 = off)
    // mode 10 packs (c1, staging rows) into ns; the handler rounds the table's
    // occupied rows up to a multiple of 8 so supports of similar size share a cubin.
    const int plane_c1 = ns & 255, plane_rows = ns >> 8;
    if (mode == 10) {                                  // whole (n_b, n_c|1) planes per block
        row_bytes = 16LL * nkx * (nky | 1);
        // The kernel's static tables live[n_b] + rowb[n_b] (int) + foff[2][PB] (long long)
        // share the block's opt-in budget with the dynamic planes; +16 B alignment
        // slack.  ffi.fft.plane_resident_bytes is the same bound (PB = 1, no staging).
        auto stat = [&](long long pb) { return 5LL * nkx + 16 * pb + 16; };
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
    if (mode == 8 && rb < ns * ns) {
        std::ostringstream os;
        os << "GATE mathdx-kconv-lorentz-residency: got k-grid (" << nkx << "," << nky << "," << nkz
           << ") with ns=" << ns << ", whose " << ns * ns << " spin rows need " << ns * ns << "*16*(nk|1)="
           << ns * ns * row_bytes << " B; want <= " << smem_optin << " B of opt-in shared memory on this "
              "device; why: the Lorentz vertex sum mixes a pair's spin rows in R space, so one block holds "
              "all of them; fix: a smaller k-grid (the family has no out-of-core arm)";
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
    const std::string inc = root + "/include", cutlass = root + "/external/cutlass/include";
    if (!exists(inc + "/cufftdx.hpp")) {
        return sticky("GATE mathdx-headers",
                      "got no cufftdx.hpp under " + inc + "; want the nvidia-mathdx wheel; fix: "
                      "pip install nvidia-mathdx", ffi::ErrorCode::kFailedPrecondition);
    }
    // Options that decide the cubin (include paths do not: two installs of one
    // wheel version compile the same image).
    std::vector<std::string> defs = {
        "--std=c++17", "--device-as-default-execution-space",
        "--gpu-architecture=sm_" + std::to_string(cc_major) + std::to_string(cc_minor),
        "-DLRX_MODE=" + std::to_string(mode), "-DLRX_NX=" + std::to_string(nkx),
        "-DLRX_NY=" + std::to_string(nky), "-DLRX_NZ=" + std::to_string(nkz),
        "-DLRX_NS=" + std::to_string(mode == 10 ? plane_c1 : ns),
        "-DLRX_RB=" + std::to_string(mode == 10 ? plane_minb : rb),
        "-DLRX_F32=" + std::string(f32 ? "1" : "0"),
        "-DLRX_SM=" + std::to_string(cc_major * 100 + cc_minor * 10)};
    // Mode 10: planes per block; a block that has its SM alone runs 512 threads.
    const int plane_threads = mode == 10 && plane_minb == 1 ? 512 : kThreads;
    if (mode == 10) {
        defs.push_back("-DLRX_PB=" + std::to_string(rb));
        defs.push_back("-DLRX_ROWS=" + std::to_string(plane_rows));
        defs.push_back("-DLRX_STAGE=" + std::string(plane_stage > 0 ? "1" : "0"));
        defs.push_back("-DLRX_THREADS=" + std::to_string(plane_threads));
    }
    std::vector<std::string> o = defs;
    for (const std::string& d : {"-I" + inc, "-I" + cutlass, "-I" + cuda_inc, "-I" + cuda_inc + "/cccl"})
        o.push_back(d);

    // Disk cache key: source, deciding options and the toolchain (file header).
    int nv_major = 0, nv_minor = 0;
    nvrtcVersion(&nv_major, &nv_minor);
    const char* src = mode == 10 ? kPlaneSrc : kSrc;
    namespace ag = lorrax_ffi::async_gather;
    const bool uses_async = std::string_view(src).find(ag::kHeaderName) != std::string_view::npos;
    uint64_t h = fnv1a(src);
    if (uses_async) h = fnv1a(ag::kHeaderSrc, fnv1a("\x1d", h));   // the embedded header decides the image too
    for (const auto& d : defs) h = fnv1a(d, fnv1a("\x1f", h));
    const std::string cccl = exists(cuda_inc + "/cccl/cuda/std/__cccl/version.h")
        ? cuda_inc + "/cccl/cuda/std/__cccl/version.h" : cuda_inc + "/cuda/std/__cccl/version.h";
    std::string missing;
    for (const std::string& f : {inc + "/cufftdx/cufftdx_version.hpp", inc + "/commondx/commondx_version.hpp",
                                 cutlass + "/cutlass/version.h", cccl}) {
        const std::string text = read_file(f);
        if (text.empty()) missing += (missing.empty() ? "" : ", ") + f;
        h = fnv1a(text, fnv1a("\x1e" + f.substr(f.find_last_of('/') + 1), h));
    }
    h = fnv1a("nvrtc" + std::to_string(nv_major) + "." + std::to_string(nv_minor) + "@" +
              nvrtc_library_realpath(), h);
    h = fnv1a("mathdx-dist:" + mathdx_dist_info(root), h);
    const std::string key_hex = hex16(h);
    const std::string dir(missing.empty() ? std::string(cubin_dir) : std::string());
    if (!missing.empty() && !std::string(cubin_dir).empty() && (mklpin::announce_here() || log_enabled()))
        std::fprintf(stderr, "[kconv_mathdx] disk cubin cache OFF for this build: empty version header(s) %s "
                     "would drop out of the key\n", missing.c_str());
    std::string path;
    if (!dir.empty()) {
        std::ostringstream name;
        name << dir << "/kconv_m" << mode << "_" << nkx << "x" << nky << "x" << nkz << "_ns" << ns
             << (f32 ? "_c64" : "") << "_sm" << cc_major << cc_minor << "_" << key_hex << ".cubin";
        path = name.str();
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto compile = [&](std::vector<char>* image, std::string* where, std::string* err) {
        std::vector<const char*> opts;
        for (auto& x : o) opts.push_back(x.c_str());
        nvrtcProgram prog = nullptr;
        const char* hdr_src[] = {ag::kHeaderSrc};
        const char* hdr_name[] = {ag::kHeaderName};
        nvrtcResult nr = nvrtcCreateProgram(&prog, src, "lrx_kconv_mathdx.cu", uses_async ? 1 : 0,
                                            uses_async ? hdr_src : nullptr, uses_async ? hdr_name : nullptr);
        if (nr != NVRTC_SUCCESS) { *where = "nvrtcCreateProgram"; *err = nvrtcGetErrorString(nr); return false; }
        nr = nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
        if (nr != NVRTC_SUCCESS) {
            size_t n = 0; std::string log;
            if (nvrtcGetProgramLogSize(prog, &n) == NVRTC_SUCCESS && n > 1) { log.resize(n); nvrtcGetProgramLog(prog, &log[0]); }
            nvrtcDestroyProgram(&prog);
            *where = "nvrtcCompileProgram";
            *err = std::string(nvrtcGetErrorString(nr)) + " -- " + log.substr(0, 4000);
            return false;
        }
        size_t n = 0;
        if (nvrtcGetCUBINSize(prog, &n) != NVRTC_SUCCESS || n == 0) {
            nvrtcDestroyProgram(&prog); *where = "nvrtcGetCUBINSize"; *err = "empty cubin"; return false;
        }
        image->assign(n, 0);
        nr = nvrtcGetCUBIN(prog, image->data());
        nvrtcDestroyProgram(&prog);
        if (nr != NVRTC_SUCCESS || !is_elf(*image)) {
            *where = "nvrtcGetCUBIN";
            *err = nr != NVRTC_SUCCESS ? nvrtcGetErrorString(nr) : "image is not an ELF cubin";
            return false;
        }
        return true;
    };
    std::vector<char> cubin;
    // A disk image must frame, hash AND be an ELF; one the driver then refuses
    // is deleted and rebuilt once below.
    bool from_disk = !path.empty() && disk_load(path, key_hex, &cubin) && is_elf(cubin);
    bool stored = false, rebuilt_bad = false;
    std::string where, err;
    if (!from_disk) {
        if (!compile(&cubin, &where, &err)) return sticky(where.c_str(), err);
        if (!path.empty()) stored = disk_store(dir, path, key_hex, cubin);
    }
    CUmodule module = nullptr;
    cr = api.ModuleLoadData(&module, cubin.data());
    if (cr != CUDA_SUCCESS && from_disk) {
        unlink(path.c_str());
        from_disk = false; rebuilt_bad = true;
        if (!compile(&cubin, &where, &err)) return sticky(where.c_str(), err);
        stored = disk_store(dir, path, key_hex, cubin);
        cr = api.ModuleLoadData(&module, cubin.data());
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (cr != CUDA_SUCCESS) return sticky("cuModuleLoadData", cu_err(cr));
    Built b;
    cr = api.ModuleGetFunction(&b.fn, module, "lrx_kconv");
    if (cr != CUDA_SUCCESS) return sticky("cuModuleGetFunction", cu_err(cr));
    b.rb = static_cast<int>(rb);
    b.threads = plane_threads;
    b.smem = static_cast<int>(rb * (row_bytes + plane_stage));
    if (mode == 10) {                                  // persistent blocks: the resident count
        int sms = 0;
        LRX_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SM count");
        b.grid_cap = static_cast<long long>(sms) * plane_minb;
    }
    b.compile_ms = ms;
    // Mode 10 always sets the dynamic limit: its static tables count against the
    // 48 KiB default too, so a plane just under 48 KiB would fail at launch.
    if (b.smem > 49152 || mode == 10) {
        cr = api.FuncSetAttribute(b.fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, b.smem);
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
    int64_t nky, int64_t nkz, double scale, std::string_view mathdx_root, std::string_view cubin_dir) {
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
          U->dimensions()[1] == ns && U->dimensions()[2] == ml / ns && U->dimensions()[3] == ns &&
          U->dimensions()[4] == nl / ns))
        return bad("want c128 Gp=Gt (np,ml,nl); s32 row,trs,kout (nk), lsrc (nk,ml), rsrc (nk,nl); c128 "
                   "mph (nk,ml), nph (nk,nl), spin (nk,ns,ns), V (nk,ml/ns,nl/ns); U "
                   "(n_out,ns,ml/ns,ns,nl/ns), n_out = nk without kout");
    const int64_t pairs = (ml / ns) * (nl / ns);
    if (pairs == 0) return ffi::Error::Success();
    const Built* k = nullptr;
    ffi::Error e = build(7, static_cast<int>(nkx), static_cast<int>(nky), static_cast<int>(nkz),
                         static_cast<int>(ns), false, mathdx_root, cubin_dir, &k);
    if (!e.success()) return e;
    UnfoldTab t{static_cast<const int*>(row.untyped_data()), static_cast<const int*>(trs.untyped_data()),
                static_cast<const int*>(lsrc.untyped_data()), static_cast<const int*>(rsrc.untyped_data()),
                static_cast<const double*>(mph.untyped_data()), static_cast<const double*>(nph.untyped_data()),
                static_cast<const double*>(spin.untyped_data()), ml, nl,
                kout ? static_cast<const int*>(kout->untyped_data()) : nullptr};
    const void* gpp = Gp.untyped_data();
    const void* gtp = Gt.untyped_data();
    const void* vp = V.untyped_data();
    void* up = U->untyped_data();
    double sc = scale;
    void* args[] = {(void*)&gpp, (void*)&gtp, (void*)&vp, (void*)&up, (void*)&t, (void*)&sc};
    const long long rows = pairs * ns * ns;
    const long long blocks = (rows + k->rb - 1) / k->rb;
    if (blocks > 2147483647LL) return bad("grid.x overflow");
    CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, kThreads, 1, 1,
                                            static_cast<unsigned>(k->smem),
                                            reinterpret_cast<CUstream>(stream), args, nullptr);
    if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
    return ffi::Error::Success();
}

static ffi::Error KleadUnfoldRowsConv(
    cudaStream_t stream, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    ffi::AnyBuffer kout, ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky,
    int64_t nkz, double scale, std::string_view mathdx_root, std::string_view cubin_dir) {
    return KleadUnfoldImpl(stream, Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, &kout, V, U, nkx, nky,
                           nkz, scale, mathdx_root, cubin_dir);
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
static ffi::Error KleadLorentzImpl(
    cudaStream_t stream, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    const ffi::AnyBuffer* kout, ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx,
    int64_t nky, int64_t nkz, double scale_g, double scale_f, double mult,
    ffi::Span<const int64_t> perm_l, ffi::Span<const int64_t> phase_l, ffi::Span<const int64_t> perm_r,
    ffi::Span<const int64_t> phase_r, std::string_view mathdx_root, std::string_view cubin_dir) {
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
                kout ? static_cast<const int*>(kout->untyped_data()) : nullptr};
    const void* gpp = Gp.untyped_data();
    const void* gtp = Gt.untyped_data();
    const void* vp = V.untyped_data();
    void* up = U->untyped_data();
    void* args[] = {(void*)&gpp, (void*)&gtp, (void*)&vp, (void*)&up, (void*)&t, (void*)&v};
    const long long rows = pairs * ns * ns;
    const long long blocks = (rows + k->rb - 1) / k->rb;
    if (blocks > 2147483647LL) return bad("grid.x overflow");
    CUresult cr = driver_api().LaunchKernel(k->fn, static_cast<unsigned>(blocks), 1, 1, kThreads, 1, 1,
                                            static_cast<unsigned>(k->smem),
                                            reinterpret_cast<CUstream>(stream), args, nullptr);
    if (cr != CUDA_SUCCESS) return fail("cuLaunchKernel", cu_err(cr));
    return ffi::Error::Success();
}

static ffi::Error KleadLorentzRowsConv(
    cudaStream_t stream, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    ffi::AnyBuffer kout, ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky,
    int64_t nkz, double scale_g, double scale_f, double mult, ffi::Span<const int64_t> perm_l,
    ffi::Span<const int64_t> phase_l, ffi::Span<const int64_t> perm_r, ffi::Span<const int64_t> phase_r,
    std::string_view mathdx_root, std::string_view cubin_dir) {
    return KleadLorentzImpl(stream, Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, &kout, V, U, nkx, nky, nkz,
                            scale_g, scale_f, mult, perm_l, phase_l, perm_r, phase_r, mathdx_root, cubin_dir);
}
static ffi::Error KleadLorentzConv(
    cudaStream_t stream, ffi::AnyBuffer Gp, ffi::AnyBuffer Gt, ffi::AnyBuffer row, ffi::AnyBuffer trs,
    ffi::AnyBuffer lsrc, ffi::AnyBuffer rsrc, ffi::AnyBuffer mph, ffi::AnyBuffer nph, ffi::AnyBuffer spin,
    ffi::AnyBuffer V, ffi::Result<ffi::AnyBuffer> U, int64_t nkx, int64_t nky, int64_t nkz,
    double scale_g, double scale_f, double mult, ffi::Span<const int64_t> perm_l,
    ffi::Span<const int64_t> phase_l, ffi::Span<const int64_t> perm_r, ffi::Span<const int64_t> phase_r,
    std::string_view mathdx_root, std::string_view cubin_dir) {
    return KleadLorentzImpl(stream, Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, nullptr, V, U, nkx, nky,
                            nkz, scale_g, scale_f, mult, perm_l, phase_l, perm_r, phase_r, mathdx_root,
                            cubin_dir);
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
    // ns carries (c1, staging rows): the occupied rows rounded up to 8, at most n_b.
    const int64_t stage_rows = std::min<int64_t>(nb, (gd[0] + 7) / 8 * 8);
    ffi::Error e = build(10, static_cast<int>(nb), static_cast<int>(nc), static_cast<int>(b1),
                         static_cast<int>(c1 | (stage_rows << 8)), false, mathdx_root, cubin_dir, &k);
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
    KConvMathdxKleadLorentzRowsCudaFfi, lorrax_ffi::kconv_mathdx::KleadLorentzRowsConv,
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
