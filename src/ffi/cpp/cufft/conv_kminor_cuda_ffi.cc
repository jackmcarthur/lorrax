// conv_kminor_cuda_ffi.cc — the k-MINOR FUSED CONVOLUTION handler, CUDA leg.
//
//   U[b,m,n,t,s,:] = scale · FFT_k( IFFT_k( T[b,m,n,t,s,:] ) · W[m,n,:] )
//
// ONE kernel.  One read of T, one read of W, one write of U — the whole
// ifft·multiply·fft chain plus the decode-layout permutation, on the
// contiguous k-minor tile the caller already holds.  ZERO layout change on
// the caller's side.
//
// ---------------------------------------------------------------------------
// WHY THIS EXISTS (measured; do not re-derive it from first principles)
// ---------------------------------------------------------------------------
// The BSE ladder-W rung computes exactly the expression above on
// T = (nb, μ, ν, t, s, nk), k MINOR.  Two independent measurements said the
// existing engines cannot serve it:
//
//   O7 (evidence/opt_fftffi/RESULTS.txt) — routing the rung to the flat-k
//     cuFFT handler needs a k-LEADING tile, and cufftPlanMany64(istride=T,
//     idist=1) degrades with the batch stride, which here IS the μ·ν·s² tile:
//     fused/XLA = 0.83 at nk=9 but 1.61 at nk=64 and 4.00 at nk=216.  The
//     strided-plan arm is dead above nk≈16.
//
//   O9 (evidence/opt_kernels/RESULTS.txt) — the XLA emission of this chain is
//     8.1 HBM passes over the T tile (pre-FFT transpose 1 | ifft 2 radix + 1
//     'ortho' scal_kernel_val | multiply 1.12 | fft 2 radix | post-FFT
//     transpose 1) where a fused kernel needs 1.12.  At the gnppm fixture
//     (n_rmu=399, nk=9, ns=2, nb=1) that chain is 3637 µs = 74.3% of the
//     4897 µs the whole ladder matvec spends on the GPU, and the two
//     chain-sized transposes alone are 1864 µs at 246–975 GB/s.
//
// The transform itself is NOT the expensive part: nk is 9…216 points, i.e.
// 144 B…3.4 KB per row, and there are ~6.4·10⁵ rows at the fixture.  This is a
// huge batch of TINY transforms, and the roofline (O9 §C) puts the binding
// constraint on memory, not FLOPs.  So the kernel is written to move each row
// exactly once and do everything while it is resident.
//
// ---------------------------------------------------------------------------
// GENERIC BY CONSTRUCTION (owner ask: kernels reusable by other routines)
// ---------------------------------------------------------------------------
// Nothing here knows about the BSE.  The contract is
//
//   T : (d0, d1, d2, d3, d4, nk) c128, contiguous, nk minor-most
//   W : (d1, d2, nk)             c128, contiguous, broadcast over d0/d3/d4
//   U : out_layout=0 → shape(T)                  (aliasable, in-place)
//       out_layout=1 → (d0, nk, d3, d1, d4, d2)  (the STORE permutation)
//
// i.e. "an ifft·multiply·fft over a minor-most k axis, with the multiplicand
// broadcast over three of the five leading axes".  Any consumer that can name
// its leading axes in that order — the Σ gw_conv family included, by merging
// its free axes into d0/d3 — can call it.  The BSE rung is the first caller,
// not a special case: d0=n_trial, d1=μ, d2=ν, d3=t, d4=s.
//
// out_layout=1 is `contract_bands_block_reshard(extra="leading")`'s canonical
// O layout (b, k, t, μ, s, ν).  Emitting it from the STORE deletes the
// production `jnp.transpose(U, (0,5,3,1,4,2))` — 1488 µs/matvec, 30% of the
// whole ladder matvec — instead of moving it.
//
// WHY W AND NOT W_q.  The handler takes W ALREADY inverse-transformed
// (`bse_feast.ensure_W_R` → `bse_densify.make_w_densifier` builds
// W_R = ifftn(W_q, axes=k, norm='ortho') ONCE PER SOLVE and caches it; the
// tile is 22.9 MB at the fixture).  Transforming W inside the handler would
// redo that 22.9 MB transform on EVERY matvec — hundreds of times per solve —
// to save a call the caller already made.  The alternative composes worse, so
// it is not offered: this handler multiplies, it does not transform W.
// `scale` therefore carries BOTH remaining norm factors (1/√nk from the
// inverse, 1/√nk from the forward, at norm='ortho') as ONE constant, folded
// into the last forward pass.  That also deletes the two `scal_kernel_val`
// passes XLA emits for cuFFT's missing 'ortho' normalisation (271 µs/matvec).
// Value-level identical to the decomposed chain up to reassociation; gated at
// ~1e-15, never claimed bit-exact — the same class as the gw_conv scale-fold.
//
// ---------------------------------------------------------------------------
// THE TRANSFORM: a DIRECT per-axis DFT, and it is SIZE-AGNOSTIC ON PURPOSE
// ---------------------------------------------------------------------------
// THERE IS NO RADIX IN THIS FILE.  No butterflies, no factorisation, no
// per-size code path, no per-size compilation.  Each of the three axes is a
// DIRECT DFT of its own runtime length against a precomputed twiddle ring:
//
//     y[m] = Σ_j x[j] · W_len^(±m·j),   W_len = exp(-2πi/len)
//
// with `len` = nkx / nky / nkz read from the call's ATTRIBUTES.  The same
// instructions serve 1, 2, 3, 7, 11, 13, 24 — primes included — because
// nothing about the loop knows what `len` factors into.  Any (nkx,nky,nkz)
// the caller names is executed by the same kernel; the only bound is
// residency (below), and it is checked and REFUSED BY NAME, never guessed.
//
// WHY A RING AND NOT AN n×n TWIDDLE MATRIX.  The matrix form is the obvious
// spelling of a direct DFT and it is strictly worse here: it costs len²
// shared-memory words instead of len, and this kernel is SHARED-MEMORY
// BANDWIDTH BOUND (measured: 10–18 TB/s against an A100's ~19.5 TB/s roof),
// so spending shared traffic on twiddles is spending exactly the resource
// that binds.  The ring stores len entries and the inner loop walks the
// exponent with `m += mid; if (m >= len) m -= len` — one add and one
// predicated subtract per term, no multiply, no modulo, and the same FLOP
// count as the matrix.  Same arithmetic, len storage instead of len².
//
// WHY NOT O(n log n).  Because the FLOPs are not what the kernel waits on.
// The roofline for this operator is bytes: at the campaign geometry the whole
// chain moves 183 MB per application and computes 5.6 GFLOP, and the measured
// kernel sits at 400–570 GB/s of HBM while the fp64 pipes idle.  A radix
// ladder would cut an O(len²) term that is already free and would buy it with
// per-size code — the one thing the operator must not have.  If a deck ever
// arrives whose axis length makes the DFT term bind (it would take len in the
// hundreds, far above anything a k-grid reaches), `lrx_pass` is the single
// seam to put a factorisation behind, and it would go behind the SAME runtime
// signature.
//
// WHY NOT cuFFTDx.  It is a CUDA-C++ TEMPLATE library: the transform size is
// a template parameter, so covering a runtime (nkx,nky,nkz) means compiling a
// specialisation per size — the combinatorial per-size family this design
// exists to avoid — and it needs a real nvcc compile of a TU that instantiates
// it.  The MathDx headers also ship in neither the Perlmutter jax Shifter
// image nor any module on this system (checked), and vendoring them is a
// licence question nobody has answered.  Two independent reasons; the
// generality one is the one that would still hold if the headers appeared.
//
// THE ONE BOUND, and it is about RESIDENCY, not about size class: the fused
// chain needs the whole k-row live between the inverse and forward halves —
// that is what makes it one kernel and one HBM round trip instead of eight.
// So a row must fit in a block's shared memory.  The limit is DERIVED from
// the device at run time (see plan_launch) rather than assumed, and a row
// over it is refused with the number, the device's own maximum, and the
// sibling handler to use instead.  MEASURED on an A100 (dynamic shared max
// 166 912 B): any k-grid whose PRODUCT is ≤ ~10 350 points is served — every
// axis extent in [1,24] individually, every mixed radix, every prime, and
// every (nkx,nky,nkz) in [1,24]³ up to that product; 24³ = 13 824 is the one
// corner of that box that does not fit, and it refuses with the arithmetic.
// The exact number depends on the twiddle rings (n0+n1+n2) and is computed,
// not tabulated — the refusal quotes it.
//
// ---------------------------------------------------------------------------
// KERNEL SHAPE
// ---------------------------------------------------------------------------
// A "row" is one (d0,d1,d2,d3,d4) index: nk contiguous complex128 values.
// R = d0·d1·d2·d3·d4 rows (6.4·10⁵ at the fixture).
//
//   block = (TPR, RB)      RB rows per block, TPR threads per row
//   each thread owns EPT = ceil(nk/TPR) elements of its row (k = lane + e·TPR)
//   the block's RB rows live in shared memory for the whole chain
//
//   1. flat, fully-coalesced global→shared copy of RB·nk complex values
//   2. inverse: three axis passes (z, y, x), each thread accumulating its own
//      elements into registers, then a barrier, then written back
//   3. the W multiply is FUSED into the last inverse pass (no extra barrier,
//      no extra shared round trip)
//   4. forward: three axis passes; `scale` is FUSED into the last one
//   5. the STORE, whose thread→element map depends on out_layout
//
// THE STORE PATTERN, which is the point of out_layout=1.  The natural map
// (consecutive threads walk k) would scatter 16-byte writes at stride
// d3·d1·d4·d2 — the transpose's cost, moved into this kernel rather than
// removed.  Instead the store re-reads shared memory with consecutive threads
// walking the ROW axis: consecutive rows differ in d4, then d3, then d2, and
// d2 is the MINOR axis of the out_layout=1 destination.  So a block covering
// RB = nn·(d3·d4) consecutive rows writes nn contiguous destination elements
// per (k,d3,d4) — nn·16 bytes of real transaction.  RB is rounded down to a
// multiple of d3·d4 for exactly this reason.
//
// Shared-memory row stride is SP = nk|1 (forced ODD).  The store phase reads
// smem[j·SP + k] with j consecutive across threads; an even stride puts every
// thread of a phase on the same bank (8-way conflict at nk=16, 64, 216).  One
// padding element per row removes it.
//
// ---------------------------------------------------------------------------
// WHAT MOVED THE NUMBER: A JAX -> CUDA CATALOG
// ---------------------------------------------------------------------------
// This handler replaced a native-JAX expression that XLA already compiled
// competently.  Everything below is a measured delta, in the order it was
// found, so the next agent porting a JAX expression to a CUDA kernel can read
// the list instead of rediscovering it.  The prose version, written for that
// reader rather than for this file, is docs/dev/cuda_kernel_migration.md; ONE
// of the two owns each fact and this one owns the numbers.
//
// A. WHAT THE FUSION ITSELF BOUGHT  (this is the big one; do this first)
//    1. PASS FUSION.  XLA emits ifft / scale / multiply / fft / transpose as
//       separate kernels, each a full HBM round trip of the chain buffer:
//       8.1 passes over a 91.7 MB tile.  One kernel that keeps the k-row live
//       between the halves needs 1.12.  Measured on the rung chain:
//       1.367 ms -> 0.354 ms (3.87x), and XLA's 91.7 MB temp buffer goes to
//       ZERO because the intermediate never exists.
//    2. THE STORE PATTERN.  The consumer wanted (b,k,t,mu,s,nu) and XLA spent
//       a whole kernel transposing into it — 1488 us/matvec, 30% of the entire
//       ladder matvec, running at 246 GB/s.  Writing that permutation FROM THE
//       STORE deletes it rather than moving it.  Cost: reordering which thread
//       stores which element so the destination's minor axis is what
//       consecutive threads walk.  Measured free — the permuted store is not
//       slower than the coalesced one.
//    3. FOLDED SCALES.  cuFFT implements no norm convention, so XLA emits two
//       scal_kernel_val passes (271 us/matvec) for 'ortho'.  Both factors, and
//       any caller multiplier, commute with the linear transforms: fold them
//       into ONE constant applied once, inside the kernel.  Free.
//    Together: 3637 us of a 4897 us matvec replaced by 940 us.
//
// B. WHAT THE FIRST WORKING KERNEL GOT WRONG  (all four are invisible to a
//    correctness test, and a rewrite reintroduces them silently)
//    4. NO 64-BIT INTEGER DIVISION IN A PER-ELEMENT LOOP.  NVIDIA hardware has
//       no integer divide; a 64-bit div/mod is ~100 emulated instructions.
//       Decomposing a row index into five dimensions PER STORED ELEMENT cost
//       more than the memory traffic it served.  Fix: hoist to a PER-ROW
//       shared table (the destination is affine in k, so dst = dbase +
//       k*kstride), and carry loop indices incrementally instead of
//       recomputing i/nk.  Part of ~250 -> ~400 GB/s.
//    5. 128-BIT ACCESSES.  A complex128 spelled as two `double` loads is two
//       instructions each covering half a sector.  One 16-byte aligned struct
//       makes it one transaction.
//    6. COMPILE-TIME ELEMENT COUNTS.  A register array indexed by a runtime
//       loop bound is NOT in registers — nvcc puts it in LOCAL memory, i.e.
//       DRAM behind L1.  Template the body on the per-thread element count and
//       give each value its OWN entry point: a single kernel switching over
//       the instantiations pays the heaviest arm's register count on every
//       launch (measured 148 registers, which will not even fit a 576-thread
//       block).  Check `-Xptxas -v` for "0 bytes spill stores"; treat any
//       spill as a defect, not a tuning note.
//    7. ODD SHARED-MEMORY ROW STRIDE.  Threads reading smem[j*SP + k] with j
//       consecutive hit one bank if SP is even.  `SP = nk | 1` costs one
//       padding element per row and removes an 8-way conflict at nk = 16, 64
//       and 216.
//
// C. WHAT THE SECOND ROUND FOUND: THE BOTTLENECK MOVED
//    8. AFTER FUSION, SHARED MEMORY BINDS, NOT HBM.  Re-derive the roofline
//       after each change rather than assuming the original one still holds.
//       Measured here: 9:1 to 22:1 shared bytes per HBM byte against ~15:1
//       available, i.e. 53-91% of the SHARED roof while HBM idled at
//       400-650 GB/s.  The fix is never "move more bytes faster", it is "move
//       fewer bytes": see 9 and 10.
//    9. SKIP IDENTITY WORK.  An axis of extent 1 is the identity transform.
//       Every 2-D k-grid has one, and the pass for it was a full shared round
//       trip and two barriers computing nothing.  Skipping it is a runtime
//       branch on a kernel argument — uniform across the block, so free.
//   10. CONTIGUOUS OWNERSHIP.  Give a thread a contiguous run of the transform
//       axis and that axis's DFT sums over the thread's OWN registers: no
//       shared traffic, no barrier.  The trap is 6 above — the general form
//       (a thread owning a 2-D plane) needs a runtime-indexed register array
//       and silently spills.  A LINE works because the sum runs over the whole
//       array, making both indices compile-time loop counters, which is what
//       lets the extents stay runtime values.  See the CONTIGUOUS-OWNERSHIP
//       arm below.
//
// D. WHAT MADE IT SHIPPABLE RATHER THAN A DEMO
//   11. RUNTIME-COMPILED, RUNTIME-TARGETED.  NVRTC compiles the kernels once
//       per process for the compute capability queried FROM THE DEVICE, so one
//       .so serves every GPU generation and the build needs no nvcc.  Prefer a
//       native cubin, fall back to PTX.  Cache the failure as well as the
//       success: an NVRTC error is deterministic, and without a negative cache
//       it recompiles on every dispatch before failing again.
//   12. DERIVED BOUNDS, NAMED REFUSALS.  Every limit this kernel has is
//       computed from the device (shared memory per block) or from the
//       operands, never assumed; and every operand it cannot serve is refused
//       with the arithmetic, the device's own maximum, and the name of the
//       handler to use instead.  A kernel that silently mis-handles one shape
//       is worse than one that refuses it.
//   13. TWO ARMS, ONE REFERENCE.  Where a faster structure only applies to
//       some shapes, keep the general one as a plan-time fallback and gate
//       BOTH against the same native-JAX expression.  Then a shape that does
//       not fit the fast path is slow, not wrong.
//
// ---------------------------------------------------------------------------
// DEVICE CODE WITHOUT NVCC
// ---------------------------------------------------------------------------
// Same house pattern as the sibling `fft_flat_k_cuda_ffi.cc`: this TU is
// compiled by g++ against the CUDA headers, and the kernel lives in an NVRTC
// source string compiled ONCE per process for the compute capability queried
// from the runtime device, loaded through driver-API entry points resolved by
// dlsym (libcuda is already in the process — JAX loaded it; we add no
// link-time libcuda dependency).  This TU links libnvrtc and nothing else new
// — it does NOT use cuFFT at all.
//
// The ~120 lines of driver/NVRTC glue are DELIBERATELY not shared with
// fft_flat_k_cuda_ffi.cc.  NVRTC failures are cached per CUDA context per
// module, so one source string serving both handlers means a compile error in
// this new, uncertified kernel would disable the certified flat-k handler in
// the same process.  Two modules, two negative caches, one blast radius each.
// When this handler is certified, fold them.
//
// Envelope-honesty: every extent, stride and launch dimension comes from the
// runtime buffer dimensions and attributes.  Nothing is specialized to a deck.
// Concurrency: the kernel touches no shared arena and needs no plan cache, so
// the only global state is the per-context NVRTC module (built under one
// mutex).  There is no cross-call device state at all.

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <dlfcn.h>

#include "../common/mkl_thread_pin.h"   // rank-scoped log gate only

#include <cuda.h>            // driver-API types only; entry points via dlsym
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "xla/ffi/api/ffi.h"

namespace lorrax_ffi::conv_kminor {

namespace ffi = ::xla::ffi;
using C128 = std::complex<double>;

// ---------------------------------------------------------------------------
//  Launch-shape policy.  Both numbers are here, named, because they are the
//  only tuning in the file and a reader must be able to find them.
// ---------------------------------------------------------------------------
//: PREFERRED shared-memory budget per block, bytes.  32 KB keeps several
//: blocks resident per SM while holding a useful number of rows.  It is a
//: preference, not a limit: a k-grid whose SINGLE ROW needs more than this
//: grows the request up to the DEVICE's own opt-in maximum (queried, see
//: KernelArms::smem_max), because the alternative for such a grid is
//: refusing it.
//: Occupancy is what is traded away there, and only for rows that large.
static constexpr int kSmemPreferred = 32768;
//: The device's maximum dynamic shared memory per block is queried once per
//: context and carried in KernelArms::smem_max — NOT in a global.  It is the
//: ONE number that decides which k-grids this handler can serve, and it is a
//: property of the LOADED KERNELS (cuFuncSetAttribute has to have succeeded on
//: them for a launch to be allowed to ask for it), so binding it to the table
//: that owns those functions is what makes "planned" and "permitted" the same
//: fact.  A free-standing global could be read before any module was built.
//: Elements per thread.  There is a compiled kernel arm for each value in
//: 1..kEptMax, and ptxas prices them: 42 registers at EPT=1, 64 through EPT=5,
//: 128 at EPT=8 (all with ZERO spill).  kEptPref is what the launch plan aims
//: for — staying at or under 4 keeps every arm in the 64-register class, which
//: is worth more than the slightly larger blocks a higher EPT would allow.
//: kEptMax is the hard ceiling: past it there is no compiled arm.
static constexpr int kEptMax = 8;
static constexpr int kEptPref = 4;
//: Threads per block, capped.  Matches LRX_MAX_BLOCK in the kernel source,
//: where __launch_bounds__ turns it into ptxas's register budget — 65536/512 =
//: 128 registers, which is exactly what the heaviest arm needs.  The two
//: constants are one fact and must move together.
static constexpr int kMaxBlock = 512;
//: Target block size.  Blocks smaller than this waste occupancy on the small
//: grids, where RB is capped rather than shared-memory-bound.
static constexpr int kBlockTarget = 256;
//: Rows per block, capped.  Bigger blocks amortise the twelve barriers and the
//: per-row metadata over more elements, and lengthen the out_layout=1 store's
//: contiguous run (RB/(d3·d4) elements per k); 64 rows of a 9-point grid is a
//: 256 B run at the campaign fixture.  Past this the block hits the
//: 1024-thread ceiling anyway.
static constexpr int kRowsMax = 64;
//: Contiguous-arm capacities.  There is a compiled arm for CMAX = 2, 4 and 8;
//: ptxas spills at 16 under the 512-thread launch bound, which would put the
//: register array this arm exists to hold back into local memory.  A k-grid
//: whose innermost non-trivial axis is longer than kCtgMax takes the strided
//: arm instead — correct, just without the register-resident pass.
static constexpr int kCtgMax = 8;
static constexpr int kCtgArms[3] = {2, 4, 8};
//: Rows per block for the contiguous arm.  It can afford more than the strided
//: arm because its threads-per-row is FIXED at nk/C (usually small), so rows
//: are what fills the block — and a longer row run lengthens the out_layout=1
//: store's contiguous destination run for free.
static constexpr int kRowsMaxCtg = 128;

static bool log_enabled() {
    static const bool on = [] {
        return mklpin::log_here("LORRAX_CONV_KMINOR_LOG") ||
               mklpin::log_here("LORRAX_FFT_FFI_LOG");
    }();
    return on;
}

static ffi::Error fail(const char* where, const std::string& detail) {
    std::ostringstream os;
    os << "conv_kminor (k-minor fused conv CUDA FFI): " << where
       << " failed — " << detail;
    return ffi::Error(ffi::ErrorCode::kInternal, os.str());
}

#define LRX_CUDA_CHECK(expr, where)                                      \
    do {                                                                 \
        cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) {                                         \
            return fail((where), cudaGetErrorString(_e));                \
        }                                                                \
    } while (0)

// ---------------------------------------------------------------------------
//  Driver-API entry points via dlsym.  See the file header for why there is no
//  link-time libcuda.
// ---------------------------------------------------------------------------
struct DriverApi {
    CUresult (*ModuleLoadData)(CUmodule*, const void*) = nullptr;
    CUresult (*ModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*ModuleUnload)(CUmodule) = nullptr;
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                             unsigned, unsigned, unsigned, unsigned,
                             CUstream, void**, void**) = nullptr;
    CUresult (*CtxGetCurrent)(CUcontext*) = nullptr;
    CUresult (*GetErrorString)(CUresult, const char**) = nullptr;
    // Raises a kernel's ALLOWED dynamic-shared maximum past the 48 KB default.
    // Opt-in only: it changes what a launch MAY request, never what a launch
    // does request, so small launches keep their L1 carve-out untouched.
    CUresult (*FuncSetAttribute)(CUfunction, int, int) = nullptr;
    bool ok = false;
    std::string err;
};

static const DriverApi& driver_api() {
    static DriverApi api = [] {
        DriverApi a;
        void* h = RTLD_DEFAULT;
        dlerror();
        if (dlsym(h, "cuLaunchKernel") == nullptr) {
            h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
            if (h == nullptr) {
                const char* e = dlerror();
                a.err = std::string("dlopen(libcuda.so.1): ") +
                        (e ? e : "(dlerror returned no detail)");
                return a;
            }
        }
        auto need = [&](const char* name) -> void* {
            dlerror();
            void* p = dlsym(h, name);
            if (p == nullptr) {
                const char* e = dlerror();
                if (!a.err.empty()) a.err += "; ";
                a.err += std::string("dlsym(") + name + "): " +
                         (e ? e : "symbol not found (no dlerror detail)");
            }
            return p;
        };
        a.ModuleLoadData = reinterpret_cast<decltype(a.ModuleLoadData)>(
            need("cuModuleLoadData"));
        a.ModuleGetFunction = reinterpret_cast<decltype(a.ModuleGetFunction)>(
            need("cuModuleGetFunction"));
        a.ModuleUnload = reinterpret_cast<decltype(a.ModuleUnload)>(
            need("cuModuleUnload"));
        a.LaunchKernel = reinterpret_cast<decltype(a.LaunchKernel)>(
            need("cuLaunchKernel"));
        a.CtxGetCurrent = reinterpret_cast<decltype(a.CtxGetCurrent)>(
            need("cuCtxGetCurrent"));
        a.GetErrorString = reinterpret_cast<decltype(a.GetErrorString)>(
            need("cuGetErrorString"));
        a.FuncSetAttribute = reinterpret_cast<decltype(a.FuncSetAttribute)>(
            need("cuFuncSetAttribute"));
        a.ok = a.ModuleLoadData && a.ModuleGetFunction && a.ModuleUnload &&
               a.LaunchKernel && a.CtxGetCurrent && a.GetErrorString &&
               a.FuncSetAttribute;
        return a;
    }();
    return api;
}

static std::string cu_err(CUresult r) {
    const DriverApi& api = driver_api();
    const char* s = nullptr;
    if (api.GetErrorString && api.GetErrorString(r, &s) == CUDA_SUCCESS && s) {
        return s;
    }
    std::ostringstream os;
    os << "CUresult=" << static_cast<int>(r);
    return os.str();
}

// ---------------------------------------------------------------------------
//  THE KERNEL.  NVRTC source; raw `double` indexing (2i / 2i+1) rather than
//  double2 so the source needs no header at all under NVRTC — the same rule
//  the sibling flat-k kernels follow.
//
//  Index algebra, once, so the three passes can share one function:
//    a row is nk contiguous elements, k = ((ix·n1) + iy)·n2 + iz
//    an axis pass is fully described by (len, stride):
//        axis x : len=n0  stride=n1·n2
//        axis y : len=n1  stride=n2
//        axis z : len=n2  stride=1
//    with k = outer·(len·stride) + mid·stride + inner,  inner < stride,
//    the pass replaces element k by  Σ_j row[outer·len·stride + inner +
//    j·stride] · tw[(mid·j) mod len],  tw[m] = exp(-2πi·m/len) (sgn=+1,
//    forward) or its conjugate (sgn=-1, inverse).
// ---------------------------------------------------------------------------
static const char* kKernelSrc = R"__lrx__(

// 16-byte complex128, so every global and shared access is ONE 128-bit
// transaction instead of two 64-bit ones.  Declared here rather than using the
// toolkit's `double2` to keep this source header-free under NVRTC, which is
// the sibling flat-k kernel's rule and costs one line.
struct __align__(16) lrx_c2 { double x, y; };

// ONE separable axis pass over this thread's EPT elements.
//
// TEMPLATED ON EPT, and that is not style.  With a runtime element count the
// accumulator array cannot be proven register-resident and nvcc puts it in
// LOCAL memory — i.e. in DRAM behind L1 — so every one of the six passes pays
// a round trip that does not appear anywhere in the algorithm.  A compile-time
// EPT makes acc[] registers.  The host picks EPT and the kernel switches once,
// uniformly, at entry.
//
// TEMPLATED ON AXIS for the same class of reason.  The element's 3-D index
// (kx,ky,kz) is FIXED for a thread across all six passes, so it is decomposed
// ONCE by the caller; each pass then derives its (base, mid) from it with
// multiplies.  Spelling the pass generically over (len, stride) instead —
// which the first working version did — forces `k % stride`, `k / stride`,
// `. % len`, `. / len` per element per pass: 24 runtime integer divisions per
// element, ~20 emulated instructions each, against ~140 instructions of actual
// arithmetic.  That version ran the whole sweep at a flat ~320 GB/s and the
// index math, not the memory system, was what it was waiting on.
template <int EPT, int AXIS>
__device__ __forceinline__ void lrx_pass(
    const lrx_c2* rowp, const lrx_c2* twv,
    int nk, int lane, int tpr, int n0, int n1, int n2,
    const int (&kx)[EPT], const int (&ky)[EPT], const int (&kz)[EPT],
    double sgn, lrx_c2 (&acc)[EPT])
{
    const int len    = (AXIS == 0) ? n0 : ((AXIS == 1) ? n1 : n2);
    const int stride = (AXIS == 0) ? (n1 * n2) : ((AXIS == 1) ? n2 : 1);
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int k = lane + e * tpr;
        if (k < nk) {
            int b, md;
            if (AXIS == 0)      { b = ky[e] * n2 + kz[e];      md = kx[e]; }
            else if (AXIS == 1) { b = kx[e] * n1 * n2 + kz[e]; md = ky[e]; }
            else                { b = (kx[e] * n1 + ky[e]) * n2; md = kz[e]; }
            double ar = 0.0, ai = 0.0;
            int m = 0;                     // (md·j) mod len, carried
            for (int j = 0; j < len; ++j) {
                const lrx_c2 v = rowp[b + j * stride];
                const lrx_c2 w = twv[m];
                const double wi = sgn * w.y;
                ar += v.x * w.x - v.y * wi;
                ai += v.x * wi + v.y * w.x;
                m += md;
                if (m >= len) m -= len;
            }
            acc[e].x = ar;
            acc[e].y = ai;
        }
    }
}

template <int EPT>
__device__ __forceinline__ void lrx_writeback(
    lrx_c2* rowp, int nk, int lane, int tpr, const lrx_c2 (&acc)[EPT])
{
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int k = lane + e * tpr;
        if (k < nk) rowp[k] = acc[e];
    }
}

template <int EPT>
__device__ __forceinline__ void lrx_conv_body(
    const lrx_c2* tin, const lrx_c2* __restrict__ wgt, lrx_c2* uout,
    long long R, long long d1, long long d2, long long d3, long long d4,
    int nk, int n0, int n1, int n2, int SP, double scale, int out_layout,
    lrx_c2* sm)
{
    const int tpr  = blockDim.x;
    const int RB   = blockDim.y;
    const int tid  = threadIdx.y * tpr + threadIdx.x;
    const int nthr = tpr * RB;
    const int lane = threadIdx.x;
    const int ntw  = n0 + n1 + n2;
    const int tile = RB * nk;

    lrx_c2* tw = sm + (long long)RB * SP;
    long long* meta = (long long*)(tw + ntw);   // [0,RB) W offset; [RB,2RB) dst
    const long long r0 = (long long)blockIdx.x * RB;
    const long long kstride = d3 * d1 * d4 * d2;   // out_layout=1 k stride

    // --- twiddle rings: tw_x[n0], tw_y[n1], tw_z[n2], exp(-2πi m/len) ---
    for (int i = tid; i < ntw; i += nthr) {
        int m, len;
        if (i < n0)            { m = i;           len = n0; }
        else if (i < n0 + n1)  { m = i - n0;      len = n1; }
        else                   { m = i - n0 - n1; len = n2; }
        double s, c;
        // sincospi, not sincos(2*pi*x): the CUDA math API is available to
        // NVRTC with no include, and sincospi takes HALF-TURNS, so it never
        // rounds a multiple of pi into the angle.  sincos(-2*PI*m/len) would
        // carry ~1e-15 of angle error straight into the twiddle and eat the
        // whole parity budget.
        sincospi(-2.0 * (double)m / (double)len, &s, &c);
        tw[i].x = c;
        tw[i].y = s;
    }
    const lrx_c2* twx = tw;
    const lrx_c2* twy = tw + n0;
    const lrx_c2* twz = tw + n0 + n1;

    // --- PER-ROW METADATA.  The ONLY 64-bit div/mod in the kernel, and it
    //     runs RB times per block instead of RB·nk.  Row r decomposes as
    //     (i0,i1,i2,i3,i4) with i4 fastest; from that,
    //        W row  = (r / (d3·d4)) mod (d1·d2) = i1·d2 + i2
    //        dst(k) = dbase + k·kstride,  because
    //                 ((((i0·nk+k)·d3+i3)·d1+i1)·d4+i4)·d2+i2
    //               = i0·nk·kstride + k·kstride + ((i3·d1+i1)·d4+i4)·d2 + i2
    //     Rows past the end still decompose in range on i1..i4 (only i0 runs
    //     over), so the W offset is always a legal read and the store is
    //     guarded instead.
    for (int j = tid; j < RB; j += nthr) {
        long long q  = r0 + j;
        const long long i4 = q % d4; q /= d4;
        const long long i3 = q % d3; q /= d3;
        const long long i2 = q % d2; q /= d2;
        const long long i1 = q % d1; q /= d1;
        const long long i0 = q;
        meta[j]      = (i1 * d2 + i2) * nk;
        meta[RB + j] = i0 * (long long)nk * kstride
                     + ((i3 * d1 + i1) * d4 + i4) * d2 + i2;
    }

    // --- LOAD: flat and fully coalesced.  (j,k) advances INCREMENTALLY, so
    //     the loop carries no division; the one at entry is 32-bit. ---
    {
        int i = tid;
        int j = i / nk, k = i - j * nk;
        const int dj = nthr / nk, dk = nthr - dj * nk;
        for (; i < tile; i += nthr) {
            const long long r = r0 + j;
            lrx_c2 v;
            v.x = 0.0; v.y = 0.0;
            if (r < R) v = tin[r * (long long)nk + k];
            sm[(long long)j * SP + k] = v;
            k += dk;
            if (k >= nk) { k -= nk; j += 1; }
            j += dj;
        }
    }
    __syncthreads();

    lrx_c2* rowp = sm + (long long)threadIdx.y * SP;
    lrx_c2 acc[EPT];

    // --- the element's 3-D index, decomposed ONCE for all six passes ---
    int kx[EPT], ky[EPT], kz[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        int t = lane + e * tpr;
        if (t >= nk) t = 0;                 // inactive lane: keep it in range
        kz[e] = t % n2; t /= n2;
        ky[e] = t % n1;
        kx[e] = t / n1;
    }

    // ================= THE SIX PASSES =================
    // A length-1 axis is the IDENTITY transform, so its pass is a shared-memory
    // round trip and two barriers that compute nothing.  Every 2-D k-grid the
    // BSE runs (3x3x1, 4x4x1) has one, and this kernel is SHARED-MEMORY
    // BANDWIDTH bound — measured 10-18 TB/s of an A100's ~19.5 TB/s roof — so
    // skipping them is not a micro-optimisation, it is 4 of the 22 shared
    // accesses per element at 3x3x1.  The conditions are kernel ARGUMENTS, so
    // every thread in the block takes the same arm and the __syncthreads()
    // inside them are reached uniformly.
#define LRX_MULT_W                                                            \
    {                                                                         \
        const lrx_c2* wp = wgt + meta[threadIdx.y];                           \
        _Pragma("unroll")                                                     \
        for (int e = 0; e < EPT; ++e) {                                       \
            const int k = lane + e * tpr;                                     \
            if (k < nk) {                                                     \
                const lrx_c2 w = wp[k];                                       \
                const double ar = acc[e].x, ai = acc[e].y;                    \
                acc[e].x = ar * w.x - ai * w.y;                               \
                acc[e].y = ar * w.y + ai * w.x;                               \
            }                                                                 \
        }                                                                     \
    }
#define LRX_SCALE                                                             \
    {                                                                         \
        _Pragma("unroll")                                                     \
        for (int e = 0; e < EPT; ++e) {                                       \
            acc[e].x *= scale;                                                \
            acc[e].y *= scale;                                                \
        }                                                                     \
    }
    // AX = axis index, TWV = its ring, LEN = its extent, SGN = direction,
    // FOLD = the thing fused into the LAST executed pass of this direction
    // (the W multiply on the inverse, the folded constant on the forward), so
    // neither ever costs a round trip of its own.
#define LRX_DO_PASS(AX, TWV, LEN, SGN, FOLD)                                  \
    if ((LEN) > 1) {                                                          \
        lrx_pass<EPT, AX>(rowp, TWV, nk, lane, tpr, n0, n1, n2,               \
                          kx, ky, kz, SGN, acc);                              \
        __syncthreads();                                                      \
        if (--rem == 0) { FOLD }                                              \
        lrx_writeback<EPT>(rowp, nk, lane, tpr, acc);                         \
        __syncthreads();                                                      \
    }

    const int nact = (n0 > 1) + (n1 > 1) + (n2 > 1);
    if (nact == 0) {
        // Every axis is length 1: both transforms are the identity and the
        // whole kernel is U = scale * T * W.  Stated rather than assumed —
        // with nact==0 the loops below would leave the W multiply and the
        // constant unapplied, silently.
        const lrx_c2* wp = wgt + meta[threadIdx.y];
#pragma unroll
        for (int e = 0; e < EPT; ++e) {
            const int k = lane + e * tpr;
            if (k < nk) {
                const lrx_c2 v = rowp[k], w = wp[k];
                rowp[k].x = (v.x * w.x - v.y * w.y) * scale;
                rowp[k].y = (v.x * w.y + v.y * w.x) * scale;
            }
        }
        __syncthreads();
    } else {
        int rem = nact;
        LRX_DO_PASS(2, twz, n2, -1.0, LRX_MULT_W)
        LRX_DO_PASS(1, twy, n1, -1.0, LRX_MULT_W)
        LRX_DO_PASS(0, twx, n0, -1.0, LRX_MULT_W)
        rem = nact;
        LRX_DO_PASS(2, twz, n2, 1.0, LRX_SCALE)
        LRX_DO_PASS(1, twy, n1, 1.0, LRX_SCALE)
        LRX_DO_PASS(0, twx, n0, 1.0, LRX_SCALE)
    }
#undef LRX_DO_PASS
#undef LRX_SCALE
#undef LRX_MULT_W

    // ============================ THE STORE ============================
    if (out_layout == 0) {
        // same layout as the operand: the flat map, mirror of the load.
        int i = tid;
        int j = i / nk, k = i - j * nk;
        const int dj = nthr / nk, dk = nthr - dj * nk;
        for (; i < tile; i += nthr) {
            const long long r = r0 + j;
            if (r < R) uout[r * (long long)nk + k] = sm[(long long)j * SP + k];
            k += dk;
            if (k >= nk) { k -= nk; j += 1; }
            j += dj;
        }
    } else {
        // (d0, nk, d3, d1, d4, d2): consecutive threads walk the ROW axis, so
        // consecutive d2 — the destination's MINOR axis — land adjacent.  A
        // block covering RB = nn·(d3·d4) consecutive rows therefore writes nn
        // contiguous elements per (k, d3, d4); RB is rounded to a multiple of
        // d3·d4 host-side for exactly this.
        int i = tid;
        int k = i / RB, j = i - k * RB;
        const int dk = nthr / RB, dj = nthr - dk * RB;
        for (; i < tile; i += nthr) {
            if (r0 + j < R) {
                uout[meta[RB + j] + (long long)k * kstride] =
                    sm[(long long)j * SP + k];
            }
            j += dj;
            if (j >= RB) { j -= RB; k += 1; }
            k += dk;
        }
    }
}

// ---------------------------------------------------------------------------
//  EIGHT ENTRY POINTS, one per EPT, instead of one kernel with a switch.
//
//  ptxas allocates registers for the WHOLE entry function, so a single kernel
//  switching over eight instantiations pays the EPT=8 arm's 148 registers on
//  every launch — and 148 registers x a 576-thread block is 85k registers
//  against the SM's 64k, i.e. a launch failure at the small grids where the
//  block is largest.  Separate entry points let each arm cost what it costs.
//
//  EPT is therefore NOT a kernel argument any more: it is the entry point's
//  identity.  The host picks the name from its own launch plan, so the two
//  cannot disagree.
//
//  __launch_bounds__(LRX_MAX_BLOCK) then makes the remaining case impossible
//  rather than merely unlikely: it tells ptxas the block will never exceed
//  that size, so it caps registers accordingly and every launch the host can
//  construct is guaranteed to fit.  The host's own block cap is the same
//  constant, named once on each side.
// ---------------------------------------------------------------------------
#define LRX_MAX_BLOCK 512

#define LRX_ENTRY(N)                                                          \
extern "C" __global__ __launch_bounds__(LRX_MAX_BLOCK)                        \
void lrx_conv_kminor_c128_e##N(                                               \
    const lrx_c2* tin,                  /* NOT __restrict__: may alias uout */\
    const lrx_c2* __restrict__ wgt,                                           \
    lrx_c2* uout,                                                             \
    long long R, long long MN,                                                \
    long long d1, long long d2, long long d3, long long d4,                   \
    int nk, int n0, int n1, int n2,                                           \
    int SP,                                                                   \
    double scale, int out_layout)                                             \
{                                                                             \
    extern __shared__ lrx_c2 sm[];                                            \
    lrx_conv_body<N>(tin, wgt, uout, R, d1, d2, d3, d4, nk,                   \
                     n0, n1, n2, SP, scale, out_layout, sm);                  \
}

LRX_ENTRY(1) LRX_ENTRY(2) LRX_ENTRY(3) LRX_ENTRY(4)
LRX_ENTRY(5) LRX_ENTRY(6) LRX_ENTRY(7) LRX_ENTRY(8)
#undef LRX_ENTRY

// ===========================================================================
//  ARM 2 of 2: CONTIGUOUS-OWNERSHIP.  Fewer shared-memory round trips.
// ===========================================================================
//  The strided arm above moves every element through shared memory SIX times
//  (once per axis pass), because a thread owning k = lane + e*tpr shares no
//  axis line with itself: every pass is a cross-thread read.  Measured, that
//  makes the kernel SHARED-MEMORY BANDWIDTH bound (10-18 TB/s of an A100's
//  roof) while HBM idles at 400-650 GB/s.
//
//  THE FIX: give a thread a CONTIGUOUS run of k, sized to one whole line along
//  the innermost non-trivial axis.  That axis's transform then sums over the
//  thread's OWN elements — no shared memory, no barrier — and only the
//  remaining axes stay cross-thread.  Shared accesses per element fall from
//  2 + 2*(n0+n1+n2) + 6 to 2 + 2*(sum of the CROSS axes) + 4.
//
//  THE PROBLEM THIS HAD TO SOLVE, and it is the whole reason the arm looks
//  like this.  A register array can only be indexed by a COMPILE-TIME
//  constant; index it with a runtime value and nvcc silently moves it to
//  LOCAL memory, which is DRAM behind L1 — strictly worse than the shared
//  memory it was meant to replace.  A thread owning a 2-D (y,z) plane needs
//  acc[(e/n2)*n2 + j], whose index depends on the runtime n2, so the plane
//  variant cannot be written without making n1 and n2 template parameters —
//  i.e. a per-size compiled family, which is exactly what this handler must
//  not have.
//
//  A LINE is different, and that is the insight the arm rests on: the DFT
//  along the owned axis sums over ALL of the thread's own elements, so the
//  inner loop runs s = 0..C-1 over the WHOLE register array.  Both the output
//  index o and the source index s are then compile-time loop counters of
//  unrolled loops, and the array stays in registers — with C, and every axis
//  extent, still a RUNTIME value.  One code path, no per-size compilation, any
//  (nkx,nky,nkz).
//
//  WHY THE TWIDDLE READS DO NOT GIVE THE SAVING BACK.  The intra-thread pass
//  reads twv[(o*s) mod C] from shared memory, which looks like it just trades
//  one shared access for another.  It does not: o, s and C are identical
//  across every lane of the warp at each unrolled step, so the address is
//  WARP-UNIFORM and the read is a broadcast — one transaction, not 32.  The
//  data reads it replaces were 32 distinct addresses.  Bandwidth, not
//  instruction count, is what binds here, so the trade is real.
//
//  CMAX is the compile-time capacity of the register arrays.  The host picks
//  the smallest instantiation with CMAX >= C and FALLS BACK TO THE STRIDED ARM
//  when C exceeds the largest — a plan-time selection, stated, with both arms
//  gated to the same reference.  Slots past C are never executed (the unrolled
//  loops `break`), so a loose fit costs only register footprint.
//
//  WHY CMAX STOPS AT 8, and it is measured, not chosen: the own-pass is an
//  O(CMAX^2) fully-unrolled body, so acc[]+tmp[] is 4*CMAX doubles and the
//  code grows quadratically.  ptxas reports ZERO spill at CMAX = 2, 4 and 8
//  and 1168 bytes of SPILL STORES at 16 under the 512-thread launch bound —
//  i.e. the 16 arm puts the very array this design exists to keep in registers
//  back into local memory, which would make it slower than the arm it
//  replaces.  Every k-grid in the campaign sweep has an innermost non-trivial
//  extent of 2, 3, 4 or 6, so all of them take this arm; grids with a larger
//  innermost axis take the strided arm and are correct, just not faster.
// ===========================================================================

//: One cross-thread axis pass, for a thread owning C contiguous elements.
//: The axis is described entirely by (len, stride, base0, mid): because the
//: owned run is a line along the INNERMOST non-trivial axis, every element of
//: the run has the same `mid` and its source base is simply base0 + o.  The
//: host derives those four numbers from (nkx,nky,nkz) and the thread's fixed
//: outer indices; the kernel does no division at all.
template <int CMAX>
__device__ __forceinline__ void lrx_ctg_cross(
    const lrx_c2* rowp, const lrx_c2* twv,
    int C, int len, int stride, int base0, int mid, double sgn,
    lrx_c2 (&acc)[CMAX])
{
#pragma unroll
    for (int o = 0; o < CMAX; ++o) {
        if (o >= C) break;
        const int b = base0 + o;
        double ar = 0.0, ai = 0.0;
        int m = 0;
        for (int j = 0; j < len; ++j) {
            const lrx_c2 v = rowp[b + j * stride];
            const lrx_c2 w = twv[m];
            const double wi = sgn * w.y;
            ar += v.x * w.x - v.y * wi;
            ai += v.x * wi + v.y * w.x;
            m += mid;
            if (m >= len) m -= len;
        }
        acc[o].x = ar;
        acc[o].y = ai;
    }
}

//: The INTRA-thread pass over the owned axis: no shared data traffic and no
//: barrier.  Both loop counters are compile-time, which is what keeps acc[]
//: and tmp[] in registers with C still a runtime value.
template <int CMAX>
__device__ __forceinline__ void lrx_ctg_own(
    const lrx_c2* twv, int C, double sgn, lrx_c2 (&acc)[CMAX])
{
    lrx_c2 tmp[CMAX];
#pragma unroll
    for (int o = 0; o < CMAX; ++o) {
        if (o >= C) break;
        double ar = 0.0, ai = 0.0;
        int m = 0;
#pragma unroll
        for (int s = 0; s < CMAX; ++s) {
            if (s >= C) break;
            const lrx_c2 v = acc[s];        // COMPILE-TIME index: registers
            const lrx_c2 w = twv[m];        // warp-uniform: a broadcast read
            const double wi = sgn * w.y;
            ar += v.x * w.x - v.y * wi;
            ai += v.x * wi + v.y * w.x;
            m += o;
            if (m >= C) m -= C;
        }
        tmp[o].x = ar;
        tmp[o].y = ai;
    }
#pragma unroll
    for (int o = 0; o < CMAX; ++o) {
        if (o >= C) break;
        acc[o] = tmp[o];
    }
}

template <int CMAX>
__device__ __forceinline__ void lrx_ctg_store_regs(
    lrx_c2* rowp, int C, int kbase, const lrx_c2 (&acc)[CMAX])
{
#pragma unroll
    for (int o = 0; o < CMAX; ++o) {
        if (o >= C) break;
        rowp[kbase + o] = acc[o];
    }
}

template <int CMAX>
__device__ __forceinline__ void lrx_conv_body_ctg(
    const lrx_c2* tin, const lrx_c2* __restrict__ wgt, lrx_c2* uout,
    long long R, long long d1, long long d2, long long d3, long long d4,
    int nk, int n0, int n1, int n2, int SP, int C, int oa,
    double scale, int out_layout, lrx_c2* sm)
{
    const int tpr  = blockDim.x;
    const int RB   = blockDim.y;
    const int tid  = threadIdx.y * tpr + threadIdx.x;
    const int nthr = tpr * RB;
    const int lane = threadIdx.x;
    const int ntw  = n0 + n1 + n2;
    const int tile = RB * nk;

    lrx_c2* tw = sm + (long long)RB * SP;
    long long* meta = (long long*)(tw + ntw);
    const long long r0 = (long long)blockIdx.x * RB;
    const long long kstride = d3 * d1 * d4 * d2;

    for (int i = tid; i < ntw; i += nthr) {
        int m, len;
        if (i < n0)            { m = i;           len = n0; }
        else if (i < n0 + n1)  { m = i - n0;      len = n1; }
        else                   { m = i - n0 - n1; len = n2; }
        double s, c;
        sincospi(-2.0 * (double)m / (double)len, &s, &c);
        tw[i].x = c;
        tw[i].y = s;
    }
    const lrx_c2* twx = tw;
    const lrx_c2* twy = tw + n0;
    const lrx_c2* twz = tw + n0 + n1;

    for (int j = tid; j < RB; j += nthr) {
        long long q  = r0 + j;
        const long long i4 = q % d4; q /= d4;
        const long long i3 = q % d3; q /= d3;
        const long long i2 = q % d2; q /= d2;
        const long long i1 = q % d1; q /= d1;
        const long long i0 = q;
        meta[j]      = (i1 * d2 + i2) * nk;
        meta[RB + j] = i0 * (long long)nk * kstride
                     + ((i3 * d1 + i1) * d4 + i4) * d2 + i2;
    }

    {   // coalesced global -> shared, identical to the strided arm
        int i = tid;
        int j = i / nk, k = i - j * nk;
        const int dj = nthr / nk, dk = nthr - dj * nk;
        for (; i < tile; i += nthr) {
            const long long r = r0 + j;
            lrx_c2 v;
            v.x = 0.0; v.y = 0.0;
            if (r < R) v = tin[r * (long long)nk + k];
            sm[(long long)j * SP + k] = v;
            k += dk;
            if (k >= nk) { k -= nk; j += 1; }
            j += dj;
        }
    }
    __syncthreads();

    lrx_c2* rowp = sm + (long long)threadIdx.y * SP;
    const int kbase = lane * C;

    // --- THE THREAD'S FIXED OUTER INDICES.  The owned run is a line along the
    //     innermost non-trivial axis, so the two outer indices are CONSTANT
    //     over the run — no per-slot decomposition, no per-slot division, and
    //     no index arrays competing with acc[] for registers.
    int ix = 0, iy = 0;
    if (oa == 2)      { ix = lane / n1; iy = lane - ix * n1; }
    else if (oa == 1) { ix = lane; }
    // oa == 0: the whole row belongs to this thread; ix, iy unused.

    // Cross-axis descriptors (len, stride, base0, mid), derived once.  See the
    // header for the algebra; each is exact, not an approximation of it.
    int yc_len = 0, yc_str = 0, yc_b0 = 0, yc_mid = 0;
    int xc_len = 0, xc_str = 0, xc_b0 = 0, xc_mid = 0;
    if (oa == 2) {
        if (n1 > 1) { yc_len = n1; yc_str = n2; yc_b0 = ix * n1 * n2; yc_mid = iy; }
        if (n0 > 1) { xc_len = n0; xc_str = n1 * n2; xc_b0 = iy * n2;  xc_mid = ix; }
    } else if (oa == 1) {
        if (n0 > 1) { xc_len = n0; xc_str = n1;      xc_b0 = 0;        xc_mid = ix; }
    }

    lrx_c2 acc[CMAX];
#pragma unroll
    for (int o = 0; o < CMAX; ++o) {
        if (o >= C) break;
        acc[o] = rowp[kbase + o];
    }

    // ===================== INVERSE (unnormalised, sgn = -1) =====================
    if (C > 1) {
        lrx_ctg_own<CMAX>((oa == 2) ? twz : ((oa == 1) ? twy : twx),
                          C, -1.0, acc);
    }
    if (yc_len > 1) {
        __syncthreads();                       // everyone has read; safe to write
        lrx_ctg_store_regs<CMAX>(rowp, C, kbase, acc);
        __syncthreads();
        lrx_ctg_cross<CMAX>(rowp, twy, C, yc_len, yc_str, yc_b0, yc_mid,
                            -1.0, acc);
    }
    if (xc_len > 1) {
        __syncthreads();
        lrx_ctg_store_regs<CMAX>(rowp, C, kbase, acc);
        __syncthreads();
        lrx_ctg_cross<CMAX>(rowp, twx, C, xc_len, xc_str, xc_b0, xc_mid,
                            -1.0, acc);
    }

    // --- the stored kernel's multiply, in registers between the halves ---
    {
        const lrx_c2* wp = wgt + meta[threadIdx.y] + kbase;
#pragma unroll
        for (int o = 0; o < CMAX; ++o) {
            if (o >= C) break;
            const lrx_c2 w = wp[o];
            const double ar = acc[o].x, ai = acc[o].y;
            acc[o].x = ar * w.x - ai * w.y;
            acc[o].y = ar * w.y + ai * w.x;
        }
    }

    // ===================== FORWARD (unnormalised, sgn = +1) =====================
    if (xc_len > 1) {
        __syncthreads();
        lrx_ctg_store_regs<CMAX>(rowp, C, kbase, acc);
        __syncthreads();
        lrx_ctg_cross<CMAX>(rowp, twx, C, xc_len, xc_str, xc_b0, xc_mid,
                            1.0, acc);
    }
    if (yc_len > 1) {
        __syncthreads();
        lrx_ctg_store_regs<CMAX>(rowp, C, kbase, acc);
        __syncthreads();
        lrx_ctg_cross<CMAX>(rowp, twy, C, yc_len, yc_str, yc_b0, yc_mid,
                            1.0, acc);
    }
    if (C > 1) {
        lrx_ctg_own<CMAX>((oa == 2) ? twz : ((oa == 1) ? twy : twx),
                          C, 1.0, acc);
    }

    // --- the single folded constant, applied once on the way back out ---
#pragma unroll
    for (int o = 0; o < CMAX; ++o) {
        if (o >= C) break;
        acc[o].x *= scale;
        acc[o].y *= scale;
    }
    __syncthreads();
    lrx_ctg_store_regs<CMAX>(rowp, C, kbase, acc);
    __syncthreads();

    // ============================ THE STORE ============================
    // Byte-identical to the strided arm's: the store pattern is a property of
    // the OUTPUT layout, not of how the transform was decomposed.
    if (out_layout == 0) {
        int i = tid;
        int j = i / nk, k = i - j * nk;
        const int dj = nthr / nk, dk = nthr - dj * nk;
        for (; i < tile; i += nthr) {
            const long long r = r0 + j;
            if (r < R) uout[r * (long long)nk + k] = sm[(long long)j * SP + k];
            k += dk;
            if (k >= nk) { k -= nk; j += 1; }
            j += dj;
        }
    } else {
        int i = tid;
        int k = i / RB, j = i - k * RB;
        const int dk = nthr / RB, dj = nthr - dk * RB;
        for (; i < tile; i += nthr) {
            if (r0 + j < R) {
                uout[meta[RB + j] + (long long)k * kstride] =
                    sm[(long long)j * SP + k];
            }
            j += dj;
            if (j >= RB) { j -= RB; k += 1; }
            k += dk;
        }
    }
}

#define LRX_ENTRY_CTG(N)                                                      \
extern "C" __global__ __launch_bounds__(LRX_MAX_BLOCK)                        \
void lrx_conv_kminor_c128_ctg##N(                                             \
    const lrx_c2* tin,                                                        \
    const lrx_c2* __restrict__ wgt,                                           \
    lrx_c2* uout,                                                             \
    long long R, long long MN,                                                \
    long long d1, long long d2, long long d3, long long d4,                   \
    int nk, int n0, int n1, int n2,                                           \
    int SP, int C, int oa,                                                    \
    double scale, int out_layout)                                             \
{                                                                             \
    extern __shared__ lrx_c2 sm[];                                            \
    lrx_conv_body_ctg<N>(tin, wgt, uout, R, d1, d2, d3, d4, nk,               \
                         n0, n1, n2, SP, C, oa, scale, out_layout, sm);       \
}

LRX_ENTRY_CTG(2) LRX_ENTRY_CTG(4) LRX_ENTRY_CTG(8)
#undef LRX_ENTRY_CTG
)__lrx__";

// ---------------------------------------------------------------------------
//  Per-context NVRTC module, with a NEGATIVE cache: an NVRTC/module failure is
//  deterministic for a given process+context and this runs per FFI dispatch,
//  so a persistent failure must not re-run the compile on every call.
// ---------------------------------------------------------------------------
static std::mutex g_mu;

//: The eight entry points, indexed by EPT-1.  ONE NVRTC module carries them
//: all (one compile, one negative cache); ptxas gives each its own register
//: budget, which is the whole reason they are separate kernels.
struct KernelArms {
    CUfunction fn[kEptMax] = {nullptr};    //: strided arm, indexed by EPT-1
    CUfunction ctg[3] = {nullptr};         //: contiguous arm, by kCtgArms index
    int smem_max = 0;      //: bytes of dynamic shared memory a launch may ask
};

static ffi::Error ensure_kernels(const KernelArms** out) {
    static std::map<CUcontext, KernelArms> cache;
    static std::map<CUcontext, std::string> fail_cache;
    const DriverApi& api = driver_api();
    if (!api.ok) {
        return fail("driver-api resolve",
                    std::string("the CUDA driver entry points this handler "
                                "needs could not be resolved — ") +
                        (api.err.empty()
                             ? "no detail recorded (is a CUDA driver present "
                               "on this node?)"
                             : api.err));
    }
    CUcontext ctx = nullptr;
    CUresult cr = api.CtxGetCurrent(&ctx);
    if (cr != CUDA_SUCCESS) {
        return fail("cuCtxGetCurrent",
                    "CUresult=" + std::to_string(static_cast<int>(cr)) +
                        " (" + cu_err(cr) + ")");
    }
    if (ctx == nullptr) {
        LRX_CUDA_CHECK(cudaFree(nullptr), "context bind (cudaFree(0))");
        cr = api.CtxGetCurrent(&ctx);
        if (cr != CUDA_SUCCESS) {
            return fail("cuCtxGetCurrent after context bind",
                        "CUresult=" +
                            std::to_string(static_cast<int>(cr)) + " (" +
                            cu_err(cr) + ")");
        }
        if (ctx == nullptr) {
            return fail("context bind", "no current CUDA context");
        }
    }
    auto it = cache.find(ctx);
    if (it != cache.end()) {
        *out = &it->second;
        return ffi::Error::Success();
    }
    auto fit = fail_cache.find(ctx);
    if (fit != fail_cache.end()) {
        return fail("kernel build (cached failure, NVRTC not re-run)",
                    fit->second);
    }
    auto fail_sticky = [&](const char* where,
                           const std::string& detail) -> ffi::Error {
        fail_cache.emplace(ctx, std::string(where) + " — " + detail);
        return fail(where, detail);
    };

    int dev = 0, cc_major = 0, cc_minor = 0, smem_optin = 0;
    LRX_CUDA_CHECK(cudaGetDevice(&dev), "cudaGetDevice");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(
                       &cc_major, cudaDevAttrComputeCapabilityMajor, dev),
                   "query compute capability (major)");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(
                       &cc_minor, cudaDevAttrComputeCapabilityMinor, dev),
                   "query compute capability (minor)");
    // THE SIZE ENVELOPE, measured from the hardware.  The largest k-row this
    // handler can hold — and therefore the largest (nkx,nky,nkz) it can serve
    // — is decided by this one number.  Asking the device beats assuming
    // 48 KB: on an A100 the opt-in maximum is ~3.4x that, which is the
    // difference between serving a 12^3 grid and refusing it.
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(
                       &smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                       dev),
                   "query max dynamic shared memory per block (opt-in)");

    nvrtcProgram prog = nullptr;
    nvrtcResult nr = nvrtcCreateProgram(&prog, kKernelSrc,
                                        "lrx_conv_kminor.cu",
                                        0, nullptr, nullptr);
    if (nr != NVRTC_SUCCESS) {
        return fail_sticky("nvrtcCreateProgram", nvrtcGetErrorString(nr));
    }
    char arch[64];
    std::snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d",
                  cc_major, cc_minor);
    // FMA contraction is left ON (the nvrtc default).  It is both faster and
    // MORE accurate here — one rounding per complex multiply-accumulate
    // instead of two — and bit-equality with XLA's radix emission was never
    // on offer anyway (different algorithm); the gate is 1e-15, value-level.
    const char* opts[] = {arch};
    nr = nvrtcCompileProgram(prog, 1, opts);
    if (nr != NVRTC_SUCCESS) {
        size_t log_sz = 0;
        std::string log;
        if (nvrtcGetProgramLogSize(prog, &log_sz) == NVRTC_SUCCESS &&
            log_sz > 1) {
            log.resize(log_sz);
            nvrtcGetProgramLog(prog, &log[0]);
        }
        nvrtcDestroyProgram(&prog);
        return fail_sticky("nvrtcCompileProgram",
                           std::string(nvrtcGetErrorString(nr)) + " — " + log);
    }
    std::vector<char> image;
    size_t sz = 0;
    bool used_cubin = false;
    if (nvrtcGetCUBINSize(prog, &sz) == NVRTC_SUCCESS && sz > 0) {
        image.resize(sz);
        nr = nvrtcGetCUBIN(prog, image.data());
        used_cubin = true;
    } else if (nvrtcGetPTXSize(prog, &sz) == NVRTC_SUCCESS && sz > 0) {
        image.resize(sz);
        nr = nvrtcGetPTX(prog, image.data());
    } else {
        nr = NVRTC_ERROR_INTERNAL_ERROR;
    }
    nvrtcDestroyProgram(&prog);
    if (nr != NVRTC_SUCCESS || image.empty()) {
        return fail_sticky("nvrtc get cubin/ptx", nvrtcGetErrorString(nr));
    }

    CUmodule mod = nullptr;
    cr = api.ModuleLoadData(&mod, image.data());
    if (cr != CUDA_SUCCESS) {
        return fail_sticky("cuModuleLoadData", cu_err(cr));
    }
    KernelArms arms;
    for (int e = 1; e <= kEptMax; ++e) {
        char name[64];
        std::snprintf(name, sizeof(name), "lrx_conv_kminor_c128_e%d", e);
        cr = api.ModuleGetFunction(&arms.fn[e - 1], mod, name);
        if (cr != CUDA_SUCCESS) {
            // Unload before failing: the module is unreachable after this
            // return, and without the unload it leaked device memory on every
            // retry before the negative cache existed.
            api.ModuleUnload(mod);
            return fail_sticky("cuModuleGetFunction",
                               std::string(name) + ": " + cu_err(cr));
        }
        // Dynamic shared-memory opt-in is part of the launch contract.  Never
        // advertise a ceiling that was not established for every function the
        // planner may select.
        if (smem_optin > 49152) {
            cr = api.FuncSetAttribute(
                arms.fn[e - 1],
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_optin);
            if (cr != CUDA_SUCCESS) {
                api.ModuleUnload(mod);
                return fail_sticky(
                    "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
                    std::string(name) + ": requested " +
                        std::to_string(smem_optin) + " B: CUresult=" +
                        std::to_string(static_cast<int>(cr)) + " (" +
                        cu_err(cr) + ")");
            }
        }
    }
    for (int a = 0; a < 3; ++a) {
        char name[64];
        std::snprintf(name, sizeof(name), "lrx_conv_kminor_c128_ctg%d",
                      kCtgArms[a]);
        cr = api.ModuleGetFunction(&arms.ctg[a], mod, name);
        if (cr != CUDA_SUCCESS) {
            api.ModuleUnload(mod);
            return fail_sticky("cuModuleGetFunction",
                               std::string(name) + ": " + cu_err(cr));
        }
        if (smem_optin > 49152) {
            cr = api.FuncSetAttribute(
                arms.ctg[a],
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_optin);
            if (cr != CUDA_SUCCESS) {
                api.ModuleUnload(mod);
                return fail_sticky(
                    "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
                    std::string(name) + ": requested " +
                        std::to_string(smem_optin) + " B: CUresult=" +
                        std::to_string(static_cast<int>(cr)) + " (" +
                        cu_err(cr) + ")");
            }
        }
    }
    // Record the envelope only once every arm is loaded and raised, so a plan
    // can never be made against a limit the kernels do not actually have.
    arms.smem_max = (smem_optin > 49152) ? smem_optin : 49152;
    if (log_enabled()) {
        std::fprintf(stderr,
                     "[conv_kminor] NVRTC kernels compiled for sm_%d%d "
                     "(%zu B %s, device %d, %d strided + 3 contiguous arms); "
                     "dynamic shared max %d B -> largest k-row %d points\n",
                     cc_major, cc_minor, image.size(),
                     used_cubin ? "cubin" : "ptx", dev, kEptMax,
                     arms.smem_max, arms.smem_max / 16 - 2);
    }
    auto res = cache.emplace(ctx, arms);
    *out = &res.first->second;
    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
//  Launch geometry.  Derived from nk and the inner group size G = d3·d4;
//  nothing else.  See the file header for what each constant buys.
// ---------------------------------------------------------------------------
struct LaunchCfg {
    int rb = 1;      // rows per block   (blockDim.y)
    int tpr = 1;     // threads per row  (blockDim.x)
    int ept = 1;     // elements per thread   (strided arm)
    int sp = 1;      // shared row stride, elements (odd)
    int smem = 0;    // dynamic shared bytes
    // -- arm selection, decided here and nowhere else ----------------------
    bool contig = false;  //: take the contiguous-ownership arm
    int  own_c = 1;       //: its owned run length (the innermost non-trivial
                          //   axis extent); also selects the CMAX arm
    int  own_ax = 0;      //: which axis that is: 0=x, 1=y, 2=z
    int  ctg_arm = 0;     //: index into kCtgArms
};

//: The owned axis is the INNERMOST axis with extent > 1, and the run length is
//: its extent.  That choice is forced, not preferred: a thread's owned set has
//: to be contiguous in k for the load/store walks to stay coalesced, and the
//: only contiguous line in the layout k = (ix*n1 + iy)*n2 + iz is along the
//: innermost non-trivial axis (every axis inside it has extent 1, so it
//: contributes stride 1).  Returns the run length; sets *ax.
static int owned_axis(int n0, int n1, int n2, int* ax) {
    if (n2 > 1) { *ax = 2; return n2; }
    if (n1 > 1) { *ax = 1; return n1; }
    *ax = 0;
    return n0;
}

static bool plan_launch(int nk, int n0, int n1, int n2, long long group,
                        int smem_max, LaunchCfg* cfg, std::string* why) {
    const int ntw = n0 + n1 + n2;                // the three twiddle rings
    const int sp = nk | 1;                       // force odd: bank conflicts
    // Shared bytes for `r` rows: the tile (r·sp), the twiddle rings (ntw) and
    // the per-row metadata (r int64 pairs = r complex-sized slots).
    auto smem_for = [&](long long r) {
        return 16 * (r * (long long)(sp + 1) + (long long)ntw);
    };
    // Prefer the occupancy budget; grow to the DEVICE's opt-in maximum only
    // when a single row does not fit in it.  Occupancy is traded away exactly
    // for the grids that would otherwise be refused, and for no others.
    long long budget = kSmemPreferred;
    if (smem_for(1) > budget) budget = smem_max;
    long long rb = (budget / 16 - (long long)ntw) / (long long)(sp + 1);
    if (rb < 1) {
        const long long nk_max = (long long)smem_max / 16 - 2 - ntw;
        std::ostringstream os;
        os << "k-grid product nk=" << nk << " needs " << smem_for(1)
           << " B of shared memory to hold ONE k-row, and this device's "
              "maximum dynamic shared memory per block is " << smem_max
           << " B.  THE BOUND IS RESIDENCY, NOT SIZE CLASS: the whole k-row "
              "must stay live between the inverse and forward halves, which "
              "is what makes this ONE kernel and ONE HBM round trip instead "
              "of eight.  This device serves any (nkx,nky,nkz) whose PRODUCT "
              "is <= " << (nk_max > 0 ? nk_max : 0)
           << " points, primes and mixed radices included, with no per-size "
              "code path.  For a larger transform use the fused-conv family's "
              "k-STRIDED member (lorrax_mklfft_gw_conv), which is plan-based "
              "and has no residency bound — it needs a k-LEADING operand.";
        *why = os.str();
        return false;
    }
    const long long rb_cap = rb;      // the shared-memory ceiling on rows

    // ---------------------------------------------------------------------
    //  ARM SELECTION.  Prefer the CONTIGUOUS-ownership arm: it runs the owned
    //  axis's transform in registers, which removes that axis's shared-memory
    //  round trip AND its barrier pair from both halves of the chain.  It
    //  applies whenever the owned run fits a compiled CMAX and its FIXED
    //  threads-per-row (nk/C) fits the block cap; otherwise the strided arm
    //  below serves the same call, correctly, with more shared traffic.
    //  Selected HERE so exactly one place decides, and reported by the log
    //  line so a measurement can never be attributed to the wrong arm.
    // ---------------------------------------------------------------------
    {
        int ax = 0;
        const int C = owned_axis(n0, n1, n2, &ax);
        int arm = -1;
        for (int a = 0; a < 3; ++a) {
            if (C <= kCtgArms[a]) { arm = a; break; }
        }
        // C MUST BE ODD, and this is the same argument that forces SP = nk|1
        // one level down.  A thread's owned run starts at lane*C, so its
        // shared accesses have an ELEMENT stride of C between consecutive
        // lanes — 4*C words.  With C odd, gcd(4C, 32) = 4 and an eight-thread
        // phase covers all 32 banks: conflict-free.  With C even the stride
        // shares a factor of 8 or 16 with the bank count and the same accesses
        // land on two or four banks.  The strided arm does not have this
        // problem (its lanes walk consecutive k), so an even C trades a real
        // conflict for a nominal traffic saving.
        //
        // MEASURED, mu=nu=200, ffi/xla on the rung chain, contiguous vs
        // strided at each grid — the rule and the data agree at every point:
        //     3x3x1  C=3 odd    0.34  vs 0.40   contiguous WINS
        //     3x3x3  C=3 odd    0.27  vs 0.30   contiguous WINS
        //     4x4x1  C=4 even   0.29  vs 0.30   within noise
        //     4x4x2  C=2 even   0.30  vs 0.31   within noise
        //     4x4x4  C=4 even   0.37  vs 0.33   contiguous LOSES
        //     6x6x6  C=6 even   0.71  vs 0.49   contiguous LOSES badly
        // So the arm is taken on the cause (odd C), not on a size threshold,
        // and the two even cases it declines were the two it did not help.
        if (C >= 2 && (C & 1) == 1 && arm >= 0 && (nk % C) == 0) {
            const long long tpr_c = nk / C;
            if (tpr_c >= 1 && tpr_c <= kMaxBlock) {
                long long rb_c = std::min<long long>(rb_cap, kRowsMaxCtg);
                rb_c = std::min<long long>(rb_c, kMaxBlock / tpr_c);
                if (group > 0 && rb_c >= group) rb_c -= rb_c % group;
                if (rb_c >= 1) {
                    cfg->contig = true;
                    cfg->own_c = C;
                    cfg->own_ax = ax;
                    cfg->ctg_arm = arm;
                    cfg->rb = (int)rb_c;
                    cfg->tpr = (int)tpr_c;
                    cfg->ept = 1;              // unused by this arm
                    cfg->sp = sp;
                    cfg->smem = (int)smem_for(rb_c);
                    return true;
                }
            }
        }
    }

    rb = std::min<long long>(rb, kRowsMax);
    // Round DOWN to a multiple of the inner group so a block covers whole
    // (d3,d4) groups and the out_layout=1 store writes contiguous runs of
    // rb/group destination elements.
    if (group > 0 && rb >= group) rb -= rb % group;
    if (rb < 1) rb = 1;

    // Threads per row: enough that ept lands at or under kEptPref (the
    // 64-register arms), and enough that the block reaches kBlockTarget.  Then
    // PREFER A DIVISOR of nk — with tpr | nk every lane owns exactly ept
    // elements and the `k < nk` predicate in the six passes is always true, so
    // no lane idles through the transform.  A non-divisor is taken only when
    // the divisor would overrun the block cap (long prime nk), and then the
    // ragged tail is the honest cost of that shape.
    long long tpr_want = (nk + kEptPref - 1) / kEptPref;
    if (tpr_want * rb < kBlockTarget) {
        tpr_want = std::max<long long>(tpr_want, kBlockTarget / rb);
    }
    tpr_want = std::min<long long>(std::max<long long>(tpr_want, 1), nk);
    // Shrinking ROWS to afford the divisor is the right trade: a divisor tpr
    // removes the ragged tail from all six passes, while rb only costs a
    // little block size and a shorter (still >= 128 B) store run.
    auto fit_block = [&](long long t, long long r) {
        while (t * r > kMaxBlock && r > 1) {
            r -= 1;
            if (group > 0 && r >= group) r -= r % group;
        }
        return r;
    };
    long long tpr = 0;
    for (long long t = tpr_want; t <= nk; ++t) {
        if (nk % t == 0) { tpr = t; break; }
    }
    if (tpr != 0) {
        const long long rb_div = fit_block(tpr, rb);
        if (tpr * rb_div <= kMaxBlock) rb = rb_div;
        else tpr = 0;
    }
    if (tpr == 0) tpr = tpr_want;          // no divisor fits: ragged tail
    rb = fit_block(tpr, rb);
    if (tpr * rb > kMaxBlock) {
        std::ostringstream os;
        os << "nk=" << nk << " needs " << tpr
           << " threads per row, over the " << kMaxBlock
           << "-thread block cap this kernel's register budget is built for.";
        *why = os.str();
        return false;
    }
    const long long ept = (nk + tpr - 1) / tpr;
    if (ept > kEptMax) {
        std::ostringstream os;
        os << "internal: ept=" << ept << " exceeds kEptMax=" << kEptMax
           << " at nk=" << nk << " tpr=" << tpr;
        *why = os.str();
        return false;
    }
    cfg->rb = (int)rb;
    cfg->tpr = (int)tpr;
    cfg->ept = (int)ept;
    cfg->sp = sp;
    // tile (rb·sp complex) + twiddle rings (ntw complex) + the per-row
    // metadata table (2·rb int64: the W row offset and the out_layout=1
    // destination base — see the kernel's PER-ROW METADATA block).
    cfg->smem = (int)smem_for(rb);
    if (cfg->smem > smem_max) {
        // Unreachable by construction (rb was derived from this bound), so it
        // is an INTERNAL error rather than a user-facing size refusal, and it
        // says so — a plan that disagrees with its own budget is a defect in
        // this function, not a k-grid the caller should change.
        std::ostringstream os;
        os << "internal: planned " << cfg->smem << " B of dynamic shared "
              "memory at nk=" << nk << " rb=" << rb << ", over this device's "
           << smem_max << " B maximum.";
        *why = os.str();
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
//  The handler.
// ---------------------------------------------------------------------------
static ffi::Error ConvKMinorDispatch(
    cudaStream_t stream,
    ffi::AnyBuffer T, ffi::AnyBuffer W, ffi::Result<ffi::AnyBuffer> U,
    int64_t nkx, int64_t nky, int64_t nkz, double scale, int64_t out_layout)
{
    // ---- dtype: c128 ONLY, refused BY NAME (never demoted) ----
    if (T.element_type() != ffi::DataType::C128 ||
        W.element_type() != ffi::DataType::C128 ||
        U->element_type() != ffi::DataType::C128) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "conv_kminor: complex128 only — this handler implements no other "
            "element type and does NOT up-cast.  A complex64 caller (e.g. the "
            "fp32-GMRES ladder arm) must refuse here, not silently change the "
            "arithmetic it is measuring.");
    }
    auto td = T.dimensions();
    auto wd = W.dimensions();
    auto ud = U->dimensions();
    if (td.size() != 6 || wd.size() != 3 || ud.size() != 6) {
        std::ostringstream os;
        os << "conv_kminor: expected T rank 6 (d0,d1,d2,d3,d4,nk), W rank 3 "
              "(d1,d2,nk), U rank 6; got ranks " << td.size() << "/"
           << wd.size() << "/" << ud.size() << ".";
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    const int64_t d0 = td[0], d1 = td[1], d2 = td[2], d3 = td[3], d4 = td[4];
    const int64_t nk = td[5];
    if (nkx < 1 || nky < 1 || nkz < 1 || nk != nkx * nky * nkz) {
        std::ostringstream os;
        os << "conv_kminor: minor extent " << nk << " != nkx*nky*nkz = "
           << nkx << "*" << nky << "*" << nkz << ".";
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    if (wd[0] != d1 || wd[1] != d2 || wd[2] != nk) {
        std::ostringstream os;
        os << "conv_kminor: W must be (d1,d2,nk) = (" << d1 << "," << d2 << ","
           << nk << "); got (" << wd[0] << "," << wd[1] << "," << wd[2] << ").";
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    int64_t want[6];
    if (out_layout == 0) {
        want[0] = d0; want[1] = d1; want[2] = d2;
        want[3] = d3; want[4] = d4; want[5] = nk;
    } else if (out_layout == 1) {
        want[0] = d0; want[1] = nk; want[2] = d3;
        want[3] = d1; want[4] = d4; want[5] = d2;
    } else {
        std::ostringstream os;
        os << "conv_kminor: out_layout must be 0 (same as T) or 1 "
              "(d0,nk,d3,d1,d4,d2); got " << out_layout << ".";
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    for (int i = 0; i < 6; ++i) {
        if (ud[i] != want[i]) {
            std::ostringstream os;
            os << "conv_kminor: out_layout=" << out_layout
               << " requires U of shape (" << want[0] << "," << want[1] << ","
               << want[2] << "," << want[3] << "," << want[4] << ","
               << want[5] << "); got (" << ud[0] << "," << ud[1] << ","
               << ud[2] << "," << ud[3] << "," << ud[4] << "," << ud[5] << ").";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
    const int64_t R = d0 * d1 * d2 * d3 * d4;
    if (R == 0 || nk == 0) return ffi::Error::Success();   // nothing to do

    // ORDER MATTERS: the kernels are loaded FIRST because loading them is
    // what establishes this device's shared-memory envelope, and the launch
    // plan — including whether this k-grid can be served at all — is derived
    // from that envelope.  Planning first would have to guess it.
    const KernelArms* arms = nullptr;
    ffi::Error e = ffi::Error::Success();
    {
        std::lock_guard<std::mutex> lock(g_mu);
        e = ensure_kernels(&arms);
    }
    if (!e.success()) return e;

    LaunchCfg cfg;
    std::string why;
    if (!plan_launch((int)nk, (int)nkx, (int)nky, (int)nkz, d3 * d4,
                     arms->smem_max, &cfg, &why)) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "conv_kminor: " + why);
    }
    CUfunction fn = cfg.contig ? arms->ctg[cfg.ctg_arm]
                               : arms->fn[cfg.ept - 1];

    const auto* t_in = static_cast<const double*>(T.untyped_data());
    const auto* w_in = static_cast<const double*>(W.untyped_data());
    auto* u_out = static_cast<double*>(U->untyped_data());

    if (log_enabled()) {
        static std::atomic<bool> once{false};
        if (!once.exchange(true)) {
            std::fprintf(stderr,
                         "[conv_kminor] first call: kgrid=(%lld,%lld,%lld) "
                         "nk=%lld T=(%lld,%lld,%lld,%lld,%lld) rows=%lld "
                         "out_layout=%lld scale=%.9e | ARM=%s rb=%d tpr=%d "
                         "ept/C=%d sp=%d smem=%d B grid=%lld inplace=%d\n",
                         (long long)nkx, (long long)nky, (long long)nkz,
                         (long long)nk, (long long)d0, (long long)d1,
                         (long long)d2, (long long)d3, (long long)d4,
                         (long long)R, (long long)out_layout, scale,
                         cfg.contig ? "contiguous" : "strided",
                         cfg.rb, cfg.tpr,
                         cfg.contig ? cfg.own_c : cfg.ept, cfg.sp, cfg.smem,
                         (long long)((R + cfg.rb - 1) / cfg.rb),
                         (int)(static_cast<const void*>(t_in) ==
                               static_cast<const void*>(u_out)));
        }
    }

    long long a_R = R, a_MN = d1 * d2;
    long long a_d1 = d1, a_d2 = d2, a_d3 = d3, a_d4 = d4;
    int a_nk = (int)nk, a_n0 = (int)nkx, a_n1 = (int)nky, a_n2 = (int)nkz;
    // NOTE neither arm takes an `ept`/`CMAX` argument: it is the ENTRY POINT's
    // identity (lrx_conv_kminor_c128_e<ept> / _ctg<CMAX>), so the plan and the
    // kernel cannot disagree about it.  The contiguous arm does take the run
    // length C and the owned axis, because those are runtime facts about the
    // k-grid rather than compile-time capacities.
    int a_sp = cfg.sp, a_layout = (int)out_layout;
    int a_C = cfg.own_c, a_oa = cfg.own_ax;
    double a_scale = scale;
    void* args_strided[] = {(void*)&t_in, (void*)&w_in, (void*)&u_out,
                            &a_R, &a_MN, &a_d1, &a_d2, &a_d3, &a_d4,
                            &a_nk, &a_n0, &a_n1, &a_n2,
                            &a_sp, &a_scale, &a_layout};
    void* args_contig[] = {(void*)&t_in, (void*)&w_in, (void*)&u_out,
                           &a_R, &a_MN, &a_d1, &a_d2, &a_d3, &a_d4,
                           &a_nk, &a_n0, &a_n1, &a_n2,
                           &a_sp, &a_C, &a_oa, &a_scale, &a_layout};
    void** args = cfg.contig ? args_contig : args_strided;

    const long long nblocks = (R + cfg.rb - 1) / cfg.rb;
    if (nblocks > 2147483647LL) {
        std::ostringstream os;
        os << "conv_kminor: " << nblocks
           << " blocks exceeds the grid.x limit; split the leading axis.";
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    CUresult cr = driver_api().LaunchKernel(
        fn, (unsigned)nblocks, 1, 1,
        (unsigned)cfg.tpr, (unsigned)cfg.rb, 1,
        (unsigned)cfg.smem, reinterpret_cast<CUstream>(stream), args, nullptr);
    if (cr != CUDA_SUCCESS) {
        return fail("cuLaunchKernel", cu_err(cr));
    }
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::conv_kminor

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CufftConvKMinorCudaFfi,
    lorrax_ffi::conv_kminor::ConvKMinorDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()      // T (d0,d1,d2,d3,d4,nk) c128
        .Arg<xla::ffi::AnyBuffer>()      // W (d1,d2,nk) c128, R-space already
        .Ret<xla::ffi::AnyBuffer>()      // U (may alias T when out_layout=0)
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")           // ONE folded constant (both norms)
        .Attr<int64_t>("out_layout"));   // 0 = shape(T); 1 = (d0,nk,d3,d1,d4,d2)
