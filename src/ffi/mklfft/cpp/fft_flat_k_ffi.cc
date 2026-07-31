// fft_flat_k_ffi.cc — flat-k batched 3-D MKL FFT (DFTI API) host handlers,
// HOST platform (JAX CPU backend).  PROTOTYPE (FFT-FFI workstream,
// 2026-07-28) — gated behind LORRAX_FFT_FFI / LORRAX_FFT_FFI_FUSED on the
// Python side (common/fft_helpers.py); the default XLA path is untouched.
//
// WHAT THIS IS: MKL's FFT engine driven through the DFTI descriptor API —
// an O(N log N) fast Fourier transform at ANY k-count (mixed radix,
// arbitrary lengths).  It is NOT a DFT-as-matmul (that formulation is
// owner-vetoed); "DFTI" is Intel's descriptor-API name for its FFT, and
// every transform below is a genuine FFT.
//
// WHY IT EXISTS: XLA:CPU's fft custom-call (DUCC) requires the transformed
// axes minor-most, while the Σ τ-kernel's producers/consumers (G-build
// einsum, G·W multiply, ψ-projection dots) hold the tile in flat-k
// "dot layout" — k-major: (nk, s, μ_X, s', μ_Y), c128[16,2,624,2,624]
// (~398 MB/rank at nb=128/P=64).  XLA therefore transposes the full tile
// to c128[2,624,2,624,4,4,1] before EVERY fft and back after — measured
// at ~60-65% of the 1.51 s/τ (G_ifft 79.6 + GW_mult_fft 95.7 + V_ifft
// 16.5 = 192 s of 272 s sigma.exec; closure in wk_REL/sigma_perf_results.md:
// the layout anchor is the fft custom-call itself, present in ANY XLA-side
// re-arrangement).  MKL FFT (DFTI API) has no minor-most requirement:
// STRIDE DESCRIPTORS read the dot-layout tile exactly where it lies —
// FFT-axis element strides {nky*nkz*T, nkz*T, T}, batch of T transforms at
// DISTANCE 1 along the unit-stride trail — so the layout copies vanish
// instead of being moved around.
//
// Handlers (registered by ffi_loader.py under platform="cpu"):
//   MklFftFlatKHostFfi  (target lorrax_mklfft_flat_k)
//       X (nk, *trail) c128  ->  Y same shape.  One batched 3-D FFT over
//       the LEADING flat-k axis; direction + total scale are attributes,
//       so the jnp.fft norm conventions ('ortho'/'backward'/'forward')
//       live in ONE place, the Python helper.
//   MklFftGwConvHostFfi (target lorrax_mklfft_gw_conv)
//       G (nk, a, mx, b, my) c128, W (nk, mx, my) c128 -> S = shape(G).
//       The fused Σ τ convolution step
//           S = FFT[ IFFT[G] * IFFT[W][:,None,:,None,:] * mult ]
//       chunked over the trail so the big R-space intermediate NEVER
//       materializes (per-thread compact buffer, ~4 MiB); `mult` is folded
//       into the forward scale by the Python wrapper.
//
// In-place: the Python wrappers alias operand 0 to the result
// (input_output_aliases={0:0}).  When XLA grants the alias the transform
// runs DFTI_INPLACE in the operand buffer — the terminal form of buffer
// donation (zero extra tiles); when XLA must keep the operand alive it
// passes distinct buffers and the handler runs DFTI_NOT_INPLACE.  Both
// paths are exercised by the unit gate.
//
// Threading: the handler parallelizes its CHUNK loop with OpenMP (this TU
// is compiled -fopenmp; the .so already links gomp via mkl_gnu_thread) and
// pins MKL to ONE thread inside each team member — the MklThreadScope
// pattern from scalapack/cpp/blacs_grid.h (workstream AW), locally
// duplicated below WITHOUT the mpi.h/slate deps so this TU stays
// comms-free; fold into a shared header if the prototype graduates.
// Team size: LORRAX_MKLFFT_THREADS (auto/off/N, strict grammar per the AW
// audit fix) — parsed per call (cheap vs the >=10 ms transforms) so the
// unit gate can sweep it in-process; the measured cap policy is recorded
// in wk_REL/ffi_fft_proto_notes.md.
//
// Envelope-honesty: every extent and stride is taken from the runtime
// buffer dimensions / attributes; nothing is specialized to a deck.  The
// FFT is batched over whatever trail the caller shards onto this rank, so
// no N_mu^2 global tile is ever required (LORRAX scaling target).

#include <algorithm>
#include <atomic>
#include <cctype>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <dlfcn.h>
#include <omp.h>

#include <mkl_dfti.h>

#include "../../common/cpp/mkl_thread_pin.h"
#include "xla/ffi/api/ffi.h"

namespace lorrax_ffi::mklfft {

namespace ffi = ::xla::ffi;
using C128 = std::complex<double>;

// ---------------------------------------------------------------------------
//  MKL thread pinning (workstream-AW pattern).  The mechanism lives in
//  common/cpp/mkl_thread_pin.h — an MPI-free, CUDA-free header, which is why
//  it is not scalapack/cpp/blacs_grid.h (that one includes <mpi.h> at :40 and
//  this TU must stay comms-free).
//
//  This file previously held its own copy that resolved the pin with a bare
//  `dlsym(RTLD_DEFAULT, ...)` while the mklblas copy used RTLD_DEFAULT then
//  RTLD_NEXT.  Under a local-scope dlopen the GEMM handler pinned MKL and
//  this one silently did not (2026-07-30 divergence audit).  There is now one
//  resolver; `MklLocalPin` is the old local name for what is now
//  `mklpin::MklThreadScope`, kept so the two pin sites below read unchanged.
// ---------------------------------------------------------------------------
using MklLocalPin = mklpin::MklThreadScope;
using mklpin::str_ieq;

// OpenMP team size for the chunk loop.  Strict full-string grammar (the
// blacs_grid.h AW-audit lesson: a typo must not silently pick a known-bad
// policy).  "auto" (default) = the ambient omp_get_max_threads() — the
// production harness exports OMP_NUM_THREADS per rank; whether a smaller
// cap wins (the owner's 6-way-cap question) is a measurement recorded in
// the proto notes, and the unit gate sweeps this knob per call.
static int team_threads() {
    const int maxt = std::max(1, omp_get_max_threads());
    const char* v = std::getenv("LORRAX_MKLFFT_THREADS");
    if (!v || !*v || str_ieq(v, "auto")) return maxt;
    if (str_ieq(v, "off")) return 1;
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end != v && *end == '\0' && parsed >= 1 && parsed <= 4096) {
        return static_cast<int>(parsed);
    }
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
        std::fprintf(
            stderr,
            "*** LORRAX_MKLFFT_THREADS='%s' is not a recognized value "
            "(accepted, case-insensitive: 'auto', 'off', or a positive "
            "integer <= 4096).  Falling back to 'auto' (%d threads). ***\n",
            v, maxt);
    }
    return maxt;
}

// Trail elements per chunk.  Default: size the per-thread compact buffer
// (nk · chunk · 16 B) to ~512 KiB so it stays resident in a CLX core's
// 1 MiB L2 — the dimension-by-dimension radix passes of the small 3-D FFT
// then run in cache and the big tile is streamed ONCE per direction
// (measured: the strided→strided form, whose radix passes all stream the
// full tile, ran 2.8× slower single-thread).  LORRAX_MKLFFT_CHUNK
// overrides for experiments; the unit gate sweeps it and exercises ragged
// chunks explicitly.
static int64_t chunk_elems(int64_t nk) {
    const char* v = std::getenv("LORRAX_MKLFFT_CHUNK");
    if (v && *v) {
        char* end = nullptr;
        const long long parsed = std::strtoll(v, &end, 10);
        if (end != v && *end == '\0' && parsed >= 1 &&
            parsed <= (1LL << 30)) {
            return static_cast<int64_t>(parsed);
        }
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            std::fprintf(stderr,
                         "*** LORRAX_MKLFFT_CHUNK='%s' unrecognized — using "
                         "the auto policy. ***\n",
                         v);
        }
    }
    const int64_t c = (512LL * 1024) / (16 * std::max<int64_t>(nk, 1));
    return std::max<int64_t>(c, 64);
}

// Opt-in debug logging.  Presence-tested as before, but now rank-scoped:
// rank 0 by default, every rank with LORRAX_MKLFFT_LOG=all.  See the
// log_here() note in common/cpp/mkl_thread_pin.h for why the default
// flipped — at P=1000 the all-ranks default multiplied a 5-line answer by
// the process count, and (at the commit site below) by the OpenMP team size
// on top of that.
static bool log_enabled() {
    static const bool on = mklpin::log_here("LORRAX_MKLFFT_LOG");
    return on;
}

// ---------------------------------------------------------------------------
//  DFTI descriptor cache — per thread (sidesteps any question of concurrent
//  DftiCompute* on one handle; each OpenMP team member owns its handles).
//  A handful of keys per process: (dims, in/out trail strides, transforms
//  per call, scales, placement).  Handles live for the process.
// ---------------------------------------------------------------------------
struct DescKey {
    MKL_LONG d0, d1, d2;
    MKL_LONG t_in, t_out;  // trail-stride multiplier (elements) in/out
    MKL_LONG n;            // transforms per compute call
    double sf, sb;         // forward / backward scale
    int inplace;

    bool operator<(const DescKey& o) const {
        return std::tie(d0, d1, d2, t_in, t_out, n, sf, sb, inplace) <
               std::tie(o.d0, o.d1, o.d2, o.t_in, o.t_out, o.n, o.sf, o.sb,
                        o.inplace);
    }
};

static bool dfti_ok(MKL_LONG status) {
    return status == 0 || DftiErrorClass(status, DFTI_NO_ERROR);
}

// Create+commit (or fetch) the descriptor for one chunk geometry.  The
// batch rides the unit-stride trail: NUMBER_OF_TRANSFORMS=n at DISTANCE 1,
// FFT-axis strides {d1*d2*T, d2*T, T} — the documented MKL "transform
// along a non-minor axis" layout (one-to-one because n <= T).  Both scales
// are set so one handle serves forward and backward computes.
static MKL_LONG get_descriptor(const DescKey& k, DFTI_DESCRIPTOR_HANDLE* out) {
    static thread_local std::map<DescKey, DFTI_DESCRIPTOR_HANDLE> cache;
    auto it = cache.find(k);
    if (it != cache.end()) {
        *out = it->second;
        return 0;
    }
    DFTI_DESCRIPTOR_HANDLE h = nullptr;
    MKL_LONG dims3[3] = {k.d0, k.d1, k.d2};
    MKL_LONG st = DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 3, dims3);
    if (!dfti_ok(st)) return st;
    MKL_LONG istr[4] = {0, k.d1 * k.d2 * k.t_in, k.d2 * k.t_in, k.t_in};
    MKL_LONG ostr[4] = {0, k.d1 * k.d2 * k.t_out, k.d2 * k.t_out, k.t_out};
    st = DftiSetValue(h, DFTI_PLACEMENT,
                      k.inplace ? DFTI_INPLACE : DFTI_NOT_INPLACE);
    if (dfti_ok(st)) st = DftiSetValue(h, DFTI_INPUT_STRIDES, istr);
    if (dfti_ok(st) && !k.inplace)
        st = DftiSetValue(h, DFTI_OUTPUT_STRIDES, ostr);
    if (dfti_ok(st)) st = DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, k.n);
    if (dfti_ok(st)) st = DftiSetValue(h, DFTI_INPUT_DISTANCE, (MKL_LONG)1);
    if (dfti_ok(st) && !k.inplace)
        st = DftiSetValue(h, DFTI_OUTPUT_DISTANCE, (MKL_LONG)1);
    if (dfti_ok(st)) st = DftiSetValue(h, DFTI_FORWARD_SCALE, k.sf);
    if (dfti_ok(st)) st = DftiSetValue(h, DFTI_BACKWARD_SCALE, k.sb);
    // One MKL thread per handle: the chunk loop is the parallel dimension.
    if (dfti_ok(st)) st = DftiSetValue(h, DFTI_THREAD_LIMIT, (MKL_LONG)1);
    if (dfti_ok(st)) st = DftiCommitDescriptor(h);
    if (!dfti_ok(st)) {
        if (h) DftiFreeDescriptor(&h);
        return st;
    }
    // ONE line per distinct geometry per PROCESS.  The descriptor cache
    // above is `thread_local`, so this site is reached once per (thread,
    // key): before the 2026-07-30 audit the log was multiplied by the
    // OpenMP team size (28/rank in production) and then again by the
    // process count.  The diagnostic question this line answers — "which
    // descriptor geometries did this process build?" — is a per-process
    // property, so a process-wide seen-set preserves the whole answer and
    // drops the multiplier.  The thread number is kept and now names the
    // thread that FIRST committed the geometry.
    if (log_enabled()) {
        static std::mutex seen_mu;
        static std::map<DescKey, char> seen;
        bool first = false;
        {
            std::lock_guard<std::mutex> g(seen_mu);
            first = seen.emplace(k, '\0').second;
        }
        if (first) {
            std::fprintf(stderr,
                         "[mklfft] commit dims=(%ld,%ld,%ld) t_in=%ld "
                         "t_out=%ld n=%ld sf=%.3e sb=%.3e inplace=%d "
                         "(first committed by thread %d)\n",
                         (long)k.d0, (long)k.d1, (long)k.d2, (long)k.t_in,
                         (long)k.t_out, (long)k.n, k.sf, k.sb, k.inplace,
                         omp_get_thread_num());
        }
    }
    cache.emplace(k, h);
    *out = h;
    return 0;
}

static MKL_LONG compute(DFTI_DESCRIPTOR_HANDLE h, bool forward, bool inplace,
                        const C128* in, C128* out) {
    // DFTI_NOT_INPLACE never writes the input buffer, so the const_cast is
    // safe for the XLA (read-only) operand; DFTI_INPLACE is only selected
    // when XLA granted the output alias (in == out).
    void* pin = const_cast<void*>(static_cast<const void*>(in));
    if (inplace) {
        return forward ? DftiComputeForward(h, static_cast<void*>(out))
                       : DftiComputeBackward(h, static_cast<void*>(out));
    }
    return forward ? DftiComputeForward(h, pin, static_cast<void*>(out))
                   : DftiComputeBackward(h, pin, static_cast<void*>(out));
}

// Aggregate the first DFTI failure out of an OpenMP region.
struct ErrSink {
    std::atomic<bool> failed{false};
    std::mutex mu;
    std::string msg;

    void record(MKL_LONG status, const char* where) {
        if (failed.exchange(true)) return;
        std::lock_guard<std::mutex> lock(mu);
        std::ostringstream os;
        os << "mklfft (MKL FFT via the DFTI API): " << where
           << " failed, status=" << (long)status << " — "
           << DftiErrorMessage(status);
        msg = os.str();
    }
};

// The batched-FFT engine: transform `T` trail elements of `in` -> `out`
// (both in the strided flat-k layout with trail stride multiplier T),
// chunked over the trail across an OpenMP team.
//
// Each chunk transforms strided-input -> a per-thread COMPACT buffer
// (trail stride = chunk), then scatter-copies the nk contiguous k-lines
// to the strided output.  Rationale (measured, unit gate 7878708): the
// small 3-D FFT is computed dimension-by-dimension, so a strided→strided
// transform streams the ~400 MB tile once per radix pass (161.7 ms at 28
// threads, SLOWER than XLA's transpose+DUCC), while the compact form
// keeps those passes in the per-core L2 and streams the big tile once
// in + once out (the fused-conv path already worked this way and ran
// ~2x faster per transform).  The extra cost is one L2->memory copy of
// the chunk — 16 contiguous memcpys per chunk.
// The `in == out` (XLA granted the alias) case needs no DFTI_INPLACE
// here: the chunk's k-lines are read into the compact buffer before the
// scatter-copy rewrites exactly those locations.
static ffi::Error run_flat_batch(
    MKL_LONG d0, MKL_LONG d1, MKL_LONG d2, int64_t T, bool forward,
    double scale, bool inplace, const C128* in, C128* out)
{
    (void)inplace;  // handled by construction (see above)
    const int64_t nk = (int64_t)d0 * d1 * d2;
    const int64_t C = std::min<int64_t>(chunk_elems(nk), T);
    const int64_t n_chunks = (T + C - 1) / C;
    const int nthr = static_cast<int>(
        std::min<int64_t>(team_threads(), n_chunks));
    // 'ortho'/'backward'/'forward' totals arrive pre-folded from Python;
    // the same value is planted on the direction actually computed.
    const double sf = forward ? scale : 1.0;
    const double sb = forward ? 1.0 : scale;
    ErrSink err;

#pragma omp parallel num_threads(nthr)
    {
        MklLocalPin pin(1);  // chunk loop is the parallel dimension
        static thread_local std::vector<C128> tl_buf;
        if ((int64_t)tl_buf.size() < nk * C) tl_buf.resize(nk * C);
        C128* buf = tl_buf.data();
#pragma omp for schedule(dynamic)
        for (int64_t ci = 0; ci < n_chunks; ++ci) {
            if (err.failed.load(std::memory_order_relaxed)) continue;
            const int64_t t0 = ci * C;
            const MKL_LONG c = static_cast<MKL_LONG>(
                std::min<int64_t>(C, T - t0));
            DescKey key{d0, d1, d2, (MKL_LONG)T, c, c, sf, sb, 0};
            DFTI_DESCRIPTOR_HANDLE h = nullptr;
            MKL_LONG st = get_descriptor(key, &h);
            if (!dfti_ok(st)) { err.record(st, "descriptor"); continue; }
            st = compute(h, forward, /*inplace=*/false, in + t0, buf);
            if (!dfti_ok(st)) { err.record(st, "compute"); continue; }
            for (int64_t k = 0; k < nk; ++k) {
                std::memcpy(out + k * T + t0, buf + k * c,
                            (size_t)c * sizeof(C128));
            }
        }
    }
    if (err.failed.load()) {
        return ffi::Error(ffi::ErrorCode::kInternal, err.msg);
    }
    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
//  Handler 1: plain batched flat-k FFT/IFFT.
// ---------------------------------------------------------------------------
static ffi::Error FlatKDispatch(
    ffi::AnyBuffer X, ffi::Result<ffi::AnyBuffer> Y,
    int64_t nkx, int64_t nky, int64_t nkz, int64_t forward, double scale)
{
    if (X.element_type() != ffi::DataType::C128 ||
        Y->element_type() != ffi::DataType::C128) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "mklfft.flat_k: buffers must be complex128");
    }
    auto dims = X.dimensions();
    auto odims = Y->dimensions();
    if (dims.size() < 1 || dims.size() != odims.size() ||
        !std::equal(dims.begin(), dims.end(), odims.begin())) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "mklfft.flat_k: output shape must equal input shape");
    }
    const int64_t nk = dims[0];
    if (nkx < 1 || nky < 1 || nkz < 1 || nk != nkx * nky * nkz) {
        std::ostringstream os;
        os << "mklfft.flat_k: leading flat-k extent " << nk
           << " != nkx*nky*nkz = " << nkx << "*" << nky << "*" << nkz;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    int64_t T = 1;
    for (size_t i = 1; i < dims.size(); ++i) T *= dims[i];
    if (T == 0) return ffi::Error::Success();  // empty trail: nothing to do

    const auto* in = static_cast<const C128*>(X.untyped_data());
    auto* out = static_cast<C128*>(Y->untyped_data());
    const bool inplace = (static_cast<const void*>(in) ==
                          static_cast<const void*>(out));
    if (log_enabled()) {
        static std::atomic<bool> once{false};
        if (!once.exchange(true)) {
            std::fprintf(stderr,
                         "[mklfft] flat_k first call: nk=(%ld,%ld,%ld) T=%ld "
                         "fwd=%ld scale=%.6e inplace=%d threads=%d chunk=%ld\n",
                         (long)nkx, (long)nky, (long)nkz, (long)T,
                         (long)forward, scale, (int)inplace, team_threads(),
                         (long)chunk_elems(nk));
        }
    }
    return run_flat_batch((MKL_LONG)nkx, (MKL_LONG)nky, (MKL_LONG)nkz, T,
                          forward != 0, scale, inplace, in, out);
}

// ---------------------------------------------------------------------------
//  Handler 2: fused Σ τ convolution  S = FFT[ IFFT[G] · IFFT[W](bcast) ].
//  The R-space G tile exists only as per-thread ~4 MiB compact chunks; the
//  IFFT'd W (V_R, the small (nk, mx, my) tile) is staged once per call in a
//  reused process arena (malloc'd, invisible to the XLA planner — same
//  class as the scalapack workspace, logged under LORRAX_MKLFFT_LOG).
// ---------------------------------------------------------------------------
struct Arena {
    std::mutex mu;
    std::vector<C128> buf;
};

static Arena& vr_arena() {
    static Arena a;
    return a;
}

static ffi::Error GwConvDispatch(
    ffi::AnyBuffer G, ffi::AnyBuffer W, ffi::Result<ffi::AnyBuffer> S,
    int64_t nkx, int64_t nky, int64_t nkz, double scale_i, double scale_f)
{
    if (G.element_type() != ffi::DataType::C128 ||
        W.element_type() != ffi::DataType::C128 ||
        S->element_type() != ffi::DataType::C128) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "mklfft.gw_conv: buffers must be complex128");
    }
    auto gd = G.dimensions();
    auto wd = W.dimensions();
    auto sd = S->dimensions();
    if (gd.size() != 5 || wd.size() != 3 || sd.size() != 5 ||
        !std::equal(gd.begin(), gd.end(), sd.begin())) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "mklfft.gw_conv: expected G/S (nk, a, mx, b, my) and W (nk, mx, my)");
    }
    const int64_t nk = gd[0], a = gd[1], mx = gd[2], b = gd[3], my = gd[4];
    if (nkx < 1 || nky < 1 || nkz < 1 || nk != nkx * nky * nkz ||
        wd[0] != nk || wd[1] != mx || wd[2] != my) {
        std::ostringstream os;
        os << "mklfft.gw_conv: shape mismatch — G(nk=" << nk << ",a=" << a
           << ",mx=" << mx << ",b=" << b << ",my=" << my << ") vs W("
           << wd[0] << "," << wd[1] << "," << wd[2] << ") vs kgrid ("
           << nkx << "," << nky << "," << nkz << ")";
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    const int64_t Tg = a * mx * b * my;
    const int64_t Tv = mx * my;
    if (Tg == 0 || Tv == 0) return ffi::Error::Success();

    const auto* g_in = static_cast<const C128*>(G.untyped_data());
    const auto* w_in = static_cast<const C128*>(W.untyped_data());
    auto* s_out = static_cast<C128*>(S->untyped_data());
    const bool aliased = (static_cast<const void*>(g_in) ==
                          static_cast<const void*>(s_out));

    const MKL_LONG d0 = (MKL_LONG)nkx, d1 = (MKL_LONG)nky, d2 = (MKL_LONG)nkz;

    // --- stage 1: V_R = IFFT[W] into the arena (strided, trail stride Tv).
    Arena& ar = vr_arena();
    std::unique_lock<std::mutex> ar_lock(ar.mu);  // serializes concurrent convs
    if ((int64_t)ar.buf.size() < nk * Tv) ar.buf.resize(nk * Tv);
    C128* vr = ar.buf.data();
    {
        ffi::Error e = run_flat_batch(d0, d1, d2, Tv, /*forward=*/false,
                                      scale_i, /*inplace=*/false, w_in, vr);
        if (!e.success()) return e;
    }

    // --- stage 2: chunked IFFT(G) -> multiply by V_R (broadcast over a, b)
    //              -> FFT back into S, per ~4 MiB compact buffer.
    const int64_t C = std::min<int64_t>(chunk_elems(nk), Tg);
    const int64_t n_chunks = (Tg + C - 1) / C;
    const int nthr = static_cast<int>(
        std::min<int64_t>(team_threads(), n_chunks));
    ErrSink err;

    if (log_enabled()) {
        static std::atomic<bool> once{false};
        if (!once.exchange(true)) {
            std::fprintf(stderr,
                         "[mklfft] gw_conv first call: nk=(%ld,%ld,%ld) "
                         "G trail (a=%ld,mx=%ld,b=%ld,my=%ld) Tg=%ld Tv=%ld "
                         "scale_i=%.6e scale_f=%.6e aliased=%d threads=%d "
                         "chunk=%ld arena=%.1f MB\n",
                         (long)nkx, (long)nky, (long)nkz, (long)a, (long)mx,
                         (long)b, (long)my, (long)Tg, (long)Tv, scale_i,
                         scale_f, (int)aliased, nthr, (long)C,
                         nk * Tv * 16.0 / 1e6);
        }
    }

#pragma omp parallel num_threads(nthr)
    {
        MklLocalPin pin(1);
        // Compact chunk buffer, cached across calls (XLA host-callback
        // threads are a stable pool, so this settles after the first τ).
        static thread_local std::vector<C128> tl_buf;
        if ((int64_t)tl_buf.size() < nk * C) tl_buf.resize(nk * C);
        C128* buf = tl_buf.data();

#pragma omp for schedule(dynamic)
        for (int64_t ci = 0; ci < n_chunks; ++ci) {
            if (err.failed.load(std::memory_order_relaxed)) continue;
            const int64_t t0 = ci * C;
            const MKL_LONG c = static_cast<MKL_LONG>(
                std::min<int64_t>(C, Tg - t0));

            // (a) backward: strided G chunk -> compact buffer (scale_i).
            DescKey kb{d0, d1, d2, (MKL_LONG)Tg, c, c, 1.0, scale_i, 0};
            DFTI_DESCRIPTOR_HANDLE hb = nullptr;
            MKL_LONG st = get_descriptor(kb, &hb);
            if (!dfti_ok(st)) { err.record(st, "conv bwd descriptor"); continue; }
            st = compute(hb, /*forward=*/false, /*inplace=*/false,
                         g_in + t0, buf);
            if (!dfti_ok(st)) { err.record(st, "conv bwd compute"); continue; }

            // (b) multiply by V_R with the (a, b) broadcast: trail index
            //     t = ((ai*mx + x)*b + bi)*my + y maps to V trail x*my + y.
            //     Walk contiguous y-segments so the V reads stay streamed.
            {
                int64_t rem = t0;
                int64_t y = rem % my; rem /= my;
                int64_t bi = rem % b; rem /= b;
                int64_t x = rem % mx;
                int64_t j = 0;
                while (j < c) {
                    const int64_t seg = std::min<int64_t>(my - y, c - j);
                    const int64_t vrow = x * my + y;
                    for (int64_t k = 0; k < nk; ++k) {
                        C128* bp = buf + k * c + j;
                        const C128* vp = vr + k * Tv + vrow;
                        for (int64_t s2 = 0; s2 < seg; ++s2) bp[s2] *= vp[s2];
                    }
                    j += seg;
                    y += seg;
                    if (y == my) {
                        y = 0;
                        if (++bi == b) { bi = 0; if (++x == mx) x = 0; }
                    }
                }
            }

            // (c) forward: compact buffer -> strided S chunk (scale_f, with
            //     the caller's multiplier folded in).  Under the G->S alias
            //     this rewrites exactly the trail range read in (a) — safe.
            DescKey kf{d0, d1, d2, c, (MKL_LONG)Tg, c, scale_f, 1.0, 0};
            DFTI_DESCRIPTOR_HANDLE hf = nullptr;
            st = get_descriptor(kf, &hf);
            if (!dfti_ok(st)) { err.record(st, "conv fwd descriptor"); continue; }
            st = compute(hf, /*forward=*/true, /*inplace=*/false,
                         buf, s_out + t0);
            if (!dfti_ok(st)) { err.record(st, "conv fwd compute"); continue; }
        }
    }

    if (err.failed.load()) {
        return ffi::Error(ffi::ErrorCode::kInternal, err.msg);
    }
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::mklfft

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MklFftFlatKHostFfi,
    lorrax_ffi::mklfft::FlatKDispatch,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()      // X (nk, *trail) c128
        .Ret<xla::ffi::AnyBuffer>()      // Y same shape (may alias X)
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<int64_t>("forward")        // 0 = ifftn, 1 = fftn
        .Attr<double>("scale"));         // total jnp-convention norm scale

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MklFftGwConvHostFfi,
    lorrax_ffi::mklfft::GwConvDispatch,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()      // G (nk, a, mx, b, my) c128
        .Arg<xla::ffi::AnyBuffer>()      // W (nk, mx, my) c128
        .Ret<xla::ffi::AnyBuffer>()      // S shape(G) (may alias G)
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale_i")         // inverse-transform scale (both IFFTs)
        .Attr<double>("scale_f"));       // forward scale × caller multiplier
