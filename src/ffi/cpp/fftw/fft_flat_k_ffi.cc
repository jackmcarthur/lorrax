// fft_flat_k_ffi.cc — the host (cpu) leg of the ffi.fft k-convolution router:
// flat-k batched 3-D FFTs through the FFTW3 advanced interface
// (fftw_plan_many_dft), bound at RUN time by dlsym to whichever FFTW3 engine
// the process has (MKL's FFTW3 export, cray-fftw, FFTW, AOCL; the ladder and
// its refusal are docs/architecture/ffi_layout.md §3c).  Nothing FFT is
// linked.  The k-convolution router's cpu (plan) backend; required at startup
// by ffi.fft.require_kconv on a cpu mesh.
//
// WHY: XLA:CPU's fft custom call needs the transformed axes minor-most, while
// the Σ τ tiles live k-major in "dot layout" (nk, s, μ_X, s', μ_Y); XLA would
// transpose the whole tile around every FFT.  The advanced interface's strides
// read the tile where it lies: FFT-axis element strides {nky·nkz·T, nkz·T, T},
// a batch of T transforms at distance 1 along the unit-stride trail.
//
// Handlers (registered by ffi_loader.py under platform="cpu"):
//   MklFftFlatKHostFfi  (target lorrax_mklfft_flat_k)
//       X (nk, *trail) c128 -> Y same shape: one batched 3-D FFT over the
//       LEADING flat-k axis; direction and total scale are attributes, so the
//       jnp.fft norm conventions live in one place, the Python helper.
//   MklFftGwConvHostFfi (target lorrax_mklfft_gw_conv)
//       G (nk, a, mx, b, my), W (nk, mx, my) c128 -> S = shape(G):
//           S = FFT[ IFFT[G] * IFFT[W][:,None,:,None,:] * mult ]
//       V_R = IFFT[W] is staged once per call in a reused host arena; the
//       R-space G tile exists only in per-thread compact chunks.
// (The target and symbol names keep their historical MKL spelling.)
//
// In place: the Python wrappers alias operand 0 to the result; every plan is
// out-of-place into a per-thread compact buffer, then scatter-copied back, and
// a chunk's k-lines are fully read before the scatter rewrites them.
//
// Threading: an OpenMP team over the trail chunks (this TU is -fopenmp), each
// member pinning a resident MKL to one thread (common/mkl_thread_pin.h).
// Team size LORRAX_FFT_FFI_THREADS (auto|off|N), parsed per call.
// Plans are FFTW_ESTIMATE | FFTW_UNALIGNED (MEASURE would overwrite the XLA
// operand) and cached per shape under one mutex.
//
// Every extent and stride comes from the runtime buffer dimensions and
// attributes; the batch is whatever trail the caller shards onto this rank.

#include <algorithm>
#include <array>
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

// NO vendor FFT header.  The FFTW3 entry points are resolved at RUN time by
// dlsym (see the table below), so this TU must compile on a site that has no
// FFT development headers at all — and it must not bake in whose FFT it is.
// The five declarations below are the FFTW3 ABI, which is stable and
// identical across FFTW3, cray-fftw, AOCL and MKL's native FFTW3 interface:
//   fftw_plan     is an opaque pointer
//   fftw_complex  is double[2], layout-compatible with std::complex<double>
//   the flag/sign values are fixed by the FFTW3 API contract
using fftw_plan_t = void*;
static constexpr int kFftwForward = -1;
static constexpr int kFftwBackward = +1;
static constexpr unsigned kFftwEstimate = 1U << 6;   // FFTW_ESTIMATE
static constexpr unsigned kFftwUnaligned = 1U << 1;  // FFTW_UNALIGNED

#include "../common/mkl_thread_pin.h"
#include "xla/ffi/api/ffi.h"

namespace lorrax_ffi::mklfft {

namespace ffi = ::xla::ffi;
using C128 = std::complex<double>;

// ---------------------------------------------------------------------------
//  MKL thread pinning (workstream-AW pattern).  The mechanism lives in
//  cpp/common/mkl_thread_pin.h — an MPI-free, CUDA-free header, which is why
//  it is not cpp/scalapack/blacs_grid.h (that one includes <mpi.h> at :40 and
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

// OpenMP team size for the chunk loop.  Strict full-string grammar via the
// shared parser (mklpin::parse_thread_knob — the blacs_grid.h AW-audit
// lesson: a typo must not silently pick a known-bad policy; the parse used
// to exist three times and drifted, 2026-07-31 audit).  "auto" (default) =
// the ambient omp_get_max_threads() — the production harness exports
// OMP_NUM_THREADS per rank; whether a smaller cap wins (the owner's
// 6-way-cap question) is a measurement recorded in the proto notes, and the
// unit gate sweeps this knob per call.
static int team_threads() {
    const int maxt = std::max(1, omp_get_max_threads());
    const char* v = std::getenv("LORRAX_FFT_FFI_THREADS");
    const mklpin::ThreadKnob k = mklpin::parse_thread_knob(v, 1, 4096);
    switch (k.kind) {
        case mklpin::ThreadKnob::kOff: return 1;
        case mklpin::ThreadKnob::kInt: return k.value;
        case mklpin::ThreadKnob::kBad: {
            static std::atomic<bool> warned{false};
            // Rank-scoped (announce_here): pre-audit this printed on every
            // rank — P lines for one misspelling (P1.13).
            if (!warned.exchange(true) && mklpin::announce_here()) {
                std::fprintf(
                    stderr,
                    "*** LORRAX_FFT_FFI_THREADS='%s' is not a recognized "
                    "value (accepted, case-insensitive: 'auto', 'off', or a "
                    "positive integer <= 4096).  Falling back to 'auto' "
                    "(%d threads). ***\n",
                    v, maxt);
            }
            return maxt;
        }
        case mklpin::ThreadKnob::kAuto:
        default:
            return maxt;
    }
}

// Trail elements per chunk.  Default: size the per-thread compact buffer
// (nk · chunk · 16 B) to ~512 KiB so it stays resident in a CLX core's
// 1 MiB L2 — the dimension-by-dimension radix passes of the small 3-D FFT
// then run in cache and the big tile is streamed ONCE per direction
// (measured: the strided→strided form, whose radix passes all stream the
// full tile, ran 2.8× slower single-thread).  LORRAX_FFT_FFI_CHUNK
// overrides for experiments; the
// unit gate sweeps it and exercises ragged chunks explicitly.
static int64_t chunk_elems(int64_t nk) {
    const char* v = std::getenv("LORRAX_FFT_FFI_CHUNK");
    if (v && *v) {
        char* end = nullptr;
        const long long parsed = std::strtoll(v, &end, 10);
        if (end != v && *end == '\0' && parsed >= 1 &&
            parsed <= (1LL << 30)) {
            return static_cast<int64_t>(parsed);
        }
        static std::atomic<bool> warned{false};
        // Rank-scoped announce (P1.13) — was an every-rank print.
        if (!warned.exchange(true) && mklpin::announce_here()) {
            std::fprintf(stderr,
                         "*** LORRAX_FFT_FFI_CHUNK='%s' unrecognized — using "
                         "the auto policy. ***\n",
                         v);
        }
    }
    const int64_t c = (512LL * 1024) / (16 * std::max<int64_t>(nk, 1));
    return std::max<int64_t>(c, 64);
}

// Native FFT detail follows the one rank-zero driver debug stream.
static bool log_enabled() {
    static const bool on = mklpin::debug_print_here();
    return on;
}

// ---------------------------------------------------------------------------
//  Plan key: one fftw_plan_many_dft plan per (dims, in/out element stride,
//  transforms per call, sign), cached for the process under plan_mutex().
//  Every plan is out-of-place into a compact chunk (see run_flat_batch).
// ---------------------------------------------------------------------------
struct DescKey {
    // The fields map one-for-one onto fftw_plan_many_dft's advanced-interface
    // parameters.  t_in/t_out ARE istride/ostride; n IS howmany; idist and
    // odist default to 1 for the flat-k layout.  Contiguous k-minor
    // trajectories use t_in=t_out=1 and dist_in=dist_out=nk instead.
    long d0, d1, d2;
    long t_in, t_out;      // element stride (istride / ostride)
    long n;                // transforms per execute call (howmany)
    int  sign;             // kFftwForward | kFftwBackward
    long dist_in = 1, dist_out = 1;

    bool operator<(const DescKey& o) const {
        return std::tie(d0, d1, d2, t_in, t_out, n, sign, dist_in, dist_out) <
               std::tie(o.d0, o.d1, o.d2, o.t_in, o.t_out, o.n, o.sign,
                        o.dist_in, o.dist_out);
    }
};
// ---------------------------------------------------------------------------
//  THE FFTW3 SYMBOL TABLE, resolved at RUN time.
//
//  Design: docs/architecture/ffi_layout.md §7.  This is the pattern the GEMM
//  service already proved (cpp/cblas/gemm_batch_ffi.cc) -- resolve each
//  vendor entry point through mklpin::resolve_sym (RTLD_DEFAULT then
//  RTLD_NEXT) rather than taking a link-time dependency.  Three consequences,
//  all of them the point:
//
//   * ONE SOURCE, NO FORK.  On Frontera the symbols resolve out of
//     libmkl_intel_lp64, already on the link line, because MKL exports the
//     FFTW3 C interface natively -- no wrapper build, no new link dep.  On a
//     Cray/AMD site the engine is brought in by the dlopen ladder below.
//     The engine is named by what the process can load, never by a knob --
//     invariant 2 (§2).
//
//  CORRECTED 2026-08-06 -- WHY THERE IS A dlopen LADDER AND NOT A LINK.
//  --------------------------------------------------------------------
//  This comment used to read "the LORRAX_HOST_HAVE_FFTW3 CMake leg links the
//  system FFTW and the SAME dlsym finds it".  That is what the build did, and
//  it was self-defeating.  Linking FFTW3 puts its SONAME in DT_NEEDED, which
//  is a LOAD-TIME dependency: the dynamic linker must resolve it before any
//  code in this library runs.  So the library became unloadable anywhere that
//  exact SONAME was missing -- and cray-fftw's is `libfftw3.so.mpi31.3`,
//  version- AND MPI-flavour-stamped, not a string that survives a move to
//  another site, another cray-fftw, or (measured, 2026-08-06) into the
//  Shifter container, which does not bind-mount /opt/cray/pe/fftw at all.
//  Nineteen ScaLAPACK/GEMM contract tests that never touch an FFT
//  became skips because the FFT engine's library was not on disk.
//
//  Resolving symbols at run time is only half of run-time resolution.  The
//  library that DEFINES them has to be loaded at run time too, or the
//  link-time coupling is still there -- just moved from the symbol table to
//  the DT_NEEDED list, where the recorded invariant
//  (`nm -D --undefined-only | grep -c fftw_` -> 0) could not see it.  Hence:
//
//      stage 1  resolve_sym            -- already-loaded provider.  This is
//                                        the MKL site: libmkl_intel_lp64 is
//                                        in DT_NEEDED for ScaLAPACK anyway
//                                        and exports the FFTW3 C interface,
//                                        so Frontera never reaches stage 2.
//      stage 2  dlopen(candidates)     -- non-MKL site.  RTLD_GLOBAL so the
//                                        symbols enter the global scope and
//                                        stage 1's resolver finds them on the
//                                        retry.  Nothing is added to
//                                        DT_NEEDED, so failure here costs the
//                                        FFT handlers and NOTHING ELSE.
//      stage 3  refuse, loudly         -- the router's startup check
//                                        (require_kconv), naming the
//                                        symbol AND every candidate tried.
//
//  The candidate list is ordered most-specific-first.  LORRAX_FFTW3_SO_HINT is
//  the absolute path CMake recorded for the FFTW3 it was configured against --
//  that is how "the engine is named by the build" survives the link removal.
//  LORRAX_FFTW3_SO is deployment plumbing (GATES.md's "not gates" list), for a
//  site whose SONAME nobody guessed; it is not a gate and selects nothing
//  numerically -- every candidate implements the same FFTW3 advanced ABI.
//   * NO NEW ENVIRONMENT VARIABLE.  Deliberate.  A missing handler refuses
//     at STARTUP via ffi.fft.require_kconv.
//   * ABSENCE IS LOUD.  A missing engine is a refusal naming the symbol,
//     never a silent demotion to a slower path and never a wrong number.
// ---------------------------------------------------------------------------
struct FftwApi {
    fftw_plan_t (*plan_many)(int, const int*, int,
                             void*, const int*, int, int,
                             void*, const int*, int, int,
                             int, unsigned) = nullptr;
    void (*execute_dft)(fftw_plan_t, void*, void*) = nullptr;
    void (*destroy_plan)(fftw_plan_t) = nullptr;
    bool ok = false;
    const char* missing = nullptr;
    // Where the engine came from, and — when it did not come at all — every
    // candidate that was tried.  Both feed the refusal message: "no FFT
    // engine" without the list of names looked for is an unactionable error.
    std::string provider;
    std::string tried;
};

// The dlopen candidate ladder.  Most specific first; every entry is the same
// FFTW3 advanced ABI, so the order is about WHICH FILE, never about which
// numerics.  Empty entries are skipped, so an unset env var costs nothing.
static std::vector<std::string> fftw3_candidates() {
    std::vector<std::string> out;
    auto push = [&out](const char* s) {
        if (s && *s) out.emplace_back(s);
    };
    // 1. Deployment plumbing: a site whose SONAME nobody guessed.
    push(std::getenv("LORRAX_FFTW3_SO"));
    // 2. What THIS BUILD was configured against, as an absolute path.  This
    //    is how the artifact keeps naming its own engine now that the link
    //    is gone.
#ifdef LORRAX_FFTW3_SO_HINT
    push(LORRAX_FFTW3_SO_HINT);
#endif
    // 3. Portable SONAMEs, in decreasing likelihood.  `libfftw3.so.3` is the
    //    upstream FFTW3 SONAME everywhere; `.so.mpi31.3` is cray-fftw's
    //    MPI-flavour-stamped one; the bare `.so` is a devel symlink, present
    //    only where headers are installed, hence last.
    push("libfftw3.so.3");
    push("libfftw3.so.mpi31.3");
    push("libmkl_rt.so");
    push("libfftw3.so");
    return out;
}

static const FftwApi& fftw_api() {
    static const FftwApi api = [] {
        FftwApi a;
        auto bind = [&a] {
            a.plan_many = reinterpret_cast<decltype(a.plan_many)>(
                mklpin::resolve_sym("fftw_plan_many_dft"));
            a.execute_dft = reinterpret_cast<decltype(a.execute_dft)>(
                mklpin::resolve_sym("fftw_execute_dft"));
            a.destroy_plan = reinterpret_cast<decltype(a.destroy_plan)>(
                mklpin::resolve_sym("fftw_destroy_plan"));
            return a.plan_many && a.execute_dft && a.destroy_plan;
        };

        // STAGE 1 — an engine already in the process.  The MKL site lands
        // here: libmkl_intel_lp64 is in DT_NEEDED for ScaLAPACK and exports
        // the FFTW3 C interface natively, so nothing is dlopen'd on Frontera
        // and the behaviour there is byte-for-byte what it was.
        if (bind()) {
            a.provider = "already loaded (no dlopen needed)";
        } else {
            // STAGE 2 — bring one in.  RTLD_GLOBAL is load-bearing: the
            // symbols must enter the global scope for the stage-1 resolver to
            // find them on the retry.  RTLD_NOW so a broken engine fails HERE,
            // naming itself, rather than at the first transform.
            std::ostringstream tried;
            bool first = true;
            for (const std::string& cand : fftw3_candidates()) {
                if (!first) tried << ", ";
                first = false;
                tried << cand;
                void* h = dlopen(cand.c_str(), RTLD_NOW | RTLD_GLOBAL);
                if (h == nullptr) {
                    if (log_enabled()) {
                        const char* e = dlerror();
                        std::fprintf(stderr, "[mklfft] dlopen(%s) failed: %s\n",
                                     cand.c_str(), e ? e : "(no dlerror)");
                    }
                    continue;
                }
                // Deliberately never dlclose'd: the plans below hold state
                // inside the engine for the life of the process.
                if (bind()) {
                    a.provider = cand;
                    break;
                }
                if (log_enabled()) {
                    std::fprintf(stderr,
                                 "[mklfft] dlopen(%s) succeeded but does not "
                                 "export the FFTW3 advanced interface\n",
                                 cand.c_str());
                }
            }
            a.tried = tried.str();
        }

        if (!a.plan_many)         a.missing = "fftw_plan_many_dft";
        else if (!a.execute_dft)  a.missing = "fftw_execute_dft";
        else if (!a.destroy_plan) a.missing = "fftw_destroy_plan";
        a.ok = (a.missing == nullptr);
        if (log_enabled()) {
            if (a.ok) {
                std::fprintf(stderr,
                             "[mklfft] FFTW3 engine resolved "
                             "(fftw_plan_many_dft et al.) from %s\n",
                             a.provider.c_str());
            } else {
                std::fprintf(stderr,
                             "[mklfft] FFTW3 engine NOT RESOLVED (%s); "
                             "tried: %s\n",
                             a.missing, a.tried.c_str());
            }
        }
        return a;
    }();
    return api;
}

// THREE FFTW HAZARDS THIS DESIGN IS BUILT AROUND
// ----------------------------------------------
// 1. THE PLANNER IS NOT THREAD-SAFE.  fftw_plan_* mutates global planner
//    state; only fftw_execute* is re-entrant.  The chunk loop below is an
//    OpenMP parallel region, so every plan lookup/creation is serialised by
//    plan_mutex() and execution uses the NEW-ARRAY entry fftw_execute_dft,
//    which is documented thread-safe.
// 2. FFTW_MEASURE OVERWRITES ITS BUFFERS.  Planning with MEASURE would
//    destroy the XLA operand we were handed.  FFTW_ESTIMATE is documented
//    not to touch the arrays, so planning against live buffers is safe.
//    MEASURE is a deliberate non-goal here.
// 3. NEW-ARRAY EXECUTION HAS AN ALIGNMENT CONTRACT.  fftw_execute_dft may
//    only be handed arrays whose alignment matches the planning arrays,
//    because a plan can bake in SIMD assumptions.  Our pointers are chunk
//    offsets into XLA buffers and are NOT alignment-stable, so every plan is
//    created FFTW_UNALIGNED.  That costs some vectorisation and is the
//    correct trade: the alternative is a wrong answer or a SIGSEGV on an
//    unlucky offset.
static std::mutex& plan_mutex() {
    static std::mutex m;
    return m;
}

// Returns nullptr if the engine is absent or the planner failed.
static fftw_plan_t get_descriptor(const DescKey& k, void* in_hint,
                                  void* out_hint) {
    const FftwApi& api = fftw_api();
    if (!api.ok) return nullptr;
    static std::map<DescKey, fftw_plan_t> cache;
    std::lock_guard<std::mutex> g(plan_mutex());
    auto it = cache.find(k);
    if (it != cache.end()) return it->second;

    int n3[3] = {static_cast<int>(k.d0), static_cast<int>(k.d1),
                 static_cast<int>(k.d2)};
    // inembed/onembed NULL => taken as n, so element (i0,i1,i2) of transform
    // j sits at base + j*dist + (i0*d1*d2 + i1*d2 + i2)*stride.
    fftw_plan_t p = api.plan_many(
        3, n3, static_cast<int>(k.n),
        in_hint,  nullptr, static_cast<int>(k.t_in),  static_cast<int>(k.dist_in),
        out_hint, nullptr, static_cast<int>(k.t_out), static_cast<int>(k.dist_out),
        k.sign, kFftwEstimate | kFftwUnaligned);
    if (!p) return nullptr;

    if (log_enabled()) {
        static std::map<DescKey, char> seen;
        if (seen.emplace(k, '\0').second) {
            std::fprintf(stderr,
                         "[mklfft] plan dims=(%ld,%ld,%ld) istride=%ld "
                         "ostride=%ld howmany=%ld sign=%d "
                         "(ESTIMATE|UNALIGNED)\n",
                         k.d0, k.d1, k.d2, k.t_in, k.t_out, k.n, k.sign);
        }
    }
    cache.emplace(k, p);
    return p;
}

static inline void execute(fftw_plan_t p, const C128* in, C128* out) {
    // Plans are never FFTW_IN_PLACE, so the transform never writes the input
    // buffer and the const_cast is safe for the XLA (read-only) operand.  The
    // XLA-granted alias (in == out at the HANDLER level) is handled by the
    // chunk engine's read-before-scatter ordering.
    fftw_api().execute_dft(p, const_cast<void*>(static_cast<const void*>(in)),
                           static_cast<void*>(out));
}

// FFTW normalises nothing in either direction, so the caller's pre-folded
// jnp-convention scale is an explicit streaming multiply over the chunk.
static inline void scale_contig(C128* p, long n, double s) {
    if (s == 1.0) return;
    for (long i = 0; i < n; ++i) p[i] *= s;
}

static inline void scale_strided(C128* p, long nk, long howmany, long stride,
                                 double s) {
    if (s == 1.0) return;
    for (long k = 0; k < nk; ++k) {
        C128* row = p + k * stride;
        for (long j = 0; j < howmany; ++j) row[j] *= s;
    }
}

// The one refusal, phrased so the fix is in the message.
static ffi::Error engine_absent_error() {
    const FftwApi& api = fftw_api();
    std::ostringstream os;
    os << "mklfft: no FFTW3 engine in this process — could not resolve "
       << (api.missing ? api.missing : "the FFTW3 entry points")
       << ".  The flat-k FFT handlers call the FFTW3 ADVANCED interface, "
          "which Intel MKL exports natively (libmkl_intel_lp64) and which "
          "cray-fftw / stock FFTW3 / AOCL also provide.  No engine was "
          "already loaded, and dlopen found none of: "
       << (api.tried.empty() ? std::string("(no candidates tried)") : api.tried)
       << ".  FIX: put an FFTW3 where the dynamic linker can see it "
          "(LD_LIBRARY_PATH, or a bind-mount if you are in a container — note "
          "that /opt/cray/pe/fftw is NOT mounted inside the Shifter image), "
          "or name the file outright with LORRAX_FFTW3_SO=/path/to/libfftw3.so"
          ".  Do NOT try to fix this by linking FFTW3 into "
          "liblorrax_ffi_host.so: that was the pre-2026-08-06 arrangement, and "
          "it made the ENTIRE library — ScaLAPACK and GEMM handlers "
          "included — fail to load wherever the FFTW3 SONAME was absent.";
    return ffi::Error(ffi::ErrorCode::kInternal, os.str());
}

struct ErrSink {
    std::atomic<bool> failed{false};
    std::mutex mu;
    std::string msg;

    // FFTW has no status codes (plan creation returns NULL), so `where`
    // carries the whole diagnostic.
    void record(long status, const char* where) {
        if (failed.exchange(true)) return;
        std::lock_guard<std::mutex> lock(mu);
        std::ostringstream os;
        os << "mklfft (FFTW3 advanced interface): " << where << " failed";
        if (status) os << ", status=" << status;
        const FftwApi& api = fftw_api();
        if (!api.ok && api.missing) os << " — no FFTW3 engine: dlsym could not resolve " << api.missing;
        msg = os.str();
    }
};

static ffi::Error run_flat_batch(
    long d0, long d1, long d2, int64_t T, bool forward,
    double scale, const C128* in, C128* out)
{
    const int64_t nk = (int64_t)d0 * d1 * d2;
    const int64_t C = std::min<int64_t>(chunk_elems(nk), T);
    const int64_t n_chunks = (T + C - 1) / C;
    const int nthr = static_cast<int>(
        std::min<int64_t>(team_threads(), n_chunks));
    if (!fftw_api().ok) return engine_absent_error();
    const int sign = forward ? kFftwForward : kFftwBackward;
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
            const long c = static_cast<long>(
                std::min<int64_t>(C, T - t0));
            DescKey key{(long)d0, (long)d1, (long)d2, (long)T, (long)c,
                        (long)c, sign};
            fftw_plan_t h = get_descriptor(
                key, const_cast<C128*>(in + t0), buf);
            if (!h) { err.record(0, "plan creation"); continue; }
            execute(h, in + t0, buf);
            scale_contig(buf, (long)(nk * c), scale);
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
    return run_flat_batch((long)nkx, (long)nky, (long)nkz, T,
                          forward != 0, scale, in, out);
}

// ---------------------------------------------------------------------------
//  Handler 2: fused Σ τ convolution  S = FFT[ IFFT[G] · IFFT[W](bcast) ].
//  The R-space G tile exists only as per-thread ~4 MiB compact chunks; the
//  IFFT'd W (V_R, the small (nk, mx, my) tile) is staged once per call in a
//  reused process arena (malloc'd, invisible to the XLA planner — same
//  class as the scalapack workspace, logged under LORRAX_DEBUG_PRINT).
// ---------------------------------------------------------------------------
struct Arena {
    std::mutex mu;
    std::vector<C128> buf;
};

static Arena& vr_arena() {
    static Arena a;
    return a;
}

static ffi::Error run_klead_stored(
    long d0, long d1, long d2, int64_t a, int64_t mx, int64_t b, int64_t my,
    double scale_i, double scale_f, const C128* g_in, const C128* vr, C128* s_out)
{
    const int64_t nk = d0 * d1 * d2;
    const int64_t Tg = a * mx * b * my, Tv = mx * my;
    const bool aliased = g_in == s_out;
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
                         "chunk=%ld kernel=%.1f MB\n",
                         (long)d0, (long)d1, (long)d2, (long)a, (long)mx,
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
            const long c = static_cast<long>(
                std::min<int64_t>(C, Tg - t0));

            // (a) backward: strided G chunk -> compact buffer (scale_i).
            DescKey kb{(long)d0, (long)d1, (long)d2, (long)Tg, (long)c,
                       (long)c, kFftwBackward};
            fftw_plan_t hb = get_descriptor(
                kb, const_cast<C128*>(g_in + t0), buf);
            if (!hb) { err.record(0, "conv bwd plan"); continue; }
            execute(hb, g_in + t0, buf);
            scale_contig(buf, (long)(nk * c), scale_i);

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
            DescKey kf{(long)d0, (long)d1, (long)d2, (long)c, (long)Tg,
                       (long)c, kFftwForward};
            fftw_plan_t hf = get_descriptor(kf, buf, s_out + t0);
            if (!hf) { err.record(0, "conv fwd plan"); continue; }
            execute(hf, buf, s_out + t0);
            scale_strided(s_out + t0, (long)nk, (long)c, (long)Tg, scale_f);
        }
    }

    if (err.failed.load()) {
        return ffi::Error(ffi::ErrorCode::kInternal, err.msg);
    }
    return ffi::Error::Success();
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
    if (!fftw_api().ok) return engine_absent_error();

    const auto* g_in = static_cast<const C128*>(G.untyped_data());
    const auto* w_in = static_cast<const C128*>(W.untyped_data());
    auto* s_out = static_cast<C128*>(S->untyped_data());
    const long d0 = (long)nkx, d1 = (long)nky, d2 = (long)nkz;

    // --- stage 1: V_R = IFFT[W] into the arena (strided, trail stride Tv).
    Arena& ar = vr_arena();
    std::unique_lock<std::mutex> ar_lock(ar.mu);  // serializes concurrent convs
    if ((int64_t)ar.buf.size() < nk * Tv) ar.buf.resize(nk * Tv);
    C128* vr = ar.buf.data();
    {
        ffi::Error e = run_flat_batch(d0, d1, d2, Tv, /*forward=*/false,
                                      scale_i, w_in, vr);
        if (!e.success()) return e;
    }

    return run_klead_stored(d0, d1, d2, a, mx, b, my,
                            scale_i, scale_f, g_in, vr, s_out);
}

// BSE k-leading convolution uses the same chunk engine as GW, with the
// caller's real-space kernel directly: no W transform or private W arena.
static ffi::Error KleadStoredDispatch(
    ffi::AnyBuffer X, ffi::AnyBuffer K, ffi::Result<ffi::AnyBuffer> Y,
    int64_t nkx, int64_t nky, int64_t nkz, double scale)
{
    auto xd = X.dimensions();
    auto kd = K.dimensions();
    auto yd = Y->dimensions();
    if (X.element_type() != ffi::DataType::C128 ||
        K.element_type() != ffi::DataType::C128 ||
        Y->element_type() != ffi::DataType::C128 ||
        xd.size() != 5 || kd.size() != 3 || yd.size() != 5 ||
        !std::equal(xd.begin(), xd.end(), yd.begin())) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "mklfft.kconv_klead: expected complex128 X/Y "
                          "(nk,a,mx,b,my) and K_R (nk,mx,my)");
    }
    if (nkx < 1 || nky < 1 || nkz < 1 || xd[0] != nkx * nky * nkz ||
        kd[0] != xd[0] || kd[1] != xd[2] || kd[2] != xd[4]) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "mklfft.kconv_klead: X/K_R shapes disagree with the k grid");
    }
    if (xd[1] * xd[2] * xd[3] * xd[4] == 0) return ffi::Error::Success();
    if (!fftw_api().ok) return engine_absent_error();
    return run_klead_stored((long)nkx, (long)nky, (long)nkz,
                            xd[1], xd[2], xd[3], xd[4], 1.0, scale,
                            static_cast<const C128*>(X.untyped_data()),
                            static_cast<const C128*>(K.untyped_data()),
                            static_cast<C128*>(Y->untyped_data()));
}

// BSE convolution with K_R already stored in real space.  Each contiguous
// k trajectory is transformed in bounded per-thread chunks; neither a
// k-leading transpose nor a full R-space X intermediate is materialized.
static ffi::Error KminorConvDispatch(
    ffi::AnyBuffer X, ffi::AnyBuffer K, ffi::Result<ffi::AnyBuffer> Y,
    int64_t nkx, int64_t nky, int64_t nkz, double scale, int64_t out_layout)
{
    if (X.element_type() != ffi::DataType::C128 ||
        K.element_type() != ffi::DataType::C128 ||
        Y->element_type() != ffi::DataType::C128) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "mklfft.kconv_kminor: buffers must be complex128");
    }
    auto xd = X.dimensions();
    auto kd = K.dimensions();
    auto yd = Y->dimensions();
    if (xd.size() != 6 || kd.size() != 3 || yd.size() != 6 ||
        (out_layout != 0 && out_layout != 1)) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "mklfft.kconv_kminor: expected X rank 6, K rank 3, "
                          "Y rank 6 and out_layout 0 or 1");
    }
    const int64_t d0 = xd[0], d1 = xd[1], d2 = xd[2];
    const int64_t d3 = xd[3], d4 = xd[4], nk = xd[5];
    const std::array<int64_t, 6> expected = out_layout == 0
        ? std::array<int64_t, 6>{d0, d1, d2, d3, d4, nk}
        : std::array<int64_t, 6>{d0, nk, d3, d1, d4, d2};
    if (nkx < 1 || nky < 1 || nkz < 1 || nk != nkx * nky * nkz ||
        kd[0] != d1 || kd[1] != d2 || kd[2] != nk ||
        !std::equal(yd.begin(), yd.end(), expected.begin())) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "mklfft.kconv_kminor: X/K/Y shapes disagree with "
                          "the k grid, broadcast axes or output layout");
    }
    const int64_t rows = d0 * d1 * d2 * d3 * d4;
    if (rows == 0) return ffi::Error::Success();
    if (!fftw_api().ok) return engine_absent_error();
    const auto* x_in = static_cast<const C128*>(X.untyped_data());
    const auto* k_in = static_cast<const C128*>(K.untyped_data());
    auto* y_out = static_cast<C128*>(Y->untyped_data());
    if (out_layout == 1 && x_in == y_out) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "mklfft.kconv_kminor: transposed output cannot alias X");
    }
    const int64_t C = std::min<int64_t>(chunk_elems(nk), rows);
    const int64_t n_chunks = (rows + C - 1) / C;
    const int nthr = static_cast<int>(std::min<int64_t>(team_threads(), n_chunks));
    ErrSink err;

#pragma omp parallel num_threads(nthr)
    {
        MklLocalPin pin(1);
        static thread_local std::vector<C128> backward_buf, forward_buf;
        if ((int64_t)backward_buf.size() < nk * C) backward_buf.resize(nk * C);
        // Layout 0 writes its forward transform straight to Y.  Layout 1
        // needs one more compact chunk before the permuted store.
        if (out_layout == 1 && (int64_t)forward_buf.size() < nk * C)
            forward_buf.resize(nk * C);

#pragma omp for schedule(dynamic)
        for (int64_t ci = 0; ci < n_chunks; ++ci) {
            if (err.failed.load(std::memory_order_relaxed)) continue;
            const int64_t t0 = ci * C;
            const long c = static_cast<long>(std::min<int64_t>(C, rows - t0));
            C128* buf = backward_buf.data();
            DescKey kb{(long)nkx, (long)nky, (long)nkz, 1, 1, c,
                       kFftwBackward, (long)nk, (long)nk};
            fftw_plan_t hb = get_descriptor(kb, const_cast<C128*>(x_in + t0 * nk), buf);
            if (!hb) { err.record(0, "kminor conv bwd plan"); continue; }
            execute(hb, x_in + t0 * nk, buf);
            for (int64_t j = 0; j < c; ++j) {
                const int64_t kernel_row = ((t0 + j) / (d3 * d4)) % (d1 * d2);
                C128* bp = buf + j * nk;
                const C128* kp = k_in + kernel_row * nk;
                for (int64_t k = 0; k < nk; ++k) bp[k] *= kp[k] * scale;
            }
            C128* dest = out_layout == 0 ? y_out + t0 * nk : forward_buf.data();
            DescKey kf{(long)nkx, (long)nky, (long)nkz, 1, 1, c,
                       kFftwForward, (long)nk, (long)nk};
            fftw_plan_t hf = get_descriptor(kf, buf, dest);
            if (!hf) { err.record(0, "kminor conv fwd plan"); continue; }
            execute(hf, buf, dest);
            if (out_layout == 1) {
                for (int64_t j = 0; j < c; ++j) {
                    int64_t r = t0 + j;
                    const int64_t i4 = r % d4; r /= d4;
                    const int64_t i3 = r % d3; r /= d3;
                    const int64_t i2 = r % d2; r /= d2;
                    const int64_t i1 = r % d1; r /= d1;
                    const int64_t base = ((((r * nk) * d3 + i3) * d1 + i1) * d4 + i4) * d2 + i2;
                    const int64_t stride = d3 * d1 * d4 * d2;
                    for (int64_t k = 0; k < nk; ++k)
                        y_out[base + k * stride] = dest[j * nk + k];
                }
            }
        }
    }
    if (err.failed.load()) return ffi::Error(ffi::ErrorCode::kInternal, err.msg);
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MklFftKminorConvHostFfi,
    lorrax_ffi::mklfft::KminorConvDispatch,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()      // X (d0,d1,d2,d3,d4,nk), c128
        .Arg<xla::ffi::AnyBuffer>()      // K_R (d1,d2,nk), c128
        .Ret<xla::ffi::AnyBuffer>()      // layout 0 may alias X
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("out_layout"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MklFftKleadStoredHostFfi,
    lorrax_ffi::mklfft::KleadStoredDispatch,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()      // X (nk,a,mx,b,my), c128
        .Arg<xla::ffi::AnyBuffer>()      // K_R (nk,mx,my), c128
        .Ret<xla::ffi::AnyBuffer>()      // may alias X
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale"));
