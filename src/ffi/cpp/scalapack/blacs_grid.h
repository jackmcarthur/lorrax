// blacs_grid.h — the ScaLAPACK/BLACS seam shared by every handler in
// src/ffi/cpp/scalapack/.
//
// ScaLAPACK ships no C header we can rely on across vendors (MKL's
// mkl_scalapack.h exists but is MKL-only and drags in mkl.h), so the
// Fortran ABI is declared by hand here — ONCE — together with the
// per-MeshCtx BLACS context cache the handlers share.  32-bit integer
// interface (LP64: `libmkl_scalapack_lp64` on Frontera, `libsci_*` on
// Cray); every `int` below is the ScaLAPACK integer.
//
// Grid contract (identical for every handler): the BLACS grid is built on
// the MeshCtx's rank-remapped comm (src/ffi/cpp/mesh/context.cc) with
// BLACS "C" (column-major) grid order, so comm rank (mx + my*Px) lands at
// grid coords (mx, my) == the JAX mesh coords.  Combined with the square
// block g = N / max(Px, Py) the block-cyclic distribution then coincides
// exactly with JAX's contiguous ``P('x','y')`` shards on square and 1-D
// meshes — which is why those are the only meshes the wrappers accept.
//
// The context is created COLLECTIVELY on first use per MeshCtx and cached
// for the process lifetime (BLACS grids are freed at exit).
//
// Character arguments: the Fortran ABI appends hidden string lengths that
// we do not pass.  Every ScaLAPACK routine compares them through LSAME,
// which only ever reads the first character, so a `const char*` is safe —
// the same convention the pXgetrf/pXgetrs calls have used in production
// since workstream L.

#pragma once

#include <cctype>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include <dlfcn.h>
#include <mpi.h>

#include "../common/mkl_thread_pin.h"
#include "../mesh/ctx.h"

// ---------------------------------------------------------------------------
//  C-BLACS + ScaLAPACK tools (Fortran ABI).
// ---------------------------------------------------------------------------
extern "C" {
int  Csys2blacs_handle(MPI_Comm comm);
void Cblacs_gridinit(int* ictxt, const char* order, int nprow, int npcol);
void Cblacs_gridinfo(int ictxt, int* nprow, int* npcol, int* myrow, int* mycol);

int  numroc_(const int* n, const int* nb, const int* iproc,
             const int* isrcproc, const int* nprocs);
void descinit_(int* desc, const int* m, const int* n, const int* mb,
               const int* nb, const int* irsrc, const int* icsrc,
               const int* ictxt, const int* lld, int* info);

// LU (solve_lu_ffi.cc)
void pdgetrf_(const int* m, const int* n, double* a, const int* ia,
              const int* ja, const int* desca, int* ipiv, int* info);
void pdgetrs_(const char* trans, const int* n, const int* nrhs,
              const double* a, const int* ia, const int* ja,
              const int* desca, const int* ipiv, double* b, const int* ib,
              const int* jb, const int* descb, int* info);
void pzgetrf_(const int* m, const int* n, std::complex<double>* a,
              const int* ia, const int* ja, const int* desca, int* ipiv,
              int* info);
void pzgetrs_(const char* trans, const int* n, const int* nrhs,
              const std::complex<double>* a, const int* ia, const int* ja,
              const int* desca, const int* ipiv, std::complex<double>* b,
              const int* ib, const int* jb, const int* descb, int* info);

// Hermitian/symmetric eigensolver, divide & conquer (eigh_ffi.cc).
// NOTE the ABI asymmetry: the real routine has no RWORK.
void pdsyevd_(const char* jobz, const char* uplo, const int* n,
              double* a, const int* ia, const int* ja, const int* desca,
              double* w,
              double* z, const int* iz, const int* jz, const int* descz,
              double* work, const int* lwork,
              int* iwork, const int* liwork, int* info);
void pzheevd_(const char* jobz, const char* uplo, const int* n,
              std::complex<double>* a, const int* ia, const int* ja,
              const int* desca,
              double* w,
              std::complex<double>* z, const int* iz, const int* jz,
              const int* descz,
              std::complex<double>* work, const int* lwork,
              double* rwork, const int* lrwork,
              int* iwork, const int* liwork, int* info);
}

namespace lorrax_ffi::scalapack {

// BLACS context per MeshCtx, created collectively on first use.  The map
// is tiny (one entry per mesh shape) and lives for the process.  Inline =>
// exactly one cache across all translation units of the library.
inline int blacs_ctxt_for(lorrax_ffi::mesh::MeshCtx* ctx) {
    static std::mutex mu;
    static std::map<int64_t, int> cache;
    const int64_t key = reinterpret_cast<int64_t>(ctx);
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    int ictxt = Csys2blacs_handle(ctx->comm);
    // "C": comm rank r -> grid (r % p, r / p).  With the MeshCtx remap
    // (context.cc) this lands JAX shard (mx, my) at grid (mx, my).
    Cblacs_gridinit(&ictxt, "C", ctx->p, ctx->q);
    cache[key] = ictxt;
    return ictxt;
}

// Object that defines `name` in this process's resolution scope, or "" when
// the symbol does not resolve at all.  Empty is NOT an error here: it only
// means the provenance question cannot be answered (a static ScaLAPACK, a
// stripped provider).
inline const std::string& scalapack_symbol_provider(const char* name) {
    static std::mutex mu;
    static std::map<std::string, std::string> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    std::string origin;
    void* addr = lorrax_ffi::mklpin::resolve_sym(name);
    Dl_info info{};
    if (addr != nullptr && dladdr(addr, &info) != 0 && info.dli_fname != nullptr) {
        origin = info.dli_fname;
    }
    return cache.emplace(name, std::move(origin)).first->second;
}

// ---------------------------------------------------------------------------
//  WHO IS ANSWERING?  (LORRAX_DEBUG_PRINT)
//
//  The whole point of calling a published API is that any of several
//  packages can satisfy it — MKL here, Cray LibSci on a Cray, AOCL on AMD,
//  netlib anywhere.  The cost of
//  that freedom is that NOTHING at run time otherwise says which one you
//  actually got: `ffi_loader` keys only on LORRAX's own handler symbols, and
//  a vendor named in a docstring or a directory name is decoration.
//
//  This makes the answer visible on demand rather than guessable.  It prints
//  the defining object for each of the ELEVEN names, once per process, on
//  rank 0 (or every rank with `=all`), reusing the log grammar in
//  mkl_thread_pin.h.  Opt-in, because in a healthy build it is noise; the
//  moment a port misbehaves it is the first question worth asking.
// ---------------------------------------------------------------------------
inline void scalapack_log_providers_once() {
    // Resolved ONCE per process: this runs on EVERY handler dispatch, and the
    // knob is a process-lifetime env fact like every other knob in this
    // header (the thread pin below) — re-reading the environment per solve bought
    // nothing but a getenv on the hot path (2026-08-01 seam-audit leftover).
    static const bool enabled =
        lorrax_ffi::mklpin::debug_print_here();
    if (!enabled) return;
    static std::once_flag once;
    std::call_once(once, [] {
        static const char* kNames[] = {
            "pzheevd_", "pdsyevd_", "pzgetrf_", "pdgetrf_", "pzgetrs_",
            "pdgetrs_", "numroc_", "descinit_", "Csys2blacs_handle",
            "Cblacs_gridinit", "Cblacs_gridinfo"};
        std::fprintf(stderr,
                     "[scalapack] provider map (the 11 names this library "
                     "calls; anything implementing them can back it):\n");
        for (const char* n : kNames) {
            const std::string& lib = scalapack_symbol_provider(n);
            std::fprintf(stderr, "[scalapack]   %-18s <- %s\n", n,
                         lib.empty() ? "<unresolved>" : lib.c_str());
        }
        std::fflush(stderr);
    });
}

// ---------------------------------------------------------------------------
//  MKL thread pinning around ScaLAPACK calls (workstream AW, 2026-07-27).
//
//  THE MEASUREMENT (wk_ENV aw_mkl_matrix, pz_bench = the exact eigh_ffi.cc
//  geometry, n=2448, provider mlx):
//
//    grid 12x12 g=204 (P=144, the production shape):
//        MKL threads 14 -> 11.28 s/q      MKL threads 4 -> 0.463 s/q (24x)
//        MKL threads  1 ->  0.585 s/q     (threads 28, oversub: see logs)
//    grid 4x4 g=612 (P=16): FLAT — 0.87..0.94 s/q across 1/4/14/28.
//
//  At scale the block-cyclic panels are small (g=204), each pXheevd step
//  is hundreds of tiny BLAS calls interleaved with latency-bound BLACS
//  collectives, and MKL's threading layer (fork/join + spin-wait between
//  kernels) starves MPI progress — the more threads, the worse.  A small
//  team (~4) keeps the panel GEMMs parallel without the spin-wait tax.
//
//  Mechanism: mkl_set_num_threads_local on the CALLING thread only, via the
//  shared dlsym resolver in cpp/common/mkl_thread_pin.h, so this header adds
//  no MKL link dependency and the pin degrades to a no-op on a non-MKL
//  ScaLAPACK (LibSci).  That header is where the mechanism now lives for all
//  three families (2026-07-30 divergence audit: this copy used a bare
//  dlsym(RTLD_DEFAULT) while mklblas used RTLD_DEFAULT->RTLD_NEXT, so under a
//  local-scope dlopen this pin would silently no-op).  The global
//  MKL_NUM_THREADS (28 in production — right for the LOCAL zheevd_ plan-A
//  route and for XLA-adjacent BLAS) is untouched; the previous local
//  value is restored when the scope closes.
//
//  Env: LORRAX_SCALAPACK_MKL_THREADS (values case-insensitive)
//        unset / "" / "auto" -> min(mkl_get_max_threads(), 4) [measured best]
//        "0" / "off"         -> leave MKL threading alone (pre-AW behaviour)
//        <positive integer>  -> pin exactly N inside the handlers
//        anything else       -> ONE loud stderr line naming the value and
//                               this grammar, then the "auto" policy.
//        Rationale (audit fix/zq 2026-07-28): a typo ("Auto", "on", "4x",
//        a negative number) used to fall through atoi() to 0 == "off",
//        i.e. silently inherit the global MKL_NUM_THREADS (28 in
//        production) — the measured-24x-slower configuration this knob
//        exists to prevent, with a mysteriously slow eigh as the only
//        symptom.  Doctrine 3/pattern #8: a malformed explicit request
//        must not silently select the known-bad policy; falling back to
//        AUTO is the safe direction and the announcement makes it
//        diagnosable.
// ---------------------------------------------------------------------------
// Mechanism (resolver, pin RAII, str_ieq) is shared — see the header note in
// cpp/common/mkl_thread_pin.h.  These aliases keep every existing
// `lorrax_ffi::scalapack::`-qualified use site (eigh_ffi.cc:297,
// solve_lu_ffi.cc:189) compiling unchanged.
using mklpin::mkl_set_local_fn;
using mklpin::mkl_get_max_fn;
using mklpin::mkl_set_num_threads_local_ptr;
using mklpin::mkl_get_max_threads_ptr;
using mklpin::str_ieq;

// Threads to pin inside ScaLAPACK handlers; 0 = leave MKL alone.
inline int scalapack_mkl_threads() {
    static int resolved = [] {
        const auto auto_policy = [] {
            auto get_max = mkl_get_max_threads_ptr();
            if (get_max == nullptr) return 0;  // not MKL — no pin
            const int cur = get_max();
            return cur > 4 ? 4 : 0;            // cap only if it would shrink
        };
        // Strict full-string grammar via the SHARED parser
        // (mklpin::parse_thread_knob — the parse used to exist three times,
        // here + mklblas + mklfft, and had drifted; 2026-07-31 audit).
        // Only the parse is shared: atoi() previously mapped any typo to
        // 0 == "off" (the measured-24x-slower policy) and silently accepted
        // trailing junk ("4x" -> 4).  min 0: "0" == "off" (documented).
        const char* v = std::getenv("LORRAX_SCALAPACK_MKL_THREADS");
        const mklpin::ThreadKnob k = mklpin::parse_thread_knob(v, 0, 4096);
        switch (k.kind) {
            case mklpin::ThreadKnob::kOff: return 0;
            case mklpin::ThreadKnob::kInt: return k.value;
            case mklpin::ThreadKnob::kBad: break;   // announce below
            case mklpin::ThreadKnob::kAuto:
            default: return auto_policy();
        }
        // Unrecognized value: announce loudly ONCE (this init runs once
        // per process; rank-scoped since the 2026-07-31 announce-hygiene
        // pass — one line per JOB, not per rank) and take the safe
        // direction — AUTO, never "off" (see the grammar note above;
        // audit fix/zq 2026-07-28).
        if (mklpin::announce_here()) {
            std::fprintf(
                stderr,
                "*** LORRAX_SCALAPACK_MKL_THREADS='%s' is not a recognized "
                "value (accepted, case-insensitive: 'auto', 'off', '0', or a "
                "positive integer <= 4096).  Falling back to 'auto' (cap the "
                "handler-local MKL team at 4) — NOT to 'off', which would "
                "silently inherit the global MKL_NUM_THREADS (measured 24x "
                "slower pzheevd at the 12x12 production grid, workstream AW). "
                "***\n",
                v);
        }
        return auto_policy();
    }();
    return resolved;
}

// RAII: pin the calling thread's MKL team for the duration of a handler
// body; restore the previous thread-local setting on exit.  No-op when
// the pin is disabled or MKL is absent.  Defined in
// cpp/common/mkl_thread_pin.h; aliased here so the two ScaLAPACK handlers
// keep spelling it `lorrax_ffi::scalapack::MklThreadScope`.
using mklpin::MklThreadScope;

}  // namespace lorrax_ffi::scalapack
