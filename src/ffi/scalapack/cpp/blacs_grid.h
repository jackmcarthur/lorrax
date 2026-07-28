// blacs_grid.h — the ScaLAPACK/BLACS seam shared by every handler in
// src/ffi/scalapack/cpp/.
//
// ScaLAPACK ships no C header we can rely on across vendors (MKL's
// mkl_scalapack.h exists but is MKL-only and drags in mkl.h), so the
// Fortran ABI is declared by hand here — ONCE — together with the
// per-SlateCtx BLACS context cache the handlers share.  32-bit integer
// interface (LP64: `libmkl_scalapack_lp64` on Frontera, `libsci_*` on
// Cray); every `int` below is the ScaLAPACK integer.
//
// Grid contract (identical for every handler): the BLACS grid is built on
// the SlateCtx's rank-remapped comm (src/ffi/slate/cpp/context.cc) with
// BLACS "C" (column-major) grid order, so comm rank (mx + my*Px) lands at
// grid coords (mx, my) == the JAX mesh coords.  Combined with the square
// block g = N / max(Px, Py) the block-cyclic distribution then coincides
// exactly with JAX's contiguous ``P('x','y')`` shards on square and 1-D
// meshes — which is why those are the only meshes the wrappers accept.
//
// The context is created COLLECTIVELY on first use per SlateCtx and cached
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

#include <dlfcn.h>
#include <mpi.h>

#include "../../slate/cpp/ctx.h"

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

// BLACS context per SlateCtx, created collectively on first use.  The map
// is tiny (one entry per mesh shape) and lives for the process.  Inline =>
// exactly one cache across all translation units of the library.
inline int blacs_ctxt_for(lorrax_ffi::slate::SlateCtx* ctx) {
    static std::mutex mu;
    static std::map<int64_t, int> cache;
    const int64_t key = reinterpret_cast<int64_t>(ctx);
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    int ictxt = Csys2blacs_handle(ctx->comm);
    // "C": comm rank r -> grid (r % p, r / p).  With the SlateCtx remap
    // (context.cc) this lands JAX shard (mx, my) at grid (mx, my).
    Cblacs_gridinit(&ictxt, "C", ctx->p, ctx->q);
    cache[key] = ictxt;
    return ictxt;
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
//  Mechanism: mkl_set_num_threads_local on the CALLING thread only, via
//  dlsym(RTLD_DEFAULT) so this header adds no MKL link dependency and the
//  pin degrades to a no-op on a non-MKL ScaLAPACK (LibSci).  The global
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
using mkl_set_local_fn = int (*)(int);
using mkl_get_max_fn   = int (*)();

inline mkl_set_local_fn mkl_set_num_threads_local_ptr() {
    static mkl_set_local_fn fn = reinterpret_cast<mkl_set_local_fn>(
        dlsym(RTLD_DEFAULT, "MKL_Set_Num_Threads_Local"));
    return fn;
}

inline mkl_get_max_fn mkl_get_max_threads_ptr() {
    static mkl_get_max_fn fn = reinterpret_cast<mkl_get_max_fn>(
        dlsym(RTLD_DEFAULT, "MKL_Get_Max_Threads"));
    return fn;
}

// Case-insensitive string equality (ASCII; env-value vocabulary only).
inline bool str_ieq(const char* a, const char* b) {
    for (;; ++a, ++b) {
        const int ca = std::tolower(static_cast<unsigned char>(*a));
        const int cb = std::tolower(static_cast<unsigned char>(*b));
        if (ca != cb) return false;
        if (ca == 0) return true;
    }
}

// Threads to pin inside ScaLAPACK handlers; 0 = leave MKL alone.
inline int scalapack_mkl_threads() {
    static int resolved = [] {
        const auto auto_policy = [] {
            auto get_max = mkl_get_max_threads_ptr();
            if (get_max == nullptr) return 0;  // not MKL — no pin
            const int cur = get_max();
            return cur > 4 ? 4 : 0;            // cap only if it would shrink
        };
        const char* v = std::getenv("LORRAX_SCALAPACK_MKL_THREADS");
        if (!v || !*v || str_ieq(v, "auto")) return auto_policy();
        if (str_ieq(v, "off")) return 0;
        // Strict full-string integer parse: atoi() previously mapped any
        // typo to 0 == "off" (the measured-24x-slower policy) and
        // silently accepted trailing junk ("4x" -> 4).
        char* end = nullptr;
        const long parsed = std::strtol(v, &end, 10);
        if (end != v && *end == '\0' && parsed >= 0 && parsed <= 4096) {
            return static_cast<int>(parsed);   // "0" == "off" (documented)
        }
        // Unrecognized value: announce loudly ONCE (this init runs once
        // per process) and take the safe direction — AUTO, never "off"
        // (see the grammar note above; audit fix/zq 2026-07-28).
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
        return auto_policy();
    }();
    return resolved;
}

// RAII: pin the calling thread's MKL team for the duration of a handler
// body; restore the previous thread-local setting on exit.  No-op when
// the pin is disabled or MKL is absent.
class MklThreadScope {
  public:
    explicit MklThreadScope(int nthreads) {
        if (nthreads <= 0) return;
        auto set_local = mkl_set_num_threads_local_ptr();
        if (set_local == nullptr) return;
        prev_ = set_local(nthreads);
        active_ = true;
    }
    ~MklThreadScope() {
        if (active_) {
            mkl_set_num_threads_local_ptr()(prev_);
        }
    }
    MklThreadScope(const MklThreadScope&) = delete;
    MklThreadScope& operator=(const MklThreadScope&) = delete;

  private:
    bool active_ = false;
    int  prev_ = 0;   // 0 = "follow the global setting" per MKL docs
};

}  // namespace lorrax_ffi::scalapack
