// blacs_grid.h — the ScaLAPACK/BLACS seam shared by every handler in
// src/ffi/cpp/scalapack/.
//
// ScaLAPACK ships no C header we can rely on across vendors (MKL's
// mkl_scalapack.h exists but is MKL-only and drags in mkl.h), so the
// Fortran ABI is declared by hand here — ONCE — together with the
// per-SlateCtx BLACS context cache the handlers share.  32-bit integer
// interface (LP64: `libmkl_scalapack_lp64` on Frontera, `libsci_*` on
// Cray); every `int` below is the ScaLAPACK integer.
//
// Grid contract (identical for every handler): the BLACS grid is built on
// the SlateCtx's rank-remapped comm (src/ffi/cpp/slate/context.cc) with
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
#include <string>

#include <dlfcn.h>
#include <mpi.h>

#include "../common/mkl_thread_pin.h"
#include "../slate/ctx.h"

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

// General matrix multiply (gemm_ffi.cc).
void pdgemm_(const char* transa, const char* transb,
             const int* m, const int* n, const int* k,
             const double* alpha,
             const double* a, const int* ia, const int* ja, const int* desca,
             const double* b, const int* ib, const int* jb, const int* descb,
             const double* beta,
             double* c, const int* ic, const int* jc, const int* descc);
void pzgemm_(const char* transa, const char* transb,
             const int* m, const int* n, const int* k,
             const std::complex<double>* alpha,
             const std::complex<double>* a, const int* ia, const int* ja,
             const int* desca,
             const std::complex<double>* b, const int* ib, const int* jb,
             const int* descb,
             const std::complex<double>* beta,
             std::complex<double>* c, const int* ic, const int* jc,
             const int* descc);

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
//  WHICH LIBRARY IS ACTUALLY BEHIND THESE THIRTEEN NAMES  (2026-08-16)
//
//  ScaLAPACK is an API with several implementations, and this file's whole
//  premise is that LORRAX does not care which one answers — MKL on Frontera,
//  Cray LibSci on Perlmutter, netlib, AOCL.  There is exactly ONE
//  implementation for which that indifference is false, and it is reachable
//  by accident, so it is detected here.
//
//  SLATE ships an OPTIONAL ScaLAPACK-compatibility layer
//  (`scalapack_api/`, built as `libslate_scalapack_api.so`) whose documented
//  use is `LD_PRELOAD` interception: it re-DEFINES the ScaLAPACK entry
//  points and forwards them to `slate::`.  MEASURED against this repo's own
//  thirteen names (SLATE v2025.05.28, built 2026-07-31, `nm -D`):
//
//      DEFINED (8)  pzheevd_ pdsyevd_ pzgetrf_ pdgetrf_ pzgetrs_ pdgetrs_
//                   pzgemm_ pdgemm_
//      UNDEF   (2)  numroc_ Cblacs_gridinfo      <- it CALLS these
//      absent  (3)  descinit_ Csys2blacs_handle Cblacs_gridinit
//
//  So it is an OVERLAY, never a provider: it must sit on top of a real
//  ScaLAPACK+BLACS (it also consumes Cblacs_get / Cblacs_pcoord /
//  Cblacs_pinfo / indxl2g_).  The eight it does define are exactly the eight
//  compute routines LORRAX calls, i.e. an `LD_PRELOAD` — or an
//  `-lslate_scalapack_api` ahead of MKL in LORRAX_SCALAPACK_LIBRARIES —
//  silently replaces EVERY solve these two handlers make, with nothing on
//  the Python side able to observe it: `ffi_loader` keys only on LORRAX's
//  own handler symbols (ScalapackEighHostFfi, ...), so
//  `resolve_backend('eigh', 'scalapack')` still returns 'scalapack' and
//  still promises a callable backend.
//
//  IT DOES RUN — that is exactly why it needs a guard rather than a shrug.
//  MEASURED end to end (job 7883874, 1x1 CPU mesh, this .so, the overlay in
//  LD_PRELOAD, waiver set): eigh `max|dW| = 1.24e-14`,
//  `||AZ-ZW||/||A|| = 1.44e-15`, `||Z^H Z - I|| = 8.43e-15`; LU
//  `||AX-B||/||B|| = 1.23e-14`.  Both CORRECT, no SIGSEGV.  An interposition
//  that merely crashed would be self-limiting; one that returns machine
//  precision on the geometry you test and silently wrong answers on the
//  geometry you ship is not.
//
//  THREE REASONS SUBSTITUTION IS NOT SAFE HERE.  The first two are read
//  straight out of the shim source and hold at every size:
//
//   1. THE RANK<->SHARD MAPPING IS THE WHOLE DEFECT, AND IT IS EXACTLY ONE
//      PERMUTATION.  Every shim body builds its SLATE matrix with
//      `fromScaLAPACK(..., MPI_COMM_WORLD)`, hard-coded
//      (scalapack_getrf.cc:109, scalapack_getrs.cc:116, heevd:134), i.e. it
//      wants shard (mx,my) on WORLD rank mx+my*p (GridOrder::Col).  LORRAX
//      puts it on mx*q+my (C-order mesh reshape) and compensates inside
//      SlateCtx with a rank remap (cpp/slate/ctx.h).  The two agree only
//      when p==1 or q==1.
//
//      MEASURED (job 7883978, 4 ranks, 2x2 mesh, n=64) — the mesh's device
//      order is the ONLY variable, and it swaps which provider is right:
//
//        mesh device order   provider   eigh                    LU
//        C (what we ship)    MKL        CORRECT   8.5e-14       CORRECT 4.3e-16
//        C (what we ship)    SLATE      WRONG     1.5e-01       WRONG   1.55e-01
//        F (Fortran)         MKL        WRONG     1.5e-01       WRONG   1.55e-01
//        F (Fortran)         SLATE      CORRECT   1.4e-15       CORRECT 4.3e-16
//
//      The two WRONG rows agree to four digits, which is what makes this a
//      proof rather than a coincidence: it is one permutation, applied once
//      too few times or once too many.  On a 4x1 mesh (remap == identity)
//      the overlay's LU is CORRECT (4.33e-16) and its eigh REFUSES loudly
//      (SLATE requires a square process grid, heev.cc:102).
//
//      SO: the overlay is NOT broken, and this is NOT merely "unvalidated"
//      — under the mesh order LORRAX currently ships it is wrong by ~15%
//      with no error of any kind, on every production shape (4x4, 8x8,
//      12x12).  The 1x1 numbers above are from the one geometry that
//      cannot show it.
//
//      THE FIX IS LORRAX-SIDE, NEEDS NO UPSTREAM PATCH, AND IS NOT FREE:
//      a Fortran-order mesh makes the overlay correct at machine precision
//      — and makes MKL wrong on the same mesh.  They are mutually
//      exclusive, so the device order becomes a PROVIDER-DEPENDENT choice.
//      The clean version is to stop assuming C-order here: derive this
//      BLACS grid's permutation from the mesh's ACTUAL device order instead
//      of hard-coding the C-order remap, after which MKL is correct under
//      both orders and F-order is a free knob that enables the overlay.
//      That touches every consumer of the mesh and needs its own gate.
//      Reproducer: wk_REL/harness/slalias_mesh.sbatch (6 cells, MKL controls
//      on both orders).
//
//      `slate_scalapack_blacs_grid_order()` (scalapack_slate.hh:60) is a
//      second, independent way to get this wrong: it infers row/col-major
//      from `Cblacs_pcoord` on the DEFAULT system context
//      (`Cblacs_get(-1,0,...)`), not from desca's context — which is not
//      the grid these handlers built.
//   2. `*info` IS HARD-WIRED TO 0 ("todo: extract the real info from
//      getrf/heevd" — scalapack_getrf.cc:147, scalapack_heevd.cc:113).  Both
//      handlers below report success iff info==0, so a singular pivot or a
//      non-converged block returns garbage as a clean result.  That deletes
//      the only failure signal these handlers have.
//   3. eigh lands in the routine bug L-2 is recorded against — and the
//      measurement says L-2 IS NOT IN THAT ROUTINE.  `slate_pheevd`
//      (scalapack_heevd.cc:146) is a direct call to `slate::heev`, the host
//      routine `ffi/linalg/resolve.py` guard 2b refuses because it "SIGSEGVs
//      deterministically ... n = 64/512/1200, mesh 1x1".  Through the
//      overlay it returned CORRECT at n = 32/64/128/512 on a 1x1 mesh
//      (job 7883880: max|dW| 1.24e-14 -> 5.76e-13, ||AZ-ZW||/||A|| <= 3.2e-15),
//      while in the SAME job, SAME mesh and sizes, `ffi.slate`'s own host
//      handler died with SIGSEGV (rc 139) at n=32 AND n=64.  Same
//      libslate.so, same MKL, same process image, opposite outcomes — so the
//      fault is in LORRAX's `cpp/slate/host_ffi.cc` call path (its Z is
//      SLATE-managed + copied out tile-by-tile; the shim's Z wraps the user
//      buffer via fromScaLAPACK, and it passes MaxPanelThreads /
//      InnerBlocking / an explicit grid_order that host_ffi.cc does not),
//      NOT in `slate::heev` as L-2 records.  Reproducer:
//      wk_REL/harness/slalias_l2.sbatch.  That does not make the overlay
//      safe — reasons 1 and 2 are untouched — but the L-2 sentence in
//      docs/dev/linalg_ffi.md needs correcting, and the CPU SLATE eigh
//      capability may be recoverable.
//
//  Hence: DETECT and REFUSE, rather than trust or ignore.  `dlsym` +
//  `dladdr` answer "which object defines the symbol the PLT will bind" —
//  `dlsym(RTLD_DEFAULT, ...)` (via the shared resolver) returns the
//  DEFINITION address, not this library's PLT stub, so `dladdr` names the
//  real provider.  Cached once per name per process.
//
//  Escape hatch: LORRAX_SCALAPACK_ALLOW_SLATE_API=1 (or 'on'/'true'/'yes')
//  downgrades the refusal to one loud stderr line, for deliberately
//  measuring the SLATE route.  Any other value announces itself and takes
//  the safe direction (refuse) — the same grammar discipline as
//  LORRAX_SCALAPACK_MKL_THREADS below.
// ---------------------------------------------------------------------------

// Object that defines `name` in this process's resolution scope, or "" when
// the symbol does not resolve at all.  Empty is NOT an error here: it only
// means the provenance question cannot be answered (a static ScaLAPACK, a
// stripped provider), and every caller treats "unknown" as "not SLATE".
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

// True when `name` is answered by SLATE's ScaLAPACK-compatibility overlay.
// Matched on the SONAME substring, which is upstream's fixed library name
// (GNUmakefile: lib/libslate_scalapack_api.${so}) and is what both the
// LD_PRELOAD and the -l link routes produce.
inline bool scalapack_symbol_is_slate_api(const char* name) {
    const std::string& lib = scalapack_symbol_provider(name);
    return lib.find("libslate_scalapack_api") != std::string::npos;
}

// Was the refusal deliberately waived?  Resolved once; a malformed value is
// announced and takes the safe direction (refuse), never the permissive one.
inline bool scalapack_slate_api_allowed() {
    static const bool allowed = [] {
        const char* v = std::getenv("LORRAX_SCALAPACK_ALLOW_SLATE_API");
        if (v == nullptr || *v == '\0') return false;
        using lorrax_ffi::mklpin::str_ieq;
        if (str_ieq(v, "1") || str_ieq(v, "on") || str_ieq(v, "true")
            || str_ieq(v, "yes")) {
            return true;
        }
        if (str_ieq(v, "0") || str_ieq(v, "off") || str_ieq(v, "false")
            || str_ieq(v, "no")) {
            return false;
        }
        std::fprintf(
            stderr,
            "*** LORRAX_SCALAPACK_ALLOW_SLATE_API='%s' is not a recognized "
            "value (accepted, case-insensitive: 1/on/true/yes, "
            "0/off/false/no).  Taking the SAFE direction — the SLATE "
            "ScaLAPACK overlay stays REFUSED. ***\n",
            v);
        return false;
    }();
    return allowed;
}

// WHICH DEVICE the overlay will use, and whether that was chosen or
// inherited.  SLATE's shim picks its execution target from the environment
// (`SLATE_SCALAPACK_TARGET`, scalapack_slate.hh:170-188) and DEFAULTS TO
// `HostTask` — so a SLATE built `gpu_backend=cuda` still runs on the CPU
// unless someone sets it.  That is a silent demotion of exactly the
// capability this route exists to provide, and doctrine 3 says a demotion
// must announce itself.  The value is a per-process env fact, so it is
// reported from the same rank and the same line as the provider.
inline const char* scalapack_slate_target_note() {
    const char* v = std::getenv("SLATE_SCALAPACK_TARGET");
    if (v == nullptr || *v == '\0') {
        return "SLATE_SCALAPACK_TARGET is UNSET, so SLATE will run on the "
               "CPU (its default is HostTask) EVEN IF this SLATE was built "
               "gpu_backend=cuda/hip -- set it to 'devices' for the GPU";
    }
    if (lorrax_ffi::mklpin::str_ieq(v, "devices")) {
        return "SLATE_SCALAPACK_TARGET=devices, so SLATE will use the GPU if "
               "this build has one (gpu_backend=cuda/hip); a gpu_backend=none "
               "build silently stays on the CPU";
    }
    return "SLATE_SCALAPACK_TARGET is set to a host target, so SLATE will "
           "run on the CPU";
}

// One stderr line, once per process, naming the provider of `name`.  Used on
// the waived path so a deliberately-SLATE run is never silent about it.
inline void scalapack_announce_slate_api(const char* op, const char* name) {
    static std::mutex mu;
    static std::map<std::string, bool> said;
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!said.emplace(name, true).second) return;
    }
    if (!lorrax_ffi::mklpin::announce_here()) return;
    std::fprintf(
        stderr,
        "*** scalapack.%s: %s is provided by %s (SLATE's ScaLAPACK overlay), "
        "NOT by the ScaLAPACK this library was linked against.  Allowed only "
        "because LORRAX_SCALAPACK_ALLOW_SLATE_API is set.  It WILL return a "
        "plausible answer — measured to machine precision on a 1x1 mesh — "
        "and that is the hazard, not a reassurance: every shim hard-codes "
        "MPI_COMM_WORLD while this grid is the ctx's rank-remapped comm, so "
        "on any p>1,q>1 mesh (every production shape) it assembles tiles "
        "from the WRONG RANKS, silently; and every shim hard-wires info=0, "
        "so a singular or non-converged solve reports success.  Trust "
        "nothing measured here on a 2-D mesh.  DEVICE: %s. ***\n",
        op, name, scalapack_symbol_provider(name).c_str(),
        scalapack_slate_target_note());
    std::fflush(stderr);
}

// ---------------------------------------------------------------------------
//  WHO IS ANSWERING?  (LORRAX_SCALAPACK_PROVIDER_LOG)
//
//  The whole point of calling a published API is that any of several
//  packages can satisfy it — MKL here, Cray LibSci on a Cray, AOCL on AMD,
//  netlib anywhere, SLATE's overlay in front of any of them.  The cost of
//  that freedom is that NOTHING at run time otherwise says which one you
//  actually got: `ffi_loader` keys only on LORRAX's own handler symbols, and
//  a vendor named in a docstring or a directory name is decoration.
//
//  This makes the answer visible on demand rather than guessable.  It prints
//  the defining object for each of the THIRTEEN names, once per process, on
//  rank 0 (or every rank with `=all`), reusing the log grammar in
//  mkl_thread_pin.h.  Opt-in, because in a healthy build it is noise; the
//  moment a port misbehaves it is the first question worth asking.
// ---------------------------------------------------------------------------
inline void scalapack_log_providers_once() {
    // Resolved ONCE per process: this runs on EVERY handler dispatch (via
    // scalapack_slate_api_refusal), and the knob is a process-lifetime env
    // fact like every other knob in this header (the waiver above, the
    // thread pin below) — re-reading the environment per solve bought
    // nothing but a getenv on the hot path (2026-08-01 seam-audit leftover).
    static const bool enabled =
        lorrax_ffi::mklpin::log_here("LORRAX_SCALAPACK_PROVIDER_LOG");
    if (!enabled) return;
    static std::once_flag once;
    std::call_once(once, [] {
        static const char* kNames[] = {
            "pzheevd_", "pdsyevd_", "pzgetrf_", "pdgetrf_", "pzgetrs_",
            "pdgetrs_", "pzgemm_", "pdgemm_", "numroc_", "descinit_",
            "Csys2blacs_handle", "Cblacs_gridinit", "Cblacs_gridinfo"};
        std::fprintf(stderr,
                     "[scalapack] provider map (the 13 names this library "
                     "calls; anything implementing them can back it):\n");
        for (const char* n : kNames) {
            const std::string& lib = scalapack_symbol_provider(n);
            std::fprintf(stderr, "[scalapack]   %-18s <- %s\n", n,
                         lib.empty() ? "<unresolved>" : lib.c_str());
        }
        std::fflush(stderr);
    });
}

// The refusal text a handler returns when `name` is SLATE's and the waiver
// is not set.  Non-empty == refuse.
inline std::string scalapack_slate_api_refusal(const char* op,
                                               const char* name) {
    scalapack_log_providers_once();
    if (!scalapack_symbol_is_slate_api(name)) return std::string();
    if (scalapack_slate_api_allowed()) {
        scalapack_announce_slate_api(op, name);
        return std::string();
    }
    return std::string("scalapack.") + op + ": " + name + " resolves to "
        + scalapack_symbol_provider(name)
        + " — SLATE's ScaLAPACK-compatibility overlay has been interposed "
          "(LD_PRELOAD, or -lslate_scalapack_api ahead of the real "
          "ScaLAPACK in LORRAX_SCALAPACK_LIBRARIES).  REFUSING: this "
          "backend's promise is the LINKED ScaLAPACK, and the overlay is "
          "not equivalent to it UNDER THE MESH DEVICE ORDER LORRAX SHIPS.  "
          "(1) MEASURED (job 7883978, 4 ranks, 2x2 mesh, n=64): with the "
          "C-order mesh this library builds, the overlay returns eigh with "
          "||AZ-ZW||/||A||=1.5e-01 and ||Z^H Z-I||=6.98 and LU with "
          "||AX-B||/||B||=1.55e-01 — wrong by ~15%, silently, where MKL is "
          "correct to 1e-15 on the same mesh.  Cause: the shims hard-code "
          "MPI_COMM_WORLD and want shard (mx,my) on rank mx+my*p, while "
          "LORRAX puts it on mx*q+my; the two agree only when p==1 or q==1. "
          "Under a FORTRAN-order mesh the overlay is CORRECT (1.4e-15) and "
          "MKL is the wrong one — so this is one permutation, not a broken "
          "library, and it is fixable on LORRAX's side with no upstream "
          "patch.  Until that lands the two providers cannot share a mesh.  "
          "(2) Every shim hard-wires info=0 ('todo: extract the real "
          "info'), so a singular pivot or a non-converged block is reported "
          "as success — LORRAX's LU ridge covers the singular case, nothing "
          "covers a non-converged eigh.  (3) SLATE's heev additionally "
          "requires a SQUARE process grid and refuses a 4x1 outright.  "
          "Fix: drop it from LD_PRELOAD / the link line.  The route to "
          "making it work is src/ffi/PORTING.md §0b.  To measure it anyway, "
          "set LORRAX_SCALAPACK_ALLOW_SLATE_API=1.";
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
