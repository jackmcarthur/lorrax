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

#include <complex>
#include <cstdint>
#include <map>
#include <mutex>

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

}  // namespace lorrax_ffi::scalapack
