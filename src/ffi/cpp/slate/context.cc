// context.cc — SLATE per-process context lifecycle (init MPI if needed,
// dup MPI_COMM_WORLD, tear down).  Exposes C entry points that ffi_loader
// binds via ctypes.
//
// SLATE requirements:
//   - MPI_Init_thread with MPI_THREAD_MULTIPLE (SLATE's internal OMP+MPI
//     pattern is not safe below that).
//   - An arbitrary MPI_Comm (not necessarily MPI_COMM_WORLD).  We dup WORLD
//     so our comm is independent of what phdf5 / cusolvermp do.
//
// The MPI_Init_thread call is no-op if phdf5 has already done it.  That's
// the usual case: phdf5_init_mpi() runs early in the driver and SLATE
// piggybacks.  Kept here anyway so this FFI is usable without phdf5.

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <unistd.h>
#include <mpi.h>

#include "ctx.h"
#include "../common/c_abi.h"
#include "../common/mpi_thread_guard.h"

namespace lorrax_ffi::slate {

// GUARD (2026-07-30 FFI divergence audit; the phdf5 twin is
// cpp/phdf5/context.cc:201-219).  SLATE's stated requirement at the top of
// this file — "MPI_Init_thread with MPI_THREAD_MULTIPLE (SLATE's internal
// OMP+MPI pattern is not safe below that)" — was REQUESTED here and never
// VERIFIED: `provided` was captured and discarded, and MPI_Query_thread
// appeared nowhere in cpp/slate/.  Requesting MULTIPLE proves nothing,
// because the request is a no-op whenever someone else initialised MPI
// first: jax's MPI CPU collectives via MPItrampoline request FUNNELED.  The
// measured consequence of
// running concurrent MPI progress below MULTIPLE on this stack was a ~29%
// provider-independent segfault/hang rate at P=16 x 8 nodes (3 segfaults +
// 1 hang in 14 runs; scorecard AS.4b).  The certified site launch must make
// the live grant MULTIPLE (patched adapter on Frontera; Cray async progress on
// Perlmutter). This guard refuses a hazardous configuration up front instead
// of dying minutes later with a backtrace that names neither cause nor fix.
//
// Placement matters: the query runs on BOTH paths — after we initialise AND
// after an early return because someone else already did.  The
// already-initialised path is the hazardous one, so a guard that only ran
// when we called MPI_Init_thread ourselves would be void by construction.
//
// Mechanism shared with the phdf5 twin in cpp/common/mpi_thread_guard.h
// (2026-08-01 dedup; the copies had begun to drift).  The family-specific
// hazard sentence stays here.
static void require_thread_multiple() {
    lorrax_ffi::mpiguard::require_thread_multiple(
        "slate", "SLATE's internal OpenMP+MPI task engine");
}

static void ensure_mpi_initialized() {
    int inited = 0;
    MPI_Initialized(&inited);
    if (!inited) {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        // Important: MPICH needs srun --mpi=cray_shasta (not pmi2 / pmix)
        // inside the shifter gpu,mpich module to actually bootstrap a
        // multi-rank MPI_COMM_WORLD.  run_shifter.sh sets this default.  If
        // MPI_COMM_WORLD is singleton here on a world_size > 1 allocation,
        // check LORRAX_MPI_TYPE and/or the srun --mpi= flag.
        //
        // `provided` is NOT inspected here on purpose: MPI_Query_thread
        // below answers the same question on both paths, and reading it
        // only here is precisely the bug this guard fixes.
    }
    require_thread_multiple();
}

}  // namespace lorrax_ffi::slate

extern "C" {

// Create a SLATE context for the given (p, q) grid.  Returns an opaque
// handle (pointer cast to int64_t).  All ranks must call this
// collectively; the dup'd MPI comm participates.
int64_t LRX_C_ENTRY(lrx_slate_context_create)(
    int rank, int world_size, int p, int q,
    char* err_buf, int err_buf_len)
{
    try {
        if (p <= 0 || q <= 0 || p * q != world_size) {
            if (err_buf_len > 0) {
                std::snprintf(err_buf, err_buf_len,
                    "slate: invalid grid p=%d q=%d world=%d (need p*q==world)",
                    p, q, world_size);
            }
            return 0;
        }

        lorrax_ffi::slate::ensure_mpi_initialized();

        auto* ctx = new lorrax_ffi::slate::SlateCtx{};
        ctx->rank = rank;
        ctx->world_size = world_size;
        ctx->p = p;
        ctx->q = q;

        // Comm remap so SLATE's hardcoded GridOrder::Col in fromDevices
        // assigns tile (mx, my) to the same physical process as JAX's
        // P('x','y') shard at mesh (mx, my).
        //   JAX: shard (mx, my) -> process (mx*q + my)  (C-order reshape)
        //   SLATE Col: tile (ti, tj) -> rank (ti + tj*p)
        // For (mx, my) -> (ti=mx, tj=my): SLATE rank = mx + my*p,
        // assigned to JAX process (mx*q + my).  MPI_Comm_split with
        // key = mx + my*p = (jax_rank/q) + (jax_rank%q)*p reorders the
        // group accordingly.  Trivial (identity) when p==1 or q==1.
        //
        // This pairs with JAX-side local-transpose (shard_map +
        // jnp.transpose) on inputs to make SLATE see the correct
        // global matrix on any p x q mesh.  See ffi.slate.cholesky /
        // ffi.slate.trsm for the Python side.
        const int new_rank = (rank / q) + (rank % q) * p;
        MPI_Comm dup = MPI_COMM_NULL;
        int rc = MPI_Comm_split(MPI_COMM_WORLD, 0, new_rank, &dup);
        if (rc != MPI_SUCCESS) {
            delete ctx;
            if (err_buf_len > 0) {
                std::snprintf(err_buf, err_buf_len,
                    "slate: MPI_Comm_dup failed rc=%d", rc);
            }
            return 0;
        }
        ctx->comm = dup;
        ctx->owns_comm = true;

        return reinterpret_cast<int64_t>(ctx);
    } catch (const std::exception& ex) {
        if (err_buf_len > 0) {
            std::snprintf(err_buf, err_buf_len,
                "slate: context_create threw: %s", ex.what());
        }
        return 0;
    }
}

// Create a SLATE context for the per-X-row Y-axis sub-communicator, given a
// full (Px, Py) process grid.  Intended for batched ops where each X-row
// cooperatively processes an independent slice of a (Nbatch, N, N) input
// (batch distributed along X, each N×N distributed across Y).
//
// All world ranks must call this collectively.  The call performs
// MPI_Comm_split with color = x_rank and key = y_rank, yielding a per-
// X-row sub-comm of size Py.  Since the sub-grid is (1, Py), SLATE's
// GridOrder::Col tile (0, j) → rank j already matches JAX's shard (0, j)
// → rank j — no further remap needed.
int64_t LRX_C_ENTRY(lrx_slate_subrow_context_create)(
    int rank, int world_size, int Px, int Py,
    char* err_buf, int err_buf_len)
{
    try {
        if (Px <= 0 || Py <= 0 || Px * Py != world_size) {
            if (err_buf_len > 0) {
                std::snprintf(err_buf, err_buf_len,
                    "slate subrow: invalid grid Px=%d Py=%d world=%d "
                    "(need Px*Py==world)", Px, Py, world_size);
            }
            return 0;
        }

        lorrax_ffi::slate::ensure_mpi_initialized();

        const int x_rank = rank / Py;
        const int y_rank = rank % Py;

        MPI_Comm sub = MPI_COMM_NULL;
        int rc = MPI_Comm_split(MPI_COMM_WORLD, x_rank, y_rank, &sub);
        if (rc != MPI_SUCCESS) {
            if (err_buf_len > 0) {
                std::snprintf(err_buf, err_buf_len,
                    "slate subrow: MPI_Comm_split failed rc=%d", rc);
            }
            return 0;
        }

        auto* ctx = new lorrax_ffi::slate::SlateCtx{};
        ctx->rank = y_rank;           // rank within sub-comm
        ctx->world_size = Py;         // size of sub-comm
        ctx->p = 1;
        ctx->q = Py;
        ctx->comm = sub;
        ctx->owns_comm = true;
        return reinterpret_cast<int64_t>(ctx);
    } catch (const std::exception& ex) {
        if (err_buf_len > 0) {
            std::snprintf(err_buf, err_buf_len,
                "slate subrow: context_create threw: %s", ex.what());
        }
        return 0;
    }
}

void LRX_C_ENTRY(lrx_slate_context_destroy)(int64_t handle) {
    auto* ctx = reinterpret_cast<lorrax_ffi::slate::SlateCtx*>(handle);
    if (ctx == nullptr) return;
    if (ctx->owns_comm && ctx->comm != MPI_COMM_NULL) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Comm_free(&ctx->comm);
        }
    }
    delete ctx;
}

// Early init hook — caller can invoke this before any jit/compile to hoist
// the MPI_Init_thread cost out of the hot path.  Idempotent.
void LRX_C_ENTRY(lrx_slate_init_mpi)() {
    lorrax_ffi::slate::ensure_mpi_initialized();
}

}  // extern "C"
