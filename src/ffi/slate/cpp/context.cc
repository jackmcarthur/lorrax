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

#include <atomic>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <mpi.h>

#include "ctx.h"

namespace lorrax_ffi::slate {

static void ensure_mpi_initialized() {
    int inited = 0;
    MPI_Initialized(&inited);
    if (inited) return;
    int provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    // Important: MPICH needs srun --mpi=cray_shasta (not pmi2 / pmix)
    // inside the shifter gpu,mpich module to actually bootstrap a
    // multi-rank MPI_COMM_WORLD.  run_shifter.sh sets this default.  If
    // MPI_COMM_WORLD is singleton here on a world_size > 1 allocation,
    // check LORRAX_MPI_TYPE and/or the srun --mpi= flag.
}

}  // namespace lorrax_ffi::slate

extern "C" {

// Create a SLATE context for the given (p, q) grid.  Returns an opaque
// handle (pointer cast to int64_t).  All ranks must call this
// collectively; the dup'd MPI comm participates.
int64_t lrx_slate_context_create(
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

        MPI_Comm dup = MPI_COMM_NULL;
        int rc = MPI_Comm_dup(MPI_COMM_WORLD, &dup);
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

void lrx_slate_context_destroy(int64_t handle) {
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
void lrx_slate_init_mpi() {
    lorrax_ffi::slate::ensure_mpi_initialized();
}

}  // extern "C"
