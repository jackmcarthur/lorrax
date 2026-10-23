// context.cc — the MPI mesh context lifecycle (init MPI if needed, split
// MPI_COMM_WORLD with the JAX rank remap, tear down).  C entry points bound by
// distrib_la.loader via ctypes.  Pure MPI: the ScaLAPACK (host) and ELPA (ROCM)
// handlers run on this comm.
//
// The MPI_Init_thread call is a no-op if phdf5 has already done it (the usual
// case: phdf5_init_mpi() runs early in the driver).  Kept here so this FFI is
// usable without phdf5.

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <unistd.h>
#include <mpi.h>

#include "ctx.h"
#include "../common/c_abi.h"
#include "../common/ctx_registry.h"
#include "../common/mpi_thread_guard.h"

namespace lorrax_ffi::mesh {

// GUARD (2026-07-30 FFI divergence audit; the phdf5 twin is
// cpp/phdf5/context.cc).  Requesting MPI_THREAD_MULTIPLE proves nothing,
// because the request is a no-op whenever someone else initialised MPI first:
// jax's MPI CPU collectives via MPItrampoline request FUNNELED.  The measured
// consequence of running concurrent MPI progress below MULTIPLE on this stack
// was a ~29% provider-independent segfault/hang rate at P=16 x 8 nodes
// (scorecard AS.4b).  The query runs on BOTH paths — after we initialise AND
// after an early return because someone else already did — since the
// already-initialised path is the hazardous one.  Mechanism shared with the
// phdf5 twin in cpp/common/mpi_thread_guard.h.
static void ensure_mpi_initialized() {
    int inited = 0;
    MPI_Initialized(&inited);
    if (!inited) {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        // MPICH needs srun --mpi=cray_shasta (not pmi2 / pmix) inside the
        // shifter gpu,mpich module to bootstrap a multi-rank MPI_COMM_WORLD.
        // `provided` is NOT inspected here on purpose: MPI_Query_thread in
        // the guard answers the same question on both paths.
    }
    lorrax_ffi::mpiguard::require_thread_multiple(
        "mesh", "the ScaLAPACK/ELPA handlers' MPI collectives from XLA threads");
}

}  // namespace lorrax_ffi::mesh

extern "C" {

// Create a mesh context for the given (p, q) grid.  Returns an opaque handle
// (pointer cast to int64_t).  All ranks must call this collectively.
int64_t LRX_C_ENTRY(lrx_mesh_context_create)(
    int rank, int world_size, int p, int q,
    char* err_buf, int err_buf_len)
{
    try {
        if (p <= 0 || q <= 0 || p * q != world_size) {
            if (err_buf_len > 0) {
                std::snprintf(err_buf, err_buf_len,
                    "mesh: invalid grid p=%d q=%d world=%d (need p*q==world)",
                    p, q, world_size);
            }
            return 0;
        }

        lorrax_ffi::mesh::ensure_mpi_initialized();

        auto* ctx = new lorrax_ffi::mesh::MeshCtx{};
        ctx->rank = rank;
        ctx->world_size = world_size;
        ctx->p = p;
        ctx->q = q;

        // Comm remap so a column-major process grid (BLACS "C", ELPA's
        // process_row/col) puts tile (mx, my) on the process holding JAX's
        // P('x','y') shard (mx, my):
        //   JAX: shard (mx, my) -> process (mx*q + my)  (C-order reshape)
        //   Col: tile (ti, tj)  -> rank (ti + tj*p)
        // key = mx + my*p = (jax_rank/q) + (jax_rank%q)*p.  Identity when
        // p==1 or q==1.
        const int new_rank = (rank / q) + (rank % q) * p;
        MPI_Comm dup = MPI_COMM_NULL;
        int rc = MPI_Comm_split(MPI_COMM_WORLD, 0, new_rank, &dup);
        if (rc != MPI_SUCCESS) {
            delete ctx;
            if (err_buf_len > 0) {
                std::snprintf(err_buf, err_buf_len,
                    "mesh: MPI_Comm_split failed rc=%d", rc);
            }
            return 0;
        }
        ctx->comm = dup;
        ctx->owns_comm = true;

        return reinterpret_cast<int64_t>(ctx);
    } catch (const std::exception& ex) {
        if (err_buf_len > 0) {
            std::snprintf(err_buf, err_buf_len,
                "mesh: context_create threw: %s", ex.what());
        }
        return 0;
    }
}

void LRX_C_ENTRY(lrx_mesh_context_destroy)(int64_t handle) {
    // A cached executable run after teardown must refuse, not dereference.
    lorrax_ffi::ctx_registry::forget_handle(handle);
    auto* ctx = reinterpret_cast<lorrax_ffi::mesh::MeshCtx*>(handle);
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
void LRX_C_ENTRY(lrx_mesh_init_mpi)() {
    lorrax_ffi::mesh::ensure_mpi_initialized();
}

}  // extern "C"
