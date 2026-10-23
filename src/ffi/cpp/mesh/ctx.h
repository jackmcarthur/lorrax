// ctx.h — LORRAX's MPI mesh context (MeshCtx), shared by context.cc and the
// ScaLAPACK (host) and ELPA (ROCM) handlers.
#pragma once

#include <cstdint>
#include <mutex>
#include <mpi.h>

namespace lorrax_ffi::mesh {

// One MeshCtx per (p, q) mesh shape.  Built by lrx_mesh_context_create in
// context.cc, lifetime-managed by distrib_la._mesh_ctx's atexit.
//
// The comm is MPI_COMM_WORLD split with a rank reorder, so a column-major
// process grid on it matches JAX's P('x','y') shard layout:
//   JAX shard (x, y)            -> JAX rank (x*q + y)
//   column-major grid (ti, tj)  -> comm rank (ti + tj*p)
// MPI_Comm_split with key = (jax_rank / q) + (jax_rank % q) * p.
// Identity for p==1 or q==1; nontrivial pairwise swaps for p, q > 1.
struct MeshCtx {
    int      rank = -1;            // JAX-side rank
    int      world_size = 0;
    int      p = 0, q = 0;
    MPI_Comm comm = MPI_COMM_NULL; // the remapped comm
    bool     owns_comm = false;
};

// Serialization for the HOST-platform handlers (scalapack/*_ffi.cc).  XLA's
// CPU runtime may execute independent ffi_calls concurrently on its intra-op
// thread pool; two handlers issuing MPI collectives on the same comm from
// different threads match collectives in a rank-dependent order — silent
// corruption (observed e2e: per-q potrf calls, one corrupted q tile,
// intermittent).  Device handlers are serialized by the XLA stream and do not
// take this lock.  One mutex per shared object (inline static), covering every
// comm the ctx cache hands out.
inline std::mutex& host_collective_mutex() {
    static std::mutex m;
    return m;
}

}  // namespace lorrax_ffi::mesh
