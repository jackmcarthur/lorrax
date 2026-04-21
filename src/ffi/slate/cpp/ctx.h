// ctx.h — shared SlateCtx used by context.cc and the per-op handlers.
#pragma once

#include <cstdint>
#include <mpi.h>

namespace lorrax_ffi::slate {

// One SlateCtx per (p, q) mesh shape.  Built by lrx_slate_context_create
// in context.cc, lifetime-managed by the Python wrapper's atexit.
//
// MPI: SLATE needs MPI_THREAD_MULTIPLE plus an MPI_Comm.  We dup
// MPI_COMM_WORLD with a rank reorder so SLATE's hardcoded
// GridOrder::Col in fromDevices matches JAX's P('x','y') shard layout
// (paired with the per-rank local-transpose on the Python side; see
// src/ffi/slate/README.md for the full layout rationale).
//
// Mapping (any p × q):
//   JAX shard (x, y)            -> JAX rank (x*q + y)
//   SLATE Col tile (ti, tj)     -> SLATE rank (ti + tj*p)
// Want JAX rank holding shard (mx, my) to be SLATE rank for tile
// (mx, my), so SLATE rank = mx + my*p.  MPI_Comm_split with
//   key = (jax_rank / q) + (jax_rank % q) * p.
// Identity for p==1 or q==1; nontrivial pairwise swaps for p, q > 1.
struct SlateCtx {
    int      rank = -1;            // JAX-side rank
    int      world_size = 0;
    int      p = 0, q = 0;
    MPI_Comm comm = MPI_COMM_NULL; // remapped comm passed to SLATE
    bool     owns_comm = false;
};

}  // namespace lorrax_ffi::slate
