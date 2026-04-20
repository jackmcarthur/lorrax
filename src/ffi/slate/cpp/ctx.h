// ctx.h — shared SlateCtx used by context.cc and eigh_ffi.cc.
#pragma once

#include <cstdint>
#include <mpi.h>

namespace lorrax_ffi::slate {

struct SlateCtx {
    // identity
    int rank = -1;
    int world_size = 0;
    int p = 0, q = 0;

    // MPI: SLATE needs MPI_THREAD_MULTIPLE.  We build a dup'd
    // MPI_COMM_WORLD with ranks REORDERED so SLATE's hardcoded
    // GridOrder::Col in fromDevices matches JAX's P('x','y') shard layout.
    //
    // JAX mesh layout (Mesh(reshape(p, q), ('x','y'))) puts shard (x, y)
    // on process x*q + y.
    //
    // SLATE GridOrder::Col (fromDevices default) puts tile (ti, tj) on
    // rank ti + tj*p.
    //
    // For JAX shard (x, y) to land on SLATE tile (x, y), we need:
    //   SLATE_rank(tile (x, y)) == JAX_proc(shard (x, y))
    //   x + y*p              == x*q + y
    // Only true when p == q == 1.  Otherwise remap: make the FFI comm
    // assign new rank = x + y*p to process (x*q + y), i.e. to JAX rank
    // (jax_rank / q) + (jax_rank % q) * p.
    MPI_Comm comm = MPI_COMM_NULL;
    bool     owns_comm = false;

    // Default 2-D block-cyclic tile size (nb).  n/p is the simplest choice
    // (one SLATE tile per rank = matches JAX's block sharding, no
    // reshuffling).  Smaller nb = more parallelism in the panel factor but
    // more communication.  256 is a reasonable middle ground for A100.
    int64_t default_nb = 256;
};

}  // namespace lorrax_ffi::slate
