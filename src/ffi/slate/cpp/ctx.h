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

    // MPI: SLATE needs MPI_THREAD_MULTIPLE.  We dup MPI_COMM_WORLD so
    // SLATE's internal splits/teardown don't affect phdf5 or other FFI MPI
    // users in the same process.
    MPI_Comm comm = MPI_COMM_NULL;
    bool     owns_comm = false;

    // Default 2-D block-cyclic tile size (nb).  n/p is the simplest choice
    // (one SLATE tile per rank = matches JAX's block sharding, no
    // reshuffling).  Smaller nb = more parallelism in the panel factor but
    // more communication.  256 is a reasonable middle ground for A100.
    int64_t default_nb = 256;
};

}  // namespace lorrax_ffi::slate
