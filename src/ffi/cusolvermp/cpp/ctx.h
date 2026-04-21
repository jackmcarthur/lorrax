// ctx.h — shared Ctx struct used by context.cc and eigh_ffi.cc.
#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <nccl.h>
#include <cal.h>
#include <cusolverMp.h>

namespace lorrax_ffi::cusolvermp {

// "Shim" struct passed through CAL's user-data slot to our three allgather
// callbacks.  Lives for the lifetime of the LorraxCusolverMpCtx.
struct CalNcclShim {
    ncclComm_t    nccl_comm  = nullptr;
    cudaStream_t  stream     = nullptr;
    // Scratch device buffer reused across allgathers; grown on demand.
    void*         d_scratch       = nullptr;
    size_t        d_scratch_bytes = 0;
};

struct LorraxCusolverMpCtx {
    // identity
    int rank = -1;
    int world_size = 0;
    int p = 0, q = 0;
    bool grid_layout_col_major = true;
    int local_device_id = 0;

    // owned resources
    ncclComm_t          nccl_comm  = nullptr;
    cudaStream_t        stream     = nullptr;
    cusolverMpHandle_t  handle     = nullptr;
    cal_comm_t          cal_comm   = nullptr;   // our CAL wrapper over NCCL
    cusolverMpGrid_t    grid       = nullptr;
    CalNcclShim         shim{};                 // stable address — CAL holds a ptr

    // Pooled events for cross-stream joins.  Created once in ctor, destroyed
    // once in dtor.  cudaEventRecord on an already-recorded event just
    // updates the record point.  Avoids the +750 ms stalls that phdf5 hit
    // with per-call cudaEventDestroy under cuda_malloc_async (see
    // src/ffi/phdf5/ARCHITECTURE.md §2.2).
    cudaEvent_t  ev_xla_in  = nullptr;   // signal on xla_stream → wait on ctx
    cudaEvent_t  ev_ctx_out = nullptr;   // signal on ctx_stream → wait on xla

    // per-call scratch reused across invocations
    void*   d_workspace       = nullptr;
    size_t  d_workspace_bytes = 0;
    void*   h_workspace       = nullptr;
    size_t  h_workspace_bytes = 0;
    int*    d_info            = nullptr;
};

// Grow (d_workspace, h_workspace) if needed; keeps largest allocation.
void ensure_workspace(LorraxCusolverMpCtx* ctx, size_t d_need, size_t h_need);

}  // namespace lorrax_ffi::cusolvermp
