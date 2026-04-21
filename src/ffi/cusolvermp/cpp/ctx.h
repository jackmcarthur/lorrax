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

// ---------------------------------------------------------------------------
//  Sub-row context — per-X-row sub-comm of the process mesh.
// ---------------------------------------------------------------------------
// Used by the batched potrf/potrs path.  Each X-row of a (Px, Py) process
// mesh is an independent comm of size Py, built via
// `ncclCommSplit(world, color=x_rank, key=y_rank)`.  Each such sub-comm
// gets its own `cal_comm_t` (wrapping a per-sub-comm `CalNcclShim`) and
// its own `cusolverMpGrid_t` of shape (1, Py).
//
// cuSOLVERMp restriction: "only one handle per process per GPU" — so this
// struct holds its own handle.  If the world-wide `LorraxCusolverMpCtx`
// has already created a handle on this process, creating a second one
// will fail.  In practice the batched path is used in isolation
// (GWJAX-style workflows), not mixed with world-wide eigh.
struct LorraxCusolverMpSubRowCtx {
    // identity
    int Px = 0, Py = 0;          // process-mesh shape
    int x_rank = -1, y_rank = -1;
    int world_rank = -1;
    int local_device_id = 0;

    // owned resources
    ncclComm_t          nccl_comm = nullptr;   // per-X-row sub-comm (size Py)
    cudaStream_t        stream    = nullptr;
    cusolverMpHandle_t  handle    = nullptr;
    cal_comm_t          cal_comm  = nullptr;
    cusolverMpGrid_t    grid      = nullptr;   // (1, Py)
    CalNcclShim         shim{};

    // Pooled cross-stream join events (see LorraxCusolverMpCtx for rationale).
    cudaEvent_t  ev_xla_in  = nullptr;
    cudaEvent_t  ev_ctx_out = nullptr;

    // reused scratchpads
    void*   d_workspace       = nullptr;
    size_t  d_workspace_bytes = 0;
    void*   h_workspace       = nullptr;
    size_t  h_workspace_bytes = 0;
    int*    d_info            = nullptr;
};

void ensure_subrow_workspace(LorraxCusolverMpSubRowCtx* ctx,
                             size_t d_need, size_t h_need);

}  // namespace lorrax_ffi::cusolvermp
