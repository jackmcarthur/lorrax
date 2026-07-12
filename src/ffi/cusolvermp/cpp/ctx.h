// ctx.h — shared Ctx struct used by context.cc and eigh_ffi.cc.
#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <nccl.h>
#include <cusolverMp.h>
#include <cublasmp.h>

// CAL note: cuSolverMp/cuBLASMp dropped the CAL communicator ABI at 0.7.0 /
// 0.5.0 respectively — cusolverMpCreateDeviceGrid and cublasMpGridCreate now
// take an ncclComm_t directly (see the cu13 headers).  The legacy cal_comm_t
// path (and the CAL->NCCL allgather shim it required) has been removed so the
// FFI builds against the pure-pip, CAL-free CUDA-13 stack.  See
// src/ffi/PORTING.md.

namespace lorrax_ffi::cusolvermp {

struct LorraxCusolverMpCtx {
    // identity
    int rank = -1;
    int world_size = 0;
    int p = 0, q = 0;
    bool grid_layout_col_major = true;
    int local_device_id = 0;

    // Loaded-library version, MAJOR*1000 + MINOR*100 + PATCH (e.g. 720,
    // 900).  Read once at create_context via cusolverMpGetVersion; used for
    // the startup banner and the NCCL>=2.27 sanity warning.
    int cusolvermp_version = 0;

    // owned resources
    ncclComm_t          nccl_comm  = nullptr;
    cudaStream_t        stream     = nullptr;
    cusolverMpHandle_t  handle     = nullptr;
    cusolverMpGrid_t    grid       = nullptr;

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

    // cuBLASMp handle + grid — shares the NCCL comm and CUDA stream with
    // the cuSOLVERMp side so the two are interoperable in fused FFIs.
    // Lazily populated by ensure_cublasmp(): the first fused-kernel
    // call builds the handle/grid, subsequent calls reuse them.
    cublasMpHandle_t  cublasmp_handle = nullptr;
    cublasMpGrid_t    cublasmp_grid   = nullptr;
};

// Grow (d_workspace, h_workspace) if needed; keeps largest allocation.
void ensure_workspace(LorraxCusolverMpCtx* ctx, size_t d_need, size_t h_need);

// Lazy-initialise the cuBLASMp handle + grid, sharing the existing NCCL
// comm and CUDA stream with cuSOLVERMp.  No-op on repeat calls.  Callers
// that don't need cuBLASMp never pay the init cost.  Throws on failure.
void ensure_cublasmp(LorraxCusolverMpCtx* ctx);

}  // namespace lorrax_ffi::cusolvermp
