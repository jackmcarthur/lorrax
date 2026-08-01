// ctx.h — shared Ctx struct used by context.cc and eigh_ffi.cc.
#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <nccl.h>
// LORRAX_FFI_HAVE_CAL defaults to 1 (unset -> Perlmutter/Cray behaviour).
// The Frontera eigh-only build compiles with -DLORRAX_FFI_HAVE_CAL=0
// because cuSOLVERMp >= 0.7 (the pip wheels) is NCCL-native and ships no
// cal.h / libcal.
#ifndef LORRAX_FFI_HAVE_CAL
#define LORRAX_FFI_HAVE_CAL 1
#endif
#if LORRAX_FFI_HAVE_CAL
#include <cal.h>
#endif
#include <cusolverMp.h>
#include <cublasmp.h>

namespace lorrax_ffi::cusolvermp {

#if LORRAX_FFI_HAVE_CAL
// "Shim" struct passed through CAL's user-data slot to our three allgather
// callbacks.  Lives for the lifetime of the LorraxCusolverMpCtx.
struct CalNcclShim {
    ncclComm_t    nccl_comm  = nullptr;
    cudaStream_t  stream     = nullptr;
    // Scratch device buffer reused across allgathers; grown on demand.
    void*         d_scratch       = nullptr;
    size_t        d_scratch_bytes = 0;
};
#endif

struct LorraxCusolverMpCtx {
    // identity
    int rank = -1;
    int world_size = 0;
    int p = 0, q = 0;
    bool grid_layout_col_major = true;
    int local_device_id = 0;

    // Loaded-library version, MAJOR*1000 + MINOR*100 + PATCH (e.g. 600,
    // 720, 800).  Read once at create_context via cusolverMpGetVersion.
    // Used to dispatch CAL vs NCCL comm path in cusolverMpCreateDeviceGrid
    // and cublasMpGridCreate (the parameter type changed at 0.7.0).
    int cusolvermp_version = 0;

    // owned resources
    ncclComm_t          nccl_comm  = nullptr;
    cudaStream_t        stream     = nullptr;
    cusolverMpHandle_t  handle     = nullptr;
#if LORRAX_FFI_HAVE_CAL
    cal_comm_t          cal_comm   = nullptr;   // our CAL wrapper over NCCL
#endif
    cusolverMpGrid_t    grid       = nullptr;
#if LORRAX_FFI_HAVE_CAL
    CalNcclShim         shim{};                 // stable address — CAL holds a ptr
#endif

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

    // cuBLASMp handle + grid — shares the CAL comm and CUDA stream with
    // the cuSOLVERMp side so the two are interoperable in fused FFIs.
    // Lazily populated by ensure_cublasmp(): the first fused-kernel
    // call builds the handle/grid, subsequent calls reuse them.
    cublasMpHandle_t  cublasmp_handle = nullptr;
    cublasMpGrid_t    cublasmp_grid   = nullptr;
};

// Grow (d_workspace, h_workspace) if needed; keeps largest allocation.
void ensure_workspace(LorraxCusolverMpCtx* ctx, size_t d_need, size_t h_need);

// Lazy-initialise the cuBLASMp handle + grid, sharing the existing CAL
// comm and CUDA stream with cuSOLVERMp.  No-op on repeat calls.  Callers
// that don't need cuBLASMp never pay the init cost.  Throws on failure.
void ensure_cublasmp(LorraxCusolverMpCtx* ctx);

}  // namespace lorrax_ffi::cusolvermp
