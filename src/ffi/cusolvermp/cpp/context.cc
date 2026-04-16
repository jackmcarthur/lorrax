// context.cc — NCCL + cusolverMp context construction/teardown.
//
// The `LorraxCusolverMpCtx` struct is ONLY lived inside the .so.  Python
// sees it as an opaque int64 handle (reinterpret_cast<uintptr_t>).

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <nccl.h>
#include <cal.h>
#include <cusolverMp.h>

#include "../../common/cpp/xla_ffi_glue.h"
#include "ctx.h"

namespace lorrax_ffi::cusolvermp {

// ---------------------------------------------------------------------------
// CAL → NCCL bridge
// ---------------------------------------------------------------------------
// CAL's three user callbacks implement an allgather over host buffers.  We
// route that through ncclAllGather by staging through a per-shim device
// scratch buffer.
//
// Signature: cal_comm_create_params_t::allgather
//   calError_t allgather(void* src, void* recv, size_t size,
//                        void* data, void** request);
// where `size` is bytes *per rank* (recv_buf has nranks * size bytes).
// ---------------------------------------------------------------------------

static calError_t cal_nccl_allgather(void* src, void* recv, size_t size,
                                     void* data, void** request) {
    auto* shim = static_cast<CalNcclShim*>(data);

    int world = 0;
    if (ncclCommCount(shim->nccl_comm, &world) != ncclSuccess) return CAL_ERROR;
    const size_t total_bytes = size * static_cast<size_t>(world);

    // (Re)allocate device scratch large enough for [send | recv].
    const size_t need = size + total_bytes;
    if (need > shim->d_scratch_bytes) {
        if (shim->d_scratch) cudaFree(shim->d_scratch);
        if (cudaMalloc(&shim->d_scratch, need) != cudaSuccess) return CAL_ERROR_CUDA;
        shim->d_scratch_bytes = need;
    }
    void* d_send = shim->d_scratch;
    void* d_recv = static_cast<char*>(d_send) + size;

    // H2D our rank's payload.
    if (cudaMemcpyAsync(d_send, src, size, cudaMemcpyHostToDevice, shim->stream)
        != cudaSuccess) return CAL_ERROR_CUDA;

    // Communicate as raw bytes (ncclChar = 1 byte).
    if (ncclAllGather(d_send, d_recv, size, ncclChar,
                      shim->nccl_comm, shim->stream) != ncclSuccess) {
        return CAL_ERROR;
    }

    // D2H into CAL's recv buffer.
    if (cudaMemcpyAsync(recv, d_recv, total_bytes, cudaMemcpyDeviceToHost,
                        shim->stream) != cudaSuccess) return CAL_ERROR_CUDA;

    // Block until done — CAL expects the data to be ready when req_test returns
    // CAL_OK.  We signal completion by stamping the request pointer with a
    // "done" sentinel; the cheapest synchronous completion is to sync the
    // stream here.
    if (cudaStreamSynchronize(shim->stream) != cudaSuccess) return CAL_ERROR_CUDA;

    if (request) *request = shim;  // non-null sentinel so req_test/free see a handle
    return CAL_OK;
}

static calError_t cal_nccl_req_test(void* request) {
    // Our allgather already synced the stream, so every submitted request
    // is immediately complete.
    (void)request;
    return CAL_OK;
}

static calError_t cal_nccl_req_free(void* request) {
    (void)request;  // shim is owned by the Ctx; nothing to free per-request
    return CAL_OK;
}

// ----- tiny throwing checks (for setup path; FFI handlers use Error) ------
static void throw_if_cuda(cudaError_t st, const char* what) {
    if (st == cudaSuccess) return;
    std::ostringstream os;
    os << what << ": " << cudaGetErrorName(st) << " (" << cudaGetErrorString(st) << ")";
    throw std::runtime_error(os.str());
}
static void throw_if_nccl(ncclResult_t st, const char* what) {
    if (st == ncclSuccess) return;
    std::ostringstream os;
    os << what << ": nccl=" << ncclGetErrorString(st);
    throw std::runtime_error(os.str());
}
static void throw_if_cusolver(cusolverStatus_t st, const char* what) {
    if (st == CUSOLVER_STATUS_SUCCESS) return;
    std::ostringstream os;
    os << what << ": cusolver status=" << static_cast<int>(st);
    throw std::runtime_error(os.str());
}

// ----- context lifecycle --------------------------------------------------
int64_t create_context(int rank, int world_size,
                       uintptr_t nccl_unique_id_addr,
                       int nccl_unique_id_nbytes,
                       int p, int q,
                       bool grid_layout_col_major)
{
    if (p * q != world_size) {
        std::ostringstream os;
        os << "create_cusolvermp_context: p*q (=" << p*q
           << ") must equal world_size (=" << world_size << ")";
        throw std::runtime_error(os.str());
    }
    if (nccl_unique_id_nbytes != static_cast<int>(sizeof(ncclUniqueId))) {
        std::ostringstream os;
        os << "create_cusolvermp_context: nccl_unique_id_nbytes ("
           << nccl_unique_id_nbytes << ") != sizeof(ncclUniqueId) ("
           << sizeof(ncclUniqueId) << ")";
        throw std::runtime_error(os.str());
    }

    auto* ctx = new LorraxCusolverMpCtx{};
    ctx->rank       = rank;
    ctx->world_size = world_size;
    ctx->p          = p;
    ctx->q          = q;
    ctx->grid_layout_col_major = grid_layout_col_major;

    // Ensure this thread's current CUDA device matches the caller's local
    // rank (NERSC: srun binds 1 GPU per rank; cudaGetDevice returns 0 in
    // the narrow view, but that *is* the correct local GPU for this proc).
    //
    // Importantly, the subsequent stream create MUST happen while that
    // device is current — a known cusolverMp pitfall (see NVIDIA devtalk
    // thread 313054).
    int current_device = 0;
    throw_if_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
    throw_if_cuda(cudaSetDevice(current_device), "cudaSetDevice");
    // Ensure CUDA primary context is live on this device.
    throw_if_cuda(cudaFree(nullptr), "cudaFree(nullptr)");
    ctx->local_device_id = current_device;

    // ncclCommInitRank: collective among all `world_size` processes.  The
    // unique id was generated by rank 0 and broadcast by the caller.
    ncclUniqueId uid;
    std::memcpy(&uid, reinterpret_cast<void*>(nccl_unique_id_addr),
                sizeof(ncclUniqueId));
    throw_if_nccl(ncclCommInitRank(&ctx->nccl_comm, world_size, uid, rank),
                  "ncclCommInitRank");

    // Private CUDA stream for cusolverMp operations.  XLA still passes us
    // its own per-call stream; we serialise between the two in the FFI
    // handler.
    throw_if_cuda(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking),
                  "cudaStreamCreate");

    // cusolverMp handle — tied to (localDeviceId, stream).
    throw_if_cusolver(
        cusolverMpCreate(&ctx->handle, ctx->local_device_id, ctx->stream),
        "cusolverMpCreate");

    // Single process grid used for all matrices this context will solve
    // (p × q).  Larger matrices can reuse the same grid across many
    // FFI calls.
    cusolverMpGridMapping_t layout = grid_layout_col_major
        ? CUSOLVERMP_GRID_MAPPING_COL_MAJOR
        : CUSOLVERMP_GRID_MAPPING_ROW_MAJOR;
    // Build a real cal_comm via cal_comm_create, plumbing the three
    // required allgather/req_test/req_free callbacks to NCCL.  This is the
    // canonical non-MPI path — far cleaner than the reinterpret_cast trick
    // that the NVIDIA mp_syevd.c sample uses (that trick relies on C's
    // lax pointer conversion plus an MPI-initialised libcal, neither of
    // which applies in a JAX-only Python process).
    ctx->shim.nccl_comm = ctx->nccl_comm;
    ctx->shim.stream    = ctx->stream;
    cal_comm_create_params_t cp{};
    cp.allgather    = &cal_nccl_allgather;
    cp.req_test     = &cal_nccl_req_test;
    cp.req_free     = &cal_nccl_req_free;
    cp.data         = &ctx->shim;
    cp.nranks       = world_size;
    cp.rank         = rank;
    cp.local_device = ctx->local_device_id;
    calError_t cal_st = cal_comm_create(cp, &ctx->cal_comm);
    if (cal_st != CAL_OK) {
        std::ostringstream os;
        os << "cal_comm_create failed: calError=" << (int)cal_st;
        throw std::runtime_error(os.str());
    }

    throw_if_cusolver(
        cusolverMpCreateDeviceGrid(ctx->handle, &ctx->grid,
                                   ctx->cal_comm,
                                   /*numRowDevices=*/p,
                                   /*numColDevices=*/q, layout),
        "cusolverMpCreateDeviceGrid");

    // Device buffer for d_info (int) used by every Syevd call.
    throw_if_cuda(cudaMalloc(&ctx->d_info, sizeof(int)), "cudaMalloc(d_info)");
    throw_if_cuda(cudaMemset(ctx->d_info, 0, sizeof(int)), "cudaMemset(d_info)");

    // Workspace scratchpads start empty; grown on demand in the FFI.
    ctx->d_workspace = nullptr;
    ctx->d_workspace_bytes = 0;
    ctx->h_workspace = nullptr;
    ctx->h_workspace_bytes = 0;

    return reinterpret_cast<int64_t>(ctx);
}

void destroy_context(int64_t ctx_handle) {
    if (ctx_handle == 0) return;
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);

    if (ctx->grid)     { cusolverMpDestroyGrid(ctx->grid);       ctx->grid = nullptr; }
    if (ctx->handle)   { cusolverMpDestroy(ctx->handle);         ctx->handle = nullptr; }
    if (ctx->cal_comm) { cal_comm_destroy(ctx->cal_comm);        ctx->cal_comm = nullptr; }
    if (ctx->shim.d_scratch) {
        cudaFree(ctx->shim.d_scratch);
        ctx->shim.d_scratch = nullptr;
        ctx->shim.d_scratch_bytes = 0;
    }
    if (ctx->stream)   { cudaStreamDestroy(ctx->stream);         ctx->stream = nullptr; }
    if (ctx->nccl_comm){ ncclCommDestroy(ctx->nccl_comm);        ctx->nccl_comm = nullptr; }
    if (ctx->d_info)  { cudaFree(ctx->d_info);                   ctx->d_info = nullptr; }
    if (ctx->d_workspace) { cudaFree(ctx->d_workspace);          ctx->d_workspace = nullptr; }
    if (ctx->h_workspace) { free(ctx->h_workspace);              ctx->h_workspace = nullptr; }

    delete ctx;
}

// Grow workspace on demand; preserves the larger allocation on future calls.
void ensure_workspace(LorraxCusolverMpCtx* ctx, size_t d_need, size_t h_need) {
    if (d_need > ctx->d_workspace_bytes) {
        if (ctx->d_workspace) cudaFree(ctx->d_workspace);
        throw_if_cuda(cudaMalloc(&ctx->d_workspace, d_need),
                      "cudaMalloc(d_workspace)");
        ctx->d_workspace_bytes = d_need;
    }
    if (h_need > ctx->h_workspace_bytes) {
        if (ctx->h_workspace) free(ctx->h_workspace);
        ctx->h_workspace = malloc(h_need);
        if (!ctx->h_workspace)
            throw std::runtime_error("malloc(h_workspace) failed");
        ctx->h_workspace_bytes = h_need;
    }
}

// ----- smoke: ncclAllReduce on a device float buffer ---------------------
int smoke_allreduce_sum(int64_t ctx_handle,
                        uintptr_t device_ptr,
                        int nelems)
{
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    ncclResult_t st = ncclAllReduce(
        reinterpret_cast<const void*>(device_ptr),
        reinterpret_cast<void*>(device_ptr),
        static_cast<size_t>(nelems),
        ncclFloat32,
        ncclSum,
        ctx->nccl_comm,
        ctx->stream);
    if (st != ncclSuccess) return static_cast<int>(st);
    cudaError_t ce = cudaStreamSynchronize(ctx->stream);
    if (ce != cudaSuccess) return 1000 + static_cast<int>(ce);
    return 0;
}

}  // namespace lorrax_ffi::cusolvermp
