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
#include <cublasmp.h>

#include "../../common/cpp/ffi_helpers.h"
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
static void throw_if_cublasmp(cublasMpStatus_t st, const char* what) {
    if (st == CUBLASMP_STATUS_SUCCESS) return;
    std::ostringstream os;
    os << what << ": cublasmp status=" << static_cast<int>(st);
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

    // -- Version detection: cusolverMpCreateDeviceGrid's third argument
    //    changed from cal_comm_t (≤0.6.x) to ncclComm_t (≥0.7.0).  Both
    //    are opaque pointers so the C compiler doesn't catch the mismatch;
    //    runtime symptom on 0.7+ with a CAL pointer is rank-asymmetric
    //    "cusolverMpGetrs status=7" with all per-rank info=0.  See
    //    https://docs.nvidia.com/cuda/cusolvermp/usage/initialization/cal_to_nccl.html
    //
    //    The same FFI binary supports BOTH paths: detect the loaded
    //    library version via cusolverMpGetVersion (an int returned as
    //    MAJOR*1000 + MINOR*100 + PATCH, e.g. 600/700/720/800), choose
    //    the correct comm type, and warn on known-broken combinations.
    int mp_version = 0;
    throw_if_cusolver(cusolverMpGetVersion(ctx->handle, &mp_version),
                      "cusolverMpGetVersion");
    ctx->cusolvermp_version = mp_version;
    const int mp_major = mp_version / 1000;
    const int mp_minor = (mp_version / 100) % 10;
    const int mp_patch = mp_version % 100;
    const bool use_nccl_comm_path = (mp_version >= 700);

    // NCCL version (returned as e.g. 22603 = 2.26.3, encoded
    // major*10000 + minor*100 + patch).  Used to flag the known
    // ncclCommWindowRegister symbol gap that cusolverMp 0.8+ requires.
    int nccl_version = 0;
    (void)ncclGetVersion(&nccl_version);
    const int nccl_major = nccl_version / 10000;
    const int nccl_minor = (nccl_version / 100) % 100;
    const int nccl_patch = nccl_version % 100;

    // Rank-0 startup banner so misconfigurations are visible without
    // having to grep through XLA traces.
    if (rank == 0) {
        std::fprintf(stderr,
            "[lorrax cusolverMp] library %d.%d.%d, NCCL %d.%d.%d, "
            "comm path: %s, grid: %dx%d (%s)\n",
            mp_major, mp_minor, mp_patch,
            nccl_major, nccl_minor, nccl_patch,
            use_nccl_comm_path ? "NCCL" : "CAL",
            p, q, grid_layout_col_major ? "col-major" : "row-major");
    }

    // Actionable warnings for known-broken combinations:
    //   * 0.6.x + Px>1 AND Py>1: getrf/getrs returns silent wrong answers.
    //     Reproduced 2026-05-09 across (N, NRHS) on a 2x2 mesh; the
    //     cuSolverMp 0.7.0 release notes' "non-square grid" fix is in
    //     fact the CAL→NCCL ABI shift below and applies to ALL 2D grids
    //     (square or not) once the comm type is correct.  1×N and N×1
    //     are unaffected because the broken degenerate-block-cyclic path
    //     isn't triggered there.
    //   * 0.8.0 + NCCL < 2.27: dlopen fails with "undefined symbol:
    //     ncclCommWindowRegister".  We can't reach this point if the
    //     symbol is missing (load fails before this constructor runs),
    //     but a soft warning at NCCL < 2.27 is still useful for any
    //     future cusolverMp ≥ 0.8 binary.
    if (rank == 0) {
        if (mp_version < 700 && p > 1 && q > 1) {
            std::fprintf(stderr,
                "[lorrax cusolverMp] WARNING: cuSolverMp %d.%d.%d on a 2D "
                "process grid (%dx%d) has a known correctness bug in "
                "getrf/getrs that returns wrong answers silently.  "
                "Use a 1xN or Nx1 mesh, or upgrade to ≥0.7.0 (where the "
                "comm-handle ABI shifts from CAL to NCCL).\n",
                mp_major, mp_minor, mp_patch, p, q);
        }
        if (mp_version >= 800 && nccl_version < 22700) {
            std::fprintf(stderr,
                "[lorrax cusolverMp] WARNING: cuSolverMp %d.%d.%d expects "
                "NCCL ≥ 2.27 (ncclCommWindowRegister); loaded NCCL %d.%d.%d "
                "is too old.  Stage NCCL ≥ 2.27 ahead of the container's "
                "bundled libnccl.so.2 to use this version.\n",
                mp_major, mp_minor, mp_patch,
                nccl_major, nccl_minor, nccl_patch);
        }
    }

    cusolverMpGridMapping_t layout = grid_layout_col_major
        ? CUSOLVERMP_GRID_MAPPING_COL_MAJOR
        : CUSOLVERMP_GRID_MAPPING_ROW_MAJOR;

    if (!use_nccl_comm_path) {
        // 0.6.x: cusolverMpCreateDeviceGrid takes cal_comm_t.  Build a
        // real cal_comm via cal_comm_create, plumbing the three required
        // allgather/req_test/req_free callbacks to NCCL.
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
                                       reinterpret_cast<cal_comm_t>(ctx->cal_comm),
                                       /*numRowDevices=*/p,
                                       /*numColDevices=*/q, layout),
            "cusolverMpCreateDeviceGrid");
    } else {
        // 0.7+: cusolverMpCreateDeviceGrid takes ncclComm_t directly.
        // The cast is needed because the system cusolverMp.h on this
        // box may declare the parameter as cal_comm_t (depending on
        // which SDK we're built against), but the LOADED library
        // dereferences it as ncclComm_t — we pass the right object
        // and let the linker do its thing.  cal_comm_t and ncclComm_t
        // are both opaque pointers at the ABI level.
        throw_if_cusolver(
            cusolverMpCreateDeviceGrid(
                ctx->handle, &ctx->grid,
                reinterpret_cast<cal_comm_t>(ctx->nccl_comm),
                /*numRowDevices=*/p,
                /*numColDevices=*/q, layout),
            "cusolverMpCreateDeviceGrid");
    }

    // Device buffer for d_info (int) used by every Syevd call.
    throw_if_cuda(cudaMalloc(&ctx->d_info, sizeof(int)), "cudaMalloc(d_info)");
    throw_if_cuda(cudaMemset(ctx->d_info, 0, sizeof(int)), "cudaMemset(d_info)");

    // Pooled cross-stream events.  Created once, reused for every FFI call.
    throw_if_cuda(cudaEventCreateWithFlags(&ctx->ev_xla_in,
                                           cudaEventDisableTiming),
                  "cudaEventCreate(ev_xla_in)");
    throw_if_cuda(cudaEventCreateWithFlags(&ctx->ev_ctx_out,
                                           cudaEventDisableTiming),
                  "cudaEventCreate(ev_ctx_out)");

    // Host workspace starts empty; grown on demand in the FFI.
    // Device workspace is now allocated per-call via ffi::ScratchAllocator.
    ctx->h_workspace = nullptr;
    ctx->h_workspace_bytes = 0;

    return reinterpret_cast<int64_t>(ctx);
}

void destroy_context(int64_t ctx_handle) {
    if (ctx_handle == 0) return;
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);

    // cuBLASMp first (it shares the CAL comm with cuSOLVERMp — tear down
    // the client handles before we destroy the comm underneath them).
    if (ctx->cublasmp_grid)   { cublasMpGridDestroy(ctx->cublasmp_grid);     ctx->cublasmp_grid = nullptr; }
    if (ctx->cublasmp_handle) { cublasMpDestroy(ctx->cublasmp_handle);       ctx->cublasmp_handle = nullptr; }
    if (ctx->grid)     { cusolverMpDestroyGrid(ctx->grid);       ctx->grid = nullptr; }
    if (ctx->handle)   { cusolverMpDestroy(ctx->handle);         ctx->handle = nullptr; }
    if (ctx->cal_comm) { cal_comm_destroy(ctx->cal_comm);        ctx->cal_comm = nullptr; }
    if (ctx->shim.d_scratch) {
        cudaFree(ctx->shim.d_scratch);
        ctx->shim.d_scratch = nullptr;
        ctx->shim.d_scratch_bytes = 0;
    }
    if (ctx->ev_xla_in)  { cudaEventDestroy(ctx->ev_xla_in);     ctx->ev_xla_in = nullptr; }
    if (ctx->ev_ctx_out) { cudaEventDestroy(ctx->ev_ctx_out);    ctx->ev_ctx_out = nullptr; }
    if (ctx->stream)   { cudaStreamDestroy(ctx->stream);         ctx->stream = nullptr; }
    if (ctx->nccl_comm){ ncclCommDestroy(ctx->nccl_comm);        ctx->nccl_comm = nullptr; }
    if (ctx->d_info)      { cudaFree(ctx->d_info);                ctx->d_info = nullptr; }
    if (ctx->h_workspace) { free(ctx->h_workspace);              ctx->h_workspace = nullptr; }

    delete ctx;
}

// Lazy init for cuBLASMp.  Reuses the cuSOLVERMp ctx's stream and comm.
// cublasMpGridCreate's communicator-parameter ABI shifted from cal_comm_t
// to ncclComm_t at the same time cuSolverMp's did (0.7.0).  We follow
// the same dispatch as the cuSolverMp grid: CAL for ≤0.6.x, NCCL for
// ≥0.7.0.  See create_context() above for the full rationale.
void ensure_cublasmp(LorraxCusolverMpCtx* ctx) {
    if (ctx->cublasmp_handle && ctx->cublasmp_grid) return;
    const bool use_nccl_comm_path = (ctx->cusolvermp_version >= 700);
    if (!use_nccl_comm_path && !ctx->cal_comm) {
        throw std::runtime_error(
            "ensure_cublasmp: cuSOLVERMp ctx has no CAL comm — "
            "cuBLASMp must be initialised after cuSOLVERMp.");
    }
    if (use_nccl_comm_path && !ctx->nccl_comm) {
        throw std::runtime_error(
            "ensure_cublasmp: cuSOLVERMp ctx has no NCCL comm.");
    }
    if (!ctx->cublasmp_handle) {
        throw_if_cublasmp(
            cublasMpCreate(&ctx->cublasmp_handle, ctx->stream),
            "cublasMpCreate");
    }
    if (!ctx->cublasmp_grid) {
        cublasMpGridLayout_t layout = ctx->grid_layout_col_major
            ? CUBLASMP_GRID_LAYOUT_COL_MAJOR
            : CUBLASMP_GRID_LAYOUT_ROW_MAJOR;
        // cal_comm_t / ncclComm_t are both opaque pointers; cast to the
        // declared header type and pass the correct concrete object.
        cal_comm_t comm_arg = use_nccl_comm_path
            ? reinterpret_cast<cal_comm_t>(ctx->nccl_comm)
            : ctx->cal_comm;
        throw_if_cublasmp(
            cublasMpGridCreate(ctx->p, ctx->q, layout,
                                comm_arg, &ctx->cublasmp_grid),
            "cublasMpGridCreate");
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
