// eigh_ffi.cc — XLA FFI handler for cusolverMpSyevd (Hermitian / real-symm
//                eigensolver across a 2-D process grid).
//
// Structure (mirrors jaxlib/gpu/solver_kernels_ffi.cc):
//
//   EighImpl<T>       — does the actual cuSOLVERMp dance for one scalar type
//   EighDispatch      — shape/dtype validation + dtype switch
//   XLA_FFI_DEFINE_HANDLER_SYMBOL(EighMpFfi, EighDispatch, Bind()...)
//
// Patterns adopted from jaxlib:
//   - ffi::ScratchAllocator for per-call device workspace (it comes from
//     XLA's own pool, so it is inside the planners' budget)
//   - FFI_RETURN_IF_ERROR / LORRAX_*_CHECK for one-line error paths
//   - per-dtype cusolvermp::mp::{SyevdBufferSize,Syevd}<T> shims
//
// LORRAX-specific pieces that stay on the Ctx (not scratch):
//   - cusolverMpHandle_t / cal_comm_t / cusolverMpGrid_t  (one-time setup)
//   - host-side workspace buffer (cuSOLVERMp needs a host malloc; XLA's
//     scratch allocator is device-only)
//   - d_info (tiny; allocated once on ctx; zeroed before and read after
//     every solve, info.h)
//   - NCCL scratch in the CAL→NCCL shim (lives outside any FFI call)

#include "../common/ctx_registry.h"
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#include <cuda_runtime.h>
#include <cusolverMp.h>

#include "xla/ffi/api/ffi.h"

#include "../common/ffi_helpers.h"
#include "cusolvermp_interface.h"
#include "ctx.h"
#include "mp_plan_cache.h"
#include "info.h"

namespace lorrax_ffi::cusolvermp {

namespace ffi = ::xla::ffi;

// The scratch is the vendor workspace alone. The operand the solver destroys
// is a result aliased to the input (ABI 10, the third Ret): XLA owns it, as an
// intermediate the program hands over or a copy of a live parameter, so no
// private operand tile sits in the scratch any more (EIGHMEM). Both the
// execution and query entry points use this size.
static size_t eigh_scratch_bytes(size_t workspace_bytes) {
    return (workspace_bytes + 255) & ~size_t(255);
}

// ---------------------------------------------------------------------------
//  Profile switch
// ---------------------------------------------------------------------------
static inline bool profile_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("LORRAX_FFI_PROFILE");
        return e != nullptr && *e == '1';
    }();
    return v;
}

// ---------------------------------------------------------------------------
//  Cross-stream join using pooled events on the ctx.  Avoids the +750 ms
//  per-call stalls that phdf5 hit with `cudaEventDestroy` under
//  `cuda_malloc_async` (see the "cudaEventDestroy" note in
//  src/ffi/cpp/phdf5/write_ffi.cc).
// ---------------------------------------------------------------------------
static ffi::Error cross_stream_wait_pooled(cudaStream_t waiter,
                                           cudaStream_t signaller,
                                           cudaEvent_t  ev) {
    LORRAX_GPU_CHECK(cudaEventRecord(ev, signaller));
    LORRAX_GPU_CHECK(cudaStreamWaitEvent(waiter, ev, 0));
    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
//  Impl — the actual cuSOLVERMp solve for one scalar type.
// ---------------------------------------------------------------------------
template <typename T>
static ffi::Error EighImpl(
    int64_t n, int64_t mb, int64_t nb,
    bool compute_evecs,
    cudaStream_t xla_stream,
    ffi::ScratchAllocator& scratch,
    LorraxCusolverMpCtx* ctx,
    const T* d_A,
    T* d_operand,
    mp::RealOf_t<T>* d_W,
    T* d_Q)
{
    const bool prof = profile_enabled();
    cudaEvent_t ev[6];
    if (prof) {
        for (int i = 0; i < 6; ++i)
            LORRAX_GPU_CHECK(cudaEventCreate(&ev[i]));
        LORRAX_GPU_CHECK(cudaEventRecord(ev[0], ctx->stream));
    }

    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        ctx->stream, xla_stream, ctx->ev_xla_in));
    if (prof) LORRAX_GPU_CHECK(cudaEventRecord(ev[1], ctx->stream));

    const int64_t llda = (n + ctx->p - 1) / ctx->p;   // = n/p when exact
    // The descriptors and the Syevd workspace sizes, once per shape (mp_plan_cache.h): no
    // bufferSize query, and so no cuBLASMp grid or communicator, on the per-call path.
    const char jobz = compute_evecs ? 'V' : 'N';
    const MpPlan* plan = nullptr;
    const int plan_rc = mp_plan(MpPlanKey{kMpEigh, reinterpret_cast<int64_t>(ctx),
            int64_t(mp::CudaDataTypeOf<T>::value), n, mb, nb, llda, int64_t(jobz), 0, 0, 0},
        &plan, [&](MpPlan& p) -> int {
            auto st = cusolverMpCreateMatrixDesc(&p.desc[0], ctx->grid, mp::CudaDataTypeOf<T>::value,
                                                 n, n, mb, nb, 0, 0, llda);
            if (st == CUSOLVER_STATUS_SUCCESS)
                st = cusolverMpCreateMatrixDesc(&p.desc[1], ctx->grid, mp::CudaDataTypeOf<T>::value,
                                                n, n, mb, nb, 0, 0, llda);
            if (st == CUSOLVER_STATUS_SUCCESS)
                st = mp::SyevdBufferSize<T>(ctx->handle, jobz, CUBLAS_FILL_MODE_LOWER, n,
                                            d_A, 1, 1, p.desc[0], d_W, d_Q, 1, 1, p.desc[1],
                                            &p.d_ws, &p.h_ws);
            return int(st);
        });
    if (plan_rc) {
        std::ostringstream os;
        os << "cusolverMp eigh plan (descriptors, Syevd_bufferSize) failed: status=" << plan_rc;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    cusolverMpMatrixDescriptor_t descA = plan->desc[0], descQ = plan->desc[1];
    if (prof) LORRAX_GPU_CHECK(cudaEventRecord(ev[2], ctx->stream));
    const size_t d_ws_bytes = plan->d_ws, h_ws_bytes = plan->h_ws;
    cusolverStatus_t mp_st = CUSOLVER_STATUS_SUCCESS;
    if (prof) LORRAX_GPU_CHECK(cudaEventRecord(ev[3], ctx->stream));

    // Device workspace from XLA's scratch pool, i.e. inside XLA's reserved
    // pool.  cuSOLVERMp's own internals (NCCL, handles, the context's
    // grow-only workspace) allocate outside it: up to 4.72 GB/rank at P4.
    // At fraction 0.89 a 40 GB card leaves 0.11 x 42.4 = 4.66 GB outside
    // the reservation, so those bytes fit only because the driver releases
    // idle reserved memory to them on demand (docs/environment/overview.md
    // §2.1).  What never fits is XLA's LIVE bytes plus these.
    const size_t scratch_bytes = eigh_scratch_bytes(d_ws_bytes);
    auto ws_opt = scratch.Allocate(scratch_bytes);
    if (!ws_opt.has_value()) {
        std::ostringstream os;
        os << "eigh: XLA scratch allocator failed to provide "
           << scratch_bytes << " bytes";
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, os.str());
    }
    void* d_workspace = *ws_opt;
    // input_output_aliases={0: 2} makes the destroyed-operand result the input
    // buffer itself; the copy below runs only if a caller bound them apart.
    if (static_cast<const void*>(d_operand) != static_cast<const void*>(d_A)) {
        const size_t operand_bytes = size_t(n / ctx->p) * (n / ctx->q) * sizeof(T);
        LORRAX_GPU_CHECK(cudaMemcpyAsync(d_operand, d_A, operand_bytes,
                                         cudaMemcpyDeviceToDevice, ctx->stream));
    }

    // Host workspace stays on Ctx (scratch is device-only).
    if (h_ws_bytes > ctx->h_workspace_bytes) {
        if (ctx->h_workspace) std::free(ctx->h_workspace);
        ctx->h_workspace = std::malloc(h_ws_bytes);
        if (!ctx->h_workspace) {
            return ffi::Error(ffi::ErrorCode::kInternal,
                              "malloc(host_workspace) failed");
        }
        ctx->h_workspace_bytes = h_ws_bytes;
    }

    // The status reports API errors only; a STEDC failure is in d_info (info.h).
    FFI_RETURN_IF_ERROR(reset_info(ctx));

    // cuSOLVERMp overwrites its operand during tridiagonalisation: it is
    // given the destroyed-operand result, never the caller's A.
    mp_st = mp::Syevd<T>(
        ctx->handle, jobz, CUBLAS_FILL_MODE_LOWER, n,
        d_operand, 1, 1, descA,
        d_W,
        d_Q, 1, 1, descQ,
        d_workspace, d_ws_bytes,
        ctx->h_workspace, h_ws_bytes,
        ctx->d_info);
    if (prof) LORRAX_GPU_CHECK(cudaEventRecord(ev[4], ctx->stream));


    if (mp_st != CUSOLVER_STATUS_SUCCESS) {
        std::ostringstream os;
        os << "cusolverMpSyevd failed: status=" << (int)mp_st;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    FFI_RETURN_IF_ERROR(read_info(ctx, "cusolverMpSyevd",
        "the tridiagonal eigensolver did not converge", n, mb));

    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        xla_stream, ctx->stream, ctx->ev_ctx_out));
    if (prof) LORRAX_GPU_CHECK(cudaEventRecord(ev[5], ctx->stream));

    if (prof) {
        LORRAX_GPU_CHECK(cudaEventSynchronize(ev[5]));
        float ts[5];
        cudaEventElapsedTime(&ts[0], ev[0], ev[1]);
        cudaEventElapsedTime(&ts[1], ev[1], ev[2]);
        cudaEventElapsedTime(&ts[2], ev[2], ev[3]);
        cudaEventElapsedTime(&ts[3], ev[3], ev[4]);
        cudaEventElapsedTime(&ts[4], ev[4], ev[5]);
        if (ctx->rank == 0) {
            std::fprintf(stderr,
                "[lorrax.ffi.prof] eigh n=%ld mb=%ld nb=%ld  "
                "cross-wait=%.2f ms  desc=%.2f ms  "
                "bufferSize=%.2f ms  syevd=%.2f ms  tail=%.2f ms  "
                "TOTAL(events)=%.2f ms\n",
                (long)n, (long)mb, (long)nb,
                ts[0], ts[1], ts[2], ts[3], ts[4],
                ts[0] + ts[1] + ts[2] + ts[3] + ts[4]);
        }
        for (int i = 0; i < 6; ++i) cudaEventDestroy(ev[i]);
    }

    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
//  Dispatch — validates shapes, looks up Ctx, switches on dtype.
// ---------------------------------------------------------------------------
static ffi::Error EighDispatch(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> W_out,
    ffi::Result<ffi::AnyBuffer> Q_out,
    ffi::Result<ffi::AnyBuffer> A_out,
    int64_t n, int64_t mb, int64_t nb,
    int64_t ctx_key, bool compute_evecs)
{
    const int64_t ctx_handle = ::lorrax_ffi::ctx_registry::resolve(ctx_key, "cusolvermp");
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "eigh: ctx_handle is null (context not initialized "
                          "on this process?)");
    }

    const auto dtype = A.element_type();
    if (A_out->element_type() != dtype ||
        A_out->size_bytes() != A.size_bytes()) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "eigh: the destroyed-operand result must match A (aliased to it)");
    }
    if (Q_out->element_type() != dtype) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "eigh: output Q and input A must have matching element type");
    }
    if (W_out->element_type() != ffi::DataType::F64) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "eigh: eigenvalue output W must be F64 (cuSOLVERMp convention)");
    }

    // Shape checks are light — Python side enforces the hard invariants
    // (n % p == 0, square matrix, correct mesh axes).  If a caller ever
    // violates this we'll get a cuSOLVERMp status error below, which is
    // still actionable.

    switch (dtype) {
        case ffi::DataType::F64:
            return EighImpl<double>(
                n, mb, nb, compute_evecs, stream, scratch, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<double*>(A_out->untyped_data()),
                static_cast<double*>(W_out->untyped_data()),
                static_cast<double*>(Q_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return EighImpl<C128>(
                n, mb, nb, compute_evecs, stream, scratch, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<C128*>(A_out->untyped_data()),
                static_cast<double*>(W_out->untyped_data()),
                static_cast<C128*>(Q_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "eigh: unsupported dtype " << static_cast<int>(dtype)
               << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::cusolvermp

// ---------------------------------------------------------------------------
//  FFI binding.
// ---------------------------------------------------------------------------
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    EighMpFfi, lorrax_ffi::cusolvermp::EighDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()              // A
        .Ret<xla::ffi::AnyBuffer>()              // W (evals)
        .Ret<xla::ffi::AnyBuffer>()              // Q (evecs)
        .Ret<xla::ffi::AnyBuffer>()              // A, destroyed (aliased to the Arg)
        .Attr<int64_t>("n")
        .Attr<int64_t>("mb")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_key")
        .Attr<bool>("compute_evecs"));

// Query-only planning entry point. No matrix or workspace allocation and no solve.
// ctx_handle==0 selects the device-local cuSOLVER route; otherwise the live
// cuSOLVERMp grid supplies the descriptors EighImpl builds at block ``mb``
// (the block the solve runs at, distrib_la._cusolvermp.solve_layout).
#include <cusolverDn.h>
#include <algorithm>
extern "C" int lrx_eigh_workspace_bytes_block(
    int64_t ctx_handle, int64_t n, int64_t mb, int complex128,
    uint64_t* device_bytes, uint64_t* host_bytes);

// One tile per rank (mb = n/p): the descriptor of a solve at its own tile.
extern "C" int lrx_eigh_workspace_bytes(
    int64_t ctx_handle, int64_t n, int complex128,
    uint64_t* device_bytes, uint64_t* host_bytes) {
    int64_t mb = n;
    if (ctx_handle != 0) {
        auto* ctx = reinterpret_cast<lorrax_ffi::cusolvermp::LorraxCusolverMpCtx*>(ctx_handle);
        mb = ctx->p > 0 ? n / ctx->p : n;
    }
    return lrx_eigh_workspace_bytes_block(ctx_handle, n, mb, complex128,
                                          device_bytes, host_bytes);
}

extern "C" int lrx_eigh_workspace_bytes_block(
    int64_t ctx_handle, int64_t n, int64_t mb, int complex128,
    uint64_t* device_bytes, uint64_t* host_bytes) {
    if (!device_bytes || !host_bytes || n < 1 || n > INT32_MAX ||
        (complex128 != 0 && complex128 != 1)) return -1;
    *device_bytes = 0; *host_bytes = 0;
    if (ctx_handle == 0) {
        // One handle for the process, created once (a function-local static: thread-safe
        // initialization) and never destroyed: cusolverDnDestroy (cublasDestroy) synchronizes
        // the device, so a query issued after XLA programs were dispatched waited on all of
        // them (4+ min in cuCtxSynchronize at n 48640, run30k try 7).
        static const cusolverDnHandle_t handle = [] {
            cusolverDnHandle_t h = nullptr;
            return cusolverDnCreate(&h) == CUSOLVER_STATUS_SUCCESS ? h : nullptr;
        }();
        if (!handle) return int(CUSOLVER_STATUS_NOT_INITIALIZED);
        cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
        // The 64-bit generic query (size_t workspace, int64 n): the legacy 32-bit
        // Zheevd_bufferSize answered INVALID_VALUE at n 30000 (int lwork; run30k).
        cusolverDnParams_t params = nullptr;
        size_t ws_device = 0, ws_host = 0;
        const cudaDataType type = complex128 ? CUDA_C_64F : CUDA_R_64F;
        status = cusolverDnCreateParams(&params);
        if (status == CUSOLVER_STATUS_SUCCESS)
            status = cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR,
                CUBLAS_FILL_MODE_LOWER, n, type, nullptr, n, CUDA_R_64F, nullptr, type,
                &ws_device, &ws_host);
        if (params) cusolverDnDestroyParams(params);
        int lwork = 0;
        // JAX selects Jacobi for small matrices. Cover that path as well.
        if (status == CUSOLVER_STATUS_SUCCESS && n <= 32) {
            syevjInfo_t info = nullptr;
            status = cusolverDnCreateSyevjInfo(&info);
            int jacobi = 0;
            if (status == CUSOLVER_STATUS_SUCCESS) {
                if (complex128) status = cusolverDnZheevj_bufferSize(handle,
                    CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                    int(n), nullptr, int(n), nullptr, &jacobi, info);
                else status = cusolverDnDsyevj_bufferSize(handle,
                    CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                    int(n), nullptr, int(n), nullptr, &jacobi, info);
                cusolverDnDestroySyevjInfo(info);
                lwork = std::max(lwork, jacobi);
            }
        }
        if (status == CUSOLVER_STATUS_SUCCESS) {
            *device_bytes = std::max<uint64_t>(ws_device, uint64_t(lwork) * (complex128 ? 16 : 8));
            *host_bytes = ws_host;
        }
        return int(status);
    }
    using namespace lorrax_ffi::cusolvermp;
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (ctx->p != ctx->q || n % ctx->p || n % ctx->q || mb < 1 || (n / ctx->p) % mb)
        return -2;
    // The plan EighImpl uses for this shape (mp_plan_cache.h): sized once, then cached. Mp
    // sizing rejects null pointers, although it reads no matrix data: the context's device-info
    // allocation is the address token.
    const int64_t llda = n / ctx->p;
    const auto dtype = complex128 ? CUDA_C_64F : CUDA_R_64F;
    const MpPlan* plan = nullptr;
    const int rc = mp_plan(MpPlanKey{kMpEigh, reinterpret_cast<int64_t>(ctx), int64_t(dtype), n, mb,
                                     mb, llda, int64_t('V'), 0, 0, 0},
        &plan, [&](MpPlan& p) -> int {
            auto st = cusolverMpCreateMatrixDesc(&p.desc[0], ctx->grid, dtype, n, n, mb, mb, 0, 0, llda);
            if (st == CUSOLVER_STATUS_SUCCESS)
                st = cusolverMpCreateMatrixDesc(&p.desc[1], ctx->grid, dtype, n, n, mb, mb, 0, 0, llda);
            if (st != CUSOLVER_STATUS_SUCCESS) return int(st);
            if (complex128)
                return int(mp::SyevdBufferSize<std::complex<double>>(ctx->handle, 'V',
                    CUBLAS_FILL_MODE_LOWER, n,
                    reinterpret_cast<const std::complex<double>*>(ctx->d_info), 1, 1, p.desc[0],
                    reinterpret_cast<double*>(ctx->d_info),
                    reinterpret_cast<std::complex<double>*>(ctx->d_info), 1, 1, p.desc[1],
                    &p.d_ws, &p.h_ws));
            return int(mp::SyevdBufferSize<double>(ctx->handle, 'V', CUBLAS_FILL_MODE_LOWER, n,
                reinterpret_cast<const double*>(ctx->d_info), 1, 1, p.desc[0],
                reinterpret_cast<double*>(ctx->d_info), reinterpret_cast<double*>(ctx->d_info),
                1, 1, p.desc[1], &p.d_ws, &p.h_ws));
        });
    if (rc) return rc;
    *device_bytes = eigh_scratch_bytes(plan->d_ws);
    *host_bytes = plan->h_ws;
    return 0;
}
