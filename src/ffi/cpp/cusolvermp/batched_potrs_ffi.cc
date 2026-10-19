// batched_potrs_ffi.cc — per-q cuSOLVERMp potrs on the world-wide
// (Px, Py) grid.  See batched_potrf_ffi.cc for the rationale for the
// plain C++ for-loop + reused descriptors pattern.
//
// Sharding contract:
//   L : (Nq, N, N)        P(None, 'y', 'x')  — batched_cholesky output
//                         (inner dims pre-transposed, col-major bytes).
//   B : (Nq, N, Mrhs)     P(None, 'x', 'y')  — Python transposes inner
//                         dims → (Nq, Mrhs/Py, N/Px) row-major ≡
//                         (N/Px × Mrhs/Py) col-major per slice.
//   X : same shape/layout as B; overwritten in place by potrs.
//
// descA : (N, N) with mb=N/Px, nb=N/Py, lld=N/Px.
// descB : (N, Mrhs) with mb_B=N/Px (= A's local row block — inner-dim
//         alignment for the forward solve), nb_B=Mrhs/Py (one col-tile
//         per rank → matches JAX's contiguous P(None, 'x', 'y') slab),
//         lld=N/Px.

#include "../common/ctx_registry.h"
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#include <cuda_runtime.h>
#include <cusolverMp.h>
#include <nvtx3/nvToolsExt.h>

#include "xla/ffi/api/ffi.h"

#include "../common/ffi_helpers.h"
#include "cusolvermp_interface.h"
#include "ctx.h"
#include "mp_plan_cache.h"
#include "info.h"

namespace lorrax_ffi::cusolvermp_batched_potrs {

namespace ffi = ::xla::ffi;
using lorrax_ffi::cusolvermp::LorraxCusolverMpCtx;
using lorrax_ffi::cusolvermp::ensure_workspace;
namespace mp = lorrax_ffi::cusolvermp::mp;
using lorrax_ffi::cusolvermp::MpPlan;
using lorrax_ffi::cusolvermp::MpPlanKey;
using lorrax_ffi::cusolvermp::mp_plan;
using lorrax_ffi::cusolvermp::kMpPotrs;

struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

static ffi::Error cross_stream_wait_pooled(cudaStream_t waiter,
                                           cudaStream_t signaller,
                                           cudaEvent_t  ev) {
    NvtxRange range("potrs.cross_stream_wait");
    LORRAX_GPU_CHECK(cudaEventRecord(ev, signaller));
    LORRAX_GPU_CHECK(cudaStreamWaitEvent(waiter, ev, 0));
    return ffi::Error::Success();
}

template <typename T>
static ffi::Error BatchedPotrsImpl(
    int64_t nq, int64_t n, int64_t mrhs,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    cudaStream_t xla_stream,
    LorraxCusolverMpCtx* ctx,
    ffi::ScratchAllocator& scratch,
    const T* d_L, const T* d_B_in, T* d_X_out)
{
    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        ctx->stream, xla_stream, ctx->ev_xla_in));

    const int Px = ctx->p;
    const int Py = ctx->q;
    const int64_t lld_A = (n + Px - 1) / Px;
    const int64_t lld_B = lld_A;             // B rows distributed same way as A
    const int64_t A_local_cols = (n    + Py - 1) / Py;
    const int64_t B_local_cols = (mrhs + Py - 1) / Py;
    const int64_t A_slice = lld_A * A_local_cols;
    const int64_t B_slice = lld_B * B_local_cols;

    // Skip the memcpy when input_output_aliases={1: 0} on the
    // ffi_call gives us B === X (XLA reuses the buffer).  See
    // batched_potrf_ffi.cc for the rationale.
    if (d_X_out != static_cast<const T*>(d_B_in)) {
        NvtxRange range("potrs.copy_B_to_X");
        LORRAX_GPU_CHECK(cudaMemcpyAsync(
            d_X_out, d_B_in,
            nq * B_slice * sizeof(T),
            cudaMemcpyDeviceToDevice, ctx->stream));
    }

    // The descriptors and the Potrs workspace sizes, once per shape (mp_plan_cache.h).
    const MpPlan* plan = nullptr;
    const int plan_rc = mp_plan(MpPlanKey{kMpPotrs, reinterpret_cast<int64_t>(ctx),
            int64_t(mp::CudaDataTypeOf<T>::value), n, mrhs, mb_a, nb_a, mb_b, nb_b, lld_A, lld_B},
        &plan, [&](MpPlan& p) -> int {
            NvtxRange range("potrs.plan");
            auto st = cusolverMpCreateMatrixDesc(&p.desc[0], ctx->grid, mp::CudaDataTypeOf<T>::value,
                                                 n, n, mb_a, nb_a, 0, 0, lld_A);
            if (st == CUSOLVER_STATUS_SUCCESS)
                st = cusolverMpCreateMatrixDesc(&p.desc[1], ctx->grid, mp::CudaDataTypeOf<T>::value,
                                                n, mrhs, mb_b, nb_b, 0, 0, lld_B);
            if (st == CUSOLVER_STATUS_SUCCESS)
                st = mp::PotrsBufferSize<T>(ctx->handle, CUBLAS_FILL_MODE_LOWER, n, mrhs,
                                            d_L, 1, 1, p.desc[0], d_X_out, 1, 1, p.desc[1],
                                            &p.d_ws, &p.h_ws);
            return int(st);
        });
    if (plan_rc) {
        std::ostringstream os;
        os << "cusolverMp potrs plan (descriptors, Potrs_bufferSize) failed: status=" << plan_rc;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    cusolverMpMatrixDescriptor_t descA = plan->desc[0], descB = plan->desc[1];
    size_t d_ws = plan->d_ws, h_ws = plan->h_ws;
    cusolverStatus_t mp_st = CUSOLVER_STATUS_SUCCESS;
    // Device workspace from XLA's scratch allocator (inside the pool, released
    // when the call's stream work is done -- the exit cross-stream wait orders
    // it), as eigh_ffi.cc does.  The context's grow-only cudaMalloc workspace
    // used here before stayed resident outside the pool after the call:
    // 23.5 GB/rank after a one-tile 51200 c128 potrf on 2x2, which every later
    // op in the process then ran out of memory against.  Host side unchanged.
    try {
        NvtxRange range("potrs.ensure_workspace");
        ensure_workspace(ctx, 0, h_ws);
    } catch (const std::exception& ex) {
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, ex.what());
    }
    void* d_work = nullptr;
    if (d_ws > 0) {
        auto ws_opt = scratch.Allocate(d_ws);
        if (!ws_opt.has_value()) {
            std::ostringstream os;
            os << "cuSOLVERMp potrs: XLA scratch allocator failed to provide " << d_ws
               << " bytes of workspace (n=" << n << ")";
            return ffi::Error(ffi::ErrorCode::kResourceExhausted, os.str());
        }
        d_work = *ws_opt;
    }

    for (int64_t q = 0; q < nq; ++q) {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "potrs.Potrs[q=%lld]",
                      static_cast<long long>(q));
        NvtxRange range(tag);
        T* A_slice_ptr = const_cast<T*>(d_L) + q * A_slice;
        T* X_slice_ptr = d_X_out + q * B_slice;
        if (auto err = reset_info(ctx); !err.success()) {
            return err;
        }
        mp_st = mp::Potrs<T>(
            ctx->handle, CUBLAS_FILL_MODE_LOWER, n, mrhs,
            A_slice_ptr, 1, 1, descA,
            X_slice_ptr, 1, 1, descB,
            d_work, d_ws,
            ctx->h_workspace, ctx->h_workspace_bytes,
            ctx->d_info);
        if (mp_st != CUSOLVER_STATUS_SUCCESS) {
            std::ostringstream os;
            os << "cusolverMpPotrs (q=" << q << ") failed: status=" << (int)mp_st;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
        if (auto err = read_info(ctx, "cusolverMpPotrs", "the solve failed", n, mb_a, q);
                !err.success()) {
            return err;
        }
    }

    {
        NvtxRange range("potrs.DestroyMatrixDesc[A]");
    }
    {
        NvtxRange range("potrs.DestroyMatrixDesc[B]");
    }

    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        xla_stream, ctx->stream, ctx->ev_ctx_out));
    return ffi::Error::Success();
}

static ffi::Error BatchedPotrsDispatch(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::AnyBuffer L,
    ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> X_out,
    int64_t nq, int64_t n, int64_t mrhs,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    int64_t ctx_key)
{
    const int64_t ctx_handle = ::lorrax_ffi::ctx_registry::resolve(ctx_key, "cusolvermp");
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "batched_potrs: ctx_handle is null");
    }
    const auto dtype = L.element_type();
    if (B.element_type() != dtype || X_out->element_type() != dtype) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "batched_potrs: L, B, X must share dtype");
    }
    switch (dtype) {
        case ffi::DataType::F64:
            return BatchedPotrsImpl<double>(
                nq, n, mrhs, mb_a, nb_a, mb_b, nb_b, stream, ctx, scratch,
                static_cast<const double*>(L.untyped_data()),
                static_cast<const double*>(B.untyped_data()),
                static_cast<double*>(X_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return BatchedPotrsImpl<C128>(
                nq, n, mrhs, mb_a, nb_a, mb_b, nb_b, stream, ctx, scratch,
                static_cast<const C128*>(L.untyped_data()),
                static_cast<const C128*>(B.untyped_data()),
                static_cast<C128*>(X_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "batched_potrs: unsupported dtype " << (int)dtype
               << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

}  // namespace lorrax_ffi::cusolvermp_batched_potrs

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CusolverMpBatchedPotrsFfi,
    lorrax_ffi::cusolvermp_batched_potrs::BatchedPotrsDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()      // L
        .Arg<xla::ffi::AnyBuffer>()      // B
        .Ret<xla::ffi::AnyBuffer>()      // X
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("mrhs")
        .Attr<int64_t>("mb_a")
        .Attr<int64_t>("nb_a")
        .Attr<int64_t>("mb_b")
        .Attr<int64_t>("nb_b")
        .Attr<int64_t>("ctx_key"));
