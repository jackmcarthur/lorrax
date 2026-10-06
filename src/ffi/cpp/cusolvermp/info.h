// info.h — the cuSOLVERMp solvers' info output, read after every call.
//
// A cusolverStatus_t reports API and launch errors only. A numerical failure
// (LAPACK convention info > 0: STEDC non-convergence in syevd, a leading
// minor that is not positive in potrf, an exactly zero U(i,i) in getrf) and an
// illegal argument (info < 0) reach the caller only through the device int
// d_info. Each call zeroes it on the context stream first and copies it to
// the host after (one int and one stream sync per call); a nonzero value
// returns as an error naming the routine, info, its meaning and the shape,
// never as a result.
#pragma once

#include <cstdint>
#include <sstream>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

#include "../common/ffi_helpers.h"
#include "ctx.h"

namespace lorrax_ffi::cusolvermp {

inline ::xla::ffi::Error reset_info(LorraxCusolverMpCtx* ctx) {
    LORRAX_CUDA_CHECK(cudaMemsetAsync(ctx->d_info, 0, sizeof(int), ctx->stream));
    return ::xla::ffi::Error::Success();
}

// ``positive`` names what info = i > 0 means for this routine; ``q`` is the
// batch member (-1 for an unbatched call).
inline ::xla::ffi::Error read_info(LorraxCusolverMpCtx* ctx, const char* routine,
                                   const char* positive, int64_t n, int64_t mb,
                                   int64_t q = -1) {
    int info = 0;
    LORRAX_CUDA_CHECK(cudaMemcpyAsync(&info, ctx->d_info, sizeof(int),
                                      cudaMemcpyDeviceToHost, ctx->stream));
    LORRAX_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    if (info == 0) return ::xla::ffi::Error::Success();
    std::ostringstream os;
    os << routine << " returned info=" << info << " (n=" << n << ", block=" << mb;
    if (q >= 0) os << ", q=" << q;
    os << ", rank " << ctx->rank << "): ";
    if (info > 0) os << positive << " at index " << info;
    else os << "argument " << -info << " is invalid";
    os << "; the result is not returned";
    return ::xla::ffi::Error(::xla::ffi::ErrorCode::kInternal, os.str());
}

}  // namespace lorrax_ffi::cusolvermp
