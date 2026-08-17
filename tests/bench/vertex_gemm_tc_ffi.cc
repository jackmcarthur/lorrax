// Benchmark-only CUDA FFI probe for row-major complex128 strided-batch GEMM.
//
// This translation unit is deliberately not part of liblorrax_ffi.so and is
// not registered by LORRAX's production FFI loader.  It exists solely to
// compare XLA's CUTLASS complex-dot lowering against the public cuBLAS
// cublasZgemmStridedBatched entry at the vertex-contraction shapes.

#include <complex>
#include <cstdint>
#include <sstream>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include "xla/ffi/api/ffi.h"

namespace lorrax_vertex_gemm_probe {

namespace ffi = ::xla::ffi;
using C128 = std::complex<double>;

static ffi::Error fail(const char* where, cublasStatus_t status) {
  std::ostringstream os;
  os << "vertex_gemm_tc_probe: " << where
     << " failed with cublasStatus=" << static_cast<int>(status);
  return ffi::Error(ffi::ErrorCode::kInternal, os.str());
}

static ffi::Error dispatch(
    cudaStream_t stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> c, int64_t math_mode) {
  if (a.element_type() != ffi::DataType::C128 ||
      b.element_type() != ffi::DataType::C128 ||
      c->element_type() != ffi::DataType::C128) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "vertex_gemm_tc_probe: buffers must be complex128");
  }
  auto ad = a.dimensions();
  auto bd = b.dimensions();
  auto cd = c->dimensions();
  if (ad.size() != 3 || bd.size() != 3 || cd.size() != 3 ||
      ad[0] != bd[0] || ad[0] != cd[0] || ad[1] != cd[1] ||
      ad[2] != bd[1] || bd[2] != cd[2]) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "vertex_gemm_tc_probe: need A(batch,m,k), B(batch,k,n), C(batch,m,n)");
  }

  const int batch = static_cast<int>(ad[0]);
  const int m = static_cast<int>(ad[1]);
  const int k = static_cast<int>(ad[2]);
  const int n = static_cast<int>(bd[2]);
  if (batch == 0 || m == 0 || n == 0) return ffi::Error::Success();

  // Handles are thread-local because XLA may dispatch independent custom
  // calls from different host threads.  cublasSetStream orders the work on
  // the stream supplied by XLA; no synchronization is performed here.
  thread_local cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    cublasStatus_t st = cublasCreate(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) return fail("cublasCreate", st);
  }
  cublasStatus_t st = cublasSetStream(handle, stream);
  if (st != CUBLAS_STATUS_SUCCESS) return fail("cublasSetStream", st);

  cublasMath_t mode = CUBLAS_DEFAULT_MATH;
  if (math_mode == 1) mode = CUBLAS_TENSOR_OP_MATH;
#if defined(CUBLAS_PEDANTIC_MATH)
  if (math_mode == 2) mode = CUBLAS_PEDANTIC_MATH;
#endif
  st = cublasSetMathMode(handle, mode);
  if (st != CUBLAS_STATUS_SUCCESS) return fail("cublasSetMathMode", st);

  const auto* ap = static_cast<const cuDoubleComplex*>(a.untyped_data());
  const auto* bp = static_cast<const cuDoubleComplex*>(b.untyped_data());
  auto* cp = static_cast<cuDoubleComplex*>(c->untyped_data());
  const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
  const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

  // Inputs and output are row-major.  cuBLAS is column-major, so compute
  // C^T = B^T A^T: swap A/B and m/n while leaving the stored bytes alone.
  st = cublasZgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
      bp, n, static_cast<long long>(k) * n,
      ap, k, static_cast<long long>(m) * k, &beta,
      cp, n, static_cast<long long>(m) * n, batch);
  if (st != CUBLAS_STATUS_SUCCESS) {
    return fail("cublasZgemmStridedBatched", st);
  }
  return ffi::Error::Success();
}

}  // namespace lorrax_vertex_gemm_probe

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LorraxVertexZgemmStridedBatchedProbeFfi,
    lorrax_vertex_gemm_probe::dispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("math_mode"));
