// Experiment: exact active-prefix cuSOLVER eigensolve with XLA-owned workspace.
#include "xla/ffi/api/ffi.h"
#include <algorithm>
#include <climits>
#include <complex>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <string>
namespace ffi = xla::ffi;
namespace lorrax_ffi::active_subspace {
struct Handle {
  cusolverDnHandle_t h = nullptr;
  Handle() { cusolverDnCreate(&h); }
  ~Handle() {
    if (h)
      cusolverDnDestroy(h);
  }
};
static cusolverDnHandle_t handle() {
  thread_local Handle h;
  return h.h;
}
extern "C" int lrx_active_eigh_lwork(int n) {
  int size = 0;
  auto rc = cusolverDnZheevd_bufferSize(handle(), CUSOLVER_EIG_MODE_VECTOR,
                                        CUBLAS_FILL_MODE_LOWER, n, nullptr, n,
                                        nullptr, &size);
  return rc == CUSOLVER_STATUS_SUCCESS ? size : -int(rc);
}
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    auto err = (call);                                                         \
    if (err != cudaSuccess)                                                    \
      return ffi::Error::Internal(cudaGetErrorString(err));                    \
  } while (0)
#define SOLVER_CHECK(call)                                                     \
  do {                                                                         \
    auto err = (call);                                                         \
    if (err != CUSOLVER_STATUS_SUCCESS)                                        \
      return ffi::Error::Internal("cuSOLVER status " +                         \
                                  std::to_string(int(err)));                   \
  } while (0)
static ffi::Error
active_eigh(cudaStream_t stream, ffi::BufferR2<ffi::C128> h,
            ffi::BufferR0<ffi::S32> active,
            ffi::Result<ffi::BufferR1<ffi::F64>> eigenvalues,
            ffi::Result<ffi::BufferR2<ffi::C128>> coefficients,
            ffi::Result<ffi::BufferR1<ffi::C128>> work,
            ffi::Result<ffi::BufferR1<ffi::F64>> evalwork,
            ffi::Result<ffi::BufferR1<ffi::S32>> info) {
  if (h.dimensions()[0] < 1 || h.dimensions()[0] > INT_MAX ||
      eigenvalues->dimensions()[0] < 1 ||
      eigenvalues->dimensions()[0] > h.dimensions()[0])
    return ffi::Error::InvalidArgument("active_eigh dimension overflow");
  const int cap = h.dimensions()[0], b = eigenvalues->dimensions()[0];
  if (h.dimensions()[1] != cap || coefficients->dimensions()[0] != cap ||
      coefficients->dimensions()[1] != b || evalwork->dimensions()[0] != cap ||
      info->dimensions()[0] != 1)
    return ffi::Error::InvalidArgument("active_eigh buffer geometry mismatch");
  int n = 0;
  CUDA_CHECK(cudaMemcpyAsync(&n, active.typed_data(), sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (n < 1 || n > cap)
    return ffi::Error::InvalidArgument(
        "active_eigh active size outside capacity");
  auto a = reinterpret_cast<cuDoubleComplex *>(work->typed_data());
  const int64_t declared = work->dimensions()[0] - int64_t(cap) * cap;
  if (declared < 1 || declared > INT_MAX)
    return ffi::Error::InvalidArgument("active_eigh workspace geometry");
  const int lwork = declared;
  auto scratch = a + int64_t(cap) * cap;
  int needed = 0;
  SOLVER_CHECK(cusolverDnSetStream(handle(), stream));
  SOLVER_CHECK(cusolverDnZheevd_bufferSize(handle(), CUSOLVER_EIG_MODE_VECTOR,
                                           CUBLAS_FILL_MODE_LOWER, n, a, cap,
                                           evalwork->typed_data(), &needed));
  if (needed > lwork)
    return ffi::Error::InvalidArgument(
        "active_eigh insufficient declared workspace");
  CUDA_CHECK(cudaMemcpy2DAsync(a, cap * sizeof(cuDoubleComplex), h.typed_data(),
                               cap * sizeof(cuDoubleComplex),
                               n * sizeof(cuDoubleComplex), n,
                               cudaMemcpyDeviceToDevice, stream));
  SOLVER_CHECK(cusolverDnZheevd(
      handle(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, a, cap,
      evalwork->typed_data(), scratch, lwork, info->typed_data()));
  const int columns = std::min(n, b);
  CUDA_CHECK(cudaMemsetAsync(eigenvalues->typed_data(), 0, b * sizeof(double),
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(eigenvalues->typed_data(), evalwork->typed_data(),
                             columns * sizeof(double), cudaMemcpyDeviceToDevice,
                             stream));
  CUDA_CHECK(cudaMemsetAsync(coefficients->typed_data(), 0,
                             int64_t(cap) * b * sizeof(cuDoubleComplex),
                             stream));
  CUDA_CHECK(cudaMemcpy2DAsync(
      coefficients->typed_data(), cap * sizeof(cuDoubleComplex), a,
      cap * sizeof(cuDoubleComplex), n * sizeof(cuDoubleComplex), columns,
      cudaMemcpyDeviceToDevice, stream));
  return ffi::Error::Success();
}
} // namespace lorrax_ffi::active_subspace
using namespace lorrax_ffi::active_subspace;
XLA_FFI_DEFINE_HANDLER_SYMBOL(ActiveSubspaceEighFfi, active_eigh,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::BufferR2<ffi::C128>>()
                                  .Arg<ffi::BufferR0<ffi::S32>>()
                                  .Ret<ffi::BufferR1<ffi::F64>>()
                                  .Ret<ffi::BufferR2<ffi::C128>>()
                                  .Ret<ffi::BufferR1<ffi::C128>>()
                                  .Ret<ffi::BufferR1<ffi::F64>>()
                                  .Ret<ffi::BufferR1<ffi::S32>>());
