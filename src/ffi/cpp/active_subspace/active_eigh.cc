// Exact active-prefix Hermitian eigensolve with XLA-owned workspace.
// Leg-neutral: cuSOLVERDn on the CUDA leg, rocSOLVER on the ROCM leg, both through
// common/gpu_solver.h (its CUDA spellings are the cuSOLVER calls this file made before).
#include "xla/ffi/api/ffi.h"
#include <algorithm>
#include <climits>
#include <complex>
#include <cstdint>
#include <string>

#include "../common/c_abi.h"
#include "../common/gpu_runtime.h"
#include "../common/gpu_solver.h"
namespace ffi = xla::ffi;
namespace lorrax_ffi::active_subspace {
namespace solver = lrx::gpu::solver;
struct Handle {
  solver::Handle h = nullptr;
  Handle() { solver::create(&h); }
  ~Handle() {
    if (h)
      solver::destroy(h);
  }
};
static solver::Handle handle() {
  thread_local Handle h;
  return h.h;
}
// lrx_active_eigh_lwork on CUDA, lrx_active_eigh_lwork_rocm on ROCM (c_abi.h).
extern "C" int LRX_C_ENTRY(lrx_active_eigh_lwork)(int n) {
  int size = 0;
  auto rc = solver::zheevd_buffer_size(handle(), n, nullptr, n, nullptr, &size);
  return rc == solver::kSuccess ? size : -int(rc);
}
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    auto err = (call);                                                         \
    if (err != lrx::gpu::kSuccess)                                             \
      return ffi::Error::Internal(lrx::gpu::error_string(err));                \
  } while (0)
#define SOLVER_CHECK(call)                                                     \
  do {                                                                         \
    auto err = (call);                                                         \
    if (err != solver::kSuccess)                                               \
      return ffi::Error::Internal(LRX_SOLVER_API " status " +                  \
                                  std::to_string(int(err)));                   \
  } while (0)
static ffi::Error
active_eigh(lrx::gpu::Stream stream, ffi::BufferR2<ffi::C128> h,
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
  if (const char *why = solver::unavailable())
    return ffi::Error(ffi::ErrorCode::kFailedPrecondition, why);
  int n = 0;
  CUDA_CHECK(lrx::gpu::memcpy_async(&n, active.typed_data(), sizeof(int),
                                    lrx::gpu::kDeviceToHost, stream));
  CUDA_CHECK(lrx::gpu::stream_synchronize(stream));
  if (n < 1 || n > cap)
    return ffi::Error::InvalidArgument(
        "active_eigh active size outside capacity");
  auto a = reinterpret_cast<solver::Complex *>(work->typed_data());
  const int64_t declared = work->dimensions()[0] - int64_t(cap) * cap;
  if (declared < 1 || declared > INT_MAX)
    return ffi::Error::InvalidArgument("active_eigh workspace geometry");
  const int lwork = declared;
  auto scratch = a + int64_t(cap) * cap;
  int needed = 0;
  SOLVER_CHECK(solver::set_stream(handle(), stream));
  SOLVER_CHECK(solver::zheevd_buffer_size(handle(), n, a, cap,
                                          evalwork->typed_data(), &needed));
  if (needed > lwork)
    return ffi::Error::InvalidArgument(
        "active_eigh insufficient declared workspace");
  CUDA_CHECK(lrx::gpu::memcpy_2d_async(a, cap * sizeof(solver::Complex), h.typed_data(),
                                      cap * sizeof(solver::Complex),
                                      n * sizeof(solver::Complex), n,
                                      lrx::gpu::kDeviceToDevice, stream));
  SOLVER_CHECK(solver::zheevd(handle(), n, a, cap, evalwork->typed_data(),
                              scratch, lwork, info->typed_data()));
  const int columns = std::min(n, b);
  CUDA_CHECK(lrx::gpu::memset_async(eigenvalues->typed_data(), 0, b * sizeof(double),
                                    stream));
  CUDA_CHECK(lrx::gpu::memcpy_async(eigenvalues->typed_data(), evalwork->typed_data(),
                                    columns * sizeof(double), lrx::gpu::kDeviceToDevice,
                                    stream));
  CUDA_CHECK(lrx::gpu::memset_async(coefficients->typed_data(), 0,
                                    int64_t(cap) * b * sizeof(solver::Complex),
                                    stream));
  CUDA_CHECK(lrx::gpu::memcpy_2d_async(
      coefficients->typed_data(), cap * sizeof(solver::Complex), a,
      cap * sizeof(solver::Complex), n * sizeof(solver::Complex), columns,
      lrx::gpu::kDeviceToDevice, stream));
  return ffi::Error::Success();
}
} // namespace lorrax_ffi::active_subspace
using namespace lorrax_ffi::active_subspace;
XLA_FFI_DEFINE_HANDLER_SYMBOL(ActiveSubspaceEighFfi, active_eigh,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<lrx::gpu::Stream>>()
                                  .Arg<ffi::BufferR2<ffi::C128>>()
                                  .Arg<ffi::BufferR0<ffi::S32>>()
                                  .Ret<ffi::BufferR1<ffi::F64>>()
                                  .Ret<ffi::BufferR2<ffi::C128>>()
                                  .Ret<ffi::BufferR1<ffi::C128>>()
                                  .Ret<ffi::BufferR1<ffi::F64>>()
                                  .Ret<ffi::BufferR1<ffi::S32>>());
