#include "xla/ffi/api/ffi.h"
#include <climits>
#include <complex>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <string>
namespace ffi = xla::ffi;
namespace lorrax_ffi::active_subspace {
using B = ffi::BufferR2<ffi::C128>;
using I = ffi::BufferR1<ffi::S32>;
using R = ffi::Result<B>;
using Scratch = ffi::BufferR1<ffi::U8>;
using ScratchResult = ffi::Result<Scratch>;
#define CUDA(call)                                                             \
  do {                                                                         \
    auto rc = (call);                                                          \
    if (rc != cudaSuccess)                                                     \
      return ffi::Error::Internal(cudaGetErrorString(rc));                     \
  } while (0)
#define BLAS(call)                                                             \
  do {                                                                         \
    auto rc = (call);                                                          \
    if (rc != CUBLAS_STATUS_SUCCESS)                                           \
      return ffi::Error::Internal("cublas " + std::to_string(rc));             \
  } while (0)
struct BlasHandle {
  cublasHandle_t h = nullptr;
  BlasHandle() { cublasCreate(&h); }
  ~BlasHandle() {
    if (h)
      cublasDestroy(h);
  }
};
static cublasHandle_t handle() {
  thread_local BlasHandle h;
  return h.h;
}
static cuDoubleComplex *ptr(B a) {
  return reinterpret_cast<cuDoubleComplex *>(a.typed_data());
}
static cuDoubleComplex *ptr(R a) {
  return reinterpret_cast<cuDoubleComplex *>(a->typed_data());
}
static ffi::Error project(cudaStream_t stream, B v, B hv, B h, I active, R out,
                          ScratchResult scratch) {
  if (active.dimensions()[0] != 3 || v.dimensions()[0] < 1 ||
      v.dimensions()[0] > INT_MAX || v.dimensions()[1] < 1 ||
      v.dimensions()[1] > INT_MAX)
    return ffi::Error::InvalidArgument("project descriptor/dimension overflow");
  int q[3];
  CUDA(cudaMemcpyAsync(q, active.typed_data(), sizeof(q),
                       cudaMemcpyDeviceToHost, stream));
  CUDA(cudaStreamSynchronize(stream));
  int cap = v.dimensions()[0], d = v.dimensions()[1], m = q[0], start = q[1],
      r = q[2];
  if (hv.dimensions()[0] != cap || hv.dimensions()[1] != d ||
      h.dimensions()[0] != cap || h.dimensions()[1] != cap ||
      out->dimensions()[0] != cap || out->dimensions()[1] != cap)
    return ffi::Error::InvalidArgument("project buffer geometry");
  if (m > cap || m < 1 || start < 0 || r < 0 || int64_t(start) + r != m)
    return ffi::Error::InvalidArgument("project active geometry");
  BLAS(cublasSetStream(handle(), stream));
  BLAS(cublasSetWorkspace(handle(), scratch->typed_data(),
                          scratch->dimensions()[0]));
  if (out->typed_data() != h.typed_data())
    return ffi::Error::InvalidArgument(
        "active projection requires declared input/output alias");
  const cuDoubleComplex one{1, 0}, zero{0, 0};
  if (r) {
    BLAS(cublasZgemm(handle(), CUBLAS_OP_C, CUBLAS_OP_N, m, r, d, &one, ptr(v),
                     d, ptr(hv) + int64_t(start) * d, d, &zero,
                     ptr(out) + int64_t(start) * cap, cap));
    BLAS(cublasZgemm(handle(), CUBLAS_OP_C, CUBLAS_OP_N, r, m, d, &one,
                     ptr(hv) + int64_t(start) * d, d, ptr(v), d, &zero,
                     ptr(out) + start, cap));
  }
  return ffi::Error::Success();
}
static ffi::Error reconstruct(cudaStream_t stream, B v, B hv, B c, I active,
                              R x, R hx, ScratchResult scratch) {
  if (active.dimensions()[0] != 3 || v.dimensions()[0] < 1 ||
      v.dimensions()[0] > INT_MAX || v.dimensions()[1] < 1 ||
      v.dimensions()[1] > INT_MAX || x->dimensions()[0] > INT_MAX)
    return ffi::Error::InvalidArgument(
        "reconstruct descriptor/dimension overflow");
  int q[3];
  CUDA(cudaMemcpyAsync(q, active.typed_data(), sizeof(q),
                       cudaMemcpyDeviceToHost, stream));
  CUDA(cudaStreamSynchronize(stream));
  int cap = v.dimensions()[0], d = v.dimensions()[1], b = x->dimensions()[0],
      m = q[0], columns = q[1];
  if (hv.dimensions()[0] != cap || hv.dimensions()[1] != d ||
      c.dimensions()[0] != cap || c.dimensions()[1] != b ||
      x->dimensions()[1] != d || hx->dimensions()[0] != b ||
      hx->dimensions()[1] != d)
    return ffi::Error::InvalidArgument("reconstruct buffer geometry");
  if (m < 1 || m > cap || columns < 1 || columns > b)
    return ffi::Error::InvalidArgument("reconstruct active geometry");
  BLAS(cublasSetStream(handle(), stream));
  BLAS(cublasSetWorkspace(handle(), scratch->typed_data(),
                          scratch->dimensions()[0]));
  const cuDoubleComplex one{1, 0}, zero{0, 0};
  if (columns < b) {
    CUDA(cudaMemsetAsync(ptr(x) + int64_t(columns) * d, 0,
                         int64_t(b - columns) * d * sizeof(cuDoubleComplex),
                         stream));
  }
  BLAS(cublasZgemm(handle(), CUBLAS_OP_N, CUBLAS_OP_N, d, columns, m, &one,
                   ptr(v), d, ptr(c), cap, &zero, ptr(x), d));
  if (q[2]) {
    if (columns < b)
      CUDA(cudaMemsetAsync(ptr(hx) + int64_t(columns) * d, 0,
                           int64_t(b - columns) * d * sizeof(cuDoubleComplex),
                           stream));
    BLAS(cublasZgemm(handle(), CUBLAS_OP_N, CUBLAS_OP_N, d, columns, m, &one,
                     ptr(hv), d, ptr(c), cap, &zero, ptr(hx), d));
  } else {
    CUDA(cudaMemsetAsync(ptr(hx), 0, int64_t(b) * d * sizeof(cuDoubleComplex),
                         stream));
  }
  return ffi::Error::Success();
}
static ffi::Error orthogonalize(cudaStream_t stream, B v, B p, I active, R out,
                                R work, ScratchResult scratch) {
  if (active.dimensions()[0] != 1 || v.dimensions()[0] < 1 ||
      v.dimensions()[0] > INT_MAX || v.dimensions()[1] < 1 ||
      v.dimensions()[1] > INT_MAX || p.dimensions()[0] < 1 ||
      p.dimensions()[0] > INT_MAX)
    return ffi::Error::InvalidArgument(
        "orthogonalize descriptor/dimension overflow");
  int m;
  CUDA(cudaMemcpyAsync(&m, active.typed_data(), sizeof(m),
                       cudaMemcpyDeviceToHost, stream));
  CUDA(cudaStreamSynchronize(stream));
  int cap = v.dimensions()[0], d = v.dimensions()[1], b = p.dimensions()[0];
  if (p.dimensions()[1] != d || out->dimensions()[0] != b ||
      out->dimensions()[1] != d || work->dimensions()[0] != cap ||
      work->dimensions()[1] != b)
    return ffi::Error::InvalidArgument("orthogonalize buffer geometry");
  if (m < 1 || m > cap)
    return ffi::Error::InvalidArgument("ortho active geometry");
  BLAS(cublasSetStream(handle(), stream));
  BLAS(cublasSetWorkspace(handle(), scratch->typed_data(),
                          scratch->dimensions()[0]));
  const cuDoubleComplex one{1, 0}, zero{0, 0}, minus{-1, 0};
  CUDA(cudaMemcpyAsync(out->typed_data(), p.typed_data(),
                       int64_t(b) * d * sizeof(cuDoubleComplex),
                       cudaMemcpyDeviceToDevice, stream));
  for (int pass = 0; pass < 2; ++pass) {
    BLAS(cublasZgemm(handle(), CUBLAS_OP_C, CUBLAS_OP_N, m, b, d, &one, ptr(v),
                     d, ptr(out), d, &zero, ptr(work), cap));
    BLAS(cublasZgemm(handle(), CUBLAS_OP_N, CUBLAS_OP_N, d, b, m, &minus,
                     ptr(v), d, ptr(work), cap, &one, ptr(out), d));
  }
  return ffi::Error::Success();
}
// XLA aliases the first two operands to the two results. Only the active
// inserted block is copied; neither the capacity tail nor a false branch is.
static ffi::Error store(cudaStream_t stream, B v, B hv, B p, B hp, I range,
                        R out, R hout) {
  if (range.dimensions()[0] != 2 || v.dimensions()[0] != hv.dimensions()[0] ||
      v.dimensions()[1] != hv.dimensions()[1] ||
      p.dimensions()[0] != hp.dimensions()[0] ||
      p.dimensions()[1] != hp.dimensions()[1] ||
      v.dimensions()[1] != p.dimensions()[1])
    return ffi::Error::InvalidArgument("active store buffer geometry");
  if (out->typed_data() != v.typed_data() ||
      hout->typed_data() != hv.typed_data())
    return ffi::Error::InvalidArgument(
        "active store requires declared input/output aliases");
  int q[2];
  CUDA(cudaMemcpyAsync(q, range.typed_data(), sizeof(q), cudaMemcpyDeviceToHost,
                       stream));
  CUDA(cudaStreamSynchronize(stream));
  int64_t start = q[0], count = q[1], d = v.dimensions()[1];
  if (start < 0 || count < 0 || count > p.dimensions()[0] ||
      start + count > v.dimensions()[0])
    return ffi::Error::InvalidArgument("active store range outside capacity");
  if (count) {
    CUDA(cudaMemcpyAsync(ptr(out) + start * d, ptr(p),
                         count * d * sizeof(cuDoubleComplex),
                         cudaMemcpyDeviceToDevice, stream));
    CUDA(cudaMemcpyAsync(ptr(hout) + start * d, ptr(hp),
                         count * d * sizeof(cuDoubleComplex),
                         cudaMemcpyDeviceToDevice, stream));
  }
  return ffi::Error::Success();
}
} // namespace lorrax_ffi::active_subspace
using namespace lorrax_ffi::active_subspace;
XLA_FFI_DEFINE_HANDLER_SYMBOL(ActiveSubspaceProjectFfi, project,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<B>()
                                  .Arg<B>()
                                  .Arg<B>()
                                  .Arg<I>()
                                  .Ret<B>()
                                  .Ret<Scratch>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(ActiveSubspaceReconstructFfi, reconstruct,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<B>()
                                  .Arg<B>()
                                  .Arg<B>()
                                  .Arg<I>()
                                  .Ret<B>()
                                  .Ret<B>()
                                  .Ret<Scratch>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(ActiveSubspaceOrthoFfi, orthogonalize,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<B>()
                                  .Arg<B>()
                                  .Arg<I>()
                                  .Ret<B>()
                                  .Ret<B>()
                                  .Ret<Scratch>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(ActiveSubspaceStoreFfi, store,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<B>()
                                  .Arg<B>()
                                  .Arg<B>()
                                  .Arg<B>()
                                  .Arg<I>()
                                  .Ret<B>()
                                  .Ret<B>());
