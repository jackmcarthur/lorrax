// Local exact-range batched GEMM through classic BLAS: cuBLAS on the CUDA leg, rocBLAS on the
// ROCM leg (common/gpu_blas.h; one source, the same symbols, as phdf5 and slate).
//
// Each process owns complete contraction axes and independent centroid tiles:
//   A[nq,m,k] @ B[nq,k,n] -> C[nq,m,n].
// JAX presents row-major buffers.  Reinterpreting them as column-major gives
// C^T = B^T A^T, so an interval [lo,hi) is a pair of pointer views with the
// original leading dimensions.  No selected operand or distributed context is
// created, and no collective is issued.
//
// Two bindings.  The aliased one takes C and applies beta.  The `_out` one takes
// no C: beta is 0, so every batch row is written here (the GEMM over a live
// interval, a memset of an empty one) and the output needs no zero fill first.
//
// Leg-neutral: every runtime call goes through lrx::gpu:: (gpu_runtime.h) and every BLAS
// call through lrx::gpu::blas:: (gpu_blas.h).  On the CUDA leg those are inline forwards to
// the cudaX / cublasX calls this file made before, with the same arguments, so the CUDA
// handler is the one it was.  The bounds read (Dispatch / OutDispatch) is a stream-ordered
// D2H of 2*n_bounds int32 and a stream sync on both legs: the A100 trace kept it (it idles the
// GPU <= 1.2%, the queue refills within ~20 us: _active_local_cuda.py, kernel lessons).

#include "xla/ffi/api/ffi.h"
#include "common/ffi_helpers.h"

#include <algorithm>
#include <climits>
#include <complex>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "common/gpu_blas.h"

namespace ffi = xla::ffi;

namespace lorrax_ffi::cublas_local_active_gemm {

constexpr int64_t kWorkspaceBytes = 4 * 1024 * 1024;

namespace blas = lrx::gpu::blas;

#define LORRAX_LOCAL_GPU(call)                                                 \
  do {                                                                         \
    const lrx::gpu::Error status = (call);                                      \
    if (status != lrx::gpu::kSuccess)                                           \
      return ffi::Error::Internal(                                              \
          std::string("local active GEMM " LRX_GPU_VENDOR ": ") +              \
          lrx::gpu::error_string(status));                                      \
  } while (0)

static ffi::Error BlasError(const char* operation, blas::Status status) {
  std::ostringstream message;
  message << "local active GEMM " << operation << ": " LRX_BLAS_NAME " status "
          << static_cast<int>(status);
  return ffi::Error::Internal(message.str());
}

// A distinct handle is required here.  The active-subspace service binds its
// own handle to short-lived XLA workspace and therefore must not be shared.
struct LocalHandle {
  blas::Handle value = nullptr;
  blas::Status creation_status = blas::kNotInitialized;

  LocalHandle() { creation_status = blas::create(&value); }
  ~LocalHandle() {
    if (value != nullptr) blas::destroy(value);
  }
};

static ffi::Error GetHandle(lrx::gpu::Stream stream, void* workspace,
                            size_t workspace_bytes, blas::Handle* result) {
  // ROCM: rocBLAS is the process's mapped copy (GATE rocblas-mapped); CUDA: linked, nullptr.
  if (const char* why = blas::unavailable())
    return ffi::Error(ffi::ErrorCode::kFailedPrecondition, why);
  thread_local LocalHandle state;
  if (state.creation_status != blas::kSuccess || state.value == nullptr)
    return BlasError("handle creation", state.creation_status);
  // cublasSetStream resets a user-provided workspace, so the order is fixed (rocBLAS keeps
  // the same order).
  blas::Status status = blas::set_stream(state.value, stream);
  if (status != blas::kSuccess)
    return BlasError("stream binding", status);
  status = blas::set_workspace(state.value, workspace, workspace_bytes);
  if (status != blas::kSuccess)
    return BlasError("workspace binding", status);
  *result = state.value;
  return ffi::Error::Success();
}

template <typename T>
static ffi::Error ScaleEmpty(blas::Handle handle, T* output, int64_t count,
                             T beta) {
  // The caller handles beta=0 with a stream memset and beta=1 as a no-op.
  if (beta == T(0) || beta == T(1))
    return ffi::Error::InvalidArgument("local active GEMM internal scale misuse");
  for (int64_t offset = 0; offset < count;) {
    const int length = static_cast<int>(
        std::min<int64_t>(count - offset, std::numeric_limits<int>::max()));
    const blas::Status status = blas::scal(handle, length, &beta, output + offset, 1);
    if (status != blas::kSuccess) return BlasError("empty scaling", status);
    offset += length;
  }
  return ffi::Error::Success();
}

template <typename T>
static ffi::Error Run(lrx::gpu::Stream stream, blas::Handle handle, const T* a,
                      const T* b, T* output,
                      const std::vector<int32_t>& bounds, int64_t n_bounds,
                      int64_t nq, int64_t m, int64_t k, int64_t n, T alpha,
                      T beta) {
  const int64_t stride_a = m * k;
  const int64_t stride_b = k * n;
  const int64_t stride_c = m * n;
  for (int64_t first = 0; first < nq;) {
    const int64_t bound_index = n_bounds == 1 ? 0 : first;
    const int32_t lo = bounds[2 * bound_index];
    const int32_t hi = bounds[2 * bound_index + 1];
    int64_t end = first + 1;
    if (n_bounds == 1) {
      end = nq;
    } else {
      while (end < nq && bounds[2 * end] == lo && bounds[2 * end + 1] == hi)
        ++end;
    }
    const int batch_count = static_cast<int>(end - first);
    T* c_run = output + first * stride_c;
    if (lo == hi) {
      if (beta == T(0)) {
        LORRAX_LOCAL_GPU(lrx::gpu::memset_async(
            c_run, 0, (end - first) * stride_c * sizeof(T), stream));
      } else if (beta != T(1)) {
        FFI_RETURN_IF_ERROR(
            ScaleEmpty(handle, c_run, (end - first) * stride_c, beta));
      }
      first = end;
      continue;
    }
    // Row-major C=A*B is column-major C^T=B^T*A^T.  The active B rows
    // start at lo*n; the active A columns start at lo and retain lda=k.
    const blas::Status status = blas::gemm_strided_batched(
        handle, static_cast<int>(n), static_cast<int>(m), hi - lo, &alpha,
        b + first * stride_b + int64_t(lo) * n, static_cast<int>(n), stride_b,
        a + first * stride_a + lo, static_cast<int>(k), stride_a, &beta,
        c_run, static_cast<int>(n), stride_c, batch_count);
    if (status != blas::kSuccess) return BlasError("GEMM", status);
    first = end;
  }
  return ffi::Error::Success();
}

static bool ProductFitsInt64(int64_t x, int64_t y) {
  return x == 0 || y <= std::numeric_limits<int64_t>::max() / x;
}

// c_in == nullptr: the `_out` binding (no C operand, beta = 0).
static ffi::Error DispatchWithBounds(
    lrx::gpu::Stream stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
    const ffi::AnyBuffer* c_in,
    ffi::Result<ffi::AnyBuffer> c_out,
    ffi::Result<ffi::BufferR1<ffi::U8>> workspace,
    const std::vector<int32_t>& host_bounds, int64_t n_bounds, int64_t nq,
    int64_t m, int64_t k, int64_t n, double alpha_re, double alpha_im,
    double beta_re, double beta_im) {
  if (nq < 1 || m < 1 || k < 1 || n < 1 || nq > INT_MAX || m > INT_MAX ||
      k > INT_MAX || n > INT_MAX || !ProductFitsInt64(m, k) ||
      !ProductFitsInt64(k, n) || !ProductFitsInt64(m, n) ||
      !ProductFitsInt64(nq, m * k) || !ProductFitsInt64(nq, k * n) ||
      !ProductFitsInt64(nq, m * n))
    return ffi::Error::InvalidArgument("local active GEMM dimension overflow");
  const auto c_dims = c_in != nullptr ? c_in->dimensions() : c_out->dimensions();
  const ffi::DataType c_type = c_in != nullptr ? c_in->element_type() : c_out->element_type();
  if (a.dimensions().size() != 3 || b.dimensions().size() != 3 ||
      c_dims.size() != 3 || c_out->dimensions().size() != 3 ||
      a.dimensions()[0] != nq || a.dimensions()[1] != m ||
      a.dimensions()[2] != k || b.dimensions()[0] != nq ||
      b.dimensions()[1] != k || b.dimensions()[2] != n ||
      c_dims[0] != nq || c_dims[1] != m ||
      c_dims[2] != n || c_out->dimensions()[0] != nq ||
      c_out->dimensions()[1] != m || c_out->dimensions()[2] != n)
    return ffi::Error::InvalidArgument("local active GEMM buffer geometry");
  if (n_bounds != 1 && n_bounds != nq)
    return ffi::Error::InvalidArgument(
        "local active GEMM needs one or nq bound pairs");
  if (a.element_type() != b.element_type() ||
      a.element_type() != c_type ||
      a.element_type() != c_out->element_type())
    return ffi::Error::InvalidArgument("local active GEMM operand dtype mismatch");
  if (c_in != nullptr && c_in->untyped_data() != c_out->untyped_data())
    return ffi::Error::InvalidArgument(
        "local active GEMM requires declared C input/output alias");
  if (c_in == nullptr && (beta_re != 0.0 || beta_im != 0.0))
    return ffi::Error::InvalidArgument(
        "local active GEMM without C requires beta = 0");
  if (workspace->dimensions()[0] < kWorkspaceBytes)
    return ffi::Error::InvalidArgument("local active GEMM workspace too small");

  for (int64_t i = 0; i < n_bounds; ++i) {
    if (host_bounds[2 * i] < 0 ||
        host_bounds[2 * i] > host_bounds[2 * i + 1] ||
        host_bounds[2 * i + 1] > k)
      return ffi::Error::InvalidArgument(
          "local active GEMM requires 0 <= lo <= hi <= storage K");
  }

  blas::Handle handle = nullptr;
  FFI_RETURN_IF_ERROR(GetHandle(stream, workspace->typed_data(),
                               workspace->dimensions()[0], &handle));
  if (a.element_type() == ffi::DataType::F64) {
    if (alpha_im != 0.0 || beta_im != 0.0)
      return ffi::Error::InvalidArgument(
          "local active GEMM real operands require real alpha/beta");
    return Run(stream, handle, static_cast<const double*>(a.untyped_data()),
               static_cast<const double*>(b.untyped_data()),
               static_cast<double*>(c_out->untyped_data()), host_bounds,
               n_bounds, nq, m, k, n, alpha_re, beta_re);
  }
  if (a.element_type() == ffi::DataType::C128) {
    using T = std::complex<double>;
    return Run(stream, handle, static_cast<const T*>(a.untyped_data()),
               static_cast<const T*>(b.untyped_data()),
               static_cast<T*>(c_out->untyped_data()), host_bounds, n_bounds,
               nq, m, k, n, T(alpha_re, alpha_im), T(beta_re, beta_im));
  }
  return ffi::Error::InvalidArgument(
      "local active GEMM supports float64 and complex128");
}

static ffi::Error Dispatch(
    lrx::gpu::Stream stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::BufferR2<ffi::S32> bounds, ffi::AnyBuffer c_in,
    ffi::Result<ffi::AnyBuffer> c_out,
    ffi::Result<ffi::BufferR1<ffi::U8>> workspace, int64_t nq, int64_t m,
    int64_t k, int64_t n, double alpha_re, double alpha_im, double beta_re,
    double beta_im) {
  if (bounds.dimensions()[1] != 2 ||
      (bounds.dimensions()[0] != 1 && bounds.dimensions()[0] != nq))
    return ffi::Error::InvalidArgument(
        "local active GEMM bounds must have shape (1|nq,2)");
  const int64_t n_bounds = bounds.dimensions()[0];
  std::vector<int32_t> host_bounds(2 * n_bounds);
  LORRAX_LOCAL_GPU(lrx::gpu::memcpy_async(
      host_bounds.data(), bounds.typed_data(), host_bounds.size() * sizeof(int32_t),
      lrx::gpu::kDeviceToHost, stream));
  LORRAX_LOCAL_GPU(lrx::gpu::stream_synchronize(stream));
  return DispatchWithBounds(
      stream, a, b, &c_in, c_out, workspace, host_bounds, n_bounds, nq, m, k, n,
      alpha_re, alpha_im, beta_re, beta_im);
}

// The `_out` binding: no C operand, beta = 0 (the handler writes every row).
static ffi::Error OutDispatch(
    lrx::gpu::Stream stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::BufferR2<ffi::S32> bounds, ffi::Result<ffi::AnyBuffer> c_out,
    ffi::Result<ffi::BufferR1<ffi::U8>> workspace, int64_t nq, int64_t m,
    int64_t k, int64_t n, double alpha_re, double alpha_im) {
  if (bounds.dimensions()[1] != 2 ||
      (bounds.dimensions()[0] != 1 && bounds.dimensions()[0] != nq))
    return ffi::Error::InvalidArgument(
        "local active GEMM bounds must have shape (1|nq,2)");
  const int64_t n_bounds = bounds.dimensions()[0];
  std::vector<int32_t> host_bounds(2 * n_bounds);
  LORRAX_LOCAL_GPU(lrx::gpu::memcpy_async(
      host_bounds.data(), bounds.typed_data(), host_bounds.size() * sizeof(int32_t),
      lrx::gpu::kDeviceToHost, stream));
  LORRAX_LOCAL_GPU(lrx::gpu::stream_synchronize(stream));
  return DispatchWithBounds(
      stream, a, b, nullptr, c_out, workspace, host_bounds, n_bounds, nq, m, k, n,
      alpha_re, alpha_im, 0.0, 0.0);
}

static ffi::Error PreparedDispatch(
    lrx::gpu::Stream stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::AnyBuffer c_in, ffi::Result<ffi::AnyBuffer> c_out,
    ffi::Result<ffi::BufferR1<ffi::U8>> workspace, int64_t nq, int64_t m,
    int64_t k, int64_t n, double alpha_re, double alpha_im, double beta_re,
    double beta_im, ffi::Span<const int64_t> active_bounds) {
  const size_t pair_count = active_bounds.size() / 2;
  if (nq < 1 || active_bounds.size() % 2 != 0 ||
      (pair_count != 1 && pair_count != static_cast<size_t>(nq)))
    return ffi::Error::InvalidArgument(
        "local prepared active GEMM active_bounds must contain 1 or nq pairs");
  const int64_t n_bounds = static_cast<int64_t>(pair_count);
  std::vector<int32_t> host_bounds(active_bounds.size());
  for (size_t i = 0; i < active_bounds.size(); ++i) {
    if (active_bounds[i] < 0 || active_bounds[i] > INT_MAX)
      return ffi::Error::InvalidArgument(
          "local prepared active GEMM bounds must fit nonnegative int32");
    host_bounds[i] = static_cast<int32_t>(active_bounds[i]);
  }
  return DispatchWithBounds(
      stream, a, b, &c_in, c_out, workspace, host_bounds, n_bounds, nq, m, k, n,
      alpha_re, alpha_im, beta_re, beta_im);
}

}  // namespace lorrax_ffi::cublas_local_active_gemm

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CublasLocalActiveRangeGemmFfi,
    lorrax_ffi::cublas_local_active_gemm::Dispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::U8>>()
        .Attr<int64_t>("nq")
        .Attr<int64_t>("m")
        .Attr<int64_t>("k")
        .Attr<int64_t>("n")
        .Attr<double>("alpha_re")
        .Attr<double>("alpha_im")
        .Attr<double>("beta_re")
        .Attr<double>("beta_im"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CublasLocalActiveRangeGemmOutFfi,
    lorrax_ffi::cublas_local_active_gemm::OutDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::U8>>()
        .Attr<int64_t>("nq")
        .Attr<int64_t>("m")
        .Attr<int64_t>("k")
        .Attr<int64_t>("n")
        .Attr<double>("alpha_re")
        .Attr<double>("alpha_im"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CublasLocalPreparedActiveRangeGemmFfi,
    lorrax_ffi::cublas_local_active_gemm::PreparedDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::BufferR1<ffi::U8>>()
        .Attr<int64_t>("nq")
        .Attr<int64_t>("m")
        .Attr<int64_t>("k")
        .Attr<int64_t>("n")
        .Attr<double>("alpha_re")
        .Attr<double>("alpha_im")
        .Attr<double>("beta_re")
        .Attr<double>("beta_im")
        .Attr<ffi::Span<const int64_t>>("active_bounds"));
