#include "xla/ffi/api/ffi.h"
#include <climits>
#include <complex>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <mutex>
#include <nccl.h>
#include <string>

#include "../cusolvermp/ctx.h"
namespace ffi = xla::ffi;
namespace lorrax_ffi::active_subspace {
using B = ffi::BufferR2<ffi::C128>;
using C = ffi::BufferR1<ffi::C128>;
using I = ffi::BufferR1<ffi::S32>;
using R = ffi::Result<B>;
using CR = ffi::Result<C>;
using IR = ffi::Result<I>;
using Scratch = ffi::BufferR1<ffi::U8>;
using ScratchResult = ffi::Result<Scratch>;
using lorrax_ffi::cusolvermp::LorraxCusolverMpCtx;
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
#define NCCL(call)                                                             \
  do {                                                                         \
    auto rc = (call);                                                          \
    if (rc != ncclSuccess)                                                     \
      return ffi::Error::Internal(std::string("nccl ") +                      \
                                  ncclGetErrorString(rc));                     \
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
static cuDoubleComplex *ptr(CR a) {
  return reinterpret_cast<cuDoubleComplex *>(a->typed_data());
}

// A cuSolverMp context owns a private stream and pooled events.  Once armed,
// this guard makes XLA wait for that stream even when a later library call
// returns an error, so XLA-owned scratch cannot be recycled while native work
// is still using it.
class ContextStreamJoin {
public:
  ContextStreamJoin(cudaStream_t xla_stream, LorraxCusolverMpCtx *ctx)
      : xla_stream_(xla_stream), ctx_(ctx) {}

  ffi::Error start() {
    CUDA(cudaEventRecord(ctx_->active_subspace_ev_xla_in, xla_stream_));
    CUDA(cudaStreamWaitEvent(ctx_->stream,
                             ctx_->active_subspace_ev_xla_in, 0));
    armed_ = true;
    return ffi::Error::Success();
  }

  ffi::Error finish() {
    CUDA(cudaEventRecord(ctx_->active_subspace_ev_ctx_out, ctx_->stream));
    CUDA(cudaStreamWaitEvent(xla_stream_,
                             ctx_->active_subspace_ev_ctx_out, 0));
    armed_ = false;
    return ffi::Error::Success();
  }

  ~ContextStreamJoin() {
    if (armed_) {
      // Error recovery is best effort: destructors cannot return an FFI
      // error, but retaining the dependency is safer than letting XLA reuse
      // result scratch while already-enqueued context work is in flight.
      if (cudaEventRecord(ctx_->active_subspace_ev_ctx_out,
                          ctx_->stream) == cudaSuccess)
        (void)cudaStreamWaitEvent(xla_stream_,
                                  ctx_->active_subspace_ev_ctx_out, 0);
    }
  }

private:
  cudaStream_t xla_stream_;
  LorraxCusolverMpCtx *ctx_;
  bool armed_ = false;
};
// Apply the two projection GEMMs after the caller has validated the buffers
// and copied the small range descriptor to the host.  The fused store/project
// handler uses this helper so it does not need a second device-to-host copy or
// stream synchronization between inserting vectors and projecting them.
static ffi::Error project_host(cudaStream_t stream, B v, B hv, int cap, int d,
                               int m, int start, int r, R out,
                               ScratchResult scratch) {
  BLAS(cublasSetStream(handle(), stream));
  BLAS(cublasSetWorkspace(handle(), scratch->typed_data(),
                          scratch->dimensions()[0]));
  const cuDoubleComplex one{1, 0}, zero{0, 0};
  if (r) {
    BLAS(cublasZgemm(handle(), CUBLAS_OP_C, CUBLAS_OP_N, m, r, d, &one,
                     ptr(v), d, ptr(hv) + int64_t(start) * d, d, &zero,
                     ptr(out) + int64_t(start) * cap, cap));
    BLAS(cublasZgemm(handle(), CUBLAS_OP_C, CUBLAS_OP_N, r, m, d, &one,
                     ptr(hv) + int64_t(start) * d, d, ptr(v), d, &zero,
                     ptr(out) + start, cap));
  }
  return ffi::Error::Success();
}

static ffi::Error project(cudaStream_t stream, B v, B hv, B h, I active,
                          R out, ScratchResult scratch) {
  if (active.dimensions()[0] != 3 || v.dimensions()[0] < 1 ||
      v.dimensions()[0] > INT_MAX || v.dimensions()[1] < 1 ||
      v.dimensions()[1] > INT_MAX)
    return ffi::Error::InvalidArgument("project descriptor/dimension overflow");
  int cap = v.dimensions()[0], d = v.dimensions()[1];
  if (hv.dimensions()[0] != cap || hv.dimensions()[1] != d ||
      h.dimensions()[0] != cap || h.dimensions()[1] != cap ||
      out->dimensions()[0] != cap || out->dimensions()[1] != cap)
    return ffi::Error::InvalidArgument("project buffer geometry");
  if (out->typed_data() != h.typed_data())
    return ffi::Error::InvalidArgument(
        "active projection requires declared input/output alias");
  int q[3];
  CUDA(cudaMemcpyAsync(q, active.typed_data(), sizeof(q),
                       cudaMemcpyDeviceToHost, stream));
  CUDA(cudaStreamSynchronize(stream));
  int m = q[0], start = q[1], r = q[2];
  if (m > cap || m < 1 || start < 0 || r < 0 || int64_t(start) + r != m)
    return ffi::Error::InvalidArgument("project active geometry");
  return project_host(stream, v, hv, cap, d, m, start, r, out, scratch);
}
static ffi::Error reconstruct(cudaStream_t stream, B v, B hv, B c, I active,
                              R x, R hx, ScratchResult scratch) {
  if (active.dimensions()[0] != 4 || v.dimensions()[0] < 1 ||
      v.dimensions()[0] > INT_MAX || v.dimensions()[1] < 1 ||
      v.dimensions()[1] > INT_MAX || x->dimensions()[0] > INT_MAX)
    return ffi::Error::InvalidArgument(
        "reconstruct descriptor/dimension overflow");
  int q[4];
  CUDA(cudaMemcpyAsync(q, active.typed_data(), sizeof(q),
                       cudaMemcpyDeviceToHost, stream));
  CUDA(cudaStreamSynchronize(stream));
  int cap = v.dimensions()[0], d = v.dimensions()[1], b = x->dimensions()[0],
      m = q[0], columns = q[1], start = q[3];
  if (hv.dimensions()[0] != cap || hv.dimensions()[1] != d ||
      c.dimensions()[0] != cap || c.dimensions()[1] != b ||
      x->dimensions()[1] != d || hx->dimensions()[0] != b ||
      hx->dimensions()[1] != d)
    return ffi::Error::InvalidArgument("reconstruct buffer geometry");
  if (m < 0 || start < 0 || int64_t(start) + m > cap || columns < 0 || columns > b)
    return ffi::Error::InvalidArgument("reconstruct active geometry");
  BLAS(cublasSetStream(handle(), stream));
  BLAS(cublasSetWorkspace(handle(), scratch->typed_data(),
                          scratch->dimensions()[0]));
  const cuDoubleComplex one{1, 0}, zero{0, 0};
  if (!m || !columns) {
    CUDA(cudaMemsetAsync(ptr(x), 0, int64_t(b)*d*sizeof(cuDoubleComplex), stream));
    CUDA(cudaMemsetAsync(ptr(hx), 0, int64_t(b)*d*sizeof(cuDoubleComplex), stream));
    return ffi::Error::Success();
  }
  if (columns < b) {
    CUDA(cudaMemsetAsync(ptr(x) + int64_t(columns) * d, 0,
                         int64_t(b - columns) * d * sizeof(cuDoubleComplex),
                         stream));
  }
  BLAS(cublasZgemm(handle(), CUBLAS_OP_N, CUBLAS_OP_N, d, columns, m, &one,
                   ptr(v) + int64_t(start)*d, d, ptr(c) + start, cap, &zero, ptr(x), d));
  if (q[2]) {
    if (columns < b)
      CUDA(cudaMemsetAsync(ptr(hx) + int64_t(columns) * d, 0,
                           int64_t(b - columns) * d * sizeof(cuDoubleComplex),
                           stream));
    BLAS(cublasZgemm(handle(), CUBLAS_OP_N, CUBLAS_OP_N, d, columns, m, &one,
                     ptr(hv) + int64_t(start)*d, d, ptr(c) + start, cap, &zero, ptr(hx), d));
  } else {
    CUDA(cudaMemsetAsync(ptr(hx), 0, int64_t(b) * d * sizeof(cuDoubleComplex),
                         stream));
  }
  return ffi::Error::Success();
}
static ffi::Error orthogonalize(cudaStream_t stream, B v, B p, I active, R out,
                                R work, ScratchResult scratch) {
  if (active.dimensions()[0] != 2 || v.dimensions()[0] < 1 ||
      v.dimensions()[0] > INT_MAX || v.dimensions()[1] < 1 ||
      v.dimensions()[1] > INT_MAX || p.dimensions()[0] < 1 ||
      p.dimensions()[0] > INT_MAX)
    return ffi::Error::InvalidArgument(
        "orthogonalize descriptor/dimension overflow");
  int q[2];
  CUDA(cudaMemcpyAsync(q, active.typed_data(), sizeof(q),
                       cudaMemcpyDeviceToHost, stream));
  CUDA(cudaStreamSynchronize(stream));
  int start = q[0], m = q[1];
  int cap = v.dimensions()[0], d = v.dimensions()[1], b = p.dimensions()[0];
  if (p.dimensions()[1] != d || out->dimensions()[0] != b ||
      out->dimensions()[1] != d || work->dimensions()[0] != cap ||
      work->dimensions()[1] != b)
    return ffi::Error::InvalidArgument("orthogonalize buffer geometry");
  if (m < 0 || start < 0 || int64_t(start) + m > cap)
    return ffi::Error::InvalidArgument("ortho active geometry");
  BLAS(cublasSetStream(handle(), stream));
  BLAS(cublasSetWorkspace(handle(), scratch->typed_data(),
                          scratch->dimensions()[0]));
  const cuDoubleComplex one{1, 0}, zero{0, 0}, minus{-1, 0};
  // The legacy wrapper returns a fresh buffer.  New callers may declare the
  // same operand/result alias and avoid this block-sized device copy.
  if (out->typed_data() != p.typed_data())
    CUDA(cudaMemcpyAsync(out->typed_data(), p.typed_data(),
                         int64_t(b) * d * sizeof(cuDoubleComplex),
                         cudaMemcpyDeviceToDevice, stream));
  if (!m) return ffi::Error::Success();
  for (int pass = 0; pass < 2; ++pass) {
    BLAS(cublasZgemm(handle(), CUBLAS_OP_C, CUBLAS_OP_N, m, b, d, &one, ptr(v) + int64_t(start)*d,
                     d, ptr(out), d, &zero, ptr(work), cap));
    BLAS(cublasZgemm(handle(), CUBLAS_OP_N, CUBLAS_OP_N, d, b, m, &minus,
                     ptr(v) + int64_t(start)*d, d, ptr(work), cap, &one, ptr(out), d));
  }
  return ffi::Error::Success();
}

// Two-pass classical Gram-Schmidt for a vector block distributed over every
// process.  V and P remain process-local; only the count-by-block overlap
// coefficients cross the all-process NCCL communicator.  A fixed-size range
// exchange precedes the variable-size reductions so one disagreeing rank is
// rejected by every peer before any peer can enter a different collective.
static ffi::Error distributed_orthogonalize(
    cudaStream_t xla_stream, B v, B p, I range, R out, CR coefficients,
    IR gathered_ranges, ScratchResult scratch, int64_t ctx_handle) {
  if (!ctx_handle)
    return ffi::Error::InvalidArgument(
        "distributed orthogonalize context is null");
  auto *ctx = reinterpret_cast<LorraxCusolverMpCtx *>(ctx_handle);
  if (!ctx->nccl_comm || !ctx->stream ||
      !ctx->active_subspace_ev_xla_in ||
      !ctx->active_subspace_ev_ctx_out || ctx->world_size < 1 || ctx->rank < 0 ||
      ctx->rank >= ctx->world_size || ctx->world_size > INT_MAX / 2)
    return ffi::Error::InvalidArgument(
        "distributed orthogonalize context geometry");
  if (range.dimensions()[0] != 2 || v.dimensions()[0] < 1 ||
      v.dimensions()[0] > INT_MAX || v.dimensions()[1] < 1 ||
      v.dimensions()[1] > INT_MAX || p.dimensions()[0] < 1 ||
      p.dimensions()[0] > INT_MAX)
    return ffi::Error::InvalidArgument(
        "distributed orthogonalize descriptor/dimension overflow");

  const int64_t cap = v.dimensions()[0], d = v.dimensions()[1];
  const int64_t b = p.dimensions()[0];
  if (p.dimensions()[1] != d || out->dimensions()[0] != b ||
      out->dimensions()[1] != d ||
      coefficients->dimensions()[0] != cap * b ||
      gathered_ranges->dimensions()[0] != int64_t(2) * ctx->world_size)
    return ffi::Error::InvalidArgument(
        "distributed orthogonalize buffer geometry");
  if (out->typed_data() != p.typed_data())
    return ffi::Error::InvalidArgument(
        "distributed orthogonalize requires declared P output alias");

  // XLA gives every rank the same static descriptor geometry.  Serialize
  // this context's active-subspace submissions locally; the fixed-size first
  // collective then makes the dynamic range agreement explicit globally.
  std::lock_guard<std::mutex> lock(ctx->active_subspace_mutex);
  auto &host_ranges = ctx->active_subspace_host_ranges;
  if (host_ranges.size() != static_cast<size_t>(2) *
                                static_cast<size_t>(ctx->world_size))
    return ffi::Error::InvalidArgument(
        "distributed orthogonalize host range buffer geometry");
  ContextStreamJoin streams(xla_stream, ctx);
  auto error = streams.start();
  if (!error.success())
    return error;
  NCCL(ncclAllGather(range.typed_data(), gathered_ranges->typed_data(), 2,
                     ncclInt32, ctx->nccl_comm, ctx->stream));
  CUDA(cudaMemcpyAsync(host_ranges.data(), gathered_ranges->typed_data(),
                       host_ranges.size() * sizeof(int32_t),
                       cudaMemcpyDeviceToHost, ctx->stream));
  CUDA(cudaStreamSynchronize(ctx->stream));

  const int64_t start = host_ranges[0], count = host_ranges[1];
  for (int rank = 1; rank < ctx->world_size; ++rank) {
    if (host_ranges[2 * rank] != start ||
        host_ranges[2 * rank + 1] != count)
      return ffi::Error::InvalidArgument(
          "distributed orthogonalize range differs across ranks");
  }
  if (start < 0 || count < 0 || start + count > cap)
    return ffi::Error::InvalidArgument(
        "distributed orthogonalize range outside capacity");
  if (!count)
    return streams.finish();

  BLAS(cublasSetStream(handle(), ctx->stream));
  BLAS(cublasSetWorkspace(handle(), scratch->typed_data(),
                          scratch->dimensions()[0]));
  const cuDoubleComplex one{1, 0}, zero{0, 0}, minus{-1, 0};
  auto *c = ptr(coefficients);
  const size_t coefficient_count = static_cast<size_t>(count * b);
  for (int pass = 0; pass < 2; ++pass) {
    BLAS(cublasZgemm(handle(), CUBLAS_OP_C, CUBLAS_OP_N,
                     static_cast<int>(count), static_cast<int>(b),
                     static_cast<int>(d), &one,
                     ptr(v) + start * d, static_cast<int>(d), ptr(out),
                     static_cast<int>(d), &zero, c,
                     static_cast<int>(count)));
    NCCL(ncclAllReduce(c, c, 2 * coefficient_count, ncclDouble, ncclSum,
                       ctx->nccl_comm, ctx->stream));
    BLAS(cublasZgemm(handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                     static_cast<int>(d), static_cast<int>(b),
                     static_cast<int>(count), &minus,
                     ptr(v) + start * d, static_cast<int>(d), c,
                     static_cast<int>(count), &one, ptr(out),
                     static_cast<int>(d)));
  }
  return streams.finish();
}

// Apply one distributed Gram-Schmidt correction after JAX has reduced the
// coefficient matrix.  The vectors are stored as rows by JAX, so cuBLAS sees
// V as a d-by-capacity column-major matrix and P as d-by-block.  C is already
// column-major with leading dimension capacity.  When next_coefficients is
// present, form the next process-local V^H P panel after updating P; the
// caller will reduce that panel before the second subtraction.
static ffi::Error subtract_host(cudaStream_t stream, B v, B p, B coefficients,
                                int cap, int d, int b, int start, int count,
                                R out, cuDoubleComplex *next_coefficients,
                                ScratchResult scratch) {
  if (!count) {
    if (next_coefficients)
      CUDA(cudaMemsetAsync(next_coefficients, 0,
                           int64_t(cap) * b * sizeof(cuDoubleComplex), stream));
    return ffi::Error::Success();
  }
  BLAS(cublasSetStream(handle(), stream));
  BLAS(cublasSetWorkspace(handle(), scratch->typed_data(),
                          scratch->dimensions()[0]));
  const cuDoubleComplex one{1, 0}, zero{0, 0}, minus{-1, 0};
  BLAS(cublasZgemm(handle(), CUBLAS_OP_N, CUBLAS_OP_N, d, b, count,
                   &minus, ptr(v) + int64_t(start) * d, d,
                   ptr(coefficients) + start, cap, &one, ptr(out), d));
  if (next_coefficients) {
    CUDA(cudaMemsetAsync(next_coefficients, 0,
                         int64_t(cap) * b * sizeof(cuDoubleComplex), stream));
    BLAS(cublasZgemm(handle(), CUBLAS_OP_C, CUBLAS_OP_N, count, b, d,
                     &one, ptr(v) + int64_t(start) * d, d, ptr(out), d,
                     &zero, next_coefficients + start, cap));
  }
  return ffi::Error::Success();
}

static ffi::Error validate_subtract_geometry(B v, B p, B coefficients,
                                             I range, R out) {
  if (range.dimensions()[0] != 2 || v.dimensions()[0] < 1 ||
      v.dimensions()[0] > INT_MAX || v.dimensions()[1] < 1 ||
      v.dimensions()[1] > INT_MAX || p.dimensions()[0] < 1 ||
      p.dimensions()[0] > INT_MAX)
    return ffi::Error::InvalidArgument(
        "active subspace subtract descriptor/dimension overflow");
  const int64_t cap = v.dimensions()[0], d = v.dimensions()[1];
  const int64_t b = p.dimensions()[0];
  if (p.dimensions()[1] != d || coefficients.dimensions()[0] != cap ||
      coefficients.dimensions()[1] != b || out->dimensions()[0] != b ||
      out->dimensions()[1] != d)
    return ffi::Error::InvalidArgument(
        "active subspace subtract buffer geometry");
  if (out->typed_data() != p.typed_data())
    return ffi::Error::InvalidArgument(
        "active subspace subtract requires declared P output alias");
  return ffi::Error::Success();
}

// In logical row-major notation, apply
// P -= C[start:start+count].T @ V[start:start+count].  cuBLAS sees the same
// buffers transposed: P and V are d-by-block/active column-major matrices.
static ffi::Error subtract(cudaStream_t stream, B v, B p, B coefficients,
                           I range, R out, ScratchResult scratch) {
  auto error = validate_subtract_geometry(v, p, coefficients, range, out);
  if (!error.success())
    return error;
  int q[2];
  CUDA(cudaMemcpyAsync(q, range.typed_data(), sizeof(q),
                       cudaMemcpyDeviceToHost, stream));
  CUDA(cudaStreamSynchronize(stream));
  const int64_t cap = v.dimensions()[0], d = v.dimensions()[1];
  const int64_t b = p.dimensions()[0], start = q[0], count = q[1];
  if (start < 0 || count < 0 || start + count > cap)
    return ffi::Error::InvalidArgument(
        "active subspace subtract range outside capacity");
  return subtract_host(stream, v, p, coefficients, static_cast<int>(cap),
                       static_cast<int>(d), static_cast<int>(b),
                       static_cast<int>(start), static_cast<int>(count), out,
                       nullptr, scratch);
}

// Perform the first correction and overwrite C with the process-local
// coefficients for the second CGS pass.  Clearing all of C is part of the
// contract: a subsequent full-capacity JAX reduction must never communicate
// stale entries outside the selected basis interval.
static ffi::Error subtract_gram(cudaStream_t stream, B v, B p,
                                B coefficients, I range, R out,
                                R next_coefficients,
                                ScratchResult scratch) {
  auto error = validate_subtract_geometry(v, p, coefficients, range, out);
  if (!error.success())
    return error;
  if (next_coefficients->dimensions()[0] != coefficients.dimensions()[0] ||
      next_coefficients->dimensions()[1] != coefficients.dimensions()[1] ||
      next_coefficients->typed_data() != coefficients.typed_data())
    return ffi::Error::InvalidArgument(
        "active subspace subtract/Gram requires declared C output alias");
  int q[2];
  CUDA(cudaMemcpyAsync(q, range.typed_data(), sizeof(q),
                       cudaMemcpyDeviceToHost, stream));
  CUDA(cudaStreamSynchronize(stream));
  const int64_t cap = v.dimensions()[0], d = v.dimensions()[1];
  const int64_t b = p.dimensions()[0], start = q[0], count = q[1];
  if (start < 0 || count < 0 || start + count > cap)
    return ffi::Error::InvalidArgument(
        "active subspace subtract/Gram range outside capacity");
  return subtract_host(stream, v, p, coefficients, static_cast<int>(cap),
                       static_cast<int>(d), static_cast<int>(b),
                       static_cast<int>(start), static_cast<int>(count), out,
                       ptr(next_coefficients), scratch);
}

static ffi::Error gram(cudaStream_t stream, B v, B p, I range, R out,
                       ScratchResult scratch) {
  if (range.dimensions()[0] != 2 || v.dimensions()[0] < 1 ||
      v.dimensions()[0] > INT_MAX || v.dimensions()[1] < 1 ||
      v.dimensions()[1] > INT_MAX || p.dimensions()[0] < 1 ||
      p.dimensions()[0] > INT_MAX)
    return ffi::Error::InvalidArgument("gram descriptor/dimension overflow");
  int q[2];
  CUDA(cudaMemcpyAsync(q, range.typed_data(), sizeof(q), cudaMemcpyDeviceToHost, stream));
  CUDA(cudaStreamSynchronize(stream));
  int cap = v.dimensions()[0], d = v.dimensions()[1], b = p.dimensions()[0];
  int start = q[0], count = q[1];
  if (p.dimensions()[1] != d || out->dimensions()[0] != cap ||
      out->dimensions()[1] != b || start < 0 || count < 0 || int64_t(start)+count > cap)
    return ffi::Error::InvalidArgument("gram buffer/range geometry");
  CUDA(cudaMemsetAsync(ptr(out), 0, int64_t(cap)*b*sizeof(cuDoubleComplex), stream));
  if (!count) return ffi::Error::Success();
  BLAS(cublasSetStream(handle(), stream));
  BLAS(cublasSetWorkspace(handle(), scratch->typed_data(), scratch->dimensions()[0]));
  const cuDoubleComplex one{1,0}, zero{0,0};
  BLAS(cublasZgemm(handle(), CUBLAS_OP_C, CUBLAS_OP_N, count, b, d, &one,
                  ptr(v)+int64_t(start)*d, d, ptr(p), d, &zero, ptr(out)+start, cap));
  return ffi::Error::Success();
}
// Copy an already validated insertion range. Only the selected rows are
// touched; the allocated capacity beyond them remains unchanged.
static ffi::Error store_host(cudaStream_t stream, B p, B hp, int64_t start,
                             int64_t count, int64_t d, R out, R hout) {
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
  return store_host(stream, p, hp, start, count, d, out, hout);
}

// Insert p/hp and update the corresponding projected row and column of h.
// Davidson always performs these operations as a pair. Reading [start,count]
// once preserves runtime range validation while avoiding the second host wait.
// A zero count is an exact no-op, including when start is zero.
static ffi::Error store_project(cudaStream_t stream, B v, B hv, B p, B hp, B h,
                                I range, R out, R hout, R h_out,
                                ScratchResult scratch) {
  if (range.dimensions()[0] != 2 || v.dimensions()[0] < 1 ||
      v.dimensions()[0] > INT_MAX || v.dimensions()[1] < 1 ||
      v.dimensions()[1] > INT_MAX)
    return ffi::Error::InvalidArgument(
        "active store/project descriptor/dimension overflow");

  int64_t cap = v.dimensions()[0], d = v.dimensions()[1];
  if (hv.dimensions()[0] != cap || hv.dimensions()[1] != d ||
      p.dimensions()[0] != hp.dimensions()[0] ||
      p.dimensions()[1] != hp.dimensions()[1] ||
      p.dimensions()[1] != d || h.dimensions()[0] != cap ||
      h.dimensions()[1] != cap || out->dimensions()[0] != cap ||
      out->dimensions()[1] != d || hout->dimensions()[0] != cap ||
      hout->dimensions()[1] != d || h_out->dimensions()[0] != cap ||
      h_out->dimensions()[1] != cap)
    return ffi::Error::InvalidArgument("active store/project buffer geometry");
  if (out->typed_data() != v.typed_data() ||
      hout->typed_data() != hv.typed_data() ||
      h_out->typed_data() != h.typed_data())
    return ffi::Error::InvalidArgument(
        "active store/project requires declared input/output aliases");

  int q[2];
  CUDA(cudaMemcpyAsync(q, range.typed_data(), sizeof(q),
                       cudaMemcpyDeviceToHost, stream));
  CUDA(cudaStreamSynchronize(stream));
  int64_t start = q[0], count = q[1];
  if (start < 0 || count < 0 || count > p.dimensions()[0] ||
      start + count > cap)
    return ffi::Error::InvalidArgument(
        "active store/project range outside capacity");
  if (!count)
    return ffi::Error::Success();

  auto error = store_host(stream, p, hp, start, count, d, out, hout);
  if (!error.success())
    return error;
  return project_host(stream, v, hv, static_cast<int>(cap),
                      static_cast<int>(d), static_cast<int>(start + count),
                      static_cast<int>(start), static_cast<int>(count), h_out,
                      scratch);
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ActiveSubspaceDistributedOrthoFfi, distributed_orthogonalize,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<B>()
        .Arg<B>()
        .Arg<I>()
        .Ret<B>()
        .Ret<C>()
        .Ret<I>()
        .Ret<Scratch>()
        .Attr<int64_t>("ctx_handle"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ActiveSubspaceSubtractFfi, subtract,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<B>()
        .Arg<B>()
        .Arg<B>()
        .Arg<I>()
        .Ret<B>()
        .Ret<Scratch>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ActiveSubspaceSubtractGramFfi, subtract_gram,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<B>()
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(ActiveSubspaceStoreProjectFfi, store_project,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<B>()
                                  .Arg<B>()
                                  .Arg<B>()
                                  .Arg<B>()
                                  .Arg<B>()
                                  .Arg<I>()
                                  .Ret<B>()
                                  .Ret<B>()
                                  .Ret<B>()
                                  .Ret<Scratch>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(ActiveSubspaceGramFfi, gram,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Arg<B>().Arg<B>().Arg<I>().Ret<B>().Ret<Scratch>());
