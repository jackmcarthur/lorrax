// platform_seam.h — THE three compile-time seams that let the parallel-HDF5
// FFI core compile, unmodified, into both platform libraries.
//
//   liblorrax_ffi.so       (CUDA)  — LORRAX_FFI_NO_CUDA undefined
//   liblorrax_ffi_host.so  (cpu)   — -DLORRAX_FFI_NO_CUDA=1
//
// The collective MPI-IO core itself (hyperslab arithmetic, H5Dread/H5Dwrite,
// the FIFO writer thread, the empty-selection / valid_shape clipping) is
// hardware-agnostic and byte-identical on both platforms.  Only these three
// things differ, and they all live here so the two TUs that need them
// (read_ffi.cc, write_ffi.cc) cannot drift apart:
//
//   1. HANDLER BINDING — the CUDA handlers take XLA's platform stream as a
//      leading Ctx and are named ``Phdf*Ffi``; the host handlers take no
//      stream (XLA's CPU runtime calls the handler on the calling thread with
//      host buffers) and are named ``Phdf*HostFfi`` so both .so's can be
//      dlopen'd RTLD_GLOBAL in one process.  ffi_loader.py registers the CUDA
//      names under platform="CUDA" and the Host names under platform="cpu",
//      both against the SAME jax.ffi target strings, so every ffi_call site
//      stays platform-agnostic (the split jaxlib uses for cpu-lapack vs
//      cuda-cusolver).
//   2. INDEX COPY-IN — the small (ndim × int64) offset / valid_shape / count
//      control buffers are device-resident on CUDA (cudaMemcpy D2H) and
//      already host-resident on cpu (a plain read).
//   3. DEVICE STAGING — the bulk payload.  CUDA stages through the ctx's
//      page-locked ``pinned_buf`` with an async copy on ``ctx->stream``
//      signalled by a pooled event; on cpu the XLA buffer IS host memory, so
//      the read tail is a std::memcpy and the WRITE path hands H5Dwrite the
//      XLA buffer pointer directly (no staging copy, no second 12 GB).
//      The staging seam is per-TU (read tail vs write source) and lives in
//      read_ffi.cc / write_ffi.cc respectively; seams 1 and 2 are here.

#pragma once

#include <cstring>
#include <string>

#ifndef LORRAX_FFI_NO_CUDA
#include <cuda_runtime.h>
#endif

// ---- Seam 1a: the leading dispatch/Impl parameter for the platform stream.
// The CUDA handlers bind ``.Ctx<PlatformStream<cudaStream_t>>()`` and thread
// the stream through to the H2D staging; the host handlers take no stream.
#ifdef LORRAX_FFI_NO_CUDA
#define LRX_STREAM_PARAM
#define LRX_STREAM_ARG
#else
#define LRX_STREAM_PARAM cudaStream_t xla_stream,
#define LRX_STREAM_ARG   xla_stream,
#endif

// ---- Seam 1b: handler symbol name + the stream Ctx in the Ffi::Bind chain.
#ifdef LORRAX_FFI_NO_CUDA
#  define LRX_PHDF_HANDLER(name) name##HostFfi
#  define LRX_PHDF_STREAM_CTX
#else
#  define LRX_PHDF_HANDLER(name) name##Ffi
#  define LRX_PHDF_STREAM_CTX \
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
#endif

namespace lorrax_ffi::phdf5 {

// ---- Seam 2: copy a small (N × int64) index buffer out of an FFI input into
// host ``dst``.  Host: the buffer is already host-resident.  CUDA: D2H copy.
// Returns false on failure and fills ``err`` (CUDA path only).
static inline bool copy_index_to_host(
    void* dst, const void* src, size_t nbytes, std::string* err)
{
#ifdef LORRAX_FFI_NO_CUDA
    std::memcpy(dst, src, nbytes);
    (void)err;
    return true;
#else
    cudaError_t ce = cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) {
        if (err) *err = cudaGetErrorString(ce);
        return false;
    }
    return true;
#endif
}

}  // namespace lorrax_ffi::phdf5
