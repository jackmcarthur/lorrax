// lrx_async_gather.h -- the shared asynchronous gather layer of the NVRTC
// kernel families (cufft/kconv_mathdx_cuda_ffi.cc; the k-box and plane FFTs).
//
// The device source is embedded as a string and handed to NVRTC as the header
// "lrx_async_gather.cuh" (a sealed bundle ships no source tree, so the header
// cannot be read from disk at run time).  A kernel that includes it gets:
//
//   lrx_async::cell16(dst_smem, src_gmem, valid)
//       one 16-byte cell global -> shared; !valid zero-fills the cell without
//       reading global memory.  sm_80+: cp.async.cg (L2 only, bypassing L1),
//       so the copy is in flight while the block computes; older archs: a
//       plain load and store (same result, no overlap).
//   lrx_async::copy<B>(dst_smem, src_gmem)
//       one B-byte (4, 8, 16) copy global -> shared, L1-cached (cp.async.ca).
//   lrx_async::commit()           close the current group of issued copies
//   lrx_async::wait_all()         wait for every issued copy of this thread
//   lrx_async::wait_prior<N>()    wait until at most N groups are pending
//   lrx_async::gather(dst, src, idx, n)
//       block-cooperative indexed gather dst[j] = idx[j] >= 0 ? src[idx[j]] : 0,
//       j in [0, n), 16-byte elements; issues and commits, does not wait.
//
// The usual double-buffered use: issue the next tile's gather into a staging
// buffer right after the current tile's staging has been consumed, compute,
// then wait_all() + __syncthreads() before consuming it.  Every copy is
// followed by a __syncthreads() before another thread reads the cell: wait_*
// covers only the calling thread's own copies.
//
// Host side: a caller sizes the staging from device attributes (opt-in shared
// memory per block, shared memory per SM) and keeps the plain in-place gather
// when staging does not fit; there are no per-size constants here.
#pragma once

namespace lorrax_ffi::async_gather {

inline constexpr const char* kHeaderName = "lrx_async_gather.cuh";

inline constexpr const char* kHeaderSrc = R"__lrx__(
#pragma once
namespace lrx_async {

__device__ __forceinline__ void cell16(void* dst, const void* src, bool valid) {
#if __CUDA_ARCH__ >= 800
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s), "l"(src), "r"(valid ? 16 : 0) : "memory");
#else
    double2 v = valid ? *static_cast<const double2*>(src) : make_double2(0.0, 0.0);
    *static_cast<double2*>(dst) = v;
#endif
}

// One BYTES-sized copy (4, 8 or 16) global -> shared, cached in L1 (cp.async.ca);
// the strided/tile loaders of the k-box stage use it.
template <int BYTES>
__device__ __forceinline__ void copy(void* dst, const void* src) {
    static_assert(BYTES == 4 || BYTES == 8 || BYTES == 16, "cp.async copies 4, 8 or 16 bytes");
#if __CUDA_ARCH__ >= 800
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(s), "l"(src), "n"(BYTES) : "memory");
#else
    if constexpr (BYTES == 16) *static_cast<double2*>(dst) = *static_cast<const double2*>(src);
    else if constexpr (BYTES == 8) *static_cast<double*>(dst) = *static_cast<const double*>(src);
    else *static_cast<float*>(dst) = *static_cast<const float*>(src);
#endif
}

__device__ __forceinline__ void commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void wait_all() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
}

template <int N>
__device__ __forceinline__ void wait_prior() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}

// dst[j] = idx[j] >= 0 ? src[idx[j]] : 0 for j in [0, n); T is 16 bytes.
template <class T>
__device__ __forceinline__ void gather(T* dst, const T* src, const int* idx, int n) {
    static_assert(sizeof(T) == 16, "lrx_async::gather moves 16-byte cells");
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const int k = idx[j];
        cell16(dst + j, src + (k >= 0 ? k : 0), k >= 0);
    }
    commit();
}

}  // namespace lrx_async
)__lrx__";

}  // namespace lorrax_ffi::async_gather
