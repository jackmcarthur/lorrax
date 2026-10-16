// gpu_blas.h — the classic-BLAS vocabulary of LORRAX's device legs (gpu_runtime.h's
// "library handles get their own header" rule, first used by cublas/local_active_gemm_ffi.cc).
//
// One spelling per operation, column-major, both operands untransposed: the handle, its
// stream and workspace binding, the strided-batched GEMM and the vector scale, for float64
// and complex128 (std::complex<double>, layout-equal to cuDoubleComplex and
// rocblas_double_complex).
//
//   CUDA  cuBLAS (cublas_v2.h, linked).  Every wrapper is the inline call it replaced, with
//         the same arguments and the same reinterpret casts, so the CUDA leg's code is
//         unchanged.
//   ROCM  rocBLAS, resolved from the copy this process already mapped (the JAX ROCm plugin
//         maps librocblas.so.5 from the pip wheel; no header, no link: one ROCm per process,
//         as rocsolver/batched_eigh_ffi.cc).  The C ABI is spelled here: handles are opaque
//         pointers, enums C ints, rocblas_int int32 (LP64 build), rocblas_stride int64.  A
//         process without the mapped library refuses by GATE rocblas-mapped (unavailable()).
//         Each handle is created with atomics NOT allowed, so a GEMM is deterministic run
//         to run, as cuBLAS's default GEMM is.
//
// Thin by design: no state beyond the ROCm symbol table, no policy.
#pragma once
#include "gpu_runtime.h"

#include <complex>
#include <cstddef>
#include <cstdint>

#if defined(LRX_LEG_ROCM)
#  include <dlfcn.h>
#  include <string>
#  define LRX_BLAS_NAME "rocBLAS"
#else
#  include <cublas_v2.h>
#  define LRX_BLAS_NAME "cuBLAS"
#endif

namespace lrx::gpu::blas {

using Z = std::complex<double>;

#if defined(LRX_LEG_ROCM)
using Handle = void*;                       // rocblas_handle
using Status = int;                         // rocblas_status
inline constexpr Status kSuccess = 0;       // rocblas_status_success
inline constexpr Status kNotInitialized = 6;  // rocblas_status_internal_error (no handle)

namespace detail {
constexpr int kOpNone = 111;                // rocblas_operation_none
constexpr int kAtomicsNotAllowed = 0;       // rocblas_atomics_not_allowed
struct Api {
    Status (*create)(Handle*) = nullptr;
    Status (*destroy)(Handle) = nullptr;
    Status (*set_stream)(Handle, Stream) = nullptr;
    Status (*set_workspace)(Handle, void*, std::size_t) = nullptr;
    Status (*set_atomics)(Handle, int) = nullptr;
    Status (*dgemm_sb)(Handle, int, int, int32_t, int32_t, int32_t, const double*,
                       const double*, int32_t, int64_t, const double*, int32_t, int64_t,
                       const double*, double*, int32_t, int64_t, int32_t) = nullptr;
    Status (*zgemm_sb)(Handle, int, int, int32_t, int32_t, int32_t, const void*, const void*,
                       int32_t, int64_t, const void*, int32_t, int64_t, const void*, void*,
                       int32_t, int64_t, int32_t) = nullptr;
    Status (*dscal)(Handle, int32_t, const double*, double*, int32_t) = nullptr;
    Status (*zscal)(Handle, int32_t, const void*, void*, int32_t) = nullptr;
    std::string err;                        // empty: every symbol bound
};
inline const Api& api() {
    static const Api a = [] {
        Api r;
        constexpr const char* kSoname = "librocblas.so.5";
        void* h = dlopen(kSoname, RTLD_NOW | RTLD_NOLOAD);
        if (h == nullptr) {
            r.err = std::string("GATE rocblas-mapped: ") + kSoname + " is not mapped in this "
                    "process; want the copy the JAX ROCm plugin maps (one ROCm per process); "
                    "fix: import jax with the rocm plugin before the first call";
            return r;
        }
        std::string missing;
        auto need = [&](const char* name) {
            void* p = dlsym(h, name);
            if (p == nullptr) missing += std::string(missing.empty() ? "" : ", ") + name;
            return p;
        };
#define LRX_BLAS_BIND(field, name) r.field = reinterpret_cast<decltype(r.field)>(need(name))
        LRX_BLAS_BIND(create, "rocblas_create_handle");
        LRX_BLAS_BIND(destroy, "rocblas_destroy_handle");
        LRX_BLAS_BIND(set_stream, "rocblas_set_stream");
        LRX_BLAS_BIND(set_workspace, "rocblas_set_workspace");
        LRX_BLAS_BIND(set_atomics, "rocblas_set_atomics_mode");
        LRX_BLAS_BIND(dgemm_sb, "rocblas_dgemm_strided_batched");
        LRX_BLAS_BIND(zgemm_sb, "rocblas_zgemm_strided_batched");
        LRX_BLAS_BIND(dscal, "rocblas_dscal");
        LRX_BLAS_BIND(zscal, "rocblas_zscal");
#undef LRX_BLAS_BIND
        if (!missing.empty())
            r.err = "GATE rocblas-symbols: dlsym found no " + missing + " in " + kSoname;
        return r;
    }();
    return a;
}
}  // namespace detail

// Why the library is unusable on this leg (a GATE message), or nullptr when it is usable.
inline const char* unavailable() {
    const detail::Api& a = detail::api();
    return a.err.empty() ? nullptr : a.err.c_str();
}
inline Status create(Handle* h) {
    if (unavailable() != nullptr) return kNotInitialized;
    const Status s = detail::api().create(h);
    if (s != kSuccess) return s;
    return detail::api().set_atomics(*h, detail::kAtomicsNotAllowed);
}
inline Status destroy(Handle h) {
    return unavailable() != nullptr ? kNotInitialized : detail::api().destroy(h);
}
inline Status set_stream(Handle h, Stream s) { return detail::api().set_stream(h, s); }
inline Status set_workspace(Handle h, void* p, std::size_t n) {
    return detail::api().set_workspace(h, p, n);
}
inline Status gemm_strided_batched(Handle h, int m, int n, int k, const double* alpha,
                                   const double* a, int lda, int64_t sa, const double* b,
                                   int ldb, int64_t sb, const double* beta, double* c, int ldc,
                                   int64_t sc, int batch) {
    return detail::api().dgemm_sb(h, detail::kOpNone, detail::kOpNone, m, n, k, alpha, a, lda,
                                  sa, b, ldb, sb, beta, c, ldc, sc, batch);
}
inline Status gemm_strided_batched(Handle h, int m, int n, int k, const Z* alpha, const Z* a,
                                   int lda, int64_t sa, const Z* b, int ldb, int64_t sb,
                                   const Z* beta, Z* c, int ldc, int64_t sc, int batch) {
    return detail::api().zgemm_sb(h, detail::kOpNone, detail::kOpNone, m, n, k, alpha, a, lda,
                                  sa, b, ldb, sb, beta, c, ldc, sc, batch);
}
inline Status scal(Handle h, int n, const double* alpha, double* x, int incx) {
    return detail::api().dscal(h, n, alpha, x, incx);
}
inline Status scal(Handle h, int n, const Z* alpha, Z* x, int incx) {
    return detail::api().zscal(h, n, alpha, x, incx);
}

#else  // CUDA: cuBLAS, the calls the CUDA handlers made before this header existed.
using Handle = cublasHandle_t;
using Status = cublasStatus_t;
inline constexpr Status kSuccess = CUBLAS_STATUS_SUCCESS;
inline constexpr Status kNotInitialized = CUBLAS_STATUS_NOT_INITIALIZED;

inline constexpr const char* unavailable() { return nullptr; }   // linked: always usable
inline Status create(Handle* h) { return cublasCreate(h); }
inline Status destroy(Handle h) { return cublasDestroy(h); }
inline Status set_stream(Handle h, Stream s) { return cublasSetStream(h, s); }
inline Status set_workspace(Handle h, void* p, std::size_t n) {
    return cublasSetWorkspace(h, p, n);
}
inline Status gemm_strided_batched(Handle h, int m, int n, int k, const double* alpha,
                                   const double* a, int lda, int64_t sa, const double* b,
                                   int ldb, int64_t sb, const double* beta, double* c, int ldc,
                                   int64_t sc, int batch) {
    return cublasDgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a, lda, sa,
                                     b, ldb, sb, beta, c, ldc, sc, batch);
}
inline Status gemm_strided_batched(Handle h, int m, int n, int k, const Z* alpha, const Z* a,
                                   int lda, int64_t sa, const Z* b, int ldb, int64_t sb,
                                   const Z* beta, Z* c, int ldc, int64_t sc, int batch) {
    return cublasZgemmStridedBatched(
        h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, reinterpret_cast<const cuDoubleComplex*>(alpha),
        reinterpret_cast<const cuDoubleComplex*>(a), lda, sa,
        reinterpret_cast<const cuDoubleComplex*>(b), ldb, sb,
        reinterpret_cast<const cuDoubleComplex*>(beta), reinterpret_cast<cuDoubleComplex*>(c),
        ldc, sc, batch);
}
inline Status scal(Handle h, int n, const double* alpha, double* x, int incx) {
    return cublasDscal(h, n, alpha, x, incx);
}
inline Status scal(Handle h, int n, const Z* alpha, Z* x, int incx) {
    return cublasZscal(h, n, reinterpret_cast<const cuDoubleComplex*>(alpha),
                       reinterpret_cast<cuDoubleComplex*>(x), incx);
}
#endif

}  // namespace lrx::gpu::blas
