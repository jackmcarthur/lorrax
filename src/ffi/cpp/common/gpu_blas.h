// gpu_blas.h — the dense-BLAS vocabulary of LORRAX's device legs (gpu_runtime.h's
// library sibling, the "gpu_blas.h" it names).
//
// A family written against `lrx::gpu::blas::` compiles unchanged into the CUDA leg
// (cuBLAS) and the ROCM leg (rocBLAS).  Only what a ported family calls is here:
// handle lifetime, stream and workspace binding, and complex128 GEMM.
//
//   CUDA  every name is the cuBLAS one it replaced (types are aliases, functions inline
//         forwards), so a CUDA translation unit compiles to the calls it made before.
//         cuBLAS is linked (CMake CUBLAS_LIBRARY), as before.
//   ROCM  rocBLAS is NOT linked: its C ABI is spelled here (handles opaque pointers,
//         enums C ints, rocblas_int int32 in the LP64 build) and resolved once from the
//         librocblas.so.5 this process already mapped (the JAX ROCm plugin's hipBLAS /
//         hipSOLVER pull it in), so the process keeps ONE ROCm -- the rule
//         rocsolver/batched_eigh_ffi.cc set.  A process without it refuses by
//         GATE rocblas-mapped (unavailable() carries the message; every entry point
//         then returns kUnavailable instead of calling through a null pointer).
//
// Thin by design: no policy.  Handles are created and kept by the caller (thread_local in
// the active-subspace family, as on CUDA); on ROCM destroy() deliberately leaves the handle
// to process exit (no teardown-order hazard against the HIP runtime at thread exit).
#pragma once

#include "gpu_runtime.h"

#include <cstddef>

#if defined(LRX_LEG_ROCM)
#  include <dlfcn.h>
#  include <string>
#else
#  include <cublas_v2.h>
#endif

namespace lrx::gpu::blas {

#if defined(LRX_LEG_ROCM)

#define LRX_BLAS_API "rocblas"
using Handle  = void*;                  // rocblas_handle
using Status  = int;                    // rocblas_status
using Op      = int;                    // rocblas_operation
struct Complex { double x, y; };        // rocblas_double_complex's layout
inline constexpr Status kSuccess     = 0;      // rocblas_status_success
inline constexpr Status kUnavailable = 1000;   // not a rocblas_status: library not mapped
inline constexpr Op     kOpN = 111;            // rocblas_operation_none
inline constexpr Op     kOpC = 113;            // rocblas_operation_conjugate_transpose

namespace detail {

// One resolution per process.  `library` stays open (RTLD_NOLOAD took a reference to an
// already-mapped object), so the bound pointers live as long as the process.
struct Api {
    Status (*create)(Handle*) = nullptr;
    Status (*set_stream)(Handle, Stream) = nullptr;
    Status (*set_workspace)(Handle, void*, std::size_t) = nullptr;
    Status (*start_query)(Handle) = nullptr;
    Status (*stop_query)(Handle, std::size_t*) = nullptr;
    Status (*zgemm)(Handle, Op, Op, int, int, int, const Complex*, const Complex*, int,
                    const Complex*, int, const Complex*, Complex*, int) = nullptr;
    void* library = nullptr;
    std::string err;
};

inline const Api& api() {
    static const Api a = [] {
        Api r;
        constexpr const char* kSoname = "librocblas.so.5";
        r.library = dlopen(kSoname, RTLD_NOW | RTLD_NOLOAD);
        if (r.library == nullptr) {
            r.err = std::string("GATE rocblas-mapped: ") + kSoname + " is not mapped in this "
                    "process; want the copy the JAX ROCm plugin maps (one ROCm per process); "
                    "fix: initialize the JAX ROCm backend before the first call";
            return r;
        }
        auto need = [&](const char* name) {
            void* p = dlsym(r.library, name);
            if (p == nullptr) r.err += std::string(r.err.empty() ? "" : ", ") + name;
            return p;
        };
#define LRX_BLAS_BIND(field, name) r.field = reinterpret_cast<decltype(r.field)>(need(name))
        LRX_BLAS_BIND(create, "rocblas_create_handle");
        LRX_BLAS_BIND(set_stream, "rocblas_set_stream");
        LRX_BLAS_BIND(set_workspace, "rocblas_set_workspace");
        LRX_BLAS_BIND(start_query, "rocblas_start_device_memory_size_query");
        LRX_BLAS_BIND(stop_query, "rocblas_stop_device_memory_size_query");
        LRX_BLAS_BIND(zgemm, "rocblas_zgemm");
#undef LRX_BLAS_BIND
        if (!r.err.empty()) r.err = "GATE rocblas-symbols: dlsym found no " + r.err + " in " + kSoname;
        return r;
    }();
    return a;
}

}  // namespace detail

// nullptr when the library is usable, else the GATE message.
inline const char* unavailable() {
    const detail::Api& a = detail::api();
    return a.err.empty() ? nullptr : a.err.c_str();
}
inline Status create(Handle* h) {
    if (unavailable()) { *h = nullptr; return kUnavailable; }
    return detail::api().create(h);
}
inline Status destroy(Handle) { return kSuccess; }   // kept to process exit (header note)
inline Status set_stream(Handle h, Stream s) {
    return unavailable() ? kUnavailable : detail::api().set_stream(h, s);
}
inline Status set_workspace(Handle h, void* p, std::size_t bytes) {
    return unavailable() ? kUnavailable : detail::api().set_workspace(h, p, bytes);
}
inline Status zgemm(Handle h, Op ta, Op tb, int m, int n, int k, const Complex* alpha,
                    const Complex* a, int lda, const Complex* b, int ldb,
                    const Complex* beta, Complex* c, int ldc) {
    return unavailable() ? kUnavailable
                         : detail::api().zgemm(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#else  // CUDA leg: the cuBLAS names, unchanged

#define LRX_BLAS_API "cublas"
using Handle  = cublasHandle_t;
using Status  = cublasStatus_t;
using Op      = cublasOperation_t;
using Complex = cuDoubleComplex;
inline constexpr Status kSuccess = CUBLAS_STATUS_SUCCESS;
inline constexpr Op     kOpN = CUBLAS_OP_N;
inline constexpr Op     kOpC = CUBLAS_OP_C;

inline constexpr const char* unavailable() { return nullptr; }   // linked: always usable
inline Status create(Handle* h)  { return cublasCreate(h); }
inline Status destroy(Handle h)  { return cublasDestroy(h); }
inline Status set_stream(Handle h, Stream s) { return cublasSetStream(h, s); }
inline Status set_workspace(Handle h, void* p, std::size_t bytes) {
    return cublasSetWorkspace(h, p, bytes);
}
inline Status zgemm(Handle h, Op ta, Op tb, int m, int n, int k, const Complex* alpha,
                    const Complex* a, int lda, const Complex* b, int ldb,
                    const Complex* beta, Complex* c, int ldc) {
    return cublasZgemm(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif

}  // namespace lrx::gpu::blas
