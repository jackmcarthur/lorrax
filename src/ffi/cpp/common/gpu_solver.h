// gpu_solver.h — the dense-solver vocabulary of LORRAX's device legs (gpu_blas.h's
// sibling): the one-matrix complex128 Hermitian eigensolver with eigenvectors, LOWER
// triangle (column-major sense), and caller-owned workspace.
//
//   CUDA  cuSOLVERDn: Handle = cusolverDnHandle_t; zheevd_buffer_size / zheevd forward to
//         cusolverDnZheevd_bufferSize / cusolverDnZheevd (CUSOLVER_EIG_MODE_VECTOR,
//         CUBLAS_FILL_MODE_LOWER), the workspace counted in complex128 elements -- the calls
//         active_subspace/active_eigh.cc made before, unchanged.
//   ROCM  rocSOLVER, resolved from the mapped librocsolver.so.0 (not linked; the rule of
//         gpu_blas.h and rocsolver/batched_eigh_ffi.cc).  rocSOLVER takes its workspace from
//         the rocBLAS handle (Handle = the rocBLAS handle) and needs one more device array, E
//         (n doubles, the tridiagonal off-diagonal).  To keep cuSOLVER's contract -- ONE
//         caller-owned buffer of `lwork` complex128 elements -- zheevd_buffer_size reports
//         E plus the rocBLAS size-query bytes plus alignment slack in complex128 elements, and
//         zheevd carves E and the rocBLAS workspace out of that buffer and binds the latter
//         with rocblas_set_workspace (no allocation inside rocSOLVER; stream-ordered).
//
// Thin by design: no policy, no state beyond the one-time symbol resolution.
#pragma once

#include "gpu_runtime.h"

#include <cstddef>
#include <cstdint>

#if defined(LRX_LEG_ROCM)
#  include "gpu_blas.h"
#else
#  include <cusolverDn.h>
#endif

namespace lrx::gpu::solver {

#if defined(LRX_LEG_ROCM)

#define LRX_SOLVER_API "rocSOLVER"
using Handle  = blas::Handle;
using Status  = blas::Status;
using Complex = blas::Complex;
inline constexpr Status kSuccess     = blas::kSuccess;
inline constexpr Status kUnavailable = blas::kUnavailable;

namespace detail {
inline constexpr int kEvectOriginal = 211;   // rocblas_evect_original
inline constexpr int kFillLower     = 122;   // rocblas_fill_lower
inline constexpr std::size_t kAlign = 256;   // E and the rocBLAS workspace start on 256 B

using Zheevd = Status (*)(Handle, int, int, int, Complex*, int, double*, double*, int*);
struct Api {
    Zheevd zheevd = nullptr;
    std::string err;
};
inline const Api& api() {
    static const Api a = [] {
        Api r;
        if (const char* why = blas::unavailable()) { r.err = why; return r; }
        constexpr const char* kSoname = "librocsolver.so.0";
        void* h = dlopen(kSoname, RTLD_NOW | RTLD_NOLOAD);
        if (h == nullptr) {
            r.err = std::string("GATE rocsolver-mapped: ") + kSoname + " is not mapped in this "
                    "process; want the copy the JAX ROCm plugin maps (one ROCm per process); "
                    "fix: initialize the JAX ROCm backend before the first call";
            return r;
        }
        r.zheevd = reinterpret_cast<Zheevd>(dlsym(h, "rocsolver_zheevd"));
        if (r.zheevd == nullptr)
            r.err = std::string("GATE rocsolver-symbols: dlsym found no rocsolver_zheevd in ") + kSoname;
        return r;
    }();
    return a;
}
inline std::uintptr_t align_up(std::uintptr_t p) { return (p + kAlign - 1) & ~(kAlign - 1); }
}  // namespace detail

inline const char* unavailable() {
    const detail::Api& a = detail::api();
    return a.err.empty() ? nullptr : a.err.c_str();
}
inline Status create(Handle* h) {
    if (unavailable()) { *h = nullptr; return kUnavailable; }
    return blas::create(h);
}
inline Status destroy(Handle h) { return blas::destroy(h); }
inline Status set_stream(Handle h, Stream s) {
    return unavailable() ? kUnavailable : blas::set_stream(h, s);
}

// Workspace, in complex128 elements, for an n x n problem with leading dimension lda.
// rocBLAS size-query mode runs only rocSOLVER's argument checks and size computation, which
// reject null pointers for n > 0: the query passes stand-in addresses that are never
// dereferenced (nothing executes on the device in query mode).
inline Status zheevd_buffer_size(Handle h, int n, Complex* a, int lda, double* w, int* elements) {
    if (unavailable()) return kUnavailable;
    const blas::detail::Api& B = blas::detail::api();
    auto* stand_in = reinterpret_cast<Complex*>(detail::kAlign);
    double* e = reinterpret_cast<double*>(detail::kAlign);
    int* info = reinterpret_cast<int*>(detail::kAlign);
    Status st = B.start_query(h);
    if (st != kSuccess) return st;
    const Status q = detail::api().zheevd(h, detail::kEvectOriginal, detail::kFillLower, n,
                                          a ? a : stand_in, lda, w ? w : e, e, info);
    std::size_t bytes = 0;
    st = B.stop_query(h, &bytes);
    if (st != kSuccess) return st;
    // In query mode the solver answers rocblas_status_size_increased/unchanged (9/10) or
    // success; anything else is a genuine argument error.
    if (q != kSuccess && q != 9 && q != 10) return q;
    const std::size_t total = detail::kAlign                                     // start alignment
        + detail::align_up(static_cast<std::size_t>(n) * sizeof(double))         // E
        + bytes;                                                                 // rocBLAS
    const std::size_t count = (total + sizeof(Complex) - 1) / sizeof(Complex);
    if (count > static_cast<std::size_t>(INT32_MAX)) return 3;                   // invalid_size
    *elements = static_cast<int>(count);
    return kSuccess;
}

// Eigenvalues ascending in w[0:n], eigenvectors in place of a; info on the device.
inline Status zheevd(Handle h, int n, Complex* a, int lda, double* w, Complex* work, int lwork,
                     int* info) {
    if (unavailable()) return kUnavailable;
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(work);
    const std::uintptr_t end = base + static_cast<std::uintptr_t>(lwork) * sizeof(Complex);
    const std::uintptr_t e = detail::align_up(base);
    const std::uintptr_t ws = detail::align_up(e + static_cast<std::size_t>(n) * sizeof(double));
    if (ws > end) return 5;                                                      // memory_error
    Status st = blas::set_workspace(h, reinterpret_cast<void*>(ws), end - ws);
    if (st != kSuccess) return st;
    return detail::api().zheevd(h, detail::kEvectOriginal, detail::kFillLower, n, a, lda, w,
                                reinterpret_cast<double*>(e), info);
}

#else  // CUDA leg: the cuSOLVERDn calls, unchanged

#define LRX_SOLVER_API "cuSOLVER"
using Handle  = cusolverDnHandle_t;
using Status  = cusolverStatus_t;
using Complex = cuDoubleComplex;
inline constexpr Status kSuccess = CUSOLVER_STATUS_SUCCESS;

inline constexpr const char* unavailable() { return nullptr; }   // linked: always usable
inline Status create(Handle* h)  { return cusolverDnCreate(h); }
inline Status destroy(Handle h)  { return cusolverDnDestroy(h); }
inline Status set_stream(Handle h, Stream s) { return cusolverDnSetStream(h, s); }
inline Status zheevd_buffer_size(Handle h, int n, Complex* a, int lda, double* w, int* elements) {
    return cusolverDnZheevd_bufferSize(h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, a,
                                       lda, w, elements);
}
inline Status zheevd(Handle h, int n, Complex* a, int lda, double* w, Complex* work, int lwork,
                     int* info) {
    return cusolverDnZheevd(h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, a, lda, w,
                            work, lwork, info);
}

#endif

}  // namespace lrx::gpu::solver
