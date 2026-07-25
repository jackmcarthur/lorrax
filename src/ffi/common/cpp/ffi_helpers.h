// ffi_helpers.h — small error-propagation helpers shared by LORRAX FFI
// handlers.  Intentionally minimal — only the macros that make handler
// code noticeably shorter.  Extend when a second handler actually needs
// something, not speculatively.

#pragma once

#include <cstdio>
#include <sstream>
#include <string>

// LORRAX_FFI_NO_CUDA is defined by the host-platform build (see
// common/cpp/host/CMakeLists.txt).  The CUDA-free host TUs (phdf5 read,
// future host handlers) include this header for FFI_RETURN_IF_ERROR and the
// error-formatting helpers, but must not pull in <cuda_runtime.h> or the
// CUDA-only macros.  The device build leaves the flag undefined and gets the
// full set unchanged.
#ifndef LORRAX_FFI_NO_CUDA
#include <cuda_runtime.h>
#endif

#include "xla/ffi/api/ffi.h"

namespace lorrax_ffi {

namespace ffi = ::xla::ffi;

#ifndef LORRAX_FFI_NO_CUDA
// Format a cudaError_t as an FFI Internal error (for paths that can't
// use the LORRAX_CUDA_CHECK macro because they need to branch on the
// specific error before returning, e.g. tolerating PeerAccessAlreadyEnabled).
inline ffi::Error cuda_error(const char* expr, cudaError_t status) {
    std::ostringstream os;
    os << "CUDA error (" << expr << "): "
       << cudaGetErrorName(status) << " — " << cudaGetErrorString(status);
    return ffi::Error(ffi::ErrorCode::kInternal, os.str());
}
#endif  // !LORRAX_FFI_NO_CUDA

// If the expression evaluates to a non-success ffi::Error, return it.
#define FFI_RETURN_IF_ERROR(...)                       \
    do {                                               \
        ::xla::ffi::Error _err = (__VA_ARGS__);        \
        if (!_err.success()) return _err;              \
    } while (0)

#ifndef LORRAX_FFI_NO_CUDA
// CUDA runtime call returning cudaSuccess on OK.  Returns an FFI error
// (not throwing) so callers can use it inside handler functions.
#define LORRAX_CUDA_CHECK(expr)                                       \
    do {                                                                  \
        cudaError_t _st = (expr);                                         \
        if (_st != cudaSuccess) {                                         \
            std::ostringstream _os;                                       \
            _os << "CUDA error at " __FILE__ ":" << __LINE__              \
                << " (" #expr "): " << cudaGetErrorName(_st) << " — "     \
                << cudaGetErrorString(_st);                               \
            return ::xla::ffi::Error(::xla::ffi::ErrorCode::kInternal,    \
                                     _os.str());                          \
        }                                                                 \
    } while (0)
#endif  // !LORRAX_FFI_NO_CUDA

// Generic library-status check.  SUCCESS is the value that means OK,
// `lib` a short descriptive label (e.g. "cusolverMp").
#define LORRAX_LIB_CHECK(expr, SUCCESS, lib)                          \
    do {                                                                  \
        auto _st = (expr);                                                \
        if (_st != (SUCCESS)) {                                           \
            std::ostringstream _os;                                       \
            _os << lib << " error at " __FILE__ ":" << __LINE__           \
                << " (" #expr "): status=" << static_cast<int>(_st);      \
            return ::xla::ffi::Error(::xla::ffi::ErrorCode::kInternal,    \
                                     _os.str());                          \
        }                                                                 \
    } while (0)

}  // namespace lorrax_ffi
