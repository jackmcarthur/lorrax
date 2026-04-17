// ffi_helpers.h — small error-propagation helpers shared by LORRAX FFI
// handlers.  Intentionally minimal — only the macros that make handler
// code noticeably shorter.  Extend when a second handler actually needs
// something, not speculatively.

#pragma once

#include <cstdio>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace lorrax_ffi {

namespace ffi = ::xla::ffi;

// If the expression evaluates to a non-success ffi::Error, return it.
#define FFI_RETURN_IF_ERROR(...)                       \
    do {                                               \
        ::xla::ffi::Error _err = (__VA_ARGS__);        \
        if (!_err.success()) return _err;              \
    } while (0)

// CUDA runtime call returning cudaSuccess on OK.  Returns an FFI error
// (not throwing) so callers can use it inside handler functions.
#define LORRAX_CUDA_CHECK_FFI(expr)                                       \
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

// Generic library-status check.  SUCCESS is the value that means OK,
// `lib` a short descriptive label (e.g. "cusolverMp").
#define LORRAX_LIB_CHECK_FFI(expr, SUCCESS, lib)                          \
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
