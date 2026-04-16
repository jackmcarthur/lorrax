// xla_ffi_glue.h — tiny helpers for XLA FFI handlers in LORRAX.
//
// Kept intentionally small: error formatting, dtype check macros.
// For anything bigger, add a .cc and link it into lorrax_ffi.

#pragma once

#include <cstdio>
#include <sstream>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace lorrax_ffi {

namespace ffi = ::xla::ffi;

// Format a CUDA error as an FFI Internal error.
inline ffi::Error cuda_error(const char *expr, cudaError_t status) {
    std::ostringstream os;
    os << "CUDA error (" << expr << "): "
       << cudaGetErrorName(status) << " — " << cudaGetErrorString(status);
    return ffi::Error(ffi::ErrorCode::kInternal, os.str());
}

// CUDA_CHECK returns from the enclosing ffi::Error-returning function.
#define LORRAX_CUDA_CHECK(expr)                                           \
    do {                                                                  \
        cudaError_t _st = (expr);                                         \
        if (_st != cudaSuccess) {                                         \
            return ::lorrax_ffi::cuda_error(#expr, _st);                  \
        }                                                                 \
    } while (0)

// Generic CHECK that expects a specific success status.  Caller provides
// the success constant and a descriptive library name.
#define LORRAX_LIB_CHECK(expr, success_val, libname)                      \
    do {                                                                  \
        auto _st = (expr);                                                \
        if (_st != (success_val)) {                                       \
            std::ostringstream _os;                                       \
            _os << libname << " error in '" << #expr                      \
                << "': status=" << static_cast<int>(_st);                 \
            return ::xla::ffi::Error(                                     \
                ::xla::ffi::ErrorCode::kInternal, _os.str());             \
        }                                                                 \
    } while (0)

}  // namespace lorrax_ffi
