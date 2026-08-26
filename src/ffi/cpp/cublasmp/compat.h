#pragma once

#include <cstddef>

#include <cublasmp.h>

namespace lorrax_ffi::cublasmp {

// cuBLASMp 0.8 renamed the matmul-descriptor attribute accessors and
// removed the legacy spelling in later releases.  Keep the call sites on
// one LORRAX-owned name so the CUDA-12 baseline and current CUDA-13 SDKs
// compile from the same source.
inline cublasMpStatus_t set_matmul_descriptor_attribute(
    cublasMpMatmulDescriptor_t descriptor,
    cublasMpMatmulDescriptorAttribute_t attribute,
    const void* value,
    std::size_t value_bytes) {
#if CUBLASMP_VERSION >= 800
    return cublasMpMatmulDescriptorSetAttribute(
        descriptor, attribute, value, value_bytes);
#else
    return cublasMpMatmulDescriptorAttributeSet(
        descriptor, attribute, value, value_bytes);
#endif
}

}  // namespace lorrax_ffi::cublasmp
