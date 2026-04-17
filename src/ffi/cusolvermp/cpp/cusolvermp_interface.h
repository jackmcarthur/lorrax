// cusolvermp_interface.h — per-dtype templates that thin-wrap cuSOLVERMp.
//
// Pattern copied from jaxlib/gpu/solver_interface.h: one template per
// cuSOLVERMp routine, specialized for each supported scalar type.  The FFI
// Impl<T>() functions call these templates, not the raw cusolverMp API.
// This keeps the CUDA_R_64F / CUDA_C_64F dtype enum out of the handler
// code and centralizes any per-dtype quirks in one file.

#pragma once

#include <complex>
#include <cstdint>

#include <cusolverMp.h>

namespace lorrax_ffi::cusolvermp::mp {

// Type trait: map scalar type -> cudaDataType_t for cuSOLVERMp.
template <typename T> struct CudaDataTypeOf;
template <> struct CudaDataTypeOf<float>                { static constexpr cudaDataType_t value = CUDA_R_32F; };
template <> struct CudaDataTypeOf<double>               { static constexpr cudaDataType_t value = CUDA_R_64F; };
template <> struct CudaDataTypeOf<std::complex<float>>  { static constexpr cudaDataType_t value = CUDA_C_32F; };
template <> struct CudaDataTypeOf<std::complex<double>> { static constexpr cudaDataType_t value = CUDA_C_64F; };

// Eigenvalue-output type for Syevd (always real, even for complex input).
template <typename T> struct RealOf                    { using type = T; };
template <> struct RealOf<std::complex<float>>         { using type = float; };
template <> struct RealOf<std::complex<double>>        { using type = double; };
template <typename T> using RealOf_t = typename RealOf<T>::type;

// ---------------------------------------------------------------------------
//  Syevd — Hermitian / real-symmetric eigensolver
// ---------------------------------------------------------------------------

template <typename T>
inline cusolverStatus_t SyevdBufferSize(
    cusolverMpHandle_t handle,
    char jobz, cublasFillMode_t uplo, int64_t n,
    const T* d_A, int64_t ia, int64_t ja, cusolverMpMatrixDescriptor_t descA,
    RealOf_t<T>* d_W,
    T* d_Q, int64_t iq, int64_t jq, cusolverMpMatrixDescriptor_t descQ,
    size_t* workspace_bytes_device,
    size_t* workspace_bytes_host)
{
    return cusolverMpSyevd_bufferSize(
        handle, &jobz, uplo, n,
        const_cast<void*>(static_cast<const void*>(d_A)), ia, ja, descA,
        static_cast<void*>(d_W),
        static_cast<void*>(d_Q), iq, jq, descQ,
        CudaDataTypeOf<T>::value,
        workspace_bytes_device, workspace_bytes_host);
}

template <typename T>
inline cusolverStatus_t Syevd(
    cusolverMpHandle_t handle,
    char jobz, cublasFillMode_t uplo, int64_t n,
    T* d_A, int64_t ia, int64_t ja, cusolverMpMatrixDescriptor_t descA,
    RealOf_t<T>* d_W,
    T* d_Q, int64_t iq, int64_t jq, cusolverMpMatrixDescriptor_t descQ,
    void* d_workspace, size_t workspace_bytes_device,
    void* h_workspace, size_t workspace_bytes_host,
    int* d_info)
{
    return cusolverMpSyevd(
        handle, &jobz, uplo, n,
        static_cast<void*>(d_A), ia, ja, descA,
        static_cast<void*>(d_W),
        static_cast<void*>(d_Q), iq, jq, descQ,
        CudaDataTypeOf<T>::value,
        d_workspace, workspace_bytes_device,
        h_workspace,  workspace_bytes_host,
        d_info);
}

}  // namespace lorrax_ffi::cusolvermp::mp
