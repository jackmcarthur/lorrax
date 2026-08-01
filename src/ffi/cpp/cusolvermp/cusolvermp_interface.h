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

// ---------------------------------------------------------------------------
//  Potrf — distributed Cholesky factorization (A = L L^H, in-place)
// ---------------------------------------------------------------------------

template <typename T>
inline cusolverStatus_t PotrfBufferSize(
    cusolverMpHandle_t handle,
    cublasFillMode_t uplo, int64_t n,
    const T* d_A, int64_t ia, int64_t ja, cusolverMpMatrixDescriptor_t descA,
    size_t* workspace_bytes_device,
    size_t* workspace_bytes_host)
{
    return cusolverMpPotrf_bufferSize(
        handle, uplo, n,
        const_cast<void*>(static_cast<const void*>(d_A)), ia, ja, descA,
        CudaDataTypeOf<T>::value,
        workspace_bytes_device, workspace_bytes_host);
}

template <typename T>
inline cusolverStatus_t Potrf(
    cusolverMpHandle_t handle,
    cublasFillMode_t uplo, int64_t n,
    T* d_A, int64_t ia, int64_t ja, cusolverMpMatrixDescriptor_t descA,
    void* d_workspace, size_t workspace_bytes_device,
    void* h_workspace, size_t workspace_bytes_host,
    int* d_info)
{
    return cusolverMpPotrf(
        handle, uplo, n,
        static_cast<void*>(d_A), ia, ja, descA,
        CudaDataTypeOf<T>::value,
        d_workspace, workspace_bytes_device,
        h_workspace, workspace_bytes_host,
        d_info);
}

// ---------------------------------------------------------------------------
//  Potrs — solve A X = B given A's Cholesky factor (in-place on B)
// ---------------------------------------------------------------------------

template <typename T>
inline cusolverStatus_t PotrsBufferSize(
    cusolverMpHandle_t handle,
    cublasFillMode_t uplo, int64_t n, int64_t nrhs,
    const T* d_A, int64_t ia, int64_t ja, cusolverMpMatrixDescriptor_t descA,
    const T* d_B, int64_t ib, int64_t jb, cusolverMpMatrixDescriptor_t descB,
    size_t* workspace_bytes_device,
    size_t* workspace_bytes_host)
{
    return cusolverMpPotrs_bufferSize(
        handle, uplo, n, nrhs,
        const_cast<void*>(static_cast<const void*>(d_A)), ia, ja, descA,
        const_cast<void*>(static_cast<const void*>(d_B)), ib, jb, descB,
        CudaDataTypeOf<T>::value,
        workspace_bytes_device, workspace_bytes_host);
}

template <typename T>
inline cusolverStatus_t Potrs(
    cusolverMpHandle_t handle,
    cublasFillMode_t uplo, int64_t n, int64_t nrhs,
    T* d_A, int64_t ia, int64_t ja, cusolverMpMatrixDescriptor_t descA,
    T* d_B, int64_t ib, int64_t jb, cusolverMpMatrixDescriptor_t descB,
    void* d_workspace, size_t workspace_bytes_device,
    void* h_workspace, size_t workspace_bytes_host,
    int* d_info)
{
    return cusolverMpPotrs(
        handle, uplo, n, nrhs,
        static_cast<void*>(d_A), ia, ja, descA,
        static_cast<void*>(d_B), ib, jb, descB,
        CudaDataTypeOf<T>::value,
        d_workspace, workspace_bytes_device,
        h_workspace, workspace_bytes_host,
        d_info);
}

// ---------------------------------------------------------------------------
//  Getrf / Getrs — distributed LU factorization with partial pivoting
// ---------------------------------------------------------------------------

template <typename T>
inline cusolverStatus_t GetrfBufferSize(
    cusolverMpHandle_t handle,
    int64_t m, int64_t n,
    const T* d_A, int64_t ia, int64_t ja, cusolverMpMatrixDescriptor_t descA,
    int64_t* d_ipiv,
    size_t* workspace_bytes_device,
    size_t* workspace_bytes_host)
{
    return cusolverMpGetrf_bufferSize(
        handle, m, n,
        const_cast<void*>(static_cast<const void*>(d_A)), ia, ja, descA,
        d_ipiv,
        CudaDataTypeOf<T>::value,
        workspace_bytes_device, workspace_bytes_host);
}

template <typename T>
inline cusolverStatus_t Getrf(
    cusolverMpHandle_t handle,
    int64_t m, int64_t n,
    T* d_A, int64_t ia, int64_t ja, cusolverMpMatrixDescriptor_t descA,
    int64_t* d_ipiv,
    void* d_workspace, size_t workspace_bytes_device,
    void* h_workspace, size_t workspace_bytes_host,
    int* d_info)
{
    return cusolverMpGetrf(
        handle, m, n,
        static_cast<void*>(d_A), ia, ja, descA,
        d_ipiv,
        CudaDataTypeOf<T>::value,
        d_workspace, workspace_bytes_device,
        h_workspace, workspace_bytes_host,
        d_info);
}

template <typename T>
inline cusolverStatus_t GetrsBufferSize(
    cusolverMpHandle_t handle,
    cublasOperation_t trans, int64_t n, int64_t nrhs,
    const T* d_A, int64_t ia, int64_t ja, cusolverMpMatrixDescriptor_t descA,
    const int64_t* d_ipiv,
    const T* d_B, int64_t ib, int64_t jb, cusolverMpMatrixDescriptor_t descB,
    size_t* workspace_bytes_device,
    size_t* workspace_bytes_host)
{
    return cusolverMpGetrs_bufferSize(
        handle, trans, n, nrhs,
        const_cast<void*>(static_cast<const void*>(d_A)), ia, ja, descA,
        d_ipiv,
        const_cast<void*>(static_cast<const void*>(d_B)), ib, jb, descB,
        CudaDataTypeOf<T>::value,
        workspace_bytes_device, workspace_bytes_host);
}

template <typename T>
inline cusolverStatus_t Getrs(
    cusolverMpHandle_t handle,
    cublasOperation_t trans, int64_t n, int64_t nrhs,
    const T* d_A, int64_t ia, int64_t ja, cusolverMpMatrixDescriptor_t descA,
    const int64_t* d_ipiv,
    T* d_B, int64_t ib, int64_t jb, cusolverMpMatrixDescriptor_t descB,
    void* d_workspace, size_t workspace_bytes_device,
    void* h_workspace, size_t workspace_bytes_host,
    int* d_info)
{
    return cusolverMpGetrs(
        handle, trans, n, nrhs,
        const_cast<void*>(static_cast<const void*>(d_A)), ia, ja, descA,
        d_ipiv,
        static_cast<void*>(d_B), ib, jb, descB,
        CudaDataTypeOf<T>::value,
        d_workspace, workspace_bytes_device,
        h_workspace, workspace_bytes_host,
        d_info);
}

}  // namespace lorrax_ffi::cusolvermp::mp
