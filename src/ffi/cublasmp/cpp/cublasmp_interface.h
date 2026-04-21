// cublasmp_interface.h — per-dtype templates that thin-wrap cuBLASMp.
//
// Pattern mirrors cusolvermp_interface.h: one template per cuBLASMp routine,
// specialized for each supported scalar type.  The FFI handlers call these
// templates, not the raw cublasMp API, which keeps the CUDA_R_64F / CUDA_C_64F
// enum plumbing out of the handlers.

#pragma once

#include <complex>
#include <cstdint>

#include <cublasmp.h>

namespace lorrax_ffi::cublasmp::mp {

// Type trait: map scalar type -> cudaDataType_t for cuBLASMp matrix descriptors.
template <typename T> struct CudaDataTypeOf;
template <> struct CudaDataTypeOf<float>                { static constexpr cudaDataType_t value = CUDA_R_32F; };
template <> struct CudaDataTypeOf<double>               { static constexpr cudaDataType_t value = CUDA_R_64F; };
template <> struct CudaDataTypeOf<std::complex<float>>  { static constexpr cudaDataType_t value = CUDA_C_32F; };
template <> struct CudaDataTypeOf<std::complex<double>> { static constexpr cudaDataType_t value = CUDA_C_64F; };

// cuBLAS compute type trait — specifies internal precision for matmul.
// Using _64F for complex128 and _32F for complex64 matches standard conventions.
template <typename T> struct ComputeTypeOf;
template <> struct ComputeTypeOf<float>                 { static constexpr cublasComputeType_t value = CUBLAS_COMPUTE_32F; };
template <> struct ComputeTypeOf<double>                { static constexpr cublasComputeType_t value = CUBLAS_COMPUTE_64F; };
template <> struct ComputeTypeOf<std::complex<float>>   { static constexpr cublasComputeType_t value = CUBLAS_COMPUTE_32F; };
template <> struct ComputeTypeOf<std::complex<double>>  { static constexpr cublasComputeType_t value = CUBLAS_COMPUTE_64F; };

// ---------------------------------------------------------------------------
//  Matmul — distributed GEMM: D = alpha * op(A) * op(B) + beta * C
//
// op(A) is selected via CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA on the
// matmul descriptor (N / T / C).  D and C may share a buffer (common usage).
// ---------------------------------------------------------------------------

template <typename T>
inline cublasMpStatus_t MatmulBufferSize(
    cublasMpHandle_t handle,
    cublasMpMatmulDescriptor_t matmulDesc,
    int64_t m, int64_t n, int64_t k,
    const T* alpha,
    const T* d_A, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA,
    const T* d_B, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB,
    const T* beta,
    const T* d_C, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC,
    T* d_D, int64_t id, int64_t jd, cublasMpMatrixDescriptor_t descD,
    size_t* workspace_bytes_device,
    size_t* workspace_bytes_host)
{
    return cublasMpMatmul_bufferSize(
        handle, matmulDesc, m, n, k,
        alpha, d_A, ia, ja, descA,
        d_B, ib, jb, descB,
        beta, d_C, ic, jc, descC,
        d_D, id, jd, descD,
        workspace_bytes_device, workspace_bytes_host);
}

template <typename T>
inline cublasMpStatus_t Matmul(
    cublasMpHandle_t handle,
    cublasMpMatmulDescriptor_t matmulDesc,
    int64_t m, int64_t n, int64_t k,
    const T* alpha,
    const T* d_A, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA,
    const T* d_B, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB,
    const T* beta,
    const T* d_C, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC,
    T* d_D, int64_t id, int64_t jd, cublasMpMatrixDescriptor_t descD,
    void* d_workspace, size_t workspace_bytes_device,
    void* h_workspace, size_t workspace_bytes_host)
{
    return cublasMpMatmul(
        handle, matmulDesc, m, n, k,
        alpha, d_A, ia, ja, descA,
        d_B, ib, jb, descB,
        beta, d_C, ic, jc, descC,
        d_D, id, jd, descD,
        d_workspace, workspace_bytes_device,
        h_workspace, workspace_bytes_host);
}

// ---------------------------------------------------------------------------
//  Trsm — distributed triangular solve:  op(A) X = alpha B  (side=LEFT) or
//                                         X op(A) = alpha B  (side=RIGHT)
//  B is read and overwritten with X (in place).  A is triangular.
// ---------------------------------------------------------------------------

template <typename T>
inline cublasMpStatus_t TrsmBufferSize(
    cublasMpHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    int64_t m, int64_t n,
    const T* alpha,
    const T* d_A, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA,
    T* d_B, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB,
    size_t* workspace_bytes_device,
    size_t* workspace_bytes_host)
{
    return cublasMpTrsm_bufferSize(
        handle, side, uplo, trans, diag, m, n,
        alpha, d_A, ia, ja, descA,
        d_B, ib, jb, descB,
        ComputeTypeOf<T>::value,
        workspace_bytes_device, workspace_bytes_host);
}

template <typename T>
inline cublasMpStatus_t Trsm(
    cublasMpHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    int64_t m, int64_t n,
    const T* alpha,
    const T* d_A, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA,
    T* d_B, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB,
    void* d_workspace, size_t workspace_bytes_device,
    void* h_workspace, size_t workspace_bytes_host)
{
    return cublasMpTrsm(
        handle, side, uplo, trans, diag, m, n,
        alpha, d_A, ia, ja, descA,
        d_B, ib, jb, descB,
        ComputeTypeOf<T>::value,
        d_workspace, workspace_bytes_device,
        h_workspace, workspace_bytes_host);
}

}  // namespace lorrax_ffi::cublasmp::mp
