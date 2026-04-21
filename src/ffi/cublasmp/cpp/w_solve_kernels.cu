// w_solve_kernels.cu — CUDA helper kernels for the fused W-solve.
//
// Two per-rank element-wise ops that depend on each rank's (x_idx, y_idx)
// coordinates in the (Px, Py) process grid, because the operation targets
// the GLOBAL (i, j) position of each local element:
//
//   zero_upper_tri_lower   — zero the upper triangle of a distributed
//                            lower-triangular matrix.  Needed because
//                            cusolverMpPotrf leaves the upper triangle
//                            untouched (original A bytes), which would
//                            contaminate any downstream matmul unless
//                            we zero it explicitly.
//
//   identity_minus_T       — H[i,j] = (i==j) ? 1 - T[i,j] : -T[i,j]
//                            on a distributed matrix.  The identity
//                            diagonal lives at global (i,i), so each
//                            rank needs its (x_idx, y_idx) to figure
//                            out which of its local entries are
//                            on-diagonal.
//
// Layout: each rank holds one (mb × nb) tile in col-major with lld=mb.
// Buffer offset for (slice q, local row r, local col c) is
//     q * mb * nb + c * mb + r
// Global coords (i_glob, j_glob) = (x_idx*mb + r, y_idx*nb + c).

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>
#include <cstdint>

namespace lorrax_ffi::cublasmp_batched_gemm {

// Zero-template helpers for the per-element complex/real operation.
template <typename T> __device__ __forceinline__ T make_zero();
template <> __device__ __forceinline__ double make_zero<double>() { return 0.0; }
template <> __device__ __forceinline__ cuDoubleComplex make_zero<cuDoubleComplex>() {
    return make_cuDoubleComplex(0.0, 0.0);
}
template <typename T> __device__ __forceinline__ T make_one();
template <> __device__ __forceinline__ double make_one<double>() { return 1.0; }
template <> __device__ __forceinline__ cuDoubleComplex make_one<cuDoubleComplex>() {
    return make_cuDoubleComplex(1.0, 0.0);
}

template <typename T>
__device__ __forceinline__ T sub(T a, T b);
template <> __device__ __forceinline__ double sub<double>(double a, double b) { return a - b; }
template <> __device__ __forceinline__ cuDoubleComplex sub<cuDoubleComplex>(
    cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(cuCreal(a) - cuCreal(b),
                                 cuCimag(a) - cuCimag(b));
}

template <typename T>
__global__ void zero_upper_tri_kernel(
    T* buf,
    int64_t mb, int64_t nb, int64_t nq,
    int64_t x_idx, int64_t y_idx)
{
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t c = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t q = blockIdx.z * blockDim.z + threadIdx.z;
    if (r >= mb || c >= nb || q >= nq) return;
    const int64_t i_glob = x_idx * mb + r;
    const int64_t j_glob = y_idx * nb + c;
    // Keep iff i_glob >= j_glob.  Otherwise zero.
    if (i_glob < j_glob) {
        buf[q * mb * nb + c * mb + r] = make_zero<T>();
    }
}

template <typename T>
__global__ void identity_minus_T_kernel(
    T* H_out, const T* T_in,
    int64_t mb, int64_t nb, int64_t nq,
    int64_t x_idx, int64_t y_idx)
{
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t c = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t q = blockIdx.z * blockDim.z + threadIdx.z;
    if (r >= mb || c >= nb || q >= nq) return;
    const int64_t i_glob = x_idx * mb + r;
    const int64_t j_glob = y_idx * nb + c;
    const int64_t off = q * mb * nb + c * mb + r;
    T v = T_in[off];
    T neg_v = sub<T>(make_zero<T>(), v);
    if (i_glob == j_glob) {
        H_out[off] = sub<T>(make_one<T>(), v);
    } else {
        H_out[off] = neg_v;
    }
}

// Pad X^† into a larger buffer: copy conj-transpose of X (lower-tri col-major
// tile) into the first N columns of a wider (N + pad) buffer, zero the rest.
// For our potrs workaround, the padded buffer has N + 2*Py cols globally,
// so per rank it's n/Px rows × (N/Py + 2) cols (one extra col per rank).
// Not used yet — Python-side jnp.pad is simpler until we need it here.

// Extern "C" launchers (callable from the FFI .cc without CUDA in that TU).
extern "C" {

void lorrax_zero_upper_tri_f64(
    double* buf, int64_t mb, int64_t nb, int64_t nq,
    int64_t x_idx, int64_t y_idx, cudaStream_t stream) {
    dim3 block(16, 16, 1);
    dim3 grid((mb + 15) / 16, (nb + 15) / 16, nq);
    zero_upper_tri_kernel<double><<<grid, block, 0, stream>>>(
        buf, mb, nb, nq, x_idx, y_idx);
}
void lorrax_zero_upper_tri_c128(
    cuDoubleComplex* buf, int64_t mb, int64_t nb, int64_t nq,
    int64_t x_idx, int64_t y_idx, cudaStream_t stream) {
    dim3 block(16, 16, 1);
    dim3 grid((mb + 15) / 16, (nb + 15) / 16, nq);
    zero_upper_tri_kernel<cuDoubleComplex><<<grid, block, 0, stream>>>(
        buf, mb, nb, nq, x_idx, y_idx);
}

void lorrax_identity_minus_T_f64(
    double* H_out, const double* T_in,
    int64_t mb, int64_t nb, int64_t nq,
    int64_t x_idx, int64_t y_idx, cudaStream_t stream) {
    dim3 block(16, 16, 1);
    dim3 grid((mb + 15) / 16, (nb + 15) / 16, nq);
    identity_minus_T_kernel<double><<<grid, block, 0, stream>>>(
        H_out, T_in, mb, nb, nq, x_idx, y_idx);
}
void lorrax_identity_minus_T_c128(
    cuDoubleComplex* H_out, const cuDoubleComplex* T_in,
    int64_t mb, int64_t nb, int64_t nq,
    int64_t x_idx, int64_t y_idx, cudaStream_t stream) {
    dim3 block(16, 16, 1);
    dim3 grid((mb + 15) / 16, (nb + 15) / 16, nq);
    identity_minus_T_kernel<cuDoubleComplex><<<grid, block, 0, stream>>>(
        H_out, T_in, mb, nb, nq, x_idx, y_idx);
}

}  // extern "C"

}  // namespace lorrax_ffi::cublasmp_batched_gemm
