// eigh_mg_ffi.cc — XLA FFI handler for cusolverMgSyevd.
//
// Single-process, multi-GPU Hermitian/symmetric eigensolve.  The handler
// takes a (n, n) device-0 matrix, scatters it across all visible GPUs via
// cuSOLVERMg's column-tile distribution, runs the solve (which uses all
// GPUs internally), and gathers the result back to device 0.
//
// This is architecturally simpler than cuSOLVERMp (no NCCL/MPI bootstrap,
// no multi-process coordination) and uses only libraries already shipped
// with the NVIDIA JAX container — no HPC SDK needed.
//
// Layout:
//   Input  A : (n, n) float64 or complex128, ROW-major (JAX convention)
//   Output D : (n,) float64, eigenvalues (ascending)
//   Output Q : (n, n) same dtype as A, eigenvectors (columns)
//
// Since cuSOLVERMg operates column-major (ScaLAPACK convention), we pass
// the row-major JAX buffer AS-IS to cuSOLVERMg, which interprets it as
// A^T.  For a Hermitian matrix A = A^H the eigenvalues of A^T equal
// those of A, so evals are correct.  Eigenvectors returned are the evecs
// of A^T, which are column-wise the same as the conjugate of A's evecs;
// a Python-side .conj() recovers A's evecs for the complex path.

#include <cstdint>
#include <cstring>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverMg.h>

#include "xla/ffi/api/ffi.h"

#include "../../common/cpp/xla_ffi_glue.h"

namespace lorrax_ffi {
namespace cusolvermg {

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Persistent per-process handle.  cuSOLVERMg has no multi-handle restrictions
// but the setup (device select, peer access) is expensive to repeat.
// ---------------------------------------------------------------------------
struct MgCtx {
    cusolverMgHandle_t handle = nullptr;
    std::vector<int>   device_ids;   // [0, 1, ..., n_gpus-1]
    bool               peer_enabled = false;
};

static MgCtx& mg_ctx() {
    static MgCtx ctx;
    return ctx;
}

static ffi::Error ensure_ctx_initialised(int max_gpus) {
    MgCtx& ctx = mg_ctx();
    if (ctx.handle != nullptr) return ffi::Error::Success();

    int n_gpus = 0;
    LORRAX_CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    if (max_gpus > 0 && max_gpus < n_gpus) n_gpus = max_gpus;
    if (n_gpus <= 0) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "cuSOLVERMg FFI: no CUDA devices visible");
    }

    ctx.device_ids.resize(n_gpus);
    for (int j = 0; j < n_gpus; ++j) ctx.device_ids[j] = j;

    LORRAX_LIB_CHECK(cusolverMgCreate(&ctx.handle),
                     CUSOLVER_STATUS_SUCCESS, "cusolverMgCreate");
    LORRAX_LIB_CHECK(cusolverMgDeviceSelect(ctx.handle, n_gpus,
                                            ctx.device_ids.data()),
                     CUSOLVER_STATUS_SUCCESS, "cusolverMgDeviceSelect");

    // Enable bidirectional peer access.  Safe to be a no-op if already on.
    for (int i = 0; i < n_gpus; ++i) {
        LORRAX_CUDA_CHECK(cudaSetDevice(ctx.device_ids[i]));
        for (int j = 0; j < n_gpus; ++j) {
            if (i == j) continue;
            cudaError_t st = cudaDeviceEnablePeerAccess(ctx.device_ids[j], 0);
            // cudaErrorPeerAccessAlreadyEnabled is fine
            if (st != cudaSuccess &&
                st != cudaErrorPeerAccessAlreadyEnabled) {
                return cuda_error("cudaDeviceEnablePeerAccess", st);
            }
            cudaGetLastError();  // clear sticky
        }
    }
    LORRAX_CUDA_CHECK(cudaSetDevice(0));
    ctx.peer_enabled = true;
    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
//  Helpers adapted from NVIDIA's cusolverMg_utils.h
//  (we reimplement rather than bringing the whole header in)
// ---------------------------------------------------------------------------

// Returns the INDEX into array_d / device_ids for the device that owns
// tile col_idx under block-cyclic col distribution across n_gpus with
// block size T_A.
static inline int owner_index(int64_t col_idx, int64_t T_A, int n_gpus) {
    int64_t block_idx = col_idx / T_A;
    return static_cast<int>(block_idx % n_gpus);
}

// Allocate per-device column tiles.  array_d[j] <- device-j buffer of size
// (lda * n_cols_on_device_j * elem_size) bytes.
static ffi::Error create_mat(int64_t N, int64_t T_A, int64_t lda,
                             size_t elem_size,
                             const std::vector<int>& device_ids,
                             std::vector<void*>* array_d)
{
    const int n_gpus = static_cast<int>(device_ids.size());
    array_d->assign(n_gpus, nullptr);
    // cols per device under cyclic distribution of ceil(N/T_A) blocks
    int64_t n_blocks = (N + T_A - 1) / T_A;
    for (int j = 0; j < n_gpus; ++j) {
        int64_t blocks_here = (n_blocks + n_gpus - 1 - j) / n_gpus; // ceil
        int64_t cols_here = blocks_here * T_A;
        size_t  bytes = static_cast<size_t>(lda) * cols_here * elem_size;
        LORRAX_CUDA_CHECK(cudaSetDevice(device_ids[j]));
        LORRAX_CUDA_CHECK(cudaMalloc(&(*array_d)[j], bytes));
        LORRAX_CUDA_CHECK(cudaMemset((*array_d)[j], 0, bytes));
    }
    LORRAX_CUDA_CHECK(cudaSetDevice(0));
    return ffi::Error::Success();
}

static void destroy_mat(std::vector<void*>& array_d,
                        const std::vector<int>& device_ids)
{
    for (size_t j = 0; j < array_d.size(); ++j) {
        if (array_d[j] == nullptr) continue;
        cudaSetDevice(device_ids[j]);
        cudaFree(array_d[j]);
        array_d[j] = nullptr;
    }
    cudaSetDevice(0);
}

// Scatter a (lda × N) column-major buffer on device 0 into the block-cyclic
// distribution expected by cuSOLVERMg.  For each source column k (0..N-1)
// we compute its owner device and within-device column index.
static ffi::Error scatter_device_to_distributed(
    int64_t N, int64_t T_A, int64_t lda, size_t elem_size,
    const void* src_dev0_ptr,                     // (lda × N) on dev 0
    std::vector<void*>* array_d,
    const std::vector<int>& device_ids,
    cudaStream_t stream)
{
    const int n_gpus = static_cast<int>(device_ids.size());
    const size_t col_bytes = static_cast<size_t>(lda) * elem_size;

    for (int64_t k = 0; k < N; ++k) {
        int     dev_idx   = owner_index(k, T_A, n_gpus);
        int     dev       = device_ids[dev_idx];
        int64_t block     = k / T_A;
        int64_t local_block = block / n_gpus;
        int64_t within    = k - block * T_A;       // col offset inside its block
        int64_t local_col = local_block * T_A + within;

        const char* src = static_cast<const char*>(src_dev0_ptr) + k * col_bytes;
        char*       dst = static_cast<char*>((*array_d)[dev_idx])
                          + local_col * col_bytes;
        LORRAX_CUDA_CHECK(cudaMemcpyPeerAsync(
            dst, dev, src, device_ids[0], col_bytes, stream));
    }
    return ffi::Error::Success();
}

static ffi::Error gather_distributed_to_device(
    int64_t N, int64_t T_A, int64_t lda, size_t elem_size,
    const std::vector<void*>& array_d,
    void* dst_dev0_ptr,
    const std::vector<int>& device_ids,
    cudaStream_t stream)
{
    const int n_gpus = static_cast<int>(device_ids.size());
    const size_t col_bytes = static_cast<size_t>(lda) * elem_size;

    for (int64_t k = 0; k < N; ++k) {
        int     dev_idx   = owner_index(k, T_A, n_gpus);
        int     dev       = device_ids[dev_idx];
        int64_t block     = k / T_A;
        int64_t local_block = block / n_gpus;
        int64_t within    = k - block * T_A;
        int64_t local_col = local_block * T_A + within;

        const char* src = static_cast<const char*>(array_d[dev_idx])
                          + local_col * col_bytes;
        char*       dst = static_cast<char*>(dst_dev0_ptr) + k * col_bytes;
        LORRAX_CUDA_CHECK(cudaMemcpyPeerAsync(
            dst, device_ids[0], src, dev, col_bytes, stream));
    }
    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
//  FFI handler: real-symmetric F64
// ---------------------------------------------------------------------------
static ffi::Error EighMgF64Host(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::F64> A,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> D_out,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> Q_out,
    int64_t n, int64_t tile_size, int64_t max_gpus_attr,
    bool compute_evecs)
{
    auto e0 = ensure_ctx_initialised(static_cast<int>(max_gpus_attr));
    if (!e0.success()) return e0;
    MgCtx& ctx = mg_ctx();

    const int64_t lda = n;
    const int64_t T_A = tile_size > 0 ? tile_size : 32;

    // Stage (a): allocate distributed A and workspace
    cudaLibMgMatrixDesc_t descA = nullptr;
    cudaLibMgGrid_t grid = nullptr;
    LORRAX_LIB_CHECK(
        cusolverMgCreateDeviceGrid(&grid, /*nRowDevices=*/1,
                                   static_cast<int>(ctx.device_ids.size()),
                                   ctx.device_ids.data(),
                                   CUDALIBMG_GRID_MAPPING_COL_MAJOR),
        CUSOLVER_STATUS_SUCCESS, "cusolverMgCreateDeviceGrid");
    LORRAX_LIB_CHECK(
        cusolverMgCreateMatrixDesc(&descA, n, n, n, T_A, CUDA_R_64F, grid),
        CUSOLVER_STATUS_SUCCESS, "cusolverMgCreateMatrixDesc");

    std::vector<void*> array_d_A;
    auto e1 = create_mat(n, T_A, lda, sizeof(double), ctx.device_ids, &array_d_A);
    if (!e1.success()) return e1;

    // Scatter A from device 0 (XLA's input) into the distributed layout
    auto e2 = scatter_device_to_distributed(
        n, T_A, lda, sizeof(double),
        A.typed_data(), &array_d_A, ctx.device_ids, stream);
    if (!e2.success()) return e2;
    LORRAX_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Stage (b): workspace query
    cusolverEigMode_t jobz = compute_evecs ? CUSOLVER_EIG_MODE_VECTOR
                                           : CUSOLVER_EIG_MODE_NOVECTOR;
    int64_t lwork = 0;
    LORRAX_LIB_CHECK(
        cusolverMgSyevd_bufferSize(
            ctx.handle, jobz, CUBLAS_FILL_MODE_LOWER, n,
            array_d_A.data(),
            /*IA=*/1, /*JA=*/1,
            descA,
            /*W=*/nullptr,
            CUDA_R_64F,
            /*computeType=*/CUDA_R_64F,
            &lwork),
        CUSOLVER_STATUS_SUCCESS, "cusolverMgSyevd_bufferSize");

    // Stage (c): allocate per-device workspace
    const int n_gpus = static_cast<int>(ctx.device_ids.size());
    std::vector<void*> array_d_work(n_gpus, nullptr);
    for (int j = 0; j < n_gpus; ++j) {
        LORRAX_CUDA_CHECK(cudaSetDevice(ctx.device_ids[j]));
        LORRAX_CUDA_CHECK(cudaMalloc(&array_d_work[j], sizeof(double) * lwork));
    }
    LORRAX_CUDA_CHECK(cudaSetDevice(0));

    // Eigenvalues live in host memory for cuSOLVERMg
    std::vector<double> h_D(n, 0.0);

    // Stage (d): solve
    int info = 0;
    LORRAX_LIB_CHECK(
        cusolverMgSyevd(
            ctx.handle, jobz, CUBLAS_FILL_MODE_LOWER, n,
            array_d_A.data(),
            /*IA=*/1, /*JA=*/1, descA,
            reinterpret_cast<void*>(h_D.data()),
            CUDA_R_64F,
            /*computeType=*/CUDA_R_64F,
            array_d_work.data(), lwork, &info),
        CUSOLVER_STATUS_SUCCESS, "cusolverMgSyevd");
    LORRAX_CUDA_CHECK(cudaDeviceSynchronize());

    if (info != 0) {
        std::ostringstream os;
        os << "cusolverMgSyevd returned info=" << info;
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }

    // Stage (e): copy evals to device-0 output
    LORRAX_CUDA_CHECK(cudaMemcpyAsync(
        D_out->typed_data(), h_D.data(),
        sizeof(double) * n, cudaMemcpyHostToDevice, stream));

    // Stage (f): gather eigenvectors back to device 0
    auto e3 = gather_distributed_to_device(
        n, T_A, lda, sizeof(double),
        array_d_A, Q_out->typed_data(), ctx.device_ids, stream);
    if (!e3.success()) return e3;
    LORRAX_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Teardown of per-call state
    for (int j = 0; j < n_gpus; ++j) {
        if (array_d_work[j]) {
            cudaSetDevice(ctx.device_ids[j]);
            cudaFree(array_d_work[j]);
        }
    }
    cudaSetDevice(0);
    destroy_mat(array_d_A, ctx.device_ids);
    cusolverMgDestroyMatrixDesc(descA);
    cusolverMgDestroyGrid(grid);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    EighMgF64, EighMgF64Host,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()     // input A
        .Ret<ffi::Buffer<ffi::DataType::F64>>()     // output D (evals)
        .Ret<ffi::Buffer<ffi::DataType::F64>>()     // output Q (evecs)
        .Attr<int64_t>("n")
        .Attr<int64_t>("tile_size")
        .Attr<int64_t>("max_gpus")
        .Attr<bool>("compute_evecs"));

}  // namespace cusolvermg
}  // namespace lorrax_ffi
