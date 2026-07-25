// api.cc — extern "C" entry points for ctypes consumers.
//
// These thin wrappers let Python call the context lifecycle functions via
// ctypes (no pybind/nanobind dependency).  The XLA FFI handlers themselves
// are already exposed by XLA_FFI_DEFINE_HANDLER_SYMBOL as C symbols in
// cusolvermp/cpp/eigh_ffi.cc.

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>
#include <nccl.h>
#include <cusolverMp.h>

#include "../../cusolvermp/cpp/ctx.h"

// The parallel-HDF5 lifecycle extern-C wrappers (lrx_phdf5_*) now live in
// the CUDA-free phdf5/cpp/api.cc so the SAME TU compiles into both the CUDA
// library and the host library.  A build with -DLORRAX_FFI_HAVE_PHDF5=ON
// adds phdf5/cpp/api.cc to the source list (see common/cpp/CMakeLists.txt);
// a build without it simply omits that TU and ffi_loader.py skips the
// absent phdf5 symbols.

namespace lrx = lorrax_ffi::cusolvermp;

namespace lorrax_ffi::cusolvermp {
    // Implemented in context.cc
    int64_t create_context(int rank, int world_size,
                           uintptr_t nccl_unique_id_addr,
                           int nccl_unique_id_nbytes,
                           int p, int q,
                           bool grid_layout_col_major);
    void    destroy_context(int64_t ctx_handle);
    int     smoke_allreduce_sum(int64_t ctx_handle,
                                uintptr_t device_ptr,
                                int nelems);
}

// ---------------------------------------------------------------------------
// extern "C" ABI.  All functions set *err_out (size 512) to a message on
// failure and return nonzero; return zero on success.  This lets Python see
// C++ exceptions as structured error strings without crossing the exception
// ABI boundary.
// ---------------------------------------------------------------------------

extern "C" {

int lrx_nccl_unique_id_bytes(void) {
    return static_cast<int>(sizeof(ncclUniqueId));
}

// Fill (addr..addr+sizeof(ncclUniqueId)) with a fresh unique id.  Call ONLY
// on the rank that will broadcast.
int lrx_fill_nccl_unique_id(void* addr, char* err_out, int err_cap) {
    ncclUniqueId uid;
    ncclResult_t st = ncclGetUniqueId(&uid);
    if (st != ncclSuccess) {
        if (err_out && err_cap > 0) {
            snprintf(err_out, err_cap,
                     "ncclGetUniqueId failed: %s", ncclGetErrorString(st));
        }
        return 1;
    }
    std::memcpy(addr, &uid, sizeof(ncclUniqueId));
    return 0;
}

// Returns 0 on success, or sets err_out and returns 1.
int lrx_create_cusolvermp_context(
    int rank, int world_size,
    void* nccl_unique_id_addr,
    int nccl_unique_id_nbytes,
    int p, int q,
    int grid_layout_col_major,      // 0 = row-major, !=0 = col-major
    int64_t* ctx_out,
    char* err_out, int err_cap)
{
    try {
        int64_t h = lrx::create_context(
            rank, world_size,
            reinterpret_cast<uintptr_t>(nccl_unique_id_addr),
            nccl_unique_id_nbytes, p, q,
            /*col_major=*/(grid_layout_col_major != 0));
        *ctx_out = h;
        return 0;
    } catch (const std::exception& e) {
        if (err_out && err_cap > 0) {
            snprintf(err_out, err_cap, "%s", e.what());
        }
        return 1;
    }
}

void lrx_destroy_cusolvermp_context(int64_t ctx_handle) {
    lrx::destroy_context(ctx_handle);
}

int lrx_smoke_allreduce_sum(int64_t ctx_handle,
                            void* device_ptr,
                            int nelems) {
    return lrx::smoke_allreduce_sum(
        ctx_handle, reinterpret_cast<uintptr_t>(device_ptr), nelems);
}

// Diagnostic info: returns cuda runtime version and NCCL version.
int lrx_version_info(int* cuda_rt, int* cuda_drv, int* nccl_ver) {
    if (cuda_rt)  cudaRuntimeGetVersion(cuda_rt);
    if (cuda_drv) cudaDriverGetVersion(cuda_drv);
    if (nccl_ver) ncclGetVersion(nccl_ver);
    return 0;
}

// The parallel-HDF5 lifecycle wrappers (lrx_phdf5_open/close/init_mpi/
// ensure_dataset/open_dataset_ro) live in phdf5/cpp/api.cc (CUDA-free,
// shared with the host library).

}  // extern "C"
