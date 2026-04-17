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
#include "../../phdf5/cpp/ctx.h"

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

namespace lorrax_ffi::phdf5 {
    // Implemented in phdf5/cpp/context.cc
    PhdfCtx* open_ctx(const std::string& path, int p, int q,
                      int rank, int world_size, int mode_flag);
    void     close_ctx(PhdfCtx* ctx);
    hid_t    ensure_dataset(PhdfCtx* ctx, const std::string& ds_name,
                            int64_t n_rows, int64_t n_cols, int dtype_tag);
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

// ---------------------------------------------------------------------------
//  parallel-HDF5 subpackage lifecycle
// ---------------------------------------------------------------------------

// Open (or create) a parallel-HDF5 file collectively.
// mode_flag: 0 = 'w' truncate, 1 = 'a' append-or-create, 2 = 'r' read-only
// Returns 0 on success; sets err_out and returns 1 on failure.
int lrx_phdf5_open(
    const char* path,
    int p, int q, int rank, int world_size,
    int mode_flag,
    int64_t* ctx_out,
    char* err_out, int err_cap)
{
    try {
        auto* ctx = lorrax_ffi::phdf5::open_ctx(
            std::string(path), p, q, rank, world_size, mode_flag);
        *ctx_out = reinterpret_cast<int64_t>(ctx);
        return 0;
    } catch (const std::exception& e) {
        if (err_out && err_cap > 0) {
            snprintf(err_out, err_cap, "%s", e.what());
        }
        return 1;
    }
}

void lrx_phdf5_close(int64_t ctx_handle) {
    lorrax_ffi::phdf5::close_ctx(
        reinterpret_cast<lorrax_ffi::phdf5::PhdfCtx*>(ctx_handle));
}

// Collective H5Dcreate/H5Dopen.  All ranks must call concurrently.
// dtype_tag: 1=F32 2=F64 3=S32 4=S64 5=C64 6=C128 (matches xla::ffi::DataType).
// Returns 0 on success and writes the hid_t to *ds_id_out; sets err_out and
// returns 1 on failure.
int lrx_phdf5_ensure_dataset(
    int64_t ctx_handle,
    const char* ds_name,
    int64_t n_rows, int64_t n_cols,
    int dtype_tag,
    int64_t* ds_id_out,
    char* err_out, int err_cap)
{
    try {
        hid_t ds = lorrax_ffi::phdf5::ensure_dataset(
            reinterpret_cast<lorrax_ffi::phdf5::PhdfCtx*>(ctx_handle),
            std::string(ds_name), n_rows, n_cols, dtype_tag);
        *ds_id_out = (int64_t)ds;
        return 0;
    } catch (const std::exception& e) {
        if (err_out && err_cap > 0) snprintf(err_out, err_cap, "%s", e.what());
        return 1;
    }
}

}  // extern "C"
