// api.cc — extern "C" lifecycle entry points for the parallel-HDF5 FFI.
//
// These thin ctypes wrappers (no pybind/nanobind) drive the collective
// open/create, dataset open, and close paths implemented in context.cc.
// They are deliberately CUDA-FREE so this ONE translation unit compiles
// into BOTH platform libraries:
//   * liblorrax_ffi.so       (CUDA)  — alongside the cusolverMp/cuBLASMp
//                                       extern-C wrappers in cpp/common/api.cc
//   * liblorrax_ffi_host.so   (cpu)   — the CUDA-free host lib
// so multi-process CPU can open a collective context and drive the host
// read handlers through the same lifecycle the GPU path uses.  The
// collective MPI-IO core (context.cc) and the device-staging switch
// (read_ffi.cc) are the only phdf5 TUs that differ by platform, and they
// differ only by the LORRAX_FFI_NO_CUDA compile flag — this file is
// identical on both.
//
// All functions set *err_out (size err_cap) to a message on failure and
// return nonzero; return zero on success — so Python sees C++ exceptions as
// structured error strings without crossing the exception ABI boundary.

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <hdf5.h>

#include "ctx.h"

namespace lorrax_ffi::phdf5 {
    // Implemented in cpp/phdf5/context.cc
    PhdfCtx* open_ctx(const std::string& path, int p, int q,
                      int rank, int world_size, int mode_flag);
    void     close_ctx(PhdfCtx* ctx);
    hid_t    ensure_dataset(PhdfCtx* ctx, const std::string& ds_name,
                            const int64_t* shape, int ndim, int dtype_tag);
    hid_t    open_dataset_ro(PhdfCtx* ctx, const std::string& ds_name);
    void     ensure_mpi_initialized();
}

extern "C" {

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

// Eager MPI init.  Safe to call multiple times.  Use at program
// startup to move the ~400 ms MPI_Init_thread(THREAD_MULTIPLE) cost
// off the first-open critical path.
void lrx_phdf5_init_mpi(void) {
    lorrax_ffi::phdf5::ensure_mpi_initialized();
}

// Collective H5Dcreate/H5Dopen.  All ranks must call concurrently.
// shape[ndim] is the N-D dataset shape; ndim >= 1.  dtype_tag: 1=F32
// 2=F64 3=S32 4=S64 5=C64 6=C128 (matches xla::ffi::DataType).
// Returns 0 on success and writes the hid_t to *ds_id_out; sets err_out
// and returns 1 on failure.
int lrx_phdf5_ensure_dataset(
    int64_t ctx_handle,
    const char* ds_name,
    const int64_t* shape, int ndim,
    int dtype_tag,
    int64_t* ds_id_out,
    char* err_out, int err_cap)
{
    try {
        hid_t ds = lorrax_ffi::phdf5::ensure_dataset(
            reinterpret_cast<lorrax_ffi::phdf5::PhdfCtx*>(ctx_handle),
            std::string(ds_name), shape, ndim, dtype_tag);
        *ds_id_out = (int64_t)ds;
        return 0;
    } catch (const std::exception& e) {
        if (err_out && err_cap > 0) snprintf(err_out, err_cap, "%s", e.what());
        return 1;
    }
}

// Collective H5Dopen (read-only semantics; no create).  All ranks must
// call concurrently.  Returns 0 on success and writes the hid_t to
// *ds_id_out; sets err_out and returns 1 on failure.
int lrx_phdf5_open_dataset_ro(
    int64_t ctx_handle,
    const char* ds_name,
    int64_t* ds_id_out,
    char* err_out, int err_cap)
{
    try {
        hid_t ds = lorrax_ffi::phdf5::open_dataset_ro(
            reinterpret_cast<lorrax_ffi::phdf5::PhdfCtx*>(ctx_handle),
            std::string(ds_name));
        *ds_id_out = (int64_t)ds;
        return 0;
    } catch (const std::exception& e) {
        if (err_out && err_cap > 0) snprintf(err_out, err_cap, "%s", e.what());
        return 1;
    }
}

}  // extern "C"
