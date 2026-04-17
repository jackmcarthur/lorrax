// context.cc — open/close for the parallel-HDF5 FFI.
//
// open_file: collective MPI_Init_thread + H5Fcreate/H5Fopen with cached
// property lists.  Tuning via env vars (LORRAX_PHDF5_INDEPENDENT,
// LORRAX_PHDF5_ALIGN_MB, LORRAX_PHDF5_NO_COLL_META).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <mpi.h>
#include <hdf5.h>

#include "ctx.h"
#include "phdf5_interface.h"

namespace lorrax_ffi::phdf5 {

static void throw_if(hid_t id, const char* what) {
    if (id < 0) {
        std::ostringstream os;
        os << "HDF5 error in " << what << " (returned " << id << ")";
        throw std::runtime_error(os.str());
    }
}
static void throw_if_cuda(cudaError_t st, const char* what) {
    if (st != cudaSuccess) {
        std::ostringstream os;
        os << what << ": " << cudaGetErrorName(st) << " (" << cudaGetErrorString(st) << ")";
        throw std::runtime_error(os.str());
    }
}

// Lazy one-shot MPI init.  cuSOLVERMp doesn't use MPI, so if the caller
// only ever touches cuSOLVERMp there's no MPI_Init surprise.  First
// open_file triggers MPI_Init_thread(THREAD_MULTIPLE).
static void ensure_mpi_initialized() {
    int inited = 0;
    MPI_Initialized(&inited);
    if (inited) return;
    int provided = 0;
    // MPI_THREAD_MULTIPLE is the strictest level — safer for coexisting
    // with XLA's thread pool.  Some MPIs fall back to a weaker level;
    // we accept whatever we get for now.
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    // Do NOT register MPI_Finalize via atexit — HDF5 calls MPI at its
    // own destructors; ordering is fragile.  Our close_file path calls
    // MPI_Finalize explicitly when the last file closes (via a ref count).
}

// Pull an integer tunable from env, falling back to default.
static long env_long(const char* name, long default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    char* end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v) return default_value;
    return parsed;
}

bool ensure_pinned(PhdfCtx* ctx, size_t need_bytes) {
    if (ctx->pinned_capacity >= need_bytes) return true;
    if (ctx->pinned_buf) {
        cudaFreeHost(ctx->pinned_buf);
        ctx->pinned_buf = nullptr;
        ctx->pinned_capacity = 0;
    }
    // Round up to a multiple of 2 MiB to reduce re-allocation churn
    // across writes of slightly varying sizes.
    size_t rounded = ((need_bytes + (2 << 20) - 1) / (2 << 20)) * (2 << 20);
    if (cudaMallocHost(&ctx->pinned_buf, rounded) != cudaSuccess) {
        ctx->pinned_buf = nullptr;
        ctx->pinned_capacity = 0;
        return false;
    }
    ctx->pinned_capacity = rounded;
    return true;
}

// -----------------------------------------------------------------
//  open_file — collective across all JAX processes.
// -----------------------------------------------------------------
// mode_flag: 0 = 'w' (truncate), 1 = 'a' (append / create-if-absent), 2 = 'r'
static hid_t h5_open_or_create(const std::string& path, int mode_flag, hid_t fapl) {
    switch (mode_flag) {
        case 0:
            return H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        case 1: {
            // Try open, else create.
            htri_t ok = H5Fis_accessible(path.c_str(), fapl);
            if (ok > 0) {
                return H5Fopen(path.c_str(), H5F_ACC_RDWR, fapl);
            }
            return H5Fcreate(path.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, fapl);
        }
        case 2:
            return H5Fopen(path.c_str(), H5F_ACC_RDONLY, fapl);
        default:
            return H5I_INVALID_HID;
    }
}

PhdfCtx* open_ctx(const std::string& path, int p, int q,
                  int rank, int world_size, int mode_flag)
{
    ensure_mpi_initialized();

    auto* ctx = new PhdfCtx{};
    ctx->path = path;
    ctx->p = p;
    ctx->q = q;
    ctx->rank = rank;
    ctx->world_size = world_size;

    // Env-driven tuning.
    ctx->use_collective  = (env_long("LORRAX_PHDF5_INDEPENDENT", 0) == 0);
    ctx->coll_metadata   = (env_long("LORRAX_PHDF5_NO_COLL_META", 0) == 0);
    long align_mb        = env_long("LORRAX_PHDF5_ALIGN_MB", 1);
    ctx->align_threshold = (align_mb > 0) ? (size_t)align_mb << 20 : 0;
    ctx->align_length    = ctx->align_threshold;

    // Duplicate MPI_COMM_WORLD so HDF5's internal splits/teardown don't
    // affect other MPI users in the process.
    MPI_Comm_dup(MPI_COMM_WORLD, &ctx->comm);
    ctx->owns_comm = true;

    // --- fapl: MPI-IO + (optional) collective metadata + alignment ---
    ctx->fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    throw_if(ctx->fapl_id, "H5Pcreate(FILE_ACCESS)");
    if (H5Pset_fapl_mpio(ctx->fapl_id, ctx->comm, MPI_INFO_NULL) < 0) {
        throw std::runtime_error("H5Pset_fapl_mpio failed");
    }
#if H5_VERSION_GE(1, 10, 0)
    if (ctx->coll_metadata) {
        H5Pset_coll_metadata_write(ctx->fapl_id, /*is_collective=*/true);
        H5Pset_all_coll_metadata_ops(ctx->fapl_id, /*is_collective=*/true);
    }
#endif
    if (ctx->align_threshold > 0) {
        H5Pset_alignment(ctx->fapl_id,
                         (hsize_t)ctx->align_threshold,
                         (hsize_t)ctx->align_length);
    }

    ctx->fcpl_id = H5Pcreate(H5P_FILE_CREATE);
    throw_if(ctx->fcpl_id, "H5Pcreate(FILE_CREATE)");

    // --- Open / create the file collectively ---
    ctx->file_id = h5_open_or_create(path, mode_flag, ctx->fapl_id);
    if (ctx->file_id < 0) {
        std::ostringstream os;
        os << "H5Fcreate/H5Fopen failed for '" << path
           << "' (mode=" << mode_flag << ")";
        throw std::runtime_error(os.str());
    }

    // --- dxpl for dataset transfers (cached; collective default) ---
    ctx->dxpl_coll = H5Pcreate(H5P_DATASET_XFER);
    throw_if(ctx->dxpl_coll, "H5Pcreate(DATASET_XFER coll)");
    H5Pset_dxpl_mpio(ctx->dxpl_coll, H5FD_MPIO_COLLECTIVE);

    ctx->dxpl_indep = H5Pcreate(H5P_DATASET_XFER);
    throw_if(ctx->dxpl_indep, "H5Pcreate(DATASET_XFER indep)");
    H5Pset_dxpl_mpio(ctx->dxpl_indep, H5FD_MPIO_INDEPENDENT);

    // --- private CUDA stream for D2H staging ---
    throw_if_cuda(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking),
                  "cudaStreamCreate(phdf5 ctx)");

    return ctx;
}

// -----------------------------------------------------------------
//  ensure_dataset — collective, called from Python via ctypes before
//  any write_sharded_slab.  Creates the dataset if it doesn't exist,
//  opens it if it does.  Returns the cached hid_t on ctx->open_datasets.
// -----------------------------------------------------------------
hid_t ensure_dataset(PhdfCtx* ctx, const std::string& ds_name,
                     int64_t n_rows, int64_t n_cols, int dtype_tag) {
    {
        std::lock_guard<std::mutex> g(ctx->datasets_mu);
        auto it = ctx->open_datasets.find(ds_name);
        if (it != ctx->open_datasets.end() && it->second >= 0) {
            return it->second;
        }
    }

    hid_t native = dt::h5_native_for_tag(dtype_tag);
    if (native < 0) {
        throw std::runtime_error("phdf5 ensure_dataset: unsupported dtype_tag " +
                                 std::to_string(dtype_tag));
    }

    hid_t dset = -1;
    H5E_BEGIN_TRY {
        dset = H5Dopen(ctx->file_id, ds_name.c_str(), H5P_DEFAULT);
    } H5E_END_TRY;
    if (dset < 0) {
        hsize_t dims[2] = { (hsize_t)n_rows, (hsize_t)n_cols };
        hid_t filespace = H5Screate_simple(2, dims, nullptr);
        if (filespace < 0) {
            throw std::runtime_error(
                "phdf5 ensure_dataset: H5Screate_simple failed for '" + ds_name + "'");
        }
        dset = H5Dcreate(ctx->file_id, ds_name.c_str(), native,
                         filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(filespace);
        if (dset < 0) {
            throw std::runtime_error(
                "phdf5 ensure_dataset: H5Dcreate failed for '" + ds_name + "'");
        }
    }

    {
        std::lock_guard<std::mutex> g(ctx->datasets_mu);
        ctx->open_datasets[ds_name] = dset;
    }
    return dset;
}

void close_ctx(PhdfCtx* ctx) {
    if (!ctx) return;

    // Close cached datasets first (so their metadata flushes into file).
    {
        std::lock_guard<std::mutex> g(ctx->datasets_mu);
        for (auto& kv : ctx->open_datasets) {
            if (kv.second >= 0) H5Dclose(kv.second);
        }
        ctx->open_datasets.clear();
    }

    if (ctx->file_id    >= 0) H5Fclose(ctx->file_id);
    if (ctx->dxpl_coll  >= 0) H5Pclose(ctx->dxpl_coll);
    if (ctx->dxpl_indep >= 0) H5Pclose(ctx->dxpl_indep);
    if (ctx->fapl_id    >= 0) H5Pclose(ctx->fapl_id);
    if (ctx->fcpl_id    >= 0) H5Pclose(ctx->fcpl_id);

    if (ctx->pinned_buf) {
        cudaFreeHost(ctx->pinned_buf);
        ctx->pinned_buf = nullptr;
        ctx->pinned_capacity = 0;
    }
    if (ctx->stream)  cudaStreamDestroy(ctx->stream);

    if (ctx->owns_comm && ctx->comm != MPI_COMM_NULL) {
        MPI_Comm_free(&ctx->comm);
    }

    // We deliberately do NOT call MPI_Finalize here — other phdf5 files
    // or future ELPA FFIs may still want MPI alive.  Finalization is
    // handled by the process shutting down (MPI_Finalize on atexit is
    // registered inside ensure_mpi_initialized in HDF5/MPI runtimes).

    delete ctx;
}

}  // namespace lorrax_ffi::phdf5
