// context.cc — open/close for the parallel-HDF5 FFI.
//
// open_file: collective MPI_Init_thread + H5Fcreate/H5Fopen with cached
// property lists.
//
// Env tunables (all optional; docs/dev/env_vars.md is the registry).
// Boolean knobs accept the shared writer grammar — see env_flag below
// (mirrors Python's file_io/_slab_io_mpi_host._env_flag exactly):
//   LORRAX_PHDF5_COLLECTIVE_WRITES (1)  collective vs independent writes
//   LORRAX_PHDF5_DEDUP_REPLICAS   (1)   one canonical writer per replica
//   LORRAX_PHDF5_INDEPENDENT      (0)   force independent READS
//   LORRAX_PHDF5_COLL_META        (0)   collective metadata ops
//   LORRAX_PHDF5_ALIGN_MB         (4)   H5Pset_alignment threshold/length
//   LORRAX_PHDF5_STRIPE_COUNT     (16)  Lustre striping_factor hint
//   LORRAX_PHDF5_STRIPE_SIZE_FS   (4M)  Lustre striping_unit hint (lfs form;
//                                       legacy byte-valued
//                                       LORRAX_PHDF5_STRIPE_SIZE honoured)
//   LORRAX_PHDF5_CB_WRITE / _DS_WRITE / _CB_NODES / _CB_BUFFER_SIZE /
//   _CB_PER_NODE                (unset) ROMIO pass-throughs; unset = ROMIO
//                                       automatic policy (measured best on
//                                       Frontera — scorecard AI/AW)
//   LORRAX_PHDF5_DUMP_HINTS       (0)   print the hints ROMIO retained

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef LORRAX_FFI_NO_CUDA
#include <cuda_runtime.h>
#endif
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
#ifndef LORRAX_FFI_NO_CUDA
static void throw_if_cuda(cudaError_t st, const char* what) {
    if (st != cudaSuccess) {
        std::ostringstream os;
        os << what << ": " << cudaGetErrorName(st) << " (" << cudaGetErrorString(st) << ")";
        throw std::runtime_error(os.str());
    }
}
#endif

// One-shot MPI init.  cuSOLVERMp doesn't use MPI, so if the caller
// only ever touches cuSOLVERMp there's no MPI_Init surprise.  Either
// first open_file triggers MPI_Init_thread(THREAD_MULTIPLE), or the
// caller triggers it eagerly via the exported ``lrx_phdf5_init_mpi``
// (takes ~400 ms on first call; calling it during program startup
// before the hot path saves that time on the critical path).
// GUARD (scorecard AS.4b; twin in cpp/slate/context.cc).  This ctx runs a
// dedicated writer thread whose H5Dwrite drives MPI-IO collectives
// concurrently with any main-thread / XLA-thread MPI traffic.  That
// concurrency is only DEFINED at MPI_THREAD_MULTIPLE.  When *we* init MPI we
// request MULTIPLE — but when someone else got there first (jax's MPI CPU
// collectives via MPItrampoline request FUNNELED; an unpatched MPIwrapper
// grants it), the granted level is whatever they asked for, and the measured
// consequence at P=16 x 8 nodes was a ~29% provider-independent segfault/hang
// rate at the zeta-write/V_q boundary — two threads of one rank concurrently
// inside MPID_Progress_wait.  The certified fix is the THREAD_MULTIPLE-
// patched MPIwrapper; this guard exists so the hazardous configuration
// announces itself up front instead of dying minutes later with a backtrace
// that names neither cause nor fix.
//
// MOVED HERE from inside open_ctx (2026-07-30 FFI divergence audit).  It ran
// only on the first open_file, so the documented eager-init entry point
// `lrx_phdf5_init_mpi` (api.cc:74) — whose entire purpose is to init MPI
// early, at startup, off the critical path — took the hazardous decision and
// said nothing.  Sitting in ensure_mpi_initialized it now covers every way
// this library can reach MPI, and it queries on BOTH paths: after we
// initialise AND after an early return because someone else already did.
// The already-initialised path is the hazardous one, so a guard that only
// ran when we called MPI_Init_thread ourselves would be void by
// construction.  Rank now comes from MPI_Comm_rank rather than open_ctx's
// argument, which is what let it live at the lower level.
static void warn_if_thread_level_insufficient() {
    static std::atomic<bool> warned{false};
    int provided = 0;
    MPI_Query_thread(&provided);
    if (provided >= MPI_THREAD_MULTIPLE) return;
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return;
    if (warned.exchange(true)) return;
    std::fprintf(stderr,
        "[phdf5] WARNING: MPI granted thread level %d < "
        "MPI_THREAD_MULTIPLE (%d) and MPI was initialized before "
        "this library (jax cpu_collectives_implementation=mpi with "
        "an unpatched MPIwrapper?).  The phdf5 writer thread's "
        "collective MPI-IO is UNDEFINED at this level — measured "
        "~29%% multi-node crash rate (scorecard AS.4b).  Fix: "
        "point MPITRAMPOLINE_LIB at the MPIwrapper built by "
        "config/frontera/build_mpiwrapper.sh, which upgrades every "
        "init to MPI_THREAD_MULTIPLE (docs/dev/mpi_collectives.md).\n",
        provided, MPI_THREAD_MULTIPLE);
    std::fflush(stderr);
}

void ensure_mpi_initialized() {
    int inited = 0;
    MPI_Initialized(&inited);
    if (!inited) {
        int provided = 0;
        // MPI_THREAD_MULTIPLE is required: the dedicated writer thread
        // (ctx->writer_thread, started in open_ctx) calls H5Dwrite which
        // internally drives MPI-IO collectives concurrently with any
        // main-thread MPI calls.  `provided` is NOT inspected here on
        // purpose — MPI_Query_thread below answers the same question on
        // both paths, and reading it only here would miss the hazardous
        // case entirely (someone else initialised first).
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        // Do NOT register MPI_Finalize via atexit — HDF5 calls MPI at its
        // own destructors; ordering is fragile.  Our close_file path calls
        // MPI_Finalize explicitly when the last file closes (via a ref count).
    }
    warn_if_thread_level_insufficient();
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

// Pull a BOOLEAN tunable from env.  Grammar is defined in ONE place —
// Python's ``file_io/_slab_io_mpi_host._env_flag`` — and mirrored here
// exactly (keep the two in sync):
//   unset or empty                                  -> default
//   strip + lowercase in {"1", "true", "yes", "on"} -> true
//   anything else ("0", "false", "no", "off", ...)  -> false
// Boolean knobs used to go through env_long, whose strtol grammar
// silently kept the DEFAULT on word spellings — so e.g.
// LORRAX_PHDF5_COLLECTIVE_WRITES=false left the FFI writer collective
// while the Python phdf5_host writer went independent: the two writers
// diverged on the same environment (audit fix/zq 2026-07-28).
static bool env_flag(const char* name, bool default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    std::string s(v);
    // strip (all-whitespace strips to "" -> false, matching Python, where
    // only unset / exactly-empty return the default)
    const auto b = s.find_first_not_of(" \t\r\n");
    const auto e = s.find_last_not_of(" \t\r\n");
    s = (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

// Grow the staging buffer to >= need_bytes.  The read core H5Dreads into
// this buffer identically on both platforms; only the allocator differs —
// cudaMallocHost (page-locked, for fast async H2D) on the CUDA build vs a
// plain aligned host malloc on the host build, where the staging tail is a
// std::memcpy and page-locking would be pointless.
bool ensure_pinned(PhdfCtx* ctx, size_t need_bytes) {
    if (ctx->pinned_capacity >= need_bytes) return true;
    if (ctx->pinned_buf) {
#ifdef LORRAX_FFI_NO_CUDA
        std::free(ctx->pinned_buf);
#else
        cudaFreeHost(ctx->pinned_buf);
#endif
        ctx->pinned_buf = nullptr;
        ctx->pinned_capacity = 0;
    }
    // Round up to a multiple of 2 MiB to reduce re-allocation churn
    // across writes of slightly varying sizes.
    size_t rounded = ((need_bytes + (2 << 20) - 1) / (2 << 20)) * (2 << 20);
#ifdef LORRAX_FFI_NO_CUDA
    // 64-byte aligned so the H5Dread lands on a cache-line boundary
    // (rounded is a 2-MiB multiple, so it satisfies aligned_alloc's
    // size-multiple-of-alignment requirement).
    ctx->pinned_buf = std::aligned_alloc(64, rounded);
    if (ctx->pinned_buf == nullptr) {
        ctx->pinned_capacity = 0;
        return false;
    }
#else
    if (cudaMallocHost(&ctx->pinned_buf, rounded) != cudaSuccess) {
        ctx->pinned_buf = nullptr;
        ctx->pinned_capacity = 0;
        return false;
    }
#endif
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
    // Covers the thread-level guard too — it lives in
    // ensure_mpi_initialized now (see the note there), so the eager
    // lrx_phdf5_init_mpi path gets it as well.
    ensure_mpi_initialized();

    auto* ctx = new PhdfCtx{};
    ctx->path = path;
    ctx->p = p;
    ctx->q = q;
    ctx->rank = rank;
    ctx->world_size = world_size;

    // Env-driven tuning.  Defaults (see ctx.h): reads collective, writes
    // COLLECTIVE (default flipped 2026-07-27 to match the Python
    // phdf5_host writer, so LORRAX_PHDF5_COLLECTIVE_WRITES means the
    // same thing in every writer; measured on the production tile
    // geometry: strided 2-D tiles of a contiguous dataset are ~3 orders
    // faster under two-phase collective aggregation — scorecard AI),
    // metadata non-collective.  Env overrides:
    //   LORRAX_PHDF5_INDEPENDENT=1       → also force reads independent
    //   LORRAX_PHDF5_COLLECTIVE_WRITES=0 → back to independent writes
    //                                     (historical Cray-MPICH caution:
    //                                     ad_cray_write_coll.c OOM at
    //                                     >~1 GB/rank on that stack)
    //   LORRAX_PHDF5_COLL_META=1         → re-enable collective metadata
    //   LORRAX_PHDF5_DEDUP_REPLICAS=0    → let every rank of a replica
    //                                     group write its identical copy
    //                                     (overlapping selections are
    //                                     undefined under collective
    //                                     MPI-IO — only for debugging)
    // Boolean knobs parse through env_flag — the SAME grammar as the
    // Python writers' _env_flag (see env_flag above), so a given
    // environment means the same thing in every writer.
    const bool force_indep_read    = env_flag("LORRAX_PHDF5_INDEPENDENT", false);
    const bool coll_write          = env_flag("LORRAX_PHDF5_COLLECTIVE_WRITES", true);
    const bool force_coll_metadata = env_flag("LORRAX_PHDF5_COLL_META", false);
    ctx->use_collective_read  = !force_indep_read;
    ctx->use_collective_write = coll_write;
    ctx->coll_metadata        = force_coll_metadata;
    ctx->dedup_replicas       = env_flag("LORRAX_PHDF5_DEDUP_REPLICAS", true);
    // Alignment default matches the new striping_unit default (4 MiB) so
    // H5 objects start on Lustre stripe boundaries.
    long align_mb        = env_long("LORRAX_PHDF5_ALIGN_MB", 4);
    ctx->align_threshold = (align_mb > 0) ? (size_t)align_mb << 20 : 0;
    ctx->align_length    = ctx->align_threshold;

    // Pass MPI_COMM_WORLD straight through to H5Pset_fapl_mpio.  HDF5's
    // MPI-IO VFD takes its OWN ``MPI_Comm_dup`` of the comm internally
    // (per H5Pset_fapl_mpio docs) and frees it on file close, so an
    // outer dup here is redundant work.  It has also proven fragile in
    // the Shifter container: the dup landed in HPC-X OpenMPI's
    // ``ompi_comm_dup_with_info`` → UCX path on a 2026-05-10 allocation
    // where ``--module=mpich`` failed to fully shadow HPC-X with Cray
    // MPICH, segfaulting before any HDF5 code ran.  With the dup gone
    // we never reach OpenMPI's PMIx-derived endpoint state in this
    // open path; HDF5's own internal dup happens later under the
    // VFD's normal init sequence which has been robust.  See
    // KNOWN_SANDBOX_ERRORS.md (2026-05-10) for the underlying
    // ``--module=mpich`` shadowing investigation.
    ctx->comm = MPI_COMM_WORLD;
    ctx->owns_comm = false;

    // --- MPI_Info hints for ROMIO/MPI-IO.  POLICY (aligned with the
    // Python writer `_slab_io_mpi_host._mpi_io_hints`, 2026-07-27
    // workstream AW): the STRIPE hints are set by default — they are the
    // measured lever (scorecard AI: collective+stripe 74 → 2066 MB/s on
    // the production tile; and `lfs` does not exist inside the apptainer
    // image, so these hints are the ONLY way the layout gets set on
    // Frontera).  Everything else — romio_cb_write / romio_ds_write /
    // cb_buffer_size / cb_nodes — is left at ROMIO's automatic policy
    // unless the env asks: under H5FD_MPIO_COLLECTIVE ROMIO already
    // runs two-phase aggregation, and FORCING romio_cb_write=enable on
    // top of it measured *slower* than ROMIO's own choice (wk_AI, P=16
    // production tile: 1826 vs 2066 MB/s).  The old defaults here
    // (cb_write=enable, ds_write=disable, cb_buffer_size=64 MiB,
    // cb_nodes=world_size) were a Perlmutter/OpenMPI-era tuning
    // (0.85 → 4.4 GB/s on THAT stack); they are preserved as env
    // escape hatches, not defaults, so unset-env now means the same
    // thing in every writer: ROMIO decides.
    MPI_Info info = MPI_INFO_NULL;
    MPI_Info_create(&info);
    auto info_set = [&](const char* key, const char* val) {
        MPI_Info_set(info, const_cast<char*>(key), const_cast<char*>(val));
    };
    // Pass-through knobs (set only when the env is non-empty).
    const char* cb_write = std::getenv("LORRAX_PHDF5_CB_WRITE");
    if (cb_write && *cb_write) info_set("romio_cb_write", cb_write);
    const char* ds_write = std::getenv("LORRAX_PHDF5_DS_WRITE");
    if (ds_write && *ds_write) info_set("romio_ds_write", ds_write);
    const char* cb_buf = std::getenv("LORRAX_PHDF5_CB_BUFFER_SIZE");
    if (cb_buf && *cb_buf) info_set("cb_buffer_size", cb_buf);
    const char* cb_nodes = std::getenv("LORRAX_PHDF5_CB_NODES");
    if (cb_nodes && *cb_nodes) info_set("cb_nodes", cb_nodes);
    const char* cb_per_node = std::getenv("LORRAX_PHDF5_CB_PER_NODE");
    if (cb_per_node && *cb_per_node) {
        std::string v = std::string("*:") + cb_per_node;
        info_set("cb_config_list", v.c_str());
    }
    // striping_factor / striping_unit: MPI-IO's way to request a Lustre
    // stripe layout when it creates the file (ROMIO applies it through
    // llapi — works inside the container with no `lfs` binary on PATH).
    // Default 16 x 4 MiB (scorecard AI: +57% on top of collective; the
    // request is clamped to what Lustre grants — the banner prints the
    // request, `lfs getstripe` prints the truth).  No-ops if the file
    // inode already exists — which is why, on mode='w', rank 0 unlinks
    // the target before open in both Python backends, so the collective
    // create here sees a fresh inode and the striping hints apply.
    //
    // Env naming: LORRAX_PHDF5_STRIPE_SIZE_FS is THE documented knob
    // (env_vars.md), spelled like `lfs setstripe -S` ("4M"); it is read
    // by the three Python sites and, since workstream AW, here too —
    // previously this file read only the undocumented byte-valued
    // LORRAX_PHDF5_STRIPE_SIZE, so the documented knob silently did not
    // reach the C++ writer.  The legacy byte spelling still works as a
    // fallback.
    const char* stripe_count = std::getenv("LORRAX_PHDF5_STRIPE_COUNT");
    info_set("striping_factor", stripe_count && *stripe_count ? stripe_count : "16");
    long stripe_bytes = 4L << 20;
    if (const char* fs = std::getenv("LORRAX_PHDF5_STRIPE_SIZE_FS");
        fs && *fs) {
        char* end = nullptr;
        double v = std::strtod(fs, &end);
        long mult = 1;
        if (end && *end) {
            switch (*end) {
                case 'k': case 'K': mult = 1L << 10; break;
                case 'm': case 'M': mult = 1L << 20; break;
                case 'g': case 'G': mult = 1L << 30; break;
                default: mult = 0; break;  // unparseable -> keep default
            }
        }
        if (end != fs && mult > 0) stripe_bytes = (long)(v * (double)mult);
    } else if (const char* b = std::getenv("LORRAX_PHDF5_STRIPE_SIZE");
               b && *b) {
        long parsed = std::strtol(b, nullptr, 10);
        if (parsed > 0) stripe_bytes = parsed;
    }
    {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%ld", stripe_bytes);
        info_set("striping_unit", buf);
    }

    // --- fapl: MPI-IO + (optional) collective metadata + alignment ---
    ctx->fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    throw_if(ctx->fapl_id, "H5Pcreate(FILE_ACCESS)");
    if (H5Pset_fapl_mpio(ctx->fapl_id, ctx->comm, info) < 0) {
        throw std::runtime_error("H5Pset_fapl_mpio failed");
    }
    MPI_Info_free(&info);
    // Use the latest HDF5 format version — enables modern layout and avoids
    // the 2 GB dataset size cap in the legacy format.  Recommended by NERSC.
    H5Pset_libver_bounds(ctx->fapl_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
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

    // --- dcpl: avoid the implicit "fill the dataset with zeros on create"
    // step that HDF5 defaults to.  Without this, every H5Dcreate triggers
    // a silent N-byte zero-fill write BEFORE our H5Dwrite lands — which
    // doubles I/O and costs wall time.  Pair with ALLOC_TIME_EARLY so
    // the file extent is reserved up front (contiguous, stripe-aligned).
    ctx->dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    throw_if(ctx->dcpl_id, "H5Pcreate(DATASET_CREATE)");
    H5Pset_fill_time(ctx->dcpl_id, H5D_FILL_TIME_NEVER);
    H5Pset_alloc_time(ctx->dcpl_id, H5D_ALLOC_TIME_EARLY);

    // --- Open / create the file collectively ---
    ctx->file_id = h5_open_or_create(path, mode_flag, ctx->fapl_id);
    if (ctx->file_id < 0) {
        std::ostringstream os;
        os << "H5Fcreate/H5Fopen failed for '" << path
           << "' (mode=" << mode_flag << ")";
        throw std::runtime_error(os.str());
    }

    // --- diagnostic: dump the MPI_Info the MPI-IO driver actually
    //     retained for this file (rank 0 only).  Opt-in via env so it
    //     doesn't clutter normal output; indispensable when tuning
    //     Lustre/ROMIO performance because ROMIO silently ignores hints
    //     it doesn't understand or that the filesystem overrides.
    if (ctx->rank == 0 && env_flag("LORRAX_PHDF5_DUMP_HINTS", false)) {
        void* vfd_handle = nullptr;
        if (H5Fget_vfd_handle(ctx->file_id, ctx->fapl_id, &vfd_handle) >= 0
            && vfd_handle != nullptr) {
            MPI_File mpi_fh = *static_cast<MPI_File*>(vfd_handle);
            MPI_Info info_out = MPI_INFO_NULL;
            if (MPI_File_get_info(mpi_fh, &info_out) == MPI_SUCCESS) {
                int nkeys = 0;
                MPI_Info_get_nkeys(info_out, &nkeys);
                std::fprintf(stderr,
                    "[phdf5.hints %s] %d hints retained by ROMIO:\n",
                    path.c_str(), nkeys);
                for (int k = 0; k < nkeys; ++k) {
                    char key[MPI_MAX_INFO_KEY + 1] = {0};
                    MPI_Info_get_nthkey(info_out, k, key);
                    int vlen = 0, flag = 0;
                    MPI_Info_get_valuelen(info_out, key, &vlen, &flag);
                    std::string val(vlen + 1, '\0');
                    MPI_Info_get(info_out, key, vlen + 1, &val[0], &flag);
                    val.resize(vlen);
                    std::fprintf(stderr, "  %-40s = %s\n", key, val.c_str());
                }
                std::fflush(stderr);
                MPI_Info_free(&info_out);
            }
        }
    }

    // --- dxpl for dataset transfers (cached; collective default) ---
    ctx->dxpl_coll = H5Pcreate(H5P_DATASET_XFER);
    throw_if(ctx->dxpl_coll, "H5Pcreate(DATASET_XFER coll)");
    H5Pset_dxpl_mpio(ctx->dxpl_coll, H5FD_MPIO_COLLECTIVE);

    ctx->dxpl_indep = H5Pcreate(H5P_DATASET_XFER);
    throw_if(ctx->dxpl_indep, "H5Pcreate(DATASET_XFER indep)");
    H5Pset_dxpl_mpio(ctx->dxpl_indep, H5FD_MPIO_INDEPENDENT);

#ifndef LORRAX_FFI_NO_CUDA
    // --- private CUDA stream for D2H staging ---
    // Record the current device so the writer_thread can bind to the
    // same one; GPU pointers allocated under the main thread's JAX
    // context aren't recognised on a different device and
    // cudaMemcpyAsync returns cudaErrorMemoryAllocation.
    throw_if_cuda(cudaGetDevice(&ctx->cuda_device),
                  "cudaGetDevice(phdf5 ctx)");
    throw_if_cuda(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking),
                  "cudaStreamCreate(phdf5 ctx)");

    // --- reusable D2H/H2D completion events (see ctx.h) ---
    throw_if_cuda(cudaEventCreateWithFlags(&ctx->d2h_event, cudaEventDisableTiming),
                  "cudaEventCreate(phdf5 d2h_event)");
    throw_if_cuda(cudaEventCreateWithFlags(&ctx->h2d_event, cudaEventDisableTiming),
                  "cudaEventCreate(phdf5 h2d_event)");
#endif

    // --- start dedicated writer thread (FIFO task queue) ---
    // One thread per ctx drains ``task_queue`` in FIFO order so every
    // rank enters the H5Dwrite/H5Dread MPI-IO collectives in the same
    // program order (the collective correctness requirement).  On the
    // CUDA build the worker must attach to the CUDA primary context
    // before any task runs — read tasks call cudaMemcpyAsync H2D after
    // the H5Dread; without cudaSetDevice the runtime returns
    // cudaErrorMemoryAllocation on the first CUDA call from this thread.
    // On the host build there is no device to bind and the staging tail
    // is a plain memcpy.
    ctx->writer_thread = std::thread([ctx]() {
#ifndef LORRAX_FFI_NO_CUDA
        cudaSetDevice(ctx->cuda_device);
#endif
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(ctx->queue_mu);
                ctx->queue_cv.wait(lk, [&]{
                    return ctx->shutdown_flag || !ctx->task_queue.empty();
                });
                if (ctx->task_queue.empty()) return;
                task = std::move(ctx->task_queue.front());
                ctx->task_queue.pop_front();
            }
            task();
        }
    });

    return ctx;
}

// -----------------------------------------------------------------
//  ensure_dataset — collective, called from Python via ctypes before
//  any write_sharded_slab.  Creates the dataset if it doesn't exist,
//  opens it if it does.  Returns the cached hid_t on ctx->open_datasets.
// -----------------------------------------------------------------
hid_t ensure_dataset(PhdfCtx* ctx, const std::string& ds_name,
                     const int64_t* shape, int ndim, int dtype_tag) {
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
    if (ndim <= 0) {
        throw std::runtime_error("phdf5 ensure_dataset: ndim must be >= 1");
    }

    hid_t dset = -1;
    H5E_BEGIN_TRY {
        dset = H5Dopen(ctx->file_id, ds_name.c_str(), H5P_DEFAULT);
    } H5E_END_TRY;
    if (dset < 0) {
        std::vector<hsize_t> dims((size_t)ndim);
        for (int i = 0; i < ndim; ++i) dims[i] = (hsize_t)shape[i];
        hid_t filespace = H5Screate_simple(ndim, dims.data(), nullptr);
        if (filespace < 0) {
            throw std::runtime_error(
                "phdf5 ensure_dataset: H5Screate_simple failed for '" + ds_name + "'");
        }
        dset = H5Dcreate(ctx->file_id, ds_name.c_str(), native,
                         filespace, H5P_DEFAULT, ctx->dcpl_id, H5P_DEFAULT);
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

// -----------------------------------------------------------------
//  open_dataset_ro — collective, called from Python before any
//  read_sharded_slab.  Pure H5Dopen (no create); caches the hid_t
//  on ctx->open_datasets so repeat calls are cheap.
// -----------------------------------------------------------------
hid_t open_dataset_ro(PhdfCtx* ctx, const std::string& ds_name) {
    {
        std::lock_guard<std::mutex> g(ctx->datasets_mu);
        auto it = ctx->open_datasets.find(ds_name);
        if (it != ctx->open_datasets.end() && it->second >= 0) {
            return it->second;
        }
    }

    hid_t dset = H5Dopen(ctx->file_id, ds_name.c_str(), H5P_DEFAULT);
    if (dset < 0) {
        throw std::runtime_error(
            "phdf5 open_dataset_ro: H5Dopen failed for '" + ds_name + "'");
    }

    {
        std::lock_guard<std::mutex> g(ctx->datasets_mu);
        ctx->open_datasets[ds_name] = dset;
    }
    return dset;
}

void close_ctx(PhdfCtx* ctx) {
    if (!ctx) return;

    // Drain and stop the writer thread first — any pending H5Dwrite
    // must complete before we H5Dclose/H5Fclose below.  Setting
    // shutdown_flag with the queue non-empty still exits the loop
    // cleanly after remaining tasks run because the loop only checks
    // for shutdown when the queue is empty.
    if (ctx->writer_thread.joinable()) {
        {
            std::lock_guard<std::mutex> lk(ctx->queue_mu);
            ctx->shutdown_flag = true;
        }
        ctx->queue_cv.notify_all();
        ctx->writer_thread.join();
    }

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
    if (ctx->dcpl_id    >= 0) H5Pclose(ctx->dcpl_id);
    if (ctx->fapl_id    >= 0) H5Pclose(ctx->fapl_id);
    if (ctx->fcpl_id    >= 0) H5Pclose(ctx->fcpl_id);

    if (ctx->pinned_buf) {
#ifdef LORRAX_FFI_NO_CUDA
        std::free(ctx->pinned_buf);
#else
        cudaFreeHost(ctx->pinned_buf);
#endif
        ctx->pinned_buf = nullptr;
        ctx->pinned_capacity = 0;
    }
#ifndef LORRAX_FFI_NO_CUDA
    if (ctx->d2h_event) cudaEventDestroy(ctx->d2h_event);
    if (ctx->h2d_event) cudaEventDestroy(ctx->h2d_event);
    if (ctx->stream)    cudaStreamDestroy(ctx->stream);
#endif

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
