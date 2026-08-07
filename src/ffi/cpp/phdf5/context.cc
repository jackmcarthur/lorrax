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
//   LORRAX_PHDF5_STRIPE_COUNT  (policy) Lustre striping_factor hint; unset =
//                                       clamp(world_size, 4, 128)
//   LORRAX_PHDF5_STRIPE_SIZE_FS(policy) Lustre striping_unit hint (lfs form;
//                                       legacy byte-valued
//                                       LORRAX_PHDF5_STRIPE_SIZE honoured);
//                                       unset = the 1->4 MiB rank-count ramp.
//                                       Both policies transcribe
//                                       file_io/_slab_io_ffi._stripe_policy —
//                                       see stripe_policy_count/_unit below
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
#include "../common/mpi_thread_guard.h"

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
//
// Mechanism shared with the slate twin in cpp/common/mpi_thread_guard.h
// (2026-08-01 dedup); the hazard sentence and once-flag stay per family.
static void warn_if_thread_level_insufficient() {
    static std::atomic<bool> warned{false};
    lorrax_ffi::mpiguard::warn_if_thread_level_insufficient(
        "phdf5", "the phdf5 writer thread's collective MPI-IO", warned);
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

// ---------------------------------------------------------------------------
// The Lustre stripe POLICY for an unset LORRAX_PHDF5_STRIPE_{COUNT,SIZE_FS}.
// ---------------------------------------------------------------------------
//
// TRANSCRIPTION, not a second policy.  The one source of truth is
// ``file_io/_slab_io_ffi.py::_stripe_policy(nranks)`` — that docstring and
// the block comment above it carry every measurement (job 56389339,
// 2026-08-05) and the reasoning for both clamps.  Keep the two in step; a
// change to one without the other reinstates exactly the disagreement this
// pair replaced (Python: clamp(nranks, 4, 128) + a 1->4 MiB ramp; C++: the
// literals "16" and 1 MiB).
//
// Python, verbatim:
//     n     = max(1, nranks)
//     count = min(max(n, 4), 128)
//     unit  = 1 MiB; while unit < 4 MiB and 2*n*n > ((unit//1MiB)*32)**2:
//                        unit *= 2
// i.e. the power of two nearest IN LOG2 to (nranks/16) MiB, clamped to
// [1 MiB, 4 MiB]: 1 MiB below 16*sqrt(2) ~= 22.6 ranks, 2 MiB to
// 32*sqrt(2) ~= 45.3, 4 MiB above.  Integer arithmetic on both sides so
// neither can drift with libm.
static long stripe_policy_count(int world_size) {
    long n = world_size > 1 ? (long)world_size : 1L;
    if (n < 4) n = 4;
    if (n > 128) n = 128;
    return n;
}

static long stripe_policy_unit(int world_size) {
    const long n = world_size > 1 ? (long)world_size : 1L;
    const long kMin = 1L << 20, kMax = 4L << 20;
    long unit = kMin;
    while (unit < kMax) {
        const long anchor = (unit / kMin) * 32L;
        if (2L * n * n <= anchor * anchor) break;
        unit *= 2;
    }
    return unit;
}

// env_flag (the boolean grammar) moved to ctx.h on 2026-08-06 so every
// phdf5 translation unit shares it.  It was static here, which is why
// read_ffi.cc's LORRAX_PHDF5_TIME had to be a bare presence test.
// Boolean knobs used to go through env_long, whose strtol grammar
// silently kept the DEFAULT on word spellings — so e.g.
// LORRAX_PHDF5_COLLECTIVE_WRITES=false left the FFI writer collective
// while the Python phdf5_host writer went independent: the two writers
// diverged on the same environment (audit fix/zq 2026-07-28).

// Free one staging buffer through the allocator that made it.
static void staging_free(void* p) {
    if (!p) return;
#ifdef LORRAX_FFI_NO_CUDA
    std::free(p);
#else
    cudaFreeHost(p);
#endif
}

// Grow *slot to >= need_bytes.  H5Dread/H5Dwrite land in these buffers
// identically on both platforms; only the allocator differs — cudaMallocHost
// (page-locked, for fast async H2D) on the CUDA build vs a plain aligned host
// malloc on the host build, where the staging tail is a std::memcpy and
// page-locking would be pointless.
static bool ensure_staging(void** slot, size_t* capacity, size_t need_bytes,
                           const char* who) {
    if (*capacity >= need_bytes) return true;
    // Round up to a multiple of 2 MiB to reduce re-allocation churn across
    // transfers of slightly varying sizes.  The round-up is where an absurd
    // request stops being absurd and starts being SMALL: need_bytes near
    // SIZE_MAX wraps to a few bytes, the allocation succeeds, and the caller
    // then memsets/reads far past it.  Refuse before rounding.
    const size_t kChunk = (size_t)2 << 20;
    if (need_bytes > SIZE_MAX - (kChunk - 1)) {
        std::fprintf(stderr,
            "[phdf5 ERROR] %s: staging request of %zu bytes cannot be "
            "rounded without overflowing size_t\n", who, need_bytes);
        std::fflush(stderr);
        return false;
    }
    staging_free(*slot);
    *slot = nullptr;
    *capacity = 0;
    const size_t rounded = ((need_bytes + kChunk - 1) / kChunk) * kChunk;
#ifdef LORRAX_FFI_NO_CUDA
    // 64-byte aligned so the H5Dread lands on a cache-line boundary
    // (rounded is a 2-MiB multiple, so it satisfies aligned_alloc's
    // size-multiple-of-alignment requirement).
    *slot = std::aligned_alloc(64, rounded);
    if (*slot == nullptr) return false;
#else
    if (cudaMallocHost(slot, rounded) != cudaSuccess) {
        *slot = nullptr;
        return false;
    }
#endif
    *capacity = rounded;
    return true;
}

// WRITER-THREAD staging (ctx.h OWNERSHIP).
bool ensure_pinned(PhdfCtx* ctx, size_t need_bytes) {
#ifdef LORRAX_FFI_NO_CUDA
    // Host build: ``pinned_buf`` has exactly one producer, the writer thread
    // (H5Dwrite reads the XLA input buffer in place, so the write path never
    // stages).  Enforce it rather than assert it in a comment: the buffer's
    // whole safety argument is "one thread, one FIFO queue", and the audit's
    // item 4 was precisely a second, synchronous entry point that nothing
    // stopped anyone from adding.  Now the first such call fails, by name,
    // instead of corrupting a read months later.  Synchronous readers have
    // their own buffer — use ``ensure_read_buf``.
    const unsigned long long me =
        (unsigned long long)std::hash<std::thread::id>{}(
            std::this_thread::get_id());
    const unsigned long long owner = ctx->writer_tid_hash.load(
        std::memory_order_acquire);
    if (owner == 0 || owner != me) {
        std::fprintf(stderr,
            "[phdf5 ERROR rank=%d] ensure_pinned called off the writer "
            "thread.  ctx->pinned_buf is writer-thread-only on the host "
            "build (ctx.h OWNERSHIP); a synchronous handler that stages must "
            "use ensure_read_buf, or be routed through the writer queue.\n",
            ctx->rank);
        std::fflush(stderr);
        return false;
    }
#endif
    return ensure_staging(&ctx->pinned_buf, &ctx->pinned_capacity,
                          need_bytes, "ensure_pinned");
}

// SYNCHRONOUS-READER staging (ctx.h OWNERSHIP).
bool ensure_read_buf(PhdfCtx* ctx, size_t need_bytes) {
    return ensure_staging(&ctx->read_buf, &ctx->read_capacity,
                          need_bytes, "ensure_read_buf");
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

// Undo a PARTIALLY built ctx.
//
// ``open_ctx`` allocates a PhdfCtx and then creates, in order, an MPI_Info,
// four property lists, the file and (on CUDA) a stream and two events — and
// throws from a dozen places in between.  Every one of those throws used to
// leak the whole partial context: the ctx itself, whichever hid_t's already
// existed, and the MPI_Info.  Leaking an hid_t is not merely untidy here,
// because HDF5's library-wide id table then reports open objects at
// H5Fclose and the real error arrives wearing a second, spurious one.
// Called only before ``writer_thread`` is started, so there is no thread to
// join and no queue to drain.
static void abandon_partial_ctx(PhdfCtx* ctx, MPI_Info* info) noexcept {
    if (info && *info != MPI_INFO_NULL) MPI_Info_free(info);
    if (!ctx) return;
    if (ctx->file_id    >= 0) H5Fclose(ctx->file_id);
    if (ctx->dxpl_coll  >= 0) H5Pclose(ctx->dxpl_coll);
    if (ctx->dxpl_indep >= 0) H5Pclose(ctx->dxpl_indep);
    if (ctx->dcpl_id    >= 0) H5Pclose(ctx->dcpl_id);
    if (ctx->fapl_id    >= 0) H5Pclose(ctx->fapl_id);
    if (ctx->fcpl_id    >= 0) H5Pclose(ctx->fcpl_id);
#ifndef LORRAX_FFI_NO_CUDA
    if (ctx->d2h_event) cudaEventDestroy(ctx->d2h_event);
    if (ctx->h2d_event) cudaEventDestroy(ctx->h2d_event);
    if (ctx->stream)    cudaStreamDestroy(ctx->stream);
#endif
    delete ctx;
}

static PhdfCtx* open_ctx_impl(const std::string& path, int p, int q,
                              int rank, int world_size, int mode_flag,
                              PhdfCtx** ctx_out, MPI_Info* info_out);

PhdfCtx* open_ctx(const std::string& path, int p, int q,
                  int rank, int world_size, int mode_flag)
{
    PhdfCtx* ctx = nullptr;
    MPI_Info info = MPI_INFO_NULL;
    try {
        return open_ctx_impl(path, p, q, rank, world_size, mode_flag,
                             &ctx, &info);
    } catch (...) {
        abandon_partial_ctx(ctx, &info);
        throw;
    }
}

static PhdfCtx* open_ctx_impl(const std::string& path, int p, int q,
                              int rank, int world_size, int mode_flag,
                              PhdfCtx** ctx_out, MPI_Info* info_out)
{
    // Covers the thread-level guard too — it lives in
    // ensure_mpi_initialized now (see the note there), so the eager
    // lrx_phdf5_init_mpi path gets it as well.
    ensure_mpi_initialized();

    auto* ctx = new PhdfCtx{};
    *ctx_out = ctx;
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
    // Alignment is deliberately NOT tied to the striping_unit default.
    // It used to be justified as "matches the striping unit so H5 objects
    // start on stripe boundaries", but it is measured non-load-bearing on
    // this filesystem: at 16 x 1 MiB striping, ALIGN_MB of 4 / 1 / 0 gave
    // 0.830 / 0.809 / 0.813 GiB/s write at 1 node and 2.975 / 2.883 /
    // 2.915 at 4 nodes -- all inside the +-1.5% repeat noise (job
    // 56389339).  So it stays at 4 rather than becoming a second knob
    // that has to be kept in sync with a value it does not depend on.
    long align_mb        = env_long("LORRAX_PHDF5_ALIGN_MB", 4);
    // Clamp before the shift: a typo'd LORRAX_PHDF5_ALIGN_MB of 2^50 wraps
    // ``<< 20`` around size_t and hands H5Pset_alignment a small or absurd
    // threshold with no complaint.  16 GiB is far past any useful value.
    if (align_mb > (1L << 14)) align_mb = 1L << 14;
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
    *info_out = info;                       // so a throw below still frees it
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
    // The request is clamped to what Lustre grants — the banner prints
    // the request, `lfs getstripe` prints the truth.  No-ops if the file
    // inode already exists — which is why, on mode='w', rank 0 unlinks
    // the target before open in the Python writer, so the collective
    // create here sees a fresh inode and the striping hints apply.
    //
    // THE UNSET DEFAULT IS A FUNCTION OF THE RANK COUNT, and it is the
    // SAME function the Python writer resolves —
    // `stripe_policy_count`/`stripe_policy_unit` above transcribe
    // `file_io/_slab_io_ffi.py::_stripe_policy`, which owns the
    // measurements.  Until 2026-08-06 this side wrote the literals "16"
    // and 1 MiB while Python resolved `clamp(nranks, 4, 128)` and a
    // 1→4 MiB ramp.  In-tree that was masked, because
    // `_FfiBackend.__init__` calls `_export_striping_env()` immediately
    // before `_open_file()` and so the getenv below always found a
    // value; but anything reaching `ffi.io.open_file` DIRECTLY got the
    // old constant, silently, and TWO WRITERS REQUESTING DIFFERENT
    // LAYOUTS is worse than both being wrong the same way — it is the
    // bug class `env_flag` and the STRIPE_* refusals were introduced
    // for.  `world_size` here is the same `jax.process_count()` Python
    // passes `_stripe_policy`, so the two agree by construction now
    // rather than by call-ordering.
    //
    // Env naming: LORRAX_PHDF5_STRIPE_SIZE_FS is THE documented knob
    // (env_vars.md), spelled like `lfs setstripe -S` ("4M"); it is read
    // by the three Python sites and, since workstream AW, here too —
    // previously this file read only the undocumented byte-valued
    // LORRAX_PHDF5_STRIPE_SIZE, so the documented knob silently did not
    // reach the C++ writer.  The legacy byte spelling still works as a
    // fallback.
    // MEASURED DEFAULT (2026-08-05, job 56389339, /pscratch, 2.000 GiB
    // C128, GiB/s aggregate write/read, best of 2 reps, MPI world size
    // asserted):
    //                  1 node / 4 ranks    4 nodes / 16 ranks
    //   16 x 4 MiB      0.654 / 1.30        2.07 / 3.23   <- previous
    //   16 x 1 MiB      0.818 / 2.29        2.93 / 4.74   <- this default
    //   16 x 2 MiB      0.626 / 1.30        3.12 / 5.06   best at 4 nodes,
    //                                                     WORST at 1 node
    //   32 x 1 MiB      0.754 / 2.28        2.78 / 5.52
    //    8 x 1 MiB      0.867 / 2.28        2.52 / 3.53
    //    1 x 1 MiB      0.616 / 1.01        0.695 / 0.761  (fs default)
    // 16 x 1 MiB is the only layout that wins at BOTH geometries; 2 MiB
    // and 32-wide each win one and lose the other.  Count stays 16 for
    // the same reason.  The 1-stripe row is the /pscratch default and is
    // a PER-FILE single-OST ceiling near 0.65 GiB/s -- note it barely
    // moves from 4 to 16 ranks, so it is not the "~30 MB/s per rank" the
    // old lxrun comment claimed.  Always pre-stripe, or pass the hints.
    // Both stripe knobs REFUSE malformed input, naming the variable and the
    // grammar, because their Python siblings in file_io/_slab_io_ffi.py
    // (_stripe_count, _stripe_size_bytes) already do.  Until 2026-08-06 this
    // side did neither: STRIPE_COUNT was passed to ROMIO as an unvalidated
    // STRING (so `=sixteen` became the literal hint "striping_factor=sixteen"),
    // and STRIPE_SIZE_FS silently kept 1 MiB on a bad suffix -- the exact
    // "an explicit =4MiB A/B experiment quietly measured the default
    // configuration" failure its own sibling's docstring post-mortems.
    // Two writers that disagree about one environment is the bug class
    // env_flag was introduced for; leaving the numeric knobs split was the
    // half of that audit that did not land.
    if (const char* sc = std::getenv("LORRAX_PHDF5_STRIPE_COUNT");
        sc && *sc) {
        char* end = nullptr;
        const long parsed = std::strtol(sc, &end, 10);
        while (end && *end && std::isspace((unsigned char)*end)) ++end;
        if (end == sc || (end && *end)) {
            throw std::runtime_error(
                std::string("LORRAX_PHDF5_STRIPE_COUNT=\"") + sc +
                "\" is not a valid stripe count: expected a plain integer "
                "(e.g. 16; 0 disables the striping hints).  Refusing rather "
                "than handing ROMIO a hint it will ignore while the caller "
                "believes the layout was chosen.");
        }
        if (parsed < 0) {
            throw std::runtime_error(
                std::string("LORRAX_PHDF5_STRIPE_COUNT=\"") + sc +
                "\" is refused.  A negative striping_factor means 'every OST "
                "on the filesystem', the maximum-CONTENTION layout: 0.105 "
                "GiB/s at 64 ranks writing 32 GiB against 10.63 for the "
                "policy (job 56389339).  Pass a positive count, or unset it.");
        }
        info_set("striping_factor", sc);
    } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%ld",
                      stripe_policy_count(ctx->world_size));
        info_set("striping_factor", buf);
    }
    long stripe_bytes = stripe_policy_unit(ctx->world_size);
    if (const char* fs = std::getenv("LORRAX_PHDF5_STRIPE_SIZE_FS");
        fs && *fs) {
        char* end = nullptr;
        double v = std::strtod(fs, &end);
        long mult = 1;
        if (end == fs) {
            throw std::runtime_error(
                std::string("LORRAX_PHDF5_STRIPE_SIZE_FS=\"") + fs +
                "\" is not a valid stripe size: expected a number with an "
                "optional K/M/G suffix (e.g. 4M).");
        }
        if (end && *end) {
            switch (*end) {
                case 'k': case 'K': mult = 1L << 10; break;
                case 'm': case 'M': mult = 1L << 20; break;
                case 'g': case 'G': mult = 1L << 30; break;
                default: mult = 0; break;
            }
            // Exactly ONE suffix character, nothing after it.  Python's
            // _stripe_size_bytes tests raw[-1], so it refuses "4MiB"; if we
            // accepted it as 4 MiB the two writers would once again disagree
            // about one environment -- and "4MiB" is the very spelling that
            // docstring records someone actually exporting.
            const char* tail = end + 1;
            while (*tail && std::isspace((unsigned char)*tail)) ++tail;
            if (*tail) mult = 0;
        }
        if (mult == 0) {
            throw std::runtime_error(
                std::string("LORRAX_PHDF5_STRIPE_SIZE_FS=\"") + fs +
                "\" has an unrecognised suffix.  Accepted: K, M, G (e.g. "
                "4M), or a bare byte count.  Refusing rather than silently "
                "keeping the 1 MiB default while the knob looks in force.");
        }
        stripe_bytes = (long)(v * (double)mult);
        if (stripe_bytes <= 0) {
            throw std::runtime_error(
                std::string("LORRAX_PHDF5_STRIPE_SIZE_FS=\"") + fs +
                "\" resolves to a non-positive striping_unit, which is not a "
                "hint ROMIO can act on.");
        }
    } else if (const char* b = std::getenv("LORRAX_PHDF5_STRIPE_SIZE");
               b && *b) {
        char* end = nullptr;
        const long parsed = std::strtol(b, &end, 10);
        if (end == b || (end && *end) || parsed <= 0) {
            throw std::runtime_error(
                std::string("LORRAX_PHDF5_STRIPE_SIZE=\"") + b +
                "\" is not a valid byte count: expected a positive plain "
                "integer.");
        }
        stripe_bytes = parsed;
    }
    // Backstop only.  Every reachable way to get here non-positive now
    // throws above, naming the variable; this catches a v*mult that
    // overflowed long.  It is deliberately NOT the silent-default path it
    // used to be -- that is what let "=4MiB" measure the policy value.
    if (stripe_bytes <= 0) stripe_bytes = stripe_policy_unit(ctx->world_size);
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
    *info_out = MPI_INFO_NULL;              // handed to HDF5, ours no longer
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
        // Publish this thread's identity BEFORE any task can run; it is what
        // ``ensure_pinned`` checks on the host build (ctx.h OWNERSHIP).  A
        // task can only be picked up inside the loop below, so every legal
        // caller observes the published value.
        ctx->writer_tid_hash.store(
            (unsigned long long)std::hash<std::thread::id>{}(
                std::this_thread::get_id()),
            std::memory_order_release);
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
            // A task that escapes with an exception never sets its Promise,
            // so the Future never resolves and the rank hangs INSIDE the
            // collective its peers are waiting in — the failure signature
            // that cost 2026-08-02.  We cannot rescue the Promise from here
            // (it was moved into the task), so terminate is the honest
            // outcome; say what happened first, flushed, because a message
            // written as the process dies is otherwise lost under
            // srun+apptainer.
            try {
                task();
            } catch (const std::exception& e) {
                std::fprintf(stderr,
                    "[phdf5 ERROR rank=%d] writer-thread task threw: %s\n",
                    ctx->rank, e.what());
                std::fflush(stderr);
                throw;
            }
        }
    });

    return ctx;
}

// -----------------------------------------------------------------
//  ensure_dataset — collective, called from Python via ctypes before
//  any write_sharded_slab.  Creates the dataset if it doesn't exist,
//  opens it if it does.  Returns the cached hid_t on ctx->open_datasets.
// -----------------------------------------------------------------
// Format a dims vector for a refusal message: "[2,5,8]".
static std::string dims_to_string(const std::vector<hsize_t>& v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) { if (i) os << ","; os << v[i]; }
    os << "]";
    return os.str();
}

static std::string dims_to_string(const int64_t* p, int n) {
    std::ostringstream os;
    os << "[";
    for (int i = 0; i < n; ++i) { if (i) os << ","; os << p[i]; }
    os << "]";
    return os.str();
}

// Verify an EXISTING dataset against the requested geometry.
//
// decisions.md 2026-08-04, "Padding is SlabIO's business": identical
// logical shape and dtype => reuse (idempotent); anything else => REFUSE,
// naming both shapes.  Before this check H5Dopen simply succeeded and the
// write clipped against whatever extent the dataset happened to have, so an
// `mode='a'` rerun at a different mu silently wrote a PREFIX into the old
// geometry -- wrong physics, no symptom.  It is also what makes SlabIO's
// Python-side record of the dataset shape authoritative, which is where
// `valid_shape` is now derived from.
//
// Rank-invariant by construction: every input is a replicated control value
// and the file is the same file, so all ranks reach the same verdict.  A
// refusal on a proper subset of ranks inside a collective is a hang.
static void check_existing_geometry(
    hid_t dset, const std::string& ds_name,
    const int64_t* shape, int ndim, hid_t native)
{
    hid_t space = H5Dget_space(dset);
    if (space < 0) {
        throw std::runtime_error(
            "phdf5 ensure_dataset: H5Dget_space failed for '" + ds_name + "'");
    }
    int have_ndim = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> have((size_t)(have_ndim > 0 ? have_ndim : 0), 0);
    if (have_ndim > 0 &&
        H5Sget_simple_extent_dims(space, have.data(), nullptr) < 0) {
        H5Sclose(space);
        throw std::runtime_error(
            "phdf5 ensure_dataset: H5Sget_simple_extent_dims failed for '"
            + ds_name + "'");
    }
    H5Sclose(space);

    bool shape_ok = (have_ndim == ndim);
    if (shape_ok) {
        for (int i = 0; i < ndim; ++i) {
            if (have[(size_t)i] != (hsize_t)shape[i]) { shape_ok = false; break; }
        }
    }

    hid_t have_type = H5Dget_type(dset);
    htri_t same_type = (have_type < 0) ? -1 : H5Tequal(have_type, native);
    if (have_type >= 0) H5Tclose(have_type);

    if (shape_ok && same_type > 0) return;

    std::ostringstream os;
    os << "phdf5 ensure_dataset: dataset '" << ds_name
       << "' already exists with shape " << dims_to_string(have)
       << (same_type > 0 ? "" : " and a DIFFERENT dtype")
       << ", but was requested at shape " << dims_to_string(shape, ndim)
       << ".  SlabIO will neither delete-and-recreate it (data loss) nor "
          "write into the previous geometry (wrong extent, no symptom).  "
          "Open with mode='w', use a different dataset name, or delete the "
          "file.  Refused identically on every rank.";
    throw std::runtime_error(os.str());
}

hid_t ensure_dataset(PhdfCtx* ctx, const std::string& ds_name,
                     const int64_t* shape, int ndim, int dtype_tag) {
    hid_t native = dt::h5_native_for_tag(dtype_tag);
    if (native < 0) {
        throw std::runtime_error("phdf5 ensure_dataset: unsupported dtype_tag " +
                                 std::to_string(dtype_tag));
    }
    if (ndim <= 0) {
        throw std::runtime_error("phdf5 ensure_dataset: ndim must be >= 1");
    }
    if (ndim > H5S_MAX_RANK) {
        throw std::runtime_error(
            "phdf5 ensure_dataset: ndim " + std::to_string(ndim) +
            " exceeds HDF5's maximum dataspace rank " +
            std::to_string((int)H5S_MAX_RANK));
    }
    if (shape == nullptr) {
        throw std::runtime_error("phdf5 ensure_dataset: shape is null");
    }
    // A negative extent becomes a colossal hsize_t two lines further down;
    // H5Screate_simple then fails with a message about a shape nobody asked
    // for.  Name the actual mistake.  Rank-invariant: replicated arguments.
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] < 0) {
            throw std::runtime_error(
                "phdf5 ensure_dataset: negative extent " +
                std::to_string((long long)shape[i]) + " at dim " +
                std::to_string(i) + " of '" + ds_name + "'");
        }
    }

    // Cached hid: still CHECK it.  A second ensure_dataset for the same
    // name at a different shape would otherwise take the cache hit and
    // return a handle to geometry the caller does not think it has.
    {
        std::lock_guard<std::mutex> g(ctx->datasets_mu);
        auto it = ctx->open_datasets.find(ds_name);
        if (it != ctx->open_datasets.end() && it->second >= 0) {
            check_existing_geometry(it->second, ds_name, shape, ndim, native);
            return it->second;
        }
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
    } else {
        // Pre-existing on disk: reuse only if the geometry matches.  On a
        // refusal close the handle we just opened -- the throw skips every
        // caller-side cleanup, and leaking an hid_t here means H5Fclose
        // later reports an open-object error on top of the real one.
        try {
            check_existing_geometry(dset, ds_name, shape, ndim, native);
        } catch (...) {
            H5Dclose(dset);
            throw;
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
    //
    // These returns were dropped on the floor.  H5Dclose/H5Fclose are where a
    // collective write's metadata actually reaches the file: a failure here
    // means the run finishes rc=0 with a truncated or unflushed dataset and
    // NOTHING says so.  close_ctx cannot throw (it is called from an extern
    // "C" void, and from atexit), so announce — flushed, immediately.
    auto closed_ok = [ctx](herr_t st, const char* what) {
        if (st >= 0) return;
        std::fprintf(stderr,
            "[phdf5 ERROR rank=%d] %s failed while closing '%s'; data may "
            "not have reached the file\n",
            ctx->rank, what, ctx->path.c_str());
        std::fflush(stderr);
    };
    {
        std::lock_guard<std::mutex> g(ctx->datasets_mu);
        for (auto& kv : ctx->open_datasets) {
            if (kv.second >= 0) closed_ok(H5Dclose(kv.second), "H5Dclose");
        }
        ctx->open_datasets.clear();
    }

    if (ctx->file_id    >= 0) closed_ok(H5Fclose(ctx->file_id), "H5Fclose");
    if (ctx->dxpl_coll  >= 0) H5Pclose(ctx->dxpl_coll);
    if (ctx->dxpl_indep >= 0) H5Pclose(ctx->dxpl_indep);
    if (ctx->dcpl_id    >= 0) H5Pclose(ctx->dcpl_id);
    if (ctx->fapl_id    >= 0) H5Pclose(ctx->fapl_id);
    if (ctx->fcpl_id    >= 0) H5Pclose(ctx->fcpl_id);

    staging_free(ctx->pinned_buf);
    ctx->pinned_buf = nullptr;
    ctx->pinned_capacity = 0;
    staging_free(ctx->read_buf);
    ctx->read_buf = nullptr;
    ctx->read_capacity = 0;
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
