// ctx.h — per-file context for the parallel-HDF5 FFI.
//
// Created at open_file time on a collective MPI + H5Fcreate path,
// reused across every write to that file, torn down on close_file.
// One PhdfCtx per open file.  openPMD-api pattern: all property lists
// are cached at open time and reused; no per-call churn.

#pragma once

#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

// LORRAX_FFI_NO_CUDA is defined by the host-platform build (see
// cpp/CMakeLists.txt).  The collective MPI-IO read core in
// read_ffi.cc / context.cc is hardware-agnostic; only the device-staging
// tail (H2D vs a plain host memcpy) and the staging-buffer allocator
// (cudaMallocHost vs aligned host malloc) switch on this flag.  The CUDA
// build leaves it undefined and compiles the stream/event/pinned members
// below; the host build drops them.
#ifndef LORRAX_FFI_NO_CUDA
#include <cuda_runtime.h>
#endif
#include <mpi.h>
#include <hdf5.h>

namespace lorrax_ffi::phdf5 {

// The ONE boolean grammar for this translation unit family.  Mirrors
// Python's ``file_io/_slab_io_mpi_host._env_flag`` exactly (keep in sync):
//   unset or empty                                  -> default
//   strip + lowercase in {"1", "true", "yes", "on"} -> true
//   anything else ("0", "false", "no", "off", ...)  -> false
//
// It lives in the header, not in context.cc, because it was static there
// and read_ffi.cc could not see it -- so the former timing flag was written as
// a bare `getenv(...) != nullptr` presence test, in which `=0` turns
// timing ON.  That is the same defect this project already post-mortems
// for the Python driver debug switch and LORRAX_EXIT_AFTER_ZETA
// docstring).  A grammar only one file can reach is a grammar the next
// file will re-invent wrongly.
inline bool env_flag(const char* name, bool default_value) {
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

// ONE STRUCT BODY, TWO ABI IDENTITIES — and the second is not optional.
//
// The members under `#ifndef LORRAX_FFI_NO_CUDA` below mean the CUDA build
// and the host build lay this struct out DIFFERENTLY.  Until 2026-08-08 both
// layouts were called `lorrax_ffi::phdf5::PhdfCtx`, so both libraries also
// exported one mangled `open_ctx(...) -> PhdfCtx*`, `close_ctx(PhdfCtx*)`,
// `ensure_dataset(PhdfCtx*, ...)`, ... — one name, two incompatible meanings,
// in one RTLD_GLOBAL namespace.  Whichever library was dlopened first answered
// for both, INCLUDING for the other one's own internal calls, so a handler
// could be handed a ctx minted under the other layout and read its fields at
// the wrong offsets.  That is what KNOWN_FAILURES L1's
//
//     offset_base=[0,0,0,4596944070643295330]
//
// is: an IEEE-754 float64 (~0.19) decoded as an int64, because the reader and
// the writer disagreed about where the field was.
//
// Naming the two layouts differently makes that CATEGORY of failure
// unconstructible rather than merely unlikely: `open_ctx` mangles to
// `...PhdfCtxCudaV1...` in one library and `...PhdfCtxHostV1...` in the other,
// so the two symbols cannot collide even if some future build drops the
// visibility control in `exports_{cuda,host}.map`.  Belt AND braces, in that
// order — the version scripts are the belt, this is the braces.
//
// `using PhdfCtx = ...` below keeps every call site, every forward
// declaration and every `reinterpret_cast` in the four phdf5 TUs written the
// way they already were.  The alias is the source-level name; the struct tag
// is the ABI name.  Bump the V1 only if a layout changes in a way that a
// stale .so could be handed — the version is an ABI generation, not a
// release number.
#ifdef LORRAX_FFI_NO_CUDA
#  define LORRAX_PHDF_CTX_TYPE PhdfCtxHostV1
#else
#  define LORRAX_PHDF_CTX_TYPE PhdfCtxCudaV1
#endif

struct LORRAX_PHDF_CTX_TYPE {
    // Identity / process grid
    int rank = -1;
    int world_size = 0;
    int p = 0, q = 0;                            // 2-D mesh shape
    std::string path;

    // Owned MPI + HDF5 resources (created at open, destroyed at close).
    MPI_Comm comm            = MPI_COMM_NULL;
    bool     owns_comm       = false;            // dup'd from MPI_COMM_WORLD? then free on close
    hid_t    file_id         = H5I_INVALID_HID;
    hid_t    fapl_id         = H5I_INVALID_HID;  // mpio + coll_meta + align
    hid_t    fcpl_id         = H5I_INVALID_HID;
    hid_t    dxpl_coll       = H5I_INVALID_HID;  // H5FD_MPIO_COLLECTIVE
    hid_t    dxpl_indep      = H5I_INVALID_HID;  // H5FD_MPIO_INDEPENDENT
    // fill=NEVER, alloc=EARLY, CONTIGUOUS layout.  There is no
    // ``H5Pset_chunk`` call anywhere in this subpackage (``git grep
    // H5Pset_chunk`` is empty), so every dataset ``ensure_dataset``
    // creates with this DCPL takes HDF5's contiguous default.  The
    // comment used to say "chunked layout", which contradicted the
    // implementation, ``SlabIO.create_dataset``'s public docstring and
    // the runtime warning alike.  Chunking here would need per-dataset
    // chunk dimensions, not a context-wide property list.
    hid_t    dcpl_id         = H5I_INVALID_HID;

    // Cache of open datasets, keyed by HDF5 path ("ds_name" or
    // "/group/ds_name").  H5Dcreate on first write, H5Dopen on re-open.
    std::unordered_map<std::string, hid_t> open_datasets;
    std::mutex                             datasets_mu;

    // Staging — host buffers H5Dread/H5Dwrite land in before the
    // device-staging tail.  On the CUDA build they are cudaMallocHost-pinned
    // and copied H2D on ``stream`` with completion signalled via
    // ``h2d_event``; on the host build they are plain aligned mallocs and the
    // tail is a std::memcpy into the (host-resident) XLA output buffer.
    // The buffer + capacity fields are shared by both; the CUDA-only
    // stream/event/device fields below are compiled out on the host build.
    //
    // OWNERSHIP.  There are TWO staging buffers per ctx, split by which
    // THREAD may touch them.  The split is the fix for the race the
    // 2026-08-04 audit documented as item 4; before it there was one buffer
    // and the synchronous readers raced the writer thread on it.
    //
    //   ``pinned_buf``  — WRITER-THREAD staging.  Users:
    //        * ``async_read_kchunk_union_worker`` (both platforms), and
    //        * the CUDA async write, whose D2H is kicked from the XLA thread
    //          in ``WriteDispatch`` and consumed by the writer thread's
    //          H5Dwrite (ordered by ``d2h_event`` + the Python worker, which
    //          blocks one write's Future before dispatching the next).
    //      On the HOST build there is no write staging at all — H5Dwrite
    //      reads the XLA input buffer in place — so on the host build this
    //      buffer has exactly ONE producer, the writer thread, and FIFO task
    //      order is the whole of its synchronisation.  ``ensure_pinned``
    //      ENFORCES that on the host build: it refuses, by name, if called
    //      from any thread but ``writer_thread``.  A second synchronous
    //      entry point therefore cannot be added silently — it fails on its
    //      first call instead of corrupting a buffer months later.
    //
    //   ``read_buf``    — SYNCHRONOUS-READER staging, XLA thread.  Users:
    //      ``ReadImpl`` and ``ReadKchunkImpl``, both of which return
    //      ``ffi::Error`` (not a Future) and therefore run to completion on
    //      the calling XLA thread.  They share one buffer with each other,
    //      which is the relationship they already had; they no longer share
    //      one with the writer thread, which is the relationship that raced.
    //
    // Residual window, CUDA build only: an in-flight async write's D2H
    // (XLA thread) against a queued ``ReadKchunkUnion`` task (writer thread),
    // both on ``pinned_buf``.  Closing it means routing the write staging
    // through the writer queue — a C++ redesign, deliberately not attempted;
    // registered in KNOWN_LORRAX_ISSUES.md.  The host build, which is the
    // platform LORRAX certifies, does not have it.
    void*        pinned_buf          = nullptr;
    size_t       pinned_capacity     = 0;
    void*        read_buf            = nullptr;
    size_t       read_capacity       = 0;
    // std::hash of ``writer_thread``'s id, published by the thread itself
    // before it runs any task.  A plain integer rather than a
    // std::atomic<std::thread::id> so the guard needs nothing exotic; a hash
    // collision only means one illegal call is not caught.
    std::atomic<unsigned long long> writer_tid_hash{0};
#ifndef LORRAX_FFI_NO_CUDA
    // private CUDA stream for D2H/H2D copies.
    cudaStream_t stream              = nullptr;
    int          cuda_device         = -1;         // CUDA device id the main thread was on at open_ctx time; the writer_thread binds here before running any CUDA API (otherwise GPU pointers from main thread's context aren't recognised → cudaErrorMemoryAllocation on cudaMemcpyAsync).
    // Reused I/O-completion events.  Created in open_ctx, destroyed
    // in close_ctx.  Per-call cudaEventCreate/Destroy blocks for
    // ~800 ms on non-root ranks when xla_stream has a backlog
    // (measured 2026-04-18) — likely a cuda_async stream-ordered
    // allocator interaction.  Pooling: one event for write (D2H
    // completion signalled to writer thread), one for read (H2D
    // completion signalled to xla_stream so downstream ops wait).
    // Safe to reuse because reads and writes are serialised through
    // the Python worker / writer thread.
    cudaEvent_t  d2h_event           = nullptr;
    cudaEvent_t  h2d_event           = nullptr;
#endif

    // ─── Async-write worker ─────────────────────────────────────────
    // Single dedicated thread drains ``task_queue`` in FIFO order.
    // Every XLA-FFI-dispatched ``write_slab`` enqueues one task; the
    // worker runs the H5Dwrite MPI-IO collective.  One thread per ctx
    // (not per call) guarantees writes rendezvous in the same order
    // on every rank, which is the MPI-IO collective correctness
    // requirement.  (With detached per-call threads, OS scheduling
    // could reorder task execution between ranks and deadlock the
    // collective.)
    std::thread                       writer_thread;
    std::mutex                        queue_mu;
    std::condition_variable           queue_cv;
    std::deque<std::function<void()>> task_queue;
    bool                              shutdown_flag = false;

    // Tuning flags, read from env at open time.  Default: collective
    // reads AND collective writes (two-phase aggregation; the strided
    // 2-D tile pattern of the production writes is ~3 orders faster
    // collective — scorecard AI — and the default now matches the
    // Python phdf5_host writer so LORRAX_PHDF5_COLLECTIVE_WRITES=0
    // means "independent" in every writer.  Historical Cray-MPICH
    // caution: ad_cray_write_coll.c:669 OOM at >~1 GB/rank on that
    // stack — use the env override there).  Metadata defaults to
    // non-collective so file-level ops (H5Dcreate/extend) bypass the
    // collective driver — per ARCHITECTURE.md non-collective meta is
    // ~100 ms faster on OpenMPI small writes too.  dedup_replicas
    // drops all-but-one writer of a replica group's identical
    // hyperslab (mesh axes not consumed by the array's sharding):
    // required for correctness under collective writes (overlapping
    // selections are undefined) and pure waste-removal under
    // independent writes.  LORRAX_PHDF5_DEDUP_REPLICAS=0 disables.
    bool   use_collective_read  = true;
    bool   use_collective_write = true;
    bool   coll_metadata        = false;
    bool   dedup_replicas       = true;
    size_t align_threshold   = 1 << 20;          // 1 MiB
    size_t align_length      = 1 << 20;
};

// The source-level name.  Everything below, and every phdf5 TU, says
// `PhdfCtx`; only the emitted symbols carry the leg.
using PhdfCtx = LORRAX_PHDF_CTX_TYPE;

// Grow ctx->pinned_buf (WRITER-THREAD staging) to at least `need_bytes`.
// Idempotent; returns false on allocation failure or on an overflowing
// request.  On the host build it ALSO returns false, with a flushed
// diagnostic, when called from any thread but ``writer_thread`` — see the
// OWNERSHIP block above.
bool ensure_pinned(PhdfCtx* ctx, size_t need_bytes);

// Grow ctx->read_buf (SYNCHRONOUS-READER staging, XLA thread) to at least
// `need_bytes`.  Same allocator as ensure_pinned, no thread guard: its
// callers are the synchronous read handlers by construction.
bool ensure_read_buf(PhdfCtx* ctx, size_t need_bytes);

// ─── THE LIVE-CTX REGISTRY (SLAB_IO_ROOT_CAUSE_AUDIT.md B2) ──────────────
//
// A handler's ctx handle arrives as an int64 in a DEVICE buffer and is turned
// back into a pointer by ``reinterpret_cast<PhdfCtx*>``.  Before B1 that
// buffer was read without stream ordering, so the int64 could be stale
// allocator content: either a garbage address (immediate segfault on the
// first field access) or — worse, because it corrupts quietly — a pointer to
// a PREVIOUS iteration's already-closed ctx, i.e. a use-after-free whose
// damage surfaces somewhere else entirely, e.g. at the next close's
// writer-thread join.  That is the suspected route to S1.
//
// B1 removes the race.  This is the second line: every value that comes off
// that seam is CHECKED against the set of contexts this library currently has
// open before anything dereferences it, so a residual race — from a path we
// have not audited, from a double close, from a handle minted by the other
// platform leg — becomes a named refusal with got/want/fix instead of a
// dereference.  Insert at open (after the ctx is fully built), erase at the
// top of close (before teardown starts).
//
// ``n_live`` is an out-param rather than a second no-argument function on
// purpose: a ``size_t live_ctx_count()`` would mangle IDENTICALLY in the CUDA
// and host legs, and this header's whole ABI-identity argument above is that
// no phdf5 symbol may do that.  Taking a ``PhdfCtx*`` puts the leg tag in the
// mangled name.  (The version script hides both anyway — belt and braces, in
// that order.)
void register_live_ctx(PhdfCtx* ctx);
// Returns whether this pointer WAS live — the erase and the test are one
// locked operation so ``close_ctx`` can use it as the double-close guard
// without a check-then-erase window.
bool unregister_live_ctx(PhdfCtx* ctx);
bool ctx_is_live(const PhdfCtx* ctx, size_t* n_live);

}  // namespace lorrax_ffi::phdf5
