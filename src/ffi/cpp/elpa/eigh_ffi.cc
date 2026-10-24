// elpa/eigh_ffi.cc -- XLA FFI handler for ELPA's Hermitian eigensolver on the ROCM leg:
// the ONE-stage path only (GPU tridiagonalization, GPU divide-and-conquer, GPU
// back-transformation), ELPA 2026.02.001 built by config/frontier/build_elpa_hip.sh.
// ELPA's two-stage path is unreachable: the handle is pinned to ELPA_SOLVER_1STAGE and
// read back after setup, and every call refuses ELPA_FORCE_* environment variables
// (ELPA lets them override any elpa_set at each read).  Two-stage was slower everywhere
// measured and returns non-orthonormal eigenvectors on the 4x4 grid at n >= 47104
// (ELPA_TWO_STAGE_UPSTREAM_REPORT.md).
//
// Operand contract (distrib_la._elpa; relabeled block-cyclic layout, distrib_la._cyclic):
//   A (input):   the rank's (N/p, N/p) block, LOCALLY TRANSPOSED by the Python
//                wrapper, i.e. column-major bytes; read as ELPA's block-cyclic
//                local array with tile nb | N/p on the p x p grid, so ELPA
//                solves Pi A Pi^T.  Never written (copied into the workspace).
//   W (output):  (N,) real eigenvalues, ascending, replicated -- ELPA writes
//                them on the device (device-pointer API).
//   Q (output):  ELPA's local eigenvector array (col-major, lld = N/p): rows
//                in A's order, columns block-cyclic (the wrapper permutes them).
//   WS (output): u8 workspace, an XLA buffer: [ELPA's device pool | rocBLAS regions].
//   A_io (output, ALIASED to A): ELPA's input, destroyed in place.  A is the wrapper's
//                transposed temporary (input_output_aliases {0: 3}), never the caller's.
//
// Grid: ELPA takes the process coordinates explicitly.  Rank r of the mesh's
// MPI mesh context (cpp/mesh/; JAX process index, mesh shard (r / q, r % q)) is ELPA process
// (row r / q, col r % q) on a parent communicator in JAX rank order -- the
// mapping the standalone benchmark validated (ELPA_ROCM_REPORT.md).
//
// Device memory.  ELPA allocates every buffer through hipMallocFromC, which
// the build patch (config/frontier/build_elpa_hip.sh) routes to
// lorrax_elpa_set_device_allocator.  During the solve this handler installs a
// first-fit allocator over the workspace's pool region (buffers of at least
// n * 64 elements; smaller ones are device mallocs, see kSmallMin), so ELPA's large buffers are
// XLA's (in memory_analysis, inside the arena); past its end the allocator
// falls back to a device malloc (counted; LORRAX_DEBUG_PRINT=1 prints it).
// The free callback synchronizes the device first: hipFree does, and ELPA
// relies on that before a buffer is reused.  The ELPA handle (GPU setup:
// BLAS handles, streams) is created once per (context, N, nb, dtype) with the
// hook UNSET, so its persistent allocations never land in a per-call buffer.
//
// rocBLAS / rocSOLVER.  ELPA creates 2 * omp_get_max_threads() (+1 measured) rocBLAS
// handles per object; the patched library records them (lorrax_elpa_rocblas_handles).
// At creation each is detached (rocblas_set_workspace(h, nullptr, 0): its 32 MiB default
// is freed).  For each solve every handle of the object gets a user-owned region of the
// workspace's tail (BLAS handle 0: rb_back_bytes, solver handle 0 -- rocsolver stedc --
// rb_stedc_bytes, every other handle rb_other_bytes; distrib_la._elpa.rocblas_regions)
// and is detached again after it, so rocBLAS holds no device memory between calls.  A
// region too small makes rocBLAS/rocSOLVER return rocblas_status_memory_error without
// computing; ELPA drops most statuses, so the patched library counts every failed status
// (lorrax_elpa_gpu_status) and this handler refuses any solve during which the count
// moved: never a silent result.
//
// Threads: ELPA's OpenMP regions run with omp_threads = the process's CPU
// affinity count for float64 and 1 for complex128 (the configuration measured and
// gated; ELPA_ROCM_REPORT.md).
#include "../common/ctx_registry.h"

#include <sched.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <tuple>
#include <vector>

#include <mpi.h>
#include "../common/gpu_runtime.h"
#include <elpa/elpa.h>
#include <rocblas/rocblas.h>

#include "xla/ffi/api/ffi.h"
#include "../common/ffi_helpers.h"
#include "../mesh/ctx.h"

extern "C" {
typedef int (*lorrax_elpa_alloc_fn)(void** ptr, std::size_t bytes, void* ctx);
typedef int (*lorrax_elpa_free_fn)(void* ptr, void* ctx);
// Exported by the patched libelpa (config/frontier/build_elpa_hip.sh, patches 1 and 2).
void lorrax_elpa_set_device_allocator(lorrax_elpa_alloc_fn, lorrax_elpa_free_fn, void*);
void lorrax_elpa_gpu_status(long* failures, long* degraded, int* last_status);
int lorrax_elpa_rocblas_handles(void** handles, int* kinds, int cap);
}

namespace lorrax_ffi::elpa_ffi {

namespace ffi = ::xla::ffi;

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
//  The pool: first-fit with coalescing over one device region.
// ---------------------------------------------------------------------------
constexpr std::size_t kAlign = 256;
//: Allocations below max(kSmallMin, n * kSmallPerRow * itemsize) are device mallocs,
//: never pool blocks.  ELPA keeps buffers allocated inside a solve live past it, cached
//: on the handle: measured 262144 B at n = 1024 and n * 32 elements at n = 20480 (10 MiB
//: complex128, 5 MiB float64; ELPA 2026.02.001, one-stage).  From the pool they would
//: alias XLA memory after return, so the cap is twice the measured size; a larger buffer
//: kept past the call still refuses (the live-pool check below).  The small mallocs'
//: live high-water is reported (LORRAX_DEBUG_PRINT).
constexpr std::size_t kSmallMin = std::size_t(1) << 20;
constexpr std::size_t kSmallPerRow = 64;

struct Pool {
    char* base = nullptr;
    std::size_t size = 0;
    std::map<std::size_t, std::size_t> free_;      // offset -> length
    std::map<void*, std::pair<std::size_t, bool>> live;   // ptr -> (bytes, pooled)
    std::size_t used = 0, hw = 0, fb_bytes = 0, fb_hw = 0, small = 0, small_hw = 0;
    long fallbacks = 0, allocs = 0;
    std::size_t small_cap = kSmallMin;
    std::mutex mu;
    Pool(void* b, std::size_t n, std::size_t cap)
        : base(static_cast<char*>(b)), size(n), small_cap(std::max(kSmallMin, cap)) {
        if (n) free_[0] = n;
    }
};

static int pool_alloc(void** out, std::size_t bytes, void* vp) {
    auto* P = static_cast<Pool*>(vp);
    std::size_t want = (bytes + kAlign - 1) / kAlign * kAlign;
    if (want == 0) want = kAlign;
    std::lock_guard<std::mutex> lk(P->mu);
    ++P->allocs;
    if (want < P->small_cap) {
        void* p = nullptr;
        if (lrx::gpu::device_malloc(&p, want) != lrx::gpu::kSuccess) return 0;
        P->live[p] = {want, false};
        P->small += want;
        P->small_hw = std::max(P->small_hw, P->small);
        *out = p;
        return 1;
    }
    for (auto it = P->free_.begin(); it != P->free_.end(); ++it) {
        if (it->second < want) continue;
        const std::size_t off = it->first, len = it->second;
        P->free_.erase(it);
        if (len > want) P->free_[off + want] = len - want;
        void* p = P->base + off;
        P->live[p] = {want, true};
        P->used += want;
        P->hw = std::max(P->hw, P->used);
        *out = p;
        return 1;
    }
    void* p = nullptr;
    if (lrx::gpu::device_malloc(&p, want) != lrx::gpu::kSuccess) return 0;
    P->live[p] = {want, false};             // past the pool's end: counted fallback
    ++P->fallbacks;
    P->fb_bytes += want;
    P->fb_hw = std::max(P->fb_hw, P->fb_bytes);
    *out = p;
    return 1;
}

static int pool_free(void* p, void* vp) {
    if (p == nullptr) return 1;
    auto* P = static_cast<Pool*>(vp);
    // hipFree's implicit device synchronization, kept (ELPA reuses buffers after it).
    if (LRX_GPU_SYM(DeviceSynchronize)() != lrx::gpu::kSuccess) return 0;
    std::lock_guard<std::mutex> lk(P->mu);
    auto it = P->live.find(p);
    if (it == P->live.end())                       // not ours: an allocation from before the hook
        return LRX_GPU_SYM(Free)(p) == lrx::gpu::kSuccess;
    const auto [len, pooled] = it->second;
    P->live.erase(it);
    if (!pooled) {
        if (len < P->small_cap) P->small -= len; else P->fb_bytes -= len;
        return LRX_GPU_SYM(Free)(p) == lrx::gpu::kSuccess;
    }
    P->used -= len;
    std::size_t off = static_cast<char*>(p) - P->base, sz = len;
    auto nx = P->free_.lower_bound(off);
    if (nx != P->free_.end() && off + sz == nx->first) { sz += nx->second; nx = P->free_.erase(nx); }
    if (nx != P->free_.begin()) {
        auto pv = std::prev(nx);
        if (pv->first + pv->second == off) { pv->second += sz; return 1; }
    }
    P->free_[off] = sz;
    return 1;
}

// Installs the pool for one solve; always uninstalls.
struct PoolScope {
    explicit PoolScope(Pool* p) { lorrax_elpa_set_device_allocator(pool_alloc, pool_free, p); }
    ~PoolScope() { lorrax_elpa_set_device_allocator(nullptr, nullptr, nullptr); }
};

// ---------------------------------------------------------------------------
//  ELPA handles: one per (context, N, nb, complex), created collectively on
//  first use and kept for the process (ELPA's GPU setup costs seconds, and
//  elpa_deallocate leaks communicators with GPU kernels).
// ---------------------------------------------------------------------------
struct Handle {
    elpa_t h = nullptr;
    MPI_Comm comm = MPI_COMM_NULL;
    std::vector<rocblas_handle> rb;     // this object's rocBLAS handles, creation order
    std::vector<int> rb_kind;           // 0 BLAS, 1 solver (rocsolverCreateFromC)
};

constexpr int kMaxRb = 512;

// The registry's handles (creation order) and kinds.
static int rb_registry(std::vector<rocblas_handle>* hs, std::vector<int>* kinds) {
    std::vector<void*> h(kMaxRb);
    std::vector<int> k(kMaxRb);
    const int n = lorrax_elpa_rocblas_handles(h.data(), k.data(), kMaxRb);
    hs->clear(); kinds->clear();
    for (int i = 0; i < n && i < kMaxRb; ++i) {
        hs->push_back(static_cast<rocblas_handle>(h[i]));
        kinds->push_back(k[i]);
    }
    return n;
}

// Returns a handle to rocBLAS management with no memory (its managed buffer is freed).
static bool rb_detach(rocblas_handle h) {
    return rocblas_set_workspace(h, nullptr, 0) == rocblas_status_success;
}
static std::mutex g_mu;
static std::map<std::tuple<const void*, int64_t, int64_t, bool>, Handle> g_handles;
static std::map<rocblas_handle, bool> g_rb_owned;   // every rocBLAS handle some Handle owns
static std::map<const void*, MPI_Comm> g_comms;
static std::once_flag g_init;
static int g_init_status = ELPA_ERROR;

static int affinity_threads() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return 1;
    return std::max(1, CPU_COUNT(&set));
}

static std::string set_int(elpa_t h, const char* k, int v) {
    int err = ELPA_OK;
    elpa_set(h, k, v, &err);
    if (err != ELPA_OK) {
        std::ostringstream os;
        os << "elpa_set(" << k << "=" << v << "): " << elpa_strerr(err);
        return os.str();
    }
    return "";
}

// ELPA_FORCE_<option> overrides any elpa_set value at every read (elpa_index.c); one
// could re-select the two-stage solver or move kernels to the host.  Refused.
static std::string forced_option() {
    for (char** e = ::environ; e && *e; ++e)
        if (std::strncmp(*e, "ELPA_FORCE_", 11) == 0) return std::string(*e);
    return "";
}

static ffi::Error get_handle(lorrax_ffi::mesh::MeshCtx* ctx, int64_t n, int64_t nb,
                             bool cplx, Handle** out) {
    std::lock_guard<std::mutex> lk(g_mu);
    const auto key = std::make_tuple(static_cast<const void*>(ctx), n, nb, cplx);
    auto it = g_handles.find(key);
    if (it != g_handles.end()) { *out = &it->second; return ffi::Error::Success(); }

    std::call_once(g_init, [] { g_init_status = elpa_init(ELPA_API_VERSION); });
    if (g_init_status != ELPA_OK)
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "elpa.eigh: elpa_init(ELPA_API_VERSION) failed (library/header mismatch)");
    auto c = g_comms.find(ctx);
    if (c == g_comms.end()) {
        MPI_Comm jc = MPI_COMM_NULL;    // the context's ranks in JAX order
        if (MPI_Comm_split(ctx->comm, 0, ctx->rank, &jc) != MPI_SUCCESS)
            return ffi::Error(ffi::ErrorCode::kInternal, "elpa.eigh: MPI_Comm_split failed");
        c = g_comms.emplace(ctx, jc).first;
    }
    const int p = ctx->p, q = ctx->q;
    const int local = static_cast<int>(n / p);
    int err = ELPA_OK;
    elpa_t h = elpa_allocate(&err);
    if (h == nullptr || err != ELPA_OK)
        return ffi::Error(ffi::ErrorCode::kInternal, "elpa.eigh: elpa_allocate failed");
    int dev = 0;
    if (lrx::gpu::get_device(&dev) != lrx::gpu::kSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, "elpa.eigh: get_device failed");
    std::string e;
    const std::pair<const char*, int> pre[] = {
        {"na", (int) n}, {"nev", (int) n}, {"local_nrows", local}, {"local_ncols", local},
        {"nblk", (int) nb}, {"mpi_comm_parent", (int) MPI_Comm_c2f(c->second)},
        {"process_row", ctx->rank / q}, {"process_col", ctx->rank % q}};
    for (const auto& kv : pre) if (e.empty()) e = set_int(h, kv.first, kv.second);
    if (e.empty() && elpa_setup(h) != ELPA_OK) e = "elpa_setup failed";
    const std::pair<const char*, int> post[] = {
        {"solver", ELPA_SOLVER_1STAGE}, {"amd-gpu", 1},
        {"use_gpu_id", dev}, {"omp_threads", cplx ? 1 : affinity_threads()}};
    for (const auto& kv : post) if (e.empty()) e = set_int(h, kv.first, kv.second);
    if (e.empty()) {
        int got = -1, gerr = ELPA_OK;
        elpa_get(h, "solver", &got, &gerr);
        if (gerr != ELPA_OK || got != ELPA_SOLVER_1STAGE)
            e = "the handle's solver reads back as " + std::to_string(got) +
                ", not ELPA_SOLVER_1STAGE (two-stage is not supported)";
    }
    std::vector<rocblas_handle> before, after;
    std::vector<int> kb, ka;
    rb_registry(&before, &kb);
    if (e.empty() && elpa_setup_gpu(h) != ELPA_OK) e = "elpa_setup_gpu failed";
    if (!e.empty()) {
        int e2 = 0;
        elpa_deallocate(h, &e2);
        return ffi::Error(ffi::ErrorCode::kInternal, "elpa.eigh: " + e);
    }
    rb_registry(&after, &ka);
    Handle H{h, c->second, {}, {}};
    for (std::size_t i = 0; i < after.size(); ++i) {
        if (g_rb_owned.count(after[i])) continue;      // another object's
        H.rb.push_back(after[i]);
        H.rb_kind.push_back(ka[i]);
        g_rb_owned[after[i]] = true;
        if (!rb_detach(after[i]))
            return ffi::Error(ffi::ErrorCode::kInternal,
                              "elpa.eigh: rocblas_set_workspace(h, nullptr, 0) failed at setup");
    }
    auto& slot = g_handles[key];
    slot = std::move(H);
    *out = &slot;
    return ffi::Error::Success();
}

// The workspace tail's region for each of H's rocBLAS handles: BLAS handle 0 -> back,
// solver handle 0 -> stedc, every other -> other.  False if they do not fit in `room`.
static bool rb_layout(const Handle& H, std::size_t back, std::size_t stedc, std::size_t other,
                      std::size_t room, std::vector<std::size_t>* off, std::vector<std::size_t>* len,
                      std::size_t* need) {
    bool blas0 = false, solver0 = false;
    std::size_t at = 0;
    off->clear(); len->clear();
    for (std::size_t i = 0; i < H.rb.size(); ++i) {
        std::size_t b = other;
        if (H.rb_kind[i] == 0 && !blas0) { b = back; blas0 = true; }
        else if (H.rb_kind[i] == 1 && !solver0) { b = stedc; solver0 = true; }
        b = (b + kAlign - 1) / kAlign * kAlign;
        off->push_back(at);
        len->push_back(b);
        at += b;
    }
    *need = at;
    return at <= room;
}

struct WsAttrs {
    std::size_t pool, back, stedc, other;
};

template <typename T>
static ffi::Error EighImpl(int64_t n, int64_t nb, bool compute_evecs, lrx::gpu::Stream stream,
                           lorrax_ffi::mesh::MeshCtx* ctx, const T* d_A, T* d_Aio, double* d_W,
                           T* d_Q, void* d_ws, std::size_t ws_bytes, WsAttrs wa) {
    constexpr bool cplx = !std::is_same_v<T, double>;
    const int p = ctx->p;
    if (ctx->q != p)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "elpa.eigh: needs a square mesh");
    if (const std::string f = forced_option(); !f.empty())
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "elpa.eigh: refusing ELPA_FORCE_ environment override '" + f +
                          "' (it overrides the handler's pinned one-stage GPU settings); unset it");
    if (n % p || (n / p) % nb || nb < 1 || (nb & (nb - 1)))
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "elpa.eigh: nb must be a power of two dividing N/p "
                          "(ELPA's GPU kernels; otherwise it silently runs them on the host)");
    const int64_t local = n / p;
    const std::size_t abytes = static_cast<std::size_t>(local) * local * sizeof(T);
    if (wa.pool > ws_bytes)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "elpa.eigh: workspace smaller than its pool_bytes attribute");
    auto ce = lrx::gpu::stream_synchronize(stream);
    if (ce != lrx::gpu::kSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal,
                          std::string("elpa.eigh: StreamSynchronize(xla) failed: ") +
                          lrx::gpu::error_string(ce));
    Handle* H = nullptr;
    if (auto e = get_handle(ctx, n, nb, cplx, &H); e.failure()) return e;
    elpa_t h = H->h;

    // ELPA destroys its input: A_io is A's buffer (input_output_aliases); copy only if XLA
    // did not alias them (no extra memory: A_io is a separate output then).
    if (static_cast<const void*>(d_Aio) != static_cast<const void*>(d_A)) {
        ce = lrx::gpu::memcpy_async(d_Aio, d_A, abytes, lrx::gpu::kDeviceToDevice, stream);
        if (ce == lrx::gpu::kSuccess) ce = lrx::gpu::stream_synchronize(stream);
        if (ce != lrx::gpu::kSuccess)
            return ffi::Error(ffi::ErrorCode::kInternal,
                              std::string("elpa.eigh: A copy failed: ") + lrx::gpu::error_string(ce));
    }

    // rocBLAS: every handle of this object gets its region of the workspace's tail.
    std::vector<std::size_t> roff, rlen;
    std::size_t rneed = 0;
    if (!rb_layout(*H, wa.back, wa.stedc, wa.other, ws_bytes - wa.pool, &roff, &rlen, &rneed))
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "elpa.eigh: ELPA created " + std::to_string(H->rb.size()) +
                          " rocBLAS handles needing " + std::to_string(rneed) +
                          " workspace bytes; the workspace's rocBLAS tail holds " +
                          std::to_string(ws_bytes - wa.pool) +
                          " (distrib_la._elpa.rocblas_bytes counts 2 * elpa_threads() + 1 handles; "
                          "check OMP_NUM_THREADS)");
    char* rb_base = static_cast<char*>(d_ws) + wa.pool;
    for (std::size_t i = 0; i < H->rb.size(); ++i)
        if (rocblas_set_workspace(H->rb[i], rb_base + roff[i], rlen[i]) != rocblas_status_success) {
            for (auto rh : H->rb) rb_detach(rh);
            return ffi::Error(ffi::ErrorCode::kInternal, "elpa.eigh: rocblas_set_workspace failed");
        }
    std::vector<rocblas_handle> reg_before;
    std::vector<int> kind_before;
    rb_registry(&reg_before, &kind_before);
    long f0 = 0, d0 = 0, f1 = 0, d1 = 0;
    int last0 = 0, last1 = 0;
    lorrax_elpa_gpu_status(&f0, &d0, &last0);

    Pool pool(d_ws, wa.pool, static_cast<std::size_t>(n) * kSmallPerRow * sizeof(T));
    const double t0 = now_s();
    int err = ELPA_OK;
    {
        PoolScope scope(&pool);
        if constexpr (cplx) {
            if (compute_evecs) elpa_eigenvectors_double_complex(h, d_Aio, d_W, d_Q, &err);
            else               elpa_eigenvalues_double_complex(h, d_Aio, d_W, &err);
        } else {
            if (compute_evecs) elpa_eigenvectors_double(h, d_Aio, d_W, d_Q, &err);
            else               elpa_eigenvalues_double(h, d_Aio, d_W, &err);
        }
        // ELPA's last kernels and copies (eigenvalues, eigenvectors) are on its own
        // streams: complete before XLA reads W and Q or reuses the workspace.
        ce = LRX_GPU_SYM(DeviceSynchronize)();
    }
    const double t1 = now_s();
    lorrax_elpa_gpu_status(&f1, &d1, &last1);
    // Detach: rocBLAS must not keep pointers into the workspace past the call.  Handles
    // that appeared during the solve were unattached (rocBLAS-managed, outside the arena):
    // detached (freed) here, counted, and owned by this object from the next call.
    std::size_t rb_unattached_bytes = 0;
    int rb_new = 0;
    for (auto rh : H->rb) rb_detach(rh);
    {
        std::vector<rocblas_handle> reg_after;
        std::vector<int> kind_after;
        rb_registry(&reg_after, &kind_after);
        std::lock_guard<std::mutex> lk(g_mu);
        for (std::size_t i = 0; i < reg_after.size(); ++i) {
            if (g_rb_owned.count(reg_after[i])) continue;
            std::size_t sz = 0;
            rocblas_get_device_memory_size(reg_after[i], &sz);
            rb_unattached_bytes += sz;
            ++rb_new;
            H->rb.push_back(reg_after[i]);
            H->rb_kind.push_back(kind_after[i]);
            g_rb_owned[reg_after[i]] = true;
            rb_detach(reg_after[i]);
        }
    }
    if (err != ELPA_OK)
        return ffi::Error(ffi::ErrorCode::kInternal,
                          std::string("elpa.eigh: elpa_eigen* returned: ") + elpa_strerr(err));
    if (ce != lrx::gpu::kSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal,
                          std::string("elpa.eigh: DeviceSynchronize failed: ") +
                          lrx::gpu::error_string(ce));
    if (f1 != f0)       // ELPA would have returned a result computed past a failed call
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "elpa.eigh: " + std::to_string(f1 - f0) +
                          " rocBLAS/rocSOLVER call(s) failed during the solve (last status " +
                          std::to_string(last1) + "; " + std::to_string((int) rocblas_status_memory_error) +
                          " = rocblas_status_memory_error: a rocBLAS workspace region was too small, "
                          "see distrib_la._elpa.rocblas_regions); the result is discarded");
    std::size_t live_pooled = 0;
    for (const auto& kv : pool.live) if (kv.second.second) live_pooled += kv.second.first;
    if (live_pooled)      // a pooled buffer outliving the call would alias XLA memory
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "elpa.eigh: ELPA kept " + std::to_string(live_pooled) +
                          " pooled bytes past the call");
    const char* rep = std::getenv("LORRAX_DEBUG_PRINT");
    if (rep && rep[0] && rep[0] != '0') {
        std::fprintf(stderr,
            "[elpa.eigh] rank=%d n=%lld nb=%lld %s wall=%.2f pool=%.3fGB pool_hw=%.3fGB "
            "fallbacks=%ld fallback_hw=%.3fGB small_hw=%.6fGB small_live=%.6fGB allocs=%ld "
            "rocblas_handles=%zu rocblas_ws=%.3fGB rocblas_new=%d rocblas_unattached=%.3fGB "
            "degraded=%ld aliased=%d\n",
            ctx->rank, (long long) n, (long long) nb, cplx ? "c128" : "f64", t1 - t0,
            pool.size / 1e9, pool.hw / 1e9, pool.fallbacks, pool.fb_hw / 1e9, pool.small_hw / 1e9,
            pool.small / 1e9, pool.allocs, H->rb.size() - rb_new, rneed / 1e9, rb_new,
            rb_unattached_bytes / 1e9, d1 - d0,
            static_cast<const void*>(d_Aio) == static_cast<const void*>(d_A) ? 1 : 0);
    }
    return ffi::Error::Success();
}

static ffi::Error EighDispatch(lrx::gpu::Stream stream, ffi::AnyBuffer A,
                               ffi::Result<ffi::AnyBuffer> W_out,
                               ffi::Result<ffi::AnyBuffer> Q_out,
                               ffi::Result<ffi::AnyBuffer> WS_out,
                               ffi::Result<ffi::AnyBuffer> A_io,
                               int64_t n, int64_t nb, int64_t ctx_key,
                               bool compute_evecs, int64_t pool_bytes, int64_t rb_back_bytes,
                               int64_t rb_stedc_bytes, int64_t rb_other_bytes) {
    const int64_t ctx_handle = ::lorrax_ffi::ctx_registry::resolve(ctx_key, "mesh");
    auto* ctx = reinterpret_cast<lorrax_ffi::mesh::MeshCtx*>(ctx_handle);
    if (ctx == nullptr)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "elpa.eigh: ctx_handle is null (context not initialized?)");
    const auto dt = A.element_type();
    if (Q_out->element_type() != dt || A_io->element_type() != dt ||
        W_out->element_type() != ffi::DataType::F64)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "elpa.eigh: Q must match A's dtype and W must be float64");
    void* ws = WS_out->untyped_data();
    const std::size_t wsb = WS_out->size_bytes();
    auto* W = static_cast<double*>(W_out->untyped_data());
    if (pool_bytes < 0 || rb_back_bytes < 0 || rb_stedc_bytes < 0 || rb_other_bytes < 0)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "elpa.eigh: negative workspace attribute");
    const WsAttrs wa{static_cast<std::size_t>(pool_bytes), static_cast<std::size_t>(rb_back_bytes),
                     static_cast<std::size_t>(rb_stedc_bytes), static_cast<std::size_t>(rb_other_bytes)};
    switch (dt) {
        case ffi::DataType::F64:
            return EighImpl<double>(n, nb, compute_evecs, stream, ctx,
                                    static_cast<const double*>(A.untyped_data()),
                                    static_cast<double*>(A_io->untyped_data()), W,
                                    static_cast<double*>(Q_out->untyped_data()), ws, wsb, wa);
        case ffi::DataType::C128:
            return EighImpl<std::complex<double>>(
                n, nb, compute_evecs, stream, ctx,
                static_cast<const std::complex<double>*>(A.untyped_data()),
                static_cast<std::complex<double>*>(A_io->untyped_data()), W,
                static_cast<std::complex<double>*>(Q_out->untyped_data()), ws, wsb, wa);
        default:
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "elpa.eigh: unsupported dtype (supported: F64, C128)");
    }
}

}  // namespace lorrax_ffi::elpa_ffi

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ElpaEighFfi, lorrax_ffi::elpa_ffi::EighDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()              // A (local block, col-major bytes)
        .Ret<xla::ffi::AnyBuffer>()              // W (N,) float64, replicated
        .Ret<xla::ffi::AnyBuffer>()              // Q (local block, A's dtype)
        .Ret<xla::ffi::AnyBuffer>()              // workspace (u8): ELPA's pool + rocBLAS regions
        .Ret<xla::ffi::AnyBuffer>()              // A_io: aliased to A, destroyed by ELPA
        .Attr<int64_t>("n")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_key")
        .Attr<bool>("compute_evecs")
        .Attr<int64_t>("pool_bytes")
        .Attr<int64_t>("rb_back_bytes")
        .Attr<int64_t>("rb_stedc_bytes")
        .Attr<int64_t>("rb_other_bytes"));
