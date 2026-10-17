// elpa/eigh_ffi.cc -- XLA FFI handler for ELPA's Hermitian eigensolver on the ROCM leg:
// the one-stage path (GPU tridiagonalization, GPU divide-and-conquer, GPU
// back-transformation; distrib_la._elpa's default, measured fastest on MI250X) or the
// two-stage path (attribute solver = 2; host band -> tridiagonal), ELPA 2026.02.001
// built by config/frontier/build_elpa_hip.sh.
//
// Operand contract: the SAME as slate/eigh_ffi.cc, so distrib_la._elpa reuses
// _slate's wrapper shape (relabeled block-cyclic layout, distrib_la._cyclic):
//   A (input):   the rank's (N/p, N/p) block, LOCALLY TRANSPOSED by the Python
//                wrapper, i.e. column-major bytes; read as ELPA's block-cyclic
//                local array with tile nb | N/p on the p x p grid, so ELPA
//                solves Pi A Pi^T.  Never written (copied into the workspace).
//   W (output):  (N,) real eigenvalues, ascending, replicated -- ELPA writes
//                them on the device (device-pointer API).
//   Q (output):  ELPA's local eigenvector array (col-major, lld = N/p): rows
//                in A's order, columns block-cyclic (the wrapper permutes them).
//   WS (output): u8 workspace, an XLA buffer: [A copy | ELPA's device pool].
//
// Grid: ELPA takes the process coordinates explicitly.  Rank r of the mesh's
// SLATE context (JAX process index, mesh shard (r / q, r % q)) is ELPA process
// (row r / q, col r % q) on a parent communicator in JAX rank order -- the
// mapping the standalone benchmark validated (ELPA_ROCM_REPORT.md).
//
// Device memory.  ELPA allocates every buffer through hipMallocFromC, which
// the build patch (config/frontier/build_elpa_hip.sh) routes to
// lorrax_elpa_set_device_allocator.  During the solve this handler installs a
// first-fit allocator over the workspace's pool region (buffers of at least
// n * 64 elements; smaller ones are device mallocs, see kSmallMin), so ELPA's large buffers are
// XLA's (in memory_analysis, inside the arena); past its end the allocator
// falls back to a device malloc (counted; LORRAX_ELPA_POOL_REPORT=1 prints it).
// The free callback synchronizes the device first: hipFree does, and ELPA
// relies on that before a buffer is reused.  The ELPA handle (GPU setup:
// BLAS handles, streams) is created once per (context, N, nb, dtype) with the
// hook UNSET, so its persistent allocations never land in a per-call buffer.
// What stays outside the arena (rocBLAS/rocSOLVER handle workspace, the
// handle's persistent buffers) is measured in the report.
//
// Threads: ELPA's OpenMP regions run with omp_threads = the process's CPU
// affinity count for float64 and 1 for complex128 (ELPA 2026.02's OpenMP +
// GPU complex two-stage band reduction shares one rocBLAS handle across
// threads and faults -- measured, ELPA_ROCM_REPORT.md).
#include "../common/ctx_registry.h"

#include <sched.h>
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

#include "xla/ffi/api/ffi.h"
#include "../common/ffi_helpers.h"
#include "../slate/ctx.h"

extern "C" {
typedef int (*lorrax_elpa_alloc_fn)(void** ptr, std::size_t bytes, void* ctx);
typedef int (*lorrax_elpa_free_fn)(void* ptr, void* ctx);
// Exported by the patched libelpa (config/frontier/build_elpa_hip.sh).
void lorrax_elpa_set_device_allocator(lorrax_elpa_alloc_fn, lorrax_elpa_free_fn, void*);
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
//: live high-water is reported (LORRAX_ELPA_POOL_REPORT).
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
};
static std::mutex g_mu;
static std::map<std::tuple<const void*, int64_t, int64_t, bool, int>, Handle> g_handles;
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

static ffi::Error get_handle(lorrax_ffi::slate::SlateCtx* ctx, int64_t n, int64_t nb,
                             bool cplx, int solver, elpa_t* out) {
    std::lock_guard<std::mutex> lk(g_mu);
    const auto key = std::make_tuple(static_cast<const void*>(ctx), n, nb, cplx, solver);
    auto it = g_handles.find(key);
    if (it != g_handles.end()) { *out = it->second.h; return ffi::Error::Success(); }

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
        {"solver", solver == 2 ? ELPA_SOLVER_2STAGE : ELPA_SOLVER_1STAGE}, {"amd-gpu", 1},
        {"use_gpu_id", dev}, {"omp_threads", cplx ? 1 : affinity_threads()}};
    for (const auto& kv : post) if (e.empty()) e = set_int(h, kv.first, kv.second);
    // The stage-2 back-transformation kernel exists only on a two-stage handle: ELPA refuses
    // the key (ELPA_ERROR_ENTRY_INVALID_VALUE) once solver is one-stage.
    if (e.empty() && solver == 2)
        e = set_int(h, cplx ? "complex_kernel" : "real_kernel",
                    cplx ? (int) ELPA_2STAGE_COMPLEX_AMD_GPU : (int) ELPA_2STAGE_REAL_AMD_GPU);
    if (e.empty() && elpa_setup_gpu(h) != ELPA_OK) e = "elpa_setup_gpu failed";
    if (!e.empty()) {
        int e2 = 0;
        elpa_deallocate(h, &e2);
        return ffi::Error(ffi::ErrorCode::kInternal, "elpa.eigh: " + e);
    }
    g_handles[key] = Handle{h, c->second};
    *out = h;
    return ffi::Error::Success();
}

// Bytes of the A copy at the head of the workspace (distrib_la._elpa._a_copy_bytes).
static std::size_t a_copy_bytes(int64_t local, std::size_t itemsize) {
    return (static_cast<std::size_t>(local) * local * itemsize + kAlign - 1) / kAlign * kAlign;
}

template <typename T>
static ffi::Error EighImpl(int64_t n, int64_t nb, int solver, bool compute_evecs, lrx::gpu::Stream stream,
                           lorrax_ffi::slate::SlateCtx* ctx, const T* d_A, double* d_W, T* d_Q,
                           void* d_ws, std::size_t ws_bytes) {
    constexpr bool cplx = !std::is_same_v<T, double>;
    const int p = ctx->p;
    if (ctx->q != p)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "elpa.eigh: needs a square mesh");
    if (solver != 1 && solver != 2)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "elpa.eigh: solver must be 1 or 2");
    if (n % p || (n / p) % nb || nb < 1 || (nb & (nb - 1)))
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "elpa.eigh: nb must be a power of two dividing N/p "
                          "(ELPA's GPU kernels; otherwise it silently runs them on the host)");
    const int64_t local = n / p;
    const std::size_t abytes = a_copy_bytes(local, sizeof(T));
    if (ws_bytes < abytes)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "elpa.eigh: workspace smaller than the A copy");
    auto ce = lrx::gpu::stream_synchronize(stream);
    if (ce != lrx::gpu::kSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal,
                          std::string("elpa.eigh: StreamSynchronize(xla) failed: ") +
                          lrx::gpu::error_string(ce));
    elpa_t h = nullptr;
    if (auto e = get_handle(ctx, n, nb, cplx, solver, &h); e.failure()) return e;

    T* a_ws = static_cast<T*>(d_ws);
    ce = lrx::gpu::memcpy_async(a_ws, d_A, static_cast<std::size_t>(local) * local * sizeof(T),
                                lrx::gpu::kDeviceToDevice, stream);
    if (ce == lrx::gpu::kSuccess) ce = lrx::gpu::stream_synchronize(stream);
    if (ce != lrx::gpu::kSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal,
                          std::string("elpa.eigh: A copy failed: ") + lrx::gpu::error_string(ce));

    Pool pool(static_cast<char*>(d_ws) + abytes, ws_bytes - abytes,
              static_cast<std::size_t>(n) * kSmallPerRow * sizeof(T));
    const double t0 = now_s();
    int err = ELPA_OK;
    {
        PoolScope scope(&pool);
        if constexpr (cplx) {
            if (compute_evecs) elpa_eigenvectors_double_complex(h, a_ws, d_W, d_Q, &err);
            else               elpa_eigenvalues_double_complex(h, a_ws, d_W, &err);
        } else {
            if (compute_evecs) elpa_eigenvectors_double(h, a_ws, d_W, d_Q, &err);
            else               elpa_eigenvalues_double(h, a_ws, d_W, &err);
        }
        // ELPA's last kernels and copies (eigenvalues, eigenvectors) are on its own
        // streams: complete before XLA reads W and Q or reuses the workspace.
        ce = LRX_GPU_SYM(DeviceSynchronize)();
    }
    const double t1 = now_s();
    if (err != ELPA_OK)
        return ffi::Error(ffi::ErrorCode::kInternal,
                          std::string("elpa.eigh: elpa_eigen* returned: ") + elpa_strerr(err));
    if (ce != lrx::gpu::kSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal,
                          std::string("elpa.eigh: DeviceSynchronize failed: ") +
                          lrx::gpu::error_string(ce));
    std::size_t live_pooled = 0;
    for (const auto& kv : pool.live) if (kv.second.second) live_pooled += kv.second.first;
    if (live_pooled)      // a pooled buffer outliving the call would alias XLA memory
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "elpa.eigh: ELPA kept " + std::to_string(live_pooled) +
                          " pooled bytes past the call");
    const char* rep = std::getenv("LORRAX_ELPA_POOL_REPORT");
    if (rep && rep[0] && rep[0] != '0') {
        std::fprintf(stderr,
            "[elpa.eigh] rank=%d n=%lld nb=%lld solver=%d %s wall=%.2f pool=%.3fGB pool_hw=%.3fGB "
            "fallbacks=%ld fallback_hw=%.3fGB small_hw=%.6fGB allocs=%ld\n",
            ctx->rank, (long long) n, (long long) nb, solver, cplx ? "c128" : "f64", t1 - t0,
            pool.size / 1e9, pool.hw / 1e9, pool.fallbacks, pool.fb_hw / 1e9, pool.small_hw / 1e9,
            pool.allocs);
    }
    return ffi::Error::Success();
}

static ffi::Error EighDispatch(lrx::gpu::Stream stream, ffi::AnyBuffer A,
                               ffi::Result<ffi::AnyBuffer> W_out,
                               ffi::Result<ffi::AnyBuffer> Q_out,
                               ffi::Result<ffi::AnyBuffer> WS_out,
                               int64_t n, int64_t nb, int64_t solver, int64_t ctx_key,
                               bool compute_evecs) {
    const int64_t ctx_handle = ::lorrax_ffi::ctx_registry::resolve(ctx_key, "slate");
    auto* ctx = reinterpret_cast<lorrax_ffi::slate::SlateCtx*>(ctx_handle);
    if (ctx == nullptr)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "elpa.eigh: ctx_handle is null (context not initialized?)");
    const auto dt = A.element_type();
    if (Q_out->element_type() != dt || W_out->element_type() != ffi::DataType::F64)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "elpa.eigh: Q must match A's dtype and W must be float64");
    void* ws = WS_out->untyped_data();
    const std::size_t wsb = WS_out->size_bytes();
    auto* W = static_cast<double*>(W_out->untyped_data());
    switch (dt) {
        case ffi::DataType::F64:
            return EighImpl<double>(n, nb, (int) solver, compute_evecs, stream, ctx,
                                    static_cast<const double*>(A.untyped_data()), W,
                                    static_cast<double*>(Q_out->untyped_data()), ws, wsb);
        case ffi::DataType::C128:
            return EighImpl<std::complex<double>>(
                n, nb, (int) solver, compute_evecs, stream, ctx,
                static_cast<const std::complex<double>*>(A.untyped_data()), W,
                static_cast<std::complex<double>*>(Q_out->untyped_data()), ws, wsb);
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
        .Ret<xla::ffi::AnyBuffer>()              // workspace (u8): A copy + ELPA's pool
        .Attr<int64_t>("n")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("solver")                 // 1: ELPA 1-stage, 2: 2-stage
        .Attr<int64_t>("ctx_key")
        .Attr<bool>("compute_evecs"));
