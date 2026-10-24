// batched_solve_lu_ffi.cc — per-q cuSOLVERMp LU factor+solve on the
// world-wide (Px, Py) grid.  Does getrf followed immediately by getrs,
// per q-slice, so callers never see the pivot vector.
//
// *** Historical note — 2D-grid LU was broken in 0.6.0, fixed in 0.7.2 ***
//
// On cuSOLVERMp 0.6.0 (NVHPC 25.5) this handler returned wrong answers
// for most (N, NRHS) configurations on true 2D process grids: status=
// SUCCESS and info=0 from both Getrf and Getrs, but X buffer contained
// garbage (residual |AX-B|/|B| ≈ 0.3–0.5).  The Cholesky path
// (potrf/potrs) on the same ctx/grid/descriptor-setup worked correctly
// at all tested sizes — the bug was specific to the 2D distributed LU
// kernel in cuSOLVERMp itself.
//
// **Resolved in cuSOLVERMp 0.7.2** (current default; see
// ``stage_pypi.sh``).  Validated end-to-end on the MoS2 3×3 D3h
// bispinor smoke (2×2 mesh, transverse γ̃^i channels) — eqp0
// matches the legacy per-q ``jnp.linalg.solve`` path to float ULP.
// See ``isdf_fitting._resolve_solver_kind_transverse`` for the
// dispatch (default ``auto`` → cuSolverMp on true 2D meshes; override
// via cohsex.in ``cusolvermp_lu``).
//
// (Earlier 0.6.0-era workaround in w_isdf's low_mem W-solve — the
// symmetric Cholesky identity ``W = X (I − X† χ X)⁻¹ X†`` — is no
// longer required on 0.7.2 but is retained because it has favourable
// memory characteristics for the W solve.)
//
// Sharding contract:
//   A : (Nq, N, N)       P(None, 'x', 'y')  — Python pre-transposes the
//                        inner two dims to match cuSOLVERMp's col-major
//                        grid expectation (same trick as potrf/potrs).
//   B : (Nq, N, NRHS)    P(None, 'x', 'y')  — inner dims pre-transposed
//                        to (NRHS/Py, N/Px) row-major ≡ col-major
//                        (N/Px × NRHS/Py) per slice.
//   X : same layout as B; written in place.
//   LU: same layout as A; the factors, the second output.
//
// descA : (N, N) with mb=nb=N/Px=N/Py (caller passes mb_a, nb_a).
// descB : (N, NRHS) with mb_b=N/Px (row block matches A), nb_b=NRHS/Py.
//
// getrf and getrs work in place, so the caller of the _factors target aliases
// A -> LU and B -> X (input_output_aliases={0: 1, 1: 0}).  Every write is then
// a declared output: XLA copies an operand only while something still reads
// it, and otherwise the factors reuse A's buffer.  The ABI-6 target has no LU
// output and writes the factors into A.  The pivot vector and one info slot
// per vendor call are allocated per call via cudaMalloc, read once and freed
// before the ctx→xla cross-stream event.  cuSOLVERMp requires LOCr(M_A) + MB_A pivot entries
// per rank, including the extra block used by distributed row interchanges.

#include "../common/ctx_registry.h"
#include <cctype>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>

#include <algorithm>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <cuda_runtime.h>
#include <cusolverMp.h>

#include "xla/ffi/api/ffi.h"

#include "../common/ffi_helpers.h"
#include "cusolvermp_interface.h"
#include "ctx.h"
#include "info.h"

namespace lorrax_ffi::cusolvermp_batched_solve_lu {

namespace ffi = ::xla::ffi;
using lorrax_ffi::cusolvermp::LorraxCusolverMpCtx;
using lorrax_ffi::cusolvermp::ensure_workspace;
namespace mp = lorrax_ffi::cusolvermp::mp;

static ffi::Error cross_stream_wait_pooled(cudaStream_t waiter,
                                           cudaStream_t signaller,
                                           cudaEvent_t  ev) {
    LORRAX_GPU_CHECK(cudaEventRecord(ev, signaller));
    LORRAX_GPU_CHECK(cudaStreamWaitEvent(waiter, ev, 0));
    return ffi::Error::Success();
}

// cusolverMpGetrs's workspace grows with the RHS width far past the RHS itself:
// at n = nrhs = 30720 complex128 on 2x2 (256 blocks) it asked for 297 GB, 674 GB at
// n = nrhs = 51200 float64 (A100-80GB, dlascale pmC); on 4x4 at n 30000 (block 250),
// 0.21 GB at 1000 columns and over 2.15 GB at 3000 (run30k, 2026-10-19).  The RHS is
// therefore solved in column CHUNKS: k whole local column blocks of every process column
// (global columns [c k nb_b Py, (c+1) k nb_b Py) of the block-cyclic B, contiguous in each
// rank's buffer at column c k nb_b), each its own n x (k nb_b Py) descriptor at a pointer
// offset.  Columns are independent, so the chunked solve is the same solve, and every chunk
// repeats getrs's row interchanges and both triangular sweeps: the fewer, the faster.
//
// k is the largest divisor of the local block count whose workspace fits the cap (1 if
// none does), bisected over the divisors (the workspace does not shrink as the chunk
// widens; whatever it does, the k taken fits).  The cap is the room the caller posted for
// this (n, nrhs) (lrx_lu_getrs_room), else kGetrsDefaultCap; the split getrs always takes
// the default, because its workspace is the context's persistent buffer (ensure_workspace),
// which never shrinks.  Each width is sized once per plan, at the largest rank's figure,
// and a fused solve takes the smallest rank's cap: every rank picks the same k and issues
// the same collective calls.
static constexpr size_t kGetrsDefaultCap = size_t(2) << 30;

static std::mutex room_mu;
static std::map<std::pair<int64_t, int64_t>, uint64_t> posted_rooms;   // (n, nrhs) -> bytes/rank

static uint64_t getrs_cap(int64_t n, int64_t nrhs) {
    std::lock_guard<std::mutex> lock(room_mu);
    auto it = posted_rooms.find({n, nrhs});
    return it == posted_rooms.end() ? uint64_t(kGetrsDefaultCap) : it->second;
}

// ``v[0..count)`` reduced over the grid's ranks (ncclMax or ncclMin) through the device words
// ``d``, back on the host: a collective every rank enters at the same point.
static ffi::Error agree(LorraxCusolverMpCtx* ctx, uint64_t* d, uint64_t* v, size_t count,
                        ncclRedOp_t op) {
    LORRAX_GPU_CHECK(cudaMemcpyAsync(d, v, count * sizeof(uint64_t), cudaMemcpyHostToDevice,
                                     ctx->stream));
    if (ncclAllReduce(d, d, count, ncclUint64, op, ctx->nccl_comm, ctx->stream) != ncclSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal, "cusolvermp LU: ncclAllReduce (agree) failed");
    LORRAX_GPU_CHECK(cudaMemcpyAsync(v, d, count * sizeof(uint64_t), cudaMemcpyDeviceToHost,
                                     ctx->stream));
    LORRAX_GPU_CHECK(cudaStreamSynchronize(ctx->stream));
    return ffi::Error::Success();
}

// One process-lifetime plan per (context, dtype, n, nrhs, blocks): the descriptors a solve runs
// at and the vendor's workspace sizes. cuSOLVERMp 0.9.1's getrs_bufferSize builds a cuBLASMp
// grid for the descriptors it is given (cublasMpGridCreate -> ncclCommSplit -> ncclCommInitRank),
// so a query per call made a communicator per call; one rank deadlocked in libfabric's memory
// hooks inside that split and stalled run30k's first Dyson dispatch (2026-10-19, gdb on all 16
// ranks). The plan is built once under a lock, on the first call of every rank alike (the
// calls are collective), and never freed: its descriptors live as long as the context.
struct GetrsChunk {
    cusolverMpMatrixDescriptor_t descB = nullptr;       // n x (k nb_b Py)
    size_t d_ws = 0, h_ws = 0;                          // the largest rank's
};

struct LuPlan {
    cusolverMpMatrixDescriptor_t descA = nullptr;
    size_t d_ws_f = 0, h_ws_f = 0;
    bool getrf = false;
    std::map<int64_t, GetrsChunk> chunks;               // k -> its chunk, sized when first probed
    uint64_t* d_agree = nullptr;                        // two device words for agree()
};

static std::mutex plan_mu;

template <typename T>
static ffi::Error lu_plan(LorraxCusolverMpCtx* ctx, int64_t n, int64_t nrhs, int64_t mb_a,
                          int64_t nb_a, int64_t mb_b, int64_t nb_b, bool getrf, bool getrs,
                          T* d_A, int64_t* d_ipiv, LuPlan** out) {
    using Key = std::tuple<const void*, int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    static std::map<Key, LuPlan> plans;                 // node-based: references stay valid
    std::lock_guard<std::mutex> lock(plan_mu);
    LuPlan& p = plans[Key{ctx, int(mp::CudaDataTypeOf<T>::value), n, getrs ? nrhs : 0, mb_a, nb_a,
                          getrs ? mb_b : 0, getrs ? nb_b : 0}];
    const int64_t lld_A = (n + ctx->p - 1) / ctx->p;
    if (!p.descA) {
        cusolverMpMatrixDescriptor_t d = nullptr;
        if (cusolverMpCreateMatrixDesc(&d, ctx->grid, mp::CudaDataTypeOf<T>::value,
                                       n, n, mb_a, nb_a, 0, 0, lld_A) != CUSOLVER_STATUS_SUCCESS)
            return ffi::Error(ffi::ErrorCode::kInternal, "cusolverMpCreateMatrixDesc(A)");
        LORRAX_GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&p.d_agree), 2 * sizeof(uint64_t)));
        p.descA = d;
    }
    if (getrf && !p.getrf) {
        cusolverStatus_t st = mp::GetrfBufferSize<T>(ctx->handle, n, n, d_A, 1, 1, p.descA, d_ipiv,
                                                     &p.d_ws_f, &p.h_ws_f);
        if (st != CUSOLVER_STATUS_SUCCESS) {
            std::ostringstream os;
            os << "cusolverMpGetrf_bufferSize failed: status=" << (int)st;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
        p.getrf = true;
    }
    *out = &p;
    return ffi::Error::Success();
}

// The chunk a getrs of this plan runs at under ``cap`` (the comment above kGetrsDefaultCap).
template <typename T>
static ffi::Error getrs_chunk(LorraxCusolverMpCtx* ctx, LuPlan& p, int64_t n, int64_t nrhs,
                              int64_t mb_b, int64_t nb_b, T* d_A, int64_t* d_ipiv, T* d_X,
                              uint64_t cap, int64_t* k_out, const GetrsChunk** out) {
    const int Py = ctx->q;
    const int64_t lld = (n + ctx->p - 1) / ctx->p;
    const int64_t local = (nrhs + Py - 1) / Py, blk = local / nb_b;   // column blocks per rank
    if (blk < 1 || local % nb_b)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cusolvermp getrs: nb_b must divide the local RHS columns");
    std::lock_guard<std::mutex> lock(plan_mu);
    auto size = [&](int64_t k) -> ffi::Error {
        if (p.chunks.count(k)) return ffi::Error::Success();
        cusolverMpMatrixDescriptor_t dB = nullptr;
        if (cusolverMpCreateMatrixDesc(&dB, ctx->grid, mp::CudaDataTypeOf<T>::value,
                                       n, k * nb_b * Py, mb_b, nb_b, 0, 0, lld)
                != CUSOLVER_STATUS_SUCCESS)
            return ffi::Error(ffi::ErrorCode::kInternal, "cusolverMpCreateMatrixDesc(B chunk)");
        size_t d = 0, h = 0;
        cusolverStatus_t st = mp::GetrsBufferSize<T>(
            ctx->handle, CUBLAS_OP_N, n, k * nb_b * Py, d_A, 1, 1, p.descA, d_ipiv,
            d_X, 1, 1, dB, &d, &h);
        uint64_t ws[2] = {d, h};
        ffi::Error e = st == CUSOLVER_STATUS_SUCCESS
            ? agree(ctx, p.d_agree, ws, 2, ncclMax)
            : ffi::Error(ffi::ErrorCode::kInternal,
                         "cusolverMpGetrs_bufferSize failed: status=" + std::to_string(int(st)));
        if (!e.success()) { cusolverMpDestroyMatrixDesc(dB); return e; }
        p.chunks[k] = GetrsChunk{dB, size_t(ws[0]), size_t(ws[1])};
        return ffi::Error::Success();
    };
    std::vector<int64_t> ks;
    for (int64_t k = 1; k <= blk; ++k)
        if (blk % k == 0) ks.push_back(k);
    size_t lo = 0, hi = ks.size();          // ks[lo] is taken (k = 1 always); ks[hi] does not fit
    while (hi - lo > 1) {
        const size_t mid = (lo + hi) / 2;
        if (auto e = size(ks[mid]); !e.success()) return e;
        (p.chunks[ks[mid]].d_ws <= cap ? lo : hi) = mid;
    }
    if (auto e = size(ks[lo]); !e.success()) return e;
    *k_out = ks[lo];
    *out = &p.chunks[ks[lo]];
    return ffi::Error::Success();
}

// Under LORRAX_DEBUG_PRINT, once per process, shape and chunking: the descriptors and the getrs chunking a
// solve runs at (ws_f < 0: a getrs-only call, its getrf ran before).
static void debug_lu_layout(const char* kind, int64_t n, int64_t nrhs, int64_t mb_a, int64_t nb_a,
                            int64_t mb_b, int64_t nb_b, int64_t n_chunks, int64_t chunk_cols,
                            double ws_f, double ws_s) {
    static std::mutex mu;
    static std::set<std::tuple<std::string, int64_t, int64_t, int64_t>> seen;
    const char* dbg = std::getenv("LORRAX_DEBUG_PRINT");
    if (!dbg || !*dbg || *dbg == '0') return;
    std::lock_guard<std::mutex> lock(mu);
    if (!seen.insert({kind, n, nrhs, n_chunks}).second) return;
    std::fprintf(stderr, "[cusolvermp %s] n=%lld nrhs=%lld A mb/nb %lld/%lld B mb/nb %lld/%lld: getrs "
                 "%lld chunk(s) of %lld local columns; workspace getrf %s GB, getrs %.2f GB\n", kind,
                 (long long)n, (long long)nrhs, (long long)mb_a, (long long)nb_a, (long long)mb_b,
                 (long long)nb_b, (long long)n_chunks, (long long)chunk_cols,
                 ws_f < 0 ? "n/a" : std::to_string(ws_f / 1e9).substr(0, 5).c_str(), ws_s / 1e9);
}

template <typename T>
static ffi::Error BatchedSolveLuImpl(
    int64_t nq, int64_t n, int64_t nrhs,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    cudaStream_t xla_stream,
    LorraxCusolverMpCtx* ctx,
    const T* d_A_in, const T* d_B_in,
    T* d_A_factored_out, T* d_X_out)
{
    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        ctx->stream, xla_stream, ctx->ev_xla_in));

    const int Px = ctx->p;
    const int Py = ctx->q;
    const int64_t lld_A = (n + Px - 1) / Px;
    const int64_t A_local_cols = (n    + Py - 1) / Py;
    const int64_t B_local_cols = (nrhs + Py - 1) / Py;
    const int64_t A_slice = lld_A * A_local_cols;
    const int64_t B_slice = lld_A * B_local_cols;
    // Per-slice local ipiv length.  The cuSOLVERMp Getrf/Getrs contract is
    // LOCr(M_A) + MB_A, not LOCc(N).  This one-block-per-process-row
    // descriptor has LOCr(M_A) == lld_A.
    const int64_t ipiv_slice = lld_A + mb_a;

    // Copy A into A_factored_out (unless aliased); copy B into X_out
    // (unless aliased).  getrf and getrs are in-place on their output
    // buffers.
    if (d_A_factored_out != static_cast<const T*>(d_A_in)) {
        LORRAX_GPU_CHECK(cudaMemcpyAsync(
            d_A_factored_out, d_A_in,
            nq * A_slice * sizeof(T),
            cudaMemcpyDeviceToDevice, ctx->stream));
    }
    if (d_X_out != static_cast<const T*>(d_B_in)) {
        LORRAX_GPU_CHECK(cudaMemcpyAsync(
            d_X_out, d_B_in,
            nq * B_slice * sizeof(T),
            cudaMemcpyDeviceToDevice, ctx->stream));
    }

    // The pivots and one info slot per call (getrf and every getrs chunk of every slice, so
    // none is read before the last call) in one per-call allocation.  Plain cudaMalloc: the
    // pool-based async path correlated with non-deterministic wrong answers on 2D grids.
    const int64_t blk = B_local_cols / std::max<int64_t>(1, nb_b);   // most chunks per slice
    const int64_t n_slots = nq * (1 + blk);
    const size_t ipiv_bytes = static_cast<size_t>(nq) * ipiv_slice * sizeof(int64_t);
    const size_t scratch_bytes = ipiv_bytes + size_t(n_slots) * sizeof(int);
    int64_t* d_ipiv = nullptr;
    LORRAX_GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ipiv), scratch_bytes));
    int* d_infos = reinterpret_cast<int*>(reinterpret_cast<char*>(d_ipiv) + ipiv_bytes);
    auto cleanup = [&]() { cudaFree(d_ipiv); };
    if (cudaMemsetAsync(d_ipiv, 0, scratch_bytes, ctx->stream) != cudaSuccess) {
        cleanup();
        return ffi::Error(ffi::ErrorCode::kInternal, "cuSOLVERMp LU: cudaMemsetAsync(pivots)");
    }

    // The descriptors and the vendor's workspace sizes, once per shape (lu_plan, getrs_chunk):
    // no bufferSize query, and so no cuBLASMp grid or communicator, once a width is sized.  The
    // cap is the smallest rank's, so every rank solves in the same chunks (one 8-byte allreduce
    // and host read before getrf; another agrees the allocations below).
    LuPlan* plan = nullptr;
    uint64_t cap = getrs_cap(n, nrhs);
    int64_t kc = 1;
    const GetrsChunk* chunk = nullptr;
    ffi::Error planned = lu_plan<T>(ctx, n, nrhs, mb_a, nb_a, mb_b, nb_b, true, true,
                                    d_A_factored_out, d_ipiv, &plan);
    if (planned.success()) planned = agree(ctx, plan->d_agree, &cap, 1, ncclMin);
    if (planned.success())
        planned = getrs_chunk<T>(ctx, *plan, n, nrhs, mb_b, nb_b, d_A_factored_out, d_ipiv,
                                 d_X_out, cap, &kc, &chunk);
    if (!planned.success()) { cleanup(); return planned; }
    cusolverMpMatrixDescriptor_t descA = plan->descA, descB = chunk->descB;
    // The NVIDIA mp_getrf_getrs sample allocates SEPARATE workspaces for Getrf and Getrs --
    // sharing one workspace buffer gave non-deterministic wrong answers on 2D grids. Both are
    // allocated per call via cudaMallocAsync so the CUDA pool can recycle them.
    const size_t d_ws_f = plan->d_ws_f, h_ws_f = plan->h_ws_f;
    const size_t d_ws_s = chunk->d_ws, h_ws_s = chunk->h_ws;
    cusolverStatus_t mp_st = CUSOLVER_STATUS_SUCCESS;
    const int64_t chunk_cols = kc * nb_b;                         // local columns per chunk
    const int64_t n_chunks = blk / kc;
    debug_lu_layout("solve_lu", n, nrhs, mb_a, nb_a, mb_b, nb_b, n_chunks, chunk_cols, double(d_ws_f), double(d_ws_s));
    void* d_ws_f_buf = nullptr;
    void* d_ws_s_buf = nullptr;
    void* h_ws_f_buf = nullptr;
    void* h_ws_s_buf = nullptr;
    auto cleanup_ws = [&]() {
        if (d_ws_f_buf) cudaFreeAsync(d_ws_f_buf, ctx->stream);
        if (d_ws_s_buf) cudaFreeAsync(d_ws_s_buf, ctx->stream);
        if (h_ws_f_buf) std::free(h_ws_f_buf);
        if (h_ws_s_buf) std::free(h_ws_s_buf);
        d_ws_f_buf = d_ws_s_buf = h_ws_f_buf = h_ws_s_buf = nullptr;
    };
    // A failed workspace allocation releases what this call already holds (the
    // getrf workspace, d_ipiv, the descriptors).  It used to return through
    // LORRAX_GPU_CHECK with all of them live: on A100-80GB P4 the n=30720 c128
    // getrs workspace failed, and the leaked getrf workspace then failed every
    // later solve, potrf and eigh in the process (dlascale pmA / pmB).  The outcome
    // is agreed before the first collective call, so a rank that cannot allocate
    // fails the solve on every rank instead of leaving the others in getrf.
    std::string refused;
    auto ws_alloc = [&](void** buf, size_t bytes, const char* what) {
        if (bytes == 0 || !refused.empty()) return;
        cudaError_t st = cudaMallocAsync(buf, bytes, ctx->stream);
        if (st == cudaSuccess) return;
        *buf = nullptr;
        cudaGetLastError();                      // clear the sticky-free OOM status
        std::ostringstream os;
        os << "cudaMallocAsync(" << what << ", " << bytes << " bytes) failed: "
           << cudaGetErrorName(st);
        refused = os.str();
    };
    ws_alloc(&d_ws_f_buf, d_ws_f, "getrf workspace");
    ws_alloc(&d_ws_s_buf, d_ws_s, "getrs workspace");
    if (refused.empty() && h_ws_f > 0 && !(h_ws_f_buf = std::malloc(h_ws_f)))
        refused = "malloc(h_ws_getrf) failed";
    if (refused.empty() && h_ws_s > 0 && !(h_ws_s_buf = std::malloc(h_ws_s)))
        refused = "malloc(h_ws_getrs) failed";
    uint64_t allocated = refused.empty() ? 1 : 0;
    ffi::Error agreed = agree(ctx, plan->d_agree, &allocated, 1, ncclMin);
    if (!agreed.success() || !allocated) {
        cleanup_ws(); cleanup();
        if (!agreed.success()) return agreed;
        std::ostringstream os;
        os << "cuSOLVERMp LU: " << (refused.empty() ? "another rank's workspace allocation failed"
                                                     : refused)
           << " (n=" << n << ", nrhs=" << nrhs << ", blocks " << mb_a << "x" << nb_a
           << ", getrs chunk " << chunk_cols << " local columns, rank " << ctx->rank << ")";
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, os.str());
    }

    for (int64_t q = 0; q < nq; ++q) {
        T* A_slice_ptr = d_A_factored_out + q * A_slice;
        T* X_slice_ptr = d_X_out + q * B_slice;
        int64_t* ipiv_slice_ptr = d_ipiv + q * ipiv_slice;
        int* info_q = d_infos + q * (1 + blk);           // getrf's, then one per chunk

        mp_st = mp::Getrf<T>(
            ctx->handle, n, n,
            A_slice_ptr, 1, 1, descA,
            ipiv_slice_ptr,
            d_ws_f_buf, d_ws_f,
            h_ws_f_buf, h_ws_f,
            info_q);
        if (mp_st != CUSOLVER_STATUS_SUCCESS) {
            cleanup_ws(); cleanup();
            std::ostringstream os;
            os << "cusolverMpGetrf (q=" << q << ") failed: status=" << (int)mp_st;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
        for (int64_t c = 0; c < n_chunks; ++c) {
            mp_st = mp::Getrs<T>(
                ctx->handle, CUBLAS_OP_N, n, chunk_cols * Py,
                A_slice_ptr, 1, 1, descA,
                ipiv_slice_ptr,
                X_slice_ptr + c * chunk_cols * lld_A, 1, 1, descB,
                d_ws_s_buf, d_ws_s,
                h_ws_s_buf, h_ws_s,
                info_q + 1 + c);
            if (mp_st != CUSOLVER_STATUS_SUCCESS) {
                cleanup_ws(); cleanup();
                std::ostringstream os;
                os << "cusolverMpGetrs (q=" << q << ", chunk " << c << ") failed: status="
                   << (int)mp_st;
                return ffi::Error(ffi::ErrorCode::kInternal, os.str());
            }
        }
    }

    // One status read per solve, after the last call: no host sync between the calls.
    cleanup_ws();
    std::vector<int> infos(n_slots, 0);
    cudaError_t rd = cudaMemcpyAsync(infos.data(), d_infos, n_slots * sizeof(int),
                                     cudaMemcpyDeviceToHost, ctx->stream);
    if (rd == cudaSuccess) rd = cudaStreamSynchronize(ctx->stream);
    cleanup();
    if (rd != cudaSuccess)
        return ffi::Error(ffi::ErrorCode::kInternal,
                          std::string("cuSOLVERMp LU: info read failed: ") + cudaGetErrorName(rd));
    for (int64_t q = 0; q < nq; ++q)
        for (int64_t c = 0; c <= n_chunks; ++c)
            if (int info = infos[q * (1 + blk) + c]; info != 0)
                return c == 0 ? info_error(ctx, "cusolverMpGetrf",
                                           "U has an exactly zero diagonal entry", info, n, mb_a, q)
                              : info_error(ctx, "cusolverMpGetrs", "the solve failed", info,
                                           n, mb_a, q);

    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        xla_stream, ctx->stream, ctx->ev_ctx_out));
    return ffi::Error::Success();
}

// Split sibling of BatchedSolveLuImpl.  LU and ipiv remain device-resident
// between calls so a physics stage can pay getrf once and issue getrs for
// every RHS chunk.  ipiv_len is the LOCAL cuSOLVERMp extent
// LOCr(M_A)+MB_A;
// Python represents the rank-private rows as P(None,('x','y')), hence each
// rank stores exactly nq*(LOCr+MB) int64 values (never a replicated pivot).
template <typename T>
static ffi::Error BatchedGetrfImpl(
    int64_t nq, int64_t n, int64_t mb, int64_t nb, int64_t ipiv_len,
    cudaStream_t xla_stream, LorraxCusolverMpCtx* ctx,
    const T* d_A_in, T* d_LU_out, int64_t* d_ipiv_out)
{
    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        ctx->stream, xla_stream, ctx->ev_xla_in));
    const int Px = ctx->p;
    const int Py = ctx->q;
    const int64_t lld_A = (n + Px - 1) / Px;
    const int64_t local_cols = (n + Py - 1) / Py;
    const int64_t expected_ipiv_len = lld_A + mb;
    if (ipiv_len != expected_ipiv_len) {
        std::ostringstream os;
        os << "cusolvermp.getrf: ipiv_len=" << ipiv_len
           << " != LOCr(M_A)+MB_A=" << expected_ipiv_len;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    const int64_t A_slice = lld_A * local_cols;
    if (d_LU_out != d_A_in) {
        LORRAX_GPU_CHECK(cudaMemcpyAsync(
            d_LU_out, d_A_in, nq * A_slice * sizeof(T),
            cudaMemcpyDeviceToDevice, ctx->stream));
    }
    LORRAX_GPU_CHECK(cudaMemsetAsync(
        d_ipiv_out, 0, nq * ipiv_len * sizeof(int64_t), ctx->stream));

    int64_t* piv0 = d_ipiv_out;
    // The descriptor and the vendor's workspace size, once per shape (lu_plan).
    LuPlan* plan = nullptr;
    if (auto e = lu_plan<T>(ctx, n, 0, mb, nb, 0, 0, true, false, d_LU_out, piv0, &plan);
            !e.success())
        return e;
    cusolverMpMatrixDescriptor_t descA = plan->descA;
    const size_t d_ws = plan->d_ws_f, h_ws = plan->h_ws_f;
    cusolverStatus_t mp_st = CUSOLVER_STATUS_SUCCESS;
    try {
        ensure_workspace(ctx, d_ws, h_ws);
    } catch (const std::exception& ex) {
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, ex.what());
    }
    for (int64_t q = 0; q < nq; ++q) {
        T* LU_q = d_LU_out + q * A_slice;
        int64_t* piv_q = d_ipiv_out + q * ipiv_len;
        if (auto err = reset_info(ctx); !err.success()) {
            return err;
        }
        mp_st = mp::Getrf<T>(
            ctx->handle, n, n, LU_q, 1, 1, descA, piv_q,
            ctx->d_workspace, ctx->d_workspace_bytes,
            ctx->h_workspace, ctx->h_workspace_bytes, ctx->d_info);
        if (mp_st != CUSOLVER_STATUS_SUCCESS) {
            std::ostringstream os;
            os << "cusolverMpGetrf (q=" << q << ") failed: status="
               << (int)mp_st;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
        if (auto err = read_info(ctx, "cusolverMpGetrf", "U has an exactly zero diagonal entry",
                                 n, mb, q); !err.success()) {
            return err;
        }
    }
    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        xla_stream, ctx->stream, ctx->ev_ctx_out));
    return ffi::Error::Success();
}

template <typename T>
static ffi::Error BatchedGetrsImpl(
    int64_t nq, int64_t n, int64_t nrhs,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    int64_t ipiv_len, cudaStream_t xla_stream, LorraxCusolverMpCtx* ctx,
    const T* d_LU, const int64_t* d_ipiv, const T* d_B_in, T* d_X_out)
{
    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        ctx->stream, xla_stream, ctx->ev_xla_in));
    const int Px = ctx->p;
    const int Py = ctx->q;
    const int64_t lld_A = (n + Px - 1) / Px;
    const int64_t A_local_cols = (n + Py - 1) / Py;
    const int64_t B_local_cols = (nrhs + Py - 1) / Py;
    const int64_t expected_ipiv_len = lld_A + mb_a;
    if (ipiv_len != expected_ipiv_len) {
        std::ostringstream os;
        os << "cusolvermp.getrs: ipiv_len=" << ipiv_len
           << " != LOCr(M_A)+MB_A=" << expected_ipiv_len;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    const int64_t A_slice = lld_A * A_local_cols;
    const int64_t B_slice = lld_A * B_local_cols;
    if (d_X_out != d_B_in) {
        LORRAX_GPU_CHECK(cudaMemcpyAsync(
            d_X_out, d_B_in, nq * B_slice * sizeof(T),
            cudaMemcpyDeviceToDevice, ctx->stream));
    }
    int64_t* piv0 = const_cast<int64_t*>(d_ipiv);
    // The descriptors and the vendor's workspace size, once per shape (lu_plan).
    // Always at the default cap: this workspace is the context's persistent buffer.
    LuPlan* plan = nullptr;
    int64_t kc = 1;
    const GetrsChunk* chunk = nullptr;
    if (auto e = lu_plan<T>(ctx, n, nrhs, mb_a, nb_a, mb_b, nb_b, false, true, const_cast<T*>(d_LU),
                            piv0, &plan); !e.success())
        return e;
    if (auto e = getrs_chunk<T>(ctx, *plan, n, nrhs, mb_b, nb_b, const_cast<T*>(d_LU), piv0,
                                d_X_out, kGetrsDefaultCap, &kc, &chunk); !e.success())
        return e;
    cusolverMpMatrixDescriptor_t descA = plan->descA, descB = chunk->descB;
    const size_t d_ws = chunk->d_ws, h_ws = chunk->h_ws;
    cusolverStatus_t mp_st = CUSOLVER_STATUS_SUCCESS;
    const int64_t chunk_cols = kc * nb_b;
    const int64_t n_chunks = (B_local_cols / nb_b) / kc;
    debug_lu_layout("getrs", n, nrhs, mb_a, nb_a, mb_b, nb_b, n_chunks, chunk_cols, -1.0, double(d_ws));
    // This workspace may be the same persistent allocation getrf used.
    // Reuse is safe across the SPLIT handlers because getrf records
    // ev_ctx_out and makes the XLA stream wait before returning its LU/ipiv;
    // this getrs cannot begin until those inputs are ready, then its opening
    // xla->ctx event orders the context stream after that completion.  The
    // fused handler cannot establish that FFI boundary and therefore keeps
    // two simultaneous workspaces (see its measured 2-D-grid warning above).
    try {
        ensure_workspace(ctx, d_ws, h_ws);
    } catch (const std::exception& ex) {
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, ex.what());
    }
    for (int64_t q = 0; q < nq; ++q) {
        T* LU_q = const_cast<T*>(d_LU) + q * A_slice;
        T* X_q = d_X_out + q * B_slice;
        int64_t* piv_q = const_cast<int64_t*>(d_ipiv) + q * ipiv_len;
        if (auto err = reset_info(ctx); !err.success()) {
            return err;
        }
        for (int64_t c = 0; c < n_chunks && mp_st == CUSOLVER_STATUS_SUCCESS; ++c)
            mp_st = mp::Getrs<T>(
                ctx->handle, CUBLAS_OP_N, n, chunk_cols * Py, LU_q, 1, 1, descA, piv_q,
                X_q + c * chunk_cols * lld_A, 1, 1, descB,
                ctx->d_workspace, ctx->d_workspace_bytes,
                ctx->h_workspace, ctx->h_workspace_bytes, ctx->d_info);
        if (mp_st != CUSOLVER_STATUS_SUCCESS) {
            std::ostringstream os;
            os << "cusolverMpGetrs (q=" << q << ") failed: status="
               << (int)mp_st;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
        if (auto err = read_info(ctx, "cusolverMpGetrs", "the solve failed", n, mb_a, q);
                !err.success()) {
            return err;
        }
    }
    FFI_RETURN_IF_ERROR(cross_stream_wait_pooled(
        xla_stream, ctx->stream, ctx->ev_ctx_out));
    return ffi::Error::Success();
}

static ffi::Error BatchedGetrfDispatch(
    cudaStream_t stream, ffi::AnyBuffer A,
    ffi::Result<ffi::AnyBuffer> LU_out,
    ffi::Result<ffi::AnyBuffer> ipiv_out,
    int64_t nq, int64_t n, int64_t mb, int64_t nb, int64_t ipiv_len,
    int64_t ctx_key)
{
    const int64_t ctx_handle = ::lorrax_ffi::ctx_registry::resolve(ctx_key, "cusolvermp");
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (ctx == nullptr)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cusolvermp.getrf: ctx_handle is null");
    const auto dtype = A.element_type();
    if (LU_out->element_type() != dtype ||
        ipiv_out->element_type() != ffi::DataType::S64)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cusolvermp.getrf: LU must match A and ipiv must be int64");
    const auto ipiv_dims = ipiv_out->dimensions();
    if (ipiv_dims.size() != 2 || ipiv_dims[0] != nq ||
        ipiv_dims[1] != ipiv_len) {
        std::ostringstream os;
        os << "cusolvermp.getrf: local ipiv buffer must be (" << nq
           << ", " << ipiv_len << ") int64";
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    switch (dtype) {
        case ffi::DataType::F64:
            return BatchedGetrfImpl<double>(
                nq, n, mb, nb, ipiv_len, stream, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<double*>(LU_out->untyped_data()),
                static_cast<int64_t*>(ipiv_out->untyped_data()));
        case ffi::DataType::C128: {
            using C128 = std::complex<double>;
            return BatchedGetrfImpl<C128>(
                nq, n, mb, nb, ipiv_len, stream, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<C128*>(LU_out->untyped_data()),
                static_cast<int64_t*>(ipiv_out->untyped_data()));
        }
        default:
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "cusolvermp.getrf: supported dtypes are F64, C128");
    }
}

static ffi::Error BatchedGetrsDispatch(
    cudaStream_t stream, ffi::AnyBuffer LU, ffi::AnyBuffer ipiv,
    ffi::AnyBuffer B, ffi::Result<ffi::AnyBuffer> X_out,
    int64_t nq, int64_t n, int64_t nrhs,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    int64_t ipiv_len, int64_t ctx_key)
{
    const int64_t ctx_handle = ::lorrax_ffi::ctx_registry::resolve(ctx_key, "cusolvermp");
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (ctx == nullptr)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cusolvermp.getrs: ctx_handle is null");
    const auto dtype = LU.element_type();
    if (B.element_type() != dtype || X_out->element_type() != dtype ||
        ipiv.element_type() != ffi::DataType::S64)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cusolvermp.getrs: LU/B/X must share dtype and ipiv must be int64");
    const auto ipiv_dims = ipiv.dimensions();
    if (ipiv_dims.size() != 2 || ipiv_dims[0] != nq ||
        ipiv_dims[1] != ipiv_len) {
        std::ostringstream os;
        os << "cusolvermp.getrs: local ipiv buffer must be (" << nq
           << ", " << ipiv_len << ") int64";
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    switch (dtype) {
        case ffi::DataType::F64:
            return BatchedGetrsImpl<double>(
                nq, n, nrhs, mb_a, nb_a, mb_b, nb_b, ipiv_len,
                stream, ctx,
                static_cast<const double*>(LU.untyped_data()),
                static_cast<const int64_t*>(ipiv.untyped_data()),
                static_cast<const double*>(B.untyped_data()),
                static_cast<double*>(X_out->untyped_data()));
        case ffi::DataType::C128: {
            using C128 = std::complex<double>;
            return BatchedGetrsImpl<C128>(
                nq, n, nrhs, mb_a, nb_a, mb_b, nb_b, ipiv_len,
                stream, ctx,
                static_cast<const C128*>(LU.untyped_data()),
                static_cast<const int64_t*>(ipiv.untyped_data()),
                static_cast<const C128*>(B.untyped_data()),
                static_cast<C128*>(X_out->untyped_data()));
        }
        default:
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "cusolvermp.getrs: supported dtypes are F64, C128");
    }
}

static ffi::Error BatchedSolveLu(
    cudaStream_t stream,
    ffi::AnyBuffer A,
    ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> X_out,
    void* lu, ffi::DataType lu_type,
    int64_t nq, int64_t n, int64_t nrhs,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b,
    int64_t ctx_key)
{
    const int64_t ctx_handle = ::lorrax_ffi::ctx_registry::resolve(ctx_key, "cusolvermp");
    auto* ctx = reinterpret_cast<LorraxCusolverMpCtx*>(ctx_handle);
    if (ctx == nullptr) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "batched_solve_lu: ctx_handle is null");
    }
    const auto dtype = A.element_type();
    if (B.element_type() != dtype || X_out->element_type() != dtype || lu_type != dtype) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "batched_solve_lu: A, B, X_out, LU_out must share dtype");
    }
    switch (dtype) {
        case ffi::DataType::F64:
            return BatchedSolveLuImpl<double>(
                nq, n, nrhs, mb_a, nb_a, mb_b, nb_b, stream, ctx,
                static_cast<const double*>(A.untyped_data()),
                static_cast<const double*>(B.untyped_data()),
                static_cast<double*>(lu),
                static_cast<double*>(X_out->untyped_data()));
        case ffi::DataType::C128:
            using C128 = std::complex<double>;
            return BatchedSolveLuImpl<C128>(
                nq, n, nrhs, mb_a, nb_a, mb_b, nb_b, stream, ctx,
                static_cast<const C128*>(A.untyped_data()),
                static_cast<const C128*>(B.untyped_data()),
                static_cast<C128*>(lu),
                static_cast<C128*>(X_out->untyped_data()));
        default: {
            std::ostringstream os;
            os << "batched_solve_lu: unsupported dtype " << (int)dtype
               << " (supported: F64, C128)";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
    }
}

static ffi::Error BatchedSolveLuFactorsDispatch(
    cudaStream_t stream, ffi::AnyBuffer A, ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> X_out, ffi::Result<ffi::AnyBuffer> LU_out,
    int64_t nq, int64_t n, int64_t nrhs,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b, int64_t ctx_key)
{
    return BatchedSolveLu(stream, A, B, X_out, LU_out->untyped_data(), LU_out->element_type(),
                          nq, n, nrhs, mb_a, nb_a, mb_b, nb_b, ctx_key);
}

// The factors land in A, which this target does not declare as an output.
static ffi::Error BatchedSolveLuDispatch(
    cudaStream_t stream, ffi::AnyBuffer A, ffi::AnyBuffer B,
    ffi::Result<ffi::AnyBuffer> X_out,
    int64_t nq, int64_t n, int64_t nrhs,
    int64_t mb_a, int64_t nb_a, int64_t mb_b, int64_t nb_b, int64_t ctx_key)
{
    return BatchedSolveLu(stream, A, B, X_out, A.untyped_data(), A.element_type(),
                          nq, n, nrhs, mb_a, nb_a, mb_b, nb_b, ctx_key);
}

// Query-only (no matrix, no solve): the device and host workspace one solve allocates beside
// its operands per matrix -- getrf's and the chunked getrs's (separate buffers, both live) and
// the pivots -- at the descriptors BatchedSolveLuImpl builds, on the persistent context (no
// handle is created). Mp sizing reads no matrix data but rejects null pointers: the context's
// info allocation is the address token.
template <typename T>
static int lu_workspace(LorraxCusolverMpCtx* ctx, int64_t n, int64_t nrhs, int64_t mb_a,
                        int64_t nb_a, int64_t mb_b, int64_t nb_b, uint64_t* device_bytes,
                        uint64_t* host_bytes) {
    // Through the same plan a solve uses (lu_plan, getrs_chunk at this process's cap): sized once
    // per shape and width, then cached; collective wherever a width is sized. Mp sizing reads no
    // matrix data but rejects null pointers: the context's info allocation is the token.
    T* token = reinterpret_cast<T*>(ctx->d_info);
    int64_t* piv = reinterpret_cast<int64_t*>(ctx->d_info);
    LuPlan* plan = nullptr;
    int64_t k = 1;
    const GetrsChunk* chunk = nullptr;
    if (!lu_plan<T>(ctx, n, nrhs, mb_a, nb_a, mb_b, nb_b, true, true, token, piv, &plan).success()
        || !getrs_chunk<T>(ctx, *plan, n, nrhs, mb_b, nb_b, token, piv, token, getrs_cap(n, nrhs),
                           &k, &chunk).success())
        return -3;
    const int64_t lld_A = (n + ctx->p - 1) / ctx->p;
    *device_bytes = uint64_t(plan->d_ws_f + chunk->d_ws) + uint64_t(lld_A + mb_a) * sizeof(int64_t);
    *host_bytes = uint64_t(plan->h_ws_f + chunk->h_ws);
    return 0;
}

}  // namespace lorrax_ffi::cusolvermp_batched_solve_lu

extern "C" int lrx_lu_workspace_bytes(
    int64_t ctx_handle, int64_t n, int64_t nrhs, int64_t mb_a, int64_t nb_a, int64_t mb_b,
    int64_t nb_b, int complex128, uint64_t* device_bytes, uint64_t* host_bytes) {
    if (!device_bytes || !host_bytes || ctx_handle == 0 || n < 1 || nrhs < 1 || mb_a < 1 ||
        nb_a < 1 || mb_b < 1 || nb_b < 1 || (complex128 != 0 && complex128 != 1)) return -1;
    *device_bytes = 0; *host_bytes = 0;
    auto* ctx = reinterpret_cast<lorrax_ffi::cusolvermp::LorraxCusolverMpCtx*>(ctx_handle);
    namespace lu = lorrax_ffi::cusolvermp_batched_solve_lu;
    return complex128
        ? lu::lu_workspace<std::complex<double>>(ctx, n, nrhs, mb_a, nb_a, mb_b, nb_b,
                                                 device_bytes, host_bytes)
        : lu::lu_workspace<double>(ctx, n, nrhs, mb_a, nb_a, mb_b, nb_b, device_bytes, host_bytes);
}

// The room (bytes per rank) a fused n x nrhs solve's getrs chunks may take beside everything
// else its program holds (getrs_chunk); bytes < 0 withdraws it, back to kGetrsDefaultCap.  Each
// solve takes the smallest rank's figure.  An entry point, not a handler attribute: the compiled
// program never carries the room, and no ABI bump (lorrax_ffi_abi.h).
extern "C" int lrx_lu_getrs_room(int64_t n, int64_t nrhs, int64_t bytes) {
    namespace lu = lorrax_ffi::cusolvermp_batched_solve_lu;
    if (n < 1 || nrhs < 1) return -1;
    std::lock_guard<std::mutex> lock(lu::room_mu);
    if (bytes < 0) lu::posted_rooms.erase({n, nrhs});
    else lu::posted_rooms[{n, nrhs}] = uint64_t(bytes);
    return 0;
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CusolverMpBatchedSolveLuFactorsFfi,
    lorrax_ffi::cusolvermp_batched_solve_lu::BatchedSolveLuFactorsDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()      // A
        .Arg<xla::ffi::AnyBuffer>()      // B
        .Ret<xla::ffi::AnyBuffer>()      // X (aliased to B)
        .Ret<xla::ffi::AnyBuffer>()      // LU factors (aliased to A)
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("mb_a")
        .Attr<int64_t>("nb_a")
        .Attr<int64_t>("mb_b")
        .Attr<int64_t>("nb_b")
        .Attr<int64_t>("ctx_key"));

// Replaced by lorrax_cusolvermp_batched_solve_lu_factors; kept for ABI-6 trees, deleted at the next ABI bump.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CusolverMpBatchedSolveLuFfi,
    lorrax_ffi::cusolvermp_batched_solve_lu::BatchedSolveLuDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()      // A (the factors overwrite it)
        .Arg<xla::ffi::AnyBuffer>()      // B
        .Ret<xla::ffi::AnyBuffer>()      // X (aliased to B)
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("mb_a")
        .Attr<int64_t>("nb_a")
        .Attr<int64_t>("mb_b")
        .Attr<int64_t>("nb_b")
        .Attr<int64_t>("ctx_key"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CusolverMpBatchedGetrfFfi,
    lorrax_ffi::cusolvermp_batched_solve_lu::BatchedGetrfDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("mb")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ipiv_len")
        .Attr<int64_t>("ctx_key"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CusolverMpBatchedGetrsFfi,
    lorrax_ffi::cusolvermp_batched_solve_lu::BatchedGetrsDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nq")
        .Attr<int64_t>("n")
        .Attr<int64_t>("nrhs")
        .Attr<int64_t>("mb_a")
        .Attr<int64_t>("nb_a")
        .Attr<int64_t>("mb_b")
        .Attr<int64_t>("nb_b")
        .Attr<int64_t>("ipiv_len")
        .Attr<int64_t>("ctx_key"));
