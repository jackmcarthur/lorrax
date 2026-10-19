// rocsolver/lu_solve_ffi.cc — the ROCM leg's one-device LU solve A X = B on a correctly
// blocked triangular-solve path: rocsolver_{z,d}getrf, then every rocBLAS call on a
// sub-matrix whose byte span from its own base pointer is below 2^31.
//
// Why (ROCSOLVER_LU_REPORT.md; pool logs lurc_*): rocBLAS 5 {z,d}trsm, left side (the
// path XLA's TriangularSolve, jnp.linalg.solve and rocsolver_{z,d}getrs all reach), in its
// in-place blocked algorithm (chosen when m is a multiple of 128, ws ~ 160 m elements)
// returns a WRONG solution with status 0 once the 128-block diagonal offset of A crosses
// 2^31 BYTES: c128 n = 11648 exact, 11776 / 12288 / 14336 / 16384 wrong (resid 5e-3 .. 9e-3);
// f64 16384 exact, 16512 / 20480 / 24576 wrong.  rocblas_ztrsm_64 and rocsolver_zgetrs_64
// are wrong at the same sizes (it is not the int32 API).  rocsolver getrf is exact at every
// size measured (its panels and trailing updates are offset by rocSOLVER itself).
//
// The rule (explicit, from the root cause): no rocBLAS/rocSOLVER call below receives an
// operand spanning >= 2^31 bytes from its pointer.  Diagonal blocks are nb x nb at
// A + j0 (1 + lda) with nb = min(kNbPerf, floor(L / lda)), L = INT32_MAX / itemsize; RHS
// column chunks hold kc columns with (kc - 1) ldb + n <= L; GEMM row chunks the same.
// The offsets are formed here, in 64-bit pointer arithmetic.
//
// Two targets (the Python side owns layout; distrib_la.local_lu).  Every operand is the
// ROW-major XLA buffer read column-major: a row-major A is the column-major M = A^T.
//   RocsolverLuSolveFfi   in a [..., n, n], b [..., nrhs, n] (B^T row-major = B col-major)
//                         out lu (aliased to a), x [..., nrhs, n] (aliased to b), info [...]
//     getrf(M = A^T) = P L U, A = U^T L^T P^T: X = P (L^T)^{-1} (U^T)^{-1} B -- a forward
//     sweep op(U)=U^T, a backward sweep op(L)=L^T (transpose, not conjugate), then the
//     interchanges in reverse order (laswp, incx = -1).  No transpose of A is made.
//   RocsolverTriSolveFfi  in a [..., n, n] (read only), b [..., nrhs, n], piv [..., n] s32
//                         (1-based, mode 1 only); out x (aliased to b); attr mode:
//     1  apply jax's (lu, piv) of A (A[perm] = L U): laswp(+1), L (M upper = L^T, unit)
//        forward, U (M lower = U^T) backward  -- jax.scipy.linalg.lu_solve, trans=0
//     2  L x = b, L row-major lower (M = L^T upper): forward sweep op T
//     3  L^T x = b (M upper, op N): backward sweep; L^H x = b is mode 3 on conj(b), conj(x)
// Dtypes: f32, f64, c64, c128.
// No host-device synchronization; workspace from XLA's scratch, sized by a rocBLAS
// size-query pass over the same call sequence.
#include "../common/leg.h"
#if !defined(LRX_LEG_ROCM)
#  error "rocsolver/lu_solve_ffi.cc is ROCM-leg only (CUDA solves through jaxlib's cuSOLVER, exact at every n measured)"
#endif

#include <dlfcn.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>

#include "../common/ffi_helpers.h"
#include "../common/gpu_runtime.h"

namespace lorrax_ffi::rocsolver_lu {

namespace ffi = ::xla::ffi;

using RbHandle = void*;
using RbStatus = int;
using RbInt = int32_t;
constexpr int kOpNone = 111, kOpTrans = 112;
constexpr int kFillUpper = 121, kFillLower = 122;
constexpr int kDiagNonUnit = 131, kDiagUnit = 132;
constexpr int kSideLeft = 141;
//: Diagonal-block width when the span rule allows it: the trailing GEMMs run with k = nb.
constexpr int64_t kNbPerf = 1024;

// One C ABI per routine family; the element pointers are opaque (s, d, c, z share it).
using GetrfFn = RbStatus (*)(RbHandle, RbInt, RbInt, void*, RbInt, RbInt*, RbInt*);
using LaswpFn = RbStatus (*)(RbHandle, RbInt, void*, RbInt, RbInt, RbInt, const RbInt*, RbInt);
using TrsmFn = RbStatus (*)(RbHandle, int, int, int, int, RbInt, RbInt, const void*, const void*,
                            RbInt, void*, RbInt);
using GemmFn = RbStatus (*)(RbHandle, int, int, RbInt, RbInt, RbInt, const void*, const void*,
                            RbInt, const void*, RbInt, const void*, void*, RbInt);
enum Kind { kS = 0, kD = 1, kC = 2, kZ = 3 };

struct Api {
    RbStatus (*create_handle)(RbHandle*) = nullptr;
    RbStatus (*set_stream)(RbHandle, lrx::gpu::Stream) = nullptr;
    RbStatus (*set_workspace)(RbHandle, void*, size_t) = nullptr;
    RbStatus (*start_query)(RbHandle) = nullptr;
    RbStatus (*stop_query)(RbHandle, size_t*) = nullptr;
    GetrfFn getrf[4] = {};
    LaswpFn laswp[4] = {};
    TrsmFn trsm[4] = {};
    GemmFn gemm[4] = {};
    std::string path;
    std::string err;
};

static std::string loaded_path(void* sym) {
    Dl_info info{};
    return (dladdr(sym, &info) && info.dli_fname) ? std::string(info.dli_fname) : std::string("?");
}

static const Api& api() {
    static const Api a = [] {
        Api r;
        constexpr const char* kSoname = "librocsolver.so.0";
        void* h = dlopen(kSoname, RTLD_NOW | RTLD_NOLOAD);
        if (h == nullptr) {
            r.err = std::string("GATE rocsolver-mapped: ") + kSoname + " is not mapped in this "
                    "process; want the copy the JAX ROCm plugin maps (one ROCm per process); "
                    "fix: import jax with the rocm plugin before the first call";
            return r;
        }
        auto need = [&](const std::string& name) {
            void* p = dlsym(h, name.c_str());   // the library, then its dependencies (rocBLAS)
            if (p == nullptr) r.err += (r.err.empty() ? "" : ", ") + name;
            return p;
        };
        r.create_handle = reinterpret_cast<decltype(r.create_handle)>(need("rocblas_create_handle"));
        r.set_stream = reinterpret_cast<decltype(r.set_stream)>(need("rocblas_set_stream"));
        r.set_workspace = reinterpret_cast<decltype(r.set_workspace)>(need("rocblas_set_workspace"));
        r.start_query = reinterpret_cast<decltype(r.start_query)>(need("rocblas_start_device_memory_size_query"));
        r.stop_query = reinterpret_cast<decltype(r.stop_query)>(need("rocblas_stop_device_memory_size_query"));
        const char* p = "sdcz";
        for (int k = 0; k < 4; ++k) {
            const std::string c(1, p[k]);
            r.getrf[k] = reinterpret_cast<GetrfFn>(need("rocsolver_" + c + "getrf"));
            r.laswp[k] = reinterpret_cast<LaswpFn>(need("rocsolver_" + c + "laswp"));
            r.trsm[k] = reinterpret_cast<TrsmFn>(need("rocblas_" + c + "trsm"));
            r.gemm[k] = reinterpret_cast<GemmFn>(need("rocblas_" + c + "gemm"));
        }
        if (!r.err.empty()) {
            r.err = "GATE rocsolver-symbols: dlsym found no " + r.err + " in " +
                    loaded_path(dlsym(h, "rocsolver_get_version_string"));
            return r;
        }
        r.path = loaded_path(reinterpret_cast<void*>(r.getrf[kZ]));
        return r;
    }();
    return a;
}

struct DeviceHandle {
    RbHandle h = nullptr;
    std::mutex m;
    std::map<std::tuple<int, int, int64_t, int64_t>, size_t> ws;   // (target, dtype, n, nrhs) -> B
};

static ffi::Error device_handle(DeviceHandle** out) {
    static std::mutex mu;
    static std::map<int, std::unique_ptr<DeviceHandle>> handles;
    int dev = -1;
    LORRAX_GPU_CHECK(lrx::gpu::get_device(&dev));
    std::lock_guard<std::mutex> lk(mu);
    auto& slot = handles[dev];
    if (!slot) {
        auto d = std::make_unique<DeviceHandle>();
        LORRAX_LIB_CHECK(api().create_handle(&d->h), 0, "rocblas_create_handle");
        slot = std::move(d);
    }
    *out = slot.get();
    return ffi::Error::Success();
}

// The blocking of one (n, nrhs, itemsize) solve, from the span rule alone.
struct Blocking {
    int64_t nb = 0;      // diagonal block width
    int64_t kc = 0;      // RHS columns per chunk
    int64_t limit = 0;   // L = INT32_MAX / itemsize: the largest element span a call may get
};

static Blocking blocking(int64_t n, int64_t nrhs, size_t esize) {
    Blocking b;
    b.limit = static_cast<int64_t>(INT32_MAX) / static_cast<int64_t>(esize);
    b.nb = std::max<int64_t>(1, std::min<int64_t>({kNbPerf, n, b.limit / n}));
    b.kc = std::max<int64_t>(1, std::min<int64_t>(nrhs, (b.limit - n) / n + 1));
    return b;
}

// Rows per transposed-panel GEMM so that the panel (k x rows, ld lda) spans <= L elements.
static int64_t gemm_rows(int64_t k, int64_t lda, int64_t limit) {
    return std::max<int64_t>(1, (limit - k) / lda + 1);
}

// The calls of one element kind on one handle.  Element offsets become byte offsets here.
struct Ops {
    const Api& A;
    RbHandle h;
    int kind;
    size_t es;
    alignas(16) unsigned char one[16] = {}, mone[16] = {};
    Ops(const Api& a, RbHandle hh, int k) : A(a), h(hh), kind(k), es(k == kS ? 4 : (k == kZ ? 16 : 8)) {
        if (k == kS || k == kC) { float o = 1.f, m = -1.f; std::memcpy(one, &o, 4); std::memcpy(mone, &m, 4); }
        else { double o = 1.0, m = -1.0; std::memcpy(one, &o, 8); std::memcpy(mone, &m, 8); }
    }
    char* at(void* base, int64_t i, int64_t j, int64_t ld) const {
        return static_cast<char*>(base) + static_cast<size_t>(i + j * ld) * es;
    }
    RbStatus getrf(int64_t n, void* a, RbInt* ipiv, RbInt* info) const {
        return A.getrf[kind](h, (RbInt)n, (RbInt)n, a, (RbInt)n, ipiv, info);
    }
    RbStatus laswp(int64_t cols, void* b, int64_t ld, int64_t n, const RbInt* ipiv, int incx) const {
        return A.laswp[kind](h, (RbInt)cols, b, (RbInt)ld, 1, (RbInt)n, ipiv, incx);
    }
    RbStatus trsm(int fill, int op, int diag, int64_t m, int64_t cols, const void* a, int64_t lda,
                  void* b, int64_t ldb) const {
        return A.trsm[kind](h, kSideLeft, fill, op, diag, (RbInt)m, (RbInt)cols, one, a, (RbInt)lda, b, (RbInt)ldb);
    }
    // C (m x cols) -= op(A) B, op(A) m x k
    RbStatus gemm(int op, int64_t m, int64_t cols, int64_t k, const void* a, int64_t lda, const void* b,
                  int64_t ldb, void* c, int64_t ldc) const {
        return A.gemm[kind](h, op, kOpNone, (RbInt)m, (RbInt)cols, (RbInt)k, mone, a, (RbInt)lda, b,
                            (RbInt)ldb, one, c, (RbInt)ldc);
    }
};

// Sweeps over the column-major M (n x n, ld n) on one RHS chunk Bc (n x kc, ld n), in place.
// Returns the first non-success status; in rocBLAS size-query mode (`query`) statuses are
// size statuses and are not read (the whole sequence is issued).
struct Sweep {
    const Ops& op;
    const Blocking& bk;
    int64_t n;
    bool query;
    std::string* where;
    // op(M) = M^T, triangle stored in `fill`: forward (fill upper: op(M) lower) or backward.
    RbStatus trans(bool forward, int fill, int diag, const void* M, void* B, int64_t kc) const {
        const int64_t rows_per_gemm = gemm_rows(bk.nb, n, bk.limit);
        const int64_t last = ((n - 1) / bk.nb) * bk.nb;
        RbStatus st = 0;
        for (int64_t s = 0; s <= last; s += bk.nb) {
            const int64_t j0 = forward ? s : last - s, ib = std::min(bk.nb, n - j0), j1 = j0 + ib;
            if ((st = op.trsm(fill, kOpTrans, diag, ib, kc, op.at(const_cast<void*>(M), j0, j0, n), n,
                              op.at(B, j0, 0, n), n)) && !query) { *where = "trsm(T)"; return st; }
            const int64_t r_lo = forward ? j1 : 0, r_hi = forward ? n : j0;
            for (int64_t r0 = r_lo; r0 < r_hi; r0 += rows_per_gemm) {   // B[r] -= M[j0:j1, r]^T X_j
                const int64_t m = std::min(rows_per_gemm, r_hi - r0);
                if ((st = op.gemm(kOpTrans, m, kc, ib, op.at(const_cast<void*>(M), j0, r0, n), n,
                                  op.at(B, j0, 0, n), n, op.at(B, r0, 0, n), n)) && !query) { *where = "gemm(T)"; return st; }
            }
        }
        return 0;
    }
    // op(M) = M, M upper: backward sweep (B[0:j0] -= M[0:j0, j0:j1] X_j; panel span <= nb n).
    RbStatus none_upper(int diag, const void* M, void* B, int64_t kc) const {
        const int64_t last = ((n - 1) / bk.nb) * bk.nb;
        RbStatus st = 0;
        for (int64_t j0 = last; j0 >= 0; j0 -= bk.nb) {
            const int64_t ib = std::min(bk.nb, n - j0);
            if ((st = op.trsm(kFillUpper, kOpNone, diag, ib, kc, op.at(const_cast<void*>(M), j0, j0, n), n,
                              op.at(B, j0, 0, n), n)) && !query) { *where = "trsm(N)"; return st; }
            if (j0 > 0 && (st = op.gemm(kOpNone, j0, kc, ib, op.at(const_cast<void*>(M), 0, j0, n), n,
                                        op.at(B, j0, 0, n), n, B, n)) && !query) { *where = "gemm(N)"; return st; }
        }
        return 0;
    }
};

static int kind_of(ffi::DataType dt) {
    switch (dt) {
        case ffi::DataType::F32: return kS;
        case ffi::DataType::F64: return kD;
        case ffi::DataType::C64: return kC;
        case ffi::DataType::C128: return kZ;
        default: return -1;
    }
}

// One solve per matrix q of the stack; mode 0 = factor + solve, 1..3 = the TriSolve modes.
static RbStatus one_matrix(const Ops& op, const Blocking& bk, int mode, int64_t n, int64_t nrhs,
                           void* a, void* b, const RbInt* piv_in, RbInt* ipiv, RbInt* info,
                           bool query, std::string* where) {
    RbStatus st = 0;
    if (mode == 0 && (st = op.getrf(n, a, ipiv, info)) && !query) { *where = "getrf"; return st; }
    const RbInt* pv = mode == 0 ? ipiv : piv_in;
    const Sweep sw{op, bk, n, query, where};
    for (int64_t c0 = 0; c0 < nrhs; c0 += bk.kc) {
        const int64_t kc = std::min(bk.kc, nrhs - c0);
        void* B = op.at(b, 0, c0, n);
        switch (mode) {
            case 0:
                if ((st = sw.trans(true, kFillUpper, kDiagNonUnit, a, B, kc))) return st;
                if ((st = sw.trans(false, kFillLower, kDiagUnit, a, B, kc))) return st;
                if ((st = op.laswp(kc, B, n, n, pv, -1)) && !query) { *where = "laswp(-1)"; return st; }
                break;
            case 1:
                if ((st = op.laswp(kc, B, n, n, pv, 1)) && !query) { *where = "laswp(+1)"; return st; }
                if ((st = sw.trans(true, kFillUpper, kDiagUnit, a, B, kc))) return st;
                if ((st = sw.trans(false, kFillLower, kDiagNonUnit, a, B, kc))) return st;
                break;
            case 2:
                if ((st = sw.trans(true, kFillUpper, kDiagNonUnit, a, B, kc))) return st;
                break;
            case 3:
                if ((st = sw.none_upper(kDiagNonUnit, a, B, kc))) return st;
                break;
        }
    }
    return 0;
}

static ffi::Error run(int mode, lrx::gpu::Stream stream, ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer a, ffi::AnyBuffer b, const RbInt* piv_in, void* lu_out,
                      void* x_out, RbInt* info_out) {
    const Api& Ap = api();
    if (!Ap.err.empty()) return ffi::Error(ffi::ErrorCode::kFailedPrecondition, Ap.err);
    const auto da = a.dimensions();
    const auto db = b.dimensions();
    const size_t rank = da.size();
    if (rank < 2 || da[rank - 1] != da[rank - 2] || db.size() != rank || db[rank - 1] != da[rank - 1]) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "GATE rocsolver-lu-shape: want a [..., n, n] and b [..., nrhs, n] (B^T)");
    }
    for (size_t i = 0; i + 2 < rank; ++i) {
        if (da[i] != db[i]) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "GATE rocsolver-lu-shape: batch dims differ");
    }
    const int kind = kind_of(a.element_type());
    if (kind < 0 || b.element_type() != a.element_type()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "GATE rocsolver-lu-dtype: want float32, float64, complex64 or complex128 a and b alike");
    }
    if (mode < 0 || mode > 3) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "GATE rocsolver-lu-mode: want 0..3");
    const int64_t n = da[rank - 1], nrhs = db[rank - 2];
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < rank; ++i) batch *= da[i];
    if (n == 0 || nrhs == 0 || batch == 0) return ffi::Error::Success();
    if (n > INT32_MAX || nrhs > INT32_MAX) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "GATE rocsolver-lu-int32: n and nrhs must fit rocblas_int");
    }
    const Blocking bk = blocking(n, nrhs, kind == kS ? 4 : (kind == kZ ? 16 : 8));
    DeviceHandle* dh = nullptr;
    FFI_RETURN_IF_ERROR(device_handle(&dh));
    const Ops op(Ap, dh->h, kind);
    const size_t a_bytes = static_cast<size_t>(n) * n * op.es, b_bytes = static_cast<size_t>(n) * nrhs * op.es;
    if (lu_out != nullptr && lu_out != a.untyped_data()) {
        LORRAX_GPU_CHECK(lrx::gpu::memcpy_async(lu_out, a.untyped_data(), batch * a_bytes, lrx::gpu::kDeviceToDevice, stream));
    }
    if (x_out != b.untyped_data()) {
        LORRAX_GPU_CHECK(lrx::gpu::memcpy_async(x_out, b.untyped_data(), batch * b_bytes, lrx::gpu::kDeviceToDevice, stream));
    }
    RbInt* ip = nullptr;
    if (mode == 0) {
        std::optional<void*> piv = scratch.Allocate(static_cast<size_t>(n) * sizeof(RbInt));
        if (!piv) return ffi::Error(ffi::ErrorCode::kResourceExhausted, "scratch for the pivots");
        ip = static_cast<RbInt*>(*piv);
    }
    char* const a0 = static_cast<char*>(mode == 0 ? lu_out : const_cast<void*>(a.untyped_data()));
    char* const x0 = static_cast<char*>(x_out);
    std::lock_guard<std::mutex> lk(dh->m);
    LORRAX_LIB_CHECK(Ap.set_stream(dh->h, stream), 0, "rocblas_set_stream");
    std::string where;
    auto matrix = [&](int64_t q, bool query) {
        return one_matrix(op, bk, mode, n, nrhs, a0 + q * a_bytes, x0 + q * b_bytes,
                          piv_in ? piv_in + q * n : nullptr, ip, info_out ? info_out + q : nullptr, query, &where);
    };
    // Workspace: one size-query pass over the same call sequence (host only: nothing is enqueued).
    const auto key = std::make_tuple(mode == 0 ? 0 : 1 + mode, kind, n, nrhs);
    auto it = dh->ws.find(key);
    if (it == dh->ws.end()) {
        size_t bytes = 0;
        LORRAX_LIB_CHECK(Ap.start_query(dh->h), 0, "rocblas_start_device_memory_size_query");
        const RbStatus q = matrix(0, true);
        const RbStatus s = Ap.stop_query(dh->h, &bytes);
        if (s != 0) {
            std::ostringstream os;
            os << "rocblas_stop_device_memory_size_query status=" << s << " (solver query " << q << " at " << where << ")";
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
        it = dh->ws.emplace(key, bytes).first;
    }
    const size_t bytes = it->second;
    void* ws = nullptr;
    if (bytes > 0) {
        std::optional<void*> p = scratch.Allocate(bytes);
        if (!p) return ffi::Error(ffi::ErrorCode::kResourceExhausted, "scratch for rocBLAS workspace");
        ws = *p;
    }
    LORRAX_LIB_CHECK(Ap.set_workspace(dh->h, ws, bytes), 0, "rocblas_set_workspace");
    for (int64_t q = 0; q < batch; ++q) {
        const RbStatus st = matrix(q, false);
        if (st != 0) {
            std::ostringstream os;
            os << "rocsolver lu/tri solve (mode " << mode << "): " << where << " status=" << st << " (n=" << n
               << ", nrhs=" << nrhs << ", matrix " << q << " of " << batch << ", nb=" << bk.nb << ", kc=" << bk.kc
               << ", ws=" << bytes << " B, " << Ap.path << ")";
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
    }
    return ffi::Error::Success();
}

static ffi::Error lu_solve(lrx::gpu::Stream stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer a,
                           ffi::AnyBuffer b, ffi::Result<ffi::AnyBuffer> lu, ffi::Result<ffi::AnyBuffer> x,
                           ffi::Result<ffi::Buffer<ffi::S32>> info) {
    if (lu->element_type() != a.element_type() || x->element_type() != b.element_type()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "GATE rocsolver-lu-dtype: lu, x like a, b");
    }
    return run(0, stream, scratch, a, b, nullptr, lu->untyped_data(), x->untyped_data(), info->typed_data());
}

static ffi::Error tri_solve(lrx::gpu::Stream stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer a,
                            ffi::AnyBuffer b, ffi::Buffer<ffi::S32> piv, ffi::Result<ffi::AnyBuffer> x,
                            int64_t mode) {
    if (x->element_type() != b.element_type()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "GATE rocsolver-lu-dtype: x like b");
    }
    if (mode == 1 && piv.element_count() != a.element_count() / std::max<int64_t>(1, a.dimensions().back())) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "GATE rocsolver-lu-piv: want piv [..., n] for mode 1");
    }
    return run(static_cast<int>(mode), stream, scratch, a, b, mode == 1 ? piv.typed_data() : nullptr,
               nullptr, x->untyped_data(), nullptr);
}

}  // namespace lorrax_ffi::rocsolver_lu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RocsolverLuSolveFfi, lorrax_ffi::rocsolver_lu::lu_solve,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()                    // a  [..., n, n]   row-major A (= A^T col-major)
        .Arg<xla::ffi::AnyBuffer>()                    // b  [..., nrhs, n] B^T row-major (= B col-major)
        .Ret<xla::ffi::AnyBuffer>()                    // lu [..., n, n]   (aliased to a)
        .Ret<xla::ffi::AnyBuffer>()                    // x  [..., nrhs, n] (aliased to b)
        .Ret<xla::ffi::Buffer<xla::ffi::S32>>());      // info [...]

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RocsolverTriSolveFfi, lorrax_ffi::rocsolver_lu::tri_solve,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()                    // a   [..., n, n]  read only (LU, or L)
        .Arg<xla::ffi::AnyBuffer>()                    // b   [..., nrhs, n] B^T row-major
        .Arg<xla::ffi::Buffer<xla::ffi::S32>>()        // piv [..., n] 1-based (mode 1; else unused)
        .Ret<xla::ffi::AnyBuffer>()                    // x   [..., nrhs, n] (aliased to b)
        .Attr<int64_t>("mode"));                       // 1 lu_solve, 2 L x = b, 3 L^T x = b
