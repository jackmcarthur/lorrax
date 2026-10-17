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
// Contract (the Python side owns layout; distrib_la.local_lu):
//   in   a     [..., n, n]     c128 | f64, the ROW-major A, read column-major: A^T
//   in   b     [..., nrhs, n]  same dtype, B^T row-major = B column-major (ld n)
//   out  lu    [..., n, n]     getrf of A^T (aliased to a)
//   out  x     [..., nrhs, n]  X column-major = X^T row-major (aliased to b)
//   out  info  [...]           s32 getrf info per matrix (0 = nonsingular)
// With A^T = P L U (getrf), A = U^T L^T P^T, so X = P (L^T)^{-1} (U^T)^{-1} B: a forward
// sweep with op(U) = U^T, a backward sweep with op(L) = L^T (transpose, not conjugate),
// then the interchanges in reverse order (laswp, incx = -1).  No transpose of A is made.
// No host-device synchronization; workspace from XLA's scratch, sized by a rocBLAS
// size-query pass over the same call sequence.
#include "../common/leg.h"
#if !defined(LRX_LEG_ROCM)
#  error "rocsolver/lu_solve_ffi.cc is ROCM-leg only (CUDA solves through jaxlib's cuSOLVER, exact at every n measured)"
#endif

#include <dlfcn.h>

#include <algorithm>
#include <climits>
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

struct Api {
    RbStatus (*create_handle)(RbHandle*) = nullptr;
    RbStatus (*set_stream)(RbHandle, lrx::gpu::Stream) = nullptr;
    RbStatus (*set_workspace)(RbHandle, void*, size_t) = nullptr;
    RbStatus (*start_query)(RbHandle) = nullptr;
    RbStatus (*stop_query)(RbHandle, size_t*) = nullptr;
    RbStatus (*zgetrf)(RbHandle, RbInt, RbInt, void*, RbInt, RbInt*, RbInt*) = nullptr;
    RbStatus (*dgetrf)(RbHandle, RbInt, RbInt, double*, RbInt, RbInt*, RbInt*) = nullptr;
    RbStatus (*zlaswp)(RbHandle, RbInt, void*, RbInt, RbInt, RbInt, const RbInt*, RbInt) = nullptr;
    RbStatus (*dlaswp)(RbHandle, RbInt, double*, RbInt, RbInt, RbInt, const RbInt*, RbInt) = nullptr;
    RbStatus (*ztrsm)(RbHandle, int, int, int, int, RbInt, RbInt, const void*, const void*, RbInt,
                      void*, RbInt) = nullptr;
    RbStatus (*dtrsm)(RbHandle, int, int, int, int, RbInt, RbInt, const double*, const double*,
                      RbInt, double*, RbInt) = nullptr;
    RbStatus (*zgemm)(RbHandle, int, int, RbInt, RbInt, RbInt, const void*, const void*, RbInt,
                      const void*, RbInt, const void*, void*, RbInt) = nullptr;
    RbStatus (*dgemm)(RbHandle, int, int, RbInt, RbInt, RbInt, const double*, const double*,
                      RbInt, const double*, RbInt, const double*, double*, RbInt) = nullptr;
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
        auto need = [&](const char* name) {
            void* p = dlsym(h, name);       // the library, then its dependencies (rocBLAS)
            if (p == nullptr) r.err += std::string(r.err.empty() ? "" : ", ") + name;
            return p;
        };
#define LRX_BIND(field, name) r.field = reinterpret_cast<decltype(r.field)>(need(name))
        LRX_BIND(create_handle, "rocblas_create_handle");
        LRX_BIND(set_stream, "rocblas_set_stream");
        LRX_BIND(set_workspace, "rocblas_set_workspace");
        LRX_BIND(start_query, "rocblas_start_device_memory_size_query");
        LRX_BIND(stop_query, "rocblas_stop_device_memory_size_query");
        LRX_BIND(zgetrf, "rocsolver_zgetrf");
        LRX_BIND(dgetrf, "rocsolver_dgetrf");
        LRX_BIND(zlaswp, "rocsolver_zlaswp");
        LRX_BIND(dlaswp, "rocsolver_dlaswp");
        LRX_BIND(ztrsm, "rocblas_ztrsm");
        LRX_BIND(dtrsm, "rocblas_dtrsm");
        LRX_BIND(zgemm, "rocblas_zgemm");
        LRX_BIND(dgemm, "rocblas_dgemm");
#undef LRX_BIND
        if (!r.err.empty()) {
            r.err = "GATE rocsolver-symbols: dlsym found no " + r.err + " in " +
                    loaded_path(dlsym(h, "rocsolver_get_version_string"));
            return r;
        }
        r.path = loaded_path(reinterpret_cast<void*>(r.zgetrf));
        return r;
    }();
    return a;
}

struct DeviceHandle {
    RbHandle h = nullptr;
    std::mutex m;
    std::map<std::tuple<int, int64_t, int64_t>, size_t> ws;   // (dtype, n, nrhs) -> bytes, under m
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

// Rows per GEMM call so that the transposed panel A (k x rows, ld lda) spans <= L elements.
static int64_t gemm_rows(int64_t k, int64_t lda, int64_t limit) {
    return std::max<int64_t>(1, (limit - k) / lda + 1);
}

template <bool kCplx>
struct Ops {
    const Api& A;
    RbHandle h;
    static constexpr int64_t kStride = kCplx ? 2 : 1;      // doubles per element
    double one[2] = {1.0, 0.0}, mone[2] = {-1.0, 0.0};
    RbStatus getrf(int64_t n, double* a, RbInt* ipiv, RbInt* info) const {
        const RbInt ni = static_cast<RbInt>(n);
        return kCplx ? A.zgetrf(h, ni, ni, a, ni, ipiv, info) : A.dgetrf(h, ni, ni, a, ni, ipiv, info);
    }
    RbStatus laswp_rev(int64_t cols, double* b, int64_t ldb, int64_t n, const RbInt* ipiv) const {
        return kCplx ? A.zlaswp(h, (RbInt)cols, b, (RbInt)ldb, 1, (RbInt)n, ipiv, -1)
                     : A.dlaswp(h, (RbInt)cols, b, (RbInt)ldb, 1, (RbInt)n, ipiv, -1);
    }
    RbStatus trsm(int fill, int diag, int64_t m, int64_t cols, const double* a, int64_t lda,
                  double* b, int64_t ldb) const {
        return kCplx ? A.ztrsm(h, kSideLeft, fill, kOpTrans, diag, (RbInt)m, (RbInt)cols, one, a,
                               (RbInt)lda, b, (RbInt)ldb)
                     : A.dtrsm(h, kSideLeft, fill, kOpTrans, diag, (RbInt)m, (RbInt)cols, one, a,
                               (RbInt)lda, b, (RbInt)ldb);
    }
    // C (m x cols) -= op(A)^T-panel (A stored k x m, ld lda) * B (k x cols)
    RbStatus gemm_t(int64_t m, int64_t cols, int64_t k, const double* a, int64_t lda,
                    const double* b, int64_t ldb, double* c, int64_t ldc) const {
        return kCplx ? A.zgemm(h, kOpTrans, kOpNone, (RbInt)m, (RbInt)cols, (RbInt)k, mone, a,
                               (RbInt)lda, b, (RbInt)ldb, one, c, (RbInt)ldc)
                     : A.dgemm(h, kOpTrans, kOpNone, (RbInt)m, (RbInt)cols, (RbInt)k, mone, a,
                               (RbInt)lda, b, (RbInt)ldb, one, c, (RbInt)ldc);
    }
};

// One matrix: a (n x n, col-major A^T, factored in place), b (n x nrhs col-major, solved in
// place).  Returns the first non-success rocBLAS/rocSOLVER status (0 = all succeeded).  In
// rocBLAS size-query mode (`query`) every call returns a size status, not 0: the whole
// sequence is issued and the statuses are not read.
template <bool kCplx>
static RbStatus solve_one(const Ops<kCplx>& op, const Blocking& bk, int64_t n, int64_t nrhs,
                          double* a, double* b, RbInt* ipiv, RbInt* info, std::string* where,
                          bool query) {
    constexpr int64_t s = Ops<kCplx>::kStride;
    auto A_ = [&](int64_t i, int64_t j) { return a + (i + j * n) * s; };
    RbStatus st = op.getrf(n, a, ipiv, info);
    if (st != 0 && !query) { *where = "getrf"; return st; }
    const int64_t rows_per_gemm = gemm_rows(bk.nb, n, bk.limit);
    for (int64_t c0 = 0; c0 < nrhs; c0 += bk.kc) {
        const int64_t kc = std::min(bk.kc, nrhs - c0);
        double* B = b + c0 * n * s;
        auto B_ = [&](int64_t i) { return B + i * s; };
        // forward: U^T W = B   (blocks ascending; U^T is lower)
        for (int64_t j0 = 0; j0 < n; j0 += bk.nb) {
            const int64_t ib = std::min(bk.nb, n - j0), j1 = j0 + ib;
            if ((st = op.trsm(kFillUpper, kDiagNonUnit, ib, kc, A_(j0, j0), n, B_(j0), n)) && !query) { *where = "trsm U^T"; return st; }
            for (int64_t r0 = j1; r0 < n; r0 += rows_per_gemm) {      // B[r0:r1] -= U[j0:j1, r0:r1]^T W_j
                const int64_t m = std::min(rows_per_gemm, n - r0);
                if ((st = op.gemm_t(m, kc, ib, A_(j0, r0), n, B_(j0), n, B_(r0), n)) && !query) { *where = "gemm U^T"; return st; }
            }
        }
        // backward: L^T V = W   (blocks descending; L^T is unit upper)
        for (int64_t j0 = ((n - 1) / bk.nb) * bk.nb; j0 >= 0; j0 -= bk.nb) {
            const int64_t ib = std::min(bk.nb, n - j0);
            if ((st = op.trsm(kFillLower, kDiagUnit, ib, kc, A_(j0, j0), n, B_(j0), n)) && !query) { *where = "trsm L^T"; return st; }
            for (int64_t r0 = 0; r0 < j0; r0 += rows_per_gemm) {      // B[r0:r1] -= L[j0:j1, r0:r1]^T V_j
                const int64_t m = std::min(rows_per_gemm, j0 - r0);
                if ((st = op.gemm_t(m, kc, ib, A_(j0, r0), n, B_(j0), n, B_(r0), n)) && !query) { *where = "gemm L^T"; return st; }
            }
        }
        // X = P V: the interchanges in reverse order
        if ((st = op.laswp_rev(kc, B, n, n, ipiv)) && !query) { *where = "laswp"; return st; }
    }
    return 0;
}

static ffi::Error run(lrx::gpu::Stream stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer a,
                      ffi::AnyBuffer b, ffi::Result<ffi::AnyBuffer> lu,
                      ffi::Result<ffi::AnyBuffer> x, ffi::Result<ffi::Buffer<ffi::S32>> info) {
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
    const ffi::DataType dt = a.element_type();
    if ((dt != ffi::DataType::C128 && dt != ffi::DataType::F64) || b.element_type() != dt ||
        lu->element_type() != dt || x->element_type() != dt) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "GATE rocsolver-lu-dtype: want complex128 or float64 a, b, lu, x");
    }
    const int64_t n = da[rank - 1], nrhs = db[rank - 2];
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < rank; ++i) batch *= da[i];
    if (n == 0 || nrhs == 0 || batch == 0) return ffi::Error::Success();
    if (n > INT32_MAX || nrhs > INT32_MAX) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "GATE rocsolver-lu-int32: n and nrhs must fit rocblas_int");
    }
    const bool cplx = dt == ffi::DataType::C128;
    const size_t esize = cplx ? 16 : 8;
    const size_t a_bytes = static_cast<size_t>(n) * n * esize, b_bytes = static_cast<size_t>(n) * nrhs * esize;
    if (lu->untyped_data() != a.untyped_data()) {
        LORRAX_GPU_CHECK(lrx::gpu::memcpy_async(lu->untyped_data(), a.untyped_data(), batch * a_bytes,
                                                lrx::gpu::kDeviceToDevice, stream));
    }
    if (x->untyped_data() != b.untyped_data()) {
        LORRAX_GPU_CHECK(lrx::gpu::memcpy_async(x->untyped_data(), b.untyped_data(), batch * b_bytes,
                                                lrx::gpu::kDeviceToDevice, stream));
    }
    std::optional<void*> piv = scratch.Allocate(static_cast<size_t>(n) * sizeof(RbInt));
    if (!piv) return ffi::Error(ffi::ErrorCode::kResourceExhausted, "scratch for the pivots");
    const Blocking bk = blocking(n, nrhs, esize);
    char* const lu0 = static_cast<char*>(lu->untyped_data());
    char* const x0 = static_cast<char*>(x->untyped_data());
    RbInt* const ip = static_cast<RbInt*>(*piv);
    RbInt* const inf = info->typed_data();

    DeviceHandle* dh = nullptr;
    FFI_RETURN_IF_ERROR(device_handle(&dh));
    std::lock_guard<std::mutex> lk(dh->m);
    LORRAX_LIB_CHECK(Ap.set_stream(dh->h, stream), 0, "rocblas_set_stream");
    std::string where;
    bool query = false;
    auto one_matrix = [&](int64_t q) -> RbStatus {
        double* aq = reinterpret_cast<double*>(lu0 + q * a_bytes);
        double* bq = reinterpret_cast<double*>(x0 + q * b_bytes);
        return cplx ? solve_one(Ops<true>{Ap, dh->h}, bk, n, nrhs, aq, bq, ip, inf + q, &where, query)
                    : solve_one(Ops<false>{Ap, dh->h}, bk, n, nrhs, aq, bq, ip, inf + q, &where, query);
    };
    // Workspace: one size-query pass over the same call sequence (host only: nothing is enqueued).
    const auto key = std::make_tuple(static_cast<int>(dt), n, nrhs);
    auto it = dh->ws.find(key);
    if (it == dh->ws.end()) {
        size_t bytes = 0;
        LORRAX_LIB_CHECK(Ap.start_query(dh->h), 0, "rocblas_start_device_memory_size_query");
        query = true;
        const RbStatus q = one_matrix(0);
        query = false;
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
        const RbStatus st = one_matrix(q);
        if (st != 0) {
            std::ostringstream os;
            os << "rocsolver lu_solve: " << where << " status=" << st << " (n=" << n << ", nrhs=" << nrhs
               << ", matrix " << q << " of " << batch << ", nb=" << bk.nb << ", kc=" << bk.kc
               << ", ws=" << bytes << " B, " << Ap.path << ")";
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
    }
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::rocsolver_lu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RocsolverLuSolveFfi, lorrax_ffi::rocsolver_lu::run,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()                    // a  [..., n, n]   row-major A (= A^T col-major)
        .Arg<xla::ffi::AnyBuffer>()                    // b  [..., nrhs, n] B^T row-major (= B col-major)
        .Ret<xla::ffi::AnyBuffer>()                    // lu [..., n, n]   (aliased to a)
        .Ret<xla::ffi::AnyBuffer>()                    // x  [..., nrhs, n] (aliased to b)
        .Ret<xla::ffi::Buffer<xla::ffi::S32>>());      // info [...]
