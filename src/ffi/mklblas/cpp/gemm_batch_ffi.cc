// gemm_batch_ffi.cc — MKL batched-GEMM host handler, HOST platform (JAX CPU
// backend).  The gated CPU GEMM body of the contract_bands_block_reshard
// primitive (src/common/contract_bands.py, LORRAX_BANDS_GEMM_FFI) —
// wk_REL/RESHARD_OVERHEAD_MEMO.md Sec. 4.4 exit (b) / Sec. 7 lever 1.
//
// WHY IT EXISTS: the primitive's large right contraction is the measured
// Σ project_rs wall — XLA:CPU runs it through Eigen at 295 GF/s (promoted
// zgemm) / ~172 GF/s (split dgemm) where the same node's MKL BLAS runs the
// identical contraction at ~1263 GF/s (memo Sec. 4.4, jobs 7878883/7878907/
// 7878942).  This handler is the FFT-FFI playbook applied to that GEMM:
// dispatch on buffer dtype, MKL-internal threading under the workstream-AW
// MklThreadScope pattern, announce-or-refuse on the Python side.  The CUDA
// path is deliberately NOT served — XLA:GPU's dot lowering already hits
// cuBLAS (optimal); the Python gate refuses the flag on non-CPU meshes.
//
// Handler (registered by ffi_loader.py under platform="cpu"):
//   MklBlasGemmBatchHostFfi (target lorrax_mklblas_gemm_batch)
//       A (BA, M, K), B (BB, K, N)  ->  C (BA, M, N),  BA % BB == 0,
//       C[i] = A[i] @ B[i % BB]     (row-major, NoTrans/NoTrans),
//       dtype f64 (cblas_dgemm_batch) or c128 (cblas_zgemm_batch), both
//       operands the SAME dtype (the primitive's de-promotion policy
//       guarantees no mixed real/complex GEMM ever reaches this point).
//   The B-cycling broadcast rule serves both the plain per-k batch
//   (BA == BB == nk) and the extra-stacked batch (BA == E·nk vs the
//   k-only ψ operand, stack axis OUTERMOST in A).
//
// In-place: NONE — a (BA, M, N) GEMM output never legally aliases a
// (BA, M, K)/(BB, K, N) operand buffer, so no input_output_aliases are
// declared (contrast the in-place mklfft handlers, where shapes match).
//
// Threading: ONE cblas_*gemm_batch call per invocation; MKL parallelizes
// internally.  The calling (XLA host-callback) thread pins MKL to
// LORRAX_MKLBLAS_THREADS (auto = ambient omp_get_max_threads(), i.e. the
// harness's OMP_NUM_THREADS under taskset; strict grammar per the AW audit
// fix) via MKL_Set_Num_Threads_Local — the same dlsym'd MklThreadScope
// pattern as scalapack/cpp/blacs_grid.h and mklfft/cpp, duplicated here so
// this TU stays MPI-free.  The AW cliff (cap MKL threads INSIDE ScaLAPACK
// handlers) does not apply: this is a rank-LOCAL BLAS call, the same class
// as the plan-A local eigh that NEEDS the full thread count.
//
// Envelope-honesty: every extent comes from the runtime buffer dimensions;
// nothing is specialized to a deck.  The batch is whatever shard the
// caller's shard_map placed on this rank — no N_mu^2 global tile is ever
// required (LORRAX scaling target).

#include <algorithm>
#include <atomic>
#include <cctype>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <omp.h>

#include <mkl_cblas.h>

#include "xla/ffi/api/ffi.h"

namespace lorrax_ffi::mklblas {

namespace ffi = ::xla::ffi;
using C128 = std::complex<double>;

// ---------------------------------------------------------------------------
//  MKL thread pinning (workstream-AW pattern; dlsym so no extra link dep and
//  a no-op on a non-MKL BLAS — same local copy as mklfft/cpp).
// ---------------------------------------------------------------------------
using mkl_set_local_fn = int (*)(int);

static mkl_set_local_fn mkl_set_num_threads_local_ptr() {
    static mkl_set_local_fn fn = reinterpret_cast<mkl_set_local_fn>(
        dlsym(RTLD_DEFAULT, "MKL_Set_Num_Threads_Local"));
    return fn;
}

class MklLocalPin {
  public:
    explicit MklLocalPin(int nthreads) {
        if (nthreads <= 0) return;
        auto set_local = mkl_set_num_threads_local_ptr();
        if (set_local == nullptr) return;
        prev_ = set_local(nthreads);
        active_ = true;
    }
    ~MklLocalPin() {
        if (active_) mkl_set_num_threads_local_ptr()(prev_);
    }
    MklLocalPin(const MklLocalPin&) = delete;
    MklLocalPin& operator=(const MklLocalPin&) = delete;

  private:
    bool active_ = false;
    int prev_ = 0;  // 0 = "follow the global setting" per MKL docs
};

static bool str_ieq(const char* a, const char* b) {
    for (;; ++a, ++b) {
        const int ca = std::tolower(static_cast<unsigned char>(*a));
        const int cb = std::tolower(static_cast<unsigned char>(*b));
        if (ca != cb) return false;
        if (ca == 0) return true;
    }
}

// MKL team size for the batch call.  Strict full-string grammar (AW audit
// lesson: a typo must not silently pick a known-bad policy).  "auto"
// (default) = ambient omp_get_max_threads() — the production harness
// exports OMP_NUM_THREADS=28 per rank under taskset.
static int team_threads() {
    const int maxt = std::max(1, omp_get_max_threads());
    const char* v = std::getenv("LORRAX_MKLBLAS_THREADS");
    if (!v || !*v || str_ieq(v, "auto")) return maxt;
    if (str_ieq(v, "off")) return 1;
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end != v && *end == '\0' && parsed >= 1 && parsed <= 4096) {
        return static_cast<int>(parsed);
    }
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
        std::fprintf(
            stderr,
            "*** LORRAX_MKLBLAS_THREADS='%s' is not a recognized value "
            "(accepted, case-insensitive: 'auto', 'off', or a positive "
            "integer <= 4096).  Falling back to 'auto' (%d threads). ***\n",
            v, maxt);
    }
    return maxt;
}

static bool log_enabled() {
    static const bool on = (std::getenv("LORRAX_MKLBLAS_LOG") != nullptr);
    return on;
}

static ffi::Error GemmBatchDispatch(
    ffi::AnyBuffer A, ffi::AnyBuffer B, ffi::Result<ffi::AnyBuffer> C)
{
    const auto dt = A.element_type();
    if (dt != ffi::DataType::F64 && dt != ffi::DataType::C128) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "mklblas.gemm_batch: dtype must be f64 or c128");
    }
    if (B.element_type() != dt || C->element_type() != dt) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "mklblas.gemm_batch: A/B/C dtypes must match (the primitive's "
            "de-promotion policy must split mixed real/complex upstream)");
    }
    auto ad = A.dimensions();
    auto bd = B.dimensions();
    auto cd = C->dimensions();
    if (ad.size() != 3 || bd.size() != 3 || cd.size() != 3) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "mklblas.gemm_batch: expected A (BA, M, K), "
                          "B (BB, K, N), C (BA, M, N)");
    }
    const int64_t ba = ad[0], m = ad[1], k = ad[2];
    const int64_t bb = bd[0], n = bd[2];
    if (bd[1] != k || cd[0] != ba || cd[1] != m || cd[2] != n ||
        bb < 1 || ba < 0 || (ba % bb) != 0) {
        std::ostringstream os;
        os << "mklblas.gemm_batch: shape mismatch — A(" << ba << "," << m
           << "," << k << ") B(" << bd[0] << "," << bd[1] << "," << bd[2]
           << ") C(" << cd[0] << "," << cd[1] << "," << cd[2]
           << "); need B.K == A.K, C == (BA, M, N), BA % BB == 0, BB >= 1";
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    if (ba == 0 || m == 0 || n == 0) return ffi::Error::Success();

    const int nthr = team_threads();
    if (log_enabled()) {
        static std::atomic<bool> once{false};
        if (!once.exchange(true)) {
            std::fprintf(stderr,
                         "[mklblas] gemm_batch first call: dtype=%s BA=%ld "
                         "BB=%ld M=%ld N=%ld K=%ld threads=%d\n",
                         dt == ffi::DataType::F64 ? "f64" : "c128",
                         (long)ba, (long)bb, (long)m, (long)n, (long)k, nthr);
        }
    }

    // One group; per-batch pointer arrays express the B-cycling broadcast
    // (a constant-stride API cannot: A's batch walks e·nk while B's walks
    // k only).  Pointer-array setup is O(BA) — noise next to the GEMMs.
    const CBLAS_TRANSPOSE trans = CblasNoTrans;
    const MKL_INT gm = (MKL_INT)m, gn = (MKL_INT)n, gk = (MKL_INT)k;
    const MKL_INT lda = gk, ldb = gn, ldc = gn;
    const MKL_INT group_count = 1;
    const MKL_INT group_size = (MKL_INT)ba;

    MklLocalPin pin(nthr);  // MKL threads the batch internally
    if (dt == ffi::DataType::F64) {
        const double* a = static_cast<const double*>(A.untyped_data());
        const double* b = static_cast<const double*>(B.untyped_data());
        double* c = static_cast<double*>(C->untyped_data());
        std::vector<const double*> ap(ba), bp(ba);
        std::vector<double*> cp(ba);
        for (int64_t i = 0; i < ba; ++i) {
            ap[i] = a + i * m * k;
            bp[i] = b + (i % bb) * k * n;
            cp[i] = c + i * m * n;
        }
        const double alpha = 1.0, beta = 0.0;
        cblas_dgemm_batch(CblasRowMajor, &trans, &trans, &gm, &gn, &gk,
                          &alpha, ap.data(), &lda, bp.data(), &ldb,
                          &beta, cp.data(), &ldc, group_count, &group_size);
    } else {
        const C128* a = static_cast<const C128*>(A.untyped_data());
        const C128* b = static_cast<const C128*>(B.untyped_data());
        C128* c = static_cast<C128*>(C->untyped_data());
        std::vector<const void*> ap(ba), bp(ba);
        std::vector<void*> cp(ba);
        for (int64_t i = 0; i < ba; ++i) {
            ap[i] = a + i * m * k;
            bp[i] = b + (i % bb) * k * n;
            cp[i] = c + i * m * n;
        }
        const C128 alpha(1.0, 0.0), beta(0.0, 0.0);
        cblas_zgemm_batch(CblasRowMajor, &trans, &trans, &gm, &gn, &gk,
                          &alpha, ap.data(), &lda, bp.data(), &ldb,
                          &beta, cp.data(), &ldc, group_count, &group_size);
    }
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::mklblas

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MklBlasGemmBatchHostFfi,
    lorrax_ffi::mklblas::GemmBatchDispatch,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()      // A (BA, M, K) f64|c128
        .Arg<xla::ffi::AnyBuffer>()      // B (BB, K, N) same dtype
        .Ret<xla::ffi::AnyBuffer>());    // C (BA, M, N) same dtype
