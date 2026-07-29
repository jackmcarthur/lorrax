// gemm_batch_ffi.cc — vendor-BLAS batched-GEMM host handler, HOST platform
// (JAX CPU backend).  The gated CPU GEMM body of the
// contract_bands_block_reshard primitive (src/common/contract_bands.py,
// LORRAX_BANDS_GEMM_FFI) — wk_REL/RESHARD_OVERHEAD_MEMO.md Sec. 4.4 exit
// (b) / Sec. 7 lever 1.
//
// VENDOR PORTABILITY (2026-07-29, owner order): cblas_?gemm_batch is an
// MKL EXTENSION of CBLAS (OpenBLAS ships it too; Cray LibSci does not).
// The choice is made AT RUNTIME BY dlsym — there is NO build-time feature
// probe and no HAVE_BATCH macro:
//   cblas_{d,z}gemm_batch resolve  -> ONE batched call per invocation;
//   either one missing              -> portable fallback: a loop of plain
//                                     cblas_{d,z}gemm calls (standard
//                                     CBLAS — MKL / LibSci / OpenBLAS /
//                                     BLIS all serve those), each GEMM
//                                     threaded internally by the vendor.
// ONE BINARY SERVES EITHER VENDOR.  Rationale (owner order 2026-07-29,
// after the probe cost a gate cycle): a link-based CMake
// check_symbol_exists silently downgraded THIS handler to the slow loop
// on an MKL that has the batched entry, twice, for two different
// link-closure reasons (jobs 7879278/7879281 — see
// wk_REL/gemm_portability_bse_notes.md).  A build-time probe that can
// answer "no" for environmental reasons is a footgun: the failure is
// invisible and costs 1.6-1.9x.  dlsym asks the question in the process
// that will actually make the call, using the SAME house idiom as the
// MKL thread pin below (blacs_grid.h's MklThreadScope) — zero link
// dependency, no CMake state set/unset around a probe, nothing to get
// out of order.  The prototypes are declared HERE as function-pointer
// typedefs, so the TU never needs a header that declares the batched
// entry (plain <cblas.h> on LibSci/OpenBLAS is enough).
//   LORRAX_MKLBLAS_MKL_HEADER  defined -> compile against <mkl_cblas.h>
//                              (MKL builds); undefined -> <cblas.h>.
//                              A plain EXISTS test in CMake, not a
//                              probe: its failure mode is "handler not
//                              built at all" (loud), never "built slow".
// WHICH ENTRY IS LIVE IS ANNOUNCED ON FIRST USE, UNCONDITIONALLY (see
// announce_entry_once below) — a silent downgrade is impossible by
// construction, which is the whole point of the redesign.
// Works in principle with Intel MKL or Cray LibSci; TESTED WITH INTEL
// ONLY so far (Frontera MKL 2020.1, batched entry — jobs 7879008/7879010).
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
//       dtype f64 (cblas_dgemm[_batch]) or c128 (cblas_zgemm[_batch]),
//       both operands the SAME dtype (the primitive's de-promotion policy
//       guarantees no mixed real/complex GEMM ever reaches this point).
//   The B-cycling broadcast rule serves both the plain per-k batch
//   (BA == BB == nk) and the extra-stacked batch (BA == E·nk vs the
//   k-only ψ operand, stack axis OUTERMOST in A).
//
// In-place: NONE — a (BA, M, N) GEMM output never legally aliases a
// (BA, M, K)/(BB, K, N) operand buffer, so no input_output_aliases are
// declared (contrast the in-place mklfft handlers, where shapes match).
//
// Threading: ONE cblas_*gemm_batch call per invocation (or the plain-GEMM
// loop, one internally-threaded GEMM per slot); MKL parallelizes
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
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <omp.h>

#if defined(LORRAX_MKLBLAS_MKL_HEADER)
#include <mkl_cblas.h>
#else
#include <cblas.h>              // standard CBLAS (LibSci / OpenBLAS / BLIS)
#endif

#include "xla/ffi/api/ffi.h"

namespace lorrax_ffi::mklblas {

namespace ffi = ::xla::ffi;
using C128 = std::complex<double>;

// LP64 CBLAS integer.  MKL_INT under <mkl_cblas.h>; plain int for the
// standard CBLAS headers (LibSci/OpenBLAS/BLIS lp64 — the only builds
// LORRAX links; no ILP64 variant is configured anywhere in this repo).
#if defined(LORRAX_MKLBLAS_MKL_HEADER)
using blas_int = MKL_INT;
#else
using blas_int = int;
#endif

// ---------------------------------------------------------------------------
//  Runtime symbol resolution (the house idiom: blacs_grid.h's MklThreadScope,
//  cufft/cpp's driver-API entries, mklfft/cpp's pin — all dlsym).
//
//  RTLD_DEFAULT searches the process's global symbol scope.  That is the
//  right handle here because ffi_loader.get_lib() dlopens this .so with
//  ctypes.CDLL(..., mode=RTLD_GLOBAL) (ffi_loader.py:514), which publishes
//  the library AND its DT_NEEDED closure — libmkl_intel_lp64 among them —
//  into that scope.  RTLD_NEXT is tried as a second chance for the case
//  where this object was loaded into a local scope instead; if BOTH miss,
//  we take the plain-GEMM loop AND SAY SO (announce_entry_once).  There is
//  deliberately no third mechanism: an unresolved symbol here is a correct,
//  announced capability answer, not an error to work around.
// ---------------------------------------------------------------------------
static void* resolve_sym(const char* name) {
    void* p = dlsym(RTLD_DEFAULT, name);
#ifdef RTLD_NEXT
    if (p == nullptr) p = dlsym(RTLD_NEXT, name);
#endif
    return p;
}

// Batched-GEMM entry points, declared HERE as function-pointer typedefs so
// no header needs to declare them (that is what frees the TU from
// mkl_cblas.h).  The CBLAS enum parameters are typed `int`: every CBLAS
// spells its enums differently (CBLAS_LAYOUT / CBLAS_ORDER) but they are
// plain C enums with small values, i.e. int-sized and int-passed on every
// ABI this project builds for.  The integer parameters use blas_int, which
// IS the vendor's index type by the same rule as the plain entries below.
using dgemm_batch_fn = void (*)(int, const int*, const int*,
                                const blas_int*, const blas_int*,
                                const blas_int*, const double*,
                                const double**, const blas_int*,
                                const double**, const blas_int*,
                                const double*, double**, const blas_int*,
                                blas_int, const blas_int*);
using zgemm_batch_fn = void (*)(int, const int*, const int*,
                                const blas_int*, const blas_int*,
                                const blas_int*, const void*,
                                const void**, const blas_int*,
                                const void**, const blas_int*,
                                const void*, void**, const blas_int*,
                                blas_int, const blas_int*);

struct BatchedEntries {
    dgemm_batch_fn d = nullptr;
    zgemm_batch_fn z = nullptr;
    // BOTH or NEITHER: a BLAS that served only one would make the handler's
    // behaviour dtype-dependent for no benefit, and no real BLAS does that.
    bool ok() const { return d != nullptr && z != nullptr; }
};

static const BatchedEntries& batched_entries() {
    static const BatchedEntries e = [] {
        BatchedEntries b;
        b.d = reinterpret_cast<dgemm_batch_fn>(
            resolve_sym("cblas_dgemm_batch"));
        b.z = reinterpret_cast<zgemm_batch_fn>(
            resolve_sym("cblas_zgemm_batch"));
        return b;
    }();
    return e;
}

// Announce on rank 0 (or when the launcher is unknown — tests, single
// process).  Reading the launcher's rank env is enough: this TU is
// comms-free by design and must not link MPI.
static bool announce_here() {
    for (const char* v : {"SLURM_PROCID", "PMI_RANK", "OMPI_COMM_WORLD_RANK"}) {
        const char* s = std::getenv(v);
        if (s != nullptr && *s != '\0') return std::strcmp(s, "0") == 0;
    }
    return true;
}

// UNCONDITIONAL (not behind LORRAX_MKLBLAS_LOG): which entry is live must
// always be visible in the log, so a silent downgrade cannot happen.  Once
// per process, at first use.
static void announce_entry_once() {
    static std::atomic<bool> once{false};
    if (once.exchange(true)) return;
    if (!announce_here()) return;
    const BatchedEntries& e = batched_entries();
    if (e.ok()) {
        std::fprintf(stderr,
                     "[mklblas] GEMM entry: cblas_?gemm_batch (batched) — "
                     "resolved by dlsym at first use.\n");
    } else {
        std::fprintf(stderr,
                     "[mklblas] GEMM entry: plain cblas_?gemm loop — this "
                     "BLAS does not export cblas_%sgemm_batch (dlsym), so "
                     "the portable per-slot loop is used.  Correct, ~1.6-1.9x "
                     "below the batched entry on MKL.\n",
                     e.d == nullptr ? "d" : "z");
    }
    std::fflush(stderr);
}

// ---------------------------------------------------------------------------
//  MKL thread pinning (workstream-AW pattern; dlsym so no extra link dep and
//  a no-op on a non-MKL BLAS — same local copy as mklfft/cpp).
// ---------------------------------------------------------------------------
using mkl_set_local_fn = int (*)(int);

static mkl_set_local_fn mkl_set_num_threads_local_ptr() {
    static mkl_set_local_fn fn = reinterpret_cast<mkl_set_local_fn>(
        resolve_sym("MKL_Set_Num_Threads_Local"));
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
    // Capability answer + announcement BEFORE any work, so the log records
    // which entry ran even if the GEMM below aborts.
    announce_entry_once();
    const BatchedEntries& batched = batched_entries();
    const bool use_batched = batched.ok();

    if (log_enabled()) {
        static std::atomic<bool> once{false};
        if (!once.exchange(true)) {
            const char* entry = use_batched
                ? "cblas_?gemm_batch (batched entry)"
                : "cblas_?gemm loop (no batched entry in this BLAS)";
            std::fprintf(stderr,
                         "[mklblas] gemm_batch first call: dtype=%s BA=%ld "
                         "BB=%ld M=%ld N=%ld K=%ld threads=%d via %s\n",
                         dt == ffi::DataType::F64 ? "f64" : "c128",
                         (long)ba, (long)bb, (long)m, (long)n, (long)k,
                         nthr, entry);
        }
    }

    const blas_int gm = (blas_int)m, gn = (blas_int)n, gk = (blas_int)k;
    const blas_int lda = gk, ldb = gn, ldc = gn;

    MklLocalPin pin(nthr);  // the vendor BLAS threads each call internally
                            // (dlsym'd MKL pin; a no-op on non-MKL BLAS,
                            // where OMP_NUM_THREADS et al. govern)
    if (use_batched) {
    // One group; per-batch pointer arrays express the B-cycling broadcast
    // (a constant-stride API cannot: A's batch walks e·nk while B's walks
    // k only).  Pointer-array setup is O(BA) — noise next to the GEMMs.
    const int layout = (int)CblasRowMajor;
    const int trans = (int)CblasNoTrans;
    const blas_int group_count = 1;
    const blas_int group_size = (blas_int)ba;
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
        batched.d(layout, &trans, &trans, &gm, &gn, &gk,
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
        batched.z(layout, &trans, &trans, &gm, &gn, &gk,
                  &alpha, ap.data(), &lda, bp.data(), &ldb,
                  &beta, cp.data(), &ldc, group_count, &group_size);
    }
    } else {
    // Portable fallback: plain standard-CBLAS GEMMs, one per batch slot,
    // same B-cycling broadcast rule.  Each GEMM is threaded internally by
    // the vendor BLAS; the loop itself is sequential BY DESIGN (the
    // batches are large-M×K tiles — an outer OpenMP loop would fight the
    // BLAS's own team for the same cores).
    if (dt == ffi::DataType::F64) {
        const double* a = static_cast<const double*>(A.untyped_data());
        const double* b = static_cast<const double*>(B.untyped_data());
        double* c = static_cast<double*>(C->untyped_data());
        for (int64_t i = 0; i < ba; ++i) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        gm, gn, gk, 1.0, a + i * m * k, lda,
                        b + (i % bb) * k * n, ldb,
                        0.0, c + i * m * n, ldc);
        }
    } else {
        const C128* a = static_cast<const C128*>(A.untyped_data());
        const C128* b = static_cast<const C128*>(B.untyped_data());
        C128* c = static_cast<C128*>(C->untyped_data());
        const C128 alpha(1.0, 0.0), beta(0.0, 0.0);
        for (int64_t i = 0; i < ba; ++i) {
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        gm, gn, gk, &alpha, a + i * m * k, lda,
                        b + (i % bb) * k * n, ldb,
                        &beta, c + i * m * n, ldc);
        }
    }
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
