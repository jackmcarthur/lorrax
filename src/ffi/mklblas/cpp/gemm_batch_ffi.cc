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
//   cblas_{s,d,c,z}gemm_batch resolve -> ONE batched call per invocation;
//   missing for THAT precision        -> portable fallback: a loop of plain
//                                     cblas_{s,d,c,z}gemm calls (standard
//                                     CBLAS — MKL / LibSci / OpenBLAS /
//                                     BLIS all serve those), each GEMM
//                                     threaded internally by the vendor.
// Availability is tracked PER PRECISION, not all-or-nothing, so a BLAS
// with only the double batched entries keeps them where it has them.
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
//       dtype f64/f32/c128/c64 (cblas_{d,s,z,c}gemm[_batch]) — ALL FOUR
//       precisions since the 2026-07-29 owner order, so the BSE fp32-GMRES
//       (complex64) path rides the handler instead of being excluded;
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
#include <limits>
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

#include "../../common/cpp/mkl_thread_pin.h"
#include "xla/ffi/api/ffi.h"

namespace lorrax_ffi::mklblas {

namespace ffi = ::xla::ffi;
using C128 = std::complex<double>;
using C64  = std::complex<float>;

// LP64 CBLAS integer.  MKL_INT under <mkl_cblas.h>; plain int for the
// standard CBLAS headers (LibSci/OpenBLAS/BLIS lp64 — the only builds
// LORRAX links; no ILP64 variant is configured anywhere in this repo).
#if defined(LORRAX_MKLBLAS_MKL_HEADER)
using blas_int = MKL_INT;
#else
using blas_int = int;
#endif

// ---------------------------------------------------------------------------
//  Runtime symbol resolution (the house idiom: cufft/cpp's driver-API
//  entries, the MKL pins — all dlsym).  The resolver itself now lives in
//  common/cpp/mkl_thread_pin.h, which is MPI-free and CUDA-free so this
//  comms-free TU can use it; it was THIS file's two-stage
//  RTLD_DEFAULT->RTLD_NEXT version that the 2026-07-30 audit promoted to be
//  the shared one, because it is the strictly more capable of the two forms
//  that had diverged.  If both stages miss we take the plain-GEMM loop AND
//  SAY SO (announce_entry_once).
//
//  CORRECTION (2026-07-30): the comment that used to live here said
//  "RTLD_DEFAULT searches the process's global symbol scope.  That is the
//  right handle here because ffi_loader.get_lib() dlopens this .so with
//  ctypes.CDLL(..., mode=RTLD_GLOBAL)".  The first sentence is the man
//  page's wording for a caller in the main executable and is not what glibc
//  does for a caller inside a dlopen'd library — it searches the CALLER's
//  link-map scope, which already includes this .so's DT_NEEDED closure.  The
//  RTLD_GLOBAL load mode is therefore NOT what makes the pin resolve here,
//  and the difference between this file's resolver and the bare-RTLD_DEFAULT
//  copies in mklfft/scalapack was never behavioural.  Measured table is in
//  the mkl_thread_pin.h header; the probe is
//  tests/test_ffi_thread_pin_resolution.py.
// ---------------------------------------------------------------------------
using mklpin::resolve_sym;

// Batched-GEMM entry points, declared HERE as function-pointer typedefs so
// no header needs to declare them (that is what frees the TU from
// mkl_cblas.h).  The CBLAS enum parameters are typed `int`: every CBLAS
// spells its enums differently (CBLAS_LAYOUT / CBLAS_ORDER) but they are
// plain C enums with small values, i.e. int-sized and int-passed on every
// ABI this project builds for.  The integer parameters use blas_int, which
// IS the vendor's index type by the same rule as the plain entries below.
// Real precisions take alpha/beta BY VALUE ARRAYS of the scalar type;
// complex precisions take them as void* (the CBLAS convention).
template <typename S>
using real_gemm_batch_fn = void (*)(int, const int*, const int*,
                                    const blas_int*, const blas_int*,
                                    const blas_int*, const S*,
                                    const S**, const blas_int*,
                                    const S**, const blas_int*,
                                    const S*, S**, const blas_int*,
                                    blas_int, const blas_int*);
using cplx_gemm_batch_fn = void (*)(int, const int*, const int*,
                                    const blas_int*, const blas_int*,
                                    const blas_int*, const void*,
                                    const void**, const blas_int*,
                                    const void**, const blas_int*,
                                    const void*, void**, const blas_int*,
                                    blas_int, const blas_int*);
using dgemm_batch_fn = real_gemm_batch_fn<double>;
using sgemm_batch_fn = real_gemm_batch_fn<float>;
using zgemm_batch_fn = cplx_gemm_batch_fn;
using cgemm_batch_fn = cplx_gemm_batch_fn;

// PER-PRECISION availability (2026-07-29, owner order adding single
// precision).  Deliberately NOT all-or-nothing: a BLAS that ships the
// double batched entries but not the single ones then still gets the
// batched path where it has it and the correct plain loop where it does
// not, instead of losing the batched entry everywhere.
struct BatchedEntries {
    dgemm_batch_fn d = nullptr;
    sgemm_batch_fn s = nullptr;
    zgemm_batch_fn z = nullptr;
    cgemm_batch_fn c = nullptr;
};

static const BatchedEntries& batched_entries() {
    static const BatchedEntries e = [] {
        BatchedEntries b;
        b.d = reinterpret_cast<dgemm_batch_fn>(
            resolve_sym("cblas_dgemm_batch"));
        b.s = reinterpret_cast<sgemm_batch_fn>(
            resolve_sym("cblas_sgemm_batch"));
        b.z = reinterpret_cast<zgemm_batch_fn>(
            resolve_sym("cblas_zgemm_batch"));
        b.c = reinterpret_cast<cgemm_batch_fn>(
            resolve_sym("cblas_cgemm_batch"));
        return b;
    }();
    return e;
}

// Human name + whether the batched entry for THIS dtype resolved.
static const char* dtype_name(ffi::DataType dt) {
    switch (dt) {
        case ffi::DataType::F64:  return "f64";
        case ffi::DataType::F32:  return "f32";
        case ffi::DataType::C128: return "c128";
        case ffi::DataType::C64:  return "c64";
        default:                  return "?";
    }
}

static bool batched_for(ffi::DataType dt) {
    const BatchedEntries& e = batched_entries();
    switch (dt) {
        case ffi::DataType::F64:  return e.d != nullptr;
        case ffi::DataType::F32:  return e.s != nullptr;
        case ffi::DataType::C128: return e.z != nullptr;
        case ffi::DataType::C64:  return e.c != nullptr;
        default:                  return false;
    }
}

// Announce on rank 0 (or when the launcher is unknown — tests, single
// process).  Reading the launcher's rank env is enough: this TU is
// comms-free by design and must not link MPI.  Moved to
// common/cpp/mkl_thread_pin.h in the 2026-07-30 audit so mklfft and cufft —
// which had NO rank guard at all — can share the one implementation.
using mklpin::announce_here;

// UNCONDITIONAL (not behind LORRAX_MKLBLAS_LOG): which entry is live must
// always be visible in the log, so a silent downgrade cannot happen.  Once
// per PRECISION at that precision's first use — bounded at four lines per
// process, and it names the precision because availability is per-dtype.
static void announce_entry_once(ffi::DataType dt) {
    static std::atomic<bool> seen[4] = {};
    int slot;
    switch (dt) {
        case ffi::DataType::F64:  slot = 0; break;
        case ffi::DataType::F32:  slot = 1; break;
        case ffi::DataType::C128: slot = 2; break;
        case ffi::DataType::C64:  slot = 3; break;
        default: return;
    }
    if (seen[slot].exchange(true)) return;
    if (!announce_here()) return;
    const char* nm = dtype_name(dt);
    const char letter = (dt == ffi::DataType::F64)  ? 'd'
                      : (dt == ffi::DataType::F32)  ? 's'
                      : (dt == ffi::DataType::C128) ? 'z' : 'c';
    if (batched_for(dt)) {
        std::fprintf(stderr,
                     "[mklblas] GEMM entry (%s): cblas_%cgemm_batch "
                     "(batched) — this BLAS provides the batched entry.\n",
                     nm, letter);
    } else {
        std::fprintf(stderr,
                     "[mklblas] GEMM entry (%s): plain cblas_%cgemm loop — "
                     "this BLAS does not provide cblas_%cgemm_batch, so the "
                     "portable per-slot loop is used.  Correct, "
                     "~1.6-1.9x below the batched entry on MKL.\n",
                     nm, letter, letter);
    }
    std::fflush(stderr);
}

// ---------------------------------------------------------------------------
//  MKL thread pinning (workstream-AW pattern; dlsym so no extra link dep and
//  a no-op on a non-MKL BLAS).  Mechanism shared via
//  common/cpp/mkl_thread_pin.h; `MklLocalPin` is the old local name for what
//  is now mklpin::MklThreadScope, kept so the pin site below reads unchanged.
// ---------------------------------------------------------------------------
using MklLocalPin = mklpin::MklThreadScope;
using mklpin::str_ieq;

// MKL team size for the batch call.  Strict full-string grammar via the
// shared parser (mklpin::parse_thread_knob; the parse used to exist three
// times — here, fft_flat_k_ffi.cc, blacs_grid.h — and had drifted; only the
// PARSE is shared, the "auto" POLICY stays this family's: ambient
// omp_get_max_threads() — the production harness exports OMP_NUM_THREADS=28
// per rank under taskset.  AW audit lesson: a typo must not silently pick a
// known-bad policy.
static int team_threads() {
    const int maxt = std::max(1, omp_get_max_threads());
    const char* v = std::getenv("LORRAX_MKLBLAS_THREADS");
    const mklpin::ThreadKnob k = mklpin::parse_thread_knob(v, 1, 4096);
    switch (k.kind) {
        case mklpin::ThreadKnob::kOff: return 1;
        case mklpin::ThreadKnob::kInt: return k.value;
        case mklpin::ThreadKnob::kBad: {
            static std::atomic<bool> warned{false};
            // Rank-scoped (announce_here): pre-audit this printed on every
            // rank — P lines for one misspelling (P1.13).
            if (!warned.exchange(true) && announce_here()) {
                std::fprintf(
                    stderr,
                    "*** LORRAX_MKLBLAS_THREADS='%s' is not a recognized "
                    "value (accepted, case-insensitive: 'auto', 'off', or a "
                    "positive integer <= 4096).  Falling back to 'auto' "
                    "(%d threads). ***\n",
                    v, maxt);
            }
            return maxt;
        }
        case mklpin::ThreadKnob::kAuto:
        default:
            return maxt;
    }
}

static bool log_enabled() {
    static const bool on = (std::getenv("LORRAX_MKLBLAS_LOG") != nullptr);
    return on;
}

static ffi::Error GemmBatchDispatch(
    ffi::AnyBuffer A, ffi::AnyBuffer B, ffi::Result<ffi::AnyBuffer> C)
{
    const auto dt = A.element_type();
    if (dt != ffi::DataType::F64 && dt != ffi::DataType::C128
        && dt != ffi::DataType::F32 && dt != ffi::DataType::C64) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "mklblas.gemm_batch: dtype must be f64, f32, "
                          "c128 or c64");
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

    // blas_int narrowing refusal (P1.10): under LP64 (the only configured
    // ABI here) blas_int is 32-bit, and the casts below would silently
    // truncate a >2^31-1 extent into a wrong-answer GEMM.  Refuse, naming
    // the offending extent.  (i*m*k pointer arithmetic stays int64 and is
    // unaffected; only the extents handed to CBLAS must fit.)
    {
        constexpr int64_t blas_max =
            static_cast<int64_t>(std::numeric_limits<blas_int>::max());
        const struct { const char* name; int64_t v; } extents[] = {
            {"BA", ba}, {"M", m}, {"N", n}, {"K", k}};
        for (const auto& e : extents) {
            if (e.v > blas_max) {
                std::ostringstream os;
                os << "mklblas.gemm_batch: extent " << e.name << "=" << e.v
                   << " exceeds this build's BLAS integer (LP64 blas_int max "
                   << blas_max << ") — the CBLAS call would silently "
                   << "truncate it.  Shard the batch/band window smaller, or "
                   << "unset LORRAX_BANDS_GEMM_FFI for this call path (the "
                   << "XLA lowering has no 32-bit extent limit).";
                return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
            }
        }
    }

    const int nthr = team_threads();
    // Capability answer + announcement BEFORE any work, so the log records
    // which entry ran even if the GEMM below aborts.
    announce_entry_once(dt);
    const BatchedEntries& batched = batched_entries();
    const bool use_batched = batched_for(dt);

    if (log_enabled()) {
        static std::atomic<bool> once{false};
        if (!once.exchange(true)) {
            const char* entry = use_batched
                ? "cblas_?gemm_batch (batched entry)"
                : "cblas_?gemm loop (no batched entry in this BLAS)";
            std::fprintf(stderr,
                         "[mklblas] gemm_batch first call: dtype=%s BA=%ld "
                         "BB=%ld M=%ld N=%ld K=%ld threads=%d via %s\n",
                         dtype_name(dt),
                         (long)ba, (long)bb, (long)m, (long)n, (long)k,
                         nthr, entry);
        }
    }

    const blas_int gm = (blas_int)m, gn = (blas_int)n, gk = (blas_int)k;
    const blas_int lda = gk, ldb = gn, ldc = gn;

    MklLocalPin pin(nthr);  // the vendor BLAS threads each call internally
                            // (dlsym'd MKL pin; a no-op on non-MKL BLAS,
                            // where OMP_NUM_THREADS et al. govern)
    // One group; per-batch pointer arrays express the B-cycling broadcast
    // (a constant-stride API cannot: A's batch walks e·nk while B's walks
    // k only).  Pointer-array setup is O(BA) — noise next to the GEMMs.
    // The plain fallback is a loop of standard-CBLAS GEMMs with the SAME
    // broadcast rule; each GEMM is threaded internally by the vendor BLAS
    // and the loop is sequential BY DESIGN (the batches are large M×K
    // tiles — an outer OpenMP loop would fight the BLAS's own team).
    // Both arms are identical across the four precisions (owner order
    // 2026-07-29 added f32/c64); only the scalar type and the entry point
    // change, so the broadcast rule and MklLocalPin cannot drift apart.
    const int layout = (int)CblasRowMajor;
    const int trans = (int)CblasNoTrans;
    const blas_int group_count = 1;
    const blas_int group_size = (blas_int)ba;

    // The plain entries are called DIRECTLY (every CBLAS has them, so a
    // link reference is safe) and wrapped in lambdas because the CBLAS
    // enums cannot be passed as int in C++ — the enum constants must stay
    // inside the call.
    auto plain_d = [](blas_int M, blas_int N, blas_int K, const double* Ax,
                      blas_int LDA, const double* Bx, blas_int LDB,
                      double* Cx, blas_int LDC) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    1.0, Ax, LDA, Bx, LDB, 0.0, Cx, LDC);
    };
    auto plain_s = [](blas_int M, blas_int N, blas_int K, const float* Ax,
                      blas_int LDA, const float* Bx, blas_int LDB,
                      float* Cx, blas_int LDC) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    1.0f, Ax, LDA, Bx, LDB, 0.0f, Cx, LDC);
    };
    auto plain_z = [](blas_int M, blas_int N, blas_int K, const void* al,
                      const void* Ax, blas_int LDA, const void* Bx,
                      blas_int LDB, const void* be, void* Cx, blas_int LDC) {
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    al, Ax, LDA, Bx, LDB, be, Cx, LDC);
    };
    auto plain_c = [](blas_int M, blas_int N, blas_int K, const void* al,
                      const void* Ax, blas_int LDA, const void* Bx,
                      blas_int LDB, const void* be, void* Cx, blas_int LDC) {
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    al, Ax, LDA, Bx, LDB, be, Cx, LDC);
    };

    auto run_real = [&](auto tag, auto batch_fn, auto plain) {
        using S = decltype(tag);
        const S* a = static_cast<const S*>(A.untyped_data());
        const S* b = static_cast<const S*>(B.untyped_data());
        S* c = static_cast<S*>(C->untyped_data());
        if (use_batched) {
            std::vector<const S*> ap(ba), bp(ba);
            std::vector<S*> cp(ba);
            for (int64_t i = 0; i < ba; ++i) {
                ap[i] = a + i * m * k;
                bp[i] = b + (i % bb) * k * n;
                cp[i] = c + i * m * n;
            }
            const S alpha = S(1), beta = S(0);
            batch_fn(layout, &trans, &trans, &gm, &gn, &gk,
                     &alpha, ap.data(), &lda, bp.data(), &ldb,
                     &beta, cp.data(), &ldc, group_count, &group_size);
        } else {
            for (int64_t i = 0; i < ba; ++i) {
                plain(gm, gn, gk, a + i * m * k, lda,
                      b + (i % bb) * k * n, ldb, c + i * m * n, ldc);
            }
        }
    };

    auto run_cplx = [&](auto tag, auto batch_fn, auto plain) {
        using CT = decltype(tag);
        const CT* a = static_cast<const CT*>(A.untyped_data());
        const CT* b = static_cast<const CT*>(B.untyped_data());
        CT* c = static_cast<CT*>(C->untyped_data());
        const CT alpha(1, 0), beta(0, 0);
        if (use_batched) {
            std::vector<const void*> ap(ba), bp(ba);
            std::vector<void*> cp(ba);
            for (int64_t i = 0; i < ba; ++i) {
                ap[i] = a + i * m * k;
                bp[i] = b + (i % bb) * k * n;
                cp[i] = c + i * m * n;
            }
            batch_fn(layout, &trans, &trans, &gm, &gn, &gk,
                     &alpha, ap.data(), &lda, bp.data(), &ldb,
                     &beta, cp.data(), &ldc, group_count, &group_size);
        } else {
            for (int64_t i = 0; i < ba; ++i) {
                plain(gm, gn, gk, &alpha, a + i * m * k, lda,
                      b + (i % bb) * k * n, ldb, &beta,
                      c + i * m * n, ldc);
            }
        }
    };

    switch (dt) {
        case ffi::DataType::F64:
            run_real(double{}, batched.d, plain_d); break;
        case ffi::DataType::F32:
            run_real(float{}, batched.s, plain_s); break;
        case ffi::DataType::C128:
            run_cplx(C128{}, batched.z, plain_z); break;
        default:
            run_cplx(C64{}, batched.c, plain_c); break;
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
