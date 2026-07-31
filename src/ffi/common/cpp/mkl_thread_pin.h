// mkl_thread_pin.h — the ONE runtime-symbol resolver + MKL thread-pin RAII
// shared by every LORRAX FFI family that talks to a vendor BLAS/FFT.
//
// WHY THIS FILE EXISTS (extraction, 2026-07-30 FFI divergence audit).
// The same ~60-line block had been copied THREE times:
//     scalapack/cpp/blacs_grid.h : ~152-238   (inline, header)
//     mklblas/cpp/gemm_batch_ffi.cc : ~277-303
//     mklfft/cpp/fft_flat_k_ffi.cc  : ~96-122
// and the copies had DRIFTED: mklblas resolved through `resolve_sym`
// (RTLD_DEFAULT then RTLD_NEXT), the other two through a bare
// `dlsym(RTLD_DEFAULT, ...)`.  Measured evidence that the three copies are
// real and independent (they are function-local statics, so each holds its
// OWN cached resolution — `strings` cannot see them, `nm -C` can):
//
//   nm -C lorrax_ffi_unified/build_host_C64/liblorrax_ffi_host.so
//     b lorrax_ffi::mklfft::mkl_set_num_threads_local_ptr()::fn
//     b lorrax_ffi::mklblas::mkl_set_num_threads_local_ptr()::fn
//     u lorrax_ffi::scalapack::mkl_set_num_threads_local_ptr()::fn
//
// WHAT THE DRIFT DOES AND DOES NOT COST — measure before believing the
// scary version.  The obvious reading is "under a local-scope dlopen the
// GEMM handler still pins MKL and the FFT handler silently does not".  That
// reading is WRONG on glibc, and it comes from the man page's description
// of RTLD_DEFAULT ("the process's global symbol scope"), which is the
// behaviour for a caller in the MAIN EXECUTABLE.  glibc's `_dl_sym`
// resolves RTLD_DEFAULT against the CALLER's link-map scope (`l_scope`),
// and for an object that was itself dlopen'd — RTLD_LOCAL or not — that
// scope includes the object's own DT_NEEDED closure.  libmkl_intel_lp64.so
// is in this library's DT_NEEDED, so RTLD_DEFAULT finds MKL either way.
//
// Measured on this system (gcc 8.3.0 / glibc; the probe is
// tests/test_ffi_thread_pin_resolution.py, which builds a fake vendor lib
// and a probe .so and runs both resolvers side by side).  All six
// configurations give the bare form and the two-stage form the SAME answer:
//
//   symbol in probe's DT_NEEDED, probe RTLD_LOCAL    bare=1 shared=1
//   symbol in probe's DT_NEEDED, probe RTLD_GLOBAL   bare=1 shared=1
//   vendor dlopen'd RTLD_LOCAL,  probe RTLD_LOCAL    bare=0 shared=0
//   vendor dlopen'd RTLD_LOCAL,  probe RTLD_GLOBAL   bare=0 shared=0
//   vendor dlopen'd RTLD_GLOBAL, probe RTLD_LOCAL    bare=1 shared=1
//   vendor dlopen'd RTLD_GLOBAL, probe RTLD_GLOBAL   bare=1 shared=1
//
// So this extraction fixes a MAINTENANCE hazard — one mechanism, three
// copies, already visibly drifted, each with its own cached resolution —
// and NOT a live correctness bug.  Do not record it as the latter.  The
// two-stage form is kept as the shared one because it is the strictly more
// capable of the two (RTLD_NEXT still matters if the symbol is ever defined
// by this library itself and has to be chained past) and because having ONE
// resolver is the entire point.
//
// `docs/dev/vendor_gemm_service.md:174-186` proposed exactly this file and
// recorded it as "Not done in this wave"; TEMPLATE.md:194-195's own
// threshold ("extract when a THIRD library would copy it a third time") was
// already met.  This is that extraction.
//
// HARD CONSTRAINT — this header must include NO <mpi.h> and NO CUDA header.
// That is the entire reason the logic could not simply live in
// scalapack/cpp/blacs_grid.h, which includes <mpi.h> at line 40: mklblas
// and mklfft are comms-free TUs by design and must not acquire an MPI link
// dependency to pin a thread.  `common/cpp/scalapack_descriptor.h` is the
// precedent for a dependency-free shared header in this directory; the
// include is a plain relative path ("../../common/cpp/mkl_thread_pin.h"),
// so no build-system change is required to consume it.
//
// WHAT IS DELIBERATELY *NOT* HERE.  Each family's thread-count POLICY stays
// in that family, because the policies are genuinely different and the
// measurements behind them are not transferable:
//   scalapack  LORRAX_SCALAPACK_MKL_THREADS  cap at 4  (pzheevd starves MPI
//                                             progress: 24x at the 12x12 grid)
//   mklblas    LORRAX_MKLBLAS_THREADS        ambient omp_get_max_threads()
//   mklfft     LORRAX_MKLFFT_THREADS         ambient, then pin(1) per member
//                                            because the CHUNK LOOP is the
//                                            parallel dimension
// Unifying those would fuse three different answers to three different
// questions.  Only the mechanism is shared.

#pragma once

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <initializer_list>   // announce_here's range-for over a braced list;
                              // gemm_batch_ffi.cc only got this transitively

#include <dlfcn.h>

namespace lorrax_ffi::mklpin {

// ---------------------------------------------------------------------------
//  THE resolver.  RTLD_DEFAULT searches the CALLER's link-map scope — not,
//  as the man page's summary suggests, only the process-wide global scope;
//  that wording describes a caller in the main executable.  For this library
//  the caller's scope already contains its DT_NEEDED closure
//  (libmkl_intel_lp64 among them), which is why the vendor symbol resolves
//  whether ffi_loader dlopens us RTLD_GLOBAL (it does — ffi_loader.py:514)
//  or RTLD_LOCAL.  See the measured six-configuration table in the file
//  header before changing this.
//
//  RTLD_NEXT is kept as a second stage for the one case RTLD_DEFAULT cannot
//  answer: a symbol this library itself defines, which must be chained past
//  to reach the vendor's.  It costs one extra dlsym only on the miss path,
//  which runs once per process.  If both miss, the caller gets nullptr and
//  every consumer degrades to a documented no-op (a non-MKL BLAS is a
//  supported build).
//
//  There is deliberately no third mechanism: an unresolved symbol here is a
//  correct, announced capability answer, not an error to work around.
// ---------------------------------------------------------------------------
inline void* resolve_sym(const char* name) {
    void* p = dlsym(RTLD_DEFAULT, name);
#ifdef RTLD_NEXT
    if (p == nullptr) p = dlsym(RTLD_NEXT, name);
#endif
    return p;
}

using mkl_set_local_fn = int (*)(int);
using mkl_get_max_fn   = int (*)();

// `inline` (not `static`) is load-bearing: it gives `fn` vague linkage, so
// all TUs share ONE cached resolution instead of the three independent
// function-local statics `nm -C` found in the pre-extraction build.
inline mkl_set_local_fn mkl_set_num_threads_local_ptr() {
    static mkl_set_local_fn fn = reinterpret_cast<mkl_set_local_fn>(
        resolve_sym("MKL_Set_Num_Threads_Local"));
    return fn;
}

inline mkl_get_max_fn mkl_get_max_threads_ptr() {
    static mkl_get_max_fn fn = reinterpret_cast<mkl_get_max_fn>(
        resolve_sym("MKL_Get_Max_Threads"));
    return fn;
}

// Case-insensitive full-string equality (ASCII; env-value vocabulary only).
// Was a third copy in its own right (blacs_grid.h:168, gemm_batch:305,
// fft_flat_k:124) — same function, same three files.
inline bool str_ieq(const char* a, const char* b) {
    for (;; ++a, ++b) {
        const int ca = std::tolower(static_cast<unsigned char>(*a));
        const int cb = std::tolower(static_cast<unsigned char>(*b));
        if (ca != cb) return false;
        if (ca == 0) return true;
    }
}

// RAII: pin the CALLING thread's MKL team for the duration of a scope and
// restore the previous thread-local setting on exit.  No-op when the pin is
// disabled (nthreads <= 0) or MKL is absent.  The global MKL_NUM_THREADS is
// never touched.
class MklThreadScope {
  public:
    explicit MklThreadScope(int nthreads) {
        if (nthreads <= 0) return;
        auto set_local = mkl_set_num_threads_local_ptr();
        if (set_local == nullptr) return;
        prev_ = set_local(nthreads);
        active_ = true;
    }
    ~MklThreadScope() {
        if (active_) mkl_set_num_threads_local_ptr()(prev_);
    }
    MklThreadScope(const MklThreadScope&) = delete;
    MklThreadScope& operator=(const MklThreadScope&) = delete;

  private:
    bool active_ = false;
    int  prev_ = 0;   // 0 = "follow the global setting" per MKL docs
};

// ---------------------------------------------------------------------------
//  Rank-0 detection for stderr output, read from the LAUNCHER's environment.
//  Deliberately env-based rather than MPI_Comm_rank: the consumers (mklblas,
//  mklfft, cufft) are comms-free TUs that must not link MPI — the same
//  constraint that keeps this header MPI-free.  Returns true when the
//  launcher is unknown (unit tests, single process), so a serial run still
//  prints.
//
//  Previously the only copy was `announce_here()` at gemm_batch_ffi.cc:229;
//  mklfft and cufft had no rank guard at all, so at P=1000+ every opt-in log
//  line was multiplied by the process count (and, at fft_flat_k_ffi.cc:255,
//  by the OpenMP team size on top of that — see that call site).
// ---------------------------------------------------------------------------
inline bool announce_here() {
    for (const char* v : {"SLURM_PROCID", "PMI_RANK", "OMPI_COMM_WORLD_RANK"}) {
        const char* s = std::getenv(v);
        if (s != nullptr && *s != '\0') return std::strcmp(s, "0") == 0;
    }
    return true;
}

// Grammar for the opt-in C++ debug-log knobs (LORRAX_MKLFFT_LOG,
// LORRAX_CUFFT_LOG).  Presence-tested, as before, so every value that
// enabled logging before still does — with ONE addition: a value of "all"
// or "*" asks for every rank.  Anything else logs on rank 0 only.
//
// Why the default flipped to rank-0-only rather than staying all-ranks: the
// knob's job is "show me what the handler decided", which is a per-process
// property that is identical on every rank in every configuration this code
// supports (plan keys, arena sizes and team sizes are functions of the
// SHARDED tile shape).  At P=1000 the all-ranks default turned a 5-line
// answer into 5000 interleaved lines.  "all" is kept because a genuinely
// per-rank question (a rank whose shard is ragged) does exist, and answering
// it must remain possible.
inline bool log_here(const char* env_name) {
    const char* v = std::getenv(env_name);
    if (v == nullptr) return false;
    if (str_ieq(v, "all") || std::strcmp(v, "*") == 0) return true;
    return announce_here();
}

}  // namespace lorrax_ffi::mklpin
