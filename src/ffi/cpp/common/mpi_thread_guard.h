// mpi_thread_guard.h — the ONE thread-level-insufficient warning shared by
// every LORRAX FFI family whose handlers drive MPI from more than one thread
// (slate: internal OpenMP+MPI task engine; phdf5: dedicated writer thread
// running collective MPI-IO).
//
// EXTRACTION (2026-08-01 seam-audit leftover).  The body existed twice —
// cpp/slate/context.cc and cpp/phdf5/context.cc — differing only in the tag
// and one hazard sentence, and slate's copy carried a note deferring
// extraction until "a THIRD library would copy it a third time"
// (TEMPLATE.md:194-195).  The dedup was pulled forward: the two copies had
// already begun to drift textually, and the mechanism ("query on BOTH paths,
// after we initialise AND after an early return because someone else already
// did") is precisely the kind of load-bearing subtlety a divergent copy
// loses first.  Each family keeps its own hazard sentence and its own
// once-flag (the `knob_value` alias_warned idiom in mkl_thread_pin.h): a
// process using both libraries should hear about EACH hazard once.
//
// This header includes <mpi.h> ON PURPOSE and must therefore never be
// included from the comms-free TUs (mklblas, mklfft, cufft) — that
// constraint is why the logic cannot live in mkl_thread_pin.h (see its
// HARD CONSTRAINT note).
//
// WHY THE WARNING EXISTS (measured; scorecard AS.4b): with MPI initialized
// before these libraries at a thread level below MPI_THREAD_MULTIPLE (jax
// cpu_collectives_implementation=mpi with an unpatched MPIwrapper), two
// threads of one rank end up concurrently inside MPID_Progress_wait — a
// ~29% provider-independent multi-node segfault/hang rate at the
// zeta-write/V_q boundary, minutes after the cause, with a backtrace that
// names neither cause nor fix.  This guard makes the hazardous
// configuration announce itself up front.
//
// PLACEMENT CONTRACT for callers: invoke from ensure_mpi_initialized() so
// the query runs on BOTH paths — after we call MPI_Init_thread ourselves
// AND after an early return because someone else already initialized.  The
// already-initialized path is the hazardous one; a guard that only ran on
// self-init would be void by construction.

#pragma once

#include <atomic>
#include <cstdio>

#include <mpi.h>

namespace lorrax_ffi::mpiguard {

// One rank-0 stderr warning per (family) process lifetime when the granted
// MPI thread level is below MPI_THREAD_MULTIPLE.  `tag` names the family
// ("slate", "phdf5"); `hazard` is the family-specific sentence naming WHAT
// is undefined at the granted level; `warned` lives at the call site so the
// once-flag is per family, not per header.
inline void warn_if_thread_level_insufficient(const char* tag,
                                              const char* hazard,
                                              std::atomic<bool>& warned) {
    int provided = 0;
    MPI_Query_thread(&provided);
    if (provided >= MPI_THREAD_MULTIPLE) return;
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return;
    if (warned.exchange(true)) return;
    std::fprintf(stderr,
        "[%s] WARNING: MPI granted thread level %d < MPI_THREAD_MULTIPLE "
        "(%d) and MPI was initialized before this library (jax "
        "cpu_collectives_implementation=mpi with an unpatched MPIwrapper?).  "
        "%s is UNDEFINED at this level — the concurrent-MPI-progress hazard "
        "measured at a ~29%% multi-node crash rate (scorecard AS.4b).  Fix: "
        "point MPITRAMPOLINE_LIB at the MPIwrapper built by "
        "config/frontera/build_mpiwrapper.sh, which upgrades every init to "
        "MPI_THREAD_MULTIPLE (docs/dev/mpi_collectives.md).\n",
        tag, provided, MPI_THREAD_MULTIPLE, hazard);
    std::fflush(stderr);
}

}  // namespace lorrax_ffi::mpiguard
