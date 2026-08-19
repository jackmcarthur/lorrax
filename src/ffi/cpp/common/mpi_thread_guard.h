// mpi_thread_guard.h — the ONE thread-level requirement shared by
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
// loses first.  Each family keeps its own hazard sentence; the shared guard
// aborts the whole MPI world before either family enters a collective.
//
// This header includes <mpi.h> ON PURPOSE and must therefore never be
// included from the comms-free TUs (mklblas, mklfft, cufft) — that
// constraint is why the logic cannot live in mkl_thread_pin.h (see its
// HARD CONSTRAINT note).
//
// WHY THIS REFUSES (measured; scorecard AS.4b): with MPI initialized
// before these libraries at a thread level below MPI_THREAD_MULTIPLE, two
// threads of one rank end up concurrently inside MPID_Progress_wait — a
// ~29% provider-independent multi-node segfault/hang rate at the
// zeta-write/V_q boundary, minutes after the cause, with a backtrace that
// names neither cause nor fix.  Continuing after detecting that known-
// undefined configuration only moves the failure downstream and can strand
// peer ranks in a collective, so every rank aborts before native I/O/linalg.
//
// PLACEMENT CONTRACT for callers: invoke from ensure_mpi_initialized() so
// the query runs on BOTH paths — after we call MPI_Init_thread ourselves
// AND after an early return because someone else already initialized.  The
// already-initialized path is the hazardous one; a guard that only ran on
// self-init would be void by construction.

#pragma once

#include <cstdlib>
#include <cstdio>

#include <mpi.h>

namespace lorrax_ffi::mpiguard {

// Fail on every rank when the granted MPI thread level is insufficient.
// MPI_Abort is intentional: returning an error on one rank while peers enter
// a PHDF5/SLATE collective is a hang, not graceful failure.
inline void require_thread_multiple(const char* tag, const char* hazard) {
    int provided = 0;
    MPI_Query_thread(&provided);
    if (provided >= MPI_THREAD_MULTIPLE) return;
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::fprintf(stderr,
            "[%s] FATAL: MPI granted thread level %d < MPI_THREAD_MULTIPLE "
            "(%d).  %s is UNDEFINED at this level; this configuration has a "
            "measured ~29%% multi-node crash rate (scorecard AS.4b).  Use the "
            "site CPU-MPI launch recipe and verify the live thread grant "
            "before enabling native MPI I/O/linalg "
            "(docs/dev/mpi_collectives.md).  Aborting the MPI world before "
            "peer ranks can enter a native collective.\n",
            tag, provided, MPI_THREAD_MULTIPLE, hazard);
        std::fflush(stderr);
    }
    MPI_Abort(MPI_COMM_WORLD, 86);
    std::abort();  // MPI_Abort is specified not to return; defensive fallback.
}

}  // namespace lorrax_ffi::mpiguard
