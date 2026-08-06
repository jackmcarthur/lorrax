#!/usr/bin/env bash
# ===========================================================================
# Post-link gate (hazard S3): EXACTLY ONE cray-mpich runtime object may be
# mapped into the process.  Two libmpi means MPI_COMM_WORLD and the MPI_Comm
# handed to Csys2blacs_handle mean different things in different frames; it
# links fine and corrupts or hangs at the first collective.
#
#   usage: gate_one_mpi.sh <so-file> [expected-variant]
#
# WHY ldd AND NOT `readelf -d`:
#   readelf -d lists DIRECT DT_NEEDED entries only.  The FFI links libmpi
#   itself AND libhdf5_parallel_gnu_123, and it is HDF5s OWN DT_NEEDED
#   (libmpi_gnu_123.so.12) that can smuggle in the second MPI.  readelf
#   never sees it, so the old gate passed binaries it was written to catch.
#   ldd resolves the entire closure under the real LD_LIBRARY_PATH.
#
# WHY distinct RESOLVED FILES AND NOT distinct NAME STRINGS:
#   on the container leg the phdf5 stage ships libmpi_gnu_123.so.12 as a
#   deliberate FILENAME ALIAS symlink onto shifters libmpi_gnu_91.so.12.0.0,
#   so HDF5s _123 request and our _91 request land on ONE object.  Counting
#   distinct spellings reports 2 and fails a correct binary.  The invariant
#   that actually matters is one MAPPED libmpi, so we dedupe by realpath.
# ===========================================================================
set -uo pipefail

SO="${1:?usage: gate_one_mpi.sh <so-file> [expected-variant]}"
EXPECT="${2:-}"
TAG="${GATE_TAG:-gate_one_mpi}"

# A GATE THAT CANNOT RUN IS NOT A GATE THAT PASSED.
#
# Until 2026-08-06 a missing ldd printed "GATE SKIPPED" and exited 0, which
# both callers (build.sh, build_ffi_host.sh) read as success — a green light
# manufactured out of a missing tool, on the ONE hazard whose whole point is
# that it "links fine and corrupts or hangs at the first collective".  The
# only two honest outcomes here are PASSED and FAILED.
#
# The escape is explicit and announced, following the FFI gate contract
# (src/ffi/gate.py: an opt-out is a stated decision, never an inference):
# LORRAX_GATE_ONE_MPI=off disables the gate and says so on every run.
if [[ "${LORRAX_GATE_ONE_MPI:-on}" == "off" ]]; then
    echo "[$TAG] GATE DISABLED by LORRAX_GATE_ONE_MPI=off — hazard S3 (two" >&2
    echo "[$TAG] cray-mpich runtimes in one process) is NOT checked for this" >&2
    echo "[$TAG] artifact.  An unchecked .so must not be certified." >&2
    exit 0
fi

if ! command -v ldd >/dev/null 2>&1; then
    echo "[$TAG] GATE FAILED (S3): ldd is not available, so the library" >&2
    echo "[$TAG]   closure cannot be resolved and the one-libmpi invariant" >&2
    echo "[$TAG]   cannot be checked.  This is NOT a pass: readelf -d is not" >&2
    echo "[$TAG]   a substitute (it sees direct DT_NEEDED only and misses the" >&2
    echo "[$TAG]   second MPI that arrives through libhdf5_parallel_gnu_123)." >&2
    echo "[$TAG]   Install binutils/glibc-utils, run the gate inside the" >&2
    echo "[$TAG]   container where ldd exists, or state the risk explicitly" >&2
    echo "[$TAG]   with LORRAX_GATE_ONE_MPI=off." >&2
    exit 1
fi

LDD_OUT="$(ldd "$SO" 2>&1)"

# Unresolved deps make the closure incomplete -> the gate cannot be trusted.
if printf %s "$LDD_OUT" | grep -q "not found"; then
    echo "[$TAG] GATE FAILED (S3): unresolved dependencies; closure incomplete:" >&2
    printf %s\\n "$LDD_OUT" | grep "not found" >&2
    exit 1
fi

# Every resolved path in the closure whose basename is an MPI *runtime*
# (libmpi.so / libmpi_gnu_<N>.so).  Excludes libmpifort*, libmpi_gtl_*.
mapfile -t MPI_PATHS < <(printf %s\\n "$LDD_OUT" \
    | sed -n "s|.*=> \\(/[^ ]*\\).*|\\1|p" \
    | grep -E "/libmpi(_gnu_[0-9]+)?\\.so" || true)

if [[ ${#MPI_PATHS[@]} -eq 0 ]]; then
    echo "[$TAG] GATE FAILED (S3): no libmpi in the closure of $SO" >&2
    exit 1
fi

# Dedupe by the FILE the loader will actually map.
mapfile -t MPI_REAL < <(for p in "${MPI_PATHS[@]}"; do readlink -f "$p"; done | sort -u)
# The SONAMEs actually mapped (one per distinct object).
mapfile -t MPI_SONAMES < <(for p in "${MPI_REAL[@]}"; do
    readelf -d "$p" 2>/dev/null | sed -n "s/.*SONAME.*\\[\\(.*\\)\\].*/\\1/p"
done | sort -u)

echo "[$TAG] libmpi requests in closure:"
printf %s\\n "$LDD_OUT" | grep -E "/libmpi(_gnu_[0-9]+)?\\.so" | sed "s/^/[$TAG]   /"
echo "[$TAG] distinct mapped objects: ${#MPI_REAL[@]}"
for p in "${MPI_REAL[@]}"; do echo "[$TAG]   $p"; done
echo "[$TAG] SONAMEs: ${MPI_SONAMES[*]}"

if [[ ${#MPI_REAL[@]} -ne 1 ]]; then
    echo "[$TAG] GATE FAILED (S3): ${#MPI_REAL[@]} distinct libmpi objects mapped." >&2
    exit 1
fi

if [[ -n "$EXPECT" ]]; then
    if ! printf %s\\n "${MPI_SONAMES[@]}" | grep -q "$EXPECT"; then
        echo "[$TAG] GATE FAILED (S3): mapped ${MPI_SONAMES[*]}, expected $EXPECT" >&2
        exit 1
    fi
fi

echo "[$TAG] GATE 1 (S3, one cray-mpich runtime) PASSED: ${MPI_SONAMES[*]}"
exit 0
