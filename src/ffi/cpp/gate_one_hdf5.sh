#!/usr/bin/env bash
# ===========================================================================
# Post-link gate: EXACTLY ONE HDF5 may be reachable, and it must be the one
# the container stages.
#
#   usage: gate_one_hdf5.sh <so-file> [<so-file> ...]
#   env:   LORRAX_PHDF5_STAGE      the bind-mount stage tree (dir with lib/).
#                                  When set, the SONAME the artifact requests
#                                  must be one the stage PROVIDES, and the
#                                  stage must provide exactly one HDF5.
#          LORRAX_GATE_ONE_HDF5=off   announced opt-out (never silent).
#
# WHY THIS GATE EXISTS
# --------------------
# The two FFI legs are built in two different places against two different
# HDF5s, and until 2026-08-06 nothing compared them:
#
#   host leg    config/perlmutter/build_ffi_host.sh, BARE METAL, against the
#               cray-hdf5-parallel MODULE  -> SONAME libhdf5_parallel_gnu.so.310
#   device leg  src/ffi/cpp/build.sh, IN-CONTAINER, against whatever is
#               bind-mounted at /lorrax_phdf5 -> was HDF5 1.12,
#               SONAME libhdf5_parallel_gnu_123.so.200
#
# That is a MAJOR-VERSION skew, and it has two distinct failure modes, which
# is why this gate has two legs of its own:
#
#   (1) The SONAME the stage does not provide is simply `not found`, and the
#       WHOLE library fails to dlopen -- every handler in it, not just the
#       phdf5 ones.  Measured 2026-08-06: the host leg's only remaining
#       in-container `not found` was libhdf5_parallel_gnu.so.310 (CLAIMS 89).
#       That one is loud, and the loudness is the good case.
#
#   (2) The tempting repair for (1) is to put BOTH HDF5s where the loader can
#       see them -- stage 1.14 beside the 1.12 that the device leg needs, or
#       alias one SONAME onto the other.  That resolves, and is far worse: two
#       HDF5 major versions get MAPPED INTO ONE PROCESS, each with its own
#       error stack, file-locking state, free lists and open-file table, both
#       reachable from the same phdf5 handler code and both writing the same
#       .h5.  It loads and then misbehaves, which is the failure mode with no
#       symptom at the seam.
#
# So the invariant is not "the SONAME resolves".  It is ONE HDF5, and both
# legs agree which one.
#
# WHY ldd AND NOT `readelf -d` FOR THE MAPPING CHECK: same reason as
# gate_one_mpi.sh -- readelf sees direct DT_NEEDED only, and a second HDF5 can
# arrive transitively.  Dedupe by resolved realpath, because a stage may ship
# filename aliases onto one object on purpose (the libmpi shims do exactly
# that) and counting spellings would fail a correct artifact.
#
# WHAT THIS GATE RETURNS WHEN THE PROPERTY IS FALSE -- stated here so no
# caller has to guess, and so this is not another `nm -D | grep -c fftw_`
# (CLAIMS 88: that check read 0 while the library was completely unloadable,
# because dlsym drives it to 0 by construction):
#   - two SOVERSIONs across the inputs        -> FAIL 7a, both printed
#   - a stage carrying two SOVERSIONs         -> FAIL 7b, both printed
#   - requested SONAME absent from the stage  -> FAIL 7c, want/have printed
#   - `not found` anywhere in a closure       -> FAIL 7d, the lines printed
#   - two distinct mapped libhdf5 objects     -> FAIL 7e, both realpaths
# Each of 7b/7c is exercised against a real tree in claims/, not argued.
# ===========================================================================
set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: gate_one_hdf5.sh <so-file> [<so-file> ...]" >&2
    exit 2
fi

TAG="${GATE_TAG:-gate_one_hdf5}"
STAGE="${LORRAX_PHDF5_STAGE:-}"

if [[ "${LORRAX_GATE_ONE_HDF5:-on}" == "off" ]]; then
    echo "[$TAG] GATE DISABLED by LORRAX_GATE_ONE_HDF5=off — the one-HDF5" >&2
    echo "[$TAG] invariant is NOT checked for this artifact.  An unchecked" >&2
    echo "[$TAG] .so must not be certified." >&2
    exit 0
fi

for so in "$@"; do
    if [[ ! -f "$so" ]]; then
        echo "[$TAG] GATE FAILED: no such artifact: $so" >&2
        exit 1
    fi
done

# --- what the artifacts REQUEST (direct DT_NEEDED) -------------------------
mapfile -t REQ < <(for so in "$@"; do
    readelf -d "$so" 2>/dev/null | grep NEEDED \
        | grep -oE 'libhdf5[a-zA-Z0-9_]*\.so\.[0-9]+'
done | sort -u)

if [[ ${#REQ[@]} -eq 0 ]]; then
    # Not a vacuous pass: this is a stated fact about the artifact.  A build
    # with -DLORRAX_FFI_HAVE_PHDF5=OFF legitimately links no HDF5, and then
    # there is no skew to have.  CLAIMS 83's rule is that a gate which
    # SCANNED NOTHING must not print PASS; this one scanned and found none.
    echo "[$TAG] GATE 7 N/A: none of the $# artifact(s) link HDF5 at all"
    echo "[$TAG]   (LORRAX_FFI_HAVE_PHDF5=OFF).  Nothing to skew."
    exit 0
fi

# The SOVERSION is what a major-version skew shows up as.  Cray spells the
# same library several ways (libhdf5_parallel_gnu.so.310,
# libhdf5_parallel_gnu_123.so.200); the trailing integer is the fact.
mapfile -t REQ_SOV < <(printf '%s\n' "${REQ[@]}" | grep -oE '[0-9]+$' | sort -u)

echo "[$TAG] HDF5 requested by the artifact(s):"
printf '%s\n' "${REQ[@]}" | sed "s/^/[$TAG]   /"

if [[ ${#REQ_SOV[@]} -ne 1 ]]; then
    echo "[$TAG] GATE FAILED (7a): ${#REQ_SOV[@]} HDF5 SOVERSIONs across the" >&2
    echo "[$TAG]   artifacts: ${REQ_SOV[*]}.  The legs disagree about which" >&2
    echo "[$TAG]   HDF5 they are built for.  Whichever one is missing at run" >&2
    echo "[$TAG]   time takes its WHOLE library down at dlopen; putting both" >&2
    echo "[$TAG]   where the loader can see them is worse (two HDF5 majors in" >&2
    echo "[$TAG]   one process).  Rebuild both legs against ONE HDF5." >&2
    exit 1
fi

# --- what the STAGE PROVIDES ----------------------------------------------
if [[ -n "$STAGE" ]]; then
    if [[ ! -d "$STAGE/lib" ]]; then
        echo "[$TAG] GATE CANNOT RUN (7b): LORRAX_PHDF5_STAGE=$STAGE has no" >&2
        echo "[$TAG]   lib/ directory, so what the container will provide is" >&2
        echo "[$TAG]   unknown.  This is NOT a pass — the whole point of the" >&2
        echo "[$TAG]   gate is that the build side and the container side are" >&2
        echo "[$TAG]   populated separately and drifted apart once already." >&2
        echo "[$TAG]   fix: run src/ffi/cpp/stage/phdf5_stage_cray.sh, or name" >&2
        echo "[$TAG]        the tree with LORRAX_FFI_PHDF5_DIR." >&2
        exit 1
    fi
    mapfile -t PROV < <(for f in "$STAGE"/lib/libhdf5*.so*; do
        [[ -e "$f" && ! -L "$f" ]] || continue
        readelf -d "$f" 2>/dev/null | sed -n 's/.*SONAME.*\[\(.*\)\].*/\1/p'
    done | sort -u)
    mapfile -t PROV_SOV < <(printf '%s\n' "${PROV[@]}" \
        | grep -oE '[0-9]+$' | sort -u)

    echo "[$TAG] HDF5 provided by the stage $STAGE:"
    printf '%s\n' "${PROV[@]}" | sed "s/^/[$TAG]   /"

    if [[ ${#PROV_SOV[@]} -ne 1 ]]; then
        echo "[$TAG] GATE FAILED (7b): the stage provides ${#PROV_SOV[@]} HDF5" >&2
        echo "[$TAG]   SOVERSIONs: ${PROV_SOV[*]}.  A stage that carries two" >&2
        echo "[$TAG]   HDF5 majors lets the two FFI legs bind DIFFERENT ones" >&2
        echo "[$TAG]   and both get mapped into one process.  It resolves, so" >&2
        echo "[$TAG]   nothing complains, and the two libraries then keep" >&2
        echo "[$TAG]   independent error stacks, file locks and open-file" >&2
        echo "[$TAG]   tables over the same .h5.  Re-stage from ONE module:" >&2
        echo "[$TAG]     rm -rf $STAGE" >&2
        echo "[$TAG]     module load PrgEnv-gnu cray-hdf5-parallel/<one> cray-mpich" >&2
        echo "[$TAG]     LORRAX_FFI_PHDF5_DIR=$STAGE \\" >&2
        echo "[$TAG]       src/ffi/cpp/stage/phdf5_stage_cray.sh" >&2
        exit 1
    fi

    if ! printf '%s\n' "${PROV[@]}" | grep -qxF "${REQ[0]}"; then
        echo "[$TAG] GATE FAILED (7c): the artifact requests ${REQ[*]}," >&2
        echo "[$TAG]   the stage provides ${PROV[*]}." >&2
        echo "[$TAG]   In the container that request is a plain 'not found'" >&2
        echo "[$TAG]   and the ENTIRE library fails to dlopen — ScaLAPACK," >&2
        echo "[$TAG]   SLATE, GEMM and phdf5 handlers together, none of which" >&2
        echo "[$TAG]   touch HDF5.  Do NOT repair this with a SONAME alias: a" >&2
        echo "[$TAG]   major-version alias loads and then misbehaves." >&2
        echo "[$TAG]   Either re-stage the container from the module this" >&2
        echo "[$TAG]   build loads, or build against the staged tree — and" >&2
        echo "[$TAG]   whichever you pick, BOTH legs move together." >&2
        exit 1
    fi
fi

# --- what actually gets MAPPED --------------------------------------------
#
# THIS HALF IS ENVIRONMENT-DEPENDENT and the two above are not.  The request
# and stage halves read SONAMEs out of ELF headers and out of a directory; this
# one asks the dynamic loader to resolve a closure, and the answer belongs to
# the machine it is asked on.  A DEVICE library examined from a login node has
# libnccl / libnvshmem_host / libucc 'not found' while being perfectly correct,
# and so does a peer artifact from the other leg.
#
# LORRAX_GATE_ONE_HDF5_CLOSURE=off runs the environment-independent halves and
# SAYS SO — it does not print an unqualified PASS, because a gate that skipped
# a half must not report the whole property.  scripts/verify_ffi_build.sh sets
# it exactly when it knows this is the wrong environment, and counts the
# unchecked half as COULD NOT RUN.
if [[ "${LORRAX_GATE_ONE_HDF5_CLOSURE:-on}" == "off" ]]; then
    echo "[$TAG] GATE 7 request+stage halves PASSED: ${REQ[*]}"
    echo "[$TAG]   closure half NOT CHECKED here (LORRAX_GATE_ONE_HDF5_CLOSURE=off):"
    echo "[$TAG]   whether two distinct HDF5 objects would be MAPPED into one"
    echo "[$TAG]   process is a property of the environment the library runs in,"
    echo "[$TAG]   not of this one."
    exit 0
fi
if command -v ldd >/dev/null 2>&1; then
    ALL_LDD=""
    for so in "$@"; do
        out="$(ldd "$so" 2>&1)"
        if printf %s "$out" | grep -q "not found"; then
            echo "[$TAG] GATE FAILED (7d): unresolved dependencies in the" >&2
            echo "[$TAG]   closure of $so — the closure is incomplete, so the" >&2
            echo "[$TAG]   one-HDF5 invariant cannot be checked and the .so" >&2
            echo "[$TAG]   will not dlopen here at all:" >&2
            printf %s\\n "$out" | grep "not found" | sed "s/^/[$TAG]   /" >&2
            exit 1
        fi
        ALL_LDD+="$out"$'\n'
    done
    mapfile -t H5_REAL < <(printf %s "$ALL_LDD" \
        | sed -n 's|.*=> \(/[^ ]*\).*|\1|p' \
        | grep -E '/libhdf5[a-zA-Z0-9_]*\.so' \
        | while read -r p; do readlink -f "$p"; done | sort -u)
    echo "[$TAG] distinct mapped HDF5 objects: ${#H5_REAL[@]}"
    for p in "${H5_REAL[@]}"; do echo "[$TAG]   $p"; done
    if [[ ${#H5_REAL[@]} -ne 1 ]]; then
        echo "[$TAG] GATE FAILED (7e): ${#H5_REAL[@]} distinct HDF5 objects" >&2
        echo "[$TAG]   would be mapped into one process." >&2
        exit 1
    fi
else
    echo "[$TAG] GATE FAILED (7d): ldd is not available, so the closure" >&2
    echo "[$TAG]   cannot be resolved.  readelf -d is not a substitute: it" >&2
    echo "[$TAG]   sees direct DT_NEEDED only and misses an HDF5 arriving" >&2
    echo "[$TAG]   transitively.  Run the gate where ldd exists, or state" >&2
    echo "[$TAG]   the risk with LORRAX_GATE_ONE_HDF5=off." >&2
    exit 1
fi

echo "[$TAG] GATE 7 (one HDF5, and the stage provides it) PASSED: ${REQ[*]}"
exit 0
