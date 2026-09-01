#!/usr/bin/env bash
# The executable authority for LORRAX's ScaLAPACK/PBLAS ABI surface.
# Keep this list aligned with the declarations in blacs_grid.h; the
# distrib_la acceptance test checks that correspondence without loading an
# MPI library.
set -euo pipefail

COMPUTE=(pzheevd_ pdsyevd_ pzgetrf_ pdgetrf_ pzgetrs_ pdgetrs_ pzgemm_ pdgemm_)
OVERLAY_UNDEFINED=(numroc_ Cblacs_gridinfo)
OVERLAY_ABSENT=(descinit_ Csys2blacs_handle Cblacs_gridinit)
ALL=("${COMPUTE[@]}" "${OVERLAY_UNDEFINED[@]}" "${OVERLAY_ABSENT[@]}")

usage() {
    echo "usage: $0 --print-all | provider <shared-object> | slate-overlay <shared-object>" >&2
    exit 2
}

if [[ "${1:-}" == "--print-all" ]]; then
    printf '%s\n' "${ALL[@]}"
    exit 0
fi
[[ $# -eq 2 ]] || usage
MODE="$1"
SO="$2"
[[ -f "$SO" ]] || { echo "[scalapack-symbols] missing artifact: $SO" >&2; exit 2; }

# Strip ELF symbol versions so a conventional default-version export such as
# pdgemm_@@SCALAPACK_2.2 still satisfies dlsym("pdgemm_").  Read each table
# once; piping nm into grep -q under pipefail turns an early match into a
# false failure when nm receives SIGPIPE.
DEFINED="$(nm -D --defined-only "$SO" 2>/dev/null | awk '{print $NF}' | sed 's/@.*//')"
UNDEFINED="$(nm -D --undefined-only "$SO" 2>/dev/null | awk '{print $NF}' | sed 's/@.*//')"
[[ -n "$DEFINED" || -n "$UNDEFINED" ]] || {
    echo "[scalapack-symbols] could not read a dynamic symbol table from $SO" >&2
    exit 2
}

has_defined() { grep -Fxq "$1" <<<"$DEFINED"; }
has_undefined() { grep -Fxq "$1" <<<"$UNDEFINED"; }
bad=0

case "$MODE" in
    provider)
        for symbol in "${ALL[@]}"; do
            if has_defined "$symbol"; then
                echo "[scalapack-symbols] DEFINED $symbol"
            else
                echo "[scalapack-symbols] MISSING $symbol" >&2
                bad=1
            fi
        done
        summary="13/13 provider symbols"
        ;;
    slate-overlay)
        for symbol in "${COMPUTE[@]}"; do
            if has_defined "$symbol"; then
                echo "[scalapack-symbols] DEFINED $symbol"
            else
                echo "[scalapack-symbols] MISSING $symbol (expected DEFINED)" >&2
                bad=1
            fi
        done
        for symbol in "${OVERLAY_UNDEFINED[@]}"; do
            if has_undefined "$symbol" && ! has_defined "$symbol"; then
                echo "[scalapack-symbols] UNDEFINED $symbol"
            else
                echo "[scalapack-symbols] $symbol is not exactly UNDEFINED" >&2
                bad=1
            fi
        done
        for symbol in "${OVERLAY_ABSENT[@]}"; do
            if ! has_defined "$symbol" && ! has_undefined "$symbol"; then
                echo "[scalapack-symbols] absent $symbol"
            else
                echo "[scalapack-symbols] $symbol is not absent" >&2
                bad=1
            fi
        done
        summary="8 defined / 2 undefined / 3 absent SLATE-overlay symbols"
        ;;
    *) usage ;;
esac

[[ $bad -eq 0 ]] || {
    echo "[scalapack-symbols] FAILED: $SO violates the $MODE contract" >&2
    exit 1
}
echo "[scalapack-symbols] PASS: $summary"
