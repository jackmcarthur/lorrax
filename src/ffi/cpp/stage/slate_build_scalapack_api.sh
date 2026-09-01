#!/usr/bin/env bash
# ============================================================================
# build_scalapack_api.sh — build SLATE's OPTIONAL ScaLAPACK-compatibility
# overlay, `libslate_scalapack_api.so`, from a SLATE source tree.
#
#   LORRAX_SLATE_SRC=<slate source>  LORRAX_SLATE_HOST_INSTALL_DIR=<prefix> \
#     bash src/ffi/cpp/stage/slate_build_scalapack_api.sh [-o <outdir>]
#
# WHY THIS SCRIPT EXISTS, AND WHY THE THING IT BUILDS IS NOT A LORRAX
# DEPENDENCY.
# ----------------------------------------------------------------------------
# It is the RED TWIN for the provenance guard in
# src/ffi/cpp/scalapack/blacs_grid.h.  That guard refuses to run LORRAX's
# ScaLAPACK handlers when their pXheevd / pXgetrf have been interposed by
# SLATE's overlay; a guard whose failing case cannot be produced is void
# (wk_REL/README.md §5 lesson 1), and this script produces it.
#
# It is ALSO the measurement behind PORTING.md §0's "SLATE's own optional
# libslate_scalapack_api" row.  Run `nm -D` on the artifact and you get, for
# LORRAX's thirteen ScaLAPACK/BLACS names (SLATE v2025.05.28, measured
# 2026-07-31):
#
#     DEFINED (8)  pzheevd_ pdsyevd_ pzgetrf_ pdgetrf_ pzgetrs_ pdgetrs_
#                  pzgemm_ pdgemm_
#     UNDEF   (2)  numroc_ Cblacs_gridinfo          <- the overlay CALLS these
#     absent  (3)  descinit_ Csys2blacs_handle Cblacs_gridinit
#
# i.e. the overlay covers 100% of the OPERATIONS LORRAX asks ScaLAPACK to
# perform and 0% of the grid/descriptor infrastructure — it is a layer ON TOP
# of a real ScaLAPACK+BLACS (upstream's own link line is
# `-lslate_scalapack_api -lslate -lmkl_scalapack_lp64 ...`), never a
# replacement for one.  Do NOT put it in LORRAX_SCALAPACK_LIBRARIES as if it
# were a provider: see the three defects listed in blacs_grid.h.
#
# UPSTREAM DOES NOT BUILD THIS TARGET FROM CMAKE.  SLATE v2025.05.28's
# CMakeLists.txt has the whole `scalapack_api` block COMMENTED OUT with
# `# todo: requires ScaLAPACK` (lines 228-244) — so it is absent from a
# CMake install regardless of what -DSCALAPACK_LIBRARIES is set to.  Only the
# GNUmakefile route builds it, and that route wants a full autoconf-style
# `make config`.  This script therefore compiles the same source list the
# GNUmakefile declares (scalapack_api_src, which excludes `getri` —
# upstream: "todo: getri not finished") directly against an existing SLATE
# install, which is the cheap and reproducible way to get the artifact.
# ============================================================================
set -euo pipefail

: "${LORRAX_SLATE_SRC:?set LORRAX_SLATE_SRC to a SLATE source tree (contains scalapack_api/)}"
: "${LORRAX_SLATE_HOST_INSTALL_DIR:?set to the SLATE install prefix (contains include/ and lib64/libslate.so)}"
: "${LORRAX_MPI_INCLUDE_DIR:=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/include}"
: "${CXX:=g++}"
: "${JOBS:=$( (nproc 2>/dev/null) || echo 8 )}"

# The output directory is MANDATORY and deliberately has no default under the
# SLATE install: dropping this artifact into $LORRAX_SLATE_HOST_INSTALL_DIR/
# lib64 puts it on the rpath of every LORRAX host build, one careless
# LD_PRELOAD away from being live.  It is a test fixture; keep it in the
# campaign directory that is testing with it.
OUT="${LORRAX_SLATE_SCALAPACK_API_DIR:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        -o) OUT="$2"; shift 2 ;;
        *)  echo "usage: $0 -o <outdir>   (or set LORRAX_SLATE_SCALAPACK_API_DIR)" >&2
            exit 2 ;;
    esac
done
[ -n "$OUT" ] || {
    echo "[sapi] refusing to guess an output directory — pass -o <outdir> or set" >&2
    echo "       LORRAX_SLATE_SCALAPACK_API_DIR.  Do NOT aim it at the SLATE" >&2
    echo "       install's lib64 (see the note above this check)." >&2
    exit 2; }

SRC="$LORRAX_SLATE_SRC/scalapack_api"
INC="$LORRAX_SLATE_HOST_INSTALL_DIR/include"
LIB="$LORRAX_SLATE_HOST_INSTALL_DIR/lib64"
for p in "$SRC/scalapack_slate.hh" "$INC/slate/slate.hh" "$LIB/libslate.so" \
         "$LORRAX_MPI_INCLUDE_DIR/mpi.h"; do
    [ -e "$p" ] || { echo "[sapi] missing prerequisite: $p" >&2; exit 2; }
done

# Upstream's own list (GNUmakefile `scalapack_api_src`).  getri is excluded
# there and in the commented-out CMake block alike.
UNITS="gecon gels gemm gesv gesv_mixed gesvd getrf getrs heev heevd hemm \
her2k herk lange lanhe lansy lantr pocon posv potrf potri potrs symm syr2k \
syrk trcon trmm trsm"

# A GPU-backend SLATE install's headers include the vendor runtime header
# (blas/device.hh -> <cuda_runtime.h> / <hip/hip_runtime.h>), so anything
# compiling against it needs that include path even though this overlay
# contains no device code of its own.  Auto-detect from the toolchain that
# is on PATH; override with LORRAX_SAPI_EXTRA_INCLUDES for an unusual layout.
EXTRA_INC="${LORRAX_SAPI_EXTRA_INCLUDES:-}"
if [ -z "$EXTRA_INC" ] && grep -q 'cuda_runtime\.h' "$INC/blas/device.hh" 2>/dev/null; then
    _nvcc="$(command -v nvcc || true)"
    if [ -n "$_nvcc" ]; then
        EXTRA_INC="-I$(dirname "$(dirname "$_nvcc")")/include"
        echo "[sapi] SLATE headers want the CUDA runtime header; adding $EXTRA_INC"
    else
        echo "[sapi] WARNING: this SLATE install's headers include" >&2
        echo "       <cuda_runtime.h> but no nvcc is on PATH.  Load a CUDA" >&2
        echo "       module, or set LORRAX_SAPI_EXTRA_INCLUDES=-I<cuda>/include." >&2
    fi
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/slate_sapi.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"

echo "[sapi] CXX=$($CXX --version | head -1)"
echo "[sapi] src=$SRC"
echo "[sapi] slate=$LORRAX_SLATE_HOST_INSTALL_DIR"
echo "[sapi] out=$OUT"

n=0
for u in $UNITS; do
    [ -f "$SRC/scalapack_$u.cc" ] || {
        echo "[sapi] FATAL: $SRC/scalapack_$u.cc missing — SLATE version drift?" >&2
        exit 2; }
    n=$((n+1))
done
# xargs -P, not `wait -n` (bash 4.2 on the Frontera login image has no -n).
printf '%s\n' $UNITS | xargs -P "$JOBS" -I{} sh -c \
    "\"$CXX\" -std=c++17 -O2 -fPIC -fopenmp -I'$INC' -I'$SRC' \
     -I'$LORRAX_MPI_INCLUDE_DIR' $EXTRA_INC -c '$SRC/scalapack_{}.cc' \
     -o '$TMP/sa_{}.o' 2>'$TMP/err_{}'" || true
rc=0
for u in $UNITS; do
    if [ ! -f "$TMP/sa_$u.o" ]; then
        echo "[sapi] COMPILE FAILED: scalapack_$u.cc" >&2
        sed 's/^/    /' "$TMP/err_$u" >&2 || true
        rc=1
    fi
done
[ "$rc" -eq 0 ] || exit 1
echo "[sapi] compiled $n translation units"

# Link exactly as upstream does: the overlay in front of libslate, and NO
# ScaLAPACK of its own — the undefined numroc_ / Cblacs_* are meant to be
# satisfied by whatever real ScaLAPACK+BLACS the interposed process already
# has.  Leaving them undefined is what makes `nm -D` show the gap honestly.
SO="$OUT/libslate_scalapack_api.so"
"$CXX" -shared -fopenmp -o "$SO" "$TMP"/sa_*.o \
    -Wl,-soname,libslate_scalapack_api.so \
    -L"$LIB" -lslate -Wl,-rpath,"$LIB"
echo "[sapi] --- artifact ---"; ls -lh "$SO"

# The shared checker owns the exact thirteen-symbol surface.  For this
# artifact it requires 8 compute definitions (including pdgemm/pzgemm),
# 2 unresolved grid queries, and 3 absent grid/descriptor helpers.
bash "$(dirname "${BASH_SOURCE[0]}")/../scalapack/check_symbol_contract.sh" \
    slate-overlay "$SO"
echo "[sapi] done — this is an OVERLAY, not a ScaLAPACK."
echo "[sapi] LORRAX REFUSES it as a provider; see src/ffi/cpp/scalapack/blacs_grid.h."
