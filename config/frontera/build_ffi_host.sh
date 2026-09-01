#!/usr/bin/env bash
# ============================================================================
# build_ffi_host.sh — build liblorrax_ffi_host.so (the CUDA-free host-platform
# FFI library) on Frontera, WITH the phdf5 read AND write handlers.  RUN
# INSIDE the
# python:3.12 apptainer container (glibc 2.28+), same host Intel-MPI parallel
# HDF5 the CUDA build_ffi.sh uses for its phdf5 path — minus every
# CUDA/NCCL/cuSOLVERMp dependency.
#
#   config/frontera/build_ffi_host.sh [--fresh]
#
# Output: $LORRAX_FFI_HOST_STAGE/liblorrax_ffi_host.so
#   (default $LORRAX_FFI_STAGE_WTA/build_host; point LORRAX_FFI_HOST_SO there
#    at runtime).  A UNIQUE stage dir so it never clobbers the shared
#    $WORK/lorrax_ffi CUDA .so.
#
# ----------------------------------------------------------------------------
# ScaLAPACK is standard; SLATE is an independent opt-in
# ----------------------------------------------------------------------------
# The host library always carries the ScaLAPACK/PBLAS handlers, using MKL by
# default or LORRAX_SCALAPACK_LIBRARIES when explicitly supplied.  Setting
# LORRAX_SLATE_HOST_INSTALL_DIR adds the separate SLATE handlers; it does not
# enable or configure ScaLAPACK.
#
#   LORRAX_SLATE_HOST_INSTALL_DIR=$WORK/slate_builds/cpu/install \
#     config/frontera/build_ffi_host.sh --fresh
#
# Building that SLATE install is a SEPARATE, one-time step; SLATE is external
# source, so its two Frontera deviations cannot live in this repo's CMake and
# are recorded here instead (workstream L, 2026-07-25; the working script is
# /scratch2/08271/jackmc/lorrax_setup/wk_L/build_slate_host.sh):
#
#   1. PATCH SLATE: "LANGUAGES CXX Fortran" -> "LANGUAGES CXX".
#      SLATE declares Fortran but uses it NOWHERE (only in the commented-out
#      scalapack_api); the py312 container has no gfortran, so the configure
#      dies before compiling anything.  One line, applied to the SLATE source
#      tree before configuring:
#
#        sed -i 's/    LANGUAGES CXX Fortran/    LANGUAGES CXX/' \
#            "$SLATE_SRC/CMakeLists.txt"
#
#   2. CONFIGURE SLATE with MKL, not libsci (the Perlmutter recipe's default):
#        -Dgpu_backend=none -Dblas=mkl -Dblas_int=int -Dblas_threaded=true
#        -Dbuild_tests=no -DSCALAPACK_LIBRARIES=""
#        -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_LIBDIR=lib64
#      (SCALAPACK_LIBRARIES="" is why THIS lib must link ScaLAPACK itself —
#       see the ScaLAPACK block in src/ffi/cpp/CMakeLists.txt.)
#
#   3. BUILD ON NODE-LOCAL /tmp.  /work2 is Lustre; under a concurrent
#      40-node job even `python -c pass` from the venv stalls minutes in
#      ptlrpc_set_wait.  Stage the SLATE source + cmake/ninja ELF binaries to
#      /tmp (one sequential tar read, ~2-min build) and pass
#      -DLORRAX_XLA_FFI_INCLUDE_DIR explicitly so cmake never imports jax.
#
# Runtime for the SLATE/ScaLAPACK targets needs LD_LIBRARY_PATH ⊇ MKL libdir
# + the Intel compiler runtime (libimf/libsvml/libirng/libintlc) + SLATE's
# lib64.  KNOWN BUG L-2: SLATE host `heev` SIGSEGVs deterministically (even
# 1×1, n=64) and is REJECTED at resolve time on CPU meshes.  The distributed
# CPU eigh is ScaLAPACK `pzheevd` (ScalapackEighHostFfi, workstream V);
# potrf/trsm/ScaLAPACK getrf are clean.  See docs/dev/linalg_ffi.md.
#
# ----------------------------------------------------------------------------
# WHICH LIBRARY SUPPLIES WHAT, AND HOW ITS ABSENCE ANNOUNCES ITSELF
# ----------------------------------------------------------------------------
# Four numerical libraries can end up inside liblorrax_ffi_host.so.  THREE OF
# THE FOUR ARE INTERCHANGEABLE WITH ANOTHER VENDOR'S, because LORRAX calls a
# published API, not a product:
#
#   what is linked here           API LORRAX calls        swap it for
#   ---------------------------   ---------------------   --------------------
#   libmkl_scalapack_lp64 +       ScaLAPACK + C-BLACS     Cray LibSci
#   libmkl_blacs_intelmpi_lp64    (13 Fortran-ABI names,  (libsci_*_mpi_*),
#   (+ the mkl_intel/thread/core  hand-declared in        netlib, AOCL, or
#   layers they need)             cpp/scalapack/          netlib or AOCL
#                                 blacs_grid.h)           (LP64 ABI only)
#   libmkl_intel_lp64 (CBLAS)     cblas_dgemm/zgemm       OpenBLAS, BLIS,
#                                 (+ an OPTIONAL batched  LibSci, ATLAS
#                                 extension asked for at
#                                 runtime, never here)
#   libmkl_intel_lp64 (DFTI)      DftiCreateDescriptor/   NOTHING — DFTI is
#                                 SetValue/Compute*       Intel-only.  This is
#                                                         the one host family
#                                                         with a single-vendor
#                                                         API.
#   libslate + blaspp/lapackpp    slate:: C++ templates   NOTHING — a template
#                                                         library has no ABI a
#                                                         second vendor could
#                                                         implement.
#
# To use a different ScaLAPACK, pass the whole link line and skip the MKL
# probe entirely — no source change, nothing else to configure:
#
#   LORRAX_SCALAPACK_LIBRARIES="-L/opt/cray/pe/libsci/.../lib -lsci_gnu_mpi_mp -lsci_gnu_mp" \
#     config/frontera/build_ffi_host.sh --fresh
#
# The three groups are independent: no SLATE install still gives you the
# ScaLAPACK, GEMM and FFT handlers, and no ScaLAPACK still gives you SLATE's.
# Each group prints one configure line saying what it resolved or why it was
# skipped — READ THE CONFIGURE LOG FIRST.  If you no longer have it, the
# runtime symptoms map back like this:
#
#   cannot resolve pzheevd_/pdsyevd_/pzgetrf_/pdgetrf_/pzgetrs_/pdgetrs_/
#   pzgemm_/pdgemm_/numroc_/descinit_/Csys2blacs_handle/
#   Cblacs_gridinit/Cblacs_gridinfo
#       -> no ScaLAPACK+BLACS was linked.  Set LORRAX_MKL_ROOT, or pass
#       LORRAX_SCALAPACK_LIBRARIES.  Those THIRTEEN names are the complete
#       numerical/grid surface. The explicit link line must be LP64
#       (32-bit Fortran INTEGER); an ILP64 library is ABI-incompatible.
#   cannot resolve cblas_dgemm/cblas_zgemm  -> a CBLAS header was found but
#       its library was not on the link line.  (cblas_?gemm_batch will never
#       appear here; it is looked up at runtime and its absence just selects
#       the plain-GEMM loop.)
#   cannot resolve Dfti*  -> the MKL runtime is missing from LD_LIBRARY_PATH.
#   cannot resolve a mangled _ZN5slate...  -> libslate.so is missing from
#       LD_LIBRARY_PATH.
#   a collective hangs or aborts inside blacs_gridinit, no message  -> the
#       BLACS flavour does not match the MPI.  MKL ships one BLACS per MPI
#       (mkl_blacs_intelmpi_lp64 vs mkl_blacs_openmpi_lp64); the wrong one
#       links perfectly and only fails at the first grid call.
#
# The authoritative version of this map, with the CMake variable names, is the
# "HOST NUMERICAL LIBRARIES" block at the top of the resolution section in
# src/ffi/cpp/CMakeLists.txt.
# ============================================================================
set -euo pipefail

: "${LORRAX_ROOT:?set LORRAX_ROOT to the worktree/repo root (contains src/ffi)}"
: "${LORRAX_VENV:=$WORK/lorrax_env/.venv}"
: "${LORRAX_FFI_STAGE_WTA:=$WORK/lorrax_ffi_wtA}"
: "${LORRAX_HDF5_ROOT:=/home1/apps/intel19/impi19_0/phdf5/1.14.6}"
: "${LORRAX_IMPI_ROOT:=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64}"
# MKL supplies ScaLAPACK + BLACS (and SLATE's BLAS/LAPACK).  2020.1 is the
# newest on Frontera that ships libmkl_scalapack_lp64.so; 2019.5 also works.
: "${LORRAX_MKL_ROOT:=/opt/intel/compilers_and_libraries_2020.1.217/linux/mkl}"
# Empty disables only the independent SLATE handlers.
: "${LORRAX_SLATE_HOST_INSTALL_DIR:=}"

PY="$LORRAX_VENV/bin/python"
CMAKE="$LORRAX_VENV/bin/cmake"
NINJA="$LORRAX_VENV/bin/ninja"
SRC="$LORRAX_ROOT/src/ffi/cpp"
BUILD="${LORRAX_FFI_HOST_STAGE:-$LORRAX_FFI_STAGE_WTA/build_host}"

# cmake + ninja live in the venv (pip-installed by stage_ffi_deps.sh); put
# them on PATH so cmake's -G Ninja resolves the build program.
export PATH="$LORRAX_VENV/bin:$PATH"

for p in "$PY" "$CMAKE" "$LORRAX_HDF5_ROOT/include/H5pubconf.h" \
         "$LORRAX_IMPI_ROOT/include/mpi.h" "$SRC/CMakeLists.txt"; do
    [ -e "$p" ] || { echo "[build_host] missing prerequisite: $p" >&2; exit 2; }
done

# Intel MPI wrapper env (same as build_ffi.sh's phdf5 branch): HDF5's
# hdf5-config.cmake runs find_package(MPI); the config/frontera/cmake FindMPI
# stub (on CMAKE_MODULE_PATH) satisfies it with Intel MPI + libfabric.  The
# container has gcc/g++ but no icc, so force the wrappers to gcc/g++.
export I_MPI_ROOT="${LORRAX_IMPI_ROOT%/intel64}"
export I_MPI_CC=gcc
export I_MPI_CXX=g++
export LIBRARY_PATH="$LORRAX_IMPI_ROOT/libfabric/lib:$LORRAX_IMPI_ROOT/lib/release:$LORRAX_IMPI_ROOT/lib:${LIBRARY_PATH:-}"

ARGS=(
    -S "$SRC" -B "$BUILD" -G Ninja
    -DLORRAX_FFI_PLATFORM=host
    -DLORRAX_HOST_HAVE_SCALAPACK=ON
    -DCMAKE_MAKE_PROGRAM="$NINJA"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_COMPILER=g++
    -DPython3_EXECUTABLE="$PY"            # jax.ffi.include_dir() probe uses this
    -DLORRAX_FFI_HAVE_PHDF5=ON
    -DHDF5_ROOT="$LORRAX_HDF5_ROOT"
    -DHDF5_PREFER_PARALLEL=ON
    -DLORRAX_MPI_INCLUDE_DIR="$LORRAX_IMPI_ROOT/include"
    -DLORRAX_MPICH_LIB_DIR="$LORRAX_IMPI_ROOT/lib/release"
    -DCMAKE_MODULE_PATH="$LORRAX_ROOT/config/frontera/cmake"
    -DLORRAX_IMPI_ROOT="$LORRAX_IMPI_ROOT"
)

# ScaLAPACK provider.  CMake owns provider resolution; this site script only
# supplies the site value.  This block is deliberately outside the SLATE
# conditional below.
if [ -n "${LORRAX_SCALAPACK_LIBRARIES:-}" ]; then
    ARGS+=(-DLORRAX_SCALAPACK_LIBRARIES="$LORRAX_SCALAPACK_LIBRARIES")
    echo "[build_host] ScaLAPACK from the explicit link line:"
    echo "[build_host]   $LORRAX_SCALAPACK_LIBRARIES"
    # MKL headers still supply the independent DFTI/CBLAS handlers when the
    # site has them, even if another library supplies ScaLAPACK.
    if [ -f "$LORRAX_MKL_ROOT/include/mkl_dfti.h" ]; then
        ARGS+=(-DLORRAX_MKL_ROOT="$LORRAX_MKL_ROOT")
        echo "[build_host]   (MKL at $LORRAX_MKL_ROOT still used for the FFT/GEMM headers)"
    else
        echo "[build_host]   (no MKL headers — the DFTI FFT handlers will be skipped;"
        echo "[build_host]    set LORRAX_CBLAS_DIR for the GEMM handler's cblas.h)"
    fi
else
    [ -f "$LORRAX_MKL_ROOT/lib/intel64_lin/libmkl_scalapack_lp64.so" ] || {
        echo "[build_host] no libmkl_scalapack_lp64.so under $LORRAX_MKL_ROOT" >&2
        echo "             ScaLAPACK is an API with several implementations —" >&2
        echo "             either set LORRAX_MKL_ROOT to an MKL prefix, or set" >&2
        echo "             LORRAX_SCALAPACK_LIBRARIES to a whole link line for" >&2
        echo "             any other one (Cray LibSci, netlib, AOCL)." >&2
        exit 2; }
    ARGS+=(-DLORRAX_MKL_ROOT="$LORRAX_MKL_ROOT")
    echo "[build_host] ScaLAPACK/BLACS from MKL at $LORRAX_MKL_ROOT"
fi

# Optional SLATE provider, independent of the ScaLAPACK/PBLAS block above.
if [ -n "$LORRAX_SLATE_HOST_INSTALL_DIR" ]; then
    [ -d "$LORRAX_SLATE_HOST_INSTALL_DIR/lib64/cmake/slate" ] || {
        echo "[build_host] LORRAX_SLATE_HOST_INSTALL_DIR set but no" >&2
        echo "             $LORRAX_SLATE_HOST_INSTALL_DIR/lib64/cmake/slate" >&2
        exit 2; }
    ARGS+=(
        -DLORRAX_HOST_HAVE_SLATE=ON
        -DLORRAX_SLATE_HOST_INSTALL_DIR="$LORRAX_SLATE_HOST_INSTALL_DIR"
    )
    echo "[build_host] SLATE group ON  ($LORRAX_SLATE_HOST_INSTALL_DIR)"
else
    ARGS+=(-DLORRAX_HOST_HAVE_SLATE=OFF)
    echo "[build_host] SLATE group OFF; ScaLAPACK/PBLAS remains ON."
fi

if [ "${1:-}" == "--fresh" ]; then rm -rf "$BUILD"; fi
mkdir -p "$BUILD"

echo "[build_host] configuring -> $BUILD ..."
"$CMAKE" "${ARGS[@]}"
echo "[build_host] compiling..."
"$CMAKE" --build "$BUILD" --parallel

SO="$BUILD/liblorrax_ffi_host.so"
[ -f "$SO" ] || { echo "[build_host] FAILED: no $SO" >&2; exit 1; }
echo "[build_host] --- artifact: $SO ---"; ls -lh "$SO"
echo "[build_host] --- libraries this .so will load at run time ---"
readelf -d "$SO" | grep NEEDED || true
if readelf -d "$SO" | grep NEEDED | grep -qiE 'cuda|nccl|nvshmem|cusolver|cublas'; then
    echo "[build_host] FAILED: host lib links a CUDA-stack library." >&2
    exit 1
fi
echo "[build_host] CUDA-free OK."

# Exported handler symbols must match ffi_loader._HOST_TARGET_SYMBOLS, or
# has_target() lies and the facade's capability guard passes wrongly.
#
# Grouped exactly as the CMakeLists groups them, so a partial build is
# checked for what it actually claimed to build and nothing else.  The
# ScaLAPACK group carries the two GEMM/FFT handlers because they ride its
# link line, not because they call ScaLAPACK.
WANT="PhdfReadHostFfi PhdfReadKchunkHostFfi PhdfReadKchunkUnionHostFfi \
PhdfWriteHostFfi ScalapackBatchedSolveLuHostFfi \
ScalapackBatchedGetrfHostFfi ScalapackBatchedGetrsHostFfi \
ScalapackEighHostFfi ScalapackBatchedGemmHostFfi \
MklFftFlatKHostFfi MklFftGwConvHostFfi"
if [ -n "$LORRAX_SLATE_HOST_INSTALL_DIR" ]; then
    # SLATE group.
    WANT="$WANT SlateEighHostFfi SlatePotrfHostFfi SlateTrsmHostFfi \
SlateBatchedPotrfHostFfi SlateBatchedTrsmHostFfi \
lrx_slate_init_mpi_host lrx_slate_context_create_host"
    # The ``_host`` suffix is not cosmetic and it is not optional.
    # cpp/slate/context.cc is CUDA-free and compiles into BOTH platform
    # libraries; until 2026-08-08 both exported these names unsuffixed, and
    # since both are dlopened RTLD_GLOBAL the first one loaded answered them
    # for the other library's internal calls too (lorrax
    # tests/KNOWN_FAILURES.md, L1).  cpp/common/c_abi.h's LRX_C_ENTRY
    # appends the leg; src/ffi/cpp/exports_host.map is what keeps everything
    # else off the dynamic table.  A host .so that still exports
    # ``lrx_slate_init_mpi`` predates the fix — this gate says so by name.
fi
echo "[build_host] --- exported handlers ---"
# Read the dynamic symbol table ONCE.  `nm -D "$SO" | grep -q ...` inside a
# `set -o pipefail` script reports MISSING for every symbol grep matches
# EARLY: grep -q exits at the first hit, nm dies of SIGPIPE, and pipefail
# propagates nm's 141.  (It only "worked" for the symbols that happen to sit
# at the end of the table.)
DYNSYMS="$(nm -D "$SO")"
MISS=0
for s in $WANT; do
    if grep -qE " (T|B|D) $s\$" <<< "$DYNSYMS"; then echo "  OK      $s"
    else echo "  MISSING $s"; MISS=1; fi
done
[ "$MISS" -eq 0 ] || { echo "[build_host] FAILED: missing handlers." >&2; exit 1; }
# The ScaLAPACK provider must be a dependency regardless of whether SLATE is
# also built.  A missing dependency otherwise survives link and fails at the
# first PBLAS/ScaLAPACK call.
echo "[build_host] --- ScaLAPACK provider recorded as a dependency ---"
readelf -d "$SO" | grep -E 'scalapack|blacs|libsci' \
    || { echo "[build_host] FAILED: no ScaLAPACK/BLACS library is recorded" >&2
         echo "             as a dependency of the .so — the" >&2
         echo "             -Wl,--no-as-needed link line did not take." >&2
         exit 1; }

# ===========================================================================
# THE BUILD CONTRACT.  Unconditional, and a failure here fails the build.
#
# scripts/verify_ffi_build.sh is the machine-agnostic form of the seven gates
# that used to exist only in config/perlmutter/build_ffi_host.sh.  Frontera had
# NONE of them: no one-MPI check, no one-BLAS-flavour check, no fftw-in-
# DT_NEEDED check, no HDF5 pairing check, no ABI stamp.  It had the CUDA-free
# check and the exported-handler list above, and those are two of ten.
#
# THE HANDLER LIST ABOVE IS NOW REDUNDANT and is left in place only because
# this branch could not run a Frontera build to prove its replacement green
# (TACC login needs interactive MFA).  GATE 0 checks the same property by
# pattern, and services/distrib_la/tests/test_so_acceptance.py checks the exact
# names out of the loader's own table — which is better, because that list is a
# THIRD copy of a table that already exists twice.  Delete `WANT` and its loop
# the first time somebody can run this script end to end.
#
# ScaLAPACK/PBLAS, vendor GEMM, phdf5 and FFT are the standard host contract;
# an explicitly named gpu_backend=none SLATE install adds only SLATE.
# ===========================================================================
_expect_backends="scalapack,gemm,phdf5,fft"
if [ -n "$LORRAX_SLATE_HOST_INSTALL_DIR" ]; then
    _expect_backends="scalapack,gemm,slate,phdf5,fft"
fi
echo "[build_host] --- build contract (scripts/verify_ffi_build.sh) ---"
LORRAX_FFI_EXPECT_BACKENDS="${LORRAX_FFI_EXPECT_BACKENDS:-$_expect_backends}" \
    bash "$LORRAX_ROOT/scripts/verify_ffi_build.sh" --leg host "$SO" || {
        echo "[build_host] FAILED: the library does not meet the build" >&2
        echo "[build_host] contract.  It must not be deployed or pinned." >&2
        exit 1
    }

# ---------------------------------------------------------------------------
# Stamp the build's identity NEXT TO the .so.
#
# Stage dirs are hand-named (build_host_ONE, build_host_PADFIX, ...) and a
# harness exports the path from a shell variable that a later line may
# override, so a log line naming a path has never established which BYTES ran.
# That cost real debugging time on 2026-08-02, when the certified
# build_host_ONE turned out to predate the tree it was being compared against
# (453 exported symbols vs 475) and nothing on disk said so.
#
# ffi.common.ffi_loader.library_provenance reads this file and the runtime
# startup report prints it, so every job log now records the build it loaded.
# Keep the format flat key=value: it is parsed by that function.
# ---------------------------------------------------------------------------
SO="$BUILD/liblorrax_ffi_host.so"
if [ -f "$SO" ]; then
    _rev=$(git -C "$LORRAX_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
    _dirty=no
    git -C "$LORRAX_ROOT" diff --quiet 2>/dev/null || _dirty=yes
    {
        echo "so=$SO"
        echo "sha256=$(sha256sum "$SO" | cut -d' ' -f1)"
        echo "bytes=$(stat -c %s "$SO")"
        echo "git_rev=$_rev"
        echo "git_dirty=$_dirty"
        echo "source_tree=$LORRAX_ROOT"
        echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "built_on=$(hostname)"
        echo "slate=$([ -n "$LORRAX_SLATE_HOST_INSTALL_DIR" ] && echo on || echo off)"
        echo "exported_symbols=$(nm -D --defined-only "$SO" 2>/dev/null | wc -l)"
        echo "hdf5_root=$LORRAX_HDF5_ROOT"
        echo "impi_root=$LORRAX_IMPI_ROOT"
    } > "$BUILD/PROVENANCE"
    echo "[build_host] stamped $BUILD/PROVENANCE"
    sed 's/^/[build_host]   /' "$BUILD/PROVENANCE"
    if [ "$_dirty" = yes ]; then
        echo "[build_host] WARNING: source tree had uncommitted changes; this" >&2
        echo "             .so does not correspond to any commit." >&2
    fi
fi
echo "[build_host] done."
