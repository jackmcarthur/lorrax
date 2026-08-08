#!/usr/bin/env bash
# ============================================================================
# build_ffi_host.sh — build liblorrax_ffi_host.so (the CUDA-free host-platform
# FFI library) on NERSC Perlmutter, against the CRAY PROGRAMMING ENVIRONMENT.
# Bare metal: no container, no venv, no staged compiler.
#
#   config/perlmutter/build_ffi_host.sh [--fresh]
#
# Output: $LORRAX_FFI_HOST_STAGE/liblorrax_ffi_host.so
#   (default $LORRAX_ROOT/src/ffi/cpp/build_host, where ffi_loader.py looks;
#    override at runtime with LORRAX_FFI_HOST_SO).
#
# This is the Perlmutter twin of config/frontera/build_ffi_host.sh.  The two
# are deliberately STRUCTURALLY PARALLEL — same section headers, same variable
# names, same ordering — so `diff` between them shows only VALUES.  That
# diffability is the whole point of the per-machine config-file idiom
# (BerkeleyGW arch.mk, CP2K arch/<machine>.<toolchain>, QE make.inc).
#
# ----------------------------------------------------------------------------
# WHICH LIBRARY SUPPLIES WHAT — the Perlmutter column
# ----------------------------------------------------------------------------
# Same four families as Frontera, different vendors.  THREE OF THE FOUR are a
# published API, so they are swappable; only SLATE is single-implementation.
#
#   family     API LORRAX calls          Perlmutter provider
#   ---------  ------------------------  ---------------------------------
#   scalapack  ScaLAPACK + C-BLACS       Cray LibSci  libsci_gnu_mpi_mp
#              (the 11 Fortran-ABI       VERIFIED 2026-08-05: all ELEVEN
#              names in blacs_grid.h)    symbols present (pzheevd_ pdsyevd_
#                                        pzgetrf_ pdgetrf_ pzgetrs_ pdgetrs_
#                                        numroc_ descinit_ Csys2blacs_handle
#                                        Cblacs_gridinit Cblacs_gridinfo).
#   mklblas    CBLAS cblas_{d,z}gemm     Cray LibSci  libsci_gnu_mp
#                                        NOTE: LibSci does NOT export
#                                        cblas_?gemm_batch.  That entry is
#                                        dlsym'd at runtime, so its absence
#                                        selects the portable plain-GEMM loop
#                                        (~1.6-1.9x slower, announced on
#                                        first use).  Correctness unaffected.
#   mklfft     Intel DFTI descriptor API NOT AVAILABLE.  DFTI is Intel-only
#                                        and LibSci does not implement it, so
#                                        the flat-k FFT handlers are SKIPPED
#                                        here and LORRAX_FFT_FFI refuses at
#                                        startup.  See the FFT note below.
#   slate      slate:: C++ templates     ICL SLATE, gpu_backend=none, built
#                                        against LibSci + cray-mpich.
#
# ----------------------------------------------------------------------------
# WHY LibSci AND NOT MKL (MKL 2025.3 *does* exist at /opt/intel/oneapi/mkl)
# ----------------------------------------------------------------------------
# Three reasons, all about failing loudly instead of silently:
#
#   1. BLACS/MPI FLAVOUR.  MKL ships one BLACS per MPI (mkl_blacs_intelmpi_*
#      / mkl_blacs_openmpi_*), neither of which is cray-mpich.  A mismatch
#      LINKS CLEANLY and then hangs or aborts inside blacs_gridinit at the
#      first collective.  LibSci's libsci_gnu_mpi_mp carries
#      `NEEDED libmpi_gnu_123.so.12` in its own ELF header, so the pairing is
#      enforced by the dynamic linker rather than by a build-time choice
#      nobody re-checks.
#   2. INTEGER WIDTH.  LibSci's ScaLAPACK is LP64 ONLY — verified: zero
#      `_64`-suffixed ScaLAPACK symbols in libsci_gnu_mpi.so.  blacs_grid.h
#      declares every ScaLAPACK argument as `int`, so the classic ILP64
#      silent-descriptor-corruption hazard is STRUCTURALLY ABSENT on this
#      route.  MKL exports lp64 and ilp64 under the SAME unsuffixed symbol
#      names in different libraries; picking the wrong one corrupts silently.
#   3. ONE BLAS IN THE PROCESS.  Linking MKL for its FFT alongside LibSci for
#      ScaLAPACK would put two definitions of cblas_dgemm in one process and
#      let load order decide which one runs.
#
# The cost of this choice is the FFT handler (see below).  That is a lost
# optimisation, never a wrong answer.
#
# ----------------------------------------------------------------------------
# THE FFT GAP — read this before wondering why a CPU run refuses at startup
# ----------------------------------------------------------------------------
# src/ffi/fft.py declares LORRAX_FFT_FFI with default="on" and
# off_policy="refuse" (the native XLA flat-k arm was DELETED under the
# FFI-required ruling, decisions.md 2026-08-01), and
# runtime._enforce_required_ffi() calls Gate.enforce() at startup step 6b.
# With no DFTI provider the target lorrax_mklfft_flat_k does not exist, so a
# CPU-mesh run REFUSES at startup and `LORRAX_FFT_FFI=0` refuses too.
#
# THE FIX IS NOT MKL.  It is a second, vendor-neutral FFT backend against the
# standard FFTW3 advanced interface (fftw_plan_many_dft), which cray-fftw
# provides here (VERIFIED: fftw_plan_many_dft present in
# /opt/cray/pe/fftw/3.3.10.11/x86_milan/lib/libfftw3.so) and which FFTW,
# AOCL and MKL's own FFTW3 wrappers all implement.  The flat-k layout is one
# uniform batch (istride = howmany, idist = 1), which that call expresses
# directly.
#
# LANDED 2026-08-05 (commit d81507a), so the paragraph above is history: the
# DFTI source-lock is gone and lorrax_mklfft_flat_k is exported by this host
# library.  The engine is found at RUN time — dlsym for the entry points,
# dlopen for the library that defines them — so an absent FFTW3 costs the FFT
# handlers and nothing else, and LORRAX_FFT_FFI refuses at startup naming both
# the unresolved symbol and every candidate it tried.  It is NOT a link-time
# dependency; GATE 5 below enforces that, and the note at LORRAX_PM_FFTW
# explains what it cost when it was one.
# ============================================================================

set -euo pipefail

LORRAX_ROOT="${LORRAX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SRC="$LORRAX_ROOT/src/ffi/cpp"
BUILD="${LORRAX_FFI_HOST_STAGE:-$SRC/build_host}"

# ---------------------------------------------------------------------------
# Site constants.  Everything a port would need to change lives HERE.
# Frontera's twin sets LORRAX_MKL_ROOT / LORRAX_IMPI_ROOT in this slot.
# ---------------------------------------------------------------------------
LORRAX_PM_PRGENV="${LORRAX_PM_PRGENV:-PrgEnv-gnu}"
LORRAX_PM_LIBSCI="${LORRAX_PM_LIBSCI:-cray-libsci}"
LORRAX_PM_HDF5="${LORRAX_PM_HDF5:-cray-hdf5-parallel/1.14.3.7}"
# cray-fftw supplies the FFT engine.  The flat-k handlers call the FFTW3
# ADVANCED interface and resolve every entry point by RUNTIME dlsym, and the
# library that defines them is brought in by a RUNTIME dlopen ladder
# (src/ffi/cpp/mklfft/fft_flat_k_ffi.cc).  This module therefore exists only
# to tell the build WHERE that engine lives, so CMake can record it as the
# dlopen hint.  NOTHING FROM IT REACHES THE LINK LINE -- see GATE 5 below.
#
# WHAT THIS USED TO SAY, AND WHY IT WAS WRONG (corrected 2026-08-06)
# ------------------------------------------------------------------
# "this module exists only to put libfftw3 into the .so's DT_NEEDED for that
# dlsym to find".  It did exactly that, and a DT_NEEDED entry is a LOAD-TIME
# dependency: `libfftw3.so.mpi31.3` had to be findable before any code in the
# library ran.  cray-fftw's SONAME is version- and MPI-flavour-stamped, and
# the Shifter container does not bind-mount /opt/cray/pe/fftw at all, so the
# .so simply would not dlopen in-container.  Measured 2026-08-06: nineteen
# ScaLAPACK/SLATE/GEMM contract tests -- none of which perform an FFT --
# turned into SKIPS, and the suite reported 0 failures.  A lost FFT
# optimisation silently became a lost linear-algebra test suite.
#
# On an MKL site this module is unnecessary -- MKL exports the FFTW3 C
# interface natively from libmkl_intel_lp64 and the ladder never reaches
# dlopen.  With no engine reachable at all, LORRAX_FFT_FFI refuses at STARTUP
# naming the unresolved symbol and every non-FFT handler still works.
LORRAX_PM_FFTW="${LORRAX_PM_FFTW:-cray-fftw}"
LORRAX_PM_CMAKE="${LORRAX_PM_CMAKE:-cmake}"
# The CONTAINER stage this bare-metal .so has to agree with — GATE 7.
#
# This leg is built on the login node against the cray-hdf5-parallel
# MODULE; the device leg is built inside Shifter against whatever is
# bind-mounted at /lorrax_phdf5.  Two populations, two HDF5s, and until
# 2026-08-06 nothing compared them: the module moved to 1.14.3.7 while the
# stage stayed on 1.12, so this .so carried a SONAME the container did not
# have and would not dlopen there at all (CLAIMS 89).  The path is read
# from config/perlmutter/site_config.sh so there is ONE source of truth for
# it — the modulefile's --volume= source and this gate must not be able to
# name different trees.
if [[ -z "${LORRAX_FFI_PHDF5_DIR:-}" && -r "$(dirname "$0")/site_config.sh" ]]; then
    # shellcheck disable=SC1091
    LORRAX_FFI_PHDF5_DIR="$(. "$(dirname "$0")/site_config.sh" >/dev/null 2>&1;
                            printf %s "$LORRAX_FFI_PHDF5_DIR_DEFAULT")"
fi
LORRAX_PM_PHDF5_STAGE="${LORRAX_FFI_PHDF5_DIR:-}"
# LibSci threading flavour.  MUST match the SLATE install's, or the process
# ends up with both libsci_gnu_mpi and libsci_gnu_mpi_mp loaded and ELF load
# order silently decides which BLAS/ScaLAPACK runs.  The gpu_backend=none
# SLATE here was built threaded, so: _mp.
LORRAX_PM_LIBSCI_FLAVOUR="${LORRAX_PM_LIBSCI_FLAVOUR:-_mp}"
LORRAX_SLATE_HOST_INSTALL_DIR="${LORRAX_SLATE_HOST_INSTALL_DIR:-$HOME/software/slate_builds/cpu/install}"
# XLA FFI headers must match the RUNTIME jaxlib, not whatever python is first
# on PATH.  src/ffi/cpp/build_host.sh stages these out of the Shifter image;
# reuse that stage.
LORRAX_XLA_FFI_HEADERS_DIR="${LORRAX_XLA_FFI_HEADERS_DIR:-$HOME/software/lorrax_xla_ffi_headers/25.04-py3}"

# ---------------------------------------------------------------------------
# Modules.  craype-accel-nvidia80 / cudatoolkit are UNLOADED on purpose: with
# them loaded the CC wrapper links libmpi_gtl_cuda, whose libcuda.so.1
# dependency defeats the point of a CUDA-free host library.
# ---------------------------------------------------------------------------
if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash
fi
module load "$LORRAX_PM_PRGENV" "$LORRAX_PM_LIBSCI" "$LORRAX_PM_HDF5" "$LORRAX_PM_FFTW" "$LORRAX_PM_CMAKE"
module unload craype-accel-nvidia80 2>/dev/null || true
module unload cudatoolkit           2>/dev/null || true

: "${CRAY_LIBSCI_PREFIX_DIR:?cray-libsci did not set CRAY_LIBSCI_PREFIX_DIR}"
: "${CRAY_MPICH_DIR:?cray-mpich did not set CRAY_MPICH_DIR}"
: "${HDF5_DIR:?cray-hdf5-parallel did not set HDF5_DIR}"

# Capture the prefixes, THEN unload cray-libsci.
#
# WHY: the Cray CC wrapper auto-injects `-lsci_gnu` — the SEQUENTIAL LibSci —
# whenever the cray-libsci module is loaded.  Our own link line asks for the
# THREADED `_mp` pair (to match the gpu_backend=none SLATE, which links
# threaded), so leaving the module loaded puts BOTH flavours in the .so.  That
# is not a theoretical concern: the Cray runtime itself detects it and prints
#     [CRAYBLAS_WARNING] Application linked against multiple cray-libsci libraries
# on the first BLAS call.  ELF order makes it deterministic (the _mp pair comes
# first and wins), but "deterministic if you re-derive the link order" is not a
# property to rely on.  Unloading the module leaves the wrapper adding no
# libsci at all, and the explicit -L/-l below is then the only source.
LORRAX_PM_LIBSCI_DIR="$CRAY_LIBSCI_PREFIX_DIR"
LORRAX_PM_MPICH_DIR="$CRAY_MPICH_DIR"
LORRAX_PM_HDF5_DIR="$HDF5_DIR"
module unload cray-libsci 2>/dev/null || true

# Capture the FFTW3 prefix, THEN unload cray-fftw — the SAME hazard as
# cray-libsci above, one level more insidious.
#
# WHY: with cray-fftw loaded, its lib dir is on the CC wrapper's IMPLICIT
# link path, and CMake's FindOpenMP probes for an OpenMP runtime by link name.
# It matched cray-fftw's `libfftw3f_omp` / `libfftw3_omp` and adopted them as
# the OpenMP runtime for the WHOLE target.  Measured in the 2026-08-05 build's
# CMakeCache.txt:
#     OpenMP_CXX_LIB_NAMES:STRING=fftw3f_omp;fftw3_omp;gomp
# So the .so acquired THREE fftw DT_NEEDED entries, and only one of them came
# from the FFT leg at all — the other two were an OpenMP misdetection that
# nothing in the build reported.  Unloading the module leaves FindOpenMP with
# plain -fopenmp/-lgomp, and the FFTW3 location reaches CMake only through the
# explicit -DLORRAX_FFTW3_LIBRARY below, where it is recorded as a runtime
# dlopen hint and never linked.  Exactly the libsci idiom: capture, unload,
# hand the value over explicitly.
LORRAX_PM_FFTW_LIB=""
for _d in "${FFTW_DIR:-}" "${FFTW_ROOT:-}/lib"; do
    if [[ -n "$_d" && -e "$_d/libfftw3.so" ]]; then
        LORRAX_PM_FFTW_LIB="$_d/libfftw3.so"
        break
    fi
done
if [[ -z "$LORRAX_PM_FFTW_LIB" ]]; then
    echo "[build_ffi_host] NOTE: no libfftw3.so found via FFTW_DIR/FFTW_ROOT." >&2
    echo "[build_ffi_host]   The FFT handlers still build; the runtime dlopen" >&2
    echo "[build_ffi_host]   ladder will fall back to portable SONAMEs, and" >&2
    echo "[build_ffi_host]   LORRAX_FFT_FFI refuses at startup if none loads." >&2
else
    echo "[build_ffi_host] fftw3 runtime dlopen hint: $LORRAX_PM_FFTW_LIB"
fi
module unload cray-fftw 2>/dev/null || true

# Unloading the module also unsets CRAY_LIBSCI_PREFIX_DIR, which the
# CMakeLists' CBLAS probe reads from the ENVIRONMENT:
#     find_path(LORRAX_CBLAS_INCLUDE_DIR cblas.h
#               HINTS "$ENV{CRAY_LIBSCI_PREFIX_DIR}/include"
#                     "$ENV{LORRAX_CBLAS_DIR}/include" ...)
# Without a hint it finds no cblas.h, silently drops the vendor-BLAS GEMM
# handler, and LORRAX_BANDS_GEMM_FFI then refuses at runtime.  Hand it the
# documented site-neutral hint instead of re-exporting a Cray-private name.
export LORRAX_CBLAS_DIR="$LORRAX_PM_LIBSCI_DIR"

if [[ ! -f "$LORRAX_XLA_FFI_HEADERS_DIR/xla/ffi/api/ffi.h" ]]; then
    echo "[build_ffi_host] ERROR: XLA FFI headers not staged at" >&2
    echo "[build_ffi_host]   $LORRAX_XLA_FFI_HEADERS_DIR" >&2
    echo "[build_ffi_host] Stage them once with: bash src/ffi/cpp/build_host.sh" >&2
    exit 2
fi
if [[ ! -d "$LORRAX_SLATE_HOST_INSTALL_DIR/lib64" ]]; then
    echo "[build_ffi_host] ERROR: gpu_backend=none SLATE not found at" >&2
    echo "[build_ffi_host]   $LORRAX_SLATE_HOST_INSTALL_DIR" >&2
    echo "[build_ffi_host] Build it with: bash src/ffi/cpp/stage/slate_build_perlmutter.sh cpu" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# PRE-FLIGHT: the eleven ScaLAPACK/BLACS names must actually be in LibSci.
# A capability check, not a version check — this is what makes the script
# survive a LibSci upgrade or a move to netlib/AOCL.
# ---------------------------------------------------------------------------
SCALAPACK_SO="$LORRAX_PM_LIBSCI_DIR/lib/libsci_gnu_mpi${LORRAX_PM_LIBSCI_FLAVOUR}.so"
# Dump the symbol table ONCE.  Do not pipe nm into `grep -q` in a loop: under
# `set -o pipefail`, grep -q exits on first match and closes the pipe, nm dies
# of SIGPIPE, and the pipeline reports failure — which reads as "symbol
# missing" for every symbol that IS present.  (Cost me one false gate trip.)
_symtab="$(nm -D --defined-only "$SCALAPACK_SO" 2>/dev/null || true)"
if [[ -z "$_symtab" ]]; then
    echo "[build_ffi_host] ERROR: could not read symbols from $SCALAPACK_SO" >&2
    exit 2
fi
missing=""
for sym in pzheevd_ pdsyevd_ pzgetrf_ pdgetrf_ pzgetrs_ pdgetrs_ \
           numroc_ descinit_ Csys2blacs_handle Cblacs_gridinit Cblacs_gridinfo; do
    grep -qE "[[:space:]]${sym}\$" <<<"$_symtab" || missing="$missing $sym"
done
unset _symtab
if [[ -n "$missing" ]]; then
    echo "[build_ffi_host] ERROR: $SCALAPACK_SO is missing:$missing" >&2
    echo "[build_ffi_host] Pass an explicit -DLORRAX_SCALAPACK_LIBRARIES instead." >&2
    exit 2
fi
echo "[build_ffi_host] pre-flight: all 11 ScaLAPACK/BLACS symbols present in $(basename "$SCALAPACK_SO")"

# ---------------------------------------------------------------------------
# Configure + build.
#
# Why an explicit -DLORRAX_SCALAPACK_LIBRARIES rather than letting CMake find
# it: the CMakeLists' built-in probe searches for an MKL layout
# (libmkl_scalapack_lp64 under $MKLROOT), which does not describe LibSci.  The
# explicit link line is the CMakeLists' own documented vendor-agnostic escape
# hatch.  A FindScaLAPACK.cmake that understands both layouts would be the
# better long-term shape; noted, not built here.
#
# Explicit MPI/HDF5 paths are REQUIRED even though the Cray CC wrapper puts
# them on its implicit -I/-L line, because the CMakeLists probes with
# find_path(... NO_DEFAULT_PATH), which cannot see wrapper-implicit paths.
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--fresh" ]]; then
    echo "[build_ffi_host] --fresh: wiping $BUILD"
    rm -rf "${BUILD:?}"
fi
mkdir -p "$BUILD"
cd "$BUILD"

F="$LORRAX_PM_LIBSCI_FLAVOUR"
FFTW_ARGS=()
if [[ -n "$LORRAX_PM_FFTW_LIB" ]]; then
    FFTW_ARGS=(-DLORRAX_FFTW3_LIBRARY="$LORRAX_PM_FFTW_LIB")
fi
cmake "$SRC" \
    "${FFTW_ARGS[@]}" \
    -DLORRAX_FFI_PLATFORM=host \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=CC \
    -DLORRAX_XLA_FFI_INCLUDE_DIR="$LORRAX_XLA_FFI_HEADERS_DIR" \
    -DLORRAX_SLATE_HOST_INSTALL_DIR="$LORRAX_SLATE_HOST_INSTALL_DIR" \
    -DLORRAX_SCALAPACK_LIBRARIES="-L$LORRAX_PM_LIBSCI_DIR/lib -Wl,--no-as-needed -lsci_gnu_mpi${F} -lsci_gnu${F} -lgomp -lpthread -lm -ldl" \
    -DLORRAX_CBLAS_INCLUDE_DIR="$LORRAX_PM_LIBSCI_DIR/include" \
    -DLORRAX_CBLAS_LIBRARY="$LORRAX_PM_LIBSCI_DIR/lib/libsci_gnu${F}.so" \
    -DLORRAX_MPI_INCLUDE_DIR="$LORRAX_PM_MPICH_DIR/include" \
    -DLORRAX_MPICH_LIB_DIR="$LORRAX_PM_MPICH_DIR/lib" \
    -DHDF5_ROOT="$LORRAX_PM_HDF5_DIR"

cmake --build . --parallel "${LORRAX_BUILD_JOBS:-8}"

SO_FILE="$BUILD/liblorrax_ffi_host.so"
[[ -f "$SO_FILE" ]] || { echo "[build_ffi_host] FAILED: no .so produced." >&2; exit 1; }

# ===========================================================================
# POST-LINK GATES.  All three are HARD FAILURES, not warnings — each one
# guards a defect that links cleanly and only shows up as a hang or a wrong
# number much later.
# ===========================================================================
echo
echo "[build_ffi_host] --- post-link gates ---"
readelf -d "$SO_FILE" | grep NEEDED

# GATE 1 (hazard S3): EXACTLY ONE cray-mpich ABI variant, and it must be the
# one LibSci and SLATE carry.  Perlmutter keeps several side by side:
# libmpi_gnu_91 is cray-mpich 8.1.25 (2023) and libmpi_gnu_123 is 9.0.1.  Two
# libmpi in one process means MPI_COMM_WORLD and the MPI_Comm handle passed
# to Csys2blacs_handle mean different things in different frames.  It links
# fine and corrupts or hangs at the first collective.
# NOTE 2026-08-05: this gate USED to test `readelf -d` (DIRECT DT_NEEDED
# only) and therefore passed binaries that pulled a second libmpi in
# transitively through libhdf5_parallel_gnu_123.  It now resolves the FULL
# closure with ldd and dedupes by the resolved FILE, not the name string.
# See src/ffi/cpp/gate_one_mpi.sh for why both changes are load-bearing.
GATE_TAG=build_ffi_host "$SRC/gate_one_mpi.sh" "$SO_FILE" libmpi_gnu_123 || exit 1

# GATE 2 (hazard S8): ONE LibSci threading flavour.  Mixing libsci_gnu_mpi
# with libsci_gnu_mpi_mp puts a sequential and an OpenMP BLAS/ScaLAPACK in the
# same process and lets ELF load order pick.  Deterministic, but nobody
# re-derives the order after a link-line edit.
sci_seq=$(readelf -d "$SO_FILE" | grep NEEDED | grep -cE 'libsci_gnu(_mpi)?\.so' || true)
sci_mp=$(readelf -d "$SO_FILE"  | grep NEEDED | grep -cE 'libsci_gnu(_mpi)?_mp\.so' || true)
if [[ "$sci_seq" -gt 0 && "$sci_mp" -gt 0 ]]; then
    echo "[build_ffi_host] GATE FAILED (S8): BOTH LibSci flavours linked" >&2
    echo "[build_ffi_host]   (sequential=$sci_seq threaded=$sci_mp).  The Cray runtime" >&2
    echo "[build_ffi_host]   reports this as [CRAYBLAS_WARNING] Application linked" >&2
    echo "[build_ffi_host]   against multiple cray-libsci libraries.  Usually means the" >&2
    echo "[build_ffi_host]   cray-libsci module was still loaded at configure time and" >&2
    echo "[build_ffi_host]   the CC wrapper auto-injected the sequential libsci." >&2
    exit 1
fi
if [[ "$F" == "_mp" && "$sci_mp" -eq 0 ]]; then
    echo "[build_ffi_host] GATE FAILED (S8): asked for threaded LibSci, none linked." >&2
    exit 1
fi
echo "[build_ffi_host] GATE 2 (S8, one LibSci flavour '$F') PASSED: sequential=$sci_seq threaded=$sci_mp"

# GATE 3: CUDA-free by construction.
if readelf -d "$SO_FILE" | grep NEEDED | grep -qiE 'cuda|nccl|nvshmem|cal\.so'; then
    echo "[build_ffi_host] GATE FAILED: host lib links a CUDA-stack library." >&2
    exit 1
fi
echo "[build_ffi_host] GATE 3 (CUDA-free) PASSED"

# GATE 5: THE RUN-TIME-RESOLVED FFT ENGINE IS NOT A LOAD-TIME DEPENDENCY.
#
# This gate exists because its absence cost a test suite.  The FFT backend is
# documented as resolved at RUN time, and the structural proof recorded for
# that (CLAIMS 55) was
#     nm -D --undefined-only liblorrax_ffi_host.so | grep -c fftw_   ->  0
# which was TRUE on 2026-08-05 and TRUE on 2026-08-06 and detected nothing,
# because it is the wrong measurement.  It counts undefined SYMBOL references,
# and dlsym'ing every entry point drives that to zero BY CONSTRUCTION — it is
# what you would measure to confirm the dlsym rewrite compiled, not to confirm
# the library is unbound.  Meanwhile the build was passing -lfftw3 (plus two
# more fftw libraries via a FindOpenMP misdetection), so the .so carried
#     NEEDED libfftw3.so.mpi31.3
#     NEEDED libfftw3f_omp.so.mpi31.3
#     NEEDED libfftw3_omp.so.mpi31.3
# and DT_NEEDED is resolved by the dynamic linker BEFORE any of this library's
# code runs.  Result, measured in-container 2026-08-06: dlopen of the host
# library failed outright and 19 ScaLAPACK/SLATE/GEMM contract tests that
# never call an FFT reported as SKIPPED, with the suite green at 0 failures.
#
# So: keep the symbol check (it is still the right proof that the CALLS are
# dlsym'd), and add the check that actually guards loadability.  Both, because
# neither implies the other.
echo "[build_ffi_host] GATE 5 (FFT engine is not a load-time dependency)"
fftw_undef=$(nm -D --undefined-only "$SO_FILE" 2>/dev/null | grep -c 'fftw_' || true)
if [[ "$fftw_undef" -ne 0 ]]; then
    echo "[build_ffi_host] GATE FAILED (5a): $fftw_undef undefined fftw_ symbols." >&2
    echo "[build_ffi_host]   The flat-k TU must resolve every FFTW3 entry point" >&2
    echo "[build_ffi_host]   through mklpin::resolve_sym, never by direct call." >&2
    exit 1
fi
fftw_needed=$(readelf -d "$SO_FILE" | grep NEEDED | grep -c 'fftw' || true)
if [[ "$fftw_needed" -ne 0 ]]; then
    echo "[build_ffi_host] GATE FAILED (5b): $fftw_needed fftw entries in DT_NEEDED:" >&2
    readelf -d "$SO_FILE" | grep NEEDED | grep 'fftw' >&2
    echo "[build_ffi_host]   An FFT engine resolved at RUN time must not be a" >&2
    echo "[build_ffi_host]   LOAD-time dependency.  Any fftw in DT_NEEDED makes" >&2
    echo "[build_ffi_host]   the WHOLE library — ScaLAPACK, SLATE, GEMM, phdf5" >&2
    echo "[build_ffi_host]   handlers included — unloadable wherever that exact" >&2
    echo "[build_ffi_host]   SONAME is absent, e.g. inside the Shifter container," >&2
    echo "[build_ffi_host]   which does not bind-mount /opt/cray/pe/fftw." >&2
    echo "[build_ffi_host]   Two known causes:" >&2
    echo "[build_ffi_host]     1. an -lfftw3 on the link line (the CMakeLists" >&2
    echo "[build_ffi_host]        records the path as a dlopen hint instead);" >&2
    echo "[build_ffi_host]     2. FindOpenMP adopting cray-fftw's libfftw3*_omp" >&2
    echo "[build_ffi_host]        as the OpenMP runtime — check that this script" >&2
    echo "[build_ffi_host]        unloaded cray-fftw before invoking cmake, and" >&2
    echo "[build_ffi_host]        check OpenMP_CXX_LIB_NAMES in CMakeCache.txt." >&2
    exit 1
fi
echo "[build_ffi_host] GATE 5 PASSED: undefined fftw_ symbols=0, fftw in DT_NEEDED=0"
# GATE 5 IS HALF THE PROPERTY, AND THIS SCRIPT CANNOT CHECK THE OTHER HALF.
# Zero fftw in DT_NEEDED means nothing binds at LOAD time.  It says nothing
# about which engine — or how many — the process ends up with, because after
# GATE 5 the engine arrives by dlopen at first use, and no static tool can see
# it: `ldd` on this artifact reports nothing about FFTW3 at all.  Measured
# 2026-08-06, this exact .so, one build: the bare host maps cray-fftw and the
# container maps NOTHING (the image ships no FFTW3) — a difference GATE 5
# cannot express, and both pass it.
#
# src/ffi/cpp/gate_one_fftw.sh (GATE 8) closes it by driving one real flat-k
# FFT and reading /proc/self/maps in that process.  It is NOT called from here
# on purpose: this script runs bare-metal on a login node, where the dynamic
# leg cannot import jax and would refuse every build.  Run it where the
# process lives:
#
#   in-container:  LORRAX_FFTW3_STAGE=/lorrax_fftw \
#                    src/ffi/cpp/gate_one_fftw.sh <host.so> [<device.so>]
#   bare host:     LORRAX_GATE_FFTW_PY=<venv>/bin/python \
#                    src/ffi/cpp/gate_one_fftw.sh <host.so>
echo "[build_ffi_host] GATE 5 covers LOAD time only — run GATE 8"
echo "[build_ffi_host]   (src/ffi/cpp/gate_one_fftw.sh) where the process"
echo "[build_ffi_host]   runs, or the engine identity is unchecked."

# GATE 6: and more generally, the OpenMP runtime is an OpenMP runtime.  The
# misdetection above was invisible because nothing ever looked.  gomp (GNU) or
# iomp5/omp (LLVM/Intel) are the legitimate answers; anything else means
# FindOpenMP matched on the substring "omp" in a library that is not one.
omp_needed=$(readelf -d "$SO_FILE" | grep NEEDED \
             | grep -oE 'lib[a-z0-9_]*omp[a-z0-9_]*\.so[^]]*' || true)
bad_omp=$(grep -vE '^lib(gomp|iomp5|omp)\.so' <<<"$omp_needed" || true)
if [[ -n "${bad_omp//[[:space:]]/}" ]]; then
    echo "[build_ffi_host] GATE FAILED (6): non-OpenMP library adopted as the" >&2
    echo "[build_ffi_host]   OpenMP runtime:" >&2
    echo "$bad_omp" >&2
    echo "[build_ffi_host]   Check OpenMP_CXX_LIB_NAMES in $BUILD/CMakeCache.txt." >&2
    exit 1
fi
echo "[build_ffi_host] GATE 6 (OpenMP runtime is really OpenMP) PASSED"

# GATE 7: ONE HDF5, and the container stage provides it.
#
# GATE 5 removed the fftw SONAME from DT_NEEDED and left the host leg with
# exactly one remaining in-container `not found`: libhdf5_parallel_gnu.so.310,
# because this script loads cray-hdf5-parallel/1.14.3.7 (SOVERSION 310) and
# the stage bind-mounted at /lorrax_phdf5 held HDF5 1.12 (SOVERSION 200).
# Same shape as GATE 5 one library over: a dependency this leg cannot see is
# missing, because it is only ever missing somewhere this leg does not run.
# So the check has to be made HERE, against the OTHER population.
#
# It is deliberately not just "does the SONAME resolve".  See
# src/ffi/cpp/gate_one_hdf5.sh for why the two-HDF5s-in-one-process repair
# is worse than the failure it fixes.
if [[ -z "$LORRAX_PM_PHDF5_STAGE" ]]; then
    echo "[build_ffi_host] GATE 7 CANNOT RUN: no phdf5 stage named." >&2
    echo "[build_ffi_host]   Neither LORRAX_FFI_PHDF5_DIR nor" >&2
    echo "[build_ffi_host]   LORRAX_FFI_PHDF5_DIR_DEFAULT in site_config.sh" >&2
    echo "[build_ffi_host]   is set, so the HDF5 this .so links cannot be" >&2
    echo "[build_ffi_host]   compared against the one the container will" >&2
    echo "[build_ffi_host]   provide.  That comparison is the whole gate." >&2
    exit 1
fi
GATE_TAG=build_ffi_host LORRAX_PHDF5_STAGE="$LORRAX_PM_PHDF5_STAGE" \
    "$SRC/gate_one_hdf5.sh" "$SO_FILE" || exit 1

# GATE 9: THE EXPORTED SURFACE IS THE SANCTIONED ONE, AND NOTHING ELSE.
#
# This gate is the build-time half of the KNOWN_FAILURES L1 fix (the test-time
# half is services/distrib_la/tests/test_so_acceptance.py check 6, which
# INTERSECTS the two libraries; a build script only ever sees one of them, so
# it checks the property that makes the intersection empty instead).
#
# WHAT WENT WRONG.  Both platform .so's are dlopened RTLD_GLOBAL into one
# process, and ld.so answers a name from the FIRST object that defined it —
# for the whole process, including for the OTHER library's own internal calls.
# Measured 2026-08-07 on the pinned pair: 259 defined names in common, 25 of
# them LORRAX's own.  Among them `lorrax_ffi::phdf5::open_ctx`, whose return
# type `PhdfCtx` has a DIFFERENT LAYOUT in the two builds (the CUDA
# stream/event/pinned members compile out here).  One library's handler read
# the other's struct at the wrong offsets:
#     offset_base=[0,0,0,4596944070643295330]     <- a float64 read as int64
#
# src/ffi/cpp/exports_host.map now localises everything except the three
# sanctioned patterns.  This gate is what notices the day someone drops the
# version script from the link line — the symptom otherwise is a wrong number
# in a mixed process, weeks later.
echo "[build_ffi_host] GATE 9 (exported surface is the sanctioned one)"
stray=$(nm -D --defined-only "$SO_FILE" 2>/dev/null | awk '{print $NF}' \
        | grep -vE 'HostFfi$|^lrx_[a-z0-9_]+_host$|^lorrax_ffi_host_build_config$' \
        || true)
if [[ -n "${stray//[[:space:]]/}" ]]; then
    echo "[build_ffi_host] GATE FAILED (9): the host library exports symbols" >&2
    echo "[build_ffi_host]   outside its sanctioned surface (*HostFfi handlers," >&2
    echo "[build_ffi_host]   lrx_*_host ctypes entry points, and the build-config" >&2
    echo "[build_ffi_host]   stamp).  Anything else can be answered by — or can" >&2
    echo "[build_ffi_host]   answer for — liblorrax_ffi.so in a process that has" >&2
    echo "[build_ffi_host]   both open.  Offenders:" >&2
    printf '%s\n' "$stray" | sed 's/^/    /' >&2
    echo "[build_ffi_host]   Usual cause: -Wl,--version-script=exports_host.map" >&2
    echo "[build_ffi_host]   is not on the link line (check CMakeCache.txt and" >&2
    echo "[build_ffi_host]   src/ffi/cpp/CMakeLists.txt).  If a NEW symbol"       >&2
    echo "[build_ffi_host]   genuinely has to be exported, add it to that map"    >&2
    echo "[build_ffi_host]   deliberately — that file is the whole surface."      >&2
    exit 1
fi
echo "[build_ffi_host] GATE 9 PASSED: $(nm -D --defined-only "$SO_FILE" | wc -l) exported symbols, all sanctioned"

# GATE 4: nothing left unresolved at load time.  -Wl,--no-undefined already
# fails the LINK on a missing link-time symbol; this catches the other half —
# a NEEDED library that cannot itself be found at run time.
if command -v ldd >/dev/null 2>&1; then
    LD_LIBRARY_PATH="$LORRAX_SLATE_HOST_INSTALL_DIR/lib64:${LD_LIBRARY_PATH:-}" \
        ldd -r "$SO_FILE" 2>&1 | grep -iE 'undefined|not found' && {
            echo "[build_ffi_host] GATE FAILED: unresolved symbols at load time." >&2
            exit 1
        }
    echo "[build_ffi_host] GATE 4 (load-time resolution) PASSED"
fi

# Build provenance beside the artifact — see the note in src/ffi/cpp/build.sh.
"$LORRAX_ROOT/src/ffi/cpp/stage/stamp_provenance.sh" "$SO_FILE" \
    "leg=host" \
    "prgenv=$LORRAX_PM_PRGENV" \
    "libsci=$LORRAX_PM_LIBSCI$LORRAX_PM_LIBSCI_FLAVOUR" \
    "hdf5=$LORRAX_PM_HDF5" \
    "phdf5_stage=$LORRAX_PM_PHDF5_STAGE" \
    "fftw=$LORRAX_PM_FFTW" \
    "slate=$LORRAX_SLATE_HOST_INSTALL_DIR" || {
        echo "[build_ffi_host] WARNING: provenance stamp failed" >&2
    }

echo
echo "[build_ffi_host] done.  .so at: $SO_FILE"
echo "[build_ffi_host] run with: export LORRAX_FFI_HOST_SO=$SO_FILE"
echo "[build_ffi_host]           export LD_LIBRARY_PATH=$LORRAX_SLATE_HOST_INSTALL_DIR/lib64:\$LD_LIBRARY_PATH"
