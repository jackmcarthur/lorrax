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
#              (the 13 Fortran-ABI       verified: all thirteen
#              names in blacs_grid.h)    symbols present (pzheevd_ pdsyevd_
#                                        pzgetrf_ pdgetrf_ pzgetrs_ pdgetrs_
#                                        pzgemm_ pdgemm_
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
#   slate      slate:: C++ templates     not built by this route; independent
#                                        of the LibSci ScaLAPACK/PBLAS route.
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
# dependency; GATE 5 (scripts/verify_ffi_build.sh) enforces that, and the
# note at LORRAX_PM_FFTW
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
# dlopen hint.  NOTHING FROM IT REACHES THE LINK LINE -- see GATE 5 in
# scripts/verify_ffi_build.sh.
#
# WHAT THIS USED TO SAY, AND WHY IT WAS WRONG (corrected 2026-08-06)
# ------------------------------------------------------------------
# "this module exists only to put libfftw3 into the .so's DT_NEEDED for that
# dlsym to find".  It did exactly that, and a DT_NEEDED entry is a LOAD-TIME
# dependency: `libfftw3.so.mpi31.3` had to be findable before any code in the
# library ran.  cray-fftw's SONAME is version- and MPI-flavour-stamped, and
# the Shifter container does not bind-mount /opt/cray/pe/fftw at all, so the
# .so simply would not dlopen in-container.  Measured 2026-08-06: nineteen
# ScaLAPACK/GEMM contract tests -- none of which perform an FFT --
# turned into SKIPS, and the suite reported 0 failures.  A lost FFT
# optimisation silently became a lost linear-algebra test suite.
#
# On an MKL site this module is unnecessary -- MKL exports the FFTW3 C
# interface natively from libmkl_intel_lp64 and the ladder never reaches
# dlopen.  With no engine reachable at all, LORRAX_FFT_FFI refuses at STARTUP
# naming the unresolved symbol and every non-FFT handler still works.
LORRAX_PM_FFTW="${LORRAX_PM_FFTW:-cray-fftw}"
LORRAX_PM_CMAKE="${LORRAX_PM_CMAKE:-cmake}"
# The CONTAINER stage this bare-metal .so has to agree with — GATE 7 in
# scripts/verify_ffi_build.sh.
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
# LibSci threading flavour.  Use one explicit flavour for both the distributed
# and local BLAS dependencies; the wrapper module is unloaded below so it
# cannot inject the sequential twin as well.
LORRAX_PM_LIBSCI_FLAVOUR="${LORRAX_PM_LIBSCI_FLAVOUR:-_mp}"
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
module unload darshan               2>/dev/null || true

: "${CRAY_LIBSCI_PREFIX_DIR:?cray-libsci did not set CRAY_LIBSCI_PREFIX_DIR}"
: "${CRAY_MPICH_DIR:?cray-mpich did not set CRAY_MPICH_DIR}"
: "${HDF5_DIR:?cray-hdf5-parallel did not set HDF5_DIR}"

# Capture the prefixes, THEN unload cray-libsci.
#
# WHY: the Cray CC wrapper auto-injects `-lsci_gnu` — the SEQUENTIAL LibSci —
# whenever the cray-libsci module is loaded.  Our own link line asks for the
# THREADED `_mp` pair, so leaving the module loaded puts BOTH flavours in the
# .so.  That is not a theoretical concern: the Cray runtime itself detects it
# and prints
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
# ---------------------------------------------------------------------------
# PRE-FLIGHT: the thirteen ScaLAPACK/BLACS names must actually be in LibSci.
# A capability check, not a version check — this is what makes the script
# survive a LibSci upgrade or a move to netlib/AOCL.
# ---------------------------------------------------------------------------
SCALAPACK_SO="$LORRAX_PM_LIBSCI_DIR/lib/libsci_gnu_mpi${LORRAX_PM_LIBSCI_FLAVOUR}.so"
bash "$LORRAX_ROOT/src/ffi/cpp/scalapack/check_symbol_contract.sh" \
    provider "$SCALAPACK_SO"

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
    -DLORRAX_HOST_HAVE_SCALAPACK=ON \
    -DLORRAX_HOST_HAVE_SLATE=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=CC \
    -DLORRAX_XLA_FFI_INCLUDE_DIR="$LORRAX_XLA_FFI_HEADERS_DIR" \
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
# POST-LINK GATES — DELEGATED.
#
# Every gate that used to be written out here now lives in
# scripts/verify_ffi_build.sh, and so does every argument for why each
# measurement is the right one: the ldd-not-readelf reasoning, the
# dedupe-by-realpath reasoning, the "nm -D | grep -c fftw_ read 0 while the
# library was unloadable" history, the FindOpenMP misdetection, the
# two-HDF5s-in-one-process explanation.  They were the crown jewels of this
# script and they are not diminished by moving — they are now enforced on
# FIVE build paths instead of one, which is the whole point.  Nothing was
# dropped; read that file for the full text.
#
# WHAT STAYS HERE IS WHAT IS PERLMUTTER'S: the EXPECTATIONS.  A gate that
# infers its own expectation from the artifact cannot fail, so the site states
# them and the machine-agnostic gates check them.
#
# ONE GATE ALSO STAYS HERE IN FULL — GATE 9, the export-table ratchet, below.
# It is not Perlmutter's, but tests/KNOWN_FAILURES.md's L1 closure names this
# file and src/ffi/cpp/build.sh as its two build-time ratchets, and a ratchet
# that has moved out from under its own record is not a ratchet.  The
# artifact-level enforcement is the acceptance tier (check 6 and the
# sanctioned-surface cell), which sees BOTH libraries at once.
# ===========================================================================
echo
echo "[build_ffi_host] --- post-link gates (scripts/verify_ffi_build.sh) ---"
readelf -d "$SO_FILE" | grep NEEDED

# GATE 7 needs a phdf5 stage to compare against, and on this machine there is
# no honest way to run without one.  The host leg is built bare-metal against
# the cray-hdf5-parallel MODULE; the device leg is built in-container against
# whatever is bind-mounted at /lorrax_phdf5.  Two populations, two HDF5s, and
# until 2026-08-06 nothing compared them: the module moved to 1.14.3.7 while
# the stage stayed on 1.12, so this .so carried a SONAME the container did not
# have and would not dlopen there at all (CLAIMS 89).
if [[ -z "$LORRAX_PM_PHDF5_STAGE" ]]; then
    echo "[build_ffi_host] GATE 7 CANNOT RUN: no phdf5 stage named." >&2
    echo "[build_ffi_host]   Neither LORRAX_FFI_PHDF5_DIR nor" >&2
    echo "[build_ffi_host]   LORRAX_FFI_PHDF5_DIR_DEFAULT in site_config.sh" >&2
    echo "[build_ffi_host]   is set, so the HDF5 this .so links cannot be" >&2
    echo "[build_ffi_host]   compared against the one the container will" >&2
    echo "[build_ffi_host]   provide.  That comparison is the whole gate." >&2
    exit 1
fi

# THE SOVERSION THE RUNTIME WILL MOUNT, read from the stage that will mount it.
#
# THIS IS THE TRAP THAT COST THE KCHUNK BRANCH A BUILD, stated as a gate.
# LORRAX_FFI_PHDF5_DIR does NOT choose the HDF5 this leg links — the Cray
# compiler wrappers do, via the module named in LORRAX_PM_HDF5.  Set only the
# former and you get a .so.310 library that agrees with nothing, links
# cleanly, and is structurally unloadable beside a .so.200 device library.
# Deriving the expectation from the STAGE and comparing it against what the
# MODULE produced is what makes that disagreement fail here instead of at
# somebody's first read_slabs.
_stage_sov=""
for _f in "$LORRAX_PM_PHDF5_STAGE"/lib/libhdf5*.so*; do
    [[ -e "$_f" && ! -L "$_f" ]] || continue
    _s="$(readelf -d "$_f" 2>/dev/null | sed -n 's/.*SONAME.*\[\(.*\)\].*/\1/p')"
    _s="${_s##*.}"
    [[ -n "$_s" ]] && _stage_sov="$_s"
done
if [[ -n "$_stage_sov" ]]; then
    echo "[build_ffi_host] phdf5 stage provides HDF5 SOVERSION $_stage_sov"
else
    echo "[build_ffi_host] NOTE: could not read a SOVERSION out of" >&2
    echo "[build_ffi_host]   $LORRAX_PM_PHDF5_STAGE/lib — GATE 7 falls back to" >&2
    echo "[build_ffi_host]   the stage-provides comparison alone." >&2
fi

# GATE 9 STAYS HERE, and it is the one exception to the delegation above.
# tests/KNOWN_FAILURES.md's L1 closure names THIS FILE and src/ffi/cpp/build.sh
# as the two build-time ratchets of the ODR fix, so the ratchet lives where the
# record says it lives.  The artifact-level twin is the acceptance tier's
# check 6, which INTERSECTS the two libraries; a build script only ever sees
# one of them, so it checks the property that keeps that intersection empty.
#
# GATE 9: NO LORRAX-OWNED INTERNAL IS ON THE DYNAMIC TABLE.
#
# The build-time half of the KNOWN_FAILURES L1 fix.  The test-time half is
# services/distrib_la/tests/test_so_acceptance.py check 6, which INTERSECTS
# the two libraries; a build script only ever sees one of them, so it checks
# the property that makes the LORRAX part of that intersection empty.
#
# WHAT WENT WRONG.  Both platform .so's are dlopened RTLD_GLOBAL into one
# process, and ld.so answers a name from the FIRST object that defined it --
# for the whole process, including for the OTHER library's own internal
# calls.  Measured 2026-08-07 on the pinned pair: 259 defined names in
# common, 25 of them LORRAX's own.  Among them `lorrax_ffi::phdf5::open_ctx`,
# whose `PhdfCtx` return type has a DIFFERENT LAYOUT in the two builds (the
# CUDA stream/event/pinned members compile out here).  One library's handler
# read the other's struct at the wrong offsets:
#     offset_base=[0,0,0,4596944070643295330]     <- a float64 read as int64
#
# TWO THINGS ARE CHECKED, because the fix has two halves:
#   (a) nothing `lorrax_ffi`-namespaced is exported -- src/ffi/cpp/
#       exports_host.map localises it (the build-config stamp is the one
#       deliberate exception and is named here as such);
#   (b) every exported `lrx_*` ctypes entry point carries the `_host` leg
#       suffix -- cpp/common/c_abi.h's LRX_C_ENTRY.  Those nine cannot be
#       hidden (Python dlsyms them), so renaming is what stops them
#       colliding.
#
# This gate is what notices the day the version script falls off the link
# line or LRX_C_ENTRY stops being applied.  The symptom otherwise is a wrong
# number in a mixed process, weeks later.
echo "[build_ffi_host] GATE 9 (no LORRAX internal on the dynamic table)"
_dynsyms=$(nm -D --defined-only "$SO_FILE" 2>/dev/null | awk '{print $NF}')
leaked=$(grep -E 'lorrax_ffi' <<<"$_dynsyms" \
         | grep -vE '^lorrax_ffi_host_(build_config|abi_version)$' || true)
if [[ -n "${leaked//[[:space:]]/}" ]]; then
    echo "[build_ffi_host] GATE FAILED (9a): LORRAX-owned internals are on" >&2
    echo "[build_ffi_host]   the dynamic table.  liblorrax_ffi.so defines" >&2
    echo "[build_ffi_host]   these names too, and in a process with both" >&2
    echo "[build_ffi_host]   open the first one loaded answers them for" >&2
    echo "[build_ffi_host]   BOTH -- including for this library's own calls." >&2
    printf '%s\n' "$leaked" | head -20 | sed 's/^/    /' >&2
    echo "[build_ffi_host]   Cause: -Wl,--version-script=exports_host.map is" >&2
    echo "[build_ffi_host]   not on the link line (check CMakeCache.txt)." >&2
    exit 1
fi
bare_lrx=$(grep -E '^lrx_' <<<"$_dynsyms" | grep -vE '_host$' || true)
if [[ -n "${bare_lrx//[[:space:]]/}" ]]; then
    echo "[build_ffi_host] GATE FAILED (9b): unsuffixed lrx_* entry points" >&2
    printf '%s\n' "$bare_lrx" | sed 's/^/    /' >&2
    echo "[build_ffi_host]   cpp/phdf5/api.cc and cpp/slate/context.cc"    >&2
    echo "[build_ffi_host]   compile into BOTH libraries; their entry"      >&2
    echo "[build_ffi_host]   points must go through cpp/common/c_abi.h's"   >&2
    echo "[build_ffi_host]   LRX_C_ENTRY so this leg's carry '_host'."      >&2
    exit 1
fi
echo "[build_ffi_host] GATE 9 PASSED: 0 lorrax_ffi internals exported, all $(grep -cE '^lrx_' <<<"$_dynsyms") lrx_* entry points leg-suffixed"
unset _dynsyms

# GATE 4 (load-time resolution) used to be written out here; it is one of the
# gates that moved into scripts/verify_ffi_build.sh.

# The Perlmutter host recipe builds its four required host backends and keeps
# the independent SLATE host backend off.  Declaring the required set turns
# "ScaLAPACK/BLACS not found" from a CMake WARNING into a build failure — the
# Aug-7 deployed library shipped with scalapack=0 because nothing anywhere
# said what it was supposed to contain.
LORRAX_FFI_EXPECT_BACKENDS="${LORRAX_FFI_EXPECT_BACKENDS:-scalapack,gemm,phdf5,fft}" \
LORRAX_FFI_EXPECT_MPI="${LORRAX_FFI_EXPECT_MPI:-libmpi_gnu}" \
LORRAX_FFI_EXPECT_HDF5_SOVERSION="${LORRAX_FFI_EXPECT_HDF5_SOVERSION:-$_stage_sov}" \
LORRAX_PHDF5_STAGE="$LORRAX_PM_PHDF5_STAGE" \
GATE_TAG=build_ffi_host \
    bash "$LORRAX_ROOT/scripts/verify_ffi_build.sh" --leg host "$SO_FILE" || {
        echo "[build_ffi_host] FAILED: the library does not meet the build" >&2
        echo "[build_ffi_host] contract.  The .so is left on disk so it can be" >&2
        echo "[build_ffi_host] inspected, but it must not be deployed or" >&2
        echo "[build_ffi_host] pinned.  Every failure above names its own fix." >&2
        exit 1
    }

# GATE 8 (FFT engine identity) is reported as NOT RUN by the verifier on a
# login node, and that is honest rather than a gap: it drives a real flat-k
# FFT and reads /proc/self/maps, and shifter cannot bind-mount $HOME on a
# login node.  Run it inside an allocation, where the process lives:
#
#   in-container:  LORRAX_FFTW3_STAGE=/lorrax_fftw \
#                    src/ffi/cpp/gate_one_fftw.sh <host.so> [<device.so>]
#   bare host:     LORRAX_GATE_FFTW_PY=<venv>/bin/python \
#                    src/ffi/cpp/gate_one_fftw.sh <host.so>
#
# or re-run this verifier there with LORRAX_GATE_FFTW_PY set, which is what
# the acceptance pytest tier does.

# Build provenance beside the artifact — see the note in src/ffi/cpp/build.sh.
"$LORRAX_ROOT/src/ffi/cpp/stage/stamp_provenance.sh" "$SO_FILE" \
    "leg=host" \
    "prgenv=$LORRAX_PM_PRGENV" \
    "libsci=$LORRAX_PM_LIBSCI$LORRAX_PM_LIBSCI_FLAVOUR" \
    "hdf5=$LORRAX_PM_HDF5" \
    "phdf5_stage=$LORRAX_PM_PHDF5_STAGE" \
    "fftw=$LORRAX_PM_FFTW" \
    "slate=off" || {
        echo "[build_ffi_host] WARNING: provenance stamp failed" >&2
    }

echo
echo "[build_ffi_host] done.  .so at: $SO_FILE"
echo "[build_ffi_host] run with: export LORRAX_FFI_HOST_SO=$SO_FILE"
