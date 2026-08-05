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
# directly.  NOT YET IMPLEMENTED — tracked separately; until it lands, this
# host lib supports the ScaLAPACK/SLATE/GEMM/phdf5 targets and CPU chain
# drivers refuse at startup.
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
LORRAX_PM_CMAKE="${LORRAX_PM_CMAKE:-cmake}"
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
module load "$LORRAX_PM_PRGENV" "$LORRAX_PM_LIBSCI" "$LORRAX_PM_HDF5" "$LORRAX_PM_CMAKE"
module unload craype-accel-nvidia80 2>/dev/null || true
module unload cudatoolkit           2>/dev/null || true

: "${CRAY_LIBSCI_PREFIX_DIR:?cray-libsci did not set CRAY_LIBSCI_PREFIX_DIR}"
: "${CRAY_MPICH_DIR:?cray-mpich did not set CRAY_MPICH_DIR}"
: "${HDF5_DIR:?cray-hdf5-parallel did not set HDF5_DIR}"

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
SCALAPACK_SO="$CRAY_LIBSCI_PREFIX_DIR/lib/libsci_gnu_mpi${LORRAX_PM_LIBSCI_FLAVOUR}.so"
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
cmake "$SRC" \
    -DLORRAX_FFI_PLATFORM=host \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=CC \
    -DLORRAX_XLA_FFI_INCLUDE_DIR="$LORRAX_XLA_FFI_HEADERS_DIR" \
    -DLORRAX_SLATE_HOST_INSTALL_DIR="$LORRAX_SLATE_HOST_INSTALL_DIR" \
    -DLORRAX_SCALAPACK_LIBRARIES="-L$CRAY_LIBSCI_PREFIX_DIR/lib -Wl,--no-as-needed -lsci_gnu_mpi${F} -lsci_gnu${F} -lgomp -lpthread -lm -ldl" \
    -DLORRAX_MPI_INCLUDE_DIR="$CRAY_MPICH_DIR/include" \
    -DLORRAX_MPICH_LIB_DIR="$CRAY_MPICH_DIR/lib" \
    -DHDF5_ROOT="$HDF5_DIR"

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
mpi_variants=$(readelf -d "$SO_FILE" | grep NEEDED | grep -oE 'libmpi_gnu_[0-9]+' | sort -u)
mpi_count=$(printf '%s\n' "$mpi_variants" | grep -c . || true)
if [[ "$mpi_count" -ne 1 ]]; then
    echo "[build_ffi_host] GATE FAILED (S3): expected exactly ONE libmpi_gnu_*," >&2
    echo "[build_ffi_host]   found $mpi_count: $mpi_variants" >&2
    exit 1
fi
if [[ "$mpi_variants" != "libmpi_gnu_123" ]]; then
    echo "[build_ffi_host] GATE FAILED (S3): linked $mpi_variants, expected" >&2
    echo "[build_ffi_host]   libmpi_gnu_123 (cray-mpich 9.0.1 — what LibSci and" >&2
    echo "[build_ffi_host]   the SLATE install carry).  Check the loaded modules." >&2
    exit 1
fi
echo "[build_ffi_host] GATE 1 (S3, one cray-mpich ABI) PASSED: $mpi_variants"

# GATE 2 (hazard S8): ONE LibSci threading flavour.  Mixing libsci_gnu_mpi
# with libsci_gnu_mpi_mp puts a sequential and an OpenMP BLAS/ScaLAPACK in the
# same process and lets ELF load order pick.  Deterministic, but nobody
# re-derives the order after a link-line edit.
sci_seq=$(readelf -d "$SO_FILE" | grep NEEDED | grep -cE 'libsci_gnu(_mpi)?\.so' || true)
sci_mp=$(readelf -d "$SO_FILE"  | grep NEEDED | grep -cE 'libsci_gnu(_mpi)?_mp\.so' || true)
if [[ "$F" == "_mp" && "$sci_mp" -eq 0 ]]; then
    echo "[build_ffi_host] GATE FAILED (S8): asked for threaded LibSci, none linked." >&2
    exit 1
fi
echo "[build_ffi_host] GATE 2 (S8, LibSci flavour) : sequential=$sci_seq threaded=$sci_mp (want flavour '$F')"

# GATE 3: CUDA-free by construction.
if readelf -d "$SO_FILE" | grep NEEDED | grep -qiE 'cuda|nccl|nvshmem|cal\.so'; then
    echo "[build_ffi_host] GATE FAILED: host lib links a CUDA-stack library." >&2
    exit 1
fi
echo "[build_ffi_host] GATE 3 (CUDA-free) PASSED"

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

echo
echo "[build_ffi_host] done.  .so at: $SO_FILE"
echo "[build_ffi_host] run with: export LORRAX_FFI_HOST_SO=$SO_FILE"
echo "[build_ffi_host]           export LD_LIBRARY_PATH=$LORRAX_SLATE_HOST_INSTALL_DIR/lib64:\$LD_LIBRARY_PATH"
