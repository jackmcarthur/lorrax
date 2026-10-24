#!/usr/bin/env bash
# ============================================================================
# build_ffi_host.sh — build liblorrax_ffi_host.so (the CUDA-free host-platform
# FFI library) on NERSC Perlmutter: plain g++ against Cray MPICH, LibSci, HDF5.
# Bare metal: no container or staged compiler; JAX headers come from the venv.
#
#   config/perlmutter/build_ffi_host.sh [--fresh]
#
# Output: <checkout>/src/ffi/cpp/build_host/liblorrax_ffi_host.so, where
#   ffi_loader.py looks by default (LORRAX_FFI_HOST_STAGE moves it).
# Needs: `uv sync` in this checkout (the venv's jaxlib supplies
#   the XLA FFI headers).
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
# Same three families as Frontera, different vendors.  Each is a published
# API, so they are swappable.
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
#                                        here and ffi.fft.require_kconv refuses at
#                                        startup.  See the FFT note below.
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
# The k-convolution router's cpu backend is these host FFT handlers, and
# ffi.fft.require_kconv() checks them at startup step 6b.  With no FFT provider
# the target lorrax_mklfft_flat_k does not exist, so a CPU-mesh run REFUSES at
# startup.
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
# handlers and nothing else, and ffi.fft.require_kconv refuses at startup naming both
# the unresolved symbol and every candidate it tried.  It is NOT a link-time
# dependency; GATE 5 (scripts/verify_ffi_build.sh) enforces that.  (When it
# was one, the Shifter container, which does not mount cray-fftw, could not
# dlopen the library, and nineteen ScaLAPACK/GEMM tests silently became skips.)
# ============================================================================

set -euo pipefail

# The checkout is the one this script lives in.  An exported LORRAX_ROOT is
# ignored on purpose: a shell that loaded a module setting it would otherwise
# build that module's release snapshot instead of this checkout.
LORRAX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LORRAX_VENV_PY="$LORRAX_ROOT/.venv/bin/python"
SRC="$LORRAX_ROOT/src/ffi/cpp"
BUILD="${LORRAX_FFI_HOST_STAGE:-$SRC/build_host}"
# shellcheck disable=SC1091
source "$LORRAX_ROOT/config/perlmutter/ffi_mpi.sh"

# ---------------------------------------------------------------------------
# Compiler and paths.  Plain g++ (gcc-native/14, the compiler PrgEnv-gnu's CC
# wraps) with every library named explicitly: config/perlmutter/ffi_site.cmake,
# through the perlmutter-host preset in src/ffi/cpp/CMakePresets.json.  The
# Cray CC wrapper is not used, so no loaded module adds to the link line: it
# used to add the sequential -lsci_gnu beside the threaded pair, libmpi_gtl_cuda
# with craype-accel-nvidia80 loaded, libdarshan with darshan loaded, and an
# unconditional -lxpmem, and cray-fftw's lib dir on its implicit path made
# CMake's FindOpenMP adopt libfftw3_omp as the OpenMP runtime.  The one flag of
# its that changes code, -march=znver3, is in the site file.
#
#   LORRAX_HOST_HAVE_SCALAPACK=OFF   GPU-only install: no ScaLAPACK backend
#                                    (a CUDA run never calls it; CPU-mesh
#                                    eigh/solve_lu = scalapack would refuse).
# ---------------------------------------------------------------------------
[[ -x "$LORRAX_VENV_PY" ]] || {
    echo "[build_ffi_host] ERROR: no venv at $LORRAX_ROOT/.venv." >&2
    echo "[build_ffi_host]   Run 'uv sync' in $LORRAX_ROOT first (add --extra cuda13 for GPU runs)." >&2
    exit 2
}
# CMakePresets.json needs cmake >= 3.23 (the system cmake is older).
if ! printf '3.23\n%s\n' "$(cmake --version 2>/dev/null | awk 'NR==1{print $3}')" | sort -VC; then
    # shellcheck disable=SC1091
    type module >/dev/null 2>&1 || source /opt/cray/pe/lmod/lmod/init/bash
    module load cmake
fi
SCALAPACK="${LORRAX_HOST_HAVE_SCALAPACK:-ON}"

# ---------------------------------------------------------------------------
# PRE-FLIGHT: the eleven ScaLAPACK/BLACS names must actually be in LibSci.
# A capability check, not a version check — this is what makes the script
# survive a LibSci upgrade or a move to netlib/AOCL.
# ---------------------------------------------------------------------------
if [[ "$SCALAPACK" == ON ]]; then
    SCALAPACK_SO="$LORRAX_PM_LIBSCI_ROOT/lib/libsci_gnu_mpi_mp.so"
    # Dump the symbol table ONCE.  Do not pipe nm into `grep -q` in a loop: under
    # `set -o pipefail`, grep -q exits on first match and closes the pipe, nm dies
    # of SIGPIPE, and the pipeline reports failure — which reads as "symbol
    # missing" for every symbol that IS present.  (Cost me one false gate trip.)
    _symtab="$(nm -D --defined-only "$SCALAPACK_SO" 2>/dev/null || true)"
    [[ -n "$_symtab" ]] || { echo "[build_ffi_host] ERROR: could not read symbols from $SCALAPACK_SO" >&2; exit 2; }
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
fi

# ---------------------------------------------------------------------------
# Configure + build.  The post-build step (cmake/LorraxPostBuild.cmake) writes
# the dependency closure (<so>.closure.json, which the loader opens and
# enforces) and runs scripts/verify_ffi_build.sh with Perlmutter's
# expectations from ffi_site.cmake, so the build fails if any gate does.
# ---------------------------------------------------------------------------
[[ "${1:-}" == "--fresh" ]] && { echo "[build_ffi_host] --fresh: wiping $BUILD"; rm -rf "${BUILD:?}"; }
GEN=(); command -v ninja >/dev/null 2>&1 && GEN=(-G Ninja)   # /usr/bin on the login nodes
(cd "$SRC" && cmake --preset perlmutter-host --fresh -B "$BUILD" "${GEN[@]}" \
    -DLORRAX_HOST_HAVE_SCALAPACK="$SCALAPACK")
cmake --build "$BUILD" --parallel "${LORRAX_BUILD_JOBS:-8}"

SO_FILE="$BUILD/liblorrax_ffi_host.so"
[[ -f "$SO_FILE" ]] || { echo "[build_ffi_host] FAILED: no .so produced." >&2; exit 1; }
readelf -d "$SO_FILE" | grep NEEDED

# GATE 9 STAYS HERE.  tests/KNOWN_FAILURES.md's L1 closure names THIS FILE and
# src/ffi/cpp/build.sh as the two build-time ratchets of the ODR fix, so the
# ratchet lives where the record says it lives.  The artifact-level twin is the
# acceptance tier's check 6 (services/distrib_la/tests/test_so_acceptance.py),
# which INTERSECTS the two libraries; a build script only ever sees one of
# them, so it checks the property that keeps that intersection empty.
#
# GATE 9: NO LORRAX-OWNED INTERNAL IS ON THE DYNAMIC TABLE.  Both platform
# .so's are dlopened RTLD_GLOBAL into one process, and ld.so answers a name
# from the FIRST object that defined it, for the whole process.  Measured
# 2026-08-07: 25 LORRAX names in common, among them
# `lorrax_ffi::phdf5::open_ctx`, whose `PhdfCtx` has a different layout in the
# two builds; one library's handler read the other's struct at the wrong
# offsets.  Checked: (a) nothing `lorrax_ffi`-namespaced is exported
# (exports_host.map; the build-config stamp is the one exception), (b) every
# exported `lrx_*` ctypes entry point carries the `_host` suffix (LRX_C_ENTRY).
echo "[build_ffi_host] GATE 9 (no LORRAX internal on the dynamic table)"
_dynsyms=$(nm -D --defined-only "$SO_FILE" 2>/dev/null | awk '{print $NF}')
leaked=$(grep -E 'lorrax_ffi' <<<"$_dynsyms" \
         | grep -vE '^lorrax_ffi_host_(build_config|abi_version)$' || true)
if [[ -n "${leaked//[[:space:]]/}" ]]; then
    echo "[build_ffi_host] GATE FAILED (9a): LORRAX-owned internals are on the dynamic table:" >&2
    printf '%s\n' "$leaked" | head -20 | sed 's/^/    /' >&2
    echo "[build_ffi_host]   Cause: -Wl,--version-script=exports_host.map is not on the link line." >&2
    exit 1
fi
bare_lrx=$(grep -E '^lrx_' <<<"$_dynsyms" | grep -vE '_host$' || true)
if [[ -n "${bare_lrx//[[:space:]]/}" ]]; then
    echo "[build_ffi_host] GATE FAILED (9b): unsuffixed lrx_* entry points" >&2
    printf '%s\n' "$bare_lrx" | sed 's/^/    /' >&2
    echo "[build_ffi_host]   they must go through cpp/common/c_abi.h's LRX_C_ENTRY." >&2
    exit 1
fi
echo "[build_ffi_host] GATE 9 PASSED: 0 lorrax_ffi internals exported, all $(grep -cE '^lrx_' <<<"$_dynsyms") lrx_* entry points leg-suffixed"
unset _dynsyms

# GATE 8 (FFT engine identity) is reported as NOT RUN by the verifier when
# shifter cannot bind-mount $HOME; run it inside an allocation:
#   LORRAX_GATE_FFTW_PY=<venv>/bin/python src/ffi/cpp/gate_one_fftw.sh <host.so>

"$LORRAX_VENV_PY" "$LORRAX_ROOT/scripts/dep_check.py" ffi "$SO_FILE" || {
    echo "[build_ffi_host] FAILED: the library resolves dependencies outside the install (above)" >&2
    exit 1; }

# Build provenance beside the artifact — see the note in src/ffi/cpp/build.sh.
_closure_sha="$("$LORRAX_VENV_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["record_sha256"])' "$SO_FILE.closure.json")"
"$LORRAX_ROOT/src/ffi/cpp/stage/stamp_provenance.sh" "$SO_FILE" \
    "leg=host" \
    "compiler=$(g++ --version 2>/dev/null | head -1)" \
    "mpi=$LORRAX_PM_MPICH_MODULE" \
    "mpi_root=$LORRAX_PM_MPICH_ROOT" \
    "libsci=${LORRAX_PM_LIBSCI_MODULE}_mp" \
    "scalapack=$SCALAPACK" \
    "hdf5=$LORRAX_PM_HDF5_MODULE" \
    "phdf5_stage=$LORRAX_PM_HDF5_ROOT" \
    "fftw=cray-fftw" \
    "closure_sha256=$_closure_sha" || {
        echo "[build_ffi_host] WARNING: provenance stamp failed" >&2
    }

echo
echo "[build_ffi_host] done.  .so at: $SO_FILE"
echo "[build_ffi_host] the loader opens and enforces $(basename "$SO_FILE").closure.json; seal_bundle.py makes the pair."
