#!/usr/bin/env bash
# ============================================================================
# build_ffi_host.sh — build liblorrax_ffi_host.so (the CUDA-free, ROCm-free
# host-platform FFI library) on OLCF Frontier, against the CRAY PROGRAMMING
# ENVIRONMENT.  Bare metal: no container, no staged compiler.
#
#   config/frontier/build_ffi_host.sh [--fresh]
#
# Output: <checkout>/src/ffi/cpp/build_host/liblorrax_ffi_host.so
#   (LORRAX_FFI_HOST_STAGE moves it).  Pin it with LORRAX_FFI_HOST_SO; the
#   run environment is config/frontier/ffi_env.sh.
# Needs:
#   * the venv that will LOAD the library (its jaxlib supplies the XLA FFI
#     headers): $LORRAX_OLCF_VENV, default $HOME/envs/lx-jax091-rocm7141.
#
# This is the Frontier twin of config/perlmutter/build_ffi_host.sh, written
# section for section in parallel so `diff` shows values.  The long arguments
# for each step (why LibSci not MKL, why capture-then-unload cray-libsci and
# cray-fftw, GATE 7 and GATE 9 history) live in that file and in
# scripts/verify_ffi_build.sh; they apply here unchanged because Frontier and
# Perlmutter run the same Cray PE.
#
# ----------------------------------------------------------------------------
# WHICH LIBRARY SUPPLIES WHAT — the Frontier column
# ----------------------------------------------------------------------------
#   family     API LORRAX calls          Frontier provider
#   ---------  ------------------------  ---------------------------------
#   scalapack  ScaLAPACK + C-BLACS       Cray LibSci 25.09.0 libsci_gnu_mpi_mp
#                                        (all 11 names checked at pre-flight)
#   gemm       CBLAS cblas_{d,z}gemm     Cray LibSci libsci_gnu_mp (no
#                                        cblas_?gemm_batch: the portable
#                                        loop is used, announced at first use)
#   fft        FFTW3 advanced interface  cray-fftw 3.3.10.11, found at RUN time
#                                        by dlopen; never DT_NEEDED (GATE 5)
#   phdf5      parallel HDF5             cray-hdf5-parallel 1.14.3.7 (.so.310)
#   MPI        one libmpi                cray-mpich 9.0.1, libmpi_gnu_123.so.12
# The versions and the reasons for them are in config/frontier/ffi_mpi.sh.
#
# ROCm: craype-accel-amd-gfx90a and rocm are UNLOADED.  With the accel module
# loaded the CC wrapper links libmpi_gtl_hsa (and through it libamdhip64),
# which a host-only library must not carry (GATE 3's intent, for AMD).
# ============================================================================

set -euo pipefail
ulimit -c 0

LORRAX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$LORRAX_ROOT/src/ffi/cpp"
BUILD="${LORRAX_FFI_HOST_STAGE:-$SRC/build_host}"

# ---------------------------------------------------------------------------
# Site constants.  Everything a port would need to change lives HERE and in
# config/frontier/ffi_mpi.sh.
# ---------------------------------------------------------------------------
# shellcheck disable=SC1091
source "$LORRAX_ROOT/config/frontier/ffi_mpi.sh"
LORRAX_OLCF_PRGENV="${LORRAX_OLCF_PRGENV:-$LORRAX_OLCF_PRGENV_MODULE}"
LORRAX_OLCF_LIBSCI="${LORRAX_OLCF_LIBSCI:-$LORRAX_OLCF_LIBSCI_MODULE}"
LORRAX_OLCF_HDF5="${LORRAX_OLCF_HDF5:-$LORRAX_OLCF_HDF5_MODULE}"
LORRAX_OLCF_FFTW="${LORRAX_OLCF_FFTW:-$LORRAX_OLCF_FFTW_MODULE}"
LORRAX_OLCF_CMAKE="${LORRAX_OLCF_CMAKE:-$LORRAX_OLCF_CMAKE_MODULE}"
# GATE 7's have-side.  Bare host, both legs load the module's HDF5, so the
# stage defaults to that module's HDF5_DIR (set below, once it is loaded).
LORRAX_OLCF_PHDF5_STAGE="${LORRAX_FFI_PHDF5_DIR:-}"
# LibSci threading flavour: ONE per process (the threaded _mp pair).
LORRAX_OLCF_LIBSCI_FLAVOUR="${LORRAX_OLCF_LIBSCI_FLAVOUR:-_mp}"
LORRAX_OLCF_VENV="${LORRAX_OLCF_VENV:-$HOME/envs/lx-jax091-rocm7141}"
LORRAX_VENV_PY="$LORRAX_OLCF_VENV/bin/python"
# XLA FFI headers must match the RUNTIME jaxlib.
if [[ -z "${LORRAX_XLA_FFI_HEADERS_DIR:-}" ]]; then
    [[ -x "$LORRAX_VENV_PY" ]] || {
        echo "[build_ffi_host] ERROR: no python at $LORRAX_VENV_PY (set LORRAX_OLCF_VENV)." >&2
        exit 2
    }
    LORRAX_XLA_FFI_HEADERS_DIR="$("$LORRAX_VENV_PY" -c 'import jax.ffi; print(jax.ffi.include_dir())')"
fi

# ---------------------------------------------------------------------------
# Modules.  PrgEnv first (it resets cray-mpich), then the pinned MPI, then the
# families built against it.  GPU modules UNLOADED (see header).
# ---------------------------------------------------------------------------
if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /opt/cray/pe/lmod/lmod/init/bash
fi
module load "$LORRAX_OLCF_PRGENV"
lorrax_olcf_pin_mpi
module load "$LORRAX_OLCF_LIBSCI" "$LORRAX_OLCF_HDF5" "$LORRAX_OLCF_FFTW" "$LORRAX_OLCF_CMAKE"
module unload craype-accel-amd-gfx90a 2>/dev/null || true
module unload rocm amd-mixed           2>/dev/null || true
[[ "$CRAY_MPICH_DIR" == "$LORRAX_OLCF_MPICH_ROOT" ]] || {
    echo "[build_ffi_host] ERROR: cray-mpich resolved to $CRAY_MPICH_DIR, not the" >&2
    echo "[build_ffi_host]   pinned $LORRAX_OLCF_MPICH_ROOT (config/frontier/ffi_mpi.sh)." >&2
    exit 2
}
: "${CRAY_LIBSCI_PREFIX_DIR:?cray-libsci did not set CRAY_LIBSCI_PREFIX_DIR}"
: "${HDF5_DIR:?cray-hdf5-parallel did not set HDF5_DIR}"
[[ "$CRAY_LIBSCI_PREFIX_DIR" == "$LORRAX_OLCF_LIBSCI_ROOT" && "$HDF5_DIR" == "$LORRAX_OLCF_HDF5_ROOT" ]] || {
    echo "[build_ffi_host] ERROR: LibSci $CRAY_LIBSCI_PREFIX_DIR / HDF5 $HDF5_DIR are not the" >&2
    echo "[build_ffi_host]   run-time pins in config/frontier/ffi_mpi.sh; change that file." >&2
    exit 2
}

# Capture the prefixes, THEN unload cray-libsci (else the wrapper injects the
# sequential -lsci_gnu beside our threaded _mp pair: two LibSci flavours).
LORRAX_OLCF_LIBSCI_DIR="$CRAY_LIBSCI_PREFIX_DIR"
LORRAX_OLCF_MPICH_DIR="$CRAY_MPICH_DIR"
LORRAX_OLCF_HDF5_DIR="$HDF5_DIR"
LORRAX_OLCF_PHDF5_STAGE="${LORRAX_OLCF_PHDF5_STAGE:-$HDF5_DIR}"
module unload cray-libsci 2>/dev/null || true

# Capture the FFTW3 library, THEN unload cray-fftw (else FindOpenMP adopts
# libfftw3_omp as the OpenMP runtime and fftw lands in DT_NEEDED: GATE 5).
# It reaches CMake only as -DLORRAX_FFTW3_LIBRARY, a runtime dlopen hint.
LORRAX_OLCF_FFTW_LIB=""
for _d in "${FFTW_DIR:-}" "${FFTW_ROOT:-}/lib"; do
    if [[ -n "$_d" && -e "$_d/libfftw3.so" ]]; then
        LORRAX_OLCF_FFTW_LIB="$_d/libfftw3.so"
        break
    fi
done
if [[ -z "$LORRAX_OLCF_FFTW_LIB" ]]; then
    echo "[build_ffi_host] NOTE: no libfftw3.so via FFTW_DIR/FFTW_ROOT; the FFT handlers" >&2
    echo "[build_ffi_host]   still build and the runtime ladder falls back to portable SONAMEs." >&2
else
    echo "[build_ffi_host] fftw3 runtime dlopen hint: $LORRAX_OLCF_FFTW_LIB"
fi
module unload cray-fftw 2>/dev/null || true

# The CMakeLists' CBLAS probe reads a hint from the environment.
export LORRAX_CBLAS_DIR="$LORRAX_OLCF_LIBSCI_DIR"

if [[ ! -f "$LORRAX_XLA_FFI_HEADERS_DIR/xla/ffi/api/ffi.h" ]]; then
    echo "[build_ffi_host] ERROR: XLA FFI headers not at $LORRAX_XLA_FFI_HEADERS_DIR" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# PRE-FLIGHT 1: the eleven ScaLAPACK/BLACS names must be in LibSci.
# ---------------------------------------------------------------------------
SCALAPACK_SO="$LORRAX_OLCF_LIBSCI_DIR/lib/libsci_gnu_mpi${LORRAX_OLCF_LIBSCI_FLAVOUR}.so"
_symtab="$(nm -D --defined-only "$SCALAPACK_SO" 2>/dev/null || true)"
[[ -n "$_symtab" ]] || { echo "[build_ffi_host] ERROR: cannot read $SCALAPACK_SO" >&2; exit 2; }
missing=""
for sym in pzheevd_ pdsyevd_ pzgetrf_ pdgetrf_ pzgetrs_ pdgetrs_ \
           numroc_ descinit_ Csys2blacs_handle Cblacs_gridinit Cblacs_gridinfo; do
    grep -qE "[[:space:]]${sym}\$" <<<"$_symtab" || missing="$missing $sym"
done
unset _symtab
if [[ -n "$missing" ]]; then
    echo "[build_ffi_host] ERROR: $SCALAPACK_SO is missing:$missing" >&2
    exit 2
fi
echo "[build_ffi_host] pre-flight: all 11 ScaLAPACK/BLACS symbols present in $(basename "$SCALAPACK_SO")"

# ---------------------------------------------------------------------------
# Configure + build.  Explicit ScaLAPACK link line (the CMake probe expects an
# MKL layout) and explicit MPI/HDF5 paths (the CMakeLists probes with
# NO_DEFAULT_PATH, which cannot see wrapper-implicit paths).
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--fresh" ]]; then
    echo "[build_ffi_host] --fresh: wiping $BUILD"
    rm -rf "${BUILD:?}"
fi
mkdir -p "$BUILD"
cd "$BUILD"

F="$LORRAX_OLCF_LIBSCI_FLAVOUR"
FFTW_ARGS=()
if [[ -n "$LORRAX_OLCF_FFTW_LIB" ]]; then
    FFTW_ARGS=(-DLORRAX_FFTW3_LIBRARY="$LORRAX_OLCF_FFTW_LIB")
fi
cmake "$SRC" \
    "${FFTW_ARGS[@]}" \
    -DLORRAX_FFI_PLATFORM=host \
    -DLORRAX_FFI_POST_BUILD=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=CC \
    -DLORRAX_XLA_FFI_INCLUDE_DIR="$LORRAX_XLA_FFI_HEADERS_DIR" \
    -DLORRAX_SCALAPACK_LIBRARIES="-L$LORRAX_OLCF_LIBSCI_DIR/lib -Wl,--no-as-needed -lsci_gnu_mpi${F} -lsci_gnu${F} -lgomp -lpthread -lm -ldl" \
    -DLORRAX_CBLAS_INCLUDE_DIR="$LORRAX_OLCF_LIBSCI_DIR/include" \
    -DLORRAX_CBLAS_LIBRARY="$LORRAX_OLCF_LIBSCI_DIR/lib/libsci_gnu${F}.so" \
    -DLORRAX_MPI_INCLUDE_DIR="$LORRAX_OLCF_MPICH_DIR/include" \
    -DLORRAX_MPICH_LIB_DIR="$LORRAX_OLCF_MPICH_DIR/lib" \
    -DLORRAX_MPI_LIBRARY="$LORRAX_OLCF_MPI_LIBRARY" \
    -DHDF5_ROOT="$LORRAX_OLCF_HDF5_DIR"

cmake --build . --parallel "${LORRAX_BUILD_JOBS:-16}"

SO_FILE="$BUILD/liblorrax_ffi_host.so"
[[ -f "$SO_FILE" ]] || { echo "[build_ffi_host] FAILED: no .so produced." >&2; exit 1; }

# ===========================================================================
# POST-LINK GATES — delegated to scripts/verify_ffi_build.sh; what stays here
# is Frontier's EXPECTATIONS, plus GATE 9 (the export-table ratchet), which
# the Perlmutter twin carries in full for the reason given there.
# ===========================================================================
echo
echo "[build_ffi_host] --- post-link gates (scripts/verify_ffi_build.sh) ---"
readelf -d "$SO_FILE" | grep NEEDED

_stage_sov=""
for _f in "$LORRAX_OLCF_PHDF5_STAGE"/lib/libhdf5*.so*; do
    [[ -e "$_f" && ! -L "$_f" ]] || continue
    _s="$(readelf -d "$_f" 2>/dev/null | sed -n 's/.*SONAME.*\[\(.*\)\].*/\1/p')"
    _s="${_s##*.}"
    [[ -n "$_s" ]] && _stage_sov="$_s"
done
if [[ -n "$_stage_sov" ]]; then
    echo "[build_ffi_host] phdf5 stage provides HDF5 SOVERSION $_stage_sov"
else
    echo "[build_ffi_host] NOTE: no SOVERSION readable from $LORRAX_OLCF_PHDF5_STAGE/lib" >&2
fi

echo "[build_ffi_host] GATE 9 (no LORRAX internal on the dynamic table)"
_dynsyms=$(nm -D --defined-only "$SO_FILE" 2>/dev/null | awk '{print $NF}')
leaked=$(grep -E 'lorrax_ffi' <<<"$_dynsyms" \
         | grep -vE '^lorrax_ffi_host_(build_config|abi_version)$' || true)
if [[ -n "${leaked//[[:space:]]/}" ]]; then
    echo "[build_ffi_host] GATE FAILED (9a): LORRAX-owned internals are exported:" >&2
    printf '%s\n' "$leaked" | head -20 | sed 's/^/    /' >&2
    exit 1
fi
bare_lrx=$(grep -E '^lrx_' <<<"$_dynsyms" | grep -vE '_host$' || true)
if [[ -n "${bare_lrx//[[:space:]]/}" ]]; then
    echo "[build_ffi_host] GATE FAILED (9b): unsuffixed lrx_* entry points" >&2
    printf '%s\n' "$bare_lrx" | sed 's/^/    /' >&2
    exit 1
fi
echo "[build_ffi_host] GATE 9 PASSED: 0 lorrax_ffi internals exported, all $(grep -cE '^lrx_' <<<"$_dynsyms") lrx_* entry points leg-suffixed"
unset _dynsyms

# Dependency closure (lxkit.native_provider), as on Perlmutter: the pinned
# MPI/HDF5/LibSci files, enforced by the loader.
"$LORRAX_VENV_PY" "$LORRAX_ROOT/src/ffi/cpp/stage/write_checkout_closure.py" \
    "$SO_FILE" \
    --library-dirs "$LORRAX_OLCF_RUNTIME_LIB_DIRS" || {
        echo "[build_ffi_host] FAILED: dependency closure" >&2; exit 1; }
_closure_sha="$("$LORRAX_VENV_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["record_sha256"])' "$SO_FILE.closure.json")"

# All four host backends are built here.  GATE 8 needs a python that imports
# jax and must run inside an allocation; LORRAX_GATE_FFTW_PY is passed so a
# run on a compute node executes it (on a login node it is not evidence).
LORRAX_FFI_EXPECT_BACKENDS="${LORRAX_FFI_EXPECT_BACKENDS:-scalapack,gemm,phdf5,fft}" \
LORRAX_FFI_EXPECT_MPI="${LORRAX_FFI_EXPECT_MPI:-$LORRAX_OLCF_MPI_SONAME}" \
LORRAX_FFI_EXPECT_HDF5_SOVERSION="${LORRAX_FFI_EXPECT_HDF5_SOVERSION:-$_stage_sov}" \
LORRAX_PHDF5_STAGE="$LORRAX_OLCF_PHDF5_STAGE" \
LD_LIBRARY_PATH="$LORRAX_OLCF_RUNTIME_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
LORRAX_GATE_FFTW_PY="${LORRAX_GATE_FFTW_PY:-}" \
GATE_TAG=build_ffi_host \
    bash "$LORRAX_ROOT/scripts/verify_ffi_build.sh" --leg host "$SO_FILE" || {
        echo "[build_ffi_host] FAILED: the library does not meet the build contract." >&2
        echo "[build_ffi_host] The .so is left on disk for inspection; do not pin it." >&2
        exit 1
    }

"$LORRAX_ROOT/src/ffi/cpp/stage/stamp_provenance.sh" "$SO_FILE" \
    "leg=host" \
    "site=frontier" \
    "closure_sha256=$_closure_sha" \
    "prgenv=$LORRAX_OLCF_PRGENV" \
    "mpi=$LORRAX_OLCF_MPICH_MODULE" \
    "mpi_root=$LORRAX_OLCF_MPICH_ROOT" \
    "libsci=$LORRAX_OLCF_LIBSCI$LORRAX_OLCF_LIBSCI_FLAVOUR" \
    "hdf5=$LORRAX_OLCF_HDF5" \
    "phdf5_stage=$LORRAX_OLCF_PHDF5_STAGE" \
    "fftw=$LORRAX_OLCF_FFTW" \
    "venv=$LORRAX_OLCF_VENV" || {
        echo "[build_ffi_host] WARNING: provenance stamp failed" >&2
    }

echo
echo "[build_ffi_host] done.  .so at: $SO_FILE"
echo "[build_ffi_host] run environment: source config/frontier/ffi_env.sh"
