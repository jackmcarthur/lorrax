#!/usr/bin/env bash
# ============================================================================
# build_slate_cpu.sh — the gpu_backend=none SLATE (+ BLAS++, LAPACK++) that
# liblorrax_ffi_host.so links on OLCF Frontier.  Frontier twin of
# src/ffi/cpp/stage/slate_build_perlmutter.sh cpu.
#
#   config/frontier/build_slate_cpu.sh [--fresh]
#
# Produces (ROOT = $LORRAX_SLATE_ROOT, default $HOME/software/slate_builds):
#   $ROOT/src/slate        pinned source checkout
#   $ROOT/cpu/build        build tree
#   $ROOT/cpu/install      install prefix = the CMakeLists' default
#                          LORRAX_SLATE_HOST_INSTALL_DIR
#
# Module stack: PrgEnv-gnu + the pins in config/frontier/ffi_mpi.sh, with
# craype-accel-amd-gfx90a / rocm UNLOADED so the wrapper links no
# libmpi_gtl_hsa and BLAS++ cannot detect HIP.
#
# Why not the OLCF slate modules: see config/frontier/ffi_mpi.sh (they link
# OpenBLAS, which beside LibSci ScaLAPACK is a second BLAS: GATE 2).
#
# Tests (build_tests) are OFF by default to keep the compile within a login
# session; LORRAX's own SLATE contract tests exercise the install.
# LORRAX_SLATE_BUILD_TESTS=yes builds test/tester (then SCALAPACK_LIBRARIES=""
# applies, as on Perlmutter).
# ============================================================================
set -euo pipefail
ulimit -c 0
FRESH="${1:-}"
LORRAX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${LORRAX_SLATE_ROOT:-$HOME/software/slate_builds}"
REPO="${LORRAX_SLATE_REPO:-https://github.com/icl-utk-edu/slate.git}"
COMMIT="${LORRAX_SLATE_COMMIT:-ded15290}"     # v2025.05.28-1, as on Perlmutter
JOBS="${LORRAX_SLATE_MAKE_J:-32}"
TESTS="${LORRAX_SLATE_BUILD_TESTS:-no}"
SRC="$ROOT/src/slate"; BUILD="$ROOT/cpu/build"; PREFIX="$ROOT/cpu/install"

if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash 2>/dev/null || source /opt/cray/pe/lmod/lmod/init/bash
fi
# shellcheck disable=SC1091
source "$LORRAX_ROOT/config/frontier/ffi_mpi.sh"
module load "$LORRAX_OLCF_PRGENV_MODULE"
lorrax_olcf_pin_mpi            # AFTER PrgEnv: it resets cray-mpich to the default
module load "$LORRAX_OLCF_LIBSCI_MODULE" "$LORRAX_OLCF_CMAKE_MODULE"
module unload craype-accel-amd-gfx90a 2>/dev/null || true
module unload rocm amd-mixed           2>/dev/null || true
[[ "$CRAY_MPICH_DIR" == "$LORRAX_OLCF_MPICH_ROOT" && "$CRAY_LIBSCI_PREFIX_DIR" == "$LORRAX_OLCF_LIBSCI_ROOT" ]] || {
    echo "[slate-build] ERROR: modules resolved MPICH=$CRAY_MPICH_DIR LIBSCI=$CRAY_LIBSCI_PREFIX_DIR," >&2
    echo "[slate-build]   not the pins in config/frontier/ffi_mpi.sh" >&2; exit 2; }
echo "[slate-build] modules: $(module -t list 2>&1 | tr '\n' ' ')"

mkdir -p "$ROOT/src"
[[ -d "$SRC/.git" ]] || git clone --recursive "$REPO" "$SRC"
git -C "$SRC" fetch --tags --quiet 2>/dev/null || true
git -C "$SRC" checkout --quiet "$COMMIT"
git -C "$SRC" submodule update --init --recursive --quiet
echo "[slate-build] source at $(git -C "$SRC" describe --tags --always)"

if [[ "$FRESH" == "--fresh" ]]; then rm -rf "$BUILD" "$PREFIX"; fi
mkdir -p "$BUILD"
cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=CC -DCMAKE_C_COMPILER=cc -DCMAKE_Fortran_COMPILER=ftn \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -Dblas=libsci \
    -Dgpu_backend=none \
    -DSCALAPACK_LIBRARIES="" \
    -Dbuild_tests="$TESTS"
cmake --build "$BUILD" --parallel "$JOBS"
cmake --install "$BUILD"

echo "[slate-build] cpu done. install: $PREFIX"
for f in "$PREFIX"/lib64/libslate.so "$PREFIX"/lib64/libblaspp.so "$PREFIX"/lib64/liblapackpp.so; do
    echo "[slate-build] $(basename "$(readlink -f "$f")"): $(readelf -d "$f" | grep NEEDED | sed 's/.*\[//;s/\]//' | tr '\n' ' ')"
done
# A gpu_backend=none BLAS++ must not name any HIP/CUDA runtime.
if readelf -d "$PREFIX"/lib64/libblaspp.so | grep -qiE 'amdhip|rocblas|cudart|cublas'; then
    echo "[slate-build] ERROR: BLAS++ links a GPU runtime" >&2; exit 1
fi
