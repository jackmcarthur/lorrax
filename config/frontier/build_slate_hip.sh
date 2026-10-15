#!/usr/bin/env bash
# ============================================================================
# build_slate_hip.sh — the gpu_backend=hip SLATE (+ BLAS++, LAPACK++) that
# liblorrax_ffi_rocm.so links on OLCF Frontier (MI250X, gfx90a).  The ROCM
# twin of build_slate_cpu.sh, and of src/ffi/cpp/stage/slate_build_perlmutter.sh
# cuda on the CUDA leg.
#
#   config/frontier/build_slate_hip.sh [--fresh]
#
# Produces (ROOT = $LORRAX_SLATE_ROOT, default $HOME/software/slate_builds):
#   $ROOT/src/slate        the pinned source checkout SHARED with the cpu build
#                          (the HIP sources are hipify-perl'd into src/hip/ by
#                          SLATE's own CMake; the cpu build never compiles them)
#   $ROOT/hip/build        build tree
#   $ROOT/hip/install      install prefix = the ROCM block's default
#                          LORRAX_SLATE_ROCM_INSTALL_DIR
#
# ONE ROCm RUNTIME PER PROCESS.  JAX maps the pip ROCm 7.14.1 wheel
# (_rocm_sdk_core: libamdhip64.so.7; _rocm_sdk_libraries: librocblas.so.5,
# librocsolver.so.0).  SLATE is built against module rocm/7.14.1, the SAME
# release with the SAME SONAMEs, so at run time its NEEDED entries are
# satisfied by the objects JAX already mapped (ld.so keys loaded objects by
# SONAME).  The run-time check is distrib_la's slate gate, which reads
# /proc/self/maps; this script checks the link-time half (the SONAMEs).
#
# One MPI and one BLAS per process: PrgEnv-gnu + the pins in ffi_mpi.sh,
# craype-accel-amd-gfx90a UNLOADED so the Cray wrapper links no libmpi_gtl_hsa
# (SLATE stages tiles through host memory for MPI; it needs no GPU-aware MPI).
# Host BLAS/LAPACK = LibSci's SEQUENTIAL flavour libsci_gnu, passed EXPLICITLY
# with cray-libsci captured-then-UNLOADED (else the wrapper injects its own).
# SEQUENTIAL because SLATE calls host BLAS concurrently from its OpenMP task
# threads and assumes a sequential BLAS (its own default blas_threaded=false;
# it has no BLAS-thread control).  MEASURED 2026-10-15, P=4 2x2, heev n=2048
# (nb=1024): against libsci_gnu_mp every run SIGSEGVs in
# openblas_dgemm_oncopy <- dtrmm on a SLATE worker thread; OMP_NUM_THREADS=7,1
# still crashes, OMP_NUM_THREADS=1 passes, and LD_PRELOAD of the sequential
# libsci_gnu.so.6 passes with full threading.  n=512 (nb=256) passes either
# way: below OpenBLAS's threading threshold.
# Earlier measurement, module left loaded: BLAS++ (linked by the HIP driver,
# one .hip source) got the wrapper's implicit -lsci_gnu_mpi -lsci_gnu beside
# the threaded pair -- both LibSci flavours in one library (GATE 2a).
# Device code: SLATE's src/hip kernels are compiled by hipcc; BLAS++ enables
# the CMake HIP language.  Both target gfx90a (no xnack feature, so the image
# runs under either HSA_XNACK mode).
# ============================================================================
set -euo pipefail
ulimit -c 0
FRESH="${1:-}"
LORRAX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${LORRAX_SLATE_ROOT:-$HOME/software/slate_builds}"
REPO="${LORRAX_SLATE_REPO:-https://github.com/icl-utk-edu/slate.git}"
COMMIT="${LORRAX_SLATE_COMMIT:-ded15290}"     # v2025.05.28-1, as the cpu build and Perlmutter
JOBS="${LORRAX_SLATE_MAKE_J:-16}"
ROCM_MOD="${LORRAX_OLCF_ROCM:-rocm/7.14.1}"   # == the pip rocm-sdk JAX loads
GFX="${LORRAX_SLATE_GFX:-gfx90a}"
SRC="$ROOT/src/slate"; BUILD="$ROOT/hip/build"; PREFIX="$ROOT/hip/install"

if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash 2>/dev/null || source /opt/cray/pe/lmod/lmod/init/bash
fi
export LMOD_PAGER=none
# shellcheck disable=SC1091
source "$LORRAX_ROOT/config/frontier/ffi_mpi.sh"
module load "$LORRAX_OLCF_PRGENV_MODULE"
lorrax_olcf_pin_mpi            # AFTER PrgEnv: it resets cray-mpich to the default
module load "$LORRAX_OLCF_LIBSCI_MODULE" "$LORRAX_OLCF_CMAKE_MODULE" "$ROCM_MOD"
module unload craype-accel-amd-gfx90a 2>/dev/null || true
[[ "$CRAY_MPICH_DIR" == "$LORRAX_OLCF_MPICH_ROOT" && "$CRAY_LIBSCI_PREFIX_DIR" == "$LORRAX_OLCF_LIBSCI_ROOT" ]] || {
    echo "[slate-hip] ERROR: modules resolved MPICH=$CRAY_MPICH_DIR LIBSCI=$CRAY_LIBSCI_PREFIX_DIR," >&2
    echo "[slate-hip]   not the pins in config/frontier/ffi_mpi.sh" >&2; exit 2; }
: "${ROCM_PATH:?module $ROCM_MOD did not set ROCM_PATH}"
# Capture LibSci, THEN unload it (else the wrapper injects sequential -lsci_gnu).
LIBSCI_LIB="$CRAY_LIBSCI_PREFIX_DIR/lib/libsci_gnu${LORRAX_SLATE_LIBSCI_FLAVOUR:-}.so"   # sequential
[[ -f "$LIBSCI_LIB" ]] || { echo "[slate-hip] ERROR: no $LIBSCI_LIB" >&2; exit 2; }
module unload cray-libsci 2>/dev/null || true
echo "[slate-hip] modules: $(module -t list 2>&1 | tr '\n' ' ')"
echo "[slate-hip] ROCM_PATH=$ROCM_PATH"

mkdir -p "$ROOT/src"
[[ -d "$SRC/.git" ]] || git clone --recursive "$REPO" "$SRC"
git -C "$SRC" fetch --tags --quiet 2>/dev/null || true
git -C "$SRC" checkout --quiet "$COMMIT"
git -C "$SRC" submodule update --init --recursive --quiet
echo "[slate-hip] source at $(git -C "$SRC" describe --tags --always)"

# SLATE's src/hip objects are add_custom_command(hipcc ...) without an arch
# flag; hipcc reads the target from here.
export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=$GFX"
if [[ "$FRESH" == "--fresh" ]]; then rm -rf "$BUILD" "$PREFIX"; fi
mkdir -p "$BUILD"
cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=CC -DCMAKE_C_COMPILER=cc -DCMAKE_Fortran_COMPILER=ftn \
    -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
    -DCMAKE_HIP_ARCHITECTURES="$GFX" -DGPU_TARGETS="$GFX" -DAMDGPU_TARGETS="$GFX" \
    -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DBLAS_LIBRARIES="$LIBSCI_LIB" -DLAPACK_LIBRARIES="$LIBSCI_LIB" \
    -Dgpu_backend=hip \
    -DSCALAPACK_LIBRARIES="" \
    -Dbuild_tests=no
cmake --build "$BUILD" --parallel "$JOBS"
cmake --install "$BUILD"

echo "[slate-hip] hip done. install: $PREFIX"
fail=0
for f in "$PREFIX"/lib64/libslate.so "$PREFIX"/lib64/libblaspp.so "$PREFIX"/lib64/liblapackpp.so; do
    echo "[slate-hip] $(basename "$(readlink -f "$f")"): $(readelf -d "$f" | grep NEEDED | sed 's/.*\[//;s/\]//' | tr '\n' ' ')"
done
# The link-time half of "one ROCm, one MPI per process".
need() { readelf -d "$1" | grep NEEDED | sed 's/.*\[//;s/\]//'; }
need "$PREFIX/lib64/libblaspp.so"  | grep -qx 'librocblas.so.5'   || { echo "[slate-hip] ERROR: BLAS++ has no rocBLAS (gpu_backend=hip not taken)" >&2; fail=1; }
need "$PREFIX/lib64/liblapackpp.so" | grep -qx 'librocsolver.so.0' || { echo "[slate-hip] ERROR: LAPACK++ has no rocSOLVER" >&2; fail=1; }
for f in "$PREFIX"/lib64/lib{slate,blaspp,lapackpp}.so; do
    if need "$f" | grep -qE 'cudart|cublas|libmpi_gtl|libopenblas'; then
        echo "[slate-hip] ERROR: $(basename "$f") links a CUDA runtime / GTL / second BLAS" >&2; fail=1; fi
    if need "$f" | grep -E '^libamdhip64' | grep -qvx 'libamdhip64.so.7'; then
        echo "[slate-hip] ERROR: $(basename "$f") links a HIP runtime other than libamdhip64.so.7" >&2; fail=1; fi
    if need "$f" | grep -E '^libmpi' | grep -qvx "$LORRAX_OLCF_MPI_SONAME"; then
        echo "[slate-hip] ERROR: $(basename "$f") links an MPI other than $LORRAX_OLCF_MPI_SONAME" >&2; fail=1; fi
done
# One LibSci threading flavour across the whole SLATE closure (GATE 2a's rule,
# applied to the three libraries the leg pulls in).
_sci="$(for f in "$PREFIX"/lib64/lib{slate,blaspp,lapackpp}.so; do need "$f"; done | grep -E '^libsci' | sort -u)"
if printf '%s\n' "$_sci" | grep -qE '^libsci_gnu(_mpi)?\.so' && printf '%s\n' "$_sci" | grep -qE '_mp\.so'; then
    echo "[slate-hip] ERROR: both LibSci threading flavours in the SLATE closure: $(echo $_sci)" >&2; fail=1; fi
if [[ -z "${LORRAX_SLATE_LIBSCI_FLAVOUR:-}" ]] && printf '%s\n' "$_sci" | grep -qE '_mp\.so'; then
    echo "[slate-hip] ERROR: a threaded LibSci in the SLATE closure ($(echo $_sci)); SLATE needs the sequential one" >&2; fail=1; fi
echo "[slate-hip] LibSci in closure: $(echo $_sci)"
exit $fail
