#!/usr/bin/env bash
# build_perlmutter.sh — reproducible SLATE builds on Perlmutter (GPU + CPU).
#
# Usage:
#   src/ffi/cpp/stage/slate_build_perlmutter.sh gpu   [--fresh]
#   src/ffi/cpp/stage/slate_build_perlmutter.sh cpu   [--fresh]
#
# Produces:
#   $HOME/software/slate_builds/src/slate        shared source checkout (pinned)
#   $HOME/software/slate_builds/gpu/{build,install}   gpu_backend=cuda
#   $HOME/software/slate_builds/cpu/{build,install}   gpu_backend=none
#
# Where to run: the script is host-side (NO shifter container — SLATE is
# built against the Cray PE so it gets libsci BLAS/LAPACK/ScaLAPACK and
# GPU-aware Cray MPICH).  cmake configure is fine on a login node; the
# compile is ~15 min at -j32, so if the login node is loaded run the whole
# thing through the allocation instead:
#
#   srun --jobid=$SLURM_JOBID --overlap -N1 -n1 -c 64 \
#       bash src/ffi/cpp/stage/slate_build_perlmutter.sh gpu
#
# (Compilation only needs the x86 host side — building the `cpu` variant
# on a GPU node is fine; both node types are AMD Milan.)
#
# Module stack (NERSC-recommended, see
# https://docs.nersc.gov/development/programming-models/cuda/ and
# .../mpi/cray-mpich/):
#   GPU:  PrgEnv-gnu cray-libsci cmake cudatoolkit craype-accel-nvidia80
#   CPU:  PrgEnv-gnu cray-libsci cmake        (accel + cudatoolkit UNLOADED)
#
# craype-accel-nvidia80 matters twice for the GPU build: it makes nvcc
# target sm_80 AND it makes the CC wrapper link libmpi_gtl_cuda (the GPU
# Transport Layer), which Cray MPICH requires at runtime whenever
# MPICH_GPU_SUPPORT_ENABLED=1.  The CPU build must NOT link the GTL: it
# drags in libcuda.so.1 (the driver), which does not exist on CPU nodes.
#
# Overrides (env):
#   LORRAX_SLATE_BUILDS_DIR   root dir     (default $HOME/software/slate_builds)
#   LORRAX_SLATE_REPO         git URL/path (default github icl-utk-edu/slate)
#   LORRAX_SLATE_COMMIT       commit/tag   (default ded15290 = v2025.05.28-1,
#                                           same as the $HOME/software/slate
#                                           evaluation build)
#   LORRAX_SLATE_CUDATOOLKIT  cudatoolkit module version (default 12.9 — must
#                             stay CUDA-12 to match the nvcr.io/nvidia/jax
#                             container the LORRAX FFI runs in; libcudart
#                             ABI is compatible within a major version)
#   LORRAX_SLATE_MAKE_J       parallel build jobs (default 32)

set -euo pipefail

VARIANT="${1:?usage: build_perlmutter.sh gpu|cpu [--fresh]}"
case "${VARIANT}" in gpu|cpu) ;; *)
    echo "build_perlmutter.sh: variant must be 'gpu' or 'cpu', got '${VARIANT}'" >&2
    exit 2 ;;
esac
FRESH="${2:-}"

ROOT="${LORRAX_SLATE_BUILDS_DIR:-$HOME/software/slate_builds}"
REPO="${LORRAX_SLATE_REPO:-https://github.com/icl-utk-edu/slate.git}"
COMMIT="${LORRAX_SLATE_COMMIT:-ded15290}"
CTK_VER="${LORRAX_SLATE_CUDATOOLKIT:-12.9}"
JOBS="${LORRAX_SLATE_MAKE_J:-32}"

SRC="${ROOT}/src/slate"
BUILD="${ROOT}/${VARIANT}/build"
PREFIX="${ROOT}/${VARIANT}/install"

# ---------------------------------------------------------------------------
# Modules.  `module` is a shell function; when this script runs under srun
# (non-login bash) it isn't defined yet — init Lmod explicitly.
# ---------------------------------------------------------------------------
if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash
fi

module load PrgEnv-gnu       # gcc-native compiler under the CC/cc/ftn wrappers
module load cray-libsci      # BLAS/LAPACK/ScaLAPACK (wrapper links it implicitly)
module load cmake
if [[ "${VARIANT}" == gpu ]]; then
    module load "cudatoolkit/${CTK_VER}"
    module load craype-accel-nvidia80
else
    # Unload so the CC wrapper emits no -lmpi_gtl_cuda and blaspp cannot
    # auto-detect CUDA.  `|| true`: they may not be loaded to begin with.
    module unload craype-accel-nvidia80 2>/dev/null || true
    module unload cudatoolkit           2>/dev/null || true
fi
echo "[slate-build] modules:"
module -t list 2>&1 | sed 's/^/[slate-build]   /'

# ---------------------------------------------------------------------------
# Source checkout, pinned.  Submodules (blaspp, lapackpp, testsweeper) come
# from the same org and are pinned by the superproject commit.
# ---------------------------------------------------------------------------
mkdir -p "${ROOT}/src"
if [[ ! -d "${SRC}/.git" ]]; then
    echo "[slate-build] cloning ${REPO} -> ${SRC}"
    git clone --recursive "${REPO}" "${SRC}"
fi
git -C "${SRC}" fetch --tags --quiet 2>/dev/null || true   # offline-tolerant
git -C "${SRC}" checkout --quiet "${COMMIT}"
git -C "${SRC}" submodule update --init --recursive --quiet
echo "[slate-build] source at $(git -C "${SRC}" describe --tags --always)"

# ---------------------------------------------------------------------------
# Configure + build + install
# ---------------------------------------------------------------------------
if [[ "${FRESH}" == "--fresh" ]]; then
    echo "[slate-build] --fresh: wiping ${BUILD} and ${PREFIX}"
    rm -rf "${BUILD}" "${PREFIX}"
fi
mkdir -p "${BUILD}"

if [[ "${VARIANT}" == gpu ]]; then
    BACKEND=cuda
    # A100 only; skipping other archs keeps nvcc time down.
    EXTRA_CMAKE=(-DCMAKE_CUDA_ARCHITECTURES=80)
else
    BACKEND=none
    EXTRA_CMAKE=()
fi

# Flag notes:
#  * CXX=CC etc.: Cray wrappers, so libsci + MPICH (+ GTL on gpu) are linked
#    with the right ABI automatically.
#  * -Dblas=libsci: tell BLAS++'s search to look for Cray LibSci instead of
#    probing openblas/mkl.
#  * -DSCALAPACK_LIBRARIES="" — THE GOTCHA.  SLATE's test/CMakeLists.txt
#    defaults this to "scalapack" (-lscalapack), which does not exist as a
#    standalone lib on Cray: ScaLAPACK lives inside libsci_gnu_mpi, which
#    the CC wrapper already links.  Empty string keeps the tester's
#    ScaLAPACK reference checks compiled in (SLATE_HAVE_SCALAPACK) while
#    linking nothing extra, letting the wrapper-provided libsci satisfy the
#    p* symbols.  (SCALAPACK_LIBRARIES="none" would instead compile the
#    reference checks OUT — then `tester --ref y` cannot cross-check.)
#  * No -DSLATE_HAVE_MT_BCAST: ICL's INSTALL.md warns the multi-threaded
#    bcast path hangs on some systems ("particularly Frontier") — and
#    Frontier portability is the point of this exercise.
#  * build_tests=yes (default) so we get test/tester for validation.
cmake -S "${SRC}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=CC \
    -DCMAKE_C_COMPILER=cc \
    -DCMAKE_Fortran_COMPILER=ftn \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -Dblas=libsci \
    -Dgpu_backend="${BACKEND}" \
    -DSCALAPACK_LIBRARIES="" \
    -Dbuild_tests=yes \
    "${EXTRA_CMAKE[@]}"

cmake --build "${BUILD}" --parallel "${JOBS}"
cmake --install "${BUILD}"

echo
echo "[slate-build] ${VARIANT} done."
echo "[slate-build]   install : ${PREFIX}"
echo "[slate-build]   tester  : ${BUILD}/test/tester"
ls -l "${PREFIX}"/lib64/libslate.so* 2>/dev/null || ls -l "${PREFIX}"/lib/libslate.so* 2>/dev/null
