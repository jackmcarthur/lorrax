#!/usr/bin/env bash
# ============================================================================
# build_openblas_seq.sh — the host BLAS/LAPACK the HIP SLATE links on OLCF
# Frontier: OpenBLAS, SEQUENTIAL (USE_THREAD=0) and LOCKING (USE_LOCKING=1).
#
#   config/frontier/build_openblas_seq.sh
#
# Produces $ROOT/openblas/install (ROOT = $LORRAX_SLATE_ROOT, default
# $HOME/software/slate_builds) with a LORRAX_BUILD_FLAGS stamp that
# build_slate_hip.sh gates on.
#
# WHY NOT LIBSCI.  SLATE calls host BLAS/LAPACK concurrently from its OpenMP
# task threads and assumes a sequential BLAS that is safe for concurrent
# callers.  Neither LibSci flavour is: the threaded one SIGSEGVs there
# (SLATE_HIP_REPORT §2), and the sequential libsci_gnu is an OpenBLAS
# single-thread build WITHOUT locking, whose shared buffers make concurrent
# calls return silently wrong numbers.  MEASURED 2026-10-15, P=4 2x2 and P=1,
# heev at nb < n/p (several tiles per rank): eigenvalue errors 1e-3 .. 4 and
# eigenvector residuals 0.04 .. 10 with OpenMP threading; exact (1e-15) with
# OMP_NUM_THREADS=1 or with this library preloaded (SLATE_PERF_REPORT).  At
# one tile per rank few tasks overlap, which is why it went unseen.
# OpenBLAS documents USE_LOCKING=1 as the switch that makes a USE_THREAD=0
# build safe to call from several threads.
# ============================================================================
set -euo pipefail
ulimit -c 0
ROOT="${LORRAX_SLATE_ROOT:-$HOME/software/slate_builds}"
TAG="${LORRAX_OPENBLAS_TAG:-v0.3.28}"
JOBS="${LORRAX_SLATE_MAKE_J:-16}"
SRC="$ROOT/openblas/src"; PREFIX="$ROOT/openblas/install"
FLAGS="TARGET=ZEN USE_THREAD=0 USE_OPENMP=0 USE_LOCKING=1 NUM_THREADS=1 NO_STATIC=1"
if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash 2>/dev/null || source /opt/cray/pe/lmod/lmod/init/bash
fi
export LMOD_PAGER=none
module load PrgEnv-gnu/8.7.0 >/dev/null 2>&1
mkdir -p "$ROOT/openblas"
[[ -d "$SRC/.git" ]] || git clone --quiet --depth 1 -b "$TAG" https://github.com/OpenMathLib/OpenBLAS "$SRC"
# shellcheck disable=SC2086
make -C "$SRC" -s -j"$JOBS" CC=gcc FC=gfortran $FLAGS libs netlib shared
# shellcheck disable=SC2086
make -C "$SRC" -s PREFIX="$PREFIX" $FLAGS install
echo "$TAG $FLAGS" > "$PREFIX/LORRAX_BUILD_FLAGS"
echo "[openblas-seq] $PREFIX/lib/libopenblas.so ($(cat "$PREFIX/LORRAX_BUILD_FLAGS"))"
