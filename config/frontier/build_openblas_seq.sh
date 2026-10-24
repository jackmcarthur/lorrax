#!/usr/bin/env bash
# ============================================================================
# build_openblas_seq.sh — the ROCM leg's one host BLAS/LAPACK on OLCF Frontier
# (ELPA and its netlib ScaLAPACK link it: config/frontier/build_elpa_hip.sh):
# OpenBLAS, SEQUENTIAL (USE_THREAD=0) and LOCKING (USE_LOCKING=1).
#
#   config/frontier/build_openblas_seq.sh
#
# Produces $ROOT/install (ROOT = $LORRAX_OPENBLAS_ROOT, default
# $HOME/software/openblas_seq) with a LORRAX_BUILD_FLAGS stamp that
# build_elpa_hip.sh gates on.
#
# WHY NOT LIBSCI.  Host BLAS/LAPACK is called concurrently from OpenMP threads.
# Neither LibSci flavour is safe for that: the threaded one SIGSEGVed, and the
# sequential libsci_gnu is an OpenBLAS single-thread build WITHOUT locking,
# whose shared buffers made concurrent calls return silently wrong numbers
# (measured 2026-10-15, P=4 2x2 and P=1: eigenvalue errors 1e-3 .. 4; exact
# with OMP_NUM_THREADS=1 or with this library preloaded).  OpenBLAS documents
# USE_LOCKING=1 as the switch that makes a USE_THREAD=0 build safe to call from
# several threads.
# ============================================================================
set -euo pipefail
ulimit -c 0
ROOT="${LORRAX_OPENBLAS_ROOT:-$HOME/software/openblas_seq}"
TAG="${LORRAX_OPENBLAS_TAG:-v0.3.28}"
JOBS="${LORRAX_OPENBLAS_MAKE_J:-16}"
SRC="$ROOT/src"; PREFIX="$ROOT/install"
FLAGS="TARGET=ZEN USE_THREAD=0 USE_OPENMP=0 USE_LOCKING=1 NUM_THREADS=1 NO_STATIC=1"
if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash 2>/dev/null || source /opt/cray/pe/lmod/lmod/init/bash
fi
export LMOD_PAGER=none
module load PrgEnv-gnu/8.7.0 >/dev/null 2>&1
mkdir -p "$ROOT"
[[ -d "$SRC/.git" ]] || git clone --quiet --depth 1 -b "$TAG" https://github.com/OpenMathLib/OpenBLAS "$SRC"
# shellcheck disable=SC2086
make -C "$SRC" -s -j"$JOBS" CC=gcc FC=gfortran $FLAGS libs netlib shared
# shellcheck disable=SC2086
make -C "$SRC" -s PREFIX="$PREFIX" $FLAGS install
echo "$TAG $FLAGS" > "$PREFIX/LORRAX_BUILD_FLAGS"
echo "[openblas-seq] $PREFIX/lib/libopenblas.so ($(cat "$PREFIX/LORRAX_BUILD_FLAGS"))"
