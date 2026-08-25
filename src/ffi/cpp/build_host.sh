#!/usr/bin/env bash
# build_host.sh — build liblorrax_ffi_host.so, the CUDA-free host-platform
# FFI library (SLATE host handlers; see CMakeLists.txt, host leg).
#
# Runs HOST-SIDE under the Cray PE (no Shifter container) against the
# gpu_backend=none SLATE install.  cmake + the ~30 s compile are fine on a
# login node; to route through the allocation instead:
#
#   srun --jobid=$SLURM_JOBID --overlap -N1 -n1 -c 32 \
#       bash src/ffi/cpp/build_host.sh
#
# XLA FFI headers: they must match the RUNTIME jaxlib — the Shifter
# container's jax, NOT the host venv's (which may ship a newer XLA FFI API;
# newer-headers-on-older-runtime is the unsupported direction).  This
# script stages the container's jax.ffi.include_dir() tree to
# $HOME/software/lorrax_xla_ffi_headers/<image tag>/ once and reuses it.
#
# Output: src/ffi/cpp/build_host/liblorrax_ffi_host.so
#         (ffi_loader.py finds it there; override with LORRAX_FFI_HOST_SO)
#
# Overrides (env):
#   LORRAX_FFI_HOST_BUILD_DIR      alternate build dir
#   LORRAX_SLATE_HOST_INSTALL_DIR  SLATE none-backend install
#                                  (default $HOME/software/slate_builds/cpu/install)
#   LORRAX_XLA_FFI_HEADERS_DIR     pre-staged header dir (skips staging)
#   LORRAX_FFI_HOST_IMAGE          required Shifter image to stage headers
#                                  from. It must be the JAX-0.9 image under
#                                  which the .so will be loaded.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LORRAX_ROOT="${LORRAX_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# ===========================================================================
# ON A MACHINE THAT HAS A SITE RECIPE, USE THE SITE RECIPE.
#
# THIS IS THE FIX FOR THE ACCIDENT THAT MOTIVATED THE WHOLE BUILD CONTRACT.
# The library deployed on 2026-08-07 was produced by running THIS script on
# Perlmutter.  It configures no ScaLAPACK — CMake's built-in probe searches for
# an MKL layout, which does not describe Cray LibSci — so "ScaLAPACK/BLACS not
# found" printed as a WARNING, the link succeeded, and the library shipped with
# `scalapack=0` and zero Scalapack* handlers.  It also left cray-libsci loaded
# at configure time, so the CC wrapper auto-injected the sequential LibSci
# beside the threaded pair the SLATE install needs.  Two defects, one command,
# no failure.
#
# The verifier at the bottom of this file would now catch both.  Catching them
# is worse than not producing them: config/perlmutter/build_ffi_host.sh already
# knows every one of those answers, and a person who ran the obvious command
# should get the right library, not a lecture.  So the obvious command hands
# over.
#
# The escape is explicit and announced, per the gate contract in
# src/ffi/gate.py — an opt-out is a stated decision, never an inference:
#   LORRAX_FFI_GENERIC_BUILD=1   build here anyway, on this machine, with
#                                whatever this script alone can resolve.
# ===========================================================================
_site=""
case "${NERSC_HOST:-}${TACC_SYSTEM:-}" in
    perlmutter) _site=perlmutter ;;
    frontera)   _site=frontera ;;
esac
if [[ -n "$_site" && -f "${LORRAX_ROOT}/config/${_site}/build_ffi_host.sh" \
      && "${LORRAX_FFI_GENERIC_BUILD:-0}" != "1" ]]; then
    echo "[build_host] this is ${_site}, and it has a site recipe."
    echo "[build_host] handing over to config/${_site}/build_ffi_host.sh, which"
    echo "[build_host] knows this machine's ScaLAPACK link line, its LibSci"
    echo "[build_host] threading flavour, its HDF5 module and its phdf5 stage."
    echo "[build_host] (LORRAX_FFI_GENERIC_BUILD=1 builds here instead.)"
    exec bash "${LORRAX_ROOT}/config/${_site}/build_ffi_host.sh" "$@"
fi
if [[ -n "$_site" ]]; then
    echo "[build_host] NOTE: LORRAX_FFI_GENERIC_BUILD=1 on ${_site} — building" >&2
    echo "[build_host]   with the generic recipe, which resolves less than the" >&2
    echo "[build_host]   site one.  The verifier at the end will name whatever" >&2
    echo "[build_host]   this build does not contain." >&2
fi

BUILD_DIR="${LORRAX_FFI_HOST_BUILD_DIR:-${SCRIPT_DIR}/build_host}"
SLATE_DIR="${LORRAX_SLATE_HOST_INSTALL_DIR:-$HOME/software/slate_builds/cpu/install}"
# Staging headers from a different generation is an ABI skew.  Do not hide
# that choice behind a default—the old default was a retired JAX-0.7 image.
IMAGE="${LORRAX_FFI_HOST_IMAGE:-}"
if [[ -z "${IMAGE}" ]]; then
    echo "[build_host] LORRAX_FFI_HOST_IMAGE is required (JAX 0.9)." >&2
    exit 2
fi
case "${IMAGE}" in
    *jax-2025-07-21*|*25.04-py3*)
        echo "[build_host] refusing retired pre-0.9 image ${IMAGE}." >&2
        exit 2
        ;;
esac
IMAGE_TAG="${IMAGE##*:}"
HDR_DIR="${LORRAX_XLA_FFI_HEADERS_DIR:-$HOME/software/lorrax_xla_ffi_headers/${IMAGE_TAG}}"

if [[ ! -d "${SLATE_DIR}/lib64" ]]; then
    echo "[build_host] ERROR: SLATE gpu_backend=none install not found at" >&2
    echo "[build_host]   ${SLATE_DIR}" >&2
    echo "[build_host] Build it first:" >&2
    echo "[build_host]   bash src/ffi/cpp/stage/slate_build_perlmutter.sh cpu" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Modules (same pattern as cpp/stage/slate_build_perlmutter.sh cpu): Cray PE,
# no CUDA — with craype-accel-nvidia80 loaded the CC wrapper would link
# libmpi_gtl_cuda, whose libcuda.so.1 dependency defeats the purpose.
# ---------------------------------------------------------------------------
if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash
fi
module load PrgEnv-gnu
module load cray-libsci
module load cmake
module unload craype-accel-nvidia80 2>/dev/null || true
module unload cudatoolkit           2>/dev/null || true

# ---------------------------------------------------------------------------
# Stage the container's XLA FFI headers (once).
# ---------------------------------------------------------------------------
if [[ ! -f "${HDR_DIR}/xla/ffi/api/ffi.h" ]]; then
    echo "[build_host] staging XLA FFI headers from ${IMAGE} -> ${HDR_DIR}"
    mkdir -p "${HDR_DIR}"
    STAGE_CMD=(shifter "--image=${IMAGE}" bash -c
        "cp -r \$(python3 -c 'import jax.ffi,sys;sys.stdout.write(jax.ffi.include_dir())')/xla '${HDR_DIR}/'")
    if [[ -n "${SLURM_JOBID:-}" ]]; then
        srun --jobid="${SLURM_JOBID}" --overlap -N1 -n1 "${STAGE_CMD[@]}"
    else
        "${STAGE_CMD[@]}"
    fi
fi
if [[ ! -f "${HDR_DIR}/xla/ffi/api/ffi.h" ]]; then
    echo "[build_host] ERROR: header staging failed (no xla/ffi/api/ffi.h" >&2
    echo "[build_host] under ${HDR_DIR}).  If shifter is flaky on the login" >&2
    echo "[build_host] node, rerun with SLURM_JOBID set (see KNOWN_SANDBOX_ERRORS)." >&2
    exit 2
fi

echo "[build_host] SLATE (none) = ${SLATE_DIR}"
echo "[build_host] XLA headers  = ${HDR_DIR}"
echo "[build_host] build dir    = ${BUILD_DIR}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [[ "${1:-}" == "--fresh" ]]; then
    echo "[build_host] --fresh: wiping ${BUILD_DIR}"
    rm -rf "${BUILD_DIR:?}/"* "${BUILD_DIR:?}/".??* 2>/dev/null || true
fi

cmake "${SCRIPT_DIR}" \
    -DLORRAX_FFI_PLATFORM=host \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=CC \
    -DLORRAX_XLA_FFI_INCLUDE_DIR="${HDR_DIR}" \
    -DLORRAX_SLATE_HOST_INSTALL_DIR="${SLATE_DIR}"

cmake --build . --parallel

SO_FILE="${BUILD_DIR}/liblorrax_ffi_host.so"
if [[ ! -f "${SO_FILE}" ]]; then
    echo "[build_host] FAILED: no .so produced." >&2
    exit 1
fi

# ===========================================================================
# THE BUILD CONTRACT.  Unconditional, and a failure here fails the build.
#
# This replaces the single inline CUDA-free check that used to live here — one
# of the seven properties an FFI library has to have, checked in one of the
# five places a library can be produced.  scripts/verify_ffi_build.sh is the
# single source for all of them; see its header for the gate list and for why
# it is a script rather than a CMake POST_BUILD rule.
#
# On a site with no ScaLAPACK/GEMM this build legitimately contains fewer
# backends than the default expectation.  SAY SO — the declaration is the
# gate:
#     LORRAX_FFI_EXPECT_BACKENDS=slate,phdf5,fft bash src/ffi/cpp/build_host.sh
# ===========================================================================
echo
echo "[build_host] --- build contract (scripts/verify_ffi_build.sh) ---"
bash "${LORRAX_ROOT}/scripts/verify_ffi_build.sh" --leg host "${SO_FILE}" || {
    echo "[build_host] FAILED: the library does not meet the build contract." >&2
    echo "[build_host] The .so is left on disk so it can be inspected, but it" >&2
    echo "[build_host] must not be deployed or pinned.  Read the gate output" >&2
    echo "[build_host] above; every failure names its own fix." >&2
    exit 1
}

echo
echo "[build_host] done.  .so at: ${SO_FILE}"
