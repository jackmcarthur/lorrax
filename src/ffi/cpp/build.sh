#!/usr/bin/env bash
# Build liblorrax_ffi.so inside the Shifter container.
#
# Typical usage (from lorrax root):
#   export LX_BASE_MODULE=lorrax_A LORRAX_CHECKOUT=$PWD
#   lx run -N 1 -G 0 -n 1 -- bash src/ffi/cpp/build.sh
#
# `lx` enters the selected site image with the HPC SDK bind-mounted at
# /lorrax_nvhpc so the build can see cuSOLVERMp headers and libraries.
# `run_shifter.sh` remains the internal/manual composition tool for porting.
#
# Output: src/ffi/cpp/build/liblorrax_ffi*.so

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# LORRAX_FFI_BUILD_DIR: alternate build dir (default: in-tree build/).
# Useful for building against a different SLATE install without racing a
# concurrently-used in-tree .so; point LORRAX_FFI_SO at the result.
BUILD_DIR="${LORRAX_FFI_BUILD_DIR:-${SCRIPT_DIR}/build}"

# WHICH staged cuSOLVERMp this .so is COMPILED AND LINKED against.
#
# THIS MUST NOT BE HARDCODED, and was until 2026-08-06 — the same defect as
# run_shifter.sh's LORRAX_NVHPC_SUBPATH (fixed 2026-08-05), one layer down.
# The default here was `/lorrax_nvhpc/25.5_cuda12.9`, i.e. cuSolverMp 0.6.0,
# which config/perlmutter/site_config.sh documents as silently returning
# WRONG getrf/getrs answers on any Px>1 AND Py>1 mesh — while site_config's
# LORRAX_NVHPC_SUBPATH selects 0.7.2 for the RUN.  Every stage exports the
# same SONAME (libcusolverMp.so.0), so building against one and running
# against the other links cleanly and warns about nothing.  It has been
# latent rather than live only because 0.7.2 happens to come first on
# LD_LIBRARY_PATH; that is luck, one path edit deep.
#
# THERE IS NO SAFE DEFAULT, which is the reason this refuses instead of
# picking a better one.  The stages differ in more than a version number:
#   25.5_cuda12.9  cuSolverMp 0.6.0, ships cal.h + libcal -> the CAL comm
#                  path (LORRAX_FFI_HAVE_CAL=ON, the CMake default)
#   0.7.2_cuda12.9 cuSolverMp 0.7.2, NCCL-native, ships NO cal.h/libcal ->
#                  needs -DLORRAX_FFI_HAVE_CAL=OFF (CMakeLists.txt ~L40)
# So a default does not merely guess a version, it silently picks a
# COMMUNICATION PATH.  Defaulting to 0.7.2 would trade one silent
# substitution for another.  State it.
#
# Normally you state it once, in the site config, and never think about it:
# run_shifter.sh exports LORRAX_NVHPC_ROOT derived from the SAME
# LORRAX_NVHPC_SUBPATH that decides the runtime library, so a build launched
# through it agrees with the run it is built for BY CONSTRUCTION.
NVHPC_ROOT="${LORRAX_NVHPC_ROOT:-}"
NVHPC_CUDA="${LORRAX_NVHPC_CUDA:-12.9}"
NVHPC_MOUNT="${LORRAX_NVHPC_MOUNT:-/lorrax_nvhpc}"

if [[ -z "${NVHPC_ROOT}" ]]; then
    # Second source of truth, same fact: the runtime subpath.  Its first
    # component IS the stage directory.
    if [[ -n "${LORRAX_NVHPC_SUBPATH:-}" ]]; then
        NVHPC_ROOT="${NVHPC_MOUNT}/${LORRAX_NVHPC_SUBPATH%%/*}"
        echo "[build] NVHPC_ROOT derived from LORRAX_NVHPC_SUBPATH='${LORRAX_NVHPC_SUBPATH}'" >&2
    else
        echo "[build] REFUSED — LORRAX_NVHPC_ROOT is not set." >&2
        echo "  rule   this selects the cuSOLVERMp the .so is COMPILED and" >&2
        echo "         LINKED against.  Every stage exports the same SONAME," >&2
        echo "         so building against one and running against another" >&2
        echo "         links cleanly and fails later, as wrong getrf/getrs" >&2
        echo "         answers on a Px>1 AND Py>1 mesh (site_config.sh)." >&2
        echo "  got    neither LORRAX_NVHPC_ROOT nor LORRAX_NVHPC_SUBPATH." >&2
        echo "  wanted the stage that this build's RUNS will load." >&2
        echo "  fix    launch through src/ffi/cpp/run_shifter.sh, which" >&2
        echo "         exports both from config/perlmutter/site_config.sh;" >&2
        echo "         or name it:  LORRAX_NVHPC_ROOT=${NVHPC_MOUNT}/<stage>" >&2
        echo "  note   this script no longer guesses 25.5_cuda12.9" >&2
        echo "         (cuSolverMp 0.6.0).  Stages present under" >&2
        echo "         ${NVHPC_MOUNT}:" >&2
        if [[ -d "${NVHPC_MOUNT}" ]]; then
            for _s in "${NVHPC_MOUNT}"/*/; do
                [[ -d "$_s" ]] || continue
                _s="$(basename "$_s")"
                # Only real stages: a dir with the cuSOLVERMp header tree.
                [[ -f "${NVHPC_MOUNT}/${_s}/math_libs/${NVHPC_CUDA}/targets/x86_64-linux/include/cusolverMp.h" ]] || continue
                if [[ -f "${NVHPC_MOUNT}/${_s}/math_libs/${NVHPC_CUDA}/targets/x86_64-linux/include/cal.h" ]]; then
                    echo "           ${_s}  (has cal.h -> LORRAX_FFI_HAVE_CAL=ON)" >&2
                else
                    echo "           ${_s}  (no cal.h -> needs -DLORRAX_FFI_HAVE_CAL=OFF)" >&2
                fi
            done
        else
            echo "           (${NVHPC_MOUNT} is not mounted here)" >&2
        fi
        exit 2
    fi
fi

# MPI stack sanity check.  The CMake config (see CMakeLists.txt ~L283-305)
# reads LORRAX_MPI_INCLUDE_DIR + LORRAX_MPICH_LIB_DIR and silently falls
# back to /opt/hpcx/ompi/lib if either is unset.  That fallback links the
# FFI against HPC-X OpenMPI (DT_NEEDED libmpi.so.40), which then races
# the Cray MPICH bind-mount at runtime — see KNOWN_SANDBOX_ERRORS.md
# 2026-05-10.  This script is normally invoked via run_shifter.sh which
# exports both vars; fail loudly when they're missing instead of producing
# a silently-broken .so.  Set LORRAX_FFI_ALLOW_DEFAULT_MPI=1 to opt out
# (e.g. for the OpenMPI build path on non-Cray sites).
if [[ -z "${LORRAX_FFI_ALLOW_DEFAULT_MPI:-}" ]]; then
    if [[ -z "${LORRAX_MPI_INCLUDE_DIR:-}" || -z "${LORRAX_MPICH_LIB_DIR:-}" ]]; then
        echo "[build] ERROR: LORRAX_MPI_INCLUDE_DIR / LORRAX_MPICH_LIB_DIR not set." >&2
        echo "[build]   include='${LORRAX_MPI_INCLUDE_DIR:-<unset>}'" >&2
        echo "[build]   libdir ='${LORRAX_MPICH_LIB_DIR:-<unset>}'"   >&2
        echo "[build] Without these CMake falls back to HPC-X OpenMPI at /opt/hpcx/ompi" >&2
        echo "[build] and the produced .so will ask for libmpi.so.40 at run time —" >&2
        echo "[build] the wrong MPI library name for Cray MPICH, which segfaults" >&2
        echo "[build] the runtime path" >&2
        echo "[build] (see KNOWN_SANDBOX_ERRORS.md 2026-05-10)." >&2
        echo "[build] Invoke via src/ffi/cpp/run_shifter.sh, which exports both," >&2
        echo "[build] or set LORRAX_FFI_ALLOW_DEFAULT_MPI=1 to bypass this check." >&2
        exit 2
    fi
fi

echo "[build] NVHPC_ROOT = ${NVHPC_ROOT}"
echo "[build] CUDA sub   = ${NVHPC_CUDA}"
echo "[build] sources    = ${SCRIPT_DIR}"
echo "[build] build dir  = ${BUILD_DIR}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [[ "${1:-}" == "--fresh" ]]; then
    echo "[build] --fresh: wiping ${BUILD_DIR}"
    rm -rf "${BUILD_DIR:?}/"* "${BUILD_DIR:?}/".??* 2>/dev/null || true
fi

# Force the container's Python (/usr/bin/python3) so we pick up the
# jaxlib headers that match the runtime (/opt/jaxlibs), not some external
# virtualenv's jaxlib that happens to be on PATH first.  The XLA FFI API
# version is baked into the header — headers and runtime must match.
PYTHON_EXE="${LORRAX_FFI_PYTHON:-/usr/bin/python3}"

# CAL vs NCCL comm path must MATCH the stage selected above.  cuSOLVERMp
# >= 0.7 is NCCL-native and ships no cal.h/libcal; <= 0.6.x needs both.
# CMake's LORRAX_FFI_HAVE_CAL defaults ON, so the 0.7.x stage with no flag
# fails deep inside the compile on a missing cal.h — a confusing error for
# a decision that is fully determined by the tree we just chose.  Check it
# here, where the tree is in hand, and name the flag.
CAL_INC="${NVHPC_ROOT}/math_libs/${NVHPC_CUDA}/targets/x86_64-linux/include/cal.h"
CAL_ARGS=()
if [[ -n "${LORRAX_FFI_HAVE_CAL:-}" ]]; then
    # Explicit wins, unexamined: a port may know something this check does not.
    CAL_ARGS=(-DLORRAX_FFI_HAVE_CAL="${LORRAX_FFI_HAVE_CAL}")
    echo "[build] CAL comm path: LORRAX_FFI_HAVE_CAL=${LORRAX_FFI_HAVE_CAL} (explicit)"
elif [[ ! -f "${CAL_INC}" ]]; then
    echo "[build] REFUSED — the selected cuSOLVERMp stage ships no cal.h." >&2
    echo "  rule   the CAL and NCCL comm paths are different code; the" >&2
    echo "         stage decides which one is correct, not the default." >&2
    echo "  got    NVHPC_ROOT=${NVHPC_ROOT}" >&2
    echo "         with no ${CAL_INC}" >&2
    echo "         (cuSOLVERMp >= 0.7 is NCCL-native and drops CAL)," >&2
    echo "         but CMake's LORRAX_FFI_HAVE_CAL defaults ON." >&2
    echo "  fix    rebuild with LORRAX_FFI_HAVE_CAL=OFF, i.e." >&2
    echo "           LORRAX_FFI_HAVE_CAL=OFF bash src/ffi/cpp/build.sh" >&2
    echo "         or select a stage that ships cal.h (25.5_cuda12.9 is" >&2
    echo "         cuSolverMp 0.6.0 — see site_config.sh for why you" >&2
    echo "         probably do not want it)." >&2
    exit 2
else
    echo "[build] CAL comm path: cal.h present in the stage -> HAVE_CAL=ON"
fi

cmake "${SCRIPT_DIR}" \
    -DLORRAX_FFI_PLATFORM=cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="${PYTHON_EXE}" \
    -DNVHPC_ROOT="${NVHPC_ROOT}" \
    -DNVHPC_CUDA_SUBDIR="${NVHPC_CUDA}" \
    "${CAL_ARGS[@]}"

cmake --build . --parallel

echo
echo "[build] --- artifacts ---"
ls -lh "${BUILD_DIR}"/liblorrax_ffi*.so 2>/dev/null || {
    echo "[build] FAILED: no .so produced."
    exit 1
}

SO_FILE=$(ls "${BUILD_DIR}"/liblorrax_ffi*.so 2>/dev/null | head -1)
echo
echo "[build] --- ldd check ---"
ldd "${SO_FILE}" | grep -Ei 'cusolver|nccl|cuda|cal' || true

# ===========================================================================
# THE BUILD CONTRACT.  Unconditional, and a failure here fails the build.
#
# This leg used to call gate_one_mpi.sh and gate_one_hdf5.sh directly and
# nothing else; scripts/verify_ffi_build.sh runs those two plus the backend
# declaration, the load-resolution probe and the ABI stamp, and is the same
# file the host leg and the pytest tier use.  See its header for why one
# script rather than five copies.
#
# No expected MPI variant is pinned: inside the Shifter container the ONLY
# cray-mpich available is shifter's --module=mpich bind-mount (8.1.25 /
# libmpi_gnu_91), and the phdf5 stage deliberately aliases
# libmpi_gnu_123.so.12 onto it so HDF5 and the FFI share ONE runtime.  What
# must hold is one mapped object, not a version.
#
# On this leg the phdf5 stage is what we built against, so GATE 7's want/have
# half is near-tautological here — what it catches is a STALE build dir (a
# CMakeCache pinned to a stage since re-populated from a different module) and
# a stage carrying two HDF5 majors at once.  The CROSS-LEG half is enforced on
# the host side, where the two populations can actually be compared; point
# LORRAX_FFI_EXPECT_PEER_SO at the host .so to run it from here too.
# ===========================================================================
echo
echo "[build] --- build contract (scripts/verify_ffi_build.sh) ---"
LORRAX_ROOT="${LORRAX_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
LORRAX_PHDF5_STAGE="${LORRAX_PHDF5_STAGE:-${LORRAX_PHDF5_MOUNT:-/lorrax_phdf5}}" \
    bash "${LORRAX_ROOT}/scripts/verify_ffi_build.sh" --leg cuda "${SO_FILE}" || {
        echo "[build] FAILED: the library does not meet the build contract." >&2
        echo "[build] The .so is left on disk so it can be inspected, but it" >&2
        echo "[build] must not be deployed or pinned.  Read the gate output" >&2
        echo "[build] above; every failure names its own fix." >&2
        exit 1
    }

# GATE 9: NO LORRAX-OWNED INTERNAL IS ON THE DYNAMIC TABLE.
#
# The device twin of the gate in config/perlmutter/build_ffi_host.sh, which
# carries the full argument.  Both platform .so's are dlopened RTLD_GLOBAL
# into one process and ld.so answers a name from the FIRST object that
# defined it -- including for the OTHER library's internal calls.  Measured
# 2026-08-07 on the pinned pair: 259 defined names in common, 25 of them
# LORRAX's own, among them `lorrax_ffi::phdf5::open_ctx` -- whose `PhdfCtx`
# return type has a different LAYOUT in the two builds.  That is
# KNOWN_FAILURES L1.  src/ffi/cpp/exports_cuda.map localises them; this gate
# notices the day it falls off the link line.
#
# TWO NAMES ARE EXEMPT, and exports_cuda.map's `global:` clause is where the
# reason is written: common/build_config.cc joined this leg's sources on
# 2026-08-08 and its two entry points are `lorrax_ffi_cuda_*`, so the pattern
# below would take them.  They are declared APIs read from outside the
# library, and their host twins carry `_host`, so nothing collides.
echo "[build] GATE 9 (no LORRAX internal on the dynamic table)"
_leaked=$(nm -D --defined-only "${SO_FILE}" 2>/dev/null | awk '{print $NF}' \
          | grep -E 'lorrax_ffi' \
          | grep -vE '^lorrax_ffi_cuda_(build_config|abi_version)$' || true)
if [ -n "$(printf %s "${_leaked}" | tr -d '[:space:]')" ]; then
    echo "[build] GATE FAILED (9): LORRAX-owned internals are on the" >&2
    echo "[build]   dynamic table.  liblorrax_ffi_host.so defines these" >&2
    echo "[build]   names too, and in a process with both open the first" >&2
    echo "[build]   one loaded answers them for BOTH." >&2
    printf '%s\n' "${_leaked}" | head -20 | sed 's/^/    /' >&2
    echo "[build]   Cause: -Wl,--version-script=exports_cuda.map is not on" >&2
    echo "[build]   the link line (check CMakeCache.txt)." >&2
    exit 1
fi
echo "[build] GATE 9 PASSED: 0 lorrax_ffi internals exported"

# Build provenance beside the artifact.  ffi_loader.build_provenance()
# prints it in every run's startup report; without it the loader can only say
# "NO PROVENANCE FILE (pre-stamp build)".  On 2026-08-05 a 4-node GPU log was
# analysed against an on-disk liblorrax_ffi.so that had been rebuilt SEVEN
# MINUTES after that log's last write, and nothing on disk recorded the fact.
"${SCRIPT_DIR}/stage/stamp_provenance.sh" "${SO_FILE}" \
    "leg=cuda" \
    "nvhpc_cuda=${NVHPC_CUDA}" \
    "build_dir=${BUILD_DIR}" || {
        echo "[build] WARNING: provenance stamp failed (build itself is fine)" >&2
    }

echo
echo "[build] done.  .so at: ${SO_FILE}"
