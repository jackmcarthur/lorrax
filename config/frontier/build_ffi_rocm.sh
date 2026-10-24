#!/bin/bash
# Build liblorrax_ffi_rocm.so, the ROCM leg (docs/architecture/ffi_layout.md,
# "Platform legs"), on OLCF Frontier (MI250X, gfx90a).
#
#   bash config/frontier/build_ffi_rocm.sh [--fresh]
#
# One MPI and one HDF5 per process (docs/installation/ffi-build.md): the ROCM
# leg is built on the SAME module stack as the host leg on this site
# (PrgEnv-gnu, cray-mpich, cray-hdf5-parallel), and against the ROCm release
# whose HIP runtime the JAX plugin loads (pip rocm-sdk 7.14.1 == module
# rocm/7.14.1), so the process maps one libamdhip64.so.7.  The Cray wrapper
# CC supplies the HDF5 and MPI include paths, as on the host leg.
# Overrides: LORRAX_OLCF_VENV, LORRAX_OLCF_ROCM, LORRAX_OLCF_MPICH,
# LORRAX_OLCF_HDF5, LORRAX_ROCM_BUILD_DIR.
set -eo pipefail
ulimit -c 0
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=${LORRAX_OLCF_VENV:-$HOME/envs/lx-jax091-rocm7141}
ROCM_MOD=${LORRAX_OLCF_ROCM:-rocm/7.14.1}
MPICH_MOD=${LORRAX_OLCF_MPICH:-cray-mpich/9.0.1}
HDF5_MOD=${LORRAX_OLCF_HDF5:-cray-hdf5-parallel/1.14.3.7}
BUILD=${LORRAX_ROCM_BUILD_DIR:-$ROOT/src/ffi/cpp/build_rocm}

export LMOD_PAGER=none
module load PrgEnv-gnu/8.7.0 "$MPICH_MOD" "$HDF5_MOD" "$ROCM_MOD" cmake cray-python/3.12.12
# No wrapper-injected LibSci: PrgEnv-gnu loads the site-default cray-libsci and
# the CC wrapper would add its threaded pair to the link.  The leg's one host
# BLAS is the sequential, locking OpenBLAS (build_openblas_seq.sh), which libelpa
# links and resolves from its own prefix.
module unload cray-libsci 2>/dev/null || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"         # cmake probes jax.ffi.include_dir() from here

[ "${1:-}" = "--fresh" ] && rm -rf "$BUILD"
# ELPA, the ROCM distributed eigh (config/frontier/build_elpa_hip.sh): its
# content-keyed prefix, built on demand (login node, ~4 min; a no-op verify after).
# Prefix selection and build must see the same explicit controls, while module
# state and library search paths stay out of the child. ELPA initializes its
# own pinned module environment; a login shell would reintroduce user state.
# The system profile alone (no user dotfiles) supplies the site MODULEPATH:
# under a bare env -i, Lmod's init leaves MODULEPATH undefined and every module
# is unknown on Frontier ("Lmod Warning: MODULEPATH is undefined").
_build_elpa() {
    local name
    local child=(HOME="$HOME" PATH=/usr/bin:/bin)
    [[ -z "${USER+x}" ]] || child+=(USER="$USER")
    for name in LORRAX_OLCF_VENV LORRAX_OLCF_ROCM LORRAX_ROCM_BLAS_DIR \
                LORRAX_ELPA_ROOT LORRAX_ELPA_GFX LORRAX_ELPA_OPENMP LORRAX_ELPA_MAKE_J; do
        [[ -z "${!name+x}" ]] || child+=("$name=${!name}")
    done
    # $0 stays "bash" and the startup flags stay default: Frontier's profile
    # scripts detect the shell from them and skip Lmod otherwise.
    env -i "${child[@]}" bash -c \
        'p=$1; shift; source "$p" >/dev/null 2>&1; exec bash "$@"' bash \
        "${LORRAX_OLCF_SYSTEM_PROFILE:-/etc/profile}" \
        "$ROOT/config/frontier/build_elpa_hip.sh" "$@"
}
ELPA_DIR=${LORRAX_ELPA_ROCM_DIR:-$(_build_elpa --print-prefix)}
[ -e "$ELPA_DIR/LORRAX_ELPA_COMPLETE" ] || _build_elpa || {
    echo "[build_ffi_rocm] FAILED: ELPA" >&2; exit 1; }
cmake -S "$ROOT/src/ffi/cpp" -B "$BUILD" \
    -DLORRAX_FFI_PLATFORM=rocm \
    -DLORRAX_ELPA_ROCM_DIR="$ELPA_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=CC \
    -DHDF5_ROOT="$HDF5_DIR" \
    -DLORRAX_MPI_INCLUDE_DIR="$MPICH_DIR/include" \
    -DLORRAX_MPICH_LIB_DIR="$MPICH_DIR/lib"
cmake --build "$BUILD" -j 16
SO="$BUILD/liblorrax_ffi_rocm.so"
echo "[build_ffi_rocm] built $SO"
readelf -d "$SO" | grep -E "NEEDED|RUNPATH|RPATH"
echo "[build_ffi_rocm] exported LORRAX surface:"
nm -D --defined-only "$SO" | grep -E " T (lrx_|lorrax_ffi_|Phdf)" | awk '{print "  " $3}'

# Dependency closure (lxkit.native_provider).  The ROCm libraries resolve from
# the directories the venv's JAX ROCm plugin maps them from (the rocm-sdk
# wheels: one rocBLAS/rocSOLVER/hipBLASLt/RCCL/HIP runtime per process), the
# ELPA prefix (libelpa and its OpenBLAS) from the leg's own RUNPATH, MPI/HDF5 from the
# pinned directories.  The loader opens exactly those files and refuses a
# process that maps another copy of any of them.
# shellcheck disable=SC1091
source "$ROOT/config/frontier/ffi_mpi.sh"
"$VENV/bin/python" "$ROOT/src/ffi/cpp/stage/write_checkout_closure.py" "$SO" \
    --python "$VENV/bin/python" --library-dirs "$LORRAX_OLCF_RUNTIME_LIB_DIRS" || {
        echo "[build_ffi_rocm] FAILED: dependency closure" >&2; exit 1; }
# Acceptance uses the recorded wheel files through the runtime loader, not
# the build module's alternate ROCm search path. No GPU is initialized here;
# a compute-node GPU smoke remains necessary before runtime certification.
for _control in LORRAX_FFI_VERIFY LORRAX_GATE_ONE_MPI LORRAX_GATE_ONE_HDF5; do
    if [[ "${!_control:-on}" == off ]]; then
        echo "[build_ffi_rocm] FAILED: $_control=off disables mandatory acceptance; provenance was not stamped." >&2
        exit 1
    fi
done
LORRAX_FFI_EXPECT_BACKENDS="${LORRAX_FFI_EXPECT_BACKENDS:-phdf5,elpa,kdft}" \
LORRAX_FFI_EXPECT_MPI="${LORRAX_FFI_EXPECT_MPI:-$LORRAX_OLCF_MPI_SONAME}" \
LORRAX_FFI_EXPECT_HDF5_SOVERSION="${LORRAX_FFI_EXPECT_HDF5_SOVERSION:-310}" \
LORRAX_PHDF5_STAGE="$LORRAX_OLCF_HDF5_ROOT" \
LORRAX_GATE_FFTW_PY="$VENV/bin/python" \
LORRAX_FFI_VERIFY_STRICT=1 \
LORRAX_ROOT="$ROOT" \
GATE_TAG=build_ffi_rocm \
    bash "$ROOT/scripts/verify_ffi_build.sh" --leg rocm "$SO" || {
        echo "[build_ffi_rocm] FAILED: the library does not meet the build contract." >&2
        echo "[build_ffi_rocm] The .so is left on disk for inspection; do not pin it." >&2
        exit 1; }
"$VENV/bin/python" "$ROOT/scripts/dep_check.py" ffi "$SO" || {
    echo "[build_ffi_rocm] FAILED: the library resolves dependencies outside the install." >&2; exit 1; }
"$ROOT/src/ffi/cpp/stage/stamp_provenance.sh" "$SO" \
    leg=rocm site=frontier rocm_module="$ROCM_MOD" mpi="$MPICH_MOD" hdf5="$HDF5_MOD" \
    venv="$VENV" \
    closure_sha256="$("$VENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["record_sha256"])' "$SO.closure.json")" \
    || echo "[build_ffi_rocm] WARNING: provenance stamp failed" >&2
