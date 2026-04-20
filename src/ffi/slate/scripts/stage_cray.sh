#!/usr/bin/env bash
# stage_cray.sh — populate a /pscratch dir with copies of the Cray runtime
# libs that SLATE's Cray-built .so's NEED but aren't in nvcr.io/nvidia/jax.
#
# Why:
#   SLATE is built against the host Cray PE (CC=cc CXX=CC FC=ftn) so it can
#   use libsci for optimal BLAS/LAPACK and GPU-aware MPI via libmpi_gtl_cuda.
#   Inside the Shifter container those libs live under /opt/cray, which
#   Shifter forbids as a --volume source (NERSC siteFs only allows /pscratch
#   /global/* /dvs_*).  The fix is the same as phdf5's stage_cray.sh: copy
#   the libs once into /pscratch and bind-mount at runtime.
#
# Run on a login node with the needed modules loaded:
#
#   module load cray-libsci cray-mpich cray-pmi
#   src/ffi/slate/scripts/stage_cray.sh
#
# Then use with `LORRAX_SLATE_STACK=cray` in run_shifter.sh (bind-mounted
# at /lorrax_slate inside the container).

set -euo pipefail

: "${LORRAX_FFI_SLATE_DIR:=/pscratch/sd/${USER:0:1}/${USER}/lorrax_slate_cray/stage}"

mkdir -p "${LORRAX_FFI_SLATE_DIR}/lib"

# The libs SLATE's host-built .so NEEDs that aren't in nvcr/jax.  Each is a
# dereferenced copy (-L) because the CRAY_PE paths use relative symlinks
# that'd dangle when lifted out of /opt/cray/pe/.
#
# libdarshan.so.0 lives under /global/common (already in shifter siteFs) so
# isn't staged here; we just add that path to LD_LIBRARY_PATH at runtime.
# libmpi_gnu_123.so.12 is shimmed by the phdf5 stage dir.

copy_lib () {
    local src="$1"
    local dst="${LORRAX_FFI_SLATE_DIR}/lib/$(basename "${src}")"
    if [[ -e "${src}" ]]; then
        cp -L "${src}" "${dst}"
        echo "[stage]   ${src}"
    else
        echo "[stage]   MISSING ${src}" >&2
        exit 2
    fi
}

# Cray libsci (BLAS/LAPACK/ScaLAPACK) — parallel + serial variants
copy_lib "/opt/cray/pe/lib64/libsci_gnu_mpi_mp.so.6"
copy_lib "/opt/cray/pe/lib64/libsci_gnu_mp.so.6"

# GPU-aware MPI transport
copy_lib "/opt/cray/pe/mpich/9.0.1/gtl/lib/libmpi_gtl_cuda.so.0"

# Cray-specific runtime deps (profiling / XPMEM shared memory / Lustre)
copy_lib "/usr/lib64/libxpmem.so.0"
copy_lib "/usr/lib64/liblustreapi.so.1"

# liblustreapi transitively NEEDs libreadline.so.7, which nvcr/jax doesn't
# have (it ships libreadline.so.8).  SLATE doesn't actually call any
# readline API paths (lustreapi is linked by the Cray wrapper but
# referenced only for file-striping queries we don't exercise).  Shim 7->8:
# ABI-compat for the tiny surface that gets pulled in at library init.
# Absolute target — resolved relative to container root at runtime, dangles
# harmlessly on host (host SUSE uses /usr/lib64, not /usr/lib/x86_64-linux-gnu).
ln -sf "/usr/lib/x86_64-linux-gnu/libreadline.so.8" "${LORRAX_FFI_SLATE_DIR}/lib/libreadline.so.7"
echo "[stage]   shim libreadline.so.7 -> /usr/lib/x86_64-linux-gnu/libreadline.so.8"

echo "[stage] staged libs in ${LORRAX_FFI_SLATE_DIR}/lib:"
ls -la "${LORRAX_FFI_SLATE_DIR}/lib/"
echo
echo "[stage] run_shifter.sh picks this up when LORRAX_SLATE_STACK=cray."
echo "[stage] libdarshan.so.0 is NOT staged — reachable via siteFs at"
echo "[stage]   /global/common/software/nersc9/darshan/default/lib"
echo "[stage] libmpi_gnu_123.so.12 uses phdf5's existing shim."
