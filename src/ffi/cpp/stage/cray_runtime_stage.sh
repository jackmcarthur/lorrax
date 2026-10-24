#!/usr/bin/env bash
# cray_runtime_stage.sh — populate a staging dir (default
# $HOME/software/lorrax_cray_runtime/stage, see LORRAX_FFI_CRAY_DIR below) with
# copies of the Cray runtime libs a Cray-built FFI leg and GPU-aware MPI NEED
# but nvcr.io/nvidia/jax does not ship.
#
# Why: inside the Shifter container those libs live under /opt/cray, which
# Shifter forbids as a --volume source (NERSC siteFs allows /pscratch
# /global/* /dvs_* but not /opt).  The fix is the same as phdf5's stage: copy
# the libs once into $HOME/software (off purged $SCRATCH) and bind-mount them
# at /lorrax_cray at run time (run_shifter.sh, the lorrax modulefile).
#
# Run on a login node with the needed modules loaded:
#
#   module load cray-libsci cray-mpich cray-pmi
#   src/ffi/cpp/stage/cray_runtime_stage.sh
set -euo pipefail
: "${LORRAX_FFI_CRAY_DIR:=$HOME/software/lorrax_cray_runtime/stage}"
mkdir -p "${LORRAX_FFI_CRAY_DIR}/lib"
# Each is a dereferenced copy (-L) because the CRAY_PE paths use relative
# symlinks that would dangle when lifted out of /opt/cray/pe/.
#
# libdarshan.so.0 lives under /global/common (already in shifter siteFs) so
# isn't staged here; run_shifter.sh adds that path to LD_LIBRARY_PATH.
# libmpi_gnu_123.so.12 is shimmed by the phdf5 stage dir.
copy_lib () {
    local src="$1"
    local dst="${LORRAX_FFI_CRAY_DIR}/lib/$(basename "${src}")"
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
# Cray-specific runtime deps (XPMEM shared memory / Lustre)
copy_lib "/usr/lib64/libxpmem.so.0"
copy_lib "/usr/lib64/liblustreapi.so.1"
# liblustreapi transitively NEEDs libreadline.so.7, which nvcr/jax doesn't have
# (it ships libreadline.so.8).  Nothing calls a readline API (lustreapi is
# linked by the Cray wrapper for file-striping queries we don't exercise).
# Shim 7->8: ABI-compat for the tiny surface pulled in at library init.  The
# absolute target resolves relative to the container root at run time.
ln -sf "/usr/lib/x86_64-linux-gnu/libreadline.so.8" "${LORRAX_FFI_CRAY_DIR}/lib/libreadline.so.7"
echo "[stage]   shim libreadline.so.7 -> /usr/lib/x86_64-linux-gnu/libreadline.so.8"

echo "[stage] staged libs in ${LORRAX_FFI_CRAY_DIR}/lib:"
ls -la "${LORRAX_FFI_CRAY_DIR}/lib/"
