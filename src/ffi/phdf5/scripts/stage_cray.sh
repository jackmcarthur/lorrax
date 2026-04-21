#!/usr/bin/env bash
# stage_cray.sh — populate a /pscratch dir with a copy of the host's
# cray-hdf5-parallel module, plus a shim symlink that lets it load
# inside Shifter when paired with --module=mpich.
#
# Why not just bind-mount /opt/cray directly?  Shifter at NERSC blocks
# --volume from /opt/ system paths.  Why not bind-mount from $HOME?
# Shifter's udiRoot.conf on Perlmutter only accepts --volume sources
# under /pscratch (and a handful of other paths).  So we copy the
# module to /pscratch, where Shifter is willing to bind-mount it.
#
# This is the default staged-deps target as of 2026-04-20.  Historical
# note: we used to ship `stage_openmpi.sh` as default because Cray
# MPICH's collective-write path (`ad_cray_write_coll.c:669`) OOMs at
# ≥ 1 GB/rank.  That's worked around now by forcing independent writes
# (LORRAX_PHDF5_INDEPENDENT=1 defaults) and non-collective metadata
# (LORRAX_PHDF5_COLL_META=0 defaults) — see `src/ffi/phdf5/cpp/ctx.h`.
# stage_openmpi.sh is kept as a fallback for non-Cray clusters.
#
# Run on a NERSC login node, after `module load cray-hdf5-parallel
# cray-mpich` (so $HDF5_DIR and $MPICH_DIR are set).  No container, no GPU:
#
#   module load cray-hdf5-parallel cray-mpich
#   src/ffi/phdf5/scripts/stage_cray.sh
#
# Then use with `LORRAX_PHDF5_MPI_STACK=mpich` in run_shifter.sh
# (the module's `lxrun` uses this stack by default).

set -euo pipefail

: "${LORRAX_FFI_PHDF5_DIR:=/pscratch/sd/${USER:0:1}/${USER}/lorrax_phdf5_cray/stage}"

# The cray-hdf5-parallel module's install root.  If env sets HDF5_DIR
# (from `module load cray-hdf5-parallel`), prefer that.  Otherwise fall
# back to the path observed on Perlmutter as of April 2026.
: "${CRAY_HDF5_PATH:=${HDF5_DIR:-/opt/cray/pe/hdf5-parallel/1.12.2.9/gnu/12.3}}"

# Likewise for cray-mpich headers.  We need mpi.h + friends at build
# time for the FFI; they're not shipped by the shifter mpich module
# (runtime-only).
: "${CRAY_MPICH_PATH:=${MPICH_DIR:-/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3}}"

# Where shifter --module=mpich places the MPICH-ABI libmpi inside the
# container.  The shim symlinks below map cray-pe compiler-specific
# SONAMEs (libmpi_gnu_{91,110,123}.so.12) onto this generic MPICH-ABI
# lib.  The loader follows the symlink at container startup — ABI is
# equivalent because every variant is MPICH 4.x libmpi.so.12 underneath.
SHIM_TARGET="/opt/udiImage/modules/mpich/libmpi.so.12"

if [[ ! -d "${CRAY_HDF5_PATH}" ]]; then
    echo "stage_cray.sh: CRAY_HDF5_PATH=${CRAY_HDF5_PATH} not found."
    echo "  Run 'module load cray-hdf5-parallel' and try again, or set"
    echo "  CRAY_HDF5_PATH explicitly."
    exit 2
fi

echo "[stage] src hdf5:  ${CRAY_HDF5_PATH}"
echo "[stage] src mpich: ${CRAY_MPICH_PATH}"
echo "[stage] dst:       ${LORRAX_FFI_PHDF5_DIR}"
mkdir -p "${LORRAX_FFI_PHDF5_DIR}"

# Copy bin, include, lib (DEREFERENCE symlinks — cray-hdf5-parallel
# uses relative symlinks like ../../../include/hdf5.h that break when
# we copy only the compiler-specific subtree).  ~12 MB total.
for sub in bin include lib; do
    if [[ -d "${CRAY_HDF5_PATH}/${sub}" ]]; then
        cp -rL "${CRAY_HDF5_PATH}/${sub}" "${LORRAX_FFI_PHDF5_DIR}/"
    fi
done

# Copy MPICH headers into the stage include/ so the FFI build can find
# mpi.h.  Small (~1 MB); only C headers, Fortran .mod files skipped.
if [[ -d "${CRAY_MPICH_PATH}/include" ]]; then
    for h in mpi.h mpio.h mpi_proto.h cray_version.h mpi_kt.h; do
        if [[ -f "${CRAY_MPICH_PATH}/include/${h}" ]]; then
            cp "${CRAY_MPICH_PATH}/include/${h}" \
               "${LORRAX_FFI_PHDF5_DIR}/include/"
        fi
    done
fi

# Shim: every cray-pe libmpi SONAME that libhdf5.so could NEED points
# at the shifter mpich module's generic libmpi.so.12.
for soname in libmpi_gnu_91.so.12 libmpi_gnu_110.so.12 libmpi_gnu_123.so.12; do
    ln -sf "${SHIM_TARGET}" "${LORRAX_FFI_PHDF5_DIR}/lib/${soname}"
done

echo "[stage] done."
echo "[stage] inspect libhdf5 NEEDED:"
readelf -d "${LORRAX_FFI_PHDF5_DIR}/lib/libhdf5.so" | \
    grep -E "SONAME|NEEDED" | head -10
echo "[stage] shims:"
ls -la "${LORRAX_FFI_PHDF5_DIR}/lib/" | grep -E "^l.*libmpi" | head -5
echo
echo "[stage] run_shifter.sh picks this up when"
echo "[stage] LORRAX_PHDF5_MPI_STACK=mpich."
