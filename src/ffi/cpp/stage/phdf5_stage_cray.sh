#!/usr/bin/env bash
# stage_cray.sh — populate a staging dir (default $HOME/software, see
# LORRAX_FFI_PHDF5_DIR below) with a copy of the host's cray-hdf5-parallel
# module, plus a shim symlink that lets it load inside Shifter when paired
# with --module=mpich.
#
# Why not just bind-mount /opt/cray directly?  Shifter at NERSC blocks
# --volume from /opt/ system paths (not in udiRoot.conf siteFs).  $HOME
# (/global/homes -> /global/u1) and /global/common ARE valid siteFs
# --volume sources (verified 2026-06-29), so we stage to $HOME/software —
# off the purged $SCRATCH — and bind-mount from there.
#
# Relocated to $HOME/software 2026-06-24 (was /pscratch).  Historical
# note: we used to ship `stage_openmpi.sh` as default because Cray
# MPICH's collective-write path (`ad_cray_write_coll.c:669`) OOMs at
# ≥ 1 GB/rank.  That's worked around now by forcing independent writes
# (LORRAX_PHDF5_INDEPENDENT=1 defaults) and non-collective metadata
# (LORRAX_PHDF5_COLL_META=0 defaults) — see `src/ffi/cpp/phdf5/ctx.h`.
# stage_openmpi.sh is kept as a fallback for non-Cray clusters.
#
# Run on a NERSC login node, after `module load cray-hdf5-parallel
# cray-mpich` (so $HDF5_DIR and $MPICH_DIR are set).  No container, no GPU:
#
#   module load cray-hdf5-parallel cray-mpich
#   src/ffi/cpp/stage/phdf5_stage_cray.sh
#
# Then use with `LORRAX_PHDF5_MPI_STACK=mpich` in run_shifter.sh
# (the module's `lxrun` uses this stack by default).

set -euo pipefail

: "${LORRAX_FFI_PHDF5_DIR:=$HOME/software/lorrax_phdf5_cray/stage}"

# The cray-hdf5-parallel module's install root, and the cray-mpich root
# for the mpi.h the FFI build needs (the shifter mpich module is
# runtime-only and ships no headers).
#
# BOTH REFUSE WHEN UNSET.  Until 2026-08-06 each fell back to a path
# observed on Perlmutter in April 2026 —
#     ${HDF5_DIR:-/opt/cray/pe/hdf5-parallel/1.12.2.9/gnu/12.3}
#     ${MPICH_DIR:-/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3}
# — which is the "quietly substitute a default for a missing environment
# fact" shape, in the one place where it does the most damage.  What gets
# staged here is what every later build LINKS against, and the resulting
# .so does not fail at link time when the guess is wrong; it fails much
# later, as a wrong answer or a hang, with nothing on disk recording that
# a different HDF5 was substituted.  The guess is also demonstrably stale:
# config/perlmutter/build_ffi_host.sh loads cray-hdf5-parallel/1.14.3.7,
# so a run of THIS script with no module loaded would stage 1.12 under a
# tree the host build fills with 1.14.  Staging against the wrong HDF5 is
# exactly the class that "links cleanly, fails later".
#
# The fix is one command, and it is in the usage block above:
#   module load cray-hdf5-parallel cray-mpich
# Or name the trees outright with CRAY_HDF5_PATH / CRAY_MPICH_PATH, which
# still take precedence — an explicit path is a stated fact, not a guess.
if [[ -z "${CRAY_HDF5_PATH:-}" && -z "${HDF5_DIR:-}" ]]; then
    echo "phdf5_stage_cray.sh: REFUSED — HDF5_DIR is not set." >&2
    echo "  rule   the staged HDF5 is what every later FFI build links" >&2
    echo "         against; a substituted default links cleanly and fails" >&2
    echo "         later, as a wrong answer or a hang." >&2
    echo "  got    neither HDF5_DIR nor CRAY_HDF5_PATH in the environment." >&2
    echo "  wanted HDF5_DIR, as exported by the module." >&2
    echo "  fix    module load cray-hdf5-parallel cray-mpich" >&2
    echo "         (or set CRAY_HDF5_PATH=<hdf5 install root> explicitly)." >&2
    echo "  note   this script no longer guesses" >&2
    echo "         /opt/cray/pe/hdf5-parallel/1.12.2.9/gnu/12.3; the host" >&2
    echo "         FFI build uses cray-hdf5-parallel/1.14.3.7." >&2
    exit 2
fi
: "${CRAY_HDF5_PATH:=${HDF5_DIR}}"

if [[ -z "${CRAY_MPICH_PATH:-}" && -z "${MPICH_DIR:-}" ]]; then
    echo "phdf5_stage_cray.sh: REFUSED — MPICH_DIR is not set." >&2
    echo "  rule   the mpi.h staged here is the ABI the FFI compiles" >&2
    echo "         against; the wrong one links and then corrupts or hangs" >&2
    echo "         at the first collective (hazard S3)." >&2
    echo "  got    neither MPICH_DIR nor CRAY_MPICH_PATH in the" >&2
    echo "         environment." >&2
    echo "  wanted MPICH_DIR, as exported by the module." >&2
    echo "  fix    module load cray-mpich" >&2
    echo "         (or set CRAY_MPICH_PATH=<mpich install root>)." >&2
    exit 2
fi
: "${CRAY_MPICH_PATH:=${MPICH_DIR}}"

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
