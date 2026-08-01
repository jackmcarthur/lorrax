#!/usr/bin/env bash
# stage_openmpi.sh — stage a conda-forge parallel-HDF5 build (linked
# against OpenMPI 4.x libmpi.so.40) into a /pscratch dir Shifter can
# bind-mount.  Pair with `shifter --module=gpu` (no --module=mpich) so
# the container's HPC-X OpenMPI at /opt/hpcx/ompi satisfies libmpi.
#
# This is the default / verified stack.  See src/ffi/PORTING.md for
# the Cray MPICH alternative (stage_cray.sh), which is currently
# unstable for large collective writes.
#
# Run on a NERSC login node (no container, no GPU):
#
#   src/ffi/cpp/stage/phdf5_stage_openmpi.sh
#
# Output: $LORRAX_FFI_PHDF5_DIR populated with HDF5 + libaec .so's and
# the HDF5 + libaec headers.  Default destination:
#
#   /pscratch/sd/${USER:0:1}/${USER}/lorrax_phdf5_openmpi/stage
#
# Destination has to be under /pscratch (or /global/cfs) because NERSC
# Shifter's udiRoot.conf only accepts --volume sources under those paths.
# A cache of the downloaded .conda files lives next to this script
# (not under /pscratch), so repeat runs don't re-download.

set -euo pipefail

: "${LORRAX_FFI_PHDF5_DIR:=/pscratch/sd/${USER:0:1}/${USER}/lorrax_phdf5_openmpi/stage}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cache="${here}/cache"
mkdir -p "${cache}" "${LORRAX_FFI_PHDF5_DIR}"

# conda-forge binaries built against openmpi 4 + libaec (for SZIP).
# Pinned to specific builds so stages are reproducible.  When HDF5 bumps
# to 1.16, update these two URLs; the FFI code itself is API-stable.
HDF5_URL="https://conda.anaconda.org/conda-forge/linux-64/hdf5-1.14.6-mpi_openmpi_h8367ee7_8.conda"
AEC_URL="https://conda.anaconda.org/conda-forge/linux-64/libaec-1.1.3-h59595ed_0.conda"

fetch() {
    local url="$1" target="$2"
    if [[ ! -f "${target}" ]]; then
        echo "[stage] fetch $(basename "${target}")"
        curl -sSL -o "${target}" "${url}"
    fi
}
fetch "${HDF5_URL}" "${cache}/hdf5.conda"
fetch "${AEC_URL}" "${cache}/libaec.conda"

# .conda files are zip archives containing pkg-*.tar.zst (the payload)
# and info-*.tar.zst (metadata).  We only need pkg-*.
extract_conda() {
    local conda_file="$1" dst="$2"
    local tmp; tmp="$(mktemp -d)"
    unzip -o -q "${conda_file}" -d "${tmp}"
    zstd -d --stdout "${tmp}"/pkg-*.tar.zst | tar -x -C "${dst}"
    rm -rf "${tmp}"
}

echo "[stage] extract HDF5 → ${LORRAX_FFI_PHDF5_DIR}"
extract_conda "${cache}/hdf5.conda"   "${LORRAX_FFI_PHDF5_DIR}"
echo "[stage] extract libaec → ${LORRAX_FFI_PHDF5_DIR}"
extract_conda "${cache}/libaec.conda" "${LORRAX_FFI_PHDF5_DIR}"

echo "[stage] done.  libhdf5:"
readelf -d "${LORRAX_FFI_PHDF5_DIR}/lib/libhdf5.so" | \
    grep -E "SONAME|NEEDED" | head -8
echo
echo "[stage] dst: ${LORRAX_FFI_PHDF5_DIR}"
echo "[stage] run_shifter.sh picks this up automatically when"
echo "[stage] LORRAX_PHDF5_MPI_STACK=openmpi (the default)."
