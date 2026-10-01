# shellcheck shell=bash
# ============================================================================
# The ONE MPI every Perlmutter FFI build links: sourced by BOTH legs.
#   host leg: config/perlmutter/build_ffi_host.sh
#   CUDA leg: config/perlmutter/build_ffi_cuda.sh
#   SLATE:    src/ffi/cpp/stage/slate_build_perlmutter.sh (after PrgEnv-gnu)
#
# In a GPU run both legs are dlopened into one process, so they must name the
# same libmpi.  cray-hdf5-parallel/1.14.3.7 (the phdf5 stage both legs share)
# and the SLATE host install carry DT_NEEDED libmpi_gnu_123.so.12, which is
# cray-mpich 9.0.1.  So everything pins to 9.0.1:
#   - cray-mpich/9.0.1;
#   - the LibSci that links it, cray-libsci/25.09.0.  The site-default 26.03.0
#     links libmpi_gnu.so.12, which is 9.1.0.
#
# Measured 2026-09-24: an unpinned host leg took the site defaults
# (cray-mpich 9.1.0, cray-libsci 26.03.0) plus the darshan module's
# libdarshan.so.0.  It linked libmpi_gnu.so.12 beside HDF5's
# libmpi_gnu_123.so.12, which puts two MPIs in one process, while the CUDA leg
# linked 9.0.1 only.  GATE 1's regex missed the 9.1.0 SONAME, so both legs
# passed their own gates and the pair could not be sealed.
#
# To move to a new MPI, change these values and nothing else.  Every value
# must still agree with the phdf5 stage's and SLATE's DT_NEEDED; GATE 1
# (gate_one_mpi.sh) and gate_one_odr.py (Gate 10) enforce that.
# ============================================================================
LORRAX_PM_MPICH_MODULE="cray-mpich/9.0.1"
LORRAX_PM_LIBSCI_MODULE="cray-libsci/25.09.0"
LORRAX_PM_MPICH_ROOT="/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3"
LORRAX_PM_MPI_LIBRARY="$LORRAX_PM_MPICH_ROOT/lib/libmpi_gnu_123.so"
LORRAX_PM_MPI_SONAME="libmpi_gnu_123.so.12"

# Load the pinned MPI and unload darshan.  The Cray CC wrapper links the
# loaded cray-mpich implicitly and injects libdarshan.so.0 whenever the darshan
# module is loaded, so both steps are needed even though the MPI library is
# also named explicitly.
lorrax_pm_pin_mpi() {
    module load "$LORRAX_PM_MPICH_MODULE"
    module unload darshan 2>/dev/null || true
}

# GATE 1 for a library the sealed bundle preloads by absolute path (the private
# SLATE closure).  It resolves its machine libraries through its own DT_RPATH,
# then LD_LIBRARY_PATH, where a run has /opt/cray/pe/lib64: the site-default
# LibSci, whose 26.03 links cray-mpich 9.1.0.  So the gate runs under that
# search path and must find only the pinned MPI.  Measured 2026-09-30: B3's
# SLATE (RPATH = its install prefix and darshan) failed this gate, mapped 9.1.0
# beside 9.0.1 in every process, and every CPU run hung in its second MPI_Init.
lorrax_pm_gate_private_lib() {
    local so="$1" fabric
    fabric="$(ldd "$LORRAX_PM_MPI_LIBRARY" | awk '/libfabric/ {print $3; exit}')"
    [[ -n "$fabric" ]] || { echo "cannot resolve libfabric from $LORRAX_PM_MPI_LIBRARY" >&2; return 1; }
    LD_LIBRARY_PATH="/opt/cray/pe/lib64:$(dirname "$fabric")" GATE_TAG="private $(basename "$so")" \
        bash "$(dirname "${BASH_SOURCE[0]}")/../../src/ffi/cpp/gate_one_mpi.sh" \
        "$so" "$LORRAX_PM_MPI_SONAME"
}
