# shellcheck shell=bash
# ============================================================================
# The ONE MPI every Frontier (OLCF) FFI build links, with the LibSci and HDF5
# built against it, and the directories a run resolves them from.  Sourced by
#   host leg: config/frontier/build_ffi_host.sh
#   runtime:  config/frontier/ffi_env.sh
# and, when it exists, by the ROCm leg's build, which must link the same libmpi.
# Frontier twin of config/perlmutter/ffi_mpi.sh.
#
# WHY THESE VERSIONS (measured on Frontier 2026-10-14, readelf -d):
#   cray-mpich 8.1.31 (site default) ... 9.0.1  -> libmpi_gnu_123.so.12
#   cray-mpich 9.1.0                            -> libmpi_gnu.so.12
#   cray-hdf5-parallel 1.12.2.11 .. 1.14.3.7 gnu/12.3 NEED libmpi_gnu_123.so.12;
#     1.14.3.9 NEEDs libmpi_gnu.so.12.
#   cray-libsci 24.03 .. 25.09 GNU/12.3 NEED libmpi_gnu_123.so.12;
#     26.03.0 NEEDs libmpi_gnu.so.12.
# So the stack must be all-123 or all-9.1.0.  The 123 generation is chosen
# because it is also Frontier's site default (/opt/cray/pe/lib64 and the
# default modules map libmpi_gnu_123.so.12), so a library that falls back to
# a site default still names the same MPI SONAME.  Within it, the versions are
# the ones the host leg is validated with on Perlmutter (ffi_mpi.sh there);
# all three exist on Frontier at byte-identical /opt/cray/pe paths.
#
# To move to a new MPI, change these values and nothing else.  Every value
# must still agree with the phdf5 module's DT_NEEDED; GATE 1
# (gate_one_mpi.sh) enforces that.
# ============================================================================
LORRAX_OLCF_PRGENV_MODULE="PrgEnv-gnu/8.7.0"          # gcc-native/13.2
LORRAX_OLCF_MPICH_MODULE="cray-mpich/9.0.1"
LORRAX_OLCF_LIBSCI_MODULE="cray-libsci/25.09.0"
LORRAX_OLCF_HDF5_MODULE="cray-hdf5-parallel/1.14.3.7"
LORRAX_OLCF_FFTW_MODULE="cray-fftw/3.3.10.11"
LORRAX_OLCF_CMAKE_MODULE="cmake/3.31.11"
LORRAX_OLCF_MPICH_ROOT="/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3"
LORRAX_OLCF_MPI_LIBRARY="$LORRAX_OLCF_MPICH_ROOT/lib/libmpi_gnu_123.so"
LORRAX_OLCF_MPI_SONAME="libmpi_gnu_123.so.12"
LORRAX_OLCF_LIBSCI_ROOT="/opt/cray/pe/libsci/25.09.0/GNU/12.3/x86_64"
LORRAX_OLCF_HDF5_ROOT="/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.3"

# The legs' only search path is $ORIGIN (a RUNPATH), so the run environment
# names these directories, ahead of /opt/cray/pe/lib64 (site defaults:
# cray-mpich 8.1.31, libsci 24.11.0, hdf5 1.14.3.3).
LORRAX_OLCF_RUNTIME_LIB_DIRS="$LORRAX_OLCF_HDF5_ROOT/lib:$LORRAX_OLCF_MPICH_ROOT/lib:$LORRAX_OLCF_LIBSCI_ROOT/lib"

# Load the pinned MPI and unload darshan.  On Frontier the module is
# darshan-runtime (Perlmutter: darshan); with it loaded the Cray wrappers inject
# libdarshan.so into every link.
lorrax_olcf_pin_mpi() {
    module load "$LORRAX_OLCF_MPICH_MODULE"
    module unload darshan-runtime 2>/dev/null || true
}
