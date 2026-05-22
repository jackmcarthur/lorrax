# config/mpi_stacks/cray_mpich.sh
# Single source of truth for the Cray MPICH stack constants.
# Sourced by run_shifter.sh and read by 0.1.0.lua via os.getenv().
# The companion cray_mpich.cmake file contains the same values for CMake.
#
# --mpi=cray_shasta is the PMI protocol that Shifter's --module=mpich libmpi
# speaks.  pmi2/pmix both produce singleton MPI_COMM_WORLDs on Perlmutter
# (each rank sees world_size==1) — observed while bringing up the SLATE FFI.
export LORRAX_MPI_STACK_NAME="cray_mpich"
export LORRAX_MPI_TYPE_DEFAULT="cray_shasta"
export LORRAX_SHIFTER_MODULES="gpu,mpich"
export LORRAX_MPI_LIB_DIR_CT="/opt/udiImage/modules/mpich"
export LORRAX_MPI_INCLUDE_DIR_CT="/lorrax_phdf5/include"
export LORRAX_GTL_PRELOAD="/lorrax_slate/lib/libmpi_gtl_cuda.so.0"
