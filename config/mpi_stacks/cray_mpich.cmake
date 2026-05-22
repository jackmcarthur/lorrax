# config/mpi_stacks/cray_mpich.cmake
# Single source of truth for the Cray MPICH stack constants.
# Consumed by CMakeLists.txt (include() this file) and read at build time.
# The companion cray_mpich.sh file exports the same values for shell consumers
# (run_shifter.sh, 0.1.0.lua via os.getenv).
#
# --mpi=cray_shasta is the PMI protocol that Shifter's --module=mpich libmpi
# speaks.  pmi2/pmix both produce singleton MPI_COMM_WORLDs on Perlmutter
# (each rank sees world_size==1) — observed while bringing up the SLATE FFI.
set(LORRAX_MPI_STACK_NAME       "cray_mpich")
set(LORRAX_MPI_TYPE_DEFAULT     "cray_shasta")     # --mpi= arg to srun
set(LORRAX_SHIFTER_MODULES      "gpu,mpich")
set(LORRAX_MPI_LIB_DIR_CT       "/opt/udiImage/modules/mpich")
set(LORRAX_MPI_INCLUDE_DIR_CT   "/lorrax_phdf5/include")
set(LORRAX_GTL_PRELOAD          "/lorrax_slate/lib/libmpi_gtl_cuda.so.0")
