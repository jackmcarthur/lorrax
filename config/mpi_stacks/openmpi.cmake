# config/mpi_stacks/openmpi.cmake
# TODO: not yet wired; see Blitz #3 follow-up.
# Placeholder for the future Apptainer port (Frontier, Polaris, non-Cray
# clusters).  Wire this when Apptainer companion image work is scheduled.
#
# Expected values (not yet validated):
#   LORRAX_MPI_STACK_NAME      "openmpi"
#   LORRAX_MPI_TYPE_DEFAULT    "pmix"
#   LORRAX_SHIFTER_MODULES     "gpu"
#   LORRAX_MPI_LIB_DIR_CT      "/opt/hpcx/ompi/lib"
#   LORRAX_MPI_INCLUDE_DIR_CT  "/opt/hpcx/ompi/include"
#   LORRAX_GTL_PRELOAD         ""   (UCX handles GPU-Direct; no gtl shim)
