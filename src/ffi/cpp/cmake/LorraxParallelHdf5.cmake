# LorraxParallelHdf5.cmake — parallel HDF5 (HDF5_IS_PARALLEL) and its MPI.
#
# Shared by the host and ROCM legs; the CUDA leg still carries its own copy
# with container defaults (/lorrax_phdf5, HPC-X) and migrates when it can be
# rebuilt and verified.  A MACRO: it runs in the caller's scope, as the inline
# block it replaced did, and sets HDF5_INCLUDE_DIRS, HDF5_LIBRARIES,
# LORRAX_MPI_INCLUDE_DIR and LORRAX_MPI_LIBRARY there.
#
# Inputs: -DHDF5_ROOT / $HDF5_ROOT (a parallel install; on Cray, load
# cray-hdf5-parallel), -DLORRAX_MPI_INCLUDE_DIR / $LORRAX_MPI_INCLUDE_DIR
# (default: beside the HDF5 headers), -DLORRAX_MPICH_LIB_DIR /
# $LORRAX_MPICH_LIB_DIR (the directory holding libmpi).
macro(lorrax_find_parallel_hdf5)
    set(HDF5_PREFER_PARALLEL ON)
    find_package(HDF5 COMPONENTS C REQUIRED)
    if(NOT HDF5_IS_PARALLEL)
        message(FATAL_ERROR
            "HDF5 at ${HDF5_INCLUDE_DIRS} is NOT built with MPI-IO.  Point "
            "-DHDF5_ROOT / $HDF5_ROOT at a parallel install.")
    endif()
    message(STATUS "HDF5: ${HDF5_VERSION} at ${HDF5_INCLUDE_DIRS} (parallel)")

    # MPI — pinned explicitly (no find_package(MPI); see CUDA CMakeLists note).
    if(NOT DEFINED LORRAX_MPI_INCLUDE_DIR AND DEFINED ENV{LORRAX_MPI_INCLUDE_DIR})
        set(LORRAX_MPI_INCLUDE_DIR "$ENV{LORRAX_MPI_INCLUDE_DIR}" CACHE PATH "MPI header dir")
    endif()
    if(NOT DEFINED LORRAX_MPI_INCLUDE_DIR)
        find_path(LORRAX_MPI_INCLUDE_DIR mpi.h
            HINTS "${HDF5_INCLUDE_DIRS}" NO_DEFAULT_PATH)
    endif()
    if(NOT LORRAX_MPI_INCLUDE_DIR OR NOT EXISTS "${LORRAX_MPI_INCLUDE_DIR}/mpi.h")
        message(FATAL_ERROR "mpi.h not found.  Set -DLORRAX_MPI_INCLUDE_DIR.")
    endif()
    if(NOT DEFINED LORRAX_MPICH_LIB_DIR AND DEFINED ENV{LORRAX_MPICH_LIB_DIR})
        set(LORRAX_MPICH_LIB_DIR "$ENV{LORRAX_MPICH_LIB_DIR}" CACHE PATH "MPI runtime lib dir")
    endif()
    find_library(LORRAX_MPI_LIBRARY NAMES mpi
        HINTS "${LORRAX_MPICH_LIB_DIR}" NO_DEFAULT_PATH)
    if(NOT LORRAX_MPI_LIBRARY)
        message(FATAL_ERROR "libmpi not found in ${LORRAX_MPICH_LIB_DIR}.")
    endif()
    message(STATUS "MPI:  include=${LORRAX_MPI_INCLUDE_DIR} lib=${LORRAX_MPI_LIBRARY}")
endmacro()
