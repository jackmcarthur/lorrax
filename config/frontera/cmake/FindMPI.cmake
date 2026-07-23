# Frontera stub FindMPI — shadows CMake's builtin (via CMAKE_MODULE_PATH).
#
# HDF5's hdf5-config.cmake does find_package(MPI REQUIRED); CMake's builtin
# FindMPI insists on a compile test that fails in the python:3.12 container
# (no icc for Intel MPI's wrappers, and libmpi.so DT_NEEDs Intel's bundled
# libfabric which isn't on the default link path).  LORRAX itself links MPI
# explicitly via LORRAX_MPI_LIBRARY, so all we need here is to hand HDF5 a
# valid MPI::MPI_C / MPI::MPI_CXX target pointing at Intel MPI + libfabric.
#
# Requires -DLORRAX_IMPI_ROOT=<.../linux/mpi/intel64>.
if(NOT LORRAX_IMPI_ROOT)
    message(FATAL_ERROR "stub FindMPI: pass -DLORRAX_IMPI_ROOT=<intel mpi intel64 dir>")
endif()

set(_mpi_inc "${LORRAX_IMPI_ROOT}/include")
set(_mpi_libs
    "${LORRAX_IMPI_ROOT}/lib/release/libmpi.so"
    "${LORRAX_IMPI_ROOT}/libfabric/lib/libfabric.so")

foreach(_lang C CXX)
    set(MPI_${_lang}_FOUND TRUE)
    set(MPI_${_lang}_INCLUDE_DIRS "${_mpi_inc}")
    set(MPI_${_lang}_INCLUDE_PATH "${_mpi_inc}")   # legacy var name
    set(MPI_${_lang}_LIBRARIES "${_mpi_libs}")
    set(MPI_${_lang}_COMPILER "")
    if(NOT TARGET MPI::MPI_${_lang})
        add_library(MPI::MPI_${_lang} INTERFACE IMPORTED)
        set_target_properties(MPI::MPI_${_lang} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_mpi_inc}"
            INTERFACE_LINK_LIBRARIES      "${_mpi_libs}")
    endif()
endforeach()

set(MPI_FOUND TRUE)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPI REQUIRED_VARS MPI_C_LIBRARIES MPI_CXX_LIBRARIES)
