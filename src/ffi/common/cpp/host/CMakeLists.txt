cmake_minimum_required(VERSION 3.22)
# LANGUAGES CXX ONLY — deliberate, do not add C or Fortran.  Nothing here is
# Fortran (the ScaLAPACK calls go through MKL's C bindings) and the py312
# apptainer container this builds in has no gfortran, so enabling Fortran
# fails the configure outright.  SLATE's OWN CMakeLists declares
# `LANGUAGES CXX Fortran` while using Fortran nowhere and needs the same
# one-line patch before it will configure here — see
# config/frontera/build_ffi_host.sh, step "patch SLATE".
project(lorrax_ffi_host LANGUAGES CXX)

# liblorrax_ffi_host.so — the host-platform (JAX CPU backend) FFI library.
#
# Deliberately separate from ../CMakeLists.txt (the CUDA build): that build
# hard-requires cuSOLVERMp / NCCL / the CUDA language and runs inside the
# Shifter/apptainer container; this one is CUDA-free BY CONSTRUCTION.  It
# compiles the SAME phdf5 collective MPI-IO core (context.cc + read_ffi.cc)
# as the CUDA lib, with -DLORRAX_FFI_NO_CUDA so the device-staging tail is a
# std::memcpy into the host-resident XLA output buffer instead of a
# cudaMemcpyAsync H2D (see phdf5/cpp/read_ffi.cc).  ffi_loader.py registers
# this library's handlers under platform="cpu" (the CUDA library's under
# platform="CUDA") with the same jax.ffi target names — the jaxlib
# cpu-lapack / cuda-cusolver registration split.
#
# Two optional component groups, each CUDA-free:
#   LORRAX_FFI_HAVE_PHDF5  (default ON)  parallel-HDF5 read handlers + lifecycle
#   LORRAX_HOST_HAVE_SLATE (default ON)  SLATE gpu_backend=none host handlers
#                                        AND the ScaLAPACK solve_lu + eigh
#                                        handlers (they reuse slate/cpp/ctx.h)
#                                        + their ScaLAPACK/BLACS link line.
# SLATE downgrades to "skipped" if its install isn't found, so a phdf5-only
# host lib still builds on a cluster with no SLATE host install.
#
# The full 10-target lib (phdf5×3, slate×5, scalapack×2) is built by
# config/frontera/build_ffi_host.sh (9 targets first built 2026-07-25;
# scalapack eigh added 2026-07-26, workstream V).

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LORRAX_FFI_HAVE_PHDF5  "Build the phdf5 host read handlers (needs parallel HDF5)" ON)
option(LORRAX_HOST_HAVE_SLATE "Build the SLATE host handlers (needs a gpu_backend=none SLATE install)" ON)

# ----------------------------------------------------------------------------
# XLA FFI headers.  They must match the RUNTIME jaxlib (the container's jax,
# NOT whatever python is first on the host PATH — newer XLA-FFI headers on an
# older runtime is the unsupported direction).  Prefer an explicit
# LORRAX_XLA_FFI_INCLUDE_DIR (build_host.sh stages one out of the container);
# otherwise fall back to probing jax.ffi.include_dir() with Python3 — correct
# when this cmake runs INSIDE the runtime container/venv (the Frontera path).
# ----------------------------------------------------------------------------
if(NOT DEFINED LORRAX_XLA_FFI_INCLUDE_DIR AND DEFINED ENV{LORRAX_XLA_FFI_INCLUDE_DIR})
    set(LORRAX_XLA_FFI_INCLUDE_DIR "$ENV{LORRAX_XLA_FFI_INCLUDE_DIR}" CACHE PATH
        "Directory containing xla/ffi/api/ffi.h (staged from the runtime jaxlib)")
endif()
if(NOT LORRAX_XLA_FFI_INCLUDE_DIR OR NOT EXISTS "${LORRAX_XLA_FFI_INCLUDE_DIR}/xla/ffi/api/ffi.h")
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c
                "import jax.ffi,sys; d=jax.ffi.include_dir(); sys.stdout.write(d if isinstance(d,str) else d.decode())"
            OUTPUT_VARIABLE _jax_ffi_dir
            RESULT_VARIABLE _jax_status
            ERROR_VARIABLE  _jax_err)
        if(_jax_status EQUAL 0 AND EXISTS "${_jax_ffi_dir}/xla/ffi/api/ffi.h")
            set(LORRAX_XLA_FFI_INCLUDE_DIR "${_jax_ffi_dir}")
        endif()
    endif()
endif()
if(NOT LORRAX_XLA_FFI_INCLUDE_DIR OR NOT EXISTS "${LORRAX_XLA_FFI_INCLUDE_DIR}/xla/ffi/api/ffi.h")
    message(FATAL_ERROR
        "xla/ffi/api/ffi.h not found under '${LORRAX_XLA_FFI_INCLUDE_DIR}'.  "
        "Set -DLORRAX_XLA_FFI_INCLUDE_DIR to a header stage matching the "
        "runtime jaxlib, or run this cmake with a Python3 whose jax matches "
        "the runtime (jax.ffi.include_dir() is probed automatically).")
endif()
message(STATUS "XLA FFI headers: ${LORRAX_XLA_FFI_INCLUDE_DIR}")

set(LORRAX_FFI_HOST_SOURCES "")
set(_host_link_libs "")
set(_host_rpaths "")

# ----------------------------------------------------------------------------
# parallel-HDF5 + MPI (CUDA-free).  Same probe as the CUDA CMakeLists, minus
# every CUDA/NCCL/cuSOLVERMp dependency: the host read handlers only need
# HDF5 (MPI-IO) + MPI.  On Frontera these are the host Intel-MPI parallel
# HDF5 (-DHDF5_ROOT=...) + Intel MPI (-DLORRAX_MPI_INCLUDE_DIR / _MPICH_LIB_DIR),
# with the config/frontera/cmake FindMPI stub on CMAKE_MODULE_PATH.
# ----------------------------------------------------------------------------
if(LORRAX_FFI_HAVE_PHDF5)
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

    list(APPEND LORRAX_FFI_HOST_SOURCES
        ../../../phdf5/cpp/api.cc
        ../../../phdf5/cpp/context.cc
        ../../../phdf5/cpp/read_ffi.cc
    )
    list(APPEND _host_link_libs ${HDF5_LIBRARIES} "${LORRAX_MPI_LIBRARY}")
    get_filename_component(_hdf5_libdir "${HDF5_C_LIBRARY_hdf5}" DIRECTORY)
    if(_hdf5_libdir)
        list(APPEND _host_rpaths "${_hdf5_libdir}")
    endif()
    if(LORRAX_MPICH_LIB_DIR)
        list(APPEND _host_rpaths "${LORRAX_MPICH_LIB_DIR}")
    endif()
endif()

# ----------------------------------------------------------------------------
# SLATE — the gpu_backend=none install (optional; skipped if absent).  Adds
# the slate host handlers + the ScaLAPACK solve-lu host handler (which reuses
# slate/cpp/ctx.h's host_collective_mutex).
# ----------------------------------------------------------------------------
if(LORRAX_HOST_HAVE_SLATE)
    if(NOT DEFINED LORRAX_SLATE_HOST_INSTALL_DIR AND DEFINED ENV{LORRAX_SLATE_HOST_INSTALL_DIR})
        set(LORRAX_SLATE_HOST_INSTALL_DIR "$ENV{LORRAX_SLATE_HOST_INSTALL_DIR}" CACHE PATH
            "SLATE gpu_backend=none install prefix")
    endif()
    if(NOT DEFINED LORRAX_SLATE_HOST_INSTALL_DIR)
        set(LORRAX_SLATE_HOST_INSTALL_DIR "$ENV{HOME}/software/slate_builds/cpu/install"
            CACHE PATH "SLATE gpu_backend=none install prefix")
    endif()
    set(slate_DIR    "${LORRAX_SLATE_HOST_INSTALL_DIR}/lib64/cmake/slate")
    set(blaspp_DIR   "${LORRAX_SLATE_HOST_INSTALL_DIR}/lib64/cmake/blaspp")
    set(lapackpp_DIR "${LORRAX_SLATE_HOST_INSTALL_DIR}/lib64/cmake/lapackpp")
    if(NOT TARGET MPI::MPI_CXX)
        add_library(MPI::MPI_CXX INTERFACE IMPORTED)
    endif()
    find_package(slate CONFIG
        NO_DEFAULT_PATH PATHS "${LORRAX_SLATE_HOST_INSTALL_DIR}/lib64/cmake/slate")
    if(slate_FOUND)
        message(STATUS "SLATE (host): found via ${LORRAX_SLATE_HOST_INSTALL_DIR}")
        list(APPEND LORRAX_FFI_HOST_SOURCES
            ../../../slate/cpp/context.cc
            ../../../slate/cpp/host_ffi.cc
            ../../../scalapack/cpp/solve_lu_ffi.cc
            ../../../scalapack/cpp/eigh_ffi.cc)
        list(APPEND _host_link_libs slate)
        list(APPEND _host_rpaths "${LORRAX_SLATE_HOST_INSTALL_DIR}/lib64" "/lorrax_slate/lib")

        # --------------------------------------------------------------
        # ScaLAPACK + BLACS — required by scalapack/cpp/*_ffi.cc
        # (pXgetrf / pXgetrs / pXheevd / blacs_gridinit / descinit / numroc;
        # prototypes in scalapack/cpp/blacs_grid.h).
        #
        # The SLATE package does NOT supply them: SLATE is configured with
        # -DSCALAPACK_LIBRARIES="" (its optional scalapack_api layer is
        # not built), so `slate` brings in BLAS/LAPACK only.  Workstream L
        # (2026-07-25) had to hand-inject the link line via
        # -DCMAKE_SHARED_LINKER_FLAGS on every configure — that gap is
        # closed here.
        #
        # Resolution order:
        #   1. -DLORRAX_SCALAPACK_LIBRARIES="<link line>" — vendor-agnostic
        #      escape hatch (Cray LibSci, netlib, AOCL, ...).  A CMake list
        #      or a space-separated string; raw `-l`/`-L` flags are fine.
        #   2. MKL, probed under -DLORRAX_MKL_ROOT / $MKLROOT.  This is the
        #      Frontera path (MKL 2020.1 at
        #      /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl).
        #   3. nothing found -> loud WARNING; the .so still links (the
        #      symbols stay undefined) but the first scalapack call dies.
        #
        # `-Wl,--no-as-needed` is mandatory: MKL's layered libraries are
        # reached transitively and ld10+ drops them from DT_NEEDED
        # otherwise, producing a .so that dlopens and then fails to
        # resolve pzgetrf at call time.
        # --------------------------------------------------------------
        if(NOT DEFINED LORRAX_SCALAPACK_LIBRARIES AND DEFINED ENV{LORRAX_SCALAPACK_LIBRARIES})
            set(LORRAX_SCALAPACK_LIBRARIES "$ENV{LORRAX_SCALAPACK_LIBRARIES}" CACHE STRING
                "Full ScaLAPACK+BLACS link line (overrides the MKL probe)")
        endif()
        if(LORRAX_SCALAPACK_LIBRARIES)
            separate_arguments(_scalapack_link UNIX_COMMAND
                              "${LORRAX_SCALAPACK_LIBRARIES}")
            message(STATUS "ScaLAPACK: explicit -> ${LORRAX_SCALAPACK_LIBRARIES}")
        else()
            if(NOT DEFINED LORRAX_MKL_ROOT AND DEFINED ENV{LORRAX_MKL_ROOT})
                set(LORRAX_MKL_ROOT "$ENV{LORRAX_MKL_ROOT}" CACHE PATH "MKL install prefix")
            endif()
            if(NOT LORRAX_MKL_ROOT AND DEFINED ENV{MKLROOT})
                set(LORRAX_MKL_ROOT "$ENV{MKLROOT}" CACHE PATH "MKL install prefix")
            endif()
            # BLACS flavour MUST match the MPI actually linked above
            # (Intel MPI on Frontera); a mismatch links fine and then
            # hangs/aborts inside blacs_gridinit.
            set(LORRAX_MKL_BLACS "mkl_blacs_intelmpi_lp64" CACHE STRING
                "MKL BLACS library, must match the MPI (…_intelmpi_lp64 | …_openmpi_lp64)")
            set(LORRAX_MKL_THREAD_LIB "mkl_gnu_thread" CACHE STRING
                "MKL threading layer (mkl_gnu_thread for g++, mkl_sequential to disable)")
            find_library(LORRAX_MKL_SCALAPACK_LIBRARY NAMES mkl_scalapack_lp64
                HINTS "${LORRAX_MKL_ROOT}/lib/intel64_lin"
                      "${LORRAX_MKL_ROOT}/lib/intel64"
                      "${LORRAX_MKL_ROOT}/lib")
            if(LORRAX_MKL_SCALAPACK_LIBRARY)
                get_filename_component(_mkl_libdir
                                       "${LORRAX_MKL_SCALAPACK_LIBRARY}" DIRECTORY)
                set(_scalapack_link
                    "-L${_mkl_libdir}" "-Wl,--no-as-needed"
                    "-lmkl_scalapack_lp64" "-l${LORRAX_MKL_BLACS}"
                    "-lmkl_intel_lp64" "-l${LORRAX_MKL_THREAD_LIB}" "-lmkl_core")
                if(LORRAX_MKL_THREAD_LIB STREQUAL "mkl_gnu_thread")
                    list(APPEND _scalapack_link "-lgomp")
                endif()
                list(APPEND _scalapack_link "-lpthread" "-lm" "-ldl")
                list(APPEND _host_rpaths "${_mkl_libdir}")
                message(STATUS "ScaLAPACK: MKL at ${_mkl_libdir} "
                               "(blacs=${LORRAX_MKL_BLACS}, thread=${LORRAX_MKL_THREAD_LIB})")
            else()
                set(_scalapack_link "")
                message(WARNING
                    "ScaLAPACK/BLACS not found — building "
                    "scalapack/cpp/solve_lu_ffi.cc with UNDEFINED pXgetrf/BLACS "
                    "symbols.  ffi_loader.has_target('lorrax_scalapack_batched_solve_lu') "
                    "will report the handler present and the first call will die.  "
                    "Set -DLORRAX_MKL_ROOT (or \$MKLROOT), or pass a full "
                    "-DLORRAX_SCALAPACK_LIBRARIES link line.")
            endif()
        endif()
        list(APPEND _host_link_libs ${_scalapack_link})
    else()
        message(STATUS
            "SLATE (host): not found at ${LORRAX_SLATE_HOST_INSTALL_DIR} — "
            "slate/scalapack host handlers skipped.")
    endif()
endif()

if(NOT LORRAX_FFI_HOST_SOURCES)
    message(FATAL_ERROR
        "No host FFI components selected/found (phdf5 off and SLATE absent). "
        "Nothing to build.")
endif()

# ----------------------------------------------------------------------------
# Build liblorrax_ffi_host.so
# ----------------------------------------------------------------------------
add_library(lorrax_ffi_host SHARED ${LORRAX_FFI_HOST_SOURCES})

# The defining flag: every phdf5 TU compiled here switches its device-staging
# tail + staging-buffer allocator to the host path and drops <cuda_runtime.h>.
target_compile_definitions(lorrax_ffi_host PRIVATE LORRAX_FFI_NO_CUDA=1)

set_target_properties(lorrax_ffi_host PROPERTIES
    OUTPUT_NAME "lorrax_ffi_host"
    PREFIX "lib"
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "${_host_rpaths}"
)

target_include_directories(lorrax_ffi_host PRIVATE
    "${LORRAX_XLA_FFI_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
)
if(LORRAX_FFI_HAVE_PHDF5)
    target_include_directories(lorrax_ffi_host PRIVATE
        "${HDF5_INCLUDE_DIRS}"
        "${LORRAX_MPI_INCLUDE_DIR}")
endif()

find_package(Threads REQUIRED)
target_link_libraries(lorrax_ffi_host PRIVATE ${_host_link_libs} Threads::Threads)

target_compile_options(lorrax_ffi_host PRIVATE -Wall -Wextra
    -Wno-unused-parameter -Wno-unused-variable)

message(STATUS "lorrax_ffi_host will build sources:")
foreach(_s ${LORRAX_FFI_HOST_SOURCES})
    message(STATUS "  ${_s}")
endforeach()
