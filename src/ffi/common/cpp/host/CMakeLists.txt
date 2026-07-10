cmake_minimum_required(VERSION 3.22)
project(lorrax_ffi_host LANGUAGES CXX)

# liblorrax_ffi_host.so — the host-platform (JAX CPU backend) FFI library.
#
# Deliberately separate from ../CMakeLists.txt (the CUDA build): that build
# hard-requires cuSOLVERMp / NCCL / the CUDA language and runs inside the
# Shifter container; this one is CUDA-free by construction, links SLATE's
# gpu_backend=none install + MPI only, and builds host-side under the Cray
# PE.  ffi_loader.py registers this library's handlers under
# platform="cpu" (the CUDA library's under platform="CUDA") with the same
# target names — the jaxlib cpu-lapack / cuda-cusolver registration split.
#
# Sources: the slate context (pure MPI, shared verbatim with the CUDA lib)
# + the slate host handlers.  Future host backends (e.g. ScaLAPACK) add
# their handler TUs here.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----------------------------------------------------------------------------
# XLA FFI headers.  NO python probing here: the headers must match the
# RUNTIME jaxlib (the Shifter container's — see ../build.sh's PYTHON_EXE
# note), not whatever python is first on the host PATH.  build_host.sh
# stages them out of the container once and passes the directory in.
# ----------------------------------------------------------------------------
if(NOT DEFINED LORRAX_XLA_FFI_INCLUDE_DIR AND DEFINED ENV{LORRAX_XLA_FFI_INCLUDE_DIR})
    set(LORRAX_XLA_FFI_INCLUDE_DIR "$ENV{LORRAX_XLA_FFI_INCLUDE_DIR}" CACHE PATH
        "Directory containing xla/ffi/api/ffi.h (staged from the runtime jaxlib)")
endif()
if(NOT LORRAX_XLA_FFI_INCLUDE_DIR OR NOT EXISTS "${LORRAX_XLA_FFI_INCLUDE_DIR}/xla/ffi/api/ffi.h")
    message(FATAL_ERROR
        "xla/ffi/api/ffi.h not found under "
        "'${LORRAX_XLA_FFI_INCLUDE_DIR}'.  Set LORRAX_XLA_FFI_INCLUDE_DIR to "
        "a header stage matching the runtime jaxlib — build_host.sh stages "
        "one from the Shifter container automatically.")
endif()
message(STATUS "XLA FFI headers: ${LORRAX_XLA_FFI_INCLUDE_DIR}")

# ----------------------------------------------------------------------------
# SLATE — the gpu_backend=none install (see slate/scripts/build_perlmutter.sh).
# ----------------------------------------------------------------------------
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

# slateConfig expects an MPI::MPI_CXX target.  Under the Cray CC wrapper
# MPI headers and libmpi come implicitly, so an empty stand-in suffices —
# and it keeps find_package(MPI) from running (same trick as the CUDA
# build, which must dodge the container's OpenMPI mpicc).
if(NOT TARGET MPI::MPI_CXX)
    add_library(MPI::MPI_CXX INTERFACE IMPORTED)
endif()

find_package(slate CONFIG REQUIRED
    NO_DEFAULT_PATH PATHS "${LORRAX_SLATE_HOST_INSTALL_DIR}/lib64/cmake/slate")
message(STATUS "SLATE (host): found via ${LORRAX_SLATE_HOST_INSTALL_DIR}")

# ----------------------------------------------------------------------------
# Build liblorrax_ffi_host.so
# ----------------------------------------------------------------------------
add_library(lorrax_ffi_host SHARED
    ../../../slate/cpp/context.cc
    ../../../slate/cpp/host_ffi.cc
)

set_target_properties(lorrax_ffi_host PROPERTIES
    OUTPUT_NAME "lorrax_ffi_host"
    PREFIX "lib"
    # RPATH: the SLATE host install for bare-metal runs, plus the container
    # stage path (/lorrax_slate/lib carries libsci etc. inside Shifter).
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "${LORRAX_SLATE_HOST_INSTALL_DIR}/lib64;/lorrax_slate/lib"
)

target_include_directories(lorrax_ffi_host PRIVATE
    "${LORRAX_XLA_FFI_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
)

target_link_libraries(lorrax_ffi_host PRIVATE slate)

target_compile_options(lorrax_ffi_host PRIVATE -Wall -Wextra
    -Wno-unused-parameter -Wno-unused-variable)

message(STATUS "lorrax_ffi_host will link:")
message(STATUS "  SLATE = ${LORRAX_SLATE_HOST_INSTALL_DIR} (gpu_backend=none)")
message(STATUS "  MPI   = Cray CC wrapper (implicit)")
