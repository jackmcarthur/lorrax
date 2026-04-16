cmake_minimum_required(VERSION 3.22)
project(lorrax_ffi LANGUAGES C CXX)

# ----------------------------------------------------------------------------
# Options / paths
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# NVHPC root holding cuSOLVERMp + libcal.  Bind-mounted into Shifter at
# runtime; resolvable at build time from the same path.  The default is
# overridden via -DNVHPC_ROOT=... in build.sh.
set(NVHPC_ROOT "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5" CACHE PATH
    "Directory containing math_libs/<cuda>/targets/.../include/cusolverMp.h")
set(NVHPC_CUDA_SUBDIR "12.9" CACHE STRING
    "CUDA minor version string inside NVHPC math_libs")

set(CUSOLVERMP_INCLUDE
    "${NVHPC_ROOT}/math_libs/${NVHPC_CUDA_SUBDIR}/targets/x86_64-linux/include")
set(CUSOLVERMP_LIBDIR
    "${NVHPC_ROOT}/math_libs/${NVHPC_CUDA_SUBDIR}/lib64")

if(NOT EXISTS "${CUSOLVERMP_INCLUDE}/cusolverMp.h")
    message(FATAL_ERROR
        "cusolverMp.h not found at ${CUSOLVERMP_INCLUDE}/. "
        "Pass -DNVHPC_ROOT=<host_path_inside_container> to point at an "
        "NVHPC install.  On Perlmutter bind-mount /opt/nvidia/hpc_sdk/... "
        "into the Shifter container first.")
endif()

# ----------------------------------------------------------------------------
# Python (for jax.ffi.include_dir)
# ----------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import jax.ffi, sys; d = jax.ffi.include_dir(); sys.stdout.write(d if isinstance(d, str) else d.decode())"
    OUTPUT_VARIABLE JAX_FFI_INCLUDE_DIR
    RESULT_VARIABLE _jax_status
    ERROR_VARIABLE  _jax_stderr)
if(NOT _jax_status EQUAL 0 OR NOT JAX_FFI_INCLUDE_DIR)
    message(FATAL_ERROR
        "Could not locate jax.ffi.include_dir(): ${_jax_stderr}")
endif()
message(STATUS "JAX FFI headers: ${JAX_FFI_INCLUDE_DIR}")

# ----------------------------------------------------------------------------
# CUDA / NCCL — use the container's (not HPC SDK's)
# ----------------------------------------------------------------------------
set(CUDA_TOOLKIT_ROOT "/usr/local/cuda" CACHE PATH "Container CUDA toolkit root")
if(NOT EXISTS "${CUDA_TOOLKIT_ROOT}/include/cuda_runtime.h")
    message(FATAL_ERROR
        "cuda_runtime.h not at ${CUDA_TOOLKIT_ROOT}/include/. "
        "Set -DCUDA_TOOLKIT_ROOT if building outside the NVIDIA JAX container.")
endif()

set(NCCL_INCLUDE "/usr/include" CACHE PATH "Directory containing nccl.h")
if(NOT EXISTS "${NCCL_INCLUDE}/nccl.h")
    # Try the HPC SDK one as a fallback
    set(NCCL_INCLUDE
        "${NVHPC_ROOT}/comm_libs/${NVHPC_CUDA_SUBDIR}/nccl/include")
endif()
if(NOT EXISTS "${NCCL_INCLUDE}/nccl.h")
    message(FATAL_ERROR "nccl.h not found; pass -DNCCL_INCLUDE=<path>.")
endif()

find_library(CUDART_LIBRARY cudart
    PATHS "${CUDA_TOOLKIT_ROOT}/lib64" "${CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/lib"
    NO_DEFAULT_PATH)
find_library(CUSOLVER_LIBRARY cusolver
    PATHS "${CUDA_TOOLKIT_ROOT}/lib64" "${CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/lib"
    NO_DEFAULT_PATH)
if(NOT CUSOLVER_LIBRARY)
    message(FATAL_ERROR "libcusolver (single-GPU) not found")
endif()
find_library(CUSOLVERMG_LIBRARY cusolverMg
    PATHS "${CUDA_TOOLKIT_ROOT}/lib64" "${CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/lib"
    NO_DEFAULT_PATH)
if(NOT CUSOLVERMG_LIBRARY)
    message(FATAL_ERROR "libcusolverMg (multi-GPU single-process) not found")
endif()
find_library(NCCL_LIBRARY nccl
    PATHS "/lib/x86_64-linux-gnu" "/usr/lib/x86_64-linux-gnu"
          "${NVHPC_ROOT}/comm_libs/${NVHPC_CUDA_SUBDIR}/nccl/lib"
    NO_DEFAULT_PATH)
if(NOT CUDART_LIBRARY)
    message(FATAL_ERROR "libcudart not found")
endif()
if(NOT NCCL_LIBRARY)
    message(FATAL_ERROR "libnccl not found")
endif()

# ----------------------------------------------------------------------------
# Build lorrax_ffi.so
# ----------------------------------------------------------------------------
set(LORRAX_FFI_SOURCES
    api.cc
    ../../cusolvermp/cpp/context.cc
    ../../cusolvermp/cpp/eigh_ffi.cc
    ../../cusolvermg/cpp/eigh_mg_ffi.cc
)

add_library(lorrax_ffi SHARED ${LORRAX_FFI_SOURCES})

set_target_properties(lorrax_ffi PROPERTIES
    OUTPUT_NAME "lorrax_ffi"
    PREFIX "lib"
    # RPATH points at the NVHPC directory so the runtime linker finds
    # libcusolverMp.so and libcal.so without LD_LIBRARY_PATH.
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "${CUSOLVERMP_LIBDIR};${NVHPC_ROOT}/math_libs/${NVHPC_CUDA_SUBDIR}/lib64"
)

target_include_directories(lorrax_ffi PRIVATE
    "${JAX_FFI_INCLUDE_DIR}"
    "${CUSOLVERMP_INCLUDE}"
    "${NCCL_INCLUDE}"
    "${CUDA_TOOLKIT_ROOT}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
)

target_link_libraries(lorrax_ffi PRIVATE
    "${CUSOLVERMP_LIBDIR}/libcusolverMp.so"
    "${NCCL_LIBRARY}"
    "${CUDART_LIBRARY}"
    "${CUSOLVER_LIBRARY}"
    "${CUSOLVERMG_LIBRARY}"
)

target_compile_options(lorrax_ffi PRIVATE
    -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable
)

message(STATUS "lorrax_ffi will link:")
message(STATUS "  cuSOLVERMp = ${CUSOLVERMP_LIBDIR}/libcusolverMp.so")
message(STATUS "  NCCL       = ${NCCL_LIBRARY}")
message(STATUS "  CUDART     = ${CUDART_LIBRARY}")
