cmake_minimum_required(VERSION 3.22)
project(lorrax_ffi LANGUAGES C CXX)

# ----------------------------------------------------------------------------
# Options / paths
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----------------------------------------------------------------------------
#  Locate cuSOLVERMp + libcal.  BGW-style: prefer -D options, else env vars,
#  else well-known paths.  See src/ffi/PORTING.md for per-cluster notes.
# ----------------------------------------------------------------------------
set(NVHPC_ROOT "" CACHE PATH
    "Explicit NVHPC root (leave empty to autodetect from env vars and standard paths)")
set(NVHPC_CUDA_SUBDIR "" CACHE STRING
    "CUDA minor version string inside NVHPC math_libs (e.g. 12.9); empty = newest present")
set(CUSOLVERMP_INCLUDE_DIR "" CACHE PATH
    "Alternative: explicit directory containing cusolverMp.h + cal.h")
set(CUSOLVERMP_LIB_DIR "" CACHE PATH
    "Alternative: explicit directory containing libcusolverMp.so + libcal.so")

# Autodetection ladder — first match wins:
#   (a) explicit CUSOLVERMP_{INCLUDE,LIB}_DIR                (override)
#   (b) explicit NVHPC_ROOT
#   (c) $NVHPC_ROOT / $HPCSDK_ROOT / $NVHPC_SDK_PATH env vars
#   (d) /opt/nvidia/hpc_sdk/... or /lorrax_nvhpc/...
#   (e) CUDA Toolkit layout (CUDA 12.8+ standalone cuda-cusolvermp-12-X pkg)

if(CUSOLVERMP_INCLUDE_DIR AND CUSOLVERMP_LIB_DIR)
    set(CUSOLVERMP_INCLUDE "${CUSOLVERMP_INCLUDE_DIR}")
    set(CUSOLVERMP_LIBDIR  "${CUSOLVERMP_LIB_DIR}")
    message(STATUS "cuSOLVERMp: using explicit paths from CUSOLVERMP_{INCLUDE,LIB}_DIR")
else()
    # (b, c) env vars
    if(NOT NVHPC_ROOT)
        foreach(_envvar NVHPC_ROOT HPCSDK_ROOT NVHPC_SDK_PATH)
            if(DEFINED ENV{${_envvar}} AND IS_DIRECTORY "$ENV{${_envvar}}")
                set(NVHPC_ROOT "$ENV{${_envvar}}")
                message(STATUS "cuSOLVERMp: NVHPC_ROOT from \$${_envvar} = ${NVHPC_ROOT}")
                break()
            endif()
        endforeach()
    endif()

    # (d) common NVHPC install locations
    if(NOT NVHPC_ROOT)
        file(GLOB _nvhpc_candidates
            "/opt/nvidia/hpc_sdk/Linux_x86_64/*"
            "/lorrax_nvhpc/*"                        # Shifter bind-mount target
            "/usr/local/nvhpc/Linux_x86_64/*")
        list(SORT _nvhpc_candidates ORDER DESCENDING)  # newest first
        foreach(_c ${_nvhpc_candidates})
            if(EXISTS "${_c}/math_libs")
                set(NVHPC_ROOT "${_c}")
                message(STATUS "cuSOLVERMp: NVHPC_ROOT autodetected = ${NVHPC_ROOT}")
                break()
            endif()
        endforeach()
    endif()

    if(NVHPC_ROOT)
        # Pick CUDA subdir under NVHPC_ROOT (newest present with cusolverMp.h)
        if(NOT NVHPC_CUDA_SUBDIR)
            file(GLOB _cuda_vers RELATIVE "${NVHPC_ROOT}/math_libs" "${NVHPC_ROOT}/math_libs/*")
            list(SORT _cuda_vers ORDER DESCENDING)
            foreach(_v ${_cuda_vers})
                if(EXISTS "${NVHPC_ROOT}/math_libs/${_v}/targets/x86_64-linux/include/cusolverMp.h")
                    set(NVHPC_CUDA_SUBDIR "${_v}")
                    message(STATUS "cuSOLVERMp: CUDA subdir autodetected = ${NVHPC_CUDA_SUBDIR}")
                    break()
                endif()
            endforeach()
        endif()
        if(NOT NVHPC_CUDA_SUBDIR)
            message(FATAL_ERROR
                "cuSOLVERMp: no usable CUDA subdir found under ${NVHPC_ROOT}/math_libs/")
        endif()
        set(CUSOLVERMP_INCLUDE
            "${NVHPC_ROOT}/math_libs/${NVHPC_CUDA_SUBDIR}/targets/x86_64-linux/include")
        set(CUSOLVERMP_LIBDIR
            "${NVHPC_ROOT}/math_libs/${NVHPC_CUDA_SUBDIR}/lib64")
    else()
        # (e) CUDA Toolkit layout — CUDA 12.8+ ships `cuda-cusolvermp-12-X`
        # as a standalone package that installs into the toolkit tree
        # right next to the single-GPU cuSOLVER.
        foreach(_cuda_root "/usr/local/cuda" "/usr/local/cuda-12.9"
                          "/usr/local/cuda-12.8")
            set(_inc "${_cuda_root}/targets/x86_64-linux/include/cusolverMp.h")
            set(_lib "${_cuda_root}/targets/x86_64-linux/lib/libcusolverMp.so")
            if(EXISTS "${_inc}" AND EXISTS "${_lib}")
                set(CUSOLVERMP_INCLUDE "${_cuda_root}/targets/x86_64-linux/include")
                set(CUSOLVERMP_LIBDIR  "${_cuda_root}/targets/x86_64-linux/lib")
                message(STATUS "cuSOLVERMp: found alongside CUDA Toolkit at ${_cuda_root}")
                break()
            endif()
        endforeach()
    endif()

    if(NOT CUSOLVERMP_INCLUDE OR NOT CUSOLVERMP_LIBDIR)
        message(FATAL_ERROR
            "cusolverMp.h not found.  Pass one of:\n"
            "  -DNVHPC_ROOT=<dir with math_libs/>\n"
            "  -DCUSOLVERMP_INCLUDE_DIR=<dir with cusolverMp.h> "
            "-DCUSOLVERMP_LIB_DIR=<dir with libcusolverMp.so>\n"
            "or set \$NVHPC_ROOT / \$HPCSDK_ROOT, or install "
            "cuda-cusolvermp-12-X.  See src/ffi/PORTING.md.")
    endif()
endif()

if(NOT EXISTS "${CUSOLVERMP_INCLUDE}/cusolverMp.h")
    message(FATAL_ERROR
        "cusolverMp.h not found at ${CUSOLVERMP_INCLUDE}/.\n"
        "See src/ffi/PORTING.md for where cuSOLVERMp usually lives and how "
        "to point the build at it on a new cluster.")
endif()
if(NOT EXISTS "${CUSOLVERMP_LIBDIR}/libcusolverMp.so")
    message(FATAL_ERROR
        "libcusolverMp.so not found at ${CUSOLVERMP_LIBDIR}/.")
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

# ----------------------------------------------------------------------------
#  ABI sanity check: CUDA toolkit (the one JAX links against at runtime)
#  must be compatible with the CUDA subdir under NVHPC where we found
#  cusolverMp.  Major version mismatch → guaranteed ABI breakage.  Minor
#  mismatch → usually OK within 12.X, but worth a warning.
# ----------------------------------------------------------------------------
if(EXISTS "${CUDA_TOOLKIT_ROOT}/version.json")
    file(READ "${CUDA_TOOLKIT_ROOT}/version.json" _cuda_ver_json)
    string(REGEX MATCH "\"cuda\"[^{]*{[^\"]*\"version\"[ :]*\"([0-9]+\\.[0-9]+)" _ _cuda_ver_json)
    if(CMAKE_MATCH_1)
        set(TOOLKIT_CUDA_VERSION "${CMAKE_MATCH_1}")
    endif()
endif()
if(NOT TOOLKIT_CUDA_VERSION AND EXISTS "${CUDA_TOOLKIT_ROOT}/bin/nvcc")
    execute_process(COMMAND "${CUDA_TOOLKIT_ROOT}/bin/nvcc" --version
                    OUTPUT_VARIABLE _nvcc_out OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" _ "${_nvcc_out}")
    if(CMAKE_MATCH_1)
        set(TOOLKIT_CUDA_VERSION "${CMAKE_MATCH_1}")
    endif()
endif()

if(TOOLKIT_CUDA_VERSION AND NVHPC_CUDA_SUBDIR)
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" _ "${TOOLKIT_CUDA_VERSION}")
    set(_tk_major "${CMAKE_MATCH_1}")
    set(_tk_minor "${CMAKE_MATCH_2}")
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" _ "${NVHPC_CUDA_SUBDIR}")
    set(_hp_major "${CMAKE_MATCH_1}")
    set(_hp_minor "${CMAKE_MATCH_2}")
    if(_tk_major AND _hp_major AND NOT "${_tk_major}" STREQUAL "${_hp_major}")
        message(FATAL_ERROR
            "CUDA major-version mismatch:\n"
            "  toolkit (${CUDA_TOOLKIT_ROOT}) = ${TOOLKIT_CUDA_VERSION}\n"
            "  NVHPC math_libs subdir        = ${NVHPC_CUDA_SUBDIR}\n"
            "libcusolverMp.so is ABI-incompatible across CUDA major "
            "versions.  Use an NVHPC install built against CUDA "
            "${_tk_major}.x, or bump the container's CUDA.")
    elseif(_tk_minor AND _hp_minor AND NOT "${_tk_minor}" STREQUAL "${_hp_minor}")
        message(WARNING
            "CUDA minor-version mismatch — toolkit ${TOOLKIT_CUDA_VERSION} "
            "vs NVHPC math_libs ${NVHPC_CUDA_SUBDIR}.  Usually OK within "
            "12.X (forward compatibility), but test before shipping.")
    endif()
    message(STATUS "CUDA toolkit version: ${TOOLKIT_CUDA_VERSION} "
                   "(NVHPC math_libs: ${NVHPC_CUDA_SUBDIR}) — compatible")
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
#  Parallel HDF5 + MPI — for the ffi.phdf5 subpackage.
# ----------------------------------------------------------------------------
# Default stage dir (bind-mounted by run_shifter.sh).  Callers can
# override with -DHDF5_ROOT=... or $HDF5_ROOT / $HDF5_DIR env vars,
# which CMake's find_package(HDF5) honours natively.
if(NOT DEFINED HDF5_ROOT AND NOT DEFINED ENV{HDF5_ROOT} AND NOT DEFINED ENV{HDF5_DIR})
    if(IS_DIRECTORY "/lorrax_phdf5/lib")
        set(HDF5_ROOT "/lorrax_phdf5" CACHE PATH
            "Root of parallel HDF5 install (bind-mounted stage by default)")
        message(STATUS "HDF5: using bind-mounted stage at /lorrax_phdf5")
    endif()
endif()

set(HDF5_PREFER_PARALLEL ON)
find_package(HDF5 COMPONENTS C REQUIRED)
if(NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR
        "HDF5 found at ${HDF5_INCLUDE_DIRS} is NOT built with MPI-IO.\n"
        "Load a parallel module (cray-hdf5-parallel, hdf5/1.14-openmpi, ...)\n"
        "or set HDF5_ROOT / -DHDF5_DIR=... to a parallel install.\n"
        "See src/ffi/PORTING.md for per-cluster notes.")
endif()
message(STATUS "HDF5: ${HDF5_VERSION} at ${HDF5_INCLUDE_DIRS} (parallel)")

find_package(MPI REQUIRED COMPONENTS C)
message(STATUS "MPI:  ${MPI_C_COMPILER} / ${MPI_C_LIBRARIES}")

# ----------------------------------------------------------------------------
# Build lorrax_ffi.so
# ----------------------------------------------------------------------------
set(LORRAX_FFI_SOURCES
    api.cc
    ../../cusolvermp/cpp/context.cc
    ../../cusolvermp/cpp/eigh_ffi.cc
    ../../cusolvermg/cpp/eigh_mg_ffi.cc
    ../../phdf5/cpp/context.cc
    ../../phdf5/cpp/write_ffi.cc
)

add_library(lorrax_ffi SHARED ${LORRAX_FFI_SOURCES})

set_target_properties(lorrax_ffi PROPERTIES
    OUTPUT_NAME "lorrax_ffi"
    PREFIX "lib"
    # RPATH points at the NVHPC directory so the runtime linker finds
    # libcusolverMp.so and libcal.so without LD_LIBRARY_PATH.
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "${CUSOLVERMP_LIBDIR};${NVHPC_ROOT}/math_libs/${NVHPC_CUDA_SUBDIR}/lib64;/lorrax_phdf5/lib"
)

target_include_directories(lorrax_ffi PRIVATE
    "${JAX_FFI_INCLUDE_DIR}"
    "${CUSOLVERMP_INCLUDE}"
    "${NCCL_INCLUDE}"
    "${CUDA_TOOLKIT_ROOT}/include"
    "${HDF5_INCLUDE_DIRS}"
    "${MPI_C_INCLUDE_DIRS}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
)

target_link_libraries(lorrax_ffi PRIVATE
    "${CUSOLVERMP_LIBDIR}/libcusolverMp.so"
    "${NCCL_LIBRARY}"
    "${CUDART_LIBRARY}"
    "${CUSOLVER_LIBRARY}"
    "${CUSOLVERMG_LIBRARY}"
    ${HDF5_LIBRARIES}
    MPI::MPI_C
)

target_compile_options(lorrax_ffi PRIVATE
    -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable
)

message(STATUS "lorrax_ffi will link:")
message(STATUS "  cuSOLVERMp = ${CUSOLVERMP_LIBDIR}/libcusolverMp.so")
message(STATUS "  NCCL       = ${NCCL_LIBRARY}")
message(STATUS "  CUDART     = ${CUDART_LIBRARY}")
message(STATUS "  HDF5       = ${HDF5_LIBRARIES}")
message(STATUS "  MPI        = ${MPI_C_LIBRARIES}")
