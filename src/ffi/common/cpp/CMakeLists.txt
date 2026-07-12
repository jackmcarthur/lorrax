cmake_minimum_required(VERSION 3.22)
project(lorrax_ffi LANGUAGES C CXX)

# ----------------------------------------------------------------------------
# Options / paths
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----------------------------------------------------------------------------
# Subpackage selection.  Each subpackage links a distinct native stack, so a
# site only needs the deps for the subpackages it enables:
#   * CUSOLVERMP → cuSolverMp + cuBLASMp + cuSolverMg + NCCL   (no MPI, no HDF5)
#   * PHDF5      → MPI + parallel HDF5
#   * SLATE      → SLATE (+ blaspp/lapackpp) + MPI
# Turn OFF the ones you don't need, e.g. a cuSolverMp-only build with neither
# MPI nor HDF5 present:
#   cmake ... -DLORRAX_FFI_ENABLE_PHDF5=OFF -DLORRAX_FFI_ENABLE_SLATE=OFF
# ----------------------------------------------------------------------------
option(LORRAX_FFI_ENABLE_CUSOLVERMP
    "Build cuSolverMp/cuBLASMp/cuSolverMg GPU dense-linalg handlers (needs NCCL, no MPI)" ON)
option(LORRAX_FFI_ENABLE_PHDF5
    "Build parallel-HDF5 collective-I/O handlers (needs MPI + parallel HDF5)" ON)
option(LORRAX_FFI_ENABLE_SLATE
    "Build SLATE handlers (needs SLATE + MPI); auto-skips if SLATE not found" ON)

if(NOT LORRAX_FFI_ENABLE_CUSOLVERMP AND NOT LORRAX_FFI_ENABLE_PHDF5
   AND NOT LORRAX_FFI_ENABLE_SLATE)
    message(FATAL_ERROR
        "No FFI subpackage enabled — set at least one of "
        "LORRAX_FFI_ENABLE_{CUSOLVERMP,PHDF5,SLATE}=ON.")
endif()

# MPI is shared by PHDF5 and SLATE.
if(LORRAX_FFI_ENABLE_PHDF5 OR LORRAX_FFI_ENABLE_SLATE)
    set(LORRAX_FFI_NEED_MPI ON)
else()
    set(LORRAX_FFI_NEED_MPI OFF)
endif()

message(STATUS "lorrax_ffi subpackages: "
    "cusolvermp=${LORRAX_FFI_ENABLE_CUSOLVERMP} "
    "phdf5=${LORRAX_FFI_ENABLE_PHDF5} "
    "slate=${LORRAX_FFI_ENABLE_SLATE} (need_mpi=${LORRAX_FFI_NEED_MPI})")

# ----------------------------------------------------------------------------
# CUDA toolkit root — needed by every subpackage (all TUs use cuda_runtime.h
# and link libcudart).  The CUDA *language* (nvcc) is only enabled when the
# cuSolverMp group is built, since its .cu kernels are the only nvcc TUs.
# ----------------------------------------------------------------------------
set(CUDA_TOOLKIT_ROOT "/usr/local/cuda" CACHE PATH "CUDA toolkit root")
if(NOT EXISTS "${CUDA_TOOLKIT_ROOT}/include/cuda_runtime.h")
    message(FATAL_ERROR
        "cuda_runtime.h not at ${CUDA_TOOLKIT_ROOT}/include/. "
        "Set -DCUDA_TOOLKIT_ROOT if building outside the NVIDIA JAX container.")
endif()
find_library(CUDART_LIBRARY cudart
    PATHS "${CUDA_TOOLKIT_ROOT}/lib64" "${CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/lib"
    NO_DEFAULT_PATH)
if(NOT CUDART_LIBRARY)
    message(FATAL_ERROR "libcudart not found under ${CUDA_TOOLKIT_ROOT}")
endif()

# ----------------------------------------------------------------------------
# Python (for jax.ffi.include_dir) — needed by every subpackage's handlers.
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

# ============================================================================
#  cuSolverMp / cuBLASMp / cuSolverMg + NCCL   (LORRAX_FFI_ENABLE_CUSOLVERMP)
# ============================================================================
if(LORRAX_FFI_ENABLE_CUSOLVERMP)
    # CUDA language (for the .cu kernel TUs in the cuBLASMp group).
    if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "${CUDA_TOOLKIT_ROOT}/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT}/bin/nvcc")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80)  # A100 default; override with -D for other GPUs
    endif()

    # ------------------------------------------------------------------------
    #  Locate cuSolverMp.  Prefer -D options, else env vars, else well-known
    #  paths.  See src/ffi/PORTING.md for per-cluster notes.  (CAL is no longer
    #  used — cuSolverMp >= 0.7.0 takes an ncclComm_t directly.)
    # ------------------------------------------------------------------------
    set(NVHPC_ROOT "" CACHE PATH
        "Explicit NVHPC root (leave empty to autodetect from env vars and standard paths)")
    set(NVHPC_CUDA_SUBDIR "" CACHE STRING
        "CUDA minor version string inside NVHPC math_libs (e.g. 12.9); empty = newest present")
    set(CUSOLVERMP_INCLUDE_DIR "" CACHE PATH
        "Alternative: explicit directory containing cusolverMp.h")
    set(CUSOLVERMP_LIB_DIR "" CACHE PATH
        "Alternative: explicit directory containing libcusolverMp.so (CAL no longer required)")
    set(CUSOLVERMP_LIB "" CACHE FILEPATH
        "Explicit path to libcusolverMp.so[.0]; empty = <CUSOLVERMP_LIB_DIR>/libcusolverMp.so")
    set(CUBLASMP_LIB "" CACHE FILEPATH
        "Explicit path to libcublasmp.so[.0]; empty = next to libcusolverMp (pip cuBLASMp wheel puts it in a separate dir)")

    # Autodetection ladder — first match wins:
    #   (a) explicit CUSOLVERMP_{INCLUDE,LIB}_DIR                (override)
    #   (b) explicit NVHPC_ROOT
    #   (c) $NVHPC_ROOT / $HPCSDK_ROOT / $NVHPC_SDK_PATH env vars
    #   (d) /opt/nvidia/hpc_sdk/... or /lorrax_nvhpc/...
    #   (e) CUDA Toolkit layout (CUDA 12.8+ standalone cuda-cusolvermp-12-X pkg)
    if(CUSOLVERMP_INCLUDE_DIR AND CUSOLVERMP_LIB_DIR)
        set(CUSOLVERMP_INCLUDE "${CUSOLVERMP_INCLUDE_DIR}")
        set(CUSOLVERMP_LIBDIR  "${CUSOLVERMP_LIB_DIR}")
        message(STATUS "cuSolverMp: using explicit paths from CUSOLVERMP_{INCLUDE,LIB}_DIR")
    else()
        # (b, c) env vars
        if(NOT NVHPC_ROOT)
            foreach(_envvar NVHPC_ROOT HPCSDK_ROOT NVHPC_SDK_PATH)
                if(DEFINED ENV{${_envvar}} AND IS_DIRECTORY "$ENV{${_envvar}}")
                    set(NVHPC_ROOT "$ENV{${_envvar}}")
                    message(STATUS "cuSolverMp: NVHPC_ROOT from \$${_envvar} = ${NVHPC_ROOT}")
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
                    message(STATUS "cuSolverMp: NVHPC_ROOT autodetected = ${NVHPC_ROOT}")
                    break()
                endif()
            endforeach()
        endif()
        if(NVHPC_ROOT)
            if(NOT NVHPC_CUDA_SUBDIR)
                file(GLOB _cuda_vers RELATIVE "${NVHPC_ROOT}/math_libs" "${NVHPC_ROOT}/math_libs/*")
                list(SORT _cuda_vers ORDER DESCENDING)
                foreach(_v ${_cuda_vers})
                    if(EXISTS "${NVHPC_ROOT}/math_libs/${_v}/targets/x86_64-linux/include/cusolverMp.h")
                        set(NVHPC_CUDA_SUBDIR "${_v}")
                        message(STATUS "cuSolverMp: CUDA subdir autodetected = ${NVHPC_CUDA_SUBDIR}")
                        break()
                    endif()
                endforeach()
            endif()
            if(NOT NVHPC_CUDA_SUBDIR)
                message(FATAL_ERROR
                    "cuSolverMp: no usable CUDA subdir found under ${NVHPC_ROOT}/math_libs/")
            endif()
            set(CUSOLVERMP_INCLUDE
                "${NVHPC_ROOT}/math_libs/${NVHPC_CUDA_SUBDIR}/targets/x86_64-linux/include")
            set(CUSOLVERMP_LIBDIR
                "${NVHPC_ROOT}/math_libs/${NVHPC_CUDA_SUBDIR}/lib64")
        else()
            # (e) CUDA Toolkit layout — CUDA 12.8+ ships `cuda-cusolvermp-12-X`
            foreach(_cuda_root "/usr/local/cuda" "/usr/local/cuda-12.9"
                              "/usr/local/cuda-12.8")
                set(_inc "${_cuda_root}/targets/x86_64-linux/include/cusolverMp.h")
                set(_lib "${_cuda_root}/targets/x86_64-linux/lib/libcusolverMp.so")
                if(EXISTS "${_inc}" AND EXISTS "${_lib}")
                    set(CUSOLVERMP_INCLUDE "${_cuda_root}/targets/x86_64-linux/include")
                    set(CUSOLVERMP_LIBDIR  "${_cuda_root}/targets/x86_64-linux/lib")
                    message(STATUS "cuSolverMp: found alongside CUDA Toolkit at ${_cuda_root}")
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
                "or set \$NVHPC_ROOT / \$HPCSDK_ROOT, or install the pip "
                "wheels (nvidia-cusolvermp-cu13).  See src/ffi/PORTING.md.")
        endif()
    endif()

    # Resolve the concrete lib files (accept versioned soname for pip wheels,
    # which ship libcusolverMp.so.0 with no bare-.so dev symlink).
    if(NOT CUSOLVERMP_LIB)
        if(EXISTS "${CUSOLVERMP_LIBDIR}/libcusolverMp.so")
            set(CUSOLVERMP_LIB "${CUSOLVERMP_LIBDIR}/libcusolverMp.so")
        elseif(EXISTS "${CUSOLVERMP_LIBDIR}/libcusolverMp.so.0")
            set(CUSOLVERMP_LIB "${CUSOLVERMP_LIBDIR}/libcusolverMp.so.0")
        endif()
    endif()
    if(NOT CUBLASMP_LIB)
        if(EXISTS "${CUSOLVERMP_LIBDIR}/libcublasmp.so")
            set(CUBLASMP_LIB "${CUSOLVERMP_LIBDIR}/libcublasmp.so")
        elseif(EXISTS "${CUSOLVERMP_LIBDIR}/libcublasmp.so.0")
            set(CUBLASMP_LIB "${CUSOLVERMP_LIBDIR}/libcublasmp.so.0")
        endif()
    endif()
    if(NOT EXISTS "${CUSOLVERMP_INCLUDE}/cusolverMp.h")
        message(FATAL_ERROR "cusolverMp.h not found at ${CUSOLVERMP_INCLUDE}/. See src/ffi/PORTING.md.")
    endif()
    if(NOT CUSOLVERMP_LIB OR NOT EXISTS "${CUSOLVERMP_LIB}")
        message(FATAL_ERROR
            "libcusolverMp.so[.0] not found in ${CUSOLVERMP_LIBDIR}/ "
            "(set -DCUSOLVERMP_LIB=<full path>).")
    endif()
    if(NOT CUBLASMP_LIB OR NOT EXISTS "${CUBLASMP_LIB}")
        message(FATAL_ERROR
            "libcublasmp.so[.0] not found (set -DCUBLASMP_LIB=<full path>; the "
            "pip cuBLASMp wheel installs it under nvidia/cublasmp/.../lib).")
    endif()

    # ------------------------------------------------------------------------
    #  ABI sanity check: CUDA toolkit major must match the NVHPC math_libs
    #  subdir where cusolverMp was found (major mismatch → ABI breakage).
    #  Only meaningful for the NVHPC path (NVHPC_CUDA_SUBDIR set).
    # ------------------------------------------------------------------------
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
                "libcusolverMp.so is ABI-incompatible across CUDA major versions.")
        elseif(_tk_minor AND _hp_minor AND NOT "${_tk_minor}" STREQUAL "${_hp_minor}")
            message(WARNING
                "CUDA minor-version mismatch — toolkit ${TOOLKIT_CUDA_VERSION} "
                "vs NVHPC math_libs ${NVHPC_CUDA_SUBDIR}.  Usually OK within a "
                "major line, but test before shipping.")
        endif()
        message(STATUS "CUDA toolkit version: ${TOOLKIT_CUDA_VERSION} "
                       "(NVHPC math_libs: ${NVHPC_CUDA_SUBDIR}) — compatible")
    endif()

    # NCCL header + libs.
    set(NCCL_INCLUDE "/usr/include" CACHE PATH "Directory containing nccl.h")
    if(NOT EXISTS "${NCCL_INCLUDE}/nccl.h" AND NVHPC_ROOT)
        set(NCCL_INCLUDE "${NVHPC_ROOT}/comm_libs/${NVHPC_CUDA_SUBDIR}/nccl/include")
    endif()
    if(NOT EXISTS "${NCCL_INCLUDE}/nccl.h")
        message(FATAL_ERROR "nccl.h not found; pass -DNCCL_INCLUDE=<path>.")
    endif()
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
    if(NOT NCCL_LIBRARY)
        # pip nvidia-nccl wheel ships libnccl.so.2 under site-packages.
        message(FATAL_ERROR "libnccl not found; pass -DNCCL_LIBRARY=<full path to libnccl.so[.2]>.")
    endif()
endif()  # LORRAX_FFI_ENABLE_CUSOLVERMP

# ============================================================================
#  Parallel HDF5   (LORRAX_FFI_ENABLE_PHDF5)
# ============================================================================
if(LORRAX_FFI_ENABLE_PHDF5)
    # Default stage dir (bind-mounted by run_shifter.sh).  Override with
    # -DHDF5_ROOT / $HDF5_ROOT / $HDF5_DIR (find_package(HDF5) honours these).
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
            "Install a parallel build (conda-forge 'hdf5=*mpi_openmpi_*', "
            "cray-hdf5-parallel, ...) or set HDF5_ROOT / -DHDF5_DIR=...\n"
            "See src/ffi/PORTING.md.")
    endif()
    message(STATUS "HDF5: ${HDF5_VERSION} at ${HDF5_INCLUDE_DIRS} (parallel)")
endif()

# ============================================================================
#  MPI (shared by PHDF5 + SLATE)
# ============================================================================
if(LORRAX_FFI_NEED_MPI)
    # We do NOT use find_package(MPI): the NVIDIA JAX container ships HPC-X
    # OpenMPI at /opt/hpcx/ompi, so mpicc would be OpenMPI regardless of which
    # MPI ABI the staged HDF5 was built against.  run_shifter.sh picks a stack
    # (LORRAX_PHDF5_MPI_STACK = openmpi | mpich) and exports the paths; either
    # var can also be set directly (-DLORRAX_MPI_INCLUDE_DIR=, etc.).
    if(NOT DEFINED LORRAX_MPI_INCLUDE_DIR AND DEFINED ENV{LORRAX_MPI_INCLUDE_DIR})
        set(LORRAX_MPI_INCLUDE_DIR "$ENV{LORRAX_MPI_INCLUDE_DIR}" CACHE PATH "MPI header dir")
    endif()
    if(NOT DEFINED LORRAX_MPI_INCLUDE_DIR)
        find_path(LORRAX_MPI_INCLUDE_DIR mpi.h
            HINTS "${HDF5_INCLUDE_DIRS}" "/opt/hpcx/ompi/include" "/usr/lib/x86_64-linux-gnu/openmpi/include"
            NO_DEFAULT_PATH)
    endif()
    if(NOT LORRAX_MPI_INCLUDE_DIR OR NOT EXISTS "${LORRAX_MPI_INCLUDE_DIR}/mpi.h")
        message(FATAL_ERROR
            "mpi.h not found.  Set LORRAX_MPI_INCLUDE_DIR (run_shifter.sh "
            "does this automatically via LORRAX_PHDF5_MPI_STACK).")
    endif()

    if(NOT DEFINED LORRAX_MPICH_LIB_DIR AND DEFINED ENV{LORRAX_MPICH_LIB_DIR})
        set(LORRAX_MPICH_LIB_DIR "$ENV{LORRAX_MPICH_LIB_DIR}" CACHE PATH "MPI runtime lib dir")
    endif()
    if(NOT DEFINED LORRAX_MPICH_LIB_DIR)
        # Fallback chases the OpenMPI path (verified default).  On Cray/MPICH
        # sites this produces DT_NEEDED libmpi.so.40, wrong under --module=mpich
        # (see KNOWN_SANDBOX_ERRORS 2026-05-10) — invoke build.sh via
        # run_shifter.sh so LORRAX_MPICH_LIB_DIR is exported, or pass -D.
        message(WARNING
            "LORRAX_MPICH_LIB_DIR not set — falling back to /opt/hpcx/ompi/lib "
            "(HPC-X OpenMPI).  Correct only for the OpenMPI build path.")
        set(LORRAX_MPICH_LIB_DIR "/opt/hpcx/ompi/lib" CACHE PATH "MPI runtime lib dir")
    endif()
    # NOTE on SONAME resolution: cmake's find_library follows the libmpi.so
    # symlink and the linker records the target's SONAME as DT_NEEDED.  On Cray
    # this yields libmpi_gnu_91.so.12 (the Cray MPICH ABI tag); the staged phdf5
    # dir aliases HDF5's DT_NEEDED onto the same library so both resolve to one
    # process-global libmpi.
    find_library(LORRAX_MPI_LIBRARY NAMES mpi
        HINTS "${LORRAX_MPICH_LIB_DIR}" NO_DEFAULT_PATH)
    if(NOT LORRAX_MPI_LIBRARY)
        message(FATAL_ERROR "libmpi not found in ${LORRAX_MPICH_LIB_DIR}.")
    endif()
    message(STATUS "MPI:  include=${LORRAX_MPI_INCLUDE_DIR} lib=${LORRAX_MPI_LIBRARY}")
endif()

# ============================================================================
#  SLATE   (LORRAX_FFI_ENABLE_SLATE — auto-skips if not found)
# ============================================================================
set(LORRAX_HAVE_SLATE 0)
if(LORRAX_FFI_ENABLE_SLATE)
    if(NOT DEFINED LORRAX_SLATE_INSTALL_DIR AND DEFINED ENV{LORRAX_SLATE_INSTALL_DIR})
        set(LORRAX_SLATE_INSTALL_DIR "$ENV{LORRAX_SLATE_INSTALL_DIR}" CACHE PATH "SLATE install prefix")
    endif()
    if(NOT DEFINED LORRAX_SLATE_INSTALL_DIR)
        set(LORRAX_SLATE_INSTALL_DIR "$ENV{HOME}/software/slate/install"
            CACHE PATH "SLATE install prefix (default: \$HOME/software/slate/install)")
    endif()
    set(slate_DIR    "${LORRAX_SLATE_INSTALL_DIR}/lib64/cmake/slate")
    set(blaspp_DIR   "${LORRAX_SLATE_INSTALL_DIR}/lib64/cmake/blaspp")
    set(lapackpp_DIR "${LORRAX_SLATE_INSTALL_DIR}/lib64/cmake/lapackpp")
    # slateConfig expects MPI::MPI_CXX; synthesise it from the pinned MPI stack.
    if(NOT TARGET MPI::MPI_CXX)
        add_library(MPI::MPI_CXX INTERFACE IMPORTED)
        set_target_properties(MPI::MPI_CXX PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${LORRAX_MPI_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES      "${LORRAX_MPI_LIBRARY}")
    endif()
    find_package(slate CONFIG)
    if(slate_FOUND)
        message(STATUS "SLATE: found via ${LORRAX_SLATE_INSTALL_DIR}")
        set(LORRAX_HAVE_SLATE 1)
    else()
        message(STATUS
            "SLATE: not found at ${LORRAX_SLATE_INSTALL_DIR} — ffi.slate targets "
            "skipped.  Set -DLORRAX_SLATE_INSTALL_DIR=... to include them.")
    endif()
endif()

# ============================================================================
# Build lorrax_ffi.so
# ============================================================================
set(LORRAX_FFI_SOURCES api.cc)
if(LORRAX_FFI_ENABLE_CUSOLVERMP)
    list(APPEND LORRAX_FFI_SOURCES
        ../../cusolvermp/cpp/context.cc
        ../../cusolvermp/cpp/eigh_ffi.cc
        ../../cusolvermp/cpp/batched_potrf_ffi.cc
        ../../cusolvermp/cpp/batched_potrs_ffi.cc
        ../../cusolvermp/cpp/batched_solve_lu_ffi.cc
        ../../cublasmp/cpp/batched_gemm_ffi.cc
        ../../cublasmp/cpp/batched_w_solve_ffi.cc
        ../../cublasmp/cpp/w_solve_kernels.cu
        ../../cusolvermg/cpp/eigh_mg_ffi.cc
    )
endif()
if(LORRAX_FFI_ENABLE_PHDF5)
    list(APPEND LORRAX_FFI_SOURCES
        ../../phdf5/cpp/context.cc
        ../../phdf5/cpp/write_ffi.cc
        ../../phdf5/cpp/read_ffi.cc
    )
endif()
if(LORRAX_HAVE_SLATE)
    list(APPEND LORRAX_FFI_SOURCES
        ../../slate/cpp/context.cc
        ../../slate/cpp/eigh_ffi.cc
        ../../slate/cpp/potrf_ffi.cc
        ../../slate/cpp/trsm_ffi.cc
        ../../slate/cpp/batched_potrf_ffi.cc
        ../../slate/cpp/batched_trsm_ffi.cc
    )
endif()

add_library(lorrax_ffi SHARED ${LORRAX_FFI_SOURCES})

# Compile-time subpackage flags — api.cc (and any shared TU) guards its
# per-subpackage entry points on these so a partial build has no dangling refs.
if(LORRAX_FFI_ENABLE_CUSOLVERMP)
    target_compile_definitions(lorrax_ffi PRIVATE LORRAX_FFI_HAVE_CUSOLVERMP=1)
endif()
if(LORRAX_FFI_ENABLE_PHDF5)
    target_compile_definitions(lorrax_ffi PRIVATE LORRAX_FFI_HAVE_PHDF5=1)
endif()
if(LORRAX_HAVE_SLATE)
    target_compile_definitions(lorrax_ffi PRIVATE LORRAX_FFI_HAVE_SLATE=1)
endif()

# RPATH: only include dirs that a given subpackage actually needs.
set(_lorrax_rpath "")
if(LORRAX_FFI_ENABLE_CUSOLVERMP)
    list(APPEND _lorrax_rpath "${CUSOLVERMP_LIBDIR}")
    if(NVHPC_ROOT)
        list(APPEND _lorrax_rpath "${NVHPC_ROOT}/math_libs/${NVHPC_CUDA_SUBDIR}/lib64")
    endif()
    get_filename_component(_cublasmp_dir "${CUBLASMP_LIB}" DIRECTORY)
    list(APPEND _lorrax_rpath "${_cublasmp_dir}")
    get_filename_component(_nccl_dir "${NCCL_LIBRARY}" DIRECTORY)
    list(APPEND _lorrax_rpath "${_nccl_dir}")
endif()
if(LORRAX_FFI_ENABLE_PHDF5)
    list(APPEND _lorrax_rpath "/lorrax_phdf5/lib")
endif()
if(LORRAX_HAVE_SLATE)
    list(APPEND _lorrax_rpath "/lorrax_slate/lib" "${LORRAX_SLATE_INSTALL_DIR}/lib64")
endif()
list(JOIN _lorrax_rpath ";" _lorrax_rpath_str)
set_target_properties(lorrax_ffi PROPERTIES
    OUTPUT_NAME "lorrax_ffi"
    PREFIX "lib"
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "${_lorrax_rpath_str}"
)

# Include dirs — always JAX FFI + CUDA; per-subpackage dirs when enabled.
target_include_directories(lorrax_ffi PRIVATE
    "${JAX_FFI_INCLUDE_DIR}"
    "${CUDA_TOOLKIT_ROOT}/include"
    "${CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
)
if(LORRAX_FFI_ENABLE_CUSOLVERMP)
    target_include_directories(lorrax_ffi PRIVATE
        "${CUSOLVERMP_INCLUDE}" "${NCCL_INCLUDE}")
    get_filename_component(_cublasmp_inc "${CUBLASMP_LIB}" DIRECTORY)
    target_include_directories(lorrax_ffi PRIVATE "${_cublasmp_inc}/../include")
endif()
if(LORRAX_FFI_ENABLE_PHDF5)
    target_include_directories(lorrax_ffi PRIVATE "${HDF5_INCLUDE_DIRS}")
endif()
if(LORRAX_FFI_NEED_MPI)
    target_include_directories(lorrax_ffi PRIVATE "${LORRAX_MPI_INCLUDE_DIR}")
endif()

find_package(Threads REQUIRED)

# Link — always cudart + threads; per-subpackage libs when enabled.  CAL is
# not linked: cuSolverMp >= 0.7.0 / cuBLASMp >= 0.5.0 take an ncclComm_t
# directly, matching the pure-pip CUDA-13 stack (no libcal ships at all).
target_link_libraries(lorrax_ffi PRIVATE "${CUDART_LIBRARY}" Threads::Threads)
if(LORRAX_FFI_ENABLE_CUSOLVERMP)
    target_link_libraries(lorrax_ffi PRIVATE
        "${CUSOLVERMP_LIB}"
        "${CUBLASMP_LIB}"
        "${NCCL_LIBRARY}"
        "${CUSOLVER_LIBRARY}"
        "${CUSOLVERMG_LIBRARY}"
    )
endif()
if(LORRAX_FFI_ENABLE_PHDF5)
    target_link_libraries(lorrax_ffi PRIVATE ${HDF5_LIBRARIES})
endif()
if(LORRAX_FFI_NEED_MPI)
    target_link_libraries(lorrax_ffi PRIVATE "${LORRAX_MPI_LIBRARY}")
endif()
if(LORRAX_HAVE_SLATE)
    target_link_libraries(lorrax_ffi PRIVATE slate)
endif()

target_compile_options(lorrax_ffi PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-variable>
)

if(LORRAX_FFI_ENABLE_CUSOLVERMP)
    # Ensure nvcc passes -fPIC to the host compiler for .cu TUs.
    set_target_properties(lorrax_ffi PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON)
endif()

message(STATUS "lorrax_ffi will link:")
if(LORRAX_FFI_ENABLE_CUSOLVERMP)
    message(STATUS "  cuSolverMp = ${CUSOLVERMP_LIB}")
    message(STATUS "  cuBLASMp   = ${CUBLASMP_LIB}")
    message(STATUS "  NCCL       = ${NCCL_LIBRARY}")
endif()
message(STATUS "  CUDART     = ${CUDART_LIBRARY}")
if(LORRAX_FFI_ENABLE_PHDF5)
    message(STATUS "  HDF5       = ${HDF5_LIBRARIES}")
endif()
if(LORRAX_FFI_NEED_MPI)
    message(STATUS "  MPI        = ${LORRAX_MPI_LIBRARY}")
endif()
