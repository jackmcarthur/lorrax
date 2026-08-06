cmake_minimum_required(VERSION 3.22)

# ============================================================================
# THE one FFI build entry point — both platform legs of LORRAX's FFI live
# behind this file (layout: docs/architecture/ffi_layout.md):
#
#   cmake -S src/ffi/cpp -DLORRAX_FFI_PLATFORM=cuda ...  -> liblorrax_ffi.so
#       (cuSOLVERMp/cuBLASMp/cuFFT/NCCL + phdf5 + SLATE; runs in the CUDA
#        container; entry scripts: build.sh, config/frontera/build_ffi.sh)
#   cmake -S src/ffi/cpp -DLORRAX_FFI_PLATFORM=host ...  -> liblorrax_ffi_host.so
#       (CUDA-free by construction: phdf5 + ScaLAPACK + MKL FFT/GEMM + SLATE
#        gpu_backend=none; entry scripts: build_host.sh,
#        config/frontera/build_ffi_host.sh)
#
# The selector is EXPLICIT and refuses when unset (announce-or-refuse): the
# old scheme selected the leg by which directory cmake was pointed at, an
# implicit dial.  Both legs register the same jax.ffi target names under
# different C++ symbols — see src/ffi/common/ffi_loader.py.
# ============================================================================
set(LORRAX_FFI_PLATFORM "" CACHE STRING
    "Which FFI library to configure: cuda | host (REQUIRED, no default)")

if(LORRAX_FFI_PLATFORM STREQUAL "cuda")
# ============================== CUDA leg ====================================
project(lorrax_ffi LANGUAGES C CXX)

# ----------------------------------------------------------------------------
# Options / paths
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----------------------------------------------------------------------------
# Optional-target switches.  Both default ON so every existing (Perlmutter /
# Cray) invocation is byte-identical — this block is a no-op there.  A
# port that lacks a dependency turns the relevant target OFF from its own
# build script (see config/frontera/):
#   -DLORRAX_FFI_HAVE_CAL=OFF     cuSOLVERMp >= 0.7 is NCCL-native and needs
#                                 no CAL (libcal / cal.h absent from the pip
#                                 wheels).  OFF drops the cal.h include, the
#                                 CAL->NCCL shim, the pre-0.7 CAL comm path,
#                                 and the libcal link; the NCCL comm path
#                                 (create_context, ensure_cublasmp) passes
#                                 ncclComm_t directly.
#   -DLORRAX_FFI_HAVE_PHDF5=OFF   drop the phdf5 subpackage (no parallel HDF5
#                                 / MPI at build time).  The cusolverMp /
#                                 cuBLASMp distributed-linalg targets still
#                                 build; ffi_loader.py skips the absent
#                                 phdf5 handler symbols.
option(LORRAX_FFI_HAVE_CAL   "Build the CAL (<=0.6.x cuSOLVERMp) comm path"      ON)
option(LORRAX_FFI_HAVE_PHDF5 "Build the phdf5 subpackage (needs parallel HDF5)" ON)
# cuBLASMp fused gemm / W-solve carry the only .cu translation unit
# (w_solve_kernels.cu), so they need a full nvcc.  OFF builds an eigh-only
# .so with NO CUDA-language step — the Frontera pip toolchain ships ptxas
# but not the nvcc driver, and the distributed eigensolver needs neither.
# libcublasmp is still linked (cuSOLVERMp's own runtime dependency) and
# ensure_cublasmp (pure host C++) still compiles.
option(LORRAX_FFI_HAVE_CUBLASMP "Build cuBLASMp fused-kernel FFIs (needs nvcc)" ON)
# cuFFT strided flat-k FFT handlers (cpp/cufft/fft_flat_k_cuda_ffi.cc) — the
# CUDA platform mirror of the mklfft host handlers (FFT-FFI, 2026-07-29).
# Plain C++ TU compiled by the host compiler against the CUDA headers (its
# two device kernels are NVRTC-compiled at runtime), so it needs NO nvcc —
# it builds in the Frontera pip toolchain where LORRAX_FFI_HAVE_CUBLASMP is
# OFF.  Auto-downgrades to a loud skip if cufft.h / libcufft / nvrtc are
# absent (LORRAX_FFT_FFI on CUDA then refuses at runtime via probe_target).
option(LORRAX_FFI_HAVE_CUFFT "Build the cuFFT strided flat-k FFT handlers" ON)

# ----------------------------------------------------------------------------
# CUDA language (for .cu kernel translation units compiled by nvcc).  We
# point CMAKE_CUDA_COMPILER at the container's nvcc so it matches the
# toolkit whose headers/runtime we already use for the C++ TUs.  A100 =
# compute_80 / sm_80; the Shifter container on Perlmutter is A100-only.
# Gated on LORRAX_FFI_HAVE_CUBLASMP: the eigh-only build has no .cu sources
# and must not require an nvcc.
# ----------------------------------------------------------------------------
if(LORRAX_FFI_HAVE_CUBLASMP)
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        if(EXISTS "/usr/local/cuda/bin/nvcc")
            set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
        endif()
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80)  # A100 on Perlmutter
    endif()
endif()

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
#  cuFFT + NVRTC — for the LORRAX_FFI_HAVE_CUFFT flat-k handlers.  All four
#  pieces ship in the container toolkit (/usr/local/cuda) AND in the pip
#  nvidia-cufft-cu12 / nvidia-cuda-nvrtc-cu12 wheels staged into the Frontera
#  cuda_root by config/frontera/stage_ffi_deps.sh.
# ----------------------------------------------------------------------------
set(LORRAX_CUFFT_ON 0)
if(LORRAX_FFI_HAVE_CUFFT)
    find_library(CUFFT_LIBRARY cufft
        PATHS "${CUDA_TOOLKIT_ROOT}/lib64"
              "${CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/lib"
        NO_DEFAULT_PATH)
    find_library(NVRTC_LIBRARY nvrtc
        PATHS "${CUDA_TOOLKIT_ROOT}/lib64"
              "${CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/lib"
        NO_DEFAULT_PATH)
    find_path(CUFFT_INCLUDE_DIR cufft.h
        PATHS "${CUDA_TOOLKIT_ROOT}/include"
              "${CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/include"
        NO_DEFAULT_PATH)
    find_path(NVRTC_INCLUDE_DIR nvrtc.h
        PATHS "${CUDA_TOOLKIT_ROOT}/include"
              "${CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/include"
        NO_DEFAULT_PATH)
    if(CUFFT_LIBRARY AND NVRTC_LIBRARY AND CUFFT_INCLUDE_DIR AND NVRTC_INCLUDE_DIR)
        set(LORRAX_CUFFT_ON 1)
        message(STATUS "cufft: strided flat-k FFT handlers ON "
                       "(${CUFFT_LIBRARY}; nvrtc=${NVRTC_LIBRARY})")
    else()
        message(WARNING
            "cufft: cufft.h/libcufft/nvrtc.h/libnvrtc not all found under "
            "${CUDA_TOOLKIT_ROOT} (cufft.h=${CUFFT_INCLUDE_DIR} "
            "lib=${CUFFT_LIBRARY} nvrtc.h=${NVRTC_INCLUDE_DIR} "
            "libnvrtc=${NVRTC_LIBRARY}) — the flat-k FFT handlers are "
            "SKIPPED and LORRAX_FFT_FFI on CUDA meshes will refuse at "
            "runtime.  On Frontera re-run config/frontera/stage_ffi_deps.sh "
            "(it merges the nvidia-cufft/nvrtc wheels into cuda_root).")
    endif()
endif()

# ----------------------------------------------------------------------------
#  Parallel HDF5 + MPI — for the ffi.phdf5 subpackage.  Gated on
#  LORRAX_FFI_HAVE_PHDF5 so a port without parallel HDF5 / MPI (e.g. an
#  eigh-only cuSOLVERMp build) can skip the whole probe.
# ----------------------------------------------------------------------------
if(LORRAX_FFI_HAVE_PHDF5)
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

# MPI — we do NOT use find_package(MPI) because the NVIDIA JAX
# container ships HPC-X OpenMPI at /opt/hpcx/ompi, so mpicc would be
# OpenMPI regardless of which MPI ABI the staged HDF5 was built
# against.  run_shifter.sh picks a stack (LORRAX_PHDF5_MPI_STACK =
# openmpi | mpich) and exports LORRAX_MPI_INCLUDE_DIR + LORRAX_MPICH_LIB_DIR
# pointing at the right paths inside the container.  Either env var
# can also be set directly (-DLORRAX_MPI_INCLUDE_DIR=, etc.).
if(NOT DEFINED LORRAX_MPI_INCLUDE_DIR AND DEFINED ENV{LORRAX_MPI_INCLUDE_DIR})
    set(LORRAX_MPI_INCLUDE_DIR "$ENV{LORRAX_MPI_INCLUDE_DIR}" CACHE PATH "MPI header dir")
endif()
if(NOT DEFINED LORRAX_MPI_INCLUDE_DIR)
    # Last-ditch: look for mpi.h next to HDF5 headers (Cray stage layout).
    find_path(LORRAX_MPI_INCLUDE_DIR mpi.h
        HINTS "${HDF5_INCLUDE_DIRS}" "/opt/hpcx/ompi/include"
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
    # Default chases the OpenMPI path (verified default) so bare cmake
    # runs inside a shifter-gpu container still find something.  This is
    # ONLY appropriate for the LORRAX_PHDF5_MPI_STACK=openmpi build path;
    # on Cray/MPICH sites it produces DT_NEEDED libmpi.so.40 which can't
    # be resolved at runtime under --module=mpich (see KNOWN_SANDBOX_ERRORS
    # 2026-05-10).  build.sh refuses to run if LORRAX_MPICH_LIB_DIR is
    # unset; cmake-by-hand callers should pass -DLORRAX_MPICH_LIB_DIR=...
    # explicitly.
    message(WARNING
        "LORRAX_MPICH_LIB_DIR not set — falling back to /opt/hpcx/ompi/lib "
        "(HPC-X OpenMPI).  This is correct only for the OpenMPI build "
        "path; on Cray/MPICH sites invoke build.sh via run_shifter.sh so "
        "LORRAX_MPICH_LIB_DIR=/opt/udiImage/modules/mpich is exported.")
    set(LORRAX_MPICH_LIB_DIR "/opt/hpcx/ompi/lib" CACHE PATH "MPI runtime lib dir")
endif()
# NOTE on SONAME resolution: cmake's find_library follows the libmpi.so
# symlink, and the linker records the target's SONAME as DT_NEEDED.  On
# Cray Perlmutter the symlink chain inside the container is
#   /opt/udiImage/modules/mpich/libmpi.so → libmpi_gnu_91.so.12.0.0
# with SONAME `libmpi_gnu_91.so.12`, so the FFI ends up with DT_NEEDED
# libmpi_gnu_91.so.12 — the actual Cray MPICH ABI tag.  HDF5's own
# DT_NEEDED is `libmpi_gnu_123.so.12` (a filename alias provided by the
# Cray phdf5 stage); the staged dir contains a symlink mapping that
# filename onto the same MPICH library, so both resolve to one process-
# global libmpi.
find_library(LORRAX_MPI_LIBRARY
    NAMES mpi
    HINTS "${LORRAX_MPICH_LIB_DIR}"
    NO_DEFAULT_PATH)
if(NOT LORRAX_MPI_LIBRARY)
    message(FATAL_ERROR
        "libmpi not found in ${LORRAX_MPICH_LIB_DIR}.")
endif()
message(STATUS "MPI:  include=${LORRAX_MPI_INCLUDE_DIR} lib=${LORRAX_MPI_LIBRARY}")
endif()  # LORRAX_FFI_HAVE_PHDF5

# ----------------------------------------------------------------------------
#  SLATE — for the ffi.slate subpackage.  Found via find_package(slate)
#  pointing at the host install (accessible inside the container via
#  siteFs mount of /global/u2).  Runtime deps (Cray libsci etc.) come from
#  the stage dir populated by src/ffi/cpp/stage/slate_stage_cray.sh.
# ----------------------------------------------------------------------------
if(NOT DEFINED LORRAX_SLATE_INSTALL_DIR AND DEFINED ENV{LORRAX_SLATE_INSTALL_DIR})
    set(LORRAX_SLATE_INSTALL_DIR "$ENV{LORRAX_SLATE_INSTALL_DIR}" CACHE PATH
        "SLATE install prefix")
endif()
if(NOT DEFINED LORRAX_SLATE_INSTALL_DIR)
    set(LORRAX_SLATE_INSTALL_DIR "$ENV{HOME}/software/slate/install"
        CACHE PATH "SLATE install prefix (default: host build under \$HOME/software/slate/install)")
endif()

# slateConfig.cmake internally find_dependency(blaspp, lapackpp), which
# live alongside slate in ${LORRAX_SLATE_INSTALL_DIR}.  Point all three
# at their CMake config dirs so the find chain resolves.
set(slate_DIR    "${LORRAX_SLATE_INSTALL_DIR}/lib64/cmake/slate")
set(blaspp_DIR   "${LORRAX_SLATE_INSTALL_DIR}/lib64/cmake/blaspp")
set(lapackpp_DIR "${LORRAX_SLATE_INSTALL_DIR}/lib64/cmake/lapackpp")

# slateConfig expects MPI::MPI_CXX (from find_package(MPI)).  We deliberately
# avoid that search above so we can pin to the chosen MPI stack.  Create a
# stand-in IMPORTED target that points at the same libmpi / mpi.h.
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
        "will be skipped.  Build SLATE (see sources/SLATE) or set "
        "-DLORRAX_SLATE_INSTALL_DIR=... to include them.")
    set(LORRAX_HAVE_SLATE 0)
endif()

# ----------------------------------------------------------------------------
# Build lorrax_ffi.so
# ----------------------------------------------------------------------------
set(LORRAX_FFI_SOURCES
    common/api.cc
    cusolvermp/context.cc
    cusolvermp/eigh_ffi.cc
    cusolvermp/batched_potrf_ffi.cc
    cusolvermp/batched_potrs_ffi.cc
    cusolvermp/batched_solve_lu_ffi.cc
)
if(LORRAX_FFI_HAVE_CUBLASMP)
    list(APPEND LORRAX_FFI_SOURCES
        cublasmp/batched_gemm_ffi.cc
        cublasmp/batched_w_solve_ffi.cc
        cublasmp/w_solve_kernels.cu
    )
endif()
if(LORRAX_CUFFT_ON)
    list(APPEND LORRAX_FFI_SOURCES
        cufft/fft_flat_k_cuda_ffi.cc
    )
endif()
if(LORRAX_FFI_HAVE_PHDF5)
    list(APPEND LORRAX_FFI_SOURCES
        phdf5/api.cc
        phdf5/context.cc
        phdf5/write_ffi.cc
        phdf5/read_ffi.cc
    )
endif()
if(LORRAX_HAVE_SLATE)
    list(APPEND LORRAX_FFI_SOURCES
        slate/context.cc
        slate/eigh_ffi.cc
        slate/potrf_ffi.cc
        slate/trsm_ffi.cc
        slate/batched_potrf_ffi.cc
        slate/batched_trsm_ffi.cc
    )
endif()

add_library(lorrax_ffi SHARED ${LORRAX_FFI_SOURCES})

set_target_properties(lorrax_ffi PROPERTIES
    OUTPUT_NAME "lorrax_ffi"
    PREFIX "lib"
    # RPATH points at the NVHPC directory so the runtime linker finds
    # libcusolverMp.so and libcal.so without LD_LIBRARY_PATH.
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "${CUSOLVERMP_LIBDIR};${NVHPC_ROOT}/math_libs/${NVHPC_CUDA_SUBDIR}/lib64;/lorrax_phdf5/lib;/lorrax_slate/lib;${LORRAX_SLATE_INSTALL_DIR}/lib64"
)

# Third-party headers (XLA FFI + vendor toolkits) as SYSTEM — they are not
# warning-clean under -Wall -Wextra and are not ours to fix; LORRAX's own
# sources stay under the full warning set (no blanket -Wno-unused-*).
target_include_directories(lorrax_ffi SYSTEM PRIVATE
    "${JAX_FFI_INCLUDE_DIR}"
    "${CUSOLVERMP_INCLUDE}"
    "${NCCL_INCLUDE}"
    "${CUDA_TOOLKIT_ROOT}/include"
    "${CUDA_TOOLKIT_ROOT}/targets/x86_64-linux/include"
    "${HDF5_INCLUDE_DIRS}"
    "${LORRAX_MPI_INCLUDE_DIR}"
)
target_include_directories(lorrax_ffi PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/common"
)

find_package(Threads REQUIRED)

# libcal must be linked EXPLICITLY: context.cc references cal_comm_*
# for the <=0.6.x CAL comm path.  The CAL-era cuBLASMp (<=0.4.x) NEEDed
# libcal.so.0 itself, so the symbols used to resolve transitively; the
# NCCL-era cuBLASMp (>=0.5.0) doesn't, and without this line the .so
# fails to dlopen with "undefined symbol: cal_comm_destroy".
target_link_libraries(lorrax_ffi PRIVATE
    "${CUSOLVERMP_LIBDIR}/libcusolverMp.so"
    "${CUSOLVERMP_LIBDIR}/libcublasmp.so"
    "${NCCL_LIBRARY}"
    "${CUDART_LIBRARY}"
    "${CUSOLVER_LIBRARY}"
    Threads::Threads
)
# CAL is only linked for the <=0.6.x comm path; NCCL-native cuSOLVERMp
# (>=0.7, the pip wheels) ships no libcal.
if(LORRAX_FFI_HAVE_CAL)
    target_link_libraries(lorrax_ffi PRIVATE "${CUSOLVERMP_LIBDIR}/libcal.so")
endif()
if(LORRAX_CUFFT_ON)
    # ${CMAKE_DL_LIBS}: the cufft TU resolves the driver API (cuModule*/
    # cuLaunchKernel) via dlsym — no link-time libcuda (absent on
    # build/login nodes); NVRTC compiles its two device kernels at runtime.
    target_link_libraries(lorrax_ffi PRIVATE
        "${CUFFT_LIBRARY}" "${NVRTC_LIBRARY}" ${CMAKE_DL_LIBS})
    target_include_directories(lorrax_ffi PRIVATE
        "${CUFFT_INCLUDE_DIR}" "${NVRTC_INCLUDE_DIR}")
endif()
if(LORRAX_FFI_HAVE_PHDF5)
    target_link_libraries(lorrax_ffi PRIVATE ${HDF5_LIBRARIES} "${LORRAX_MPI_LIBRARY}")
endif()
if(LORRAX_HAVE_SLATE)
    target_link_libraries(lorrax_ffi PRIVATE slate)
endif()

# Propagate the switches to the C++ sources (api.cc, cusolvermp/context.cc).
target_compile_definitions(lorrax_ffi PRIVATE
    LORRAX_FFI_HAVE_CAL=$<BOOL:${LORRAX_FFI_HAVE_CAL}>
    LORRAX_FFI_HAVE_PHDF5=$<BOOL:${LORRAX_FFI_HAVE_PHDF5}>
)

# -Wno-unused-parameter/-Wno-unused-variable removed 2026-08-01 (seam-audit
# leftover): the blanket opt-outs also hid genuinely dead locals.  Warnings
# are not errors here (no -Werror); host-leg sources were rebuilt clean —
# CUDA-only TUs get their warning pass at the next rtx/Perlmutter build.
target_compile_options(lorrax_ffi PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
)

# Ensure nvcc passes -fPIC to the host compiler for .cu TUs linked into
# the shared lib.  CMAKE_POSITION_INDEPENDENT_CODE usually handles this,
# but being explicit avoids relocation R_X86_64_PC32 link errors.
set_target_properties(lorrax_ffi PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

message(STATUS "lorrax_ffi will link:")
message(STATUS "  cuSOLVERMp = ${CUSOLVERMP_LIBDIR}/libcusolverMp.so")
message(STATUS "  NCCL       = ${NCCL_LIBRARY}")
message(STATUS "  CUDART     = ${CUDART_LIBRARY}")
message(STATUS "  HDF5       = ${HDF5_LIBRARIES}")
message(STATUS "  MPI        = ${LORRAX_MPI_LIBRARY}")

elseif(LORRAX_FFI_PLATFORM STREQUAL "host")
# ============================== host leg ====================================
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
# Deliberately a separate LEG from the CUDA leg above (same file, selected
# by -DLORRAX_FFI_PLATFORM): that leg hard-requires cuSOLVERMp / NCCL / the
# CUDA language and runs inside the Shifter/apptainer container; this one is
# CUDA-free BY CONSTRUCTION.  It
# compiles the SAME phdf5 collective MPI-IO core (context.cc + read_ffi.cc +
# write_ffi.cc) as the CUDA lib, with -DLORRAX_FFI_NO_CUDA so the
# device-staging seams collapse: the read tail becomes a std::memcpy into the
# host-resident XLA output buffer instead of a cudaMemcpyAsync H2D, and the
# write path hands H5Dwrite the XLA input buffer directly instead of staging
# it D2H through a pinned buffer (see cpp/phdf5/platform_seam.h).
# ffi_loader.py registers
# this library's handlers under platform="cpu" (the CUDA library's under
# platform="CUDA") with the same jax.ffi target names — the jaxlib
# cpu-lapack / cuda-cusolver registration split.
#
# Three optional component groups, each CUDA-free and each INDEPENDENT of the
# others (they were not independent before 2026-07-31 — see the "HOST
# NUMERICAL LIBRARIES" block further down for what that cost):
#   LORRAX_FFI_HAVE_PHDF5     (ON)  parallel-HDF5 read + write handlers
#                                   + lifecycle
#   LORRAX_HOST_HAVE_SCALAPACK(ON)  ScaLAPACK distributed eigh + fused LU,
#                                   plus the CBLAS GEMM and MKL-FFT handlers
#                                   that ride the same link line
#   LORRAX_HOST_HAVE_SLATE    (ON)  SLATE gpu_backend=none host handlers
# Every group downgrades to a "skipped" STATUS line if its dependency is not
# found, so a phdf5-only host lib still builds on a cluster with neither a
# SLATE install nor a ScaLAPACK.
#
# The full 11-target lib (phdf5×4, slate×5, scalapack×2) is built by
# config/frontera/build_ffi_host.sh (9 targets first built 2026-07-25;
# scalapack eigh added 2026-07-26, workstream V; the phdf5 WRITE handler
# added 2026-07-26, workstream AE — it is what makes the mpi4py +
# h5py-parallel overlay unnecessary on CPU).

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LORRAX_FFI_HAVE_PHDF5  "Build the phdf5 host read+write handlers (needs parallel HDF5)" ON)
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
        phdf5/api.cc
        phdf5/context.cc
        phdf5/read_ffi.cc
        phdf5/write_ffi.cc
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

#  ===========================================================================
#  HOST NUMERICAL LIBRARIES — WHAT IS LINKED, WHICH API IT IMPLEMENTS, AND
#  WHAT ITS ABSENCE LOOKS LIKE.       Read this before debugging a link error.
#  ===========================================================================
#
#  Four of the host handler families call a numerical library.  THREE OF THE
#  FOUR TARGET A PUBLISHED, MULTI-VENDOR API and are therefore not tied to any
#  one vendor; only SLATE is a single-implementation library.  The table is
#  the map from "symbol the linker could not find" back to "thing that was
#  not resolved here".
#
#  family    API it calls              implementations that provide it
#  --------  ------------------------  -------------------------------------
#  scalapack ScaLAPACK + C-BLACS       Intel MKL (libmkl_scalapack_lp64
#            (Fortran ABI; the ELEVEN  defines 8 + libmkl_blacs_<mpi>_lp64
#            symbols listed below,     defines 3 = all eleven, measured),
#            declared in scalapack/    Cray LibSci (libsci_*_mpi_*), netlib
#            cpp/blacs_grid.h)         ScaLAPACK, AMD AOCL.  NOT SLATE's
#                                      libslate_scalapack_api: 6/11, and
#                                      refused -- see RESOLVE 1/4 below.
#  mklblas   CBLAS (cblas_dgemm /      Intel MKL, Cray LibSci, OpenBLAS,
#            cblas_zgemm; the batched  BLIS, ATLAS.  Only the *_batch entry
#            entry is an extension)    is vendor-variable, and that is asked
#                                      at RUNTIME, never here.
#  mklfft    Intel DFTI descriptor     Intel MKL ONLY.  This is the one host
#            API (DftiCreateDescriptor family whose API has a single vendor;
#            / DftiSetValue / ...)     see the note in the mklfft block.
#  slate     slate:: C++ templates     ICL SLATE ONLY (a C++ library, not an
#                                      ABI — no second implementation exists)
#
#  THE THREE FAMILY NAMES ARE HISTORICAL, NOT VENDOR ASSERTIONS.  `mklblas`
#  builds and runs against any CBLAS; `mklfft`'s Python module also drives the
#  cuFFT handlers on the CUDA platform under the SAME jax.ffi target names.
#  Nothing at runtime can observe which vendor supplied ScaLAPACK or CBLAS —
#  see src/ffi/common/ffi_loader.py, which keys only on LORRAX's own handler
#  symbol names (ScalapackEighHostFfi, ...).
#
#  RESOLUTION IS ONE BLOCK, FOUR CONSUMERS.  Before 2026-07-31 the ScaLAPACK,
#  CBLAS and DFTI resolutions were all nested inside `if(slate_FOUND)`, and
#  the DFTI one additionally inside the MKL branch of the ScaLAPACK probe.
#  Two capabilities were silently lost as a result:
#    * no SLATE install  ->  ScaLAPACK eigh/LU, GEMM and FFT handlers ALL
#      dropped, though none of them calls SLATE (proved by compiling
#      cpp/scalapack/{eigh,solve_lu}_ffi.cc and mkl{fft,blas}/cpp/*.cc with
#      only the XLA-FFI and MPI header paths — they need no SLATE header, and
#      the resulting objects reference no slate symbol);
#    * ScaLAPACK given explicitly via -DLORRAX_SCALAPACK_LIBRARIES  ->  the
#      FFT handlers were not merely skipped, they were never reached, so the
#      configure log said nothing at all about them.
#  Both are fixed by resolving first and appending second, below.
#
#  WHAT A MISSING PIECE LOOKS LIKE, AND WHERE TO FIX IT
#  ----------------------------------------------------
#  Every one of these is announced at configure time by the `message(STATUS)`
#  lines below — read the configure log FIRST; only reach for the runtime
#  symptom if you no longer have it.
#
#  unresolved at load: any of
#         pzheevd_  pdsyevd_  pzgetrf_  pdgetrf_  pzgetrs_  pdgetrs_
#         numroc_   descinit_ Csys2blacs_handle  Cblacs_gridinit
#         Cblacs_gridinfo
#      -> no ScaLAPACK+BLACS on the link line.  Set -DLORRAX_MKL_ROOT (or
#         $MKLROOT), or pass a whole -DLORRAX_SCALAPACK_LIBRARIES link line.
#         Those ELEVEN names are the entire ScaLAPACK surface LORRAX uses —
#         the complete undefined-symbol set of solve_lu_ffi.cc + eigh_ffi.cc
#         outside libc, measured, not estimated.  Anything implementing them
#         can back these handlers.
#  unresolved at load: cblas_dgemm / cblas_zgemm
#      -> the CBLAS came from a header-only prefix with no library on the
#         link line.  cblas_?gemm_batch will NEVER appear here: it is looked
#         up at runtime and its absence selects the plain-GEMM loop.
#  unresolved at load: Dfti*
#      -> the MKL runtime is not on the load path.  DFTI lives inside
#         libmkl_intel_lp64/libmkl_core, which the ScaLAPACK link line
#         already brings in, so this can only mean a load-path problem.
#  unresolved at load: slate::* (mangled _ZN5slate...)
#      -> libslate.so is not on the load path.  Note this is the ONLY family
#         whose absence cannot be worked around with another vendor.
#  configure says "SLATE (host): not found" and you wanted SLATE
#      -> -DLORRAX_SLATE_HOST_INSTALL_DIR must point at the prefix that
#         CONTAINS lib64/cmake/slate/slateConfig.cmake.
#  hangs or aborts inside blacs_gridinit with no error
#      -> BLACS flavour does not match the MPI actually linked (MKL ships one
#         BLACS per MPI: mkl_blacs_intelmpi_lp64 vs mkl_blacs_openmpi_lp64).
#         This links cleanly and only fails at the first collective.
#
#  The runtime load path for all of the above is set by
#  config/frontera/build_ffi_host.sh (which bakes the resolved directories
#  into the library's own search path) — see its header for the Frontera
#  values.
#  ===========================================================================

option(LORRAX_HOST_HAVE_SCALAPACK
       "Build the ScaLAPACK host handlers (distributed eigh + fused LU)" ON)

# Accumulators.  Nothing is appended to LORRAX_FFI_HOST_SOURCES /
# _host_link_libs / _host_rpaths in the RESOLVE half — the APPEND half
# further down does that, in a fixed order, so the emitted link line is
# independent of which optional pieces were found.
set(_host_math_link   "")   # ScaLAPACK+BLACS+BLAS+OpenMP, in link order
set(_host_math_rpaths "")   # directories the resolved libraries live in
set(_mkl_incdir       "")   # MKL header dir, when MKL is the ScaLAPACK source
set(_scalapack_on     0)
set(_mklfft_on        0)
set(_mklblas_on       0)

# ----------------------------------------------------------------------------
#  RESOLVE 1/4 — ScaLAPACK + C-BLACS.  Required by cpp/scalapack/*_ffi.cc
#  (pXgetrf / pXgetrs / pXheevd / Cblacs_gridinit / descinit / numroc;
#  prototypes hand-declared in cpp/scalapack/blacs_grid.h because ScaLAPACK
#  ships no vendor-neutral C header).
#
#  SLATE DOES NOT SUPPLY THEM, AND CANNOT (measured 2026-07-31, job 7883830).
#  The `slate` target brings in BLAS/LAPACK only, and the reason is NOT the
#  -DSCALAPACK_LIBRARIES="" this repo's SLATE build passes — as this comment
#  used to say.  Two independent facts:
#    * SLATE v2025.05.28's own CMakeLists.txt has the whole `scalapack_api`
#      block COMMENTED OUT ("# todo: requires ScaLAPACK"), so a CMake install
#      never contains libslate_scalapack_api regardless of that setting.
#    * Even when built from the GNUmakefile (or
#      cpp/stage/slate_build_scalapack_api.sh), that library DEFINES only 6 of
#      the eleven names below — every compute routine, none of the
#      grid/descriptor tools, and it CALLS numroc_ / Cblacs_gridinfo itself.
#      It is an OVERLAY on a real ScaLAPACK, never a provider.  LORRAX
#      REFUSES it at run time (cpp/scalapack/blacs_grid.h) because its
#      pXheevd is bug L-2, its shims hard-code MPI_COMM_WORLD, and they
#      hard-wire info=0.  Full measurement: src/ffi/PORTING.md §0b.
#  Workstream L (2026-07-25) had to hand-inject the link line via
#  -DCMAKE_SHARED_LINKER_FLAGS on every configure — that gap is closed here.
#
#  Resolution order:
#    1. -DLORRAX_SCALAPACK_LIBRARIES="<link line>" — vendor-agnostic escape
#       hatch (Cray LibSci, netlib, AOCL, ...; NOT SLATE's overlay — see
#       above).  A CMake list or a space-separated string; raw `-l`/`-L`
#       flags are fine.
#    2. MKL, probed under -DLORRAX_MKL_ROOT / $MKLROOT.  This is the Frontera
#       path (MKL 2020.1 at
#       /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl).
#    3. nothing found -> loud WARNING and the ScaLAPACK handlers are NOT
#       built, so ffi_loader.has_target() answers honestly.  (Before
#       2026-07-31 they were built with the routines left unresolved, and
#       has_target() reported the handler present until the first call died.)
#
#  `-Wl,--no-as-needed` is mandatory: MKL's layered libraries are reached
#  transitively, and ld 2.30+ otherwise drops them from the library's
#  recorded dependency list, producing a .so that loads and then cannot
#  resolve pzgetrf at call time.
# ----------------------------------------------------------------------------
if(LORRAX_HOST_HAVE_SCALAPACK)
    if(NOT DEFINED LORRAX_SCALAPACK_LIBRARIES AND DEFINED ENV{LORRAX_SCALAPACK_LIBRARIES})
        set(LORRAX_SCALAPACK_LIBRARIES "$ENV{LORRAX_SCALAPACK_LIBRARIES}" CACHE STRING
            "Full ScaLAPACK+BLACS link line (overrides the MKL probe)")
    endif()
    if(LORRAX_SCALAPACK_LIBRARIES)
        separate_arguments(_host_math_link UNIX_COMMAND
                          "${LORRAX_SCALAPACK_LIBRARIES}")
        set(_scalapack_on 1)
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
            set(_host_math_link
                "-L${_mkl_libdir}" "-Wl,--no-as-needed"
                "-lmkl_scalapack_lp64" "-l${LORRAX_MKL_BLACS}"
                "-lmkl_intel_lp64" "-l${LORRAX_MKL_THREAD_LIB}" "-lmkl_core")
            if(LORRAX_MKL_THREAD_LIB STREQUAL "mkl_gnu_thread")
                list(APPEND _host_math_link "-lgomp")
            endif()
            list(APPEND _host_math_link "-lpthread" "-lm" "-ldl")
            list(APPEND _host_math_rpaths "${_mkl_libdir}")
            get_filename_component(_mkl_incdir
                                   "${_mkl_libdir}/../../include" ABSOLUTE)
            set(_scalapack_on 1)
            message(STATUS "ScaLAPACK: MKL at ${_mkl_libdir} "
                           "(blacs=${LORRAX_MKL_BLACS}, thread=${LORRAX_MKL_THREAD_LIB})")
        else()
            message(WARNING
                "ScaLAPACK/BLACS not found — the ScaLAPACK host handlers "
                "(distributed eigh + fused LU) are NOT built, and "
                "ffi_loader.has_target('lorrax_scalapack_eigh') will "
                "correctly answer no.  Set -DLORRAX_MKL_ROOT (or \$MKLROOT), "
                "or pass a full -DLORRAX_SCALAPACK_LIBRARIES link line "
                "(Cray LibSci, netlib, AOCL all implement the same API).")
        endif()
    endif()
endif()

# ----------------------------------------------------------------------------
#  RESOLVE 2/4 — CBLAS header for the vendor-BLAS GEMM host handler
#  (cpp/mklblas/gemm_batch_ffi.cc; contract_bands FFI GEMM body, 2026-07-28 —
#  RESHARD_OVERHEAD_MEMO Sec. 4.4 exit (b)).  VENDOR-PORTABLE since
#  2026-07-29 (owner order): it builds against ANY standard CBLAS — Intel MKL
#  via mkl_cblas.h (the Frontera path; zero new link deps) or Cray LibSci /
#  OpenBLAS / BLIS via cblas.h.  cblas_?gemm_batch is an MKL EXTENSION
#  (OpenBLAS ships it too) and is looked up AT RUNTIME inside the TU —
#  present -> the batched entry; absent -> the TU's portable plain
#  cblas_{d,z}gemm loop; the live entry is announced on first use.  Works in
#  principle with Intel MKL or Cray LibSci; TESTED WITH INTEL ONLY so far.
#  Compiled -fopenmp for omp_get_max_threads() in its thread-cap policy.
#
#  NO BUILD-TIME FEATURE PROBE — deliberate (owner order 2026-07-29).  A
#  link-based check_symbol_exists answers "no" for ENVIRONMENTAL reasons, and
#  its "no" silently selects the ~1.6-1.9x slower plain loop.  It did exactly
#  that on Frontera twice, for two different link-closure reasons (jobs
#  7879278 / 7879281 — a try_compile EXECUTABLE must close the whole
#  shared-library graph, so BLACS's open MPI_* references failed it, and then
#  libmpi's own fi_*@FABRIC_* references failed it again).  A build-time
#  question whose wrong answer is invisible and costly does not belong in the
#  build; the runtime lookup asks it in the process that makes the call.
#
#  What remains below is a plain EXISTS test choosing WHICH CBLAS header to
#  include.  That is not a probe in the same sense: its failure mode is "GEMM
#  handler not built at all", announced right here and refused loudly at
#  runtime — never "built, but silently slow".
# ----------------------------------------------------------------------------
#  HEADER *AND* LIBRARY (audit P1.7, 2026-07-31).  The gate used to key on
#  find_path(cblas.h) alone, but gemm_batch_ffi.cc's plain-loop fallback
#  calls cblas_dgemm/cblas_zgemm as LINK-TIME symbols (only the *_batch
#  entries are dlsym'd): on a non-MKL site with a header-only CBLAS prefix
#  the .so built with undefined symbols and died at the FIRST GEMM call
#  instead of refusing at build/configure time.  On the MKL path the link
#  line already carries libmkl_intel_lp64 (which defines cblas_?gemm); on
#  the cblas.h path a real library must be found — or an explicit
#  -DLORRAX_SCALAPACK_LIBRARIES link line may supply it (Cray LibSci defines
#  CBLAS inside libsci) — otherwise the handler is skipped, announced.  The
#  -Wl,--no-undefined at the link step below is the backstop that makes any
#  remaining gap a BUILD failure, never a first-call crash.
set(_gemm_cblas_header "")
set(_gemm_cblas_incdir "")
set(_gemm_cblas_lib    "")
if(_mkl_incdir AND EXISTS "${_mkl_incdir}/mkl_cblas.h")
    set(_gemm_cblas_header "mkl_cblas.h")
    set(_gemm_cblas_incdir "${_mkl_incdir}")
    # cblas_?gemm live in libmkl_intel_lp64, already on _host_math_link.
    set(_mklblas_on 1)
else()
    find_path(LORRAX_CBLAS_INCLUDE_DIR cblas.h
        HINTS "$ENV{CRAY_LIBSCI_PREFIX_DIR}/include"
              "$ENV{LORRAX_CBLAS_DIR}/include"
              "$ENV{LORRAX_CBLAS_DIR}")
    if(LORRAX_CBLAS_INCLUDE_DIR)
        find_library(LORRAX_CBLAS_LIBRARY
            NAMES cblas openblas blis sci_gnu sci_cray
            HINTS "$ENV{CRAY_LIBSCI_PREFIX_DIR}/lib"
                  "$ENV{LORRAX_CBLAS_DIR}/lib64"
                  "$ENV{LORRAX_CBLAS_DIR}/lib"
                  "$ENV{LORRAX_CBLAS_DIR}")
        if(LORRAX_CBLAS_LIBRARY)
            set(_gemm_cblas_header "cblas.h")
            set(_gemm_cblas_incdir "${LORRAX_CBLAS_INCLUDE_DIR}")
            set(_gemm_cblas_lib    "${LORRAX_CBLAS_LIBRARY}")
            set(_mklblas_on 1)
        elseif(LORRAX_SCALAPACK_LIBRARIES)
            # LibSci-style: the explicit ScaLAPACK link line is trusted to
            # define cblas_?gemm (libsci does); -Wl,--no-undefined verifies.
            set(_gemm_cblas_header "cblas.h")
            set(_gemm_cblas_incdir "${LORRAX_CBLAS_INCLUDE_DIR}")
            set(_mklblas_on 1)
            message(STATUS
                "mklblas: cblas.h at ${LORRAX_CBLAS_INCLUDE_DIR}; no "
                "standalone CBLAS library found — relying on the explicit "
                "LORRAX_SCALAPACK_LIBRARIES link line to define "
                "cblas_?gemm (the -Wl,--no-undefined link will fail loudly "
                "if it does not).")
        else()
            message(STATUS
                "mklblas: cblas.h found at ${LORRAX_CBLAS_INCLUDE_DIR} but "
                "NO CBLAS library (searched cblas/openblas/blis/sci_* under "
                "CRAY_LIBSCI_PREFIX_DIR / LORRAX_CBLAS_DIR) — GEMM handler "
                "skipped rather than built with undefined cblas_?gemm "
                "(LORRAX_BANDS_GEMM_FFI=1 will refuse at runtime; the auto "
                "default stays on the XLA lowering).")
        endif()
    endif()
endif()
if(NOT _mklblas_on AND NOT _gemm_cblas_header)
    message(STATUS
        "mklblas: no CBLAS header found (mkl_cblas.h under the "
        "MKL prefix, or cblas.h — set LORRAX_CBLAS_DIR / "
        "CRAY_LIBSCI_PREFIX_DIR) — GEMM handler skipped "
        "(LORRAX_BANDS_GEMM_FFI=1 will refuse at runtime; the "
        "auto default stays on the XLA lowering).")
endif()

# ----------------------------------------------------------------------------
#  RESOLVE 3/4 — MKL FFT (DFTI API) flat-k batched-FFT host handlers
#  (cpp/mklfft/fft_flat_k_ffi.cc; FFT-FFI prototype 2026-07-28).  O(N log N)
#  FFT at any k-count via MKL's DFTI descriptor API — NOT a DFT-as-matmul.
#  Rides the ScaLAPACK link line already resolved above (DFTI lives inside
#  libmkl_intel_lp64 / libmkl_core / the threading layer — zero new link
#  deps); needs only mkl_dfti.h.  The one TU is compiled -fopenmp for its
#  chunk-loop threading.
#
#  THIS IS THE ONE HOST FAMILY WITH A SINGLE-VENDOR API.  DFTI is Intel's,
#  and Cray LibSci does not implement it, so a LibSci-only site gets no host
#  FFT handler and LORRAX_FFT_FFI refuses at runtime (the default XLA lowering
#  is untouched — this is a lost optimisation, never a wrong answer).  If that
#  ever needs fixing, the portable target is the FFTW3 ADVANCED interface
#  (fftw_plan_many_dft): the flat-k layout is one uniform batch —
#  istride = howmany, idist = 1 — which that call expresses directly (the
#  guru interface's arbitrary nested loops are not needed; this comment
#  used to name fftw_plan_guru_dft — corrected 2026-07-31 to match
#  docs/architecture/ffi_layout.md §7).  fftw_plan_many_dft is implemented
#  by FFTW, cray-fftw, AOCL — and by MKL itself, whose libmkl_intel_lp64
#  exports fftw_plan_many_dft / fftw_execute alongside the Dfti* entry
#  points.  That is a rewrite of the descriptor construction, not a
#  link-line change, and it is unmeasured.
# ----------------------------------------------------------------------------
#  UPDATED 2026-08-05: the flat-k TU no longer calls DFTI.  It calls the
#  FFTW3 ADVANCED interface (fftw_plan_many_dft), resolved at RUN time by
#  dlsym — the pattern the GEMM service proved.  So there is nothing to
#  detect for MKL sites at all: MKL exports the FFTW3 C interface natively
#  from libmkl_intel_lp64, which the ScaLAPACK link line already carries, and
#  the same dlsym finds it.  ONE source serves both engines; there is no
#  build-time engine selector and no environment variable, because "which
#  FFT am I running" must be answerable from what the .so links (invariant 2,
#  ffi_layout.md §2) rather than from a knob somebody set three months ago.
#
#  CORRECTED 2026-08-06 — THIS LEG USED TO LINK FFTW3, AND THAT WAS THE BUG.
#  ------------------------------------------------------------------------
#  Until today this leg put a real FFTW3 into the library's DT_NEEDED "so the
#  runtime dlsym has something to find".  That defeats the entire point of
#  resolving at run time.  A DT_NEEDED entry is a LOAD-TIME dependency: the
#  dynamic linker must find that exact SONAME before a single byte of this
#  library runs.  So the .so became unloadable on any machine that could not
#  see `libfftw3.so.mpi31.3` — and cray-fftw's SONAME is not merely versioned,
#  it is MPI-flavour-stamped, so it is not even the same string on two Cray
#  sites.  Measured consequence (2026-08-06): the Shifter container does not
#  bind-mount /opt/cray/pe/fftw at all, so `dlopen` of this library failed at
#  DT_NEEDED resolution and NINETEEN ScaLAPACK/SLATE/GEMM contract tests —
#  none of which touch FFT — turned into skips.  A lost FFT optimisation
#  silently became a lost linear-algebra test suite.
#
#  The invariant recorded as proof that "nothing is link-bound"
#  (`nm -D --undefined-only | grep -c fftw_` -> 0) was TRUE THROUGHOUT and did
#  not catch it: it counts undefined SYMBOL references, and dlsym'ing every
#  entry point correctly drives that to zero.  It says nothing about
#  DT_NEEDED, which is what actually gates loading.  Necessary, not
#  sufficient.  The sufficient check is now GATE 5 in
#  config/perlmutter/build_ffi_host.sh: zero fftw in DT_NEEDED.
#
#  So this leg no longer contributes to the link line.  What it contributes
#  instead is a compile-time ABSOLUTE-PATH HINT that heads the runtime dlopen
#  candidate ladder in mklfft/fft_flat_k_ffi.cc.  The build still names the
#  engine it was configured against — invariant 2 is preserved, and the answer
#  to "which FFT am I running" is still baked into the artifact rather than
#  set by a knob — but naming it can no longer prevent the library loading.
#  Absence of the engine remains a STARTUP refusal from LORRAX_FFT_FFI
#  (Gate.enforce) naming the unresolved symbol, which is the loud failure this
#  was always supposed to have.
option(LORRAX_HOST_HAVE_FFTW3
       "Record a system FFTW3 as the runtime dlopen hint (NOT linked; not needed on MKL sites)"
       ON)

set(_fftw3_on 0)
if(LORRAX_HOST_HAVE_FFTW3)
    # Detection ladder, first hit wins: a real package description before a
    # hand-held path, and the raw -D override last.
    #   1. find_package(FFTW3 CONFIG)  -- upstream FFTW >= 3.3.7 ships this
    #   2. pkg-config fftw3            -- cray-fftw ships fftw3.pc
    #   3. FFTW_INC / FFTW_DIR / FFTW_ROOT hints (the cray-fftw modulefile
    #      sets all three)
    #   4. -DLORRAX_FFTW3_INCLUDE_DIR / -DLORRAX_FFTW3_LIBRARY (last resort)
    if(NOT LORRAX_FFTW3_LIBRARY)
        find_package(FFTW3 CONFIG QUIET)
    endif()
    if(TARGET FFTW3::fftw3)
        set(_fftw3_on 1)
        # We want a PATH, not a link target: the result is a dlopen hint, not
        # a link-line entry.  IMPORTED_LOCATION is where the imported target
        # keeps the actual file.
        get_target_property(_fftw3_hint FFTW3::fftw3 IMPORTED_LOCATION)
        if(NOT _fftw3_hint)
            get_target_property(_fftw3_hint FFTW3::fftw3
                                IMPORTED_LOCATION_RELEASE)
        endif()
        message(STATUS "fftw3: found via FFTW3Config.cmake -> ${_fftw3_hint}")
    else()
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(PC_FFTW3 QUIET fftw3)
        endif()
        find_path(LORRAX_FFTW3_INCLUDE_DIR fftw3.h
            HINTS ${PC_FFTW3_INCLUDE_DIRS}
                  "$ENV{FFTW_INC}" "$ENV{FFTW_ROOT}/include")
        find_library(LORRAX_FFTW3_LIBRARY NAMES fftw3
            HINTS ${PC_FFTW3_LIBRARY_DIRS}
                  "$ENV{FFTW_DIR}" "$ENV{FFTW_ROOT}/lib")
        if(LORRAX_FFTW3_LIBRARY)
            set(_fftw3_on 1)
            set(_fftw3_hint "${LORRAX_FFTW3_LIBRARY}")
            message(STATUS "fftw3: ${LORRAX_FFTW3_LIBRARY}")
        endif()
    endif()
endif()

# The flat-k FFT handlers are ALWAYS built: the TU declares the FFTW3 ABI
# itself and resolves every entry point by dlsym, so it needs no vendor
# header and no vendor library at COMPILE time.  Whether an engine is
# actually present is a RUN-time question, answered by the dlsym table and
# reported by LORRAX_FFT_FFI's announce-or-refuse.
set(_mklfft_on 1)
if(_fftw3_on)
    message(STATUS
        "mklfft: flat-k FFT handlers ON — FFTW3 advanced interface "
        "(fftw_plan_many_dft) resolved at runtime by dlsym.  System FFTW3 "
        "recorded as the runtime dlopen HINT (${_fftw3_hint}) and "
        "deliberately NOT linked — see the DT_NEEDED note above.")
else()
    message(STATUS
        "mklfft: flat-k FFT handlers ON — FFTW3 advanced interface "
        "(fftw_plan_many_dft) resolved at runtime by dlsym.  No standalone "
        "FFTW3 linked; this is CORRECT on an MKL site (libmkl_intel_lp64 "
        "exports the FFTW3 C interface natively).  On a non-MKL site load "
        "cray-fftw (or any FFTW3) and reconfigure, or LORRAX_FFT_FFI will "
        "refuse at startup naming the unresolved symbol.")
endif()

# ----------------------------------------------------------------------------
#  RESOLVE 4/4 — SLATE, the gpu_backend=none install (optional; skipped if
#  absent).  This is the only host dependency with no second implementation:
#  it is a C++ template library, so there is no ABI another vendor could
#  supply.  Its absence costs the slate potrf/trsm/batched host handlers and
#  NOTHING ELSE — the ScaLAPACK, GEMM and FFT handlers above are independent
#  of it (see the header note).
#
#  cpp/slate/context.cc and cpp/slate/ctx.h are NOT SLATE code despite the
#  directory: they are LORRAX's own MPI mesh context (a rank-remapped
#  MPI_Comm plus the host-collective mutex) in namespace lorrax_ffi::slate,
#  and they include only <mpi.h>.  The ScaLAPACK handlers use that context,
#  which is why context.cc is compiled whenever EITHER family is on.
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
    else()
        message(STATUS
            "SLATE (host): not found at ${LORRAX_SLATE_HOST_INSTALL_DIR} — "
            "slate potrf/trsm/batched host handlers skipped.  The "
            "ScaLAPACK, GEMM and FFT host handlers do NOT depend on SLATE "
            "and are unaffected.")
    endif()
endif()

# ----------------------------------------------------------------------------
#  APPEND — sources, link libraries and library search paths, in a FIXED
#  order.  Keep this order: it is what makes the emitted link line for the
#  full 11-target Frontera build byte-identical to the pre-2026-07-31 one,
#  which is how this restructure was checked without a compute node.
# ----------------------------------------------------------------------------
# LORRAX's MPI mesh context — needed by the slate AND the scalapack handlers.
if(slate_FOUND OR _scalapack_on)
    list(APPEND LORRAX_FFI_HOST_SOURCES slate/context.cc)
endif()
if(slate_FOUND)
    list(APPEND LORRAX_FFI_HOST_SOURCES slate/host_ffi.cc)
    list(APPEND _host_link_libs slate)
    list(APPEND _host_rpaths "${LORRAX_SLATE_HOST_INSTALL_DIR}/lib64" "/lorrax_slate/lib")
endif()
if(_scalapack_on)
    list(APPEND LORRAX_FFI_HOST_SOURCES
        scalapack/solve_lu_ffi.cc
        scalapack/getrf_getrs_ffi.cc
        scalapack/eigh_ffi.cc)
endif()
if(_mklfft_on)
    list(APPEND LORRAX_FFI_HOST_SOURCES mklfft/fft_flat_k_ffi.cc)
    set_source_files_properties(
        mklfft/fft_flat_k_ffi.cc
        PROPERTIES COMPILE_OPTIONS "-fopenmp")
    # The TU's own OpenMP pragmas need gomp even if the MKL threading layer
    # is switched to mkl_sequential (mkl_gnu_thread already pulls it in).
    if(NOT LORRAX_MKL_THREAD_LIB STREQUAL "mkl_gnu_thread")
        list(APPEND _host_math_link "-lgomp")
    endif()
    # NOT `list(APPEND _host_math_link ...)`.  The detected FFTW3 is handed to
    # the TU as a compile-time dlopen HINT and never reaches the link line —
    # linking it is what made the whole library unloadable wherever that exact
    # SONAME was absent (see the DT_NEEDED note at the detection block).  No
    # rpath entry either, for the same reason: an rpath pointing at a
    # bare-metal-only directory is how a container-invisible dependency gets
    # baked in without anyone noticing.
    if(_fftw3_on AND _fftw3_hint)
        set_property(SOURCE mklfft/fft_flat_k_ffi.cc APPEND PROPERTY
            COMPILE_DEFINITIONS
            "LORRAX_FFTW3_SO_HINT=\"${_fftw3_hint}\"")
    endif()
endif()
if(_mklblas_on)
    list(APPEND LORRAX_FFI_HOST_SOURCES mklblas/gemm_batch_ffi.cc)
    set(LORRAX_MKLBLAS_INCLUDE_DIR "${_gemm_cblas_incdir}")
    set(_gemm_defs "")
    if(_gemm_cblas_header STREQUAL "mkl_cblas.h")
        list(APPEND _gemm_defs "LORRAX_MKLBLAS_MKL_HEADER=1")
    endif()
    set_source_files_properties(
        mklblas/gemm_batch_ffi.cc
        PROPERTIES COMPILE_OPTIONS "-fopenmp"
                   COMPILE_DEFINITIONS "${_gemm_defs}")
    # The CBLAS library resolved above (cblas.h path only; the MKL path's
    # cblas_?gemm ride libmkl_intel_lp64, already on the math link line).
    if(_gemm_cblas_lib)
        list(APPEND _host_math_link "${_gemm_cblas_lib}")
        get_filename_component(_gemm_cblas_libdir "${_gemm_cblas_lib}" DIRECTORY)
        list(APPEND _host_math_rpaths "${_gemm_cblas_libdir}")
    endif()
    # gomp for the TU's omp_get_max_threads(); harmless duplicate when the
    # MKL branch already appended it via mkl_gnu_thread.
    list(APPEND _host_math_link "-lgomp")
    # NOTE: the substring "chosen at RUNTIME by dlsym" is grepped out of the
    # build log by the probe-removal gate (wk_REL/harness/gbp_dlsym.sbatch:68).
    # `dlsym` stays in THIS line for that reason and because a build log is
    # the one place naming the mechanism is the point; the LORRAX-facing
    # runtime announcement (cpp/mklblas/gemm_batch_ffi.cc) says it in plain
    # words instead.
    message(STATUS
        "mklblas: GEMM host handler ON (${_gemm_cblas_header} at "
        "${_gemm_cblas_incdir}); batched-vs-plain entry is chosen "
        "at RUNTIME by dlsym and announced on first use — no "
        "build-time feature probe by design")
endif()
list(APPEND _host_rpaths ${_host_math_rpaths})
list(APPEND _host_link_libs ${_host_math_link})

list(APPEND LORRAX_FFI_HOST_SOURCES common/build_config.cc)

if(NOT LORRAX_FFI_HOST_SOURCES)
    message(FATAL_ERROR
        "No host FFI components selected/found (phdf5 off and SLATE absent). "
        "Nothing to build.")
endif()

# ----------------------------------------------------------------------------
# Build liblorrax_ffi_host.so
# ----------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#  GENERATED CAPABILITY HEADER — what the build ACTUALLY resolved, recorded
#  in the library instead of only in a configure log nobody kept.
#  lorrax_ffi_host_build_config() returns it as one line, so a run (or a
#  `strings` on a .so found in a scratch directory) can report its own
#  provenance.  Note it records whether an FFTW3 was FOUND AT CONFIGURE TIME
#  and recorded as the runtime dlopen hint — not that one was linked (none is,
#  since 2026-08-06) and not which engine will actually bind: that is a
#  runtime dlopen/dlsym fact by design.
# ---------------------------------------------------------------------------
if(_fftw3_on)
    set(LORRAX_CFG_FFTW3 1)
else()
    set(LORRAX_CFG_FFTW3 0)
endif()
if(_scalapack_on)
    set(LORRAX_CFG_SCALAPACK 1)
else()
    set(LORRAX_CFG_SCALAPACK 0)
endif()
if(_mklblas_on)
    set(LORRAX_CFG_GEMM 1)
else()
    set(LORRAX_CFG_GEMM 0)
endif()
if(slate_FOUND)
    set(LORRAX_CFG_SLATE 1)
else()
    set(LORRAX_CFG_SLATE 0)
endif()
if(LORRAX_FFI_HAVE_PHDF5)
    set(LORRAX_CFG_PHDF5 1)
else()
    set(LORRAX_CFG_PHDF5 0)
endif()
string(REPLACE ";" " " LORRAX_CFG_MATH_LINK "${_host_math_link}")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/common/lorrax_config.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/lorrax_config.h" @ONLY)

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

# The XLA FFI headers are THIRD-PARTY (vendored with jaxlib) and are not
# warning-clean under -Wall -Wextra; SYSTEM keeps their noise out of the
# build log so LORRAX's own sources can be held to the no-opt-out bar above.
target_include_directories(lorrax_ffi_host SYSTEM PRIVATE
    "${LORRAX_XLA_FFI_INCLUDE_DIR}"
)
target_include_directories(lorrax_ffi_host PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
)
if(LORRAX_MKLFFT_INCLUDE_DIR)
    target_include_directories(lorrax_ffi_host PRIVATE
        "${LORRAX_MKLFFT_INCLUDE_DIR}")
endif()
if(LORRAX_MKLBLAS_INCLUDE_DIR)
    # CBLAS headers for the GEMM handler (same dir as mklfft under MKL;
    # the LibSci/OpenBLAS cblas.h dir on non-MKL builds).
    target_include_directories(lorrax_ffi_host PRIVATE
        "${LORRAX_MKLBLAS_INCLUDE_DIR}")
endif()
if(LORRAX_FFI_HAVE_PHDF5)
    target_include_directories(lorrax_ffi_host PRIVATE
        "${HDF5_INCLUDE_DIRS}"
        "${LORRAX_MPI_INCLUDE_DIR}")
endif()

find_package(Threads REQUIRED)
# ${CMAKE_DL_LIBS}: blacs_grid.h resolves MKL_Set_Num_Threads_Local via
# dlsym(RTLD_DEFAULT) so the MKL-thread pin needs no MKL link dependency
# (and is a clean no-op on a non-MKL ScaLAPACK).
target_link_libraries(lorrax_ffi_host PRIVATE ${_host_link_libs}
                      Threads::Threads ${CMAKE_DL_LIBS})

# No -Wno-unused-* here (removed 2026-08-01): the host sources compile clean
# under plain -Wall -Wextra; keep them that way rather than re-adding the
# blanket opt-out.
target_compile_options(lorrax_ffi_host PRIVATE -Wall -Wextra)

# Announce-or-refuse at LINK time (audit P1.7): every link-time symbol the
# handlers reference must resolve when the .so is built — an undefined
# cblas_?gemm (or anything else) fails HERE, loudly, instead of building a
# library that loads cleanly and dies at the first call.  Runtime-optional
# entries (cblas_?gemm_batch, MKL_Set_Num_Threads_Local, the CUDA driver
# API) are dlsym'd and deliberately NOT link references, so this flag does
# not constrain them.
target_link_options(lorrax_ffi_host PRIVATE "-Wl,--no-undefined")

message(STATUS "lorrax_ffi_host will build sources:")
foreach(_s ${LORRAX_FFI_HOST_SOURCES})
    message(STATUS "  ${_s}")
endforeach()

else()
    message(FATAL_ERROR
        "LORRAX_FFI_PLATFORM is '${LORRAX_FFI_PLATFORM}' — set it explicitly:\n"
        "  -DLORRAX_FFI_PLATFORM=cuda  -> liblorrax_ffi.so (CUDA leg)\n"
        "  -DLORRAX_FFI_PLATFORM=host  -> liblorrax_ffi_host.so (CUDA-free host leg)\n"
        "There is no default on purpose: the leg used to be selected by which\n"
        "directory cmake was pointed at, and a silent default would hide a\n"
        "wrong-platform build until runtime.")
endif()
