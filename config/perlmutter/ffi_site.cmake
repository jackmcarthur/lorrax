# Perlmutter's library paths for src/ffi/cpp, read by the perlmutter-* presets
# in src/ffi/cpp/CMakePresets.json (CMakeLists.txt includes it before
# configuring).  The MPI, LibSci and HDF5 pins themselves stay in ffi_mpi.sh,
# which the module and the run environment also source; this file only hands
# them to CMake.  Plain gcc/g++ (gcc-native) build both legs: no Cray compiler
# wrapper, so no loaded module decides what is linked.

execute_process(
    COMMAND bash -c "source '${CMAKE_CURRENT_LIST_DIR}/ffi_mpi.sh' && printf '%s;%s;%s;%s;%s;%s' \"$LORRAX_PM_MPICH_ROOT\" \"$LORRAX_PM_MPI_LIBRARY\" \"$LORRAX_PM_LIBSCI_ROOT\" \"$LORRAX_PM_HDF5_ROOT\" \"$LORRAX_PM_RUNTIME_LIB_DIRS\" \"$LORRAX_PM_MPI_SONAME\""
    OUTPUT_VARIABLE _pm RESULT_VARIABLE _pm_rc)
if(NOT _pm_rc EQUAL 0)
    message(FATAL_ERROR "ffi_site.cmake: could not read ${CMAKE_CURRENT_LIST_DIR}/ffi_mpi.sh")
endif()
list(GET _pm 0 _mpich)
list(GET _pm 1 _mpi_lib)
list(GET _pm 2 _libsci)
list(GET _pm 3 _hdf5)
list(GET _pm 4 _runtime_dirs)
list(GET _pm 5 _mpi_soname)

set(_gcc /opt/cray/pe/gcc-native/14/bin)
set(CMAKE_C_COMPILER   "${_gcc}/gcc" CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER "${_gcc}/g++" CACHE FILEPATH "")
# ccache (/usr/bin on the login nodes): the build scripts reconfigure --fresh
# every time, and a repeat build of an unchanged tree then takes 0.3 s / 0.2 s
# (CUDA / host leg) instead of 17.6 s / 6.8 s.  The output is the compiler's
# own, cached; -DCMAKE_CXX_COMPILER_LAUNCHER= (empty) turns it off.
find_program(_ccache ccache)
if(_ccache)
    foreach(_lang C CXX CUDA)
        set(CMAKE_${_lang}_COMPILER_LAUNCHER "${_ccache}" CACHE STRING "")
    endforeach()
endif()
set(HDF5_ROOT "${_hdf5}" CACHE PATH "")
set(LORRAX_MPI_INCLUDE_DIR "${_mpich}/include" CACHE PATH "")
set(LORRAX_MPICH_LIB_DIR   "${_mpich}/lib" CACHE PATH "")
set(LORRAX_MPI_LIBRARY     "${_mpi_lib}" CACHE FILEPATH "")
# The pinned run-time directories (ffi_mpi.sh): the closure record's search path.
set(LORRAX_CLOSURE_LIBRARY_DIRS "${_runtime_dirs}" CACHE STRING "")
# Both legs: one cray-mpich, and the HDF5 SOVERSION the run maps (1.14.3.7).
set(_verify "LORRAX_FFI_EXPECT_MPI=${_mpi_soname};LORRAX_FFI_EXPECT_HDF5_SOVERSION=310;LORRAX_PHDF5_STAGE=${_hdf5}")

if(LORRAX_FFI_PLATFORM STREQUAL "host")
    # What the Cray CC wrapper added for craype-x86-milan, kept so the host code
    # is the same machine code.  Its other additions were an unconditional
    # -lxpmem and, with cray-libsci loaded, the sequential -lsci_gnu beside the
    # threaded pair below; neither is wanted, which is why the old script had
    # to unload modules.
    set(CMAKE_CXX_FLAGS "-march=znver3" CACHE STRING "")
    # LibSci's threaded (_mp) pair, ONE flavour per process.  The CBLAS GEMM
    # handler takes the same library.
    set(LORRAX_SCALAPACK_LIBRARIES
        "-L${_libsci}/lib -Wl,--no-as-needed -lsci_gnu_mpi_mp -lsci_gnu_mp -lgomp -lpthread -lm -ldl"
        CACHE STRING "")
    set(LORRAX_CBLAS_INCLUDE_DIR "${_libsci}/include" CACHE PATH "")
    set(LORRAX_CBLAS_LIBRARY "${_libsci}/lib/libsci_gnu_mp.so" CACHE FILEPATH "")
    # cray-fftw's engine: a RUN-time dlopen hint, never linked (GATE 5).
    set(LORRAX_FFTW3_LIBRARY /opt/cray/pe/fftw/3.3.10.11/x86_milan/lib/libfftw3.so CACHE FILEPATH "")
    # -DLORRAX_HOST_HAVE_SCALAPACK=OFF: a GPU-only install.  A CUDA run never
    # calls ScaLAPACK (distrib_la.resolve offers it on cpu meshes only) and
    # does not open the host leg at all; CPU-mesh runs would lose
    # eigh_backend/solve_lu = scalapack.
    set(_backends scalapack,gemm,phdf5,fft)
    if(DEFINED LORRAX_HOST_HAVE_SCALAPACK AND NOT LORRAX_HOST_HAVE_SCALAPACK)
        set(_backends gemm,phdf5,fft)
    endif()
    set(LORRAX_VERIFY_ENV
        "${_verify};LORRAX_FFI_EXPECT_BACKENDS=${_backends};GATE_TAG=build_ffi_host"
        CACHE STRING "")
    set(LORRAX_VERIFY_LIBRARY_PATH "${_runtime_dirs}" CACHE STRING "")
elseif(LORRAX_FFI_PLATFORM STREQUAL "cuda")
    # The CUDA toolchain and libraries are the venv's cuda13 wheels (uv.lock):
    # nvcc, and the same files the JAX plugin maps at run time.  No cudatoolkit
    # module, so the HPC SDK's cuSOLVERMp 0.8 never comes near the build.
    get_filename_component(_venv_py "${CMAKE_CURRENT_LIST_DIR}/../../.venv/bin/python" ABSOLUTE)
    execute_process(COMMAND "${_venv_py}" -c "import sysconfig; print(sysconfig.get_paths()['purelib'], end='')"
                    OUTPUT_VARIABLE _site RESULT_VARIABLE _site_rc)
    if(NOT _site_rc EQUAL 0)
        message(FATAL_ERROR "ffi_site.cmake: no venv python at ${_venv_py}; run 'uv sync --extra cuda13'")
    endif()
    set(_cu13 "${_site}/nvidia/cu13")
    # Perlmutter is A100-only: sm_80 SASS (what runs there) plus compute_80
    # PTX, not the generic six-target default.  Measured, the build is no
    # faster for it (one .cu file; the critical path is the 16.5 s g++ TU
    # kconv_mathdx_cuda_ffi.cc), but the fatbin carries only what runs here.
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "")
    set(CMAKE_CUDA_COMPILER "${_cu13}/bin/nvcc" CACHE FILEPATH "")
    set(CMAKE_CUDA_HOST_COMPILER "${_gcc}/g++" CACHE FILEPATH "")
    set(CUDA_TOOLKIT_ROOT "${_cu13}" CACHE PATH "")
    # The wheels ship versioned names only; name each file so find_library
    # needs no bare-.so links inside the venv.
    set(CUDART_LIBRARY   "${_cu13}/lib/libcudart.so.13" CACHE FILEPATH "")
    set(CUBLAS_LIBRARY   "${_cu13}/lib/libcublas.so.13" CACHE FILEPATH "")
    set(CUSOLVER_LIBRARY "${_cu13}/lib/libcusolver.so.12" CACHE FILEPATH "")
    set(CUFFT_LIBRARY    "${_cu13}/lib/libcufft.so.12" CACHE FILEPATH "")
    set(NVRTC_LIBRARY    "${_cu13}/lib/libnvrtc.so.13" CACHE FILEPATH "")
    set(CUFFT_INCLUDE_DIR "${_cu13}/include" CACHE PATH "")
    set(NVRTC_INCLUDE_DIR "${_cu13}/include" CACHE PATH "")
    set(NCCL_INCLUDE "${_site}/nvidia/nccl/include" CACHE PATH "")
    set(NCCL_LIBRARY "${_site}/nvidia/nccl/lib/libnccl.so.2" CACHE FILEPATH "")
    # cuSOLVERMp/cuBLASMp: the venv's wheels (the cuda13 extra), linked into a
    # directory of their own, because the closure record classes them as the
    # engine-private set by directory and the CMakeLists links bare .so names.
    # Not under a directory named build: seal_bundle.py refuses those.
    get_filename_component(_mp "${CMAKE_CURRENT_LIST_DIR}/../../.build/cusolvermp" ABSOLUTE)
    file(MAKE_DIRECTORY "${_mp}/include" "${_mp}/lib")
    foreach(_pair
            "${_cu13}/include/cusolverMp.h|include/cusolverMp.h"
            "${_site}/nvidia/cublasmp/cu13/include/cublasmp.h|include/cublasmp.h"
            "${_cu13}/lib/libcusolverMp.so.0|lib/libcusolverMp.so.0"
            "${_cu13}/lib/libcusolverMp.so.0|lib/libcusolverMp.so"
            "${_site}/nvidia/cublasmp/cu13/lib/libcublasmp.so.0|lib/libcublasmp.so.0"
            "${_site}/nvidia/cublasmp/cu13/lib/libcublasmp.so.0|lib/libcublasmp.so")
        string(REPLACE "|" ";" _pair "${_pair}")
        list(GET _pair 0 _from)
        list(GET _pair 1 _to)
        if(NOT EXISTS "${_from}")
            message(FATAL_ERROR "ffi_site.cmake: ${_from} is missing; run 'uv sync --extra cuda13' (the extra installs cuSOLVERMp/cuBLASMp)")
        endif()
        file(CREATE_LINK "${_from}" "${_mp}/${_to}" SYMBOLIC)
    endforeach()
    set(CUSOLVERMP_INCLUDE_DIR "${_mp}/include" CACHE PATH "")
    set(CUSOLVERMP_LIB_DIR     "${_mp}/lib" CACHE PATH "")
    set(LORRAX_VERIFY_ENV
        "${_verify};LORRAX_FFI_EXPECT_BACKENDS=cusolvermp,cufft,phdf5;LORRAX_FFI_VERIFY_ENV=runtime;GATE_TAG=build_ffi_cuda"
        CACHE STRING "")
    set(LORRAX_VERIFY_LIBRARY_PATH "${_mp}/lib:${_site}/nvidia/nccl/lib:${_cu13}/lib:${_runtime_dirs}" CACHE STRING "")
endif()
