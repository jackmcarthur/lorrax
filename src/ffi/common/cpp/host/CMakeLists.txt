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
# compiles the SAME phdf5 collective MPI-IO core (context.cc + read_ffi.cc +
# write_ffi.cc) as the CUDA lib, with -DLORRAX_FFI_NO_CUDA so the
# device-staging seams collapse: the read tail becomes a std::memcpy into the
# host-resident XLA output buffer instead of a cudaMemcpyAsync H2D, and the
# write path hands H5Dwrite the XLA input buffer directly instead of staging
# it D2H through a pinned buffer (see phdf5/cpp/platform_seam.h).
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
        ../../../phdf5/cpp/api.cc
        ../../../phdf5/cpp/context.cc
        ../../../phdf5/cpp/read_ffi.cc
        ../../../phdf5/cpp/write_ffi.cc
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
#      scalapack/cpp/{eigh,solve_lu}_ffi.cc and mkl{fft,blas}/cpp/*.cc with
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
#  RESOLVE 1/4 — ScaLAPACK + C-BLACS.  Required by scalapack/cpp/*_ffi.cc
#  (pXgetrf / pXgetrs / pXheevd / Cblacs_gridinit / descinit / numroc;
#  prototypes hand-declared in scalapack/cpp/blacs_grid.h because ScaLAPACK
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
#      slate/scripts/build_scalapack_api.sh), that library DEFINES only 6 of
#      the eleven names below — every compute routine, none of the
#      grid/descriptor tools, and it CALLS numroc_ / Cblacs_gridinfo itself.
#      It is an OVERLAY on a real ScaLAPACK, never a provider.  LORRAX
#      REFUSES it at run time (scalapack/cpp/blacs_grid.h) because its
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
#  (mklblas/cpp/gemm_batch_ffi.cc; contract_bands FFI GEMM body, 2026-07-28 —
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
#  (mklfft/cpp/fft_flat_k_ffi.cc; FFT-FFI prototype 2026-07-28).  O(N log N)
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
#  ever needs fixing, the portable target is the FFTW3 guru interface
#  (fftw_plan_guru_dft), which takes the same arbitrary input/output strides
#  this handler depends on and is implemented by FFTW, cray-fftw, AOCL — and
#  by MKL itself, whose libmkl_intel_lp64 exports fftw_plan_many_dft /
#  fftw_execute alongside the Dfti* entry points.  That is a rewrite of the
#  descriptor construction, not a link-line change, and it is unmeasured.
# ----------------------------------------------------------------------------
if(_mkl_incdir AND EXISTS "${_mkl_incdir}/mkl_dfti.h")
    set(_mklfft_on 1)
    set(LORRAX_MKLFFT_INCLUDE_DIR "${_mkl_incdir}")
elseif(_mkl_incdir)
    message(STATUS
        "mklfft: ${_mkl_incdir}/mkl_dfti.h not found — "
        "flat-k FFT handlers skipped (LORRAX_FFT_FFI will "
        "refuse at runtime).")
else()
    # Reached on the explicit -DLORRAX_SCALAPACK_LIBRARIES route and on any
    # non-MKL site.  Before 2026-07-31 this case produced NO configure output
    # at all, because the whole block lived inside the MKL branch above.
    message(STATUS
        "mklfft: no MKL header directory resolved (the DFTI API is "
        "Intel-only, and ScaLAPACK did not come from an MKL prefix) — "
        "flat-k FFT handlers skipped.  Set -DLORRAX_MKL_ROOT as well as "
        "-DLORRAX_SCALAPACK_LIBRARIES to build them against MKL's FFT "
        "while linking someone else's ScaLAPACK; LORRAX_FFT_FFI otherwise "
        "refuses at runtime and the default XLA lowering stands.")
endif()

# ----------------------------------------------------------------------------
#  RESOLVE 4/4 — SLATE, the gpu_backend=none install (optional; skipped if
#  absent).  This is the only host dependency with no second implementation:
#  it is a C++ template library, so there is no ABI another vendor could
#  supply.  Its absence costs the slate potrf/trsm/batched host handlers and
#  NOTHING ELSE — the ScaLAPACK, GEMM and FFT handlers above are independent
#  of it (see the header note).
#
#  slate/cpp/context.cc and slate/cpp/ctx.h are NOT SLATE code despite the
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
    list(APPEND LORRAX_FFI_HOST_SOURCES ../../../slate/cpp/context.cc)
endif()
if(slate_FOUND)
    list(APPEND LORRAX_FFI_HOST_SOURCES ../../../slate/cpp/host_ffi.cc)
    list(APPEND _host_link_libs slate)
    list(APPEND _host_rpaths "${LORRAX_SLATE_HOST_INSTALL_DIR}/lib64" "/lorrax_slate/lib")
endif()
if(_scalapack_on)
    list(APPEND LORRAX_FFI_HOST_SOURCES
        ../../../scalapack/cpp/solve_lu_ffi.cc
        ../../../scalapack/cpp/eigh_ffi.cc)
endif()
if(_mklfft_on)
    list(APPEND LORRAX_FFI_HOST_SOURCES ../../../mklfft/cpp/fft_flat_k_ffi.cc)
    set_source_files_properties(
        ../../../mklfft/cpp/fft_flat_k_ffi.cc
        PROPERTIES COMPILE_OPTIONS "-fopenmp")
    # The TU's own OpenMP pragmas need gomp even if the MKL threading layer
    # is switched to mkl_sequential (mkl_gnu_thread already pulls it in).
    if(NOT LORRAX_MKL_THREAD_LIB STREQUAL "mkl_gnu_thread")
        list(APPEND _host_math_link "-lgomp")
    endif()
    message(STATUS
        "mklfft: MKL FFT (DFTI API) flat-k handlers ON "
        "(headers at ${LORRAX_MKLFFT_INCLUDE_DIR})")
endif()
if(_mklblas_on)
    list(APPEND LORRAX_FFI_HOST_SOURCES ../../../mklblas/cpp/gemm_batch_ffi.cc)
    set(LORRAX_MKLBLAS_INCLUDE_DIR "${_gemm_cblas_incdir}")
    set(_gemm_defs "")
    if(_gemm_cblas_header STREQUAL "mkl_cblas.h")
        list(APPEND _gemm_defs "LORRAX_MKLBLAS_MKL_HEADER=1")
    endif()
    set_source_files_properties(
        ../../../mklblas/cpp/gemm_batch_ffi.cc
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
    # runtime announcement (mklblas/cpp/gemm_batch_ffi.cc) says it in plain
    # words instead.
    message(STATUS
        "mklblas: GEMM host handler ON (${_gemm_cblas_header} at "
        "${_gemm_cblas_incdir}); batched-vs-plain entry is chosen "
        "at RUNTIME by dlsym and announced on first use — no "
        "build-time feature probe by design")
endif()
list(APPEND _host_rpaths ${_host_math_rpaths})
list(APPEND _host_link_libs ${_host_math_link})

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

target_compile_options(lorrax_ffi_host PRIVATE -Wall -Wextra
    -Wno-unused-parameter -Wno-unused-variable)

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
