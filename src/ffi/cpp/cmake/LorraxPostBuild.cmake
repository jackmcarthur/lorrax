# ============================================================================
# After the link: write <leg>.closure.json, then run scripts/verify_ffi_build.sh.
#
# The loader refuses a leg that has no dependency-closure record
# (lxkit.native_provider), so the build writes one: a plain `cmake --build`
# produces a leg that loads.  verify_ffi_build.sh stays the one acceptance
# contract (its header explains why it is a script); this only calls it.
# Both are outputs of their own (<leg>.closure.json, <leg>.verified), not
# POST_BUILD commands, so a failing gate fails the build but leaves the .so on
# disk to inspect, and the next build re-runs exactly the step that failed.
#
#   -DLORRAX_FFI_POST_BUILD=OFF        the build script writes the record and
#                                      runs the verifier itself (every lane
#                                      that predates this file passes it)
#   -DLORRAX_CLOSURE_LIBRARY_DIRS=A:B  write_checkout_closure.py --library-dirs.
#                                      Default: the directory of every library
#                                      this leg links (MPI, HDF5, ScaLAPACK,
#                                      CBLAS, CUDA math), so the record pins
#                                      what the link used.
#   -DLORRAX_CLOSURE_ARGS=a;b          further write_checkout_closure.py
#                                      arguments (;-list)
#   -DLORRAX_VERIFY_ENV=N=v;...        the verifier's EXPECT_* environment
#                                      (;-list); unset expectations take the
#                                      verifier's defaults, the full backend set
#   -DLORRAX_VERIFY_LIBRARY_PATH=A:B   prepended to the build shell's
#                                      LD_LIBRARY_PATH for the verifier
# ============================================================================
option(LORRAX_FFI_POST_BUILD
    "Write <leg>.closure.json and run scripts/verify_ffi_build.sh after linking" ON)
set(LORRAX_CLOSURE_LIBRARY_DIRS "" CACHE STRING
    "Colon-separated pinned library directories for the closure record (default: the linked libraries' directories)")
set(LORRAX_CLOSURE_ARGS "" CACHE STRING
    "Further write_checkout_closure.py arguments (;-list)")
set(LORRAX_VERIFY_ENV "" CACHE STRING
    "NAME=value;... environment for scripts/verify_ffi_build.sh")
set(LORRAX_VERIFY_LIBRARY_PATH "" CACHE STRING
    "Colon-separated directories prepended to LD_LIBRARY_PATH for scripts/verify_ffi_build.sh")

get_filename_component(LORRAX_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

# lorrax_leg_post_build(<target> <host|cuda|rocm>
#                       [PRIVATE_DIRS d...] [PROCESS_DIRS d...] [LINKED item...])
# LINKED: link items (absolute library paths, -L<dir> flags, other flags);
# their directories are the default pinned set.
function(lorrax_leg_post_build target leg)
    if(NOT LORRAX_FFI_POST_BUILD)
        return()
    endif()
    cmake_parse_arguments(PB "" "" "PRIVATE_DIRS;PROCESS_DIRS;LINKED" ${ARGN})

    if(Python3_EXECUTABLE)
        set(_py "${Python3_EXECUTABLE}")
    elseif(EXISTS "${LORRAX_REPO_ROOT}/.venv/bin/python")
        set(_py "${LORRAX_REPO_ROOT}/.venv/bin/python")
    else()
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        set(_py "${Python3_EXECUTABLE}")
    endif()

    set(_dirs "${LORRAX_CLOSURE_LIBRARY_DIRS}")
    if(NOT _dirs)
        set(_list "")
        foreach(_item IN LISTS PB_LINKED)
            separate_arguments(_words UNIX_COMMAND "${_item}")
            foreach(_w IN LISTS _words)
                if(_w MATCHES "^-L(.+)$")
                    list(APPEND _list "${CMAKE_MATCH_1}")
                elseif(IS_ABSOLUTE "${_w}" AND EXISTS "${_w}" AND NOT IS_DIRECTORY "${_w}")
                    get_filename_component(_d "${_w}" DIRECTORY)
                    list(APPEND _list "${_d}")
                endif()
            endforeach()
        endforeach()
        list(REMOVE_DUPLICATES _list)
        string(REPLACE ";" ":" _dirs "${_list}")
    endif()

    set(_args --python "${_py}" --library-dirs "${_dirs}")
    foreach(_d IN LISTS PB_PRIVATE_DIRS)
        list(APPEND _args --private-dir "${_d}")
    endforeach()
    foreach(_d IN LISTS PB_PROCESS_DIRS)
        list(APPEND _args --process-dir "${_d}")
    endforeach()
    list(APPEND _args ${LORRAX_CLOSURE_ARGS})
    message(STATUS "post-build: closure pinned dirs = ${_dirs}")

    # The linked file (OUTPUT cannot name it by $<TARGET_FILE>).
    get_target_property(_name ${target} OUTPUT_NAME)
    set(_so "${CMAKE_CURRENT_BINARY_DIR}/lib${_name}.so")
    set(_verify bash "${LORRAX_REPO_ROOT}/scripts/verify_ffi_build.sh" --leg ${leg} "${_so}")
    if(LORRAX_VERIFY_LIBRARY_PATH)
        # Prepended at build time, so the shell's own entries (libfabric's
        # module, ...) stay behind it, as they will at run time.
        set(_verify sh -c "LD_LIBRARY_PATH=\"$0\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\" exec \"$@\""
                    "${LORRAX_VERIFY_LIBRARY_PATH}" ${_verify})
    endif()
    add_custom_command(OUTPUT "${_so}.closure.json"
        COMMAND "${_py}" "${CMAKE_CURRENT_SOURCE_DIR}/stage/write_checkout_closure.py"
                "${_so}" ${_args}
        DEPENDS ${target}
        COMMENT "dependency closure: ${_so}.closure.json"
        VERBATIM)
    add_custom_command(OUTPUT "${_so}.verified"
        COMMAND "${CMAKE_COMMAND}" -E env "LORRAX_ROOT=${LORRAX_REPO_ROOT}"
                "LORRAX_GATE_FFTW_PY=${_py}" ${LORRAX_VERIFY_ENV} ${_verify}
        COMMAND "${CMAKE_COMMAND}" -E touch "${_so}.verified"
        DEPENDS ${target} "${_so}.closure.json"
        COMMENT "scripts/verify_ffi_build.sh --leg ${leg}"
        VERBATIM)
    add_custom_target(${target}_accept ALL DEPENDS "${_so}.closure.json" "${_so}.verified")
endfunction()
