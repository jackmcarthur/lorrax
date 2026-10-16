# LorraxXlaFfi.cmake — locate the XLA FFI headers (xla/ffi/api/ffi.h).
#
# Shared by the host and ROCM legs (the CUDA leg still carries its own
# container-aware copy; migrate it when it can be rebuilt and verified).
# A MACRO, not a function: it runs in the caller's scope exactly as the
# inline block it replaced did, so it sets LORRAX_XLA_FFI_INCLUDE_DIR there.
#
# The headers must match the RUNTIME jaxlib.  Prefer an explicit
# LORRAX_XLA_FFI_INCLUDE_DIR; otherwise probe jax.ffi.include_dir() with the
# Python3 this cmake runs under (correct inside the runtime venv/container).
macro(lorrax_find_xla_ffi)
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
endmacro()
