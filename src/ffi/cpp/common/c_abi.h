// c_abi.h — THE naming rule for the ctypes lifecycle entry points.
//
// WHY THIS FILE EXISTS.  `cpp/phdf5/api.cc` and `cpp/slate/context.cc` are
// CUDA-free and therefore compile into BOTH platform libraries.  Before
// 2026-08-08 they exported the SAME NINE `extern "C"` names out of both:
//
//     lrx_phdf5_open  lrx_phdf5_close  lrx_phdf5_init_mpi
//     lrx_phdf5_ensure_dataset  lrx_phdf5_open_dataset_ro
//     lrx_slate_context_create  lrx_slate_subrow_context_create
//     lrx_slate_context_destroy lrx_slate_init_mpi
//
// Both libraries are dlopened RTLD_GLOBAL, which publishes every one of those
// names into ONE process-global namespace with two definitions behind it.
// That is a one-definition-rule violation across shared objects, and it is
// half of the defect KNOWN_FAILURES registered as L1 (the other half is the
// mangled `lorrax_ffi::phdf5::*` core, closed by the version scripts
// `exports_{cuda,host}.map` and by PhdfCtx's split ABI identity in
// `phdf5/ctx.h`).
//
// THE RULE.  A symbol that leaves an FFI library carries the LEG in its name.
// The handler surface already did — `PhdfReadFfi` on CUDA,
// `PhdfReadHostFfi` on host (`phdf5/platform_seam.h`'s LRX_PHDF_HANDLER, and
// `slate/host_ffi.cc`'s `*HostFfi`).  This macro extends the same rule to the
// ctypes surface, which is the only other thing either library exports:
//
//     CUDA leg   lrx_phdf5_open          (unchanged — it was never ambiguous)
//     host leg   lrx_phdf5_open_host
//
// so `nm -D --defined-only` on the two files intersects in NOTHING.  That is
// checkable by set intersection instead of by an argument about dlsym scoping
// rules, and `services/distrib_la/tests/test_so_acceptance.py`'s check 6
// checks it on every run, with a red twin.
//
// SOURCE STAYS GREPPABLE.  Write the definition as
//
//     void LRX_C_ENTRY(lrx_phdf5_init_mpi)(void) { ... }
//
// so `grep lrx_phdf5_init_mpi` still finds the definition; only the emitted
// symbol changes.  The Python side keeps calling `lib.lrx_phdf5_init_mpi` —
// `ffi_loader._bind_c_abi` binds the leg's real symbol under that name once,
// at load, rather than making every call site carry a suffix.

#pragma once

// LORRAX_FFI_NO_CUDA is defined by the host-platform build only (see
// cpp/CMakeLists.txt, `target_compile_definitions(lorrax_ffi_host ...)`), so
// it is the same seam the rest of the shared TUs already switch on.  Keep it
// that way: a SECOND way to ask "which leg is this" is a second thing to get
// out of sync.
#ifdef LORRAX_FFI_NO_CUDA
#  define LRX_C_ENTRY(name) name##_host
#else
#  define LRX_C_ENTRY(name) name
#endif
