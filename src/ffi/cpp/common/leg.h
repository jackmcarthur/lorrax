// leg.h — THE one answer to "which leg is this translation unit built for".
//
// One C++ tree builds three libraries (docs/architecture/ffi_layout.md,
// "Platform legs").  CMakeLists.txt defines at most one selector per leg:
//
//   host  liblorrax_ffi_host.so   -DLORRAX_FFI_NO_CUDA=1   (no device runtime)
//   CUDA  liblorrax_ffi.so        (neither defined)        (CUDA runtime)
//   ROCM  liblorrax_ffi_rocm.so   -DLORRAX_FFI_ROCM=1      (HIP runtime)
//
// Every per-leg spelling in the shared sources is derived HERE and nowhere
// else: the C entry-point suffix (c_abi.h), the build-config and ABI stamp
// names (build_config.cc), the phdf5 context struct tag (phdf5/ctx.h).
// Source that only needs "is there a device runtime" tests LRX_LEG_DEVICE;
// the device runtime calls themselves go through gpu_runtime.h, which is the
// only file that names a vendor's runtime API.
//
// The CUDA and host spellings are the historical ones, unchanged, so a
// rebuilt CUDA or host library exports exactly the symbols it did.
#pragma once

#if defined(LORRAX_FFI_NO_CUDA) && defined(LORRAX_FFI_ROCM)
#  error "leg.h: LORRAX_FFI_NO_CUDA (host leg) and LORRAX_FFI_ROCM (ROCM leg) are both defined; CMake selects exactly one leg"
#endif

#if defined(LORRAX_FFI_NO_CUDA)
#  define LRX_LEG_HOST   1
#  define LRX_LEG_DEVICE 0
#  define LRX_LEG_NAME   "host"
#elif defined(LORRAX_FFI_ROCM)
#  define LRX_LEG_ROCM   1
#  define LRX_LEG_DEVICE 1
#  define LRX_LEG_NAME   "rocm"
#else
#  define LRX_LEG_CUDA   1
#  define LRX_LEG_DEVICE 1
#  define LRX_LEG_NAME   "cuda"
#endif

// Token-paste the leg's suffix onto a C entry point (c_abi.h's LRX_C_ENTRY)
// and onto per-leg type tags.  Two levels so the argument expands first.
#if defined(LRX_LEG_HOST)
#  define LRX_LEG_PASTE(name) name##_host
#  define LRX_LEG_TAG(prefix, version) prefix##Host##version
#elif defined(LRX_LEG_ROCM)
#  define LRX_LEG_PASTE(name) name##_rocm
#  define LRX_LEG_TAG(prefix, version) prefix##Rocm##version
#else
#  define LRX_LEG_PASTE(name) name
#  define LRX_LEG_TAG(prefix, version) prefix##Cuda##version
#endif
