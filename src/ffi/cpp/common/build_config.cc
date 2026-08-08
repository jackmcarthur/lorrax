// build_config.cc — one exported C entry point per leg reporting what this
// FFI library was actually built against, and which handler-signature ABI it
// speaks.
//
// Consumed by ffi_loader (dlsym'd, optional — an older .so simply does not
// have it, and the caller can say so rather than guess) and surfaced in the
// runtime startup report.  Deliberately plain C linkage and a plain string:
// the point is that it is trivially readable from Python, from `nm`, and
// from `strings` on a .so someone found in a scratch directory with no
// memory of how it was configured.
//
// Everything here is a compile-time constant, so the string is baked into
// .rodata and costs nothing at run time.
//
// ---------------------------------------------------------------------------
// TWO LEGS, TWO SYMBOL NAMES — deliberately, and this is load-bearing
// ---------------------------------------------------------------------------
// liblorrax_ffi.so and liblorrax_ffi_host.so are BOTH dlopened RTLD_GLOBAL in
// a GPU process, and sixteen symbol names are already defined by both (the
// lrx_phdf5_* / lrx_slate_* entry points and seven mangled
// lorrax_ffi::phdf5:: helpers).  Once both are open the first one loaded
// answers those names for both — the cross-.so ODR hazard registered as L1 in
// tests/KNOWN_FAILURES.md.  A build stamp that collided the same way would
// report the OTHER library's configuration and its ABI, which is precisely the
// question it exists to answer; a mispaired process would then certify itself.
//
// So the names are per-leg, selected by the same LORRAX_FFI_NO_CUDA macro the
// host target already defines.  The fix for L1 is not attempted here, and in
// particular NOT with a blanket `local:*` version script: the ODR branch
// measured that hiding everything by default SEGFAULTS SLATE, whose templates
// need their symbols visible across the blaspp/lapackpp boundary.

#include "lorrax_config.h"
#include "lorrax_ffi_abi.h"

#define LORRAX_CFG_STR2(x) #x
#define LORRAX_CFG_STR(x) LORRAX_CFG_STR2(x)

namespace {
// The leg name and the ABI version lead, because they are the two facts a
// human reading `strings` needs before any of the rest means anything: which
// library is this, and will my Python talk to it.
const char kBuildConfig[] =
#ifdef LORRAX_FFI_NO_CUDA
    "leg=host"
#else
    "leg=cuda"
#endif
    " abi=" LORRAX_CFG_STR(LORRAX_FFI_ABI_VERSION)
#ifdef LORRAX_FFI_NO_CUDA
    // The host leg's original five keys, in their original order and spelling.
    // scripts/verify_ffi_build.sh and services/distrib_la/tests/
    // test_so_acceptance.py both search for `scalapack=[01]` as literal bytes;
    // keep the spelling.
    " linked_fftw3=" LORRAX_CFG_STR(LORRAX_CFG_LINKED_FFTW3)
    " scalapack="    LORRAX_CFG_STR(LORRAX_CFG_HAVE_SCALAPACK)
    " gemm="         LORRAX_CFG_STR(LORRAX_CFG_HAVE_GEMM)
    " slate="        LORRAX_CFG_STR(LORRAX_CFG_HAVE_SLATE)
    " phdf5="        LORRAX_CFG_STR(LORRAX_CFG_HAVE_PHDF5)
    " math_link="    LORRAX_CFG_MATH_LINK;
#else
    // The device leg's capabilities.  Until 2026-08-08 this leg carried NO
    // stamp at all — `strings liblorrax_ffi.so | grep scalapack=` on the
    // deployed device library returns nothing — so the one artifact-level
    // question anybody could ask of it was "which symbols are exported".
    " cusolvermp=" LORRAX_CFG_STR(LORRAX_CFG_HAVE_CUSOLVERMP)
    " cublasmp="   LORRAX_CFG_STR(LORRAX_CFG_HAVE_CUBLASMP)
    " cufft="      LORRAX_CFG_STR(LORRAX_CFG_HAVE_CUFFT)
    " slate="      LORRAX_CFG_STR(LORRAX_CFG_HAVE_SLATE)
    " phdf5="      LORRAX_CFG_STR(LORRAX_CFG_HAVE_PHDF5)
    " cal="        LORRAX_CFG_STR(LORRAX_CFG_HAVE_CAL);
#endif
}  // namespace

extern "C" {

#ifdef LORRAX_FFI_NO_CUDA

const char* lorrax_ffi_host_build_config() { return kBuildConfig; }

// The ABI as an INTEGER as well as a substring of the line above.  Both, on
// purpose: the string is what `strings` on an unknown file gives you, and the
// integer is what a loader reads without parsing prose.  A parser that has to
// find "abi=" inside a line whose other fields include a full link command is
// a parser that will one day match the wrong thing.
int lorrax_ffi_host_abi_version() { return LORRAX_FFI_ABI_VERSION; }

#else

const char* lorrax_ffi_cuda_build_config() { return kBuildConfig; }
int lorrax_ffi_cuda_abi_version() { return LORRAX_FFI_ABI_VERSION; }

#endif

}  // extern "C"
