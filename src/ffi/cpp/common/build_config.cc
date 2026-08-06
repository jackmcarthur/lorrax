// build_config.cc — one exported C entry point reporting what this
// liblorrax_ffi_host.so was actually built against.
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

#include "lorrax_config.h"

#define LORRAX_CFG_STR2(x) #x
#define LORRAX_CFG_STR(x) LORRAX_CFG_STR2(x)

namespace {
const char kBuildConfig[] =
    "linked_fftw3=" LORRAX_CFG_STR(LORRAX_CFG_LINKED_FFTW3)
    " scalapack=" LORRAX_CFG_STR(LORRAX_CFG_HAVE_SCALAPACK)
    " gemm="      LORRAX_CFG_STR(LORRAX_CFG_HAVE_GEMM)
    " slate="     LORRAX_CFG_STR(LORRAX_CFG_HAVE_SLATE)
    " phdf5="     LORRAX_CFG_STR(LORRAX_CFG_HAVE_PHDF5)
    " math_link=" LORRAX_CFG_MATH_LINK;
}  // namespace

extern "C" const char* lorrax_ffi_host_build_config() { return kBuildConfig; }
