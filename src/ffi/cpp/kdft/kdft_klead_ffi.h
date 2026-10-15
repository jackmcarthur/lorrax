// kdft_klead_ffi.h -- HOST: the k-leading pass builder (kdft_klead_ffi.cc), shared with the
// k-minor split arm (kdft_kminor_ffi.cc), whose column passes are k-leading passes (modes 3, 4).
#pragma once

#include <string_view>

#include "../common/ffi_helpers.h"
#include "../common/rtc_build.h"
#include "kdft_klead_plan.h"

namespace lorrax_ffi::kdft {

// The run-time-compiled image of one pass (kdft_klead_kernel.h lrx_kdft_pass), built once per
// process and disk-cached in cache_dir.  Kernel arguments: (t, v, inner, ntile, mx, nb, my,
// scale, forward, kob, m0, m1).  A failure names GATE kdft-build.
ffi::Error build_pass(const ::kdft::Pass& p, bool f32, std::string_view cache_dir, rtc::Function* out);

}  // namespace lorrax_ffi::kdft
