// The k-minor handler's run-time allocation, from its existing shared plan.
// CPU only: no device/driver initialization and no duplicate plan policy.
#pragma once

#include <climits>
#include <cstdint>
#include "kdft_kminor_plan.h"

namespace kdft {

inline std::int64_t kminor_workspace_bytes(const MinorPlan& plan, int mode,
                                           int out, std::int64_t input_bytes) {
    return plan.split && mode == 4 && out == 1 ? input_bytes : 0;
}

// Negative means a geometry the actual handler cannot execute; never zero as
// a substitute for an unsupported plan. Modes 4/5 are the native k-minor pair.
inline std::int64_t kminor_workspace_bytes(int nx, int ny, int nz, int mode,
        int out, std::int64_t rows, std::int64_t m, int cbytes) {
    if ((mode != 4 && mode != 5) || (out != 0 && out != 1) ||
        (mode == 5 && out != 0) || (cbytes != 8 && cbytes != 16)) return -1;
    if (nx < 1 || ny < 1 || nz < 1 || nx > kAxisMax || ny > kAxisMax || nz > kAxisMax)
        return -2;
    const std::int64_t nk = std::int64_t(nx) * ny * nz;
    if (nk > kNkMax || rows < 0 || rows > INT_MAX) return -3;
    if (rows == 0) return 0;
    if (m < 1 || m > rows || rows % m) return -4;
    const MinorPlan plan = plan_kminor(nx, ny, nz, mode, 0, out, cbytes, mode == 5 ? 1 : m);
    if (!plan.rows.c || (plan.split && rows * plan.ot > INT_MAX)) return -5;
    return kminor_workspace_bytes(plan, mode, out, rows * nk * cbytes);
}

}  // namespace kdft
