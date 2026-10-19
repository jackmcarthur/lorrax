// Process-lifetime cuSOLVERMp plans: a call site's matrix descriptors and the vendor's workspace
// sizes, keyed by shape, built once under a lock and never freed (they live as long as the
// context they reference; the process exits). cuSOLVERMp 0.9.1's bufferSize queries build
// cuBLASMp grids for the descriptors they are given (cublasMpGridCreate -> ncclCommSplit ->
// ncclCommInitRank): one per call was one communicator per call, and a rank deadlocked in
// libfabric's memory hooks inside such a split (run30k, 2026-10-19, gdb on all 16 ranks).
// Collective calls reach a site in the same order on every rank, so every rank builds alike.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>

#include <cusolverMp.h>

namespace lorrax_ffi::cusolvermp {

struct MpPlan {
    cusolverMpMatrixDescriptor_t desc[2] = {nullptr, nullptr};
    size_t d_ws = 0, h_ws = 0;
    bool built = false;
};

//: [0] the call site (MpSite), [1] the context, [2] the dtype, then the site's shape fields.
using MpPlanKey = std::array<int64_t, 11>;
enum MpSite : int64_t { kMpEigh = 1, kMpPotrf = 2, kMpPotrs = 3 };

// The plan for ``key``: ``build(MpPlan&)`` (0 on success, else a status) runs once per key; on
// failure the descriptors it made are destroyed and its status returned.
inline int mp_plan(const MpPlanKey& key, const MpPlan** out, const std::function<int(MpPlan&)>& build) {
    static std::mutex mu;
    static std::map<MpPlanKey, MpPlan> plans;                // node-based: references stay valid
    std::lock_guard<std::mutex> lock(mu);
    MpPlan& p = plans[key];
    if (!p.built) {
        if (const int rc = build(p)) {
            for (auto& d : p.desc)
                if (d) { cusolverMpDestroyMatrixDesc(d); d = nullptr; }
            p.d_ws = p.h_ws = 0;
            return rc;
        }
        p.built = true;
    }
    *out = &p;
    return 0;
}

}  // namespace lorrax_ffi::cusolvermp
