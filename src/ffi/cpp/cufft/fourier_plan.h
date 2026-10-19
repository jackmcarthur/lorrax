// Shared between fourier_plan.cu (the remap kernel) and fourier_plan_cuda_ffi.cc.
#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace lorrax_ffi::fourier_plan {

// out[b, o_0, …] = s · in[b, map_0[o_0], …], 0 where any map entry is < 0.
struct RemapShape {
    int d;                 // transform axes, 1..3 (trailing)
    int64_t batch;         // product of the leading axes
    int64_t out_ext[3];    // output extents of the transform axes
    int64_t in_ext[3];     // input extents
    const int32_t* map[3]; // device: out index -> in index, -1 = zero; nullptr = identity
};

void launch_remap(const void* in, void* out, const RemapShape& sh, double sr, double si,
                  cudaStream_t stream);

}  // namespace lorrax_ffi::fourier_plan
