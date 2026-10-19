// LocalFourierPlan remap kernel: the one data-movement pass of the FFT arm.
//
// out[b, o_0, …, o_{d-1}] = s · in[b, m_0[o_0], …, m_{d-1}[o_{d-1}]], and 0 where
// any m_a[o_a] < 0.  A zero-filled embedding (sphere → box) and a restriction
// (box → sphere) are both this kernel: the embedding's map carries -1 off the
// support, so the box is zero-filled and gathered in one write.
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>

#include "fourier_plan.h"

namespace lorrax_ffi::fourier_plan {

__global__ void remap(const cuDoubleComplex* __restrict__ in, cuDoubleComplex* __restrict__ out,
                      RemapShape sh, int64_t total, double sr, double si) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += int64_t(gridDim.x) * blockDim.x) {
        int64_t rem = i, src = 0, stride = 1;
        bool zero = false;
        for (int a = sh.d - 1; a >= 0; --a) {
            const int64_t o = rem % sh.out_ext[a];
            rem /= sh.out_ext[a];
            const int64_t j = sh.map[a] ? sh.map[a][o] : o;
            zero |= (j < 0);
            src += j * stride;
            stride *= sh.in_ext[a];
        }
        src += rem * stride;           // rem is the batch index
        cuDoubleComplex v = zero ? make_cuDoubleComplex(0.0, 0.0) : in[src];
        out[i] = make_cuDoubleComplex(sr * v.x - si * v.y, sr * v.y + si * v.x);
    }
}

void launch_remap(const void* in, void* out, const RemapShape& sh, double sr, double si,
                  cudaStream_t stream) {
    int64_t total = sh.batch;
    for (int a = 0; a < sh.d; ++a) total *= sh.out_ext[a];
    if (total == 0) return;
    const int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    if (blocks > 65535LL * 64) blocks = 65535LL * 64;
    remap<<<unsigned(blocks), threads, 0, stream>>>(
        static_cast<const cuDoubleComplex*>(in), static_cast<cuDoubleComplex*>(out), sh, total,
        sr, si);
}

}  // namespace lorrax_ffi::fourier_plan
