// Coalesced register-tiled A[o,e] += p[o] c[e], complex128.
// c is the single already-transformed Keldysh correlation difference.
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>
namespace lorrax_ffi::contour {
__global__ void contour_accumulate_tiled(const cuDoubleComplex* input,
    const cuDoubleComplex* contribution, const cuDoubleComplex* projection,
    cuDoubleComplex* output, int64_t elements, int64_t outputs) {
    const int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= elements) return;
    const cuDoubleComplex c = contribution[e];
    // Four independent output rows per thread, contiguous elements per warp.
    // No reduction or time-node reordering; shared c stays in registers.
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int64_t o = int64_t(blockIdx.y) * 4 + j;
        if (o < outputs) {
            const cuDoubleComplex p = projection[o];
            const int64_t offset = o * elements + e;
            const cuDoubleComplex a = input[offset];
            output[offset] = make_cuDoubleComplex(
                a.x + (p.x * c.x - p.y * c.y),
                a.y + (p.x * c.y + p.y * c.x));
        }
    }
}
void launch(const void* a, const void* c, const void* p, void* out,
            int64_t elements, int64_t outputs, cudaStream_t stream) {
    const dim3 grid((elements + 127) / 128, (outputs + 3) / 4);
    contour_accumulate_tiled<<<grid, 128, 0, stream>>>(
        static_cast<const cuDoubleComplex*>(a),
        static_cast<const cuDoubleComplex*>(c),
        static_cast<const cuDoubleComplex*>(p),
        static_cast<cuDoubleComplex*>(out), elements, outputs);
}
}
