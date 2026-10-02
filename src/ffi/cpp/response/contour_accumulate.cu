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
            // Match XLA complex multiply then add: each scalar operation
            // rounds independently. Compiler FMA contraction changes stream bytes.
            output[offset] = make_cuDoubleComplex(
                __dadd_rn(a.x, __dsub_rn(__dmul_rn(p.x, c.x), __dmul_rn(p.y, c.y))),
                __dadd_rn(a.y, __dadd_rn(__dmul_rn(p.x, c.y), __dmul_rn(p.y, c.x))));
        }
    }
}
// Block form: A[o, q, m0+m, n0+n] += sum_s p[s,o] c[s, q, m, n] for m < mv, n < nv (in place;
// entries past the valid counts are not touched).  The s terms add in order, each as the tiled
// kernel's rounding, so two calls (s = 0, then 1) of the full form give the same bytes.  origin
// (null: none): a runtime (m0, n0) added to the static one, e.g. a scanned pass's row offset;
// entries it moves outside A are not touched.
__global__ void contour_accumulate_block(const cuDoubleComplex* contribution,
    const cuDoubleComplex* projection, const int* valid, const int* origin, cuDoubleComplex* output,
    int64_t outputs, int64_t nq, int64_t M, int64_t N, int64_t bm, int64_t bn,
    int64_t m0, int64_t n0, int64_t terms) {
    const int64_t block = nq * bm * bn;
    const int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= block) return;
    if (origin) { m0 += origin[0]; n0 += origin[1]; }
    const int64_t n = e % bn, m = (e / bn) % bm, q = e / (bm * bn);
    if (m >= valid[0] || n >= valid[1] || m0 + m < 0 || m0 + m >= M || n0 + n < 0 || n0 + n >= N) return;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int64_t o = int64_t(blockIdx.y) * 4 + j;
        if (o >= outputs) break;
        const int64_t offset = ((o * nq + q) * M + (m0 + m)) * N + (n0 + n);
        cuDoubleComplex a = output[offset];
        for (int64_t s = 0; s < terms; ++s) {
            const cuDoubleComplex c = contribution[s * block + e];
            const cuDoubleComplex p = projection[s * outputs + o];
            a = make_cuDoubleComplex(
                __dadd_rn(a.x, __dsub_rn(__dmul_rn(p.x, c.x), __dmul_rn(p.y, c.y))),
                __dadd_rn(a.y, __dadd_rn(__dmul_rn(p.x, c.y), __dmul_rn(p.y, c.x))));
        }
        output[offset] = a;
    }
}
void launch_block(const void* c, const void* p, const void* valid, const void* origin, void* out,
                  int64_t outputs, int64_t nq, int64_t M, int64_t N, int64_t bm, int64_t bn, int64_t m0,
                  int64_t n0, int64_t terms, cudaStream_t stream) {
    const dim3 grid((nq * bm * bn + 127) / 128, (outputs + 3) / 4);
    contour_accumulate_block<<<grid, 128, 0, stream>>>(
        static_cast<const cuDoubleComplex*>(c), static_cast<const cuDoubleComplex*>(p),
        static_cast<const int*>(valid), static_cast<const int*>(origin), static_cast<cuDoubleComplex*>(out),
        outputs, nq, M, N, bm, bn, m0, n0, terms);
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
