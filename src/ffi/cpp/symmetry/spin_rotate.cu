// U G U^dagger at independent (k,mu,nu), with no global intermediate.
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>

namespace lorrax_ffi::symmetry {
template <int S>
__global__ void spin_rotate(const cuDoubleComplex* input,
                           const cuDoubleComplex* spin,
                           cuDoubleComplex* output,
                           int64_t nk, int64_t mu, int64_t nu) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nk * mu * nu) return;
    const int64_t k = i / (mu * nu), m = (i / nu) % mu, n = i % nu;
    cuDoubleComplex u[S][S], g[S][S], left[S][S];
    #pragma unroll
    for (int a = 0; a < S; ++a) {
        #pragma unroll
        for (int b = 0; b < S; ++b) {
            u[a][b] = spin[(k * S + a) * S + b];
            g[a][b] = input[(((k * mu + m) * S + a) * nu + n) * S + b];
        }
    }
    #pragma unroll
    for (int a = 0; a < S; ++a) {
        #pragma unroll
        for (int d = 0; d < S; ++d) {
            cuDoubleComplex v = make_cuDoubleComplex(0., 0.);
            #pragma unroll
            for (int c = 0; c < S; ++c)
                v = cuCadd(v, cuCmul(u[a][c], g[c][d]));
            left[a][d] = v;
        }
    }
    // All input entries are consumed before any potentially aliased store.
    #pragma unroll
    for (int a = 0; a < S; ++a) {
        #pragma unroll
        for (int b = 0; b < S; ++b) {
            cuDoubleComplex v = make_cuDoubleComplex(0., 0.);
            #pragma unroll
            for (int d = 0; d < S; ++d)
                v = cuCadd(v, cuCmul(left[a][d], cuConj(u[b][d])));
            output[(((k * mu + m) * S + a) * nu + n) * S + b] = v;
        }
    }
}

void launch_spin_rotate(const void* g, const void* u, void* out,
                        int64_t nk, int s, int64_t mu, int64_t nu,
                        cudaStream_t stream) {
    const auto* input = static_cast<const cuDoubleComplex*>(g);
    const auto* spin = static_cast<const cuDoubleComplex*>(u);
    auto* output = static_cast<cuDoubleComplex*>(out);
    const unsigned blocks = (nk * mu * nu + 127) / 128;
    if (s == 2) spin_rotate<2><<<blocks, 128, 0, stream>>>(input, spin, output, nk, mu, nu);
    else spin_rotate<4><<<blocks, 128, 0, stream>>>(input, spin, output, nk, mu, nu);
}
}  // namespace lorrax_ffi::symmetry
