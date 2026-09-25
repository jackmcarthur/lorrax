// Gate + bench for src/ffi/cpp/cufft/kbox_stage.cuh.
//
// Per k-grid and mode, on a k-leading (nk, ncols) c128 tile:
//   today   a verbatim copy of the family's resident modes 2/3 kernel (axis_pass/transform3 on
//           RB rows of SP = nk|1, lrx_mul, scale at the store), compiled in this TU
//   stage   the header's arm chosen by lrx_kbox::kbox_plan from the device's opt-in smem
//   ref     cuFFT rank-3 (element stride ncols) + elementwise kernels
// Reports: bitwise(stage, today), rel(stage, ref), rel(today, ref), times.
// Mode 8 (Lorentz spin group, ns 4) runs the split arm's group pencil against ref only.
// usage: bench_kbox_stage MODE NX NY NZ M    (mode 3: ncols = M; mode 2: 4 M^2; mode 8: 16 M^2)
// build (one A100; MATHDX = the nvidia-mathdx wheel's nvidia/mathdx directory):
//   nvcc -O3 -std=c++17 -arch=sm_80 --expt-relaxed-constexpr -I src/ffi/cpp/cufft \
//        -I$MATHDX/include -I$MATHDX/external/cutlass/include -I$MATH_LIBS/include \
//        tests/bench/bench_kbox_stage.cu -o bench_kbox_stage -L$MATH_LIBS/lib64 -lcufft
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftdx.hpp>
#include "kbox_stage.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#define CK(x) do { auto _e = (x); if (_e != 0) { std::fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, int(_e)); std::exit(1); } } while (0)
constexpr int ARCH = 800;
struct __align__(16) C { double x, y; };
__device__ __forceinline__ C lrx_mul(C a, C b) { C z = {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x}; return z; }

// ---------------- today's resident kernel (modes 2/3), verbatim structure ----------------
template <int N, int STRIDE, cufftdx::fft_direction Dir, int NK, int SP, int RB>
__device__ __forceinline__ void t_axis_pass(C* bank) {
    if constexpr (N > 1) {
        using F = lrx_kbox::ThreadFFT<N, ARCH, Dir, C>;
        using V = typename F::value_type;
        constexpr int lines = NK / N;
        for (int l = threadIdx.x; l < RB * lines; l += blockDim.x) {
            const int j = l / lines, li = l % lines;
            C* p = bank + j * SP + (li / STRIDE) * N * STRIDE + (li % STRIDE);
            V v[F::storage_size];
#pragma unroll
            for (int e = 0; e < N; ++e) { v[e].x = p[e * STRIDE].x; v[e].y = p[e * STRIDE].y; }
            F().execute(v);
#pragma unroll
            for (int e = 0; e < N; ++e) { p[e * STRIDE].x = v[e].x; p[e * STRIDE].y = v[e].y; }
        }
    }
    __syncthreads();
}
template <int NX, int NY, int NZ, int RB, cufftdx::fft_direction Dir>
__device__ void t_transform3(C* bank) {
    constexpr int NK = NX * NY * NZ, SP = NK | 1;
    t_axis_pass<NZ, 1, Dir, NK, SP, RB>(bank);
    t_axis_pass<NY, NZ, Dir, NK, SP, RB>(bank);
    t_axis_pass<NX, NY * NZ, Dir, NK, SP, RB>(bank);
}
template <int MODE, int NX, int NY, int NZ, int RB>
__global__ void __launch_bounds__(256) today_kernel(const C* x, const C* kern, C* y, long long rows,
                                                    long long m0, long long m1, long long m2, double scale) {
    constexpr int NK = NX * NY * NZ, SP = NK | 1;
    extern __shared__ C sm[];
    const long long r0 = (long long)blockIdx.x * RB;
    using namespace cufftdx;
    for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
        const int k = i / RB, j = i % RB;
        const long long row = r0 + j;
        C v = {0.0, 0.0};
        if (row < rows) v = x[(long long)k * rows + row];
        sm[j * SP + k] = v;
    }
    __syncthreads();
    if (MODE == 2) t_transform3<NX, NY, NZ, RB, fft_direction::inverse>(sm);
    else t_transform3<NX, NY, NZ, RB, fft_direction::forward>(sm);
    if (MODE == 2) {
        for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
            const int k = i / RB, j = i % RB;
            const long long row = r0 + j;
            if (row < rows) {
                const long long kidx = (long long)k * (m2 * m0) + ((row / m1) % m2) * m0 + row % m0;
                sm[j * SP + k] = lrx_mul(sm[j * SP + k], kern[kidx]);
            }
        }
        __syncthreads();
        t_transform3<NX, NY, NZ, RB, fft_direction::forward>(sm);
    }
    for (int i = threadIdx.x; i < RB * NK; i += blockDim.x) {
        const int k = i / RB, j = i % RB;
        const long long row = r0 + j;
        if (row < rows) {
            const C v = sm[j * SP + k];
            C w; w.x = v.x * scale; w.y = v.y * scale;
            y[(long long)k * rows + row] = w;
        }
    }
}

// ---------------- the header's arms, wrapped as the family's modes would ----------------
struct PlainLoad { static constexpr bool kDirect = false, kFinish = false; const C* x; long long n;
    __device__ const C* stage(int k, long long c) const { return x + (long long)k * n + c; } };
// The same elements through the direct-load contract (plain loads, no cp.async): must be bitwise.
struct DirectLoad { static constexpr bool kDirect = true, kFinish = false; const C* x; long long n;
    template <class View>
    __device__ void direct(const View& v, int k0, int k1, long long c0, int width, long long nc) const {
        for (int i = threadIdx.x; i < (k1 - k0) * width; i += blockDim.x) {
            const int j = i % width, k = k0 + i / width;
            C z; z.x = 0; z.y = 0;
            if (c0 + j < nc) z = x[(long long)k * n + c0 + j];
            v(k, j) = z;
        }
    } };
struct IdGroup { __device__ void group(int, long long, C*) const {} };
struct ScaleStore { C* y; long long n; double s;
    __device__ void put(int k, long long c, C v) const { C w; w.x = v.x * s; w.y = v.y * s; y[(long long)k * n + c] = w; } };
struct VMid { const C* V; long long m0, m1, m2;
    __device__ C operator()(int k, long long r, C a) const {
        return lrx_mul(a, V[(long long)k * (m2 * m0) + ((r / m1) % m2) * m0 + r % m0]); } };
struct IdMid { __device__ C operator()(int, long long, C a) const { return a; } };

template <int MODE, int NX, int NY, int NZ, int TR, int THREADS = 256>
__global__ void __launch_bounds__(THREADS) stage_single(const C* x, C* y, const C* V, long long n,
                                                    long long m0, long long m1, long long m2, double s) {
    extern __shared__ C bank[];
    using namespace cufftdx;
    for (long long c0 = (long long)blockIdx.x * TR; c0 < n; c0 += (long long)gridDim.x * TR) {
        lrx_kbox::stage_tile<NX, NY, NZ, TR>(bank, c0, n, PlainLoad{x, n});
        if constexpr (MODE == 3) {
            lrx_kbox::transform3<NX, NY, NZ, TR, ARCH, fft_direction::forward>(bank);
        } else {
            lrx_kbox::transform3<NX, NY, NZ, TR, ARCH, fft_direction::inverse>(bank);
            lrx_kbox::mid_tile<NX, NY, NZ, TR>(bank, c0, n, VMid{V, m0, m1, m2});
            lrx_kbox::transform3<NX, NY, NZ, TR, ARCH, fft_direction::forward>(bank);
        }
        lrx_kbox::store_tile<NX, NY, NZ, TR>(bank, c0, n, ScaleStore{y, n, s});
    }
}
// Mode 3 through the direct Load and an identity group Mid (GROUP 2): the new contracts' gate.
template <int NX, int NY, int NZ, int TR, int THREADS = 256>
__global__ void __launch_bounds__(THREADS) stage_direct3(const C* x, C* y, long long n, double s) {
    extern __shared__ C bank[];
    for (long long c0 = (long long)blockIdx.x * TR; c0 < n; c0 += (long long)gridDim.x * TR) {
        lrx_kbox::stage_tile<NX, NY, NZ, TR>(bank, c0, n, DirectLoad{x, n});
        lrx_kbox::transform3<NX, NY, NZ, TR, ARCH, cufftdx::fft_direction::forward>(bank);
        lrx_kbox::mid_group_tile<NX, NY, NZ, TR, 2>(bank, c0, n, IdGroup{});
        lrx_kbox::store_tile<NX, NY, NZ, TR>(bank, c0, n, ScaleStore{y, n, s});
    }
}
template <int NX, int NY, int NZ, cufftdx::fft_direction Dir, class Ld, class St>
__global__ void __launch_bounds__(256) stage_plane(Ld ld, St st, long long n) {
    extern __shared__ C sm[];
    lrx_kbox::plane_pass<NX, NY, NZ, ARCH, Dir, 16>(sm, n, ld, st);
}
template <int NX, int NY, int NZ, bool CONV>
__global__ void __launch_bounds__(256) stage_pencil(C* y, long long n, const C* V, long long m0, long long m1, long long m2) {
    if constexpr (CONV) lrx_kbox::pencil_pass<NX, NY, NZ, ARCH, true, cufftdx::fft_direction::forward>(y, n, VMid{V, m0, m1, m2});
    else lrx_kbox::pencil_pass<NX, NY, NZ, ARCH, false, cufftdx::fft_direction::forward>(y, n, IdMid{});
}

// mode 8: G/U columns ((a*m + x)*4 + b)*m + y, V (nk, m, 4, m, 4) in R space
__constant__ int c_pl[16], c_pr[16], c_cl[16], c_cr[16];   // perms and quarter-turn codes
__constant__ C c_hl[16], c_hr[16];
__device__ __forceinline__ C cconj(C a) { C z = {a.x, -a.y}; return z; }
struct Cols8 { long long m; __device__ long long col(long long inst, int member) const {
    const long long x = inst / m, y = inst % m; const int a = member / 4, b = member % 4;
    return ((a * m + x) * 4 + b) * m + y; } };
// The mode-8 Mid on the header's group pencil: V[k, x, A, y, B] staged per block by cp.async
// (kAux = 16 per instance), each thread's vertex tables in registers (source member and
// quarter-turn code per (A, B)): out[a,b] = sum_AB i^(cl-cr) G[pl_A(a), pr_B(b)] V[A, B].
__device__ __forceinline__ C qturn(C g, int code) {       // g * i^code, exact
    switch (code & 3) { case 0: return g; case 1: return C{-g.y, g.x}; case 2: return C{-g.x, -g.y}; default: return C{g.y, -g.x}; }
}
__constant__ int c_src[256], c_code[256];                 // [out member][A*4+B]: source member, quarter code
template <int NX, int NY, int NZ, int TY>
struct Mid8 {
    static constexpr int kAux = 16;
    const C* V; long long m, n_inst;
    __device__ void stage_aux(C* saux, long long p, long long inst0, int ld) const {
        for (int i = threadIdx.x; i < NX * 16 * TY; i += blockDim.x) {
            const int B = i & 3, t = (i >> 2) % TY, A = (i / (4 * TY)) & 3, kx = i / (16 * TY);
            long long inst = inst0 + t; if (inst >= n_inst) inst = inst0;
            const long long x = inst / m, y = inst % m, k = (long long)kx * NY * NZ + p;
            lrx_kbox::cp_async<16>(saux + (kx * TY + t) * ld + A * 4 + B, V + ((((k * m + x) * 4 + A) * m + y) * 4 + B));
        }
    }
    struct Bound {
        int src[16], code[16];
        __device__ C operator()(const C* grp, const C* aux) const {
            C acc = {0, 0};
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const C t = lrx_mul(qturn(grp[src[e] * TY], code[e]), aux[e]);
                acc.x += t.x; acc.y += t.y;
            }
            return acc;
        }
    };
    __device__ Bound bind(int member) const {
        Bound b;
#pragma unroll
        for (int e = 0; e < 16; ++e) { b.src[e] = c_src[member * 16 + e]; b.code[e] = c_code[member * 16 + e]; }
        return b;
    }
};
template <int NX, int NY, int NZ, int TY>
__global__ void __launch_bounds__(16 * TY) stage_pencil8(C* y, long long n, const C* V, long long m) {
    extern __shared__ C sm8[];
    lrx_kbox::pencil_group_pass<NX, NY, NZ, ARCH, 16, TY>(y, sm8, n, m * m, Cols8{m}, Mid8<NX, NY, NZ, TY>{V, m, m * m});
}

// ---------------- reference ----------------
__global__ void ref_mul(C* A, const C* V, long long nk, long long n, long long m0, long long m1, long long m2) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < nk * n; i += (long long)gridDim.x * blockDim.x) {
        const long long k = i / n, r = i % n;
        A[i] = lrx_mul(A[i], V[k * (m2 * m0) + ((r / m1) % m2) * m0 + r % m0]);
    }
}
__global__ void ref_vertex8(const C* G, C* U, const C* V, long long nk, long long m) {
    const long long n = 16 * m * m;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < nk * n; i += (long long)gridDim.x * blockDim.x) {
        const long long k = i / n, r = i % n;
        const long long y = r % m, x = (r / (4 * m)) % m; const int b = int((r / m) % 4), a = int(r / (4 * m * m));
        C acc = {0, 0};
        for (int A = 0; A < 4; ++A)
            for (int B = 0; B < 4; ++B) {
                const C g = G[k * n + ((c_pl[A * 4 + a] * m + x) * 4 + c_pr[B * 4 + b]) * m + y];
                const C t = lrx_mul(lrx_mul(lrx_mul(c_hl[A * 4 + a], cconj(c_hr[B * 4 + b])), g), V[(((k * m + x) * 4 + A) * m + y) * 4 + B]);
                acc.x += t.x; acc.y += t.y;
            }
        U[i] = acc;
    }
}
__global__ void ref_scale(C* A, long long len, double s) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < len; i += (long long)gridDim.x * blockDim.x) { A[i].x *= s; A[i].y *= s; }
}

static float time_ms(const std::function<void()>& f, int reps = 10) {
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i = 0; i < 2; ++i) f();
    CK(cudaDeviceSynchronize());
    std::vector<float> t;
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(e0)); f(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}
static double rel(const C* a_d, const C* b_d, long long len, bool* bitwise = nullptr) {
    std::vector<C> a(len), b(len);
    CK(cudaMemcpy(a.data(), a_d, len * 16, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), b_d, len * 16, cudaMemcpyDeviceToHost));
    double md = 0, mr = 0;
    for (long long i = 0; i < len; ++i) {
        md = std::max(md, std::hypot(a[i].x - b[i].x, a[i].y - b[i].y));
        mr = std::max(mr, std::hypot(b[i].x, b[i].y));
    }
    if (bitwise) *bitwise = std::memcmp(a.data(), b.data(), len * 16) == 0;
    return md / mr;
}
template <class K>
static int grid_of(K k, int threads, size_t smem) {
    if (smem > 48 * 1024) CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
    int per = 0, sms = 0; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, threads, smem));
    return std::max(1, per) * sms;
}

template <int MODE, int NX, int NY, int NZ>
static void run(long long m) {
    constexpr int NK = NX * NY * NZ, SP = NK | 1;
    constexpr int RB = std::min(64, int((48 * 1024) / (SP * 16)) > 0 ? int((48 * 1024) / (SP * 16)) : 1);
    const long long n = MODE == 3 ? m : MODE == 2 ? 4 * m * m : 16 * m * m;
    const long long len = NK * n, vlen = MODE == 2 ? NK * m * m : MODE == 8 ? NK * 16 * m * m : 1;
    const long long m0 = m, m1 = 2 * m, m2 = m;
    const double s = MODE == 3 ? 1.0 : 1.0 / NK;
    C *x, *yt, *yh, *yr, *tmp, *V;
    CK(cudaMalloc(&x, len * 16)); CK(cudaMalloc(&yt, len * 16)); CK(cudaMalloc(&yh, len * 16));
    CK(cudaMalloc(&yr, len * 16)); CK(cudaMalloc(&tmp, len * 16)); CK(cudaMalloc(&V, vlen * 16));
    {
        unsigned long long st = 777 + NK;
        auto rnd = [&] { st = st * 6364136223846793005ULL + 1442695040888963407ULL; return double(st >> 11) / 9007199254740992.0 - 0.5; };
        std::vector<C> h(len); for (auto& v : h) v = C{rnd(), rnd()};
        CK(cudaMemcpy(x, h.data(), len * 16, cudaMemcpyHostToDevice));
        std::vector<C> hv(vlen); for (auto& v : hv) v = C{rnd(), rnd()};
        CK(cudaMemcpy(V, hv.data(), vlen * 16, cudaMemcpyHostToDevice));
        int pl[16], pr[16], cl[16], cr[16]; C hl[16], hr[16]; const C q[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        for (int A = 0; A < 4; ++A) {
            int p[4] = {0, 1, 2, 3};
            for (int i = 3; i > 0; --i) std::swap(p[i], p[int((rnd() + 0.5) * (i + 1)) % (i + 1)]);
            for (int a = 0; a < 4; ++a) { pl[A * 4 + a] = p[a]; cl[A * 4 + a] = int((rnd() + 0.5) * 4) % 4; hl[A * 4 + a] = q[cl[A * 4 + a]]; }
            for (int i = 3; i > 0; --i) std::swap(p[i], p[int((rnd() + 0.5) * (i + 1)) % (i + 1)]);
            for (int a = 0; a < 4; ++a) { pr[A * 4 + a] = p[a]; cr[A * 4 + a] = int((rnd() + 0.5) * 4) % 4; hr[A * 4 + a] = q[cr[A * 4 + a]]; }
        }
        CK(cudaMemcpyToSymbol(c_cl, cl, sizeof cl)); CK(cudaMemcpyToSymbol(c_cr, cr, sizeof cr));
        int src[256], code[256];
        for (int o = 0; o < 16; ++o)
            for (int A = 0; A < 4; ++A)
                for (int B = 0; B < 4; ++B) {
                    src[o * 16 + A * 4 + B] = pl[A * 4 + o / 4] * 4 + pr[B * 4 + o % 4];
                    code[o * 16 + A * 4 + B] = cl[A * 4 + o / 4] - cr[B * 4 + o % 4];
                }
        CK(cudaMemcpyToSymbol(c_src, src, sizeof src)); CK(cudaMemcpyToSymbol(c_code, code, sizeof code));
        CK(cudaMemcpyToSymbol(c_pl, pl, sizeof pl)); CK(cudaMemcpyToSymbol(c_pr, pr, sizeof pr));
        CK(cudaMemcpyToSymbol(c_hl, hl, sizeof hl)); CK(cudaMemcpyToSymbol(c_hr, hr, sizeof hr));
    }
    // reference
    cufftHandle plan; int nn[3] = {NX, NY, NZ};
    CK(cufftPlanMany(&plan, 3, nn, nn, int(n), 1, nn, int(n), 1, CUFFT_Z2Z, int(n)));
    auto Z = [](C* p) { return reinterpret_cast<cufftDoubleComplex*>(p); };
    if (MODE == 3) {
        CK(cufftExecZ2Z(plan, Z(x), Z(yr), CUFFT_FORWARD));
    } else if (MODE == 2) {
        CK(cufftExecZ2Z(plan, Z(x), Z(yr), CUFFT_INVERSE));
        ref_mul<<<4096, 256>>>(yr, V, NK, n, m0, m1, m2);
        CK(cufftExecZ2Z(plan, Z(yr), Z(yr), CUFFT_FORWARD));
        ref_scale<<<4096, 256>>>(yr, len, s);
    } else {
        CK(cufftExecZ2Z(plan, Z(x), Z(tmp), CUFFT_INVERSE));
        ref_vertex8<<<4096, 256>>>(tmp, yr, V, NK, m);
        CK(cufftExecZ2Z(plan, Z(yr), Z(yr), CUFFT_FORWARD));
        ref_scale<<<4096, 256>>>(yr, len, s);
    }
    CK(cudaDeviceSynchronize());
    int optin = 0; CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    const auto P = lrx_kbox::kbox_plan(NX, NY, NZ, MODE == 8 ? 16 : 1, 1, 16, optin, MODE == 3 ? 1 : 2);
    const double floor_us = (2.0 * len + vlen) * 16 / 1.555e12 * 1e6;
    std::printf("mode %d k-grid %dx%dx%d ncols=%lld (%.0f MB): plan arm=%s tr=%d threads=%d smem=%lld; floor %.1f us\n",
                MODE, NX, NY, NZ, n, len * 16 / 1e6, P.arm == 0 ? "single" : "split", P.tr, P.threads, P.smem, floor_us);
    // today (modes 2/3)
    float tt = 0; double rt = 0;
    if (MODE != 8) {
        const size_t smt = size_t(RB) * SP * 16;
        auto kt = today_kernel<MODE == 8 ? 3 : MODE, NX, NY, NZ, RB>;
        grid_of(kt, 256, smt);
        const long long gt = (n + RB - 1) / RB;
        auto ft = [&] { kt<<<unsigned(gt), 256, smt>>>(x, V, yt, n, m0, m1, m2, s); };
        tt = time_ms(ft); ft(); CK(cudaDeviceSynchronize());
        rt = rel(yt, yr, len);
    }
    // the header's arm (fd: mode 3 through the direct Load (+ identity group Mid on the single arm))
    std::function<void()> fh, fd;
    C* yd = nullptr;
    if (MODE == 3) CK(cudaMalloc(&yd, len * sizeof(C)));
    if (P.arm == 0 && MODE != 8) {
        constexpr int MD = MODE == 8 ? 3 : MODE;
        auto launch = [&](auto k, int T) {
            const int g = grid_of(k, T, size_t(P.smem));
            fh = [=] { k<<<g, T, size_t(P.smem)>>>(x, yh, V, n, m0, m1, m2, s); };
        };
#define TRC(TRV) case TRV: if (P.threads == 512) launch(stage_single<MD, NX, NY, NZ, TRV, 512>, 512); \
                           else launch(stage_single<MD, NX, NY, NZ, TRV, 256>, 256); break;
        switch (P.tr) {
            TRC(2) TRC(4) TRC(8) TRC(16) TRC(32) TRC(64)
            default: std::printf("  tr %d not instantiated\n", P.tr); return;
        }
#undef TRC
        if (MODE == 3) {
            auto ld = [&](auto k, int T) { const int g = grid_of(k, T, size_t(P.smem));
                                           fd = [=] { k<<<g, T, size_t(P.smem)>>>(x, yd, n, s); }; };
#define TRD(TRV) case TRV: ld(stage_direct3<NX, NY, NZ, TRV>, 256); break;
            switch (P.tr) { TRD(2) TRD(4) TRD(8) TRD(16) TRD(32) TRD(64) default: break; }
#undef TRD
        }
    } else {
        const size_t psm = size_t(16) * lrx_kbox::Geo<NX, NY, NZ>::PR * 16;
        using PL = lrx_kbox::Plain<C>;
        auto pi = stage_plane<NX, NY, NZ, cufftdx::fft_direction::inverse, PlainLoad, PL>;
        auto pf = stage_plane<NX, NY, NZ, cufftdx::fft_direction::forward, PlainLoad, PL>;
        auto pfs = stage_plane<NX, NY, NZ, cufftdx::fft_direction::forward, PlainLoad, ScaleStore>;
        const int gp = grid_of(pi, 256, psm); grid_of(pf, 256, psm); grid_of(pfs, 256, psm);
        if (MODE == 3) {
            auto pc = stage_pencil<NX, NY, NZ, false>; const int g2 = grid_of(pc, 256, 0);
            fh = [=] { pf<<<gp, 256, psm>>>(PlainLoad{x, n}, PL{yh, n}, n); pc<<<g2, 256>>>(yh, n, V, m0, m1, m2); };
            auto pd = stage_plane<NX, NY, NZ, cufftdx::fft_direction::forward, DirectLoad, PL>; grid_of(pd, 256, psm);
            fd = [=] { pd<<<gp, 256, psm>>>(DirectLoad{x, n}, PL{yd, n}, n); pc<<<g2, 256>>>(yd, n, V, m0, m1, m2); };
        } else if (MODE == 2) {
            auto pc = stage_pencil<NX, NY, NZ, true>; const int g2 = grid_of(pc, 256, 0);
            fh = [=] { pi<<<gp, 256, psm>>>(PlainLoad{x, n}, PL{yh, n}, n); pc<<<g2, 256>>>(yh, n, V, m0, m1, m2);
                       pfs<<<gp, 256, psm>>>(PlainLoad{yh, n}, ScaleStore{yh, n, s}, n); };
        } else {
            constexpr int TY8 = 8;
            auto p8 = stage_pencil8<NX, NY, NZ, TY8>;
            const size_t s8 = (size_t(NX) * 16 * TY8 + size_t(NX) * TY8 * 17) * 16;
            const int g2 = grid_of(p8, 16 * TY8, s8);
            fh = [=] { pi<<<gp, 256, psm>>>(PlainLoad{x, n}, PL{yh, n}, n); p8<<<g2, 16 * TY8, s8>>>(yh, n, V, m);
                       pfs<<<gp, 256, psm>>>(PlainLoad{yh, n}, ScaleStore{yh, n, s}, n); };
            const float t1 = time_ms([=] { pi<<<gp, 256, psm>>>(PlainLoad{x, n}, PL{yh, n}, n); });
            const float t2 = time_ms([=] { p8<<<g2, 16 * TY8, s8>>>(yh, n, V, m); });
            const float t3 = time_ms([=] { pfs<<<gp, 256, psm>>>(PlainLoad{yh, n}, ScaleStore{yh, n, s}, n); });
            const double gb = len * 16 / 1e9;
            std::printf("  split passes: plane_inv %.1f us (%.0f GB/s), pencil8 %.1f us (%.0f GB/s), plane_fwd %.1f us; 3-pass floor %.1f us\n",
                        t1 * 1e3, 2 * gb / (t1 * 1e-3), t2 * 1e3, 3 * gb / (t2 * 1e-3), t3 * 1e3, 7 * gb * 1e9 / 1.555e12 * 1e6);
        }
    }
    const float th = time_ms(fh); fh(); CK(cudaDeviceSynchronize());
    if (fd) {
        const float td = time_ms(fd); fd(); CK(cudaDeviceSynchronize());
        bool bd = false;
        const double dd = rel(yd, yh, len, &bd);
        std::printf("  direct Load%s: %9.1f us, vs the cp.async stage: %s (%.1e)\n",
                    P.arm == 0 ? " + group Mid" : "", td * 1e3, bd ? "BITWISE" : "DIFFERS", dd);
    }
    if (yd) cudaFree(yd);
    bool bw = false;
    const double rh = rel(yh, yr, len);
    double dht = 0;
    if (MODE != 8) dht = rel(yh, yt, len, &bw);
    if (MODE != 8)
        std::printf("  today %9.1f us (rel %.1e)   stage %9.1f us (rel %.1e, %3.0f%% of floor)   stage vs today: %s (%.1e)  %.2fx\n",
                    tt * 1e3, rt, th * 1e3, rh, 100 * floor_us / (th * 1e3), bw ? "BITWISE" : "round-off", dht, tt / th);
    else
        std::printf("  stage %9.1f us (rel %.1e, %3.0f%% of floor)\n", th * 1e3, rh, 100 * floor_us / (th * 1e3));
    cufftDestroy(plan);
    cudaFree(x); cudaFree(yt); cudaFree(yh); cudaFree(yr); cudaFree(tmp); cudaFree(V);
}

int main(int argc, char** argv) {
    const int mode = std::atoi(argv[1]), nx = std::atoi(argv[2]), ny = std::atoi(argv[3]), nz = std::atoi(argv[4]);
    const long long m = std::atoll(argv[5]);
#define G(MD, X, Y, Z) if (mode == MD && nx == X && ny == Y && nz == Z) { run<MD, X, Y, Z>(m); return 0; }
    G(3, 8, 8, 8) G(3, 10, 10, 10) G(3, 12, 12, 12) G(3, 16, 16, 16) G(3, 6, 6, 1) G(3, 8, 8, 1) G(3, 4, 4, 4)
    G(2, 8, 8, 8) G(2, 10, 10, 10) G(2, 12, 12, 12) G(2, 16, 16, 16) G(2, 6, 6, 1) G(2, 8, 8, 1) G(2, 4, 4, 4)
    G(8, 8, 8, 8) G(8, 10, 10, 10) G(8, 12, 12, 12) G(8, 16, 16, 16) G(8, 4, 4, 4)
    std::printf("case not instantiated\n");
    return 1;
}
