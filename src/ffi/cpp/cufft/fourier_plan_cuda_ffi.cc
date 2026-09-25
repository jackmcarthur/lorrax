// lorrax_fourier_plan: one local separable DFT with per-axis supports, as ONE
// custom call (the CUDA leg of ffi.fft.make_fourier_plan).
//
//   y = R_out · F_{sign} · E_in · x   over the d ≤ 3 trailing axes of x,
//
// row-major in and out: x (B…, K_0, …, K_{d-1}) → y (B…, K'_0, …, K'_{d-1}).
// Each axis is a GEMM with a Fourier matrix stored in the plan (built once per
// device and plan key, in float64 with exact integer phase reduction) or part
// of one cuFFT group.  The Python door chooses the backend per axis and the
// stage order; this handler executes it:
//
//   * GEMM axis a, current shape (L, K, R): R = 1 is one cublasZgemm (op T on
//     the matrix); otherwise one cublasZgemmStridedBatched over L with the
//     matrix's stride 0.  No transposes, the output stays row-major.
//   * FFT group: one remap pass that zero-fills and gathers the embedded axes
//     (never a separate zero pass), one cuFFT Z2Z plan per contiguous run of
//     FFT axes (a GEMM axis between two FFT axes splits the group; the
//     transform is separable, so the runs compose to the group's transform),
//     one remap pass for the restricted axes.  The FFT axes' scale rides the
//     first GEMM's alpha, else a remap pass, else a final scale remap.
//
// Plans (device matrices, remap tables, cuFFT plans) are cached per
// (device, attributes, shape) for the process lifetime.  cuFFT's work areas
// are NOT owned by the plans: auto-allocation is off, and each call takes
// the largest work area from XLA's scratch allocator (inside the pool).
// Every cuFFT and cuBLAS size is checked against INT_MAX and refused by name.
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "xla/ffi/api/ffi.h"

#include "fourier_plan.h"

namespace lorrax_ffi::fourier_plan {

namespace ffi = ::xla::ffi;

static ffi::Error err(const std::string& what) {
    return ffi::Error(ffi::ErrorCode::kInternal, "lorrax_fourier_plan: " + what);
}
static ffi::Error bad(const std::string& what) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "lorrax_fourier_plan: " + what);
}

// One executable step of a plan.
struct Step {
    enum Kind { kGemm, kRemap, kFft } kind;
    int axis = -1;                         // GEMM axis
    cuDoubleComplex* mat = nullptr;        // device (N' × K) row-major
    RemapShape remap{};                    // remap geometry (maps are device pointers)
    double sr = 1.0, si = 0.0;             // remap scale, or GEMM alpha
    std::vector<cufftHandle> ffts;         // one contiguous FFT run: one handle, looped `loops` times
    int64_t loops = 1, loop_stride = 0;
    size_t work = 0;                       // its cuFFT work area, bytes (from XLA scratch per call)
    int64_t in_ext[3]{}, out_ext[3]{};     // current extents before / after the step
};

struct Plan {
    int d = 0;
    int64_t batch = 1;
    int sign = -1;
    std::vector<Step> steps;
    int64_t max_elems = 0;                 // largest intermediate
    size_t max_work = 0;                   // largest cuFFT work area over the steps
    std::vector<void*> owned;              // device allocations
};

constexpr int64_t kIntMax = 2147483647LL;

// cuFFT's plan API and cuBLAS take int sizes: refuse, by name, anything past INT_MAX.
static bool fits_int(std::initializer_list<int64_t> v) {
    for (int64_t x : v) if (x < 0 || x > kIntMax) return false;
    return true;
}

constexpr double kPi = 3.14159265358979323846;

static std::mutex g_mu;
static std::map<std::string, std::unique_ptr<Plan>> g_plans;

// A[j', j] = s·exp(sign·2πi·(o[j']·i[j] mod n)/n), row-major (n_out × n_in).
static std::vector<cuDoubleComplex> fourier_matrix(int64_t n, const int64_t* o, int64_t n_out,
                                                    const int64_t* in, int64_t n_in, int sign,
                                                    double s) {
    std::vector<cuDoubleComplex> a(size_t(n_out) * size_t(n_in));
    for (int64_t r = 0; r < n_out; ++r)
        for (int64_t c = 0; c < n_in; ++c) {
            int64_t m = ((o[r] % n) * (in[c] % n)) % n;
            if (2 * m > n) m -= n;
            const double ang = sign * 2.0 * kPi * double(m) / double(n);
            a[size_t(r) * n_in + c] = make_cuDoubleComplex(s * std::cos(ang), s * std::sin(ang));
        }
    return a;
}

template <class T>
static ffi::Error upload(Plan& p, const std::vector<T>& h, T** dptr) {
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, h.size() * sizeof(T)) != cudaSuccess) return err("cudaMalloc");
    if (cudaMemcpy(ptr, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
        return err("cudaMemcpy");
    p.owned.push_back(ptr);
    *dptr = static_cast<T*>(ptr);
    return ffi::Error::Success();
}

static int64_t prod(const int64_t* e, int d) {
    int64_t r = 1;
    for (int a = 0; a < d; ++a) r *= e[a];
    return r;
}

// Build a plan.  Attributes (per transform axis a = 0..d-1):
//   n[a] full extent; kin[a], kout[a] compact extents; in_idx/out_idx the
//   supports concatenated (identity ranges where an axis has none);
//   sup_in/sup_out 0/1; gemm 0/1; scale[a] the axis' jnp.fft scale;
//   order: GEMM axes in execution order with -1 where the FFT group runs.
static ffi::Error build_plan(Plan& p, int64_t batch, ffi::Span<const int64_t> n,
                             ffi::Span<const int64_t> kin, ffi::Span<const int64_t> kout,
                             ffi::Span<const int64_t> in_idx, ffi::Span<const int64_t> out_idx,
                             ffi::Span<const int64_t> sup_in, ffi::Span<const int64_t> sup_out,
                             ffi::Span<const int64_t> gemm, ffi::Span<const double> scale,
                             ffi::Span<const int64_t> order, int sign) {
    const int d = int(n.size());
    p.d = d;
    p.batch = batch;
    p.sign = sign;
    std::vector<int64_t> ioff(d, 0), ooff(d, 0);
    for (int a = 1; a < d; ++a) {
        ioff[a] = ioff[a - 1] + kin[a - 1];
        ooff[a] = ooff[a - 1] + kout[a - 1];
    }
    int64_t cur[3] = {1, 1, 1};
    for (int a = 0; a < d; ++a) cur[a] = kin[a];
    p.max_elems = batch * prod(cur, d);

    double fft_scale = 1.0;
    bool any_fft = false, any_gemm = false;
    for (int a = 0; a < d; ++a) {
        if (gemm[a]) any_gemm = true;
        else { any_fft = true; fft_scale *= scale[a]; }
    }
    // The FFT axes' scale rides the first GEMM's alpha (a scalar commutes with
    // every stage); with no GEMM it rides a remap pass of the FFT group.
    bool fft_scale_pending = any_fft && fft_scale != 1.0 && !any_gemm;
    bool alpha_pending = any_fft && fft_scale != 1.0 && any_gemm;

    auto push_remap = [&](const int64_t* out_ext, std::vector<std::vector<int32_t>>& maps,
                          double sr) -> ffi::Error {
        Step st;
        st.kind = Step::kRemap;
        st.remap.d = d;
        st.remap.batch = batch;
        for (int a = 0; a < d; ++a) {
            st.remap.in_ext[a] = cur[a];
            st.remap.out_ext[a] = out_ext[a];
            st.remap.map[a] = nullptr;
            if (!maps[a].empty()) {
                int32_t* dm = nullptr;
                if (auto e = upload(p, maps[a], &dm); e.failure()) return e;
                st.remap.map[a] = dm;
            }
            st.in_ext[a] = cur[a];
            st.out_ext[a] = out_ext[a];
            cur[a] = out_ext[a];
        }
        st.sr = sr;
        p.steps.push_back(st);
        p.max_elems = std::max(p.max_elems, batch * prod(cur, d));
        return ffi::Error::Success();
    };

    for (int64_t ax : order) {
        if (ax >= 0) {                                    // one GEMM axis
            const int a = int(ax);
            Step st;
            st.kind = Step::kGemm;
            st.axis = a;
            auto h = fourier_matrix(n[a], out_idx.begin() + ooff[a], kout[a],
                                    in_idx.begin() + ioff[a], kin[a], sign, scale[a]);
            if (auto e = upload(p, h, &st.mat); e.failure()) return e;
            if (alpha_pending) {
                st.sr = fft_scale;
                alpha_pending = false;
            }
            for (int b = 0; b < d; ++b) st.in_ext[b] = cur[b];
            cur[a] = kout[a];
            for (int b = 0; b < d; ++b) st.out_ext[b] = cur[b];
            p.steps.push_back(st);
            p.max_elems = std::max(p.max_elems, batch * prod(cur, d));
            continue;
        }
        // The FFT group: embed (zero-fill + gather), transform, restrict.
        std::vector<std::vector<int32_t>> maps(d);
        int64_t ext[3] = {1, 1, 1};
        bool embed = false;
        for (int a = 0; a < d; ++a) {
            ext[a] = cur[a];
            if (!gemm[a] && sup_in[a]) {
                embed = true;
                ext[a] = n[a];
                maps[a].assign(size_t(n[a]), -1);
                for (int64_t j = 0; j < kin[a]; ++j)
                    maps[a][size_t(in_idx[ioff[a] + j] % n[a])] = int32_t(j);
            }
        }
        if (embed) {
            if (auto e = push_remap(ext, maps, fft_scale_pending ? fft_scale : 1.0); e.failure())
                return e;
            fft_scale_pending = false;
        }
        // cuFFT over each maximal contiguous run [f0, f1] of FFT axes.
        for (int f0 = 0; f0 < d;) {
            if (gemm[f0]) { ++f0; continue; }
            int f1 = f0;
            while (f1 + 1 < d && !gemm[f1 + 1]) ++f1;
            Step st;
            st.kind = Step::kFft;
            int rank = f1 - f0 + 1;
            int nn[3];
            for (int a = f0; a <= f1; ++a) nn[a - f0] = int(n[a]);
            const int64_t left = batch * prod(cur, f0);
            int64_t right = 1;
            for (int a = f1 + 1; a < d; ++a) right *= cur[a];
            int64_t box = 1;
            for (int q = 0; q < rank; ++q) box *= nn[q];
            if (!fits_int({box, left, right, box * right})) {
                std::ostringstream os;
                os << "GATE fourier-plan-int32: the FFT run over axes [" << f0 << "," << f1 << "] has box "
                   << box << ", " << left << " lines before and " << right << " after (box*after "
                   << box * right << "); want each <= 2^31-1 (cuFFT's plan sizes are int); fix: split "
                      "the batch";
                return bad(os.str());
            }
            cufftHandle h;
            if (cufftCreate(&h) != CUFFT_SUCCESS) return err("cufftCreate");
            // The work area comes from XLA's scratch allocator on every call.
            if (cufftSetAutoAllocation(h, 0) != CUFFT_SUCCESS) return err("cufftSetAutoAllocation");
            cufftResult r;
            size_t work = 0;
            if (right == 1) {
                r = cufftMakePlanMany(h, rank, nn, nn, 1, int(box), nn, 1, int(box), CUFFT_Z2Z, int(left),
                                      &work);
                st.loops = 1;
            } else if (left <= right) {
                r = cufftMakePlanMany(h, rank, nn, nn, int(right), 1, nn, int(right), 1, CUFFT_Z2Z,
                                      int(right), &work);
                st.loops = left;
                st.loop_stride = box * right;
            } else {
                r = cufftMakePlanMany(h, rank, nn, nn, int(right), int(box * right), nn, int(right),
                                      int(box * right), CUFFT_Z2Z, int(left), &work);
                st.loops = right;
                st.loop_stride = 1;
            }
            if (r != CUFFT_SUCCESS) {
                cufftDestroy(h);
                std::ostringstream os;
                os << "cufftMakePlanMany failed (" << int(r) << ")";
                return err(os.str());
            }
            st.ffts.push_back(h);
            st.work = work;
            p.max_work = std::max(p.max_work, work);
            for (int b = 0; b < d; ++b) st.in_ext[b] = st.out_ext[b] = cur[b];
            p.steps.push_back(st);
            f0 = f1 + 1;
        }
        // Restriction of the FFT axes with an output support.
        std::vector<std::vector<int32_t>> tmaps(d);
        int64_t text[3] = {1, 1, 1};
        bool take = false;
        for (int a = 0; a < d; ++a) {
            text[a] = cur[a];
            if (!gemm[a] && sup_out[a]) {
                take = true;
                text[a] = kout[a];
                tmaps[a].resize(size_t(kout[a]));
                for (int64_t j = 0; j < kout[a]; ++j)
                    tmaps[a][size_t(j)] = int32_t(out_idx[ooff[a] + j] % n[a]);
            }
        }
        if (take || fft_scale_pending) {
            if (auto e = push_remap(text, tmaps, fft_scale_pending ? fft_scale : 1.0); e.failure())
                return e;
            fft_scale_pending = false;
        }
    }
    return ffi::Error::Success();
}

static std::string plan_key(int dev, int64_t batch, ffi::Span<const int64_t> n,
                            ffi::Span<const int64_t> kin, ffi::Span<const int64_t> kout,
                            ffi::Span<const int64_t> in_idx, ffi::Span<const int64_t> out_idx,
                            ffi::Span<const int64_t> sup_in, ffi::Span<const int64_t> sup_out,
                            ffi::Span<const int64_t> gemm, ffi::Span<const double> scale,
                            ffi::Span<const int64_t> order, int64_t sign) {
    std::string k;
    auto add = [&](const void* p, size_t bytes) { k.append(static_cast<const char*>(p), bytes); };
    add(&dev, sizeof dev);
    add(&batch, sizeof batch);
    add(&sign, sizeof sign);
    for (auto s : {n, kin, kout, in_idx, out_idx, sup_in, sup_out, gemm, order}) {
        const int64_t len = int64_t(s.size());
        add(&len, sizeof len);
        add(s.begin(), s.size() * sizeof(int64_t));
    }
    add(scale.begin(), scale.size() * sizeof(double));
    return k;
}

static ffi::Error run(cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer x,
                      ffi::Result<ffi::AnyBuffer> y, ffi::Span<const int64_t> n,
                      ffi::Span<const int64_t> kin, ffi::Span<const int64_t> kout,
                      ffi::Span<const int64_t> in_idx, ffi::Span<const int64_t> out_idx,
                      ffi::Span<const int64_t> sup_in, ffi::Span<const int64_t> sup_out,
                      ffi::Span<const int64_t> gemm, ffi::Span<const double> scale,
                      ffi::Span<const int64_t> order, int64_t sign) {
    if (x.element_type() != ffi::DataType::C128 || y->element_type() != ffi::DataType::C128)
        return bad("complex128 only");
    const int d = int(n.size());
    auto xd = x.dimensions();
    if (d < 1 || d > 3 || int(xd.size()) < d) return bad("1 to 3 trailing transform axes");
    int64_t batch = 1;
    for (size_t i = 0; i + d < xd.size(); ++i) batch *= xd[i];
    for (int a = 0; a < d; ++a)
        if (xd[xd.size() - d + a] != kin[a]) return bad("x extent does not match kin");
    if (batch == 0) return ffi::Error::Success();

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return err("cudaGetDevice");
    const std::string key =
        plan_key(dev, batch, n, kin, kout, in_idx, out_idx, sup_in, sup_out, gemm, scale, order, sign);
    Plan* p = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_plans.find(key);
        if (it == g_plans.end()) {
            auto np = std::make_unique<Plan>();
            if (auto e = build_plan(*np, batch, n, kin, kout, in_idx, out_idx, sup_in, sup_out,
                                    gemm, scale, order, int(sign));
                e.failure())
                return e;
            it = g_plans.emplace(key, std::move(np)).first;
        }
        p = it->second.get();
    }

    thread_local std::map<int, cublasHandle_t> handles;
    cublasHandle_t& hb = handles[dev];
    if (hb == nullptr && cublasCreate(&hb) != CUBLAS_STATUS_SUCCESS) return err("cublasCreate");
    if (cublasSetStream(hb, stream) != CUBLAS_STATUS_SUCCESS) return err("cublasSetStream");

    // cuFFT's work area for this call, from XLA's scratch allocator.
    void* work = nullptr;
    if (p->max_work > 0) {
        auto w = scratch.Allocate(p->max_work);
        if (!w.has_value()) return err("scratch allocation (cuFFT work area)");
        work = *w;
    }
    // Ping-pong through two scratch buffers; the last step writes y.
    const size_t bytes = size_t(p->max_elems) * sizeof(cuDoubleComplex);
    void* buf[2] = {nullptr, nullptr};
    if (p->steps.size() > 1) {
        auto s0 = scratch.Allocate(bytes);
        if (!s0.has_value()) return err("scratch allocation");
        buf[0] = *s0;
        if (p->steps.size() > 2) {
            auto s1 = scratch.Allocate(bytes);
            if (!s1.has_value()) return err("scratch allocation");
            buf[1] = *s1;
        }
    }
    const void* src = x.untyped_data();
    int which = 0;
    for (size_t i = 0; i < p->steps.size(); ++i) {
        Step& st = p->steps[i];
        const bool last = (i + 1 == p->steps.size());
        void* dst = last ? y->untyped_data() : buf[which];
        if (st.kind == Step::kGemm) {
            const int a = st.axis;
            int64_t L = batch, R = 1;
            for (int b = 0; b < a; ++b) L *= st.in_ext[b];
            for (int b = a + 1; b < d; ++b) R *= st.in_ext[b];
            const int64_t K = st.in_ext[a], Np = st.out_ext[a];
            if (!fits_int({L, R, K, Np})) {
                std::ostringstream os;
                os << "GATE fourier-plan-int32: the GEMM on axis " << a << " has (L, K, N', R) = (" << L
                   << ", " << K << ", " << Np << ", " << R << "); want each <= 2^31-1 (cuBLAS sizes are "
                      "int); fix: split the batch";
                return bad(os.str());
            }
            const cuDoubleComplex alpha = make_cuDoubleComplex(st.sr, st.si);
            const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
            const auto* X = static_cast<const cuDoubleComplex*>(src);
            auto* Y = static_cast<cuDoubleComplex*>(dst);
            cublasStatus_t s;
            if (R == 1) {   // Y (L × N') = X (L × K) · Aᵀ, row-major
                s = cublasZgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, int(Np), int(L), int(K), &alpha,
                                st.mat, int(K), X, int(K), &beta, Y, int(Np));
            } else {        // per l: Y_l (N' × R) = A (N' × K) · X_l (K × R), A shared
                s = cublasZgemmStridedBatched(hb, CUBLAS_OP_N, CUBLAS_OP_N, int(R), int(Np), int(K),
                                              &alpha, X, int(R), K * R, st.mat, int(K), 0, &beta,
                                              Y, int(R), Np * R, int(L));
            }
            if (s != CUBLAS_STATUS_SUCCESS) {
                std::ostringstream os;
                os << "cuBLAS failed (" << int(s) << ") on axis " << a;
                return err(os.str());
            }
        } else if (st.kind == Step::kRemap) {
            launch_remap(src, dst, st.remap, st.sr, st.si, stream);
        } else {
            const auto dir = p->sign < 0 ? CUFFT_FORWARD : CUFFT_INVERSE;
            cufftHandle h = st.ffts[0];
            if (cufftSetStream(h, stream) != CUFFT_SUCCESS) return err("cufftSetStream");
            if (st.work > 0 && cufftSetWorkArea(h, work) != CUFFT_SUCCESS) return err("cufftSetWorkArea");
            auto* in = const_cast<cufftDoubleComplex*>(static_cast<const cufftDoubleComplex*>(src));
            auto* out = static_cast<cufftDoubleComplex*>(dst);
            for (int64_t l = 0; l < st.loops; ++l) {
                if (cufftExecZ2Z(h, in + l * st.loop_stride, out + l * st.loop_stride, dir) !=
                    CUFFT_SUCCESS)
                    return err("cufftExecZ2Z");
            }
        }
        if (cudaPeekAtLastError() != cudaSuccess) return err("kernel launch");
        src = dst;
        which ^= 1;
    }
    if (p->steps.empty() &&
        cudaMemcpyAsync(y->untyped_data(), x.untyped_data(), size_t(batch) * prod(kin.begin(), d) *
                                                                 sizeof(cuDoubleComplex),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
        return err("cudaMemcpyAsync");
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::fourier_plan

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LorraxFourierPlanCudaFfi, lorrax_ffi::fourier_plan::run,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<xla::ffi::Span<const int64_t>>("n")
        .Attr<xla::ffi::Span<const int64_t>>("kin")
        .Attr<xla::ffi::Span<const int64_t>>("kout")
        .Attr<xla::ffi::Span<const int64_t>>("in_idx")
        .Attr<xla::ffi::Span<const int64_t>>("out_idx")
        .Attr<xla::ffi::Span<const int64_t>>("sup_in")
        .Attr<xla::ffi::Span<const int64_t>>("sup_out")
        .Attr<xla::ffi::Span<const int64_t>>("gemm")
        .Attr<xla::ffi::Span<const double>>("scale")
        .Attr<xla::ffi::Span<const int64_t>>("order")
        .Attr<int64_t>("sign"));
